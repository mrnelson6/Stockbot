#!/usr/bin/env python3
"""Random portfolio trader — a standalone control/experiment bot.

Invests money entirely at random, reusing the existing Alpaca execution plumbing
but running its OWN isolated portfolio (separate Alpaca account/keys) so it never
interferes with the ML bot.

Strategy (all randomness lives in stockbot.random_bot.RandomAllocator):
- Per-tick probability: on each poll it trades with probability --trade-prob.
- Random partial churn: each trade event sells a random subset of holdings...
- ...and buys a random number of new names with Dirichlet-weighted slices of cash.

Isolation: an Alpaca account IS its API keys. For a truly separate portfolio, give
this bot its own paper account via RANDOM_ALPACA_API_KEY / RANDOM_ALPACA_SECRET_KEY
(falls back to ALPACA_* if those are unset — but then it shares the ML bot's account).

Safety: trades are SIMULATED unless you pass --execute, and paper mode defaults on.

Usage:
    python scripts/random_portfolio.py                              # dry run (no orders)
    python scripts/random_portfolio.py --universe-size 25 --seed 1  # smaller universe
    python scripts/random_portfolio.py --execute                    # place paper orders
"""

import argparse
import os
import signal
import sys
import time
from decimal import Decimal
from typing import Optional

from dotenv import load_dotenv

from stockbot.config.settings import AlpacaConfig
from stockbot.config.universe import get_universe
from stockbot.core.models import Order
from stockbot.core.types import OrderSide, OrderType, Price, Quantity, Symbol, Timestamp
from stockbot.data.providers.alpaca import AlpacaDataProvider
from stockbot.execution.broker_alpaca import AlpacaBroker
from stockbot.monitoring import setup_logging
from stockbot.monitoring.logger import get_logger
from stockbot.random_bot import RandomAllocator, fetch_dynamic_universe
from stockbot.web.recorder import SnapshotRecorder

logger = get_logger("random_portfolio")


def load_random_alpaca_config() -> AlpacaConfig:
    """Build an AlpacaConfig for the random bot's own (separate) account.

    Reads RANDOM_ALPACA_API_KEY / RANDOM_ALPACA_SECRET_KEY / RANDOM_ALPACA_PAPER.
    Falls back to the shared ALPACA_* vars if the RANDOM_* ones are not set (in
    which case this bot shares the ML bot's account — not recommended).
    """
    load_dotenv()

    api_key = os.getenv("RANDOM_ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("RANDOM_ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca credentials not configured. Set RANDOM_ALPACA_API_KEY and "
            "RANDOM_ALPACA_SECRET_KEY (preferred, for an isolated portfolio) or the "
            "shared ALPACA_API_KEY / ALPACA_SECRET_KEY."
        )

    paper_str = os.getenv("RANDOM_ALPACA_PAPER", os.getenv("ALPACA_PAPER", "true"))
    paper = paper_str.lower() == "true"

    if not os.getenv("RANDOM_ALPACA_API_KEY"):
        logger.warning(
            "RANDOM_ALPACA_API_KEY not set - falling back to shared ALPACA_* keys. "
            "This bot will share the ML bot's account and may fight over positions."
        )

    return AlpacaConfig(api_key=api_key, secret_key=secret_key, paper=paper)


class RandomPortfolioTrader:
    """Drives a RandomAllocator against a (separate) Alpaca account."""

    def __init__(
        self,
        symbols: list[str],
        allocator: RandomAllocator,
        capital: float = 100_000.0,
        execute_trades: bool = False,
        feed: str = "iex",
        recorder: Optional[SnapshotRecorder] = None,
    ) -> None:
        self._symbols = symbols
        self._allocator = allocator
        self._execute = execute_trades
        self._feed = feed
        self._recorder = recorder

        self._config = load_random_alpaca_config()
        self._data_provider = AlpacaDataProvider(self._config, feed=feed)
        self._broker = AlpacaBroker(self._config) if execute_trades else None

        # Simulated portfolio state (used only in dry-run mode).
        self._sim_cash = capital
        self._sim_positions: dict[str, int] = {s: 0 for s in symbols}

        self._running = False
        self._tick = 0

    # -- market data ---------------------------------------------------------

    def _get_current_prices(self) -> dict[str, float]:
        """Latest price per symbol (mirrors live_portfolio._get_current_prices)."""
        prices: dict[str, float] = {}
        for symbol in self._symbols:
            try:
                bar = self._data_provider.get_latest(Symbol(symbol))
                if bar:
                    prices[symbol] = float(bar.close)
            except Exception:
                pass
        return prices

    # -- account / positions -------------------------------------------------

    def _get_positions(self) -> dict[str, int]:
        """Current long share count per symbol."""
        if not self._broker:
            return {s: q for s, q in self._sim_positions.items() if q > 0}
        try:
            positions = self._broker.get_positions()
            result: dict[str, int] = {}
            for symbol in self._symbols:
                pos = positions.get(Symbol(symbol))
                if pos and int(pos.quantity) > 0:
                    result[symbol] = int(pos.quantity)
            return result
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            return {}

    def _get_cash(self) -> float:
        if not self._broker:
            return self._sim_cash
        try:
            return float(self._broker.get_cash())
        except Exception:
            return 0.0

    def _get_account_value(self) -> float:
        if not self._broker:
            return self._sim_cash + sum(self._sim_positions.values())  # rough; refined per tick
        try:
            return float(self._broker.get_equity())
        except Exception:
            return self._sim_cash

    # -- execution -----------------------------------------------------------

    def _execute_intent(self, intent, prices: dict[str, float]) -> None:
        """Turn a TradeIntent into orders (or simulate them in dry-run mode)."""
        if intent.is_empty:
            print("  (random plan: hold - nothing to trade)")
            return

        # ----- SELLS first (free up cash) -----
        for symbol, shares in intent.sells.items():
            price = prices.get(symbol, 0.0)
            if shares <= 0:
                continue
            if not self._execute or not self._broker:
                proceeds = shares * price
                self._sim_cash += proceeds
                self._sim_positions[symbol] = max(0, self._sim_positions.get(symbol, 0) - shares)
                print(f"    [SIM] SELL {shares:5} {symbol:6} @ ${price:.2f}  (+${proceeds:,.0f})")
                self._record_trade(symbol, "SELL", shares, price)
                continue
            try:
                order_id = self._broker.submit_order(
                    Order(
                        symbol=Symbol(symbol),
                        side=OrderSide.SELL,
                        quantity=Quantity(Decimal(shares)),
                        order_type=OrderType.MARKET,
                        created_at=Timestamp(int(time.time() * 1_000_000_000)),
                    )
                )
                print(f"    SOLD {shares:5} {symbol:6} @ ${price:.2f}")
                self._record_trade(symbol, "SELL", shares, price, order_id)
            except Exception as e:
                logger.error(f"Failed to sell {symbol}: {e}")

        # Let sells settle and refresh available cash/buying power.
        if self._execute and self._broker and intent.sells:
            time.sleep(2)
            try:
                buying_power = float(self._broker.get_buying_power())
            except Exception:
                buying_power = 0.0
        else:
            buying_power = self._sim_cash

        if intent.buys and buying_power < 100:
            print("  WARNING: insufficient cash/buying power - skipping buys")
            return

        # ----- BUYS (Dirichlet-sized dollar targets -> whole shares) -----
        spent = 0.0
        for symbol, target_value in sorted(intent.buys.items(), key=lambda kv: kv[1], reverse=True):
            price = prices.get(symbol, 0.0)
            if price <= 0:
                continue
            budget = min(target_value, buying_power * 0.95 - spent)
            shares = int(budget / price)
            if shares < 1:
                print(f"    SKIP {symbol:6} - insufficient cash for 1 share")
                continue
            cost = shares * price
            if not self._execute or not self._broker:
                self._sim_cash -= cost
                self._sim_positions[symbol] = self._sim_positions.get(symbol, 0) + shares
                spent += cost
                print(f"    [SIM] BUY  {shares:5} {symbol:6} @ ${price:.2f}  (-${cost:,.0f})")
                self._record_trade(symbol, "BUY", shares, price)
                continue
            try:
                order_id = self._broker.submit_order(
                    Order(
                        symbol=Symbol(symbol),
                        side=OrderSide.BUY,
                        quantity=Quantity(Decimal(shares)),
                        order_type=OrderType.MARKET,
                        created_at=Timestamp(int(time.time() * 1_000_000_000)),
                    )
                )
                spent += cost
                print(f"    BOUGHT {shares:5} {symbol:6} @ ${price:.2f} (${cost:,.0f})")
                self._record_trade(symbol, "BUY", shares, price, order_id)
            except Exception as e:
                logger.error(f"Failed to buy {symbol}: {e}")

        if spent > 0:
            print(f"  Deployed ${spent:,.2f} this event")

    # -- dashboard recording -------------------------------------------------

    def _record_trade(
        self, symbol: str, side: str, qty: int, price: float, order_id: Optional[str] = None
    ) -> None:
        if self._recorder:
            self._recorder.record_trade(symbol, side, float(qty), price, order_id)

    def _snapshot_state(self, prices: dict[str, float]) -> tuple[float, float, list[dict]]:
        """Build (equity, cash, positions) for the dashboard from broker or sim state."""
        positions: list[dict] = []
        if self._broker:
            cash = self._get_cash()
            try:
                equity = float(self._broker.get_equity())
            except Exception:
                equity = cash
            try:
                for sym, pos in self._broker.get_positions().items():
                    qty = float(pos.quantity)
                    if qty == 0:
                        continue
                    avg = float(pos.average_price)
                    price = prices.get(str(sym), avg)
                    positions.append(
                        {
                            "symbol": str(sym),
                            "qty": qty,
                            "avg_price": avg,
                            "market_value": qty * price,
                            "unrealized_pnl": float(pos.unrealized_pnl),
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to read positions for snapshot: {e}")
            return equity, cash, positions

        # Simulated book: mark to current prices.
        cash = self._sim_cash
        equity = cash
        for sym, qty in self._sim_positions.items():
            if qty <= 0:
                continue
            price = prices.get(sym, 0.0)
            mv = qty * price
            equity += mv
            positions.append(
                {
                    "symbol": sym,
                    "qty": float(qty),
                    "avg_price": price,
                    "market_value": mv,
                    "unrealized_pnl": None,
                }
            )
        return equity, cash, positions

    # -- main loop -----------------------------------------------------------

    def _process_tick(self) -> None:
        self._tick += 1
        print(f"\n=== Tick {self._tick} ===", flush=True)

        # Fetch prices every tick: needed both for trading and to mark the
        # portfolio to market for the dashboard's near-real-time equity curve.
        prices = self._get_current_prices()
        if not prices:
            logger.warning("No prices available - skipping tick")
            return

        if self._allocator.should_trade():
            positions = self._get_positions()
            cash = self._get_cash()
            print(f"  Trade event! cash ~${cash:,.0f}, holdings={len(positions)}")

            intent = self._allocator.plan_trade(
                current_positions={Symbol(s): q for s, q in positions.items()},
                prices={Symbol(s): p for s, p in prices.items()},
                cash=cash,
                universe=[Symbol(s) for s in self._symbols],
            )
            # Keys come back as Symbol (str subtype); index dicts with plain str.
            intent.sells = {str(s): q for s, q in intent.sells.items()}
            intent.buys = {str(s): v for s, v in intent.buys.items()}
            self._execute_intent(intent, prices)
        else:
            print("  No trade this tick (random gate).")

        # Record state for the dashboard regardless of whether we traded.
        if self._recorder:
            equity, cash, positions = self._snapshot_state(prices)
            self._recorder.snapshot(equity, cash, positions)

    def run(self, poll_interval: float = 60.0, run_24_7: bool = True) -> None:
        self._running = True

        def _stop(signum, frame):
            print("\nStopping...")
            self._running = False

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        mode = "EXECUTE (real orders)" if self._execute else "DRY RUN (simulated)"
        print(f"Random portfolio trader started - {mode}")
        print(f"Universe: {len(self._symbols)} symbols, poll every {poll_interval}s")
        if self._broker:
            print(f"Account paper={self._config.paper}")

        while self._running:
            # Only honor market hours when actually placing orders.
            if self._execute and self._broker and not self._broker.is_market_open():
                if run_24_7:
                    print("Market closed - waiting 60s...", flush=True)
                    time.sleep(min(60.0, poll_interval))
                    continue
                print("Market closed - exiting.")
                break

            try:
                self._process_tick()
            except Exception as e:
                logger.error(f"Tick failed: {e}")

            for _ in range(int(poll_interval)):
                if not self._running:
                    break
                time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Random portfolio trader (control bot)")
    parser.add_argument(
        "--universe-source",
        type=str,
        default="alpaca",
        choices=["static", "alpaca"],
        help="'alpaca' (default) = liquid equities fetched live; 'static' = hand-curated list",
    )
    parser.add_argument(
        "--universe-size",
        type=int,
        default=25,
        choices=[10, 25, 50, 100],
        help="Size of the static universe (ignored when --universe-source alpaca)",
    )
    # Dynamic (alpaca) universe filters
    parser.add_argument("--min-price", type=float, default=5.0, help="[alpaca] Min share price")
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=10_000_000.0,
        help="[alpaca] Min average daily dollar-volume",
    )
    parser.add_argument(
        "--max-symbols", type=int, default=200, help="[alpaca] Cap on number of names (most liquid kept)"
    )
    parser.add_argument(
        "--lookback-days", type=int, default=30, help="[alpaca] Days of daily bars used to gauge liquidity"
    )
    parser.add_argument("--poll-interval", type=float, default=60.0, help="Seconds between ticks")
    parser.add_argument("--trade-prob", type=float, default=0.1, help="Per-tick probability of trading")
    parser.add_argument("--churn-sell-prob", type=float, default=0.3, help="Per-holding sell probability per event")
    parser.add_argument("--min-buys", type=int, default=1)
    parser.add_argument("--max-buys", type=int, default=5)
    parser.add_argument("--deploy-min", type=float, default=0.9, help="Min fraction of cash deployed per event")
    parser.add_argument("--deploy-max", type=float, default=1.0, help="Max fraction of cash deployed per event")
    parser.add_argument("--max-position-value", type=float, default=20_000.0)
    parser.add_argument("--capital", type=float, default=100_000.0, help="Starting cash for dry-run sim")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--feed", type=str, default="iex", choices=["iex", "sip"])
    parser.add_argument("--execute", action="store_true", help="Place real (paper/live) orders")
    parser.add_argument(
        "--record-db",
        type=str,
        default=None,
        help="Path to a SQLite file to record equity/positions/trades for the web dashboard",
    )
    parser.add_argument("--label", type=str, default="Random Bot", help="Display name on the dashboard")
    args = parser.parse_args()

    setup_logging()

    if args.universe_source == "alpaca":
        symbols = fetch_dynamic_universe(
            load_random_alpaca_config(),
            feed=args.feed,
            min_price=args.min_price,
            min_dollar_volume=args.min_dollar_volume,
            max_symbols=args.max_symbols,
            lookback_days=args.lookback_days,
        )
        if not symbols:
            print(
                "ERROR: dynamic universe is empty - loosen --min-price / --min-dollar-volume "
                "or check Alpaca data access.",
                file=sys.stderr,
            )
            return 1
    else:
        symbols = get_universe(args.universe_size)

    allocator = RandomAllocator(
        trade_prob=args.trade_prob,
        churn_sell_prob=args.churn_sell_prob,
        min_buys=args.min_buys,
        max_buys=args.max_buys,
        deploy_fraction_range=(args.deploy_min, args.deploy_max),
        max_position_value=args.max_position_value,
        seed=args.seed,
    )

    recorder = None
    if args.record_db:
        # The recorder needs a data provider for SPY pricing; build it up front so
        # the trader and recorder share one.
        recorder = SnapshotRecorder(
            args.record_db,
            AlpacaDataProvider(load_random_alpaca_config(), feed=args.feed),
            label=args.label,
        )

    trader = RandomPortfolioTrader(
        symbols=symbols,
        allocator=allocator,
        capital=args.capital,
        execute_trades=args.execute,
        feed=args.feed,
        recorder=recorder,
    )
    trader.run(poll_interval=args.poll_interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
