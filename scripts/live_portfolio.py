#!/usr/bin/env python3
"""Live portfolio monitoring and trading with ML agent.

Monitors real-time prices for all stocks in the universe and uses
the trained portfolio agent to decide optimal allocations.

Usage:
    python scripts/live_portfolio.py                    # Monitor only
    python scripts/live_portfolio.py --execute          # Execute trades
    python scripts/live_portfolio.py --train-live       # Keep learning
    python scripts/live_portfolio.py --universe-size 25 # Smaller universe
"""

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np

from stockbot.config.settings import load_settings
from stockbot.config.universe import get_universe
from stockbot.core.models import Bar
from stockbot.core.types import Price, Quantity, Symbol, Timeframe, Timestamp
from stockbot.data.providers.alpaca import AlpacaDataProvider
from stockbot.data.storage import load_bars
from stockbot.execution.broker_alpaca import AlpacaBroker
from stockbot.learning.features import FeatureExtractor, MarketFeatures
from stockbot.learning.portfolio_agent import PortfolioAgent
from stockbot.monitoring import setup_logging
from stockbot.monitoring.logger import get_logger

logger = get_logger("live_portfolio")


class LivePortfolioTrader:
    """Live multi-asset portfolio trader."""

    def __init__(
        self,
        symbols: list[str],
        agent_path: Path,
        data_dir: Path,
        capital: float = 100_000.0,
        execute_trades: bool = False,
        continue_learning: bool = False,
        max_position_value: float = 20_000.0,  # Max $ per position
    ) -> None:
        self._symbols = symbols
        self._agent_path = agent_path
        self._data_dir = data_dir
        self._capital = capital
        self._execute = execute_trades
        self._learn = continue_learning
        self._max_position = max_position_value

        # Load settings
        self._settings = load_settings()
        if self._settings.alpaca is None:
            raise ValueError("Alpaca credentials not configured")

        # Initialize components
        self._data_provider = AlpacaDataProvider(self._settings.alpaca)
        self._broker = AlpacaBroker(self._settings.alpaca) if execute_trades else None

        # Feature extractor
        self._feature_extractor = FeatureExtractor(
            lookback_periods=[5, 10, 20, 50],
            include_volume=True,
            include_volatility=True,
            include_momentum=True,
            include_mean_reversion=True,
        )

        # Load agent
        self._agent = self._load_or_create_agent()

        # State: bars for each symbol
        self._bars: dict[str, list[Bar]] = {s: [] for s in symbols}
        self._running = False

        # Current positions (simulated if not executing)
        self._positions: dict[str, int] = {s: 0 for s in symbols}
        self._position_values: dict[str, float] = {s: 0.0 for s in symbols}

        # Performance tracking
        self._decisions: list[dict] = []
        self._last_features_dict: Optional[dict[str, MarketFeatures]] = None
        self._last_allocations: Optional[dict[str, float]] = None

    def _load_or_create_agent(self) -> PortfolioAgent:
        """Load trained agent or create new."""
        agent = PortfolioAgent(
            symbols=self._symbols,
            feature_extractor=self._feature_extractor,
            hidden_sizes=[256, 128, 64],
            learning_rate=0.0003,  # Lower for live
            gamma=0.95,
            epsilon_start=0.1 if self._learn else 0.0,
            epsilon_end=0.02,
            epsilon_decay=0.9998,
            batch_size=32,
            max_position_per_asset=0.20,
        )

        if self._agent_path.exists():
            try:
                agent.load(self._agent_path)
                print(f"Loaded agent from {self._agent_path}")
                self._show_agent_info(agent)
            except Exception as e:
                print(f"Warning: Could not load agent: {e}")
                print("Starting with untrained agent")
        else:
            print("No trained agent found - starting fresh")
            print(f"Train first: python scripts/train_portfolio.py --universe-size {len(self._symbols)}")

        return agent

    def _show_agent_info(self, agent: PortfolioAgent) -> None:
        """Display agent statistics."""
        stats = agent.get_stats()

        print(f"\nAgent Statistics:")
        print(f"  Training steps: {stats['steps']:,}")
        print(f"  Total P&L: {stats['total_pnl_pct']:.2f}%")
        print(f"  Epsilon: {stats['epsilon']:.4f}")

        print(f"\nTop learned positions:")
        for symbol, weight in agent.get_top_positions(5):
            if abs(weight) > 0.01:
                direction = "LONG" if weight > 0 else "SHORT"
                print(f"  {symbol:6} {direction:5} {abs(weight):.1%}")

    def _load_historical(self) -> None:
        """Load historical bars for all symbols."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)  # 30 days should be enough

        start_ts = Timestamp(int(start.timestamp() * 1_000_000_000))
        end_ts = Timestamp(int(end.timestamp() * 1_000_000_000))

        print(f"\nLoading historical data for {len(self._symbols)} symbols...")

        for i, symbol in enumerate(self._symbols, 1):
            sym = Symbol(symbol)

            # Try cache first
            try:
                cached = list(load_bars(self._data_dir, sym, Timeframe.DAY_1))
                recent = [b for b in cached if b.timestamp >= start_ts]
                if len(recent) >= 20:
                    self._bars[symbol] = recent
                    continue
            except Exception:
                pass

            # Fetch
            try:
                bars = list(self._data_provider.get_bars(
                    sym, start_ts, end_ts, Timeframe.DAY_1
                ))
                self._bars[symbol] = bars
            except Exception as e:
                logger.warning(f"Failed to get history for {symbol}: {e}")

            if i % 20 == 0:
                print(f"  Loaded {i}/{len(self._symbols)}...")

        loaded = sum(1 for bars in self._bars.values() if len(bars) >= 20)
        print(f"Loaded history for {loaded}/{len(self._symbols)} symbols")

    def _get_current_prices(self) -> dict[str, float]:
        """Get current prices for all symbols."""
        prices = {}
        for symbol in self._symbols:
            try:
                bar = self._data_provider.get_latest(Symbol(symbol))
                if bar:
                    prices[symbol] = float(bar.close)
            except Exception:
                pass
        return prices

    def _update_bars(self, prices: dict[str, float]) -> None:
        """Update bars with latest prices."""
        now = datetime.now(timezone.utc)
        ts = Timestamp(int(now.timestamp() * 1_000_000_000))

        for symbol, price in prices.items():
            if symbol not in self._bars:
                continue

            # Create a simple bar from the price
            bar = Bar(
                symbol=Symbol(symbol),
                timestamp=ts,
                open=Price(Decimal(str(price))),
                high=Price(Decimal(str(price))),
                low=Price(Decimal(str(price))),
                close=Price(Decimal(str(price))),
                volume=0,
            )

            self._bars[symbol].append(bar)
            # Keep last 200 bars
            if len(self._bars[symbol]) > 200:
                self._bars[symbol] = self._bars[symbol][-200:]

    def _build_features(self) -> dict[str, MarketFeatures]:
        """Build features for all symbols."""
        features_dict = {}
        for symbol, bars in self._bars.items():
            if len(bars) < 60:
                continue
            features = self._feature_extractor.extract(bars)
            if features is not None:
                features_dict[symbol] = features
        return features_dict

    def _get_account_value(self) -> float:
        """Get current account value."""
        if self._broker:
            try:
                return float(self._broker.get_equity())
            except Exception:
                pass
        return self._capital

    def _get_current_positions(self) -> dict[str, tuple[int, float]]:
        """Get current positions (shares, value) for each symbol."""
        if not self._broker:
            return {s: (self._positions[s], self._position_values[s])
                    for s in self._symbols}

        try:
            positions = self._broker.get_positions()
            result = {}
            for symbol in self._symbols:
                if symbol in positions:
                    pos = positions[symbol]
                    shares = int(pos.quantity)
                    value = float(pos.market_value)
                    result[symbol] = (shares, value)
                else:
                    result[symbol] = (0, 0.0)
            return result
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            return {s: (0, 0.0) for s in self._symbols}

    def _execute_rebalance(
        self,
        target_allocations: dict[str, float],
        prices: dict[str, float],
        account_value: float,
    ) -> None:
        """Rebalance portfolio to target allocations."""
        current_positions = self._get_current_positions()

        trades_needed = []
        for symbol in self._symbols:
            if symbol not in prices or symbol not in target_allocations:
                continue

            target_value = target_allocations[symbol] * account_value
            target_value = np.clip(target_value, -self._max_position, self._max_position)
            target_shares = int(target_value / prices[symbol]) if prices[symbol] > 0 else 0

            current_shares, _ = current_positions.get(symbol, (0, 0.0))
            delta = target_shares - current_shares

            if abs(delta) >= 1:
                trades_needed.append({
                    "symbol": symbol,
                    "current": current_shares,
                    "target": target_shares,
                    "delta": delta,
                    "price": prices[symbol],
                    "value": abs(delta) * prices[symbol],
                })

        if not trades_needed:
            return

        # Sort by value (largest trades first)
        trades_needed.sort(key=lambda x: x["value"], reverse=True)

        print(f"\n  Rebalancing {len(trades_needed)} positions:")
        for trade in trades_needed[:10]:  # Show top 10
            action = "BUY" if trade["delta"] > 0 else "SELL"
            print(f"    {trade['symbol']:6} {action:4} {abs(trade['delta']):4} "
                  f"shares @ ${trade['price']:.2f} (${trade['value']:.0f})")

        if len(trades_needed) > 10:
            print(f"    ... and {len(trades_needed) - 10} more")

        if not self._execute or not self._broker:
            # Simulate
            for trade in trades_needed:
                self._positions[trade["symbol"]] = trade["target"]
                self._position_values[trade["symbol"]] = trade["target"] * trade["price"]
            return

        # Execute trades
        from stockbot.core.models import Order
        from stockbot.core.types import OrderSide, OrderType

        for trade in trades_needed:
            try:
                side = OrderSide.BUY if trade["delta"] > 0 else OrderSide.SELL
                order = Order(
                    symbol=Symbol(trade["symbol"]),
                    side=side,
                    quantity=Quantity(Decimal(abs(trade["delta"]))),
                    order_type=OrderType.MARKET,
                    timestamp=Timestamp(int(time.time() * 1_000_000_000)),
                )
                self._broker.submit_order(order)
                action = "BOUGHT" if trade["delta"] > 0 else "SOLD"
                print(f"    EXECUTED: {action} {abs(trade['delta'])} {trade['symbol']}")

            except Exception as e:
                logger.error(f"Failed to execute {trade['symbol']}: {e}")

    def _process_tick(self) -> None:
        """Process one tick of market data."""
        # Get current prices
        prices = self._get_current_prices()
        if len(prices) < len(self._symbols) // 2:
            logger.warning(f"Only got prices for {len(prices)} symbols")
            return

        # Update bars
        self._update_bars(prices)

        # Build features
        features_dict = self._build_features()
        if len(features_dict) < len(self._symbols) // 2:
            logger.warning(f"Only got features for {len(features_dict)} symbols")
            return

        # Get account value
        account_value = self._get_account_value()

        # Get allocations from agent
        allocations, info = self._agent.get_allocations(
            features_dict, account_value, training=self._learn
        )

        # Display
        timestamp = datetime.now(timezone.utc)
        current_positions = self._get_current_positions()

        # Calculate portfolio stats
        total_long = sum(v for v in allocations.values() if v > 0)
        total_short = sum(abs(v) for v in allocations.values() if v < 0)
        n_long = sum(1 for v in allocations.values() if v > 0.01)
        n_short = sum(1 for v in allocations.values() if v < -0.01)

        print(f"\n{'=' * 70}")
        print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Portfolio Update")
        print(f"{'=' * 70}")
        print(f"Account Value: ${account_value:,.0f}")
        print(f"Symbols with data: {len(features_dict)}")
        print(f"Target: {n_long} LONG ({total_long:.0%}), {n_short} SHORT ({total_short:.0%})")
        print(f"Gross Exposure: {info['gross_exposure']:.0%}, Net Exposure: {info['net_exposure']:.0%}")

        # Top positions
        sorted_allocs = sorted(allocations.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\nTop Target Allocations:")
        for symbol, alloc in sorted_allocs[:10]:
            if abs(alloc) < 0.01:
                continue
            direction = "LONG " if alloc > 0 else "SHORT"
            price = prices.get(symbol, 0)
            value = abs(alloc) * account_value
            curr_shares, curr_value = current_positions.get(symbol, (0, 0.0))
            print(f"  {symbol:6} {direction} {abs(alloc):>5.1%} "
                  f"(${value:>7,.0f}) @ ${price:>8.2f}  [current: {curr_shares:>4} shares]")

        # Record decision
        self._decisions.append({
            "time": timestamp.isoformat(),
            "allocations": allocations,
            "info": info,
        })

        # Execute rebalance
        self._execute_rebalance(allocations, prices, account_value)

        # Learn from previous
        if self._learn and self._last_features_dict and self._last_allocations:
            reward = self._agent.observe_result(
                features_dict,
                self._last_features_dict,
                self._last_allocations,
                account_value,
                done=False,
            )
            if abs(reward) > 0.1:
                print(f"\nLearning: reward={reward:.4f}")

        self._last_features_dict = features_dict
        self._last_allocations = allocations

    def run(self, poll_interval: float = 60.0) -> None:
        """Run live trading loop."""
        self._running = True
        self._setup_signals()

        print("\n" + "=" * 70)
        print("LIVE PORTFOLIO TRADER")
        print("=" * 70)
        print(f"Symbols: {len(self._symbols)}")
        print(f"Capital: ${self._capital:,.0f}")
        print(f"Max Position: ${self._max_position:,.0f}")
        print(f"Execute Trades: {self._execute}")
        print(f"Continue Learning: {self._learn}")
        print("=" * 70)

        self._load_historical()

        if self._broker:
            if not self._broker.is_market_open():
                print("\nMarket closed. Will act when open.\n")
            else:
                print("\nMarket is OPEN.\n")

        print("\nMonitoring... (Ctrl+C to stop)\n")

        try:
            while self._running:
                try:
                    self._process_tick()
                except Exception as e:
                    logger.error(f"Error in tick: {e}")

                if self._running:
                    time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n\nStopped by user")

        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Clean shutdown."""
        print("\nShutting down...")

        if self._learn:
            self._agent.save(self._agent_path)
            print(f"Saved agent to {self._agent_path}")

        self._print_summary()

    def _print_summary(self) -> None:
        """Print session summary."""
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)

        print(f"Total Decisions: {len(self._decisions)}")

        if self._learn:
            stats = self._agent.get_stats()
            print(f"\nAgent Updated:")
            print(f"  Steps: {stats['steps']:,}")
            print(f"  Epsilon: {stats['epsilon']:.4f}")
            print(f"  Total P&L: {stats['total_pnl_pct']:.2f}%")

        print("=" * 70 + "\n")

    def _setup_signals(self) -> None:
        """Setup signal handlers."""
        def handler(signum, frame):
            if not self._running:
                sys.exit(1)
            print("\nStopping...")
            self._running = False

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def stop(self) -> None:
        self._running = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live portfolio trading with ML agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/live_portfolio.py                    # Monitor 100 stocks
    python scripts/live_portfolio.py --universe-size 25 # Monitor 25 stocks
    python scripts/live_portfolio.py --execute          # Execute trades
    python scripts/live_portfolio.py --train-live       # Keep learning
        """,
    )

    parser.add_argument("--universe-size", type=int, default=100,
                        choices=[10, 25, 50, 100],
                        help="Number of stocks (default: 100)")
    parser.add_argument("--agent", type=Path, default=Path("./data/portfolio_agent.json"))
    parser.add_argument("--execute", action="store_true",
                        help="Execute trades (paper account)")
    parser.add_argument("--train-live", action="store_true",
                        help="Continue learning from live data")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Trading capital (default: 100000)")
    parser.add_argument("--max-position", type=float, default=20_000.0,
                        help="Max $ per position (default: 20000)")
    parser.add_argument("--poll-interval", type=float, default=60.0,
                        help="Seconds between updates (default: 60)")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/portfolio"))
    parser.add_argument("--log-level", type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)

    symbols = get_universe(args.universe_size)

    if args.execute:
        print("\n" + "!" * 70)
        print("WARNING: Execute mode - will place real orders!")
        print("!" * 70)
        if input("\nContinue? [y/N]: ").lower() != 'y':
            return 0

    try:
        trader = LivePortfolioTrader(
            symbols=symbols,
            agent_path=args.agent,
            data_dir=args.data_dir,
            capital=args.capital,
            execute_trades=args.execute,
            continue_learning=args.train_live,
            max_position_value=args.max_position,
        )
        trader.run(poll_interval=args.poll_interval)

    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
