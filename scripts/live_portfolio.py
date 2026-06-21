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
        continue_learning: bool = True,  # Default ON
        max_position_value: float = 20_000.0,  # Max $ per long position
        max_short_value: float = 20_000.0,  # Max $ per short position
        feed: str = "iex",  # "iex" (free, default) or "sip" (paid)
        allow_shorting: bool = False,  # Whether to allow short positions
    ) -> None:
        self._symbols = symbols
        self._agent_path = agent_path
        self._data_dir = data_dir
        self._capital = capital
        self._execute = execute_trades
        self._learn = continue_learning
        self._max_position = max_position_value
        self._max_short = max_short_value
        self._feed = feed
        self._allow_shorting = allow_shorting

        # Load settings
        self._settings = load_settings()
        if self._settings.alpaca is None:
            raise ValueError("Alpaca credentials not configured")

        # Initialize components
        self._data_provider = AlpacaDataProvider(self._settings.alpaca, feed=feed)
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
        start = end - timedelta(days=90)  # 90 days lookback

        start_ts = Timestamp(int(start.timestamp() * 1_000_000_000))
        end_ts = Timestamp(int(end.timestamp() * 1_000_000_000))

        print(f"\nLoading historical data for {len(self._symbols)} symbols...")

        for i, symbol in enumerate(self._symbols, 1):
            sym = Symbol(symbol)
            cache_hit = False

            # Try cache first - check both minute and daily data
            for timeframe in [Timeframe.MINUTE_1, Timeframe.DAY_1]:
                try:
                    cached = list(load_bars(self._data_dir, sym, timeframe))
                    if len(cached) >= 60:
                        # Use most recent bars
                        self._bars[symbol] = cached[-200:]
                        cache_hit = True
                        break
                except Exception:
                    pass

            # If no cache hit, try to fetch daily data
            if symbol not in self._bars or len(self._bars[symbol]) < 60:
                try:
                    bars = list(self._data_provider.get_bars(
                        sym, start_ts, end_ts, Timeframe.DAY_1
                    ))
                    if len(bars) >= 20:
                        self._bars[symbol] = bars
                except Exception as e:
                    logger.warning(f"Failed to get history for {symbol}: {e}")

            if i % 5 == 0 or i <= 5:
                status = "cache" if cache_hit else "fetched"
                bars_count = len(self._bars.get(symbol, []))
                print(f"  Loaded {i}/{len(self._symbols)} ({symbol}: {bars_count} bars, {status})...", flush=True)

        loaded = sum(1 for bars in self._bars.values() if len(bars) >= 60)
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
        """Rebalance portfolio to target allocations with buying power awareness.

        Handles both long and short positions with proper margin management:
        1. First close positions (sell longs, cover shorts) - frees up capital
        2. Then open new positions (buy longs, sell shorts) - uses capital
        """
        current_positions = self._get_current_positions()

        # Categorize trades by type
        close_longs = []    # Sell shares we own (frees cash)
        cover_shorts = []   # Buy to cover short positions (uses cash, frees margin)
        open_longs = []     # Buy new long positions (uses cash)
        open_shorts = []    # Sell short (uses margin)

        for symbol in self._symbols:
            if symbol not in prices or symbol not in target_allocations:
                continue

            target_value = target_allocations[symbol] * account_value

            # If shorting not allowed, treat negative targets as 0 (flat)
            if not self._allow_shorting and target_value < 0:
                target_value = 0

            # Clip to max position limits (long and short separately)
            if target_value > 0:
                target_value = min(target_value, self._max_position)
            else:
                target_value = max(target_value, -self._max_short)

            target_shares = int(target_value / prices[symbol]) if prices[symbol] > 0 else 0

            current_shares, _ = current_positions.get(symbol, (0, 0.0))
            delta = target_shares - current_shares

            if abs(delta) < 1:
                continue

            trade = {
                "symbol": symbol,
                "current": current_shares,
                "target": target_shares,
                "delta": delta,
                "price": prices[symbol],
                "value": abs(delta) * prices[symbol],
            }

            if current_shares > 0:
                # Currently LONG
                if delta < 0:
                    # Reduce or close long position
                    sell_qty = min(abs(delta), current_shares)
                    trade["delta"] = -sell_qty
                    trade["value"] = sell_qty * prices[symbol]
                    close_longs.append(trade)

                    # If target is negative, we also need to open a short after closing
                    if target_shares < 0 and self._allow_shorting:
                        short_trade = trade.copy()
                        short_trade["delta"] = target_shares  # negative
                        short_trade["value"] = abs(target_shares) * prices[symbol]
                        open_shorts.append(short_trade)
                else:
                    # Increase long position
                    open_longs.append(trade)

            elif current_shares < 0:
                # Currently SHORT
                if delta > 0:
                    # Reduce or close short position (buy to cover)
                    cover_qty = min(delta, abs(current_shares))
                    trade["delta"] = cover_qty
                    trade["value"] = cover_qty * prices[symbol]
                    cover_shorts.append(trade)

                    # If target is positive, we also need to open a long after covering
                    if target_shares > 0:
                        long_trade = trade.copy()
                        long_trade["delta"] = target_shares
                        long_trade["value"] = target_shares * prices[symbol]
                        open_longs.append(long_trade)
                else:
                    # Increase short position
                    if self._allow_shorting:
                        open_shorts.append(trade)
            else:
                # Currently FLAT
                if delta > 0:
                    open_longs.append(trade)
                elif delta < 0 and self._allow_shorting:
                    open_shorts.append(trade)

        total_trades = len(close_longs) + len(cover_shorts) + len(open_longs) + len(open_shorts)
        if total_trades == 0:
            return

        print(f"\n  Rebalancing: {len(close_longs)} close longs, {len(cover_shorts)} cover shorts, "
              f"{len(open_longs)} open longs, {len(open_shorts)} open shorts")

        if not self._execute or not self._broker:
            # Simulate
            for trades in [close_longs, cover_shorts, open_longs, open_shorts]:
                for trade in trades:
                    self._positions[trade["symbol"]] = trade["target"]
                    self._position_values[trade["symbol"]] = trade["target"] * trade["price"]
            return

        from stockbot.core.models import Order
        from stockbot.core.types import OrderSide, OrderType

        # Step 1: Close positions first (frees up capital/margin)
        if close_longs:
            print("  Closing long positions...")
            for trade in sorted(close_longs, key=lambda x: x["value"], reverse=True):
                try:
                    order = Order(
                        symbol=Symbol(trade["symbol"]),
                        side=OrderSide.SELL,
                        quantity=Quantity(Decimal(abs(trade["delta"]))),
                        order_type=OrderType.MARKET,
                        created_at=Timestamp(int(time.time() * 1_000_000_000)),
                    )
                    self._broker.submit_order(order)
                    print(f"    SOLD {abs(trade['delta']):4} {trade['symbol']:6} @ ${trade['price']:.2f}")
                except Exception as e:
                    logger.error(f"Failed to close long {trade['symbol']}: {e}")

        if cover_shorts:
            print("  Covering short positions...")
            for trade in sorted(cover_shorts, key=lambda x: x["value"], reverse=True):
                try:
                    order = Order(
                        symbol=Symbol(trade["symbol"]),
                        side=OrderSide.BUY,
                        quantity=Quantity(Decimal(trade["delta"])),
                        order_type=OrderType.MARKET,
                        created_at=Timestamp(int(time.time() * 1_000_000_000)),
                    )
                    self._broker.submit_order(order)
                    print(f"    COVERED {trade['delta']:4} {trade['symbol']:6} @ ${trade['price']:.2f}")
                except Exception as e:
                    logger.error(f"Failed to cover short {trade['symbol']}: {e}")

        # Wait for closes to settle
        if (close_longs or cover_shorts) and (open_longs or open_shorts):
            print("  Waiting for closes to settle...")
            time.sleep(2)

        # Step 2: Get updated buying power
        try:
            buying_power = float(self._broker.get_buying_power())
        except Exception:
            buying_power = 0.0

        print(f"  Available buying power: ${buying_power:,.2f}")

        # If no buying power, skip opening new positions
        if buying_power < 100:
            if open_longs or open_shorts:
                print("  WARNING: No buying power available - skipping new positions")
                print("  (Close some positions to free up capital)")
            return

        # Step 3: Open new long positions
        if open_longs:
            print("  Opening long positions...")
            spent = 0.0
            for trade in sorted(open_longs, key=lambda x: x["value"], reverse=True):
                cost = trade["value"]

                if spent + cost > buying_power * 0.95:
                    affordable = int((buying_power * 0.95 - spent) / trade["price"])
                    if affordable < 1:
                        print(f"    SKIP {trade['symbol']:6} - insufficient buying power")
                        continue
                    trade["delta"] = affordable
                    cost = affordable * trade["price"]

                try:
                    order = Order(
                        symbol=Symbol(trade["symbol"]),
                        side=OrderSide.BUY,
                        quantity=Quantity(Decimal(trade["delta"])),
                        order_type=OrderType.MARKET,
                        created_at=Timestamp(int(time.time() * 1_000_000_000)),
                    )
                    self._broker.submit_order(order)
                    spent += cost
                    print(f"    BOUGHT {trade['delta']:4} {trade['symbol']:6} @ ${trade['price']:.2f} (${cost:.0f})")
                except Exception as e:
                    logger.error(f"Failed to buy {trade['symbol']}: {e}")

            if spent > 0:
                print(f"  Long positions cost: ${spent:,.2f}")
                buying_power -= spent

        # Step 4: Open new short positions (requires margin)
        if open_shorts and self._allow_shorting:
            print("  Opening short positions...")
            margin_used = 0.0
            # Shorts typically require 50% margin (Reg T) - be conservative
            margin_available = buying_power * 0.45

            for trade in sorted(open_shorts, key=lambda x: x["value"], reverse=True):
                margin_required = trade["value"] * 0.5  # 50% margin requirement

                if margin_used + margin_required > margin_available:
                    # Try smaller position
                    affordable = int((margin_available - margin_used) * 2 / trade["price"])
                    if affordable < 1:
                        print(f"    SKIP {trade['symbol']:6} - insufficient margin")
                        continue
                    trade["delta"] = -affordable
                    margin_required = affordable * trade["price"] * 0.5

                try:
                    order = Order(
                        symbol=Symbol(trade["symbol"]),
                        side=OrderSide.SELL,
                        quantity=Quantity(Decimal(abs(trade["delta"]))),
                        order_type=OrderType.MARKET,
                        created_at=Timestamp(int(time.time() * 1_000_000_000)),
                    )
                    self._broker.submit_order(order)
                    margin_used += margin_required
                    print(f"    SHORTED {abs(trade['delta']):4} {trade['symbol']:6} @ ${trade['price']:.2f} "
                          f"(margin: ${margin_required:.0f})")
                except Exception as e:
                    logger.error(f"Failed to short {trade['symbol']}: {e}")

            if margin_used > 0:
                print(f"  Margin used for shorts: ${margin_used:,.2f}")

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

    def _wait_for_market_open(self) -> bool:
        """Wait for market to open, sleeping efficiently.

        Returns:
            True if should continue, False if stopped
        """
        if not self._broker:
            return True

        clock = self._broker.get_market_clock()
        if clock["is_open"]:
            return True

        next_open = clock.get("next_open")
        if next_open is None:
            print("  Could not determine next market open. Waiting 1 hour...")
            for _ in range(60):  # Check every minute for 1 hour
                if not self._running:
                    return False
                time.sleep(60)
            return True

        # Calculate sleep time
        now = datetime.now(timezone.utc)
        if hasattr(next_open, 'tzinfo') and next_open.tzinfo is None:
            next_open = next_open.replace(tzinfo=timezone.utc)

        time_until_open = (next_open - now).total_seconds()

        if time_until_open <= 0:
            return True

        # Wake up 5 minutes early to prepare
        wake_early_seconds = 300
        sleep_seconds = max(0, time_until_open - wake_early_seconds)

        # Format for display
        hours, remainder = divmod(int(time_until_open), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\n{'=' * 70}")
        print("MARKET CLOSED")
        print(f"{'=' * 70}")
        print(f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Next open:    {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Time until open: {hours}h {minutes}m {seconds}s")
        print(f"Sleeping until: {(now + timedelta(seconds=sleep_seconds)).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"{'=' * 70}\n")

        # Sleep in chunks so we can respond to stop signals
        chunk_size = 60  # Check every minute
        remaining = sleep_seconds

        while remaining > 0 and self._running:
            sleep_time = min(chunk_size, remaining)
            time.sleep(sleep_time)
            remaining -= sleep_time

            # Show progress every 30 minutes
            if remaining > 0 and int(remaining) % 1800 == 0:
                hours, remainder = divmod(int(remaining), 3600)
                minutes, _ = divmod(remainder, 60)
                print(f"  ... {hours}h {minutes}m until wake up")

        if not self._running:
            return False

        print("\n" + "=" * 70)
        print("WAKING UP - Market opens soon!")
        print("=" * 70 + "\n")

        # Refresh historical data before market opens
        print("Refreshing historical data...")
        self._load_historical()

        return True

    def run(self, poll_interval: float = 60.0, run_24_7: bool = True) -> None:
        """Run live trading loop.

        Args:
            poll_interval: Seconds between updates when market is open
            run_24_7: If True, sleep during market close and resume when open
        """
        self._running = True
        self._setup_signals()
        error_count = 0
        max_errors = 10  # Max consecutive errors before longer backoff

        print("\n" + "=" * 70)
        print("LIVE PORTFOLIO TRADER")
        print("=" * 70)
        print(f"Symbols: {len(self._symbols)}")
        print(f"Capital: ${self._capital:,.0f}")
        print(f"Max Long: ${self._max_position:,.0f}")
        print(f"Max Short: ${self._max_short:,.0f}")
        print(f"Execute Trades: {self._execute}")
        print(f"Continue Learning: {self._learn}")
        print(f"Data Feed: {self._feed.upper()}")
        print(f"Allow Shorting: {self._allow_shorting}")
        print(f"24/7 Mode: {run_24_7}")
        print("=" * 70)

        self._load_historical()
        print("Historical data loaded, entering main loop...", flush=True)

        try:
            while self._running:
                # Check market status
                print("Checking market status...", flush=True)
                if self._broker:
                    print(f"  Calling is_market_open()...", flush=True)
                    if not self._broker.is_market_open():
                        if run_24_7:
                            if not self._wait_for_market_open():
                                break
                        else:
                            print("\nMarket closed. Exiting (use --24-7 to wait).\n")
                            break

                # Process tick
                try:
                    self._process_tick()
                    error_count = 0  # Reset on success
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error in tick ({error_count}/{max_errors}): {e}")

                    if error_count >= max_errors:
                        print(f"\n{max_errors} consecutive errors. Backing off for 5 minutes...")
                        time.sleep(300)
                        error_count = 0

                # Sleep until next tick
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
    python scripts/live_portfolio.py                    # Monitor 100 stocks (learning ON)
    python scripts/live_portfolio.py --universe-size 25 # Monitor 25 stocks
    python scripts/live_portfolio.py --execute          # Execute trades
    python scripts/live_portfolio.py --no-learn         # Disable live learning
    python scripts/live_portfolio.py --feed sip         # Use paid SIP data feed
        """,
    )

    parser.add_argument("--universe-size", type=int, default=100,
                        choices=[10, 25, 50, 100],
                        help="Number of stocks (default: 100)")
    parser.add_argument("--agent", type=Path, default=Path("./data/portfolio_agent.json"))
    parser.add_argument("--execute", action="store_true",
                        help="Execute trades (paper account)")
    parser.add_argument("--no-learn", dest="train_live", action="store_false",
                        help="Disable live learning (learning is ON by default)")
    parser.add_argument("--capital", type=float, default=100_000.0,
                        help="Trading capital (default: 100000)")
    parser.add_argument("--max-position", type=float, default=20_000.0,
                        help="Max $ per long position (default: 20000)")
    parser.add_argument("--max-short", type=float, default=20_000.0,
                        help="Max $ per short position (default: 20000)")
    parser.add_argument("--poll-interval", type=float, default=60.0,
                        help="Seconds between updates (default: 60)")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/portfolio"))
    parser.add_argument("--feed", type=str, default="iex",
                        choices=["iex", "sip"],
                        help="Data feed: iex (free, default) or sip (paid)")
    parser.add_argument("--allow-shorting", action="store_true",
                        help="Allow short selling (requires margin account)")
    parser.add_argument("--24-7", dest="run_24_7", action="store_true",
                        help="Run 24/7, sleeping when market is closed")
    parser.add_argument("--log-level", type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.set_defaults(train_live=True)  # Learning ON by default

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)

    symbols = get_universe(args.universe_size)

    # Check if paper or live account
    settings = load_settings()
    is_paper = settings.alpaca.paper if settings.alpaca else True

    if args.execute:
        if is_paper:
            print("\n" + "-" * 70)
            print("Executing trades on PAPER account.")
            print("-" * 70 + "\n")
        else:
            print("\n" + "!" * 70)
            print("WARNING: Execute mode on LIVE account - real money at risk!")
            print("!" * 70)
            if input("\nType 'yes' to confirm: ").lower() != 'yes':
                print("Aborted.")
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
            max_short_value=args.max_short,
            feed=args.feed,
            allow_shorting=args.allow_shorting,
        )
        trader.run(poll_interval=args.poll_interval, run_24_7=args.run_24_7)

    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
