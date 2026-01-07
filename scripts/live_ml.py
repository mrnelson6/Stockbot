#!/usr/bin/env python3
"""Live ML agent that monitors prices, learns, and trades.

The agent decides:
1. Direction: long, short, or flat
2. Size: how much of available capital to deploy

Usage:
    python scripts/live_ml.py SPY
    python scripts/live_ml.py SPY --execute  # Actually trade
    python scripts/live_ml.py SPY --train-live  # Keep learning
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
from stockbot.core.models import Bar
from stockbot.core.types import Price, Quantity, Symbol, Timeframe, Timestamp
from stockbot.data.providers.alpaca import AlpacaDataProvider
from stockbot.data.storage import load_bars
from stockbot.execution.broker_alpaca import AlpacaBroker
from stockbot.learning.features import FeatureExtractor, MarketFeatures
from stockbot.learning.rl_agent import POSITION_LEVELS, TradingAgent
from stockbot.monitoring import setup_logging
from stockbot.monitoring.logger import get_logger

logger = get_logger("live_ml")


class LiveMLTrader:
    """Live ML trading agent with position sizing."""

    def __init__(
        self,
        symbol: Symbol,
        agent_path: Path,
        data_dir: Path,
        capital: float = 10000.0,
        execute_trades: bool = False,
        continue_learning: bool = True,
    ) -> None:
        self._symbol = symbol
        self._agent_path = agent_path
        self._data_dir = data_dir
        self._capital = capital
        self._execute = execute_trades
        self._learn = continue_learning

        # Load settings
        self._settings = load_settings()
        if self._settings.alpaca is None:
            raise ValueError("Alpaca credentials not configured")

        # Initialize components
        self._data_provider = AlpacaDataProvider(self._settings.alpaca)
        self._broker = AlpacaBroker(self._settings.alpaca) if execute_trades else None

        # Feature extractor (must match training)
        self._feature_extractor = FeatureExtractor(
            lookback_periods=[5, 10, 20, 50],
            include_volume=True,
            include_volatility=True,
            include_momentum=True,
            include_mean_reversion=True,
        )

        # Load agent
        self._agent = self._load_or_create_agent()

        # State
        self._running = False
        self._bars: list[Bar] = []
        self._current_shares = 0
        self._current_position_value = 0.0
        self._last_features: Optional[MarketFeatures] = None
        self._last_action_idx: Optional[int] = None
        self._last_target_fraction = 0.0

        # Performance
        self._session_pnl = 0.0
        self._decisions: list[dict] = []

    def _load_or_create_agent(self) -> TradingAgent:
        """Load trained agent or create new."""
        agent = TradingAgent(
            feature_extractor=self._feature_extractor,
            hidden_sizes=[128, 64],
            learning_rate=0.0005,  # Lower LR for live
            gamma=0.95,
            epsilon_start=0.1 if self._learn else 0.0,
            epsilon_end=0.02,
            epsilon_decay=0.9995,
            batch_size=32,
        )

        if self._agent_path.exists():
            agent.load(self._agent_path)
            print(f"Loaded agent from {self._agent_path}")
            self._show_agent_info(agent)
        else:
            print("No trained agent found - starting fresh")
            print("Train first: python scripts/train_ml.py SPY")

        return agent

    def _show_agent_info(self, agent: TradingAgent) -> None:
        """Show what the agent learned."""
        stats = agent.get_stats()

        print(f"\nAgent Statistics:")
        print(f"  Training steps: {stats['steps']:,}")
        print(f"  Win rate: {stats['win_rate']:.1%}")
        print(f"  Total P&L: {stats['total_pnl_pct']:.2f}%")

        prefs = agent.get_position_preferences()
        if prefs:
            print(f"\nPosition preferences:")
            for pos, pct in sorted(prefs.items(), key=lambda x: float(x[0].replace('%', ''))):
                if pct > 5:
                    print(f"    {pos}: {pct:.0f}%")

        print(f"\nTop features:")
        for name, score in agent.get_feature_importance()[:5]:
            print(f"    {name}: {score:.4f}")

    def _load_historical(self) -> None:
        """Load historical bars for features."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=90)

        start_ts = Timestamp(int(start.timestamp() * 1_000_000_000))
        end_ts = Timestamp(int(end.timestamp() * 1_000_000_000))

        # Try cache
        try:
            cached = list(load_bars(self._data_dir, self._symbol, Timeframe.DAY_1))
            recent = [b for b in cached if b.timestamp >= start_ts]
            if len(recent) >= 60:
                self._bars = recent
                print(f"Loaded {len(self._bars)} bars from cache")
                return
        except Exception:
            pass

        # Fetch
        print("Fetching historical data...")
        bars = list(self._data_provider.get_bars(
            self._symbol, start_ts, end_ts, Timeframe.DAY_1
        ))
        self._bars = bars
        print(f"Loaded {len(self._bars)} bars")

    def _get_current_bar(self) -> Optional[Bar]:
        """Get latest bar."""
        try:
            return self._data_provider.get_latest(self._symbol)
        except Exception as e:
            logger.warning(f"Failed to get bar: {e}")
            return None

    def _get_account_value(self) -> float:
        """Get current account value."""
        if self._broker:
            try:
                return float(self._broker.get_equity())
            except Exception:
                pass
        return self._capital

    def _get_current_position(self) -> tuple[int, float]:
        """Get current shares and value."""
        if not self._broker:
            return self._current_shares, self._current_position_value

        try:
            positions = self._broker.get_positions()
            if self._symbol in positions:
                pos = positions[self._symbol]
                shares = int(pos.quantity)
                value = float(pos.quantity * pos.market_value / pos.quantity) if pos.quantity else 0
                return shares, value
            return 0, 0.0
        except Exception as e:
            logger.warning(f"Failed to get position: {e}")
            return self._current_shares, self._current_position_value

    def _execute_rebalance(self, target_shares: int, price: float) -> None:
        """Rebalance to target position."""
        current_shares, _ = self._get_current_position()
        delta = target_shares - current_shares

        if abs(delta) < 1:
            return

        if not self._execute or not self._broker:
            if delta > 0:
                print(f"  >> RECOMMEND: BUY {delta} shares @ ${price:.2f}")
            else:
                print(f"  >> RECOMMEND: SELL {abs(delta)} shares @ ${price:.2f}")
            self._current_shares = target_shares
            self._current_position_value = target_shares * price
            return

        # Execute
        try:
            from stockbot.core.models import Order
            from stockbot.core.types import OrderSide, OrderType

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            qty = abs(delta)

            order = Order(
                symbol=self._symbol,
                side=side,
                quantity=Quantity(Decimal(qty)),
                order_type=OrderType.MARKET,
                timestamp=Timestamp(int(time.time() * 1_000_000_000)),
            )

            self._broker.submit_order(order)
            action = "BOUGHT" if delta > 0 else "SOLD"
            print(f"  >> EXECUTED: {action} {qty} shares @ ~${price:.2f}")

        except Exception as e:
            logger.error(f"Failed to execute: {e}")

    def _process_bar(self, bar: Bar) -> None:
        """Process new bar and decide position."""
        # Skip if already processed
        if self._bars and bar.timestamp <= self._bars[-1].timestamp:
            return

        self._bars.append(bar)
        if len(self._bars) > 200:
            self._bars = self._bars[-200:]

        # Extract features
        features = self._feature_extractor.extract(self._bars)
        if features is None:
            return

        # Get account value
        account_value = self._get_account_value()

        # Agent decides position size
        position_dollars, info = self._agent.get_position_size(
            features, account_value, training=self._learn
        )

        target_fraction = info["target_fraction"]
        q_values = info["q_values"]

        # Calculate target shares
        if target_fraction != 0:
            target_shares = int(position_dollars / features.price)
        else:
            target_shares = 0

        # Current state
        current_shares, current_value = self._get_current_position()
        current_fraction = current_value / account_value if account_value > 0 else 0

        # Display
        timestamp = datetime.fromtimestamp(bar.timestamp / 1_000_000_000, tz=timezone.utc)

        print(f"\n[{timestamp.strftime('%Y-%m-%d %H:%M')}] {self._symbol} @ ${features.price:.2f}")
        print(f"  Account: ${account_value:,.0f}")
        print(f"  Current: {current_shares} shares ({current_fraction:.0%})")
        print(f"  Target:  {target_shares} shares ({target_fraction:.0%})")

        # Show Q-values
        q_str = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(q_values.items(),
                         key=lambda x: float(x[0].replace('%', ''))))
        print(f"  Q-values: {q_str}")

        # Record
        self._decisions.append({
            "time": timestamp.isoformat(),
            "price": features.price,
            "current_fraction": current_fraction,
            "target_fraction": target_fraction,
            "q_values": q_values,
        })

        # Execute rebalance if needed
        if abs(target_shares - current_shares) >= 1:
            self._execute_rebalance(target_shares, features.price)

        # Learn from previous decision
        if self._learn and self._last_features is not None and self._last_action_idx is not None:
            reward = self._agent.observe_result(
                self._last_features,
                self._last_action_idx,
                features,
                position_held=self._last_target_fraction,
                done=False,
            )
            if abs(reward) > 0.1:
                print(f"  Learning: reward={reward:.3f}")

        self._last_features = features
        self._last_action_idx = info["action_idx"]
        self._last_target_fraction = target_fraction

    def run(self, poll_interval: float = 60.0) -> None:
        """Run live trading loop."""
        self._running = True
        self._setup_signals()

        print(f"\n{'=' * 60}")
        print("LIVE ML TRADER")
        print(f"{'=' * 60}")
        print(f"Symbol: {self._symbol}")
        print(f"Capital: ${self._capital:,.0f}")
        print(f"Execute: {self._execute}")
        print(f"Learning: {self._learn}")
        print(f"Position Levels: {[f'{int(p*100)}%' for p in POSITION_LEVELS]}")
        print(f"{'=' * 60}")

        self._load_historical()

        if self._broker and not self._broker.is_market_open():
            print("\nMarket closed. Will act when open.\n")

        print("\nMonitoring... (Ctrl+C to stop)\n")

        try:
            while self._running:
                try:
                    bar = self._get_current_bar()
                    if bar:
                        self._process_bar(bar)
                except Exception as e:
                    logger.error(f"Error: {e}")

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
        print(f"\n{'=' * 60}")
        print("SESSION SUMMARY")
        print(f"{'=' * 60}")

        print(f"Decisions: {len(self._decisions)}")

        if self._decisions:
            # Position distribution
            fractions = [d["target_fraction"] for d in self._decisions]
            print(f"Avg position: {np.mean(fractions):.0%}")
            print(f"Max position: {max(fractions):.0%}")
            print(f"Min position: {min(fractions):.0%}")

        if self._learn:
            stats = self._agent.get_stats()
            print(f"\nAgent updated:")
            print(f"  Steps: {stats['steps']}")
            print(f"  Epsilon: {stats['epsilon']:.4f}")

            prefs = self._agent.get_position_preferences()
            if prefs:
                print(f"\nCurrent position preferences:")
                for pos, pct in sorted(prefs.items(), key=lambda x: float(x[0].replace('%', ''))):
                    if pct > 3:
                        print(f"    {pos}: {pct:.0f}%")

        print(f"{'=' * 60}\n")

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
        description="Live ML trading agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/live_ml.py SPY                    # Monitor only
    python scripts/live_ml.py SPY --execute          # Execute trades
    python scripts/live_ml.py SPY --train-live       # Keep learning
    python scripts/live_ml.py SPY --capital 50000    # Set capital
        """,
    )

    parser.add_argument("symbol", type=str, default="SPY", nargs="?")
    parser.add_argument("--agent", type=Path, default=Path("./data/ml_agent.json"))
    parser.add_argument("--execute", action="store_true",
                        help="Execute trades (paper account)")
    parser.add_argument("--train-live", action="store_true",
                        help="Continue learning")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Trading capital (default: 10000)")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--log-level", type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)

    symbol = Symbol(args.symbol.upper())

    if args.execute:
        print("\n" + "!" * 60)
        print("WARNING: Execute mode - will place real orders!")
        print("!" * 60)
        if input("\nContinue? [y/N]: ").lower() != 'y':
            return 0

    try:
        trader = LiveMLTrader(
            symbol=symbol,
            agent_path=args.agent,
            data_dir=args.data_dir,
            capital=args.capital,
            execute_trades=args.execute,
            continue_learning=args.train_live,
        )
        trader.run(poll_interval=args.poll_interval)

    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
