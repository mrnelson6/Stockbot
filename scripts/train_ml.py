#!/usr/bin/env python3
"""Train a machine learning agent on market data.

The agent learns:
1. Which market features are predictive
2. When to be long, short, or flat
3. How much to allocate (position sizing)

Usage:
    python scripts/train_ml.py SPY
    python scripts/train_ml.py SPY --training-months 24 --epochs 10
"""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from stockbot.config.settings import load_settings
from stockbot.core.models import Bar
from stockbot.core.types import Symbol, Timeframe, Timestamp
from stockbot.data.providers.alpaca import AlpacaDataProvider
from stockbot.data.storage import load_bars, save_bars
from stockbot.learning.features import FeatureExtractor
from stockbot.learning.rl_agent import POSITION_LEVELS, TradingAgent
from stockbot.monitoring import setup_logging
from stockbot.monitoring.logger import get_logger

logger = get_logger("train_ml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ML trading agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/train_ml.py SPY
    python scripts/train_ml.py SPY --training-months 24
    python scripts/train_ml.py SPY --epochs 20
        """,
    )

    parser.add_argument("symbol", type=str, default="SPY", nargs="?",
                        help="Symbol to train on (default: SPY)")
    parser.add_argument("--training-months", type=int, default=12,
                        help="Months of historical data (default: 12)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs (default: 10)")
    parser.add_argument("--hidden-sizes", type=str, default="128,64",
                        help="Hidden layer sizes (default: 128,64)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output", type=Path, default=Path("./data/ml_agent.json"))
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fetch", action="store_true")

    return parser.parse_args()


def calculate_date_range(training_months: int) -> tuple[str, str]:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=training_months * 30)
    return start.isoformat(), end.isoformat()


def ensure_data(symbol: Symbol, start_date: str, end_date: str,
                data_dir: Path, fetch_missing: bool = True) -> list[Bar]:
    """Ensure data is available."""
    data_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    start_ts = Timestamp(int(start_dt.timestamp() * 1_000_000_000))
    end_ts = Timestamp(int(end_dt.timestamp() * 1_000_000_000))

    # Try local first
    try:
        bars = list(load_bars(data_dir, symbol, Timeframe.DAY_1))
        if bars:
            buffer_ns = 7 * 24 * 60 * 60 * 1_000_000_000
            if bars[0].timestamp <= start_ts + buffer_ns and bars[-1].timestamp >= end_ts - buffer_ns:
                logger.info(f"Loaded {len(bars)} existing bars for {symbol}")
                return bars
    except Exception:
        pass

    if not fetch_missing:
        raise ValueError(f"No data for {symbol} and --no-fetch specified")

    # Fetch
    logger.info(f"Fetching {symbol} data...")
    settings = load_settings()
    if settings.alpaca is None:
        raise ValueError("Alpaca credentials not configured")

    provider = AlpacaDataProvider(settings.alpaca)
    bars = list(provider.get_bars(symbol, start_ts, end_ts, Timeframe.DAY_1))

    if not bars:
        raise ValueError(f"No data returned for {symbol}")

    save_bars(bars, data_dir, symbol, Timeframe.DAY_1, append=False)
    logger.info(f"Downloaded {len(bars)} bars")

    return bars


def train_epoch(agent: TradingAgent, bars: list[Bar],
                feature_extractor: FeatureExtractor, epoch: int,
                capital: float = 100000.0) -> dict:
    """Run one training epoch."""
    agent.reset_episode()

    min_bars = 60
    total_reward = 0.0
    steps = 0

    prev_features = None
    prev_action_idx = None

    for i in range(min_bars, len(bars)):
        historical = bars[:i+1]
        features = feature_extractor.extract(historical)
        if features is None:
            continue

        # Get agent's position decision
        position_dollars, info = agent.get_position_size(features, capital, training=True)
        target_fraction = info["target_fraction"]
        action_idx = info["action_idx"]

        # Learn from previous step
        if prev_features is not None and prev_action_idx is not None:
            done = (i == len(bars) - 1)
            reward = agent.observe_result(
                prev_features, prev_action_idx, features,
                position_held=prev_target_fraction, done=done
            )
            total_reward += reward

        prev_features = features
        prev_action_idx = action_idx
        prev_target_fraction = target_fraction
        steps += 1

    return {
        "epoch": epoch,
        "steps": steps,
        "total_reward": total_reward,
        "avg_reward": total_reward / steps if steps > 0 else 0,
    }


def print_results(agent: TradingAgent, epochs_data: list[dict]) -> None:
    """Print training results."""
    stats = agent.get_stats()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nTraining Statistics:")
    print(f"  Total Steps: {stats['steps']:,}")
    print(f"  Final Epsilon: {stats['epsilon']:.4f}")
    print(f"  Experience Buffer: {stats['buffer_size']:,}")

    print(f"\nTrading Performance:")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Total P&L: {stats['total_pnl_pct']:.2f}%")
    print(f"  Avg Position: {stats['avg_position']:.1%}")

    # Position preferences
    prefs = agent.get_position_preferences()
    if prefs:
        print(f"\nLearned Position Preferences:")
        for pos, pct in sorted(prefs.items(), key=lambda x: float(x[0].replace('%', ''))):
            bar = "#" * int(pct / 2)
            print(f"  {pos:>5}: {pct:5.1f}% {bar}")

    # Feature importance
    print(f"\nTop Learned Features:")
    for name, score in agent.get_feature_importance()[:10]:
        bar = "#" * int(score * 100)
        print(f"  {name:<30} {score:.4f} {bar}")

    print("=" * 60 + "\n")


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    np.random.seed(args.seed)

    symbol = Symbol(args.symbol.upper())
    start_date, end_date = calculate_date_range(args.training_months)

    print(f"\n{'=' * 60}")
    print("ML AGENT TRAINING")
    print(f"{'=' * 60}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Epochs: {args.epochs}")
    print(f"Position Levels: {[f'{int(p*100)}%' for p in POSITION_LEVELS]}")
    print(f"{'=' * 60}\n")

    # Get data
    try:
        bars = ensure_data(symbol, start_date, end_date, args.data_dir,
                          fetch_missing=not args.no_fetch)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    print(f"Training on {len(bars)} daily bars\n")

    # Create feature extractor
    feature_extractor = FeatureExtractor(
        lookback_periods=[5, 10, 20, 50],
        include_volume=True,
        include_volatility=True,
        include_momentum=True,
        include_mean_reversion=True,
    )

    print(f"Feature space: {feature_extractor.feature_count} features")
    print("Agent will learn which features matter and optimal position sizes.\n")

    # Create agent
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    agent = TradingAgent(
        feature_extractor=feature_extractor,
        hidden_sizes=hidden_sizes,
        learning_rate=args.learning_rate,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=32,
        seed=args.seed,
    )

    # Train
    print("Training...\n")
    epochs_data = []

    for epoch in range(1, args.epochs + 1):
        result = train_epoch(agent, bars, feature_extractor, epoch)
        epochs_data.append(result)

        stats = agent.get_stats()
        print(
            f"Epoch {epoch:>2}/{args.epochs}: "
            f"reward={result['avg_reward']:>7.3f}, "
            f"epsilon={stats['epsilon']:.3f}, "
            f"trades={stats['total_trades']:>3}, "
            f"win_rate={stats['win_rate']:.0%}, "
            f"avg_pos={stats['avg_position']:>5.0%}"
        )

    # Save
    agent.save(args.output)
    print(f"\nAgent saved to: {args.output}")

    print_results(agent, epochs_data)

    print(f"To run live: python scripts/live_ml.py {symbol}")

    return 0


if __name__ == "__main__":
    exit(main())
