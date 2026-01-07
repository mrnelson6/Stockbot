#!/usr/bin/env python3
"""Train portfolio agent on 100 stocks with minute-level data.

Downloads 2 years of 1-minute data and trains the multi-asset
portfolio agent to learn optimal allocations.

Usage:
    python scripts/train_portfolio.py
    python scripts/train_portfolio.py --universe-size 25  # Smaller universe
    python scripts/train_portfolio.py --training-months 6  # Less history
"""

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from stockbot.config.settings import load_settings
from stockbot.config.universe import get_universe
from stockbot.core.models import Bar
from stockbot.core.types import Symbol, Timeframe, Timestamp
from stockbot.data.providers.alpaca import AlpacaDataProvider
from stockbot.data.storage import load_bars, save_bars
from stockbot.learning.features import FeatureExtractor, MarketFeatures
from stockbot.learning.portfolio_agent import PortfolioAgent
from stockbot.monitoring import setup_logging
from stockbot.monitoring.logger import get_logger

logger = get_logger("train_portfolio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train portfolio agent on multiple stocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/train_portfolio.py                     # Full 100 stocks, 2 years
    python scripts/train_portfolio.py --universe-size 25  # 25 stocks
    python scripts/train_portfolio.py --training-months 6 # 6 months
    python scripts/train_portfolio.py --timeframe 1H      # Hourly data
        """,
    )

    parser.add_argument("--universe-size", type=int, default=100,
                        choices=[10, 25, 50, 100],
                        help="Number of stocks (default: 100)")
    parser.add_argument("--training-months", type=int, default=24,
                        help="Months of historical data (default: 24)")
    parser.add_argument("--timeframe", type=str, default="1Min",
                        choices=["1Min", "5Min", "15Min", "1H", "1D"],
                        help="Data timeframe (default: 1Min)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--batch-days", type=int, default=5,
                        help="Days per training batch (default: 5)")
    parser.add_argument("--hidden-sizes", type=str, default="256,128,64",
                        help="Hidden layer sizes (default: 256,128,64)")
    parser.add_argument("--learning-rate", type=float, default=0.0005,
                        help="Learning rate (default: 0.0005)")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/portfolio"))
    parser.add_argument("--output", type=Path, default=Path("./data/portfolio_agent.json"))
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fetch", action="store_true",
                        help="Don't fetch new data")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing agent")

    return parser.parse_args()


def timeframe_to_enum(tf_str: str) -> Timeframe:
    """Convert string timeframe to enum."""
    mapping = {
        "1Min": Timeframe.MINUTE_1,
        "5Min": Timeframe.MINUTE_5,
        "15Min": Timeframe.MINUTE_15,
        "1H": Timeframe.HOUR_1,
        "1D": Timeframe.DAY_1,
    }
    return mapping.get(tf_str, Timeframe.MINUTE_1)


def fetch_symbol_data(
    provider: AlpacaDataProvider,
    symbol: Symbol,
    start_ts: Timestamp,
    end_ts: Timestamp,
    timeframe: Timeframe,
    data_dir: Path,
    force_fetch: bool = False,
) -> list[Bar]:
    """Fetch data for a single symbol, using cache if available."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Try cache first
    if not force_fetch:
        try:
            cached = list(load_bars(data_dir, symbol, timeframe))
            if cached:
                # Check if we have enough coverage
                buffer_ns = 7 * 24 * 60 * 60 * 1_000_000_000  # 7 days buffer
                if (cached[0].timestamp <= start_ts + buffer_ns and
                    cached[-1].timestamp >= end_ts - buffer_ns):
                    logger.debug(f"Using cached data for {symbol}: {len(cached)} bars")
                    return cached
        except Exception:
            pass

    # Fetch from API
    logger.info(f"Fetching {symbol}...")
    try:
        bars = list(provider.get_bars(symbol, start_ts, end_ts, timeframe))
        if bars:
            save_bars(bars, data_dir, symbol, timeframe, append=False)
            logger.info(f"  Downloaded {len(bars):,} bars for {symbol}")
        return bars
    except Exception as e:
        logger.warning(f"  Failed to fetch {symbol}: {e}")
        return []


def fetch_all_data(
    symbols: list[str],
    start_ts: Timestamp,
    end_ts: Timestamp,
    timeframe: Timeframe,
    data_dir: Path,
    fetch_missing: bool = True,
) -> dict[str, list[Bar]]:
    """Fetch data for all symbols."""
    settings = load_settings()
    if settings.alpaca is None:
        raise ValueError("Alpaca credentials not configured")

    provider = AlpacaDataProvider(settings.alpaca)
    all_data: dict[str, list[Bar]] = {}

    print(f"\nFetching data for {len(symbols)} symbols...")
    print(f"This may take a while for minute-level data.\n")

    for i, symbol in enumerate(symbols, 1):
        sym = Symbol(symbol)
        print(f"[{i}/{len(symbols)}] {symbol}...", end=" ", flush=True)

        bars = fetch_symbol_data(
            provider, sym, start_ts, end_ts, timeframe, data_dir,
            force_fetch=False
        )

        if bars:
            all_data[symbol] = bars
            print(f"{len(bars):,} bars")
        else:
            print("SKIPPED (no data)")

        # Rate limiting
        if fetch_missing and i % 10 == 0:
            time.sleep(1)  # Be nice to the API

    print(f"\nLoaded data for {len(all_data)}/{len(symbols)} symbols")
    return all_data


def align_timestamps(all_data: dict[str, list[Bar]]) -> list[int]:
    """Find common timestamps across all symbols."""
    if not all_data:
        return []

    # Get timestamps for each symbol
    timestamp_sets = [
        set(bar.timestamp for bar in bars)
        for bars in all_data.values()
    ]

    # Find intersection (timestamps where ALL symbols have data)
    common = timestamp_sets[0]
    for ts_set in timestamp_sets[1:]:
        common &= ts_set

    return sorted(common)


def build_features_at_timestamp(
    all_data: dict[str, list[Bar]],
    feature_extractor: FeatureExtractor,
    timestamp: int,
    lookback: int = 60,
) -> Optional[dict[str, MarketFeatures]]:
    """Build features for all symbols at a specific timestamp."""
    features_dict: dict[str, MarketFeatures] = {}

    for symbol, bars in all_data.items():
        # Find index of this timestamp
        idx = None
        for i, bar in enumerate(bars):
            if bar.timestamp == timestamp:
                idx = i
                break

        if idx is None or idx < lookback:
            continue

        # Get lookback window
        window = bars[max(0, idx - lookback):idx + 1]
        features = feature_extractor.extract(window)

        if features is not None:
            features_dict[symbol] = features

    return features_dict if features_dict else None


def train_epoch(
    agent: PortfolioAgent,
    all_data: dict[str, list[Bar]],
    feature_extractor: FeatureExtractor,
    common_timestamps: list[int],
    epoch: int,
    batch_size: int = 1000,
    capital: float = 1_000_000.0,
) -> dict:
    """Train one epoch over the data."""
    agent.reset()

    total_reward = 0.0
    steps = 0
    prev_features_dict: Optional[dict[str, MarketFeatures]] = None
    prev_allocations: Optional[dict[str, float]] = None

    # Process in batches to manage memory
    n_batches = (len(common_timestamps) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(common_timestamps))
        batch_timestamps = common_timestamps[start_idx:end_idx]

        for ts in batch_timestamps:
            features_dict = build_features_at_timestamp(
                all_data, feature_extractor, ts, lookback=60
            )

            if features_dict is None or len(features_dict) < len(agent.symbols) // 2:
                continue

            # Get allocations
            allocations, info = agent.get_allocations(
                features_dict, capital, training=True
            )

            # Learn from previous step
            if prev_features_dict is not None and prev_allocations is not None:
                reward = agent.observe_result(
                    features_dict,
                    prev_features_dict,
                    prev_allocations,
                    capital,
                    done=False,
                )
                total_reward += reward

            prev_features_dict = features_dict
            prev_allocations = allocations
            steps += 1

        # Progress
        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch {batch_idx + 1}/{n_batches}, steps={steps}, reward={total_reward:.2f}")

    return {
        "epoch": epoch,
        "steps": steps,
        "total_reward": total_reward,
        "avg_reward": total_reward / steps if steps > 0 else 0,
    }


def print_results(agent: PortfolioAgent, epochs_data: list[dict]) -> None:
    """Print training results."""
    stats = agent.get_stats()

    print("\n" + "=" * 70)
    print("PORTFOLIO TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nTraining Statistics:")
    print(f"  Total Steps: {stats['steps']:,}")
    print(f"  Epsilon: {stats['epsilon']:.4f}")
    print(f"  Buffer Size: {stats['buffer_size']:,}")
    print(f"  Average Loss: {stats['avg_loss']:.6f}")

    print(f"\nPortfolio Statistics:")
    print(f"  Total P&L: {stats['total_pnl_pct']:.2f}%")
    print(f"  Gross Exposure: {stats['gross_exposure']:.1%}")
    print(f"  Net Exposure: {stats['net_exposure']:.1%}")
    print(f"  Active Positions: {stats['n_positions']}")

    print(f"\nTop Positions:")
    for symbol, weight in agent.get_top_positions(10):
        if abs(weight) > 0.01:
            direction = "LONG" if weight > 0 else "SHORT"
            print(f"  {symbol:6} {direction:5} {abs(weight):.1%}")

    # Learning curve
    if epochs_data:
        print(f"\nLearning Curve:")
        for ed in epochs_data:
            bar = "#" * int(ed["avg_reward"] * 10) if ed["avg_reward"] > 0 else ""
            print(f"  Epoch {ed['epoch']}: avg_reward={ed['avg_reward']:.4f} {bar}")

    print("=" * 70 + "\n")


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)
    np.random.seed(args.seed)

    # Get universe
    symbols = get_universe(args.universe_size)
    timeframe = timeframe_to_enum(args.timeframe)

    # Calculate date range
    end_date = datetime.now(timezone.utc) - timedelta(days=1)
    start_date = end_date - timedelta(days=args.training_months * 30)

    start_ts = Timestamp(int(start_date.timestamp() * 1_000_000_000))
    end_ts = Timestamp(int(end_date.timestamp() * 1_000_000_000))

    print("\n" + "=" * 70)
    print("PORTFOLIO AGENT TRAINING")
    print("=" * 70)
    print(f"Universe: {args.universe_size} stocks")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Epochs: {args.epochs}")
    print(f"Data Directory: {args.data_dir}")
    print("=" * 70)

    # Fetch all data
    all_data = fetch_all_data(
        symbols, start_ts, end_ts, timeframe, args.data_dir,
        fetch_missing=not args.no_fetch
    )

    if len(all_data) < 5:
        print("ERROR: Not enough symbols with data")
        return 1

    # Filter to symbols with data
    active_symbols = list(all_data.keys())
    print(f"\nTraining on {len(active_symbols)} symbols")

    # Find common timestamps
    print("Aligning timestamps across symbols...")
    common_timestamps = align_timestamps(all_data)
    print(f"Found {len(common_timestamps):,} common timestamps")

    if len(common_timestamps) < 1000:
        print("WARNING: Very few common timestamps. Consider using daily data.")

    # Create feature extractor
    feature_extractor = FeatureExtractor(
        lookback_periods=[5, 10, 20, 50],
        include_volume=True,
        include_volatility=True,
        include_momentum=True,
        include_mean_reversion=True,
    )

    print(f"Feature space: {feature_extractor.feature_count} features per asset")
    print(f"Total input size: {len(active_symbols) * feature_extractor.feature_count + 10}")

    # Create agent
    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    agent = PortfolioAgent(
        symbols=active_symbols,
        feature_extractor=feature_extractor,
        hidden_sizes=hidden_sizes,
        learning_rate=args.learning_rate,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        batch_size=64,
        max_position_per_asset=0.20,  # Max 20% per asset
        seed=args.seed,
    )

    # Resume if requested
    if args.resume and args.output.exists():
        agent.load(args.output)
        print(f"Resumed from {args.output}")

    # Training
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70 + "\n")

    epochs_data = []
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        result = train_epoch(
            agent, all_data, feature_extractor,
            common_timestamps, epoch,
            batch_size=args.batch_days * 390,  # ~390 minutes per day
            capital=1_000_000.0,
        )
        epochs_data.append(result)

        stats = agent.get_stats()
        print(f"  Result: steps={result['steps']}, reward={result['avg_reward']:.4f}, "
              f"epsilon={stats['epsilon']:.3f}, positions={stats['n_positions']}")

        # Save checkpoint
        agent.save(args.output)

    # Final results
    print_results(agent, epochs_data)

    print(f"Agent saved to: {args.output}")
    print(f"\nTo run live: python scripts/live_portfolio.py")

    return 0


if __name__ == "__main__":
    exit(main())
