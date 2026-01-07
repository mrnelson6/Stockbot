#!/usr/bin/env python3
"""Train strategy selectors using historical backtests.

This script automatically fetches required data and runs walk-forward
training where the selector learns which strategies perform best.

Usage:
    python scripts/train_selector.py AAPL MSFT GOOGL
    python scripts/train_selector.py AAPL --training-months 12 --selector thompson
"""

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from stockbot.config.settings import BacktestConfig, RiskConfig, load_settings
from stockbot.core.interfaces import Strategy
from stockbot.core.types import Price, Quantity, Symbol, Timeframe, Timestamp
from stockbot.data.providers.alpaca import AlpacaDataProvider
from stockbot.data.storage import save_bars, load_bars
from stockbot.engine.backtest import BacktestEngine
from stockbot.learning.callbacks import SelectorCallback
from stockbot.learning.selector import (
    EnsembleStrategy,
    EpsilonGreedySelector,
    StrategySelector,
    ThompsonSamplingSelector,
    UCBSelector,
)
from stockbot.monitoring import setup_logging
from stockbot.monitoring.logger import get_logger
from stockbot.strategy.baseline import BuyAndHoldStrategy, SMAcrossoverStrategy

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train strategy selectors on historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on last 12 months of AAPL data
  python scripts/train_selector.py AAPL

  # Train on multiple symbols with Thompson sampling
  python scripts/train_selector.py AAPL MSFT GOOGL --selector thompson

  # Train on last 24 months with more epochs
  python scripts/train_selector.py AAPL --training-months 24 --epochs 3
        """,
    )

    parser.add_argument(
        "symbols",
        nargs="+",
        help="Symbols to train on (e.g., AAPL MSFT)",
    )

    parser.add_argument(
        "--training-months",
        type=int,
        default=12,
        help="Months of historical data to train on (default: 12)",
    )

    parser.add_argument(
        "--selector",
        type=str,
        default="ucb",
        choices=["ucb", "thompson", "epsilon"],
        help="Selector algorithm (default: ucb)",
    )

    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Training window size in days (default: 30)",
    )

    parser.add_argument(
        "--step-days",
        type=int,
        default=7,
        help="Step between training windows in days (default: 7)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of passes through the data (default: 2)",
    )

    parser.add_argument(
        "--reward-type",
        type=str,
        default="return_pct",
        choices=["pnl", "return_pct", "binary"],
        help="How to calculate reward (default: return_pct)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory for parquet data files",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/selector_state.json"),
        help="Output path for trained selector state",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Don't fetch missing data (fail if data not available)",
    )

    # Strategy-specific parameters
    parser.add_argument(
        "--fast-periods",
        type=str,
        default="5,10,15",
        help="Comma-separated fast SMA periods to test (default: 5,10,15)",
    )

    parser.add_argument(
        "--slow-periods",
        type=str,
        default="20,30,50",
        help="Comma-separated slow SMA periods to test (default: 20,30,50)",
    )

    return parser.parse_args()


def calculate_date_range(training_months: int) -> tuple[str, str]:
    """Calculate training date range based on today's date.

    Args:
        training_months: Number of months to look back

    Returns:
        Tuple of (start_date, end_date) as ISO strings
    """
    # End date is yesterday (to ensure complete data)
    end = datetime.now(timezone.utc).date() - timedelta(days=1)

    # Start date is training_months ago
    # Approximate months as 30 days
    start = end - timedelta(days=training_months * 30)

    return start.isoformat(), end.isoformat()


def ensure_data_available(
    symbols: list[Symbol],
    start_date: str,
    end_date: str,
    data_dir: Path,
    fetch_missing: bool = True,
) -> bool:
    """Ensure required data is available, fetching if needed.

    Args:
        symbols: Symbols to check
        start_date: Required start date
        end_date: Required end date
        data_dir: Data directory
        fetch_missing: Whether to fetch missing data

    Returns:
        True if all data available, False otherwise
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Parse dates
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    start_ts = Timestamp(int(start_dt.timestamp() * 1_000_000_000))
    end_ts = Timestamp(int(end_dt.timestamp() * 1_000_000_000))

    missing_symbols = []

    for symbol in symbols:
        # Check if data file exists and has required range
        try:
            bars = list(load_bars(data_dir, symbol, Timeframe.DAY_1))
            if bars:
                data_start = bars[0].timestamp
                data_end = bars[-1].timestamp

                # Check if we have enough coverage (allow some buffer)
                buffer_ns = 7 * 24 * 60 * 60 * 1_000_000_000  # 7 days
                if data_start <= start_ts + buffer_ns and data_end >= end_ts - buffer_ns:
                    logger.info(f"Data available for {symbol}: {len(bars)} bars")
                    continue

            logger.info(f"Insufficient data for {symbol}, needs fetch")
            missing_symbols.append(symbol)

        except Exception:
            logger.info(f"No data found for {symbol}")
            missing_symbols.append(symbol)

    if not missing_symbols:
        return True

    if not fetch_missing:
        logger.error(f"Missing data for: {', '.join(missing_symbols)}")
        logger.error("Run with --no-fetch removed to auto-fetch data")
        return False

    # Fetch missing data
    logger.info(f"Fetching data for {len(missing_symbols)} symbols...")

    try:
        settings = load_settings()
        if settings.alpaca is None:
            logger.error("Alpaca credentials not configured")
            logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
            return False

        provider = AlpacaDataProvider(settings.alpaca)

        for symbol in missing_symbols:
            logger.info(f"Downloading {symbol} from {start_date} to {end_date}...")

            try:
                bars = list(provider.get_bars(
                    symbol, start_ts, end_ts, Timeframe.DAY_1
                ))

                if not bars:
                    logger.warning(f"No data returned for {symbol}")
                    continue

                output_path = save_bars(
                    bars=bars,
                    data_dir=data_dir,
                    symbol=symbol,
                    timeframe=Timeframe.DAY_1,
                    append=False,
                )

                logger.info(f"Saved {len(bars)} bars to {output_path}")

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                return False

        return True

    except Exception as e:
        logger.error(f"Failed to initialize data provider: {e}")
        return False


def create_strategies(
    symbols: list[Symbol],
    fast_periods: list[int],
    slow_periods: list[int],
) -> list[Strategy]:
    """Create a diverse set of strategies to compete."""
    strategies: list[Strategy] = []

    # Add buy and hold baseline
    strategies.append(BuyAndHoldStrategy(symbols=symbols))

    # Add SMA crossover variants
    for fast in fast_periods:
        for slow in slow_periods:
            if fast < slow:
                strategies.append(
                    SMAcrossoverStrategy(
                        symbols=symbols,
                        fast_period=fast,
                        slow_period=slow,
                    )
                )

    return strategies


def create_selector(
    selector_type: str,
    strategies: list[Strategy],
    seed: int,
) -> StrategySelector:
    """Create a strategy selector."""
    if selector_type == "ucb":
        return UCBSelector(
            strategies=strategies,
            exploration_constant=2.0,
            seed=seed,
        )
    elif selector_type == "thompson":
        return ThompsonSamplingSelector(
            strategies=strategies,
            prior_alpha=1.0,
            prior_beta=1.0,
            seed=seed,
        )
    elif selector_type == "epsilon":
        return EpsilonGreedySelector(
            strategies=strategies,
            epsilon=0.3,
            epsilon_decay=0.99,
            min_epsilon=0.05,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")


def generate_training_windows(
    start_date: str,
    end_date: str,
    window_days: int,
    step_days: int,
) -> list[tuple[str, str]]:
    """Generate training windows for walk-forward training."""
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    window_delta = timedelta(days=window_days)
    step_delta = timedelta(days=step_days)

    windows = []
    current_start = start

    while current_start + window_delta <= end:
        window_end = current_start + window_delta
        windows.append((
            current_start.strftime("%Y-%m-%d"),
            window_end.strftime("%Y-%m-%d"),
        ))
        current_start += step_delta

    return windows


def train_on_window(
    window_start: str,
    window_end: str,
    selector: StrategySelector,
    callback: SelectorCallback,
    symbols: list[Symbol],
    data_dir: Path,
    seed: int,
) -> dict:
    """Run training on a single time window."""
    strategy = selector.select()
    strategy.reset()

    config = BacktestConfig(
        start_date=window_start,
        end_date=window_end,
        symbols=symbols,
        initial_capital=Price(Decimal("100000")),
        timeframe=Timeframe.DAY_1,
        commission=Decimal("0"),
        slippage_pct=Decimal("0.001"),
        seed=seed,
    )

    risk_config = RiskConfig(
        max_position_size=Quantity(Decimal("1000")),
        max_position_value=Price(Decimal("50000")),
    )

    engine = BacktestEngine(
        config=config,
        risk_config=risk_config,
        data_dir=data_dir,
        strategy=strategy,
        trade_callback=callback,
    )

    try:
        result = engine.run()
        return {
            "strategy": strategy.name,
            "total_return_pct": float(result.total_return_pct),
            "trades": result.total_trades,
            "win_rate": float(result.win_rate),
        }
    except Exception as e:
        logger.warning(f"Window failed: {e}")
        return {
            "strategy": strategy.name,
            "total_return_pct": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "error": str(e),
        }


def print_training_summary(
    selector: StrategySelector,
    window_results: list[dict],
    output_path: Path,
) -> None:
    """Print training summary."""
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nTotal training windows: {len(window_results)}")

    # Aggregate by strategy
    by_strategy: dict[str, list[dict]] = {}
    for result in window_results:
        name = result["strategy"]
        if name not in by_strategy:
            by_strategy[name] = []
        by_strategy[name].append(result)

    print(f"\n{'Strategy':<30} {'Windows':>8} {'Avg Return':>12} {'Avg WinRate':>12}")
    print("-" * 70)

    for name, results in sorted(
        by_strategy.items(),
        key=lambda x: sum(r["total_return_pct"] for r in x[1]) / len(x[1]),
        reverse=True,
    ):
        avg_return = sum(r["total_return_pct"] for r in results) / len(results)
        avg_win_rate = sum(r["win_rate"] for r in results) / len(results)
        print(f"{name:<30} {len(results):>8} {avg_return:>11.2f}% {avg_win_rate:>11.1f}%")

    # Print learned rankings
    print("\n" + "-" * 70)
    print("LEARNED STRATEGY RANKINGS")
    print("-" * 70)

    ranked = sorted(
        selector.stats.values(),
        key=lambda s: s.mean_reward,
        reverse=True,
    )

    for i, stats in enumerate(ranked[:5], 1):
        print(
            f"  {i}. {stats.name}: "
            f"reward={stats.mean_reward:.4f}, "
            f"selections={stats.n_selections}, "
            f"win_rate={stats.success_rate:.1%}"
        )

    print("=" * 70)
    print(f"\nSelector state saved to: {output_path}")
    print("\nTo use the trained selector:")
    print(f"  python scripts/run_with_selector.py SYMBOLS --selector-state {output_path}")
    print()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    setup_logging(level=args.log_level)

    # Parse symbols
    symbols = [Symbol(s.upper()) for s in args.symbols]

    # Calculate date range automatically
    start_date, end_date = calculate_date_range(args.training_months)

    print(f"\n{'=' * 70}")
    print("STRATEGY SELECTOR TRAINING")
    print(f"{'=' * 70}")
    print(f"Symbols: {', '.join(str(s) for s in symbols)}")
    print(f"Training period: {start_date} to {end_date} ({args.training_months} months)")
    print(f"Selector: {args.selector}")
    print(f"Window: {args.window_days} days, step: {args.step_days} days")
    print(f"Epochs: {args.epochs}")
    print(f"{'=' * 70}\n")

    # Ensure data is available
    print("Checking data availability...")
    if not ensure_data_available(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        data_dir=args.data_dir,
        fetch_missing=not args.no_fetch,
    ):
        return 1

    print()

    # Parse strategy parameters
    fast_periods = [int(x) for x in args.fast_periods.split(",")]
    slow_periods = [int(x) for x in args.slow_periods.split(",")]

    # Create strategies
    strategies = create_strategies(symbols, fast_periods, slow_periods)
    print(f"Created {len(strategies)} strategies to evaluate")

    # Create selector
    selector = create_selector(args.selector, strategies, args.seed)

    # Create callback with persistence
    args.output.parent.mkdir(parents=True, exist_ok=True)
    callback = SelectorCallback(
        selector=selector,
        reward_type=args.reward_type,
        persistence_path=args.output,
    )

    # Generate training windows
    windows = generate_training_windows(
        start_date,
        end_date,
        args.window_days,
        args.step_days,
    )
    print(f"Generated {len(windows)} training windows")
    print(f"\nStarting training...\n")

    # Training loop
    all_results: list[dict] = []
    total_windows = len(windows) * args.epochs

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        for i, (window_start, window_end) in enumerate(windows):
            result = train_on_window(
                window_start=window_start,
                window_end=window_end,
                selector=selector,
                callback=callback,
                symbols=symbols,
                data_dir=args.data_dir,
                seed=args.seed + epoch * 1000 + i,
            )

            all_results.append(result)

            # Progress
            completed = epoch * len(windows) + i + 1
            pct = completed / total_windows * 100
            print(f"  [{completed}/{total_windows}] {pct:.0f}% - {result['strategy']}: {result['total_return_pct']:.2f}%")

    # Save final state
    callback.on_session_end()

    # Print summary
    print_training_summary(selector, all_results, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
