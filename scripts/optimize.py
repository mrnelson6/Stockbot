#!/usr/bin/env python3
"""Script to optimize strategy parameters."""

import argparse
from decimal import Decimal
from pathlib import Path

from stockbot.config.settings import BacktestConfig, RiskConfig
from stockbot.core.types import Price, Quantity, Symbol, Timeframe
from stockbot.learning.optimizer import (
    GridSearchOptimizer,
    ParameterSpace,
    RandomSearchOptimizer,
    StrategyFactory,
    print_optimization_result,
)
from stockbot.monitoring import setup_logging
from stockbot.strategy.baseline import SMAcrossoverStrategy


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize strategy parameters")

    parser.add_argument(
        "symbols",
        nargs="+",
        help="Symbols to optimize on (e.g., AAPL MSFT)",
    )

    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (ISO format: 2024-01-01)",
    )

    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (ISO format: 2024-12-31)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Optimization method (default: grid)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return_pct", "win_rate", "profit_factor"],
        help="Metric to optimize (default: sharpe_ratio)",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials for random search (default: 50)",
    )

    parser.add_argument(
        "--fast-min",
        type=int,
        default=5,
        help="Minimum fast SMA period (default: 5)",
    )

    parser.add_argument(
        "--fast-max",
        type=int,
        default=20,
        help="Maximum fast SMA period (default: 20)",
    )

    parser.add_argument(
        "--slow-min",
        type=int,
        default=15,
        help="Minimum slow SMA period (default: 15)",
    )

    parser.add_argument(
        "--slow-max",
        type=int,
        default=50,
        help="Maximum slow SMA period (default: 50)",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Step size for grid search (default: 5)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing parquet data files",
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

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    setup_logging(level=args.log_level)

    # Convert symbols
    symbols = [Symbol(s.upper()) for s in args.symbols]

    # Create parameter space for SMA crossover strategy
    param_space = ParameterSpace()
    param_space.add_int("fast_period", args.fast_min, args.fast_max, step=args.step)
    param_space.add_int("slow_period", args.slow_min, args.slow_max, step=args.step)

    # Create backtest config
    backtest_config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        initial_capital=Price(Decimal("100000")),
        timeframe=Timeframe.DAY_1,
        commission=Decimal("0"),
        slippage_pct=Decimal("0.001"),
        seed=args.seed,
    )

    # Create risk config
    risk_config = RiskConfig(
        max_position_size=Quantity(Decimal("1000")),
        max_position_value=Price(Decimal("50000")),
    )

    # Create strategy factory
    factory = StrategyFactory(
        strategy_class=SMAcrossoverStrategy,
        symbols=symbols,
    )

    print(f"\n{'=' * 60}")
    print("STRATEGY PARAMETER OPTIMIZATION")
    print(f"{'=' * 60}")
    print(f"Strategy: SMA Crossover")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Method: {args.method}")
    print(f"Metric: {args.metric}")
    print(f"Fast Period Range: {args.fast_min} - {args.fast_max}")
    print(f"Slow Period Range: {args.slow_min} - {args.slow_max}")

    if args.method == "grid":
        print(f"Grid Size: {param_space.grid_size()} combinations")
        optimizer = GridSearchOptimizer(
            strategy_factory=factory,
            backtest_config=backtest_config,
            risk_config=risk_config,
            data_dir=args.data_dir,
            metric=args.metric,
        )
    else:
        print(f"Random Trials: {args.n_trials}")
        optimizer = RandomSearchOptimizer(
            strategy_factory=factory,
            backtest_config=backtest_config,
            risk_config=risk_config,
            data_dir=args.data_dir,
            metric=args.metric,
            n_trials=args.n_trials,
            seed=args.seed,
        )

    print(f"{'=' * 60}\n")

    # Run optimization
    result = optimizer.optimize(param_space)

    # Print results
    print_optimization_result(result)

    # Suggest best parameters
    if result.best_params:
        print("To run backtest with best parameters:")
        params_str = " ".join(
            f"--{k.replace('_', '-')} {v}" for k, v in result.best_params.items()
        )
        print(
            f"  python scripts/run_backtest.py {' '.join(symbols)} "
            f"--start {args.start} --end {args.end} --strategy sma {params_str}"
        )
        print()

    return 0


if __name__ == "__main__":
    exit(main())
