#!/usr/bin/env python3
"""Script to run backtests."""

import argparse
from decimal import Decimal
from pathlib import Path

from stockbot.config.settings import BacktestConfig, RiskConfig
from stockbot.core.types import Price, Quantity, Symbol, Timeframe
from stockbot.engine.backtest import run_backtest
from stockbot.monitoring.metrics import calculate_all_metrics, print_metrics
from stockbot.strategy.baseline import BuyAndHoldStrategy, SMAcrossoverStrategy


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a backtest")

    parser.add_argument(
        "symbols",
        nargs="+",
        help="Symbols to trade (e.g., AAPL MSFT)",
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
        "--strategy",
        type=str,
        default="sma",
        choices=["sma", "buyhold"],
        help="Strategy to use (default: sma)",
    )

    parser.add_argument(
        "--fast-period",
        type=int,
        default=10,
        help="Fast SMA period (for sma strategy)",
    )

    parser.add_argument(
        "--slow-period",
        type=int,
        default=20,
        help="Slow SMA period (for sma strategy)",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        help="Bar timeframe (default: 1d)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory containing parquet data files",
    )

    parser.add_argument(
        "--max-position-size",
        type=int,
        default=1000,
        help="Maximum position size in shares",
    )

    parser.add_argument(
        "--max-position-value",
        type=float,
        default=10000,
        help="Maximum position value in dollars",
    )

    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Per-share commission",
    )

    parser.add_argument(
        "--slippage",
        type=float,
        default=0.001,
        help="Slippage percentage (0.001 = 0.1%%)",
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


def timeframe_from_string(tf_str: str) -> Timeframe:
    """Convert string to Timeframe enum."""
    mapping = {
        "1m": Timeframe.MINUTE_1,
        "5m": Timeframe.MINUTE_5,
        "15m": Timeframe.MINUTE_15,
        "30m": Timeframe.MINUTE_30,
        "1h": Timeframe.HOUR_1,
        "4h": Timeframe.HOUR_4,
        "1d": Timeframe.DAY_1,
    }
    return mapping[tf_str]


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Convert symbols
    symbols = [Symbol(s.upper()) for s in args.symbols]

    # Create backtest config
    backtest_config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        initial_capital=Price(Decimal(str(args.capital))),
        timeframe=timeframe_from_string(args.timeframe),
        commission=Decimal(str(args.commission)),
        slippage_pct=Decimal(str(args.slippage)),
        seed=args.seed,
    )

    # Create risk config
    risk_config = RiskConfig(
        max_position_size=Quantity(Decimal(str(args.max_position_size))),
        max_position_value=Price(Decimal(str(args.max_position_value))),
    )

    # Create strategy
    if args.strategy == "sma":
        strategy = SMAcrossoverStrategy(
            symbols=symbols,
            fast_period=args.fast_period,
            slow_period=args.slow_period,
        )
    else:
        strategy = BuyAndHoldStrategy(symbols=symbols)

    print(f"\nRunning backtest: {args.start} to {args.end}")
    print(f"Strategy: {strategy.name}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print()

    # Run backtest
    result = run_backtest(
        config=backtest_config,
        risk_config=risk_config,
        data_dir=args.data_dir,
        strategy=strategy,
        log_level=args.log_level,
    )

    # Calculate and print metrics
    metrics = calculate_all_metrics(
        equity_curve=result.equity_curve,
        trades=result.trades,
        initial_capital=result.initial_capital,
    )

    print_metrics(metrics)

    # Print summary
    print(f"Strategy: {result.strategy_name}")
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Final Equity: ${float(result.final_equity):,.2f}")
    print(f"Total Return: ${float(result.total_return):,.2f} ({float(result.total_return_pct):.2f}%)")
    print(f"Total Trades: {result.total_trades}")

    return 0


if __name__ == "__main__":
    exit(main())
