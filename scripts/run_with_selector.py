#!/usr/bin/env python3
"""Run backtests or paper trading with a trained strategy selector.

This script loads a previously trained selector and uses it to
dynamically choose strategies based on learned performance.
"""

import argparse
from decimal import Decimal
from pathlib import Path

from stockbot.config.settings import AlpacaConfig, BacktestConfig, RiskConfig, load_settings
from stockbot.core.types import Price, Quantity, Symbol, Timeframe
from stockbot.engine.backtest import BacktestEngine
from stockbot.engine.paper import PaperTradingConfig, PaperTradingEngine
from stockbot.learning.callbacks import SelectorCallback
from stockbot.learning.selector import (
    EnsembleStrategy,
    EpsilonGreedySelector,
    ThompsonSamplingSelector,
    UCBSelector,
)
from stockbot.monitoring import setup_logging
from stockbot.monitoring.logger import get_logger
from stockbot.strategy.baseline import BuyAndHoldStrategy, SMAcrossoverStrategy

logger = get_logger("run_selector")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run with a trained strategy selector"
    )

    parser.add_argument(
        "symbols",
        nargs="+",
        help="Symbols to trade (e.g., AAPL MSFT)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="backtest",
        choices=["backtest", "paper"],
        help="Execution mode (default: backtest)",
    )

    parser.add_argument(
        "--selector-state",
        type=Path,
        default=Path("./data/selector_state.json"),
        help="Path to trained selector state",
    )

    parser.add_argument(
        "--selector",
        type=str,
        default="ucb",
        choices=["ucb", "thompson", "epsilon"],
        help="Selector algorithm (default: ucb)",
    )

    parser.add_argument(
        "--combination",
        type=str,
        default="best",
        choices=["best", "vote", "weighted"],
        help="How to combine strategies: best (use selected), vote (majority), weighted (by performance)",
    )

    # Backtest options
    parser.add_argument(
        "--start",
        type=str,
        help="Backtest start date (ISO format)",
    )

    parser.add_argument(
        "--end",
        type=str,
        help="Backtest end date (ISO format)",
    )

    # Common options
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
        help="Random seed",
    )

    parser.add_argument(
        "--continue-learning",
        action="store_true",
        help="Continue learning during execution (updates selector state)",
    )

    # Strategy parameters (should match training)
    parser.add_argument(
        "--fast-periods",
        type=str,
        default="5,10,15",
        help="Comma-separated fast SMA periods (default: 5,10,15)",
    )

    parser.add_argument(
        "--slow-periods",
        type=str,
        default="20,30,50",
        help="Comma-separated slow SMA periods (default: 20,30,50)",
    )

    return parser.parse_args()


def create_strategies(symbols, fast_periods, slow_periods):
    """Create the same strategies as used in training."""
    strategies = []

    # Buy and hold baseline
    strategies.append(BuyAndHoldStrategy(symbols=symbols))

    # SMA crossover variants
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


def create_selector(selector_type, strategies, seed):
    """Create selector matching the trained type."""
    if selector_type == "ucb":
        return UCBSelector(strategies=strategies, seed=seed)
    elif selector_type == "thompson":
        return ThompsonSamplingSelector(strategies=strategies, seed=seed)
    elif selector_type == "epsilon":
        return EpsilonGreedySelector(
            strategies=strategies,
            epsilon=0.05,  # Low exploration for execution
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown selector: {selector_type}")


def run_backtest(args, ensemble, callback):
    """Run backtest with ensemble strategy."""
    if not args.start or not args.end:
        print("Error: --start and --end required for backtest mode")
        return 1

    symbols = [Symbol(s.upper()) for s in args.symbols]

    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols,
        initial_capital=Price(Decimal("100000")),
        timeframe=Timeframe.DAY_1,
        commission=Decimal("0"),
        slippage_pct=Decimal("0.001"),
        seed=args.seed,
    )

    risk_config = RiskConfig(
        max_position_size=Quantity(Decimal("1000")),
        max_position_value=Price(Decimal("50000")),
    )

    engine = BacktestEngine(
        config=config,
        risk_config=risk_config,
        data_dir=args.data_dir,
        strategy=ensemble,
        trade_callback=callback if args.continue_learning else None,
    )

    print(f"\n{'=' * 60}")
    print("RUNNING BACKTEST WITH TRAINED SELECTOR")
    print(f"{'=' * 60}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Combination method: {args.combination}")
    print(f"{'=' * 60}\n")

    result = engine.run()

    # Print results
    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Strategy: {result.strategy_name}")
    print(f"Total Return: ${result.total_return:,.2f} ({result.total_return_pct:.2f}%)")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"{'=' * 60}\n")

    if args.continue_learning:
        callback.on_session_end()
        print(f"Updated selector state saved to: {args.selector_state}")

    return 0


def run_paper(args, ensemble, callback):
    """Run paper trading with ensemble strategy."""
    symbols = [Symbol(s.upper()) for s in args.symbols]

    try:
        settings = load_settings()
        alpaca_config = settings.alpaca
    except Exception as e:
        print(f"Error loading Alpaca config: {e}")
        print("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set")
        return 1

    risk_config = RiskConfig(
        max_position_size=Quantity(Decimal("100")),
        max_position_value=Price(Decimal("10000")),
    )

    config = PaperTradingConfig(
        symbols=symbols,
        alpaca_config=alpaca_config,
        risk_config=risk_config,
        poll_interval_seconds=60.0,
        timeframe=Timeframe.MINUTE_1,
    )

    print(f"\n{'=' * 60}")
    print("STARTING PAPER TRADING WITH TRAINED SELECTOR")
    print(f"{'=' * 60}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Combination method: {args.combination}")
    print(f"Continue learning: {args.continue_learning}")
    print(f"{'=' * 60}\n")
    print("Press Ctrl+C to stop\n")

    engine = PaperTradingEngine(
        config=config,
        strategy=ensemble,
        trade_callback=callback if args.continue_learning else None,
    )

    engine.run()

    if args.continue_learning:
        print(f"\nUpdated selector state saved to: {args.selector_state}")

    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()

    setup_logging(level=args.log_level)

    symbols = [Symbol(s.upper()) for s in args.symbols]
    fast_periods = [int(x) for x in args.fast_periods.split(",")]
    slow_periods = [int(x) for x in args.slow_periods.split(",")]

    # Create strategies
    strategies = create_strategies(symbols, fast_periods, slow_periods)
    logger.info(f"Created {len(strategies)} strategies")

    # Create selector
    selector = create_selector(args.selector, strategies, args.seed)

    # Create callback to load trained state
    callback = SelectorCallback(
        selector=selector,
        reward_type="return_pct",
        persistence_path=args.selector_state,
    )

    # Check if state was loaded
    total_selections = sum(s.n_selections for s in selector.stats.values())
    if total_selections > 0:
        print(f"Loaded trained selector state ({total_selections} historical selections)")

        # Show top strategies
        ranked = sorted(
            selector.stats.values(),
            key=lambda s: s.mean_reward,
            reverse=True,
        )
        print("Top strategies by learned performance:")
        for i, stats in enumerate(ranked[:3], 1):
            print(f"  {i}. {stats.name} (mean_reward={stats.mean_reward:.4f})")
    else:
        print("Warning: No trained state found, selector will start fresh")

    # Create ensemble strategy
    ensemble = EnsembleStrategy(
        selector=selector,
        combination_method=args.combination,
    )

    # Run
    if args.mode == "backtest":
        return run_backtest(args, ensemble, callback)
    else:
        return run_paper(args, ensemble, callback)


if __name__ == "__main__":
    exit(main())
