#!/usr/bin/env python3
"""Script to run paper trading."""

import argparse
from decimal import Decimal

from stockbot.config import load_settings
from stockbot.config.settings import RiskConfig
from stockbot.core.types import Price, Quantity, Symbol, Timeframe
from stockbot.engine.paper import PaperTradingConfig, run_paper_trading
from stockbot.strategy.baseline import BuyAndHoldStrategy, SMAcrossoverStrategy


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run paper trading")

    parser.add_argument(
        "symbols",
        nargs="+",
        help="Symbols to trade (e.g., AAPL MSFT)",
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
        "--poll-interval",
        type=float,
        default=60.0,
        help="Seconds between market checks (default: 60)",
    )

    parser.add_argument(
        "--max-position-size",
        type=int,
        default=100,
        help="Maximum position size in shares (default: 100)",
    )

    parser.add_argument(
        "--max-position-value",
        type=float,
        default=5000,
        help="Maximum position value in dollars (default: 5000)",
    )

    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=500,
        help="Maximum daily loss before halt (default: 500)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load settings (for Alpaca credentials)
    try:
        settings = load_settings()
    except Exception as e:
        print(f"Failed to load settings: {e}")
        print("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in .env")
        return 1

    if settings.alpaca is None:
        print("Alpaca credentials not configured")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
        return 1

    # Ensure we're using paper trading
    if not settings.alpaca.paper:
        print("WARNING: Alpaca is configured for LIVE trading!")
        print("Set ALPACA_PAPER=true in .env for paper trading")
        response = input("Continue with LIVE trading? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted")
            return 1

    # Convert symbols
    symbols = [Symbol(s.upper()) for s in args.symbols]

    # Create risk config with conservative limits for paper trading
    risk_config = RiskConfig(
        max_position_size=Quantity(Decimal(str(args.max_position_size))),
        max_position_value=Price(Decimal(str(args.max_position_value))),
        max_daily_loss=Price(Decimal(str(args.max_daily_loss))),
        max_open_positions=5,
        max_orders_per_minute=5,
    )

    # Create paper trading config
    paper_config = PaperTradingConfig(
        symbols=symbols,
        alpaca_config=settings.alpaca,
        risk_config=risk_config,
        poll_interval_seconds=args.poll_interval,
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

    print(f"\n{'='*50}")
    print("PAPER TRADING")
    print(f"{'='*50}")
    print(f"Strategy: {strategy.name}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Poll Interval: {args.poll_interval}s")
    print(f"Max Position Size: {args.max_position_size} shares")
    print(f"Max Position Value: ${args.max_position_value}")
    print(f"Max Daily Loss: ${args.max_daily_loss}")
    print(f"{'='*50}")
    print("\nPress Ctrl+C to stop\n")

    # Run paper trading
    try:
        run_paper_trading(
            config=paper_config,
            strategy=strategy,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        print("\nStopped by user")

    return 0


if __name__ == "__main__":
    exit(main())
