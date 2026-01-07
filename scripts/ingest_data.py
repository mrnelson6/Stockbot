#!/usr/bin/env python3
"""Script to ingest historical data from Alpaca."""

import argparse
from datetime import datetime
from pathlib import Path

from stockbot.config import load_settings
from stockbot.core.types import Symbol, Timeframe
from stockbot.data.providers.alpaca import AlpacaDataProvider
from stockbot.data.storage import save_bars
from stockbot.monitoring import get_logger, setup_logging

logger = get_logger("ingest")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest historical market data from Alpaca"
    )

    parser.add_argument(
        "symbols",
        nargs="+",
        help="Symbols to download (e.g., AAPL MSFT GOOGL)",
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
        "--timeframe",
        type=str,
        default="1m",
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
        help="Bar timeframe (default: 1m)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Output directory for parquet files",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing files instead of overwriting",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
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
        "1w": Timeframe.WEEK_1,
    }
    return mapping[tf_str]


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Load settings (for Alpaca credentials)
    try:
        settings = load_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        logger.error("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set")
        return 1

    if settings.alpaca is None:
        logger.error("Alpaca credentials not configured")
        logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
        return 1

    # Create data provider
    provider = AlpacaDataProvider(settings.alpaca)

    # Parse dates
    from datetime import timezone

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    from stockbot.core.types import Timestamp

    start_ts = Timestamp(int(start_dt.timestamp() * 1_000_000_000))
    end_ts = Timestamp(int(end_dt.timestamp() * 1_000_000_000))

    timeframe = timeframe_from_string(args.timeframe)

    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)

    # Download data for each symbol
    success_count = 0
    for symbol_str in args.symbols:
        symbol = Symbol(symbol_str.upper())

        logger.info(
            f"Downloading {symbol} from {args.start} to {args.end} ({args.timeframe})"
        )

        try:
            bars = list(provider.get_bars(symbol, start_ts, end_ts, timeframe))

            if not bars:
                logger.warning(f"No data found for {symbol}")
                continue

            # Save to parquet
            output_path = save_bars(
                bars=bars,
                data_dir=args.data_dir,
                symbol=symbol,
                timeframe=timeframe,
                append=args.append,
            )

            logger.info(f"Saved {len(bars)} bars to {output_path}")
            success_count += 1

        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")

    logger.info(f"Successfully downloaded {success_count}/{len(args.symbols)} symbols")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    exit(main())
