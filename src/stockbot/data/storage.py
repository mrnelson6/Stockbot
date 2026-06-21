"""Parquet storage utilities for market data."""

from decimal import Decimal
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stockbot.core.models import Bar
from stockbot.core.types import Price, Quantity, Symbol, Timeframe, Timestamp


def bars_to_dataframe(bars: Iterable[Bar]) -> pd.DataFrame:
    """Convert bars to a pandas DataFrame.

    Args:
        bars: Iterable of Bar objects

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    records = []
    for bar in bars:
        records.append(
            {
                "timestamp": int(bar.timestamp),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )

    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    return pd.DataFrame(records)


def dataframe_to_bars(
    df: pd.DataFrame,
    symbol: Symbol,
    timeframe: Timeframe = Timeframe.MINUTE_1,
) -> list[Bar]:
    """Convert a DataFrame to a list of Bars.

    Args:
        df: DataFrame with OHLCV columns
        symbol: Symbol for all bars
        timeframe: Timeframe for all bars

    Returns:
        List of Bar objects
    """
    bars = []
    for _, row in df.iterrows():
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=Timestamp(int(row["timestamp"])),
                open=Price(Decimal(str(row["open"]))),
                high=Price(Decimal(str(row["high"]))),
                low=Price(Decimal(str(row["low"]))),
                close=Price(Decimal(str(row["close"]))),
                volume=Quantity(Decimal(str(row["volume"]))),
                timeframe=timeframe,
            )
        )
    return bars


def save_bars(
    bars: Iterable[Bar],
    data_dir: Path,
    symbol: Symbol,
    timeframe: Timeframe,
    append: bool = False,
) -> Path:
    """Save bars to a parquet file.

    Args:
        bars: Bars to save
        data_dir: Root data directory
        symbol: Symbol for the bars
        timeframe: Timeframe for the bars
        append: If True, append to existing file; otherwise overwrite

    Returns:
        Path to the saved file
    """
    # Create directory structure
    symbol_dir = data_dir / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    file_path = symbol_dir / f"{timeframe.value}.parquet"

    # Convert to DataFrame
    new_df = bars_to_dataframe(bars)

    if append and file_path.exists():
        # Load existing data and append
        existing_df = pq.read_table(file_path).to_pandas()
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Remove duplicates based on timestamp
        combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
        combined_df = combined_df.sort_values("timestamp")
        df_to_save = combined_df
    else:
        df_to_save = new_df.sort_values("timestamp")

    # Define schema for consistent storage
    schema = pa.schema(
        [
            ("timestamp", pa.int64()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
        ]
    )

    # Save to parquet
    table = pa.Table.from_pandas(df_to_save, schema=schema, preserve_index=False)
    pq.write_table(table, file_path, compression="snappy")

    return file_path


def load_bars(
    data_dir: Path,
    symbol: Symbol,
    timeframe: Timeframe,
    start: Timestamp | None = None,
    end: Timestamp | None = None,
) -> list[Bar]:
    """Load bars from a parquet file.

    Args:
        data_dir: Root data directory
        symbol: Symbol to load
        timeframe: Timeframe to load
        start: Optional start timestamp filter
        end: Optional end timestamp filter

    Returns:
        List of Bar objects
    """
    file_path = data_dir / symbol / f"{timeframe.value}.parquet"

    if not file_path.exists():
        return []

    df = pq.read_table(file_path).to_pandas()

    # Apply time filters
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] < end]

    df = df.sort_values("timestamp")

    return dataframe_to_bars(df, symbol, timeframe)
