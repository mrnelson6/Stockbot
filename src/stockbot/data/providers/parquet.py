"""Parquet-based data provider for local data access."""

from decimal import Decimal
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import pyarrow.parquet as pq

from stockbot.core.exceptions import DataNotFoundError
from stockbot.core.models import Bar
from stockbot.core.types import Price, Quantity, Symbol, Timeframe, Timestamp
from stockbot.data.providers.base import BaseDataProvider


class ParquetDataProvider(BaseDataProvider):
    """Data provider reading from local Parquet files.

    Expected directory structure:
        data_dir/
            AAPL/
                1m.parquet
                5m.parquet
                1d.parquet
            MSFT/
                1m.parquet
                ...
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the Parquet data provider.

        Args:
            data_dir: Root directory containing symbol subdirectories
        """
        self._data_dir = Path(data_dir)
        self._cache: dict[tuple[Symbol, Timeframe], pd.DataFrame] = {}

    def _get_file_path(self, symbol: Symbol, timeframe: Timeframe) -> Path:
        """Get the parquet file path for a symbol/timeframe."""
        return self._data_dir / symbol / f"{timeframe.value}.parquet"

    def _load_dataframe(self, symbol: Symbol, timeframe: Timeframe) -> pd.DataFrame:
        """Load and cache a dataframe for a symbol/timeframe."""
        cache_key = (symbol, timeframe)

        if cache_key not in self._cache:
            file_path = self._get_file_path(symbol, timeframe)
            if not file_path.exists():
                raise DataNotFoundError(
                    f"No data file found for {symbol} at {timeframe.value}: {file_path}"
                )

            df = pq.read_table(file_path).to_pandas()

            # Ensure timestamp column is in nanoseconds
            if "timestamp" in df.columns:
                if df["timestamp"].dtype != "int64":
                    # Convert datetime to nanosecond timestamps
                    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64")

            # Sort by timestamp
            df = df.sort_values("timestamp")
            self._cache[cache_key] = df

        return self._cache[cache_key]

    def _row_to_bar(
        self,
        row: pd.Series,  # type: ignore[type-arg]
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> Bar:
        """Convert a DataFrame row to a Bar."""
        return Bar(
            symbol=symbol,
            timestamp=Timestamp(int(row["timestamp"])),
            open=Price(Decimal(str(row["open"]))),
            high=Price(Decimal(str(row["high"]))),
            low=Price(Decimal(str(row["low"]))),
            close=Price(Decimal(str(row["close"]))),
            volume=Quantity(Decimal(str(row["volume"]))),
            timeframe=timeframe,
        )

    def get_bars(
        self,
        symbol: Symbol,
        start: Timestamp,
        end: Timestamp,
        timeframe: Timeframe = Timeframe.MINUTE_1,
    ) -> Iterator[Bar]:
        """Yield bars from parquet files.

        Args:
            symbol: Ticker symbol
            start: Start timestamp (nanoseconds)
            end: End timestamp (nanoseconds)
            timeframe: Bar resolution

        Yields:
            Bar objects in chronological order
        """
        df = self._load_dataframe(symbol, timeframe)

        # Filter by time range
        mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
        filtered = df[mask]

        for _, row in filtered.iterrows():
            yield self._row_to_bar(row, symbol, timeframe)

    def get_latest(self, symbol: Symbol) -> Optional[Bar]:
        """Get the most recent bar for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Most recent bar, or None if no data available
        """
        # Try each timeframe, starting with smallest
        for timeframe in [Timeframe.MINUTE_1, Timeframe.DAY_1]:
            try:
                df = self._load_dataframe(symbol, timeframe)
                if not df.empty:
                    row = df.iloc[-1]
                    return self._row_to_bar(row, symbol, timeframe)
            except DataNotFoundError:
                continue
        return None

    def get_symbols(self) -> list[Symbol]:
        """Get list of symbols with available data.

        Returns:
            List of symbols that have data directories
        """
        if not self._data_dir.exists():
            return []

        symbols = []
        for path in self._data_dir.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                # Check if it has any parquet files
                if list(path.glob("*.parquet")):
                    symbols.append(Symbol(path.name))

        return sorted(symbols)

    def clear_cache(self) -> None:
        """Clear the in-memory data cache."""
        self._cache.clear()
