"""Base data provider interface."""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from stockbot.core.models import Bar
from stockbot.core.types import Symbol, Timeframe, Timestamp


class BaseDataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_bars(
        self,
        symbol: Symbol,
        start: Timestamp,
        end: Timestamp,
        timeframe: Timeframe = Timeframe.MINUTE_1,
    ) -> Iterator[Bar]:
        """Yield bars for symbol in time range.

        Args:
            symbol: Ticker symbol
            start: Start timestamp (inclusive, nanoseconds)
            end: End timestamp (exclusive, nanoseconds)
            timeframe: Bar resolution

        Yields:
            Bar objects in chronological order
        """
        ...

    @abstractmethod
    def get_latest(self, symbol: Symbol) -> Optional[Bar]:
        """Get the most recent bar for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Most recent bar, or None if not available
        """
        ...

    @abstractmethod
    def get_symbols(self) -> list[Symbol]:
        """Get list of available symbols.

        Returns:
            List of symbols with available data
        """
        ...

    def get_bars_list(
        self,
        symbol: Symbol,
        start: Timestamp,
        end: Timestamp,
        timeframe: Timeframe = Timeframe.MINUTE_1,
    ) -> list[Bar]:
        """Get bars as a list instead of iterator.

        Convenience method for when you need all bars at once.

        Args:
            symbol: Ticker symbol
            start: Start timestamp
            end: End timestamp
            timeframe: Bar resolution

        Returns:
            List of bars
        """
        return list(self.get_bars(symbol, start, end, timeframe))
