"""Alpaca data provider for historical market data."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterator, Literal, Optional

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
from alpaca.data.timeframe import TimeFrameUnit

from stockbot.config.settings import AlpacaConfig
from stockbot.core.models import Bar
from stockbot.core.types import Price, Quantity, Symbol, Timeframe, Timestamp
from stockbot.data.providers.base import BaseDataProvider


def _timeframe_to_alpaca(tf: Timeframe) -> AlpacaTimeFrame:
    """Convert our Timeframe to Alpaca TimeFrame."""
    mapping = {
        Timeframe.MINUTE_1: AlpacaTimeFrame.Minute,
        Timeframe.MINUTE_5: AlpacaTimeFrame(5, TimeFrameUnit.Minute),
        Timeframe.MINUTE_15: AlpacaTimeFrame(15, TimeFrameUnit.Minute),
        Timeframe.MINUTE_30: AlpacaTimeFrame(30, TimeFrameUnit.Minute),
        Timeframe.HOUR_1: AlpacaTimeFrame.Hour,
        Timeframe.HOUR_4: AlpacaTimeFrame(4, TimeFrameUnit.Hour),
        Timeframe.DAY_1: AlpacaTimeFrame.Day,
        Timeframe.WEEK_1: AlpacaTimeFrame.Week,
    }
    return mapping[tf]


def _datetime_to_timestamp(dt: datetime) -> Timestamp:
    """Convert datetime to nanosecond timestamp."""
    return Timestamp(int(dt.timestamp() * 1_000_000_000))


def _timestamp_to_datetime(ts: Timestamp) -> datetime:
    """Convert nanosecond timestamp to datetime."""
    return datetime.fromtimestamp(ts / 1_000_000_000, tz=timezone.utc)


class AlpacaDataProvider(BaseDataProvider):
    """Data provider using Alpaca's historical data API."""

    def __init__(
        self,
        config: AlpacaConfig,
        feed: Literal["iex", "sip"] = "sip",
    ) -> None:
        """Initialize the Alpaca data provider.

        Args:
            config: Alpaca API configuration
            feed: Data feed to use - "iex" (free) or "sip" (paid subscription)
        """
        self._client = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )
        self._feed = DataFeed.IEX if feed == "iex" else DataFeed.SIP
        self._cache: dict[Symbol, list[Bar]] = {}

    def get_bars(
        self,
        symbol: Symbol,
        start: Timestamp,
        end: Timestamp,
        timeframe: Timeframe = Timeframe.MINUTE_1,
    ) -> Iterator[Bar]:
        """Fetch bars from Alpaca API.

        Args:
            symbol: Ticker symbol
            start: Start timestamp (nanoseconds)
            end: End timestamp (nanoseconds)
            timeframe: Bar resolution

        Yields:
            Bar objects in chronological order
        """
        start_dt = _timestamp_to_datetime(start)
        end_dt = _timestamp_to_datetime(end)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_timeframe_to_alpaca(timeframe),
            start=start_dt,
            end=end_dt,
            feed=self._feed,
        )

        bars_data = self._client.get_stock_bars(request)

        # BarSet uses string keys, not our Symbol NewType
        symbol_str = str(symbol)

        try:
            symbol_bars = bars_data[symbol_str]
        except (KeyError, TypeError):
            symbol_bars = None

        if symbol_bars:
            for alpaca_bar in symbol_bars:
                yield Bar(
                    symbol=symbol,
                    timestamp=_datetime_to_timestamp(alpaca_bar.timestamp),
                    open=Price(Decimal(str(alpaca_bar.open))),
                    high=Price(Decimal(str(alpaca_bar.high))),
                    low=Price(Decimal(str(alpaca_bar.low))),
                    close=Price(Decimal(str(alpaca_bar.close))),
                    volume=Quantity(Decimal(str(alpaca_bar.volume))),
                    timeframe=timeframe,
                )

    def get_latest(self, symbol: Symbol) -> Optional[Bar]:
        """Get the most recent bar for a symbol.

        Note: This fetches the latest available bar from Alpaca.
        For real-time data, consider using the streaming API.

        Args:
            symbol: Ticker symbol

        Returns:
            Most recent bar, or None if not available
        """
        from alpaca.data.requests import StockLatestBarRequest

        request = StockLatestBarRequest(symbol_or_symbols=symbol, feed=self._feed)
        latest = self._client.get_stock_latest_bar(request)

        if symbol in latest:
            alpaca_bar = latest[symbol]
            return Bar(
                symbol=symbol,
                timestamp=_datetime_to_timestamp(alpaca_bar.timestamp),
                open=Price(Decimal(str(alpaca_bar.open))),
                high=Price(Decimal(str(alpaca_bar.high))),
                low=Price(Decimal(str(alpaca_bar.low))),
                close=Price(Decimal(str(alpaca_bar.close))),
                volume=Quantity(Decimal(str(alpaca_bar.volume))),
                timeframe=Timeframe.MINUTE_1,
            )
        return None

    def get_symbols(self) -> list[Symbol]:
        """Get list of tradable symbols from Alpaca.

        Returns:
            List of active stock symbols
        """
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest

        # Note: This requires trading API access
        # For data-only access, you would need to maintain your own symbol list
        trading_client = TradingClient(
            api_key=self._client._api_key,  # type: ignore[attr-defined]
            secret_key=self._client._secret_key,  # type: ignore[attr-defined]
            paper=True,
        )

        request = GetAssetsRequest(asset_class="us_equity", status="active")
        assets = trading_client.get_all_assets(request)

        return [Symbol(asset.symbol) for asset in assets if asset.tradable]
