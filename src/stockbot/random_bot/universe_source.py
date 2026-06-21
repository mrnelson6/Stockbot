"""Dynamic trading universe sourced from Alpaca.

Instead of the hand-curated list in ``stockbot.config.universe``, this builds a
universe at runtime by:
1. Listing every active, tradable US equity from Alpaca (`get_all_assets`).
2. Pulling recent daily bars to measure liquidity (price + avg dollar-volume).
3. Filtering by a minimum price and minimum average dollar-volume, then keeping
   the most liquid ``max_symbols`` names.

The network fetch (`fetch_dynamic_universe`) is kept separate from the pure
ranking/filtering logic (`rank_by_liquidity`) so the latter can be unit tested.

Note on the data feed: the free "iex" feed reports IEX-only volume (a fraction of
consolidated tape), which understates dollar-volume but still rank-orders names
sensibly. Use "sip" for true consolidated volume if you have the subscription.
"""

from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from stockbot.config.settings import AlpacaConfig
from stockbot.monitoring.logger import get_logger

logger = get_logger("random_bot.universe")

# Major US equity exchanges (excludes OTC, which is illiquid/risky for market orders).
DEFAULT_EXCHANGES = {
    AssetExchange.NYSE,
    AssetExchange.NASDAQ,
    AssetExchange.ARCA,
    AssetExchange.AMEX,
    AssetExchange.BATS,
}

_BAR_CHUNK = 200  # symbols per historical-bars request


def rank_by_liquidity(
    stats: dict[str, tuple[float, float]],
    *,
    min_price: float,
    min_dollar_volume: float,
    max_symbols: int,
) -> list[str]:
    """Filter and rank symbols by liquidity (pure, no I/O).

    Args:
        stats: Symbol -> (last_price, avg_dollar_volume).
        min_price: Drop symbols priced below this (avoids penny stocks).
        min_dollar_volume: Drop symbols whose avg daily $-volume is below this.
        max_symbols: Keep at most this many, most-liquid first.

    Returns:
        Symbols sorted by average dollar-volume, descending.
    """
    filtered = [
        (sym, price, dv)
        for sym, (price, dv) in stats.items()
        if price >= min_price and dv >= min_dollar_volume
    ]
    filtered.sort(key=lambda row: row[2], reverse=True)
    return [sym for sym, _, _ in filtered[:max_symbols]]


def fetch_dynamic_universe(
    config: AlpacaConfig,
    *,
    feed: str = "iex",
    min_price: float = 5.0,
    min_dollar_volume: float = 10_000_000.0,
    max_symbols: int = 200,
    lookback_days: int = 30,
    exchanges: set | None = None,
) -> list[str]:
    """Build a liquid, tradable equity universe from Alpaca.

    Args:
        config: Alpaca credentials (the random bot's own account is fine; this is
            read-only — listing assets and historical bars).
        feed: "iex" (free) or "sip" (paid, true consolidated volume).
        min_price: Minimum last price to include.
        min_dollar_volume: Minimum average daily dollar-volume to include.
        max_symbols: Cap on the number of names returned (most liquid kept).
        lookback_days: Calendar days of daily bars used to measure liquidity.
        exchanges: Allowed exchanges (defaults to major US exchanges).

    Returns:
        List of ticker symbols, most liquid first. Empty if nothing qualifies.
    """
    exchanges = exchanges or DEFAULT_EXCHANGES

    # 1. List active, tradable US equities.
    trading = TradingClient(config.api_key, config.secret_key, paper=config.paper)
    assets = trading.get_all_assets(
        GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    )
    candidates = [
        a.symbol
        for a in assets
        if a.tradable
        and a.exchange in exchanges
        and "." not in a.symbol  # skip class shares / preferreds that Alpaca dot-encodes
        and "/" not in a.symbol
    ]
    logger.info(f"Found {len(candidates)} tradable US equities; measuring liquidity...")
    print(f"Found {len(candidates)} tradable US equities; measuring liquidity...", flush=True)

    # 2. Pull recent daily bars in chunks to compute price + avg dollar-volume.
    data_client = StockHistoricalDataClient(config.api_key, config.secret_key)
    feed_enum = DataFeed.IEX if feed == "iex" else DataFeed.SIP
    # Pad the end back a bit: the free IEX feed cannot serve the most recent ~15 min.
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=lookback_days)

    stats: dict[str, tuple[float, float]] = {}
    for i in range(0, len(candidates), _BAR_CHUNK):
        chunk = candidates[i : i + _BAR_CHUNK]
        try:
            barset = data_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end,
                    feed=feed_enum,
                )
            )
        except Exception as e:
            logger.warning(f"Bars request failed for chunk {i // _BAR_CHUNK}: {e}")
            continue

        for sym, bars in barset.data.items():
            closes = [float(b.close) for b in bars if b.close]
            dollar_vols = [
                float(b.close) * float(b.volume)
                for b in bars
                if b.close and b.volume
            ]
            if not closes or not dollar_vols:
                continue
            stats[sym] = (closes[-1], sum(dollar_vols) / len(dollar_vols))

        print(
            f"  scanned {min(i + _BAR_CHUNK, len(candidates))}/{len(candidates)} "
            f"({len(stats)} with data)...",
            flush=True,
        )

    # 3. Filter + rank.
    selected = rank_by_liquidity(
        stats,
        min_price=min_price,
        min_dollar_volume=min_dollar_volume,
        max_symbols=max_symbols,
    )
    logger.info(
        f"Selected {len(selected)} symbols "
        f"(min_price=${min_price}, min_$vol=${min_dollar_volume:,.0f}, cap={max_symbols})"
    )
    print(
        f"Selected {len(selected)} liquid symbols "
        f"(min price ${min_price:g}, min $-vol ${min_dollar_volume:,.0f}, cap {max_symbols}).",
        flush=True,
    )
    return selected
