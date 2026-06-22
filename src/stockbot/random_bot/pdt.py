"""Pattern Day Trader (PDT) guard.

FINRA's PDT rule limits accounts under $25,000 equity to 3 day trades per rolling
5 business days; exceeding it gets the account restricted for up to 90 days. A
"day trade" is opening and closing the same security on the same day.

A frequently-churning random bot can trip this easily on a small account. The
simplest robust guard is to never sell shares that were *bought today* — that
guarantees zero day trades (every sale closes a position opened on a prior day),
so the day-trade count stays at 0 and the account is never restricted.

This module holds the pure decision logic so it can be unit tested; the trader
tracks same-day buys and the current market date and calls in here.
"""

from datetime import date, datetime, timezone

# FINRA pattern-day-trader equity threshold (USD).
PDT_EQUITY_THRESHOLD = 25_000.0


def market_date() -> date:
    """Current US market calendar date (Eastern), used to bucket day trades.

    Falls back to UTC if timezone data is unavailable (e.g. a bare Windows dev
    box without tzdata); the Linux deployment box has zoneinfo data.
    """
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return datetime.now(timezone.utc).date()


def pdt_guarded_sells(
    sells: dict[str, float],
    bought_today: dict[str, float],
    account_value: float,
    *,
    threshold: float = PDT_EQUITY_THRESHOLD,
    enabled: bool = True,
) -> dict[str, float]:
    """Trim sell intents so they never close shares opened the same day.

    Args:
        sells: Symbol -> shares the bot wants to sell (the full held quantity).
        bought_today: Symbol -> shares bought so far today.
        account_value: Current account equity. The guard only applies below
            ``threshold`` (at/above it the PDT rule doesn't restrict day trades).
        threshold: PDT equity threshold.
        enabled: Master switch; when False the sells pass through unchanged.

    Returns:
        Adjusted sells. A symbol is dropped if all held shares were bought today;
        otherwise its quantity is reduced to the non-same-day (overnight) shares.
    """
    if not enabled or account_value >= threshold:
        return dict(sells)

    guarded: dict[str, float] = {}
    for symbol, held_qty in sells.items():
        sellable = held_qty - bought_today.get(symbol, 0.0)
        if sellable > 1e-9:
            guarded[symbol] = sellable
    return guarded
