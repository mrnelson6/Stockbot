"""FIFO cost-basis accounting for the dashboard.

Pure functions (no I/O) that match sells against buy lots first-in-first-out, so
the dashboard can show per-position cost basis and acquisition dates, realized
P&L per sale, and an all-time best/worst leaderboard. Kept separate from the DB
so the algorithm is unit-testable.

A "lot" is one buy: {"qty", "price", "opened_ts", ...}. Sells consume the oldest
lots first. Realized P&L on a sale = proceeds - cost basis of the matched shares.
"""

from typing import Any

_EPS = 1e-9

# Regulatory fees charged on SELLS (Alpaca itself is commission-free, but passes
# these through). Rates are set by the SEC/FINRA and CHANGE periodically — these
# are approximate current defaults, so the dashboard figure is an ESTIMATE.
SEC_FEE_RATE = 0.0000278   # SEC Section 31 fee: ~$27.80 per $1,000,000 of proceeds
TAF_PER_SHARE = 0.000166   # FINRA Trading Activity Fee per share sold
TAF_CAP = 8.30             # FINRA TAF per-trade cap


def estimate_fees(proceeds: float, qty: float) -> float:
    """Estimated SEC + FINRA regulatory fees for a sale (USD).

    SEC fee scales with dollar proceeds; FINRA TAF scales with share count (capped).
    """
    sec = max(0.0, proceeds) * SEC_FEE_RATE
    taf = min(qty * TAF_PER_SHARE, TAF_CAP)
    return sec + taf


def weighted_hold_ms(consumed: list[dict], sell_ts: int) -> float:
    """Share-weighted holding time (ms) of the lots consumed by a sale."""
    total = sum(c["qty"] for c in consumed)
    if total <= _EPS:
        return 0.0
    return sum(c["qty"] * (sell_ts - c["opened_ts"]) for c in consumed) / total


def fifo_sell(lots: list[dict], qty: float, price: float) -> dict[str, Any]:
    """Consume ``qty`` shares from ``lots`` (oldest first) at sale ``price``.

    Args:
        lots: Open lots for one symbol, oldest first. Not mutated.
        qty: Shares being sold.
        price: Sale price per share.

    Returns:
        dict with:
          cost_basis   - cost of the matched shares (sum qty*lot_price)
          proceeds     - matched_qty * price
          realized     - proceeds - cost_basis
          realized_pct - realized / cost_basis (0 if cost_basis is 0)
          matched_qty  - shares actually matched to lots (< qty if under-lotted)
          remaining_lots - lots after consumption (emptied lots dropped)
    """
    remaining = qty
    cost_basis = 0.0
    out_lots: list[dict] = []
    consumed: list[dict] = []

    for lot in lots:
        lot = dict(lot)
        if remaining > _EPS and lot["qty"] > _EPS:
            take = min(lot["qty"], remaining)
            cost_basis += take * lot["price"]
            consumed.append({"qty": take, "price": lot["price"], "opened_ts": lot["opened_ts"]})
            lot["qty"] -= take
            remaining -= take
        if lot["qty"] > _EPS:
            out_lots.append(lot)

    matched_qty = qty - remaining
    proceeds = matched_qty * price
    realized = proceeds - cost_basis
    realized_pct = realized / cost_basis if cost_basis > _EPS else 0.0

    return {
        "cost_basis": cost_basis,
        "proceeds": proceeds,
        "realized": realized,
        "realized_pct": realized_pct,
        "matched_qty": matched_qty,
        "remaining_lots": out_lots,
        "consumed": consumed,
    }


def replay_trades(trades: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Replay an ordered trade list to derive open lots and per-sell realized P&L.

    Args:
        trades: Trades in chronological order, each with keys
            id, ts, symbol, side ("BUY"/"SELL"), qty, price.

    Returns:
        (open_lots_by_symbol, sell_results) where:
          open_lots_by_symbol: symbol -> remaining lots [{qty, price, opened_ts}]
          sell_results: per SELL trade, {id, cost_basis, proceeds, realized, realized_pct}
    """
    lots_by_symbol: dict[str, list[dict]] = {}
    sell_results: list[dict] = []

    for t in trades:
        symbol = t["symbol"]
        side = str(t["side"]).upper()
        if side == "BUY":
            lots_by_symbol.setdefault(symbol, []).append(
                {"qty": float(t["qty"]), "price": float(t["price"]), "opened_ts": int(t["ts"])}
            )
        else:  # SELL
            lots = lots_by_symbol.get(symbol, [])
            res = fifo_sell(lots, float(t["qty"]), float(t["price"]))
            lots_by_symbol[symbol] = res["remaining_lots"]
            sell_results.append(
                {
                    "id": t.get("id"),
                    "cost_basis": res["cost_basis"],
                    "proceeds": res["proceeds"],
                    "realized": res["realized"],
                    "realized_pct": res["realized_pct"],
                    "fees": estimate_fees(res["proceeds"], res["matched_qty"]),
                    "hold_ms": weighted_hold_ms(res["consumed"], int(t["ts"])),
                }
            )

    return lots_by_symbol, sell_results
