"""Pure portfolio-analytics helpers for the dashboard.

Equity-curve and trade statistics with no I/O, so they're unit-testable. The DB
layer pulls raw rows and calls these to build the summary payload.
"""

import math
from datetime import datetime, timezone

_MS_PER_DAY = 86_400_000
_TRADING_DAYS = 252  # for annualizing daily vol / Sharpe


def max_drawdown(values: list[float]) -> float:
    """Largest peak-to-trough decline as a fraction <= 0 (e.g. -0.18 = -18%)."""
    peak = None
    mdd = 0.0
    for v in values:
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = dd
    return mdd


def daily_last_equity(snapshots: list[tuple[int, float]]) -> list[float]:
    """Collapse (ts_ms, equity) snapshots (sorted asc) to one equity per UTC day."""
    by_day: dict[str, float] = {}
    for ts, equity in snapshots:
        day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        by_day[day] = equity  # later snapshot for the day wins
    return [by_day[d] for d in sorted(by_day)]


def returns(series: list[float]) -> list[float]:
    """Simple period-over-period returns of an equity series."""
    out = []
    for i in range(1, len(series)):
        if series[i - 1]:
            out.append(series[i] / series[i - 1] - 1.0)
    return out


def volatility(rets: list[float], periods: int = _TRADING_DAYS) -> float | None:
    """Annualized standard deviation of returns."""
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(periods)


def sharpe(rets: list[float], periods: int = _TRADING_DAYS) -> float | None:
    """Annualized Sharpe ratio (risk-free = 0)."""
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return None
    return (mean / sd) * math.sqrt(periods)


def best_worst(rets: list[float]) -> tuple[float | None, float | None]:
    """Best and worst single-day return."""
    if not rets:
        return (None, None)
    return (max(rets), min(rets))


def profit_factor(realized: list[float]) -> float | None:
    """Gross wins / gross losses. None if there are no losses yet."""
    gains = sum(r for r in realized if r > 0)
    losses = -sum(r for r in realized if r < 0)
    if losses <= 0:
        return None
    return gains / losses


def avg_win_loss(realized: list[float]) -> tuple[float | None, float | None]:
    """Average winning trade and average losing trade (loss is negative)."""
    wins = [r for r in realized if r > 0]
    losses = [r for r in realized if r < 0]
    aw = sum(wins) / len(wins) if wins else None
    al = sum(losses) / len(losses) if losses else None
    return aw, al


def streaks(realized: list[float]) -> tuple[int, int]:
    """Longest consecutive win and loss streaks (realized in chronological order)."""
    max_w = cur_w = max_l = cur_l = 0
    for r in realized:
        if r > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        elif r < 0:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
        else:
            cur_w = cur_l = 0
    return max_w, max_l


def active_days(first_ts_ms: int | None, last_ts_ms: int | None) -> float:
    """Span of the equity history in days (>= 1)."""
    if not first_ts_ms or not last_ts_ms or last_ts_ms <= first_ts_ms:
        return 1.0
    return max(1.0, (last_ts_ms - first_ts_ms) / _MS_PER_DAY)
