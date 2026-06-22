"""SQLite persistence for the public dashboard.

One file holds three things:
- ``equity_snapshots``: the portfolio equity time series, with the matching SPY
  price at each point so the frontend can plot a normalized benchmark comparison.
- ``positions``: the *current* holdings (replaced wholesale on each snapshot).
- ``trades``: an append-only log of executed trades.

The DB is opened in WAL mode so the trading process (writer) and the web process
(reader) can use it concurrently. Every helper opens a short-lived connection,
which keeps things thread-safe under FastAPI without a shared connection/lock.
"""

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from stockbot.web import accounting, metrics

# Bump when the accounting algorithm changes to force a one-time rebuild.
_ACCOUNTING_VERSION = "2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts        INTEGER PRIMARY KEY,   -- unix milliseconds
    equity    REAL NOT NULL,
    cash      REAL NOT NULL,
    spy_price REAL                    -- nullable if SPY unavailable
);
CREATE TABLE IF NOT EXISTS positions (
    symbol        TEXT PRIMARY KEY,
    qty           REAL NOT NULL,
    avg_price     REAL NOT NULL,
    market_value  REAL NOT NULL,
    unrealized_pnl REAL,
    updated_ts    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    qty         REAL NOT NULL,
    price       REAL NOT NULL,
    order_id    TEXT,
    realized_pnl REAL,   -- SELLs only: proceeds - cost basis of matched lots
    cost_basis  REAL,    -- SELLs only: FIFO cost of the shares sold
    proceeds    REAL,    -- SELLs only: qty * price
    fees        REAL,    -- SELLs only: estimated SEC + FINRA regulatory fees
    hold_ms     REAL     -- SELLs only: share-weighted holding time of sold lots
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
-- Open buy lots for cost-basis / FIFO accounting (one row per surviving buy).
CREATE TABLE IF NOT EXISTS lots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT NOT NULL,
    qty       REAL NOT NULL,
    price     REAL NOT NULL,
    opened_ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lots_symbol ON lots(symbol, opened_ts);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Columns added after the first release, applied to existing DBs via _migrate.
_TRADE_MIGRATIONS = {
    "realized_pnl": "ALTER TABLE trades ADD COLUMN realized_pnl REAL",
    "cost_basis": "ALTER TABLE trades ADD COLUMN cost_basis REAL",
    "proceeds": "ALTER TABLE trades ADD COLUMN proceeds REAL",
    "fees": "ALTER TABLE trades ADD COLUMN fees REAL",
    "hold_ms": "ALTER TABLE trades ADD COLUMN hold_ms REAL",
}


def now_ms() -> int:
    """Current unix time in milliseconds."""
    return int(time.time() * 1000)


def resolve_dashboard_db(paper: Optional[bool] = None) -> str:
    """Pick the dashboard DB path so it follows the trading account.

    Both the bot (writer) and the web app (reader) call this, so flipping the
    account in .env (RANDOM_ALPACA_PAPER) points them at the same file without
    any other config change: paper -> dashboard-paper.db, live -> dashboard-live.db.

    Precedence:
    1. STOCKBOT_WEB_DB, if set, is an explicit override (used as-is).
    2. Otherwise <STOCKBOT_DATA_DIR or ./data>/dashboard-{paper|live}.db, where the
       mode comes from the ``paper`` arg if given, else RANDOM_ALPACA_PAPER /
       ALPACA_PAPER (default paper=true).
    """
    override = os.getenv("STOCKBOT_WEB_DB")
    if override:
        return override

    data_dir = os.getenv("STOCKBOT_DATA_DIR", "./data")
    if paper is None:
        flag = os.getenv("RANDOM_ALPACA_PAPER", os.getenv("ALPACA_PAPER", "true"))
        paper = flag.lower() == "true"
    suffix = "paper" if paper else "live"
    return str(Path(data_dir) / f"dashboard-{suffix}.db")


@contextmanager
def _connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection, commit on success, and always close.

    Note: `with sqlite3.connect(...)` commits but does NOT close the connection,
    which would leak handles in the long-running bot/web processes. This wrapper
    guarantees the connection is closed.
    """
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first release to an existing DB."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(trades)")}
    for column, ddl in _TRADE_MIGRATIONS.items():
        if column not in existing:
            conn.execute(ddl)


def _rebuild_accounting(conn: sqlite3.Connection) -> None:
    """Recompute lots and per-sell realized P&L by replaying the whole trade log.

    Used to backfill an existing DB after an upgrade, and as a self-heal if the
    accounting version changes. Idempotent.
    """
    trades = [
        dict(r)
        for r in conn.execute(
            "SELECT id, ts, symbol, side, qty, price FROM trades ORDER BY ts ASC, id ASC"
        )
    ]
    open_lots, sell_results = accounting.replay_trades(trades)

    conn.execute("DELETE FROM lots")
    conn.execute("UPDATE trades SET realized_pnl=NULL, cost_basis=NULL, proceeds=NULL")
    for res in sell_results:
        conn.execute(
            "UPDATE trades SET realized_pnl=?, cost_basis=?, proceeds=?, fees=?, hold_ms=? WHERE id=?",
            (res["realized"], res["cost_basis"], res["proceeds"], res["fees"], res["hold_ms"], res["id"]),
        )
    for symbol, lots in open_lots.items():
        for lot in lots:
            conn.execute(
                "INSERT INTO lots(symbol, qty, price, opened_ts) VALUES(?,?,?,?)",
                (symbol, lot["qty"], lot["price"], lot["opened_ts"]),
            )


def init_db(path: str | Path, *, label: Optional[str] = None) -> None:
    """Create/upgrade the schema, backfill accounting once, set the label."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        if label is not None:
            _set_meta(conn, "label", label)
        # One-time (per version) backfill of lots + realized P&L for existing data.
        built = conn.execute(
            "SELECT value FROM meta WHERE key='accounting_version'"
        ).fetchone()
        if not built or built["value"] != _ACCOUNTING_VERSION:
            _rebuild_accounting(conn)
            _set_meta(conn, "accounting_version", _ACCOUNTING_VERSION)


# ----- writes (called by the recorder) ------------------------------------


def insert_snapshot(
    path: str | Path,
    *,
    equity: float,
    cash: float,
    spy_price: Optional[float],
    ts: Optional[int] = None,
) -> None:
    """Append one equity/SPY datapoint to the time series."""
    ts = ts if ts is not None else now_ms()
    with _connect(path) as conn:
        conn.execute(
            "INSERT INTO equity_snapshots(ts, equity, cash, spy_price) VALUES(?,?,?,?) "
            "ON CONFLICT(ts) DO UPDATE SET equity=excluded.equity, cash=excluded.cash, "
            "spy_price=excluded.spy_price",
            (ts, equity, cash, spy_price),
        )


def replace_positions(
    path: str | Path,
    positions: list[dict[str, Any]],
    *,
    ts: Optional[int] = None,
) -> None:
    """Replace the current-positions table with the latest snapshot."""
    ts = ts if ts is not None else now_ms()
    with _connect(path) as conn:
        conn.execute("DELETE FROM positions")
        conn.executemany(
            "INSERT INTO positions(symbol, qty, avg_price, market_value, unrealized_pnl, updated_ts) "
            "VALUES(:symbol, :qty, :avg_price, :market_value, :unrealized_pnl, :updated_ts)",
            [
                {
                    "symbol": p["symbol"],
                    "qty": p["qty"],
                    "avg_price": p["avg_price"],
                    "market_value": p["market_value"],
                    "unrealized_pnl": p.get("unrealized_pnl"),
                    "updated_ts": ts,
                }
                for p in positions
            ],
        )


def insert_trade(
    path: str | Path,
    *,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    order_id: Optional[str] = None,
    ts: Optional[int] = None,
) -> None:
    """Append one executed trade and update FIFO cost-basis lots.

    A BUY opens a lot. A SELL consumes the oldest lots first; its realized P&L
    (proceeds - cost basis of the matched shares) is stored on the trade row.
    """
    ts = ts if ts is not None else now_ms()
    side_u = side.upper()
    with _connect(path) as conn:
        if side_u == "BUY":
            conn.execute(
                "INSERT INTO trades(ts, symbol, side, qty, price, order_id) VALUES(?,?,?,?,?,?)",
                (ts, symbol, side_u, qty, price, order_id),
            )
            conn.execute(
                "INSERT INTO lots(symbol, qty, price, opened_ts) VALUES(?,?,?,?)",
                (symbol, qty, price, ts),
            )
            return

        # SELL: match against open lots FIFO.
        lot_rows = [
            dict(r)
            for r in conn.execute(
                "SELECT id, qty, price, opened_ts FROM lots WHERE symbol=? ORDER BY opened_ts ASC, id ASC",
                (symbol,),
            )
        ]
        res = accounting.fifo_sell(lot_rows, qty, price)
        fees = accounting.estimate_fees(res["proceeds"], res["matched_qty"])
        hold_ms = accounting.weighted_hold_ms(res["consumed"], ts)
        conn.execute(
            "INSERT INTO trades(ts, symbol, side, qty, price, order_id, realized_pnl, cost_basis, "
            "proceeds, fees, hold_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (ts, symbol, side_u, qty, price, order_id,
             res["realized"], res["cost_basis"], res["proceeds"], fees, hold_ms),
        )
        # Persist the consumed lots: delete this symbol's lots and re-insert survivors.
        conn.execute("DELETE FROM lots WHERE symbol=?", (symbol,))
        for lot in res["remaining_lots"]:
            conn.execute(
                "INSERT INTO lots(symbol, qty, price, opened_ts) VALUES(?,?,?,?)",
                (symbol, lot["qty"], lot["price"], lot["opened_ts"]),
            )


# ----- reads (called by the API) ------------------------------------------


def get_positions(path: str | Path) -> list[dict[str, Any]]:
    """Current holdings enriched with cost basis, acquisition dates and P&L.

    Cost basis and acquisition dates come from the FIFO lots (our trade history);
    current market value comes from the latest broker snapshot. Each row adds:
    cost_basis_per_share, cost_basis_total, unrealized_pnl, unrealized_pnl_pct,
    acquired_first_ts, acquired_last_ts, n_lots.
    """
    with _connect(path) as conn:
        positions = [
            dict(r)
            for r in conn.execute(
                "SELECT symbol, qty, avg_price, market_value, unrealized_pnl, updated_ts "
                "FROM positions ORDER BY market_value DESC"
            )
        ]
        lot_agg = {
            r["symbol"]: dict(r)
            for r in conn.execute(
                "SELECT symbol, SUM(qty*price) AS cost_total, SUM(qty) AS lots_qty, "
                "MIN(opened_ts) AS acquired_first, MAX(opened_ts) AS acquired_last, "
                "COUNT(*) AS n_lots FROM lots GROUP BY symbol"
            )
        }

    for p in positions:
        agg = lot_agg.get(p["symbol"])
        if agg and agg["lots_qty"] and agg["lots_qty"] > 0:
            cost_total = float(agg["cost_total"])
            cost_per_share = cost_total / float(agg["lots_qty"])
            p["acquired_first_ts"] = agg["acquired_first"]
            p["acquired_last_ts"] = agg["acquired_last"]
            p["n_lots"] = agg["n_lots"]
        else:
            # No lots (e.g. pre-existing shares) -> fall back to broker avg price.
            cost_per_share = p["avg_price"]
            cost_total = p["avg_price"] * p["qty"]
            p["acquired_first_ts"] = p["updated_ts"]
            p["acquired_last_ts"] = p["updated_ts"]
            p["n_lots"] = 1
        unrealized = p["market_value"] - cost_total
        p["cost_basis_per_share"] = cost_per_share
        p["cost_basis_total"] = cost_total
        p["unrealized_pnl"] = unrealized
        p["unrealized_pnl_pct"] = (unrealized / cost_total) if cost_total > 1e-9 else None
    return positions


def get_trades(path: str | Path, *, limit: int = 100) -> list[dict[str, Any]]:
    """Most recent trades, newest first, with realized P&L on sells."""
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT ts, symbol, side, qty, price, order_id, realized_pnl, cost_basis, proceeds, "
            "fees, hold_ms FROM trades ORDER BY ts DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        cb = d.get("cost_basis")
        d["realized_pnl_pct"] = (
            d["realized_pnl"] / cb if (d.get("realized_pnl") is not None and cb and cb > 1e-9) else None
        )
        out.append(d)
    return out


def get_leaderboard(path: str | Path, *, n: int = 5) -> dict[str, list[dict[str, Any]]]:
    """All-time best and worst closed trades by realized P&L."""

    def _rows(order: str) -> list[dict[str, Any]]:
        with _connect(path) as conn:
            rows = conn.execute(
                "SELECT ts, symbol, qty, price, realized_pnl, cost_basis, proceeds "
                f"FROM trades WHERE side='SELL' AND realized_pnl IS NOT NULL "
                f"ORDER BY realized_pnl {order} LIMIT ?",
                (n,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            cb = d.get("cost_basis")
            d["realized_pnl_pct"] = d["realized_pnl"] / cb if cb and cb > 1e-9 else None
            result.append(d)
        return result

    return {"best": _rows("DESC"), "worst": _rows("ASC")}


def get_equity_series(
    path: str | Path, *, since_ms: Optional[int] = None, max_points: int = 5000
) -> list[dict[str, Any]]:
    """Equity + SPY time series (oldest first), optionally bounded by time."""
    query = "SELECT ts, equity, cash, spy_price FROM equity_snapshots"
    params: tuple = ()
    if since_ms is not None:
        query += " WHERE ts >= ?"
        params = (since_ms,)
    query += " ORDER BY ts ASC LIMIT ?"
    params = params + (max_points,)
    with _connect(path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_summary(path: str | Path) -> dict[str, Any]:
    """Headline numbers and full performance analytics for the dashboard.

    Returns/percentages are fractions (0.05 == +5%). P&L and fees are dollars.
    Risk stats (volatility, Sharpe, max drawdown, best/worst day) are derived
    from the daily equity curve; trade stats from realized sells.
    """
    with _connect(path) as conn:
        label_row = conn.execute("SELECT value FROM meta WHERE key='label'").fetchone()
        first = conn.execute(
            "SELECT ts, equity, spy_price FROM equity_snapshots ORDER BY ts ASC LIMIT 1"
        ).fetchone()
        last = conn.execute(
            "SELECT ts, equity, cash, spy_price FROM equity_snapshots ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        n_positions = conn.execute("SELECT COUNT(*) AS n FROM positions").fetchone()["n"]
        n_trades = conn.execute("SELECT COUNT(*) AS n FROM trades").fetchone()["n"]
        fees_total = conn.execute(
            "SELECT COALESCE(SUM(fees), 0) AS f FROM trades WHERE side='SELL'"
        ).fetchone()["f"]
        # Closed sells in chronological order for streaks / averages.
        sells = [
            dict(r)
            for r in conn.execute(
                "SELECT realized_pnl, qty, hold_ms FROM trades "
                "WHERE side='SELL' AND realized_pnl IS NOT NULL ORDER BY ts ASC, id ASC"
            )
        ]
        equity_curve = [
            (r["ts"], r["equity"])
            for r in conn.execute("SELECT ts, equity FROM equity_snapshots ORDER BY ts ASC")
        ]

    realized_list = [s["realized_pnl"] for s in sells]
    wins = sum(1 for r in realized_list if r > 0)
    losses = sum(1 for r in realized_list if r < 0)
    closed = wins + losses
    avg_win, avg_loss = metrics.avg_win_loss(realized_list)
    longest_win, longest_loss = metrics.streaks(realized_list)

    # Share-weighted average holding period (ms -> days at the frontend).
    hold_num = sum((s["hold_ms"] or 0) * s["qty"] for s in sells if s["hold_ms"] is not None)
    hold_den = sum(s["qty"] for s in sells if s["hold_ms"] is not None)
    avg_hold_ms = (hold_num / hold_den) if hold_den > 0 else None

    # Daily-equity-derived risk stats.
    daily = metrics.daily_last_equity(equity_curve)
    daily_rets = metrics.returns(daily)
    best_day, worst_day = metrics.best_worst(daily_rets)

    # Position concentration / cash.
    positions = get_positions(path)
    unrealized_total = sum(
        p["unrealized_pnl"] for p in positions if p.get("unrealized_pnl") is not None
    )
    total_mv = sum(p["market_value"] for p in positions)
    largest_mv = max((p["market_value"] for p in positions), default=0.0)

    realized_total = sum(realized_list)
    equity = last["equity"] if last else None
    start_equity = first["equity"] if first else None

    summary: dict[str, Any] = {
        "label": label_row["value"] if label_row else "Random Bot",
        "equity": equity,
        "cash": last["cash"] if last else None,
        "start_equity": start_equity,
        "start_ts": first["ts"] if first else None,
        "last_ts": last["ts"] if last else None,
        "bot_return": None,
        "spy_return": None,
        "n_positions": n_positions,
        "n_trades": n_trades,
        # P&L
        "realized_pnl": realized_total,
        "unrealized_pnl": unrealized_total,
        "total_pnl": realized_total + unrealized_total,
        "fees_total": fees_total or 0.0,
        # trade stats
        "n_wins": wins,
        "n_losses": losses,
        "win_rate": (wins / closed) if closed else None,
        "profit_factor": metrics.profit_factor(realized_list),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "longest_win_streak": longest_win,
        "longest_loss_streak": longest_loss,
        "avg_hold_ms": avg_hold_ms,
        "trades_per_day": (
            n_trades / metrics.active_days(first["ts"] if first else None, last["ts"] if last else None)
            if n_trades
            else 0.0
        ),
        # risk / curve
        "max_drawdown": metrics.max_drawdown([e for _, e in equity_curve]) if equity_curve else None,
        "volatility": metrics.volatility(daily_rets),
        "sharpe": metrics.sharpe(daily_rets),
        "best_day": best_day,
        "worst_day": worst_day,
        # exposure
        "largest_position_pct": (largest_mv / equity) if equity else None,
        "cash_pct": (last["cash"] / equity) if (last and equity) else None,
        "invested_pct": (total_mv / equity) if equity else None,
    }
    if first and last and start_equity:
        summary["bot_return"] = equity / start_equity - 1.0
        summary["total_pnl_pct"] = summary["total_pnl"] / start_equity
        if first["spy_price"] and last["spy_price"]:
            summary["spy_return"] = last["spy_price"] / first["spy_price"] - 1.0
    else:
        summary["total_pnl_pct"] = None
    return summary
