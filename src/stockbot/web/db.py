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
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       INTEGER NOT NULL,
    symbol   TEXT NOT NULL,
    side     TEXT NOT NULL,
    qty      REAL NOT NULL,
    price    REAL NOT NULL,
    order_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


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


def init_db(path: str | Path, *, label: Optional[str] = None) -> None:
    """Create the schema if needed and optionally record a display label."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(_SCHEMA)
        if label is not None:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('label', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (label,),
            )


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
    """Append one executed trade to the log."""
    ts = ts if ts is not None else now_ms()
    with _connect(path) as conn:
        conn.execute(
            "INSERT INTO trades(ts, symbol, side, qty, price, order_id) VALUES(?,?,?,?,?,?)",
            (ts, symbol, side, qty, price, order_id),
        )


# ----- reads (called by the API) ------------------------------------------


def get_positions(path: str | Path) -> list[dict[str, Any]]:
    """Current holdings, largest market value first."""
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT symbol, qty, avg_price, market_value, unrealized_pnl, updated_ts "
            "FROM positions ORDER BY market_value DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_trades(path: str | Path, *, limit: int = 100) -> list[dict[str, Any]]:
    """Most recent trades, newest first."""
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT ts, symbol, side, qty, price, order_id FROM trades "
            "ORDER BY ts DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


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
    """Headline numbers: latest equity, returns, and SPY return since inception.

    Returns are computed from the first and latest snapshots. ``bot_return`` and
    ``spy_return`` are fractions (0.05 == +5%) so the frontend can compare them.
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

    summary: dict[str, Any] = {
        "label": label_row["value"] if label_row else "Random Bot",
        "equity": None,
        "cash": None,
        "start_equity": None,
        "start_ts": None,
        "last_ts": None,
        "bot_return": None,
        "spy_return": None,
        "n_positions": n_positions,
        "n_trades": n_trades,
    }
    if last:
        summary.update(equity=last["equity"], cash=last["cash"], last_ts=last["ts"])
    if first and last and first["equity"]:
        summary["start_equity"] = first["equity"]
        summary["start_ts"] = first["ts"]
        summary["bot_return"] = last["equity"] / first["equity"] - 1.0
        if first["spy_price"] and last["spy_price"]:
            summary["spy_return"] = last["spy_price"] / first["spy_price"] - 1.0
    return summary
