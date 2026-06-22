"""Read-only FastAPI dashboard for the random bot.

Serves a small JSON API over the SQLite file the bot writes, plus a static
single-page frontend. This process is strictly read-only: it never holds Alpaca
credentials and exposes no way to place orders.

The DB path comes from the STOCKBOT_WEB_DB env var (default ./data/dashboard.db),
and must point at the same file the bot records to (``--record-db``).
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from stockbot.web import db

# Load .env so the DB path can follow the trading account (RANDOM_ALPACA_PAPER).
load_dotenv()
DB_PATH = db.resolve_dashboard_db()
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Random Bot Dashboard", docs_url=None, redoc_url=None)

# Public, read-only data: allow any origin to GET the API (e.g. if the frontend
# is ever hosted elsewhere). No credentials, no write methods.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _ensure_db() -> None:
    # Make sure the schema exists so reads don't error before the bot's first write.
    db.init_db(DB_PATH)


@app.get("/api/summary")
def summary() -> dict:
    """Headline numbers and bot-vs-SPY returns since inception."""
    return db.get_summary(DB_PATH)


@app.get("/api/positions")
def positions() -> list[dict]:
    """Current holdings, largest first."""
    return db.get_positions(DB_PATH)


@app.get("/api/trades")
def trades(limit: int = Query(100, ge=1, le=1000)) -> list[dict]:
    """Recent executed trades, newest first (sells include realized P&L)."""
    return db.get_trades(DB_PATH, limit=limit)


@app.get("/api/leaderboard")
def leaderboard(n: int = Query(5, ge=1, le=50)) -> dict:
    """All-time best and worst closed trades by realized P&L."""
    return db.get_leaderboard(DB_PATH, n=n)


@app.get("/api/equity")
def equity(since_ms: int | None = Query(None, ge=0)) -> list[dict]:
    """Equity + SPY time series (oldest first)."""
    return db.get_equity_series(DB_PATH, since_ms=since_ms)


# Frontend (index.html etc.). Mounted last so /api/* takes precedence.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
