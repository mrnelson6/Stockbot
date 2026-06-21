"""Public read-only web dashboard for the random bot.

A small, self-hosted stack that exposes the bot's portfolio publicly:
- ``db``: SQLite persistence (equity time series, latest positions, trade log).
- ``recorder``: snapshots the live account (and SPY benchmark) each tick.
- ``app``: a read-only FastAPI app + static single-page frontend.

The trading process (writer) and the web process (reader) run separately and
share one SQLite file in WAL mode. The web side never touches Alpaca credentials.
"""
