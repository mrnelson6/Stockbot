"""Tests for the dashboard SQLite persistence layer."""

import pytest

from stockbot.web import db


class TestResolveDashboardDb:
    def _clear(self, mp):
        for var in ("STOCKBOT_WEB_DB", "STOCKBOT_DATA_DIR", "RANDOM_ALPACA_PAPER", "ALPACA_PAPER"):
            mp.delenv(var, raising=False)

    def test_explicit_override_wins(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("STOCKBOT_WEB_DB", "/tmp/custom.db")
        assert db.resolve_dashboard_db() == "/tmp/custom.db"
        # even when a mode is passed
        assert db.resolve_dashboard_db(paper=False) == "/tmp/custom.db"

    def test_paper_and_live_paths(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("STOCKBOT_DATA_DIR", "/data")
        assert db.resolve_dashboard_db(paper=True).replace("\\", "/") == "/data/dashboard-paper.db"
        assert db.resolve_dashboard_db(paper=False).replace("\\", "/") == "/data/dashboard-live.db"

    def test_reads_paper_flag_from_env(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("STOCKBOT_DATA_DIR", "/data")
        monkeypatch.setenv("RANDOM_ALPACA_PAPER", "false")
        assert db.resolve_dashboard_db().replace("\\", "/") == "/data/dashboard-live.db"
        monkeypatch.setenv("RANDOM_ALPACA_PAPER", "true")
        assert db.resolve_dashboard_db().replace("\\", "/") == "/data/dashboard-paper.db"

    def test_defaults_to_paper(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("STOCKBOT_DATA_DIR", "/data")
        assert db.resolve_dashboard_db().replace("\\", "/") == "/data/dashboard-paper.db"


def test_init_and_empty_summary(tmp_path):
    p = tmp_path / "dash.db"
    db.init_db(p, label="Test Bot")
    s = db.get_summary(p)
    assert s["label"] == "Test Bot"
    assert s["equity"] is None
    assert s["n_positions"] == 0
    assert s["n_trades"] == 0
    assert db.get_positions(p) == []
    assert db.get_trades(p) == []
    assert db.get_equity_series(p) == []


def test_snapshot_and_returns(tmp_path):
    p = tmp_path / "dash.db"
    db.init_db(p)
    db.insert_snapshot(p, equity=100_000.0, cash=100_000.0, spy_price=400.0, ts=1_000)
    db.insert_snapshot(p, equity=110_000.0, cash=50_000.0, spy_price=420.0, ts=2_000)

    series = db.get_equity_series(p)
    assert [r["ts"] for r in series] == [1_000, 2_000]  # oldest first

    s = db.get_summary(p)
    assert s["equity"] == 110_000.0
    assert s["start_equity"] == 100_000.0
    assert abs(s["bot_return"] - 0.10) < 1e-9   # 100k -> 110k
    assert abs(s["spy_return"] - 0.05) < 1e-9   # 400 -> 420


def test_snapshot_handles_missing_spy(tmp_path):
    p = tmp_path / "dash.db"
    db.init_db(p)
    db.insert_snapshot(p, equity=100_000.0, cash=100_000.0, spy_price=None, ts=1_000)
    db.insert_snapshot(p, equity=120_000.0, cash=60_000.0, spy_price=None, ts=2_000)
    s = db.get_summary(p)
    assert abs(s["bot_return"] - 0.20) < 1e-9
    assert s["spy_return"] is None  # no SPY data -> no benchmark return


def test_replace_positions_is_wholesale(tmp_path):
    p = tmp_path / "dash.db"
    db.init_db(p)
    db.replace_positions(
        p,
        [
            {"symbol": "AAPL", "qty": 10, "avg_price": 190, "market_value": 1900, "unrealized_pnl": 50},
            {"symbol": "MSFT", "qty": 5, "avg_price": 400, "market_value": 2000, "unrealized_pnl": -20},
        ],
    )
    rows = db.get_positions(p)
    assert [r["symbol"] for r in rows] == ["MSFT", "AAPL"]  # by market value desc

    # A later snapshot replaces (not appends) the set.
    db.replace_positions(p, [{"symbol": "NVDA", "qty": 3, "avg_price": 120, "market_value": 360, "unrealized_pnl": 0}])
    rows = db.get_positions(p)
    assert [r["symbol"] for r in rows] == ["NVDA"]


def test_trades_append_and_order(tmp_path):
    p = tmp_path / "dash.db"
    db.init_db(p)
    db.insert_trade(p, symbol="AAPL", side="BUY", qty=10, price=190.0, ts=1_000, order_id="a")
    db.insert_trade(p, symbol="AAPL", side="SELL", qty=10, price=195.0, ts=2_000, order_id="b")
    trades = db.get_trades(p, limit=10)
    assert [t["side"] for t in trades] == ["SELL", "BUY"]  # newest first
    assert db.get_summary(p)["n_trades"] == 2

    limited = db.get_trades(p, limit=1)
    assert len(limited) == 1 and limited[0]["order_id"] == "b"
