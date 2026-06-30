"""Tests for the dashboard SQLite persistence layer."""

import pytest

from stockbot.web import accounting, db, metrics


class TestMetrics:
    def test_max_drawdown(self):
        assert metrics.max_drawdown([100, 120, 90, 150, 75]) == pytest.approx(-0.5)
        assert metrics.max_drawdown([100, 110, 120]) == pytest.approx(0.0)

    def test_profit_factor(self):
        assert metrics.profit_factor([10, -5, 20, -5]) == pytest.approx(3.0)
        assert metrics.profit_factor([10, 20]) is None  # no losses

    def test_avg_win_loss(self):
        aw, al = metrics.avg_win_loss([10, -4, 20, -6])
        assert aw == pytest.approx(15) and al == pytest.approx(-5)

    def test_streaks(self):
        assert metrics.streaks([1, 1, -1, 1, -1, -1, -1, 1]) == (2, 3)

    def test_daily_last_equity_buckets_by_day(self):
        day1, day2 = 1_000, 1_000 + 86_400_000
        snaps = [(day1, 100.0), (day1 + 60_000, 105.0), (day2, 110.0)]
        assert metrics.daily_last_equity(snaps) == [105.0, 110.0]  # last of day1, then day2

    def test_volatility_and_sharpe_need_two_points(self):
        assert metrics.volatility([0.01]) is None
        assert metrics.sharpe([0.01, 0.02, -0.01]) is not None

    def test_period_return_uses_value_at_cutoff(self):
        # value 100 @ t=1000, 110 @ t=2000, 120 @ t=3000 (latest).
        pts = [(1_000, 100.0), (2_000, 110.0), (3_000, 120.0)]
        # Cutoff at t=2000 -> baseline 110 -> 120/110 - 1.
        assert metrics.period_return(pts, 2_000) == pytest.approx(120 / 110 - 1)
        # Cutoff between points takes the last value at or before it.
        assert metrics.period_return(pts, 2_500) == pytest.approx(120 / 110 - 1)

    def test_period_return_anchors_to_inception_when_cutoff_predates(self):
        pts = [(1_000, 100.0), (2_000, 120.0)]
        # Cutoff before all history -> baseline = first value (since inception).
        assert metrics.period_return(pts, 0) == pytest.approx(0.20)

    def test_period_return_skips_null_values(self):
        # SPY price missing for some snapshots; baseline/latest skip the Nones.
        pts = [(1_000, None), (2_000, 400.0), (3_000, None), (4_000, 440.0)]
        assert metrics.period_return(pts, 2_000) == pytest.approx(0.10)
        assert metrics.period_return([(1, None), (2, None)], 1) is None


class TestFees:
    def test_estimate_fees(self):
        # $10k proceeds, 100 shares: SEC ~ 10000*0.0000278, TAF ~ 100*0.000166
        f = accounting.estimate_fees(10_000.0, 100)
        assert f == pytest.approx(10_000 * 0.0000278 + 100 * 0.000166)

    def test_taf_is_capped(self):
        f = accounting.estimate_fees(0.0, 1_000_000)  # huge share count
        assert f == pytest.approx(accounting.TAF_CAP)

    def test_weighted_hold_ms(self):
        consumed = [{"qty": 10, "opened_ts": 0}, {"qty": 30, "opened_ts": 100}]
        # weighted: (10*200 + 30*100)/40 = 125  (sell_ts=200)
        assert accounting.weighted_hold_ms(consumed, 200) == pytest.approx(125.0)


class TestFifoAccounting:
    def test_spans_multiple_lots(self):
        lots = [
            {"qty": 10, "price": 100.0, "opened_ts": 1},
            {"qty": 5, "price": 120.0, "opened_ts": 2},
        ]
        r = accounting.fifo_sell(lots, 12, 130.0)
        assert r["cost_basis"] == pytest.approx(10 * 100 + 2 * 120)  # 1240
        assert r["proceeds"] == pytest.approx(12 * 130)              # 1560
        assert r["realized"] == pytest.approx(320.0)
        assert r["matched_qty"] == pytest.approx(12)
        assert r["remaining_lots"] == [{"qty": 3, "price": 120.0, "opened_ts": 2}]

    def test_partial_single_lot(self):
        r = accounting.fifo_sell([{"qty": 10, "price": 50.0, "opened_ts": 1}], 4, 60.0)
        assert r["realized"] == pytest.approx(4 * (60 - 50))
        assert r["remaining_lots"][0]["qty"] == pytest.approx(6)

    def test_under_lotted_sell_matches_what_exists(self):
        # Selling more than we hold: only matched shares count toward basis.
        r = accounting.fifo_sell([{"qty": 5, "price": 100.0, "opened_ts": 1}], 10, 110.0)
        assert r["matched_qty"] == pytest.approx(5)
        assert r["cost_basis"] == pytest.approx(500)
        assert r["proceeds"] == pytest.approx(5 * 110)  # only matched shares
        assert r["remaining_lots"] == []

    def test_replay_builds_lots_and_sell_results(self):
        trades = [
            {"id": 1, "ts": 1, "symbol": "AAPL", "side": "BUY", "qty": 10, "price": 100.0},
            {"id": 2, "ts": 2, "symbol": "AAPL", "side": "BUY", "qty": 10, "price": 110.0},
            {"id": 3, "ts": 3, "symbol": "AAPL", "side": "SELL", "qty": 15, "price": 120.0},
        ]
        lots, sells = accounting.replay_trades(trades)
        # 15 sold FIFO: 10@100 + 5@110 = 1550 basis; 5@110 left.
        assert sells[0]["id"] == 3
        assert sells[0]["realized"] == pytest.approx(15 * 120 - (10 * 100 + 5 * 110))
        assert lots["AAPL"] == [{"qty": 5, "price": 110.0, "opened_ts": 2}]


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


def test_period_returns_windows():
    # now = 2026-06-30 12:00 UTC. Snapshots: ~1y ago, ~1w ago, start-of-today, now.
    DAY = 86_400_000
    now = 1_782_820_800_000  # 2026-06-30T12:00:00Z
    snaps = [
        (now - 400 * DAY, 100_000.0, 400.0),  # before the 1Y window
        (now - 8 * DAY, 110_000.0, 420.0),     # before the 1W window
        (now - DAY, 120_000.0, 440.0),         # yesterday (today's baseline)
        (now, 132_000.0, 462.0),               # latest
    ]
    by_key = {p["key"]: p for p in db._period_returns(snaps, now=now)}
    # Today: 132k vs yesterday's 120k = +10%; SPY 462 vs 440 = +5%.
    assert by_key["1d"]["bot"] == pytest.approx(0.10)
    assert by_key["1d"]["spy"] == pytest.approx(0.05)
    # 1W baseline is the 8-day-old snapshot (110k / 420).
    assert by_key["1w"]["bot"] == pytest.approx(132 / 110 - 1)
    assert by_key["1w"]["spy"] == pytest.approx(462 / 420 - 1)
    # 1Y baseline is the oldest snapshot (within a year).
    assert by_key["1y"]["bot"] == pytest.approx(0.32)
    # 5Y window predates all history -> anchors to inception (same as 1Y here).
    assert by_key["5y"]["bot"] == pytest.approx(0.32)
    assert [p["label"] for p in db._period_returns(snaps, now=now)] == [
        "Today", "1W", "1M", "YTD", "1Y", "5Y"
    ]


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


def test_realized_pnl_on_sell(tmp_path):
    p = tmp_path / "dash.db"
    db.init_db(p)
    db.insert_trade(p, symbol="AAPL", side="BUY", qty=10, price=100.0, ts=1_000)
    db.insert_trade(p, symbol="AAPL", side="BUY", qty=10, price=120.0, ts=2_000)
    db.insert_trade(p, symbol="AAPL", side="SELL", qty=15, price=130.0, ts=3_000)

    sell = db.get_trades(p, limit=1)[0]
    assert sell["side"] == "SELL"
    assert sell["realized_pnl"] == pytest.approx(15 * 130 - (10 * 100 + 5 * 120))  # 350
    assert sell["cost_basis"] == pytest.approx(1600)
    assert sell["realized_pnl_pct"] == pytest.approx(350 / 1600)

    s = db.get_summary(p)
    assert s["realized_pnl"] == pytest.approx(350)
    assert s["n_wins"] == 1 and s["n_losses"] == 0
    assert s["win_rate"] == pytest.approx(1.0)
    # New analytics present and sane.
    assert sell["fees"] is not None and sell["fees"] > 0
    assert s["fees_total"] == pytest.approx(sell["fees"])
    assert s["total_pnl"] == pytest.approx(s["realized_pnl"] + s["unrealized_pnl"])
    assert s["longest_win_streak"] == 1 and s["longest_loss_streak"] == 0
    # FIFO: 10 sh held 2000ms (lot@1000) + 5 sh held 1000ms (lot@2000), sold ts 3000.
    assert s["avg_hold_ms"] == pytest.approx(25_000 / 15)


def test_leaderboard_best_and_worst(tmp_path):
    p = tmp_path / "dash.db"
    db.init_db(p)
    # winner: buy 1@100 sell 1@150 (+50); loser: buy 1@100 sell 1@70 (-30)
    db.insert_trade(p, symbol="WIN", side="BUY", qty=1, price=100.0, ts=1)
    db.insert_trade(p, symbol="WIN", side="SELL", qty=1, price=150.0, ts=2)
    db.insert_trade(p, symbol="LOSE", side="BUY", qty=1, price=100.0, ts=3)
    db.insert_trade(p, symbol="LOSE", side="SELL", qty=1, price=70.0, ts=4)

    lb = db.get_leaderboard(p, n=5)
    assert lb["best"][0]["symbol"] == "WIN"
    assert lb["best"][0]["realized_pnl"] == pytest.approx(50)
    assert lb["worst"][0]["symbol"] == "LOSE"
    assert lb["worst"][0]["realized_pnl"] == pytest.approx(-30)


def test_positions_enriched_with_cost_basis(tmp_path):
    p = tmp_path / "dash.db"
    db.init_db(p)
    # Two buys -> avg cost 110; a snapshot marks current value at 130/share.
    db.insert_trade(p, symbol="AAPL", side="BUY", qty=10, price=100.0, ts=1_000)
    db.insert_trade(p, symbol="AAPL", side="BUY", qty=10, price=120.0, ts=2_000)
    db.replace_positions(
        p,
        [{"symbol": "AAPL", "qty": 20, "avg_price": 110.0, "market_value": 2600.0, "unrealized_pnl": 0}],
        ts=3_000,
    )
    pos = db.get_positions(p)[0]
    assert pos["cost_basis_per_share"] == pytest.approx(110.0)
    assert pos["cost_basis_total"] == pytest.approx(2200.0)
    assert pos["unrealized_pnl"] == pytest.approx(400.0)            # 2600 - 2200
    assert pos["unrealized_pnl_pct"] == pytest.approx(400 / 2200)
    assert pos["acquired_first_ts"] == 1_000
    assert pos["n_lots"] == 2


def test_backfill_rebuilds_accounting_for_legacy_rows(tmp_path):
    p = tmp_path / "dash.db"
    db.init_db(p)
    # Simulate pre-accounting rows: trades present but no lots / realized, and the
    # version marker absent so init_db's backfill must reconstruct them.
    with db._connect(p) as conn:
        conn.execute("DELETE FROM lots")
        conn.execute("INSERT INTO trades(ts,symbol,side,qty,price) VALUES(1,'AAPL','BUY',10,100.0)")
        conn.execute("INSERT INTO trades(ts,symbol,side,qty,price) VALUES(2,'AAPL','SELL',4,150.0)")
        conn.execute("DELETE FROM meta WHERE key='accounting_version'")
    db.init_db(p)  # should backfill
    sell = [t for t in db.get_trades(p) if t["side"] == "SELL"][0]
    assert sell["realized_pnl"] == pytest.approx(4 * (150 - 100))   # 200
    # 6 shares remain as an open lot
    pos_lots = db.get_positions  # ensure lots exist via leaderboard/summary path
    assert db.get_summary(p)["realized_pnl"] == pytest.approx(200)
