"""Tests for the random trading bot's allocator."""

import pytest

from stockbot.core.types import Symbol
from stockbot.random_bot import RandomAllocator, TradeIntent, rank_by_liquidity


UNIVERSE = [Symbol(s) for s in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"]]
PRICES = {s: 100.0 for s in UNIVERSE}


def make_allocator(**kwargs) -> RandomAllocator:
    defaults = dict(seed=42, min_buys=2, max_buys=4, max_position_value=20_000.0)
    defaults.update(kwargs)
    return RandomAllocator(**defaults)


class TestConstruction:
    def test_rejects_bad_trade_prob(self):
        with pytest.raises(ValueError):
            RandomAllocator(trade_prob=1.5)

    def test_rejects_bad_buy_range(self):
        with pytest.raises(ValueError):
            RandomAllocator(min_buys=5, max_buys=2)

    def test_rejects_bad_deploy_range(self):
        with pytest.raises(ValueError):
            RandomAllocator(deploy_fraction_range=(0.8, 0.2))


class TestShouldTrade:
    def test_frequency_matches_trade_prob(self):
        alloc = make_allocator(trade_prob=0.25, seed=1)
        n = 20_000
        hits = sum(alloc.should_trade() for _ in range(n))
        assert 0.22 < hits / n < 0.28

    def test_prob_zero_never_trades(self):
        alloc = make_allocator(trade_prob=0.0)
        assert not any(alloc.should_trade() for _ in range(1000))

    def test_prob_one_always_trades(self):
        alloc = make_allocator(trade_prob=1.0)
        assert all(alloc.should_trade() for _ in range(1000))


class TestPlanTrade:
    def test_buys_respect_cash_and_position_cap(self):
        alloc = make_allocator(max_position_value=5_000.0, deploy_fraction_range=(1.0, 1.0))
        intent = alloc.plan_trade({}, PRICES, cash=100_000.0, universe=UNIVERSE)
        assert intent.buys
        assert sum(intent.buys.values()) <= 100_000.0 + 1e-6
        assert all(v <= 5_000.0 + 1e-6 for v in intent.buys.values())

    def test_buy_count_within_range_when_cap_not_binding(self):
        # Small cash relative to the per-position cap -> count stays in [min, max].
        alloc = make_allocator(min_buys=2, max_buys=4, max_position_value=20_000.0)
        for _ in range(50):
            intent = alloc.plan_trade({}, PRICES, cash=15_000.0, universe=UNIVERSE)
            assert 2 <= len(intent.buys) <= 4

    def test_deploys_nearly_all_cash(self):
        # With a non-binding cap and full deploy, Dirichlet targets sum to ~cash.
        alloc = make_allocator(deploy_fraction_range=(1.0, 1.0), max_position_value=1e12)
        for _ in range(20):
            intent = alloc.plan_trade({}, PRICES, cash=100_000.0, universe=UNIVERSE)
            assert abs(sum(intent.buys.values()) - 100_000.0) < 1.0  # essentially fully deployed

    def test_auto_picks_enough_names_to_absorb_cash(self):
        # cash/cap = 100k/20k = 5 -> at least 5 names even if min/max_buys are smaller.
        alloc = make_allocator(min_buys=1, max_buys=2, max_position_value=20_000.0,
                               deploy_fraction_range=(1.0, 1.0))
        intent = alloc.plan_trade({}, PRICES, cash=100_000.0, universe=UNIVERSE)
        assert len(intent.buys) >= 5
        assert abs(sum(intent.buys.values()) - 100_000.0) < 1.0

    def test_water_fill_redistributes_capped_overflow(self):
        # Cash far exceeds total capacity (8 names * 20k = 160k): all names cap out,
        # overflow is redistributed rather than dropped early.
        alloc = make_allocator(max_position_value=20_000.0, deploy_fraction_range=(1.0, 1.0))
        intent = alloc.plan_trade({}, PRICES, cash=1_000_000.0, universe=UNIVERSE)
        assert len(intent.buys) == len(UNIVERSE)
        assert all(abs(v - 20_000.0) < 1e-6 for v in intent.buys.values())
        assert abs(sum(intent.buys.values()) - 160_000.0) < 1e-6

    def test_churn_sells_are_subset_of_holdings(self):
        alloc = make_allocator(churn_sell_prob=0.5)
        holdings = {Symbol("AAPL"): 10, Symbol("MSFT"): 5, Symbol("JPM"): 3}
        for _ in range(50):
            intent = alloc.plan_trade(holdings, PRICES, cash=10_000.0, universe=UNIVERSE)
            for sym, qty in intent.sells.items():
                assert sym in holdings
                assert qty == holdings[sym]  # full exits

    def test_churn_prob_one_sells_everything(self):
        alloc = make_allocator(churn_sell_prob=1.0)
        holdings = {Symbol("AAPL"): 10, Symbol("MSFT"): 5}
        intent = alloc.plan_trade(holdings, PRICES, cash=0.0, universe=UNIVERSE)
        assert intent.sells == {Symbol("AAPL"): 10, Symbol("MSFT"): 5}

    def test_churn_prob_zero_sells_nothing(self):
        alloc = make_allocator(churn_sell_prob=0.0)
        holdings = {Symbol("AAPL"): 10, Symbol("MSFT"): 5}
        intent = alloc.plan_trade(holdings, PRICES, cash=50_000.0, universe=UNIVERSE)
        assert intent.sells == {}

    def test_determinism_same_seed(self):
        a = make_allocator(seed=7)
        b = make_allocator(seed=7)
        holdings = {Symbol("AAPL"): 10}
        ia = a.plan_trade(holdings, PRICES, cash=50_000.0, universe=UNIVERSE)
        ib = b.plan_trade(holdings, PRICES, cash=50_000.0, universe=UNIVERSE)
        assert ia.sells == ib.sells
        assert ia.buys == ib.buys


class TestEdgeCases:
    def test_zero_cash_no_holdings_produces_no_buys(self):
        alloc = make_allocator()
        intent = alloc.plan_trade({}, PRICES, cash=0.0, universe=UNIVERSE)
        assert intent.buys == {}
        assert intent.is_empty

    def test_freed_cash_funds_buys_when_starting_cash_zero(self):
        # No starting cash, but selling a held position frees cash to redeploy.
        alloc = make_allocator(churn_sell_prob=1.0, deploy_fraction_range=(1.0, 1.0))
        holdings = {Symbol("AAPL"): 100}  # 100 * $100 = $10k freed
        intent = alloc.plan_trade(holdings, PRICES, cash=0.0, universe=UNIVERSE)
        assert intent.sells == {Symbol("AAPL"): 100}
        assert intent.buys  # freed cash redeployed

    def test_k_larger_than_universe_is_clamped(self):
        small_universe = [Symbol("AAPL"), Symbol("MSFT")]
        small_prices = {s: 100.0 for s in small_universe}
        alloc = make_allocator(min_buys=5, max_buys=5)
        intent = alloc.plan_trade({}, small_prices, cash=100_000.0, universe=small_universe)
        assert len(intent.buys) <= len(small_universe)

    def test_names_without_prices_are_not_bought(self):
        prices = {Symbol("AAPL"): 100.0}  # only one priced
        alloc = make_allocator(min_buys=1, max_buys=8)
        for _ in range(20):
            intent = alloc.plan_trade({}, prices, cash=100_000.0, universe=UNIVERSE)
            assert all(sym == Symbol("AAPL") for sym in intent.buys)


class TestRankByLiquidity:
    # stats: symbol -> (last_price, avg_dollar_volume)
    STATS = {
        "BIG": (100.0, 500_000_000.0),   # very liquid
        "MID": (50.0, 50_000_000.0),     # liquid
        "SMALL": (20.0, 8_000_000.0),    # below default $-vol floor
        "PENNY": (1.5, 200_000_000.0),   # liquid but below price floor
    }

    def test_filters_by_price_floor(self):
        out = rank_by_liquidity(self.STATS, min_price=5.0, min_dollar_volume=0.0, max_symbols=10)
        assert "PENNY" not in out

    def test_filters_by_dollar_volume_floor(self):
        out = rank_by_liquidity(self.STATS, min_price=0.0, min_dollar_volume=10_000_000.0, max_symbols=10)
        assert "SMALL" not in out

    def test_sorted_by_dollar_volume_desc(self):
        out = rank_by_liquidity(self.STATS, min_price=5.0, min_dollar_volume=10_000_000.0, max_symbols=10)
        assert out == ["BIG", "MID"]

    def test_respects_max_symbols_cap(self):
        out = rank_by_liquidity(self.STATS, min_price=0.0, min_dollar_volume=0.0, max_symbols=2)
        assert len(out) == 2
        assert out[0] == "BIG"  # most liquid first

    def test_empty_when_nothing_qualifies(self):
        out = rank_by_liquidity(self.STATS, min_price=1000.0, min_dollar_volume=0.0, max_symbols=10)
        assert out == []
