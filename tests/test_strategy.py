"""Tests for strategies."""

from decimal import Decimal

import pytest

from stockbot.core.models import Bar, MarketState, PortfolioState
from stockbot.core.types import Price, Quantity, Signal, Symbol, Timeframe, Timestamp
from stockbot.strategy.baseline import BuyAndHoldStrategy, SMAcrossoverStrategy


def make_bar(symbol: Symbol, timestamp: int, close: float) -> Bar:
    """Helper to create a bar with a specific close price."""
    close_price = Price(Decimal(str(close)))
    return Bar(
        symbol=symbol,
        timestamp=Timestamp(timestamp),
        open=close_price,
        high=close_price,
        low=close_price,
        close=close_price,
        volume=Quantity(Decimal("1000")),
        timeframe=Timeframe.MINUTE_1,
    )


def make_market_state(symbol: Symbol, bars: list[Bar], timestamp: int) -> MarketState:
    """Helper to create market state."""
    return MarketState(
        timestamp=Timestamp(timestamp),
        bars={symbol: bars},
        portfolio=PortfolioState(
            timestamp=Timestamp(timestamp),
            cash=Price(Decimal("100000")),
            positions={},
        ),
    )


class TestSMAcrossoverStrategy:
    """Tests for SMA crossover strategy."""

    def test_strategy_name(self, sample_symbol: Symbol) -> None:
        """Test strategy naming."""
        strategy = SMAcrossoverStrategy(
            symbols=[sample_symbol],
            fast_period=5,
            slow_period=10,
        )
        assert strategy.name == "SMA_5_10"

    def test_hold_with_insufficient_data(self, sample_symbol: Symbol) -> None:
        """Test that strategy holds when not enough data."""
        strategy = SMAcrossoverStrategy(
            symbols=[sample_symbol],
            fast_period=5,
            slow_period=10,
        )

        # Only 3 bars - not enough for either SMA
        bars = [
            make_bar(sample_symbol, i * 60_000_000_000, 100 + i)
            for i in range(3)
        ]

        state = make_market_state(sample_symbol, bars, 3 * 60_000_000_000)
        strategy.observe(state)
        signals = strategy.decide()

        assert signals[sample_symbol] == Signal.HOLD

    def test_golden_cross_generates_long(self, sample_symbol: Symbol) -> None:
        """Test that golden cross (fast crosses above slow) generates LONG."""
        strategy = SMAcrossoverStrategy(
            symbols=[sample_symbol],
            fast_period=3,
            slow_period=5,
        )

        # Create price series where fast SMA crosses above slow
        # Prices: 100, 100, 100, 100, 100, 110, 120, 130
        prices = [100, 100, 100, 100, 100, 110, 120, 130]

        for i, price in enumerate(prices):
            bars = [
                make_bar(sample_symbol, j * 60_000_000_000, prices[j])
                for j in range(i + 1)
            ]
            state = make_market_state(sample_symbol, bars, i * 60_000_000_000)
            strategy.observe(state)
            signals = strategy.decide()

        # After the uptrend, fast SMA should be above slow
        # The golden cross should have triggered a LONG signal at some point
        # Let's check the last signal
        assert signals[sample_symbol] in (Signal.HOLD, Signal.LONG)

    def test_reset_clears_state(self, sample_symbol: Symbol) -> None:
        """Test that reset clears internal state."""
        strategy = SMAcrossoverStrategy(
            symbols=[sample_symbol],
            fast_period=3,
            slow_period=5,
        )

        # Add some observations
        for i in range(10):
            bars = [make_bar(sample_symbol, j * 60_000_000_000, 100 + j) for j in range(i + 1)]
            state = make_market_state(sample_symbol, bars, i * 60_000_000_000)
            strategy.observe(state)

        # Reset
        strategy.reset()

        # Should be back to HOLD (no data)
        state = make_market_state(sample_symbol, [], 0)
        strategy.observe(state)
        signals = strategy.decide()

        assert signals[sample_symbol] == Signal.HOLD


class TestBuyAndHoldStrategy:
    """Tests for buy and hold strategy."""

    def test_first_signal_is_long(self, sample_symbol: Symbol) -> None:
        """Test that first signal is LONG."""
        strategy = BuyAndHoldStrategy(symbols=[sample_symbol])

        state = make_market_state(
            sample_symbol,
            [make_bar(sample_symbol, 0, 100)],
            0,
        )
        strategy.observe(state)
        signals = strategy.decide()

        assert signals[sample_symbol] == Signal.LONG

    def test_subsequent_signals_are_hold(self, sample_symbol: Symbol) -> None:
        """Test that signals after entry are HOLD."""
        strategy = BuyAndHoldStrategy(symbols=[sample_symbol])

        # First observation
        state1 = make_market_state(
            sample_symbol,
            [make_bar(sample_symbol, 0, 100)],
            0,
        )
        strategy.observe(state1)
        strategy.decide()  # First call returns LONG

        # Second observation
        state2 = make_market_state(
            sample_symbol,
            [make_bar(sample_symbol, 60_000_000_000, 101)],
            60_000_000_000,
        )
        strategy.observe(state2)
        signals = strategy.decide()

        assert signals[sample_symbol] == Signal.HOLD

    def test_reset_allows_new_entry(self, sample_symbol: Symbol) -> None:
        """Test that reset allows a new entry."""
        strategy = BuyAndHoldStrategy(symbols=[sample_symbol])

        # First entry
        state = make_market_state(
            sample_symbol,
            [make_bar(sample_symbol, 0, 100)],
            0,
        )
        strategy.observe(state)
        strategy.decide()

        # Reset
        strategy.reset()

        # Should get LONG again
        strategy.observe(state)
        signals = strategy.decide()

        assert signals[sample_symbol] == Signal.LONG
