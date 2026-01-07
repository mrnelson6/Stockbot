"""Baseline strategy implementations."""

from collections import deque
from decimal import Decimal
from typing import Optional

from stockbot.core.models import MarketState
from stockbot.core.types import Price, Signal, Symbol
from stockbot.strategy.base import BaseStrategy


class SMAcrossoverStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy.

    Classic trend-following strategy:
    - Go LONG when fast SMA crosses above slow SMA
    - Go FLAT when fast SMA crosses below slow SMA

    This is a baseline strategy for testing the system.
    Not intended for actual trading.
    """

    def __init__(
        self,
        symbols: list[Symbol],
        fast_period: int = 10,
        slow_period: int = 20,
    ) -> None:
        """Initialize the SMA crossover strategy.

        Args:
            symbols: Symbols to trade
            fast_period: Fast SMA period
            slow_period: Slow SMA period
        """
        super().__init__(symbols, strategy_name=f"SMA_{fast_period}_{slow_period}")
        self._fast_period = fast_period
        self._slow_period = slow_period

        # Price history for each symbol
        self._prices: dict[Symbol, deque[Price]] = {
            symbol: deque(maxlen=slow_period) for symbol in symbols
        }

        # Previous signal to detect crossovers
        self._previous_fast_above: dict[Symbol, Optional[bool]] = {
            symbol: None for symbol in symbols
        }

    def _on_observe(self, state: MarketState) -> None:
        """Update price history from market state."""
        for symbol in self._symbols:
            bar = state.latest_bar(symbol)
            if bar:
                self._prices[symbol].append(bar.close)

    def _calculate_sma(self, symbol: Symbol, period: int) -> Optional[Decimal]:
        """Calculate SMA for a symbol.

        Args:
            symbol: Symbol to calculate for
            period: SMA period

        Returns:
            SMA value or None if insufficient data
        """
        prices = self._prices[symbol]
        if len(prices) < period:
            return None

        recent = list(prices)[-period:]
        return sum(recent) / Decimal(period)

    def decide(self) -> dict[Symbol, Signal]:
        """Generate signals based on SMA crossover.

        Returns:
            Dict mapping symbols to signals
        """
        signals: dict[Symbol, Signal] = {}

        for symbol in self._symbols:
            signal = self._decide_for_symbol(symbol)
            signals[symbol] = signal

        return signals

    def _decide_for_symbol(self, symbol: Symbol) -> Signal:
        """Generate signal for a single symbol."""
        fast_sma = self._calculate_sma(symbol, self._fast_period)
        slow_sma = self._calculate_sma(symbol, self._slow_period)

        # Need both SMAs to make a decision
        if fast_sma is None or slow_sma is None:
            return Signal.HOLD

        fast_above = fast_sma > slow_sma
        previous = self._previous_fast_above[symbol]

        # Update state
        self._previous_fast_above[symbol] = fast_above

        # First observation - no crossover yet
        if previous is None:
            return Signal.HOLD

        # Detect crossovers
        if fast_above and not previous:
            # Golden cross - fast crossed above slow
            self._logger.signal(symbol, Signal.LONG, self.name, fast_sma=float(fast_sma), slow_sma=float(slow_sma))
            return Signal.LONG

        if not fast_above and previous:
            # Death cross - fast crossed below slow
            self._logger.signal(symbol, Signal.FLAT, self.name, fast_sma=float(fast_sma), slow_sma=float(slow_sma))
            return Signal.FLAT

        # No crossover
        return Signal.HOLD

    def _on_reset(self) -> None:
        """Reset strategy state."""
        for symbol in self._symbols:
            self._prices[symbol].clear()
            self._previous_fast_above[symbol] = None


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy.

    Buys on first bar and holds forever.
    Useful as a benchmark.
    """

    def __init__(self, symbols: list[Symbol]) -> None:
        super().__init__(symbols, strategy_name="BuyAndHold")
        self._has_entered: dict[Symbol, bool] = {s: False for s in symbols}

    def _on_observe(self, state: MarketState) -> None:
        """No observation needed for buy and hold."""
        pass

    def decide(self) -> dict[Symbol, Signal]:
        """Buy on first opportunity, then hold."""
        signals: dict[Symbol, Signal] = {}

        for symbol in self._symbols:
            if not self._has_entered[symbol]:
                signals[symbol] = Signal.LONG
                self._has_entered[symbol] = True
            else:
                signals[symbol] = Signal.HOLD

        return signals

    def _on_reset(self) -> None:
        """Reset entry state."""
        self._has_entered = {s: False for s in self._symbols}
