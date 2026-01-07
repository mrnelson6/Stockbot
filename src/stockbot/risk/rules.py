"""Risk rules for signal validation."""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

from stockbot.core.models import PortfolioState
from stockbot.core.types import Price, Quantity, Signal, Symbol


class RiskRule(ABC):
    """Abstract base class for risk rules.

    Each rule validates a specific aspect of risk management.
    Rules return (allowed, reason) where reason explains any rejection.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this rule for logging."""
        ...

    @abstractmethod
    def validate(
        self,
        signal: Signal,
        symbol: Symbol,
        portfolio: PortfolioState,
        price: Optional[Price] = None,
    ) -> tuple[bool, Optional[str]]:
        """Validate if signal is allowed.

        Args:
            signal: Proposed trading signal
            symbol: Symbol for the signal
            portfolio: Current portfolio state
            price: Current price (optional)

        Returns:
            Tuple of (allowed, rejection_reason)
        """
        ...


class MaxPositionSizeRule(RiskRule):
    """Limit maximum position size in shares."""

    def __init__(self, max_shares: Quantity) -> None:
        self._max_shares = max_shares

    @property
    def name(self) -> str:
        return "MaxPositionSize"

    def validate(
        self,
        signal: Signal,
        symbol: Symbol,
        portfolio: PortfolioState,
        price: Optional[Price] = None,
    ) -> tuple[bool, Optional[str]]:
        # CLOSE/FLAT signals always allowed
        if signal in (Signal.CLOSE, Signal.FLAT, Signal.HOLD):
            return True, None

        # Check current position
        position = portfolio.positions.get(symbol)
        current_qty = abs(position.quantity) if position else Decimal("0")

        if current_qty >= self._max_shares:
            return False, f"Position size {current_qty} at or exceeds max {self._max_shares}"

        return True, None


class MaxPositionValueRule(RiskRule):
    """Limit maximum position value in dollars."""

    def __init__(self, max_value: Price) -> None:
        self._max_value = max_value

    @property
    def name(self) -> str:
        return "MaxPositionValue"

    def validate(
        self,
        signal: Signal,
        symbol: Symbol,
        portfolio: PortfolioState,
        price: Optional[Price] = None,
    ) -> tuple[bool, Optional[str]]:
        # CLOSE/FLAT signals always allowed
        if signal in (Signal.CLOSE, Signal.FLAT, Signal.HOLD):
            return True, None

        if price is None:
            return True, None  # Can't validate without price

        position = portfolio.positions.get(symbol)
        if position:
            current_value = abs(position.quantity * price)
            if current_value >= self._max_value:
                return False, f"Position value ${current_value} at or exceeds max ${self._max_value}"

        return True, None


class MaxDailyLossRule(RiskRule):
    """Halt trading if daily loss exceeds threshold."""

    def __init__(self, max_loss: Price, initial_capital: Price) -> None:
        self._max_loss = max_loss
        self._initial_capital = initial_capital
        self._day_start_equity: Optional[Price] = None

    @property
    def name(self) -> str:
        return "MaxDailyLoss"

    def set_day_start_equity(self, equity: Price) -> None:
        """Set the equity at the start of the trading day."""
        self._day_start_equity = equity

    def validate(
        self,
        signal: Signal,
        symbol: Symbol,
        portfolio: PortfolioState,
        price: Optional[Price] = None,
    ) -> tuple[bool, Optional[str]]:
        # Always allow closing positions
        if signal in (Signal.CLOSE, Signal.FLAT):
            return True, None

        # HOLD doesn't open new positions
        if signal == Signal.HOLD:
            return True, None

        if self._day_start_equity is None:
            self._day_start_equity = self._initial_capital

        current_equity = portfolio.equity
        daily_pnl = current_equity - self._day_start_equity

        if daily_pnl <= -self._max_loss:
            return False, f"Daily loss ${abs(daily_pnl)} exceeds max ${self._max_loss}"

        return True, None


class MaxOpenPositionsRule(RiskRule):
    """Limit maximum number of concurrent positions."""

    def __init__(self, max_positions: int) -> None:
        self._max_positions = max_positions

    @property
    def name(self) -> str:
        return "MaxOpenPositions"

    def validate(
        self,
        signal: Signal,
        symbol: Symbol,
        portfolio: PortfolioState,
        price: Optional[Price] = None,
    ) -> tuple[bool, Optional[str]]:
        # CLOSE/FLAT signals always allowed
        if signal in (Signal.CLOSE, Signal.FLAT, Signal.HOLD):
            return True, None

        # Count current open positions
        open_positions = sum(
            1 for p in portfolio.positions.values() if not p.is_flat
        )

        # Check if we already have a position in this symbol
        has_position = (
            symbol in portfolio.positions
            and not portfolio.positions[symbol].is_flat
        )

        if not has_position and open_positions >= self._max_positions:
            return False, f"Open positions ({open_positions}) at max ({self._max_positions})"

        return True, None


class MaxDrawdownRule(RiskRule):
    """Halt new positions if drawdown exceeds threshold."""

    def __init__(
        self,
        max_drawdown_pct: Decimal,
        initial_equity: Price,
    ) -> None:
        """Initialize drawdown rule.

        Args:
            max_drawdown_pct: Maximum drawdown percentage (e.g., 10 for 10%)
            initial_equity: Starting equity for tracking
        """
        self._max_drawdown_pct = max_drawdown_pct
        self._peak_equity = initial_equity

    @property
    def name(self) -> str:
        return "MaxDrawdown"

    def update_peak(self, equity: Price) -> None:
        """Update peak equity if current is higher."""
        if equity > self._peak_equity:
            self._peak_equity = equity

    def validate(
        self,
        signal: Signal,
        symbol: Symbol,
        portfolio: PortfolioState,
        price: Optional[Price] = None,
    ) -> tuple[bool, Optional[str]]:
        # Always allow closing positions
        if signal in (Signal.CLOSE, Signal.FLAT):
            return True, None

        # HOLD doesn't open new positions
        if signal == Signal.HOLD:
            return True, None

        current_equity = portfolio.equity

        # Update peak
        self.update_peak(current_equity)

        # Calculate drawdown
        if self._peak_equity > Decimal("0"):
            drawdown = (self._peak_equity - current_equity) / self._peak_equity * 100
        else:
            drawdown = Decimal("0")

        if drawdown >= self._max_drawdown_pct:
            return False, f"Drawdown {drawdown:.1f}% exceeds max {self._max_drawdown_pct}%"

        return True, None
