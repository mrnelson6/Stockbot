"""Risk manager implementation."""

from decimal import Decimal
from typing import Optional

from stockbot.config.settings import RiskConfig
from stockbot.core.models import PortfolioState
from stockbot.core.types import Price, Quantity, Signal, Symbol
from stockbot.monitoring.logger import get_logger
from stockbot.risk.rules import (
    MaxDailyLossRule,
    MaxOpenPositionsRule,
    MaxPositionSizeRule,
    MaxPositionValueRule,
    RiskRule,
)

logger = get_logger("risk")


class BasicRiskManager:
    """Basic risk manager with configurable rules.

    Validates signals against risk rules and calculates position sizes.
    """

    def __init__(self, config: RiskConfig) -> None:
        """Initialize the risk manager.

        Args:
            config: Risk configuration settings
        """
        self._config = config
        self._emergency_stop = False
        self._initial_capital: Price = Price(Decimal("100000"))  # Default

        # Initialize rules
        self._rules: list[RiskRule] = [
            MaxPositionSizeRule(config.max_position_size),
            MaxPositionValueRule(config.max_position_value),
            MaxDailyLossRule(config.max_daily_loss, self._initial_capital),
            MaxOpenPositionsRule(config.max_open_positions),
        ]

    def set_initial_capital(self, capital: Price) -> None:
        """Set initial capital for PnL calculations."""
        self._initial_capital = capital
        # Update daily loss rule
        for rule in self._rules:
            if isinstance(rule, MaxDailyLossRule):
                rule._initial_capital = capital

    def set_day_start_equity(self, equity: Price) -> None:
        """Set equity at start of trading day."""
        for rule in self._rules:
            if isinstance(rule, MaxDailyLossRule):
                rule.set_day_start_equity(equity)

    def validate_signal(
        self,
        signal: Signal,
        symbol: Symbol,
        portfolio: PortfolioState,
        price: Optional[Price] = None,
    ) -> tuple[bool, Optional[str]]:
        """Validate if a signal is allowed.

        Args:
            signal: Proposed trading signal
            symbol: Symbol for the signal
            portfolio: Current portfolio state
            price: Current price (optional but recommended)

        Returns:
            Tuple of (allowed, rejection_reason)
        """
        # Check emergency stop first
        if self._emergency_stop:
            if signal not in (Signal.CLOSE, Signal.FLAT):
                return False, "Emergency stop active - only closing positions allowed"

        # Check each rule
        for rule in self._rules:
            allowed, reason = rule.validate(signal, symbol, portfolio, price)
            if not allowed:
                logger.risk_rejection(symbol, signal, f"{rule.name}: {reason}")
                return False, f"{rule.name}: {reason}"

        return True, None

    def calculate_position_size(
        self,
        signal: Signal,
        symbol: Symbol,
        price: Price,
        portfolio: PortfolioState,
    ) -> Quantity:
        """Calculate the allowed position size.

        Uses the minimum of various constraints:
        - Max position size in shares
        - Max position value / price
        - Available cash for buying
        - Portfolio risk percentage

        Args:
            signal: Trading signal
            symbol: Symbol to trade
            price: Current price
            portfolio: Current portfolio state

        Returns:
            Maximum allowed quantity
        """
        if signal in (Signal.HOLD, Signal.FLAT, Signal.CLOSE):
            # For close signals, return current position size
            position = portfolio.positions.get(symbol)
            if position:
                return Quantity(abs(position.quantity))
            return Quantity(Decimal("0"))

        # Start with max position size
        max_qty = self._config.max_position_size

        # Limit by max position value
        if price > Decimal("0"):
            value_limit = Quantity(self._config.max_position_value / price)
            max_qty = min(max_qty, value_limit)

        # For buys, limit by available cash
        if signal == Signal.LONG:
            if price > Decimal("0"):
                cash_limit = Quantity(portfolio.cash / price * Decimal("0.95"))  # 5% buffer
                max_qty = min(max_qty, cash_limit)

        # Apply portfolio risk percentage
        risk_amount = portfolio.equity * self._config.max_portfolio_risk
        if price > Decimal("0"):
            risk_limit = Quantity(risk_amount / price)
            max_qty = min(max_qty, risk_limit)

        # Subtract current position if adding to it
        position = portfolio.positions.get(symbol)
        if position and not position.is_flat:
            if (signal == Signal.LONG and position.is_long) or (
                signal == Signal.SHORT and position.is_short
            ):
                current = abs(position.quantity)
                max_qty = Quantity(max(Decimal("0"), max_qty - current))

        return Quantity(max(Decimal("0"), max_qty))

    def check_emergency_stop(self, portfolio: PortfolioState) -> bool:
        """Check if emergency stop should be triggered.

        Args:
            portfolio: Current portfolio state

        Returns:
            True if emergency stop is active
        """
        if self._emergency_stop:
            return True

        # Check if daily loss exceeds threshold
        for rule in self._rules:
            if isinstance(rule, MaxDailyLossRule):
                allowed, _ = rule.validate(
                    Signal.LONG, Symbol(""), portfolio
                )
                if not allowed:
                    self._emergency_stop = True
                    logger.warning("Emergency stop triggered due to daily loss limit")
                    return True

        return False

    def trigger_emergency_stop(self, reason: str) -> None:
        """Manually trigger emergency stop.

        Args:
            reason: Reason for triggering
        """
        self._emergency_stop = True
        logger.error(f"Emergency stop triggered: {reason}")

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop (use with caution)."""
        self._emergency_stop = False
        logger.info("Emergency stop reset")

    @property
    def is_emergency_stop_active(self) -> bool:
        """Check if emergency stop is currently active."""
        return self._emergency_stop
