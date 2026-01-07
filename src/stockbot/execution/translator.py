"""Signal to order translation."""

from decimal import Decimal
from typing import Optional

from stockbot.core.models import Order, PortfolioState
from stockbot.core.types import OrderSide, OrderType, Price, Quantity, Signal, Symbol, Timestamp


class SignalTranslator:
    """Translates strategy signals into executable orders.

    Handles the conversion from abstract intent (signals) to concrete
    order instructions, including position sizing and order construction.
    """

    def translate(
        self,
        symbol: Symbol,
        signal: Signal,
        price: Price,
        quantity: Quantity,
        portfolio: PortfolioState,
        timestamp: Timestamp,
        strategy_id: Optional[str] = None,
    ) -> Optional[Order]:
        """Translate a signal into an order.

        Args:
            symbol: Symbol for the signal
            signal: Trading signal
            price: Current market price
            quantity: Desired position size (from risk manager)
            portfolio: Current portfolio state
            timestamp: Current timestamp
            strategy_id: Optional strategy identifier

        Returns:
            Order if action is needed, None otherwise
        """
        current_position = portfolio.positions.get(symbol)
        current_qty = current_position.quantity if current_position else Quantity(Decimal("0"))

        if signal == Signal.HOLD:
            # No action needed
            return None

        if signal == Signal.LONG:
            return self._handle_long_signal(
                symbol, current_qty, quantity, timestamp, strategy_id
            )

        if signal == Signal.SHORT:
            return self._handle_short_signal(
                symbol, current_qty, quantity, timestamp, strategy_id
            )

        if signal in (Signal.FLAT, Signal.CLOSE):
            return self._handle_close_signal(symbol, current_qty, timestamp, strategy_id)

        return None

    def _handle_long_signal(
        self,
        symbol: Symbol,
        current_qty: Quantity,
        target_qty: Quantity,
        timestamp: Timestamp,
        strategy_id: Optional[str],
    ) -> Optional[Order]:
        """Handle a LONG signal."""
        if current_qty >= target_qty:
            # Already at or above target long position
            return None

        if current_qty < Decimal("0"):
            # Currently short - close short position first, then go long
            # For simplicity, we issue an order to achieve target long position
            order_qty = Quantity(target_qty - current_qty)
        else:
            # Currently flat or long - add to position
            order_qty = Quantity(target_qty - current_qty)

        if order_qty <= Decimal("0"):
            return None

        return Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=order_qty,
            order_type=OrderType.MARKET,
            created_at=timestamp,
            strategy_id=strategy_id,
        )

    def _handle_short_signal(
        self,
        symbol: Symbol,
        current_qty: Quantity,
        target_qty: Quantity,
        timestamp: Timestamp,
        strategy_id: Optional[str],
    ) -> Optional[Order]:
        """Handle a SHORT signal."""
        target_short = -target_qty  # Short position is negative

        if current_qty <= target_short:
            # Already at or below target short position
            return None

        # Calculate how much to sell
        order_qty = Quantity(current_qty - target_short)

        if order_qty <= Decimal("0"):
            return None

        return Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=order_qty,
            order_type=OrderType.MARKET,
            created_at=timestamp,
            strategy_id=strategy_id,
        )

    def _handle_close_signal(
        self,
        symbol: Symbol,
        current_qty: Quantity,
        timestamp: Timestamp,
        strategy_id: Optional[str],
    ) -> Optional[Order]:
        """Handle a CLOSE/FLAT signal."""
        if current_qty == Decimal("0"):
            # Already flat
            return None

        if current_qty > Decimal("0"):
            # Close long position
            return Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=Quantity(abs(current_qty)),
                order_type=OrderType.MARKET,
                created_at=timestamp,
                strategy_id=strategy_id,
            )
        else:
            # Close short position
            return Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=Quantity(abs(current_qty)),
                order_type=OrderType.MARKET,
                created_at=timestamp,
                strategy_id=strategy_id,
            )
