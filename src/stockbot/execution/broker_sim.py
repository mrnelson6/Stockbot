"""Simulated broker for backtesting and paper trading."""

from decimal import Decimal
from typing import Optional
from uuid import uuid4

import numpy as np

from stockbot.core.exceptions import OrderRejectedError
from stockbot.core.models import Bar, Fill, Order, Position
from stockbot.core.types import (
    OrderSide,
    OrderStatus,
    OrderType,
    Price,
    Quantity,
    Symbol,
    Timestamp,
)


class SimulatedBroker:
    """Simulated broker for backtesting.

    Handles order submission, fill simulation, and position tracking.
    Supports configurable slippage and commission models.
    """

    def __init__(
        self,
        initial_capital: Price,
        commission: Decimal = Decimal("0"),
        slippage_pct: Decimal = Decimal("0.001"),
        seed: int = 42,
    ) -> None:
        """Initialize the simulated broker.

        Args:
            initial_capital: Starting cash balance
            commission: Per-share commission
            slippage_pct: Slippage as percentage (0.001 = 0.1%)
            seed: Random seed for reproducible slippage
        """
        self._cash = initial_capital
        self._commission = commission
        self._slippage_pct = slippage_pct
        self._rng = np.random.default_rng(seed)

        # Order tracking
        self._orders: dict[str, Order] = {}
        self._order_status: dict[str, OrderStatus] = {}
        self._fills: dict[str, list[Fill]] = {}

        # Position tracking
        self._positions: dict[Symbol, Position] = {}

        # Current market prices (set by engine)
        self._current_prices: dict[Symbol, Price] = {}
        self._current_time: Timestamp = Timestamp(0)

    def set_market_state(
        self,
        prices: dict[Symbol, Price],
        timestamp: Timestamp,
    ) -> None:
        """Update current market prices.

        Called by the engine on each bar.

        Args:
            prices: Current prices by symbol
            timestamp: Current timestamp
        """
        self._current_prices = prices
        self._current_time = timestamp

        # Update unrealized PnL for positions
        for symbol, position in self._positions.items():
            if symbol in prices and not position.is_flat:
                current_price = prices[symbol]
                if position.is_long:
                    pnl = (current_price - position.average_price) * position.quantity
                else:
                    pnl = (position.average_price - current_price) * abs(position.quantity)
                position.unrealized_pnl = Price(pnl)

    def submit_order(self, order: Order) -> str:
        """Submit an order for execution.

        For market orders, executes immediately at current price with slippage.
        Limit orders are not yet implemented.

        Args:
            order: Order to submit

        Returns:
            Order ID
        """
        if order.order_type != OrderType.MARKET:
            raise NotImplementedError("Only market orders are currently supported")

        if order.symbol not in self._current_prices:
            raise OrderRejectedError(
                order.id,
                f"No market data available for {order.symbol}",
            )

        # Store order
        self._orders[order.id] = order
        self._order_status[order.id] = OrderStatus.SUBMITTED
        self._fills[order.id] = []

        # Execute immediately for market orders
        self._execute_market_order(order)

        return order.id

    def _execute_market_order(self, order: Order) -> None:
        """Execute a market order with slippage simulation."""
        base_price = self._current_prices[order.symbol]

        # Apply slippage (adverse direction)
        slippage_multiplier = Decimal("1") + (
            self._slippage_pct
            if order.side == OrderSide.BUY
            else -self._slippage_pct
        )
        fill_price = Price(base_price * slippage_multiplier)

        # Calculate commission
        commission = Price(self._commission * order.quantity)

        # Check if we have enough cash for buys
        if order.side == OrderSide.BUY:
            total_cost = fill_price * order.quantity + commission
            if total_cost > self._cash:
                self._order_status[order.id] = OrderStatus.REJECTED
                return

        # Create fill
        fill = Fill(
            order_id=order.id,
            fill_id=str(uuid4()),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=self._current_time,
            commission=commission,
        )

        self._fills[order.id].append(fill)
        self._order_status[order.id] = OrderStatus.FILLED

        # Update cash
        if order.side == OrderSide.BUY:
            self._cash = Price(self._cash - (fill_price * order.quantity) - commission)
        else:
            self._cash = Price(self._cash + (fill_price * order.quantity) - commission)

        # Update position
        self._update_position(fill)

    def _update_position(self, fill: Fill) -> None:
        """Update position based on a fill."""
        symbol = fill.symbol

        if symbol not in self._positions:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=Quantity(Decimal("0")),
                average_price=Price(Decimal("0")),
            )

        position = self._positions[symbol]
        old_quantity = position.quantity
        fill_quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

        new_quantity = Quantity(old_quantity + fill_quantity)

        if new_quantity == Decimal("0"):
            # Position closed - calculate realized PnL
            if position.is_long:
                pnl = (fill.price - position.average_price) * abs(old_quantity)
            else:
                pnl = (position.average_price - fill.price) * abs(old_quantity)

            position.realized_pnl = Price(position.realized_pnl + pnl)
            position.quantity = new_quantity
            position.average_price = Price(Decimal("0"))
            position.unrealized_pnl = Price(Decimal("0"))

        elif (old_quantity > 0 and new_quantity > 0) or (
            old_quantity < 0 and new_quantity < 0
        ):
            # Adding to position - update average price
            if old_quantity == Decimal("0"):
                new_avg = fill.price
            else:
                total_cost = abs(old_quantity) * position.average_price + abs(
                    fill_quantity
                ) * fill.price
                new_avg = Price(total_cost / abs(new_quantity))

            position.quantity = new_quantity
            position.average_price = new_avg

        else:
            # Reducing or reversing position
            # First close existing position
            closed_qty = min(abs(old_quantity), abs(fill_quantity))
            if position.is_long:
                pnl = (fill.price - position.average_price) * closed_qty
            else:
                pnl = (position.average_price - fill.price) * closed_qty

            position.realized_pnl = Price(position.realized_pnl + pnl)

            # Handle any remainder (reversal)
            remainder = abs(fill_quantity) - closed_qty
            if remainder > Decimal("0"):
                # New position in opposite direction
                position.quantity = Quantity(
                    remainder if fill.side == OrderSide.BUY else -remainder
                )
                position.average_price = fill.price
            else:
                position.quantity = new_quantity
                if position.quantity == Decimal("0"):
                    position.average_price = Price(Decimal("0"))

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: Order to cancel

        Returns:
            True if cancelled successfully
        """
        if order_id not in self._order_status:
            return False

        status = self._order_status[order_id]
        if status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            self._order_status[order_id] = OrderStatus.CANCELLED
            return True

        return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order."""
        return self._order_status.get(order_id, OrderStatus.REJECTED)

    def get_fills(self, order_id: str) -> list[Fill]:
        """Get fills for an order."""
        return self._fills.get(order_id, [])

    def get_positions(self) -> dict[Symbol, Position]:
        """Get current positions."""
        return {k: v for k, v in self._positions.items() if not v.is_flat}

    def get_position(self, symbol: Symbol) -> Optional[Position]:
        """Get position for a specific symbol."""
        return self._positions.get(symbol)

    def get_cash(self) -> Price:
        """Get current cash balance."""
        return self._cash

    def get_equity(self) -> Price:
        """Get total portfolio value (cash + positions)."""
        position_value = sum(
            (p.quantity * p.average_price + p.unrealized_pnl)
            for p in self._positions.values()
        )
        return Price(self._cash + position_value)

    def get_all_fills(self) -> list[Fill]:
        """Get all fills across all orders."""
        all_fills = []
        for fills in self._fills.values():
            all_fills.extend(fills)
        return sorted(all_fills, key=lambda f: f.timestamp)
