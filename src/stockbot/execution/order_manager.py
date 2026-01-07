"""Order manager for order lifecycle tracking and safety controls."""

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Protocol

from stockbot.core.exceptions import OrderRejectedError
from stockbot.core.models import Fill, Order, PortfolioState
from stockbot.core.types import OrderStatus, Price, Quantity, Signal, Symbol, Timestamp
from stockbot.execution.translator import SignalTranslator
from stockbot.monitoring.logger import get_logger

logger = get_logger("order_manager")


class Broker(Protocol):
    """Protocol for broker implementations."""

    def submit_order(self, order: Order) -> str: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_order_status(self, order_id: str) -> OrderStatus: ...
    def get_fills(self, order_id: str) -> list[Fill]: ...


@dataclass
class OrderRecord:
    """Record of an order and its state."""

    order: Order
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: Optional[float] = None
    filled_at: Optional[float] = None
    fills: list[Fill] = field(default_factory=list)
    error_message: Optional[str] = None


class OrderManager:
    """Manages order lifecycle with safety controls.

    Features:
    - Rate limiting
    - Duplicate order detection
    - Order tracking and state management
    - Pending order management
    """

    def __init__(
        self,
        broker: Broker,
        max_orders_per_minute: int = 10,
        duplicate_window_seconds: float = 5.0,
    ) -> None:
        """Initialize the order manager.

        Args:
            broker: Broker to submit orders to
            max_orders_per_minute: Maximum orders per minute (rate limit)
            duplicate_window_seconds: Window to detect duplicate orders
        """
        self._broker = broker
        self._translator = SignalTranslator()
        self._max_orders_per_minute = max_orders_per_minute
        self._duplicate_window = duplicate_window_seconds

        # Order tracking
        self._orders: dict[str, OrderRecord] = {}
        self._pending_orders: dict[str, OrderRecord] = {}

        # Rate limiting
        self._order_timestamps: deque[float] = deque()

        # Duplicate detection
        self._recent_orders: deque[tuple[float, str]] = deque()

    def submit_signal(
        self,
        symbol: Symbol,
        signal: Signal,
        price: Price,
        quantity: Quantity,
        portfolio: PortfolioState,
        timestamp: Timestamp,
        strategy_id: Optional[str] = None,
    ) -> Optional[str]:
        """Submit a signal, translating it to an order if appropriate.

        Args:
            symbol: Symbol for the signal
            signal: Trading signal
            price: Current price
            quantity: Position size
            portfolio: Current portfolio state
            timestamp: Current timestamp
            strategy_id: Optional strategy ID

        Returns:
            Order ID if order was submitted, None otherwise

        Raises:
            OrderRejectedError: If order is rejected
        """
        # Translate signal to order
        order = self._translator.translate(
            symbol=symbol,
            signal=signal,
            price=price,
            quantity=quantity,
            portfolio=portfolio,
            timestamp=timestamp,
            strategy_id=strategy_id,
        )

        if order is None:
            return None

        return self.submit_order(order)

    def submit_order(self, order: Order) -> str:
        """Submit an order with safety checks.

        Args:
            order: Order to submit

        Returns:
            Order ID

        Raises:
            OrderRejectedError: If order is rejected by safety checks
        """
        current_time = time.time()

        # Check rate limit
        self._check_rate_limit(current_time)

        # Check for duplicates
        self._check_duplicate(order, current_time)

        # Create order record
        record = OrderRecord(
            order=order,
            status=OrderStatus.PENDING,
            submitted_at=current_time,
        )
        self._orders[order.id] = record

        try:
            # Submit to broker
            broker_order_id = self._broker.submit_order(order)
            record.broker_order_id = broker_order_id
            record.status = OrderStatus.SUBMITTED

            # Track for rate limiting
            self._order_timestamps.append(current_time)
            self._recent_orders.append((current_time, self._order_key(order)))

            logger.info(
                f"Order submitted: {order.id} -> {broker_order_id}",
                symbol=order.symbol,
                side=order.side.name,
                quantity=str(order.quantity),
            )

            return order.id

        except OrderRejectedError:
            record.status = OrderStatus.REJECTED
            raise

        except Exception as e:
            record.status = OrderStatus.REJECTED
            record.error_message = str(e)
            raise OrderRejectedError(order.id, str(e))

    def _check_rate_limit(self, current_time: float) -> None:
        """Check if we're within rate limits."""
        # Remove timestamps older than 1 minute
        cutoff = current_time - 60.0
        while self._order_timestamps and self._order_timestamps[0] < cutoff:
            self._order_timestamps.popleft()

        if len(self._order_timestamps) >= self._max_orders_per_minute:
            raise OrderRejectedError(
                "rate_limit",
                f"Rate limit exceeded: {len(self._order_timestamps)} orders in last minute",
            )

    def _check_duplicate(self, order: Order, current_time: float) -> None:
        """Check for duplicate orders."""
        # Remove old entries
        cutoff = current_time - self._duplicate_window
        while self._recent_orders and self._recent_orders[0][0] < cutoff:
            self._recent_orders.popleft()

        # Check for duplicate
        order_key = self._order_key(order)
        for _, key in self._recent_orders:
            if key == order_key:
                raise OrderRejectedError(
                    order.id,
                    f"Duplicate order detected within {self._duplicate_window}s window",
                )

    def _order_key(self, order: Order) -> str:
        """Generate a key for duplicate detection."""
        return f"{order.symbol}:{order.side.name}:{order.quantity}:{order.order_type.name}"

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get status of an order.

        Args:
            order_id: Order ID

        Returns:
            Current status
        """
        record = self._orders.get(order_id)
        if record is None:
            return OrderStatus.REJECTED

        # If order is pending/submitted, check with broker
        if record.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
            if record.broker_order_id:
                record.status = self._broker.get_order_status(record.broker_order_id)

        return record.status

    def get_fills(self, order_id: str) -> list[Fill]:
        """Get fills for an order.

        Args:
            order_id: Order ID

        Returns:
            List of fills
        """
        record = self._orders.get(order_id)
        if record is None or record.broker_order_id is None:
            return []

        # Fetch fills from broker
        record.fills = self._broker.get_fills(record.broker_order_id)
        return record.fills

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID

        Returns:
            True if cancelled
        """
        record = self._orders.get(order_id)
        if record is None or record.broker_order_id is None:
            return False

        success = self._broker.cancel_order(record.broker_order_id)
        if success:
            record.status = OrderStatus.CANCELLED

        return success

    def cancel_all_pending(self) -> int:
        """Cancel all pending orders.

        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        for order_id, record in self._orders.items():
            if record.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                if self.cancel_order(order_id):
                    cancelled += 1

        logger.info(f"Cancelled {cancelled} pending orders")
        return cancelled

    def get_pending_orders(self) -> list[Order]:
        """Get all pending/submitted orders.

        Returns:
            List of pending orders
        """
        pending = []
        for record in self._orders.values():
            if record.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
                pending.append(record.order)
        return pending

    def get_order_record(self, order_id: str) -> Optional[OrderRecord]:
        """Get full order record.

        Args:
            order_id: Order ID

        Returns:
            OrderRecord or None
        """
        return self._orders.get(order_id)

    def update_order_statuses(self) -> None:
        """Update statuses of all non-terminal orders."""
        for order_id, record in self._orders.items():
            if record.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
                if record.broker_order_id:
                    new_status = self._broker.get_order_status(record.broker_order_id)
                    if new_status != record.status:
                        logger.info(
                            f"Order {order_id} status: {record.status.name} -> {new_status.name}"
                        )
                        record.status = new_status

                        if new_status == OrderStatus.FILLED:
                            record.filled_at = time.time()
                            record.fills = self._broker.get_fills(record.broker_order_id)
