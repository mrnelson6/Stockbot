"""Alpaca broker adapter for paper and live trading."""

from decimal import Decimal
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import OrderStatus as AlpacaOrderStatus
from alpaca.trading.enums import OrderType as AlpacaOrderType
from alpaca.trading.enums import TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, LimitOrderRequest

from stockbot.config.settings import AlpacaConfig
from stockbot.core.exceptions import BrokerError, OrderRejectedError
from stockbot.core.models import Fill, Order, Position
from stockbot.core.types import (
    OrderSide,
    OrderStatus,
    OrderType,
    Price,
    Quantity,
    Symbol,
    Timestamp,
)
from stockbot.monitoring.logger import get_logger

logger = get_logger("broker.alpaca")


def _convert_order_side(side: OrderSide) -> AlpacaOrderSide:
    """Convert our OrderSide to Alpaca's."""
    return AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL


def _convert_alpaca_status(status: AlpacaOrderStatus) -> OrderStatus:
    """Convert Alpaca's order status to ours."""
    mapping = {
        AlpacaOrderStatus.NEW: OrderStatus.SUBMITTED,
        AlpacaOrderStatus.ACCEPTED: OrderStatus.SUBMITTED,
        AlpacaOrderStatus.PENDING_NEW: OrderStatus.PENDING,
        AlpacaOrderStatus.PARTIALLY_FILLED: OrderStatus.PARTIAL,
        AlpacaOrderStatus.FILLED: OrderStatus.FILLED,
        AlpacaOrderStatus.CANCELED: OrderStatus.CANCELLED,
        AlpacaOrderStatus.REJECTED: OrderStatus.REJECTED,
        AlpacaOrderStatus.EXPIRED: OrderStatus.CANCELLED,
    }
    return mapping.get(status, OrderStatus.PENDING)


def _datetime_to_timestamp(dt) -> Timestamp:
    """Convert datetime to nanosecond timestamp."""
    if dt is None:
        return Timestamp(0)
    return Timestamp(int(dt.timestamp() * 1_000_000_000))


class AlpacaBroker:
    """Broker adapter for Alpaca paper and live trading."""

    def __init__(self, config: AlpacaConfig) -> None:
        """Initialize the Alpaca broker.

        Args:
            config: Alpaca API configuration
        """
        self._config = config
        self._client = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )

        # Local order tracking (maps our order ID to Alpaca order ID)
        self._order_mapping: dict[str, str] = {}
        self._alpaca_orders: dict[str, object] = {}

        logger.info(
            f"Alpaca broker initialized",
            paper=config.paper,
            base_url=config.base_url,
        )

    def submit_order(self, order: Order) -> str:
        """Submit an order to Alpaca.

        Args:
            order: Order to submit

        Returns:
            Order ID from Alpaca

        Raises:
            OrderRejectedError: If order is rejected
            BrokerError: If there's a communication error
        """
        try:
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=str(order.symbol),
                    qty=float(order.quantity),
                    side=_convert_order_side(order.side),
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type == OrderType.LIMIT:
                if order.limit_price is None:
                    raise OrderRejectedError(order.id, "Limit order requires limit_price")
                request = LimitOrderRequest(
                    symbol=str(order.symbol),
                    qty=float(order.quantity),
                    side=_convert_order_side(order.side),
                    time_in_force=TimeInForce.DAY,
                    limit_price=float(order.limit_price),
                )
            else:
                raise OrderRejectedError(
                    order.id, f"Unsupported order type: {order.order_type}"
                )

            alpaca_order = self._client.submit_order(request)

            # Store mapping
            self._order_mapping[order.id] = alpaca_order.id
            self._alpaca_orders[alpaca_order.id] = alpaca_order

            logger.order(order, "submitted to Alpaca")
            logger.info(f"Alpaca order ID: {alpaca_order.id}")

            return str(alpaca_order.id)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to submit order: {error_msg}")

            if "insufficient" in error_msg.lower():
                raise OrderRejectedError(order.id, "Insufficient buying power")
            elif "rejected" in error_msg.lower():
                raise OrderRejectedError(order.id, error_msg)
            else:
                raise BrokerError(f"Failed to submit order: {error_msg}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Our order ID or Alpaca order ID

        Returns:
            True if cancelled successfully
        """
        try:
            # Get Alpaca order ID
            alpaca_id = self._order_mapping.get(order_id, order_id)

            self._client.cancel_order_by_id(alpaca_id)
            logger.info(f"Order cancelled: {alpaca_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order.

        Args:
            order_id: Our order ID or Alpaca order ID

        Returns:
            Current order status
        """
        try:
            alpaca_id = self._order_mapping.get(order_id, order_id)
            alpaca_order = self._client.get_order_by_id(alpaca_id)
            return _convert_alpaca_status(alpaca_order.status)

        except Exception as e:
            logger.warning(f"Failed to get order status {order_id}: {e}")
            return OrderStatus.REJECTED

    def get_fills(self, order_id: str) -> list[Fill]:
        """Get fills for an order.

        Args:
            order_id: Our order ID or Alpaca order ID

        Returns:
            List of fills
        """
        try:
            alpaca_id = self._order_mapping.get(order_id, order_id)
            alpaca_order = self._client.get_order_by_id(alpaca_id)

            fills = []

            if alpaca_order.filled_qty and float(alpaca_order.filled_qty) > 0:
                # Alpaca doesn't provide individual fills for market orders,
                # so we create a single fill record
                fills.append(
                    Fill(
                        order_id=order_id,
                        fill_id=f"{alpaca_id}-fill",
                        symbol=Symbol(alpaca_order.symbol),
                        side=OrderSide.BUY if alpaca_order.side == AlpacaOrderSide.BUY else OrderSide.SELL,
                        quantity=Quantity(Decimal(str(alpaca_order.filled_qty))),
                        price=Price(Decimal(str(alpaca_order.filled_avg_price or 0))),
                        timestamp=_datetime_to_timestamp(alpaca_order.filled_at),
                        commission=Price(Decimal("0")),  # Alpaca is commission-free
                    )
                )

            return fills

        except Exception as e:
            logger.warning(f"Failed to get fills for {order_id}: {e}")
            return []

    def get_positions(self) -> dict[Symbol, Position]:
        """Get current positions from Alpaca.

        Returns:
            Dict mapping symbols to positions
        """
        try:
            alpaca_positions = self._client.get_all_positions()
            positions = {}

            for pos in alpaca_positions:
                symbol = Symbol(pos.symbol)
                qty = Decimal(str(pos.qty))

                # Alpaca returns positive qty for long, we use negative for short
                if pos.side == "short":
                    qty = -qty

                positions[symbol] = Position(
                    symbol=symbol,
                    quantity=Quantity(qty),
                    average_price=Price(Decimal(str(pos.avg_entry_price))),
                    unrealized_pnl=Price(Decimal(str(pos.unrealized_pl))),
                    realized_pnl=Price(Decimal("0")),  # Would need to track separately
                )

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def get_position(self, symbol: Symbol) -> Optional[Position]:
        """Get position for a specific symbol.

        Args:
            symbol: Symbol to look up

        Returns:
            Position or None if no position
        """
        positions = self.get_positions()
        return positions.get(symbol)

    def get_cash(self) -> Price:
        """Get current cash balance.

        Returns:
            Available cash
        """
        try:
            account = self._client.get_account()
            return Price(Decimal(str(account.cash)))
        except Exception as e:
            logger.error(f"Failed to get cash balance: {e}")
            return Price(Decimal("0"))

    def get_equity(self) -> Price:
        """Get total portfolio value.

        Returns:
            Total equity
        """
        try:
            account = self._client.get_account()
            return Price(Decimal(str(account.equity)))
        except Exception as e:
            logger.error(f"Failed to get equity: {e}")
            return Price(Decimal("0"))

    def get_buying_power(self) -> Price:
        """Get available buying power.

        Returns:
            Buying power
        """
        try:
            account = self._client.get_account()
            return Price(Decimal(str(account.buying_power)))
        except Exception as e:
            logger.error(f"Failed to get buying power: {e}")
            return Price(Decimal("0"))

    def close_all_positions(self) -> bool:
        """Close all open positions.

        Returns:
            True if successful
        """
        try:
            self._client.close_all_positions(cancel_orders=True)
            logger.warning("Closed all positions")
            return True
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

    def is_market_open(self) -> bool:
        """Check if the market is currently open.

        Returns:
            True if market is open
        """
        try:
            clock = self._client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to get market clock: {e}")
            return False

    def get_account_info(self) -> dict:
        """Get account information.

        Returns:
            Dict with account details
        """
        try:
            account = self._client.get_account()
            return {
                "id": account.id,
                "status": account.status,
                "cash": str(account.cash),
                "equity": str(account.equity),
                "buying_power": str(account.buying_power),
                "portfolio_value": str(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
