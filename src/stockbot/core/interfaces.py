"""Core interfaces and protocols for the trading system.

These abstractions enable the same strategy code to run across all environments
(backtest, paper, live) by defining clear contracts for each component.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Protocol

from stockbot.core.models import Bar, Fill, MarketState, Order, PortfolioState, Position
from stockbot.core.types import (
    OrderStatus,
    Price,
    Quantity,
    Signal,
    Symbol,
    Timeframe,
    Timestamp,
)


class Strategy(ABC):
    """Base class for all trading strategies.

    This is THE key abstraction that enables environment-agnostic strategy code.
    Strategies:
    - Receive market state through observe()
    - Emit signals through decide()
    - Never interact with the outside world directly
    - Cannot tell what environment they're running in
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this strategy."""
        ...

    @abstractmethod
    def observe(self, state: MarketState) -> None:
        """Update internal state with new market data.

        Called on every bar/tick before decide().
        Strategy should store any data needed for decision making.

        Args:
            state: Current market state including bars and portfolio
        """
        ...

    @abstractmethod
    def decide(self) -> dict[Symbol, Signal]:
        """Generate trading signals based on observed state.

        Returns trading intent (signals), NOT orders.
        The execution layer translates signals to orders.

        Returns:
            Dict mapping symbols to desired signals
        """
        ...

    def update(self, reward: float) -> None:
        """Update strategy with reward signal.

        Called after execution to provide feedback for learning strategies.
        Default implementation does nothing (for non-learning strategies).

        Args:
            reward: Reward signal (typically PnL or risk-adjusted return)
        """
        pass

    def reset(self) -> None:
        """Reset strategy state for a new episode/session.

        Called at the start of a new backtest or trading session.
        """
        pass


class DataProvider(Protocol):
    """Protocol for market data providers.

    Implementations handle the specifics of data source access.
    """

    def get_bars(
        self,
        symbol: Symbol,
        start: Timestamp,
        end: Timestamp,
        timeframe: Timeframe = Timeframe.MINUTE_1,
    ) -> Iterator[Bar]:
        """Yield bars for symbol in time range.

        Args:
            symbol: Ticker symbol
            start: Start timestamp (inclusive)
            end: End timestamp (exclusive)
            timeframe: Bar resolution

        Yields:
            Bar objects in chronological order
        """
        ...

    def get_latest(self, symbol: Symbol) -> Optional[Bar]:
        """Get the most recent bar for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Most recent bar, or None if not available
        """
        ...

    def get_symbols(self) -> list[Symbol]:
        """Get list of available symbols.

        Returns:
            List of tradable symbols
        """
        ...


class Broker(Protocol):
    """Protocol for broker implementations.

    Abstracts the differences between simulated and real brokers.
    """

    def submit_order(self, order: Order) -> str:
        """Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            Order ID (may differ from order.id if broker assigns its own)
        """
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancellation was successful
        """
        ...

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of an order.

        Args:
            order_id: Order ID

        Returns:
            Current order status
        """
        ...

    def get_fills(self, order_id: str) -> list[Fill]:
        """Get fills for an order.

        Args:
            order_id: Order ID

        Returns:
            List of fills for the order
        """
        ...

    def get_positions(self) -> dict[Symbol, Position]:
        """Get current positions.

        Returns:
            Dict mapping symbols to current positions
        """
        ...

    def get_cash(self) -> Price:
        """Get current cash balance.

        Returns:
            Available cash
        """
        ...


class RiskManager(Protocol):
    """Protocol for risk management.

    Risk manager sits between strategy signals and order execution.
    It can reject or modify any signal based on risk rules.
    """

    def validate_signal(
        self,
        signal: Signal,
        symbol: Symbol,
        portfolio: PortfolioState,
    ) -> tuple[bool, Optional[str]]:
        """Validate if a signal is allowed by risk rules.

        Args:
            signal: Proposed trading signal
            symbol: Symbol for the signal
            portfolio: Current portfolio state

        Returns:
            Tuple of (allowed, rejection_reason)
            If allowed is True, rejection_reason is None
        """
        ...

    def calculate_position_size(
        self,
        signal: Signal,
        symbol: Symbol,
        price: Price,
        portfolio: PortfolioState,
    ) -> Quantity:
        """Calculate allowed position size for a signal.

        Args:
            signal: Trading signal
            symbol: Symbol to trade
            price: Current price
            portfolio: Current portfolio state

        Returns:
            Maximum allowed quantity
        """
        ...

    def check_emergency_stop(self, portfolio: PortfolioState) -> bool:
        """Check if emergency stop should be triggered.

        Args:
            portfolio: Current portfolio state

        Returns:
            True if emergency stop condition is met
        """
        ...


class Clock(Protocol):
    """Protocol for time management.

    Abstracts the difference between simulated time (backtest)
    and real time (paper/live trading).
    """

    @property
    def now(self) -> Timestamp:
        """Get current timestamp in nanoseconds (UTC).

        Returns:
            Current timestamp
        """
        ...

    def is_market_open(self) -> bool:
        """Check if the market is currently open.

        Returns:
            True if market is open for trading
        """
        ...

    def next_market_open(self) -> Timestamp:
        """Get timestamp of next market open.

        Returns:
            Timestamp when market will next open
        """
        ...

    def next_market_close(self) -> Timestamp:
        """Get timestamp of next market close.

        Returns:
            Timestamp when market will next close
        """
        ...
