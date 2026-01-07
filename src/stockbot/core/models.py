"""Core data models for the trading system.

All models are immutable (frozen dataclasses) to prevent accidental state corruption.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional
from uuid import uuid4

from stockbot.core.types import (
    OrderSide,
    OrderStatus,
    OrderType,
    Price,
    Quantity,
    Signal,
    Symbol,
    Timeframe,
    Timestamp,
)


@dataclass(frozen=True)
class Bar:
    """OHLCV bar - the fundamental unit of market data.

    Bars are immutable and validated on creation.
    All timestamps are in nanoseconds since Unix epoch (UTC).
    """

    symbol: Symbol
    timestamp: Timestamp
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Quantity
    timeframe: Timeframe = Timeframe.MINUTE_1
    adjusted: bool = False  # Whether prices are adjusted for corporate actions

    def __post_init__(self) -> None:
        """Validate bar data integrity."""
        # Use object.__setattr__ because dataclass is frozen
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) must be >= Low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= Open and Close")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= Open and Close")
        if self.volume < Quantity(Decimal("0")):
            raise ValueError(f"Volume ({self.volume}) must be >= 0")


@dataclass(frozen=True)
class Order:
    """Order representation - immutable once created.

    Orders represent concrete execution instructions sent to the broker.
    """

    symbol: Symbol
    side: OrderSide
    quantity: Quantity
    order_type: OrderType
    id: str = field(default_factory=lambda: str(uuid4()))
    limit_price: Optional[Price] = None
    stop_price: Optional[Price] = None
    created_at: Optional[Timestamp] = None
    strategy_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate order parameters."""
        if self.quantity <= Quantity(Decimal("0")):
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require a limit price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require a stop price")


@dataclass(frozen=True)
class Fill:
    """Execution fill record - represents a completed trade."""

    order_id: str
    fill_id: str
    symbol: Symbol
    side: OrderSide
    quantity: Quantity
    price: Price
    timestamp: Timestamp
    commission: Price = Price(Decimal("0"))


@dataclass
class Position:
    """Current position in a symbol.

    Unlike other models, Position is mutable as it's updated frequently.
    Positive quantity = long, negative = short.
    """

    symbol: Symbol
    quantity: Quantity
    average_price: Price
    unrealized_pnl: Price = Price(Decimal("0"))
    realized_pnl: Price = Price(Decimal("0"))

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > Quantity(Decimal("0"))

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < Quantity(Decimal("0"))

    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.quantity == Quantity(Decimal("0"))

    @property
    def market_value(self) -> Price:
        """Current market value of position."""
        return Price(self.quantity * self.average_price)


@dataclass
class PortfolioState:
    """Snapshot of portfolio at a point in time."""

    timestamp: Timestamp
    cash: Price
    positions: dict[Symbol, Position] = field(default_factory=dict)
    pending_orders: list[Order] = field(default_factory=list)

    @property
    def equity(self) -> Price:
        """Total portfolio value (cash + positions)."""
        position_value = sum(
            (p.quantity * p.average_price + p.unrealized_pnl for p in self.positions.values()),
            Decimal("0"),
        )
        return Price(self.cash + position_value)

    @property
    def total_unrealized_pnl(self) -> Price:
        """Sum of unrealized PnL across all positions."""
        return Price(sum((p.unrealized_pnl for p in self.positions.values()), Decimal("0")))

    @property
    def total_realized_pnl(self) -> Price:
        """Sum of realized PnL across all positions."""
        return Price(sum((p.realized_pnl for p in self.positions.values()), Decimal("0")))


@dataclass(frozen=True)
class MarketState:
    """Current market state visible to strategy.

    This is the ONLY view strategies have of the world.
    Strategies cannot access the clock, broker, or any other component directly.
    """

    timestamp: Timestamp
    bars: dict[Symbol, list[Bar]]  # Symbol -> list of recent bars (oldest first)
    portfolio: PortfolioState

    def latest_bar(self, symbol: Symbol) -> Optional[Bar]:
        """Get the most recent bar for a symbol."""
        bars = self.bars.get(symbol)
        if bars:
            return bars[-1]
        return None

    def latest_price(self, symbol: Symbol) -> Optional[Price]:
        """Get the most recent close price for a symbol."""
        bar = self.latest_bar(symbol)
        if bar:
            return bar.close
        return None


@dataclass
class TradeRecord:
    """Record of a completed trade for analysis."""

    entry_time: Timestamp
    exit_time: Timestamp
    symbol: Symbol
    side: OrderSide
    quantity: Quantity
    entry_price: Price
    exit_price: Price
    pnl: Price
    commission: Price
    strategy_id: Optional[str] = None

    @property
    def return_pct(self) -> Decimal:
        """Percentage return on the trade."""
        if self.entry_price == Decimal("0"):
            return Decimal("0")
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > Decimal("0")


@dataclass
class StrategySignal:
    """A signal emitted by a strategy with metadata."""

    timestamp: Timestamp
    symbol: Symbol
    signal: Signal
    strategy_id: str
    confidence: float = 1.0  # Optional confidence score
    metadata: dict[str, object] = field(default_factory=dict)
