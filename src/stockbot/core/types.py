"""Core type definitions for the trading system."""

from decimal import Decimal
from enum import Enum, auto
from typing import NewType

# Type aliases for domain clarity and type safety
Symbol = NewType("Symbol", str)
Price = NewType("Price", Decimal)
Quantity = NewType("Quantity", Decimal)
Timestamp = NewType("Timestamp", int)  # Nanoseconds since Unix epoch (UTC)


class Signal(Enum):
    """Trading intent signals emitted by strategies.

    Signals represent desired position state, NOT orders.
    The execution layer translates signals into concrete orders.
    """

    LONG = auto()  # Desire to hold a long position
    SHORT = auto()  # Desire to hold a short position
    FLAT = auto()  # Desire to have no position
    CLOSE = auto()  # Close current position (alias for FLAT with urgency)
    HOLD = auto()  # No change to current intent


class OrderSide(Enum):
    """Order direction."""

    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    """Order execution type."""

    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


class OrderStatus(Enum):
    """Order lifecycle status."""

    PENDING = auto()  # Created but not submitted
    SUBMITTED = auto()  # Sent to broker
    PARTIAL = auto()  # Partially filled
    FILLED = auto()  # Completely filled
    CANCELLED = auto()  # Cancelled by user or system
    REJECTED = auto()  # Rejected by broker or risk manager


class Environment(Enum):
    """Execution environment type.

    Determines which adapters and behaviors are used.
    Strategy code should NOT check this directly.
    """

    BACKTEST = auto()  # Historical replay, simulated execution
    PAPER = auto()  # Live data, simulated execution
    LIVE = auto()  # Live data, real execution


class Timeframe(Enum):
    """Bar timeframe/resolution."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
