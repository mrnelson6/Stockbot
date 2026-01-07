"""Clock implementations for time management.

SimulatedClock: For backtesting with deterministic time advancement
RealClock: For paper/live trading with actual system time
"""

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from stockbot.core.types import Timestamp

# US Eastern timezone for market hours
_ET = ZoneInfo("America/New_York")

# Regular market hours (Eastern Time)
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def _datetime_to_timestamp(dt: datetime) -> Timestamp:
    """Convert datetime to nanosecond timestamp."""
    return Timestamp(int(dt.timestamp() * 1_000_000_000))


def _timestamp_to_datetime(ts: Timestamp) -> datetime:
    """Convert nanosecond timestamp to datetime."""
    return datetime.fromtimestamp(ts / 1_000_000_000, tz=timezone.utc)


class SimulatedClock:
    """Simulated clock for backtesting.

    Time is deterministic and advances only when advance() is called.
    """

    def __init__(self, start_time: Timestamp) -> None:
        """Initialize with a starting timestamp.

        Args:
            start_time: Initial timestamp in nanoseconds
        """
        self._current_time = start_time

    @property
    def now(self) -> Timestamp:
        """Get current simulated time."""
        return self._current_time

    def set_time(self, timestamp: Timestamp) -> None:
        """Set the current time directly.

        Args:
            timestamp: New timestamp in nanoseconds
        """
        self._current_time = timestamp

    def advance(self, delta_ns: int) -> None:
        """Advance time by specified nanoseconds.

        Args:
            delta_ns: Nanoseconds to advance
        """
        self._current_time = Timestamp(self._current_time + delta_ns)

    def advance_minutes(self, minutes: int) -> None:
        """Advance time by specified minutes.

        Args:
            minutes: Minutes to advance
        """
        self.advance(minutes * 60 * 1_000_000_000)

    def is_market_open(self) -> bool:
        """Check if market is open at current simulated time.

        Returns:
            True if within regular trading hours
        """
        dt = _timestamp_to_datetime(self._current_time)
        dt_et = dt.astimezone(_ET)

        # Check if weekday (0=Monday, 4=Friday)
        if dt_et.weekday() > 4:
            return False

        current_time = dt_et.time()
        return _MARKET_OPEN <= current_time < _MARKET_CLOSE

    def next_market_open(self) -> Timestamp:
        """Get timestamp of next market open.

        Returns:
            Timestamp when market will next open
        """
        dt = _timestamp_to_datetime(self._current_time)
        dt_et = dt.astimezone(_ET)

        # Start with today's open time
        next_open = dt_et.replace(
            hour=_MARKET_OPEN.hour,
            minute=_MARKET_OPEN.minute,
            second=0,
            microsecond=0,
        )

        # If we're past today's open, move to next day
        if dt_et.time() >= _MARKET_OPEN:
            next_open = next_open.replace(day=next_open.day + 1)

        # Skip weekends
        while next_open.weekday() > 4:
            next_open = next_open.replace(day=next_open.day + 1)

        return _datetime_to_timestamp(next_open)

    def next_market_close(self) -> Timestamp:
        """Get timestamp of next market close.

        Returns:
            Timestamp when market will next close
        """
        dt = _timestamp_to_datetime(self._current_time)
        dt_et = dt.astimezone(_ET)

        # Start with today's close time
        next_close = dt_et.replace(
            hour=_MARKET_CLOSE.hour,
            minute=_MARKET_CLOSE.minute,
            second=0,
            microsecond=0,
        )

        # If we're past today's close, move to next day
        if dt_et.time() >= _MARKET_CLOSE:
            next_close = next_close.replace(day=next_close.day + 1)

        # Skip weekends
        while next_close.weekday() > 4:
            next_close = next_close.replace(day=next_close.day + 1)

        return _datetime_to_timestamp(next_close)


class RealClock:
    """Real clock using system time.

    For paper and live trading.
    """

    @property
    def now(self) -> Timestamp:
        """Get current system time as nanosecond timestamp."""
        return _datetime_to_timestamp(datetime.now(timezone.utc))

    def is_market_open(self) -> bool:
        """Check if market is currently open.

        Returns:
            True if within regular trading hours
        """
        now = datetime.now(_ET)

        # Check if weekday
        if now.weekday() > 4:
            return False

        current_time = now.time()
        return _MARKET_OPEN <= current_time < _MARKET_CLOSE

    def next_market_open(self) -> Timestamp:
        """Get timestamp of next market open.

        Returns:
            Timestamp when market will next open
        """
        now = datetime.now(_ET)

        # Start with today's open time
        next_open = now.replace(
            hour=_MARKET_OPEN.hour,
            minute=_MARKET_OPEN.minute,
            second=0,
            microsecond=0,
        )

        # If we're past today's open, move to next day
        if now.time() >= _MARKET_OPEN:
            from datetime import timedelta

            next_open = next_open + timedelta(days=1)

        # Skip weekends
        while next_open.weekday() > 4:
            from datetime import timedelta

            next_open = next_open + timedelta(days=1)

        return _datetime_to_timestamp(next_open)

    def next_market_close(self) -> Timestamp:
        """Get timestamp of next market close.

        Returns:
            Timestamp when market will next close
        """
        now = datetime.now(_ET)

        # Start with today's close time
        next_close = now.replace(
            hour=_MARKET_CLOSE.hour,
            minute=_MARKET_CLOSE.minute,
            second=0,
            microsecond=0,
        )

        # If we're past today's close, move to next day
        if now.time() >= _MARKET_CLOSE:
            from datetime import timedelta

            next_close = next_close + timedelta(days=1)

        # Skip weekends
        while next_close.weekday() > 4:
            from datetime import timedelta

            next_close = next_close + timedelta(days=1)

        return _datetime_to_timestamp(next_close)
