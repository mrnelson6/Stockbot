"""Exposure tracking and risk reporting."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from stockbot.core.models import PortfolioState, Position
from stockbot.core.types import Price, Symbol, Timestamp
from stockbot.monitoring.logger import get_logger

logger = get_logger("risk.exposure")


@dataclass
class ExposureSnapshot:
    """Snapshot of portfolio exposure at a point in time."""

    timestamp: Timestamp

    # Portfolio values
    equity: Price
    cash: Price
    long_exposure: Price  # Total value of long positions
    short_exposure: Price  # Total value of short positions
    net_exposure: Price  # Long - Short
    gross_exposure: Price  # Long + Short

    # Ratios
    net_exposure_pct: Decimal  # Net exposure as % of equity
    gross_exposure_pct: Decimal  # Gross exposure as % of equity
    cash_pct: Decimal  # Cash as % of equity

    # Position counts
    num_long_positions: int
    num_short_positions: int

    # Concentration
    largest_position_pct: Decimal  # Largest position as % of equity
    largest_position_symbol: Optional[Symbol]

    # Risk metrics
    unrealized_pnl: Price
    unrealized_pnl_pct: Decimal


@dataclass
class DrawdownTracker:
    """Tracks portfolio drawdown."""

    peak_equity: Price = Price(Decimal("0"))
    current_drawdown: Price = Price(Decimal("0"))
    current_drawdown_pct: Decimal = Decimal("0")
    max_drawdown: Price = Price(Decimal("0"))
    max_drawdown_pct: Decimal = Decimal("0")
    peak_timestamp: Timestamp = Timestamp(0)
    max_drawdown_timestamp: Timestamp = Timestamp(0)

    def update(self, equity: Price, timestamp: Timestamp) -> None:
        """Update drawdown tracking with new equity value."""
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.peak_timestamp = timestamp
            self.current_drawdown = Price(Decimal("0"))
            self.current_drawdown_pct = Decimal("0")
        else:
            self.current_drawdown = Price(self.peak_equity - equity)
            if self.peak_equity > Decimal("0"):
                self.current_drawdown_pct = (
                    self.current_drawdown / self.peak_equity * 100
                )

            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
                self.max_drawdown_pct = self.current_drawdown_pct
                self.max_drawdown_timestamp = timestamp


@dataclass
class DailyPnLTracker:
    """Tracks daily P&L."""

    day_start_equity: Price = Price(Decimal("0"))
    current_equity: Price = Price(Decimal("0"))
    daily_pnl: Price = Price(Decimal("0"))
    daily_pnl_pct: Decimal = Decimal("0")
    high_water_mark: Price = Price(Decimal("0"))
    low_water_mark: Price = Price(Decimal("0"))

    def start_new_day(self, equity: Price) -> None:
        """Start tracking for a new day."""
        self.day_start_equity = equity
        self.current_equity = equity
        self.daily_pnl = Price(Decimal("0"))
        self.daily_pnl_pct = Decimal("0")
        self.high_water_mark = equity
        self.low_water_mark = equity

    def update(self, equity: Price) -> None:
        """Update with new equity value."""
        self.current_equity = equity
        self.daily_pnl = Price(equity - self.day_start_equity)

        if self.day_start_equity > Decimal("0"):
            self.daily_pnl_pct = self.daily_pnl / self.day_start_equity * 100

        if equity > self.high_water_mark:
            self.high_water_mark = equity
        if equity < self.low_water_mark:
            self.low_water_mark = equity


class ExposureTracker:
    """Tracks and reports portfolio exposure and risk metrics."""

    def __init__(self, initial_equity: Price) -> None:
        """Initialize the exposure tracker.

        Args:
            initial_equity: Starting portfolio equity
        """
        self._initial_equity = initial_equity
        self._drawdown = DrawdownTracker(peak_equity=initial_equity)
        self._daily_pnl = DailyPnLTracker()
        self._daily_pnl.start_new_day(initial_equity)

        # History
        self._exposure_history: list[ExposureSnapshot] = []
        self._max_history_size = 1000

    def calculate_exposure(
        self, portfolio: PortfolioState, timestamp: Timestamp
    ) -> ExposureSnapshot:
        """Calculate current exposure snapshot.

        Args:
            portfolio: Current portfolio state
            timestamp: Current timestamp

        Returns:
            ExposureSnapshot with all metrics
        """
        equity = portfolio.equity
        cash = portfolio.cash

        # Calculate long/short exposure
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")
        largest_value = Decimal("0")
        largest_symbol: Optional[Symbol] = None
        unrealized_pnl = Decimal("0")
        num_long = 0
        num_short = 0

        for symbol, position in portfolio.positions.items():
            position_value = abs(position.quantity * position.average_price)
            unrealized_pnl += position.unrealized_pnl

            if position.is_long:
                long_exposure += position_value
                num_long += 1
            elif position.is_short:
                short_exposure += position_value
                num_short += 1

            if position_value > largest_value:
                largest_value = position_value
                largest_symbol = symbol

        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure

        # Calculate percentages
        if equity > Decimal("0"):
            net_exposure_pct = net_exposure / equity * 100
            gross_exposure_pct = gross_exposure / equity * 100
            cash_pct = cash / equity * 100
            largest_position_pct = largest_value / equity * 100
            unrealized_pnl_pct = unrealized_pnl / equity * 100
        else:
            net_exposure_pct = Decimal("0")
            gross_exposure_pct = Decimal("0")
            cash_pct = Decimal("100")
            largest_position_pct = Decimal("0")
            unrealized_pnl_pct = Decimal("0")

        snapshot = ExposureSnapshot(
            timestamp=timestamp,
            equity=Price(equity),
            cash=Price(cash),
            long_exposure=Price(long_exposure),
            short_exposure=Price(short_exposure),
            net_exposure=Price(net_exposure),
            gross_exposure=Price(gross_exposure),
            net_exposure_pct=net_exposure_pct,
            gross_exposure_pct=gross_exposure_pct,
            cash_pct=cash_pct,
            num_long_positions=num_long,
            num_short_positions=num_short,
            largest_position_pct=largest_position_pct,
            largest_position_symbol=largest_symbol,
            unrealized_pnl=Price(unrealized_pnl),
            unrealized_pnl_pct=unrealized_pnl_pct,
        )

        # Update trackers
        self._drawdown.update(Price(equity), timestamp)
        self._daily_pnl.update(Price(equity))

        # Store in history
        self._exposure_history.append(snapshot)
        if len(self._exposure_history) > self._max_history_size:
            self._exposure_history = self._exposure_history[-self._max_history_size :]

        return snapshot

    def start_new_day(self, equity: Price) -> None:
        """Start tracking for a new trading day."""
        self._daily_pnl.start_new_day(equity)

    @property
    def drawdown(self) -> DrawdownTracker:
        """Get drawdown tracker."""
        return self._drawdown

    @property
    def daily_pnl(self) -> DailyPnLTracker:
        """Get daily P&L tracker."""
        return self._daily_pnl

    @property
    def latest_snapshot(self) -> Optional[ExposureSnapshot]:
        """Get most recent exposure snapshot."""
        if self._exposure_history:
            return self._exposure_history[-1]
        return None

    def get_risk_report(self) -> dict:
        """Generate a comprehensive risk report.

        Returns:
            Dict with all risk metrics
        """
        latest = self.latest_snapshot

        return {
            "exposure": {
                "equity": str(latest.equity) if latest else "0",
                "cash": str(latest.cash) if latest else "0",
                "net_exposure": str(latest.net_exposure) if latest else "0",
                "net_exposure_pct": f"{latest.net_exposure_pct:.1f}%" if latest else "0%",
                "gross_exposure": str(latest.gross_exposure) if latest else "0",
                "gross_exposure_pct": f"{latest.gross_exposure_pct:.1f}%" if latest else "0%",
                "long_positions": latest.num_long_positions if latest else 0,
                "short_positions": latest.num_short_positions if latest else 0,
            },
            "concentration": {
                "largest_position_pct": f"{latest.largest_position_pct:.1f}%" if latest else "0%",
                "largest_position_symbol": latest.largest_position_symbol if latest else None,
            },
            "pnl": {
                "unrealized": str(latest.unrealized_pnl) if latest else "0",
                "unrealized_pct": f"{latest.unrealized_pnl_pct:.2f}%" if latest else "0%",
                "daily": str(self._daily_pnl.daily_pnl),
                "daily_pct": f"{self._daily_pnl.daily_pnl_pct:.2f}%",
            },
            "drawdown": {
                "current": str(self._drawdown.current_drawdown),
                "current_pct": f"{self._drawdown.current_drawdown_pct:.2f}%",
                "max": str(self._drawdown.max_drawdown),
                "max_pct": f"{self._drawdown.max_drawdown_pct:.2f}%",
                "peak_equity": str(self._drawdown.peak_equity),
            },
        }

    def print_risk_report(self) -> None:
        """Print formatted risk report."""
        report = self.get_risk_report()

        print("\n" + "=" * 50)
        print("RISK REPORT")
        print("=" * 50)

        print(f"\n{'EXPOSURE':-^50}")
        for key, value in report["exposure"].items():
            print(f"  {key}: {value}")

        print(f"\n{'CONCENTRATION':-^50}")
        for key, value in report["concentration"].items():
            print(f"  {key}: {value}")

        print(f"\n{'P&L':-^50}")
        for key, value in report["pnl"].items():
            print(f"  {key}: {value}")

        print(f"\n{'DRAWDOWN':-^50}")
        for key, value in report["drawdown"].items():
            print(f"  {key}: {value}")

        print("=" * 50 + "\n")
