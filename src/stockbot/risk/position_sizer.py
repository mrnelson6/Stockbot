"""Position sizing strategies including volatility-adjusted sizing."""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Sequence

import numpy as np

from stockbot.core.models import Bar, PortfolioState
from stockbot.core.types import Price, Quantity, Symbol


class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""

    @abstractmethod
    def calculate_size(
        self,
        symbol: Symbol,
        price: Price,
        portfolio: PortfolioState,
        bars: Sequence[Bar],
    ) -> Quantity:
        """Calculate position size.

        Args:
            symbol: Symbol to size
            price: Current price
            portfolio: Current portfolio state
            bars: Recent price bars for volatility calculation

        Returns:
            Recommended position size in shares
        """
        ...


class FixedDollarSizer(PositionSizer):
    """Fixed dollar amount per position."""

    def __init__(self, dollars_per_position: Decimal) -> None:
        """Initialize with fixed dollar amount.

        Args:
            dollars_per_position: Dollar amount per position
        """
        self._dollars = dollars_per_position

    def calculate_size(
        self,
        symbol: Symbol,
        price: Price,
        portfolio: PortfolioState,
        bars: Sequence[Bar],
    ) -> Quantity:
        if price <= Decimal("0"):
            return Quantity(Decimal("0"))

        shares = self._dollars / price
        return Quantity(shares.quantize(Decimal("1")))  # Round to whole shares


class FixedPercentSizer(PositionSizer):
    """Fixed percentage of portfolio equity per position."""

    def __init__(self, percent_per_position: Decimal) -> None:
        """Initialize with percentage of equity.

        Args:
            percent_per_position: Percentage of equity (e.g., 5 for 5%)
        """
        self._percent = percent_per_position / 100

    def calculate_size(
        self,
        symbol: Symbol,
        price: Price,
        portfolio: PortfolioState,
        bars: Sequence[Bar],
    ) -> Quantity:
        if price <= Decimal("0"):
            return Quantity(Decimal("0"))

        dollars = portfolio.equity * self._percent
        shares = dollars / price
        return Quantity(shares.quantize(Decimal("1")))


class VolatilityAdjustedSizer(PositionSizer):
    """Volatility-adjusted position sizing.

    Sizes positions inversely proportional to volatility,
    so more volatile assets get smaller positions.
    """

    def __init__(
        self,
        target_risk_dollars: Decimal,
        lookback_periods: int = 20,
        use_atr: bool = True,
    ) -> None:
        """Initialize volatility-adjusted sizer.

        Args:
            target_risk_dollars: Target dollar risk per position
            lookback_periods: Periods for volatility calculation
            use_atr: If True, use ATR; if False, use standard deviation
        """
        self._target_risk = target_risk_dollars
        self._lookback = lookback_periods
        self._use_atr = use_atr

    def _calculate_atr(self, bars: Sequence[Bar]) -> Decimal:
        """Calculate Average True Range."""
        if len(bars) < 2:
            return Decimal("0")

        true_ranges = []
        for i in range(1, len(bars)):
            high = float(bars[i].high)
            low = float(bars[i].low)
            prev_close = float(bars[i - 1].close)

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        if not true_ranges:
            return Decimal("0")

        atr = np.mean(true_ranges[-self._lookback :])
        return Decimal(str(atr))

    def _calculate_std(self, bars: Sequence[Bar]) -> Decimal:
        """Calculate standard deviation of returns."""
        if len(bars) < 2:
            return Decimal("0")

        closes = [float(b.close) for b in bars[-self._lookback - 1 :]]
        if len(closes) < 2:
            return Decimal("0")

        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                ret = (closes[i] - closes[i - 1]) / closes[i - 1]
                returns.append(ret)

        if not returns:
            return Decimal("0")

        std = np.std(returns)
        # Convert to dollar volatility
        avg_price = np.mean(closes)
        dollar_vol = std * avg_price

        return Decimal(str(dollar_vol))

    def calculate_size(
        self,
        symbol: Symbol,
        price: Price,
        portfolio: PortfolioState,
        bars: Sequence[Bar],
    ) -> Quantity:
        if price <= Decimal("0") or len(bars) < self._lookback:
            return Quantity(Decimal("0"))

        # Calculate volatility measure
        if self._use_atr:
            volatility = self._calculate_atr(bars)
        else:
            volatility = self._calculate_std(bars)

        if volatility <= Decimal("0"):
            return Quantity(Decimal("0"))

        # Position size = target risk / volatility
        shares = self._target_risk / volatility
        return Quantity(max(Decimal("0"), shares.quantize(Decimal("1"))))


class RiskParitySizer(PositionSizer):
    """Risk parity position sizing.

    Sizes positions so each contributes equal risk (volatility) to portfolio.
    """

    def __init__(
        self,
        portfolio_risk_target: Decimal,
        num_positions: int,
        lookback_periods: int = 20,
    ) -> None:
        """Initialize risk parity sizer.

        Args:
            portfolio_risk_target: Target portfolio volatility (as decimal, e.g., 0.10 for 10%)
            num_positions: Expected number of positions
            lookback_periods: Periods for volatility calculation
        """
        self._portfolio_risk = portfolio_risk_target
        self._num_positions = num_positions
        self._lookback = lookback_periods

    def _calculate_volatility(self, bars: Sequence[Bar]) -> Decimal:
        """Calculate annualized volatility."""
        if len(bars) < 2:
            return Decimal("0")

        closes = [float(b.close) for b in bars[-self._lookback - 1 :]]
        if len(closes) < 2:
            return Decimal("0")

        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                ret = (closes[i] - closes[i - 1]) / closes[i - 1]
                returns.append(ret)

        if not returns:
            return Decimal("0")

        # Annualize (assuming daily data, 252 trading days)
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)

        return Decimal(str(annual_vol))

    def calculate_size(
        self,
        symbol: Symbol,
        price: Price,
        portfolio: PortfolioState,
        bars: Sequence[Bar],
    ) -> Quantity:
        if price <= Decimal("0") or len(bars) < self._lookback:
            return Quantity(Decimal("0"))

        volatility = self._calculate_volatility(bars)
        if volatility <= Decimal("0"):
            return Quantity(Decimal("0"))

        # Each position gets equal share of portfolio risk
        position_risk_target = self._portfolio_risk / Decimal(str(self._num_positions))

        # Weight = target_vol / asset_vol
        weight = position_risk_target / volatility

        # Dollar amount = weight * equity
        dollars = weight * portfolio.equity
        shares = dollars / price

        return Quantity(max(Decimal("0"), shares.quantize(Decimal("1"))))


class KellyCriterionSizer(PositionSizer):
    """Kelly Criterion position sizing.

    Uses win rate and win/loss ratio to size positions optimally.
    Typically use fractional Kelly (e.g., 0.25 or 0.5) for safety.
    """

    def __init__(
        self,
        win_rate: Decimal,
        win_loss_ratio: Decimal,
        kelly_fraction: Decimal = Decimal("0.25"),
        max_position_pct: Decimal = Decimal("0.20"),
    ) -> None:
        """Initialize Kelly sizer.

        Args:
            win_rate: Historical win rate (0-1)
            win_loss_ratio: Avg win / avg loss
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            max_position_pct: Maximum position as % of equity
        """
        self._win_rate = win_rate
        self._win_loss_ratio = win_loss_ratio
        self._kelly_fraction = kelly_fraction
        self._max_position_pct = max_position_pct

    def calculate_size(
        self,
        symbol: Symbol,
        price: Price,
        portfolio: PortfolioState,
        bars: Sequence[Bar],
    ) -> Quantity:
        if price <= Decimal("0"):
            return Quantity(Decimal("0"))

        # Kelly formula: f* = p - (1-p)/b
        # where p = win rate, b = win/loss ratio
        p = self._win_rate
        b = self._win_loss_ratio

        if b <= Decimal("0"):
            return Quantity(Decimal("0"))

        kelly_pct = p - (1 - p) / b

        # Apply Kelly fraction and cap
        position_pct = min(
            kelly_pct * self._kelly_fraction,
            self._max_position_pct,
        )

        if position_pct <= Decimal("0"):
            return Quantity(Decimal("0"))

        dollars = portfolio.equity * position_pct
        shares = dollars / price

        return Quantity(max(Decimal("0"), shares.quantize(Decimal("1"))))
