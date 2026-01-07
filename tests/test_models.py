"""Tests for core models."""

from decimal import Decimal

import pytest

from stockbot.core.models import Bar, Order, PortfolioState, Position
from stockbot.core.types import (
    OrderSide,
    OrderType,
    Price,
    Quantity,
    Symbol,
    Timeframe,
    Timestamp,
)


class TestBar:
    """Tests for Bar model."""

    def test_bar_creation(self, sample_bar: Bar) -> None:
        """Test basic bar creation."""
        assert sample_bar.symbol == Symbol("AAPL")
        assert sample_bar.open == Price(Decimal("150.00"))
        assert sample_bar.high == Price(Decimal("152.00"))
        assert sample_bar.low == Price(Decimal("149.00"))
        assert sample_bar.close == Price(Decimal("151.00"))

    def test_bar_is_immutable(self, sample_bar: Bar) -> None:
        """Test that bars are frozen."""
        with pytest.raises(AttributeError):
            sample_bar.close = Price(Decimal("999"))  # type: ignore

    def test_bar_invalid_high_low(self, sample_symbol: Symbol, sample_timestamp: Timestamp) -> None:
        """Test that invalid high/low raises error."""
        with pytest.raises(ValueError, match="High .* must be >= Low"):
            Bar(
                symbol=sample_symbol,
                timestamp=sample_timestamp,
                open=Price(Decimal("150")),
                high=Price(Decimal("140")),  # Less than low!
                low=Price(Decimal("145")),
                close=Price(Decimal("142")),
                volume=Quantity(Decimal("1000")),
            )

    def test_bar_invalid_high_open(self, sample_symbol: Symbol, sample_timestamp: Timestamp) -> None:
        """Test that open above high raises error."""
        with pytest.raises(ValueError, match="High .* must be >= Open"):
            Bar(
                symbol=sample_symbol,
                timestamp=sample_timestamp,
                open=Price(Decimal("155")),  # Above high!
                high=Price(Decimal("150")),
                low=Price(Decimal("145")),
                close=Price(Decimal("148")),
                volume=Quantity(Decimal("1000")),
            )


class TestOrder:
    """Tests for Order model."""

    def test_order_creation(self, sample_order: Order) -> None:
        """Test basic order creation."""
        assert sample_order.symbol == Symbol("AAPL")
        assert sample_order.side == OrderSide.BUY
        assert sample_order.quantity == Quantity(Decimal("100"))
        assert sample_order.order_type == OrderType.MARKET

    def test_order_has_id(self, sample_order: Order) -> None:
        """Test that orders get auto-generated IDs."""
        assert sample_order.id is not None
        assert len(sample_order.id) > 0

    def test_order_invalid_quantity(self, sample_symbol: Symbol) -> None:
        """Test that zero/negative quantity raises error."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Order(
                symbol=sample_symbol,
                side=OrderSide.BUY,
                quantity=Quantity(Decimal("0")),
                order_type=OrderType.MARKET,
            )

    def test_limit_order_requires_price(self, sample_symbol: Symbol) -> None:
        """Test that limit orders require a limit price."""
        with pytest.raises(ValueError, match="Limit orders require"):
            Order(
                symbol=sample_symbol,
                side=OrderSide.BUY,
                quantity=Quantity(Decimal("100")),
                order_type=OrderType.LIMIT,
                # Missing limit_price
            )


class TestPosition:
    """Tests for Position model."""

    def test_long_position(self, sample_position: Position) -> None:
        """Test long position properties."""
        assert sample_position.is_long
        assert not sample_position.is_short
        assert not sample_position.is_flat

    def test_short_position(self, sample_symbol: Symbol) -> None:
        """Test short position properties."""
        position = Position(
            symbol=sample_symbol,
            quantity=Quantity(Decimal("-100")),
            average_price=Price(Decimal("150.00")),
        )
        assert position.is_short
        assert not position.is_long
        assert not position.is_flat

    def test_flat_position(self, sample_symbol: Symbol) -> None:
        """Test flat position properties."""
        position = Position(
            symbol=sample_symbol,
            quantity=Quantity(Decimal("0")),
            average_price=Price(Decimal("0")),
        )
        assert position.is_flat
        assert not position.is_long
        assert not position.is_short


class TestPortfolioState:
    """Tests for PortfolioState model."""

    def test_portfolio_equity(self, sample_portfolio: PortfolioState) -> None:
        """Test equity calculation."""
        # Cash: 85000, Position: 100 shares @ 150 = 15000
        # Total: 100000
        expected = Price(Decimal("100000.00"))
        assert sample_portfolio.equity == expected

    def test_empty_portfolio_equity(self, empty_portfolio: PortfolioState) -> None:
        """Test equity with no positions."""
        assert empty_portfolio.equity == Price(Decimal("100000.00"))

    def test_portfolio_with_unrealized_pnl(
        self, sample_symbol: Symbol, sample_timestamp: Timestamp
    ) -> None:
        """Test equity includes unrealized PnL."""
        position = Position(
            symbol=sample_symbol,
            quantity=Quantity(Decimal("100")),
            average_price=Price(Decimal("150.00")),
            unrealized_pnl=Price(Decimal("500.00")),  # Gained $500
        )

        portfolio = PortfolioState(
            timestamp=sample_timestamp,
            cash=Price(Decimal("85000.00")),
            positions={sample_symbol: position},
        )

        # Cash: 85000 + Position value: 15000 + Unrealized: 500 = 100500
        expected = Price(Decimal("100500.00"))
        assert portfolio.equity == expected
