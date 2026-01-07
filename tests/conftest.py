"""Pytest fixtures for Stockbot tests."""

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import pytest

from stockbot.config.settings import BacktestConfig, RiskConfig
from stockbot.core.models import Bar, Order, PortfolioState, Position
from stockbot.core.types import (
    OrderSide,
    OrderType,
    Price,
    Quantity,
    Signal,
    Symbol,
    Timeframe,
    Timestamp,
)


@pytest.fixture
def sample_symbol() -> Symbol:
    """Sample stock symbol."""
    return Symbol("AAPL")


@pytest.fixture
def sample_timestamp() -> Timestamp:
    """Sample timestamp (2024-01-02 9:30 AM ET)."""
    return Timestamp(1704203400000000000)


@pytest.fixture
def sample_bar(sample_symbol: Symbol, sample_timestamp: Timestamp) -> Bar:
    """Sample bar for testing."""
    return Bar(
        symbol=sample_symbol,
        timestamp=sample_timestamp,
        open=Price(Decimal("150.00")),
        high=Price(Decimal("152.00")),
        low=Price(Decimal("149.00")),
        close=Price(Decimal("151.00")),
        volume=Quantity(Decimal("1000000")),
        timeframe=Timeframe.MINUTE_1,
    )


@pytest.fixture
def sample_bars(sample_symbol: Symbol) -> list[Bar]:
    """List of sample bars for testing."""
    base_ts = 1704203400000000000
    bars = []

    prices = [
        (150.00, 152.00, 149.00, 151.00),
        (151.00, 153.00, 150.00, 152.00),
        (152.00, 154.00, 151.00, 153.00),
        (153.00, 155.00, 152.00, 154.00),
        (154.00, 156.00, 153.00, 155.00),
    ]

    for i, (o, h, l, c) in enumerate(prices):
        bars.append(
            Bar(
                symbol=sample_symbol,
                timestamp=Timestamp(base_ts + i * 60_000_000_000),
                open=Price(Decimal(str(o))),
                high=Price(Decimal(str(h))),
                low=Price(Decimal(str(l))),
                close=Price(Decimal(str(c))),
                volume=Quantity(Decimal("100000")),
                timeframe=Timeframe.MINUTE_1,
            )
        )

    return bars


@pytest.fixture
def sample_order(sample_symbol: Symbol, sample_timestamp: Timestamp) -> Order:
    """Sample market order."""
    return Order(
        symbol=sample_symbol,
        side=OrderSide.BUY,
        quantity=Quantity(Decimal("100")),
        order_type=OrderType.MARKET,
        created_at=sample_timestamp,
    )


@pytest.fixture
def sample_position(sample_symbol: Symbol) -> Position:
    """Sample long position."""
    return Position(
        symbol=sample_symbol,
        quantity=Quantity(Decimal("100")),
        average_price=Price(Decimal("150.00")),
    )


@pytest.fixture
def sample_portfolio(
    sample_symbol: Symbol, sample_position: Position, sample_timestamp: Timestamp
) -> PortfolioState:
    """Sample portfolio state."""
    return PortfolioState(
        timestamp=sample_timestamp,
        cash=Price(Decimal("85000.00")),
        positions={sample_symbol: sample_position},
    )


@pytest.fixture
def empty_portfolio(sample_timestamp: Timestamp) -> PortfolioState:
    """Empty portfolio with just cash."""
    return PortfolioState(
        timestamp=sample_timestamp,
        cash=Price(Decimal("100000.00")),
        positions={},
    )


@pytest.fixture
def temp_data_dir() -> Iterator[Path]:
    """Temporary directory for test data."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_backtest_config(sample_symbol: Symbol) -> BacktestConfig:
    """Sample backtest configuration."""
    return BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-31",
        symbols=[sample_symbol],
        initial_capital=Price(Decimal("100000")),
        timeframe=Timeframe.DAY_1,
        commission=Decimal("0"),
        slippage_pct=Decimal("0.001"),
        seed=42,
    )


@pytest.fixture
def sample_risk_config() -> RiskConfig:
    """Sample risk configuration."""
    return RiskConfig(
        max_position_size=Quantity(Decimal("1000")),
        max_position_value=Price(Decimal("10000")),
        max_portfolio_risk=Decimal("0.02"),
        max_daily_loss=Price(Decimal("1000")),
        max_open_positions=10,
    )
