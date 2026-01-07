"""Configuration settings for the trading system."""

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from stockbot.core.exceptions import MissingCredentialsError
from stockbot.core.types import Environment, Price, Quantity, Symbol, Timeframe


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""

    api_key: str
    secret_key: str
    paper: bool = True  # Use paper trading endpoint

    @property
    def base_url(self) -> str:
        """Get the appropriate API base URL."""
        if self.paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"

    @property
    def data_url(self) -> str:
        """Get the data API URL."""
        return "https://data.alpaca.markets"


@dataclass
class RiskConfig:
    """Risk management configuration."""

    max_position_size: Quantity = Quantity(Decimal("1000"))  # Max shares per position
    max_position_value: Price = Price(Decimal("10000"))  # Max $ per position
    max_portfolio_risk: Decimal = Decimal("0.02")  # Max 2% portfolio risk per trade
    max_daily_loss: Price = Price(Decimal("1000"))  # Max daily loss before halt
    max_open_positions: int = 10  # Maximum concurrent positions
    max_orders_per_minute: int = 10  # Rate limiting


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    start_date: str  # ISO format: "2024-01-01"
    end_date: str  # ISO format: "2024-12-31"
    symbols: list[Symbol] = field(default_factory=list)
    initial_capital: Price = Price(Decimal("100000"))
    timeframe: Timeframe = Timeframe.MINUTE_1
    commission: Decimal = Decimal("0")  # Per-share commission
    slippage_pct: Decimal = Decimal("0.001")  # 0.1% slippage
    seed: int = 42  # Random seed for reproducibility


@dataclass
class Settings:
    """Main settings container."""

    environment: Environment = Environment.BACKTEST
    data_dir: Path = Path("./data")
    log_level: str = "INFO"

    # Sub-configs
    alpaca: Optional[AlpacaConfig] = None
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: Optional[BacktestConfig] = None


def load_settings(env_file: Optional[Path] = None) -> Settings:
    """Load settings from environment variables.

    Args:
        env_file: Optional path to .env file

    Returns:
        Populated Settings object

    Raises:
        MissingCredentialsError: If required credentials are missing for live/paper mode
    """
    # Load .env file if it exists
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Determine environment
    env_str = os.getenv("STOCKBOT_ENV", "backtest").upper()
    environment = Environment[env_str]

    # Load Alpaca config
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    alpaca_config: Optional[AlpacaConfig] = None
    if api_key and secret_key:
        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        alpaca_config = AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
    elif environment in (Environment.PAPER, Environment.LIVE):
        raise MissingCredentialsError(
            f"Alpaca credentials required for {environment.name} mode. "
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        )

    # Load data directory
    data_dir = Path(os.getenv("STOCKBOT_DATA_DIR", "./data"))

    # Load log level
    log_level = os.getenv("STOCKBOT_LOG_LEVEL", "INFO")

    return Settings(
        environment=environment,
        data_dir=data_dir,
        log_level=log_level,
        alpaca=alpaca_config,
        risk=RiskConfig(),
    )
