"""Execution context for dependency injection.

The ExecutionContext encapsulates all environment-specific components,
enabling the same strategy code to run in backtest, paper, and live modes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from stockbot.config.settings import BacktestConfig, RiskConfig, Settings
from stockbot.core.interfaces import Broker, DataProvider, RiskManager
from stockbot.core.types import Environment, Price, Symbol, Timestamp
from stockbot.data.providers.parquet import ParquetDataProvider
from stockbot.engine.clock import RealClock, SimulatedClock


@dataclass
class ExecutionContext:
    """Encapsulates environment-specific components.

    Strategy code receives this context indirectly through the engine.
    Components are swapped based on environment (backtest/paper/live).
    """

    environment: Environment
    clock: Union[SimulatedClock, RealClock]
    data_provider: DataProvider
    broker: Broker
    risk_manager: RiskManager

    # Trading configuration
    symbols: list[Symbol]
    initial_capital: Price

    @classmethod
    def for_backtest(
        cls,
        config: BacktestConfig,
        risk_config: RiskConfig,
        data_dir: Path,
    ) -> "ExecutionContext":
        """Create a context for backtesting.

        Args:
            config: Backtest configuration
            risk_config: Risk management configuration
            data_dir: Directory containing parquet data files

        Returns:
            ExecutionContext configured for backtesting
        """
        from stockbot.execution.broker_sim import SimulatedBroker
        from stockbot.risk.manager import BasicRiskManager

        # Parse start date to timestamp
        from datetime import datetime, timezone as tz

        start_dt = datetime.fromisoformat(config.start_date).replace(tzinfo=tz.utc)
        start_ts = Timestamp(int(start_dt.timestamp() * 1_000_000_000))

        return cls(
            environment=Environment.BACKTEST,
            clock=SimulatedClock(start_ts),
            data_provider=ParquetDataProvider(data_dir),
            broker=SimulatedBroker(
                initial_capital=config.initial_capital,
                commission=config.commission,
                slippage_pct=config.slippage_pct,
                seed=config.seed,
            ),
            risk_manager=BasicRiskManager(risk_config),
            symbols=config.symbols,
            initial_capital=config.initial_capital,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "ExecutionContext":
        """Create a context from settings.

        Args:
            settings: Application settings

        Returns:
            ExecutionContext configured according to settings

        Raises:
            ValueError: If settings are incomplete for the environment
        """
        if settings.environment == Environment.BACKTEST:
            if settings.backtest is None:
                raise ValueError("BacktestConfig required for backtest environment")
            return cls.for_backtest(
                config=settings.backtest,
                risk_config=settings.risk,
                data_dir=settings.data_dir,
            )

        # Paper and live trading contexts would be implemented similarly
        # but require additional components (live data feed, real broker, etc.)
        raise NotImplementedError(
            f"Environment {settings.environment.name} not yet implemented"
        )
