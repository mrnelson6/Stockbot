"""Parameter optimization for trading strategies."""

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np

from stockbot.config.settings import BacktestConfig, RiskConfig
from stockbot.core.interfaces import Strategy
from stockbot.core.types import Price, Symbol, Timeframe
from stockbot.engine.backtest import BacktestEngine, BacktestResult
from stockbot.monitoring.logger import get_logger

logger = get_logger("learning.optimizer")


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""

    name: str
    param_type: str  # "int", "float", "choice"
    low: Optional[float] = None  # For numeric types
    high: Optional[float] = None  # For numeric types
    choices: Optional[list[Any]] = None  # For choice type
    step: Optional[float] = None  # Optional step size for grid search

    def sample(self, rng: np.random.Generator) -> Any:
        """Sample a random value for this parameter."""
        if self.param_type == "int":
            return int(rng.integers(int(self.low or 0), int(self.high or 100) + 1))
        elif self.param_type == "float":
            return float(rng.uniform(self.low or 0, self.high or 1))
        elif self.param_type == "choice":
            return rng.choice(self.choices or [])
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")

    def grid_values(self) -> list[Any]:
        """Get all values for grid search."""
        if self.param_type == "choice":
            return list(self.choices or [])
        elif self.param_type == "int":
            step = int(self.step or 1)
            return list(range(int(self.low or 0), int(self.high or 100) + 1, step))
        elif self.param_type == "float":
            step = self.step or 0.1
            values = []
            v = self.low or 0
            while v <= (self.high or 1):
                values.append(v)
                v += step
            return values
        else:
            return []


@dataclass
class ParameterSpace:
    """Definition of the parameter search space."""

    parameters: list[ParameterSpec] = field(default_factory=list)

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        step: int = 1,
    ) -> "ParameterSpace":
        """Add an integer parameter."""
        self.parameters.append(
            ParameterSpec(name=name, param_type="int", low=low, high=high, step=step)
        )
        return self

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        step: Optional[float] = None,
    ) -> "ParameterSpace":
        """Add a float parameter."""
        self.parameters.append(
            ParameterSpec(name=name, param_type="float", low=low, high=high, step=step)
        )
        return self

    def add_choice(
        self,
        name: str,
        choices: list[Any],
    ) -> "ParameterSpace":
        """Add a categorical parameter."""
        self.parameters.append(
            ParameterSpec(name=name, param_type="choice", choices=choices)
        )
        return self

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample a random parameter configuration."""
        return {spec.name: spec.sample(rng) for spec in self.parameters}

    def grid(self) -> Iterator[dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        if not self.parameters:
            yield {}
            return

        names = [spec.name for spec in self.parameters]
        value_lists = [spec.grid_values() for spec in self.parameters]

        for values in itertools.product(*value_lists):
            yield dict(zip(names, values))

    def grid_size(self) -> int:
        """Calculate total number of grid search combinations."""
        if not self.parameters:
            return 1
        size = 1
        for spec in self.parameters:
            size *= len(spec.grid_values())
        return size


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    best_params: dict[str, Any]
    best_score: float
    best_result: Optional[BacktestResult]
    all_results: list[tuple[dict[str, Any], float, BacktestResult]]
    metric: str
    total_trials: int


class StrategyFactory:
    """Factory for creating strategy instances with parameters."""

    def __init__(
        self,
        strategy_class: type,
        symbols: list[Symbol],
        fixed_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the factory.

        Args:
            strategy_class: Strategy class to instantiate
            symbols: Symbols for the strategy
            fixed_params: Parameters that don't change
        """
        self._strategy_class = strategy_class
        self._symbols = symbols
        self._fixed_params = fixed_params or {}

    def create(self, params: dict[str, Any]) -> Strategy:
        """Create a strategy instance with given parameters."""
        all_params = {**self._fixed_params, **params}
        return self._strategy_class(symbols=self._symbols, **all_params)


class Optimizer(ABC):
    """Abstract base class for parameter optimizers."""

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        backtest_config: BacktestConfig,
        risk_config: RiskConfig,
        data_dir: Path,
        metric: str = "sharpe_ratio",
        higher_is_better: bool = True,
    ) -> None:
        """Initialize the optimizer.

        Args:
            strategy_factory: Factory for creating strategies
            backtest_config: Backtest configuration
            risk_config: Risk configuration
            data_dir: Directory with market data
            metric: Metric to optimize (sharpe_ratio, total_return_pct, etc.)
            higher_is_better: Whether higher metric values are better
        """
        self._factory = strategy_factory
        self._backtest_config = backtest_config
        self._risk_config = risk_config
        self._data_dir = data_dir
        self._metric = metric
        self._higher_is_better = higher_is_better

    def _run_backtest(self, params: dict[str, Any]) -> tuple[float, BacktestResult]:
        """Run a backtest with given parameters and return score."""
        strategy = self._factory.create(params)

        engine = BacktestEngine(
            config=self._backtest_config,
            risk_config=self._risk_config,
            data_dir=self._data_dir,
            strategy=strategy,
        )

        result = engine.run()

        # Extract metric
        score = self._extract_metric(result)

        return score, result

    def _extract_metric(self, result: BacktestResult) -> float:
        """Extract the optimization metric from backtest result."""
        if self._metric == "sharpe_ratio":
            # Calculate Sharpe from equity curve
            from stockbot.monitoring.metrics import calculate_sharpe_ratio, calculate_returns_from_equity
            returns = calculate_returns_from_equity(result.equity_curve)
            return float(calculate_sharpe_ratio(returns))
        elif self._metric == "total_return_pct":
            return float(result.total_return_pct)
        elif self._metric == "total_return":
            return float(result.total_return)
        elif self._metric == "win_rate":
            return float(result.win_rate)
        elif self._metric == "profit_factor":
            from stockbot.monitoring.metrics import calculate_trade_metrics
            metrics = calculate_trade_metrics(result.trades)
            return float(metrics["profit_factor"])
        elif self._metric == "max_drawdown":
            # For drawdown, lower is better, so negate
            return -float(result.max_drawdown_pct)
        else:
            raise ValueError(f"Unknown metric: {self._metric}")

    @abstractmethod
    def optimize(self, parameter_space: ParameterSpace) -> OptimizationResult:
        """Run the optimization.

        Args:
            parameter_space: Space of parameters to search

        Returns:
            OptimizationResult with best parameters found
        """
        ...


class GridSearchOptimizer(Optimizer):
    """Exhaustive grid search over parameter space."""

    def optimize(self, parameter_space: ParameterSpace) -> OptimizationResult:
        """Run grid search optimization."""
        grid_size = parameter_space.grid_size()
        logger.info(f"Starting grid search with {grid_size} combinations")

        results: list[tuple[dict[str, Any], float, BacktestResult]] = []
        best_score = float("-inf") if self._higher_is_better else float("inf")
        best_params: dict[str, Any] = {}
        best_result: Optional[BacktestResult] = None

        for i, params in enumerate(parameter_space.grid()):
            logger.info(f"Trial {i + 1}/{grid_size}: {params}")

            try:
                score, result = self._run_backtest(params)
                results.append((params, score, result))

                is_better = (
                    score > best_score if self._higher_is_better else score < best_score
                )

                if is_better:
                    best_score = score
                    best_params = params
                    best_result = result
                    logger.info(f"New best: {self._metric}={score:.4f}")

            except Exception as e:
                logger.warning(f"Trial failed: {e}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=results,
            metric=self._metric,
            total_trials=len(results),
        )


class RandomSearchOptimizer(Optimizer):
    """Random search over parameter space."""

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        backtest_config: BacktestConfig,
        risk_config: RiskConfig,
        data_dir: Path,
        metric: str = "sharpe_ratio",
        higher_is_better: bool = True,
        n_trials: int = 50,
        seed: int = 42,
    ) -> None:
        """Initialize random search optimizer.

        Args:
            n_trials: Number of random trials
            seed: Random seed for reproducibility
        """
        super().__init__(
            strategy_factory, backtest_config, risk_config, data_dir, metric, higher_is_better
        )
        self._n_trials = n_trials
        self._rng = np.random.default_rng(seed)

    def optimize(self, parameter_space: ParameterSpace) -> OptimizationResult:
        """Run random search optimization."""
        logger.info(f"Starting random search with {self._n_trials} trials")

        results: list[tuple[dict[str, Any], float, BacktestResult]] = []
        best_score = float("-inf") if self._higher_is_better else float("inf")
        best_params: dict[str, Any] = {}
        best_result: Optional[BacktestResult] = None

        for i in range(self._n_trials):
            params = parameter_space.sample(self._rng)
            logger.info(f"Trial {i + 1}/{self._n_trials}: {params}")

            try:
                score, result = self._run_backtest(params)
                results.append((params, score, result))

                is_better = (
                    score > best_score if self._higher_is_better else score < best_score
                )

                if is_better:
                    best_score = score
                    best_params = params
                    best_result = result
                    logger.info(f"New best: {self._metric}={score:.4f}")

            except Exception as e:
                logger.warning(f"Trial failed: {e}")

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=results,
            metric=self._metric,
            total_trials=len(results),
        )


def print_optimization_result(result: OptimizationResult) -> None:
    """Print optimization results in a formatted way."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nMetric: {result.metric}")
    print(f"Total Trials: {result.total_trials}")

    print(f"\n{'BEST PARAMETERS':-^60}")
    for name, value in result.best_params.items():
        print(f"  {name}: {value}")

    print(f"\n{'BEST SCORE':-^60}")
    print(f"  {result.metric}: {result.best_score:.4f}")

    if result.best_result:
        print(f"\n{'BEST BACKTEST RESULTS':-^60}")
        print(f"  Total Return: ${result.best_result.total_return:,.2f}")
        print(f"  Total Return %: {result.best_result.total_return_pct:.2f}%")
        print(f"  Total Trades: {result.best_result.total_trades}")
        print(f"  Win Rate: {result.best_result.win_rate:.1f}%")
        print(f"  Max Drawdown: {result.best_result.max_drawdown_pct:.2f}%")

    print(f"\n{'TOP 5 CONFIGURATIONS':-^60}")
    # Sort by score
    sorted_results = sorted(
        result.all_results,
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    for i, (params, score, _) in enumerate(sorted_results, 1):
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {i}. {result.metric}={score:.4f} | {params_str}")

    print("=" * 60 + "\n")
