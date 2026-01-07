"""Learning and optimization components."""

from stockbot.learning.callbacks import (
    CompositeCallback,
    MetricsCallback,
    SelectorCallback,
    TradeCallback,
)
from stockbot.learning.features import FeatureExtractor, MarketFeatures
from stockbot.learning.optimizer import (
    GridSearchOptimizer,
    OptimizationResult,
    ParameterSpace,
    RandomSearchOptimizer,
)
from stockbot.learning.rl_agent import Action, TradingAgent, create_default_agent
from stockbot.learning.selector import (
    EnsembleStrategy,
    EpsilonGreedySelector,
    StrategySelector,
    ThompsonSamplingSelector,
    UCBSelector,
)

__all__ = [
    # Callbacks
    "CompositeCallback",
    "MetricsCallback",
    "SelectorCallback",
    "TradeCallback",
    # ML Features
    "FeatureExtractor",
    "MarketFeatures",
    # RL Agent
    "Action",
    "TradingAgent",
    "create_default_agent",
    # Optimization
    "GridSearchOptimizer",
    "OptimizationResult",
    "ParameterSpace",
    "RandomSearchOptimizer",
    # Selection
    "EnsembleStrategy",
    "EpsilonGreedySelector",
    "StrategySelector",
    "ThompsonSamplingSelector",
    "UCBSelector",
]
