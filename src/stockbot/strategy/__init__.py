"""Trading strategies."""

from stockbot.strategy.base import BaseStrategy
from stockbot.strategy.baseline import SMAcrossoverStrategy

__all__ = [
    "BaseStrategy",
    "SMAcrossoverStrategy",
]
