"""Base strategy implementation."""

from abc import abstractmethod
from typing import Optional

from stockbot.core.interfaces import Strategy
from stockbot.core.models import MarketState
from stockbot.core.types import Signal, Symbol
from stockbot.monitoring.logger import get_logger


class BaseStrategy(Strategy):
    """Base class for strategy implementations.

    Provides common functionality like logging and state management.
    Subclasses implement observe() and decide() for trading logic.
    """

    def __init__(self, symbols: list[Symbol], strategy_name: Optional[str] = None) -> None:
        """Initialize the strategy.

        Args:
            symbols: Symbols this strategy will trade
            strategy_name: Optional name override
        """
        self._symbols = symbols
        self._strategy_name = strategy_name or self.__class__.__name__
        self._logger = get_logger(f"strategy.{self._strategy_name}")
        self._current_state: Optional[MarketState] = None
        self._step_count = 0

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self._strategy_name

    @property
    def symbols(self) -> list[Symbol]:
        """Get symbols being traded."""
        return self._symbols

    @property
    def current_state(self) -> Optional[MarketState]:
        """Get the most recent market state."""
        return self._current_state

    def observe(self, state: MarketState) -> None:
        """Store market state and call subclass observer.

        Args:
            state: Current market state
        """
        self._current_state = state
        self._step_count += 1
        self._on_observe(state)

    @abstractmethod
    def _on_observe(self, state: MarketState) -> None:
        """Subclass implementation of observation logic.

        Called after state is stored.

        Args:
            state: Current market state
        """
        ...

    @abstractmethod
    def decide(self) -> dict[Symbol, Signal]:
        """Generate trading signals.

        Must be implemented by subclasses.

        Returns:
            Dict mapping symbols to signals
        """
        ...

    def update(self, reward: float) -> None:
        """Update with reward signal.

        Override in learning strategies.

        Args:
            reward: Reward value
        """
        pass

    def reset(self) -> None:
        """Reset strategy state."""
        self._current_state = None
        self._step_count = 0
        self._on_reset()

    def _on_reset(self) -> None:
        """Subclass reset hook.

        Override to reset strategy-specific state.
        """
        pass
