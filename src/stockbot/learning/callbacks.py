"""Callbacks for integrating learning systems with trading engines."""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from stockbot.core.models import TradeRecord
from stockbot.learning.selector import StrategySelector, StrategyStats
from stockbot.monitoring.logger import get_logger

logger = get_logger("learning.callbacks")


class TradeCallback(ABC):
    """Abstract callback for trade events."""

    @abstractmethod
    def on_trade_complete(self, trade: TradeRecord) -> None:
        """Called when a trade is completed (position closed).

        Args:
            trade: The completed trade record
        """
        ...

    @abstractmethod
    def on_session_end(self) -> None:
        """Called when a trading session ends."""
        ...


class SelectorCallback(TradeCallback):
    """Callback that feeds trade results to a strategy selector.

    This bridges the gap between trading engines and the learning systems,
    allowing selectors to learn from actual trade performance.
    """

    def __init__(
        self,
        selector: StrategySelector,
        reward_type: str = "pnl",  # "pnl", "return_pct", "binary"
        persistence_path: Optional[Path] = None,
    ) -> None:
        """Initialize the callback.

        Args:
            selector: Strategy selector to update
            reward_type: How to calculate reward from trade
                - "pnl": Use raw P&L as reward
                - "return_pct": Use percentage return
                - "binary": +1 for profit, -1 for loss
            persistence_path: Optional path to save/load selector state
        """
        self._selector = selector
        self._reward_type = reward_type
        self._persistence_path = persistence_path
        self._trades_this_session: list[TradeRecord] = []

        # Load saved state if available
        if persistence_path and persistence_path.exists():
            self._load_state()

    def on_trade_complete(self, trade: TradeRecord) -> None:
        """Update selector with trade result."""
        self._trades_this_session.append(trade)

        # Calculate reward
        reward = self._calculate_reward(trade)
        success = float(trade.pnl) > 0

        # The trade's strategy_id tells us which strategy made this trade
        strategy_name = trade.strategy_id

        # Update the selector
        self._selector.update(strategy_name, reward, success)

        logger.debug(
            f"Updated selector: strategy={strategy_name}, "
            f"reward={reward:.4f}, success={success}"
        )

    def _calculate_reward(self, trade: TradeRecord) -> float:
        """Calculate reward from trade based on reward_type."""
        pnl = float(trade.pnl)

        if self._reward_type == "pnl":
            return pnl

        elif self._reward_type == "return_pct":
            # Return as percentage of entry value
            entry_value = float(trade.entry_price * trade.quantity)
            if entry_value > 0:
                return (pnl / entry_value) * 100
            return 0.0

        elif self._reward_type == "binary":
            return 1.0 if pnl > 0 else -1.0

        else:
            return pnl

    def on_session_end(self) -> None:
        """Save state and log summary."""
        if self._persistence_path:
            self._save_state()

        # Log session summary
        total_trades = len(self._trades_this_session)
        if total_trades > 0:
            total_pnl = sum(float(t.pnl) for t in self._trades_this_session)
            winning = sum(1 for t in self._trades_this_session if float(t.pnl) > 0)

            logger.info(
                f"Session complete: trades={total_trades}, "
                f"pnl=${total_pnl:.2f}, win_rate={winning/total_trades*100:.1f}%"
            )

        self._trades_this_session = []

    def _save_state(self) -> None:
        """Save selector state to file."""
        if not self._persistence_path:
            return

        state = {
            "stats": {
                name: {
                    "name": stats.name,
                    "n_selections": stats.n_selections,
                    "n_successes": stats.n_successes,
                    "total_reward": stats.total_reward,
                    "rewards": stats.rewards[-1000:],  # Keep last 1000
                }
                for name, stats in self._selector.stats.items()
            }
        }

        # Add selector-specific state
        if hasattr(self._selector, "_epsilon"):
            state["epsilon"] = self._selector._epsilon
        if hasattr(self._selector, "_total_selections"):
            state["total_selections"] = self._selector._total_selections

        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persistence_path, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved selector state to {self._persistence_path}")
        except Exception as e:
            logger.error(f"Failed to save selector state: {e}")

    def _load_state(self) -> None:
        """Load selector state from file."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        try:
            with open(self._persistence_path) as f:
                state = json.load(f)

            # Restore stats
            for name, saved_stats in state.get("stats", {}).items():
                if name in self._selector.stats:
                    stats = self._selector.stats[name]
                    stats.n_selections = saved_stats["n_selections"]
                    stats.n_successes = saved_stats["n_successes"]
                    stats.total_reward = saved_stats["total_reward"]
                    stats.rewards = saved_stats["rewards"]

            # Restore selector-specific state
            if "epsilon" in state and hasattr(self._selector, "_epsilon"):
                self._selector._epsilon = state["epsilon"]
            if "total_selections" in state and hasattr(
                self._selector, "_total_selections"
            ):
                self._selector._total_selections = state["total_selections"]

            logger.info(f"Loaded selector state from {self._persistence_path}")

        except Exception as e:
            logger.error(f"Failed to load selector state: {e}")

    @property
    def selector(self) -> StrategySelector:
        """Get the underlying selector."""
        return self._selector


class CompositeCallback(TradeCallback):
    """Combines multiple callbacks."""

    def __init__(self, callbacks: list[TradeCallback]) -> None:
        self._callbacks = callbacks

    def on_trade_complete(self, trade: TradeRecord) -> None:
        for callback in self._callbacks:
            try:
                callback.on_trade_complete(trade)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def on_session_end(self) -> None:
        for callback in self._callbacks:
            try:
                callback.on_session_end()
            except Exception as e:
                logger.error(f"Callback error: {e}")


class MetricsCallback(TradeCallback):
    """Callback that tracks performance metrics."""

    def __init__(self) -> None:
        self._trades: list[TradeRecord] = []
        self._equity_snapshots: list[tuple[int, float]] = []

    def on_trade_complete(self, trade: TradeRecord) -> None:
        self._trades.append(trade)

    def on_session_end(self) -> None:
        pass

    @property
    def trades(self) -> list[TradeRecord]:
        return self._trades

    @property
    def total_pnl(self) -> float:
        return sum(float(t.pnl) for t in self._trades)

    @property
    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        winning = sum(1 for t in self._trades if float(t.pnl) > 0)
        return winning / len(self._trades)

    def get_strategy_breakdown(self) -> dict[str, dict[str, float]]:
        """Get performance breakdown by strategy."""
        by_strategy: dict[str, list[TradeRecord]] = {}

        for trade in self._trades:
            if trade.strategy_id not in by_strategy:
                by_strategy[trade.strategy_id] = []
            by_strategy[trade.strategy_id].append(trade)

        result = {}
        for strategy, trades in by_strategy.items():
            pnl = sum(float(t.pnl) for t in trades)
            wins = sum(1 for t in trades if float(t.pnl) > 0)
            result[strategy] = {
                "trades": len(trades),
                "pnl": pnl,
                "win_rate": wins / len(trades) if trades else 0,
            }

        return result
