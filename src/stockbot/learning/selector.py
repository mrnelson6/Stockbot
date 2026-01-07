"""Strategy selection using multi-armed bandit algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from stockbot.core.interfaces import Strategy
from stockbot.core.models import MarketState
from stockbot.core.types import Signal, Symbol
from stockbot.monitoring.logger import get_logger

logger = get_logger("learning.selector")


@dataclass
class StrategyStats:
    """Statistics for a single strategy."""

    name: str
    n_selections: int = 0
    n_successes: int = 0  # Profitable trades
    total_reward: float = 0.0
    rewards: list[float] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        """Average reward per selection."""
        if self.n_selections == 0:
            return 0.0
        return self.total_reward / self.n_selections

    @property
    def success_rate(self) -> float:
        """Fraction of successful selections."""
        if self.n_selections == 0:
            return 0.0
        return self.n_successes / self.n_selections

    @property
    def reward_std(self) -> float:
        """Standard deviation of rewards."""
        if len(self.rewards) < 2:
            return 0.0
        return float(np.std(self.rewards))


class StrategySelector(ABC):
    """Abstract base class for strategy selectors.

    Uses multi-armed bandit algorithms to select among multiple strategies,
    balancing exploration (trying different strategies) with exploitation
    (using the best-performing strategy).
    """

    def __init__(self, strategies: list[Strategy]) -> None:
        """Initialize the selector.

        Args:
            strategies: List of strategies to choose from
        """
        self._strategies = strategies
        self._stats = {s.name: StrategyStats(name=s.name) for s in strategies}
        self._current_strategy: Optional[Strategy] = None

    @property
    def strategies(self) -> list[Strategy]:
        """Get all strategies."""
        return self._strategies

    @property
    def stats(self) -> dict[str, StrategyStats]:
        """Get statistics for all strategies."""
        return self._stats

    @property
    def current_strategy(self) -> Optional[Strategy]:
        """Get currently selected strategy."""
        return self._current_strategy

    @abstractmethod
    def select(self) -> Strategy:
        """Select a strategy for the next period.

        Returns:
            Selected strategy
        """
        ...

    def update(self, strategy_name: str, reward: float, success: bool = False) -> None:
        """Update statistics after a trading period.

        Args:
            strategy_name: Name of strategy that was used
            reward: Reward received (e.g., P&L, Sharpe contribution)
            success: Whether the trade was profitable
        """
        if strategy_name not in self._stats:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return

        stats = self._stats[strategy_name]
        stats.n_selections += 1
        stats.total_reward += reward
        stats.rewards.append(reward)

        if success:
            stats.n_successes += 1

        logger.debug(
            f"Updated {strategy_name}: selections={stats.n_selections}, "
            f"mean_reward={stats.mean_reward:.4f}"
        )

    def get_strategy_by_name(self, name: str) -> Optional[Strategy]:
        """Get a strategy by name."""
        for s in self._strategies:
            if s.name == name:
                return s
        return None

    def print_stats(self) -> None:
        """Print strategy statistics."""
        print("\n" + "=" * 70)
        print("STRATEGY SELECTION STATISTICS")
        print("=" * 70)

        print(f"\n{'Strategy':<25} {'Selections':>12} {'Mean Reward':>12} {'Success %':>12}")
        print("-" * 70)

        for stats in sorted(self._stats.values(), key=lambda s: s.mean_reward, reverse=True):
            print(
                f"{stats.name:<25} {stats.n_selections:>12} "
                f"{stats.mean_reward:>12.4f} {stats.success_rate * 100:>11.1f}%"
            )

        print("=" * 70 + "\n")


class EpsilonGreedySelector(StrategySelector):
    """Epsilon-greedy strategy selection.

    With probability epsilon, select a random strategy (explore).
    Otherwise, select the strategy with highest mean reward (exploit).
    """

    def __init__(
        self,
        strategies: list[Strategy],
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        seed: int = 42,
    ) -> None:
        """Initialize epsilon-greedy selector.

        Args:
            strategies: Strategies to choose from
            epsilon: Initial exploration probability
            epsilon_decay: Decay factor for epsilon after each selection
            min_epsilon: Minimum epsilon value
            seed: Random seed
        """
        super().__init__(strategies)
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._rng = np.random.default_rng(seed)

    @property
    def epsilon(self) -> float:
        """Current epsilon value."""
        return self._epsilon

    def select(self) -> Strategy:
        """Select strategy using epsilon-greedy."""
        if self._rng.random() < self._epsilon:
            # Explore: random selection
            strategy = self._rng.choice(self._strategies)
            logger.debug(f"Epsilon-greedy: exploring with {strategy.name}")
        else:
            # Exploit: best mean reward
            # If no data, pick randomly
            best_name = max(
                self._stats.keys(),
                key=lambda n: self._stats[n].mean_reward
                if self._stats[n].n_selections > 0
                else self._rng.random(),
            )
            strategy = self.get_strategy_by_name(best_name)
            if strategy is None:
                strategy = self._rng.choice(self._strategies)
            logger.debug(f"Epsilon-greedy: exploiting with {strategy.name}")

        # Decay epsilon
        self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)

        self._current_strategy = strategy
        return strategy


class UCBSelector(StrategySelector):
    """Upper Confidence Bound (UCB1) strategy selection.

    Balances exploration and exploitation by selecting the strategy
    with highest UCB score = mean_reward + c * sqrt(ln(total) / n_selections)
    """

    def __init__(
        self,
        strategies: list[Strategy],
        exploration_constant: float = 2.0,
        seed: int = 42,
    ) -> None:
        """Initialize UCB selector.

        Args:
            strategies: Strategies to choose from
            exploration_constant: Higher = more exploration
            seed: Random seed for tie-breaking
        """
        super().__init__(strategies)
        self._c = exploration_constant
        self._total_selections = 0
        self._rng = np.random.default_rng(seed)

    def _ucb_score(self, stats: StrategyStats) -> float:
        """Calculate UCB score for a strategy."""
        if stats.n_selections == 0:
            return float("inf")  # Unexplored strategies have infinite UCB

        if self._total_selections == 0:
            return stats.mean_reward

        exploration_bonus = self._c * np.sqrt(
            np.log(self._total_selections) / stats.n_selections
        )

        return stats.mean_reward + exploration_bonus

    def select(self) -> Strategy:
        """Select strategy using UCB1."""
        # Calculate UCB scores
        scores = {name: self._ucb_score(stats) for name, stats in self._stats.items()}

        # Select highest UCB (with random tie-breaking)
        max_score = max(scores.values())
        best_names = [name for name, score in scores.items() if score == max_score]
        best_name = self._rng.choice(best_names)

        strategy = self.get_strategy_by_name(best_name)
        if strategy is None:
            strategy = self._rng.choice(self._strategies)

        self._total_selections += 1
        self._current_strategy = strategy

        logger.debug(f"UCB: selected {strategy.name} with score {scores[best_name]:.4f}")
        return strategy


class ThompsonSamplingSelector(StrategySelector):
    """Thompson Sampling strategy selection.

    Maintains a Beta distribution for each strategy's success probability,
    samples from each, and selects the highest sample.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: int = 42,
    ) -> None:
        """Initialize Thompson Sampling selector.

        Args:
            strategies: Strategies to choose from
            prior_alpha: Prior alpha for Beta distribution
            prior_beta: Prior beta for Beta distribution
            seed: Random seed
        """
        super().__init__(strategies)
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._rng = np.random.default_rng(seed)

    def _sample_theta(self, stats: StrategyStats) -> float:
        """Sample from posterior Beta distribution."""
        alpha = self._prior_alpha + stats.n_successes
        beta = self._prior_beta + (stats.n_selections - stats.n_successes)
        return float(self._rng.beta(alpha, beta))

    def select(self) -> Strategy:
        """Select strategy using Thompson Sampling."""
        # Sample from each strategy's posterior
        samples = {name: self._sample_theta(stats) for name, stats in self._stats.items()}

        # Select highest sample
        best_name = max(samples.keys(), key=lambda n: samples[n])
        strategy = self.get_strategy_by_name(best_name)

        if strategy is None:
            strategy = self._rng.choice(self._strategies)

        self._current_strategy = strategy

        logger.debug(f"Thompson: selected {strategy.name} with sample {samples[best_name]:.4f}")
        return strategy


class EnsembleStrategy(Strategy):
    """Strategy that combines signals from multiple strategies.

    Uses a selector to weight strategies based on performance.
    """

    def __init__(
        self,
        selector: StrategySelector,
        combination_method: str = "vote",  # "vote", "weighted", "best"
    ) -> None:
        """Initialize ensemble strategy.

        Args:
            selector: Strategy selector for weighting
            combination_method: How to combine signals
        """
        self._selector = selector
        self._combination_method = combination_method
        self._current_state: Optional[MarketState] = None

    @property
    def name(self) -> str:
        return f"Ensemble_{self._combination_method}"

    def observe(self, state: MarketState) -> None:
        """Pass observation to all strategies."""
        self._current_state = state
        for strategy in self._selector.strategies:
            strategy.observe(state)

    def decide(self) -> dict[Symbol, Signal]:
        """Combine signals from all strategies."""
        if self._combination_method == "best":
            # Use only the best strategy
            strategy = self._selector.select()
            return strategy.decide()

        elif self._combination_method == "vote":
            # Majority voting
            return self._majority_vote()

        elif self._combination_method == "weighted":
            # Weighted by mean reward
            return self._weighted_vote()

        else:
            raise ValueError(f"Unknown combination method: {self._combination_method}")

    def _majority_vote(self) -> dict[Symbol, Signal]:
        """Combine signals using majority voting."""
        all_signals: dict[Symbol, list[Signal]] = {}

        for strategy in self._selector.strategies:
            signals = strategy.decide()
            for symbol, signal in signals.items():
                if symbol not in all_signals:
                    all_signals[symbol] = []
                all_signals[symbol].append(signal)

        # Take majority vote for each symbol
        result: dict[Symbol, Signal] = {}
        for symbol, signals in all_signals.items():
            # Count votes (excluding HOLD)
            from collections import Counter

            votes = Counter(s for s in signals if s != Signal.HOLD)
            if votes:
                result[symbol] = votes.most_common(1)[0][0]
            else:
                result[symbol] = Signal.HOLD

        return result

    def _weighted_vote(self) -> dict[Symbol, Signal]:
        """Combine signals using performance-weighted voting."""
        all_signals: dict[Symbol, dict[Signal, float]] = {}

        for strategy in self._selector.strategies:
            stats = self._selector.stats.get(strategy.name)
            weight = stats.mean_reward if stats and stats.n_selections > 0 else 0.1

            signals = strategy.decide()
            for symbol, signal in signals.items():
                if symbol not in all_signals:
                    all_signals[symbol] = {}
                if signal not in all_signals[symbol]:
                    all_signals[symbol][signal] = 0.0
                all_signals[symbol][signal] += max(0, weight)

        # Take weighted majority for each symbol
        result: dict[Symbol, Signal] = {}
        for symbol, votes in all_signals.items():
            # Remove HOLD from voting
            votes.pop(Signal.HOLD, None)
            if votes:
                result[symbol] = max(votes.keys(), key=lambda s: votes[s])
            else:
                result[symbol] = Signal.HOLD

        return result

    def update(self, reward: float) -> None:
        """Update selector with reward."""
        if self._selector.current_strategy:
            success = reward > 0
            self._selector.update(
                self._selector.current_strategy.name,
                reward,
                success,
            )

    def reset(self) -> None:
        """Reset all strategies."""
        for strategy in self._selector.strategies:
            strategy.reset()
