"""Reinforcement Learning agent for trading.

This is a true machine learning approach where the agent:
1. Observes market features (doesn't know which matter)
2. Decides target position size (-100% to +100%)
3. Receives rewards (trading P&L)
4. Learns which features predict profitable positions

Uses Q-learning with a neural network for function approximation.
The agent outputs a target position, not just BUY/SELL/HOLD.
"""

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from stockbot.learning.features import FeatureExtractor, MarketFeatures
from stockbot.monitoring.logger import get_logger

logger = get_logger("rl_agent")


# Position levels the agent can choose
# Negative = short, 0 = flat, positive = long
POSITION_LEVELS = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0]
# Meanings: full short, half short, flat, quarter long, half long, 3/4 long, full long


@dataclass
class Experience:
    """A single experience tuple for replay."""
    state: np.ndarray
    action: int  # Index into POSITION_LEVELS
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for stable learning."""

    def __init__(self, capacity: int = 10000) -> None:
        self._buffer: deque = deque(maxlen=capacity)

    def add(self, experience: Experience) -> None:
        self._buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        indices = np.random.choice(
            len(self._buffer),
            size=min(batch_size, len(self._buffer)),
            replace=False
        )
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class NeuralNetwork:
    """Neural network for Q-value approximation.

    Learns which features matter for predicting the value of each position size.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = [128, 64],
        output_size: int = len(POSITION_LEVELS),
        learning_rate: float = 0.001,
    ) -> None:
        self.lr = learning_rate
        self.layers = []
        self.biases = []

        # Build layers
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            # Xavier initialization
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i+1])
            self.layers.append(w)
            self.biases.append(b)

        # Cache for backprop
        self._activations = []
        self._z_values = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        x = np.atleast_2d(x)
        self._activations = [x]
        self._z_values = []

        for i, (w, b) in enumerate(zip(self.layers, self.biases)):
            z = self._activations[-1] @ w + b
            self._z_values.append(z)

            # ReLU for hidden layers, linear for output
            if i < len(self.layers) - 1:
                a = np.maximum(0, z)
            else:
                a = z
            self._activations.append(a)

        return self._activations[-1]

    def backward(self, x: np.ndarray, target: np.ndarray) -> float:
        """Backward pass with gradient descent."""
        batch_size = x.shape[0] if x.ndim > 1 else 1
        x = np.atleast_2d(x)
        target = np.atleast_2d(target)

        # Forward
        output = self.forward(x)
        loss = np.mean((output - target) ** 2)

        # Backward
        delta = 2 * (output - target) / batch_size

        for i in range(len(self.layers) - 1, -1, -1):
            # Gradient for weights and biases
            dw = self._activations[i].T @ delta
            db = np.sum(delta, axis=0)

            # Clip gradients
            dw = np.clip(dw, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)

            # Update
            self.layers[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

            # Propagate delta (if not input layer)
            if i > 0:
                delta = (delta @ self.layers[i].T) * (self._z_values[i-1] > 0)

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.atleast_2d(x))

    def copy_from(self, other: "NeuralNetwork") -> None:
        for i in range(len(self.layers)):
            self.layers[i] = other.layers[i].copy()
            self.biases[i] = other.biases[i].copy()

    def get_weights(self) -> dict:
        return {
            "layers": [l.tolist() for l in self.layers],
            "biases": [b.tolist() for b in self.biases],
        }

    def set_weights(self, weights: dict) -> None:
        self.layers = [np.array(l) for l in weights["layers"]]
        self.biases = [np.array(b) for b in weights["biases"]]


class TradingAgent:
    """RL trading agent that learns both WHAT to do and HOW MUCH.

    The agent outputs a target position size:
    - -100% = fully short
    - 0% = flat (no position)
    - +100% = fully long

    It learns the optimal position size for each market condition.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        hidden_sizes: list[int] = [128, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        target_update_freq: int = 100,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)

        self._feature_extractor = feature_extractor
        self._gamma = gamma
        self._epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq

        # Networks
        input_size = feature_extractor.feature_count
        n_actions = len(POSITION_LEVELS)

        self._q_network = NeuralNetwork(input_size, hidden_sizes, n_actions, learning_rate)
        self._target_network = NeuralNetwork(input_size, hidden_sizes, n_actions, learning_rate)
        self._target_network.copy_from(self._q_network)

        # Experience replay
        self._replay_buffer = ReplayBuffer(capacity=50000)

        # State
        self._steps = 0
        self._training_losses: list[float] = []
        self._current_position = 0.0  # Current position as fraction
        self._entry_price = 0.0
        self._last_features: Optional[MarketFeatures] = None
        self._last_action_idx: int = POSITION_LEVELS.index(0.0)  # Start flat

        # Performance tracking
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._position_history: list[float] = []

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def current_position(self) -> float:
        """Current position as fraction of capital (-1 to 1)."""
        return self._current_position

    def get_position_size(
        self,
        features: MarketFeatures,
        capital: float,
        training: bool = True
    ) -> tuple[float, dict]:
        """Get target position size in dollars.

        Args:
            features: Current market features
            capital: Available capital
            training: Whether to explore

        Returns:
            Tuple of (position_dollars, info_dict)
            Positive = long, negative = short, 0 = flat
        """
        state = features.vector

        # Epsilon-greedy
        if training and np.random.random() < self._epsilon:
            action_idx = np.random.randint(len(POSITION_LEVELS))
        else:
            q_values = self._q_network.predict(state)[0]
            action_idx = int(np.argmax(q_values))

        target_position = POSITION_LEVELS[action_idx]
        position_dollars = target_position * capital

        # Get Q-values for info
        q_values = self._q_network.predict(state)[0]

        info = {
            "target_fraction": target_position,
            "q_values": {f"{int(p*100)}%": float(q) for p, q in zip(POSITION_LEVELS, q_values)},
            "best_q": float(np.max(q_values)),
            "action_idx": action_idx,
            "epsilon": self._epsilon,
        }

        self._last_action_idx = action_idx

        return position_dollars, info

    def observe_result(
        self,
        prev_features: MarketFeatures,
        action_idx: int,
        new_features: MarketFeatures,
        position_held: float,  # Actual position fraction held
        done: bool = False,
    ) -> float:
        """Observe result and learn.

        Args:
            prev_features: State when decision was made
            action_idx: Action taken (index into POSITION_LEVELS)
            new_features: New state
            position_held: Actual position held as fraction
            done: Episode done

        Returns:
            Reward received
        """
        # Calculate reward based on P&L
        price_change = (new_features.price - prev_features.price) / prev_features.price

        # Reward = position * price_change (scaled)
        reward = position_held * price_change * 100

        # Penalty for large position changes (transaction costs)
        if self._last_features is not None:
            position_change = abs(position_held - self._current_position)
            reward -= position_change * 0.1  # 0.1% cost per position change

        # Store experience
        exp = Experience(
            state=prev_features.vector,
            action=action_idx,
            reward=reward,
            next_state=new_features.vector,
            done=done,
        )
        self._replay_buffer.add(exp)

        # Track performance
        if position_held != 0 and self._current_position == 0:
            # Opened position
            self._entry_price = new_features.price
        elif position_held == 0 and self._current_position != 0:
            # Closed position
            if self._entry_price > 0:
                pnl = (new_features.price - self._entry_price) / self._entry_price
                if self._current_position < 0:
                    pnl = -pnl
                self._total_trades += 1
                self._total_pnl += pnl
                if pnl > 0:
                    self._winning_trades += 1

        self._current_position = position_held
        self._position_history.append(position_held)
        self._last_features = new_features

        # Train
        loss = self._train_step()
        if loss is not None:
            self._training_losses.append(loss)

        # Update target network
        self._steps += 1
        if self._steps % self._target_update_freq == 0:
            self._target_network.copy_from(self._q_network)

        # Decay exploration
        self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)

        return reward

    def _train_step(self) -> Optional[float]:
        """Train on batch from replay buffer."""
        if len(self._replay_buffer) < self._batch_size:
            return None

        batch = self._replay_buffer.sample(self._batch_size)

        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        # Current Q-values
        current_q = self._q_network.predict(states)

        # Target Q-values
        next_q = self._target_network.predict(next_states)
        max_next_q = np.max(next_q, axis=1)

        # Compute targets
        targets = current_q.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self._gamma * max_next_q[i]

        return self._q_network.backward(states, targets)

    def reset_episode(self) -> None:
        """Reset for new episode."""
        self._current_position = 0.0
        self._entry_price = 0.0
        self._last_features = None
        self._last_action_idx = POSITION_LEVELS.index(0.0)

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "steps": self._steps,
            "epsilon": self._epsilon,
            "buffer_size": len(self._replay_buffer),
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate": self._winning_trades / self._total_trades if self._total_trades > 0 else 0,
            "total_pnl_pct": self._total_pnl * 100,
            "avg_loss": float(np.mean(self._training_losses[-100:])) if self._training_losses else 0,
            "avg_position": float(np.mean(self._position_history[-100:])) if self._position_history else 0,
        }

    def get_feature_importance(self) -> list[tuple[str, float]]:
        """Estimate feature importance from learned weights."""
        # Use first layer weights as importance proxy
        first_layer = self._q_network.layers[0]
        importance = np.mean(np.abs(first_layer), axis=1)
        importance = importance / (np.sum(importance) + 1e-8)

        names = self._feature_extractor.feature_names
        pairs = list(zip(names, importance))
        return sorted(pairs, key=lambda x: x[1], reverse=True)

    def get_position_preferences(self) -> dict:
        """Show what position sizes the agent prefers in different conditions."""
        if len(self._position_history) < 10:
            return {}

        # Count how often each position level was chosen
        counts = {p: 0 for p in POSITION_LEVELS}
        for pos in self._position_history:
            # Find closest level
            closest = min(POSITION_LEVELS, key=lambda x: abs(x - pos))
            counts[closest] += 1

        total = len(self._position_history)
        return {f"{int(p*100)}%": count/total*100 for p, count in counts.items()}

    def save(self, path: Path) -> None:
        """Save agent state."""
        state = {
            "q_network": self._q_network.get_weights(),
            "target_network": self._target_network.get_weights(),
            "epsilon": self._epsilon,
            "steps": self._steps,
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "total_pnl": self._total_pnl,
            "feature_names": self._feature_extractor.feature_names,
            "position_levels": POSITION_LEVELS,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved agent to {path}")

    def load(self, path: Path) -> None:
        """Load agent state."""
        with open(path) as f:
            state = json.load(f)

        self._q_network.set_weights(state["q_network"])
        self._target_network.set_weights(state["target_network"])
        self._epsilon = state["epsilon"]
        self._steps = state["steps"]
        self._total_trades = state["total_trades"]
        self._winning_trades = state["winning_trades"]
        self._total_pnl = state["total_pnl"]

        logger.info(f"Loaded agent from {path}")


# Legacy compatibility
class Action:
    """Legacy action enum for compatibility."""
    HOLD = 0
    BUY = 1
    SELL = 2


def create_default_agent(seed: int = 42) -> TradingAgent:
    """Create agent with default settings."""
    feature_extractor = FeatureExtractor(
        lookback_periods=[5, 10, 20, 50],
        include_volume=True,
        include_volatility=True,
        include_momentum=True,
        include_mean_reversion=True,
    )

    return TradingAgent(
        feature_extractor=feature_extractor,
        hidden_sizes=[128, 64],
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=32,
        seed=seed,
    )
