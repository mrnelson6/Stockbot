"""Multi-asset portfolio agent using deep reinforcement learning.

This agent manages a portfolio of multiple stocks, learning:
1. Which stocks to hold and in what proportion
2. Overall portfolio risk management
3. Cross-asset correlations and diversification
4. Optimal position sizing across the portfolio
"""

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from stockbot.learning.features import FeatureExtractor, MarketFeatures
from stockbot.monitoring.logger import get_logger

logger = get_logger("portfolio_agent")


@dataclass
class PortfolioState:
    """State of the entire portfolio."""
    features: dict[str, np.ndarray]  # symbol -> feature vector
    prices: dict[str, float]         # symbol -> current price
    timestamp: int


@dataclass
class PortfolioExperience:
    """Experience tuple for portfolio learning."""
    state: np.ndarray        # Flattened portfolio state
    actions: np.ndarray      # Position for each asset
    reward: float
    next_state: np.ndarray
    done: bool


class PortfolioReplayBuffer:
    """Experience replay for portfolio agent."""

    def __init__(self, capacity: int = 50000) -> None:
        self._buffer: deque = deque(maxlen=capacity)

    def add(self, exp: PortfolioExperience) -> None:
        self._buffer.append(exp)

    def sample(self, batch_size: int) -> list[PortfolioExperience]:
        indices = np.random.choice(
            len(self._buffer),
            size=min(batch_size, len(self._buffer)),
            replace=False
        )
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class PortfolioNetwork:
    """Neural network for portfolio allocation.

    Input: Concatenated features for all assets + portfolio state
    Output: Allocation weights for each asset (-1 to 1)
    """

    def __init__(
        self,
        n_assets: int,
        features_per_asset: int,
        hidden_sizes: list[int] = [256, 128, 64],
        learning_rate: float = 0.0005,
    ) -> None:
        self.n_assets = n_assets
        self.features_per_asset = features_per_asset
        self.lr = learning_rate

        # Input: features for each asset + market-wide features
        # Output: allocation weight for each asset
        input_size = n_assets * features_per_asset + 10  # +10 for portfolio state
        output_size = n_assets

        # Build network
        self.layers = []
        self.biases = []

        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i+1])
            self.layers.append(w)
            self.biases.append(b)

        self._activations = []
        self._z_values = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Output is tanh to bound allocations to [-1, 1]."""
        x = np.atleast_2d(x)
        self._activations = [x]
        self._z_values = []

        for i, (w, b) in enumerate(zip(self.layers, self.biases)):
            z = self._activations[-1] @ w + b
            self._z_values.append(z)

            if i < len(self.layers) - 1:
                # Hidden layers: ReLU
                a = np.maximum(0, z)
            else:
                # Output layer: tanh for bounded output
                a = np.tanh(z)

            self._activations.append(a)

        return self._activations[-1]

    def backward(self, x: np.ndarray, target: np.ndarray) -> float:
        """Backward pass."""
        batch_size = x.shape[0] if x.ndim > 1 else 1
        x = np.atleast_2d(x)
        target = np.atleast_2d(target)

        output = self.forward(x)
        loss = np.mean((output - target) ** 2)

        # Gradient of tanh
        tanh_grad = 1 - output ** 2
        delta = 2 * (output - target) / batch_size * tanh_grad

        for i in range(len(self.layers) - 1, -1, -1):
            dw = self._activations[i].T @ delta
            db = np.sum(delta, axis=0)

            dw = np.clip(dw, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)

            self.layers[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

            if i > 0:
                delta = (delta @ self.layers[i].T) * (self._z_values[i-1] > 0)

        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(np.atleast_2d(x))

    def get_weights(self) -> dict:
        return {
            "layers": [l.tolist() for l in self.layers],
            "biases": [b.tolist() for b in self.biases],
        }

    def set_weights(self, weights: dict) -> None:
        self.layers = [np.array(l) for l in weights["layers"]]
        self.biases = [np.array(b) for b in weights["biases"]]


class PortfolioAgent:
    """Deep RL agent for multi-asset portfolio management.

    The agent learns to allocate capital across multiple assets,
    considering correlations, momentum, and risk.
    """

    def __init__(
        self,
        symbols: list[str],
        feature_extractor: FeatureExtractor,
        hidden_sizes: list[int] = [256, 128, 64],
        learning_rate: float = 0.0005,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        batch_size: int = 64,
        max_position_per_asset: float = 0.20,  # Max 20% in any single asset
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)

        self._symbols = symbols
        self._symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        self._feature_extractor = feature_extractor
        self._gamma = gamma
        self._epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._batch_size = batch_size
        self._max_position = max_position_per_asset

        # Network
        n_assets = len(symbols)
        features_per_asset = feature_extractor.feature_count

        self._policy_network = PortfolioNetwork(
            n_assets, features_per_asset, hidden_sizes, learning_rate
        )

        # Experience replay
        self._replay_buffer = PortfolioReplayBuffer(capacity=100000)

        # State tracking
        self._steps = 0
        self._current_positions: dict[str, float] = {s: 0.0 for s in symbols}
        self._last_state: Optional[np.ndarray] = None
        self._last_actions: Optional[np.ndarray] = None
        self._training_losses: list[float] = []

        # Performance
        self._total_pnl = 0.0
        self._trade_count = 0

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def _build_state_vector(
        self,
        features_dict: dict[str, MarketFeatures],
        portfolio_value: float,
    ) -> np.ndarray:
        """Build flattened state vector from all asset features."""
        parts = []

        # Features for each asset (in order)
        for symbol in self._symbols:
            if symbol in features_dict:
                parts.append(features_dict[symbol].vector)
            else:
                # Missing asset - use zeros
                parts.append(np.zeros(self._feature_extractor.feature_count))

        # Portfolio state features
        positions = np.array([self._current_positions[s] for s in self._symbols])
        portfolio_features = np.array([
            np.mean(positions),           # Average position
            np.std(positions),            # Position dispersion
            np.sum(np.abs(positions)),    # Gross exposure
            np.sum(positions),            # Net exposure
            len([p for p in positions if p > 0]) / len(positions),  # % long
            len([p for p in positions if p < 0]) / len(positions),  # % short
            np.max(positions),            # Max long
            np.min(positions),            # Max short
            portfolio_value / 100000,     # Normalized portfolio value
            self._epsilon,                # Current exploration rate
        ])
        parts.append(portfolio_features)

        return np.concatenate(parts)

    def get_allocations(
        self,
        features_dict: dict[str, MarketFeatures],
        portfolio_value: float,
        training: bool = True,
    ) -> tuple[dict[str, float], dict]:
        """Get target allocation for each asset.

        Args:
            features_dict: Features for each symbol
            portfolio_value: Total portfolio value
            training: Whether to explore

        Returns:
            Tuple of (allocations_dict, info_dict)
            Allocations are fractions of portfolio (-1 to 1 per asset)
        """
        state = self._build_state_vector(features_dict, portfolio_value)

        if training and np.random.random() < self._epsilon:
            # Explore: random allocations
            raw_actions = np.random.uniform(-1, 1, len(self._symbols))
        else:
            # Exploit: use network
            raw_actions = self._policy_network.predict(state)[0]

        # Apply position limits
        actions = np.clip(raw_actions, -self._max_position, self._max_position)

        # Normalize to not exceed 100% gross exposure (optional)
        gross = np.sum(np.abs(actions))
        if gross > 1.0:
            actions = actions / gross

        # Build allocations dict
        allocations = {s: float(actions[i]) for i, s in enumerate(self._symbols)}

        info = {
            "raw_actions": raw_actions.tolist(),
            "gross_exposure": float(np.sum(np.abs(actions))),
            "net_exposure": float(np.sum(actions)),
            "n_long": int(np.sum(actions > 0.01)),
            "n_short": int(np.sum(actions < -0.01)),
            "epsilon": self._epsilon,
        }

        self._last_state = state
        self._last_actions = actions

        return allocations, info

    def observe_result(
        self,
        features_dict: dict[str, MarketFeatures],
        prev_features_dict: dict[str, MarketFeatures],
        allocations_held: dict[str, float],
        portfolio_value: float,
        done: bool = False,
    ) -> float:
        """Observe portfolio performance and learn.

        Args:
            features_dict: Current features
            prev_features_dict: Previous features
            allocations_held: Actual allocations held
            portfolio_value: Current portfolio value
            done: Episode done

        Returns:
            Total reward
        """
        # Calculate reward: sum of (allocation * return) for each asset
        total_reward = 0.0

        for symbol in self._symbols:
            if symbol in features_dict and symbol in prev_features_dict:
                price_now = features_dict[symbol].price
                price_prev = prev_features_dict[symbol].price
                ret = (price_now - price_prev) / price_prev

                allocation = allocations_held.get(symbol, 0.0)
                total_reward += allocation * ret * 100  # Scale up

        # Transaction cost penalty
        for symbol in self._symbols:
            old_pos = self._current_positions.get(symbol, 0.0)
            new_pos = allocations_held.get(symbol, 0.0)
            total_reward -= abs(new_pos - old_pos) * 0.05  # 5bp per turnover

        # Store experience
        if self._last_state is not None and self._last_actions is not None:
            new_state = self._build_state_vector(features_dict, portfolio_value)
            exp = PortfolioExperience(
                state=self._last_state,
                actions=self._last_actions,
                reward=total_reward,
                next_state=new_state,
                done=done,
            )
            self._replay_buffer.add(exp)

        # Update positions
        self._current_positions = dict(allocations_held)

        # Train
        loss = self._train_step()
        if loss is not None:
            self._training_losses.append(loss)

        # Decay exploration
        self._steps += 1
        self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)

        self._total_pnl += total_reward / 100

        return total_reward

    def _train_step(self) -> Optional[float]:
        """Train on batch."""
        if len(self._replay_buffer) < self._batch_size:
            return None

        batch = self._replay_buffer.sample(self._batch_size)

        states = np.array([e.state for e in batch])
        actions = np.array([e.actions for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])

        # Simple policy gradient update
        # Target = current action + learning signal based on reward
        current_actions = self._policy_network.predict(states)

        # Adjust actions based on reward (if positive, reinforce; if negative, oppose)
        reward_signal = rewards.reshape(-1, 1) / 10  # Normalize
        targets = current_actions + reward_signal * (actions - current_actions) * 0.1

        return self._policy_network.backward(states, targets)

    def reset(self) -> None:
        """Reset for new episode."""
        self._current_positions = {s: 0.0 for s in self._symbols}
        self._last_state = None
        self._last_actions = None

    def get_stats(self) -> dict:
        """Get training statistics."""
        positions = list(self._current_positions.values())
        return {
            "steps": self._steps,
            "epsilon": self._epsilon,
            "buffer_size": len(self._replay_buffer),
            "total_pnl_pct": self._total_pnl,
            "avg_loss": float(np.mean(self._training_losses[-100:])) if self._training_losses else 0,
            "gross_exposure": sum(abs(p) for p in positions),
            "net_exposure": sum(positions),
            "n_positions": sum(1 for p in positions if abs(p) > 0.01),
        }

    def get_top_positions(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N positions by absolute size."""
        sorted_pos = sorted(
            self._current_positions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_pos[:n]

    def save(self, path: Path) -> None:
        """Save agent."""
        state = {
            "symbols": self._symbols,
            "network": self._policy_network.get_weights(),
            "epsilon": self._epsilon,
            "steps": self._steps,
            "total_pnl": self._total_pnl,
            "current_positions": self._current_positions,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved portfolio agent to {path}")

    def load(self, path: Path) -> None:
        """Load agent."""
        with open(path) as f:
            state = json.load(f)

        # Verify symbols match
        if state["symbols"] != self._symbols:
            logger.warning("Symbol mismatch - some weights may not apply")

        self._policy_network.set_weights(state["network"])
        self._epsilon = state["epsilon"]
        self._steps = state["steps"]
        self._total_pnl = state["total_pnl"]
        self._current_positions = state.get("current_positions", {s: 0.0 for s in self._symbols})

        logger.info(f"Loaded portfolio agent from {path}")
