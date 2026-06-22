"""Core random allocation logic for the random trading bot.

This module is intentionally pure and free of any I/O (no Alpaca, no network).
All randomness flows through a single seeded ``numpy`` generator so behavior is
reproducible and unit-testable. The runner script (``scripts/random_portfolio.py``)
is responsible for turning a :class:`TradeIntent` into actual broker orders.
"""

from dataclasses import dataclass, field

import numpy as np

from stockbot.core.types import Symbol


@dataclass
class TradeIntent:
    """A single random trade event's intentions.

    Attributes:
        sells: Symbol -> number of shares to sell (full exits of churned positions).
        buys: Symbol -> target dollar value to allocate to this name.
    """

    sells: dict[Symbol, int] = field(default_factory=dict)
    buys: dict[Symbol, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True if there is nothing to do."""
        return not self.sells and not self.buys


class RandomAllocator:
    """Generates entirely random trade decisions.

    Strategy:
    - ``should_trade()`` gates whether a tick produces any trade at all
      (per-tick probability ``trade_prob``).
    - ``plan_trade()`` performs random partial churn: each currently held name is
      sold with probability ``churn_sell_prob``, then a random number of names is
      bought using Dirichlet-distributed weights over a random fraction of the
      available cash.
    """

    def __init__(
        self,
        *,
        trade_prob: float = 0.1,
        churn_sell_prob: float = 0.3,
        min_buys: int = 1,
        max_buys: int = 5,
        deploy_fraction_range: tuple[float, float] = (0.9, 1.0),
        max_position_value: float = 20_000.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the allocator.

        Args:
            trade_prob: Probability of trading on any given tick (0..1).
            churn_sell_prob: Probability each held position is fully sold per trade event.
            min_buys: Minimum number of names to buy on a trade event.
            max_buys: Maximum number of names to buy on a trade event.
            deploy_fraction_range: (low, high) uniform range for the fraction of
                available cash to deploy on a trade event.
            max_position_value: Hard cap on the dollar value targeted per name.
            seed: Optional RNG seed for reproducibility.
        """
        if not 0.0 <= trade_prob <= 1.0:
            raise ValueError(f"trade_prob must be in [0, 1], got {trade_prob}")
        if not 0.0 <= churn_sell_prob <= 1.0:
            raise ValueError(f"churn_sell_prob must be in [0, 1], got {churn_sell_prob}")
        if min_buys < 0 or max_buys < min_buys:
            raise ValueError(f"require 0 <= min_buys <= max_buys, got {min_buys}, {max_buys}")
        low, high = deploy_fraction_range
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"deploy_fraction_range must be 0<=low<=high<=1, got {deploy_fraction_range}")

        self._trade_prob = trade_prob
        self._churn_sell_prob = churn_sell_prob
        self._min_buys = min_buys
        self._max_buys = max_buys
        self._deploy_low, self._deploy_high = low, high
        self._max_position_value = max_position_value
        self._rng = np.random.default_rng(seed)

    def should_trade(self) -> bool:
        """Per-tick gate: return True with probability ``trade_prob``."""
        return bool(self._rng.random() < self._trade_prob)

    def plan_trade(
        self,
        current_positions: dict[Symbol, int],
        prices: dict[Symbol, float],
        cash: float,
        universe: list[Symbol],
    ) -> TradeIntent:
        """Build a random trade intent.

        Args:
            current_positions: Symbol -> shares currently held (long, >0).
            prices: Symbol -> latest price. Names without a price are skipped for buys.
            cash: Cash currently available to deploy (before this event's sells).
            universe: Candidate symbols to buy from.

        Returns:
            A :class:`TradeIntent`. Buy values are dollar targets, not shares; the
            executor converts to whole shares using live prices and buying power.
        """
        intent = TradeIntent()

        # 1. Random partial churn: each held position is fully sold with churn_sell_prob.
        for symbol, shares in current_positions.items():
            if shares > 0 and self._rng.random() < self._churn_sell_prob:
                intent.sells[symbol] = shares

        # 2. Random buys with Dirichlet weights over a random fraction of cash.
        #    Estimate cash freed by the churn sells so the deploy fraction reflects
        #    the capital that will realistically be available.
        freed_cash = sum(
            shares * prices[symbol]
            for symbol, shares in intent.sells.items()
            if symbol in prices
        )
        available_cash = max(0.0, cash) + freed_cash
        if available_cash <= 0:
            return intent

        # Only buy names we have a price for.
        candidates = [s for s in universe if prices.get(s, 0.0) > 0.0]
        if not candidates:
            return intent

        deploy_fraction = float(self._rng.uniform(self._deploy_low, self._deploy_high))
        deployable = available_cash * deploy_fraction
        if deployable <= 0:
            return intent

        # Pick a random number of names, but ensure there are enough to actually
        # absorb `deployable` within the per-position cap -- otherwise the cap
        # would strand cash. This keeps the portfolio close to fully invested.
        k = int(self._rng.integers(self._min_buys, self._max_buys + 1))
        if self._max_position_value > 0:
            k = max(k, int(np.ceil(deployable / self._max_position_value)))
        k = min(k, len(candidates))
        if k <= 0:
            return intent

        chosen_idx = self._rng.choice(len(candidates), size=k, replace=False)
        chosen = [candidates[i] for i in np.atleast_1d(chosen_idx)]
        weights = self._rng.dirichlet(np.ones(k))

        # Distribute `deployable` across the chosen names by their Dirichlet
        # weights, water-filling overflow from capped names onto the others so
        # the full amount lands (up to total capacity = k * cap).
        for symbol, target_value in self._water_fill(chosen, weights, deployable).items():
            if target_value > 0:
                intent.buys[symbol] = target_value

        return intent

    def _water_fill(
        self, names: list[Symbol], weights: "np.ndarray", deployable: float
    ) -> dict[Symbol, float]:
        """Allocate ``deployable`` across ``names`` by weight, respecting the cap.

        Overflow from names that hit ``max_position_value`` is redistributed by
        weight onto the remaining uncapped names, repeating until the cash is
        placed or every name is capped. Any leftover (when total capacity is
        below ``deployable``) is simply not allocated.
        """
        cap = self._max_position_value
        weight_map = {s: float(w) for s, w in zip(names, weights)}
        targets: dict[Symbol, float] = {s: 0.0 for s in names}
        active = set(names)
        remaining = deployable

        # At most one pass per name can become newly capped, so this terminates.
        for _ in range(len(names) + 1):
            if remaining <= 1e-6 or not active:
                break
            weight_sum = sum(weight_map[s] for s in active)
            newly_capped: list[Symbol] = []
            overflow = 0.0
            for s in active:
                share = remaining * weight_map[s] / weight_sum if weight_sum > 0 else remaining / len(active)
                targets[s] += share
                if cap > 0 and targets[s] >= cap:
                    overflow += targets[s] - cap
                    targets[s] = cap
                    newly_capped.append(s)
            remaining = overflow
            active.difference_update(newly_capped)

        return targets
