"""Random trading bot.

A standalone control/experiment bot that invests entirely at random:
- Dirichlet random weights across randomly chosen names
- Per-tick trade probability (it trades on a given tick with probability p)
- Random partial churn (each trade event sells a random subset and buys new names)

The randomness lives in :class:`RandomAllocator`, which is pure and seedable so it
can be unit tested and reproduced. Execution against Alpaca lives in
``scripts/random_portfolio.py``.
"""

from stockbot.random_bot.allocator import RandomAllocator, TradeIntent
from stockbot.random_bot.universe_source import (
    fetch_dynamic_universe,
    rank_by_liquidity,
)

__all__ = [
    "RandomAllocator",
    "TradeIntent",
    "fetch_dynamic_universe",
    "rank_by_liquidity",
]
