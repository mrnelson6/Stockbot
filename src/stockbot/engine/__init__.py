"""Trading engines and execution context."""

from stockbot.engine.clock import RealClock, SimulatedClock
from stockbot.engine.context import ExecutionContext

__all__ = [
    "ExecutionContext",
    "RealClock",
    "SimulatedClock",
]

# Optional imports
try:
    from stockbot.engine.paper import PaperTradingConfig, PaperTradingEngine

    __all__.extend(["PaperTradingConfig", "PaperTradingEngine"])
except ImportError:
    pass
