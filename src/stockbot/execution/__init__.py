"""Order execution components."""

from stockbot.execution.broker_sim import SimulatedBroker
from stockbot.execution.translator import SignalTranslator

__all__ = [
    "SimulatedBroker",
    "SignalTranslator",
]

# Optional imports that require Alpaca
try:
    from stockbot.execution.broker_alpaca import AlpacaBroker
    from stockbot.execution.order_manager import OrderManager

    __all__.extend(["AlpacaBroker", "OrderManager"])
except ImportError:
    pass
