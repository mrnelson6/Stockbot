"""Risk management components."""

from stockbot.risk.manager import BasicRiskManager
from stockbot.risk.rules import (
    MaxDailyLossRule,
    MaxDrawdownRule,
    MaxOpenPositionsRule,
    MaxPositionSizeRule,
    MaxPositionValueRule,
    RiskRule,
)
from stockbot.risk.exposure import (
    DrawdownTracker,
    ExposureSnapshot,
    ExposureTracker,
)
from stockbot.risk.kill_switch import KillSwitch, KillSwitchState
from stockbot.risk.position_sizer import (
    FixedDollarSizer,
    FixedPercentSizer,
    KellyCriterionSizer,
    PositionSizer,
    RiskParitySizer,
    VolatilityAdjustedSizer,
)

__all__ = [
    # Manager
    "BasicRiskManager",
    # Rules
    "MaxDailyLossRule",
    "MaxDrawdownRule",
    "MaxOpenPositionsRule",
    "MaxPositionSizeRule",
    "MaxPositionValueRule",
    "RiskRule",
    # Exposure
    "DrawdownTracker",
    "ExposureSnapshot",
    "ExposureTracker",
    # Kill switch
    "KillSwitch",
    "KillSwitchState",
    # Position sizing
    "FixedDollarSizer",
    "FixedPercentSizer",
    "KellyCriterionSizer",
    "PositionSizer",
    "RiskParitySizer",
    "VolatilityAdjustedSizer",
]
