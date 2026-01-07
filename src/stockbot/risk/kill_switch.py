"""Kill switch with persistence for emergency trading halt."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from stockbot.monitoring.logger import get_logger

logger = get_logger("risk.kill_switch")


@dataclass
class KillSwitchState:
    """Persistent state of the kill switch."""

    active: bool = False
    triggered_at: Optional[str] = None  # ISO format timestamp
    reason: Optional[str] = None
    triggered_by: Optional[str] = None  # "manual", "daily_loss", "drawdown", etc.
    auto_reset_after: Optional[str] = None  # ISO format timestamp for auto-reset


class KillSwitch:
    """Persistent kill switch for emergency trading halt.

    Features:
    - Persists state to file (survives restarts)
    - Multiple trigger conditions
    - Manual and automatic triggers
    - Optional auto-reset after time period
    """

    def __init__(
        self,
        state_file: Optional[Path] = None,
        auto_reset_hours: Optional[float] = None,
    ) -> None:
        """Initialize the kill switch.

        Args:
            state_file: Path to state file (default: ./data/kill_switch.json)
            auto_reset_hours: Hours after which to auto-reset (None = no auto-reset)
        """
        self._state_file = state_file or Path("./data/kill_switch.json")
        self._auto_reset_hours = auto_reset_hours
        self._state = self._load_state()

        # Check for auto-reset on startup
        if self._state.active and self._should_auto_reset():
            self.reset("auto_reset")

    def _load_state(self) -> KillSwitchState:
        """Load state from file."""
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    data = json.load(f)
                    state = KillSwitchState(**data)
                    if state.active:
                        logger.warning(
                            f"Kill switch is ACTIVE (triggered: {state.triggered_at}, "
                            f"reason: {state.reason})"
                        )
                    return state
            except Exception as e:
                logger.error(f"Failed to load kill switch state: {e}")

        return KillSwitchState()

    def _save_state(self) -> None:
        """Save state to file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(asdict(self._state), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save kill switch state: {e}")

    def _should_auto_reset(self) -> bool:
        """Check if kill switch should auto-reset."""
        if not self._state.active or not self._state.auto_reset_after:
            return False

        try:
            reset_time = datetime.fromisoformat(self._state.auto_reset_after)
            return datetime.now(timezone.utc) >= reset_time
        except Exception:
            return False

    @property
    def is_active(self) -> bool:
        """Check if kill switch is currently active."""
        # Check for auto-reset
        if self._state.active and self._should_auto_reset():
            self.reset("auto_reset")

        return self._state.active

    @property
    def state(self) -> KillSwitchState:
        """Get current state."""
        return self._state

    def trigger(
        self,
        reason: str,
        triggered_by: str = "manual",
    ) -> None:
        """Trigger the kill switch.

        Args:
            reason: Human-readable reason for triggering
            triggered_by: Source of trigger (manual, daily_loss, drawdown, etc.)
        """
        now = datetime.now(timezone.utc)

        self._state.active = True
        self._state.triggered_at = now.isoformat()
        self._state.reason = reason
        self._state.triggered_by = triggered_by

        # Set auto-reset time if configured
        if self._auto_reset_hours:
            from datetime import timedelta

            reset_time = now + timedelta(hours=self._auto_reset_hours)
            self._state.auto_reset_after = reset_time.isoformat()

        self._save_state()

        logger.error(
            f"KILL SWITCH TRIGGERED: {reason}",
            triggered_by=triggered_by,
            auto_reset=self._state.auto_reset_after,
        )

    def reset(self, reset_by: str = "manual") -> None:
        """Reset the kill switch.

        Args:
            reset_by: Who/what is resetting (manual, auto_reset, etc.)
        """
        was_active = self._state.active

        self._state = KillSwitchState()
        self._save_state()

        if was_active:
            logger.warning(f"Kill switch RESET by {reset_by}")

    def check_daily_loss(
        self,
        daily_pnl: float,
        max_daily_loss: float,
    ) -> bool:
        """Check if daily loss exceeds limit.

        Args:
            daily_pnl: Current daily P&L (negative = loss)
            max_daily_loss: Maximum allowed daily loss (positive number)

        Returns:
            True if kill switch was triggered
        """
        if daily_pnl <= -max_daily_loss:
            self.trigger(
                reason=f"Daily loss ${abs(daily_pnl):.2f} exceeds limit ${max_daily_loss:.2f}",
                triggered_by="daily_loss",
            )
            return True
        return False

    def check_drawdown(
        self,
        current_drawdown_pct: float,
        max_drawdown_pct: float,
    ) -> bool:
        """Check if drawdown exceeds limit.

        Args:
            current_drawdown_pct: Current drawdown percentage
            max_drawdown_pct: Maximum allowed drawdown percentage

        Returns:
            True if kill switch was triggered
        """
        if current_drawdown_pct >= max_drawdown_pct:
            self.trigger(
                reason=f"Drawdown {current_drawdown_pct:.1f}% exceeds limit {max_drawdown_pct:.1f}%",
                triggered_by="drawdown",
            )
            return True
        return False

    def check_equity(
        self,
        current_equity: float,
        min_equity: float,
    ) -> bool:
        """Check if equity falls below minimum.

        Args:
            current_equity: Current portfolio equity
            min_equity: Minimum allowed equity

        Returns:
            True if kill switch was triggered
        """
        if current_equity <= min_equity:
            self.trigger(
                reason=f"Equity ${current_equity:.2f} below minimum ${min_equity:.2f}",
                triggered_by="min_equity",
            )
            return True
        return False

    def get_status(self) -> dict:
        """Get kill switch status as dict.

        Returns:
            Status dict
        """
        return {
            "active": self._state.active,
            "triggered_at": self._state.triggered_at,
            "reason": self._state.reason,
            "triggered_by": self._state.triggered_by,
            "auto_reset_after": self._state.auto_reset_after,
        }

    def print_status(self) -> None:
        """Print kill switch status."""
        status = self.get_status()

        print("\n" + "=" * 50)
        print("KILL SWITCH STATUS")
        print("=" * 50)

        if status["active"]:
            print(f"  Status: *** ACTIVE ***")
            print(f"  Triggered At: {status['triggered_at']}")
            print(f"  Reason: {status['reason']}")
            print(f"  Triggered By: {status['triggered_by']}")
            if status["auto_reset_after"]:
                print(f"  Auto Reset After: {status['auto_reset_after']}")
        else:
            print("  Status: INACTIVE (trading allowed)")

        print("=" * 50 + "\n")
