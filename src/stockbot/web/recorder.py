"""Records the bot's state to the dashboard database each tick.

The recorder only *persists* values handed to it; the trader computes equity,
cash, and positions (it already talks to the broker or its simulated book). The
one thing the recorder fetches itself is the SPY benchmark price, since it owns a
data provider reference.
"""

from typing import Any, Optional

from stockbot.core.types import Symbol
from stockbot.monitoring.logger import get_logger
from stockbot.web import db

logger = get_logger("web.recorder")


class SnapshotRecorder:
    """Persists equity/positions/trades to a SQLite file for the web dashboard."""

    def __init__(
        self,
        db_path: str,
        data_provider: Any,
        *,
        label: str = "Random Bot",
        spy_symbol: str = "SPY",
    ) -> None:
        """Initialize and ensure the schema exists.

        Args:
            db_path: Path to the SQLite file (shared with the web process).
            data_provider: Object with ``get_latest(Symbol) -> Bar`` for SPY pricing.
            label: Display name shown on the dashboard.
            spy_symbol: Benchmark ticker (default SPY).
        """
        self._path = db_path
        self._data_provider = data_provider
        self._spy_symbol = spy_symbol
        db.init_db(db_path, label=label)
        logger.info(f"Recorder writing to {db_path}")

    def _latest_spy_price(self) -> Optional[float]:
        try:
            bar = self._data_provider.get_latest(Symbol(self._spy_symbol))
            return float(bar.close) if bar else None
        except Exception as e:
            logger.warning(f"Failed to fetch SPY price: {e}")
            return None

    def snapshot(self, equity: float, cash: float, positions: list[dict]) -> None:
        """Record one equity datapoint (with SPY) and replace current positions.

        Args:
            equity: Total portfolio value.
            cash: Available cash.
            positions: List of dicts with keys symbol, qty, avg_price,
                market_value, and optionally unrealized_pnl.
        """
        spy = self._latest_spy_price()
        ts = db.now_ms()
        try:
            db.insert_snapshot(self._path, equity=equity, cash=cash, spy_price=spy, ts=ts)
            db.replace_positions(self._path, positions, ts=ts)
        except Exception as e:
            logger.error(f"Failed to record snapshot: {e}")

    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_id: Optional[str] = None,
    ) -> None:
        """Append one executed trade to the log."""
        try:
            db.insert_trade(
                self._path,
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                order_id=order_id,
            )
        except Exception as e:
            logger.error(f"Failed to record trade {side} {symbol}: {e}")
