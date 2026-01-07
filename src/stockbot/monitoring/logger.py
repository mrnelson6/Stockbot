"""Structured logging for the trading system."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from stockbot.core.models import Fill, Order
from stockbot.core.types import Signal, Symbol


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)  # type: ignore[attr-defined]

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TradingLogger:
    """Logger with trading-specific convenience methods."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log_with_extra(
        self,
        level: int,
        message: str,
        extra_data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log with extra structured data."""
        extra = {"extra_data": extra_data} if extra_data else {}
        self._logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log_with_extra(logging.DEBUG, message, kwargs if kwargs else None)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log_with_extra(logging.INFO, message, kwargs if kwargs else None)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log_with_extra(logging.WARNING, message, kwargs if kwargs else None)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log_with_extra(logging.ERROR, message, kwargs if kwargs else None)

    def signal(
        self,
        symbol: Symbol,
        signal: Signal,
        strategy: str,
        **kwargs: Any,
    ) -> None:
        """Log a trading signal."""
        self._log_with_extra(
            logging.INFO,
            f"Signal: {signal.name} for {symbol}",
            {
                "event_type": "signal",
                "symbol": symbol,
                "signal": signal.name,
                "strategy": strategy,
                **kwargs,
            },
        )

    def order(self, order: Order, action: str = "submitted") -> None:
        """Log an order event."""
        self._log_with_extra(
            logging.INFO,
            f"Order {action}: {order.side.name} {order.quantity} {order.symbol}",
            {
                "event_type": "order",
                "action": action,
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side.name,
                "quantity": str(order.quantity),
                "order_type": order.order_type.name,
                "limit_price": str(order.limit_price) if order.limit_price else None,
            },
        )

    def fill(self, fill: Fill) -> None:
        """Log a fill event."""
        self._log_with_extra(
            logging.INFO,
            f"Fill: {fill.side.name} {fill.quantity} {fill.symbol} @ {fill.price}",
            {
                "event_type": "fill",
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.name,
                "quantity": str(fill.quantity),
                "price": str(fill.price),
                "commission": str(fill.commission),
            },
        )

    def risk_rejection(
        self,
        symbol: Symbol,
        signal: Signal,
        reason: str,
    ) -> None:
        """Log a risk rejection."""
        self._log_with_extra(
            logging.WARNING,
            f"Signal rejected: {signal.name} for {symbol} - {reason}",
            {
                "event_type": "risk_rejection",
                "symbol": symbol,
                "signal": signal.name,
                "reason": reason,
            },
        )


# Global logger registry
_loggers: dict[str, TradingLogger] = {}


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    """Configure the logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, use JSON formatting
    """
    root_logger = logging.getLogger("stockbot")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)


def get_logger(name: str = "stockbot") -> TradingLogger:
    """Get a trading logger instance.

    Args:
        name: Logger name (will be prefixed with 'stockbot.')

    Returns:
        TradingLogger instance
    """
    full_name = f"stockbot.{name}" if not name.startswith("stockbot") else name

    if full_name not in _loggers:
        logger = logging.getLogger(full_name)
        _loggers[full_name] = TradingLogger(logger)

    return _loggers[full_name]
