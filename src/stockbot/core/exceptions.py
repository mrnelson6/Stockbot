"""Custom exception hierarchy for the trading system."""


class StockbotError(Exception):
    """Base exception for all Stockbot errors."""

    pass


# Data errors
class DataError(StockbotError):
    """Base exception for data-related errors."""

    pass


class DataNotFoundError(DataError):
    """Requested data is not available."""

    pass


class DataValidationError(DataError):
    """Data failed validation checks."""

    pass


# Strategy errors
class StrategyError(StockbotError):
    """Base exception for strategy-related errors."""

    pass


class InvalidSignalError(StrategyError):
    """Strategy produced an invalid signal."""

    pass


# Risk errors
class RiskError(StockbotError):
    """Base exception for risk-related errors."""

    pass


class RiskLimitExceededError(RiskError):
    """A risk limit has been exceeded."""

    pass


class EmergencyStopError(RiskError):
    """Emergency stop has been triggered."""

    pass


# Execution errors
class ExecutionError(StockbotError):
    """Base exception for execution-related errors."""

    pass


class OrderRejectedError(ExecutionError):
    """Order was rejected by broker or risk manager."""

    def __init__(self, order_id: str, reason: str) -> None:
        self.order_id = order_id
        self.reason = reason
        super().__init__(f"Order {order_id} rejected: {reason}")


class BrokerError(ExecutionError):
    """Error communicating with broker."""

    pass


# Configuration errors
class ConfigError(StockbotError):
    """Configuration error."""

    pass


class MissingCredentialsError(ConfigError):
    """Required API credentials are missing."""

    pass
