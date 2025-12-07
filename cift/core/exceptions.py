"""
CIFT Markets - Custom Exceptions

Centralized exception hierarchy for better error handling.
"""


class CIFTException(Exception):
    """Base exception for all CIFT Markets errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Data Exceptions
class DataException(CIFTException):
    """Base exception for data-related errors."""


class DataValidationError(DataException):
    """Data failed validation checks."""


class DataIngestionError(DataException):
    """Error during data ingestion."""


class DataQualityError(DataException):
    """Data quality issues detected."""


# Model Exceptions
class ModelException(CIFTException):
    """Base exception for model-related errors."""


class ModelLoadError(ModelException):
    """Failed to load model."""


class ModelPredictionError(ModelException):
    """Error during model prediction."""


class ModelTrainingError(ModelException):
    """Error during model training."""


# Execution Exceptions
class ExecutionException(CIFTException):
    """Base exception for execution-related errors."""


class OrderExecutionError(ExecutionException):
    """Failed to execute order."""


class BrokerConnectionError(ExecutionException):
    """Failed to connect to broker."""


class InsufficientFundsError(ExecutionException):
    """Insufficient funds for order."""


# Risk Management Exceptions
class RiskException(CIFTException):
    """Base exception for risk-related errors."""


class DrawdownLimitExceeded(RiskException):
    """Portfolio drawdown exceeded limit."""


class PositionSizeExceeded(RiskException):
    """Position size exceeded limit."""


class LeverageExceeded(RiskException):
    """Portfolio leverage exceeded limit."""


# API Exceptions
class APIException(CIFTException):
    """Base exception for API-related errors."""


class AuthenticationError(APIException):
    """Authentication failed."""


class AuthorizationError(APIException):
    """Not authorized for this action."""


class RateLimitExceeded(APIException):
    """API rate limit exceeded."""


class InvalidRequest(APIException):
    """Invalid API request."""


# Infrastructure Exceptions
class InfrastructureException(CIFTException):
    """Base exception for infrastructure errors."""


class DatabaseConnectionError(InfrastructureException):
    """Failed to connect to database."""


class CacheConnectionError(InfrastructureException):
    """Failed to connect to cache."""


class MessageQueueError(InfrastructureException):
    """Error with message queue."""
