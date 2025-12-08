"""Enhanced exception hierarchy for oncotarget-lite with context information.

This module provides domain-specific exceptions with:
- Rich context for debugging
- Structured error information for logging
- HTTP status code mappings for API responses
- Error codes for programmatic handling

Usage:
    from oncotarget_lite.exceptions import (
        DataPreparationError,
        ModelLoadingError,
        PredictionError,
    )

    raise DataPreparationError(
        "Missing required columns",
        context={"file": "expression.csv", "missing": ["gene", "tpm"]},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ErrorContext:
    """Structured context for exceptions."""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    operation: str | None = None
    component: str | None = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "component": self.component,
            **self.details,
        }


class OncoTargetError(Exception):
    """
    Base exception for all oncotarget-lite errors.

    Provides:
    - Error code for programmatic handling
    - HTTP status code mapping
    - Rich context information
    - Structured logging support
    """

    error_code: str = "ONCOTARGET_ERROR"
    http_status: int = 500
    default_message: str = "An error occurred"

    def __init__(
        self,
        message: str | None = None,
        *,
        context: Dict[str, Any] | None = None,
        cause: Exception | None = None,
        operation: str | None = None,
        component: str | None = None,
    ):
        """
        Initialize exception with context.

        Args:
            message: Human-readable error message
            context: Additional context as key-value pairs
            cause: Original exception that caused this error
            operation: Name of the operation that failed
            component: Component where error occurred
        """
        self.message = message or self.default_message
        self.cause = cause
        self.context = ErrorContext(
            operation=operation,
            component=component,
            details=context or {},
        )

        # Build full message
        full_message = self.message
        if cause:
            full_message = f"{full_message}: {cause}"

        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses and logging."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "http_status": self.http_status,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, context={self.context.details})"


class DataPreparationError(OncoTargetError):
    """Exception raised when data preparation fails.

    Common causes:
    - Missing required columns in input files
    - Invalid data format
    - Empty datasets after filtering
    - File not found
    """

    error_code = "DATA_PREPARATION_ERROR"
    http_status = 400
    default_message = "Data preparation failed"


class DataValidationError(OncoTargetError):
    """Exception raised when data validation fails.

    Common causes:
    - Schema mismatch
    - Out-of-range values
    - Missing required fields
    - Data type mismatches
    """

    error_code = "DATA_VALIDATION_ERROR"
    http_status = 422
    default_message = "Data validation failed"


class ModelLoadingError(OncoTargetError):
    """Exception raised when model loading fails.

    Common causes:
    - Model file not found
    - Incompatible model version
    - Corrupted model file
    - Missing dependencies
    """

    error_code = "MODEL_LOADING_ERROR"
    http_status = 500
    default_message = "Failed to load model"


class ModelNotFoundError(ModelLoadingError):
    """Exception raised when model file is not found."""

    error_code = "MODEL_NOT_FOUND"
    http_status = 404
    default_message = "Model not found"


class PredictionError(OncoTargetError):
    """Exception raised when prediction fails.

    Common causes:
    - Feature mismatch
    - Model not loaded
    - Invalid input data
    - Inference timeout
    """

    error_code = "PREDICTION_ERROR"
    http_status = 500
    default_message = "Prediction failed"


class FeatureValidationError(PredictionError):
    """Exception raised when feature validation fails."""

    error_code = "FEATURE_VALIDATION_ERROR"
    http_status = 422
    default_message = "Feature validation failed"


class ConfigurationError(OncoTargetError):
    """Exception raised when configuration is invalid.

    Common causes:
    - Missing required configuration
    - Invalid configuration values
    - Incompatible settings
    """

    error_code = "CONFIGURATION_ERROR"
    http_status = 500
    default_message = "Invalid configuration"


class PerformanceError(OncoTargetError):
    """Exception raised when performance monitoring or optimization fails."""

    error_code = "PERFORMANCE_ERROR"
    http_status = 500
    default_message = "Performance operation failed"


class ResourceExhaustedError(PerformanceError):
    """Exception raised when system resources are exhausted."""

    error_code = "RESOURCE_EXHAUSTED"
    http_status = 503
    default_message = "System resources exhausted"


class APIError(OncoTargetError):
    """Exception raised for API-related errors."""

    error_code = "API_ERROR"
    http_status = 500
    default_message = "API error occurred"


class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded."""

    error_code = "RATE_LIMIT_EXCEEDED"
    http_status = 429
    default_message = "Rate limit exceeded"


class AuthenticationError(APIError):
    """Exception raised for authentication failures."""

    error_code = "AUTHENTICATION_ERROR"
    http_status = 401
    default_message = "Authentication failed"


class AuthorizationError(APIError):
    """Exception raised for authorization failures."""

    error_code = "AUTHORIZATION_ERROR"
    http_status = 403
    default_message = "Not authorized"


class ResilienceError(OncoTargetError):
    """Base exception for resilience framework."""

    error_code = "RESILIENCE_ERROR"
    http_status = 503
    default_message = "Service temporarily unavailable"


class CircuitBreakerOpen(ResilienceError):
    """Exception raised when a circuit breaker is open."""

    error_code = "CIRCUIT_BREAKER_OPEN"
    http_status = 503
    default_message = "Circuit breaker is open"


class RetryExhaustedError(ResilienceError):
    """Exception raised when all retry attempts are exhausted."""

    error_code = "RETRY_EXHAUSTED"
    http_status = 503
    default_message = "All retry attempts exhausted"


class TimeoutError(ResilienceError):
    """Exception raised when an operation times out."""

    error_code = "TIMEOUT"
    http_status = 504
    default_message = "Operation timed out"


class ExternalServiceError(OncoTargetError):
    """Exception raised when an external service call fails."""

    error_code = "EXTERNAL_SERVICE_ERROR"
    http_status = 502
    default_message = "External service error"


class TrainingError(OncoTargetError):
    """Exception raised when model training fails."""

    error_code = "TRAINING_ERROR"
    http_status = 500
    default_message = "Model training failed"


class EvaluationError(OncoTargetError):
    """Exception raised when model evaluation fails."""

    error_code = "EVALUATION_ERROR"
    http_status = 500
    default_message = "Model evaluation failed"


class ExplanationError(OncoTargetError):
    """Exception raised when generating explanations fails."""

    error_code = "EXPLANATION_ERROR"
    http_status = 500
    default_message = "Failed to generate explanation"


def create_error_response(error: OncoTargetError) -> Dict[str, Any]:
    """
    Create a standardized API error response.

    Args:
        error: The exception to convert

    Returns:
        Dictionary suitable for JSON response
    """
    return {
        "error": {
            "code": error.error_code,
            "message": error.message,
            "details": error.context.details,
        }
    }


def wrap_exception(
    error: Exception,
    wrapper_class: type[OncoTargetError] = OncoTargetError,
    message: str | None = None,
    **context: Any,
) -> OncoTargetError:
    """
    Wrap a generic exception in an OncoTargetError.

    Args:
        error: Original exception
        wrapper_class: Exception class to wrap with
        message: Optional custom message
        **context: Additional context to include

    Returns:
        Wrapped exception
    """
    return wrapper_class(
        message or str(error),
        cause=error,
        context=context,
    )


__all__ = [
    # Base
    "OncoTargetError",
    "ErrorContext",
    # Data
    "DataPreparationError",
    "DataValidationError",
    # Model
    "ModelLoadingError",
    "ModelNotFoundError",
    "PredictionError",
    "FeatureValidationError",
    "TrainingError",
    "EvaluationError",
    "ExplanationError",
    # Configuration
    "ConfigurationError",
    # Performance
    "PerformanceError",
    "ResourceExhaustedError",
    # API
    "APIError",
    "RateLimitError",
    "AuthenticationError",
    "AuthorizationError",
    # Resilience
    "ResilienceError",
    "CircuitBreakerOpen",
    "RetryExhaustedError",
    "TimeoutError",
    "ExternalServiceError",
    # Utilities
    "create_error_response",
    "wrap_exception",
]
