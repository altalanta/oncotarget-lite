"""Common exceptions for the oncotarget-lite package."""


class DataPreparationError(Exception):
    """Exception raised when data preparation fails."""
    pass


class ModelLoadingError(Exception):
    """Exception raised when model loading fails."""
    pass


class PredictionError(Exception):
    """Exception raised when prediction fails."""
    pass


class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""
    pass


class PerformanceError(Exception):
    """Exception raised when performance monitoring or optimization fails."""
    pass


class APIError(Exception):
    """Exception raised for API-related errors."""
    pass


class ResilienceError(Exception):
    """Base exception for resilience framework."""
    pass


class CircuitBreakerOpen(ResilienceError):
    """Exception raised when a circuit breaker is open."""
    pass

