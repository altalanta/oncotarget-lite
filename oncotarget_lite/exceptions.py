"""Custom exceptions for the oncotarget-lite pipeline.

Provides specific exception types for better error handling and debugging
across the machine learning pipeline.
"""


class OncotargetError(Exception):
    """Base exception for oncotarget-lite pipeline errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class DataValidationError(OncotargetError):
    """Raised when data validation fails."""
    pass


class ModelTrainingError(OncotargetError):
    """Raised when model training encounters an error."""
    pass


class ConfigurationError(OncotargetError):
    """Raised when configuration is invalid or missing."""
    pass


class FeatureEngineeringError(OncotargetError):
    """Raised when feature engineering fails."""
    pass


class EvaluationError(OncotargetError):
    """Raised when model evaluation fails."""
    pass


class MLflowError(OncotargetError):
    """Raised when MLflow operations fail."""
    pass