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

