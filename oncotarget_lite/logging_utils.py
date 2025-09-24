"""Advanced logging infrastructure with structured logging and performance tracking.

Provides centralized logging configuration with support for structured logging,
performance monitoring, and integration with MLflow experiment tracking.
"""

import functools
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import LoggingConfig


class PerformanceLogger:
    """Context manager and decorator for performance monitoring."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[float] = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            if exc_type is None:
                self.logger.info(f"Completed {self.operation} in {duration:.3f}s")
            else:
                self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
        return False


def time_it(operation: str = None):
    """Decorator for automatic performance logging.
    
    Args:
        operation: Description of the operation (defaults to function name)
        
    Example:
        @time_it("model training")
        def train_model():
            # training code
            pass
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            with PerformanceLogger(logger, op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with additional context."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add structured fields
        if hasattr(record, 'experiment_id'):
            record.msg = f"[{record.experiment_id}] {record.msg}"
        if hasattr(record, 'model_type'):
            record.msg = f"[{record.model_type}] {record.msg}"
        
        return super().format(record)


def setup_logging(config: LoggingConfig, logger_name: str = None) -> logging.Logger:
    """Configure logging system with specified configuration.
    
    Args:
        config: Logging configuration object
        logger_name: Name for the logger (defaults to 'oncotarget')
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from oncotarget_lite.config import load_config
        >>> config = load_config()
        >>> logger = setup_logging(config.logging)
        >>> logger.info("Training started")
    """
    logger_name = logger_name or "oncotarget"
    logger = logging.getLogger(logger_name)
    
    # Clear existing handlers
    logger.handlers.clear()
    logger.setLevel(config.level)
    
    # Create formatter
    formatter = StructuredFormatter(config.format)
    
    # Console handler
    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.file_path:
        config.file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger


def log_model_metrics(logger: logging.Logger, metrics: Dict[str, Any], 
                     model_type: str = None, experiment_id: str = None) -> None:
    """Log model metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics to log
        model_type: Type of model being evaluated
        experiment_id: MLflow experiment/run ID
    """
    extra_fields = {}
    if model_type:
        extra_fields['model_type'] = model_type
    if experiment_id:
        extra_fields['experiment_id'] = experiment_id
    
    logger.info("Model evaluation metrics:", extra=extra_fields)
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric_name}: {value:.4f}", extra=extra_fields)
        else:
            logger.info(f"  {metric_name}: {value}", extra=extra_fields)


def log_config_summary(logger: logging.Logger, config: Any) -> None:
    """Log configuration summary for reproducibility.
    
    Args:
        logger: Logger instance  
        config: Configuration object to log
    """
    logger.info("Configuration summary:")
    if hasattr(config, 'dict'):
        # Pydantic model
        config_dict = config.dict()
    else:
        config_dict = vars(config)
    
    def log_nested_dict(d: Dict[str, Any], prefix: str = ""):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"  {prefix}{key}:")
                log_nested_dict(value, prefix + "  ")
            else:
                logger.info(f"  {prefix}{key}: {value}")
    
    log_nested_dict(config_dict)


class MLflowLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically includes MLflow context."""
    
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Add MLflow context if available
        try:
            import mlflow
            active_run = mlflow.active_run()
            if active_run:
                kwargs['extra']['experiment_id'] = active_run.info.run_id[:8]
        except ImportError:
            pass
        
        return msg, kwargs


def get_logger(name: str = None, config: LoggingConfig = None) -> logging.Logger:
    """Get configured logger instance.
    
    Args:
        name: Logger name (defaults to calling module)
        config: Optional logging configuration (uses default if None)
        
    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the calling module's name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'oncotarget')
    
    if config is None:
        # Use default configuration
        from .config import LoggingConfig
        config = LoggingConfig()
    
    return setup_logging(config, name)


# Convenience function for common logging patterns
def log_data_summary(logger: logging.Logger, data, name: str = "dataset") -> None:
    """Log summary statistics for a dataset.
    
    Args:
        logger: Logger instance
        data: DataFrame or array to summarize
        name: Name of the dataset
    """
    logger.info(f"{name} summary:")
    logger.info(f"  Shape: {getattr(data, 'shape', 'N/A')}")
    
    if hasattr(data, 'dtypes'):
        logger.info(f"  Dtypes: {dict(data.dtypes)}")
    
    if hasattr(data, 'isnull'):
        missing = data.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"  Missing values: {dict(missing[missing > 0])}")
        else:
            logger.info("  No missing values detected")
    
    if hasattr(data, 'describe'):
        logger.debug(f"  Statistics:\\n{data.describe()}")