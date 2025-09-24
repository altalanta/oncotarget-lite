"""Oncotarget-lite: Production-ready ML pipeline for immunotherapy target discovery.

A comprehensive machine learning pipeline featuring:
- Robust configuration management with Pydantic validation
- Comprehensive logging and performance monitoring  
- Advanced error handling and data validation
- MLflow experiment tracking with lineage
- Bootstrap confidence intervals and model calibration
- Multiple model architectures (MLP, Random Forest, FT-Transformer)
- Interactive Streamlit application with SHAP explanations
"""

__version__ = "0.2.1"

from .config import Config, load_config
from .data import load_data, split_data
from .eval import comprehensive_evaluation, bootstrap_metrics
from .exceptions import (
    OncotargetError, 
    DataValidationError, 
    ModelTrainingError,
    ConfigurationError
)
from .logging_utils import get_logger, setup_logging, time_it
from .model import MLPClassifier, train_mlp, train_random_forest
from .performance import monitor_resources, profile_memory, optimize_memory
from .utils import set_random_seed, compute_dataset_hash

__all__ = [
    # Core functionality
    'load_data', 'split_data', 'train_mlp', 'train_random_forest',
    'comprehensive_evaluation', 'bootstrap_metrics',
    
    # Configuration and utilities
    'Config', 'load_config', 'MLPClassifier', 
    'set_random_seed', 'compute_dataset_hash',
    
    # Logging and monitoring
    'get_logger', 'setup_logging', 'time_it',
    'monitor_resources', 'profile_memory', 'optimize_memory',
    
    # Exceptions
    'OncotargetError', 'DataValidationError', 
    'ModelTrainingError', 'ConfigurationError'
]