"""
This module serves as the public API for the experimentation package.
It re-exports the main classes from the submodules for easy access.
"""

from .experimentation.dashboard import ExperimentDashboard
from .experimentation.manager import ExperimentManager
from .experimentation.optimizer import EnhancedHyperparameterOptimizer
from .experimentation.schemas import Experiment, ExperimentConfig, ExperimentTrial

__all__ = [
    "Experiment",
    "ExperimentConfig",
    "ExperimentDashboard",
    "ExperimentManager",
    "EnhancedHyperparameterOptimizer",
    "ExperimentTrial",
]

