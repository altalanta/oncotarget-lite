"""Trainer modules for different model types."""

from .base import BaseTrainer
from .logreg import LogisticRegressionTrainer
from .mlp import MLPTrainer
from .xgb import XGBTrainer

__all__ = ["BaseTrainer", "LogisticRegressionTrainer", "MLPTrainer", "XGBTrainer"]