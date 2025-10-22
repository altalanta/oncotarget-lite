"""Trainer modules for different model types."""

from .base import BaseTrainer
from .lgb import LGBTrainer
from .logreg import LogisticRegressionTrainer
from .mlp import MLPTrainer
from .xgb import XGBTrainer
from .transformer import TransformerTrainer
from .gnn import GNNTrainer

__all__ = [
    "BaseTrainer",
    "LGBTrainer",
    "LogisticRegressionTrainer",
    "MLPTrainer",
    "XGBTrainer",
    "TransformerTrainer",
    "GNNTrainer",
]

# Trainer registry for easy model selection
TRAINER_REGISTRY = {
    "logreg": LogisticRegressionTrainer,
    "xgb": XGBTrainer,
    "lgb": LGBTrainer,
    "mlp": MLPTrainer,
    "transformer": TransformerTrainer,
    "gnn": GNNTrainer,
}


def get_trainer(model_type: str, config) -> BaseTrainer:
    """Get trainer instance for the specified model type."""
    if model_type not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(TRAINER_REGISTRY.keys())}")

    trainer_class = TRAINER_REGISTRY[model_type]
    return trainer_class(config)