"""Trainer modules for different model types with lazy loading for performance."""

import logging
from typing import Dict, Type

from .base import BaseTrainer

logger = logging.getLogger(__name__)

# Lazy loading registry - only import when needed
_TRAINER_MODULES = {
    "logreg": (".logreg", "LogisticRegressionTrainer"),
    "xgb": (".xgb", "XGBTrainer"),
    "lgb": (".lgb", "LGBTrainer"),
    "mlp": (".mlp", "MLPTrainer"),
    "transformer": (".transformer", "TransformerTrainer"),
    "gnn": (".gnn", "GNNTrainer"),
}

# Cache for loaded trainer classes
_LOADED_TRAINERS: Dict[str, Type[BaseTrainer]] = {}

__all__ = [
    "BaseTrainer",
    "get_trainer",
    "get_available_trainers",
    "preload_trainer",
]


def _lazy_import_trainer(model_type: str) -> Type[BaseTrainer]:
    """Lazy import a trainer class."""
    if model_type in _LOADED_TRAINERS:
        return _LOADED_TRAINERS[model_type]

    if model_type not in _TRAINER_MODULES:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(_TRAINER_MODULES.keys())}")

    module_name, class_name = _TRAINER_MODULES[model_type]

    try:
        module = __import__(module_name, fromlist=[class_name])
        trainer_class = getattr(module, class_name)
        _LOADED_TRAINERS[model_type] = trainer_class
        logger.debug(f"Lazy loaded trainer: {model_type}")
        return trainer_class
    except ImportError as e:
        logger.error(f"Failed to import {model_type} trainer: {e}")
        raise ImportError(f"Required dependencies for {model_type} trainer not available. "
                          f"Install optional dependencies: pip install 'oncotarget-lite[{model_type}]'")


def get_trainer(model_type: str, config) -> BaseTrainer:
    """Get trainer instance for the specified model type with lazy loading."""
    trainer_class = _lazy_import_trainer(model_type)
    return trainer_class(config)


def get_available_trainers() -> Dict[str, str]:
    """Get list of available trainers with their descriptions."""
    available = {}

    # Test each trainer by attempting to import
    for model_type, (module_name, class_name) in _TRAINER_MODULES.items():
        try:
            _lazy_import_trainer(model_type)
            available[model_type] = "Available"
        except ImportError as e:
            available[model_type] = f"Missing dependencies: {str(e)}"

    return available


def preload_trainer(model_type: str) -> bool:
    """Preload a specific trainer to avoid import delays during training."""
    try:
        _lazy_import_trainer(model_type)
        return True
    except ImportError:
        return False


def preload_common_trainers() -> Dict[str, bool]:
    """Preload commonly used trainers (logreg, xgb, lgb)."""
    common_trainers = ["logreg", "xgb", "lgb"]
    results = {}

    for trainer_type in common_trainers:
        results[trainer_type] = preload_trainer(trainer_type)

    return results


# Legacy compatibility - expose trainer classes if they can be imported
try:
    from .logreg import LogisticRegressionTrainer
    _LOADED_TRAINERS["logreg"] = LogisticRegressionTrainer
except ImportError:
    pass

try:
    from .xgb import XGBTrainer
    _LOADED_TRAINERS["xgb"] = XGBTrainer
except ImportError:
    pass

try:
    from .lgb import LGBTrainer
    _LOADED_TRAINERS["lgb"] = LGBTrainer
except ImportError:
    pass

try:
    from .mlp import MLPTrainer
    _LOADED_TRAINERS["mlp"] = MLPTrainer
except ImportError:
    pass

try:
    from .transformer import TransformerTrainer
    _LOADED_TRAINERS["transformer"] = TransformerTrainer
except ImportError:
    pass

try:
    from .gnn import GNNTrainer
    _LOADED_TRAINERS["gnn"] = GNNTrainer
except ImportError:
    pass