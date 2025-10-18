"""Production deployment framework for oncotarget-lite."""

from .api import create_app
from .model_loader import ModelLoader
from .prediction_service import PredictionService

__all__ = ["create_app", "ModelLoader", "PredictionService"]


