"""Multi-layer perceptron trainer."""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer


class MLPTrainer(BaseTrainer):
    """Trainer for multi-layer perceptron models."""
    
    def create_pipeline(self) -> Pipeline:
        """Create MLP pipeline."""
        scaler = StandardScaler()
        model = MLPClassifier(**self.config.model_params)
        
        return Pipeline([
            ("scaler", scaler),
            ("clf", model),
        ])