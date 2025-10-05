"""Logistic regression trainer."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer


class LogisticRegressionTrainer(BaseTrainer):
    """Trainer for logistic regression models."""
    
    def create_pipeline(self) -> Pipeline:
        """Create logistic regression pipeline."""
        scaler = StandardScaler()
        model = LogisticRegression(**self.config.model_params)
        
        return Pipeline([
            ("scaler", scaler),
            ("clf", model),
        ])