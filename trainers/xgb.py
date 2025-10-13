"""XGBoost trainer."""

try:
    from xgboost import XGBClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer


class XGBTrainer(BaseTrainer):
    """Trainer for XGBoost models (falls back to GradientBoosting if XGBoost unavailable)."""
    
    def create_pipeline(self) -> Pipeline:
        """Create XGBoost pipeline."""
        scaler = StandardScaler()
        
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(**self.config.model_params)
        except ImportError:
            # Fall back to sklearn's GradientBoostingClassifier
            # Map XGBoost params to sklearn params
            sklearn_params = {
                "n_estimators": self.config.model_params.get("n_estimators", 100),
                "max_depth": self.config.model_params.get("max_depth", 6),
                "learning_rate": self.config.model_params.get("learning_rate", 0.1),
                "subsample": self.config.model_params.get("subsample", 0.8),
                "random_state": self.config.model_params.get("random_state", 42),
            }
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(**sklearn_params)
        
        return Pipeline([
            ("scaler", scaler),
            ("clf", model),
        ])