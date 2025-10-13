"""LightGBM trainer."""

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseTrainer

# Constants
DEFAULT_PREDICTION_THRESHOLD = 0.5


class LGBTrainer(BaseTrainer):
    """Trainer for LightGBM models."""

    def create_pipeline(self) -> Pipeline:
        """Create LightGBM pipeline."""
        if lgb is None:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")

        scaler = StandardScaler()

        # Convert sklearn-style parameters to LightGBM format
        lgb_params = self.config.model_params.copy()

        # Handle parameter name differences
        if 'n_estimators' in lgb_params:
            lgb_params['num_iterations'] = lgb_params.pop('n_estimators')
        if 'learning_rate' in lgb_params:
            lgb_params['learning_rate'] = lgb_params['learning_rate']
        if 'max_depth' in lgb_params:
            lgb_params['max_depth'] = lgb_params['max_depth']
        if 'subsample' in lgb_params:
            lgb_params['bagging_fraction'] = lgb_params.pop('subsample')
        if 'colsample_bytree' in lgb_params:
            lgb_params['feature_fraction'] = lgb_params.pop('colsample_bytree')

        # Set default parameters for binary classification
        lgb_params.setdefault('objective', 'binary')
        lgb_params.setdefault('metric', 'binary_logloss')
        lgb_params.setdefault('verbosity', -1)  # Suppress output
        lgb_params.setdefault('random_state', self.config.seed)

        # Create a wrapper that behaves like sklearn estimator
        class LGBWrapper:
            def __init__(self, params):
                self.params = params
                self.model = None

            def fit(self, X, y):
                # Convert to LightGBM format
                train_data = lgb.Dataset(X, label=y)
                self.model = lgb.train(
                    self.params,
                    train_data,
                    valid_sets=[train_data],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                return self

            def predict_proba(self, X):
                if self.model is None:
                    raise ValueError("Model not trained")
                # Get probabilities for positive class
                probs = self.model.predict(X)
                return np.column_stack([1 - probs, probs])

            def predict(self, X):
                if self.model is None:
                    raise ValueError("Model not trained")
                return (self.model.predict(X) > DEFAULT_PREDICTION_THRESHOLD).astype(int)

        model = LGBWrapper(lgb_params)

        return Pipeline([
            ("scaler", scaler),
            ("clf", model),
        ])
