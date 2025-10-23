"""Model training utilities for oncotarget-lite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import PROCESSED_DIR
from .utils import ensure_dir, load_json, save_dataframe, save_json, set_seeds
from .trainers import get_trainer, preload_common_trainers
from .performance import performance_monitor, get_performance_monitor

MODELS_DIR = Path("models")
PREDICTIONS_DIR = Path("reports")

# Constants
DEFAULT_PREDICTION_THRESHOLD = 0.5


@dataclass(slots=True)
class TrainConfig:
    # Logistic regression specific parameters
    C: float = 1.0
    penalty: str = "l2"
    max_iter: int = 500
    class_weight: str | dict[str, float] | None = "balanced"

    # Common parameters across model types
    model_type: str = "logreg"  # "logreg", "xgb", "lgb", "mlp", "transformer", "gnn"
    model_params: dict[str, Any] | None = None
    seed: int = 42


@dataclass(slots=True)
class TrainResult:
    pipeline: Pipeline
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    dataset_hash: str


class TrainingError(RuntimeError):
    """Raised when training cannot be completed."""


_DEF_MODEL_NAME = "logreg_pipeline.pkl"
_DEF_PREDICTIONS = "predictions.parquet"
_DEF_FEATURES = "feature_list.json"


def _load_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    features_path = processed_dir / "features.parquet"
    labels_path = processed_dir / "labels.parquet"
    splits_path = processed_dir / "splits.json"
    if not features_path.exists() or not labels_path.exists() or not splits_path.exists():
        raise TrainingError("Processed features/labels/splits not found; run prepare first")
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)["label"].astype(int)
    splits = load_json(splits_path)
    return features, labels, splits


def _build_pipeline(config: TrainConfig) -> Pipeline:
    """Build pipeline for different model types using lazy loading."""

    # Use model_params if provided, otherwise use legacy parameters for backward compatibility
    if config.model_params:
        params = config.model_params.copy()
    else:
        # Legacy mode - use individual parameters
        params = {
            "C": config.C,
            "penalty": config.penalty,
            "max_iter": config.max_iter,
            "class_weight": config.class_weight,
            "random_state": config.seed,
        }

    # Use lazy loading for trainer-based models (transformer, gnn, and modern models)
    if config.model_type in ["transformer", "gnn"]:
        from .trainers.base import TrainerConfig
        trainer_config = TrainerConfig(
            name=config.model_type,
            model_type=config.model_type,
            model_params=params,
            feature_type="all_features",
            seed=config.seed,
        )

        with performance_monitor(f"build_pipeline_{config.model_type}"):
            trainer = get_trainer(config.model_type, trainer_config)
            return trainer.create_pipeline()

    # Use sklearn-based models with lazy loading
    elif config.model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            C=params.get("C", config.C),
            penalty=params.get("penalty", config.penalty),
            max_iter=params.get("max_iter", config.max_iter),
            class_weight=params.get("class_weight", config.class_weight),
            solver="lbfgs",
            random_state=params.get("random_state", config.seed),
        )

    elif config.model_type == "xgb":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(**params)
        except ImportError:
            # Fallback to sklearn implementation
            from sklearn.ensemble import GradientBoostingClassifier
            sklearn_params = {
                "n_estimators": params.get("n_estimators", 100),
                "max_depth": params.get("max_depth", 6),
                "learning_rate": params.get("learning_rate", 0.1),
                "subsample": params.get("subsample", 0.8),
                "random_state": params.get("random_state", config.seed),
            }
            clf = GradientBoostingClassifier(**sklearn_params)

    elif config.model_type == "lgb":
        try:
            import lightgbm as lgb
            # Convert sklearn-style parameters to LightGBM format
            lgb_params = params.copy()
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

            clf = LGBWrapper(lgb_params)

        except ImportError:
            raise ImportError("LightGBM not available. Install with: pip install lightgbm")

    elif config.model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(**params)

    else:
        raise ValueError(f"Unknown model type: {config.model_type}. Available: logreg, xgb, lgb, mlp, transformer, gnn")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def _collect_predictions(
    model: Pipeline, features: pd.DataFrame, genes: list[str], split: str
) -> pd.DataFrame:
    subset = features.loc[genes]
    probs = model.predict_proba(subset)[:, 1]
    frame = pd.DataFrame({
        "gene": genes,
        "split": split,
        "y_prob": probs,
    })
    return frame


def _compute_metrics(labels: pd.Series, pred_frame: pd.DataFrame) -> dict[str, float]:
    y_true = labels.loc[pred_frame["gene"]].astype(int).to_numpy()
    y_prob = pred_frame["y_prob"].to_numpy()
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
    }


def train_model(
    *,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = PREDICTIONS_DIR,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train the logistic regression model and persist core artefacts."""

    cfg = config or TrainConfig()
    set_seeds(cfg.seed)

    features, labels, splits = _load_processed(processed_dir)
    train_genes = splits["train_genes"]
    test_genes = splits["test_genes"]

    pipeline = _build_pipeline(cfg)
    pipeline.fit(features.loc[train_genes], labels.loc[train_genes])

    train_preds = _collect_predictions(pipeline, features, train_genes, "train")
    test_preds = _collect_predictions(pipeline, features, test_genes, "test")

    train_metrics = _compute_metrics(labels, train_preds)
    test_metrics = _compute_metrics(labels, test_preds)

    ensure_dir(models_dir)
    ensure_dir(reports_dir)
    joblib.dump(pipeline, models_dir / _DEF_MODEL_NAME)
    save_json(models_dir / _DEF_FEATURES, {"feature_order": list(features.columns)})

    preds = pd.concat([train_preds, test_preds], ignore_index=True)
    preds["y_true"] = preds["gene"].map(labels.to_dict()).astype(int)
    save_dataframe(reports_dir / _DEF_PREDICTIONS, preds, index=False)

    metrics_payload = {
        "train": train_metrics,
        "test": test_metrics,
        "train_size": len(train_genes),
        "test_size": len(test_genes),
    }
    save_json(reports_dir / "metrics_basic.json", metrics_payload)

    return TrainResult(
        pipeline=pipeline,
        train_predictions=train_preds,
        test_predictions=test_preds,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        dataset_hash=splits.get("dataset_hash", ""),
    )
