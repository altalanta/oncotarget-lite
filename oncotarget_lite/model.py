"""Model training utilities for oncotarget-lite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import PROCESSED_DIR, PreparedData
from .utils import ensure_dir, load_json, save_dataframe, save_json, set_seeds

MODELS_DIR = Path("models")
PREDICTIONS_DIR = Path("reports")


@dataclass(slots=True)
class TrainConfig:
    C: float = 1.0
    penalty: str = "l2"
    max_iter: int = 500
    class_weight: str | dict[str, float] | None = "balanced"
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
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=config.C,
                    penalty=config.penalty,
                    max_iter=config.max_iter,
                    class_weight=config.class_weight,
                    solver="lbfgs",
                    random_state=config.seed,
                ),
            ),
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
