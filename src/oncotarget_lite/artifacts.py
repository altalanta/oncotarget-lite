"""Artifact serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import BootstrapResult, BootstrapSummary, MetricSummary


def _bootstrap_to_dict(result: BootstrapResult) -> dict[str, float]:
    return {
        "mean": result.mean,
        "lower": result.lower,
        "upper": result.upper,
        "std": result.std,
    }


def write_metrics(
    path: Path,
    train_metrics: MetricSummary,
    test_metrics: MetricSummary,
    bootstrap: BootstrapSummary,
    calibration: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": {
            "auroc": train_metrics.auroc,
            "auprc": train_metrics.auprc,
            "brier": train_metrics.brier,
            "ece": train_metrics.ece,
        },
        "test": {
            "auroc": test_metrics.auroc,
            "auprc": test_metrics.auprc,
            "brier": test_metrics.brier,
            "ece": test_metrics.ece,
        },
        "bootstrap": {
            "auroc": _bootstrap_to_dict(bootstrap.auroc),
            "auprc": _bootstrap_to_dict(bootstrap.auprc),
            "brier": _bootstrap_to_dict(bootstrap.brier),
        },
        "calibration_curve": calibration.reset_index(drop=True).to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_predictions(path: Path, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pd.concat([train_df, test_df]).reset_index().rename(columns={"index": "gene"})
    payload.to_parquet(path, index=False)


def write_feature_importances(path: Path, importances: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = importances.rename_axis("gene").reset_index(name="importance")
    df.to_parquet(path, index=False)

