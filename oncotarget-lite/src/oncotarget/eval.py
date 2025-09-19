"""Evaluation utilities for miniature target discovery models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricSummary:
    name: str
    value: float
    lower: float
    upper: float


def summarize_bootstrap(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstraps: int = 256,
    seed: int = 42,
) -> dict[str, MetricSummary]:
    """Bootstrap AUROC and AUPRC with 95% confidence intervals."""

    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError("y_true and y_scores must be the same length")

    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_y = y_true[indices]
        sample_scores = y_scores[indices]
        if np.unique(sample_y).size < 2:
            continue
        aurocs.append(roc_auc_score(sample_y, sample_scores))
        auprcs.append(average_precision_score(sample_y, sample_scores))

    def _summary(values: list[float], name: str) -> MetricSummary:
        if not values:
            return MetricSummary(name=name, value=float("nan"), lower=float("nan"), upper=float("nan"))
        array = np.array(values)
        return MetricSummary(
            name=name,
            value=float(array.mean()),
            lower=float(np.percentile(array, 2.5)),
            upper=float(np.percentile(array, 97.5)),
        )

    return {
        "auroc": _summary(aurocs, "AUROC"),
        "auprc": _summary(auprcs, "AUPRC"),
    }


def compute_reliability_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return calibration curve coordinates."""

    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins)
    return pd.DataFrame(
        {
            "fraction_of_positives": prob_true,
            "mean_predicted": prob_pred,
        }
    )


def classification_summary(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
    """Compute scalar metrics without resampling."""

    metrics = {
        "auroc": float(roc_auc_score(y_true, y_scores)),
        "auprc": float(average_precision_score(y_true, y_scores)),
        "brier": float(brier_score_loss(y_true, y_scores)),
    }
    return metrics


def compute_shap_values(
    model,
    features: pd.DataFrame,
    max_samples: int = 20,
) -> Optional[pd.DataFrame]:
    """Compute SHAP values using a kernel explainer (optional)."""

    try:
        import shap
    except ImportError:  # pragma: no cover
        return None

    import torch

    model.eval()
    subset = features.sample(n=min(max_samples, len(features)), random_state=0)
    background = subset.iloc[: min(10, len(subset))]

    def _predict(data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            inputs = torch.from_numpy(data.astype(np.float32))
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(-1)
        return probs

    explainer = shap.KernelExplainer(_predict, background.values)
    shap_values = explainer(subset.values, nsamples=100)
    values = pd.DataFrame(shap_values.values, columns=features.columns, index=subset.index)
    return values
