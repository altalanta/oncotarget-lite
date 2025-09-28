"""Metric utilities for deterministic evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

BINARY_CLASSES = 2
MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _ensure_binary(y_true: np.ndarray) -> None:
    unique = np.unique(y_true)
    if unique.size != BINARY_CLASSES:
        raise ValueError("binary metrics require exactly two classes")


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under ROC curve without sklearn dependencies."""

    y_true = y_true.astype(float)
    y_score = y_score.astype(float)
    _ensure_binary(y_true)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    positives = y_true.sum()
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    tpr = np.cumsum(y_true) / positives
    fpr = np.cumsum(1.0 - y_true) / negatives
    return float(np.trapz(tpr, fpr))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under precision-recall curve."""

    y_true = y_true.astype(float)
    y_score = y_score.astype(float)
    _ensure_binary(y_true)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    cum_tp = np.cumsum(y_true)
    cum_fp = np.cumsum(1.0 - y_true)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / positives
    # Insert starting point (0,1)
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapz(precision, recall))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y_true = y_true.astype(float)
    y_prob = y_prob.astype(float)
    bins = int(bins)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.digitize(y_prob, edges, right=True) - 1
    rows = []
    for idx in range(bins):
        mask = bin_indices == idx
        if not mask.any():
            continue
        bin_true = y_true[mask]
        bin_prob = y_prob[mask]
        rows.append(
            {
                "bin": idx,
                "lower": float(edges[idx]),
                "upper": float(edges[idx + 1]),
                "count": int(mask.sum()),
                "event_rate": float(bin_true.mean()),
                "confidence": float(bin_prob.mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(curve: pd.DataFrame, total: int) -> float:
    if curve.empty:
        return float("nan")
    weights = curve["count"].to_numpy() / total
    diffs = np.abs(curve["event_rate"].to_numpy() - curve["confidence"].to_numpy())
    return float(np.sum(weights * diffs))


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    lower: float
    upper: float
    std: float


@dataclass(frozen=True)
class MetricSummary:
    auroc: float
    auprc: float
    brier: float
    ece: float


@dataclass(frozen=True)
class BootstrapSummary:
    auroc: BootstrapResult
    auprc: BootstrapResult
    brier: BootstrapResult


def bootstrap_confidence_interval(  # noqa: PLR0913
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: MetricFn,
    samples: int,
    ci_level: float,
    seed: int,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        try:
            value = metric(y_true[indices], y_prob[indices])
        except ValueError:
            continue
        if np.isnan(value):
            continue
        stats.append(float(value))
    if not stats:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), float("nan"))
    values = np.array(stats)
    alpha = (1 - ci_level) / 2
    lower = np.quantile(values, alpha)
    upper = np.quantile(values, 1 - alpha)
    return BootstrapResult(
        mean=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        std=float(values.std(ddof=1)),
    )


def summarise_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: int,
) -> tuple[MetricSummary, pd.DataFrame]:
    roc = auroc(y_true, y_prob)
    pr = auprc(y_true, y_prob)
    brier = brier_score(y_true, y_prob)
    curve = calibration_curve(y_true, y_prob, bins)
    ece = expected_calibration_error(curve, total=len(y_true))
    return MetricSummary(roc, pr, brier, ece), curve
