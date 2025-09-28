"""Evaluation utilities for calibration-aware reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .utils import ensure_dir, load_dataframe, save_dataframe, save_json, set_seeds

REPORTS_DIR = Path("reports")


@dataclass(slots=True)
class MetricSummary:
    auroc: float
    ap: float
    brier: float
    ece: float
    accuracy: float
    f1: float
    train_auroc: float
    test_auroc: float
    overfit_gap: float


@dataclass(slots=True)
class BootstrapCI:
    mean: float
    lower: float
    upper: float


@dataclass(slots=True)
class EvaluationArtifacts:
    metrics: MetricSummary
    auroc_ci: BootstrapCI
    ap_ci: BootstrapCI
    calibration_bins: pd.DataFrame


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)
    rows = []
    ece = 0.0
    for bin_idx in range(1, bins + 1):
        mask = bin_indices == bin_idx
        if not np.any(mask):
            rows.append({
                "bin": bin_idx,
                "count": 0,
                "confidence": float((bin_edges[bin_idx - 1] + bin_edges[bin_idx]) / 2),
                "accuracy": float("nan"),
            })
            continue
        bin_true = y_true[mask]
        bin_prob = y_prob[mask]
        bin_acc = float(np.mean(bin_true))
        bin_conf = float(np.mean(bin_prob))
        rows.append({
            "bin": bin_idx,
            "count": int(mask.sum()),
            "confidence": bin_conf,
            "accuracy": bin_acc,
        })
        ece += abs(bin_acc - bin_conf) * mask.sum()
    ece /= len(y_true)
    calibration = pd.DataFrame(rows)
    return float(ece), calibration


def _bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> BootstrapCI:
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sample_true = y_true[indices]
        sample_prob = y_prob[indices]
        try:
            score = metric_fn(sample_true, sample_prob)
        except ValueError:
            continue
        scores.append(score)
    if not scores:
        value = metric_fn(y_true, y_prob)
        return BootstrapCI(mean=value, lower=value, upper=value)
    scores_arr = np.array(scores)
    value = float(scores_arr.mean())
    alpha = (1 - ci) / 2
    lower = float(np.quantile(scores_arr, alpha))
    upper = float(np.quantile(scores_arr, 1 - alpha))
    return BootstrapCI(mean=value, lower=lower, upper=upper)


def evaluate_predictions(
    *,
    reports_dir: Path = REPORTS_DIR,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    bins: int = 10,
    seed: int = 42,
) -> EvaluationArtifacts:
    """Compute calibration-aware metrics for saved predictions."""

    predictions_path = reports_dir / "predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError("predictions.parquet not found; run train before eval")

    preds = load_dataframe(predictions_path)
    train_preds = preds[preds["split"] == "train"]
    test_preds = preds[preds["split"] == "test"]

    for frame in (train_preds, test_preds):
        if frame.empty:
            raise ValueError("Missing train or test predictions for evaluation")

    train_true = train_preds["y_true"].to_numpy()
    train_prob = train_preds["y_prob"].to_numpy()
    test_true = test_preds["y_true"].to_numpy()
    test_prob = test_preds["y_prob"].to_numpy()

    auroc_test = float(roc_auc_score(test_true, test_prob))
    ap_test = float(average_precision_score(test_true, test_prob))
    brier = float(brier_score_loss(test_true, test_prob))
    accuracy = float(accuracy_score(test_true, test_prob >= 0.5))
    f1 = float(f1_score(test_true, test_prob >= 0.5))
    ece, calibration = _expected_calibration_error(test_true, test_prob, bins=bins)

    auroc_ci = _bootstrap_ci(
        test_true,
        test_prob,
        roc_auc_score,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
    )
    ap_ci = _bootstrap_ci(
        test_true,
        test_prob,
        average_precision_score,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed + 1,
    )

    metrics = MetricSummary(
        auroc=auroc_test,
        ap=ap_test,
        brier=brier,
        ece=ece,
        accuracy=accuracy,
        f1=f1,
        train_auroc=float(roc_auc_score(train_true, train_prob)),
        test_auroc=auroc_test,
        overfit_gap=float(abs(roc_auc_score(train_true, train_prob) - auroc_test)),
    )

    ensure_dir(reports_dir)
    save_json(
        reports_dir / "metrics.json",
        {
            "auroc": metrics.auroc,
            "auroc_ci": [auroc_ci.lower, auroc_ci.upper],
            "ap": metrics.ap,
            "ap_ci": [ap_ci.lower, ap_ci.upper],
            "brier": metrics.brier,
            "ece": metrics.ece,
            "accuracy": metrics.accuracy,
            "f1": metrics.f1,
            "train_auroc": metrics.train_auroc,
            "test_auroc": metrics.test_auroc,
            "overfit_gap": metrics.overfit_gap,
        },
    )
    save_json(
        reports_dir / "bootstrap.json",
        {
            "auroc": {
                "mean": auroc_ci.mean,
                "lower": auroc_ci.lower,
                "upper": auroc_ci.upper,
                "n_bootstrap": n_bootstrap,
                "confidence": ci,
            },
            "ap": {
                "mean": ap_ci.mean,
                "lower": ap_ci.lower,
                "upper": ap_ci.upper,
                "n_bootstrap": n_bootstrap,
                "confidence": ci,
            },
        },
    )
    save_dataframe(reports_dir / "calibration.csv", calibration, index=False)
    save_json(
        reports_dir / "calibration.json",
        calibration.fillna(0).to_dict(orient="list"),
    )

    _plot_curves(test_true, test_prob, calibration, reports_dir)

    return EvaluationArtifacts(
        metrics=metrics,
        auroc_ci=auroc_ci,
        ap_ci=ap_ci,
        calibration_bins=calibration,
    )


def _plot_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    calibration: pd.DataFrame,
    reports_dir: Path,
) -> None:
    ensure_dir(reports_dir)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUROC={roc_auc_score(y_true, y_prob):.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(reports_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f"PR (AP={average_precision_score(y_true, y_prob):.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(reports_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    valid = calibration.dropna()
    ax.plot(valid["confidence"], valid["accuracy"], marker="o", label="model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Reliability Curve")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(reports_dir / "calibration_plot.png", dpi=150)
    plt.close(fig)
