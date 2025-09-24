"""Enhanced evaluation with bootstrap CIs, calibration, and overfitting checks."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.utils import resample

from .utils import save_json, set_random_seed


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for metrics."""
    
    set_random_seed(random_state)
    
    auroc_scores = []
    ap_scores = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = resample(range(len(y_true)), random_state=random_state + i)
        y_boot = y_true[indices]
        pred_boot = y_pred_proba[indices]
        
        # Skip if bootstrap sample has only one class
        if len(np.unique(y_boot)) < 2:
            continue
            
        auroc_scores.append(roc_auc_score(y_boot, pred_boot))
        ap_scores.append(average_precision_score(y_boot, pred_boot))
    
    def compute_ci(scores: List[float]) -> Dict[str, float]:
        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "ci_lower": np.percentile(scores, 2.5),
            "ci_upper": np.percentile(scores, 97.5)
        }
    
    return {
        "auroc": compute_ci(auroc_scores),
        "average_precision": compute_ci(ap_scores)
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute calibration metrics (Brier score, ECE)."""
    
    # Brier score
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    calibration_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Select predictions in this bin
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            calibration_data.append({
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "accuracy": accuracy_in_bin,
                "confidence": avg_confidence_in_bin,
                "count": in_bin.sum()
            })
    
    return {
        "brier_score": brier_score,
        "ece": ece,
        "calibration_curve": calibration_data
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
    n_bins: int = 10
) -> None:
    """Plot reliability diagram for calibration."""
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
            label='Model calibration', markersize=8)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot (Reliability Diagram)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path
) -> None:
    """Plot ROC and PR curves."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)
    
    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auroc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PR curve
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    
    ax2.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
    ax2.axhline(y=y_true.mean(), color='k', linestyle='--', alpha=0.5, 
                label=f'Baseline ({y_true.mean():.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def comprehensive_evaluation(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    model_type: str = "sklearn"
) -> Dict[str, Any]:
    """Run comprehensive evaluation with all metrics and plots."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    if model_type == "mlp":
        import torch
        model.eval()
        with torch.no_grad():
            train_proba = torch.sigmoid(model(torch.FloatTensor(X_train.values))).numpy().flatten()
            test_proba = torch.sigmoid(model(torch.FloatTensor(X_test.values))).numpy().flatten()
    else:  # sklearn model
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    train_auroc = roc_auc_score(y_train, train_proba)
    test_auroc = roc_auc_score(y_test, test_proba)
    test_ap = average_precision_score(y_test, test_proba)
    
    # Bootstrap confidence intervals
    bootstrap_results = bootstrap_metrics(y_test.values, test_proba)
    
    # Calibration metrics
    calibration_results = compute_calibration_metrics(y_test.values, test_proba)
    
    # Overfitting check
    auroc_gap = train_auroc - test_auroc
    
    # Generate plots
    plot_roc_pr_curves(y_test.values, test_proba, output_dir / "roc_pr_curves.png")
    plot_calibration_curve(y_test.values, test_proba, output_dir / "calibration_curve.png")
    
    # Compile results
    results = {
        "metrics": {
            "train_auroc": train_auroc,
            "test_auroc": test_auroc,
            "test_average_precision": test_ap,
            "auroc_gap": auroc_gap,
            "brier_score": calibration_results["brier_score"],
            "ece": calibration_results["ece"]
        },
        "bootstrap": bootstrap_results,
        "calibration": calibration_results
    }
    
    # Save results
    save_json(results["metrics"], output_dir / "metrics.json")
    save_json(results["bootstrap"], output_dir / "bootstrap.json")
    save_json(results["calibration"], output_dir / "calibration.json")
    
    return results