"""Pure, dependency-light bootstrap confidence interval and classification metrics helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def bootstrap_ci(
    values: np.ndarray, alpha: float = 0.05, n_boot: int = 2000, seed: int = 1337
) -> tuple[float, float]:
    """
    Compute percentile bootstrap confidence interval.
    
    Args:
        values: Array of values to bootstrap
        alpha: Significance level (default 0.05 for 95% CI)
        n_boot: Number of bootstrap samples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (lower_bound, upper_bound) for (1-alpha)*100% confidence interval
    """
    if len(values) == 0:
        raise ValueError("Cannot compute CI for empty array")
    
    rng = np.random.RandomState(seed)
    bootstrap_samples = []
    
    for _ in range(n_boot):
        bootstrap_idx = rng.choice(len(values), size=len(values), replace=True)
        bootstrap_sample = values[bootstrap_idx]
        bootstrap_samples.append(np.mean(bootstrap_sample))
    
    bootstrap_samples = np.array(bootstrap_samples)
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_samples, lower_percentile)
    upper_bound = np.percentile(bootstrap_samples, upper_percentile)
    
    return lower_bound, upper_bound


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have same length")
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece_value = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece_value += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece_value


def compute_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_score: Predicted scores/probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary with AUROC, AUPRC, accuracy, and ECE
    """
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have same length")
    
    # Handle edge case where all labels are the same
    if len(np.unique(y_true)) == 1:
        auroc = 0.5  # Random performance when no positive or negative class
        auprc = np.mean(y_true)  # Baseline AUPRC is class proportion
    else:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    
    y_pred = (y_score >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    ece_value = ece(y_true, y_score)
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "ece": ece_value,
    }