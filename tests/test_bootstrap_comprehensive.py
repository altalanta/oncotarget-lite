"""Comprehensive tests for evaluation/bootstrap.py to boost coverage."""

import pytest
import numpy as np
from oncotarget_lite.evaluation.bootstrap import (
    bootstrap_ci, ece, compute_classification_metrics
)


def test_bootstrap_ci_basic():
    """Test basic bootstrap confidence interval computation."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower, upper = bootstrap_ci(values, alpha=0.05, n_boot=100, seed=42)
    
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower <= upper
    assert lower <= np.mean(values) <= upper


def test_bootstrap_ci_empty_array():
    """Test bootstrap_ci with empty array."""
    values = np.array([])
    
    with pytest.raises(ValueError) as exc_info:
        bootstrap_ci(values)
    
    assert "Cannot compute CI for empty array" in str(exc_info.value)


def test_bootstrap_ci_single_value():
    """Test bootstrap_ci with single value."""
    values = np.array([5.0])
    lower, upper = bootstrap_ci(values, alpha=0.05, n_boot=50, seed=42)
    
    # With a single value, CI should be very narrow around that value
    assert abs(lower - 5.0) < 0.1
    assert abs(upper - 5.0) < 0.1
    assert lower <= upper


def test_bootstrap_ci_reproducible():
    """Test that bootstrap_ci is reproducible with same seed."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    lower1, upper1 = bootstrap_ci(values, seed=123)
    lower2, upper2 = bootstrap_ci(values, seed=123)
    
    assert lower1 == lower2
    assert upper1 == upper2


def test_bootstrap_ci_different_seeds():
    """Test that different seeds give different results."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    lower1, upper1 = bootstrap_ci(values, seed=123)
    lower2, upper2 = bootstrap_ci(values, seed=456)
    
    # With different seeds, results should generally be different
    assert (lower1 != lower2) or (upper1 != upper2)


def test_bootstrap_ci_custom_alpha():
    """Test bootstrap_ci with custom alpha (confidence level)."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 90% CI (alpha=0.1) should be narrower than 95% CI (alpha=0.05)
    lower_95, upper_95 = bootstrap_ci(values, alpha=0.05, seed=42)
    lower_90, upper_90 = bootstrap_ci(values, alpha=0.1, seed=42)
    
    assert (upper_90 - lower_90) <= (upper_95 - lower_95)


def test_bootstrap_ci_custom_n_boot():
    """Test bootstrap_ci with custom number of bootstrap samples."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Should work with different n_boot values
    lower1, upper1 = bootstrap_ci(values, n_boot=100, seed=42)
    lower2, upper2 = bootstrap_ci(values, n_boot=500, seed=42)
    
    assert isinstance(lower1, float)
    assert isinstance(upper1, float)
    assert isinstance(lower2, float)
    assert isinstance(upper2, float)


def test_ece_basic():
    """Test basic ECE computation."""
    # Perfect calibration: predicted probs match true frequencies
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.3, 0.7, 0.8])
    
    ece_value = ece(y_true, y_prob, n_bins=2)
    assert isinstance(ece_value, float)
    assert ece_value >= 0.0


def test_ece_perfect_calibration():
    """Test ECE with perfect calibration."""
    # Perfectly calibrated: prob 0.5 should have 50% accuracy
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    ece_value = ece(y_true, y_prob, n_bins=10)
    assert ece_value < 0.1  # Should be very small for perfect calibration


def test_ece_poor_calibration():
    """Test ECE with poor calibration."""
    # Poor calibration: high confidence but wrong predictions
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.9, 0.95, 0.85, 0.9])
    
    ece_value = ece(y_true, y_prob, n_bins=10)
    assert ece_value > 0.5  # Should be high for poor calibration


def test_ece_mismatched_lengths():
    """Test ECE with mismatched array lengths."""
    y_true = np.array([0, 1])
    y_prob = np.array([0.3, 0.7, 0.5])
    
    with pytest.raises(ValueError) as exc_info:
        ece(y_true, y_prob)
    
    assert "y_true and y_prob must have same length" in str(exc_info.value)


def test_ece_single_bin():
    """Test ECE with single bin."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])
    
    ece_value = ece(y_true, y_prob, n_bins=1)
    assert isinstance(ece_value, float)
    assert ece_value >= 0.0


def test_ece_many_bins():
    """Test ECE with many bins."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    
    ece_value = ece(y_true, y_prob, n_bins=20)
    assert isinstance(ece_value, float)
    assert ece_value >= 0.0


def test_compute_classification_metrics_basic():
    """Test basic classification metrics computation."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    
    metrics = compute_classification_metrics(y_true, y_score)
    
    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "accuracy" in metrics
    assert "ece" in metrics
    
    assert 0 <= metrics["auroc"] <= 1
    assert 0 <= metrics["auprc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["ece"] >= 0


def test_compute_classification_metrics_perfect():
    """Test classification metrics with perfect predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.0, 0.1, 0.9, 1.0])
    
    metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)
    
    assert metrics["auroc"] == 1.0  # Perfect AUROC
    assert metrics["accuracy"] == 1.0  # Perfect accuracy


def test_compute_classification_metrics_all_positive():
    """Test classification metrics with all positive labels."""
    y_true = np.array([1, 1, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    
    metrics = compute_classification_metrics(y_true, y_score)
    
    assert metrics["auroc"] == 0.5  # Random performance for single class
    assert metrics["auprc"] == 1.0  # All samples are positive


def test_compute_classification_metrics_all_negative():
    """Test classification metrics with all negative labels."""
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    
    metrics = compute_classification_metrics(y_true, y_score)
    
    assert metrics["auroc"] == 0.5  # Random performance for single class
    assert metrics["auprc"] == 0.0  # No positive samples


def test_compute_classification_metrics_mismatched_lengths():
    """Test classification metrics with mismatched array lengths."""
    y_true = np.array([0, 1])
    y_score = np.array([0.3, 0.7, 0.5])
    
    with pytest.raises(ValueError) as exc_info:
        compute_classification_metrics(y_true, y_score)
    
    assert "y_true and y_score must have same length" in str(exc_info.value)


def test_compute_classification_metrics_custom_threshold():
    """Test classification metrics with custom threshold."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    
    # Test with different thresholds
    metrics_05 = compute_classification_metrics(y_true, y_score, threshold=0.5)
    metrics_03 = compute_classification_metrics(y_true, y_score, threshold=0.3)
    
    # Accuracy may change with different thresholds
    assert isinstance(metrics_05["accuracy"], float)
    assert isinstance(metrics_03["accuracy"], float)
    
    # AUROC and AUPRC should be the same regardless of threshold
    assert metrics_05["auroc"] == metrics_03["auroc"]
    assert metrics_05["auprc"] == metrics_03["auprc"]


def test_compute_classification_metrics_edge_cases():
    """Test classification metrics with edge cases."""
    # Single sample
    y_true = np.array([1])
    y_score = np.array([0.7])
    
    metrics = compute_classification_metrics(y_true, y_score)
    assert all(isinstance(v, float) for v in metrics.values())
    
    # Two samples, opposite classes
    y_true = np.array([0, 1])
    y_score = np.array([0.3, 0.7])
    
    metrics = compute_classification_metrics(y_true, y_score)
    assert metrics["auroc"] == 1.0  # Perfect separation
    assert metrics["accuracy"] == 1.0  # Perfect accuracy with threshold 0.5


def test_bootstrap_edge_cases():
    """Test bootstrap function with edge cases."""
    # Very small array
    values = np.array([3.14])
    lower, upper = bootstrap_ci(values, n_boot=10, seed=42)
    assert lower == upper == 3.14
    
    # Array with identical values
    values = np.array([5.0, 5.0, 5.0, 5.0])
    lower, upper = bootstrap_ci(values, seed=42)
    assert abs(lower - 5.0) < 1e-10
    assert abs(upper - 5.0) < 1e-10


def test_ece_edge_cases():
    """Test ECE function with edge cases."""
    # All predictions in one bin
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.95, 0.96, 0.97, 0.98])
    
    ece_value = ece(y_true, y_prob, n_bins=10)
    assert isinstance(ece_value, float)
    assert ece_value >= 0
    
    # Probabilities at bin boundaries
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 0.5, 1.0, 0.25])
    
    ece_value = ece(y_true, y_prob, n_bins=4)
    assert isinstance(ece_value, float)
    assert ece_value >= 0