from __future__ import annotations

import numpy as np

from oncotarget_lite.metrics import (
    auprc,
    auroc,
    bootstrap_confidence_interval,
    brier_score,
    calibration_curve,
    expected_calibration_error,
)

EXPECTED_ROC = 0.75
EXPECTED_PR = 0.7916666666666666
MIN_CALIBRATION_ROWS = 2


def test_binary_metrics_match_manual_expectations() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    roc = auroc(y_true, y_prob)
    pr = auprc(y_true, y_prob)
    brier = brier_score(y_true, y_prob)
    assert np.isclose(roc, EXPECTED_ROC, atol=1e-6)
    assert np.isclose(pr, EXPECTED_PR, atol=1e-6)
    assert np.isclose(brier, np.mean((y_prob - y_true) ** 2))


def test_calibration_curve_bins() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.2, 0.75, 0.9])
    curve = calibration_curve(y_true, y_prob, bins=4)
    assert curve.shape[0] >= MIN_CALIBRATION_ROWS
    ece = expected_calibration_error(curve, total=len(y_true))
    assert 0.0 <= ece <= 1.0


def test_bootstrap_confidence_interval_reproducible() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob = np.linspace(0.1, 0.9, num=6)
    result_a = bootstrap_confidence_interval(
        y_true, y_prob, auroc, samples=32, ci_level=0.9, seed=123
    )
    result_b = bootstrap_confidence_interval(
        y_true, y_prob, auroc, samples=32, ci_level=0.9, seed=123
    )
    assert result_a == result_b
