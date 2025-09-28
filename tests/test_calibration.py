from __future__ import annotations

import pandas as pd

from oncotarget_lite.eval import evaluate_predictions


def test_calibration_outputs_are_valid(synthetic_reports):
    artifacts = evaluate_predictions(
        reports_dir=synthetic_reports,
        n_bootstrap=100,
        ci=0.9,
        bins=8,
        seed=21,
    )

    assert 0.0 <= artifacts.metrics.ece <= 1.0

    calibration_path = synthetic_reports / "calibration.csv"
    assert calibration_path.exists()

    calibration = pd.read_csv(calibration_path)
    assert calibration["bin"].is_monotonic_increasing
    assert (calibration["confidence"] >= 0).all() and (calibration["confidence"] <= 1).all()
