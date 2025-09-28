from __future__ import annotations

from oncotarget_lite.eval import evaluate_predictions


def test_bootstrap_intervals_are_ordered(synthetic_reports):
    artifacts = evaluate_predictions(
        reports_dir=synthetic_reports,
        n_bootstrap=200,
        ci=0.95,
        bins=10,
        seed=13,
    )

    assert artifacts.auroc_ci.lower <= artifacts.auroc_ci.mean <= artifacts.auroc_ci.upper
    assert artifacts.ap_ci.lower <= artifacts.ap_ci.mean <= artifacts.ap_ci.upper

    metrics_path = synthetic_reports / "metrics.json"
    bootstrap_path = synthetic_reports / "bootstrap.json"
    assert metrics_path.exists()
    assert bootstrap_path.exists()
