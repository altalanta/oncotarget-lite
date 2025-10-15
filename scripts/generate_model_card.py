"""Generate a lightweight model card from the latest MLflow run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from oncotarget_lite.utils import ensure_dir, git_commit


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_table(rows: Iterable[tuple[str, str]]) -> str:
    header = "| Metric | Value |\n| --- | --- |"
    body = "\n".join(f"| {name} | {value} |" for name, value in rows)
    return f"{header}\n{body}"


def _find_latest_run() -> Tuple[Optional["mlflow.entities.Run"], str]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:  # pragma: no cover - dependency not available
        raise RuntimeError("MLflow is required to generate the model card") from exc

    client = MlflowClient()
    tracking_uri = mlflow.get_tracking_uri() or "mlruns"
    experiment = client.get_experiment_by_name("oncotarget-lite")
    experiment_ids = [experiment.experiment_id] if experiment else ["0"]

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    return (runs[0] if runs else None, tracking_uri)


def build_model_card(output: Path) -> Path:
    run, tracking_uri = _find_latest_run()

    metrics: Dict[str, float] = {}
    params: Dict[str, str] = {}
    tags: Dict[str, str] = {}
    run_id = "unknown"
    start_time: Optional[datetime] = None

    if run:
        metrics = run.data.metrics.copy()
        params = run.data.params.copy()
        tags = run.data.tags.copy()
        run_id = run.info.run_id
        if run.info.start_time:
            start_time = datetime.fromtimestamp(run.info.start_time / 1000.0)

    calibration = _load_json(Path("reports/calibration.json"))
    bootstrap = _load_json(Path("reports/bootstrap.json"))

    ci_rows = []
    for metric in ("auroc", "ap"):
        interval = bootstrap.get(metric, {}) if isinstance(bootstrap, dict) else {}
        if not isinstance(interval, dict) or not interval:
            continue
        mean = interval.get("mean", "n/a")
        half = interval.get("ci_half_width", "n/a")
        if isinstance(mean, (int, float)) and isinstance(half, (int, float)):
            value = f"{mean:.3f} ± {half:.3f}"
        else:
            value = f"{mean} ± {half}"
        ci_rows.append((metric.upper(), value))

    metric_keys = set(metrics.keys())
    perf_rows = [
        (
            "Train AUROC",
            f"{metrics.get('train_auroc', 'n/a'):.3f}" if "train_auroc" in metric_keys else "n/a",
        ),
        (
            "Train AP",
            f"{metrics.get('train_ap', 'n/a'):.3f}" if "train_ap" in metric_keys else "n/a",
        ),
        (
            "Test AUROC",
            f"{metrics.get('test_auroc', metrics.get('auroc', 'n/a')):.3f}" if {"test_auroc", "auroc"} & metric_keys else "n/a",
        ),
        (
            "Test AP",
            f"{metrics.get('test_ap', metrics.get('ap', 'n/a')):.3f}" if {"test_ap", "ap"} & metric_keys else "n/a",
        ),
        ("ECE", f"{metrics.get('ece', 'n/a'):.3f}" if "ece" in metric_keys else "n/a"),
        ("Brier", f"{metrics.get('brier', 'n/a'):.3f}" if "brier" in metric_keys else "n/a"),
    ]

    calibration_summary = "N/A"
    if isinstance(calibration, dict):
        bins = calibration.get("bins")
        if isinstance(bins, list) and bins:
            calibration_summary = f"{len(bins)} bins; mean expected={calibration.get('mean_expected', 'n/a')}, mean observed={calibration.get('mean_observed', 'n/a')}"

    metadata_rows = [
        ("Run ID", run_id),
        ("Timestamp", start_time.isoformat() if start_time else "n/a"),
        ("Git Commit", tags.get("git_commit", git_commit())),
        ("Dataset Hash", tags.get("dataset_hash", "n/a")),
        ("Tracking URI", tracking_uri),
    ]

    params_rows = [(key, str(value)) for key, value in sorted(params.items())]

    content = [
        "# Model Card",
        "",
        "## Metadata",
        _format_table(metadata_rows),
        "",
        "## Performance",
        _format_table(perf_rows),
    ]

    if ci_rows:
        content.extend(["", "### Confidence Intervals", _format_table(ci_rows)])

    content.extend(
        [
            "",
            "## Calibration",
            calibration_summary,
            "",
            "## Training Parameters",
            _format_table(params_rows or [("(none)", "-")]),
        ]
    )

    ensure_dir(output.parent)
    output.write_text("\n".join(content) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/model_card.md from MLflow artefacts")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/model_card.md"),
        help="Output markdown path",
    )
    args = parser.parse_args()

    build_model_card(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
