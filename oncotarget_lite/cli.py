"""Typer CLI for orchestrating the oncotarget-lite pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlflow
import typer

from . import __version__
from .data import PROCESSED_DIR, RAW_DIR, prepare_dataset
from .eval import evaluate_predictions
from .explain import SHAP_DIR, generate_shap
from .model import MODELS_DIR, TrainConfig, train_model
from .reporting import build_docs_index, generate_scorecard
from .utils import ensure_dir, git_commit

app = typer.Typer(help="oncotarget-lite lifecycle commands")

RUN_CONTEXT = Path("reports/run_context.json")


def _save_run_context(run_id: str, tracking_uri: str) -> None:
    ensure_dir(RUN_CONTEXT.parent)
    RUN_CONTEXT.write_text(json.dumps({"run_id": run_id, "tracking_uri": tracking_uri}, indent=2))


def _load_run_context() -> Optional[dict[str, str]]:
    if RUN_CONTEXT.exists():
        return json.loads(RUN_CONTEXT.read_text())
    return None


def _start_run(run_name: str | None = None) -> mlflow.ActiveRun:
    tracking_uri = mlflow.get_tracking_uri()
    run = mlflow.start_run(run_name=run_name)
    if run is None:
        raise RuntimeError("Failed to start MLflow run")
    _save_run_context(run.info.run_id, tracking_uri)
    return run


@app.command()
def prepare(
    raw_dir: Path = typer.Option(RAW_DIR, dir_okay=True, help="Directory with synthetic raw CSVs"),
    processed_dir: Path = typer.Option(PROCESSED_DIR, dir_okay=True, help="Output directory"),
    test_size: float = typer.Option(0.3, min=0.1, max=0.5, help="Test split fraction"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Create processed features, labels, and splits."""

    prepared = prepare_dataset(raw_dir=raw_dir, processed_dir=processed_dir, test_size=test_size, seed=seed)
    typer.echo(
        json.dumps(
            {
                "features": str(processed_dir / "features.parquet"),
                "labels": str(processed_dir / "labels.parquet"),
                "splits": str(processed_dir / "splits.json"),
                "dataset_hash": prepared.dataset_fingerprint,
            },
            indent=2,
        )
    )


@app.command()
def train(
    processed_dir: Path = typer.Option(PROCESSED_DIR, dir_okay=True, help="Processed data directory"),
    models_dir: Path = typer.Option(MODELS_DIR, dir_okay=True, help="Model output directory"),
    reports_dir: Path = typer.Option(Path("reports"), dir_okay=True, help="Report output directory"),
    C: float = typer.Option(1.0, help="Inverse regularisation strength"),
    max_iter: int = typer.Option(500, help="Maximum solver iterations"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Train the logistic regression model and log run metadata."""

    mlflow.set_tracking_uri(str(Path.cwd() / "mlruns"))
    mlflow.set_experiment("oncotarget-lite")

    cfg = TrainConfig(C=C, max_iter=max_iter, seed=seed)
    with _start_run(run_name="train") as run:
        train_result = train_model(processed_dir=processed_dir, models_dir=models_dir, reports_dir=reports_dir, config=cfg)

        clf = train_result.pipeline.named_steps["clf"]
        mlflow.log_params(
            {
                "algorithm": "logistic_regression",
                "seed": seed,
                "regularization_C": C,
                "max_iter": max_iter,
                "class_weight": "balanced",
                "feature_count": getattr(clf, "n_features_in_", 0),
            }
        )
        tags = {
            "git_commit": git_commit(),
            "code_version": __version__,
        }
        if train_result.dataset_hash:
            tags["dataset_hash"] = train_result.dataset_hash
        mlflow.set_tags(tags)
        mlflow.log_metrics(
            {
                "train_auroc": train_result.train_metrics["auroc"],
                "train_ap": train_result.train_metrics["ap"],
                "test_auroc": train_result.test_metrics["auroc"],
                "test_ap": train_result.test_metrics["ap"],
            }
        )
        mlflow.log_artifacts(str(models_dir))
        mlflow.log_artifacts(str(reports_dir))

        typer.echo(json.dumps({"mlflow_run_id": run.info.run_id, "tracking_uri": mlflow.get_tracking_uri()}, indent=2))


@app.command("eval")
def eval_cmd(
    reports_dir: Path = typer.Option(Path("reports"), dir_okay=True),
    n_bootstrap: int = typer.Option(1000, min=100, max=2000),
    ci: float = typer.Option(0.95, min=0.5, max=0.999),
    bins: int = typer.Option(10, min=5, max=20),
    seed: int = typer.Option(42),
) -> None:
    """Compute metrics, calibration curves, and bootstrap confidence intervals."""

    result = evaluate_predictions(reports_dir=reports_dir, n_bootstrap=n_bootstrap, ci=ci, bins=bins, seed=seed)

    context = _load_run_context()
    if context:
        mlflow.set_tracking_uri(context["tracking_uri"])
        mlflow.set_experiment("oncotarget-lite")
        with mlflow.start_run(run_id=context["run_id"]):
            mlflow.log_metrics(
                {
                    "auroc": result.metrics.auroc,
                    "ap": result.metrics.ap,
                    "brier": result.metrics.brier,
                    "ece": result.metrics.ece,
                    "accuracy": result.metrics.accuracy,
                    "f1": result.metrics.f1,
                    "overfit_gap": result.metrics.overfit_gap,
                }
            )
            mlflow.log_artifact(str(reports_dir / "metrics.json"))
            mlflow.log_artifact(str(reports_dir / "bootstrap.json"))
            mlflow.log_artifact(str(reports_dir / "calibration.json"))
            for plot in ("roc_curve.png", "pr_curve.png", "calibration_plot.png"):
                plot_path = reports_dir / plot
                if plot_path.exists():
                    mlflow.log_artifact(str(plot_path))

    typer.echo(json.dumps(result.metrics.__dict__, indent=2))


@app.command()
def explain(
    processed_dir: Path = typer.Option(PROCESSED_DIR, dir_okay=True),
    models_dir: Path = typer.Option(MODELS_DIR, dir_okay=True),
    shap_dir: Path = typer.Option(SHAP_DIR, dir_okay=True),
    seed: int = typer.Option(42),
    background_size: int = typer.Option(100, min=10, max=500),
) -> None:
    """Generate SHAP explanations and persist PNG artefacts."""

    artifacts = generate_shap(
        processed_dir=processed_dir,
        models_dir=models_dir,
        shap_dir=shap_dir,
        seed=seed,
        background_size=background_size,
    )

    context = _load_run_context()
    if context:
        mlflow.set_tracking_uri(context["tracking_uri"])
        mlflow.set_experiment("oncotarget-lite")
        with mlflow.start_run(run_id=context["run_id"]):
            mlflow.log_artifacts(str(shap_dir))

    typer.echo(json.dumps({"shap_dir": str(shap_dir), "examples": artifacts.alias_map}, indent=2))


@app.command()
def scorecard(
    reports_dir: Path = typer.Option(Path("reports"), dir_okay=True),
    shap_dir: Path = typer.Option(SHAP_DIR, dir_okay=True),
    output_path: Path = typer.Option(Path("reports/target_scorecard.html"), dir_okay=True, file_okay=True),
) -> None:
    """Build the HTML scorecard linking metrics and SHAP figures."""

    result = generate_scorecard(reports_dir=reports_dir, shap_dir=shap_dir, output_path=output_path)

    context = _load_run_context()
    if context and output_path.exists():
        mlflow.set_tracking_uri(context["tracking_uri"])
        mlflow.set_experiment("oncotarget-lite")
        with mlflow.start_run(run_id=context["run_id"]):
            mlflow.log_artifact(str(output_path))

    typer.echo(json.dumps({"scorecard": str(result)}, indent=2))


@app.command()
def snapshot(
    output_path: Path = typer.Option(Path("reports/streamlit_demo.png"), file_okay=True, dir_okay=True),
    timeout: int = typer.Option(30, help="Seconds to wait for Streamlit to boot before capturing"),
) -> None:
    """Capture a static snapshot of the Streamlit demo."""

    from .streamlit_capture import capture_streamlit

    image_path = capture_streamlit(output_path=output_path, timeout=timeout)

    context = _load_run_context()
    if context and image_path.exists():
        mlflow.set_tracking_uri(context["tracking_uri"])
        mlflow.set_experiment("oncotarget-lite")
        with mlflow.start_run(run_id=context["run_id"]):
            mlflow.log_artifact(str(image_path))

    typer.echo(json.dumps({"snapshot": str(image_path)}, indent=2))


@app.command()
def docs(
    reports_dir: Path = typer.Option(Path("reports"), dir_okay=True),
    docs_dir: Path = typer.Option(Path("docs"), dir_okay=True),
    model_card: Path = typer.Option(Path("oncotarget_lite/model_card.md"), file_okay=True),
) -> None:
    """Generate lightweight documentation landing page."""

    output = build_docs_index(reports_dir=reports_dir, docs_dir=docs_dir, model_card=model_card)
    typer.echo(json.dumps({"docs_index": str(output)}, indent=2))


@app.command()
def all(
    seed: int = typer.Option(42, help="Pipeline seed"),
) -> None:
    """Run the full end-to-end pipeline sequentially."""

    prepare(seed=seed)
    train(seed=seed)
    eval_cmd(seed=seed)
    explain(seed=seed)
    scorecard()
    docs()
    snapshot()


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
