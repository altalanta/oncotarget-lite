"""Typer CLI for orchestrating the oncotarget-lite pipeline."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
import os
from typing import Any, Dict, List, Optional

import typer

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guards runtime import
    load_dotenv = None  # type: ignore

from . import __version__
from .config import CI_PROFILE_NAME, PROFILE_ENV_VAR, profile_overrides
from .utils import ensure_dir, git_commit

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SHAP_DIR = REPORTS_DIR / "shap"


def _data_module():
    from . import data as data_module

    return data_module


def _model_module():
    from . import model as model_module

    return model_module


def _eval_module():
    from . import eval as eval_module

    return eval_module


def _explain_module():
    from . import explain as explain_module

    return explain_module


def _reporting_module():
    from . import reporting as reporting_module

    return reporting_module

app = typer.Typer(help="oncotarget-lite lifecycle commands")

if load_dotenv is not None:
    load_dotenv()


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Parameter profile to apply (e.g. ci)",
        show_default=False,
    ),
    fast: bool = typer.Option(False, "--fast", help="Use the CI profile for a rapid smoke run"),
    ci_mode: bool = typer.Option(
        False,
        "--ci",
        help="Alias for --fast to align with CI and automation scripts",
    ),
) -> None:
    """Global CLI entry-point to wire profile shortcuts before subcommands execute."""

    if fast or ci_mode:
        os.environ[PROFILE_ENV_VAR] = CI_PROFILE_NAME
    elif profile:
        os.environ[PROFILE_ENV_VAR] = profile

    if ctx.invoked_subcommand is None and not ctx.resilient_parsing:
        typer.echo(ctx.get_help())
        raise typer.Exit()

RUN_CONTEXT = Path("reports/run_context.json")


def _save_run_context(run_id: str, tracking_uri: str) -> None:
    ensure_dir(RUN_CONTEXT.parent)
    RUN_CONTEXT.write_text(json.dumps({"run_id": run_id, "tracking_uri": tracking_uri}, indent=2))


def _load_run_context() -> Optional[dict[str, str]]:
    if RUN_CONTEXT.exists():
        return json.loads(RUN_CONTEXT.read_text())
    return None


def _with_overrides(command: str, *, profile: str | None = None, fast: bool = False, ci: bool = False, base: Dict[str, Any]) -> Dict[str, Any]:
    """Return base parameters updated with profile overrides."""

    overrides = profile_overrides(command, profile=profile, fast=fast, ci=ci)
    merged = base.copy()
    merged.update(overrides)
    return merged


def _start_run(run_name: str | None = None):
    from .utils import _mlflow
    mlflow = _mlflow()
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
    profile: Optional[str] = typer.Option(
        None, help="Parameter profile to apply (e.g. ci)", show_default=False
    ),
    fast: bool = typer.Option(False, "--fast", help="Use the CI profile for a rapid smoke run"),
    ci_mode: bool = typer.Option(False, "--ci", help="Alias for --fast to align with automation scripts"),
) -> None:
    """Create processed features, labels, and splits."""

    resolved = _with_overrides(
        "prepare",
        profile=profile,
        fast=fast,
        ci=ci_mode,
        base={"test_size": test_size, "seed": seed},
    )
    test_size = float(resolved["test_size"])
    seed = int(resolved["seed"])

    data_module = _data_module()
    prepared = data_module.prepare_dataset(
        raw_dir=raw_dir, processed_dir=processed_dir, test_size=test_size, seed=seed
    )
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
    model_type: str = typer.Option("logreg", help="Model type: logreg, xgb, lgb, mlp, transformer, gnn"),
    C: float = typer.Option(1.0, help="Inverse regularisation strength (logreg only)"),
    max_iter: int = typer.Option(500, help="Maximum solver iterations (logreg only)"),
    seed: int = typer.Option(42, help="Random seed"),
    config: Optional[Path] = typer.Option(None, exists=True, help="Ablation config file"),
    all_ablations: bool = typer.Option(False, help="Run all ablation experiments"),
    profile: Optional[str] = typer.Option(
        None, help="Parameter profile to apply (e.g. ci)", show_default=False
    ),
    fast: bool = typer.Option(False, "--fast", help="Use the CI profile for a rapid smoke run"),
    ci_mode: bool = typer.Option(False, "--ci", help="Alias for --fast to align with automation scripts"),
) -> None:
    """Train a model or run ablation experiments."""

    overrides = _with_overrides(
        "train",
        profile=profile,
        fast=fast,
        ci=ci_mode,
        base={"C": C, "max_iter": max_iter, "seed": seed, "model_type": model_type},
    )
    C = float(overrides["C"])
    max_iter = int(overrides["max_iter"])
    seed = int(overrides["seed"])
    model_type = str(overrides["model_type"])

    # Handle ablation experiments
    if all_ablations or config:
        from .ablations import discover_ablation_configs, run_ablation_experiment
        
        if all_ablations:
            config_paths = discover_ablation_configs()
            if not config_paths:
                typer.echo("No ablation configs found in configs/ablations/")
                raise typer.Exit(1)
        else:
            config_paths = [config]
        
        results = []
        for config_path in config_paths:
            typer.echo(f"Running ablation: {config_path.stem}")
            result = run_ablation_experiment(
                config_path, processed_dir, models_dir, reports_dir
            )
            results.append(result)
        
        typer.echo(json.dumps({"ablations": len(results), "experiments": [r["experiment"] for r in results]}, indent=2))
        return

    # Original training logic
    from .utils import _mlflow
    mlflow = _mlflow()
    mlflow.set_tracking_uri(str(Path.cwd() / "mlruns"))
    mlflow.set_experiment("oncotarget-lite")

    model_module = _model_module()
    cfg = model_module.TrainConfig(C=C, max_iter=max_iter, model_type=model_type, seed=seed)
    with _start_run(run_name="train") as run:
        train_result = model_module.train_model(
            processed_dir=processed_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=cfg,
        )

        clf = train_result.pipeline.named_steps["clf"]
        mlflow.log_params(
            {
                "algorithm": model_type,
                "seed": seed,
                "regularization_C": C if model_type == "logreg" else None,
                "max_iter": max_iter if model_type == "logreg" else None,
                "class_weight": "balanced" if model_type == "logreg" else None,
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


@app.command()
def dashboard(
    shap_dir: Path = typer.Option(Path("reports/shap"), dir_okay=True, help="SHAP explanations directory"),
    output_dir: Path = typer.Option(Path("reports/dashboard"), dir_okay=True, help="Dashboard output directory"),
    validation_report: Optional[Path] = typer.Option(None, help="Path to validation report for enhanced analysis"),
    model_comparison: bool = typer.Option(False, help="Create model comparison dashboard"),
    model_reports: Optional[List[Path]] = typer.Option(None, help="List of validation reports for comparison"),
    export_static: bool = typer.Option(True, help="Export static images in addition to HTML"),
) -> None:
    """Generate advanced interpretability dashboard with interactive visualizations."""

    from .interpretability_dashboard import InterpretabilityDashboard

    dashboard = InterpretabilityDashboard(shap_dir)

    if model_comparison:
        if not model_reports:
            typer.echo("Error: --model-reports required for model comparison")
            raise typer.Exit(1)
        dashboard.create_model_comparison_dashboard(model_reports, output_dir / "model_comparison.html")
    else:
        # Create comprehensive dashboard
        dashboard.create_comprehensive_dashboard(output_dir / "interpretability_dashboard.html")

        # Export static images if requested
        if export_static:
            dashboard.save_static_exports(output_dir / "static")

    typer.echo(f"✅ Interpretability dashboard generated in {output_dir}")


@app.command()
def cache(
    action: str = typer.Option(..., help="Cache action: info, clear, benchmark"),
    pattern: str = typer.Option("*", help="File pattern for cache operations"),
    cache_dir: Path = typer.Option(Path("data/cache"), dir_okay=True, help="Cache directory"),
) -> None:
    """Manage data processing cache for improved performance."""

    from .scalable_loader import ScalableDataLoader

    loader = ScalableDataLoader(cache_dir)

    if action == "info":
        stats = loader.get_cache_stats()
        typer.echo(f"📊 Cache Statistics:")
        typer.echo(f"   Total files: {stats['total_files']}")
        typer.echo(f"   Total size: {stats['total_size_mb']:.2f} MB")
        typer.echo(f"   File sizes: {list(stats['file_sizes'].keys())[:5]}...")

    elif action == "clear":
        cleared = loader.clear_cache(pattern)
        typer.echo(f"🗑️  Cleared {cleared} cache files matching '{pattern}'")

    elif action == "benchmark":
        # Create a simple benchmark
        import pandas as pd
        import numpy as np

        # Create test data
        test_genes = pd.Series([f"GENE{i:04d}" for i in range(100)])

        start_time = time.time()
        loader.load_files_parallel({
            "test": (Path("data/raw/expression.csv"), {"usecols": ["gene", "median_TPM"]})
        })
        end_time = time.time()

        typer.echo(f"⏱️  Benchmark completed in {end_time - start_time:.2f}")

    else:
        typer.echo(f"❌ Unknown action: {action}. Use: info, clear, benchmark")


@app.command()
def optimize(
    processed_dir: Path = typer.Option(PROCESSED_DIR, dir_okay=True, help="Processed data directory"),
    models_dir: Path = typer.Option(MODELS_DIR, dir_okay=True, help="Model output directory"),
    reports_dir: Path = typer.Option(REPORTS_DIR, dir_okay=True, help="Report output directory"),
    model_type: str = typer.Option("logreg", help="Model type: logreg, xgb, lgb, mlp, transformer, gnn"),
    n_trials: int = typer.Option(100, min=10, max=1000, help="Number of optimization trials"),
    timeout: Optional[int] = typer.Option(None, help="Timeout in seconds (optional)"),
    metric: str = typer.Option("auroc", help="Metric to optimize: auroc, ap"),
    study_name: str = typer.Option("oncotarget_optimization", help="Optuna study name"),
    storage_path: str = typer.Option("sqlite:///reports/optuna_study.db", help="Optuna storage path"),
) -> None:
    """Run automated hyperparameter optimization using Optuna."""

    from .optimizers import HyperparameterOptimizer

    typer.echo(f"🔍 Running hyperparameter optimization for {model_type} model...")

    # Create optimizer
    optimizer = HyperparameterOptimizer(
        study_name=study_name,
        storage_path=storage_path,
        n_trials=n_trials,
        timeout=timeout,
    )

    # Run optimization
    study = optimizer.optimize(
        model_type=model_type,
        processed_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        metric=metric,
    )

    # Save results
    summary_path = reports_dir / f"optuna_summary_{model_type}.json"
    optimizer.save_study_summary(study, summary_path)

    # Display results
    typer.echo(f"✅ Optimization completed for {model_type}!")
    typer.echo(f"📊 Best {metric.upper()}: {study.best_value:.4f}")
    typer.echo(f"🏆 Best parameters: {study.best_params}")
    typer.echo(f"📁 Results saved to: {summary_path}")

    # Log to MLflow if available
    try:
        import mlflow
        with mlflow.start_run(run_name=f"optimize_{model_type}"):
            mlflow.log_params(study.best_params)
            mlflow.log_metric(f"best_{metric}", study.best_value)
            mlflow.log_artifact(str(summary_path))
    except ImportError:
        pass


@app.command("eval")
def eval_cmd(
    reports_dir: Path = typer.Option(Path("reports"), dir_okay=True),
    n_bootstrap: int = typer.Option(1000, min=100, max=2000),
    confidence_interval: float = typer.Option(0.95, "--ci", min=0.5, max=0.999),
    bins: int = typer.Option(10, min=5, max=20),
    seed: int = typer.Option(42),
    distributed: bool = typer.Option(True, help="Use distributed computing for bootstrap"),
    profile: Optional[str] = typer.Option(
        None, help="Parameter profile to apply (e.g. ci)", show_default=False
    ),
    fast: bool = typer.Option(False, "--fast", help="Use the CI profile for a rapid smoke run"),
    ci_profile: bool = typer.Option(
        False,
        "--ci-profile",
        help="Alias for --fast when --ci is already used for confidence intervals",
    ),
) -> None:
    """Compute metrics, calibration curves, and bootstrap confidence intervals."""

    overrides = _with_overrides(
        "eval",
        profile=profile,
        fast=fast,
        ci=ci_profile,
        base={
            "n_bootstrap": n_bootstrap,
            "ci": confidence_interval,
            "bins": bins,
            "seed": seed,
            "distributed": distributed,
        },
    )
    n_bootstrap = int(overrides["n_bootstrap"])
    confidence_interval = float(overrides["ci"])
    bins = int(overrides["bins"])
    seed = int(overrides["seed"])
    distributed = bool(overrides["distributed"])

    eval_module = _eval_module()
    result = eval_module.evaluate_predictions(
        reports_dir=reports_dir,
        n_bootstrap=n_bootstrap,
        ci=confidence_interval,
        bins=bins,
        seed=seed,
        distributed=distributed,
    )

    context = _load_run_context()
    if context:
        from .utils import _mlflow
        mlflow = _mlflow()
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


@app.command("ablations")
def ablations_cmd(
    reports_dir: Path = typer.Option(Path("reports"), dir_okay=True, help="Reports directory"),
    n_bootstrap: int = typer.Option(1000, min=100, max=2000, help="Bootstrap samples"),
    confidence_interval: float = typer.Option(0.95, min=0.5, max=0.999, help="Confidence interval"),
    seed: int = typer.Option(42, help="Random seed"),
    distributed: bool = typer.Option(True, help="Use distributed computing for experiments"),
    parallel_jobs: int = typer.Option(-1, help="Number of parallel jobs (-1 for all cores)"),
    profile: Optional[str] = typer.Option(
        None, help="Parameter profile to apply (e.g. ci)", show_default=False
    ),
    fast: bool = typer.Option(False, "--fast", help="Use the CI profile for a rapid smoke run"),
    ci_profile: bool = typer.Option(False, "--ci", help="Alias for --fast to align with CI"),
) -> None:
    """Aggregate ablation results and compute bootstrap CIs."""
    from .ablations import run_all_ablation_experiments
    from .data import PROCESSED_DIR, RAW_DIR
    from .model import MODELS_DIR

    overrides = _with_overrides(
        "ablations",
        profile=profile,
        fast=fast,
        ci=ci_profile,
        base={
            "n_bootstrap": n_bootstrap,
            "ci": confidence_interval,
            "seed": seed,
            "distributed": distributed,
            "parallel_jobs": parallel_jobs,
        },
    )
    n_bootstrap = int(overrides["n_bootstrap"])
    confidence_interval = float(overrides["ci"])
    seed = int(overrides["seed"])
    distributed = bool(overrides["distributed"])
    parallel_jobs = int(overrides["parallel_jobs"])

    # Run ablation experiments in parallel if distributed is enabled
    if distributed:
        print("🔄 Running ablation experiments in parallel...")
        results = run_all_ablation_experiments(
            processed_dir=PROCESSED_DIR,
            models_dir=MODELS_DIR,
            reports_dir=reports_dir,
            distributed=True,
            n_jobs=parallel_jobs,
        )
        print(f"✅ Completed {len(results)} ablation experiments")
    else:
        print("🔄 Running ablation experiments sequentially...")

    # Aggregate results
    from .ablations_eval import aggregate_ablation_results
    result = aggregate_ablation_results(
        reports_dir=reports_dir,
        n_bootstrap=n_bootstrap,
        ci=confidence_interval,
        seed=seed,
    )

    typer.echo(json.dumps({"summary": str(result)}, indent=2))


@app.command()
def explain(
    processed_dir: Path = typer.Option(PROCESSED_DIR, dir_okay=True),
    models_dir: Path = typer.Option(MODELS_DIR, dir_okay=True),
    shap_dir: Path = typer.Option(SHAP_DIR, dir_okay=True),
    seed: int = typer.Option(42),
    background_size: int = typer.Option(100, min=10, max=500),
    distributed: bool = typer.Option(True, help="Use distributed computing for SHAP"),
    max_evals: int = typer.Option(None, help="Maximum SHAP evaluations (for sampling)"),
    profile: Optional[str] = typer.Option(
        None, help="Parameter profile to apply (e.g. ci)", show_default=False
    ),
    fast: bool = typer.Option(False, "--fast", help="Use the CI profile for a rapid smoke run"),
    ci_mode: bool = typer.Option(False, "--ci", help="Alias for --fast to align with CI"),
) -> None:
    """Generate SHAP explanations and persist PNG artefacts."""

    overrides = _with_overrides(
        "explain",
        profile=profile,
        fast=fast,
        ci=ci_mode,
        base={
            "seed": seed,
            "background_size": background_size,
            "distributed": distributed,
            "max_evals": max_evals,
        },
    )
    seed = int(overrides["seed"])
    background_size = int(overrides["background_size"])
    distributed = bool(overrides["distributed"])
    max_evals_raw = overrides["max_evals"]
    max_evals = None if max_evals_raw in (None, "") else int(max_evals_raw)

    explain_module = _explain_module()
    artifacts = explain_module.generate_shap(
        processed_dir=processed_dir,
        models_dir=models_dir,
        shap_dir=shap_dir,
        seed=seed,
        background_size=background_size,
        distributed=distributed,
        max_evals=max_evals,
    )

    context = _load_run_context()
    if context:
        from .utils import _mlflow
        mlflow = _mlflow()
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

    reporting_module = _reporting_module()
    result = reporting_module.generate_scorecard(
        reports_dir=reports_dir, shap_dir=shap_dir, output_path=output_path
    )

    context = _load_run_context()
    if context and output_path.exists():
        from .utils import _mlflow
        mlflow = _mlflow()
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
        from .utils import _mlflow
        mlflow = _mlflow()
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

    reporting_module = _reporting_module()
    output = reporting_module.build_docs_index(
        reports_dir=reports_dir, docs_dir=docs_dir, model_card=model_card
    )
    typer.echo(json.dumps({"docs_index": str(output)}, indent=2))


@app.command("generate-data")
def generate_data(out_dir: str = typer.Option("data/raw", help="Output directory for synthetic data")):
    """
    Generate synthetic development data (CSV files) with comment headers.
    """
    from pathlib import Path
    from scripts.generate_synthetic_data import main as gen
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    gen(out_dir)
    typer.echo(f"Synthetic data written to {out_dir}")


@app.command("validate")
def validate(
    schema_only: bool = typer.Option(False, help="Only validate schemas, skip housekeeping"),
) -> None:
    """Validate schemas and run housekeeping checks."""
    from .schemas import validate_all_schemas

    # Schema validation
    schemas_valid = validate_all_schemas()

    if schema_only:
        if schemas_valid:
            typer.echo("✅ All schemas valid")
        else:
            typer.echo("❌ Schema validation failed")
            raise typer.Exit(1)
        return

    # Housekeeping checks
    try:
        from scripts.housekeeping import generate_housekeeping_report
        report = generate_housekeeping_report()

        status = report["summary"]["status"]
        if status == "healthy":
            typer.echo("✅ System healthy")
        elif status == "optimizable":
            typer.echo("⚠️ System optimizable")
            typer.echo("Run 'python scripts/housekeeping.py' for recommendations")
        else:
            typer.echo("❌ System needs attention")
            for issue in report["summary"]["critical_issues"]:
                typer.echo(f"  - {issue}")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"⚠️ Housekeeping check failed: {e}")
        # Don't fail hard on housekeeping errors

    if not schemas_valid:
        raise typer.Exit(1)


@app.command("security")
def security_scan(
    verbose: bool = typer.Option(False, help="Show detailed security information"),
) -> None:
    """Run security audit on dependencies and project configuration."""
    from scripts.security_scan import main as security_main

    typer.echo("🛡️ Running security scan...")

    try:
        exit_code = security_main()
        if exit_code == 0:
            typer.echo("✅ Security scan passed")
        else:
            typer.echo("❌ Security issues found")
            raise typer.Exit(exit_code)
    except Exception as e:
        typer.echo(f"❌ Security scan failed: {e}")
        raise typer.Exit(1)


@app.command("distributed")
def distributed_cmd(
    backend: str = typer.Option("joblib", help="Distributed backend (joblib, dask, ray)"),
    n_jobs: int = typer.Option(-1, help="Number of parallel jobs (-1 for all cores)"),
    verbose: int = typer.Option(0, help="Verbosity level"),
    prefer: str = typer.Option("processes", help="Joblib preference (processes, threads)"),
    memory_limit: str = typer.Option(None, help="Memory limit for dask workers"),
    ray_address: str = typer.Option(None, help="Ray cluster address"),
) -> None:
    """Configure distributed computing settings."""
    from .distributed import configure_distributed

    configure_distributed(
        backend=backend,
        n_jobs=n_jobs,
        verbose=verbose,
        prefer=prefer,
        memory_limit=memory_limit,
        ray_address=ray_address,
    )

    typer.echo(f"✅ Distributed computing configured: {backend} backend with {n_jobs} jobs")


@app.command("monitor")
def monitor_cmd(
    action: str = typer.Argument("status", help="Action: status, capture, report, alerts"),
    model_version: str = typer.Option("latest", help="Model version to monitor"),
    days: int = typer.Option(30, help="Days of history to analyze"),
    enable_alerts: bool = typer.Option(True, help="Enable alert notifications"),
    slack_webhook: str = typer.Option(None, help="Slack webhook URL for alerts"),
    email_to: str = typer.Option(None, help="Email address for alerts"),
) -> None:
    """Model performance monitoring and drift detection."""
    from .monitoring import ModelMonitor, DriftAlert

    monitor = ModelMonitor(enable_alerts=enable_alerts)

    if action == "status":
        # Show monitoring status and recent snapshots
        report = monitor.get_monitoring_report(model_version=model_version, days=days)

        if "error" in report:
            typer.echo(f"❌ {report['error']}")
            return

        typer.echo("📊 Model Monitoring Status")
        typer.echo(f"Period: {report['period_days']} days")
        typer.echo(f"Snapshots: {report['snapshots_count']}")
        typer.echo(f"Alerts: {report['alerts_count']}")

        if report['snapshots_count'] > 0:
            latest = report['latest_performance']
            typer.echo("\n📈 Latest Performance:")
            typer.echo(f"  AUROC: {latest['auroc']:.3f}")
            typer.echo(f"  AP: {latest['ap']:.3f}")
            typer.echo(f"  Accuracy: {latest['accuracy']:.3f}")
            typer.echo(f"  F1: {latest['f1']:.3f}")

        if report['alerts_count'] > 0:
            typer.echo(f"\n🚨 Recent Alerts: {report['alerts_count']}")
            for alert_data in report['recent_alerts'][:3]:
                alert = DriftAlert(**alert_data)
                typer.echo(f"  • {alert.timestamp.strftime('%Y-%m-%d %H:%M')}: {alert.message}")

    elif action == "capture":
        # Capture current performance snapshot
        predictions_path = Path("reports/predictions.parquet")
        if not predictions_path.exists():
            typer.echo("❌ No predictions found. Run training and evaluation first.")
            raise typer.Exit(1)

        typer.echo("📸 Capturing performance snapshot...")

        # Get current model info
        context = _load_run_context()
        current_model_version = context["run_id"] if context else "unknown"
        dataset_hash = "unknown"  # Would need to extract from training

        snapshot = monitor.capture_performance_snapshot(
            predictions_path=predictions_path,
            model_version=current_model_version,
            dataset_hash=dataset_hash,
        )

        typer.echo("✅ Performance snapshot captured")
        typer.echo(f"  Model: {snapshot.model_version}")
        typer.echo(f"  AUROC: {snapshot.auroc:.3f}")
        typer.echo(f"  AP: {snapshot.ap:.3f}")

    elif action == "report":
        # Generate detailed monitoring report
        typer.echo(f"📋 Generating monitoring report for {days} days...")

        report = monitor.get_monitoring_report(model_version=model_version, days=days)

        if "error" in report:
            typer.echo(f"❌ {report['error']}")
            return

        # Save report to file
        report_path = Path("reports/monitoring_report.json")
        save_json(report_path, report)

        typer.echo(f"✅ Report saved to {report_path}")

        # Show key insights
        if report['snapshots_count'] > 1:
            trends = report['trends']
            typer.echo("\n📈 Performance Trends:")
            typer.echo(f"  AUROC trend: {trends['auroc_trend_per_day']:.6f} per day ({trends['trend_direction']})")
            typer.echo(f"  AP trend: {trends['ap_trend_per_day']:.6f} per day")

    elif action == "alerts":
        # Check for drift and send alerts
        typer.echo("🔍 Checking for model drift and performance issues...")

        # Get latest snapshot for drift detection
        with sqlite3.connect("reports/monitoring.db") as conn:
            cursor = conn.execute("""
                SELECT metrics_json FROM performance_snapshots
                WHERE model_version = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (model_version,))

            row = cursor.fetchone()
            if not row:
                typer.echo("❌ No performance snapshots found for drift detection")
                return

            latest_data = json.loads(row[0])
            from .monitoring import PerformanceSnapshot
            latest_snapshot = PerformanceSnapshot(**latest_data)

        # Detect drift
        alerts = monitor.detect_drift(latest_snapshot)

        if not alerts:
            typer.echo("✅ No drift or performance issues detected")
        else:
            typer.echo(f"🚨 Found {len(alerts)} alerts:")

            for alert in alerts:
                typer.echo(f"\n  Alert: {alert.alert_type.title()}")
                typer.echo(f"  Severity: {alert.severity.upper()}")
                typer.echo(f"  Message: {alert.message}")

                # Send notifications if configured
                if slack_webhook or email_to:
                    success = monitor.send_alert_notification(
                        alert,
                        webhook_url=slack_webhook,
                        email_config={"to_email": email_to} if email_to else None,
                    )
                    if success:
                        typer.echo("  ✅ Alert sent")
                    else:
                        typer.echo("  ❌ Failed to send alert")

    else:
        typer.echo(f"❌ Unknown action: {action}")
        typer.echo("Available actions: status, capture, report, alerts")


@app.command("validate-interpretability")
def validate_interpretability_cmd(
    processed_dir: Path = typer.Option(PROCESSED_DIR, dir_okay=True),
    models_dir: Path = typer.Option(MODELS_DIR, dir_okay=True),
    output_dir: Path = typer.Option(Path("reports/interpretability_validation"), dir_okay=True),
    background_sizes: str = typer.Option("50,100,200", help="Comma-separated background sizes"),
    n_bootstrap: int = typer.Option(100, help="Number of bootstrap samples"),
    perturbation_magnitude: float = typer.Option(0.1, help="Perturbation magnitude for robustness testing"),
    summary_only: bool = typer.Option(False, help="Show only summary, don't save detailed report"),
) -> None:
    """Run comprehensive interpretability validation on SHAP explanations."""
    from .interpretability_validation import InterpretabilityValidator

    # Parse background sizes
    bg_sizes = [int(x.strip()) for x in background_sizes.split(",")]

    typer.echo("🔍 Running interpretability validation...")

    validator = InterpretabilityValidator(
        background_sizes=bg_sizes,
        n_bootstrap=n_bootstrap,
        perturbation_magnitude=perturbation_magnitude,
    )

    try:
        report = validator.validate_explanations(
            processed_dir=processed_dir,
            models_dir=models_dir,
            output_dir=output_dir,
        )

        # Generate and display summary
        summary = validator.generate_validation_summary(report)

        if not summary_only:
            typer.echo("\n" + "="*50)
            typer.echo(summary)
        else:
            typer.echo("\n📊 Interpretability Validation Summary:")
            typer.echo(f"  Overall Quality Score: {report.explanation_quality.overall_quality_score:.3f}")
            typer.echo(f"  Background Consistency: {report.explanation_quality.background_consistency:.3f}")
            typer.echo(f"  Stability Score: {report.explanation_quality.stability_score:.3f}")
            typer.echo(f"  Robustness Score: {report.explanation_quality.perturbation_robustness:.3f}")

        typer.echo(f"\n✅ Validation completed successfully")

    except Exception as e:
        typer.echo(f"❌ Interpretability validation failed: {e}")
        raise typer.Exit(1)


@app.command("retrain")
def retrain_cmd(
    config: Optional[Path] = typer.Option(None, help="Retraining configuration file"),
    dry_run: bool = typer.Option(False, help="Show what would be done without executing"),
    force: bool = typer.Option(False, help="Force retraining even if no triggers are active"),
    schedule: bool = typer.Option(False, help="Run in scheduled mode (check triggers automatically)"),
) -> None:
    """Automated model retraining with intelligent triggers."""
    from .automated_retraining import retrain_command

    retrain_command(config=config, dry_run=dry_run, force=force, schedule=schedule)


@app.command("deploy")
def deploy_cmd(
    version_id: str = typer.Argument(..., help="Model version ID to deploy"),
    confirm: bool = typer.Option(True, help="Confirm deployment"),
) -> None:
    """Deploy a model version to production."""
    from .model_deployment import deploy_cmd as deploy_func
    deploy_func(version_id=version_id, confirm=confirm)


@app.command("rollback")
def rollback_cmd(
    target_version: str = typer.Argument(..., help="Target version ID to rollback to"),
    confirm: bool = typer.Option(True, help="Confirm rollback"),
) -> None:
    """Rollback production model to a previous version."""
    from .model_deployment import rollback_cmd as rollback_func
    rollback_func(target_version=target_version, confirm=confirm)


@app.command("versions")
def versions_cmd(
    details: bool = typer.Option(False, help="Show detailed performance metrics"),
) -> None:
    """List all available model versions."""
    from .model_deployment import versions_cmd as versions_func
    versions_func(details=details)


@app.command("cleanup")
def cleanup_cmd(
    keep_production: bool = typer.Option(True, help="Keep production models"),
    keep_recent: int = typer.Option(5, help="Keep N most recent models"),
    dry_run: bool = typer.Option(False, help="Show what would be deleted"),
) -> None:
    """Clean up old model versions."""
    from .model_deployment import cleanup_cmd as cleanup_func
    cleanup_func(keep_production=keep_production, keep_recent=keep_recent, dry_run=dry_run)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Start the model serving server."""
    from .model_deployment import server_cmd as server_func
    server_func(host=host, port=port, reload=reload)


@app.command("compare")
def compare_cmd(
    criteria_config: Optional[Path] = typer.Option(None, help="JSON file with comparison criteria"),
    output_dir: Path = typer.Option(Path("reports/model_comparison"), help="Output directory"),
    generate_report: bool = typer.Option(True, help="Generate detailed comparison report"),
) -> None:
    """Compare and rank models using advanced criteria."""
    from .model_comparison import compare_models_cmd
    compare_models_cmd(
        criteria_config=criteria_config,
        output_dir=output_dir,
        generate_report=generate_report
    )


@app.command("compare-interactive")
def compare_interactive_cmd(
    criteria_config: Optional[Path] = typer.Option(None, help="JSON file with comparison criteria"),
) -> None:
    """Launch interactive model comparison dashboard."""
    from .model_comparison import interactive_comparison_cmd
    interactive_comparison_cmd(criteria_config=criteria_config)


@app.command()
def all(
    seed: int = typer.Option(42, help="Pipeline seed"),
    distributed: bool = typer.Option(True, help="Use distributed computing throughout pipeline"),
    profile: Optional[str] = typer.Option(
        None, help="Parameter profile to apply (e.g. ci)", show_default=False
    ),
    fast: bool = typer.Option(False, "--fast", help="Use the CI profile for a rapid smoke run"),
    ci_mode: bool = typer.Option(False, "--ci", help="Alias for --fast to align with CI"),
) -> None:
    """Run the full end-to-end pipeline with distributed computing."""

    # Configure distributed computing for the entire pipeline
    overrides = _with_overrides(
        "all",
        profile=profile,
        fast=fast,
        ci=ci_mode,
        base={"seed": seed, "distributed": distributed},
    )
    seed = int(overrides["seed"])
    distributed = bool(overrides["distributed"])

    if distributed:
        distributed_cmd(backend="joblib", n_jobs=-1, verbose=0)

    prepare(seed=seed)
    train(seed=seed)
    eval_cmd(seed=seed, distributed=distributed)
    explain(seed=seed, distributed=distributed)
    scorecard()
    docs()
    snapshot()


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
