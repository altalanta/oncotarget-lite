"""Command-line interface for oncotarget-lite."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer

from .config import AppConfig, ConfigError, load_config
from .data import DataContractError, build_feature_table, load_raw_data
from .features import build_features
from .metrics import summarise_metrics
from .pipeline import PipelineError, PipelineResult, run_pipeline
from .reports import ReportError, summarise_run

app = typer.Typer(help="Lean oncology target triage toolkit")
app_app = typer.Typer(help="Streamlit app commands")
app.add_typer(app_app, name="app")

TableList = list[tuple[str, pd.DataFrame]]


def _emit(payload: dict[str, Any], tables: list[tuple[str, pd.DataFrame]] | None = None) -> None:
    if sys.stdout.isatty() and tables:
        for title, frame in tables:
            typer.echo(title)
            typer.echo(frame.to_string(index=False))
            typer.echo("")
    typer.echo(json.dumps(payload, indent=2))


def _resolve_config(
    config_file: Path | None,
    seed: int | None = None,
    device: str | None = None,
    out_dir: Path | None = None,
) -> AppConfig:
    overrides: dict[str, Any] = {}
    if seed is not None:
        overrides.setdefault("training", {})["seed"] = seed
        overrides.setdefault("split", {})["seed"] = seed
        overrides.setdefault("evaluation", {})["seed"] = seed
    if device is not None:
        overrides.setdefault("training", {})["device"] = device
    if out_dir is not None:
        overrides.setdefault("artifacts", {})["base_dir"] = str(out_dir)
    return load_config(config_file, overrides=overrides)


@app.command("system-info")
def system_info() -> None:
    """Show environment information."""
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
    }
    _emit(payload)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Print the resolved configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration only"),
    config_file: Path | None = typer.Option(None, "--config-file", path_type=Path),
) -> None:
    """Inspect or validate configuration."""

    try:
        cfg = load_config(config_file)
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = {"valid": True}
    tables: list[tuple[str, pd.DataFrame]] = []
    if show:
        payload["config"] = cfg.model_dump(mode="json")
    if sys.stdout.isatty() and show:
        tables.append(("configuration", pd.DataFrame([payload["config"]["training"]])))
    _emit(payload, tables=tables)


@app.command("validate-data")
def validate_data(
    config_file: Path | None = typer.Option(None, "--config-file", path_type=Path),
) -> None:
    """Run schema checks against cached synthetic data."""

    try:
        cfg = load_config(config_file)
        bundle = load_raw_data(cfg.data)
        merged = build_feature_table(bundle)
        features = build_features(merged)
    except (ConfigError, DataContractError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "dataset": cfg.data.dataset_name,
        "genes": int(len(features.features)),
        "features": int(features.features.shape[1]),
        "positives": int(features.labels.sum()),
        "negatives": int((~features.labels).sum()),
    }
    _emit(payload)


def _summaries_for_output(result: PipelineResult) -> tuple[dict[str, Any], TableList]:
    metrics_row = pd.DataFrame([asdict(result.training.test_metrics)])
    top_targets = (
        result.scores.head(5)
        .reset_index()
        .rename(columns={"index": "gene"})
        [["gene", "score", "rank"]]
    )
    payload = {
        "run_id": result.run_id,
        "artifacts": str(result.output_dir),
        "metrics_file": str(result.metrics_path),
        "predictions_file": str(result.predictions_path),
        "importances_file": str(result.importances_path),
        "lineage_file": str(result.lineage_path),
        "model_file": str(result.model_path),
        "test_metrics": asdict(result.training.test_metrics),
        "bootstrap": asdict(result.training.bootstrap),
    }
    tables = [
        ("test metrics", metrics_row),
        ("top targets", top_targets),
    ]
    return payload, tables


@app.command()
def train(
    config_file: Path | None = typer.Option(None, "--config-file", path_type=Path),
    out: Path | None = typer.Option(None, "--out", help="Directory for artifacts", path_type=Path),
    seed: int | None = typer.Option(None, "--seed", help="Override seed"),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device selection (auto|cpu|cuda)",
        case_sensitive=False,
    ),
) -> None:
    """Train the default CPU-friendly MLP and emit artifacts."""

    device_value = device.lower()
    if device_value not in {"auto", "cpu", "cuda"}:
        typer.secho(f"invalid device: {device}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    try:
        cfg = _resolve_config(config_file, seed=seed, device=device_value, out_dir=out)
        result = run_pipeline(cfg, output_dir=out)
    except (ConfigError, PipelineError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload, tables = _summaries_for_output(result)
    _emit(payload, tables=tables)

@app.command()
def evaluate(
    run_dir: Path = typer.Option(
        ..., "--run-dir", path_type=Path, help="Path to pipeline artifacts"
    ),
) -> None:
    """Recompute metrics from stored predictions for a given run."""

    try:
        metrics_path = run_dir / "metrics.json"
        predictions = pd.read_parquet(run_dir / "predictions.parquet")
        with metrics_path.open("r", encoding="utf-8") as handle:
            stored_metrics = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    test_preds = predictions[predictions["split"] == "test"]
    calc_metrics, _ = summarise_metrics(
        test_preds["y_true"].to_numpy(),
        test_preds["y_prob"].to_numpy(),
        bins=10,
    )
    recomputed = asdict(calc_metrics)
    payload = {
        "run_dir": str(run_dir),
        "stored_test_metrics": stored_metrics.get("test", {}),
        "recomputed_test_metrics": recomputed,
    }
    table = pd.DataFrame([recomputed])
    _emit(payload, tables=[("recomputed test metrics", table)])


@app.command()
def report(
    run_dir: Path = typer.Option(
        ..., "--run-dir", path_type=Path, help="Path to pipeline artifacts"
    ),
    top_k: int = typer.Option(5, "--top-k", min=1, max=20),
) -> None:
    """Summarise a completed run with scorecard highlights."""

    try:
        summary = summarise_run(run_dir, top_k=top_k)
    except ReportError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    tables = []
    if sys.stdout.isatty():
        top_df = pd.DataFrame(summary["top_targets"])
        tables.append(("top targets", top_df))
    _emit(summary, tables=tables)


@app_app.command("run")
def run_app(
    port: int = typer.Option(8501, "--port", help="Port for Streamlit"),
) -> None:
    """Launch the Streamlit explorer."""

    script = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    cmd = [
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    typer.echo("Launching Streamlit app... (Ctrl+C to exit)")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=exc.returncode) from exc


def main() -> None:  # pragma: no cover - entry point
    app()
