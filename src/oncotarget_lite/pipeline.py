"""High-level pipeline orchestration."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from . import __version__
from .artifacts import write_feature_importances, write_metrics, write_predictions
from .config import AppConfig, dump_config
from .data import DataContractError, build_feature_table, load_raw_data
from .features import build_features
from .lineage import build_lineage, write_lineage
from .logging import LogContext, configure_logging, log_environment
from .scoring import score_targets
from .trainer import TrainingArtifacts, train_pipeline
from .utils import ensure_dir, env_flag

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency
    mlflow = None


@dataclass(frozen=True)
class PipelineResult:
    config: AppConfig
    run_id: str
    output_dir: Path
    scores: pd.DataFrame
    training: TrainingArtifacts
    metrics_path: Path
    predictions_path: Path
    importances_path: Path
    lineage_path: Path
    model_path: Path


class PipelineError(RuntimeError):
    """Raised when pipeline execution fails."""


def _resolve_output_dir(base: Path, run_id: str) -> Path:
    return ensure_dir(base / run_id)


def _save_model(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _gather_inputs(raw_dir: Path) -> Iterable[Path]:
    return sorted(raw_dir.glob("*.csv"))


def run_pipeline(config: AppConfig, output_dir: Path | None = None) -> PipelineResult:
    """Execute the full pipeline and persist artifacts."""

    run_id = uuid.uuid4().hex[:8]
    lineage_id = run_id
    base_dir = output_dir or config.artifacts.base_dir
    run_dir = _resolve_output_dir(base_dir, run_id)

    context = LogContext(run_id=run_id, lineage_id=lineage_id)
    configure_logging(config.logging.level, context, json_logs=config.logging.json_logs)
    logger = logging.getLogger(__name__)
    log_environment(logger)
    logger.info("starting pipeline", extra={"extra_fields": {"version": __version__}})

    try:
        bundle = load_raw_data(config.data)
        merged = build_feature_table(bundle)
        feature_set = build_features(merged)
        scores = score_targets(feature_set.features, feature_set.labels)
        training = train_pipeline(
            feature_set,
            split_cfg=config.split,
            train_cfg=config.training,
            eval_cfg=config.evaluation,
        )
    except (DataContractError, RuntimeError) as exc:
        logger.error("pipeline failed", extra={"extra_fields": {"error": str(exc)}})
        raise PipelineError(str(exc)) from exc

    metrics_path = run_dir / config.artifacts.metrics_filename
    predictions_path = run_dir / config.artifacts.predictions_filename
    importances_path = run_dir / config.artifacts.importances_filename
    lineage_path = run_dir / config.artifacts.lineage_filename
    model_path = run_dir / "model.pt"
    scores_path = run_dir / "scores.parquet"

    write_metrics(
        metrics_path,
        training.train_metrics,
        training.test_metrics,
        training.bootstrap,
        training.calibration,
    )
    write_predictions(predictions_path, training.train_predictions, training.test_predictions)
    write_feature_importances(importances_path, training.feature_importances)
    scores.reset_index().rename(columns={"index": "gene"}).to_parquet(scores_path, index=False)
    _save_model(training.model, model_path)

    lineage_payload = build_lineage(
        inputs=_gather_inputs(config.data.raw_dir),
        params={
            "config": dump_config(config),
            "version": __version__,
        },
        context=context,
    )
    write_lineage(lineage_payload, lineage_path)

    if env_flag("ONCOTARGET_LITE_MLFLOW", False) and mlflow is not None:  # pragma: no cover
        try:
            with mlflow.start_run(run_name=f"oncotarget-lite-{run_id}"):
                mlflow.log_params({"version": __version__, "run_id": run_id})
                mlflow.log_artifact(str(metrics_path))
                mlflow.log_artifact(str(predictions_path))
                mlflow.log_artifact(str(importances_path))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "mlflow logging failed", extra={"extra_fields": {"error": str(exc)}}
            )

    return PipelineResult(
        config=config,
        run_id=run_id,
        output_dir=run_dir,
        scores=scores,
        training=training,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        importances_path=importances_path,
        lineage_path=lineage_path,
        model_path=model_path,
    )
