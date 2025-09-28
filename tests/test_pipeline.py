from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from oncotarget_lite.config import AppConfig, load_config
from oncotarget_lite.pipeline import PipelineResult, run_pipeline


def test_artifacts_written(pipeline_result: PipelineResult) -> None:
    assert pipeline_result.metrics_path.is_file()
    assert pipeline_result.predictions_path.is_file()
    assert pipeline_result.importances_path.is_file()
    assert pipeline_result.lineage_path.is_file()
    assert pipeline_result.model_path.is_file()


def test_predictions_saved(pipeline_result: PipelineResult) -> None:
    predictions = pd.read_parquet(pipeline_result.predictions_path)
    assert set(predictions["split"]) == {"train", "test"}
    assert predictions[predictions["split"] == "test"].shape[0] > 0


def test_lineage_contains_inputs(pipeline_result: PipelineResult) -> None:
    lineage = json.loads(pipeline_result.lineage_path.read_text())
    assert lineage["inputs"]
    assert lineage["params"]["config"]["training"]["max_epochs"] >= 1


def test_pipeline_deterministic(tmp_path: Path, fast_config: AppConfig) -> None:
    overrides = fast_config.model_dump(mode="json")
    overrides["artifacts"]["base_dir"] = str(tmp_path)
    cfg = load_config(overrides=overrides)
    first = run_pipeline(cfg, output_dir=tmp_path)
    second = run_pipeline(cfg, output_dir=tmp_path)
    metrics_a = json.loads(first.metrics_path.read_text())["test"]
    metrics_b = json.loads(second.metrics_path.read_text())["test"]
    assert metrics_a == metrics_b
    preds_a = pd.read_parquet(first.predictions_path).sort_values("gene")
    preds_b = pd.read_parquet(second.predictions_path).sort_values("gene")
    pd.testing.assert_series_equal(preds_a["y_prob"], preds_b["y_prob"])

