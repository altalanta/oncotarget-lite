"""Test configuration for oncotarget-lite."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src/ layout is importable during tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oncotarget_lite.config import AppConfig, load_config
from oncotarget_lite.pipeline import PipelineResult, run_pipeline


@pytest.fixture(scope="session")
def default_config() -> AppConfig:
    return load_config()


@pytest.fixture(scope="session")
def fast_config(tmp_path_factory: pytest.TempPathFactory) -> AppConfig:
    artifacts_root = tmp_path_factory.mktemp("artifacts")
    overrides = {
        "training": {
            "max_epochs": 120,
            "patience": 15,
            "seed": 1337,
            "device": "cpu",
        },
        "evaluation": {
            "bootstrap_samples": 64,
            "seed": 42,
        },
        "split": {
            "seed": 1337,
        },
        "artifacts": {
            "base_dir": str(artifacts_root),
        },
    }
    return load_config(overrides=overrides)


@pytest.fixture(scope="session")
def pipeline_result(fast_config: AppConfig) -> PipelineResult:
    return run_pipeline(fast_config, output_dir=fast_config.artifacts.base_dir)

