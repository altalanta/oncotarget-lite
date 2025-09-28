"""Configuration models and helpers for oncotarget-lite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_RAW = _PROJECT_ROOT / "data" / "raw"
_DEFAULT_ARTIFACTS = _PROJECT_ROOT / "artifacts"


class DataConfig(BaseModel):
    """Data location and schema configuration."""

    raw_dir: Path = Field(default=_DEFAULT_DATA_RAW)
    dataset_name: str = Field(default="synthetic", min_length=3)

    @field_validator("raw_dir")
    @classmethod
    def _ensure_exists(cls, value: Path) -> Path:
        if not value.exists():
            msg = f"data directory not found: {value}"
            raise ValueError(msg)
        return value


class SplitConfig(BaseModel):
    """Train/test split settings."""

    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    seed: int = Field(default=42, ge=0)
    stratify: bool = Field(default=True)


class TrainingConfig(BaseModel):
    """MLP training hyperparameters."""

    max_epochs: int = Field(default=200, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    hidden_dims: list[int] = Field(default_factory=lambda: [64, 32])
    dropout: float = Field(default=0.15, ge=0.0, lt=1.0)
    batch_size: int = Field(default=16, gt=0)
    patience: int = Field(default=25, ge=1)
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")


class EvaluationConfig(BaseModel):
    """Evaluation, calibration, and bootstrap settings."""

    bootstrap_samples: int = Field(default=256, ge=10, le=2000)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    calibration_bins: int = Field(default=10, ge=3, le=50)
    seed: int = Field(default=7, ge=0)


class ArtifactConfig(BaseModel):
    """Artifact and lineage output configuration."""

    base_dir: Path = Field(default=_DEFAULT_ARTIFACTS)
    lineage_filename: str = Field(default="lineage.json")
    metrics_filename: str = Field(default="metrics.json")
    predictions_filename: str = Field(default="predictions.parquet")
    importances_filename: str = Field(default="feature_importances.parquet")

    @field_validator("base_dir")
    @classmethod
    def _normalize(cls, value: Path) -> Path:
        return value.resolve()


class LoggingConfig(BaseModel):
    """Logging behaviour toggles."""

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    json_logs: bool = Field(default=True)


class AppConfig(BaseModel):
    """Top-level configuration for the toolkit."""

    data: DataConfig = Field(default_factory=DataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }


class ConfigError(RuntimeError):
    """Raised when configuration loading fails."""


def _load_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"config file not found: {path}"
        raise ConfigError(msg)
    try:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}
        if not isinstance(data, dict):
            msg = "configuration root must be a mapping"
            raise ConfigError(msg)
        return data
    except yaml.YAMLError as exc:  # pragma: no cover - depends on parser internals
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc


def _deep_update(original: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(original)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load configuration from disk (YAML) and apply optional overrides."""

    base_payload = AppConfig().model_dump(mode="python")
    file_payload: dict[str, Any] = {}
    if path is not None:
        file_payload = _load_file(path)
    merged = _deep_update(base_payload, file_payload)
    if overrides:
        merged = _deep_update(merged, overrides)
    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def dump_config(config: AppConfig) -> dict[str, Any]:
    """Return a JSON-serialisable representation of the config."""

    return config.model_dump(mode="json")

