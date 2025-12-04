"""Centralized application settings with environment-based configuration.

This module provides a single source of truth for all application configuration,
using Pydantic Settings for type validation and environment variable support.

Usage:
    from oncotarget_lite.settings import get_settings

    settings = get_settings()
    print(settings.model_path)
    print(settings.api.host)

Environment variables override defaults using the ONCOTARGET_ prefix:
    ONCOTARGET_DEBUG=true
    ONCOTARGET_MODEL_PATH=/custom/path
    ONCOTARGET_API__HOST=0.0.0.0  (nested with double underscore)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ONCOTARGET_API_",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=4, ge=1, description="Number of workers")
    reload: bool = Field(default=False, description="Auto-reload for development")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )
    request_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Request timeout in seconds",
    )
    prediction_timeout: float = Field(
        default=10.0,
        ge=1.0,
        description="ML prediction timeout in seconds",
    )


class ModelSettings(BaseSettings):
    """ML model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ONCOTARGET_MODEL_",
        extra="ignore",
    )

    path: Path = Field(
        default=Path("models/logreg_pipeline.pkl"),
        description="Path to trained model",
    )
    features_path: Path = Field(
        default=Path("models/feature_list.json"),
        description="Path to feature list JSON",
    )
    version: str = Field(default="latest", description="Model version identifier")
    cache_predictions: bool = Field(
        default=True,
        description="Cache predictions for identical inputs",
    )
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="Cache TTL in seconds",
    )


class DataSettings(BaseSettings):
    """Data processing configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ONCOTARGET_DATA_",
        extra="ignore",
    )

    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Processed data directory",
    )
    cache_dir: Path = Field(
        default=Path("data/cache"),
        description="Cache directory",
    )
    test_size: float = Field(
        default=0.3,
        ge=0.1,
        le=0.5,
        description="Test split fraction",
    )
    use_optimized_processing: bool = Field(
        default=True,
        description="Use optimized data processing with Polars",
    )
    use_scalable_processing: bool = Field(
        default=True,
        description="Use scalable parallel data processing",
    )


class TrainingSettings(BaseSettings):
    """Model training configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ONCOTARGET_TRAIN_",
        extra="ignore",
    )

    seed: int = Field(default=42, description="Random seed for reproducibility")
    model_type: str = Field(
        default="logreg",
        description="Model type: logreg, xgb, lgb, mlp, transformer, gnn",
    )
    n_bootstrap: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Bootstrap samples for confidence intervals",
    )
    distributed: bool = Field(
        default=True,
        description="Use distributed computing",
    )
    n_jobs: int = Field(
        default=-1,
        ge=-1,
        description="Parallel jobs (-1 for all cores)",
    )


class MonitoringSettings(BaseSettings):
    """Observability and monitoring configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ONCOTARGET_MONITORING_",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable monitoring")
    db_path: Path = Field(
        default=Path("reports/monitoring.db"),
        description="Monitoring database path",
    )
    metrics_port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Prometheus metrics port",
    )
    drift_threshold: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Drift detection threshold",
    )
    alert_on_drift: bool = Field(default=True, description="Send alerts on drift")
    slack_webhook: str | None = Field(
        default=None,
        description="Slack webhook URL for alerts",
    )


class ResilienceSettings(BaseSettings):
    """Resilience and fault tolerance configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ONCOTARGET_RESILIENCE_",
        extra="ignore",
    )

    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts",
    )
    retry_backoff: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff factor",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Failures before circuit opens",
    )
    circuit_breaker_timeout: int = Field(
        default=30,
        ge=5,
        description="Circuit breaker recovery timeout (seconds)",
    )


class Settings(BaseSettings):
    """Main application settings aggregating all sub-configurations."""

    model_config = SettingsConfigDict(
        env_prefix="ONCOTARGET_",
        env_nested_delimiter="__",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Environment and debug settings
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    json_logs: bool = Field(
        default=False,
        description="Use JSON format for logs (production)",
    )

    # Nested configuration sections
    api: APISettings = Field(default_factory=APISettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Normalize environment name."""
        return v.lower()

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"

    def configure_for_ci(self) -> "Settings":
        """Return settings optimized for CI/CD pipelines."""
        # In CI, use faster settings
        self.training.n_bootstrap = 100
        self.training.distributed = False
        self.data.use_optimized_processing = False
        self.debug = False
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the application settings singleton.

    Settings are loaded once and cached. Environment variables
    with ONCOTARGET_ prefix override defaults.

    Returns:
        The application settings instance.
    """
    return Settings()


def get_settings_for_testing() -> Settings:
    """
    Get fresh settings for testing (bypasses cache).

    Use this in tests to ensure clean settings.

    Returns:
        A new Settings instance.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (useful for testing)."""
    get_settings.cache_clear()


# Convenience exports for common settings access
__all__ = [
    "Settings",
    "APISettings",
    "ModelSettings",
    "DataSettings",
    "TrainingSettings",
    "MonitoringSettings",
    "ResilienceSettings",
    "get_settings",
    "get_settings_for_testing",
    "reset_settings_cache",
]

