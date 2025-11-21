"""Centralized Pydantic models for configuration validation."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SearchSpaceItem(BaseModel):
    """Defines the structure for a single hyperparameter search space."""
    type: Literal["loguniform", "uniform", "int", "categorical"]
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None


class ModelConfig(BaseModel):
    """A flexible model to hold search space definitions for any model."""
    class Config:
        extra = "allow"


class HydraConfig(BaseModel):
    """Defines the structure for Hydra's internal configuration."""
    run: Dict[str, Any]
    sweep: Dict[str, Any]


class AppConfig(BaseModel):
    """The main application configuration model."""
    experiment_name: str
    metrics: List[str]
    optimization_direction: Literal["maximize", "minimize"]
    n_trials: int
    timeout: Optional[int]
    cv_folds: int
    random_seed: int
    parallel_jobs: int
    model: Dict[str, ModelConfig]
    hydra: HydraConfig










