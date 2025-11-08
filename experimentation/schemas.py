"""Schemas for ML experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from ...config import AppConfig


class ExperimentConfig(AppConfig):
    """
    Configuration for ML experiments, inheriting from the base AppConfig.
    Pydantic will automatically validate the loaded Hydra config against this schema.
    """
    pass


@dataclass
class ExperimentTrial:
    """Represents a single trial in an experiment."""
    trial_id: str
    experiment_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    model_path: Optional[Path] = None
    training_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "completed"


@dataclass
class Experiment:
    """Complete ML experiment with multiple trials."""
    experiment_id: str
    config: ExperimentConfig
    trials: List[ExperimentTrial]
    best_trial: Optional[ExperimentTrial] = None
    status: str = "running"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
