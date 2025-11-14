"""Manages ML experiments and trials."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import ensure_dir
from .schemas import Experiment, ExperimentConfig, ExperimentTrial

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages ML experiments and trials."""

    def __init__(self, experiments_dir: Path = Path("reports/experiments")):
        self.experiments_dir = experiments_dir
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: Dict[str, Experiment] = {}
        self._load_experiments()

    def _load_experiments(self):
        """Load existing experiments from disk."""
        if not self.experiments_dir.exists():
            return

        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                exp_file = exp_dir / "experiment.json"
                if exp_file.exists():
                    try:
                        with open(exp_file, 'r') as f:
                            exp_data = json.load(f)

                        config = ExperimentConfig(**exp_data['config'])
                        trials = [
                            ExperimentTrial(**trial_data)
                            for trial_data in exp_data['trials']
                        ]

                        experiment = Experiment(
                            experiment_id=exp_data['experiment_id'],
                            config=config,
                            trials=trials,
                            status=exp_data.get('status', 'completed'),
                            created_at=datetime.fromisoformat(exp_data['created_at']),
                            completed_at=datetime.fromisoformat(exp_data['completed_at']) if exp_data.get('completed_at') else None
                        )

                        self.experiments[experiment.experiment_id] = experiment

                    except Exception as e:
                        logger.warning(f"Error loading experiment {exp_dir.name}: {e}")

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment."""
        experiment_id = f"exp_{config.experiment_name}_{int(time.time())}"

        experiment = Experiment(
            experiment_id=experiment_id,
            config=config,
            trials=[]
        )

        self.experiments[experiment_id] = experiment
        self.active_experiments[experiment_id] = experiment

        # Save experiment metadata
        exp_dir = self.experiments_dir / experiment_id
        ensure_dir(exp_dir)

        exp_data = {
            'experiment_id': experiment_id,
            'config': asdict(config),
            'trials': [],
            'status': 'running',
            'created_at': experiment.created_at.isoformat(),
            'completed_at': None
        }

        with open(exp_dir / "experiment.json", 'w') as f:
            json.dump(exp_data, f, indent=2, default=str)

        logger.info(f"✅ Created experiment: {experiment_id}")
        return experiment_id

    def add_trial(self, experiment_id: str, trial: ExperimentTrial) -> None:
        """Add a trial to an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        trial.experiment_id = experiment_id
        experiment.trials.append(trial)

        # Update best trial if this one is better
        if experiment.best_trial is None:
            experiment.best_trial = trial
        else:
            primary_metric = experiment.config.metrics[0]
            current_best = experiment.best_trial.metrics.get(primary_metric, 0)
            new_value = trial.metrics.get(primary_metric, 0)

            if experiment.config.optimization_direction == "maximize":
                if new_value > current_best:
                    experiment.best_trial = trial
            else:
                if new_value < current_best:
                    experiment.best_trial = trial

        # Save trial data
        self._save_experiment(experiment)

    def _save_experiment(self, experiment: Experiment):
        """Save experiment data to disk."""
        exp_dir = self.experiments_dir / experiment.experiment_id

        exp_data = {
            'experiment_id': experiment.experiment_id,
            'config': asdict(experiment.config),
            'trials': [asdict(trial) for trial in experiment.trials],
            'best_trial': asdict(experiment.best_trial) if experiment.best_trial else None,
            'status': experiment.status,
            'created_at': experiment.created_at.isoformat(),
            'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None
        }

        with open(self.experiments_dir / experiment.experiment_id / "experiment.json", 'w') as f:
            json.dump(exp_data, f, indent=2, default=str)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.experiments.get(experiment_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with summary information."""
        experiments = []

        for exp_id, exp in self.experiments.items():
            experiments.append({
                'experiment_id': exp_id,
                'name': exp.config.experiment_name,
                'status': exp.status,
                'trials_count': len(exp.trials),
                'best_score': exp.best_trial.metrics.get(exp.config.metrics[0], 0) if exp.best_trial else 0,
                'created_at': exp.created_at.isoformat(),
                'model_types': exp.config.model_types
            })

        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)

    def complete_experiment(self, experiment_id: str):
        """Mark an experiment as completed."""
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            experiment.status = "completed"
            experiment.completed_at = datetime.now()

            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]

            self._save_experiment(experiment)
            logger.info(f"✅ Completed experiment: {experiment_id}")






