"""Enhanced hyperparameter optimization with experiment tracking."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ray
import typer
from ray import tune
from ray.tune.search.optuna import OptunaSearch

from .manager import ExperimentManager
from .schemas import Experiment, ExperimentConfig, ExperimentTrial

logger = logging.getLogger(__name__)


class EnhancedHyperparameterOptimizer:
    """Enhanced hyperparameter optimization with experiment tracking."""

    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.current_experiment: Optional[str] = None

    def optimize_with_experiment(
        self,
        config: ExperimentConfig,
        data_path: Path,
        models_dir: Path = Path("models"),
        reports_dir: Path = Path("reports")
    ) -> Experiment:
        """Run hyperparameter optimization with full experiment tracking using Ray Tune."""

        # Create experiment
        experiment_id = self.experiment_manager.create_experiment(config)
        self.current_experiment = experiment_id
        experiment = self.experiment_manager.get_experiment(experiment_id)

        if not ray.is_initialized():
            ray.init()
            
        try:
            typer.echo(f"🔬 Starting experiment: {experiment_id}")
            typer.echo(f"   Model types: {', '.join(config.model_types)}")
            typer.echo(f"   Trials: {config.n_trials} | Parallel jobs: {config.parallel_jobs}")

            search_space = self._prepare_search_space(config)
            
            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(
                        _trainable, 
                        data_path=data_path, 
                        models_dir=models_dir, 
                        reports_dir=reports_dir
                    ),
                    {"cpu": 1} # Each trial requires 1 CPU
                ),
                tune_config=tune.TuneConfig(
                    metric=config.metrics[0],
                    mode=config.optimization_direction,
                    search_alg=OptunaSearch(),
                    num_samples=config.n_trials,
                    max_parallel_trials=config.parallel_jobs,
                ),
                param_space=search_space,
            )
            
            results = tuner.fit()
            
            # Process results and add to experiment manager
            for i, result in enumerate(results):
                trial_params = result.config
                trial_metrics = result.metrics
                
                trial = ExperimentTrial(
                    trial_id=f"{experiment_id}_{trial_params['model_type']}_{i}",
                    experiment_id=experiment_id,
                    parameters=trial_params,
                    metrics=trial_metrics,
                    status="completed" if not result.error else "failed"
                )
                self.experiment_manager.add_trial(experiment_id, trial)

            self.experiment_manager.complete_experiment(experiment_id)

            typer.echo(f"\n✅ Experiment completed: {experiment_id}")
            if experiment.best_trial:
                typer.echo("🏆 Best result:")
                primary_metric = config.metrics[0]
                best_value = experiment.best_trial.metrics.get(primary_metric, 0)
                typer.echo(f"   {primary_metric.upper()}: {best_value:.4f}")

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            if experiment_id in self.experiment_manager.experiments:
                self.experiment_manager.experiments[experiment_id].status = "failed"
        
        return experiment

    def _prepare_search_space(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Prepare the search space for Ray Tune from the ExperimentConfig."""
        search_space = {}
        for model_type, space in config.search_spaces.items():
            for param, definition in space.items():
                param_name = f"{model_type}_{param}"
                if definition["type"] == "loguniform":
                    search_space[param_name] = tune.loguniform(definition["low"], definition["high"])
                elif definition["type"] == "uniform":
                    search_space[param_name] = tune.uniform(definition["low"], definition["high"])
                elif definition["type"] == "int":
                    search_space[param_name] = tune.randint(definition["low"], definition["high"])
                elif definition["type"] == "categorical":
                    search_space[param_name] = tune.choice(definition["choices"])

        search_space["model_type"] = tune.choice(config.model_types)
        return search_space


def _trainable(
    config: Dict[str, Any],
    data_path: Path,
    models_dir: Path,
    reports_dir: Path,
):
    """Trainable function for Ray Tune."""
    start_time = time.time()
    model_type = config.pop("model_type")
    
    # Reconstruct model-specific params
    params = {}
    prefix = f"{model_type}_"
    for k, v in config.items():
        if k.startswith(prefix):
            params[k[len(prefix):]] = v
            
    try:
        from ..model import train_model, TrainConfig
        from ..eval import evaluate_predictions

        train_config = TrainConfig(model_type=model_type, seed=42, **params)

        train_result = train_model(
            processed_dir=data_path.parent,
            models_dir=models_dir,
            reports_dir=reports_dir,
            config=train_config
        )

        eval_result = evaluate_predictions(
            reports_dir=reports_dir,
            n_bootstrap=200,
            ci=0.95,
            bins=10,
            seed=42,
            distributed=False # Run eval in a single process
        )
        
        training_time = time.time() - start_time
        
        metrics = {
            'auroc': eval_result.metrics.auroc,
            'ap': eval_result.metrics.ap,
            'accuracy': eval_result.metrics.accuracy,
            'f1': eval_result.metrics.f1,
            'brier': eval_result.metrics.brier,
            'ece': eval_result.metrics.ece,
            'training_time': training_time
        }
        tune.report(**metrics)

    except Exception as e:
        logger.error(f"Trial failed: {e}")
        raise

