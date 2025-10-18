"""Automated hyperparameter optimization using Optuna."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .model import TrainConfig, train_model
from .utils import ensure_dir

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimizer for oncotarget-lite models."""

    def __init__(
        self,
        study_name: str = "oncotarget_optimization",
        storage_path: str = "sqlite:///reports/optuna_study.db",
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ):
        self.study_name = study_name
        self.storage_path = storage_path
        self.n_trials = n_trials
        self.timeout = timeout

    def define_search_space(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """Define the hyperparameter search space for different model types."""

        if model_type == "logreg":
            return {
                "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            }

        elif model_type == "xgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
            }

        elif model_type == "lgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
            }

        elif model_type == "mlp":
            return {
                "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (50, 50), (100, 50)]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
                "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
            }

        elif model_type == "transformer":
            return {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
                "num_layers": trial.suggest_int("num_layers", 2, 8),
                "num_heads": trial.suggest_categorical("num_heads", [4, 8, 16]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "max_position_embeddings": trial.suggest_categorical("max_position_embeddings", [500, 1000, 2000]),
            }

        elif model_type == "gnn":
            return {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
                "num_layers": trial.suggest_int("num_layers", 2, 6),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "num_heads": trial.suggest_categorical("num_heads", [4, 8, 16]),
            }

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def objective(
        self,
        trial: optuna.Trial,
        model_type: str,
        processed_dir: Path,
        models_dir: Path,
        reports_dir: Path,
        metric: str = "auroc",
    ) -> float:
        """Objective function for Optuna optimization."""

        # Define search space
        params = self.define_search_space(trial, model_type)

        # Create training config
        config = TrainConfig(
            model_type=model_type,
            model_params=params,
            seed=42,  # Fixed seed for reproducible optimization
        )

        try:
            # Train model
            result = train_model(
                processed_dir=processed_dir,
                models_dir=models_dir,
                reports_dir=reports_dir,
                config=config,
            )

            # Get the metric to optimize (default: test AUROC)
            if metric == "auroc":
                score = result.test_metrics["auroc"]
            elif metric == "ap":
                score = result.test_metrics["ap"]
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # Report to Optuna
            trial.set_user_attr("test_auroc", result.test_metrics["auroc"])
            trial.set_user_attr("test_ap", result.test_metrics["ap"])
            trial.set_user_attr("train_auroc", result.train_metrics["auroc"])
            trial.set_user_attr("train_ap", result.train_metrics["ap"])

            return score

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            # Return a poor score for failed trials
            return 0.5

    def optimize(
        self,
        model_type: str,
        processed_dir: Path,
        models_dir: Path,
        reports_dir: Path,
        metric: str = "auroc",
    ) -> optuna.Study:
        """Run hyperparameter optimization."""

        # Create study
        study = optuna.create_study(
            study_name=f"{self.study_name}_{model_type}",
            storage=self.storage_path,
            direction="maximize",
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(
            lambda trial: self.objective(
                trial, model_type, processed_dir, models_dir, reports_dir, metric
            ),
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        return study

    def get_best_params(self, study: optuna.Study) -> Dict[str, Any]:
        """Get the best parameters from a study."""
        return study.best_params

    def get_best_score(self, study: optuna.Study) -> float:
        """Get the best score from a study."""
        return study.best_value

    def save_study_summary(self, study: optuna.Study, output_path: Path) -> None:
        """Save a summary of the optimization study."""

        ensure_dir(output_path.parent)

        summary = {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "optimization_history": [
                {
                    "trial": t.number,
                    "score": t.value,
                    "params": t.params,
                }
                for t in study.trials
            ],
        }

        import json
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
