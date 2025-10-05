"""Ablation experiment utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .trainers.base import TrainerConfig
from .trainers import LogisticRegressionTrainer, MLPTrainer, XGBTrainer


def load_ablation_config(config_path: Path) -> TrainerConfig:
    """Load ablation configuration from YAML file."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    return TrainerConfig(
        name=config_data["name"],
        model_type=config_data["model"]["type"],
        model_params=config_data["model"]["params"],
        feature_type=config_data["features"]["type"],
        feature_includes=config_data["features"].get("includes"),
        seed=config_data["training"]["seed"],
        test_size=config_data["training"]["test_size"],
    )


def create_trainer(config: TrainerConfig):
    """Create trainer instance from config."""
    if config.model_type == "logreg":
        return LogisticRegressionTrainer(config)
    elif config.model_type == "mlp":
        return MLPTrainer(config)
    elif config.model_type == "xgb":
        return XGBTrainer(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def discover_ablation_configs(configs_dir: Path = Path("configs/ablations")) -> list[Path]:
    """Discover all ablation config files."""
    if not configs_dir.exists():
        return []
    
    return list(configs_dir.glob("*.yaml"))


def run_ablation_experiment(
    config_path: Path,
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    config = load_ablation_config(config_path)
    trainer = create_trainer(config)
    
    # Create experiment-specific directories
    exp_name = config_path.stem
    exp_models_dir = models_dir / "ablations" / exp_name
    exp_reports_dir = reports_dir / "ablations" / exp_name
    
    exp_models_dir.mkdir(parents=True, exist_ok=True)
    exp_reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    result = trainer.train(processed_dir, exp_models_dir, exp_reports_dir)
    
    # Save results
    result_data = {
        "experiment": exp_name,
        "config": {
            "name": config.name,
            "model_type": config.model_type,
            "model_params": config.model_params,
            "feature_type": config.feature_type,
            "feature_includes": config.feature_includes,
        },
        "train_metrics": result.train_metrics,
        "test_metrics": result.test_metrics,
        "feature_count": len(result.feature_names),
        "feature_names": result.feature_names,
    }
    
    # Save pipeline
    import joblib
    joblib.dump(result.pipeline, exp_models_dir / "pipeline.pkl")
    
    # Save experiment metadata
    with open(exp_reports_dir / "metadata.json", "w") as f:
        json.dump(result_data, f, indent=2)
    
    return result_data