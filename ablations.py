"""Ablation experiment utilities."""

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Any, List, Optional

import yaml

from .distributed import ablation_parallel, configure_distributed
from .trainers.base import TrainerConfig
from .trainers import get_trainer


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
    return get_trainer(config.model_type, config)


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
    distributed: bool = True,
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
    joblib.dump(result.pipeline, exp_models_dir / "pipeline.pkl")
    
    # Save experiment metadata
    with open(exp_reports_dir / "metadata.json", "w") as f:
        json.dump(result_data, f, indent=2)
    
    return result_data


def run_all_ablation_experiments(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    distributed: bool = True,
    n_jobs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run all ablation experiments in parallel.

    Args:
        processed_dir: Processed data directory
        models_dir: Models output directory
        reports_dir: Reports output directory
        distributed: Whether to use distributed computing
        n_jobs: Number of parallel jobs (uses all cores if None)

    Returns:
        List of experiment results
    """
    # Configure distributed computing if enabled
    if distributed:
        configure_distributed(backend='joblib', n_jobs=n_jobs or -1, verbose=1)

    # Discover all ablation configs
    config_paths = discover_ablation_configs()

    if not config_paths:
        print("No ablation configs found in configs/ablations/")
        return []

    print(f"Running {len(config_paths)} ablation experiments in parallel...")

    # Run experiments in parallel
    results = ablation_parallel(
        ablation_configs=config_paths,
        run_experiment_fn=run_ablation_experiment,
        processed_dir=processed_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        n_jobs=n_jobs,
    )

    return results