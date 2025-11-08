#!/usr/bin/env python3
"""
Demonstration script for running an ML experiment using Hydra for configuration.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from oncotarget_lite.experimentation import ExperimentConfig, ExperimentManager, EnhancedHyperparameterOptimizer

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the ML experiment.
    """
    print("🚀 Running ML Experiment with Validated Hydra Configuration")
    print("=" * 60)
    
    # Load and validate the configuration using Pydantic
    try:
        experiment_config = ExperimentConfig(**OmegaConf.to_container(cfg, resolve=True))
        print("✅ Configuration successfully validated.")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return

    print("\n🔬 Initialized ExperimentConfig:")
    print(f"  Name: {experiment_config.experiment_name}")
    print(f"  Model types: {', '.join(experiment_config.model.keys())}")
    print(f"  Trials: {experiment_config.n_trials}")
    print(f"  Parallel jobs: {experiment_config.parallel_jobs}")

    # Initialize and run the experiment
    print("\n⚙️ Starting ExperimentManager...")
    exp_manager = ExperimentManager()
    optimizer = EnhancedHyperparameterOptimizer(exp_manager)

    study = optimizer.optimize_with_experiment(
        config=experiment_config,
        data_path=Path("data/processed") # This could also be part of the config
    )

    print("\n✅ Experiment completed!")
    if study.best_trial:
        print(f"🏆 Best trial for {study.experiment_id}:")
        best_trial = study.best_trial
        print(f"  Value: {best_trial.metrics.get(experiment_config.metrics[0], 'N/A')}")
        print("  Params: ")
        for key, value in best_trial.parameters.items():
            print(f"    {key}: {value}")

if __name__ == "__main__":
    main()

