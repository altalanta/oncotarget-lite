#!/usr/bin/env python3
"""
Demonstration script for running an ML experiment using Hydra for configuration.
"""
import hydra
from omegaconf import DictConfig, OmegaConf

from oncotarget_lite.experimentation import ExperimentConfig, ExperimentManager

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the ML experiment.
    """
    print("🚀 Running ML Experiment with Hydra Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Extract model types and search spaces from the Hydra config
    model_types = list(cfg.model.keys())
    search_spaces = {model: OmegaConf.to_container(cfg.model[model], resolve=True) for model in model_types}

    # Create the experiment configuration
    experiment_config = ExperimentConfig(
        experiment_name=cfg.experiment_name,
        model_types=model_types,
        search_spaces=search_spaces,
        metrics=list(cfg.metrics),
        n_trials=cfg.n_trials,
        timeout=cfg.timeout,
        cv_folds=cfg.cv_folds,
        random_seed=cfg.random_seed,
        optimization_direction=cfg.optimization_direction,
    )

    print("\n🔬 Initialized ExperimentConfig:")
    print(f"  Name: {experiment_config.experiment_name}")
    print(f"  Model types: {', '.join(experiment_config.model_types)}")
    print(f"  Trials: {experiment_config.n_trials}")
    print(f"  Metrics: {', '.join(experiment_config.metrics)}")

    # Initialize and run the experiment
    print("\n⚙️ Starting ExperimentManager...")
    exp_manager = ExperimentManager()
    study = exp_manager.create_or_load_study(experiment_config)
    exp_manager.run_optimization(study, experiment_config)

    print("\n✅ Experiment completed!")
    print(f"🏆 Best trial for {study.study_name}:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()

