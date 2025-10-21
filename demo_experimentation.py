#!/usr/bin/env python3
"""Demonstration script for advanced ML experimentation platform."""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from oncotarget_lite.experimentation import (
    ExperimentConfig,
    ExperimentManager,
    ExperimentDashboard
)

def demo_experiment_config():
    """Demonstrate experiment configuration capabilities."""
    print("⚙️ Experiment Configuration Demo")
    print("=" * 40)

    # Basic experiment configuration
    basic_config = ExperimentConfig(
        experiment_name="baseline_optimization",
        model_types=["logreg", "xgb"],
        search_spaces={
            "logreg": {
                "C": {"type": "loguniform", "low": 1e-4, "high": 1e2},
                "max_iter": {"type": "int", "low": 100, "high": 1000}
            },
            "xgb": {
                "n_estimators": {"type": "int", "low": 50, "high": 200},
                "max_depth": {"type": "int", "low": 3, "high": 8}
            }
        },
        metrics=["auroc", "ap", "accuracy"],
        n_trials=50
    )

    print("Basic Experiment Configuration:")
    print(f"  Name: {basic_config.experiment_name}")
    print(f"  Model types: {', '.join(basic_config.model_types)}")
    print(f"  Trials: {basic_config.n_trials}")
    print(f"  Metrics: {', '.join(basic_config.metrics)}")
    print()

    # Advanced experiment configuration
    advanced_config = ExperimentConfig(
        experiment_name="comprehensive_optimization",
        model_types=["logreg", "xgb", "lgb"],
        search_spaces={
            "logreg": {
                "C": {"type": "loguniform", "low": 1e-4, "high": 1e3},
                "max_iter": {"type": "int", "low": 100, "high": 2000},
                "class_weight": {"type": "categorical", "choices": ["balanced", None]}
            },
            "xgb": {
                "n_estimators": {"type": "int", "low": 50, "high": 1000},
                "max_depth": {"type": "int", "low": 3, "high": 12},
                "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 0.5},
                "subsample": {"type": "uniform", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "uniform", "low": 0.5, "high": 1.0}
            },
            "lgb": {
                "n_estimators": {"type": "int", "low": 50, "high": 1000},
                "max_depth": {"type": "int", "low": 3, "high": 12},
                "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 0.5},
                "subsample": {"type": "uniform", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "uniform", "low": 0.5, "high": 1.0}
            }
        },
        metrics=["auroc", "ap", "accuracy", "f1", "brier", "ece"],
        n_trials=200,
        timeout=3600  # 1 hour timeout
    )

    print("Advanced Experiment Configuration:")
    print(f"  Name: {advanced_config.experiment_name}")
    print(f"  Model types: {', '.join(advanced_config.model_types)}")
    print(f"  Trials: {advanced_config.n_trials}")
    print(f"  Timeout: {advanced_config.timeout}s")
    print()

def demo_experiment_management():
    """Demonstrate experiment management capabilities."""
    print("📋 Experiment Management Demo")
    print("=" * 40)

    # Initialize experiment manager
    exp_manager = ExperimentManager()

    print("Experiment Management Features:")
    print("• Create structured experiments with multiple model types")
    print("• Track all optimization trials and their results")
    print("• Automatically identify best parameter combinations")
    print("• Persistent storage for reproducibility")
    print("• Integration with MLflow for comprehensive tracking")
    print()

    # Show supported search space types
    print("Supported Search Space Types:")
    print("• Uniform: Continuous uniform distribution")
    print("• Loguniform: Log-uniform for scale parameters")
    print("• Int: Integer uniform distribution")
    print("• Categorical: Discrete choices")
    print("• Normal: Normal distribution")
    print()

def demo_trial_tracking():
    """Demonstrate trial tracking and analysis."""
    print("🔬 Trial Tracking Demo")
    print("=" * 40)

    print("Trial Tracking Features:")
    print("• Complete parameter tracking for each trial")
    print("• Comprehensive performance metrics")
    print("• Training time and resource usage")
    print("• Success/failure status with error details")
    print("• Automatic best trial identification")
    print()

    print("Example Trial Data Structure:")
    print("  Trial ID: exp_001_logreg_trial_15")
    print("  Parameters: {'C': 0.5, 'max_iter': 500, 'model_type': 'logreg'}")
    print("  Metrics: {'auroc': 0.85, 'ap': 0.78, 'accuracy': 0.82}")
    print("  Training time: 45.2s")
    print("  Status: completed")
    print()

def demo_visualization_capabilities():
    """Demonstrate visualization capabilities."""
    print("📊 Visualization Capabilities Demo")
    print("=" * 40)

    print("Experiment Visualizations:")
    print("• Trial progression plots showing improvement over time")
    print("• Experiment comparison charts for multiple experiments")
    print("• Parameter importance analysis")
    print("• Performance distribution analysis")
    print()

    print("Interactive Dashboards:")
    print("• Real-time experiment monitoring")
    print("• Trial-by-trial performance tracking")
    print("• Side-by-side experiment comparison")
    print("• Export capabilities for presentations")
    print()

def demo_integration_benefits():
    """Demonstrate integration benefits."""
    print("🔗 Integration Benefits Demo")
    print("=" * 40)

    print("Experimentation Platform Integration:")
    print("• Automatic model registration with versioning")
    print("• Seamless integration with automated retraining")
    print("• Quality monitoring for experiment data")
    print("• MLflow integration for comprehensive tracking")
    print("• Model serving for experiment results")
    print()

    print("Complete ML Workflow:")
    print("1. 🔄 Data Quality → Validate data before experiments")
    print("2. 🔬 Experimentation → Run systematic optimization")
    print("3. 📊 Model Comparison → Select best models")
    print("4. 🚀 Model Deployment → Deploy optimized models")
    print("5. 🌐 Model Serving → Serve predictions")
    print("6. 📈 Monitoring → Track performance and trigger retraining")
    print()

def main():
    """Main demonstration function."""
    print("🚀 Advanced ML Experimentation Platform Demonstration")
    print("=" * 60)
    print()

    demo_experiment_config()
    demo_experiment_management()
    demo_trial_tracking()
    demo_visualization_capabilities()
    demo_integration_benefits()

    print("✅ Experimentation Platform Demonstration completed!")
    print()
    print("Next steps:")
    print("1. Run 'make experiment' to start an optimization experiment")
    print("2. Run 'make experiments' to view experiment history")
    print("3. Run 'make experiment-report' for detailed analysis")
    print("4. Use custom configurations in configs/experiment_config.json")

if __name__ == "__main__":
    main()
