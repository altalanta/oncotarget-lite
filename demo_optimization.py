#!/usr/bin/env python3
"""
Demo script for testing the new hyperparameter optimization functionality.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from oncotarget_lite.optimizers import HyperparameterOptimizer


def create_simple_test_data():
    """Create simple test data for optimization testing."""
    np.random.seed(42)

    # Create 200 samples with 10 features each
    n_samples = 200
    n_features = 10

    # Generate random features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
        index=[f"gene_{i}" for i in range(n_samples)]
    )

    # Generate binary labels (some correlation with features)
    labels = pd.DataFrame({
        "gene": features.index,
        "label": (features.iloc[:, 0] + features.iloc[:, 1] + np.random.randn(n_samples) * 0.3 > 0).astype(int)
    }).set_index("gene")

    # Create train/test split
    train_genes = features.index[:140].tolist()
    test_genes = features.index[140:].tolist()

    splits = {
        "train_genes": train_genes,
        "test_genes": test_genes,
        "dataset_hash": "test_hash_456"
    }

    # Save test data
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)

    features.to_parquet(test_dir / "features.parquet")
    labels.to_parquet(test_dir / "labels.parquet")

    import json
    with open(test_dir / "splits.json", "w") as f:
        json.dump(splits, f)

    return test_dir


def test_hyperparameter_optimization():
    """Test the hyperparameter optimization functionality."""

    print("🔍 Testing Hyperparameter Optimization for oncotarget-lite")
    print("=" * 60)

    # Create test data
    print("📊 Creating test data...")
    test_dir = create_simple_test_data()
    print(f"✅ Test data created in {test_dir}")

    # Test optimization for different model types
    model_types = ["logreg", "xgb"]

    for model_type in model_types:
        print(f"\n🚀 Testing optimization for {model_type.upper()} model...")

        try:
            # Create optimizer with fewer trials for demo
            optimizer = HyperparameterOptimizer(
                study_name=f"demo_{model_type}",
                storage_path="sqlite:///test_optimization.db",
                n_trials=10,  # Small number for demo
            )

            # Run optimization
            study = optimizer.optimize(
                model_type=model_type,
                processed_dir=test_dir,
                models_dir=Path("test_models"),
                reports_dir=Path("test_reports"),
                metric="auroc",
            )

            # Save results
            summary_path = Path("test_reports") / f"optuna_summary_{model_type}.json"
            optimizer.save_study_summary(study, summary_path)

            print("✅ Optimization completed successfully!")
            print(f"   📊 Best AUROC: {study.best_value:.4f}")
            print(f"   🏆 Best parameters: {study.best_params}")
            print(f"   📁 Results saved to: {summary_path}")

        except Exception as e:
            print(f"❌ Failed to optimize {model_type} model: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 Hyperparameter optimization demo completed!")
    print("Check the test_reports/ directory for optimization results.")


if __name__ == "__main__":
    test_hyperparameter_optimization()
