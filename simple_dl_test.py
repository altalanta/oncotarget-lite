#!/usr/bin/env python3
"""
Simple test for deep learning models without full data preparation.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

from oncotarget_lite.model import TrainConfig, train_model


def create_simple_test_data():
    """Create simple test data for model testing."""
    np.random.seed(42)

    # Create 100 samples with 20 features each
    n_samples = 100
    n_features = 20

    # Generate random features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
        index=[f"gene_{i}" for i in range(n_samples)]
    )

    # Generate binary labels (some correlation with features)
    labels = pd.DataFrame({
        "gene": features.index,
        "label": (features.iloc[:, 0] + features.iloc[:, 1] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    }).set_index("gene")

    # Create train/test split
    train_genes = features.index[:70].tolist()
    test_genes = features.index[70:].tolist()

    splits = {
        "train_genes": train_genes,
        "test_genes": test_genes,
        "dataset_hash": "test_hash_123"
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


def test_deep_learning_models():
    """Test the new deep learning models with simple data."""

    print("🚀 Testing Deep Learning Models with Simple Data")
    print("=" * 50)

    # Create test data
    print("📊 Creating test data...")
    test_dir = create_simple_test_data()
    print(f"✅ Test data created in {test_dir}")

    # Test Transformer model
    print("\n🧠 Testing Transformer model...")
    try:
        config = TrainConfig(
            model_type="transformer",
            model_params={
                "hidden_dim": 32,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
            },
            seed=42
        )

        result = train_model(
            processed_dir=test_dir,
            models_dir=Path("test_models"),
            reports_dir=Path("test_reports"),
            config=config
        )

        print("✅ Transformer model trained successfully!")
        print(f"   Train AUROC: {result.train_metrics['auroc']:.3f}")
        print(f"   Test AUROC: {result.test_metrics['auroc']:.3f}")
        print(f"   Train AP: {result.train_metrics['ap']:.3f}")
        print(f"   Test AP: {result.test_metrics['ap']:.3f}")

    except Exception as e:
        print(f"❌ Failed to train Transformer model: {e}")
        import traceback
        traceback.print_exc()

    # Test GNN model
    print("\n🔗 Testing GNN model...")
    try:
        config = TrainConfig(
            model_type="gnn",
            model_params={
                "hidden_dim": 32,
                "num_layers": 2,
                "dropout": 0.2,
                "num_heads": 4,
            },
            seed=42
        )

        result = train_model(
            processed_dir=test_dir,
            models_dir=Path("test_models"),
            reports_dir=Path("test_reports"),
            config=config
        )

        print("✅ GNN model trained successfully!")
        print(f"   Train AUROC: {result.train_metrics['auroc']:.3f}")
        print(f"   Test AUROC: {result.test_metrics['auroc']:.3f}")
        print(f"   Train AP: {result.train_metrics['ap']:.3f}")
        print(f"   Test AP: {result.test_metrics['ap']:.3f}")

    except Exception as e:
        print(f"❌ Failed to train GNN model: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Deep learning model testing completed!")
    print("Check the test_models/ and test_reports/ directories for outputs.")


if __name__ == "__main__":
    test_deep_learning_models()
