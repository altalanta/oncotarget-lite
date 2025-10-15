#!/usr/bin/env python3
"""
Demo script for testing the new deep learning models (Transformer and GNN).
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oncotarget_lite.model import TrainConfig, train_model
from oncotarget_lite.data import prepare_dataset


def test_deep_learning_models():
    """Test the new deep learning models."""

    print("🚀 Testing Deep Learning Models for oncotarget-lite")
    print("=" * 50)

    # First prepare the data
    print("📊 Preparing dataset...")
    try:
        prepared = prepare_dataset(
            raw_dir=Path("data/raw"),
            processed_dir=Path("data/processed"),
            test_size=0.3,
            seed=42
        )
        print(f"✅ Dataset prepared. Hash: {prepared.dataset_fingerprint}")
    except Exception as e:
        print(f"❌ Failed to prepare dataset: {e}")
        return

    # Test Transformer model
    print("\n🧠 Testing Transformer model...")
    try:
        config = TrainConfig(
            model_type="transformer",
            model_params={
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
            },
            seed=42
        )

        result = train_model(
            processed_dir=Path("data/processed"),
            models_dir=Path("models"),
            reports_dir=Path("reports"),
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
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "num_heads": 4,
            },
            seed=42
        )

        result = train_model(
            processed_dir=Path("data/processed"),
            models_dir=Path("models"),
            reports_dir=Path("reports"),
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
    print("Check the models/ and reports/ directories for outputs.")


if __name__ == "__main__":
    test_deep_learning_models()
