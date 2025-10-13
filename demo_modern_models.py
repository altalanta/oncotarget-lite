#!/usr/bin/env python3
"""
Demonstration of Modern Model Architecture Expansion for oncotarget-lite.

This script shows how to use the newly added model architectures:
- XGBoost (already existed)
- LightGBM (newly added)
- MLP (already existed)

Run with: python demo_modern_models.py
"""

import json
from pathlib import Path

def main():
    print("🚀 oncotarget-lite Modern Model Architecture Expansion Demo")
    print("=" * 60)

    # 1. Show available model types
    print("\n📊 Available Model Types:")
    model_types = ["logreg", "xgb", "lgb", "mlp"]
    for i, model_type in enumerate(model_types, 1):
        print(f"  {i}. {model_type}")

    # 2. Show ablation configs
    print("\n🔧 Available Ablation Configurations:")
    configs_dir = Path("configs/ablations")
    if configs_dir.exists():
        configs = list(configs_dir.glob("*.yaml"))
        for config in sorted(configs):
            print(f"  • {config.stem}")
    else:
        print("  No ablation configs found")

    # 3. Show model-specific configs
    print("\n🎯 Model-Specific Ablation Configurations:")
    model_configs = {
        "logreg": "logreg.yaml",
        "xgb": "xgb.yaml",
        "lgb": "lgb.yaml",  # New!
        "mlp": "mlp.yaml"
    }

    for model_type, config_file in model_configs.items():
        config_path = configs_dir / config_file
        if config_path.exists():
            print(f"  ✅ {model_type}: {config_file}")
        else:
            print(f"  ❌ {model_type}: {config_file} (missing)")

    # 4. Show installation requirements
    print("\n📦 Installation Requirements:")
    print("  For XGBoost: pip install xgboost")
    print("  For LightGBM: pip install lightgbm")
    print("  For all modern models: pip install oncotarget-lite[modern_models]")

    # 5. Show usage examples
    print("\n💡 Usage Examples:")
    print("\n  # Train with different model types:")
    print("  python -m oncotarget_lite.cli train --model-type xgb")
    print("  python -m oncotarget_lite.cli train --model-type lgb")
    print("  python -m oncotarget_lite.cli train --model-type mlp")

    print("\n  # Run all ablation experiments:")
    print("  python -m oncotarget_lite.cli ablations --all-ablations")

    print("\n  # Run specific model ablation:")
    print("  python -m oncotarget_lite.cli train --config configs/ablations/lgb.yaml")

    print("\n✅ Modern Model Architecture Expansion Complete!")
    print("\nThe system now supports multiple state-of-the-art models while")
    print("maintaining full interpretability through SHAP explanations,")
    print("comprehensive evaluation metrics, and ablation studies.")

if __name__ == "__main__":
    main()
