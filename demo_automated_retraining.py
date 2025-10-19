#!/usr/bin/env python3
"""Demonstration script for automated model retraining functionality."""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from oncotarget_lite.automated_retraining import (
    RetrainConfig,
    AutomatedRetrainingPipeline,
    run_automated_retraining
)

def demo_config():
    """Demonstrate the retraining configuration."""
    print("🔧 Retraining Configuration Demo")
    print("=" * 40)

    config = RetrainConfig()
    print(f"Performance threshold drop: {config.performance_threshold_drop}")
    print(f"Data drift threshold: {config.data_drift_threshold}")
    print(f"Schedule interval: {config.schedule_interval_days} days")
    print(f"Auto deploy improvements: {config.auto_deploy_improvements}")
    print(f"Min improvement threshold: {config.min_improvement_threshold}")
    print()

def demo_trigger_detection():
    """Demonstrate trigger detection."""
    print("🔍 Trigger Detection Demo")
    print("=" * 40)

    config = RetrainConfig()
    pipeline = AutomatedRetrainingPipeline(config)

    should_retrain, reason, trigger_type = pipeline.check_retrain_needed()

    print(f"Retraining needed: {should_retrain}")
    print(f"Reason: {reason}")
    print(f"Trigger type: {trigger_type.value}")
    print()

def demo_dry_run():
    """Demonstrate dry run functionality."""
    print("🔍 Dry Run Demo")
    print("=" * 40)

    print("Running dry run to see what would happen...")
    try:
        run_automated_retraining(dry_run=True)
    except SystemExit:
        pass  # Expected when dry run completes
    print()

def demo_config_file():
    """Demonstrate loading configuration from file."""
    print("📁 Configuration File Demo")
    print("=" * 40)

    config_path = Path("configs/retrain_config.json")
    if config_path.exists():
        print(f"Loading configuration from: {config_path}")
        try:
            run_automated_retraining(config_path=config_path, dry_run=True)
        except SystemExit:
            pass  # Expected when dry run completes
    else:
        print(f"Configuration file not found: {config_path}")
        print("Using default configuration instead.")
    print()

def main():
    """Main demonstration function."""
    print("🚀 Automated Model Retraining Demonstration")
    print("=" * 50)
    print()

    demo_config()
    demo_trigger_detection()
    demo_dry_run()
    demo_config_file()

    print("✅ Demonstration completed!")
    print()
    print("To run automated retraining:")
    print("  python -m oncotarget_lite.cli retrain --schedule")
    print("  make retrain")
    print()
    print("To force retraining:")
    print("  python -m oncotarget_lite.cli retrain --force")
    print()
    print("To rollback to previous model:")
    print("  python -m oncotarget_lite.cli rollback")
    print("  make rollback")

if __name__ == "__main__":
    main()
