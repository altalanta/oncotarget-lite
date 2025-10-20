#!/usr/bin/env python3
"""Test script for automated retraining functionality."""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from oncotarget_lite.automated_retraining import RetrainConfig, AutomatedRetrainingPipeline

def test_retraining_config():
    """Test that the retraining configuration works."""
    config = RetrainConfig()
    print("✅ RetrainConfig created successfully")
    print(f"   Performance threshold: {config.performance_threshold_drop}")
    print(f"   Schedule interval: {config.schedule_interval_days} days")

def test_retraining_pipeline():
    """Test that the retraining pipeline can be initialized."""
    config = RetrainConfig()
    pipeline = AutomatedRetrainingPipeline(config)
    print("✅ AutomatedRetrainingPipeline created successfully")

    # Test trigger checking
    should_retrain, reason, trigger_type = pipeline.check_retrain_needed()
    print(f"   Retrain needed: {should_retrain}")
    print(f"   Reason: {reason}")
    print(f"   Trigger: {trigger_type.value}")

if __name__ == "__main__":
    print("🧪 Testing automated retraining functionality...")

    try:
        test_retraining_config()
        test_retraining_pipeline()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

