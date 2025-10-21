#!/usr/bin/env python3
"""Test script for model serving functionality."""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_server_import():
    """Test that the model server can be imported."""
    try:
        # Import just the dataclasses and models without the full chain
        from oncotarget_lite.model_server import (
            ModelVersion,
            ABTestConfig,
            PredictionRequest,
            PredictionResponse,
            ModelRegistry,
            PredictionCache
        )
        print("✅ Model server classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dataclasses():
    """Test that the dataclasses work correctly."""
    from oncotarget_lite.model_server import ModelVersion, ABTestConfig
    from datetime import datetime

    # Test ModelVersion
    model_version = ModelVersion(
        version_id="test_model_001",
        model_path=Path("models/test.pkl"),
        created_at=datetime.now(),
        performance_metrics={"auroc": 0.85, "ap": 0.78},
        feature_names=["gene_A", "gene_B"],
        model_type="logreg"
    )

    print("✅ ModelVersion created successfully")
    print(f"   Version ID: {model_version.version_id}")
    print(f"   Performance: {model_version.performance_metrics}")

    # Test ABTestConfig
    ab_test = ABTestConfig(
        test_id="experiment_001",
        model_a="model_v1",
        model_b="model_v2",
        traffic_split=0.5,
        start_time=datetime.now()
    )

    print("✅ ABTestConfig created successfully")
    print(f"   Test ID: {ab_test.test_id}")
    print(f"   Traffic split: {ab_test.traffic_split}")

    return True

def test_pydantic_models():
    """Test that the Pydantic models work correctly."""
    from oncotarget_lite.model_server import PredictionRequest, PredictionResponse
    from datetime import datetime

    # Test PredictionRequest
    request = PredictionRequest(
        features={"gene_A": 0.5, "gene_B": 0.3, "clinical_score": 0.8},
        model_version="test_model",
        request_id="req_123"
    )

    print("✅ PredictionRequest created successfully")
    print(f"   Features: {request.features}")
    print(f"   Model version: {request.model_version}")

    # Test PredictionResponse
    response = PredictionResponse(
        prediction=0.75,
        prediction_class=1,
        probabilities={"0": 0.25, "1": 0.75},
        model_version="test_model",
        request_id="req_123",
        processing_time_ms=150.5,
        timestamp=datetime.now(),
        confidence_score=0.75
    )

    print("✅ PredictionResponse created successfully")
    print(f"   Prediction: {response.prediction}")
    print(f"   Confidence: {response.confidence_score}")

    return True

def test_cache():
    """Test the prediction cache functionality."""
    from oncotarget_lite.model_server import PredictionCache

    cache = PredictionCache(max_size=100, ttl_seconds=60)

    # Test cache operations
    features = {"gene_A": 0.5, "gene_B": 0.3}
    model_version = "test_model"

    # Set a cached prediction
    prediction_data = {
        "prediction": 0.75,
        "prediction_class": 1,
        "probabilities": {"0": 0.25, "1": 0.75},
        "model_version": model_version,
        "request_id": "req_123",
        "confidence_score": 0.75
    }

    cache.set(features, model_version, prediction_data)

    # Get the cached prediction
    cached = cache.get(features, model_version)

    if cached:
        print("✅ Cache set/get operations work correctly")
        print(f"   Cached prediction: {cached['prediction']}")
    else:
        print("❌ Cache operations failed")
        return False

    return True

def main():
    """Main test function."""
    print("🧪 Testing Enhanced Model Serving & API Layer")
    print("=" * 50)
    print()

    tests = [
        ("Model Server Import", test_model_server_import),
        ("Dataclasses", test_dataclasses),
        ("Pydantic Models", test_pydantic_models),
        ("Cache Functionality", test_cache),
    ]

    all_passed = True
    for test_name, test_func in tests:
        print(f"🔍 {test_name}")
        print("-" * 30)

        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed: {e}")
            all_passed = False

        print()

    if all_passed:
        print("✅ All tests passed!")
        print()
        print("🎉 The enhanced model serving layer is ready!")
        print("   Run 'make serve' to start the server")
        print("   Visit http://localhost:8000/docs for API documentation")
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

