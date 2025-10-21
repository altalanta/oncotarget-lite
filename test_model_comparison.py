#!/usr/bin/env python3
"""Test script for advanced model comparison and selection framework."""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from oncotarget_lite.model_comparison import (
    ModelMetrics,
    ComparisonCriteria,
    ModelComparator,
    ModelComparisonDashboard
)

def test_model_metrics():
    """Test ModelMetrics dataclass functionality."""
    print("🔍 Testing ModelMetrics...")

    model = ModelMetrics(
        model_id="test_model_v1",
        model_type="logreg",
        auroc=0.85,
        ap=0.78,
        accuracy=0.82,
        f1_score=0.80,
        precision=0.79,
        recall=0.81,
        brier_score=0.15,
        ece=0.06,
        training_time=45.2,
        inference_time=2.1
    )

    # Test that all attributes are accessible
    assert model.model_id == "test_model_v1"
    assert model.auroc == 0.85
    assert model.inference_time == 2.1

    print("✅ ModelMetrics dataclass works correctly")
    return True

def test_comparison_criteria():
    """Test ComparisonCriteria dataclass functionality."""
    print("🔍 Testing ComparisonCriteria...")

    criteria = ComparisonCriteria()

    # Test default values
    assert criteria.primary_metric == "auroc"
    assert criteria.weight_auroc == 0.3
    assert criteria.min_auroc == 0.7

    # Test custom criteria
    custom_criteria = ComparisonCriteria(
        primary_metric="ap",
        weight_auroc=0.2,
        weight_ap=0.4,
        min_auroc=0.8
    )

    assert custom_criteria.primary_metric == "ap"
    assert custom_criteria.weight_ap == 0.4
    assert custom_criteria.min_auroc == 0.8

    print("✅ ComparisonCriteria dataclass works correctly")
    return True

def test_model_comparator():
    """Test ModelComparator functionality."""
    print("🔍 Testing ModelComparator...")

    # Create test models
    models = [
        ModelMetrics(
            model_id="model_a",
            model_type="logreg",
            auroc=0.82,
            ap=0.75,
            accuracy=0.78,
            f1_score=0.76,
            ece=0.08
        ),
        ModelMetrics(
            model_id="model_b",
            model_type="xgboost",
            auroc=0.87,
            ap=0.81,
            accuracy=0.83,
            f1_score=0.82,
            ece=0.06
        ),
        ModelMetrics(
            model_id="model_c",
            model_type="neural_network",
            auroc=0.89,
            ap=0.84,
            accuracy=0.85,
            f1_score=0.84,
            ece=0.04
        )
    ]

    # Test comparator initialization
    criteria = ComparisonCriteria()
    comparator = ModelComparator(criteria)

    # Add models
    for model in models:
        comparator.add_model(model)

    assert len(comparator.models) == 3

    # Test filtering
    filtered = comparator.filter_models_by_criteria()
    assert len(filtered) == 3  # All models pass default criteria

    # Test ranking
    ranked = comparator.rank_models()
    assert len(ranked) == 3

    # Check that models are ranked correctly (model_c should be first)
    top_model, top_score = ranked[0]
    assert top_model.model_id == "model_c"
    assert top_score > ranked[1][1]  # Higher score than second

    # Test recommendations
    recommendations = comparator.get_recommendations()
    assert "top_model" in recommendations
    assert recommendations["top_model"]["model_id"] == "model_c"

    print("✅ ModelComparator works correctly")
    return True

def test_dashboard_creation():
    """Test ModelComparisonDashboard functionality."""
    print("🔍 Testing ModelComparisonDashboard...")

    # Create test comparator
    criteria = ComparisonCriteria()
    comparator = ModelComparator(criteria)

    # Add test models
    models = [
        ModelMetrics(
            model_id="dashboard_test_a",
            model_type="logreg",
            auroc=0.82,
            ap=0.75,
            accuracy=0.78,
            f1_score=0.76,
            ece=0.08
        ),
        ModelMetrics(
            model_id="dashboard_test_b",
            model_type="xgboost",
            auroc=0.87,
            ap=0.81,
            accuracy=0.83,
            f1_score=0.82,
            ece=0.06
        )
    ]

    for model in models:
        comparator.add_model(model)

    # Test dashboard creation
    dashboard = ModelComparisonDashboard(comparator)

    # Test plot creation (should not raise errors)
    try:
        perf_fig = dashboard.create_performance_comparison_plot()
        ranking_fig = dashboard.create_ranking_plot()
        stats_fig = dashboard.create_statistical_significance_plot()

        # Check that figures have data
        assert perf_fig is not None
        assert ranking_fig is not None
        assert stats_fig is not None

        print("✅ Dashboard plot creation works correctly")
        return True

    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
        return False

def test_criteria_filtering():
    """Test criteria-based model filtering."""
    print("🔍 Testing criteria-based filtering...")

    # Create models with varying quality
    models = [
        ModelMetrics(
            model_id="good_model",
            model_type="logreg",
            auroc=0.85,  # Above threshold
            ap=0.78,
            accuracy=0.82,
            f1_score=0.80,
            ece=0.05  # Below threshold
        ),
        ModelMetrics(
            model_id="poor_model",
            model_type="xgboost",
            auroc=0.65,  # Below threshold
            ap=0.60,
            accuracy=0.68,
            f1_score=0.65,
            ece=0.15  # Above threshold
        ),
        ModelMetrics(
            model_id="mediocre_model",
            model_type="neural_network",
            auroc=0.75,  # Above threshold
            ap=0.70,
            accuracy=0.73,
            f1_score=0.72,
            ece=0.08  # Below threshold
        )
    ]

    # Test with strict criteria
    strict_criteria = ComparisonCriteria(
        min_auroc=0.8,
        min_calibration_error=0.06
    )

    comparator = ModelComparator(strict_criteria)
    for model in models:
        comparator.add_model(model)

    # Should only pass the good model
    filtered = comparator.filter_models_by_criteria()
    assert len(filtered) == 1
    assert filtered[0].model_id == "good_model"

    print("✅ Criteria-based filtering works correctly")
    return True

def main():
    """Main test function."""
    print("🧪 Testing Advanced Model Comparison & Selection Framework")
    print("=" * 60)
    print()

    tests = [
        ("Model Metrics", test_model_metrics),
        ("Comparison Criteria", test_comparison_criteria),
        ("Model Comparator", test_model_comparator),
        ("Dashboard Creation", test_dashboard_creation),
        ("Criteria Filtering", test_criteria_filtering),
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
            import traceback
            traceback.print_exc()
            all_passed = False

        print()

    if all_passed:
        print("✅ All tests passed!")
        print()
        print("🎉 The advanced model comparison framework is ready!")
        print("   Run 'make compare' to analyze your models")
        print("   Run 'make compare-interactive' for interactive dashboard")
    else:
        print("❌ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

