#!/usr/bin/env python3
"""Demonstration script for advanced model comparison and selection framework."""

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

def demo_comparison_criteria():
    """Demonstrate the comparison criteria system."""
    print("⚖️ Comparison Criteria Demo")
    print("=" * 40)

    # Default criteria
    criteria = ComparisonCriteria()
    print("Default Criteria:")
    print(f"  Primary metric: {criteria.primary_metric}")
    print(f"  AUROC weight: {criteria.weight_auroc}")
    print(f"  Min AUROC threshold: {criteria.min_auroc}")
    print()

    # Custom criteria for research scenario
    research_criteria = ComparisonCriteria(
        primary_metric="auroc",
        weight_auroc=0.4,
        weight_ap=0.3,
        weight_accuracy=0.1,
        weight_f1=0.1,
        weight_calibration=0.1,
        min_auroc=0.8,  # Higher threshold for research
        min_calibration_error=0.05,  # Stricter calibration
        require_statistical_significance=True,
        significance_level=0.01  # More stringent significance
    )

    print("Research-Focused Criteria:")
    print(f"  Primary metric: {research_criteria.primary_metric}")
    print(f"  AUROC weight: {research_criteria.weight_auroc}")
    print(f"  Min AUROC threshold: {research_criteria.min_auroc}")
    print(f"  Significance level: {research_criteria.significance_level}")
    print()

def demo_model_metrics():
    """Demonstrate model metrics handling."""
    print("📊 Model Metrics Demo")
    print("=" * 40)

    # Create sample model metrics
    models = [
        ModelMetrics(
            model_id="baseline_logreg_v1",
            model_type="logreg",
            auroc=0.82,
            ap=0.75,
            accuracy=0.78,
            f1_score=0.76,
            precision=0.74,
            recall=0.78,
            brier_score=0.18,
            ece=0.08,
            training_time=45.2,
            inference_time=2.1
        ),
        ModelMetrics(
            model_id="improved_xgb_v2",
            model_type="xgboost",
            auroc=0.87,
            ap=0.81,
            accuracy=0.83,
            f1_score=0.82,
            precision=0.81,
            recall=0.83,
            brier_score=0.15,
            ece=0.06,
            training_time=120.5,
            inference_time=5.3
        ),
        ModelMetrics(
            model_id="experimental_nn_v3",
            model_type="neural_network",
            auroc=0.89,
            ap=0.84,
            accuracy=0.85,
            f1_score=0.84,
            precision=0.83,
            recall=0.85,
            brier_score=0.13,
            ece=0.04,
            training_time=360.0,
            inference_time=12.5
        )
    ]

    print("Sample Models for Comparison:")
    for model in models:
        print(f"  • {model.model_id} ({model.model_type})")
        print(f"    AUROC: {model.auroc".3f"}, AP: {model.ap".3f"}")
        print(f"    Training time: {model.training_time}s, Inference: {model.inference_time}ms")
    print()

    return models

def demo_composite_scoring():
    """Demonstrate composite scoring and ranking."""
    print("🏆 Composite Scoring Demo")
    print("=" * 40)

    models = demo_model_metrics()
    criteria = ComparisonCriteria()

    # Calculate scores for each model
    print("Composite Scores (weighted average):")
    for model in models:
        score = (
            criteria.weight_auroc * model.auroc +
            criteria.weight_ap * model.ap +
            criteria.weight_accuracy * model.accuracy +
            criteria.weight_f1 * model.f1_score +
            criteria.weight_calibration * max(0, 1 - model.ece) +
            criteria.weight_efficiency * min(1.0, 100 / model.inference_time)
        )

        print(f"  {model.model_id}: {score".3f"}")
        print(f"    Breakdown: AUROC({model.auroc".3f"}) + AP({model.ap".3f"}) + "
              f"Accuracy({model.accuracy".3f"}) + F1({model.f1_score".3f"}) + "
              f"Calibration({max(0, 1 - model.ece)".3f"}) + "
              f"Efficiency({min(1.0, 100 / model.inference_time)".3f"})")
    print()

def demo_filtering_criteria():
    """Demonstrate model filtering by criteria."""
    print("🔍 Model Filtering Demo")
    print("=" * 40)

    models = demo_model_metrics()
    criteria = ComparisonCriteria()

    print("Filtering Criteria:")
    print(f"  Min AUROC: {criteria.min_auroc}")
    print(f"  Max Calibration Error: {criteria.min_calibration_error}")
    print()

    # Check which models pass criteria
    passing_models = []
    for model in models:
        passes = (
            model.auroc >= criteria.min_auroc and
            model.ece <= criteria.min_calibration_error
        )

        status = "✅ PASS" if passes else "❌ FAIL"
        print(f"  {model.model_id}: {status}")

        if passes:
            passing_models.append(model)

    print(f"\nModels passing criteria: {len(passing_models)}/{len(models)}")
    print()

def demo_statistical_testing():
    """Demonstrate statistical significance testing."""
    print("📈 Statistical Testing Demo")
    print("=" * 40)

    models = demo_model_metrics()

    print("Mock Statistical Tests (pairwise comparisons):")
    print()

    # Simulate pairwise comparisons
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models[i+1:], i+1):
            auroc_diff = abs(model1.auroc - model2.auroc)
            ap_diff = abs(model1.ap - model2.ap)

            # Mock p-values based on differences
            p_auroc = 0.01 if auroc_diff > 0.03 else 0.6
            p_ap = 0.01 if ap_diff > 0.03 else 0.6

            significant = (p_auroc < 0.05) or (p_ap < 0.05)

            status = "🔬 SIGNIFICANT" if significant else "📊 NOT SIGNIFICANT"
            print(f"  {model1.model_id} vs {model2.model_id}:")
            print(f"    AUROC diff: {auroc_diff".3f"} (p={p_auroc".3f"})")
            print(f"    AP diff: {ap_diff".3f"} (p={p_ap".3f"})")
            print(f"    Result: {status}")
            print()

def demo_visualization_info():
    """Demonstrate visualization capabilities."""
    print("📊 Visualization Demo")
    print("=" * 40)

    print("The comparison framework generates:")
    print("• Performance comparison bar charts")
    print("• Model ranking visualizations")
    print("• Statistical significance heatmaps")
    print("• Comprehensive markdown reports")
    print()

    print("Interactive visualizations include:")
    print("• Hover tooltips with detailed metrics")
    print("• Color-coded rankings and significance")
    print("• Responsive design for different screen sizes")
    print("• Export capabilities for presentations")
    print()

def demo_integration_workflow():
    """Demonstrate integration with other systems."""
    print("🔗 Integration Workflow Demo")
    print("=" * 40)

    print("Complete ML Pipeline Integration:")
    print("1. 🔄 Automated Retraining → Generate new model versions")
    print("2. 📊 Model Comparison → Evaluate and rank all models")
    print("3. 🚀 Model Deployment → Deploy best model to production")
    print("4. 🌐 Model Serving → Serve predictions via API")
    print("5. 📈 Monitoring → Track performance and trigger retraining")
    print()

    print("Example workflow commands:")
    print("  make retrain                    # Generate new models")
    print("  make compare                    # Compare and rank models")
    print("  make deploy VERSION_ID=<best>  # Deploy top model")
    print("  make serve                      # Start serving API")
    print()

def main():
    """Main demonstration function."""
    print("🚀 Advanced Model Comparison & Selection Framework")
    print("=" * 60)
    print()

    demo_comparison_criteria()
    demo_model_metrics()
    demo_composite_scoring()
    demo_filtering_criteria()
    demo_statistical_testing()
    demo_visualization_info()
    demo_integration_workflow()

    print("✅ Model Comparison Framework Demonstration completed!")
    print()
    print("Next steps:")
    print("1. Run 'make compare' to analyze your models")
    print("2. Open reports/model_comparison/ files in browser")
    print("3. Use custom criteria in configs/comparison_criteria.json")
    print("4. Deploy the recommended model with 'make deploy'")

if __name__ == "__main__":
    main()
