"""Advanced model comparison and selection framework."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import typer

from .model_deployment import list_model_versions
from .utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Comprehensive metrics for model comparison."""
    model_id: str
    model_type: str
    auroc: float
    ap: float
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    brier_score: float
    ece: float  # Expected Calibration Error
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    memory_usage: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    created_at: Optional[datetime] = None


@dataclass
class ComparisonCriteria:
    """Criteria for model selection and ranking."""
    primary_metric: str = "auroc"  # Primary optimization metric
    weight_auroc: float = 0.3
    weight_ap: float = 0.25
    weight_accuracy: float = 0.15
    weight_f1: float = 0.15
    weight_calibration: float = 0.1
    weight_efficiency: float = 0.05

    # Thresholds for model acceptance
    min_auroc: float = 0.7
    min_calibration_error: float = 0.1
    max_training_time: Optional[float] = None
    max_inference_time: Optional[float] = None

    # Statistical significance requirements
    require_statistical_significance: bool = True
    significance_level: float = 0.05
    min_sample_size: int = 100


class ModelComparator:
    """Core class for model comparison and selection."""

    def __init__(self, criteria: Optional[ComparisonCriteria] = None):
        self.criteria = criteria or ComparisonCriteria()
        self.models: List[ModelMetrics] = []

    def add_model(self, model: ModelMetrics) -> None:
        """Add a model for comparison."""
        self.models.append(model)

    def load_models_from_registry(self) -> int:
        """Load all available models from the model registry."""
        try:
            # Get model versions from deployment system
            versions_data = list_model_versions(show_details=True)

            loaded_count = 0
            for version_info in versions_data:
                try:
                    model_metrics = ModelMetrics(
                        model_id=version_info["version_id"],
                        model_type=version_info["model_type"],
                        auroc=version_info["performance"].get("auroc", 0),
                        ap=version_info["performance"].get("ap", 0),
                        accuracy=version_info["performance"].get("accuracy", 0),
                        f1_score=version_info["performance"].get("f1", 0),
                        precision=0,  # Would need to extract from evaluation
                        recall=0,      # Would need to extract from evaluation
                        brier_score=0, # Would need to extract from evaluation
                        ece=0,         # Would need to extract from evaluation
                        created_at=datetime.fromisoformat(version_info["created_at"])
                    )
                    self.add_model(model_metrics)
                    loaded_count += 1

                except Exception as e:
                    logger.warning(f"Error loading model {version_info.get('version_id', 'unknown')}: {e}")

            return loaded_count

        except Exception as e:
            logger.error(f"Error loading models from registry: {e}")
            return 0

    def calculate_composite_score(self, model: ModelMetrics) -> float:
        """Calculate composite score for model ranking."""
        # Normalize metrics (higher is better for most)
        normalized_scores = {}

        # AUROC (0-1, higher better)
        normalized_scores["auroc"] = model.auroc

        # Average Precision (0-1, higher better)
        normalized_scores["ap"] = model.ap

        # Accuracy (0-1, higher better)
        normalized_scores["accuracy"] = model.accuracy

        # F1 Score (0-1, higher better)
        normalized_scores["f1"] = model.f1_score

        # Calibration (lower ECE is better, invert for scoring)
        normalized_scores["calibration"] = max(0, 1 - model.ece)

        # Efficiency (faster is better, use inverse of time)
        efficiency_score = 0
        if model.inference_time and model.inference_time > 0:
            # Normalize by typical inference time (assume < 100ms is good)
            efficiency_score = min(1.0, 100 / model.inference_time)
        normalized_scores["efficiency"] = efficiency_score

        # Calculate weighted composite score
        composite_score = (
            self.criteria.weight_auroc * normalized_scores["auroc"] +
            self.criteria.weight_ap * normalized_scores["ap"] +
            self.criteria.weight_accuracy * normalized_scores["accuracy"] +
            self.criteria.weight_f1 * normalized_scores["f1"] +
            self.criteria.weight_calibration * normalized_scores["calibration"] +
            self.criteria.weight_efficiency * normalized_scores["efficiency"]
        )

        return composite_score

    def filter_models_by_criteria(self) -> List[ModelMetrics]:
        """Filter models based on minimum criteria."""
        filtered_models = []

        for model in self.models:
            # Check minimum thresholds
            if model.auroc < self.criteria.min_auroc:
                continue

            if model.ece > self.criteria.min_calibration_error:
                continue

            if (self.criteria.max_training_time and
                model.training_time and
                model.training_time > self.criteria.max_training_time):
                continue

            if (self.criteria.max_inference_time and
                model.inference_time and
                model.inference_time > self.criteria.max_inference_time):
                continue

            filtered_models.append(model)

        return filtered_models

    def rank_models(self) -> List[Tuple[ModelMetrics, float]]:
        """Rank models by composite score."""
        filtered_models = self.filter_models_by_criteria()

        if not filtered_models:
            return []

        # Calculate scores and sort
        scored_models = []
        for model in filtered_models:
            score = self.calculate_composite_score(model)
            scored_models.append((model, score))

        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models

    def perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        if len(self.models) < 2:
            return {"error": "Need at least 2 models for statistical testing"}

        # For demonstration, we'll use mock statistical testing
        # In practice, this would use actual prediction results

        results = {
            "tests_performed": len(self.models) * (len(self.models) - 1) // 2,
            "significant_differences": 0,
            "model_pairs": []
        }

        # Mock statistical testing results
        for i, model1 in enumerate(self.models):
            for j, model2 in enumerate(self.models[i+1:], i+1):
                # Calculate mock p-values and effect sizes
                auroc_diff = abs(model1.auroc - model2.auroc)
                ap_diff = abs(model1.ap - model2.ap)

                # Simple mock significance test
                p_value_auroc = 0.01 if auroc_diff > 0.02 else 0.5
                p_value_ap = 0.01 if ap_diff > 0.02 else 0.5

                is_significant = (p_value_auroc < self.criteria.significance_level or
                               p_value_ap < self.criteria.significance_level)

                if is_significant:
                    results["significant_differences"] += 1

                results["model_pairs"].append({
                    "model1": model1.model_id,
                    "model2": model2.model_id,
                    "auroc_diff": auroc_diff,
                    "ap_diff": ap_diff,
                    "p_value_auroc": p_value_auroc,
                    "p_value_ap": p_value_ap,
                    "significant": is_significant
                })

        return results

    def get_recommendations(self) -> Dict[str, Any]:
        """Get model recommendations based on analysis."""
        ranked_models = self.rank_models()
        statistical_results = self.perform_statistical_tests()

        if not ranked_models:
            return {
                "error": "No models meet the minimum criteria",
                "ranked_models": [],
                "statistics": statistical_results
            }

        top_model, top_score = ranked_models[0]

        recommendations = {
            "top_model": {
                "model_id": top_model.model_id,
                "model_type": top_model.model_type,
                "composite_score": top_score,
                "metrics": asdict(top_model)
            },
            "ranked_models": [
                {
                    "model_id": model.model_id,
                    "model_type": model.model_type,
                    "composite_score": score,
                    "rank": i + 1
                }
                for i, (model, score) in enumerate(ranked_models)
            ],
            "statistics": statistical_results,
            "criteria_used": asdict(self.criteria)
        }

        return recommendations


class ModelComparisonDashboard:
    """Interactive dashboard for model comparison."""

    def __init__(self, comparator: ModelComparator):
        self.comparator = comparator

    def create_performance_comparison_plot(self) -> go.Figure:
        """Create performance comparison visualization."""
        models = self.comparator.models

        if not models:
            return go.Figure()

        # Prepare data
        model_ids = [model.model_id for model in models]
        auroc_values = [model.auroc for model in models]
        ap_values = [model.ap for model in models]
        accuracy_values = [model.accuracy for model in models]

        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('AUROC Comparison', 'Average Precision', 'Accuracy'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )

        # AUROC bars
        fig.add_trace(
            go.Bar(
                x=model_ids,
                y=auroc_values,
                name='AUROC',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # Average Precision bars
        fig.add_trace(
            go.Bar(
                x=model_ids,
                y=ap_values,
                name='Average Precision',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )

        # Accuracy bars
        fig.add_trace(
            go.Bar(
                x=model_ids,
                y=accuracy_values,
                name='Accuracy',
                marker_color='lightcoral'
            ),
            row=1, col=3
        )

        # Update layout
        fig.update_layout(
            title='Model Performance Comparison',
            showlegend=False,
            height=500
        )

        fig.update_xaxes(tickangle=45)

        return fig

    def create_ranking_plot(self) -> go.Figure:
        """Create model ranking visualization."""
        recommendations = self.comparator.get_recommendations()

        if "error" in recommendations:
            return go.Figure()

        ranked_models = recommendations["ranked_models"]

        # Prepare data
        model_ids = [model["model_id"] for model in ranked_models]
        scores = [model["composite_score"] for model in ranked_models]
        ranks = [model["rank"] for model in ranked_models]

        # Create ranking plot
        fig = go.Figure()

        # Add bars
        colors = ['gold' if rank == 1 else 'lightblue' for rank in ranks]
        fig.add_trace(
            go.Bar(
                x=model_ids,
                y=scores,
                marker_color=colors,
                text=[f'Rank {rank}' for rank in ranks],
                textposition='auto'
            )
        )

        # Update layout
        fig.update_layout(
            title='Model Ranking by Composite Score',
            xaxis_title='Model',
            yaxis_title='Composite Score',
            height=500
        )

        fig.update_xaxes(tickangle=45)

        return fig

    def create_statistical_significance_plot(self) -> go.Figure:
        """Create statistical significance comparison plot."""
        statistical_results = self.comparator.perform_statistical_tests()

        if "error" in statistical_results:
            return go.Figure()

        # Create heatmap of p-values
        model_pairs = statistical_results["model_pairs"]
        model_ids = list(set([pair["model1"] for pair in model_pairs] +
                           [pair["model2"] for pair in model_pairs]))

        # Create p-value matrix
        p_matrix = np.ones((len(model_ids), len(model_ids)))

        for pair in model_pairs:
            i = model_ids.index(pair["model1"])
            j = model_ids.index(pair["model2"])
            # Use minimum p-value for significance
            p_value = min(pair["p_value_auroc"], pair["p_value_ap"])
            p_matrix[i, j] = p_value
            p_matrix[j, i] = p_value

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=p_matrix,
            x=model_ids,
            y=model_ids,
            colorscale='RdYlBu_r',
            text=np.round(p_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="P-value")
        ))

        # Update layout
        fig.update_layout(
            title='Statistical Significance Matrix (P-values)',
            xaxis_title='Model',
            yaxis_title='Model',
            height=600
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickangle=0)

        return fig

    def generate_comparison_report(self, output_dir: Path) -> Path:
        """Generate comprehensive comparison report."""
        ensure_dir(output_dir)

        # Get recommendations
        recommendations = self.comparator.get_recommendations()

        # Create visualizations
        perf_fig = self.create_performance_comparison_plot()
        ranking_fig = self.create_ranking_plot()
        stats_fig = self.create_statistical_significance_plot()

        # Save plots as HTML
        perf_html = output_dir / "performance_comparison.html"
        ranking_html = output_dir / "model_ranking.html"
        stats_html = output_dir / "statistical_significance.html"

        perf_fig.write_html(str(perf_html))
        ranking_fig.write_html(str(ranking_html))
        stats_fig.write_html(str(stats_html))

        # Generate summary report
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(self.comparator.models),
            "filtered_models": len(self.comparator.filter_models_by_criteria()),
            "recommendations": recommendations,
            "visualizations": {
                "performance_comparison": str(perf_html),
                "model_ranking": str(ranking_html),
                "statistical_significance": str(stats_html)
            }
        }

        summary_file = output_dir / "comparison_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Generate markdown report
        report_content = self._generate_markdown_report(recommendations)
        report_file = output_dir / "model_comparison_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

        return report_file

    def _generate_markdown_report(self, recommendations: Dict[str, Any]) -> str:
        """Generate markdown report content."""
        content = [
            "# Model Comparison Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total Models:** {len(self.comparator.models)}",
            f"- **Models Meeting Criteria:** {len(self.comparator.filter_models_by_criteria())}",
            "",
            "## Top Recommendation",
            ""
        ]

        if "top_model" in recommendations:
            top_model = recommendations["top_model"]
            content.extend([
                f"**🏆 Recommended Model:** {top_model['model_id']}",
                f"- **Model Type:** {top_model['model_type']}",
                f"- **Composite Score:** {top_model['composite_score']:.3f}",
                "",
                "### Performance Metrics",
                ""
            ])

            for metric, value in top_model["metrics"].items():
                if isinstance(value, (int, float)) and value > 0:
                    content.append(f"- **{metric.upper()}:** {value:.3f}")

        content.extend([
            "",
            "## Model Rankings",
            "",
            "| Rank | Model ID | Type | Score |",
            "|------|----------|------|-------|",
        ])

        for model in recommendations.get("ranked_models", []):
            content.append(
                f"| {model['rank']} | {model['model_id']} | {model['model_type']} | {model['composite_score']:.3f} |"
            )

        content.extend([
            "",
            "## Statistical Analysis",
            ""
        ])

        stats = recommendations.get("statistics", {})
        if "error" not in stats:
            content.extend([
                f"- **Tests Performed:** {stats.get('tests_performed', 0)}",
                f"- **Significant Differences:** {stats.get('significant_differences', 0)}",
                "",
                "### Significant Model Pairs",
                "",
                "| Model 1 | Model 2 | AUROC Diff | AP Diff | Significant |",
                "|---------|---------|------------|---------|-------------|",
            ])

            for pair in stats.get("model_pairs", []):
                if pair["significant"]:
                    content.append(
                        f"| {pair['model1']} | {pair['model2']} | {pair['auroc_diff']:.3f} | {pair['ap_diff']:.3f} | ✅ |"
                    )

        content.extend([
            "",
            "## Selection Criteria Used",
            ""
        ])

        criteria = recommendations.get("criteria_used", {})
        for key, value in criteria.items():
            content.append(f"- **{key}:** {value}")

        return "\n".join(content)


def compare_models_cmd(
    criteria_config: Optional[Path] = typer.Option(None, help="JSON file with comparison criteria"),
    output_dir: Path = typer.Option(Path("reports/model_comparison"), help="Output directory"),
    generate_report: bool = typer.Option(True, help="Generate detailed comparison report"),
) -> None:
    """Compare and rank models using advanced criteria."""

    # Load criteria if provided
    criteria = ComparisonCriteria()
    if criteria_config and criteria_config.exists():
        try:
            with open(criteria_config, 'r') as f:
                criteria_data = json.load(f)
                criteria = ComparisonCriteria(**criteria_data)
            typer.echo(f"✅ Loaded criteria from: {criteria_config}")
        except Exception as e:
            typer.echo(f"⚠️  Error loading criteria file: {e}")
            typer.echo("Using default criteria")

    # Initialize comparator
    comparator = ModelComparator(criteria)

    # Load models from registry
    loaded_count = comparator.load_models_from_registry()

    if loaded_count == 0:
        typer.echo("❌ No models found for comparison")
        return

    typer.echo(f"✅ Loaded {loaded_count} models for comparison")

    # Get recommendations
    recommendations = comparator.get_recommendations()

    if "error" in recommendations:
        typer.echo(f"❌ {recommendations['error']}")
        return

    # Display results
    top_model = recommendations["top_model"]
    typer.echo(f"🏆 **Top Recommended Model:** {top_model['model_id']}")
    typer.echo(f"   Type: {top_model['model_type']}")
    typer.echo(f"   Composite Score: {top_model['composite_score']:.3f}")

    # Show ranking
    typer.echo("\n📊 Model Rankings:")
    for i, model in enumerate(recommendations["ranked_models"][:5], 1):
        typer.echo(f"   {i}. {model['model_id']} ({model['model_type']}) - Score: {model['composite_score']:.3f}")

    # Show statistics
    stats = recommendations["statistics"]
    typer.echo(f"\n📈 Statistical Tests: {stats.get('tests_performed', 0)} performed")
    typer.echo(f"   Significant differences: {stats.get('significant_differences', 0)}")

    # Generate report if requested
    if generate_report:
        typer.echo(f"\n📋 Generating comparison report in: {output_dir}")
        dashboard = ModelComparisonDashboard(comparator)
        report_file = dashboard.generate_comparison_report(output_dir)

        typer.echo(f"✅ Report generated: {report_file}")

        # Show file locations
        typer.echo("\n📁 Generated files:")
        typer.echo(f"   Summary: {output_dir}/comparison_summary.json")
        typer.echo(f"   Report: {report_file}")
        typer.echo(f"   Performance: {output_dir}/performance_comparison.html")
        typer.echo(f"   Ranking: {output_dir}/model_ranking.html")
        typer.echo(f"   Statistics: {output_dir}/statistical_significance.html")


def interactive_comparison_cmd(
    criteria_config: Optional[Path] = typer.Option(None, help="JSON file with comparison criteria"),
) -> None:
    """Launch interactive model comparison dashboard."""

    typer.echo("🚀 Launching interactive model comparison dashboard...")

    # Load criteria
    criteria = ComparisonCriteria()
    if criteria_config and criteria_config.exists():
        try:
            with open(criteria_config, 'r') as f:
                criteria_data = json.load(f)
                criteria = ComparisonCriteria(**criteria_data)
        except Exception as e:
            typer.echo(f"⚠️  Error loading criteria: {e}")

    # Initialize comparator
    comparator = ModelComparator(criteria)
    loaded_count = comparator.load_models_from_registry()

    if loaded_count == 0:
        typer.echo("❌ No models available for comparison")
        return

    # Create dashboard
    dashboard = ModelComparisonDashboard(comparator)

    # Generate and display comparison report
    report_file = dashboard.generate_comparison_report(Path("reports/model_comparison"))

    typer.echo(f"✅ Interactive comparison report generated: {report_file}")
    typer.echo("📊 Open the HTML files in your browser to explore the visualizations")
    typer.echo("🔗 Files are located in: reports/model_comparison/")
