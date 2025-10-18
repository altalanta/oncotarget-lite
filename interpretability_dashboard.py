"""Advanced interpretability visualizations and interactive dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import PROCESSED_DIR
from .model import MODELS_DIR
from .utils import ensure_dir


class InterpretabilityDashboard:
    """Create advanced interpretability visualizations and interactive dashboards."""

    def __init__(self, shap_dir: Path = Path("reports/shap")):
        self.shap_dir = shap_dir
        self._load_data()

    def _load_data(self) -> None:
        """Load SHAP data for visualization."""
        try:
            # Load SHAP values
            shap_file = self.shap_dir / "shap_values.npz"
            if shap_file.exists():
                data = np.load(shap_file, allow_pickle=True)
                self.genes = data["genes"]
                self.shap_values = data["values"]
                self.feature_names = data["feature_names"]

            # Load alias map
            alias_file = self.shap_dir / "alias_map.json"
            if alias_file.exists():
                self.alias_map = json.loads(alias_file.read_text())
            else:
                self.alias_map = {}

        except Exception as e:
            print(f"Warning: Could not load SHAP data: {e}")
            self.genes = []
            self.shap_values = None
            self.feature_names = []
            self.alias_map = {}

    def create_global_importance_plot(self, top_k: int = 20) -> go.Figure:
        """Create enhanced global feature importance plot."""
        if self.shap_values is None:
            return go.Figure()

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)

        # Sort and select top features
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        top_features = [self.feature_names[i] for i in sorted_idx[:top_k]]
        top_values = mean_abs_shap[sorted_idx[:top_k]]

        # Create interactive bar plot
        fig = go.Figure()
        colors = ['#ff6b6b' if i < 5 else '#4ecdc4' for i in range(len(top_features))]

        fig.add_trace(go.Bar(
            y=top_features,
            x=top_values,
            orientation='h',
            marker=dict(color=colors),
            hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title="Global Feature Importance (Top 20)",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Features",
            template="plotly_white",
            height=600,
            showlegend=False
        )

        return fig

    def create_gene_contribution_plot(self, gene: str) -> go.Figure:
        """Create detailed contribution plot for a specific gene."""
        if self.shap_values is None or gene not in self.genes or len(self.genes) == 0 or self.genes is None:
            return go.Figure()

        # Find gene index
        gene_idx = np.where(self.genes == gene)[0][0]

        # Get SHAP values for this gene
        gene_shap = self.shap_values[gene_idx]

        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(gene_shap))[::-1]
        top_features = [self.feature_names[i] for i in sorted_idx[:15]]
        top_values = gene_shap[sorted_idx[:15]]

        # Create waterfall plot
        fig = go.Figure()

        # Add base value (mean prediction)
        base_value = 0.5  # This should be calculated from model
        fig.add_trace(go.Waterfall(
            name="SHAP Contributions",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(top_features),
            x=["Base"] + top_features,
            textposition="outside",
            text=["0.5"] + [f"{v:.3f}" for v in top_values],
            y=[base_value] + list(top_values),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title=f"SHAP Contribution Breakdown for {gene}",
            xaxis_title="Features",
            yaxis_title="Contribution to Prediction",
            template="plotly_white",
            height=500,
            showlegend=False
        )

        return fig

    def create_feature_interaction_heatmap(self, top_k: int = 10) -> go.Figure:
        """Create feature interaction heatmap."""
        if self.shap_values is None:
            return go.Figure()

        # Calculate feature-feature correlations in SHAP values
        # This is a simplified version - in practice, you'd use SHAP interaction values
        shap_corr = np.corrcoef(self.shap_values.T)

        # Get top features by importance
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
        top_features = [self.feature_names[i] for i in top_idx]
        top_corr = shap_corr[np.ix_(top_idx, top_idx)]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=top_corr,
            x=top_features,
            y=top_features,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            hovertemplate='Feature 1: %{x}<br>Feature 2: %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title="Feature Interaction Heatmap (SHAP Correlation)",
            template="plotly_white",
            height=600
        )

        return fig

    def create_stability_analysis_plot(self, validation_report_path: Path) -> go.Figure:
        """Create stability analysis visualization from validation report."""
        try:
            with open(validation_report_path) as f:
                report = json.load(f)

            # Extract stability data
            stability_data = report.get("explanation_quality", {})
            feature_ci = report.get("feature_importance_ci", {})

            # Create subplots for stability metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Background Consistency", "Feature Stability", "Perturbation Robustness", "Overall Quality"),
                specs=[[{"type": "scatter"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "indicator"}]]
            )

            # Background consistency plot
            bg_scores = stability_data.get("background_consistency_scores", {})
            if bg_scores:
                sizes = list(bg_scores.keys())
                scores = list(bg_scores.values())
                fig.add_trace(
                    go.Scatter(x=sizes, y=scores, mode='lines+markers', name="Background Consistency"),
                    row=1, col=1
                )

            # Feature stability plot
            if feature_ci:
                features = list(feature_ci.keys())[:10]  # Top 10 features
                stability_scores = [feature_ci[f].get("stability", 0) for f in features]
                fig.add_trace(
                    go.Bar(x=features, y=stability_scores, name="Feature Stability"),
                    row=1, col=2
                )

            # Perturbation robustness
            perturbation_scores = stability_data.get("perturbation_robustness", {})
            if perturbation_scores:
                epsilons = list(perturbation_scores.keys())
                scores = list(perturbation_scores.values())
                fig.add_trace(
                    go.Scatter(x=epsilons, y=scores, mode='lines+markers', name="Perturbation Robustness"),
                    row=2, col=1
                )

            # Overall quality score
            overall_score = stability_data.get("overall_quality_score", 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score,
                    title={"text": "Overall Quality Score"},
                    gauge={"axis": {"range": [0, 1]}},
                ),
                row=2, col=2
            )

            fig.update_layout(
                title="Interpretability Validation Dashboard",
                template="plotly_white",
                height=800
            )

            return fig

        except Exception as e:
            print(f"Warning: Could not create stability plot: {e}")
            return go.Figure()

    def create_comprehensive_dashboard(self, output_path: Path) -> None:
        """Create and save a comprehensive interpretability dashboard."""
        ensure_dir(output_path.parent)

        # Create subplots for main dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Global Feature Importance", "Feature Interactions", "Gene Contributions", "Validation Summary"),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]],
            vertical_spacing=0.1
        )

        # Add global importance plot
        global_fig = self.create_global_importance_plot()
        for trace in global_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # Add interaction heatmap
        interaction_fig = self.create_feature_interaction_heatmap()
        for trace in interaction_fig.data:
            fig.add_trace(trace, row=1, col=2)

        # Add gene contribution plot (for first available gene)
        if self.genes is not None and len(self.genes) > 0:
            gene_fig = self.create_gene_contribution_plot(self.genes[0])
            for trace in gene_fig.data:
                fig.add_trace(trace, row=2, col=1)

        # Add validation summary if available
        validation_path = self.shap_dir.parent / "validation_report.json"
        if validation_path.exists():
            validation_fig = self.create_stability_analysis_plot(validation_path)
            for trace in validation_fig.data:
                fig.add_trace(trace, row=2, col=2)

        fig.update_layout(
            title="Comprehensive Model Interpretability Dashboard",
            template="plotly_white",
            height=1000,
            showlegend=False
        )

        # Save as HTML
        fig.write_html(str(output_path))
        print(f"✅ Dashboard saved to {output_path}")

    def create_model_comparison_dashboard(self, model_reports: List[Path], output_path: Path) -> None:
        """Create dashboard comparing interpretability across different models."""
        ensure_dir(output_path.parent)

        # Collect data from multiple model reports
        comparison_data = []
        for report_path in model_reports:
            try:
                with open(report_path) as f:
                    report = json.load(f)

                model_name = report_path.stem.replace("validation_report_", "")
                quality = report.get("explanation_quality", {})

                comparison_data.append({
                    "model": model_name,
                    "overall_quality": quality.get("overall_quality_score", 0),
                    "background_consistency": quality.get("background_consistency", 0),
                    "stability_score": quality.get("stability_score", 0),
                    "robustness": quality.get("perturbation_robustness", 0),
                })
            except Exception as e:
                print(f"Warning: Could not load {report_path}: {e}")

        if not comparison_data:
            print("No valid model reports found for comparison")
            return

        df = pd.DataFrame(comparison_data)

        # Create comparison plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Overall Quality", "Background Consistency", "Stability Score", "Perturbation Robustness"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        metrics = ["overall_quality", "background_consistency", "stability_score", "robustness"]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for metric, (row, col) in zip(metrics, positions):
            fig.add_trace(
                go.Bar(
                    x=df["model"],
                    y=df[metric],
                    name=metric.replace("_", " ").title(),
                ),
                row=row, col=col
            )

        fig.update_layout(
            title="Model Interpretability Comparison",
            template="plotly_white",
            height=800,
            showlegend=False
        )

        fig.write_html(str(output_path))
        print(f"✅ Model comparison dashboard saved to {output_path}")

    def save_static_exports(self, output_dir: Path) -> None:
        """Save static image exports of key visualizations."""
        ensure_dir(output_dir)

        # Export global importance plot
        fig = self.create_global_importance_plot()
        fig.write_image(str(output_dir / "global_importance.png"), engine="kaleido")

        # Export interaction heatmap
        fig = self.create_feature_interaction_heatmap()
        fig.write_image(str(output_dir / "feature_interactions.png"), engine="kaleido")

        print(f"✅ Static visualizations exported to {output_dir}")
