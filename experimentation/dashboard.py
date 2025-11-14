"""Interactive dashboard for experiment analysis."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .manager import ExperimentManager
from .schemas import Experiment


class ExperimentDashboard:
    """Interactive dashboard for experiment analysis."""

    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager

    def create_experiment_comparison_plot(self) -> go.Figure:
        """Create experiment comparison visualization."""
        experiments = self.experiment_manager.list_experiments()

        if not experiments:
            return go.Figure()

        # Prepare data
        exp_names = [exp['name'] for exp in experiments]
        best_scores = [exp['best_score'] for exp in experiments]
        trial_counts = [exp['trials_count'] for exp in experiments]

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Best Scores by Experiment', 'Trial Count by Experiment'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Best scores
        fig.add_trace(
            go.Bar(
                x=exp_names,
                y=best_scores,
                name='Best Score',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # Trial counts
        fig.add_trace(
            go.Bar(
                x=exp_names,
                y=trial_counts,
                name='Trial Count',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title='Experiment Comparison Overview',
            showlegend=False,
            height=500
        )

        fig.update_xaxes(tickangle=45)

        return fig

    def create_trial_progression_plot(self, experiment_id: str) -> go.Figure:
        """Create trial progression visualization for an experiment."""
        experiment = self.experiment_manager.get_experiment(experiment_id)

        if not experiment or not experiment.trials:
            return go.Figure()

        # Prepare data
        primary_metric = experiment.config.metrics[0]

        # Extract metric values for completed trials
        metric_values = []
        trial_numbers = []

        for i, trial in enumerate(experiment.trials):
            if trial.status == "completed" and primary_metric in trial.metrics:
                metric_values.append(trial.metrics[primary_metric])
                trial_numbers.append(i + 1)

        if not metric_values:
            return go.Figure()

        # Create progression plot
        fig = go.Figure()

        # Add trial progression
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=metric_values,
                mode='lines+markers',
                name=f'{primary_metric.upper()} Progression',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            )
        )

        # Add best trial marker
        if experiment.best_trial and primary_metric in experiment.best_trial.metrics:
            best_value = experiment.best_trial.metrics[primary_metric]
            best_trial_idx = next(
                (i for i, trial in enumerate(experiment.trials)
                 if trial.trial_id == experiment.best_trial.trial_id),
                0
            )

            fig.add_trace(
                go.Scatter(
                    x=[best_trial_idx + 1],
                    y=[best_value],
                    mode='markers',
                    name='Best Trial',
                    marker=dict(color='red', size=12, symbol='star')
                )
            )

        # Update layout
        fig.update_layout(
            title=f'Trial Progression - {experiment.config.experiment_name}',
            xaxis_title='Trial Number',
            yaxis_title=primary_metric.upper(),
            height=500
        )

        return fig

    def generate_experiment_report(self, experiment_id: str, output_dir: Path) -> Path:
        """Generate comprehensive experiment report."""
        experiment = self.experiment_manager.get_experiment(experiment_id)

        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualizations
        comparison_fig = self.create_experiment_comparison_plot()
        progression_fig = self.create_trial_progression_plot(experiment_id)

        # Save plots
        comparison_html = output_dir / "experiment_comparison.html"
        progression_html = output_dir / "trial_progression.html"

        comparison_fig.write_html(str(comparison_html))
        progression_fig.write_html(str(progression_html))

        # Generate detailed report
        report_content = self._generate_experiment_report(experiment)
        report_file = output_dir / f"experiment_{experiment_id}_report.md"

        with open(report_file, 'w') as f:
            f.write(report_content)

        return report_file

    def _generate_experiment_report(self, experiment: Experiment) -> str:
        """Generate markdown report for an experiment."""
        content = [
            "# Experiment Report",
            "",
            f"**Experiment:** {experiment.config.experiment_name}",
            f"**ID:** {experiment.experiment_id}",
            f"**Status:** {experiment.status}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Experiment Configuration",
            "",
            f"- **Model Types:** {', '.join(experiment.config.model_types)}",
            f"- **Trials:** {experiment.config.n_trials}",
            f"- **Metrics:** {', '.join(experiment.config.metrics)}",
            f"- **Optimization:** {experiment.config.optimization_direction}",
            "",
            "## Results Summary",
            ""
        ]

        if experiment.best_trial:
            content.extend([
                f"### Best Trial: {experiment.best_trial.trial_id}",
                "",
                "**Parameters:**",
                ""
            ])

            for param, value in experiment.best_trial.parameters.items():
                content.append(f"- {param}: {value}")

            content.extend([
                "",
                "**Performance:**",
                ""
            ])

            for metric, value in experiment.best_trial.metrics.items():
                if isinstance(value, float):
                    content.append(f"- {metric.upper()}: {value:.4f}")

        content.extend([
            "",
            "## Trial Summary",
            "",
            f"- **Total Trials:** {len(experiment.trials)}",
            f"- **Successful Trials:** {len([t for t in experiment.trials if t.status == 'completed'])}",
            f"- **Failed Trials:** {len([t for t in experiment.trials if t.status == 'failed'])}",
            "",
            "## Trial Details",
            "",
            "| Trial ID | Model Type | Status | Score | Training Time |",
            "|----------|------------|--------|-------|---------------|",
        ])

        primary_metric = experiment.config.metrics[0] if experiment.config.metrics else "auroc"

        for trial in experiment.trials[:20]:  # Show first 20 trials
            score = trial.metrics.get(primary_metric, 0)
            model_type = trial.parameters.get('model_type', 'unknown')
            training_time = trial.training_time or 0
            status_icon = "✅" if trial.status == "completed" else "❌"
            content.append(
                f"| {trial.trial_id} | {model_type} | {status_icon} | {score:.4f} | {training_time:.2f}s |"
            )

        if len(experiment.trials) > 20:
            content.append(f"\n*... and {len(experiment.trials) - 20} more trials*")

        return "\n".join(content)






