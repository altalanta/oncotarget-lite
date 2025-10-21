"""Advanced ML experimentation platform with experiment management and automated model selection."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import typer

from .model_comparison import ModelComparator, ComparisonCriteria
from .utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for ML experiments."""
    experiment_name: str
    model_types: List[str]
    search_spaces: Dict[str, Dict[str, Any]]
    metrics: List[str] = None
    optimization_direction: str = "maximize"  # maximize or minimize
    n_trials: int = 100
    timeout: Optional[int] = None
    cv_folds: int = 5
    random_seed: int = 42

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["auroc", "ap", "accuracy", "f1"]


@dataclass
class ExperimentTrial:
    """Represents a single trial in an experiment."""
    trial_id: str
    experiment_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    model_path: Optional[Path] = None
    training_time: Optional[float] = None
    timestamp: datetime = None
    status: str = "completed"  # completed, failed, running

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Experiment:
    """Complete ML experiment with multiple trials."""
    experiment_id: str
    config: ExperimentConfig
    trials: List[ExperimentTrial]
    best_trial: Optional[ExperimentTrial] = None
    status: str = "running"  # running, completed, failed
    created_at: datetime = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ExperimentManager:
    """Manages ML experiments and trials."""

    def __init__(self, experiments_dir: Path = Path("reports/experiments")):
        self.experiments_dir = experiments_dir
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: Dict[str, Experiment] = {}
        self._load_experiments()

    def _load_experiments(self):
        """Load existing experiments from disk."""
        if not self.experiments_dir.exists():
            return

        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                exp_file = exp_dir / "experiment.json"
                if exp_file.exists():
                    try:
                        with open(exp_file, 'r') as f:
                            exp_data = json.load(f)

                        config = ExperimentConfig(**exp_data['config'])
                        trials = [
                            ExperimentTrial(**trial_data)
                            for trial_data in exp_data['trials']
                        ]

                        experiment = Experiment(
                            experiment_id=exp_data['experiment_id'],
                            config=config,
                            trials=trials,
                            status=exp_data.get('status', 'completed'),
                            created_at=datetime.fromisoformat(exp_data['created_at']),
                            completed_at=datetime.fromisoformat(exp_data['completed_at']) if exp_data.get('completed_at') else None
                        )

                        self.experiments[experiment.experiment_id] = experiment

                    except Exception as e:
                        logger.warning(f"Error loading experiment {exp_dir.name}: {e}")

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment."""
        experiment_id = f"exp_{config.experiment_name}_{int(time.time())}"

        experiment = Experiment(
            experiment_id=experiment_id,
            config=config,
            trials=[]
        )

        self.experiments[experiment_id] = experiment
        self.active_experiments[experiment_id] = experiment

        # Save experiment metadata
        exp_dir = self.experiments_dir / experiment_id
        ensure_dir(exp_dir)

        exp_data = {
            'experiment_id': experiment_id,
            'config': asdict(config),
            'trials': [],
            'status': 'running',
            'created_at': experiment.created_at.isoformat(),
            'completed_at': None
        }

        with open(exp_dir / "experiment.json", 'w') as f:
            json.dump(exp_data, f, indent=2, default=str)

        logger.info(f"✅ Created experiment: {experiment_id}")
        return experiment_id

    def add_trial(self, experiment_id: str, trial: ExperimentTrial) -> None:
        """Add a trial to an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        trial.experiment_id = experiment_id
        experiment.trials.append(trial)

        # Update best trial if this one is better
        if experiment.best_trial is None:
            experiment.best_trial = trial
        else:
            # Compare based on primary metric
            primary_metric = experiment.config.metrics[0]
            current_best = experiment.best_trial.metrics.get(primary_metric, 0)
            new_value = trial.metrics.get(primary_metric, 0)

            if experiment.config.optimization_direction == "maximize":
                if new_value > current_best:
                    experiment.best_trial = trial
            else:
                if new_value < current_best:
                    experiment.best_trial = trial

        # Save trial data
        exp_dir = self.experiments_dir / experiment_id
        self._save_experiment(experiment)

    def _save_experiment(self, experiment: Experiment):
        """Save experiment data to disk."""
        exp_dir = self.experiments_dir / experiment.experiment_id

        exp_data = {
            'experiment_id': experiment.experiment_id,
            'config': asdict(experiment.config),
            'trials': [asdict(trial) for trial in experiment.trials],
            'best_trial': asdict(experiment.best_trial) if experiment.best_trial else None,
            'status': experiment.status,
            'created_at': experiment.created_at.isoformat(),
            'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None
        }

        with open(exp_dir / "experiment.json", 'w') as f:
            json.dump(exp_data, f, indent=2, default=str)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.experiments.get(experiment_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with summary information."""
        experiments = []

        for exp_id, exp in self.experiments.items():
            experiments.append({
                'experiment_id': exp_id,
                'name': exp.config.experiment_name,
                'status': exp.status,
                'trials_count': len(exp.trials),
                'best_score': exp.best_trial.metrics.get(exp.config.metrics[0], 0) if exp.best_trial else 0,
                'created_at': exp.created_at.isoformat(),
                'model_types': exp.config.model_types
            })

        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)

    def complete_experiment(self, experiment_id: str):
        """Mark an experiment as completed."""
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            experiment.status = "completed"
            experiment.completed_at = datetime.now()

            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]

            self._save_experiment(experiment)
            logger.info(f"✅ Completed experiment: {experiment_id}")


class EnhancedHyperparameterOptimizer:
    """Enhanced hyperparameter optimization with experiment tracking."""

    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.current_experiment: Optional[str] = None

    def optimize_with_experiment(
        self,
        config: ExperimentConfig,
        data_path: Path,
        models_dir: Path = Path("models"),
        reports_dir: Path = Path("reports")
    ) -> Experiment:
        """Run hyperparameter optimization with full experiment tracking."""

        # Create experiment
        experiment_id = self.experiment_manager.create_experiment(config)
        self.current_experiment = experiment_id

        experiment = self.experiment_manager.get_experiment(experiment_id)

        try:
            typer.echo(f"🔬 Starting experiment: {experiment_id}")
            typer.echo(f"   Model types: {', '.join(config.model_types)}")
            typer.echo(f"   Trials: {config.n_trials}")

            # Import here to avoid circular imports
            from .optimizers import HyperparameterOptimizer

            # Create optimizer instance
            optimizer = HyperparameterOptimizer(
                study_name=f"exp_{experiment_id}",
                storage_path=f"sqlite:///{reports_dir}/experiments.db",
                n_trials=config.n_trials,
                timeout=config.timeout
            )

            # Run optimization for each model type
            for model_type in config.model_types:
                typer.echo(f"\n🎯 Optimizing {model_type}...")

                # Get search space for this model type
                search_space = config.search_spaces.get(model_type, {})

                if not search_space:
                    typer.echo(f"⚠️  No search space defined for {model_type}, skipping")
                    continue

                # Create study for this model type
                study = optimizer.create_study(model_type)

                # Run trials and track them
                for trial_num in range(config.n_trials):
                    try:
                        # Sample parameters
                        params = optimizer.sample_parameters(model_type, trial_num)

                        # Run trial
                        trial_result = self._run_trial(
                            model_type=model_type,
                            params=params,
                            data_path=data_path,
                            models_dir=models_dir,
                            reports_dir=reports_dir,
                            trial_id=f"{experiment_id}_{model_type}_{trial_num}"
                        )

                        # Add to experiment
                        experiment_trial = ExperimentTrial(
                            trial_id=trial_result['trial_id'],
                            experiment_id=experiment_id,
                            parameters=params,
                            metrics=trial_result['metrics'],
                            model_path=trial_result['model_path'],
                            training_time=trial_result['training_time']
                        )

                        self.experiment_manager.add_trial(experiment_id, experiment_trial)

                        # Log progress
                        if trial_num % 10 == 0:
                            typer.echo(f"   Trial {trial_num+1}/{config.n_trials} completed")

                    except Exception as e:
                        logger.error(f"Trial {trial_num} failed: {e}")
                        # Add failed trial
                        failed_trial = ExperimentTrial(
                            trial_id=f"{experiment_id}_{model_type}_{trial_num}_failed",
                            experiment_id=experiment_id,
                            parameters=params,
                            metrics={},
                            status="failed",
                            error_message=str(e)
                        )
                        self.experiment_manager.add_trial(experiment_id, failed_trial)

            # Complete experiment
            self.experiment_manager.complete_experiment(experiment_id)

            typer.echo(f"\n✅ Experiment completed: {experiment_id}")
            typer.echo(f"   Total trials: {len(experiment.trials)}")

            if experiment.best_trial:
                typer.echo("🏆 Best result:"                typer.echo(f"   Model: {experiment.best_trial.parameters.get('model_type', 'unknown')}")
                primary_metric = config.metrics[0]
                best_value = experiment.best_trial.metrics.get(primary_metric, 0)
                typer.echo(f"   {primary_metric.upper()}: {best_value".4f"}")

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            if experiment_id in self.experiment_manager.experiments:
                exp = self.experiment_manager.experiments[experiment_id]
                exp.status = "failed"

        return experiment

    def _run_trial(
        self,
        model_type: str,
        params: Dict[str, Any],
        data_path: Path,
        models_dir: Path,
        reports_dir: Path,
        trial_id: str
    ) -> Dict[str, Any]:
        """Run a single optimization trial."""
        start_time = time.time()

        try:
            # Import here to avoid circular imports
            from .model import train_model, TrainConfig
            from .eval import evaluate_predictions
            from .utils import _mlflow

            # Set up MLflow
            mlflow = _mlflow()
            mlflow.set_tracking_uri(str(Path.cwd() / "mlruns"))
            mlflow.set_experiment("oncotarget-lite-experiments")

            # Create model config
            train_config = TrainConfig(
                model_type=model_type,
                seed=params.get('seed', 42),
                **{k: v for k, v in params.items() if k != 'model_type'}
            )

            # Train model
            with mlflow.start_run(run_name=f"trial_{trial_id}") as run:
                train_result = train_model(
                    processed_dir=data_path.parent,
                    models_dir=models_dir,
                    reports_dir=reports_dir,
                    config=train_config
                )

                # Evaluate model
                eval_result = evaluate_predictions(
                    reports_dir=reports_dir,
                    n_bootstrap=200,  # Reduced for experiments
                    ci=0.95,
                    bins=10,
                    seed=params.get('seed', 42),
                    distributed=True
                )

                training_time = time.time() - start_time

                # Extract metrics
                metrics = {
                    'auroc': eval_result.metrics.auroc,
                    'ap': eval_result.metrics.ap,
                    'accuracy': eval_result.metrics.accuracy,
                    'f1': eval_result.metrics.f1,
                    'brier': eval_result.metrics.brier,
                    'ece': eval_result.metrics.ece,
                    'training_time': training_time
                }

                return {
                    'trial_id': trial_id,
                    'metrics': metrics,
                    'model_path': train_result.model_path,
                    'training_time': training_time,
                    'run_id': run.info.run_id
                }

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            raise


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
        trial_ids = [trial.trial_id for trial in experiment.trials]
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

        ensure_dir(output_dir)

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
                    content.append(f"- {metric.upper()}: {value".4f"}")

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
                f"| {trial.trial_id} | {model_type} | {status_icon} | {score".4f"} | {training_time".2f"}s |"
            )

        if len(experiment.trials) > 20:
            content.append(f"\n*... and {len(experiment.trials) - 20} more trials*")

        return "\n".join(content)


def run_experiment_cmd(
    experiment_name: str = typer.Option(..., help="Name for the experiment"),
    model_types: List[str] = typer.Option(["logreg", "xgb"], help="Model types to optimize"),
    n_trials: int = typer.Option(50, help="Number of optimization trials per model"),
    data_path: Path = typer.Option(Path("data/processed"), help="Path to processed data"),
    config_file: Optional[Path] = typer.Option(None, help="JSON file with experiment configuration"),
) -> None:
    """Run advanced ML experiment with tracking and optimization."""

    # Load configuration
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                config = ExperimentConfig(**config_data)
        except Exception as e:
            typer.echo(f"❌ Error loading config file: {e}")
            return
    else:
        # Create default search spaces
        search_spaces = {
            "logreg": {
                "C": {"type": "loguniform", "low": 1e-4, "high": 1e2},
                "max_iter": {"type": "int", "low": 100, "high": 1000}
            },
            "xgb": {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 0.3}
            }
        }

        config = ExperimentConfig(
            experiment_name=experiment_name,
            model_types=model_types,
            search_spaces=search_spaces,
            n_trials=n_trials
        )

    # Initialize experiment manager
    exp_manager = ExperimentManager()
    optimizer = EnhancedHyperparameterOptimizer(exp_manager)

    # Run experiment
    try:
        experiment = optimizer.optimize_with_experiment(
            config=config,
            data_path=data_path
        )

        typer.echo("
🎉 Experiment completed successfully!"        typer.echo(f"   Experiment ID: {experiment.experiment_id}")
        typer.echo(f"   Total trials: {len(experiment.trials)}")

        if experiment.best_trial:
            primary_metric = config.metrics[0]
            best_score = experiment.best_trial.metrics.get(primary_metric, 0)
            typer.echo(f"   Best {primary_metric.upper()}: {best_score".4f"}")

    except Exception as e:
        typer.echo(f"❌ Experiment failed: {e}")
        raise typer.Exit(1)


def compare_experiments_cmd(
    experiment_ids: List[str] = typer.Option(..., help="Experiment IDs to compare"),
    output_dir: Path = typer.Option(Path("reports/experiment_comparison"), help="Output directory"),
) -> None:
    """Compare multiple experiments and generate comparison report."""

    exp_manager = ExperimentManager()
    dashboard = ExperimentDashboard(exp_manager)

    # Validate experiment IDs
    valid_experiments = []
    for exp_id in experiment_ids:
        if exp_id in exp_manager.experiments:
            valid_experiments.append(exp_manager.experiments[exp_id])
        else:
            typer.echo(f"⚠️  Experiment {exp_id} not found")

    if len(valid_experiments) < 2:
        typer.echo("❌ Need at least 2 valid experiments for comparison")
        return

    typer.echo(f"🔍 Comparing {len(valid_experiments)} experiments...")

    # Generate comparison report
    exp_dir = output_dir / f"comparison_{int(time.time())}"
    ensure_dir(exp_dir)

    # Create comparison visualization
    comparison_fig = dashboard.create_experiment_comparison_plot()

    # Save comparison plot
    comparison_html = exp_dir / "experiment_comparison.html"
    comparison_fig.write_html(str(comparison_html))

    # Generate detailed comparison report
    report_content = [
        "# Experiment Comparison Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Compared Experiments",
        "",
        "| Experiment ID | Name | Status | Trials | Best Score |",
        "|---------------|------|--------|--------|------------|",
    ]

    primary_metric = "auroc"  # Default

    for exp in valid_experiments:
        best_score = exp.best_trial.metrics.get(primary_metric, 0) if exp.best_trial else 0
        report_content.append(
            f"| {exp.experiment_id} | {exp.config.experiment_name} | {exp.status} | {len(exp.trials)} | {best_score".4f"} |"
        )

    report_content.extend([
        "",
        "## Detailed Analysis",
        "",
        "Interactive visualizations are available in the generated HTML files.",
        "",
        "## Recommendations",
        "",
        "Based on the comparison analysis:"
    ])

    # Simple recommendation logic
    best_exp = max(valid_experiments, key=lambda x: x.best_trial.metrics.get(primary_metric, 0) if x.best_trial else 0)

    if best_exp.best_trial:
        report_content.extend([
            f"- **Recommended Experiment:** {best_exp.experiment_id}",
            f"  - Best {primary_metric.upper()}: {best_exp.best_trial.metrics[primary_metric]".4f"}",
            f"  - Model Type: {best_exp.best_trial.parameters.get('model_type', 'unknown')}",
            f"  - Total Trials: {len(best_exp.trials)}"
        ])

    # Save report
    report_file = exp_dir / "experiment_comparison_report.md"
    with open(report_file, 'w') as f:
        f.write("\n".join(report_content))

    typer.echo(f"✅ Comparison report generated: {report_file}")
    typer.echo(f"📊 Interactive plots: {exp_dir}/experiment_comparison.html")


def list_experiments_cmd(
    show_details: bool = typer.Option(False, help="Show detailed experiment information"),
) -> None:
    """List all experiments with summary information."""

    exp_manager = ExperimentManager()
    experiments = exp_manager.list_experiments()

    if not experiments:
        typer.echo("No experiments found")
        return

    typer.echo(f"📋 Experiments ({len(experiments)} total)")
    typer.echo("=" * 80)

    for exp in experiments:
        status_icon = "🏃" if exp['status'] == 'running' else "✅" if exp['status'] == 'completed' else "❌"

        typer.echo(f"\n{status_icon} {exp['name']} ({exp['experiment_id']})")
        typer.echo(f"   Status: {exp['status']}")
        typer.echo(f"   Trials: {exp['trials_count']}")
        typer.echo(f"   Best Score: {exp['best_score']".4f"}")
        typer.echo(f"   Created: {exp['created_at'][:19]}")

        if show_details:
            typer.echo(f"   Model Types: {', '.join(exp['model_types'])}")

        typer.echo()


def experiment_report_cmd(
    experiment_id: str = typer.Option(..., help="Experiment ID to generate report for"),
    output_dir: Path = typer.Option(Path("reports/experiment_reports"), help="Output directory"),
) -> None:
    """Generate detailed report for a specific experiment."""

    exp_manager = ExperimentManager()
    experiment = exp_manager.get_experiment(experiment_id)

    if not experiment:
        typer.echo(f"❌ Experiment {experiment_id} not found")
        return

    typer.echo(f"📋 Generating report for experiment: {experiment_id}")

    dashboard = ExperimentDashboard(exp_manager)
    report_file = dashboard.generate_experiment_report(experiment_id, output_dir)

    typer.echo(f"✅ Report generated: {report_file}")

    # Show key results
    if experiment.best_trial:
        primary_metric = experiment.config.metrics[0]
        best_score = experiment.best_trial.metrics.get(primary_metric, 0)
        typer.echo("
🏆 Best Result:"        typer.echo(f"   {primary_metric.upper()}: {best_score".4f"}")
        typer.echo(f"   Model Type: {experiment.best_trial.parameters.get('model_type', 'unknown')}")
