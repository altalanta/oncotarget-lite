"""Automated model retraining pipeline with intelligent triggers and deployment."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import typer

from .config import profile_overrides
from .monitoring import ModelMonitor, PerformanceSnapshot, DriftAlert
from .utils import ensure_dir, git_commit

logger = logging.getLogger(__name__)


class RetrainTrigger(Enum):
    """Types of triggers for model retraining."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    FEATURE_DRIFT = "feature_drift"


@dataclass
class RetrainConfig:
    """Configuration for automated retraining."""
    # Trigger settings
    enable_performance_trigger: bool = True
    performance_threshold_drop: float = 0.02  # 2% drop in AUROC
    enable_data_drift_trigger: bool = True
    data_drift_threshold: float = 0.1  # KS statistic threshold
    enable_scheduled_trigger: bool = True
    schedule_interval_days: int = 7  # Weekly retraining
    enable_feature_drift_trigger: bool = True
    feature_drift_threshold: float = 0.05  # Feature importance correlation threshold

    # Pipeline settings
    max_retrain_attempts: int = 3
    min_samples_for_retrain: int = 1000
    validation_split: float = 0.2

    # Deployment settings
    auto_deploy_improvements: bool = True
    min_improvement_threshold: float = 0.005  # 0.5% improvement required
    enable_rollback: bool = True
    rollback_window_hours: int = 24


@dataclass
class RetrainResult:
    """Result of a retraining attempt."""
    trigger_type: RetrainTrigger
    timestamp: datetime
    old_model_version: str
    new_model_version: str
    old_performance: Dict[str, float]
    new_performance: Dict[str, float]
    performance_improvement: Dict[str, float]
    deployed: bool
    success: bool
    error_message: Optional[str] = None


class RetrainTriggerDetector(Protocol):
    """Protocol for retraining trigger detection."""

    def should_retrain(self, config: RetrainConfig) -> tuple[bool, str, RetrainTrigger]:
        """Check if retraining should be triggered. Returns (should_retrain, reason, trigger_type)."""
        ...


class PerformanceTrigger:
    """Detects performance degradation in production models."""

    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor

    def should_retrain(self, config: RetrainConfig) -> tuple[bool, str, RetrainTrigger]:
        """Check if performance has degraded significantly."""
        try:
            report = self.monitor.get_monitoring_report(days=7)  # Last week

            if "error" in report or report["snapshots_count"] < 2:
                return False, "Insufficient performance data", RetrainTrigger.PERFORMANCE_DEGRADATION

            latest = report["latest_performance"]
            baseline = report["baseline_performance"]

            if not latest or not baseline:
                return False, "Missing performance data", RetrainTrigger.PERFORMANCE_DEGRADATION

            # Check for significant drops in key metrics
            for metric in ["auroc", "ap"]:
                if metric in latest and metric in baseline:
                    drop = baseline[metric] - latest[metric]
                    if drop >= config.performance_threshold_drop:
                        return True, f"{metric.upper()} dropped by {drop:.3f}", RetrainTrigger.PERFORMANCE_DEGRADATION

            return False, "Performance within acceptable range", RetrainTrigger.PERFORMANCE_DEGRADATION

        except Exception as e:
            logger.error(f"Error checking performance trigger: {e}")
            return False, f"Error: {e}", RetrainTrigger.PERFORMANCE_DEGRADATION


class DataDriftTrigger:
    """Detects significant changes in data distribution."""

    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor

    def should_retrain(self, config: RetrainConfig) -> tuple[bool, str, RetrainTrigger]:
        """Check for data drift using statistical tests."""
        try:
            # Get recent predictions for drift analysis
            alerts = self.monitor.get_recent_alerts(days=3)

            drift_alerts = [alert for alert in alerts if "drift" in alert.alert_type.lower()]

            if drift_alerts:
                max_severity = max(alert.severity for alert in drift_alerts)
                if max_severity == "high":
                    return True, f"High severity data drift detected: {len(drift_alerts)} alerts", RetrainTrigger.DATA_DRIFT

            return False, "No significant data drift detected", RetrainTrigger.DATA_DRIFT

        except Exception as e:
            logger.error(f"Error checking data drift trigger: {e}")
            return False, f"Error: {e}", RetrainTrigger.DATA_DRIFT


class ScheduledTrigger:
    """Triggers retraining on a schedule."""

    def __init__(self):
        self.db_path = Path("reports/automated_retraining.db")

    def should_retrain(self, config: RetrainConfig) -> tuple[bool, str, RetrainTrigger]:
        """Check if it's time for scheduled retraining."""
        if not config.enable_scheduled_trigger:
            return False, "Scheduled retraining disabled", RetrainTrigger.SCHEDULED

        try:
            ensure_dir(self.db_path.parent)

            with sqlite3.connect(self.db_path) as conn:
                # Create table if it doesn't exist
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS retrain_schedule (
                        last_retrain TIMESTAMP,
                        schedule_interval_days INTEGER
                    )
                """)

                # Get last retrain time
                cursor = conn.execute("SELECT last_retrain FROM retrain_schedule LIMIT 1")
                row = cursor.fetchone()

                if row:
                    last_retrain = datetime.fromisoformat(row[0])
                    next_retrain = last_retrain + timedelta(days=config.schedule_interval_days)
                    now = datetime.now()

                    if now >= next_retrain:
                        return True, f"Scheduled retraining due (last: {last_retrain.date()})", RetrainTrigger.SCHEDULED
                    else:
                        return False, f"Next retraining scheduled for {next_retrain.date()}", RetrainTrigger.SCHEDULED
                else:
                    # No previous retrain, schedule first one
                    return True, "First scheduled retraining", RetrainTrigger.SCHEDULED

        except Exception as e:
            logger.error(f"Error checking scheduled trigger: {e}")
            return False, f"Error: {e}", RetrainTrigger.SCHEDULED


class AutomatedRetrainingPipeline:
    """Main pipeline for automated model retraining."""

    def __init__(self, config: RetrainConfig):
        self.config = config
        self.monitor = ModelMonitor()
        self.triggers = [
            PerformanceTrigger(self.monitor),
            DataDriftTrigger(self.monitor),
            ScheduledTrigger(),
        ]
        self.db_path = Path("reports/automated_retraining.db")

    def check_retrain_needed(self) -> tuple[bool, str, RetrainTrigger]:
        """Check if any trigger indicates retraining is needed."""
        for trigger in self.triggers:
            should_retrain, reason, trigger_type = trigger.should_retrain(self.config)
            if should_retrain:
                return True, reason, trigger_type

        return False, "No triggers activated", RetrainTrigger.MANUAL

    def run_retraining(self, trigger_type: RetrainTrigger, reason: str) -> RetrainResult:
        """Execute the complete retraining pipeline."""
        logger.info(f"Starting automated retraining triggered by: {trigger_type.value}")
        logger.info(f"Trigger reason: {reason}")

        # Get current model info
        context = self._load_run_context()
        if not context:
            return RetrainResult(
                trigger_type=trigger_type,
                timestamp=datetime.now(),
                old_model_version="unknown",
                new_model_version="unknown",
                old_performance={},
                new_performance={},
                performance_improvement={},
                deployed=False,
                success=False,
                error_message="No run context found"
            )

        old_model_version = context.get("run_id", "unknown")

        # Get baseline performance
        baseline_performance = self._get_baseline_performance()

        try:
            # Step 1: Prepare fresh data
            logger.info("Step 1: Preparing fresh dataset...")
            data_result = self._prepare_data()

            if not data_result["success"]:
                raise Exception(f"Data preparation failed: {data_result['error']}")

            # Step 2: Train new model
            logger.info("Step 2: Training new model...")
            train_result = self._train_model(data_result["dataset_fingerprint"])

            # Step 3: Evaluate new model
            logger.info("Step 3: Evaluating new model...")
            eval_result = self._evaluate_model(train_result["run_id"])

            # Step 4: Compare with baseline
            logger.info("Step 4: Comparing with baseline...")
            comparison = self._compare_models(baseline_performance, eval_result["performance"])

            # Step 5: Deploy if improved
            deployed = False
            if self._should_deploy(comparison):
                logger.info("Step 5: Deploying improved model...")
                deploy_result = self._deploy_model(train_result["run_id"])
                deployed = deploy_result["success"]

                if not deployed:
                    logger.warning(f"Deployment failed: {deploy_result['error']}")
            else:
                logger.info("Model not deployed - performance not significantly improved")

            # Step 6: Update schedule
            self._update_schedule()

            return RetrainResult(
                trigger_type=trigger_type,
                timestamp=datetime.now(),
                old_model_version=old_model_version,
                new_model_version=train_result["run_id"],
                old_performance=baseline_performance,
                new_performance=eval_result["performance"],
                performance_improvement=comparison,
                deployed=deployed,
                success=True
            )

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return RetrainResult(
                trigger_type=trigger_type,
                timestamp=datetime.now(),
                old_model_version=old_model_version,
                new_model_version="unknown",
                old_performance=baseline_performance,
                new_performance={},
                performance_improvement={},
                deployed=False,
                success=False,
                error_message=str(e)
            )

    def _prepare_data(self) -> Dict[str, Any]:
        """Prepare fresh dataset for retraining."""
        try:
            from .data import PROCESSED_DIR, RAW_DIR, prepare_dataset

            result = prepare_dataset(
                raw_dir=RAW_DIR,
                processed_dir=PROCESSED_DIR,
                test_size=0.3,
                seed=42
            )

            return {
                "success": True,
                "dataset_fingerprint": result.dataset_fingerprint,
                "features_path": str(PROCESSED_DIR / "features.parquet"),
                "labels_path": str(PROCESSED_DIR / "labels.parquet"),
                "splits_path": str(PROCESSED_DIR / "splits.json")
            }

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return {"success": False, "error": str(e)}

    def _train_model(self, dataset_fingerprint: str) -> Dict[str, Any]:
        """Train a new model with current best practices."""
        try:
            from .utils import _mlflow
            from .model import MODELS_DIR, train_model, TrainConfig

            mlflow = _mlflow()
            mlflow.set_tracking_uri(str(Path.cwd() / "mlruns"))
            mlflow.set_experiment("oncotarget-lite-automated")

            # Use best model type and hyperparameters from previous runs
            best_config = self._get_best_hyperparameters()

            with mlflow.start_run(run_name=f"automated_retrain_{int(time.time())}") as run:
                train_result = train_model(
                    processed_dir=Path("data/processed"),
                    models_dir=MODELS_DIR,
                    reports_dir=Path("reports"),
                    config=TrainConfig(
                        model_type=best_config["model_type"],
                        C=best_config.get("C", 1.0),
                        max_iter=best_config.get("max_iter", 500),
                        seed=42
                    )
                )

                # Register the model in the MLflow Model Registry
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, "oncotarget-lite")
                
                return {
                    "success": True,
                    "run_id": run.info.run_id,
                    "model_path": str(train_result.model_path),
                    "dataset_hash": dataset_fingerprint
                }

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"success": False, "error": str(e)}

    def _evaluate_model(self, run_id: str) -> Dict[str, Any]:
        """Evaluate the newly trained model."""
        try:
            from .utils import _mlflow
            from .eval import evaluate_predictions

            mlflow = _mlflow()
            mlflow.set_tracking_uri(str(Path.cwd() / "mlruns"))
            mlflow.set_experiment("oncotarget-lite-automated")

            with mlflow.start_run(run_id=run_id):
                eval_result = evaluate_predictions(
                    reports_dir=Path("reports"),
                    n_bootstrap=500,  # Reduced for automated runs
                    ci=0.95,
                    bins=10,
                    seed=42,
                    distributed=True
                )

                return {
                    "success": True,
                    "performance": {
                        "auroc": eval_result.metrics.auroc,
                        "ap": eval_result.metrics.ap,
                        "accuracy": eval_result.metrics.accuracy,
                        "f1": eval_result.metrics.f1
                    }
                }

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {"success": False, "error": str(e)}

    def _compare_models(self, baseline: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
        """Compare new model performance with baseline."""
        comparison = {}

        for metric in ["auroc", "ap", "accuracy", "f1"]:
            if metric in baseline and metric in new:
                improvement = new[metric] - baseline[metric]
                comparison[metric] = improvement
                comparison[f"{metric}_relative"] = improvement / baseline[metric] if baseline[metric] > 0 else 0

        return comparison

    def _should_deploy(self, comparison: Dict[str, float]) -> bool:
        """Determine if new model should be deployed."""
        if not self.config.auto_deploy_improvements:
            return False

        # Check if there's meaningful improvement in primary metrics
        primary_metrics = ["auroc", "ap"]

        for metric in primary_metrics:
            if metric in comparison:
                improvement = comparison[metric]
                if improvement >= self.config.min_improvement_threshold:
                    logger.info(f"Model improvement detected: {metric} +{improvement:.4f}")
                    return True

        logger.info("No significant improvement detected")
        return False

    def _deploy_model(self, run_id: str) -> Dict[str, Any]:
        """Deploy the new model using the enhanced deployment system."""
        try:
            from .model_deployment import create_model_version, deploy_to_production

            # Get model performance metrics from the evaluation results
            performance_metrics = self._get_model_performance(run_id)

            # Create versioned model deployment
            version_id = create_model_version(
                run_id=run_id,
                model_path=Path("models") / "logreg_pipeline.pkl",
                performance_metrics=performance_metrics,
                model_type="logreg",
                feature_names=[],  # Would need to extract from actual model
                is_production=True
            )

            # Deploy to production
            success = deploy_to_production(version_id, confirm=False)

            if success:
                logger.info(f"Successfully deployed model version: {version_id}")
                return {"success": True, "version_id": version_id}
            else:
                logger.error("Failed to deploy model to production")
                return {"success": False, "error": "Deployment to production failed"}

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_model_performance(self, run_id: str) -> Dict[str, float]:
        """Get performance metrics for a model run."""
        # For now, return the new performance metrics from the retraining result
        # In a real implementation, this would query MLflow or evaluation results
        return {
            "auroc": self.new_performance.get("auroc", 0),
            "ap": self.new_performance.get("ap", 0),
            "accuracy": self.new_performance.get("accuracy", 0),
            "f1": self.new_performance.get("f1", 0)
        }

    def _backup_current_production_model(self, production_path: Path) -> bool:
        """Backup current production model for rollback."""
        try:
            deployment_file = production_path / "deployment.json"
            if not deployment_file.exists():
                return True  # No current model to backup

            current_info = json.loads(deployment_file.read_text())
            current_version = current_info.get("model_version")

            if current_version:
                # Create backup directory with timestamp
                backup_dir = production_path / "backups" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ensure_dir(backup_dir)

                # Copy current model files (this would need to be implemented based on actual model storage)
                # For now, just copy the deployment metadata
                import shutil
                shutil.copy2(deployment_file, backup_dir / "deployment.json")

                logger.info(f"Backed up model version {current_version} to {backup_dir}")
                return True

            return True

        except Exception as e:
            logger.error(f"Failed to backup production model: {e}")
            return False

    def _get_current_production_version(self) -> Optional[str]:
        """Get the current production model version."""
        try:
            deployment_file = Path("models/production/deployment.json")
            if deployment_file.exists():
                info = json.loads(deployment_file.read_text())
                return info.get("model_version")
            return None
        except Exception as e:
            logger.error(f"Error getting current production version: {e}")
            return None

    def rollback_model(self, hours_back: int = 24) -> Dict[str, Any]:
        """Rollback to a previous model version."""
        try:
            production_path = Path("models/production")
            backup_dir = production_path / "backups"

            if not backup_dir.exists():
                return {"success": False, "error": "No backup directory found"}

            # Find the most recent backup within the specified time window
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            backup_dirs = []

            for backup in backup_dir.iterdir():
                if backup.is_dir() and backup.name.startswith("backup_"):
                    try:
                        backup_time_str = backup.name.replace("backup_", "")
                        backup_time = datetime.strptime(backup_time_str, "%Y%m%d_%H%M%S")

                        if backup_time >= cutoff_time:
                            backup_dirs.append((backup, backup_time))
                    except ValueError:
                        continue

            if not backup_dirs:
                return {"success": False, "error": f"No suitable backups found within {hours_back} hours"}

            # Use the most recent backup
            backup_dirs.sort(key=lambda x: x[1], reverse=True)
            latest_backup, backup_time = backup_dirs[0]

            # Restore from backup (this would need to be implemented based on actual model storage)
            # For now, just restore the deployment metadata
            import shutil
            backup_deployment = latest_backup / "deployment.json"
            current_deployment = production_path / "deployment.json"

            if backup_deployment.exists():
                shutil.copy2(backup_deployment, current_deployment)
                logger.info(f"Rolled back to model version from {backup_time}")
                return {"success": True, "rollback_version": backup_time.isoformat()}

            return {"success": False, "error": "Backup deployment file not found"}

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {"success": False, "error": str(e)}

    def _get_baseline_performance(self) -> Dict[str, float]:
        """Get baseline performance from current production model."""
        try:
            # Try to get from monitoring data first
            report = self.monitor.get_monitoring_report(days=1)

            if "error" not in report and report["snapshots_count"] > 0:
                latest = report["latest_performance"]
                if latest:
                    return {
                        "auroc": latest.get("auroc", 0),
                        "ap": latest.get("ap", 0),
                        "accuracy": latest.get("accuracy", 0),
                        "f1": latest.get("f1", 0)
                    }

            # Fallback to evaluation metrics
            metrics_path = Path("reports/metrics.json")
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    return {
                        "auroc": data.get("auroc", 0),
                        "ap": data.get("ap", 0),
                        "accuracy": data.get("accuracy", 0),
                        "f1": data.get("f1", 0)
                    }

            return {}

        except Exception as e:
            logger.error(f"Error getting baseline performance: {e}")
            return {}

    def _get_best_hyperparameters(self) -> Dict[str, Any]:
        """Get best hyperparameters from previous optimization runs."""
        try:
            # Look for latest optuna results
            optuna_files = list(Path("reports").glob("optuna_summary_*.json"))

            if optuna_files:
                # Get most recent file
                latest_file = max(optuna_files, key=lambda p: p.stat().st_mtime)

                with open(latest_file, 'r') as f:
                    data = json.load(f)

                    if "best_params" in data:
                        best_params = data["best_params"].copy()
                        best_params["model_type"] = latest_file.stem.replace("optuna_summary_", "")
                        return best_params

            # Default fallback
            return {
                "model_type": "logreg",
                "C": 1.0,
                "max_iter": 500
            }

        except Exception as e:
            logger.error(f"Error getting best hyperparameters: {e}")
            return {"model_type": "logreg", "C": 1.0, "max_iter": 500}

    def _update_schedule(self):
        """Update the retraining schedule."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO retrain_schedule (last_retrain, schedule_interval_days)
                    VALUES (?, ?)
                """, (datetime.now().isoformat(), self.config.schedule_interval_days))

        except Exception as e:
            logger.error(f"Error updating schedule: {e}")

    def _load_run_context(self) -> Optional[Dict[str, str]]:
        """Load current run context."""
        context_path = Path("reports/run_context.json")
        if context_path.exists():
            try:
                return json.loads(context_path.read_text())
            except Exception as e:
                logger.error(f"Error loading run context: {e}")
        return None


def run_automated_retraining(
    config_path: Optional[Path] = None,
    dry_run: bool = False,
    force: bool = False
) -> None:
    """Main entry point for automated retraining."""

    # Load configuration
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            config = RetrainConfig(**config_data)
    else:
        config = RetrainConfig()

    # Initialize pipeline
    pipeline = AutomatedRetrainingPipeline(config)

    # Check if retraining is needed
    should_retrain, reason, trigger_type = pipeline.check_retrain_needed()

    if not should_retrain and not force:
        typer.echo(f"✅ No retraining needed: {reason}")
        return

    if dry_run:
        typer.echo(f"🔍 Dry run: Would retrain due to {trigger_type.value}")
        typer.echo(f"Reason: {reason}")
        return

    # Run retraining
    typer.echo(f"🔄 Starting automated retraining...")
    typer.echo(f"Trigger: {trigger_type.value}")
    typer.echo(f"Reason: {reason}")

    result = pipeline.run_retraining(trigger_type, reason)

    if result.success:
        typer.echo("✅ Retraining completed successfully")
        typer.echo(f"   Old model: {result.old_model_version}")
        typer.echo(f"   New model: {result.new_model_version}")
        typer.echo(f"   Deployed: {result.deployed}")

        if result.performance_improvement:
            typer.echo("   Performance changes:")
            for metric, improvement in result.performance_improvement.items():
                if not metric.endswith("_relative"):
                    typer.echo(f"     {metric.upper()}: {improvement:+.4f}")
    else:
        typer.echo(f"❌ Retraining failed: {result.error_message}")
        raise typer.Exit(1)


# CLI command integration
def retrain_command(
    config: Optional[Path] = typer.Option(None, help="Retraining configuration file"),
    dry_run: bool = typer.Option(False, help="Show what would be done without executing"),
    force: bool = typer.Option(False, help="Force retraining even if no triggers are active"),
    schedule: bool = typer.Option(False, help="Run in scheduled mode (check triggers automatically)"),
) -> None:
    """Automated model retraining with intelligent triggers."""

    if schedule:
        # Run in scheduled mode - check triggers and retrain if needed
        run_automated_retraining(config_path=config, dry_run=dry_run, force=force)
    else:
        # Manual retraining - always run
        run_automated_retraining(config_path=config, dry_run=dry_run, force=True)
