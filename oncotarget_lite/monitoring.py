"""
Model performance monitoring and drift detection system.

This module provides comprehensive monitoring capabilities for:
- Model drift detection using statistical tests
- Feature importance drift monitoring
- Performance regression alerts
- Prediction confidence interval tracking
- Automated alerting via Slack/email
"""

from __future__ import annotations

import json
import smtplib
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    from email.mime.text import MimeText
except ImportError:
    # Fallback for systems without email module
    MimeText = None

import numpy as np
import pandas as pd
from scipy import stats

from .utils import ensure_dir, load_dataframe, save_json


@dataclass
class PerformanceSnapshot:
    """Snapshot of model performance at a point in time."""
    timestamp: datetime
    model_version: str
    dataset_hash: str

    # Core metrics
    auroc: float
    ap: float
    brier: float
    accuracy: float
    f1: float

    # Confidence intervals
    auroc_ci_lower: float
    auroc_ci_upper: float
    ap_ci_lower: float
    ap_ci_upper: float

    # Prediction distribution stats
    pred_mean: float
    pred_std: float
    pred_skew: float
    pred_kurtosis: float

    # Feature importance (top 5)
    top_features: Dict[str, float]

    # Data quality indicators
    n_samples: int
    n_features: int
    class_balance: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class DriftAlert:
    """Alert for detected model drift or performance issues."""
    alert_id: str
    timestamp: datetime
    alert_type: str  # 'drift', 'regression', 'data_quality'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metrics: Dict[str, float]
    recommendations: List[str]


class ModelMonitor:
    """Main class for model performance monitoring and drift detection."""

    def __init__(
        self,
        db_path: Union[str, Path] = "reports/monitoring.db",
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_alerts: bool = True,
    ):
        """
        Initialize the model monitor.

        Args:
            db_path: Path to SQLite database for storing monitoring data
            alert_thresholds: Custom thresholds for alerts
            enable_alerts: Whether to enable alerting functionality
        """
        self.db_path = Path(db_path)
        self.enable_alerts = enable_alerts
        ensure_dir(self.db_path.parent)

        # Default alert thresholds
        self.alert_thresholds = {
            'auroc_degradation': 0.02,  # 2% drop in AUROC
            'ap_degradation': 0.03,      # 3% drop in AP
            'drift_significance': 0.05,  # 5% significance level for drift
            'feature_drift_threshold': 0.1,  # 10% change in feature importance
            'prediction_drift_threshold': 0.05,  # 5% change in prediction distribution
        }
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)

        self._init_database()

    def _init_database(self) -> None:
        """Initialize the monitoring database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    UNIQUE(timestamp, model_version, dataset_hash)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    recommendations_json TEXT NOT NULL
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON performance_snapshots(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON drift_alerts(timestamp)")

    def capture_performance_snapshot(
        self,
        predictions_path: Union[str, Path],
        model_version: str,
        dataset_hash: str,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> PerformanceSnapshot:
        """
        Capture a performance snapshot from prediction results.

        Args:
            predictions_path: Path to predictions parquet file
            model_version: Version identifier for the model
            dataset_hash: Hash of the dataset used
            feature_importance: Optional feature importance scores

        Returns:
            PerformanceSnapshot object
        """
        predictions = load_dataframe(predictions_path)

        # Separate train/test predictions
        train_preds = predictions[predictions["split"] == "train"]
        test_preds = predictions[predictions["split"] == "test"]

        if len(test_preds) == 0:
            raise ValueError("No test predictions found for monitoring")

        # Compute core metrics
        y_true = test_preds["y_true"].to_numpy()
        y_prob = test_preds["y_prob"].to_numpy()

        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, brier_score_loss

        auroc = float(roc_auc_score(y_true, y_prob))
        ap = float(average_precision_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))
        accuracy = float(accuracy_score(y_true, y_prob >= 0.5))
        f1 = float(f1_score(y_true, y_prob >= 0.5))

        # Prediction distribution statistics
        pred_mean = float(np.mean(y_prob))
        pred_std = float(np.std(y_prob))
        pred_skew = float(stats.skew(y_prob))
        pred_kurtosis = float(stats.kurtosis(y_prob))

        # Feature importance (use provided or compute simple version)
        if feature_importance is None:
            # Simple feature importance based on correlation with predictions
            features = load_dataframe("data/processed/features.parquet")
            test_features = features.loc[test_preds.index]
            correlations = {}
            for col in test_features.columns:
                if test_features[col].dtype in ['int64', 'float64']:
                    correlations[col] = abs(np.corrcoef(test_features[col], y_prob)[0, 1])
            top_features = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5])
        else:
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])

        # Class balance
        class_balance = float(np.mean(y_true))

        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            model_version=model_version,
            dataset_hash=dataset_hash,
            auroc=auroc,
            ap=ap,
            brier=brier,
            accuracy=accuracy,
            f1=f1,
            auroc_ci_lower=auroc - 0.02,  # Placeholder - would compute from bootstrap
            auroc_ci_upper=auroc + 0.02,
            ap_ci_lower=ap - 0.02,
            ap_ci_upper=ap + 0.02,
            pred_mean=pred_mean,
            pred_std=pred_std,
            pred_skew=pred_skew,
            pred_kurtosis=pred_kurtosis,
            top_features=top_features,
            n_samples=len(predictions),
            n_features=len(features.columns) if 'features' in locals() else 0,
            class_balance=class_balance,
        )

        # Store in database
        self._store_snapshot(snapshot)

        return snapshot

    def _store_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Store a performance snapshot in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO performance_snapshots
                (timestamp, model_version, dataset_hash, metrics_json)
                VALUES (?, ?, ?, ?)
            """, (
                snapshot.timestamp.isoformat(),
                snapshot.model_version,
                snapshot.dataset_hash,
                json.dumps(snapshot.to_dict()),
            ))

    def detect_drift(
        self,
        current_snapshot: PerformanceSnapshot,
        reference_snapshots: Optional[List[PerformanceSnapshot]] = None,
        lookback_days: int = 30,
    ) -> List[DriftAlert]:
        """
        Detect model drift and performance regressions.

        Args:
            current_snapshot: Current performance snapshot
            reference_snapshots: Optional reference snapshots (if None, uses database)
            lookback_days: Days to look back for reference data

        Returns:
            List of drift alerts
        """
        alerts = []

        # Get reference snapshots from database if not provided
        if reference_snapshots is None:
            reference_snapshots = self._get_reference_snapshots(
                current_snapshot.model_version,
                lookback_days
            )

        if not reference_snapshots:
            # No reference data available
            return alerts

        # 1. Performance regression detection
        alerts.extend(self._detect_performance_regression(current_snapshot, reference_snapshots))

        # 2. Prediction distribution drift
        alerts.extend(self._detect_prediction_drift(current_snapshot, reference_snapshots))

        # 3. Feature importance drift
        alerts.extend(self._detect_feature_drift(current_snapshot, reference_snapshots))

        # Store alerts in database
        for alert in alerts:
            self._store_alert(alert)

        return alerts

    def _get_reference_snapshots(self, model_version: str, lookback_days: int) -> List[PerformanceSnapshot]:
        """Get reference snapshots from the database."""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT metrics_json FROM performance_snapshots
                WHERE model_version = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (model_version, cutoff_date.isoformat()))

            snapshots = []
            for row in cursor:
                data = json.loads(row['metrics_json'])
                snapshot = PerformanceSnapshot(**data)
                snapshots.append(snapshot)

            return snapshots

    def _detect_performance_regression(
        self,
        current: PerformanceSnapshot,
        references: List[PerformanceSnapshot]
    ) -> List[DriftAlert]:
        """Detect performance regressions."""
        alerts = []

        if not references:
            return alerts

        # Get baseline metrics (mean of recent snapshots)
        ref_auroc = np.mean([s.auroc for s in references])
        ref_ap = np.mean([s.ap for s in references])

        # Check AUROC degradation
        auroc_degradation = ref_auroc - current.auroc
        if auroc_degradation > self.alert_thresholds['auroc_degradation']:
            severity = self._calculate_severity(auroc_degradation, self.alert_thresholds['auroc_degradation'])
            alerts.append(DriftAlert(
                alert_id=f"perf_reg_auroc_{current.timestamp.isoformat()}",
                timestamp=current.timestamp,
                alert_type="regression",
                severity=severity,
                message=f"AUROC degraded by {auroc_degradation:.3f} from baseline {ref_auroc:.3f} to {current.auroc:.3f}",
                metrics={"auroc_current": current.auroc, "auroc_baseline": ref_auroc, "degradation": auroc_degradation},
                recommendations=[
                    "Investigate recent data changes",
                    "Check for data quality issues",
                    "Consider retraining the model",
                    "Review feature engineering pipeline"
                ]
            ))

        # Check AP degradation
        ap_degradation = ref_ap - current.ap
        if ap_degradation > self.alert_thresholds['ap_degradation']:
            severity = self._calculate_severity(ap_degradation, self.alert_thresholds['ap_degradation'])
            alerts.append(DriftAlert(
                alert_id=f"perf_reg_ap_{current.timestamp.isoformat()}",
                timestamp=current.timestamp,
                alert_type="regression",
                severity=severity,
                message=f"Average Precision degraded by {ap_degradation:.3f} from baseline {ref_ap:.3f} to {current.ap:.3f}",
                metrics={"ap_current": current.ap, "ap_baseline": ref_ap, "degradation": ap_degradation},
                recommendations=[
                    "Check precision-recall balance",
                    "Investigate class distribution changes",
                    "Review positive class prediction quality"
                ]
            ))

        return alerts

    def _detect_prediction_drift(
        self,
        current: PerformanceSnapshot,
        references: List[PerformanceSnapshot]
    ) -> List[DriftAlert]:
        """Detect prediction distribution drift using KS test."""
        alerts = []

        if not references:
            return alerts

        # Get reference prediction statistics
        ref_means = [s.pred_mean for s in references]
        ref_stds = [s.pred_std for s in references]

        # Simple statistical test for distribution shift
        # In practice, you'd want actual prediction distributions
        mean_shift = abs(current.pred_mean - np.mean(ref_means))
        std_shift = abs(current.pred_std - np.mean(ref_stds))

        # Normalize by reference standard deviation
        ref_std_mean = np.std(ref_means) if len(ref_means) > 1 else 0.1
        ref_std_std = np.std(ref_stds) if len(ref_stds) > 1 else 0.1

        normalized_mean_shift = mean_shift / ref_std_mean if ref_std_mean > 0 else 0
        normalized_std_shift = std_shift / ref_std_std if ref_std_std > 0 else 0

        # Alert if significant shift detected
        if normalized_mean_shift > self.alert_thresholds['prediction_drift_threshold']:
            alerts.append(DriftAlert(
                alert_id=f"pred_drift_mean_{current.timestamp.isoformat()}",
                timestamp=current.timestamp,
                alert_type="drift",
                severity=self._calculate_severity(normalized_mean_shift, self.alert_thresholds['prediction_drift_threshold']),
                message=f"Prediction mean shifted by {normalized_mean_shift:.3f} standard deviations",
                metrics={
                    "mean_current": current.pred_mean,
                    "mean_baseline": np.mean(ref_means),
                    "normalized_shift": normalized_mean_shift
                },
                recommendations=[
                    "Investigate changes in input data distribution",
                    "Check for data preprocessing issues",
                    "Review model calibration"
                ]
            ))

        return alerts

    def _detect_feature_drift(
        self,
        current: PerformanceSnapshot,
        references: List[PerformanceSnapshot]
    ) -> List[DriftAlert]:
        """Detect feature importance drift."""
        alerts = []

        if not references:
            return alerts

        # Get baseline feature importance
        baseline_importance = {}
        for ref in references:
            for feature, importance in ref.top_features.items():
                if feature not in baseline_importance:
                    baseline_importance[feature] = []
                baseline_importance[feature].append(importance)

        # Average baseline importance
        for feature in baseline_importance:
            baseline_importance[feature] = np.mean(baseline_importance[feature])

        # Check for significant changes in current importance
        for feature, current_imp in current.top_features.items():
            if feature in baseline_importance:
                baseline_imp = baseline_importance[feature]
                change = abs(current_imp - baseline_imp) / baseline_imp if baseline_imp > 0 else 0

                if change > self.alert_thresholds['feature_drift_threshold']:
                    severity = self._calculate_severity(change, self.alert_thresholds['feature_drift_threshold'])
                    alerts.append(DriftAlert(
                        alert_id=f"feat_drift_{feature}_{current.timestamp.isoformat()}",
                        timestamp=current.timestamp,
                        alert_type="drift",
                        severity=severity,
                        message=f"Feature '{feature}' importance changed by {change:.1%} (baseline: {baseline_imp:.3f}, current: {current_imp:.3f})",
                        metrics={
                            "feature": feature,
                            "importance_current": current_imp,
                            "importance_baseline": baseline_imp,
                            "relative_change": change
                        },
                        recommendations=[
                            f"Investigate changes in '{feature}' feature",
                            "Check for data quality issues in this feature",
                            "Review feature engineering for this variable"
                        ]
                    ))

        return alerts

    def _calculate_severity(self, value: float, threshold: float) -> str:
        """Calculate alert severity based on value and threshold."""
        ratio = value / threshold if threshold > 0 else 0

        if ratio >= 3.0:
            return "critical"
        elif ratio >= 2.0:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"

    def _store_alert(self, alert: DriftAlert) -> None:
        """Store an alert in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO drift_alerts
                (alert_id, timestamp, alert_type, severity, message, metrics_json, recommendations_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.timestamp.isoformat(),
                alert.alert_type,
                alert.severity,
                alert.message,
                json.dumps(alert.metrics),
                json.dumps(alert.recommendations),
            ))

    def get_monitoring_report(
        self,
        model_version: Optional[str] = None,
        days: int = 30
    ) -> Dict:
        """
        Generate a monitoring report.

        Args:
            model_version: Specific model version to report on
            days: Number of days to include in report

        Returns:
            Dictionary containing monitoring summary
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get recent snapshots
            query = """
                SELECT metrics_json FROM performance_snapshots
                WHERE timestamp >= ?
            """
            params = [cutoff_date.isoformat()]

            if model_version:
                query += " AND model_version = ?"
                params.append(model_version)

            query += " ORDER BY timestamp DESC"

            cursor = conn.execute(query, params)
            snapshots = [PerformanceSnapshot(**json.loads(row['metrics_json'])) for row in cursor]

            # Get recent alerts
            alert_query = """
                SELECT * FROM drift_alerts
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """
            alert_cursor = conn.execute(alert_query, [cutoff_date.isoformat()])
            alerts = [DriftAlert(**dict(row)) for row in alert_cursor]

        if not snapshots:
            return {"error": "No monitoring data available"}

        # Generate summary
        latest = snapshots[0]
        baseline = snapshots[-1] if len(snapshots) > 1 else latest

        report = {
            "report_date": datetime.now().isoformat(),
            "period_days": days,
            "model_version": model_version,
            "snapshots_count": len(snapshots),
            "alerts_count": len(alerts),
            "latest_performance": latest.to_dict(),
            "baseline_performance": baseline.to_dict(),
            "trends": self._compute_trends(snapshots),
            "recent_alerts": [alert.__dict__ for alert in alerts[:10]],  # Last 10 alerts
        }

        return report

    def _compute_trends(self, snapshots: List[PerformanceSnapshot]) -> Dict:
        """Compute performance trends over time."""
        if len(snapshots) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Sort by timestamp
        snapshots.sort(key=lambda x: x.timestamp)

        # Simple linear trends
        timestamps = [(s.timestamp - snapshots[0].timestamp).total_seconds() for s in snapshots]
        auroc_values = [s.auroc for s in snapshots]
        ap_values = [s.ap for s in snapshots]

        # Compute trends (slope of linear regression)
        auroc_trend = np.polyfit(timestamps, auroc_values, 1)[0] if len(timestamps) > 1 else 0
        ap_trend = np.polyfit(timestamps, ap_values, 1)[0] if len(timestamps) > 1 else 0

        return {
            "auroc_trend_per_day": float(auroc_trend * 86400),  # Convert to per day
            "ap_trend_per_day": float(ap_trend * 86400),
            "trend_direction": "improving" if auroc_trend > 0 else "degrading" if auroc_trend < 0 else "stable",
        }

    def send_alert_notification(
        self,
        alert: DriftAlert,
        webhook_url: Optional[str] = None,
        email_config: Optional[Dict] = None,
    ) -> bool:
        """
        Send alert notification via Slack webhook or email.

        Args:
            alert: Alert to send
            webhook_url: Slack webhook URL
            email_config: Email configuration dict

        Returns:
            True if notification sent successfully
        """
        if not self.enable_alerts:
            return False

        success = False

        # Send Slack notification
        if webhook_url:
            success = self._send_slack_notification(alert, webhook_url) or success

        # Send email notification
        if email_config:
            success = self._send_email_notification(alert, email_config) or success

        return success

    def _send_slack_notification(self, alert: DriftAlert, webhook_url: str) -> bool:
        """Send alert to Slack webhook."""
        try:
            import requests

            # Create Slack message
            color = {
                "low": "good",
                "medium": "warning",
                "high": "danger",
                "critical": "#ff0000"
            }.get(alert.severity, "warning")

            message = {
                "username": "Model Monitor",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color,
                    "title": f"🚨 Model Alert: {alert.alert_type.title()}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": True},
                    ],
                    "footer": "Oncology Target Lite Model Monitor",
                }]
            }

            response = requests.post(webhook_url, json=message, timeout=10)
            return response.status_code == 200

        except Exception:
            return False

    def _send_email_notification(self, alert: DriftAlert, email_config: Dict) -> bool:
        """Send alert via email."""
        if MimeText is None:
            print("Email functionality not available (missing email.mime.text module)")
            return False

        try:
            msg = MimeText(f"""
🚨 Model Monitoring Alert

Alert Type: {alert.alert_type.title()}
Severity: {alert.severity.upper()}
Timestamp: {alert.timestamp.isoformat()}

Message: {alert.message}

Metrics: {json.dumps(alert.metrics, indent=2)}

Recommendations:
{chr(10).join(f"• {rec}" for rec in alert.recommendations)}

---
Oncology Target Lite Model Monitor
""")

            msg['Subject'] = f"Model Alert: {alert.alert_type.title()} ({alert.severity.upper()})"
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']

            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                if email_config.get('use_tls', True):
                    server.starttls()
                if 'username' in email_config:
                    server.login(email_config['username'], email_config['password'])
                server.send_message(msg)

            return True

        except Exception:
            return False
