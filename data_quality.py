"""Comprehensive data quality and lineage tracking system."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
import typer

from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics."""
    # Basic metrics
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]

    # Statistical metrics
    column_stats: Dict[str, Dict[str, float]]
    correlations: Dict[str, Dict[str, float]]

    # Quality indicators
    duplicate_rows: int
    duplicate_percentage: float
    data_types: Dict[str, str]

    # Validation results
    validation_errors: List[str]
    warnings: List[str]

    # Metadata
    dataset_hash: str
    created_at: datetime
    data_source: str


@dataclass
class DataLineageEntry:
    """Represents a single entry in the data lineage."""
    operation_id: str
    operation_type: str  # prepare, train, evaluate, etc.
    input_datasets: List[str]
    output_datasets: List[str]
    parameters: Dict[str, Any]
    timestamp: datetime
    git_commit: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class DataQualityProfile:
    """Complete data quality profile for a dataset."""
    dataset_id: str
    dataset_path: Path
    quality_metrics: DataQualityMetrics
    lineage: List[DataLineageEntry]
    baseline_comparison: Optional[Dict[str, Any]] = None
    drift_score: Optional[float] = None


class DataValidator:
    """Core data validation framework."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_rules = {
            'missing_threshold': self.config.get('missing_threshold', 0.1),  # 10%
            'duplicate_threshold': self.config.get('duplicate_threshold', 0.05),  # 5%
            'outlier_threshold': self.config.get('outlier_threshold', 3.0),  # 3 std devs
            'correlation_threshold': self.config.get('correlation_threshold', 0.95)
        }

    def validate_dataset(self, df: pd.DataFrame, dataset_id: str = "unknown") -> DataQualityMetrics:
        """Perform comprehensive data validation."""
        logger.info(f"Validating dataset: {dataset_id}")

        total_rows = len(df)
        total_columns = len(df.columns)

        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = (df.isnull().sum() / total_rows).to_dict()

        # Statistical analysis
        column_stats = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                stats_dict = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }
                column_stats[col] = stats_dict

        # Correlation analysis (for numeric columns only)
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = {}
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            for col in corr_matrix.columns:
                correlations[col] = corr_matrix[col].to_dict()

        # Duplicate detection
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = duplicate_rows / total_rows

        # Data type analysis
        data_types = df.dtypes.astype(str).to_dict()

        # Validation checks
        validation_errors = []
        warnings = []

        # Check missing values
        for col, pct in missing_percentage.items():
            if pct > self.validation_rules['missing_threshold']:
                validation_errors.append(f"Column '{col}' has {pct".1%"} missing values (threshold: {self.validation_rules['missing_threshold']".1%"})")

        # Check duplicates
        if duplicate_percentage > self.validation_rules['duplicate_threshold']:
            validation_errors.append(f"Dataset has {duplicate_percentage".1%"} duplicate rows (threshold: {self.validation_rules['duplicate_threshold']".1%"})")

        # Check for high correlations (potential multicollinearity)
        if correlations:
            for col1 in correlations:
                for col2, corr_val in correlations[col1].items():
                    if col1 != col2 and abs(corr_val) > self.validation_rules['correlation_threshold']:
                        warnings.append(f"High correlation ({corr_val".3f"}) between '{col1}' and '{col2}'")

        # Check for potential outliers (simple IQR method for numeric columns)
        for col in column_stats:
            q75, q25 = np.percentile(df[col].dropna(), [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - (self.validation_rules['outlier_threshold'] * iqr)
            upper_bound = q75 + (self.validation_rules['outlier_threshold'] * iqr)

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if len(outliers) > 0:
                outlier_pct = len(outliers) / total_rows
                warnings.append(f"Column '{col}' has {outlier_pct".1%"} potential outliers")

        # Generate dataset hash
        dataset_content = str(df.values.tobytes()) + str(df.columns.tolist())
        dataset_hash = hashlib.sha256(dataset_content.encode()).hexdigest()[:16]

        return DataQualityMetrics(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            column_stats=column_stats,
            correlations=correlations,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
            data_types=data_types,
            validation_errors=validation_errors,
            warnings=warnings,
            dataset_hash=dataset_hash,
            created_at=datetime.now(),
            data_source=dataset_id
        )


class DataLineageTracker:
    """Tracks data lineage throughout the ML pipeline."""

    def __init__(self, lineage_file: Path = Path("reports/data_lineage.json")):
        self.lineage_file = lineage_file
        self.lineage: List[DataLineageEntry] = []
        self._load_lineage()

    def _load_lineage(self):
        """Load existing lineage data."""
        if self.lineage_file.exists():
            try:
                with open(self.lineage_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data.get('lineage', []):
                        entry = DataLineageEntry(
                            operation_id=entry_data['operation_id'],
                            operation_type=entry_data['operation_type'],
                            input_datasets=entry_data['input_datasets'],
                            output_datasets=entry_data['output_datasets'],
                            parameters=entry_data['parameters'],
                            timestamp=datetime.fromisoformat(entry_data['timestamp']),
                            git_commit=entry_data.get('git_commit'),
                            mlflow_run_id=entry_data.get('mlflow_run_id'),
                            success=entry_data.get('success', True),
                            error_message=entry_data.get('error_message')
                        )
                        self.lineage.append(entry)
            except Exception as e:
                logger.warning(f"Error loading lineage data: {e}")

    def _save_lineage(self):
        """Save lineage data to file."""
        ensure_dir(self.lineage_file.parent)

        data = {
            'lineage': [asdict(entry) for entry in self.lineage],
            'last_updated': datetime.now().isoformat()
        }

        with open(self.lineage_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def add_entry(self, entry: DataLineageEntry):
        """Add a lineage entry."""
        self.lineage.append(entry)
        self._save_lineage()
        logger.info(f"Added lineage entry: {entry.operation_type} ({entry.operation_id})")

    def get_lineage_for_dataset(self, dataset_id: str) -> List[DataLineageEntry]:
        """Get lineage entries for a specific dataset."""
        return [
            entry for entry in self.lineage
            if dataset_id in entry.input_datasets or dataset_id in entry.output_datasets
        ]

    def get_operation_chain(self, start_dataset: str, end_dataset: str) -> List[DataLineageEntry]:
        """Get the operation chain from start to end dataset."""
        # Simple implementation - in practice would need graph traversal
        chain = []
        current_datasets = {start_dataset}

        while current_datasets and end_dataset not in current_datasets:
            next_datasets = set()
            found = False

            for entry in self.lineage:
                if current_datasets.intersection(set(entry.input_datasets)):
                    chain.append(entry)
                    next_datasets.update(entry.output_datasets)
                    found = True

            if not found:
                break

            current_datasets = next_datasets

        return chain


class DataDriftDetector:
    """Detects data drift between datasets."""

    def __init__(self, reference_dataset: pd.DataFrame):
        self.reference_dataset = reference_dataset
        self.reference_stats = self._calculate_dataset_stats(reference_dataset)

    def _calculate_dataset_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics for drift detection."""
        stats_dict = {}

        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                col_stats = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }
                stats_dict[col] = col_stats

        return stats_dict

    def detect_drift(self, current_dataset: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """Detect drift between reference and current dataset."""
        current_stats = self._calculate_dataset_stats(current_dataset)

        drift_results = {
            'overall_drift_score': 0.0,
            'column_drift': {},
            'significant_drift': []
        }

        drift_scores = []

        for col in self.reference_stats:
            if col in current_stats:
                ref_stats = self.reference_stats[col]
                curr_stats = current_stats[col]

                # Calculate drift score for this column
                # Use standardized difference in means and distributions
                mean_diff = abs(ref_stats['mean'] - curr_stats['mean'])
                std_diff = abs(ref_stats['std'] - curr_stats['std'])

                # Normalize by reference std (handle zero std case)
                ref_std = ref_stats['std'] if ref_stats['std'] > 0 else 1.0
                normalized_mean_diff = mean_diff / ref_std
                normalized_std_diff = std_diff / ref_std

                # Combined drift score for this column
                column_drift_score = (normalized_mean_diff + normalized_std_diff) / 2.0

                drift_results['column_drift'][col] = {
                    'drift_score': column_drift_score,
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    'reference_mean': ref_stats['mean'],
                    'current_mean': curr_stats['mean']
                }

                drift_scores.append(column_drift_score)

                if column_drift_score > threshold:
                    drift_results['significant_drift'].append(col)

        # Overall drift score (average across columns)
        if drift_scores:
            drift_results['overall_drift_score'] = float(np.mean(drift_scores))

        return drift_results


class DataQualityMonitor:
    """Main class for data quality monitoring and lineage tracking."""

    def __init__(self, lineage_file: Path = Path("reports/data_lineage.json")):
        self.validator = DataValidator()
        self.lineage_tracker = DataLineageTracker(lineage_file)
        self.drift_detectors: Dict[str, DataDriftDetector] = {}
        self.quality_profiles: Dict[str, DataQualityProfile] = {}

    def register_dataset(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        data_source: str = "unknown",
        parent_datasets: Optional[List[str]] = None
    ) -> DataQualityProfile:
        """Register a new dataset with quality checks and lineage tracking."""

        # Validate the dataset
        quality_metrics = self.validator.validate_dataset(df, dataset_id)

        # Create lineage entry
        operation_id = f"register_{dataset_id}_{int(datetime.now().timestamp())}"
        lineage_entry = DataLineageEntry(
            operation_id=operation_id,
            operation_type="dataset_registration",
            input_datasets=parent_datasets or [],
            output_datasets=[dataset_id],
            parameters={"data_source": data_source},
            timestamp=datetime.now(),
            success=len(quality_metrics.validation_errors) == 0
        )

        self.lineage_tracker.add_entry(lineage_entry)

        # Create quality profile
        profile = DataQualityProfile(
            dataset_id=dataset_id,
            dataset_path=Path(f"data/processed/{dataset_id}.parquet"),
            quality_metrics=quality_metrics,
            lineage=[lineage_entry]
        )

        self.quality_profiles[dataset_id] = profile

        # Save dataset if it passes validation
        if len(quality_metrics.validation_errors) == 0:
            ensure_dir(profile.dataset_path.parent)
            df.to_parquet(profile.dataset_path)
            logger.info(f"✅ Dataset registered successfully: {dataset_id}")
        else:
            logger.warning(f"⚠️ Dataset registered with validation errors: {dataset_id}")

        return profile

    def set_reference_dataset(self, dataset_id: str):
        """Set a dataset as reference for drift detection."""
        if dataset_id not in self.quality_profiles:
            raise ValueError(f"Dataset {dataset_id} not found in quality profiles")

        profile = self.quality_profiles[dataset_id]
        df = pd.read_parquet(profile.dataset_path)

        self.drift_detectors[dataset_id] = DataDriftDetector(df)
        logger.info(f"✅ Set reference dataset for drift detection: {dataset_id}")

    def check_drift(self, dataset_id: str, reference_id: str, threshold: float = 0.1) -> Dict[str, Any]:
        """Check for data drift between two datasets."""
        if reference_id not in self.drift_detectors:
            raise ValueError(f"Reference dataset {reference_id} not set for drift detection")

        if dataset_id not in self.quality_profiles:
            raise ValueError(f"Dataset {dataset_id} not found in quality profiles")

        current_profile = self.quality_profiles[dataset_id]
        current_df = pd.read_parquet(current_profile.dataset_path)

        drift_detector = self.drift_detectors[reference_id]
        drift_results = drift_detector.detect_drift(current_df, threshold)

        # Update profile with drift information
        current_profile.drift_score = drift_results['overall_drift_score']
        current_profile.baseline_comparison = {
            'reference_dataset': reference_id,
            'drift_results': drift_results
        }

        return drift_results

    def generate_quality_report(self, dataset_id: str) -> str:
        """Generate a comprehensive quality report for a dataset."""
        if dataset_id not in self.quality_profiles:
            raise ValueError(f"Dataset {dataset_id} not found")

        profile = self.quality_profiles[dataset_id]

        report = [
            "# Data Quality Report",
            "",
            f"**Dataset:** {dataset_id}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Overview",
            "",
            f"- **Total Rows:** {profile.quality_metrics.total_rows:,}","            f"- **Total Columns:** {profile.quality_metrics.total_columns}",
            f"- **Dataset Hash:** `{profile.quality_metrics.dataset_hash}`",
            f"- **Data Source:** {profile.quality_metrics.data_source}",
            "",
            "## Data Quality Metrics",
            ""
        ]

        # Missing values summary
        missing_cols = [col for col, pct in profile.quality_metrics.missing_percentage.items() if pct > 0]
        if missing_cols:
            report.append("### Missing Values")
            report.append("")
            report.append("| Column | Missing Count | Missing % |")
            report.append("|--------|---------------|-----------|")

            for col in missing_cols:
                count = profile.quality_metrics.missing_values[col]
                pct = profile.quality_metrics.missing_percentage[col]
                report.append(f"| {col} | {count:,} | {pct".1%"} |")
            report.append("")

        # Validation results
        if profile.quality_metrics.validation_errors:
            report.append("### Validation Errors")
            report.append("")
            for error in profile.quality_metrics.validation_errors:
                report.append(f"❌ {error}")
            report.append("")

        if profile.quality_metrics.warnings:
            report.append("### Warnings")
            report.append("")
            for warning in profile.quality_metrics.warnings:
                report.append(f"⚠️  {warning}")
            report.append("")

        # Drift information
        if profile.drift_score is not None:
            report.append("### Data Drift Analysis")
            report.append("")
            report.append(f"- **Overall Drift Score:** {profile.drift_score".3f"}")

            if profile.baseline_comparison:
                significant_drift = profile.baseline_comparison['drift_results'].get('significant_drift', [])
                if significant_drift:
                    report.append(f"- **Columns with Significant Drift:** {', '.join(significant_drift)}")
                else:
                    report.append("- **No significant drift detected**")
            report.append("")

        # Lineage information
        report.append("## Data Lineage")
        report.append("")
        report.append("### Operation History")
        report.append("")
        report.append("| Operation | Type | Timestamp | Success |")
        report.append("|-----------|------|-----------|---------|")

        for entry in profile.lineage:
            status = "✅" if entry.success else "❌"
            report.append(f"| {entry.operation_id} | {entry.operation_type} | {entry.timestamp.strftime('%Y-%m-%d %H:%M')} | {status} |")

        return "\n".join(report)

    def export_quality_profiles(self, output_dir: Path) -> None:
        """Export all quality profiles to JSON."""
        ensure_dir(output_dir)

        profiles_data = {
            dataset_id: {
                'dataset_id': profile.dataset_id,
                'dataset_path': str(profile.dataset_path),
                'quality_metrics': asdict(profile.quality_metrics),
                'lineage': [asdict(entry) for entry in profile.lineage],
                'baseline_comparison': profile.baseline_comparison,
                'drift_score': profile.drift_score
            }
            for dataset_id, profile in self.quality_profiles.items()
        }

        output_file = output_dir / "data_quality_profiles.json"
        with open(output_file, 'w') as f:
            json.dump(profiles_data, f, indent=2, default=str)

        logger.info(f"✅ Exported {len(profiles_data)} quality profiles to {output_file}")


def validate_data_cmd(
    data_path: Path = typer.Option(..., help="Path to data file (CSV or Parquet)"),
    dataset_id: str = typer.Option(..., help="Unique identifier for this dataset"),
    output_dir: Path = typer.Option(Path("reports/data_quality"), help="Output directory for quality report"),
    generate_report: bool = typer.Option(True, help="Generate detailed quality report"),
) -> None:
    """Validate data quality and generate comprehensive report."""

    # Load data
    try:
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    except Exception as e:
        typer.echo(f"❌ Error loading data: {e}")
        raise typer.Exit(1)

    typer.echo(f"✅ Loaded dataset: {len(df):,} rows × {len(df.columns)} columns")

    # Initialize quality monitor
    monitor = DataQualityMonitor()

    # Register and validate dataset
    profile = monitor.register_dataset(df, dataset_id, str(data_path))

    # Display validation results
    metrics = profile.quality_metrics

    typer.echo(f"\n📊 Data Quality Summary for '{dataset_id}':")
    typer.echo(f"   Total rows: {metrics.total_rows:,}","    typer.echo(f"   Total columns: {metrics.total_columns}")
    typer.echo(f"   Missing values: {sum(metrics.missing_values.values()):,}","    typer.echo(f"   Duplicate rows: {metrics.duplicate_rows:,}","    typer.echo(f"   Dataset hash: {metrics.dataset_hash}")

    if metrics.validation_errors:
        typer.echo(f"\n❌ Validation Errors ({len(metrics.validation_errors)}):")
        for error in metrics.validation_errors[:5]:  # Show first 5
            typer.echo(f"   • {error}")
        if len(metrics.validation_errors) > 5:
            typer.echo(f"   ... and {len(metrics.validation_errors) - 5} more")

    if metrics.warnings:
        typer.echo(f"\n⚠️  Warnings ({len(metrics.warnings)}):")
        for warning in metrics.warnings[:3]:  # Show first 3
            typer.echo(f"   • {warning}")
        if len(metrics.warnings) > 3:
            typer.echo(f"   ... and {len(metrics.warnings) - 3} more")

    # Generate report if requested
    if generate_report:
        typer.echo(f"\n📋 Generating quality report in: {output_dir}")
        ensure_dir(output_dir)

        # Save quality metrics
        metrics_file = output_dir / f"{dataset_id}_quality_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, default=str)

        # Generate markdown report
        report_content = monitor.generate_quality_report(dataset_id)
        report_file = output_dir / f"{dataset_id}_quality_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

        typer.echo(f"✅ Quality report generated: {report_file}")
        typer.echo(f"   Metrics: {metrics_file}")


def lineage_cmd(
    dataset_id: Optional[str] = typer.Option(None, help="Specific dataset ID to trace"),
    operation_type: Optional[str] = typer.Option(None, help="Filter by operation type"),
    output_dir: Path = typer.Option(Path("reports/data_lineage"), help="Output directory"),
) -> None:
    """Show data lineage information."""

    # Load lineage data
    lineage_file = Path("reports/data_lineage.json")
    if not lineage_file.exists():
        typer.echo("❌ No lineage data found. Run some pipeline operations first.")
        return

    tracker = DataLineageTracker(lineage_file)

    if not tracker.lineage:
        typer.echo("❌ No lineage entries found")
        return

    # Filter lineage entries
    filtered_lineage = tracker.lineage
    if dataset_id:
        filtered_lineage = [
            entry for entry in filtered_lineage
            if dataset_id in entry.input_datasets or dataset_id in entry.output_datasets
        ]
        typer.echo(f"📋 Lineage for dataset '{dataset_id}':")

    if operation_type:
        filtered_lineage = [
            entry for entry in filtered_lineage
            if entry.operation_type == operation_type
        ]
        if dataset_id:
            typer.echo(f"   (filtered by operation type: {operation_type})")

    if not filtered_lineage:
        typer.echo("❌ No matching lineage entries found")
        return

    # Display lineage entries
    typer.echo(f"\n📊 Found {len(filtered_lineage)} lineage entries:")
    typer.echo("=" * 80)

    for entry in filtered_lineage:
        status = "✅ SUCCESS" if entry.success else "❌ FAILED"
        typer.echo(f"\n🔗 Operation: {entry.operation_id}")
        typer.echo(f"   Type: {entry.operation_type}")
        typer.echo(f"   Status: {status}")
        typer.echo(f"   Timestamp: {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        if entry.input_datasets:
            typer.echo(f"   Input datasets: {', '.join(entry.input_datasets)}")
        if entry.output_datasets:
            typer.echo(f"   Output datasets: {', '.join(entry.output_datasets)}")

        if entry.parameters:
            typer.echo(f"   Parameters: {json.dumps(entry.parameters, indent=2)}")

        if not entry.success and entry.error_message:
            typer.echo(f"   Error: {entry.error_message}")


def drift_detection_cmd(
    current_dataset: str = typer.Option(..., help="Current dataset ID"),
    reference_dataset: str = typer.Option(..., help="Reference dataset ID for comparison"),
    threshold: float = typer.Option(0.1, help="Drift detection threshold"),
) -> None:
    """Detect data drift between datasets."""

    # Initialize monitor
    monitor = DataQualityMonitor()

    # Load quality profiles
    if current_dataset not in monitor.quality_profiles:
        typer.echo(f"❌ Current dataset '{current_dataset}' not found. Register it first.")
        return

    if reference_dataset not in monitor.quality_profiles:
        typer.echo(f"❌ Reference dataset '{reference_dataset}' not found. Register it first.")
        return

    # Set reference for drift detection
    monitor.set_reference_dataset(reference_dataset)

    # Check for drift
    typer.echo(f"🔍 Checking for data drift between '{reference_dataset}' and '{current_dataset}'...")
    drift_results = monitor.check_drift(current_dataset, reference_dataset, threshold)

    # Display results
    typer.echo(f"\n📊 Drift Detection Results:")
    typer.echo(f"   Overall drift score: {drift_results['overall_drift_score']".3f"}")

    if drift_results['significant_drift']:
        typer.echo(f"   Significant drift detected in {len(drift_results['significant_drift'])} columns:")
        for col in drift_results['significant_drift']:
            col_drift = drift_results['column_drift'][col]
            typer.echo(f"     • {col}: drift score {col_drift['drift_score']".3f"}")
    else:
        typer.echo("   ✅ No significant drift detected")

    # Show detailed column analysis
    typer.echo("
📋 Column-by-Column Analysis:"    for col, drift_info in drift_results['column_drift'].items():
        typer.echo(f"   {col}:")
        typer.echo(f"     Drift score: {drift_info['drift_score']".3f"}")
        typer.echo(f"     Mean difference: {drift_info['mean_diff']".3f"}")
        typer.echo(f"     Std difference: {drift_info['std_diff']".3f"}")
