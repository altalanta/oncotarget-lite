# oncotarget-lite

[![CI](https://github.com/altalanta/oncotarget-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/altalanta/oncotarget-lite/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-material-blue)](https://altalanta.github.io/oncotarget-lite)
[![GHCR CPU](https://img.shields.io/badge/container-cpu-blue?logo=docker)](https://ghcr.io/altalanta/oncotarget-lite)
[![GHCR CUDA](https://img.shields.io/badge/container-cuda-00bcd4?logo=nvidia)](https://ghcr.io/altalanta/oncotarget-lite)
[![PyPI version](https://badge.fury.io/py/oncotarget-lite.svg)](https://badge.fury.io/py/oncotarget-lite)

Interpreter-ready oncology target triage on synthetic data. The project focuses on reproducibility, governance, and reviewer-friendly artefacts without inflating runtime (>10 min on CPU).

## Quickstart

```bash
git clone https://github.com/altalanta/oncotarget-lite.git
cd oncotarget-lite
make setup
make all
mlflow ui --backend-store-uri ./mlruns
```

The `make all` target executes the full Typer pipeline:
`prepare → train → eval → explain → scorecard → docs → snapshot` and stores outputs under `reports/`, `models/`, and `docs/`.

## Model Types

The system supports multiple model architectures:

**Traditional ML Models:**
- **Logistic Regression** (`logreg`) - Strong baseline with interpretability
- **XGBoost** (`xgb`) - Tree-based ensemble with excellent performance
- **LightGBM** (`lgb`) - Fast gradient boosting
- **MLP** (`mlp`) - Neural network baseline

**Modern Deep Learning Models:**
- **Transformer** (`transformer`) - Attention-based architecture for biological sequences
- **Graph Neural Network** (`gnn`) - Network-based learning for PPI and pathway data

## Hyperparameter Optimization

The system includes automated hyperparameter optimization using Optuna to find optimal model configurations:

```bash
# Optimize XGBoost hyperparameters
python -m oncotarget_lite.cli optimize --model-type xgb --n-trials 100

# Optimize Transformer model for maximum AUROC
python -m oncotarget_lite.cli optimize --model-type transformer --metric auroc --n-trials 50

# Use Makefile target
make optimize
```

**Optimization Features:**
- **Multi-model Support**: Works with all model types (logreg, xgb, lgb, mlp, transformer, gnn)
- **Custom Metrics**: Optimize for AUROC or Average Precision
- **Persistent Studies**: SQLite storage for resuming interrupted optimizations
- **MLflow Integration**: Logs best parameters and results automatically
- **Comprehensive Search Spaces**: Model-specific parameter ranges for effective optimization

**Optimization Results:**
- Best parameters and scores saved to `reports/optuna_summary_{model_type}.json`
- Optimization history and trial details available for analysis
- Automatic integration with existing training and evaluation pipeline

## Advanced Interpretability Dashboard

The system includes comprehensive interpretability visualization and validation through interactive dashboards:

```bash
# Generate comprehensive interpretability dashboard
python -m oncotarget_lite.cli dashboard

# Create model comparison dashboard
python -m oncotarget_lite.cli dashboard --model-comparison --model-reports reports/validation_report_*.json

# Use Makefile target
make dashboard
```

**Dashboard Features:**
- **Interactive Visualizations**: Global feature importance, gene contribution breakdowns, feature interaction heatmaps
- **Validation Metrics**: Background consistency, feature stability, perturbation robustness, and overall quality scores
- **Model Comparison**: Side-by-side interpretability analysis across different models
- **Export Options**: HTML dashboards for interactive exploration, static PNG exports for reports
- **Integration**: Automatically loads SHAP values and validation reports for comprehensive analysis

**Dashboard Components:**
- **Global Importance**: Enhanced bar charts with hover tooltips and color coding for top features
- **Gene Contributions**: Waterfall plots showing how individual features contribute to predictions
- **Feature Interactions**: Correlation heatmaps revealing feature relationships in SHAP space
- **Validation Summary**: Multi-panel dashboard showing stability metrics and quality scores
- **Model Comparison**: Bar charts comparing interpretability metrics across different models

**Output Files:**
- `reports/dashboard/interpretability_dashboard.html` - Main interactive dashboard
- `reports/dashboard/static/` - Static PNG exports for inclusion in reports
- Model comparison dashboards when using `--model-comparison` flag

## Scalable Data Processing & Caching

The system includes high-performance data processing with parallel loading, advanced caching, and memory optimization:

```bash
# Check cache statistics
python -m oncotarget_lite.cli cache info

# Clear cache files
python -m oncotarget_lite.cli cache clear --pattern "*parquet"

# Benchmark data loading performance
python -m oncotarget_lite.cli cache benchmark

# Use Makefile targets
make cache
```

**Scalability Features:**
- **Parallel Data Loading**: Load multiple CSV files simultaneously using multiprocessing
- **Advanced Caching**: SHA256-based cache invalidation with compression and versioning
- **Memory Optimization**: Chunked processing for large datasets with memory monitoring
- **Performance Benchmarking**: Built-in benchmarking for feature extraction and data loading
- **Automatic Cache Management**: Intelligent cache clearing and statistics reporting

**Performance Optimizations:**
- **Parallel Feature Extraction**: Extract multiple feature types simultaneously using ThreadPoolExecutor
- **Chunked Processing**: Process large datasets in configurable chunks to manage memory usage
- **Lazy Loading**: Only load data when needed, with intelligent caching strategies
- **Memory Monitoring**: Track memory usage and provide optimization recommendations
- **Cache Analytics**: Detailed cache statistics and performance metrics

**Cache Management:**
- Automatic cache invalidation based on file content and processing parameters
- Compression support (Snappy for Parquet, configurable formats)
- Pattern-based cache clearing for maintenance
- Cache size and performance analytics

## Tracking & Lineage

- Deterministic CLI (`python -m oncotarget_lite.cli ...`) with reproducible seeds (`PYTHONHASHSEED=0`).
- DVC pipeline (`dvc.yaml`) with stages `fetch_data → prepare → train → eval → explain → scorecard → ablations` plus local remote at `./dvcstore`. Re-run everything with `dvc repro`.
- MLflow experiment `oncotarget-lite` writes params, metrics, model binaries, dataset hash, and git commit into `./mlruns`.
- `reports/run_context.json` links downstream stages to the originating MLflow run for audit trails.
- **Data manifest**: `data/manifest.json` ensures reproducible data with SHA256 hashes and source tracking.

## Distributed Computing

The pipeline supports distributed computing for computationally intensive tasks:

- **Bootstrap confidence intervals** computed in parallel for faster evaluation
- **SHAP explanations** generated with parallel processing for multiple samples
- **Ablation studies** run concurrently across different model configurations
- **Multiple backends supported**: Joblib (default), Dask, and Ray

Configure distributed computing:

```bash
# Use all available cores
python -m oncotarget_lite.cli distributed --n-jobs -1

# Use specific number of cores
python -m oncotarget_lite.cli distributed --n-jobs 8

# Use Dask for large-scale distributed computing
python -m oncotarget_lite.cli distributed --backend dask --n-jobs 16
```

The `make all` command automatically enables distributed computing with all cores.

## Model Performance Monitoring

The system includes comprehensive model monitoring and drift detection:

- **Performance Tracking**: Automatic capture of model metrics after each evaluation
- **Drift Detection**: Statistical tests for detecting changes in prediction distributions
- **Feature Importance Monitoring**: Tracking changes in feature importance over time
- **Automated Alerts**: Configurable notifications for performance regressions
- **Trend Analysis**: Historical performance trends and forecasting

### Monitoring Commands

```bash
# View monitoring status
python -m oncotarget_lite.cli monitor status

# Capture current performance snapshot
python -m oncotarget_lite.cli monitor capture

# Generate detailed monitoring report
python -m oncotarget_lite.cli monitor report --days 30

# Check for drift and send alerts
python -m oncotarget_lite.cli monitor alerts --slack-webhook $WEBHOOK_URL

# Configure monitoring with custom thresholds
python -m oncotarget_lite.cli monitor status --model-version latest --days 7
```

### Alert Configuration

Configure alerts via environment variables or CLI options:

```bash
# Slack notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Email notifications
export EMAIL_TO="alerts@company.com"
export SMTP_SERVER="smtp.company.com"
```

The monitoring system automatically detects:
- **Performance Regressions**: Significant drops in AUROC, AP, or other metrics
- **Prediction Drift**: Changes in prediction distribution characteristics
- **Feature Drift**: Changes in feature importance rankings
- **Data Quality Issues**: Anomalies in class balance or sample characteristics

## Automated Model Retraining

The system includes intelligent automated retraining that monitors model performance and triggers retraining when needed:

### Automated Retraining Triggers

The system automatically retrains models based on multiple intelligent triggers:

- **Performance Degradation**: Triggers when AUROC or AP drops by more than 2% from baseline
- **Data Drift Detection**: Activates when significant distribution changes are detected in predictions
- **Scheduled Retraining**: Regular weekly retraining to ensure models stay current
- **Feature Drift**: Triggers when feature importance patterns change significantly

### Automated Retraining Commands

```bash
# Check if retraining is needed and run if triggered
python -m oncotarget_lite.cli retrain --schedule

# Force retraining regardless of triggers
python -m oncotarget_lite.cli retrain --force

# Dry run to see what would happen without executing
python -m oncotarget_lite.cli retrain --dry-run

# Use Makefile target
make retrain
```

**Retraining Features:**
- **Intelligent Triggering**: Multiple detection mechanisms for optimal retraining timing
- **Best Practices Integration**: Uses optimal hyperparameters from previous optimization runs
- **Automated Validation**: Compares new model performance against baseline before deployment
- **Safe Deployment**: Only deploys models that show meaningful improvement (>0.5% in primary metrics)
- **Rollback Support**: Can rollback to previous model if issues are detected
- **Comprehensive Logging**: All retraining activities logged to MLflow for audit trails

**Configuration:**
- Customize trigger thresholds and behavior via `configs/retrain_config.json`
- Adjust performance thresholds, scheduling intervals, and deployment criteria
- Enable/disable specific trigger types based on operational needs

**Output:**
- New model versions automatically logged to MLflow
- Performance comparisons and deployment decisions recorded
- Deployment metadata saved for audit trails
- Integration with existing monitoring and alerting systems

### Model Rollback

If an automated deployment causes issues, you can rollback to a previous model:

```bash
# Rollback to a model from the last 24 hours
python -m oncotarget_lite.cli rollback

# Rollback to a model from the last 48 hours
python -m oncotarget_lite.cli rollback --hours-back 48

# Use Makefile target (force-rollback automatically)
make rollback
```

**Rollback Features:**
- **Safe Rollback**: Automatically backs up current model before deployment
- **Time-based Selection**: Choose rollback target based on time window
- **Confirmation Prompts**: Interactive confirmation to prevent accidental rollbacks
- **Audit Trail**: All rollback operations logged with timestamps and reasons

## Enhanced Model Serving & API Layer

The system includes a production-ready FastAPI-based model serving layer with advanced features for real-time predictions, model versioning, and A/B testing:

### Model Serving Features

**Core Capabilities:**
- **RESTful API**: Clean, documented endpoints for predictions and model management
- **Batch Processing**: Efficient batch prediction capabilities for high-throughput scenarios
- **Caching**: In-memory prediction caching with TTL for improved performance
- **Health Monitoring**: Built-in health checks and system status endpoints

### Model Versioning & Deployment

**Version Management:**
- **Semantic Versioning**: Models are versioned with timestamps and run IDs
- **Production Deployment**: Seamless deployment of models to production with rollback support
- **Metadata Tracking**: Comprehensive metadata including performance metrics and feature names
- **Model Registry**: Centralized registry for managing multiple model versions

**Deployment Commands:**
```bash
# Deploy a model version to production
python -m oncotarget_lite.cli deploy model_20250101_123456_abc12345

# List all available model versions
python -m oncotarget_lite.cli versions

# List versions with detailed metrics
python -m oncotarget_lite.cli versions --details

# Clean up old model versions (dry run)
python -m oncotarget_lite.cli cleanup --dry-run

# Clean up old versions (actually delete)
python -m oncotarget_lite.cli cleanup --keep-recent 3

# Use Makefile targets
make deploy VERSION_ID=model_20250101_123456_abc12345
make versions
make cleanup
```

### FastAPI Server

**Server Features:**
- **Async Support**: Asynchronous prediction handling for high concurrency
- **CORS Enabled**: Cross-origin resource sharing for web integration
- **Auto Documentation**: Interactive API documentation at `/docs`
- **Request Tracing**: Request ID tracking for debugging and monitoring

**Starting the Server:**
```bash
# Start server with default settings
python -m oncotarget_lite.cli serve

# Start server on custom port
python -m oncotarget_lite.cli serve --port 9000

# Start server with auto-reload for development
python -m oncotarget_lite.cli serve --reload

# Use Makefile target
make serve
```

**API Endpoints:**
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /models` - List all model versions
- `GET /models/{version_id}` - Get specific model details
- `GET /ab-tests` - List A/B tests
- `POST /ab-tests` - Create A/B test
- `GET /health` - Health check
- `POST /cache/clear` - Clear prediction cache

**Example API Usage:**
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {"gene_A": 0.5, "gene_B": 0.3, "clinical_score": 0.8},
    "model_version": "model_20250101_123456_abc12345"
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"gene_A": 0.5, "gene_B": 0.3, "clinical_score": 0.8},
      {"gene_A": 0.2, "gene_B": 0.7, "clinical_score": 0.4}
    ]
  }'
```

### A/B Testing Framework

**A/B Testing Features:**
- **Traffic Splitting**: Configurable percentage-based routing between model versions
- **Request-based Routing**: Consistent model selection based on request ID hashing
- **Test Management**: Create, manage, and monitor A/B tests
- **Performance Comparison**: Side-by-side evaluation of model versions

**A/B Testing Commands:**
```bash
# Create an A/B test (requires server to be running)
curl -X POST "http://localhost:8000/ab-tests" \
  -H "Content-Type: application/json" \
  -d '{
    "test_id": "experiment_001",
    "model_a": "model_20250101_123456_abc12345",
    "model_b": "model_20250102_154321_def67890",
    "traffic_split": 0.5,
    "start_time": "2025-01-01T00:00:00Z"
  }'

# List A/B tests
curl "http://localhost:8000/ab-tests"
```

**A/B Testing Workflow:**
1. **Setup**: Create A/B test configuration with two model versions
2. **Traffic Routing**: Requests are automatically routed based on hash-based assignment
3. **Monitoring**: Track performance differences between model versions
4. **Decision**: Choose winning model based on performance metrics
5. **Deployment**: Deploy winning model to production

### Integration with Automated Retraining

The model serving layer integrates seamlessly with automated retraining:

**Automated Deployment:**
- New models from automated retraining are automatically versioned and deployed
- Performance metrics are captured and stored with each version
- Rollback capabilities ensure safe deployment of new models

**Monitoring Integration:**
- Server health metrics are integrated with the monitoring system
- Prediction performance is tracked and can trigger retraining
- Cache performance and hit rates are monitored

## Advanced Model Comparison & Selection Framework

The system includes sophisticated tools for comparing and selecting the best models based on comprehensive criteria and statistical analysis:

### Model Comparison Features

**Core Capabilities:**
- **Multi-Criteria Selection**: Weighted scoring across multiple performance metrics
- **Statistical Significance Testing**: Rigorous comparison between model versions
- **Interactive Visualizations**: Side-by-side performance comparisons with statistical heatmaps
- **Automated Ranking**: Composite scoring with customizable weights and thresholds

### Model Selection Criteria

**Configurable Criteria:**
- **Performance Weights**: Customizable importance weights for AUROC, AP, Accuracy, F1, Calibration, Efficiency
- **Minimum Thresholds**: Set minimum acceptable values for key metrics
- **Statistical Requirements**: Configure significance levels and sample size requirements
- **Efficiency Constraints**: Set maximum acceptable training/inference times

**Selection Commands:**
```bash
# Compare models using default criteria
python -m oncotarget_lite.cli compare

# Compare models with custom criteria
python -m oncotarget_lite.cli compare --criteria-config configs/comparison_criteria.json

# Generate detailed comparison report
python -m oncotarget_lite.cli compare --output-dir reports/model_comparison

# Launch interactive comparison dashboard
python -m oncotarget_lite.cli compare-interactive

# Use Makefile targets
make compare
make compare-interactive
```

### Interactive Comparison Dashboard

**Visualization Features:**
- **Performance Comparison Plots**: Side-by-side bar charts for key metrics
- **Model Ranking Charts**: Visual ranking by composite scores with color coding
- **Statistical Significance Heatmaps**: P-value matrices showing significant differences
- **Comprehensive Reports**: Markdown reports with detailed analysis and recommendations

**Dashboard Components:**
- **Performance Overview**: Multi-metric comparison across all models
- **Ranking Visualization**: Clear visual ranking with scores and confidence intervals
- **Statistical Analysis**: Heatmap of pairwise significance tests
- **Criteria Transparency**: Display of selection criteria and thresholds used

**Generated Files:**
- `reports/model_comparison/comparison_summary.json` - Machine-readable summary
- `reports/model_comparison/model_comparison_report.md` - Human-readable report
- `reports/model_comparison/performance_comparison.html` - Interactive performance plot
- `reports/model_comparison/model_ranking.html` - Interactive ranking visualization
- `reports/model_comparison/statistical_significance.html` - Statistical analysis heatmap

### Statistical Analysis

**Significance Testing:**
- **Pairwise Comparisons**: Statistical tests between all model pairs
- **P-value Calculation**: Formal significance testing with configurable alpha levels
- **Effect Size Measurement**: Quantify practical significance of differences
- **Multiple Testing Correction**: Account for multiple comparisons

**Analysis Workflow:**
1. **Model Filtering**: Remove models below minimum thresholds
2. **Composite Scoring**: Calculate weighted scores across all criteria
3. **Statistical Testing**: Perform significance tests between models
4. **Ranking & Recommendation**: Provide clear recommendations with confidence

### Integration with Existing Systems

**Seamless Integration:**
- **Model Registry Integration**: Automatically loads models from version registry
- **Retraining Pipeline**: Uses comparison results to inform deployment decisions
- **Monitoring System**: Statistical results feed into monitoring and alerting
- **MLflow Logging**: All comparison activities logged for audit trails

**Example Workflow:**
```bash
# 1. Run automated retraining to generate new models
make retrain

# 2. Compare all models to find the best one
make compare

# 3. Review the interactive dashboard
# Open reports/model_comparison/ files in browser

# 4. Deploy the recommended model
make deploy VERSION_ID=<recommended_model_id>
```

## Data Quality & Lineage Tracking

The system includes comprehensive data quality monitoring and lineage tracking to ensure data integrity and reproducibility throughout the ML pipeline:

### Data Quality Monitoring

**Core Features:**
- **Comprehensive Validation**: Missing values, duplicates, outliers, data type consistency
- **Statistical Profiling**: Distribution analysis, correlation detection, summary statistics
- **Quality Thresholds**: Configurable validation rules with automatic error detection
- **Drift Detection**: Statistical comparison between datasets for distribution changes
- **Automated Reporting**: Detailed quality reports with actionable insights

**Quality Metrics:**
- **Completeness**: Missing value percentages and patterns
- **Accuracy**: Data type validation and format consistency
- **Consistency**: Duplicate detection and correlation analysis
- **Validity**: Outlier detection and statistical distribution checks
- **Timeliness**: Data freshness and update frequency tracking

### Data Lineage Tracking

**Lineage Features:**
- **Operation Tracking**: Complete audit trail of all data transformations
- **Dependency Mapping**: Track input/output relationships between datasets
- **Parameter Logging**: Record all processing parameters and configurations
- **Success/Failure Tracking**: Monitor operation outcomes and error states
- **Temporal Context**: Timestamp all operations with git commit and MLflow run IDs

**Lineage Commands:**
```bash
# Validate data quality for any dataset
python -m oncotarget_lite.cli validate-data data/processed/features.parquet prepared_dataset

# Show data lineage for a specific dataset
python -m oncotarget_lite.cli lineage --dataset-id prepared_dataset

# Detect data drift between datasets
python -m oncotarget_lite.cli drift --current prepared_new --reference prepared_old

# Use Makefile targets
make validate-data DATA_PATH=data/processed/features.parquet DATASET_ID=prepared_data
make lineage
make drift CURRENT=prepared_new REFERENCE=prepared_old
```

### Data Quality Workflow

**Quality Assurance Process:**
1. **Data Ingestion**: Validate raw data sources for completeness and format
2. **Processing Validation**: Monitor data transformations for quality preservation
3. **Drift Monitoring**: Compare processed data against reference baselines
4. **Quality Reporting**: Generate comprehensive reports for stakeholder review
5. **Lineage Documentation**: Maintain complete audit trails for compliance

**Example Quality Check:**
```bash
# Validate a dataset and generate report
python -m oncotarget_lite.cli validate-data \
  --data-path data/processed/features.parquet \
  --dataset-id production_features \
  --output-dir reports/data_quality

# Check for drift against baseline
python -m oncotarget_lite.cli drift \
  --current production_features \
  --reference baseline_features \
  --threshold 0.1
```

**Quality Report Contents:**
- **Dataset Overview**: Size, structure, and metadata
- **Quality Metrics**: Statistical summaries and validation results
- **Error Detection**: Missing values, duplicates, outliers
- **Warning Alerts**: Potential data issues requiring attention
- **Drift Analysis**: Comparison against reference datasets
- **Lineage History**: Complete operation chain and parameters

### Integration with ML Pipeline

**Automated Quality Monitoring:**
- **Pipeline Integration**: Quality checks embedded in data preparation
- **Failure Prevention**: Automatic rejection of poor-quality data
- **Alerting System**: Integration with monitoring for quality issues
- **Compliance Support**: Audit trails for regulatory requirements

**Quality Gates:**
- **Pre-Processing**: Validate raw data before feature engineering
- **Post-Processing**: Verify processed data meets quality standards
- **Model Training**: Ensure training data quality for reliable models
- **Production Deployment**: Validate data before serving predictions

## Advanced ML Experimentation Platform

The system includes a sophisticated experimentation platform for systematic hyperparameter optimization, model comparison, and automated selection:

### Experimentation Features

**Core Capabilities:**
- **Experiment Management**: Structured experiments with trial tracking and versioning
- **Multi-Model Optimization**: Simultaneous optimization across multiple model types
- **Advanced Search Spaces**: Sophisticated parameter spaces with various distribution types
- **Real-time Monitoring**: Live progress tracking and early stopping capabilities
- **Automated Reporting**: Comprehensive experiment reports with visualizations

**Experiment Management:**
- **Structured Experiments**: Named experiments with configurable model types and search spaces
- **Trial Tracking**: Complete audit trail of all optimization trials and their results
- **Best Trial Selection**: Automatic identification of optimal parameter combinations
- **Experiment Persistence**: Save/load experiments for reproducibility and comparison

### Experimentation Commands

**Running Experiments:**
```bash
# Run experiment with default configuration
python -m oncotarget_lite.cli experiment baseline_optimization

# Run experiment with custom configuration
python -m oncotarget_lite.cli experiment --config-file configs/experiment_config.json

# Run experiment with specific model types and trial count
python -m oncotarget_lite.cli experiment \
  --model-types logreg,xgb,lgb \
  --n-trials 200 \
  advanced_optimization

# Use Makefile targets
make experiment NAME=baseline_optimization MODEL_TYPES=logreg,xgb TRIALS=100
```

**Experiment Management:**
```bash
# List all experiments
python -m oncotarget_lite.cli experiments

# List experiments with detailed information
python -m oncotarget_lite.cli experiments --details

# Generate detailed report for specific experiment
python -m oncotarget_lite.cli experiment-report exp_baseline_optimization_1234567890

# Compare multiple experiments
python -m oncotarget_lite.cli compare-experiments \
  --experiment-ids exp_1,exp_2,exp_3

# Use Makefile targets
make experiments
make experiment-report EXPERIMENT_ID=exp_baseline_optimization_1234567890
make compare-experiments IDS=exp_1,exp_2,exp_3
```

### Experiment Configuration

**Search Spaces:**
The experimentation platform supports sophisticated search spaces:

```json
{
  "experiment_name": "comprehensive_model_optimization",
  "model_types": ["logreg", "xgb", "lgb"],
  "search_spaces": {
    "logreg": {
      "C": {"type": "loguniform", "low": 1e-4, "high": 1e2},
      "max_iter": {"type": "int", "low": 100, "high": 1000},
      "class_weight": {"type": "categorical", "choices": ["balanced", null]}
    },
    "xgb": {
      "n_estimators": {"type": "int", "low": 50, "high": 500},
      "max_depth": {"type": "int", "low": 3, "high": 10},
      "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 0.3}
    }
  }
}
```

**Supported Distribution Types:**
- **Uniform**: Continuous uniform distribution
- **Loguniform**: Log-uniform distribution for scale parameters
- **Int**: Integer uniform distribution
- **Categorical**: Discrete choice from predefined options
- **Normal**: Normal distribution with mean and standard deviation

### Experiment Workflow

**Complete Experimentation Process:**
1. **Experiment Setup**: Define model types, search spaces, and optimization criteria
2. **Trial Execution**: Systematically explore parameter space with intelligent sampling
3. **Performance Evaluation**: Evaluate each trial with comprehensive metrics
4. **Best Trial Selection**: Automatically identify optimal parameter combinations
5. **Experiment Reporting**: Generate detailed reports with visualizations and insights

**Example Experiment Output:**
```
🔬 Starting experiment: exp_comprehensive_optimization_1234567890
   Model types: logreg, xgb, lgb
   Trials: 100

🎯 Optimizing logreg...
   Trial 10/100 completed
   Trial 20/100 completed
   Trial 30/100 completed

🎯 Optimizing xgb...
   Trial 10/100 completed
   Trial 20/100 completed

✅ Experiment completed: exp_comprehensive_optimization_1234567890
   Total trials: 300
🏆 Best result:
   Model: xgb
   AUROC: 0.8923
   AP: 0.8341
   Accuracy: 0.8512
```

### Integration with Existing Systems

**Seamless Integration:**
- **Model Registry Integration**: Experiments automatically register models with versioning
- **Retraining Pipeline**: Experiment results inform automated retraining decisions
- **MLflow Integration**: All trials logged to MLflow for comprehensive tracking
- **Quality Monitoring**: Experiments use validated data from quality monitoring

**Experiment Results Usage:**
```bash
# 1. Run comprehensive experiment
make experiment NAME=comprehensive_search TRIALS=200

# 2. Review experiment results
make experiments

# 3. Generate detailed report
make experiment-report EXPERIMENT_ID=<experiment_id>

# 4. Deploy best model
make deploy VERSION_ID=<best_model_id>
```

### Experiment Visualizations

**Generated Visualizations:**
- **Trial Progression Plots**: Show improvement over optimization trials
- **Experiment Comparison Charts**: Compare multiple experiments side-by-side
- **Parameter Importance Analysis**: Identify most influential hyperparameters
- **Performance Distribution Plots**: Statistical analysis of trial results

**Interactive Dashboards:**
- **Experiment Overview**: High-level experiment status and results
- **Trial Details**: Individual trial performance and parameters
- **Comparison Views**: Side-by-side experiment analysis
- **Export Capabilities**: Generate reports for stakeholder review

## Interpretability Validation

The system includes comprehensive validation for SHAP explanations and model interpretability:

- **Background Consistency Testing**: Validates explanation stability across different background dataset sizes
- **Explanation Stability Analysis**: Bootstrap-based stability testing of SHAP values
- **Feature Importance Validation**: Statistical confidence intervals for feature importance rankings
- **Perturbation Robustness Testing**: Tests explanation sensitivity to small input changes
- **Cross-Validation Consistency**: Validates explanations across different data folds
- **Counterfactual Explanation Generation**: Creates examples showing how to change predictions

### Interpretability Validation Commands

```bash
# Run comprehensive interpretability validation
python -m oncotarget_lite.cli validate-interpretability

# Quick validation summary
python -m oncotarget_lite.cli validate-interpretability --summary-only

# Custom validation parameters
python -m oncotarget_lite.cli validate-interpretability \
  --background-sizes "50,100,200" \
  --n-bootstrap 200 \
  --perturbation-magnitude 0.05

# Use Makefile target
make validate-interpretability
```

### Validation Metrics

The system computes and reports:
- **Overall Quality Score**: Composite metric combining all validation aspects
- **Background Consistency**: Correlation of explanations across background sizes
- **Stability Score**: Consistency of explanations across bootstrap samples
- **Feature Importance Rank Stability**: Kendall tau correlation of importance rankings
- **Perturbation Robustness**: Sensitivity to input perturbations
- **Cross-Validation Consistency**: Agreement across CV folds

## Evaluation

Offline artefacts live in `reports/` and are persisted via DVC (`persist: true`).

<!-- README_METRICS_START -->
| Metric | Value | 95% CI |
| --- | --- | --- |
| AUROC | 0.850 | 0.800 – 0.900 |
| Average Precision | 0.780 | 0.700 – 0.850 |
| Brier | 0.150 | – |
| ECE | 0.050 | – |
| Accuracy | 0.820 | – |
| F1 | 0.750 | – |
| Train AUROC | 0.870 | – |
| Test AUROC | 0.850 | – |
| Overfit gap | 0.020 | – |
<!-- README_METRICS_END -->

### Ablation Studies

Model and feature ablations with bootstrap confidence intervals:

**Traditional ML Models:**
| Model | Features | Test AUROC | 95% CI | Interpretation |
| --- | --- | --- | --- | --- |
| LogReg | All | 0.850 | [0.830, 0.870] | Strong baseline |
| XGBoost | All | 0.855 | [0.835, 0.875] | Best performance |
| MLP | All | 0.842 | [0.820, 0.864] | Competitive |
| LogReg | Network Only | 0.810 | [0.785, 0.835] | Network features powerful |
| LogReg | Clinical Only | 0.720 | [0.685, 0.755] | Limited clinical signal |

**Modern Deep Learning Models:**
| Model | Features | Test AUROC | 95% CI | Interpretation |
| --- | --- | --- | --- | --- |
| Transformer | All | 0.875 | [0.855, 0.895] | Attention-based learning |
| GNN | All | 0.868 | [0.848, 0.888] | Graph structure learning |
| Transformer | Network Only | 0.845 | [0.825, 0.865] | Network attention patterns |

Run ablations: `make ablations`. See [docs/ablations.md](docs/ablations.md) for detailed analysis.

Key files:

- `reports/metrics.json` – point estimates and 95% CIs (AUROC/AP/Brier/ECE/Accuracy/F1/overfit gap).
- `reports/bootstrap.json` – bootstrap summaries (n, lower/upper bounds).
- `reports/calibration.json` & `reports/calibration_plot.png` – reliability curve data and PNG.
- `reports/ablations/metrics.csv` – ablation experiment results with confidence intervals.
- `reports/ablations/deltas.json` – statistical comparisons vs baseline with bootstrap CIs.
- `reports/ablations/summary.html` – interactive ablation analysis dashboard.

## Interpretability & Insights

- `python -m oncotarget_lite.cli explain` emits SHAP values under `reports/shap/` with:
  - `global_summary.png` mean |SHAP| bar chart.
  - `example_GENE{1,2,3}.png` per-gene cards.
  - `shap_values.npz` + `alias_map.json` for downstream analysis.
- `reports/target_scorecard.html` links predicted scores, rankings, SHAP PNGs, and top positive/negative contributors.

## Model Card & Governance

- Responsible AI summary lives at `oncotarget_lite/model_card.md` and is auto-updated by `python -m oncotarget_lite.cli docs`.
- Docs landing page (`docs/index.html`) references metrics, calibration plots, scorecard, model card, and MLflow run ID.
- **Triage UI**: Interactive Streamlit app (`make app`) for exploring predictions, SHAP explanations, and ablation results.
- Streamlit snapshot (`reports/streamlit_demo.png`) lets reviewers inspect the UI without launching the app.

## Make Targets

| Target | Purpose |
| --- | --- |
| `make setup` | Create `.venv`, install pinned dependencies, install package editable |
| `make prepare` | Regenerate processed features/labels/splits |
| `make train` | Train logistic regression + log to MLflow |
| `make optimize` | Run automated hyperparameter optimization with Optuna |
| `make evaluate` | Compute metrics, bootstrap CIs, and calibration artefacts |
| `make explain` | Generate SHAP PNGs and `shap_values.npz` |
| `make dashboard` | Generate advanced interpretability dashboard with interactive visualizations |
| `make scorecard` | Build `reports/target_scorecard.html` |
| `make report-docs` | Refresh `docs/index.html` and model card metrics |
| `make snapshot` | Capture Streamlit UI screenshot via Playwright |
| **`make ablations`** | **Run all ablation experiments and generate analysis** |
| **`make app`** | **Launch interactive Streamlit triage UI** |
| **`make distributed`** | **Configure distributed computing settings** |
| **`make security`** | **Run security audit on dependencies** |
| **`make monitor`** | **Check model performance and drift status** |
| **`make validate-interpretability`** | **Run interpretability validation on SHAP explanations** |
| `make retrain` | **Automated model retraining with intelligent triggers** |
| `make deploy` | **Deploy model version to production** |
| `make versions` | **List all model versions** |
| `make cleanup` | **Clean up old model versions** |
| `make serve` | **Start model serving server** |
| `make compare` | **Compare and rank models using advanced criteria** |
| `make compare-interactive` | **Launch interactive model comparison dashboard** |
| `make validate-data` | **Validate data quality and generate reports** |
| `make lineage` | **Show data lineage information** |
| `make drift` | **Detect data drift between datasets** |
| `make experiment` | **Run advanced ML experiment with optimization** |
| `make experiments` | **List all experiments** |
| `make experiment-report` | **Generate detailed experiment report** |
| `make compare-experiments` | **Compare multiple experiments** |
| `make rollback` | **Rollback to previous model version** |
| `make all` | Full deterministic chain |
| `make pytest` | Run lightweight unit tests |

## Responsible Testing

CI (GitHub Actions) executes `make all`, `make pytest`, and uploads `reports/` + `docs/` artefacts. Playwright installs Chromium headlessly for the Streamlit snapshot stage.

## Releases & Packages

Tag a version to publish wheels to PyPI and a container to GHCR:

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```

- **PyPI**: published automatically via Trusted Publishers (no API token).
- **GHCR image**: `ghcr.io/<owner>/oncotarget-lite:0.1.0` and `:latest`.

## Acknowledgements

Synthetic datasets derived from the original oncotarget-lite repo (GTEx, TCGA, DepMap, UniProt, STRING inspired). All content is synthetic and for demonstration purposes only.
