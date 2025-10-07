# oncotarget-lite

[![CI](https://github.com/altalanta/oncotarget-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/altalanta/oncotarget-lite/actions/workflows/ci.yml)
[![Release](https://github.com/altalanta/oncotarget-lite/actions/workflows/release.yml/badge.svg)](https://github.com/altalanta/oncotarget-lite/actions/workflows/release.yml)
[![PyPI version](https://badge.fury.io/py/oncotarget-lite.svg)](https://badge.fury.io/py/oncotarget-lite)
![License](https://img.shields.io/badge/license-MIT-blue)
![Reproducible Run](https://img.shields.io/badge/run-deterministic-success)

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

| Model | Features | Test AUROC | 95% CI | Interpretation |
| --- | --- | --- | --- | --- |
| LogReg | All | 0.850 | [0.830, 0.870] | Strong baseline |
| XGBoost | All | 0.855 | [0.835, 0.875] | Best performance |
| MLP | All | 0.842 | [0.820, 0.864] | Competitive |
| LogReg | Network Only | 0.810 | [0.785, 0.835] | Network features powerful |
| LogReg | Clinical Only | 0.720 | [0.685, 0.755] | Limited clinical signal |

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
| `make evaluate` | Compute metrics, bootstrap CIs, and calibration artefacts |
| `make explain` | Generate SHAP PNGs and `shap_values.npz` |
| `make scorecard` | Build `reports/target_scorecard.html` |
| `make report-docs` | Refresh `docs/index.html` and model card metrics |
| `make snapshot` | Capture Streamlit UI screenshot via Playwright |
| **`make ablations`** | **Run all ablation experiments and generate analysis** |
| **`make app`** | **Launch interactive Streamlit triage UI** |
| **`make distributed`** | **Configure distributed computing settings** |
| **`make security`** | **Run security audit on dependencies** |
| **`make monitor`** | **Check model performance and drift status** |
| **`make validate-interpretability`** | **Run interpretability validation on SHAP explanations** |
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
