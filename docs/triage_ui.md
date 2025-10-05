# Triage UI User Guide

The oncotarget-lite Streamlit triage UI provides an interactive interface for exploring model predictions, explanations, and ablation results.

## Quick Start

Launch the application:

```bash
make app
# or directly:
streamlit run oncotarget_lite/app.py
```

The UI will be available at `http://localhost:8501`

## Authentication

The app supports optional token-based authentication via the `ONCOTARGET_APP_TOKEN` environment variable:

```bash
export ONCOTARGET_APP_TOKEN="your-secret-token"
make app
```

If no token is set, the app runs in open mode (useful for CI screenshots).

## Interface Overview

### Sidebar Controls

**Model Run Selection**
- Select from recent MLflow runs
- Or manually enter run ID, tag, or git SHA
- Examples: `best=True`, `abc12345`, `git_sha=abc123`

**Filters**
- **Pathway/GO term filter**: Filter targets by pathway annotations
- **Minimum score**: Only show targets above prediction threshold  
- **Max uncertainty**: Filter out high-uncertainty predictions

### Main Views

#### 🎯 Ranking View
- Displays targets ranked by prediction score
- Shows pass rates and filtering metrics
- Downloadable CSV export of filtered results
- Top 20 targets with actual labels for validation

#### 🔍 Per-target SHAP View  
- Global feature importance plot
- Individual gene SHAP explanations
- Select specific genes for detailed analysis
- Visual feature attribution for model decisions

#### ⚖️ Compare Runs View
- Model performance comparison across ablations
- Feature impact analysis
- Links to detailed ablation reports
- Interactive charts and tables

## Data Sources

The UI automatically loads from the reports directory:

- `reports/metrics.json` - Current model performance
- `reports/predictions.parquet` - Target predictions and scores
- `reports/shap/` - SHAP explanations and plots
- `reports/ablations/metrics.csv` - Ablation study results

## Usage Patterns

### Target Discovery Workflow
1. Set minimum score threshold (e.g., 0.7)
2. Apply pathway filter if interested in specific biology
3. Review top-ranked targets in Ranking view
4. Investigate promising targets using SHAP explanations
5. Download filtered target list for experimental validation

### Model Validation Workflow  
1. Compare runs in the Compare view
2. Analyze feature contributions via SHAP
3. Check calibration and performance metrics
4. Review ablation results for feature importance

### Quality Assurance
- Monitor uncertainty levels to identify low-confidence predictions
- Compare actual vs predicted labels on known targets
- Use SHAP to verify biological plausibility of predictions

## Performance Notes

- The UI loads up to 50 genes for SHAP analysis (for responsiveness)
- Large prediction datasets are automatically sampled for display
- Charts and plots are cached for faster navigation

## Links and Export

The UI provides quick access to:
- Target scorecard HTML report
- Model card documentation  
- Raw CSV downloads of filtered results
- Links to detailed ablation analysis

## Screenshots

![Triage UI Overview](../reports/streamlit_demo.png)

*The CI pipeline automatically captures screenshots of the UI for documentation.*