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
