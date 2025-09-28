# oncotarget-lite

[![CI](https://github.com/altalanta/oncotarget-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/altalanta/oncotarget-lite/actions/workflows/ci.yml)
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
- DVC pipeline (`dvc.yaml`) with stages `prepare → train → eval → explain → scorecard` plus local remote at `./dvcstore`. Re-run everything with `dvc repro`.
- MLflow experiment `oncotarget-lite` writes params, metrics, model binaries, dataset hash, and git commit into `./mlruns`.
- `reports/run_context.json` links downstream stages to the originating MLflow run for audit trails.

## Evaluation

Offline artefacts live in `reports/` and are persisted via DVC (`persist: true`).

<!-- README_METRICS_START -->
_No metrics captured yet. Run `make all` to refresh this table._
<!-- README_METRICS_END -->

Key files:

- `reports/metrics.json` – point estimates and 95% CIs (AUROC/AP/Brier/ECE/Accuracy/F1/overfit gap).
- `reports/bootstrap.json` – bootstrap summaries (n, lower/upper bounds).
- `reports/calibration.json` & `reports/calibration_plot.png` – reliability curve data and PNG.

## Interpretability & Insights

- `python -m oncotarget_lite.cli explain` emits SHAP values under `reports/shap/` with:
  - `global_summary.png` mean |SHAP| bar chart.
  - `example_GENE{1,2,3}.png` per-gene cards.
  - `shap_values.npz` + `alias_map.json` for downstream analysis.
- `reports/target_scorecard.html` links predicted scores, rankings, SHAP PNGs, and top positive/negative contributors.

## Model Card & Governance

- Responsible AI summary lives at `oncotarget_lite/model_card.md` and is auto-updated by `python -m oncotarget_lite.cli docs`.
- Docs landing page (`docs/index.html`) references metrics, calibration plots, scorecard, model card, and MLflow run ID.
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
| `make all` | Full deterministic chain |
| `make pytest` | Run lightweight unit tests |

## Responsible Testing

CI (GitHub Actions) executes `make all`, `make pytest`, and uploads `reports/` + `docs/` artefacts. Playwright installs Chromium headlessly for the Streamlit snapshot stage.

## Acknowledgements

Synthetic datasets derived from the original oncotarget-lite repo (GTEx, TCGA, DepMap, UniProt, STRING inspired). All content is synthetic and for demonstration purposes only.
