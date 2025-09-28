# oncotarget-lite Model Card

## Intended Use
- **Primary goal:** educational demonstration of oncology target triage workflows on synthetic data.
- **Intended users:** ML engineers reviewing interpretability, governance, and reproducibility tooling.
- **Usage constraints:** not for clinical decision making or patient-level inference. Outputs illustrate tooling only.

## Synthetic Data Recipe
- Aggregates cached CSV snapshots emulating GTEx, TCGA, DepMap, UniProt, and protein–protein interaction statistics.
- Introduces controlled label noise (10%) and mild distributional shifts between train/test to stress calibration analyses.
- No human subject data – all records are procedurally generated.

## Preprocessing & Features
- Deterministic merge by gene symbol with per-tissue expression pivoting (normal/tumour pairs).
- Feature engineering:
  - Log2 fold-change for each tumour cohort vs. pooled normal baseline.
  - Minimum normal TPM, mean tumour TPM, mean dependency score.
  - Binary annotations: signal peptide, Ig-like domain; continuous features: protein length, PPI degree.
- Stratified 70/30 train/test split with `random_state=42`.

## Model
- Scikit-learn logistic regression (`class_weight="balanced"`, `C=1.0`, `max_iter=500`) wrapped with `StandardScaler`.
- Deterministic seeds + `PYTHONHASHSEED=0` to guarantee reproducibility.
- Training artefacts logged to MLflow (run parameters, metrics, dataset hash, git commit).

## Metrics & Calibration
The table below is auto-updated from `reports/metrics.json` and `reports/bootstrap.json` when `python -m oncotarget_lite.cli docs` runs.

<!-- METRICS_TABLE_START -->
_No evaluation metrics generated yet. Run `make all` to populate this section._
<!-- METRICS_TABLE_END -->

Calibration resources:
- Reliability curve (`reports/calibration_plot.png`).
- `reports/calibration.json` contains per-bin accuracy, confidence, and sample counts.

## Failure Modes & Risks
- **Dataset shift:** synthetic distributions may not match real tumour biology; expect drift when retrained on new cohorts.
- **Spurious correlations:** fold-change heuristics can promote genes with platform-specific artefacts.
- **Small sample bootstrap:** 95% CIs use stratified bootstrap with replacement; intervals widen dramatically for rare positives.

## Governance & Stewardship
- Pipeline orchestrated via Typer CLI (`prepare → train → eval → explain → scorecard → snapshot`).
- DVC tracks data lineage and artefact dependencies; reproduce with `dvc repro`.
- MLflow experiment `oncotarget-lite` stores run metadata; launch UI with `mlflow ui --backend-store-uri ./mlruns`.
- Schedule quarterly audits: regenerate synthetic caches, rerun pipeline with updated seeds, and review calibration drift.
- Issues / contact: open a GitHub issue in the project repository.
