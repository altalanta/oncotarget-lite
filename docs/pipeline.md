# Pipeline Overview

The oncotarget-lite pipeline is orchestrated with DVC and Typer commands. Each stage produces
versioned artefacts so that model training and evaluation remain deterministic.

## DVC Stages

- `fetch_data`: Generate or fetch synthetic raw inputs and record hashes in `data/manifest.json`.
- `prepare`: Build feature matrices and stratified splits with `oncotarget-lite prepare`.
- `train`: Fit the baseline logistic regression model and capture predictions/artefacts.
- `eval`: Compute metrics, calibration curves, and bootstrap confidence intervals.
- `explain`: Produce SHAP explanations and PNG summaries for dashboards.
- `scorecard`: Combine metrics, ablations, and interpretability into an HTML report.
- `ablations_train` / `ablations_eval`: Optional experiments over configuration sweeps.
- `scorecard_merge`: Publish the final `reports/target_scorecard_final.html` bundle.

Each target reads defaults from `params.yaml` and can be overridden via `params.ci.yaml` (used in
CI) or via the `--profile`/`--fast` CLI options.

## Command Relationship

```text
make all
 ├─ oncotarget-lite prepare
 ├─ oncotarget-lite train
 ├─ oncotarget-lite eval
 ├─ oncotarget-lite explain
 ├─ oncotarget-lite scorecard
 └─ oncotarget-lite docs
```

Use `dvc repro` to execute the end-to-end pipeline respecting the stage dependencies or
`make all FAST=1` for the compact profile used during automation.
