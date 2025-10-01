# Model Card

## Model Details

- **Model**: Two-layer PyTorch MLP classifier (`[64, 32]` hidden units, ReLU, dropout 0.15).
- **Task**: Binary classification – prioritise cell-surface proteins with favourable tumour/normal profiles.
- **Version**: `oncotarget-lite` package version (`onctotarget_lite.__version__`).
- **License**: MIT (code), CC0 (synthetic data).

## Intended Use

Educational demonstration of reproducible oncology target triage on laptops. Suitable for showcasing data contracts, deterministic ML pipelines, and artifact governance.

## Factors

- **Data**: Synthetic tables mimicking GTEx, TCGA, DepMap, UniProt, STRING distributions for ~50 genes.
- **Features**: Log2 fold-change, mean/min expression, dependency scores, protein annotations.
- **Labels**: `is_cell_surface` boolean derived from synthetic UniProt annotations.

## Metrics

- AUROC, AUPRC, Brier score, Expected Calibration Error.
- 95% bootstrap confidence intervals (default 256 resamples) written to `metrics.json`.

## Evaluation Data

Uses held-out stratified test split (20%). No external validation.

## Ethical Considerations

- Synthetic data only – do not infer real biological behaviour.
- No patient privacy concerns (all caches generated offline).
- Model is not suitable for clinical decisions or therapeutic prioritisation.

## Caveats and Recommendations

- Use as a template for engineering practices, not as a biological model.
- Replace synthetic caches with verified datasets before any real analysis and re-run schema validation.
- Retrain with domain-specific evaluation metrics and toxicity screening for realistic applications.

