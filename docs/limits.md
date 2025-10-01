# Limits

- **Synthetic-only data** – the CSV caches emulate public summaries and are not suitable for biological conclusions.
- **Toy MLP** – the PyTorch network is intentionally small and optimised for fast CPU experimentation.
- **Scorecard heuristics** – weights are hand-tuned for pedagogy and ignore toxicity, patient stratification, and downstream druggability constraints.
- **Single-modality** – no integration of genomic, proteomic, or spatial datasets beyond the provided summaries.
- **Bootstrap scope** – confidence intervals rely on 1D resampling of the tiny dataset; results should be interpreted qualitatively.
- **Optional MLflow** – tracking is disabled by default to guarantee offline use; enable via `ONCOTARGET_LITE_MLFLOW=1` only when a local server is available.

