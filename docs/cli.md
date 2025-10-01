# CLI Reference

The `oncotarget-lite` console script consolidates all pipeline entry points. Every command accepts `--config-file` to load an alternative YAML configuration and emits JSON (tables are added automatically in interactive terminals).

## `system-info`

Show Python, OS, NumPy/Pandas/Torch versions, and CUDA availability.

```bash
oncotarget-lite system-info
```

## `config`

Inspect or validate configuration.

```bash
oncotarget-lite config --show
oncotarget-lite config --validate --config-file my-config.yaml
```

## `validate-data`

Strict schema validation across cached CSVs plus feature/label summary.

```bash
oncotarget-lite validate-data
```

## `train`

Run the full pipeline and emit artifacts under the chosen directory.

```bash
oncotarget-lite train --config-file my-config.yaml --out artifacts/ --device cpu --seed 1337
```

Output fields:

- `artifacts` – run directory containing metrics, predictions, feature importances, scores, lineage, and `model.pt`.
- `test_metrics` – AUROC, AUPRC, Brier, Expected Calibration Error.
- `bootstrap` – mean, lower, upper, and std per metric.

## `evaluate`

Recompute metrics from an existing run (no retraining).

```bash
oncotarget-lite evaluate --run-dir artifacts/<run_id>
```

Outputs both stored metrics and freshly recomputed metrics from `predictions.parquet`.

## `report`

Summarise key insights (top targets, metrics, bootstrap) for dashboards or docs.

```bash
oncotarget-lite report --run-dir artifacts/<run_id> --top-k 5
```

## `app run`

Run the Streamlit explorer on the supplied port (requires `[viz]` extras).

```bash
oncotarget-lite app run --port 8501
```

