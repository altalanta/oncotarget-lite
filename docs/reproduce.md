# Reproduce Figures

The repository includes the same cached synthetic datasets used to produce the figures described in the README. Everything runs locally in well under two minutes.

1. **Train the model** (captures metrics, calibration, feature importances):

    ```bash
    oncotarget-lite train --device cpu --out artifacts/ --seed 2024
    ```

2. **Capture the run directory** by copying the `artifacts` path from the CLI JSON output.

3. **Plot calibration** (example using pandas + matplotlib):

    ```python
    import json
    import pandas as pd
    from pathlib import Path

    run_dir = Path("artifacts/<run_id>")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    calib = pd.DataFrame(metrics["calibration_curve"])
    ax = calib.plot(x="confidence", y="event_rate", marker="o", title="Calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical event rate")
    ```

4. **Scorecard highlights**: load `scores.parquet` to rank targets and break down component contributions.

5. **Streamlit explorer**: `oncotarget-lite app run --port 8501` reproduces the interactive explorer with SHAP explanations and bootstrap metrics.

Because the data, training configuration, and bootstrap seeds are deterministic, repeating the steps above yields identical metrics and score rankings, matching the original demo figures.

