# Quickstart

## Install (CPU-only)

```bash
python -m pip install --upgrade pip
pip install oncotarget-lite[viz]
```

For editable development:

```bash
git clone https://github.com/altalanta/oncotarget-lite.git
cd oncotarget-lite
pip install -e .[dev,viz]
```

## Validate the setup

```bash
oncotarget-lite system-info
oncotarget-lite validate-data
```

## Train → Evaluate → Report (≤2 minutes on a laptop CPU)

```bash
oncotarget-lite train --device cpu --out artifacts/
# copy the "artifacts" path from the JSON response
RUN_DIR="artifacts/<run_id_from_output>"
oncotarget-lite evaluate --run-dir "$RUN_DIR"
oncotarget-lite report --run-dir "$RUN_DIR"
```

!!! tip
    Every CLI command emits JSON regardless of environment. When running in a TTY you also get compact tables for key metrics and rankings.

## Launch the app

```bash
oncotarget-lite app run --port 8501
```

Then open <http://localhost:8501> to explore predictions and SHAP explanations (install with `[viz]`).

