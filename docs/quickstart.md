# Quickstart

## Install (CPU)

```bash
python -m pip install --upgrade pip
pip install oncotarget-lite
```

For editable development:

```bash
git clone https://github.com/altalanta/oncotarget-lite.git
cd oncotarget-lite
make setup        # installs via uv.lock when present
make all FAST=1   # compact CI profile (under 10 minutes on 2 cores)
```

## Run Individual Stages

```bash
oncotarget-lite --ci prepare
oncotarget-lite --ci train
oncotarget-lite --ci eval
oncotarget-lite --ci explain
oncotarget-lite scorecard --reports-dir reports --shap-dir reports/shap
```

Use `oncotarget-lite all` for the full profile or add `--fast` / `--ci` to shrink workloads.

## Streamlit Demo

```bash
oncotarget-lite app run --port 8501
```

Visit <http://localhost:8501> to explore metrics and SHAP explanations. Install the optional
`[cuda]` extra for GPU-enabled containers or `[docs]` to build the documentation locally.
