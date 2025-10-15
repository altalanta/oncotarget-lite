# Reproducibility Toolkit

The project emphasises deterministic builds and experiment tracking so that results can be
recreated on laptops or in CI.

## Environments

- `uv.lock` and `pyproject.toml` pin runtime and tooling dependencies.
- `make setup` installs the environment via `uv sync` (falling back to pip) and ensures
  datasets defined in `data/manifest.json` are present.
- `.devcontainer/devcontainer.json` bootstraps a container with DVC, MLflow, and the package in
  editable mode for consistent local development.

## Configuration Profiles

- `params.yaml` captures the default hyper-parameters used for full runs.
- `params.ci.yaml` defines the compact CI profile. Activate it with `make all FAST=1` or
  `oncotarget-lite --ci ...`.
- `.env.example` documents notification secrets that are automatically loaded by the CLI when
  present.

## Data Provenance

- `scripts/download_data.py` reads `data/manifest.json` and verifies file hashes.
- Dataset fingerprints are embedded into MLflow tags and surfaced in the generated model card.

## Artefact Tracking

- MLflow runs are stored under `mlruns/` and summarised in `docs/model_card.md`.
- DVC caches each stage output while keeping human-readable artefacts (reports, docs) outside the
  cache for convenience.
