SHELL := /bin/bash

PYTHON ?= python3
UV ?= uv
FAST ?= 0
PROFILE ?=

FAST_BOOL := $(if $(filter 1 true yes,$(FAST)),1,0)
FAST_FLAG := $(if $(filter 1 true yes,$(FAST)),--fast,)
PROFILE_ARGS :=
ifeq ($(FAST_BOOL),1)
PROFILE_ARGS += --ci
endif
ifneq ($(PROFILE),)
PROFILE_ARGS += --profile $(PROFILE)
endif

HAS_UV := $(shell command -v $(UV) >/dev/null 2>&1 && echo 1 || echo 0)
PYTHON_RUN := $(PYTHON)
RUN_CMD :=
ifeq ($(HAS_UV),1)
PYTHON_RUN := $(UV) run python
RUN_CMD := $(UV) run
endif

LOCKFILE := uv.lock
ENV_EXPORT := ONCOTARGET_LITE_FAST=$(FAST_BOOL)
CLI := $(ENV_EXPORT) $(PYTHON_RUN) -m oncotarget_lite.cli $(PROFILE_ARGS)

.DEFAULT_GOAL := all

.PHONY: setup sync download-data prepare train optimize evaluate explain dashboard cache scorecard snapshot docs docs-targets \
    monitor-report interpretability-validate model-card docs-monitoring docs-interpretability mkdocs-build \
    all clean pytest lint format mypy bandit pre-commit security ablations distributed monitor \
    validate-interpretability retrain deploy versions cleanup serve rollback compare compare-interactive \
    validate-data lineage drift experiment experiments experiment-report compare-experiments export-requirements docker.build.cpu docker.build.cuda docker.push.cpu docker.push.cuda

setup:
ifeq ($(HAS_UV),1)
	@if [ -f $(LOCKFILE) ]; then \
		$(UV) sync --frozen; \
	else \
		$(UV) sync; \
	fi
else
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
endif
	$(ENV_EXPORT) $(PYTHON_RUN) scripts/download_data.py $(FAST_FLAG)

sync: setup

download-data:
	$(ENV_EXPORT) $(PYTHON_RUN) scripts/download_data.py $(FAST_FLAG)

prepare:
	$(CLI) prepare

train:
	$(CLI) train

optimize:
	$(CLI) optimize

evaluate:
	$(CLI) eval

explain:
	$(CLI) explain

dashboard:
	$(CLI) dashboard

cache:
	$(CLI) cache

scorecard:
	$(CLI) scorecard

snapshot:
	$(CLI) snapshot

monitor:
	$(CLI) monitor status

monitor-report:
	$(CLI) monitor report || true

interpretability-validate:
	$(CLI) validate-interpretability --summary-only || true

model-card:
	$(ENV_EXPORT) $(PYTHON_RUN) scripts/generate_model_card.py

docs-monitoring: monitor-report
	$(ENV_EXPORT) $(PYTHON_RUN) - <<'PY'
from pathlib import Path
import json

output = Path("docs/monitoring.md")
output.parent.mkdir(parents=True, exist_ok=True)
report_path = Path("reports/monitoring_report.json")

lines = ["# Monitoring", ""]
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    snapshots = report.get("snapshots_count", 0)
    alerts = report.get("alerts_count", 0)
	lines.append(f"Snapshots analysed: {snapshots}")
	lines.append(f"Active alerts: {alerts}")
	latest = report.get("latest_performance") or {}
	if latest:
		lines.append("")
		lines.append("## Latest Performance")
		for key in ("auroc", "ap", "accuracy", "f1"):
			if key in latest:
				value = latest[key]
				if isinstance(value, (int, float)):
					lines.append(f"- {key.upper()}: {value:.3f}")
				else:
					lines.append(f"- {key.upper()}: {value}")
else:
    lines.append("Monitoring report not generated. Run `make monitor-report` after a pipeline run.")

output.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

docs-interpretability: interpretability-validate
	$(ENV_EXPORT) $(PYTHON_RUN) - <<'PY'
from pathlib import Path
import json

output = Path("docs/interpretability.md")
output.parent.mkdir(parents=True, exist_ok=True)
report_path = Path("reports/interpretability_validation/validation_report.json")

lines = ["# Interpretability Validation", ""]
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    quality = report.get("explanation_quality", {})
	if quality:
		lines.append("## Quality Metrics")
		for key, value in quality.items():
			label = key.replace('_', ' ').title()
			if isinstance(value, (int, float)):
				lines.append(f"- {label}: {value:.3f}")
			else:
				lines.append(f"- {label}: {value}")
    scores = report.get("background_consistency_scores", {})
	if scores:
		lines.append("")
		lines.append("## Background Consistency")
		for bg, score in scores.items():
			if isinstance(score, (int, float)):
				lines.append(f"- Background {bg}: {score:.3f}")
			else:
				lines.append(f"- Background {bg}: {score}")
else:
    lines.append("Interpretability validation report not generated. Run `make interpretability-validate`.")

output.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

docs-targets: model-card docs-monitoring docs-interpretability
	$(ENV_EXPORT) $(PYTHON_RUN) scripts/render_docs_metrics.py || true

mkdocs-build:
ifeq ($(HAS_UV),1)
	$(RUN_CMD) mkdocs build --strict
else
	mkdocs build --strict
endif

docs: docs-targets mkdocs-build

ablations:
	$(CLI) train --all-ablations
	$(CLI) ablations

distributed:
	$(CLI) distributed

validate-interpretability:
	$(CLI) validate-interpretability

retrain:
	$(CLI) retrain --schedule

rollback:
	$(CLI) rollback --force-rollback

deploy:
	@echo "Usage: make deploy VERSION_ID=<version_id>"
	@echo "Example: make deploy VERSION_ID=model_20250101_123456_abc12345"

versions:
	$(CLI) versions

cleanup:
	$(CLI) cleanup --dry-run

serve:
	$(CLI) serve

compare:
	$(CLI) compare

compare-interactive:
	$(CLI) compare-interactive

validate-data:
	@echo "Usage: make validate-data DATA_PATH=<path> DATASET_ID=<id>"
	@echo "Example: make validate-data DATA_PATH=data/processed/features.parquet DATASET_ID=prepared_data"

lineage:
	$(CLI) lineage

drift:
	@echo "Usage: make drift CURRENT=<current_id> REFERENCE=<reference_id>"
	@echo "Example: make drift CURRENT=prepared_data_new REFERENCE=prepared_data_old"

experiment:
	@echo "Usage: make experiment NAME=<experiment_name> [MODEL_TYPES=<types>] [TRIALS=<n>]"
	@echo "Example: make experiment NAME=baseline_optimization MODEL_TYPES=logreg,xgb TRIALS=100"

experiments:
	$(CLI) experiments

experiment-report:
	@echo "Usage: make experiment-report EXPERIMENT_ID=<id>"
	@echo "Example: make experiment-report EXPERIMENT_ID=exp_baseline_optimization_1234567890"

compare-experiments:
	@echo "Usage: make compare-experiments IDS=<id1,id2,id3>"
	@echo "Example: make compare-experiments IDS=exp_1,exp_2,exp_3"

all: setup prepare train optimize evaluate explain dashboard cache scorecard docs snapshot

pytest:
	$(ENV_EXPORT) $(PYTHON_RUN) -m pytest -q

lint:
	$(ENV_EXPORT) $(PYTHON_RUN) -m ruff check .

format:
	$(ENV_EXPORT) $(PYTHON_RUN) -m ruff format .

mypy:
	$(ENV_EXPORT) $(PYTHON_RUN) -m mypy oncotarget_lite

bandit:
	$(ENV_EXPORT) $(PYTHON_RUN) -m bandit -r oncotarget_lite -c pyproject.toml

pre-commit:
ifeq ($(HAS_UV),1)
	$(RUN_CMD) pre-commit run --all-files
else
	pre-commit run --all-files
endif

security:
	$(ENV_EXPORT) $(PYTHON_RUN) -m scripts.security_scan

export-requirements:
ifeq ($(HAS_UV),1)
	UV_CACHE_DIR=.uv-cache $(UV) export --format=requirements-txt --no-hashes > requirements.txt
else
	@echo "uv not available; skipping requirements export" >&2
endif

docker.build.cpu:
	docker build --file Dockerfile.cpu --target runtime --build-arg FAST=$(FAST_BOOL) --tag oncotarget-lite:cpu .

docker.build.cuda:
	docker build --file Dockerfile.cuda --target runtime --build-arg FAST=$(FAST_BOOL) --tag oncotarget-lite:cuda .

docker.push.cpu:
	docker push oncotarget-lite:cpu

docker.push.cuda:
	docker push oncotarget-lite:cuda

clean:
	rm -rf .venv __pycache__ mlruns models reports docs/site .pytest_cache .mypy_cache .ruff_cache
	rm -f reports/run_context.json
