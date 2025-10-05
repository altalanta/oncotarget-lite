PYTHON ?= python3
VENV ?= .venv
PIP ?= $(VENV)/bin/pip
PY ?= $(VENV)/bin/python

.DEFAULT_GOAL := all

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

setup: $(VENV)/bin/activate
	@echo "Virtual environment ready: $(VENV)"

.PHONY: devdata
devdata:
	python scripts/generate_synthetic_data.py

prepare: devdata
	python -m oncotarget_lite.cli prepare

train: prepare
	$(PY) -m oncotarget_lite.cli train

.PHONY: evaluate
# "eval" is a reserved keyword in some shells; expose as "evaluate"
evaluate: train
	$(PY) -m oncotarget_lite.cli eval

explain: evaluate
	$(PY) -m oncotarget_lite.cli explain

scorecard: explain
	$(PY) -m oncotarget_lite.cli scorecard

snapshot: report-docs
	$(PY) -m oncotarget_lite.cli snapshot

report-docs: scorecard
	$(PY) -m oncotarget_lite.cli docs

ablations: prepare
	$(PY) -m oncotarget_lite.cli train --all-ablations
	$(PY) -m oncotarget_lite.cli ablations

app: setup
	$(PY) -m streamlit run oncotarget_lite/app.py

all: setup devdata prepare train evaluate explain scorecard report-docs snapshot

clean:
	rm -rf $(VENV) mlruns models reports dvcstore docs/index.html
	rm -f reports/run_context.json

pytest:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

mypy:
	$(PY) -m mypy oncotarget_lite

bench:
	$(PY) -m scripts.eval_small_benchmark

docs-metrics:
	$(PY) -m scripts.render_docs_metrics

ci:
	make all
	make pytest

.PHONY: setup devdata prepare train evaluate explain scorecard snapshot report-docs ablations app all clean pytest lint format mypy ci bench docs-metrics

dvc.repro:
	dvc repro
