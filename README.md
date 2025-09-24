# oncotarget-lite

**Production-ready ML pipeline for immunotherapy target discovery with comprehensive monitoring and governance.**

A sophisticated end-to-end machine learning pipeline featuring robust configuration management, advanced logging infrastructure, performance monitoring, and comprehensive validation for oncology target discovery workflows.

## ✨ New Features (v0.2.1)

- **🔧 Robust Configuration Management**: Pydantic-based configuration with environment variable overrides
- **📊 Performance Monitoring**: Real-time resource monitoring with memory optimization
- **📝 Advanced Logging**: Structured logging with MLflow integration and performance tracking  
- **✅ Enhanced Validation**: Comprehensive data validation with detailed error handling
- **🚀 Production CLI**: Enhanced command-line interface with monitoring and configuration management
- **📈 Bootstrap CIs**: Statistical confidence intervals and model calibration analysis
- **🔧 Memory Management**: Intelligent memory optimization and GPU cache management

## Quickstart

### Basic Setup
```bash
# Create environment with comprehensive dependencies
conda env create -f environment.yml
conda activate oncotarget-lite

# Validate installation and system info
python -m oncotarget_lite.cli system-info

# Run quality checks and tests
make lint test
```

### Configuration Management
```bash
# View current configuration
python -m oncotarget_lite.cli config --show

# Validate configuration
python -m oncotarget_lite.cli config --validate

# Use custom configuration file
cp config.example.yaml my_config.yaml
python -m oncotarget_lite.cli config --config-file my_config.yaml --show
```

### Enhanced Training Pipeline
```bash
# Train with comprehensive monitoring
python -m oncotarget_lite.cli train --model-type mlp --enable-monitoring

# Train with custom configuration
python -m oncotarget_lite.cli train --config-file my_config.yaml --model-type random_forest

# Validate data before training
python -m oncotarget_lite.cli validate-data --data-dir data/raw
```

### Interactive Applications
```bash
# Launch Streamlit app with SHAP explanations
make run-app  # or streamlit run app/streamlit_app.py

# Docker deployment (production-ready)
make build-img
docker run -it --rm -p 8501:8501 oncotarget-lite:latest
```

All pipelines run locally in under two minutes on a modern laptop CPU with comprehensive monitoring and validation.

## Repository Tour
- `data/raw/` tiny cached CSVs covering normal tissues (GTEx-like), tumor medians (TCGA-like), dependencies (DepMap-like), annotations (UniProt-like), and graph connectivity (STRING/BioGRID-like). Synthetic values emulate real magnitudes for ~50 genes (EPCAM, MSLN, CD276, etc.).
- `src/oncotarget/` load/validate data, engineer features, score candidates, train a PyTorch MLP, evaluate risk/safety, and visualize findings.
- `notebooks/` step through data QC, scorecard exploration, and model training with narrative context.
- `app/streamlit_app.py` interactive score explorer with SHAP-backed explanations and ADC traffic-light heuristic.
- `tests/` pytest suite covering IO, feature math, scoring monotonicity, and model reproducibility.

## Data Provenance & Licenses
All datasets are **synthetic** and generated offline from public summary statistics trends. Headers denote intended inspirations:
- GTEx v8 (normal tissues) median TPM
- TCGA Pan-Cancer Atlas (BRCA, LUAD, COAD)
- DepMap CRISPR screens (23Q4)
- UniProtKB annotations
- STRING/BioGRID network degree

The data contains no individual-level or HIPAA-protected information. You may treat the synthetic CSVs as CC0; the code is licensed MIT (see `LICENSE`).

## Reproducibility
- Deterministic seeds across feature generation, model training, and bootstrapped evaluation.
- `environment.yml` pins Python 3.11, PyTorch CPU, and analysis tooling.
- CI (`.github/workflows/ci.yml`) runs linting, type checks, and tests on pushes/PRs.
- Notebook outputs rely solely on cached datasets—no external network calls.

## Job Description Mapping
- **Integrative data mining**: `notebooks/01_data_mining_checks.ipynb`, `src/oncotarget/io.py`, and `tests/test_io.py` showcase schema enforcement and exploratory QC.
- **Feature engineering & target ideation**: `src/oncotarget/features.py` computes tumor-vs-normal contrasts, safety surrogates, and annotation-derived flags; `src/oncotarget/scoring.py` creates an interpretable scorecard.
- **ML modeling & evaluation**: `src/oncotarget/model.py` implements a PyTorch MLP with early stopping; `src/oncotarget/eval.py` delivers AUROC/AUPRC, calibration, Brier score, and bootstrap confidence intervals; tests cover determinism.
- **Visualization & stakeholder communication**: `src/oncotarget/viz.py`, Streamlit app, and `reports/example_target_report.md` provide bar plots, calibration curves, and narrative summaries.
- **Engineering best practices**: Pre-commit hooks, type hints, pytest coverage, Dockerfile with non-root user, and Makefile automation.

## Claims & Limits
- Synthetic data approximates but does not reproduce ground-truth biomedical distributions; do not draw biological conclusions.
- The scorecard weights are hand-tuned for pedagogy, not optimized for clinical deployment.
- The neural network is a small MLP intended to overfit the toy dataset for demonstration; do not extrapolate to unseen tumors.
- SHAP explanations operate on the synthetic feature matrix and inherit its biases.
- No attempt is made to model toxicity, HLA binding, patient stratification, or multi-modal omics integration.

## FT-Transformer + Masked-Gene Pretraining

This repository now includes **FT-Transformer** (Feature Tokenizer + Transformer) models for continuous tabular data, with optional **Masked Gene Modeling (MGM)** self-supervised pretraining.

### Basic Usage

```bash
# 1) Pretrain encoder with masked-gene modeling (MGM) - optional but recommended
python -m src.oncotarget.ssl_mgm \
  --epochs 10 --mask-frac 0.15 --save-encoder artifacts/ft_encoder.pt

# 2) Train FT-Transformer from scratch on supervised task
python -m src.oncotarget.train \
  --model fttransformer --epochs 100 --output-dir artifacts/

# 3) OR fine-tune pretrained encoder on supervised task  
python -m src.oncotarget.train \
  --model fttransformer --pretrained-encoder artifacts/ft_encoder.pt \
  --epochs 100 --output-dir artifacts/

# 4) Evaluate trained model with comprehensive metrics
python -m src.oncotarget.evaluate artifacts/fttransformer_binary_model.pt \
  --bootstrap --feature-importance --save-predictions
```

### Key Features

- **FT-Transformer**: Continuous feature tokenization + multi-head attention for tabular data
- **MGM Pretraining**: Self-supervised masked gene reconstruction (15% random masking)
- **Transfer Learning**: Load pretrained encoder weights into supervised classification head
- **Backwards Compatible**: Works alongside existing MLP models and evaluation pipeline
- **Comprehensive Testing**: Unit tests cover model components, training loops, and end-to-end workflows

### Model Architecture

- **FeatureTokenizer**: Projects each continuous feature to `d_model` dimensions with layer normalization
- **TransformerEncoder**: Multi-head self-attention with configurable depth (2-4 layers recommended)
- **Task Heads**: Binary classification, regression, or feature reconstruction
- **CLS Token**: Learnable classification token for supervised tasks (disabled for reconstruction)

### Next Steps
1. Integrate experiment tracking (e.g., DVC + MLflow) and versioned data lineage.
2. Expand cohorts with real GTEx/TCGA harmonized matrices and batch effect corrections.
3. Incorporate clinical covariates (immune infiltrate, MSI, TMB) and spatial transcriptomics priors.
4. Add multi-specific epitope filters and linker-aware ADC design heuristics.
5. Explore graph neural networks using curated interactomes and ligand-receptor context.

## Responsible Use
This project is for educational illustration only. Always validate findings with domain experts, orthogonal assays, and appropriate ethical oversight before moving toward clinical translation.
