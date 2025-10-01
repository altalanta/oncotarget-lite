# Data Schema

All inputs live under `data/raw/` and are synthetic CSV caches. Each file is validated with Pydantic + polars before any training happens.

## GTEx subset (`GTEx_subset.csv`)

| Column | Type | Description |
| --- | --- | --- |
| `gene` | string | HGNC symbol |
| `tissue` | string | Tissue label (10 GTEx-inspired tissues) |
| `median_TPM` | float ≥ 0 | Median TPM across donors |

## TCGA subset (`TCGA_subset.csv`)

| Column | Type | Description |
| --- | --- | --- |
| `gene` | string | HGNC symbol |
| `tumor` | string | Tumour cohort (`BRCA`, `LUAD`, `COAD`) |
| `median_TPM` | float ≥ 0 | Median TPM across tumours |

## DepMap essentials (`DepMap_essentials_subset.csv`)

- One row per gene, wide format with columns for each cell line (float dependency scores, negative means essential).
- Parsed into a dictionary per row and validated to ensure every gene exposes at least one score.

## UniProt annotations (`uniprot_annotations.csv`)

| Column | Type | Constraints |
| --- | --- | --- |
| `gene` | string | HGNC symbol |
| `is_cell_surface` | bool | Label used for supervised training |
| `signal_peptide` | bool | Presence of signal peptide |
| `ig_like_domain` | bool | Ig-like domain indicator |
| `protein_length` | int ≥ 50 | Amino acid length |

## PPI degree (`ppi_degree_subset.csv`)

| Column | Type | Description |
| --- | --- | --- |
| `gene` | string | HGNC symbol |
| `degree` | int ≥ 0 | STRING/BioGRID-inspired degree |

## Feature Engineering Highlights

- Log2 fold-change per tumour vs. mean normal expression (`log2fc_*`).
- Minimum normal TPM (`min_normal_tpm`) and mean tumour TPM (`mean_tumor_tpm`).
- Mean dependency score across DepMap columns (`mean_dependency`).
- Safety proxies (PPI degree, signal peptide, Ig-like domain, protein length).
- Labels derived from `is_cell_surface` (positive class) after dropping rows with any missing feature.

