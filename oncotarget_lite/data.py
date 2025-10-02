"""Data preparation utilities for the oncotarget-lite pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("gene", "median_TPM")
from sklearn.model_selection import train_test_split

from .utils import dataset_hash, ensure_dir, save_dataframe, save_json, set_seeds

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


class DataPreparationError(RuntimeError):
    """Raised when cached synthetic data are missing or invalid."""


@dataclass(slots=True)
class PreparedData:
    features: pd.DataFrame
    labels: pd.Series
    train_genes: list[str]
    test_genes: list[str]
    dataset_fingerprint: str


RAW_FILES = {
    "gtex": "GTEx_subset.csv",
    "tcga": "TCGA_subset.csv",
    "depmap": "DepMap_essentials_subset.csv",
    "annotations": "uniprot_annotations.csv",
    "ppi": "ppi_degree_subset.csv",
}


_DEF_TUMOUR_PREFIX = "tumor_"
_DEF_NORMAL_PREFIX = "normal_"
_DEF_DEP_PREFIX = "dep_"
_REQUIRED_COLUMNS = {
    "is_cell_surface",
    "signal_peptide",
    "ig_like_domain",
    "ppi_degree",
    "protein_length",
}


def _read_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV loader:
    - Ignores lines starting with '#' (metadata/comment headers).
    - Strips BOM if present.
    - Provides actionable error if required columns are missing.
    """
    if not path.exists():
        raise DataPreparationError(f"Missing synthetic data file: {path}")
    
    df = pd.read_csv(path, comment="#")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required columns {missing}; "
            f"got {list(df.columns)}. Ensure comment headers start with '#'."
        )
    return df


def _load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in RAW_FILES.items():
        tables[key] = _read_csv(raw_dir / filename)
    return tables


def _wide_expression(frame: pd.DataFrame, pivot_col: str, prefix: str) -> pd.DataFrame:
    wide = frame.pivot_table(index="gene", columns=pivot_col, values="median_TPM")
    wide = wide.add_prefix(prefix)
    return wide.sort_index()


def _merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gtex_wide = _wide_expression(tables["gtex"], "tissue", _DEF_NORMAL_PREFIX)
    tcga_wide = _wide_expression(tables["tcga"], "tumor", _DEF_TUMOUR_PREFIX)

    depmap = tables["depmap"].set_index("gene").add_prefix(_DEF_DEP_PREFIX).sort_index()
    annotations = tables["annotations"].set_index("gene").sort_index()
    ppi = tables["ppi"].set_index("gene").rename(columns={"degree": "ppi_degree"}).sort_index()

    pieces: Iterable[pd.DataFrame] = (gtex_wide, tcga_wide, depmap, annotations, ppi)
    merged = pd.concat(pieces, axis=1, join="inner").sort_index()
    if merged.empty:
        raise DataPreparationError("Merged feature table is empty")
    missing = _REQUIRED_COLUMNS - set(merged.columns)
    if missing:
        raise DataPreparationError(f"Merged feature table missing columns: {sorted(missing)}")
    return merged


def build_feature_matrix(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Load cached CSVs and derive model-ready features + labels."""

    tables = _load_raw_tables(raw_dir)
    merged = _merge_tables(tables)

    normal_cols = [col for col in merged if col.startswith(_DEF_NORMAL_PREFIX)]
    tumour_cols = [col for col in merged if col.startswith(_DEF_TUMOUR_PREFIX)]
    dep_cols = [col for col in merged if col.startswith(_DEF_DEP_PREFIX)]

    normal = merged[normal_cols]
    tumour = merged[tumour_cols]
    dependency = merged[dep_cols]

    normal_mean = normal.mean(axis=1)
    tumour_mean = tumour.mean(axis=1)

    features = pd.DataFrame(index=merged.index)
    for col in tumour_cols:
        tumour_name = col.replace(_DEF_TUMOUR_PREFIX, "").lower()
        features[f"log2fc_{tumour_name}"] = np.log2((merged[col] + 1.0) / (normal_mean + 1.0))
    features["min_normal_tpm"] = normal.min(axis=1)
    features["mean_tumor_tpm"] = tumour_mean
    features["mean_dependency"] = dependency.mean(axis=1)
    features["ppi_degree"] = merged["ppi_degree"].astype(float)
    features["signal_peptide"] = merged["signal_peptide"].astype(float)
    features["ig_like_domain"] = merged["ig_like_domain"].astype(float)
    features["protein_length"] = merged["protein_length"].astype(float)

    labels = merged["is_cell_surface"].astype(int)
    return features, labels


def prepare_dataset(
    *,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    test_size: float = 0.3,
    seed: int = 42,
) -> PreparedData:
    """Generate the canonical processed artefacts for downstream stages."""

    set_seeds(seed)
    features, labels = build_feature_matrix(raw_dir)

    if labels.sum() == 0 or labels.sum() == len(labels):
        raise DataPreparationError("Labels are degenerate; need both positive and negative samples")

    train_genes, test_genes = train_test_split(
        list(features.index),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    ordered_index = sorted(features.index)
    dataset_fp = dataset_hash(features.loc[ordered_index], labels.loc[ordered_index])

    ensure_dir(processed_dir)
    save_dataframe(processed_dir / "features.parquet", features)
    save_dataframe(processed_dir / "labels.parquet", labels.to_frame(name="label"))
    save_json(
        processed_dir / "splits.json",
        {
            "train_genes": train_genes,
            "test_genes": test_genes,
            "test_size": test_size,
            "seed": seed,
            "dataset_hash": dataset_fp,
        },
    )

    return PreparedData(
        features=features,
        labels=labels,
        train_genes=train_genes,
        test_genes=test_genes,
        dataset_fingerprint=dataset_fp,
    )
