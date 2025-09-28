"""Data ingestion, schema validation, and dataset preparation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Field, ValidationError

from .config import DataConfig, SplitConfig


class DataContractError(RuntimeError):
    """Raised when cached synthetic data fail schema validation."""


class _GTExRecord(BaseModel):
    gene: str
    tissue: str
    median_TPM: float = Field(ge=0)


class _TCGARecord(BaseModel):
    gene: str
    tumor: str
    median_TPM: float = Field(ge=0)


class _AnnotationRecord(BaseModel):
    gene: str
    is_cell_surface: bool
    signal_peptide: bool
    ig_like_domain: bool
    protein_length: int = Field(ge=50)


class _DepMapRecord(BaseModel):
    gene: str
    dependency_scores: dict[str, float]


class _PPIRecord(BaseModel):
    gene: str
    degree: int = Field(ge=0)


@dataclass(frozen=True)
class RawDataBundle:
    gtex: pd.DataFrame
    tcga: pd.DataFrame
    depmap: pd.DataFrame
    annotations: pd.DataFrame
    ppi: pd.DataFrame


def _read_csv(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not path.exists():
        msg = f"required data file missing: {path.name}"
        raise DataContractError(msg)
    return pl.read_csv(path, schema=schema, comment_prefix="#")


MIN_CLASS_MEMBERS = 2


def load_raw_data(config: DataConfig) -> RawDataBundle:
    """Load and validate all raw tables."""

    raw_dir = config.raw_dir
    gtex_df = _read_csv(
        raw_dir / "GTEx_subset.csv",
        schema={"gene": pl.String, "tissue": pl.String, "median_TPM": pl.Float64},
    ).to_pandas()
    tcga_df = _read_csv(
        raw_dir / "TCGA_subset.csv",
        schema={"gene": pl.String, "tumor": pl.String, "median_TPM": pl.Float64},
    ).to_pandas()
    depmap_df = _read_csv(raw_dir / "DepMap_essentials_subset.csv", schema=None).to_pandas()
    annotations_df = _read_csv(
        raw_dir / "uniprot_annotations.csv",
        schema={
            "gene": pl.String,
            "is_cell_surface": pl.Boolean,
            "signal_peptide": pl.Boolean,
            "ig_like_domain": pl.Boolean,
            "protein_length": pl.Int64,
        },
    ).to_pandas()
    ppi_df = _read_csv(
        raw_dir / "ppi_degree_subset.csv",
        schema={"gene": pl.String, "degree": pl.Int64},
    ).to_pandas()

    try:
        [_GTExRecord(**row) for row in gtex_df.to_dict(orient="records")]
        [_TCGARecord(**row) for row in tcga_df.to_dict(orient="records")]
        [_AnnotationRecord(**row) for row in annotations_df.to_dict(orient="records")]
        [_PPIRecord(**row) for row in ppi_df.to_dict(orient="records")]
    except ValidationError as exc:
        raise DataContractError(str(exc)) from exc

    depmap_records = []
    for row in depmap_df.to_dict(orient="records"):
        gene = row.pop("gene")
        scores = {k: float(v) for k, v in row.items() if k}
        try:
            validated = _DepMapRecord(gene=gene, dependency_scores=scores)
        except ValidationError as exc:
            raise DataContractError(str(exc)) from exc
        depmap_records.append(validated)
    depmap = (
        pd.DataFrame(
            [{"gene": rec.gene, **rec.dependency_scores} for rec in depmap_records]
        )
        .set_index("gene")
        .sort_index()
    )

    bundle = RawDataBundle(
        gtex=gtex_df,
        tcga=tcga_df,
        depmap=depmap,
        annotations=annotations_df.set_index("gene").sort_index(),
        ppi=ppi_df.set_index("gene").sort_index(),
    )
    return bundle


def build_feature_table(bundle: RawDataBundle) -> pd.DataFrame:
    """Merge raw tables into a per-gene feature matrix."""

    gtex_wide = (
        bundle.gtex.pivot_table(index="gene", columns="tissue", values="median_TPM")
        .add_prefix("normal_")
        .sort_index()
    )
    tcga_wide = (
        bundle.tcga.pivot_table(index="gene", columns="tumor", values="median_TPM")
        .add_prefix("tumor_")
        .sort_index()
    )
    depmap = bundle.depmap.add_prefix("dep_")
    annotations = bundle.annotations
    ppi = bundle.ppi.rename(columns={"degree": "ppi_degree"})

    frames: Iterable[pd.DataFrame] = (gtex_wide, tcga_wide, depmap, annotations, ppi)
    merged = pd.concat(frames, axis=1, join="inner").sort_index()
    if merged.empty:
        raise DataContractError("merged feature table is empty after join")
    merged = merged.astype(float, errors="ignore")
    return merged


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _stratified_indices(
    labels: pd.Series, test_size: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    train_idx: list[str] = []
    test_idx: list[str] = []
    for label_value, members in labels.groupby(labels):
        indices = list(members.index.astype(str))
        if len(indices) < MIN_CLASS_MEMBERS:
            msg = f"class '{label_value}' has insufficient members for stratified split"
            raise DataContractError(msg)
        permuted = list(rng.permutation(indices))
        split = max(1, int(round(len(indices) * test_size)))
        test_idx.extend(permuted[:split])
        train_idx.extend(permuted[split:])
    return train_idx, test_idx


def split_dataset(features: pd.DataFrame, labels: pd.Series, split: SplitConfig) -> DatasetSplits:
    """Perform deterministic stratified train/test split."""

    if features.empty:
        raise DataContractError("feature matrix is empty")
    if not features.index.equals(labels.index):
        labels = labels.reindex(features.index)
    if labels.isna().any():
        raise DataContractError("labels contain NaN after alignment")

    if split.stratify:
        train_idx, test_idx = _stratified_indices(labels, split.test_size, split.seed)
    else:
        rng = np.random.default_rng(split.seed)
        indices = list(features.index.astype(str))
        permuted = list(rng.permutation(indices))
        split_point = int(round(len(indices) * (1 - split.test_size)))
        train_idx, test_idx = permuted[:split_point], permuted[split_point:]

    X_train = features.loc[train_idx]
    X_test = features.loc[test_idx]
    y_train = labels.loc[train_idx]
    y_test = labels.loc[test_idx]
    return DatasetSplits(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
