"""Data loading and validation utilities for oncotarget-lite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import BaseModel, Field, validator

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_RAW = _PROJECT_ROOT / "data" / "raw"


class _GTExRecord(BaseModel):
    gene: str
    tissue: str
    median_TPM: float = Field(ge=0)


class _TCGARecord(BaseModel):
    gene: str
    tumor: str
    median_TPM: float = Field(ge=0)


class _DepMapRecord(BaseModel):
    gene: str
    scores: dict[str, float]

    @validator("scores")
    def check_scores(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            msg = "dependency score dictionary must not be empty"
            raise ValueError(msg)
        return value


class _AnnotationRecord(BaseModel):
    gene: str
    is_cell_surface: bool
    signal_peptide: bool
    ig_like_domain: bool
    protein_length: int = Field(ge=50)


class _PPIRecord(BaseModel):
    gene: str
    degree: int = Field(ge=0)


@dataclass(frozen=True)
class RawDataBundle:
    """Container describing the loaded raw tables."""

    gtex: pd.DataFrame
    tcga: pd.DataFrame
    depmap: pd.DataFrame
    annotations: pd.DataFrame
    ppi: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def load_gtex(path: Path | None = None) -> pd.DataFrame:
    """Load the synthetic GTEx summary statistics table."""

    data_path = path or (_DATA_RAW / "GTEx_subset.csv")
    df = _read_csv(data_path)
    validated = [_GTExRecord(**row).dict() for row in df.to_dict(orient="records")]
    return pd.DataFrame(validated)


def load_tcga(path: Path | None = None) -> pd.DataFrame:
    """Load the synthetic TCGA summary statistics table."""

    data_path = path or (_DATA_RAW / "TCGA_subset.csv")
    df = _read_csv(data_path)
    validated = [_TCGARecord(**row).dict() for row in df.to_dict(orient="records")]
    return pd.DataFrame(validated)


def load_depmap(path: Path | None = None) -> pd.DataFrame:
    """Load gene dependency scores (negative == essential)."""

    data_path = path or (_DATA_RAW / "DepMap_essentials_subset.csv")
    df = _read_csv(data_path)
    records: list[_DepMapRecord] = []
    for record in df.to_dict(orient="records"):
        gene = record.pop("gene")
        scores = {k: float(v) for k, v in record.items()}
        records.append(_DepMapRecord(gene=gene, scores=scores))
    result = (
        pd.DataFrame([{"gene": rec.gene, **rec.scores} for rec in records])
        .set_index("gene")
        .sort_index()
    )
    return result


def load_annotations(path: Path | None = None) -> pd.DataFrame:
    """Load UniProt-style annotations used for labels and safety filters."""

    data_path = path or (_DATA_RAW / "uniprot_annotations.csv")
    df = _read_csv(data_path)
    validated = [_AnnotationRecord(**row).dict() for row in df.to_dict(orient="records")]
    return pd.DataFrame(validated).set_index("gene").sort_index()


def load_ppi(path: Path | None = None) -> pd.DataFrame:
    """Load protein-protein interaction degree annotations."""

    data_path = path or (_DATA_RAW / "ppi_degree_subset.csv")
    df = _read_csv(data_path)
    validated = [_PPIRecord(**row).dict() for row in df.to_dict(orient="records")]
    return pd.DataFrame(validated).set_index("gene").sort_index()


def load_raw_data() -> RawDataBundle:
    """Convenience helper to load all raw tables at once."""

    return RawDataBundle(
        gtex=load_gtex(),
        tcga=load_tcga(),
        depmap=load_depmap(),
        annotations=load_annotations(),
        ppi=load_ppi(),
    )


def merge_gene_feature_table(bundle: RawDataBundle | None = None) -> pd.DataFrame:
    """Merge raw tables into a wide, per-gene feature table.

    Columns are prefixed by their origin:
    - normal_*: GTEx tissues
    - tumor_*: TCGA tumor medians
    - dep_*: DepMap dependency scores
    - annotation columns retained without prefix
    - ppi_degree
    """

    bundle = bundle or load_raw_data()

    gtex_wide = (
        bundle.gtex.pivot(index="gene", columns="tissue", values="median_TPM")
        .add_prefix("normal_")
    )
    tcga_wide = (
        bundle.tcga.pivot(index="gene", columns="tumor", values="median_TPM")
        .add_prefix("tumor_")
    )
    depmap = bundle.depmap.add_prefix("dep_")
    annotations = bundle.annotations
    ppi = bundle.ppi.rename(columns={"degree": "ppi_degree"})

    frames: Iterable[pd.DataFrame] = (
        gtex_wide,
        tcga_wide,
        depmap,
        annotations,
        ppi,
    )
    merged = pd.concat(frames, axis=1, join="outer").sort_index()
    merged = merged.dropna(axis=0, how="any")
    return merged
