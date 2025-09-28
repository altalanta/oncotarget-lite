"""Feature engineering and label construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataContractError


@dataclass(frozen=True)
class FeatureSet:
    features: pd.DataFrame
    labels: pd.Series


def build_features(merged: pd.DataFrame) -> FeatureSet:
    """Derive expression, safety, and annotation features."""

    normal_cols = [col for col in merged.columns if col.startswith("normal_")]
    tumor_cols = [col for col in merged.columns if col.startswith("tumor_")]
    dep_cols = [col for col in merged.columns if col.startswith("dep_")]
    required_columns = [
        "is_cell_surface",
        "signal_peptide",
        "ig_like_domain",
        "ppi_degree",
        "protein_length",
    ]

    for column_group, cols in (
        ("normal expression", normal_cols),
        ("tumor expression", tumor_cols),
        ("dependency", dep_cols),
    ):
        if not cols:
            msg = f"merged feature table is missing {column_group} columns"
            raise DataContractError(msg)
    missing = [col for col in required_columns if col not in merged.columns]
    if missing:
        msg = f"merged feature table missing required columns: {missing}"
        raise DataContractError(msg)

    normal = merged[normal_cols]
    tumor = merged[tumor_cols]
    dependency = merged[dep_cols]

    normal_mean = normal.mean(axis=1)
    min_normal = normal.min(axis=1)
    tumor_mean = tumor.mean(axis=1)

    log2fc = {
        f"log2fc_{col.replace('tumor_', '').lower()}": np.log2(
            (merged[col] + 1.0) / (normal_mean + 1.0)
        )
        for col in tumor_cols
    }

    features = pd.DataFrame(log2fc, index=merged.index)
    features["min_normal_tpm"] = min_normal
    features["mean_tumor_tpm"] = tumor_mean
    features["mean_dependency"] = dependency.mean(axis=1)
    features["ppi_degree"] = merged["ppi_degree"].astype(float)
    features["signal_peptide"] = merged["signal_peptide"].astype(float)
    features["ig_like_domain"] = merged["ig_like_domain"].astype(float)
    features["protein_length"] = merged["protein_length"].astype(float)

    labels = merged["is_cell_surface"].astype(bool)
    return FeatureSet(features=features, labels=labels)
