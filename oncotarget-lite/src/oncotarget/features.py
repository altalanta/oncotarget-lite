"""Feature engineering for immunotherapy target discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureMatrix:
    """Bundle containing the engineered features and labels."""

    features: pd.DataFrame
    labels: pd.Series


def _require_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        msg = f"missing required columns: {missing}"
        raise ValueError(msg)


def create_feature_matrix(merged: pd.DataFrame) -> FeatureMatrix:
    """Compute expression, safety, and annotation-derived features."""

    normal_cols = [col for col in merged.columns if col.startswith("normal_")]
    tumor_cols = [col for col in merged.columns if col.startswith("tumor_")]
    dep_cols = [col for col in merged.columns if col.startswith("dep_")]
    annotation_cols = ["signal_peptide", "ig_like_domain", "protein_length"]
    _require_columns(merged, annotation_cols + ["is_cell_surface", "ppi_degree"])

    if not normal_cols or not tumor_cols or not dep_cols:
        msg = "merged table must contain normal, tumor, and dependency columns"
        raise ValueError(msg)

    normal_matrix = merged[normal_cols]
    tumor_matrix = merged[tumor_cols]
    dep_matrix = merged[dep_cols]

    normal_mean = normal_matrix.mean(axis=1)
    min_normal = normal_matrix.min(axis=1)
    mean_tumor = tumor_matrix.mean(axis=1)

    log2fc_features = {}
    for col in tumor_cols:
        tumor_name = col.replace("tumor_", "").lower()
        log2fc_features[f"log2fc_{tumor_name}"] = np.log2((merged[col] + 1.0) / (normal_mean + 1.0))

    mean_dependency = dep_matrix.mean(axis=1)

    features = pd.DataFrame(log2fc_features, index=merged.index)
    features["min_normal_tpm"] = min_normal
    features["mean_tumor_tpm"] = mean_tumor
    features["mean_dependency"] = mean_dependency
    features["ppi_degree"] = merged["ppi_degree"]
    for col in annotation_cols:
        features[col] = merged[col]

    features = features.fillna(0.0)
    labels = merged["is_cell_surface"].astype(bool)
    return FeatureMatrix(features=features, labels=labels)
