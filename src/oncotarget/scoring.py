"""Heuristic scorecard for immunotherapy target triage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for the interpretable scorecard."""

    w1_log2fc: float = 2.0
    w2_dependency: float = 1.2
    w3_surface: float = 1.5
    w4_ig_like: float = 0.6
    w5_ppi: float = 0.8
    w6_low_normal: float = 1.0


_COMPONENT_COLUMNS = [
    "log2fc_component",
    "dependency_component",
    "surface_component",
    "ig_like_component",
    "ppi_component",
    "low_normal_component",
]


def score_candidates(
    features: pd.DataFrame,
    labels: pd.Series,
    weights: ScoreWeights | None = None,
) -> pd.DataFrame:
    """Apply weighted scorecard to the engineered feature matrix."""

    weights = weights or ScoreWeights()
    if not isinstance(features.index, pd.Index):
        raise TypeError("features must have a pandas index of gene symbols")
    if not features.index.equals(labels.index):
        labels = labels.reindex(features.index)

    log_cols = [col for col in features.columns if col.startswith("log2fc_")]
    if not log_cols:
        raise ValueError("features must contain log2fc_* columns")

    log2fc_component = features[log_cols].clip(lower=0.0).mean(axis=1)
    dependency_component = features["mean_dependency"].clip(-2.0, 1.0)
    surface_component = labels.astype(float)
    ig_like_component = features["ig_like_domain"].astype(float)
    ppi_component = np.exp(-((features["ppi_degree"] - 45.0) / 45.0) ** 2)
    low_normal_component = np.clip((5.0 - features["min_normal_tpm"]) / 5.0, 0.0, 1.0)

    contributions = pd.DataFrame(
        {
            "log2fc_component": log2fc_component,
            "dependency_component": dependency_component,
            "surface_component": surface_component,
            "ig_like_component": ig_like_component,
            "ppi_component": ppi_component,
            "low_normal_component": low_normal_component,
        },
        index=features.index,
    )

    score = (
        weights.w1_log2fc * contributions["log2fc_component"]
        + weights.w2_dependency * contributions["dependency_component"]
        + weights.w3_surface * contributions["surface_component"]
        + weights.w4_ig_like * contributions["ig_like_component"]
        + weights.w5_ppi * contributions["ppi_component"]
        + weights.w6_low_normal * contributions["low_normal_component"]
    )

    result = contributions.copy()
    result["score"] = score
    result["rank"] = result["score"].rank(ascending=False, method="min").astype(int)
    return result.sort_values("score", ascending=False)


def explain_gene_score(gene: str, scores: pd.DataFrame) -> dict[str, float]:
    """Return a dict with the gene's score breakdown."""

    if gene not in scores.index:
        msg = f"gene '{gene}' not present in score table"
        raise KeyError(msg)
    row = scores.loc[gene]
    explanation = {component: float(row[component]) for component in _COMPONENT_COLUMNS}
    explanation["score"] = float(row["score"])
    explanation["rank"] = int(row["rank"])
    return explanation
