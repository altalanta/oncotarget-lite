"""Evolutionary conservation features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..utils import ensure_dir


class ConservationFeatures:
    """Compute evolutionary conservation features."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)

    def load_homolog_data(self, homolog_file: Path) -> pd.DataFrame:
        """Load homolog data."""
        if not homolog_file.exists():
            raise FileNotFoundError(f"Homolog data not found at {homolog_file}")

        homolog_df = pd.read_csv(
            homolog_file,
            sep="\t",
            names=["gene", "species", "homolog", "identity", "coverage", "evalue"]
        )

        return homolog_df

    def load_ortholog_data(self, ortholog_file: Path) -> pd.DataFrame:
        """Load ortholog data."""
        if not ortholog_file.exists():
            raise FileNotFoundError(f"Ortholog data not found at {ortholog_file}")

        ortholog_df = pd.read_csv(
            ortholog_file,
            sep="\t",
            names=["gene", "ortholog", "species", "identity", "type"]
        )

        return ortholog_df

    def compute_conservation_features(self, genes: pd.Series, homolog_data: pd.DataFrame, ortholog_data: pd.DataFrame) -> pd.DataFrame:
        """Compute conservation-based features."""

        features = pd.DataFrame(index=genes)

        # Homolog count per gene
        homolog_counts = homolog_data.groupby("gene").size()
        features["homolog_count"] = homolog_counts.reindex(genes).fillna(0)

        # Species diversity (number of unique species with homologs)
        species_counts = homolog_data.groupby("gene")["species"].nunique()
        features["homolog_species_count"] = species_counts.reindex(genes).fillna(0)

        # Average sequence identity across homologs
        identity_stats = homolog_data.groupby("gene")["identity"].agg(["mean", "std", "min", "max"])
        features["mean_homolog_identity"] = identity_stats["mean"].reindex(genes).fillna(0)
        features["std_homolog_identity"] = identity_stats["std"].reindex(genes).fillna(0)
        features["min_homolog_identity"] = identity_stats["min"].reindex(genes).fillna(0)
        features["max_homolog_identity"] = identity_stats["max"].reindex(genes).fillna(0)

        # Ortholog count per gene
        ortholog_counts = ortholog_data.groupby("gene").size()
        features["ortholog_count"] = ortholog_counts.reindex(genes).fillna(0)

        # Ortholog species diversity
        ortho_species_counts = ortholog_data.groupby("gene")["species"].nunique()
        features["ortholog_species_count"] = ortho_species_counts.reindex(genes).fillna(0)

        # Ortholog types (one-to-one, one-to-many, etc.)
        ortho_types = ortholog_data.groupby(["gene", "type"]).size().unstack(fill_value=0)
        for col in ortho_types.columns:
            features[f"ortholog_type_{col}"] = ortho_types[col].reindex(genes).fillna(0)

        # Conservation score (composite metric)
        # Higher scores indicate more conserved genes
        features["conservation_score"] = (
            features["homolog_count"] * 0.3 +
            features["homolog_species_count"] * 0.3 +
            features["mean_homolog_identity"] * 0.4
        )

        # Conservation rank (percentile within dataset)
        features["conservation_percentile"] = features["conservation_score"].rank(pct=True)

        # Highly conserved genes (top 10%)
        features["is_highly_conserved"] = (features["conservation_percentile"] > 0.9).astype(int)

        # Moderately conserved genes (top 25%)
        features["is_moderately_conserved"] = (features["conservation_percentile"] > 0.75).astype(int)

        return features

    def get_cached_conservation_features(self, genes: pd.Series, cache_key: str) -> pd.DataFrame | None:
        """Get cached conservation features if available."""
        cache_file = self.cache_dir / f"conservation_features_{cache_key}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None

    def cache_conservation_features(self, features: pd.DataFrame, cache_key: str) -> None:
        """Cache conservation features for future use."""
        cache_file = self.cache_dir / f"conservation_features_{cache_key}.parquet"
        features.to_parquet(cache_file)

    def extract_features(
        self,
        genes: pd.Series,
        homolog_file: Path | None = None,
        ortholog_file: Path | None = None,
        cache_key: str | None = None
    ) -> pd.DataFrame:
        """Extract conservation features for given genes."""

        if cache_key:
            cached = self.get_cached_conservation_features(genes, cache_key)
            if cached is not None:
                return cached

        # Default file locations
        if homolog_file is None:
            homolog_file = Path("data/raw/homologs.txt")
        if ortholog_file is None:
            ortholog_file = Path("data/raw/orthologs.txt")

        # Load conservation data
        homolog_data = pd.DataFrame()
        ortholog_data = pd.DataFrame()

        if homolog_file.exists():
            homolog_data = self.load_homolog_data(homolog_file)

        if ortholog_file.exists():
            ortholog_data = self.load_ortholog_data(ortholog_file)

        if homolog_data.empty and ortholog_data.empty:
            # Return empty features if no conservation data
            features = pd.DataFrame(index=genes)
            features["homolog_count"] = 0
            features["homolog_species_count"] = 0
            features["mean_homolog_identity"] = 0
            features["ortholog_count"] = 0
            features["ortholog_species_count"] = 0
            features["conservation_score"] = 0
            features["conservation_percentile"] = 0
            features["is_highly_conserved"] = 0
            features["is_moderately_conserved"] = 0
            return features

        # Compute conservation features
        features = self.compute_conservation_features(genes, homolog_data, ortholog_data)

        if cache_key:
            self.cache_conservation_features(features, cache_key)

        return features
