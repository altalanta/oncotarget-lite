"""Structural and biophysical features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..utils import ensure_dir


class StructuralFeatures:
    """Compute structural and biophysical features."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)

    def load_alphafold_data(self, alphafold_file: Path) -> pd.DataFrame:
        """Load AlphaFold structural data."""
        if not alphafold_file.exists():
            raise FileNotFoundError(f"AlphaFold data not found at {alphafold_file}")

        struct_df = pd.read_csv(
            alphafold_file,
            sep="\t",
            names=["gene", "pLDDT", "ptm", "iptm", "ranking_confidence", "structure_exists"]
        )

        return struct_df

    def load_physicochemical_data(self, phys_file: Path) -> pd.DataFrame:
        """Load physicochemical properties."""
        if not phys_file.exists():
            raise FileNotFoundError(f"Physicochemical data not found at {phys_file}")

        phys_df = pd.read_csv(
            phys_file,
            sep="\t",
            names=["gene", "molecular_weight", "isoelectric_point", "charge", "hydrophobicity", "polarity"]
        )

        return phys_df

    def compute_structural_features(self, genes: pd.Series, struct_data: pd.DataFrame, phys_data: pd.DataFrame) -> pd.DataFrame:
        """Compute structural and biophysical features."""

        features = pd.DataFrame(index=genes)

        # AlphaFold confidence scores
        if not struct_data.empty:
            struct_stats = struct_data.groupby("gene").agg({
                "pLDDT": ["mean", "std", "min", "max"],
                "ptm": "mean",
                "iptm": "mean",
                "ranking_confidence": "mean"
            }).fillna(0)

            features["alphafold_plddt_mean"] = struct_stats[("pLDDT", "mean")].reindex(genes).fillna(0)
            features["alphafold_plddt_std"] = struct_stats[("pLDDT", "std")].reindex(genes).fillna(0)
            features["alphafold_plddt_min"] = struct_stats[("pLDDT", "min")].reindex(genes).fillna(0)
            features["alphafold_plddt_max"] = struct_stats[("pLDDT", "max")].reindex(genes).fillna(0)
            features["alphafold_ptm"] = struct_stats[("ptm", "mean")].reindex(genes).fillna(0)
            features["alphafold_iptm"] = struct_stats[("iptm", "mean")].reindex(genes).fillna(0)
            features["alphafold_confidence"] = struct_stats[("ranking_confidence", "mean")].reindex(genes).fillna(0)

            # Structure availability
            has_structure = struct_data.groupby("gene")["structure_exists"].any()
            features["has_predicted_structure"] = has_structure.reindex(genes).fillna(False).astype(int)

            # High confidence structures (pLDDT > 70)
            high_confidence = struct_data[struct_data["pLDDT"] > 70].groupby("gene").size()
            features["high_confidence_residues"] = high_confidence.reindex(genes).fillna(0)

        # Physicochemical properties
        if not phys_data.empty:
            phys_mapped = phys_data.set_index("gene").reindex(genes).fillna(0)

            features["molecular_weight"] = phys_mapped["molecular_weight"]
            features["isoelectric_point"] = phys_mapped["isoelectric_point"]
            features["net_charge"] = phys_mapped["charge"]
            features["hydrophobicity"] = phys_mapped["hydrophobicity"]
            features["polarity"] = phys_mapped["polarity"]

            # Derived physicochemical features
            features["charge_density"] = phys_mapped["charge"] / (phys_mapped["molecular_weight"] / 1000 + 1e-10)
            features["hydrophobic_ratio"] = phys_mapped["hydrophobicity"] / (phys_mapped["polarity"] + 1e-10)

        # Membrane protein indicators (based on physicochemical properties)
        features["likely_membrane_protein"] = (
            (features["hydrophobicity"] > 0.5) &
            (features["molecular_weight"] > 10000) &
            (features["isoelectric_point"] < 8)
        ).astype(int)

        # Globular vs disordered protein prediction (simplified)
        features["likely_globular"] = (
            (features.get("alphafold_plddt_mean", 0) > 60) &
            (features.get("molecular_weight", 0) < 200000)
        ).astype(int)

        # Structural complexity score
        struct_complexity = (
            features.get("alphafold_plddt_std", 0) * 0.3 +
            features.get("molecular_weight", 0) / 100000 * 0.2 +
            features.get("hydrophobicity", 0) * 0.5
        )
        features["structural_complexity"] = struct_complexity

        return features

    def get_cached_structural_features(self, genes: pd.Series, cache_key: str) -> pd.DataFrame | None:
        """Get cached structural features if available."""
        cache_file = self.cache_dir / f"structural_features_{cache_key}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None

    def cache_structural_features(self, features: pd.DataFrame, cache_key: str) -> None:
        """Cache structural features for future use."""
        cache_file = self.cache_dir / f"structural_features_{cache_key}.parquet"
        features.to_parquet(cache_file)

    def extract_features(
        self,
        genes: pd.Series,
        alphafold_file: Path | None = None,
        phys_file: Path | None = None,
        cache_key: str | None = None
    ) -> pd.DataFrame:
        """Extract structural features for given genes."""

        if cache_key:
            cached = self.get_cached_structural_features(genes, cache_key)
            if cached is not None:
                return cached

        # Default file locations
        if alphafold_file is None:
            alphafold_file = Path("data/raw/alphafold_structures.txt")
        if phys_file is None:
            phys_file = Path("data/raw/physicochemical_properties.txt")

        # Load structural data
        struct_data = pd.DataFrame()
        phys_data = pd.DataFrame()

        if alphafold_file.exists():
            struct_data = self.load_alphafold_data(alphafold_file)

        if phys_file.exists():
            phys_data = self.load_physicochemical_data(phys_file)

        if struct_data.empty and phys_data.empty:
            # Return empty features if no structural data
            features = pd.DataFrame(index=genes)
            features["molecular_weight"] = 0
            features["isoelectric_point"] = 0
            features["net_charge"] = 0
            features["hydrophobicity"] = 0
            features["polarity"] = 0
            features["likely_membrane_protein"] = 0
            features["likely_globular"] = 0
            features["structural_complexity"] = 0
            return features

        # Compute structural features
        features = self.compute_structural_features(genes, struct_data, phys_data)

        if cache_key:
            self.cache_structural_features(features, cache_key)

        return features
