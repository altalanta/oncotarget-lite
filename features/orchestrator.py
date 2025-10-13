"""Feature engineering orchestrator for combining all advanced features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..utils import ensure_dir
from .conservation_features import ConservationFeatures
from .domain_features import DomainFeatures
from .pathway_features import PathwayFeatures
from .ppi_features import PPIFeatures
from .structural_features import StructuralFeatures


class FeatureOrchestrator:
    """Orchestrates all advanced feature engineering pipelines."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)

        # Initialize feature extractors
        self.ppi_features = PPIFeatures(cache_dir)
        self.pathway_features = PathwayFeatures(cache_dir)
        self.domain_features = DomainFeatures(cache_dir)
        self.conservation_features = ConservationFeatures(cache_dir)
        self.structural_features = StructuralFeatures(cache_dir)

    def extract_all_features(
        self,
        genes: pd.Series,
        cache_key: str | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """Extract all advanced features for given genes."""

        if cache_key:
            cache_file = self.cache_dir / f"all_advanced_features_{cache_key}.parquet"
            if cache_file.exists():
                return pd.read_parquet(cache_file)

        # Extract features from each module
        feature_dfs = []

        try:
            ppi_features = self.ppi_features.extract_features(genes, **kwargs)
            feature_dfs.append(ppi_features)
        except Exception as e:
            print(f"Warning: PPI feature extraction failed: {e}")

        try:
            pathway_features = self.pathway_features.extract_features(genes, **kwargs)
            feature_dfs.append(pathway_features)
        except Exception as e:
            print(f"Warning: Pathway feature extraction failed: {e}")

        try:
            domain_features = self.domain_features.extract_features(genes, **kwargs)
            feature_dfs.append(domain_features)
        except Exception as e:
            print(f"Warning: Domain feature extraction failed: {e}")

        try:
            conservation_features = self.conservation_features.extract_features(genes, **kwargs)
            feature_dfs.append(conservation_features)
        except Exception as e:
            print(f"Warning: Conservation feature extraction failed: {e}")

        try:
            structural_features = self.structural_features.extract_features(genes, **kwargs)
            feature_dfs.append(structural_features)
        except Exception as e:
            print(f"Warning: Structural feature extraction failed: {e}")

        # Combine all features
        if not feature_dfs:
            # Return empty DataFrame with gene index
            combined_features = pd.DataFrame(index=genes)
        else:
            # Concatenate with proper gene index alignment
            combined_features = pd.concat(feature_dfs, axis=1)

            # Ensure the index matches the input genes
            combined_features.index = genes.values

            # Handle duplicate columns by keeping the first occurrence
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

        # Cache the combined features
        if cache_key:
            combined_features.to_parquet(cache_file)

        return combined_features

    def get_feature_summary(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of extracted features."""

        summary = {
            "total_features": len(features.columns),
            "total_genes": len(features),
            "feature_categories": {},
            "missing_data": {}
        }

        # Categorize features
        categories = {
            "ppi": [col for col in features.columns if col.startswith("ppi_")],
            "pathway": [col for col in features.columns if col.startswith("pathway_")],
            "domain": [col for col in features.columns if col.startswith("domain_")],
            "conservation": [col for col in features.columns if col.startswith(("homolog_", "ortholog_", "conservation_"))],
            "structural": [col for col in features.columns if col.startswith(("alphafold_", "molecular_", "structural_"))],
        }

        for category, cols in categories.items():
            if cols:
                summary["feature_categories"][category] = len(cols)
                summary["missing_data"][category] = (features[cols] == 0).all().sum()

        return summary
