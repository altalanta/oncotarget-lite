"""Feature engineering orchestrator for combining all advanced features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import ray

from ..utils import ensure_dir
from .conservation_features import ConservationFeatures
from .domain_features import DomainFeatures
from .pathway_features import PathwayFeatures
from .ppi_features import PPIFeatures
from .structural_features import StructuralFeatures


@ray.remote
def _extract_features_task(extractor, genes: pd.Series, **kwargs) -> pd.DataFrame | None:
    """Helper function to run a single feature extraction process in a Ray task."""
    try:
        return extractor.extract_features(genes, **kwargs)
    except Exception as e:
        # Log the warning from within the remote task
        print(f"Warning: {extractor.__class__.__name__} extraction failed: {e}")
        return None


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
        """Extract all advanced features for given genes in parallel using Ray."""

        if cache_key:
            cache_file = self.cache_dir / f"all_advanced_features_{cache_key}.parquet"
            if cache_file.exists():
                return pd.read_parquet(cache_file)

        if not ray.is_initialized():
            ray.init()

        extractors = [
            self.ppi_features,
            self.pathway_features,
            self.domain_features,
            self.conservation_features,
            self.structural_features,
        ]

        # Launch all feature extraction tasks in parallel
        genes_ref = ray.put(genes)
        futures = [_extract_features_task.remote(extractor, genes_ref, **kwargs) for extractor in extractors]
        
        # Retrieve the results
        feature_dfs_raw = ray.get(futures)

        # Filter out any tasks that failed (and returned None)
        feature_dfs = [df for df in feature_dfs_raw if df is not None]

        # Combine all features
        if not feature_dfs:
            # Return empty DataFrame with gene index
            combined_features = pd.DataFrame(index=genes)
        else:
            # Concatenate with proper gene index alignment
            combined_features = pd.concat(feature_dfs, axis=1)

            # Ensure the index matches the input genes
            if not combined_features.index.equals(genes.index):
                 combined_features = combined_features.reindex(genes.index)

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
