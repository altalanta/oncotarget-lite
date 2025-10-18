"""PPI (Protein-Protein Interaction) features for oncotarget-lite."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseFeatureExtractor


class PPIFeatures(BaseFeatureExtractor):
    """Extract protein-protein interaction features."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        super().__init__(cache_dir)
        self.feature_name = "ppi"

    def extract_features(
        self,
        genes: List[str],
        **kwargs
    ) -> pd.DataFrame:
        """Extract PPI features for genes."""

        # For now, create simple synthetic PPI features
        # In a real implementation, this would query PPI databases like STRING

        features = []
        for gene in genes:
            # Simple synthetic features based on gene name
            gene_hash = hash(gene) % 1000

            feature_dict = {
                "gene": gene,
                "ppi_degree": min(gene_hash / 10, 50),  # Degree centrality (0-50)
                "ppi_clustering": (gene_hash % 100) / 100,  # Clustering coefficient (0-1)
                "ppi_betweenness": (gene_hash % 50) / 50,  # Betweenness centrality (0-1)
                "ppi_closeness": (gene_hash % 80) / 80,  # Closeness centrality (0-1)
            }

            features.append(feature_dict)

        return pd.DataFrame(features).set_index("gene")

    def get_required_files(self) -> List[str]:
        """Return list of required data files."""
        return ["ppi_network.txt"]

    def validate_inputs(self, **kwargs) -> bool:
        """Validate input data."""
        return True

