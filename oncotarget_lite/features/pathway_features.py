"""Pathway enrichment and analysis features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..utils import ensure_dir


class PathwayFeatures:
    """Compute pathway-based features."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)
        self.pathway_data = {}

    def load_kegg_pathways(self, kegg_file: Path) -> Dict[str, List[str]]:
        """Load KEGG pathway data."""
        if not kegg_file.exists():
            raise FileNotFoundError(f"KEGG data not found at {kegg_file}")

        pathways = {}
        current_pathway = None

        with open(kegg_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("ENTRY"):
                    current_pathway = line.split()[1]
                    pathways[current_pathway] = []
                elif line.startswith("GENE") and current_pathway:
                    # Extract gene symbols (simplified parsing)
                    parts = line.split()
                    if len(parts) > 1:
                        genes = [g.strip() for g in parts[1:] if not g.startswith("(")]
                        pathways[current_pathway].extend(genes)

        return pathways

    def load_reactome_pathways(self, reactome_file: Path) -> Dict[str, List[str]]:
        """Load Reactome pathway data."""
        if not reactome_file.exists():
            raise FileNotFoundError(f"Reactome data not found at {reactome_file}")

        pathways = {}
        current_pathway = None

        with open(reactome_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Pathway"):
                    current_pathway = line.split("\t")[1] if "\t" in line else line.split()[1]
                    pathways[current_pathway] = []
                elif line.startswith("Gene") and current_pathway:
                    gene = line.split("\t")[1] if "\t" in line else line.split()[1]
                    pathways[current_pathway].append(gene)

        return pathways

    def compute_pathway_enrichment(self, genes: pd.Series, pathways: Dict[str, List[str]]) -> pd.DataFrame:
        """Compute pathway enrichment features."""

        features = pd.DataFrame(index=genes)
        gene_set = set(genes.unique())

        # Pathway membership features
        pathway_membership = {}
        pathway_sizes = {}

        for pathway_id, pathway_genes in pathways.items():
            pathway_gene_set = set(pathway_genes)
            pathway_sizes[pathway_id] = len(pathway_gene_set)

            # Count overlapping genes for each gene in our dataset
            for gene in genes:
                if gene in pathway_membership:
                    pathway_membership[gene].append(pathway_id if gene in pathway_gene_set else 0)
                else:
                    pathway_membership[gene] = [pathway_id if gene in pathway_gene_set else 0]

        # Create binary pathway membership matrix
        pathway_df = pd.DataFrame(pathway_membership).T
        pathway_df.columns = list(pathways.keys())
        pathway_df = pathway_df.reindex(genes).fillna(0)

        # Pathway count features
        features["pathway_count"] = pathway_df.sum(axis=1)

        # Pathway size statistics
        pathway_sizes_series = pd.Series(pathway_sizes)
        gene_pathway_sizes = pathway_df.apply(
            lambda row: pathway_sizes_series[row[row == 1].index].mean() if row.sum() > 0 else 0,
            axis=1
        )
        features["mean_pathway_size"] = gene_pathway_sizes

        # Top pathways (most common)
        top_pathways = pathway_df.sum().nlargest(10).index
        for pathway in top_pathways:
            features[f"pathway_{pathway}_member"] = pathway_df[pathway]

        # Pathway diversity (entropy of pathway memberships)
        pathway_entropy = pathway_df.apply(
            lambda row: self._calculate_entropy(row[row > 0]) if row.sum() > 0 else 0,
            axis=1
        )
        features["pathway_entropy"] = pathway_entropy

        # Cancer-related pathway memberships
        cancer_pathways = [
            "hsa05200", "hsa05202", "hsa05203", "hsa05204", "hsa05205",
            "hsa05206", "hsa05207", "hsa05208", "hsa05210", "hsa05211",
            "hsa05212", "hsa05213", "hsa05214", "hsa05215", "hsa05216"
        ]

        cancer_cols = [col for col in pathway_df.columns if col in cancer_pathways]
        if cancer_cols:
            features["cancer_pathway_count"] = pathway_df[cancer_cols].sum(axis=1)

        return features

    def _calculate_entropy(self, values: pd.Series) -> float:
        """Calculate Shannon entropy of pathway memberships."""
        if len(values) == 0:
            return 0.0

        value_counts = values.value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
        return entropy

    def get_cached_pathway_features(self, genes: pd.Series, cache_key: str) -> pd.DataFrame | None:
        """Get cached pathway features if available."""
        cache_file = self.cache_dir / f"pathway_features_{cache_key}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None

    def cache_pathway_features(self, features: pd.DataFrame, cache_key: str) -> None:
        """Cache pathway features for future use."""
        cache_file = self.cache_dir / f"pathway_features_{cache_key}.parquet"
        features.to_parquet(cache_file)

    def extract_features(
        self,
        genes: pd.Series,
        kegg_file: Path | None = None,
        reactome_file: Path | None = None,
        cache_key: str | None = None
    ) -> pd.DataFrame:
        """Extract pathway features for given genes."""

        if cache_key:
            cached = self.get_cached_pathway_features(genes, cache_key)
            if cached is not None:
                return cached

        # Default file locations
        if kegg_file is None:
            kegg_file = Path("data/raw/kegg_pathways.txt")
        if reactome_file is None:
            reactome_file = Path("data/raw/reactome_pathways.txt")

        # Load pathway data
        kegg_pathways = {}
        reactome_pathways = {}

        if kegg_file.exists():
            kegg_pathways = self.load_kegg_pathways(kegg_file)

        if reactome_file.exists():
            reactome_pathways = self.load_reactome_pathways(reactome_file)

        # Combine all pathways
        all_pathways = {**kegg_pathways, **reactome_pathways}

        if not all_pathways:
            # Return empty features if no pathway data
            features = pd.DataFrame(index=genes)
            features["pathway_count"] = 0
            features["mean_pathway_size"] = 0
            features["pathway_entropy"] = 0
            features["cancer_pathway_count"] = 0
            return features

        # Compute pathway features
        features = self.compute_pathway_enrichment(genes, all_pathways)

        if cache_key:
            self.cache_pathway_features(features, cache_key)

        return features
