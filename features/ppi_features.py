"""Protein-Protein Interaction (PPI) network features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd

from ..utils import ensure_dir


class PPIFeatures:
    """Compute advanced PPI network features."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)

    def compute_network_features(self, genes: pd.Series, ppi_data: pd.DataFrame) -> pd.DataFrame:
        """Compute comprehensive network features from PPI data."""

        features = pd.DataFrame(index=genes)

        # Filter PPI data for genes of interest
        gene_set = set(genes.unique())
        ppi_filtered = ppi_data[
            (ppi_data["protein1"].isin(gene_set)) &
            (ppi_data["protein2"].isin(gene_set))
        ]

        # Build network
        G = nx.Graph()
        G.add_nodes_from(gene_set)

        # Add edges with confidence scores
        for _, row in ppi_filtered.iterrows():
            G.add_edge(
                row["protein1"],
                row["protein2"],
                weight=row.get("combined_score", 500)  # Default medium confidence
            )

        # Basic network metrics
        features["ppi_degree"] = pd.Series(dict(G.degree(genes)))
        features["ppi_degree"] = features["ppi_degree"].fillna(0)

        # Clustering coefficient
        clustering = nx.clustering(G, genes)
        features["ppi_clustering"] = pd.Series(clustering).fillna(0)

        # Betweenness centrality
        betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes())))
        features["ppi_betweenness"] = pd.Series(betweenness).reindex(genes).fillna(0)

        # Closeness centrality
        closeness = nx.closeness_centrality(G)
        features["ppi_closeness"] = pd.Series(closeness).reindex(genes).fillna(0)

        # Eigenvector centrality
        if len(G.nodes()) > 1:
            try:
                eigen_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                features["ppi_eigenvector"] = pd.Series(eigen_centrality).reindex(genes).fillna(0)
            except:
                features["ppi_eigenvector"] = 0
        else:
            features["ppi_eigenvector"] = 0

        # Average neighbor degree
        avg_neighbor_degree = {}
        for node in genes:
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor_degrees = [G.degree(neigh) for neigh in neighbors]
                avg_neighbor_degree[node] = np.mean(neighbor_degrees)
            else:
                avg_neighbor_degree[node] = 0

        features["ppi_avg_neighbor_degree"] = pd.Series(avg_neighbor_degree)

        # Network distance to highly connected nodes (hubs)
        hub_threshold = np.percentile([d for n, d in G.degree()], 90)  # Top 10% hubs
        hubs = [node for node, degree in G.degree() if degree >= hub_threshold]

        if hubs:
            hub_distances = {}
            for gene in genes:
                if gene in hubs:
                    hub_distances[gene] = 0
                else:
                    try:
                        distances = nx.shortest_path_length(G, source=gene, target=hubs[0])
                        hub_distances[gene] = min(distances.values()) if isinstance(distances, dict) else distances
                    except:
                        hub_distances[gene] = np.inf

            features["ppi_hub_distance"] = pd.Series(hub_distances).replace(np.inf, 10).fillna(10)

        # Community detection (if network is large enough)
        if len(G.nodes()) > 10:
            try:
                communities = nx.community.greedy_modularity_communities(G)
                community_map = {}
                for i, community in enumerate(communities):
                    for node in community:
                        community_map[node] = i

                features["ppi_community"] = pd.Series(community_map).reindex(genes).fillna(-1)
            except:
                features["ppi_community"] = -1
        else:
            features["ppi_community"] = -1

        return features

    def load_string_data(self, string_file: Path) -> pd.DataFrame:
        """Load STRING PPI data."""
        if not string_file.exists():
            raise FileNotFoundError(f"STRING data not found at {string_file}")

        # STRING format: protein1 protein2 neighborhood fusion cooccurence homology coexpression coexpression_transferred experimental knowledge knowledge_transferred textmining textmining_transferred combined_score
        ppi_df = pd.read_csv(
            string_file,
            sep=" ",
            header=None,
            names=[
                "protein1", "protein2", "neighborhood", "fusion", "cooccurence",
                "homology", "coexpression", "coexpression_transferred",
                "experimental", "knowledge", "knowledge_transferred",
                "textmining", "textmining_transferred", "combined_score"
            ]
        )

        return ppi_df

    def get_cached_ppi_features(self, genes: pd.Series, cache_key: str) -> pd.DataFrame | None:
        """Get cached PPI features if available."""
        cache_file = self.cache_dir / f"ppi_features_{cache_key}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None

    def cache_ppi_features(self, features: pd.DataFrame, cache_key: str) -> None:
        """Cache PPI features for future use."""
        cache_file = self.cache_dir / f"ppi_features_{cache_key}.parquet"
        features.to_parquet(cache_file)

    def extract_features(
        self,
        genes: pd.Series,
        ppi_file: Path | None = None,
        cache_key: str | None = None
    ) -> pd.DataFrame:
        """Extract PPI features for given genes."""

        if cache_key:
            cached = self.get_cached_ppi_features(genes, cache_key)
            if cached is not None:
                return cached

        if ppi_file is None:
            # Use default STRING file location
            ppi_file = Path("data/raw/ppi_network.txt")

        # Load PPI data
        ppi_data = self.load_string_data(ppi_file)

        # Compute features
        features = self.compute_network_features(genes, ppi_data)

        if cache_key:
            self.cache_ppi_features(features, cache_key)

        return features
