"""Protein domain architecture features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..utils import ensure_dir


class DomainFeatures:
    """Compute protein domain-based features."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)
        self.domain_data = {}

    def load_pfam_domains(self, pfam_file: Path) -> Dict[str, List[str]]:
        """Load PFAM domain data."""
        if not pfam_file.exists():
            raise FileNotFoundError(f"PFAM data not found at {pfam_file}")

        domains = {}

        with open(pfam_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">") or not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 2:
                    gene = parts[0]
                    domain = parts[1]

                    if gene not in domains:
                        domains[gene] = []
                    domains[gene].append(domain)

        return domains

    def load_interpro_domains(self, interpro_file: Path) -> Dict[str, List[str]]:
        """Load InterPro domain data."""
        if not interpro_file.exists():
            raise FileNotFoundError(f"InterPro data not found at {interpro_file}")

        domains = {}

        with open(interpro_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 2:
                    gene = parts[0]
                    domain = parts[1]

                    if gene not in domains:
                        domains[gene] = []
                    domains[gene].append(domain)

        return domains

    def compute_domain_features(self, genes: pd.Series, domains: Dict[str, List[str]]) -> pd.DataFrame:
        """Compute domain architecture features."""

        features = pd.DataFrame(index=genes)

        # Domain count
        domain_counts = {}
        for gene in genes:
            gene_domains = domains.get(gene, [])
            domain_counts[gene] = len(gene_domains)

        features["domain_count"] = pd.Series(domain_counts)

        # Domain diversity (unique domain types)
        unique_domain_counts = {}
        for gene in genes:
            gene_domains = domains.get(gene, [])
            unique_domain_counts[gene] = len(set(gene_domains))

        features["unique_domain_count"] = pd.Series(unique_domain_counts)

        # Domain diversity ratio
        features["domain_diversity_ratio"] = features["unique_domain_count"] / (features["domain_count"] + 1)

        # Domain length statistics (if available)
        # This would require additional domain length data

        # Common domain families
        all_domains = []
        for gene_domains in domains.values():
            all_domains.extend(gene_domains)

        domain_freq = pd.Series(all_domains).value_counts()
        common_domains = domain_freq.nlargest(20).index

        # Domain membership features for common domains
        for domain in common_domains:
            domain_col = {}
            for gene in genes:
                gene_domains = domains.get(gene, [])
                domain_col[gene] = 1 if domain in gene_domains else 0

            features[f"domain_{domain}"] = pd.Series(domain_col)

        # Domain family categories (simplified)
        # This would require a domain family classification system
        transmembrane_domains = ["TM", "transmembrane", "signal"]  # Simplified
        catalytic_domains = ["kinase", "phosphatase", "protease"]  # Simplified

        tm_features = {}
        catalytic_features = {}

        for gene in genes:
            gene_domains = domains.get(gene, [])

            # Check for transmembrane domains
            tm_count = sum(1 for d in gene_domains if any(tm in d.lower() for tm in transmembrane_domains))
            tm_features[gene] = tm_count

            # Check for catalytic domains
            catalytic_count = sum(1 for d in gene_domains if any(cat in d.lower() for cat in catalytic_domains))
            catalytic_features[gene] = catalytic_count

        features["transmembrane_domain_count"] = pd.Series(tm_features)
        features["catalytic_domain_count"] = pd.Series(catalytic_features)

        # Domain complexity score (entropy of domain types)
        domain_entropy = {}
        for gene in genes:
            gene_domains = domains.get(gene, [])
            if gene_domains:
                domain_freq_gene = pd.Series(gene_domains).value_counts(normalize=True)
                entropy = -np.sum(domain_freq_gene * np.log2(domain_freq_gene + 1e-10))
                domain_entropy[gene] = entropy
            else:
                domain_entropy[gene] = 0

        features["domain_entropy"] = pd.Series(domain_entropy)

        return features

    def get_cached_domain_features(self, genes: pd.Series, cache_key: str) -> pd.DataFrame | None:
        """Get cached domain features if available."""
        cache_file = self.cache_dir / f"domain_features_{cache_key}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None

    def cache_domain_features(self, features: pd.DataFrame, cache_key: str) -> None:
        """Cache domain features for future use."""
        cache_file = self.cache_dir / f"domain_features_{cache_key}.parquet"
        features.to_parquet(cache_file)

    def extract_features(
        self,
        genes: pd.Series,
        pfam_file: Path | None = None,
        interpro_file: Path | None = None,
        cache_key: str | None = None
    ) -> pd.DataFrame:
        """Extract domain features for given genes."""

        if cache_key:
            cached = self.get_cached_domain_features(genes, cache_key)
            if cached is not None:
                return cached

        # Default file locations
        if pfam_file is None:
            pfam_file = Path("data/raw/pfam_domains.txt")
        if interpro_file is None:
            interpro_file = Path("data/raw/interpro_domains.txt")

        # Load domain data
        pfam_domains = {}
        interpro_domains = {}

        if pfam_file.exists():
            pfam_domains = self.load_pfam_domains(pfam_file)

        if interpro_file.exists():
            interpro_domains = self.load_interpro_domains(interpro_file)

        # Combine all domains
        all_domains = {}
        for gene in genes.unique():
            gene_domains = []
            if gene in pfam_domains:
                gene_domains.extend(pfam_domains[gene])
            if gene in interpro_domains:
                gene_domains.extend(interpro_domains[gene])

            if gene_domains:
                all_domains[gene] = gene_domains

        if not all_domains:
            # Return empty features if no domain data
            features = pd.DataFrame(index=genes)
            features["domain_count"] = 0
            features["unique_domain_count"] = 0
            features["domain_diversity_ratio"] = 0
            features["transmembrane_domain_count"] = 0
            features["catalytic_domain_count"] = 0
            features["domain_entropy"] = 0
            return features

        # Compute domain features
        features = self.compute_domain_features(genes, all_domains)

        if cache_key:
            self.cache_domain_features(features, cache_key)

        return features
