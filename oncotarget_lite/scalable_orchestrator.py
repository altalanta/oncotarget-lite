"""Scalable feature engineering orchestrator with parallel processing."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import ensure_dir
from .features.orchestrator import FeatureOrchestrator


class ScalableFeatureOrchestrator:
    """Enhanced feature orchestrator with parallel processing and advanced caching."""

    def __init__(
        self,
        cache_dir: Path = Path("data/cache"),
        max_workers: int = None,
        chunk_size: int = 1000,
        use_memory_mapping: bool = True,
    ):
        self.cache_dir = cache_dir
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) * 2)
        self.chunk_size = chunk_size
        self.use_memory_mapping = use_memory_mapping

        ensure_dir(cache_dir)

        # Initialize the original orchestrator
        self.orchestrator = FeatureOrchestrator(cache_dir)

    def _compute_feature_hash(self, genes: pd.Series, feature_types: List[str]) -> str:
        """Compute hash for feature extraction cache key."""
        genes_str = genes.sort_values().to_json()
        types_str = json.dumps(sorted(feature_types))
        combined = f"{genes_str}_{types_str}".encode()
        return hashlib.sha256(combined).hexdigest()

    def extract_features_parallel(
        self,
        genes: pd.Series,
        feature_types: Optional[List[str]] = None,
        cache_key: str | None = None,
        **kwargs
    ) -> pd.DataFrame:
        """Extract features in parallel for better performance."""

        if feature_types is None:
            feature_types = ["ppi", "pathway", "domain", "conservation", "structural"]

        # Check cache first
        if cache_key:
            feature_hash = self._compute_feature_hash(genes, feature_types)
            cache_file = self.cache_dir / f"parallel_features_{feature_hash}.parquet"
            if cache_file.exists():
                print(f"📋 Loading cached features: {cache_file}")
                return pd.read_parquet(cache_file)

        print(f"🚀 Extracting {len(feature_types)} feature types in parallel for {len(genes)} genes...")

        # Extract features in parallel threads
        feature_results = {}
        failed_features = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit feature extraction tasks
            future_to_feature = {}

            for feature_type in feature_types:
                future = executor.submit(self._extract_single_feature_type, feature_type, genes, **kwargs)
                future_to_feature[future] = feature_type

            # Collect results as they complete
            for future in as_completed(future_to_feature):
                feature_type = future_to_feature[future]
                try:
                    result_df = future.result()
                    if result_df is not None and not result_df.empty:
                        feature_results[feature_type] = result_df
                        print(f"✅ Extracted {feature_type} features: {result_df.shape}")
                    else:
                        print(f"⚠️  No {feature_type} features extracted")
                except Exception as e:
                    failed_features.append((feature_type, str(e)))
                    print(f"❌ Failed to extract {feature_type} features: {e}")

        if failed_features:
            print(f"Warning: {len(failed_features)} feature types failed")

        # Combine all feature results
        if not feature_results:
            combined_features = pd.DataFrame(index=genes)
        else:
            combined_features = pd.concat(feature_results.values(), axis=1)

            # Ensure the index matches the input genes
            combined_features.index = genes.values

            # Handle duplicate columns
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

        # Cache the combined features
        if cache_key:
            combined_features.to_parquet(cache_file)
            print(f"💾 Cached combined features: {cache_file}")

        return combined_features

    def _extract_single_feature_type(
        self,
        feature_type: str,
        genes: pd.Series,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Extract features for a single feature type."""
        try:
            if feature_type == "ppi":
                return self.orchestrator.ppi_features.extract_features(genes, **kwargs)
            elif feature_type == "pathway":
                return self.orchestrator.pathway_features.extract_features(genes, **kwargs)
            elif feature_type == "domain":
                return self.orchestrator.domain_features.extract_features(genes, **kwargs)
            elif feature_type == "conservation":
                return self.orchestrator.conservation_features.extract_features(genes, **kwargs)
            elif feature_type == "structural":
                return self.orchestrator.structural_features.extract_features(genes, **kwargs)
            else:
                print(f"Unknown feature type: {feature_type}")
                return None
        except Exception as e:
            print(f"Error extracting {feature_type} features: {e}")
            return None

    def benchmark_feature_extraction(
        self,
        genes: pd.Series,
        feature_types: Optional[List[str]] = None,
        n_runs: int = 3,
    ) -> Dict[str, Any]:
        """Benchmark feature extraction performance."""

        if feature_types is None:
            feature_types = ["ppi", "pathway", "domain", "conservation", "structural"]

        results = {}

        for feature_type in feature_types:
            times = []
            for run in range(n_runs):
                start_time = time.time()
                try:
                    result = self._extract_single_feature_type(feature_type, genes)
                    end_time = time.time()
                    times.append(end_time - start_time)
                    if run == 0:  # Only need shape from first successful run
                        results[feature_type] = {
                            "shape": result.shape if result is not None else (0, 0),
                            "times": times,
                        }
                except Exception as e:
                    times.append(float('inf'))
                    results[feature_type] = {
                        "error": str(e),
                        "times": times,
                    }

            if feature_type in results and "error" not in results[feature_type]:
                avg_time = np.mean(results[feature_type]["times"])
                results[feature_type]["avg_time"] = avg_time
                results[feature_type]["std_time"] = np.std(results[feature_type]["times"])

        return results

    def optimize_chunk_size(self, genes: pd.Series, target_time: float = 30.0) -> int:
        """Find optimal chunk size for feature extraction."""

        chunk_sizes = [100, 500, 1000, 2000, 5000]
        best_size = self.chunk_size
        best_time = float('inf')

        print("🔍 Optimizing chunk size for feature extraction...")

        for size in chunk_sizes:
            self.chunk_size = size

            start_time = time.time()
            try:
                # Test with a subset of features
                result = self.extract_features_parallel(
                    genes[:min(100, len(genes))],
                    feature_types=["ppi"],  # Quick test
                    cache_key=None,  # No caching for benchmark
                )
                end_time = time.time()

                elapsed = end_time - start_time
                print(f"Chunk size {size}: {elapsed:.2f}s")

                if elapsed < best_time and elapsed > 0:
                    best_time = elapsed
                    best_size = size

                if elapsed < target_time:
                    break  # Good enough

            except Exception as e:
                print(f"Error with chunk size {size}: {e}")

        self.chunk_size = best_size
        print(f"✅ Optimal chunk size: {best_size} (best time: {best_time:.2f}s)")
        return best_size

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        import psutil
        import gc

        process = psutil.Process()
        memory_info = process.memory_info()

        # Force garbage collection
        gc.collect()

        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "available_mb": psutil.virtual_memory().available / (1024 * 1024),
        }

    def create_processing_report(
        self,
        genes: pd.Series,
        feature_types: List[str],
        output_path: Path,
    ) -> None:
        """Create a comprehensive processing report."""

        ensure_dir(output_path.parent)

        # Benchmark feature extraction
        benchmark_results = self.benchmark_feature_extraction(genes, feature_types)

        # Get memory usage
        memory_usage = self.get_memory_usage()

        # Create report
        report = {
            "dataset_info": {
                "n_genes": len(genes),
                "n_feature_types": len(feature_types),
                "unique_genes": genes.nunique(),
            },
            "performance": {
                "chunk_size": self.chunk_size,
                "max_workers": self.max_workers,
                "memory_usage_mb": memory_usage,
            },
            "feature_extraction_benchmarks": benchmark_results,
            "timestamp": time.time(),
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📊 Processing report saved to: {output_path}")
