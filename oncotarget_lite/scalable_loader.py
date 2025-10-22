"""Scalable data loading utilities with parallel processing and advanced caching."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .exceptions import DataPreparationError
from .utils import ensure_dir


class ScalableDataLoader:
    """High-performance data loader with parallel processing and caching."""

    def __init__(
        self,
        cache_dir: Path = Path("data/cache"),
        max_workers: int = None,
        chunk_size: int = 10000,
        use_compression: bool = True,
        cache_format: str = "parquet",
    ):
        self.cache_dir = cache_dir
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)
        self.chunk_size = chunk_size
        self.use_compression = use_compression
        self.cache_format = cache_format

        ensure_dir(cache_dir)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file for cache invalidation."""
        hash_obj = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def _get_cache_key(self, file_path: Path, processing_params: Dict[str, Any]) -> str:
        """Generate cache key based on file hash and processing parameters."""
        file_hash = self._compute_file_hash(file_path)
        params_str = json.dumps(processing_params, sort_keys=True)
        combined = f"{file_hash}_{params_str}".encode()
        return hashlib.sha256(combined).hexdigest()

    def _load_csv_chunked(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file with chunked processing for large files."""
        try:
            # First, determine file size and structure
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#'):
                    # Skip comment lines
                    header_found = False
                    for line in f:
                        if not line.strip().startswith('#'):
                            header_line = line.strip()
                            break

                    # Read the actual header
                    df_iter = pd.read_csv(
                        file_path,
                        chunksize=self.chunk_size,
                        comment='#',
                        **kwargs
                    )

                    # Process in chunks
                    chunks = []
                    for chunk in df_iter:
                        chunks.append(chunk)

                    if chunks:
                        df = pd.concat(chunks, ignore_index=True)
                        # Set the proper column names from the first chunk
                        df.columns = chunks[0].columns
                        return df
                    else:
                        return pd.DataFrame()
                else:
                    # Standard CSV reading
                    return pd.read_csv(file_path, **kwargs)

        except Exception as e:
            raise DataPreparationError(f"Failed to load {file_path}: {e}")

    def _process_file_parallel(self, file_info: Tuple[Path, Dict[str, Any]]) -> Tuple[str, pd.DataFrame]:
        """Process a single file in parallel."""
        file_path, processing_params = file_info

        # Check cache first
        cache_key = self._get_cache_key(file_path, processing_params)
        cache_file = self.cache_dir / f"{cache_key}.{self.cache_format}"

        if cache_file.exists():
            try:
                if self.cache_format == "parquet":
                    return file_path.name, pd.read_parquet(cache_file)
                else:
                    return file_path.name, pd.read_pickle(cache_file)
            except Exception:
                # Cache corrupted, will recompute
                pass

        # Load and process file
        df = self._load_csv_chunked(file_path, **processing_params)

        # Cache the result
        if self.use_compression and self.cache_format == "parquet":
            df.to_parquet(cache_file, compression="snappy")
        elif self.cache_format == "pickle":
            df.to_pickle(cache_file)

        return file_path.name, df

    def load_files_parallel(
        self,
        file_configs: Dict[str, Tuple[Path, Dict[str, Any]]],
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple files in parallel with caching."""

        # Prepare file processing tasks
        tasks = []
        for name, (file_path, params) in file_configs.items():
            if file_path.exists():
                tasks.append((name, (file_path, params)))
            else:
                print(f"Warning: File not found: {file_path}")

        if not tasks:
            raise DataPreparationError("No valid files found to load")

        # Process files in parallel
        results = {}
        failed_files = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_name = {
                executor.submit(self._process_file_parallel, task_info): name
                for name, task_info in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    file_name, df = future.result()
                    results[name] = df
                    if show_progress:
                        print(f"✅ Loaded {name}: {df.shape}")
                except Exception as e:
                    failed_files.append((name, str(e)))
                    if show_progress:
                        print(f"❌ Failed to load {name}: {e}")

        if failed_files:
            print(f"Warning: {len(failed_files)} files failed to load")

        return results

    def load_raw_tables_parallel(self, raw_dir: Path) -> Dict[str, pd.DataFrame]:
        """Load all raw tables in parallel with enhanced processing."""

        # Define file processing configurations
        file_configs = {}

        # Expression files (GTEx and TCGA)
        for key, filename in [("gtex", "expression.csv"), ("tcga", "expression.csv")]:
            file_path = raw_dir / filename
            file_configs[key] = (
                file_path,
                {
                    "comment": "#",
                    "usecols": ["gene", "median_TPM"],
                    "dtype": {"gene": str, "median_TPM": float},
                }
            )

        # Dependencies file
        file_configs["depmap"] = (
            raw_dir / "dependencies.csv",
            {
                "comment": "#",
                "usecols": ["gene", "median_TPM"],
                "dtype": {"gene": str, "median_TPM": float},
            }
        )

        # Annotations file
        file_configs["annotations"] = (
            raw_dir / "annotations.csv",
            {
                "comment": "#",
                "dtype": {col: str for col in ["gene", "is_oncogene", "is_cell_surface", "signal_peptide", "ig_like_domain"]},
            }
        )

        # PPI file
        file_configs["ppi"] = (
            raw_dir / "ppi_degree_subset.csv",
            {
                "comment": "#",
                "usecols": ["gene", "degree"],
                "dtype": {"gene": str, "degree": float},
            }
        )

        return self.load_files_parallel(file_configs)

    def clear_cache(self, pattern: str = "*") -> int:
        """Clear cache files matching pattern."""
        import glob

        cache_files = list(self.cache_dir.glob(f"{pattern}.*"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Warning: Could not remove cache file {cache_file}: {e}")

        return len(cache_files)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.parquet")) + list(self.cache_dir.glob("*.pkl"))

        total_size = 0
        file_sizes = {}

        for cache_file in cache_files:
            try:
                size = cache_file.stat().st_size
                total_size += size
                file_sizes[cache_file.name] = size
            except Exception:
                continue

        return {
            "total_files": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "file_sizes": file_sizes,
        }
