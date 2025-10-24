"""Optimized data processing using Polars for high-performance operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import polars as pl
import polars.selectors as cs
from polars import col

from .performance import performance_monitor, get_performance_monitor
from .utils import ensure_dir

logger = logging.getLogger(__name__)

# Try to import modin for distributed computing
try:
    import modin.pandas as mpd
    MODIN_AVAILABLE = True
except ImportError:
    MODIN_AVAILABLE = False


class OptimizedDataLoader:
    """High-performance data loader using Polars and optional distributed computing."""

    def __init__(self, use_polars: bool = True, use_modin: bool = False, chunk_size: int = 10000):
        self.use_polars = use_polars
        self.use_modin = use_modin and MODIN_AVAILABLE
        self.chunk_size = chunk_size
        self.monitor = get_performance_monitor()

    def load_features(self, features_path: Union[str, Path]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Load features with optimized backend."""
        features_path = Path(features_path)

        with performance_monitor(f"load_features_{features_path.name}"):
            if self.use_polars and features_path.suffix == '.parquet':
                return self._load_parquet_polars(features_path)
            elif self.use_modin:
                return self._load_with_modin(features_path)
            else:
                return self._load_with_pandas(features_path)

    def load_labels(self, labels_path: Union[str, Path]) -> Union[pd.Series, pl.Series]:
        """Load labels with optimized backend."""
        labels_path = Path(labels_path)

        with performance_monitor(f"load_labels_{labels_path.name}"):
            if self.use_polars and labels_path.suffix == '.parquet':
                df = pl.read_parquet(labels_path)
                return df['label']
            else:
                df = pd.read_parquet(labels_path)
                return df['label']

    def _load_parquet_polars(self, path: Path) -> pl.DataFrame:
        """Load parquet file using Polars."""
        return pl.read_parquet(path)

    def _load_with_modin(self, path: Path) -> pd.DataFrame:
        """Load file using Modin for distributed computing."""
        if path.suffix == '.parquet':
            return mpd.read_parquet(path)
        elif path.suffix == '.csv':
            return mpd.read_csv(path)
        else:
            return mpd.read_csv(path)

    def _load_with_pandas(self, path: Path) -> pd.DataFrame:
        """Load file using pandas."""
        if path.suffix == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix == '.csv':
            return pd.read_csv(path)
        else:
            return pd.read_csv(path)

    def process_features_streaming(
        self,
        features_path: Union[str, Path],
        operations: List[callable] = None
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Process large feature files in streaming fashion."""
        features_path = Path(features_path)

        with performance_monitor(f"streaming_process_{features_path.name}"):
            if self.use_polars:
                return self._process_polars_streaming(features_path, operations or [])
            else:
                return self._process_pandas_streaming(features_path, operations or [])

    def _process_polars_streaming(self, path: Path, operations: List[callable]) -> pl.DataFrame:
        """Process with Polars streaming API."""
        # Use lazy evaluation for better performance
        df = pl.scan_parquet(path)

        for operation in operations:
            df = operation(df)

        return df.collect()

    def _process_pandas_streaming(self, path: Path, operations: List[callable]) -> pd.DataFrame:
        """Process with pandas chunked reading."""
        chunks = []
        for chunk in pd.read_csv(path, chunksize=self.chunk_size):
            for operation in operations:
                chunk = operation(chunk)
            chunks.append(chunk)

        return pd.concat(chunks, ignore_index=True)

    def merge_datasets(
        self,
        features: Union[pd.DataFrame, pl.DataFrame],
        labels: Union[pd.Series, pl.Series],
        how: str = 'inner'
    ) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Union[pd.Series, pl.Series]]:
        """Merge features and labels efficiently."""
        with performance_monitor("merge_datasets"):
            if self.use_polars and isinstance(features, pl.DataFrame):
                # Polars merge
                if isinstance(labels, pl.Series):
                    labels_df = labels.to_frame('label')
                    merged = features.join(labels_df, how=how)
                    return merged.drop('label'), merged['label']
                else:
                    # Convert pandas series to polars
                    labels_pl = pl.from_pandas(labels.to_frame('label'))
                    merged = features.join(labels_pl, how=how)
                    return merged.drop('label'), merged['label']
            else:
                # Pandas merge
                if hasattr(features, 'index'):
                    features_df = features.copy()
                    labels_series = labels.copy()
                    if hasattr(labels_series, 'index'):
                        merged = features_df.join(labels_series.to_frame('label'), how=how)
                        return merged.drop(columns=['label']), merged['label']
                return features, labels

    def optimize_dtypes(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Optimize data types for memory efficiency."""
        with performance_monitor("optimize_dtypes"):
            if self.use_polars and isinstance(df, pl.DataFrame):
                return self._optimize_polars_dtypes(df)
            else:
                return self._optimize_pandas_dtypes(df)

    def _optimize_polars_dtypes(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimize Polars DataFrame dtypes."""
        # Downcast float64 to float32 where possible
        float64_cols = [name for name, dtype in df.schema.items()
                       if dtype == pl.Float64]

        if float64_cols:
            df = df.cast({col: pl.Float32 for col in float64_cols})

        # Convert object columns to categorical where appropriate
        # This is a simplified version - in practice you'd want more sophisticated logic
        return df

    def _optimize_pandas_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize pandas DataFrame dtypes."""
        df = df.copy()

        # Downcast numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float', errors='ignore')

        # Convert low-cardinality object columns to categorical
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.1:  # Less than 10% unique values
                df[col] = df[col].astype('category')

        return df

    def save_optimized(self, df: Union[pd.DataFrame, pl.DataFrame], path: Union[str, Path]) -> None:
        """Save DataFrame in optimized format."""
        path = Path(path)
        ensure_dir(path.parent)

        with performance_monitor(f"save_optimized_{path.name}"):
            if self.use_polars and isinstance(df, pl.DataFrame):
                # Polars parquet is highly optimized
                df.write_parquet(path, compression='snappy')
            else:
                # Convert to pandas if needed and save
                if isinstance(df, pl.DataFrame):
                    df = df.to_pandas()
                df.to_parquet(path, compression='snappy')


def create_optimized_pipeline(use_polars: bool = True, use_modin: bool = False) -> OptimizedDataLoader:
    """Factory function to create optimized data loader."""
    return OptimizedDataLoader(use_polars=use_polars, use_modin=use_modin)


# Feature extraction optimizations
class OptimizedFeatureExtractor:
    """Memory-efficient feature extraction."""

    def __init__(self, loader: OptimizedDataLoader):
        self.loader = loader
        self.cache = {}

    def extract_features_parallel(
        self,
        data_path: Union[str, Path],
        feature_functions: Dict[str, callable],
        n_jobs: int = -1
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Extract features in parallel using optimized processing."""
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing as mp

        n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()

        with performance_monitor(f"parallel_feature_extraction_{n_jobs}_jobs"):
            # Load data once
            data = self.loader.load_features(data_path)

            # Extract features in parallel
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                feature_futures = {
                    name: executor.submit(func, data)
                    for name, func in feature_functions.items()
                }

                features = {}
                for name, future in feature_futures.items():
                    features[name] = future.result()

            # Combine features
            if self.loader.use_polars:
                return pl.concat([pl.from_pandas(df) for df in features.values()], how='horizontal')
            else:
                return pd.concat(features.values(), axis=1)


# Backward compatibility utilities
def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Convert pandas DataFrame to Polars DataFrame."""
    return pl.from_pandas(df)


def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars DataFrame to pandas DataFrame."""
    return df.to_pandas()


def optimize_dataframe_memory(df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
    """Optimize any DataFrame for memory usage."""
    loader = OptimizedDataLoader()
    return loader.optimize_dtypes(df)

