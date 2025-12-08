"""Data preparation utilities for the oncotarget-lite pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datetime import datetime

REQUIRED_COLUMNS = ("gene", "median_TPM")
from sklearn.model_selection import train_test_split

from .utils import dataset_hash, ensure_dir, save_dataframe, save_json, set_seeds
from .features.orchestrator import FeatureOrchestrator
from .scalable_loader import ScalableDataLoader
from .scalable_orchestrator import ScalableFeatureOrchestrator
from .optimized_data import OptimizedDataLoader, create_optimized_pipeline
from .performance import enable_performance_monitoring, get_performance_monitor

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


from .exceptions import DataPreparationError
from .data_quality import DataQualityMonitor, DataLineageEntry
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PreparedData:
    features: pd.DataFrame
    labels: pd.Series
    train_genes: list[str]
    test_genes: list[str]
    dataset_fingerprint: str


RAW_FILES = {
    "gtex": "expression.csv",  # GTEx expression data
    "tcga": "expression.csv",  # TCGA expression data (using same file for demo)
    "depmap": "dependencies.csv",  # DepMap dependency scores
    "annotations": "annotations.csv",  # UniProt annotations (no median_TPM needed)
    "ppi": "ppi_degree_subset.csv",  # PPI degree data
}


_DEF_TUMOUR_PREFIX = "tumor_"
_DEF_NORMAL_PREFIX = "normal_"
_DEF_DEP_PREFIX = "dep_"
_REQUIRED_COLUMNS = {
    "is_cell_surface",
    "signal_peptide",
    "ig_like_domain",
    "ppi_degree",
    "protein_length",
}


def _read_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV loader:
    - Ignores lines starting with '#' (metadata/comment headers).
    - Strips BOM if present.
    - Provides actionable error if required columns are missing.
    """
    if not path.exists():
        raise DataPreparationError(f"Missing synthetic data file: {path}")

    df = pd.read_csv(path, comment="#")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Different files have different required columns
    if "expression.csv" in str(path):
        # Expression files need gene and median_TPM
        required = ["gene", "median_TPM"]
    elif "dependencies.csv" in str(path):
        # Dependencies need gene and median_TPM (we renamed dep_score to median_TPM)
        required = ["gene", "median_TPM"]
    elif "annotations.csv" in str(path):
        # Annotations need gene and annotation columns
        required = ["gene"]
    elif "ppi_degree_subset.csv" in str(path):
        # PPI data needs gene and degree
        required = ["gene", "degree"]
    else:
        # Default to original required columns for other files
        required = REQUIRED_COLUMNS

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataPreparationError(
            f"{path}: missing required columns {missing}; "
            f"got {list(df.columns)}. Ensure comment headers start with '#'."
        )
    return df


def _load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    # Load unique files, handling duplicates
    loaded_files = {}
    for key, filename in RAW_FILES.items():
        file_path = raw_dir / filename
        if file_path not in loaded_files:
            loaded_files[file_path] = _read_csv(file_path)

        # For GTEx and TCGA both using expression.csv, they need different processing
        if key == "gtex":
            tables[key] = loaded_files[file_path].copy()
        elif key == "tcga":
            tables[key] = loaded_files[file_path].copy()  # Same data, different processing in _wide_expression
        else:
            tables[key] = loaded_files[file_path]

    return tables


def _load_raw_tables_optimized(raw_dir: Path, optimized_loader: OptimizedDataLoader) -> dict[str, pd.DataFrame]:
    """Load raw tables using optimized data processing."""
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp

    tables: dict[str, pd.DataFrame] = {}

    # Load unique files in parallel using optimized loader
    unique_files = {}
    for key, filename in RAW_FILES.items():
        file_path = raw_dir / filename
        if file_path not in unique_files:
            unique_files[file_path] = key

    # Process files in parallel
    n_jobs = min(mp.cpu_count(), len(unique_files))
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        file_futures = {
            file_path: executor.submit(_read_csv_optimized, file_path, optimized_loader)
            for file_path in unique_files.keys()
        }

        loaded_files = {}
        for file_path, future in file_futures.items():
            loaded_files[file_path] = future.result()

    # Distribute loaded files to tables
    for key, filename in RAW_FILES.items():
        file_path = raw_dir / filename
        if key == "gtex":
            tables[key] = loaded_files[file_path].copy()
        elif key == "tcga":
            tables[key] = loaded_files[file_path].copy()  # Same data, different processing in _wide_expression
        else:
            tables[key] = loaded_files[file_path]

    return tables


def _read_csv_optimized(path: Path, optimized_loader: OptimizedDataLoader) -> pd.DataFrame:
    """Read CSV using optimized processing."""
    # Use optimized loader for better performance
    df = optimized_loader.load_features(path)

    # Convert back to pandas if needed for compatibility
    if hasattr(df, 'to_pandas'):
        df = df.to_pandas()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Different files have different required columns
    if "expression.csv" in str(path):
        required = ["gene", "median_TPM"]
    elif "dependencies.csv" in str(path):
        required = ["gene", "median_TPM"]
    elif "annotations.csv" in str(path):
        required = ["gene"]
    elif "ppi_degree_subset.csv" in str(path):
        required = ["gene", "degree"]
    else:
        required = REQUIRED_COLUMNS

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataPreparationError(
            f"{path}: missing required columns {missing}; "
            f"got {list(df.columns)}. Ensure comment headers start with '#'."
        )
    return df


def _wide_expression(frame: pd.DataFrame, pivot_col: str, prefix: str) -> pd.DataFrame:
    """Create wide format expression data with synthetic tissue types for demo."""
    # For demo purposes, create synthetic tissue types since we only have one expression file
    if pivot_col == "tissue":
        # Create synthetic normal tissue data
        n_genes = len(frame)
        n_tissues = 5  # Simulate 5 normal tissues
        synthetic_data = {}
        for i in range(n_tissues):
            tissue_name = f"normal_tissue_{i+1}"
            synthetic_data[tissue_name] = frame["median_TPM"] * (0.8 + np.random.random() * 0.4)

        wide = pd.DataFrame(synthetic_data, index=frame["gene"])
        wide = wide.add_prefix(prefix)
        return wide.sort_index()

    elif pivot_col == "tumor":
        # Create synthetic tumor tissue data
        n_genes = len(frame)
        n_tissues = 5  # Simulate 5 tumor types
        synthetic_data = {}
        for i in range(n_tissues):
            tissue_name = f"tumor_type_{i+1}"
            synthetic_data[tissue_name] = frame["median_TPM"] * (0.6 + np.random.random() * 0.8)

        wide = pd.DataFrame(synthetic_data, index=frame["gene"])
        wide = wide.add_prefix(prefix)
        return wide.sort_index()

    else:
        # Fallback for other pivot columns
        wide = frame.pivot_table(index="gene", columns=pivot_col, values="median_TPM")
        wide = wide.add_prefix(prefix)
        return wide.sort_index()


def _merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gtex_wide = _wide_expression(tables["gtex"], "tissue", _DEF_NORMAL_PREFIX)
    tcga_wide = _wide_expression(tables["tcga"], "tumor", _DEF_TUMOUR_PREFIX)

    depmap = tables["depmap"].set_index("gene").add_prefix(_DEF_DEP_PREFIX).sort_index()
    annotations = tables["annotations"].set_index("gene").sort_index()
    ppi = tables["ppi"].set_index("gene").rename(columns={"degree": "ppi_degree"}).sort_index()

    pieces: Iterable[pd.DataFrame] = (gtex_wide, tcga_wide, depmap, annotations, ppi)
    merged = pd.concat(pieces, axis=1, join="inner").sort_index()
    if merged.empty:
        raise DataPreparationError("Merged feature table is empty")
    missing = _REQUIRED_COLUMNS - set(merged.columns)
    if missing:
        raise DataPreparationError(f"Merged feature table missing columns: {sorted(missing)}")
    return merged


def build_feature_matrix(raw_dir: Path = RAW_DIR, use_advanced_features: bool = True, use_scalable_processing: bool = True, use_optimized_processing: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Load cached CSVs and derive model-ready features + labels with optional scalable and optimized processing."""

    # Enable performance monitoring
    enable_performance_monitoring()

    if use_optimized_processing:
        # Use optimized data loader for maximum performance
        optimized_loader = create_optimized_pipeline(use_polars=True, use_modin=False)
        tables = _load_raw_tables_optimized(raw_dir, optimized_loader)
        logger.info("data_tables_loaded", count=len(tables), mode="optimized")
    elif use_scalable_processing:
        # Use scalable data loader for better performance
        loader = ScalableDataLoader()
        tables = loader.load_raw_tables_parallel(raw_dir)
        logger.info("data_tables_loaded", count=len(tables), mode="parallel")
    else:
        # Use original sequential loading
        tables = _load_raw_tables(raw_dir)

    merged = _merge_tables(tables)

    normal_cols = [col for col in merged if col.startswith(_DEF_NORMAL_PREFIX)]
    tumour_cols = [col for col in merged if col.startswith(_DEF_TUMOUR_PREFIX)]
    dep_cols = [col for col in merged if col.startswith(_DEF_DEP_PREFIX)]

    normal = merged[normal_cols]
    tumour = merged[tumour_cols]
    dependency = merged[dep_cols]

    normal_mean = normal.mean(axis=1)
    tumour_mean = tumour.mean(axis=1)

    # Basic expression and dependency features
    features = pd.DataFrame(index=merged.index)
    for col in tumour_cols:
        tumour_name = col.replace(_DEF_TUMOUR_PREFIX, "").lower()
        features[f"log2fc_{tumour_name}"] = np.log2((merged[col] + 1.0) / (normal_mean + 1.0))
    features["min_normal_tpm"] = normal.min(axis=1)
    features["mean_tumor_tpm"] = tumour_mean
    features["mean_dependency"] = dependency.mean(axis=1)
    features["ppi_degree"] = merged["ppi_degree"].astype(float)
    features["signal_peptide"] = merged["signal_peptide"].astype(float)
    features["ig_like_domain"] = merged["ig_like_domain"].astype(float)
    features["protein_length"] = merged["protein_length"].astype(float)

    # Advanced biological features (optional)
    if use_advanced_features:
        logger.info("extracting_advanced_features")
        if use_scalable_processing:
            feature_orchestrator = ScalableFeatureOrchestrator()
            advanced_features = feature_orchestrator.extract_features_parallel(
                pd.Series(merged.index),
                cache_key="build_feature_matrix_advanced"
            )
        else:
            feature_orchestrator = FeatureOrchestrator()
            advanced_features = feature_orchestrator.extract_all_features(
                pd.Series(merged.index),
                cache_key="build_feature_matrix_advanced"
            )

        # Create cache key from dataset hash for reproducible caching
        genes = merged.index
        dataset_hash_str = str(hash(str(genes.tolist())))[:8]  # Simple hash for caching
        cache_key = f"advanced_features_{dataset_hash_str}"

        try:
            if use_scalable_processing:
                advanced_features = feature_orchestrator.extract_features_parallel(
                    genes,
                    cache_key=cache_key
                )
            else:
                advanced_features = feature_orchestrator.extract_all_features(
                    genes,
                    cache_key=cache_key
                )

            # Merge advanced features with basic features
            features = pd.concat([features, advanced_features], axis=1)

            # Log feature summary
            if hasattr(feature_orchestrator, 'get_feature_summary'):
                summary = feature_orchestrator.get_feature_summary(advanced_features)
                logger.info(
                    "advanced_features_extracted",
                    total_features=summary['total_features'],
                    total_genes=summary['total_genes'],
                    categories=summary['feature_categories'],
                )
            else:
                logger.info(
                    "advanced_features_extracted",
                    total_features=advanced_features.shape[1],
                    total_genes=advanced_features.shape[0],
                )

        except Exception as e:
            logger.warning("advanced_feature_extraction_failed", error=str(e))
            logger.info("continuing_with_basic_features")

    labels = merged["is_cell_surface"].astype(int)
    return features, labels


def prepare_dataset(
    *,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    test_size: float = 0.3,
    seed: int = 42,
    use_optimized_processing: bool = True,
) -> PreparedData:
    """Generate the canonical processed artefacts for downstream stages with optional optimization."""

    set_seeds(seed)

    # Initialize data quality monitoring
    quality_monitor = DataQualityMonitor()

    features, labels = build_feature_matrix(
        raw_dir,
        use_optimized_processing=use_optimized_processing
    )

    if labels.sum() == 0 or labels.sum() == len(labels):
        raise DataPreparationError("Labels are degenerate; need both positive and negative samples")

    train_genes, test_genes = train_test_split(
        list(features.index),
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    ordered_index = sorted(features.index)
    dataset_fp = dataset_hash(features.loc[ordered_index], labels.loc[ordered_index])

    ensure_dir(processed_dir)
    save_dataframe(processed_dir / "features.parquet", features)
    save_dataframe(processed_dir / "labels.parquet", labels.to_frame(name="label"))
    save_json(
        processed_dir / "splits.json",
        {
            "train_genes": train_genes,
            "test_genes": test_genes,
            "test_size": test_size,
            "seed": seed,
            "dataset_hash": dataset_fp,
        },
    )

    # Register prepared data with quality monitoring system
    dataset_id = f"prepared_{dataset_fp[:8]}"
    combined_df = features.copy()
    combined_df['label'] = labels

    try:
        quality_profile = quality_monitor.register_dataset(
            df=combined_df,
            dataset_id=dataset_id,
            data_source="data_preparation_pipeline",
            parent_datasets=[str(raw_dir)]  # Raw data as parent
        )

        # Create lineage entry for data preparation
        lineage_entry = DataLineageEntry(
            operation_id=f"prepare_{dataset_fp[:8]}",
            operation_type="data_preparation",
            input_datasets=[str(raw_dir)],
            output_datasets=[dataset_id],
            parameters={
                "test_size": test_size,
                "seed": seed,
                "raw_dir": str(raw_dir),
                "processed_dir": str(processed_dir)
            },
            timestamp=datetime.now(),
            success=True
        )
        quality_monitor.lineage_tracker.add_entry(lineage_entry)

    except Exception as e:
        logger.warning(f"Failed to register dataset with quality monitoring: {e}")

    return PreparedData(
        features=features,
        labels=labels,
        train_genes=train_genes,
        test_genes=test_genes,
        dataset_fingerprint=dataset_fp,
    )
