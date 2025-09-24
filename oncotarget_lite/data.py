"""Data loading and preprocessing utilities with comprehensive validation.

Provides robust data loading with extensive validation, error handling,
and logging for the oncotarget synthetic dataset.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .exceptions import DataValidationError
from .logging_utils import get_logger, log_data_summary
from .utils import set_random_seed


def _validate_datasets(gtex_df: pd.DataFrame, tcga_df: pd.DataFrame, 
                      depmap_df: pd.DataFrame, annotations_df: pd.DataFrame) -> None:
    """Validate loaded datasets for consistency and completeness.
    
    Args:
        gtex_df: GTEx normal tissue expression data
        tcga_df: TCGA tumor expression data
        depmap_df: DepMap essentiality scores
        annotations_df: UniProt annotations
        
    Raises:
        DataValidationError: If datasets fail validation checks
    """
    logger = get_logger(__name__)
    
    # Check for empty datasets
    datasets = [
        (gtex_df, "GTEx"), (tcga_df, "TCGA"), 
        (depmap_df, "DepMap"), (annotations_df, "UniProt")
    ]
    
    for df, name in datasets:
        if df.empty:
            raise DataValidationError(f"{name} dataset is empty")
        if df.isnull().all().all():
            raise DataValidationError(f"{name} dataset contains only NaN values")
    
    # Check for overlapping gene indices
    common_genes = set(gtex_df.index) & set(tcga_df.index) & set(depmap_df.index) & set(annotations_df.index)
    
    if len(common_genes) == 0:
        raise DataValidationError("No common genes found across all datasets")
    
    if len(common_genes) < 10:
        logger.warning(f"Only {len(common_genes)} common genes found across datasets")
    
    # Validate expected columns
    expected_tcga_tissues = ["BRCA", "LUAD", "COAD"]
    missing_tissues = [t for t in expected_tcga_tissues if t not in tcga_df.columns]
    if missing_tissues:
        logger.warning(f"Missing TCGA tissues: {missing_tissues}")
    
    expected_annotations = ["molecular_weight", "transmembrane", "signal_peptide"]
    missing_annotations = [c for c in expected_annotations if c not in annotations_df.columns]
    if missing_annotations:
        logger.warning(f"Missing annotation columns: {missing_annotations}")
    
    # Check data ranges
    if (gtex_df < 0).any().any():
        raise DataValidationError("GTEx data contains negative expression values")
    
    if (tcga_df < 0).any().any():
        raise DataValidationError("TCGA data contains negative expression values")
    
    logger.info(f"Dataset validation passed for {len(common_genes)} common genes")


def load_data(data_dir: Path = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the synthetic oncotarget dataset with validation.
    
    Loads synthetic datasets (GTEx, TCGA, DepMap, UniProt) and performs
    feature engineering with comprehensive data validation.
    
    Args:
        data_dir: Path to data directory (defaults to project data/raw)
        
    Returns:
        Tuple of (features_df, labels_series)
        
    Raises:
        DataValidationError: If required files are missing or data is invalid
        FileNotFoundError: If data directory doesn't exist
        
    Example:
        >>> features, labels = load_data()
        >>> print(f"Loaded {len(features)} samples with {len(features.columns)} features")
    """
    logger = get_logger(__name__)
    
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "raw"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    logger.info(f"Loading data from {data_dir}")
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "raw"
    
    # Define required files
    required_files = {
        "gtex_medians.csv": "GTEx normal tissue expression",
        "tcga_medians.csv": "TCGA tumor expression", 
        "depmap_essentiality.csv": "DepMap gene essentiality",
        "uniprot_annotations.csv": "UniProt gene annotations"
    }
    
    # Validate file existence
    for filename, description in required_files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            raise DataValidationError(
                f"Required file not found: {filename} ({description})",
                details={"missing_file": str(filepath)}
            )
    
    # Load synthetic datasets with error handling
    try:
        gtex_df = pd.read_csv(data_dir / "gtex_medians.csv", index_col=0)
        tcga_df = pd.read_csv(data_dir / "tcga_medians.csv", index_col=0) 
        depmap_df = pd.read_csv(data_dir / "depmap_essentiality.csv", index_col=0)
        annotations_df = pd.read_csv(data_dir / "uniprot_annotations.csv", index_col=0)
    except Exception as e:
        raise DataValidationError(f"Failed to load CSV files: {str(e)}")
    
    # Log data summaries
    log_data_summary(logger, gtex_df, "GTEx data")
    log_data_summary(logger, tcga_df, "TCGA data")
    log_data_summary(logger, depmap_df, "DepMap data")
    log_data_summary(logger, annotations_df, "UniProt annotations")
    
    # Validate data integrity
    _validate_datasets(gtex_df, tcga_df, depmap_df, annotations_df)
    
    # Simple feature engineering
    features = pd.DataFrame(index=gtex_df.index)
    
    # Tumor vs normal contrast features
    for tissue in ["BRCA", "LUAD", "COAD"]:
        if tissue in tcga_df.columns:
            features[f"tumor_vs_normal_{tissue}"] = (
                tcga_df[tissue] - gtex_df.mean(axis=1)
            )
    
    # Safety features from essentiality scores
    if "dependency_score" in depmap_df.columns:
        features["dependency_score"] = depmap_df["dependency_score"]
        features["is_essential"] = (depmap_df["dependency_score"] < -0.5).astype(int)
    
    # Annotation-based features
    for col in ["molecular_weight", "transmembrane", "signal_peptide"]:
        if col in annotations_df.columns:
            features[col] = annotations_df[col]
    
    # Create synthetic binary labels (high tumor expression + low essentiality)
    labels = (
        (features.filter(like="tumor_vs_normal").mean(axis=1) > 1.0) & 
        (features.get("dependency_score", 0) > -0.3)
    ).astype(int)
    
    # Validate feature matrix
    if features.empty:
        raise DataValidationError("No features generated from input data")
    
    # Drop rows with missing values
    initial_rows = len(features)
    mask = ~(features.isna().any(axis=1) | labels.isna())
    features = features[mask]
    labels = labels[mask]
    
    dropped_rows = initial_rows - len(features)
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with missing values")
    
    # Final validation
    if len(features) == 0:
        raise DataValidationError("No valid samples remaining after cleaning")
    
    if features.isnull().any().any():
        raise DataValidationError("Features still contain NaN values after cleaning")
    
    # Check label distribution
    label_counts = labels.value_counts()
    logger.info(f"Label distribution: {dict(label_counts)}")
    
    if len(label_counts) != 2:
        raise DataValidationError(f"Expected binary labels, got {len(label_counts)} classes")
    
    min_class_size = label_counts.min()
    if min_class_size < 10:
        logger.warning(f"Small minority class: {min_class_size} samples")
    
    logger.info(f"Successfully loaded {len(features)} samples with {len(features.columns)} features")
    return features, labels


def split_data(
    features: pd.DataFrame, 
    labels: pd.Series, 
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets with stratified sampling and validation.
    
    Args:
        features: Feature matrix DataFrame
        labels: Target labels Series
        test_size: Proportion for test set (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        DataValidationError: If data is invalid or split parameters are incorrect
        
    Example:
        >>> X_train, X_test, y_train, y_test = split_data(features, labels)
        >>> print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    """
    logger = get_logger(__name__)
    
    # Validate inputs
    if features.empty or labels.empty:
        raise DataValidationError("Cannot split empty datasets")
    
    if len(features) != len(labels):
        raise DataValidationError(
            f"Feature and label lengths don't match: {len(features)} vs {len(labels)}"
        )
    
    if not (0.0 < test_size < 1.0):
        raise DataValidationError(f"test_size must be between 0 and 1, got {test_size}")
    
    # Check minimum samples per class for stratification
    label_counts = labels.value_counts()
    min_samples = int(len(labels) * test_size)
    
    for label_value, count in label_counts.items():
        if count < 2:  # Need at least 2 samples per class
            raise DataValidationError(
                f"Insufficient samples for stratification: class {label_value} has {count} samples"
            )
    set_random_seed(random_state)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features, 
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
    except Exception as e:
        raise DataValidationError(f"Failed to split data: {str(e)}")
    
    # Validate splits
    logger.info(f"Split data: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"Train label distribution: {dict(y_train.value_counts())}")
    logger.info(f"Test label distribution: {dict(y_test.value_counts())}")
    
    return X_train, X_test, y_train, y_test