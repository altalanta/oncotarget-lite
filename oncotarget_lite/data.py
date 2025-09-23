"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import set_random_seed


def load_data(data_dir: Path = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the synthetic oncotarget dataset."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "raw"
    
    # Load synthetic datasets
    gtex_df = pd.read_csv(data_dir / "gtex_medians.csv", index_col=0)
    tcga_df = pd.read_csv(data_dir / "tcga_medians.csv", index_col=0) 
    depmap_df = pd.read_csv(data_dir / "depmap_essentiality.csv", index_col=0)
    annotations_df = pd.read_csv(data_dir / "uniprot_annotations.csv", index_col=0)
    
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
    
    # Drop rows with missing values
    mask = ~(features.isna().any(axis=1) | labels.isna())
    features = features[mask]
    labels = labels[mask]
    
    return features, labels


def split_data(
    features: pd.DataFrame, 
    labels: pd.Series, 
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets with stratification."""
    set_random_seed(random_state)
    
    return train_test_split(
        features, 
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )