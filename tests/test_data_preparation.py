"""Tests for data preparation functionality."""

from pathlib import Path
import pandas as pd
import pytest

from oncotarget_lite.data import build_feature_matrix, prepare_dataset, DataPreparationError


def create_synthetic_raw_data(raw_dir: Path):
    """Create minimal synthetic raw data for testing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # GTEx data
    gtex_data = pd.DataFrame({
        "gene": ["GENE001", "GENE002", "GENE003"] * 2,
        "tissue": ["Brain", "Brain", "Brain", "Heart", "Heart", "Heart"],
        "median_TPM": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]
    })
    gtex_data.to_csv(raw_dir / "GTEx_subset.csv", index=False)
    
    # TCGA data
    tcga_data = pd.DataFrame({
        "gene": ["GENE001", "GENE002", "GENE003"] * 2,
        "tumor": ["BRCA", "BRCA", "BRCA", "LUAD", "LUAD", "LUAD"],
        "median_TPM": [2.0, 3.0, 4.0, 2.5, 3.5, 4.5]
    })
    tcga_data.to_csv(raw_dir / "TCGA_subset.csv", index=False)
    
    # DepMap data
    depmap_data = pd.DataFrame({
        "gene": ["GENE001", "GENE002", "GENE003"] * 2,
        "cell_line": ["ACH-001", "ACH-001", "ACH-001", "ACH-002", "ACH-002", "ACH-002"],
        "dependency_score": [-0.5, -1.0, 0.2, -0.3, -0.8, 0.1]
    })
    depmap_data.to_csv(raw_dir / "DepMap_essentials_subset.csv", index=False)
    
    # Annotations
    annotations_data = pd.DataFrame({
        "gene": ["GENE001", "GENE002", "GENE003"],
        "is_cell_surface": [1, 0, 1],
        "signal_peptide": [1, 0, 1],
        "ig_like_domain": [0, 1, 0],
        "protein_length": [500, 300, 800]
    })
    annotations_data.to_csv(raw_dir / "uniprot_annotations.csv", index=False)
    
    # PPI data
    ppi_data = pd.DataFrame({
        "gene": ["GENE001", "GENE002", "GENE003"],
        "degree": [10, 5, 15]
    })
    ppi_data.to_csv(raw_dir / "ppi_degree_subset.csv", index=False)


def test_build_feature_matrix(tmp_path: Path):
    """Test feature matrix construction."""
    raw_dir = tmp_path / "raw"
    create_synthetic_raw_data(raw_dir)
    
    features, labels = build_feature_matrix(raw_dir)
    
    assert len(features) == 3  # 3 genes
    assert len(labels) == 3
    assert features.index.tolist() == ["GENE001", "GENE002", "GENE003"]
    
    # Check that log2fc columns are created
    log2fc_cols = [col for col in features.columns if col.startswith("log2fc_")]
    assert len(log2fc_cols) == 2  # BRCA and LUAD
    
    # Check other feature columns
    expected_cols = ["min_normal_tpm", "mean_tumor_tpm", "mean_dependency", 
                     "ppi_degree", "signal_peptide", "ig_like_domain", "protein_length"]
    for col in expected_cols:
        assert col in features.columns
    
    # Check labels
    assert labels.tolist() == [1, 0, 1]  # From is_cell_surface


def test_prepare_dataset(tmp_path: Path):
    """Test full dataset preparation."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    create_synthetic_raw_data(raw_dir)
    
    result = prepare_dataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        test_size=0.3,
        seed=42
    )
    
    assert isinstance(result.features, pd.DataFrame)
    assert isinstance(result.labels, pd.Series)
    assert len(result.train_genes) + len(result.test_genes) == 3
    assert isinstance(result.dataset_fingerprint, str)
    
    # Check that files were created
    assert (processed_dir / "features.parquet").exists()
    assert (processed_dir / "labels.parquet").exists()
    assert (processed_dir / "splits.json").exists()


def test_prepare_dataset_degenerate_labels(tmp_path: Path):
    """Test that degenerate labels (all 0 or all 1) raise error."""
    raw_dir = tmp_path / "raw"
    create_synthetic_raw_data(raw_dir)
    
    # Modify annotations to have all zeros
    annotations_data = pd.DataFrame({
        "gene": ["GENE001", "GENE002", "GENE003"],
        "is_cell_surface": [0, 0, 0],  # All zeros
        "signal_peptide": [1, 0, 1],
        "ig_like_domain": [0, 1, 0],
        "protein_length": [500, 300, 800]
    })
    annotations_data.to_csv(raw_dir / "uniprot_annotations.csv", index=False)
    
    processed_dir = tmp_path / "processed"
    
    with pytest.raises(DataPreparationError) as exc_info:
        prepare_dataset(raw_dir=raw_dir, processed_dir=processed_dir)
    
    assert "degenerate" in str(exc_info.value)