"""Tests for utility functions."""

from pathlib import Path
import pandas as pd
import pytest

from oncotarget_lite.utils import ensure_dir, save_dataframe, dataset_hash, set_seeds


def test_ensure_dir(tmp_path: Path):
    """Test directory creation utility."""
    test_dir = tmp_path / "nested" / "directory"
    assert not test_dir.exists()
    
    ensure_dir(test_dir)
    assert test_dir.exists()
    assert test_dir.is_dir()
    
    # Should not fail if directory already exists
    ensure_dir(test_dir)
    assert test_dir.exists()


def test_save_dataframe(tmp_path: Path):
    """Test dataframe saving utility."""
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })
    
    file_path = tmp_path / "test.parquet"
    save_dataframe(file_path, df)
    
    assert file_path.exists()
    
    # Load and verify
    loaded_df = pd.read_parquet(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)


def test_dataset_hash():
    """Test dataset fingerprinting."""
    features = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0],
        "feat2": [4.0, 5.0, 6.0]
    }, index=["A", "B", "C"])
    
    labels = pd.Series([0, 1, 0], index=["A", "B", "C"])
    
    hash1 = dataset_hash(features, labels)
    hash2 = dataset_hash(features, labels)
    
    # Same data should produce same hash
    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) > 0
    
    # Different data should produce different hash
    features_diff = features.copy()
    features_diff.iloc[0, 0] = 999.0
    hash3 = dataset_hash(features_diff, labels)
    assert hash1 != hash3


def test_set_seeds():
    """Test seed setting utility."""
    import numpy as np
    import random
    
    # Set seeds
    set_seeds(42)
    val1 = np.random.random()
    val2 = random.random()
    
    # Reset and check reproducibility
    set_seeds(42)
    val3 = np.random.random()
    val4 = random.random()
    
    assert val1 == val3
    assert val2 == val4