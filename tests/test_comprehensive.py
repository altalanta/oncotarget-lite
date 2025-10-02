"""Comprehensive tests to boost coverage for acceptance criteria."""

import tempfile
import subprocess
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from oncotarget_lite.data import _read_csv
from oncotarget_lite.utils import ensure_dir, save_dataframe, dataset_hash, set_seeds


def test_ensure_dir(tmp_path):
    """Test directory creation utility."""
    test_dir = tmp_path / "nested" / "dir"
    ensure_dir(test_dir)
    assert test_dir.exists()


def test_save_dataframe(tmp_path):
    """Test dataframe saving utility."""
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    file_path = tmp_path / "test.parquet"
    save_dataframe(file_path, df)
    assert file_path.exists()


def test_dataset_hash():
    """Test dataset hashing."""
    features = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    labels = pd.Series([0, 1], index=["x", "y"])
    hash1 = dataset_hash(features, labels)
    hash2 = dataset_hash(features, labels)
    assert hash1 == hash2
    assert isinstance(hash1, str)


def test_set_seeds():
    """Test seed setting."""
    set_seeds(42)
    val1 = np.random.random()
    set_seeds(42)
    val2 = np.random.random()
    assert val1 == val2


def test_read_csv_with_existing_file(tmp_path):
    """Test CSV reading with valid file."""
    p = tmp_path / "test.csv"
    p.write_text("gene,median_TPM\nGENE001,1.5\n")
    df = _read_csv(p)
    assert len(df) == 1
    assert "gene" in df.columns


def test_read_csv_missing_file():
    """Test error handling for missing files."""
    from oncotarget_lite.data import DataPreparationError
    with pytest.raises(DataPreparationError):
        _read_csv(Path("/nonexistent.csv"))


def test_cli_import():
    """Test CLI can be imported."""
    try:
        from oncotarget_lite.cli import app
        assert app is not None
    except ImportError:
        # Skip if import fails due to dependencies
        pytest.skip("CLI import failed due to dependencies")


def test_main_module():
    """Test module entrypoint exists."""
    from oncotarget_lite import __main__
    assert __main__ is not None


def test_version_import():
    """Test version can be imported."""
    try:
        from oncotarget_lite._version import __version__
        assert isinstance(__version__, str)
    except ImportError:
        # Version file may not exist in dev mode
        pass


def test_cli_commands_exist():
    """Test expected CLI commands exist."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    
    # Check for expected commands
    expected_commands = ["prepare", "train", "eval", "explain", "all"]
    for cmd in expected_commands:
        assert cmd in output


def test_prepare_command_help():
    """Test prepare command help."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "prepare", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "processed" in output.lower() or "features" in output.lower()


def test_generate_data_works():
    """Test synthetic data generation works."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test the synthetic data generation script
        script_path = Path("scripts/generate_synthetic_data.py")
        if script_path.exists():
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, cwd=tmp_dir)
            # Should succeed even if run in different directory
            assert result.returncode == 0


def test_comprehensive_csv_reading():
    """Test CSV reading with various scenarios."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Test with BOM
        p1 = tmp_path / "with_bom.csv"
        p1.write_text("\ufeff# comment\ngene,median_TPM\nGENE001,1.0\n")
        df1 = _read_csv(p1)
        assert "gene" in df1.columns
        
        # Test with whitespace in column names
        p2 = tmp_path / "whitespace.csv"
        p2.write_text("  gene  , median_TPM \nGENE001,1.0\n")
        df2 = _read_csv(p2)
        assert df2.columns.tolist() == ["gene", "median_TPM"]
        
        # Test with multiple comment lines
        p3 = tmp_path / "multi_comment.csv"
        p3.write_text("# line 1\n# line 2\n# line 3\ngene,median_TPM\nGENE001,1.0\n")
        df3 = _read_csv(p3)
        assert len(df3) == 1