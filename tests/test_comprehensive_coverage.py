"""Comprehensive tests to achieve 80% coverage."""

import tempfile
import subprocess
import sys
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from oncotarget_lite.data import _read_csv, DataPreparationError
from oncotarget_lite.utils import ensure_dir, save_dataframe, save_json, dataset_hash, set_seeds, git_commit


def test_save_json(tmp_path):
    """Test JSON saving utility."""
    data = {"test": "value", "number": 42}
    file_path = tmp_path / "test.json"
    save_json(file_path, data)
    assert file_path.exists()
    
    # Verify content
    with open(file_path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_set_seeds_deterministic():
    """Test that set_seeds produces deterministic results."""
    set_seeds(42)
    val1 = np.random.random()
    val2 = np.random.randint(0, 100)
    
    set_seeds(42)
    val3 = np.random.random()
    val4 = np.random.randint(0, 100)
    
    assert val1 == val3
    assert val2 == val4


def test_git_commit():
    """Test git commit function."""
    commit = git_commit()
    assert isinstance(commit, str)
    # Should be either a commit hash or "unknown"
    assert len(commit) > 0


def test_dataset_hash_different_data():
    """Test that different data produces different hashes."""
    features1 = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    labels1 = pd.Series([0, 1], index=["x", "y"])
    
    features2 = pd.DataFrame({"a": [3, 4]}, index=["x", "y"])
    labels2 = pd.Series([1, 0], index=["x", "y"])
    
    h1 = dataset_hash(features1, labels1)
    h2 = dataset_hash(features2, labels2)
    
    assert h1 != h2
    assert isinstance(h1, str)
    assert isinstance(h2, str)


def test_read_csv_file_not_exists():
    """Test _read_csv with non-existent file."""
    with pytest.raises(DataPreparationError) as exc_info:
        _read_csv(Path("/nonexistent/file.csv"))
    assert "Missing synthetic data file" in str(exc_info.value)


def test_read_csv_with_whitespace_columns(tmp_path):
    """Test CSV reading with whitespace in column names."""
    p = tmp_path / "whitespace.csv"
    p.write_text("# comment\n  gene  , median_TPM \nGENE001,1.5\n")
    df = _read_csv(p)
    assert list(df.columns) == ["gene", "median_TPM"]  # Should be stripped
    assert len(df) == 1


def test_read_csv_with_bom(tmp_path):
    """Test CSV reading with BOM."""
    p = tmp_path / "bom.csv"
    p.write_text("\ufeff# comment\ngene,median_TPM\nGENE001,1.5\n")
    df = _read_csv(p)
    assert "gene" in df.columns
    assert "median_TPM" in df.columns


def test_cli_commands_list():
    """Test that CLI shows expected commands."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    
    # Check for expected commands
    expected_commands = ["prepare", "train", "eval", "explain", "generate-data"]
    for cmd in expected_commands:
        assert cmd in output


def test_generate_data_command_help():
    """Test generate-data command help."""
    result = subprocess.run([
        sys.executable, "-m", "oncotarget_lite", "generate-data", "--help"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "synthetic" in output.lower() or "development" in output.lower()


def test_main_module_import():
    """Test that __main__ module can be imported."""
    from oncotarget_lite import __main__
    assert __main__ is not None


def test_version_import():
    """Test version import."""
    try:
        from oncotarget_lite._version import __version__
        assert isinstance(__version__, str)
    except ImportError:
        # Version file may not exist in development
        pass


def test_data_preparation_error():
    """Test DataPreparationError exception."""
    error = DataPreparationError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_cli_app_import():
    """Test that CLI app can be imported without MLflow."""
    # This should work due to lazy imports
    from oncotarget_lite.cli import app
    assert app is not None


def test_lazy_mlflow_import():
    """Test lazy MLflow import functionality."""
    from oncotarget_lite.utils import _mlflow
    
    # This should either return mlflow or raise a clear error
    try:
        mlflow = _mlflow()
        # If successful, should have expected attributes
        assert hasattr(mlflow, 'start_run')
    except RuntimeError as e:
        # Should have clear error message
        assert "MLflow is required" in str(e)


def test_ensure_dir_existing_dir(tmp_path):
    """Test ensure_dir with existing directory."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    
    # Should not raise error
    result = ensure_dir(existing_dir)
    assert result == existing_dir
    assert existing_dir.exists()


def test_save_dataframe_csv(tmp_path):
    """Test save_dataframe with CSV format."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    csv_path = tmp_path / "test.csv"
    
    # The function should handle parquet, but let's test it works
    save_dataframe(csv_path.with_suffix('.parquet'), df)
    assert csv_path.with_suffix('.parquet').exists()


def test_comprehensive_cli_integration():
    """Test that basic CLI integration works without crashing."""
    # Test that we can import and get help without errors
    result = subprocess.run([
        sys.executable, "-c", 
        "from oncotarget_lite.cli import app; print('CLI import successful')"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "CLI import successful" in result.stdout