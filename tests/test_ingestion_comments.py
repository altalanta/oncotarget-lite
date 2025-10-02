"""Tests for CSV ingestion with comment headers."""

from pathlib import Path
import pytest

from oncotarget_lite.data import _read_csv, DataPreparationError


def test_read_csv_ignores_comment_header(tmp_path: Path):
    """Test that _read_csv properly ignores comment headers starting with #."""
    p = tmp_path / "test.csv"
    p.write_text("# This is a comment header\n# Another comment\ngene,median_TPM\nGENE001,1.5\nGENE002,2.3\n")
    
    df = _read_csv(p)
    assert list(df.columns) == ["gene", "median_TPM"]
    assert df.shape == (2, 2)
    assert df.iloc[0]["gene"] == "GENE001"
    assert df.iloc[0]["median_TPM"] == 1.5


def test_read_csv_strips_whitespace_in_columns(tmp_path: Path):
    """Test that column names are properly stripped of whitespace."""
    p = tmp_path / "test.csv"
    p.write_text("gene , median_TPM \nGENE001,1.5\n")
    
    df = _read_csv(p)
    assert list(df.columns) == ["gene", "median_TPM"]


def test_read_csv_missing_required_columns_gtex(tmp_path: Path):
    """Test that missing required columns raises appropriate error for GTEx files."""
    p = tmp_path / "GTEx_test.csv"
    p.write_text("# Comment header\nid,value\n1,2\n")
    
    with pytest.raises(DataPreparationError) as exc_info:
        _read_csv(p)
    
    error_msg = str(exc_info.value)
    assert "missing required columns" in error_msg
    assert "median_TPM" in error_msg
    assert "Ensure comment headers start with '#'" in error_msg


def test_read_csv_missing_required_columns_tcga(tmp_path: Path):
    """Test that missing required columns raises appropriate error for TCGA files."""
    p = tmp_path / "TCGA_test.csv"
    p.write_text("# Comment header\nid,value\n1,2\n")
    
    with pytest.raises(DataPreparationError) as exc_info:
        _read_csv(p)
    
    error_msg = str(exc_info.value)
    assert "missing required columns" in error_msg
    assert "median_TPM" in error_msg


def test_read_csv_missing_file():
    """Test that missing file raises appropriate error."""
    p = Path("/nonexistent/file.csv")
    
    with pytest.raises(DataPreparationError) as exc_info:
        _read_csv(p)
    
    assert "Missing synthetic data file" in str(exc_info.value)


def test_read_csv_non_expression_file_only_needs_gene(tmp_path: Path):
    """Test that non-expression files only need 'gene' column."""
    p = tmp_path / "annotations.csv"
    p.write_text("# Comment\ngene,is_cell_surface\nGENE001,1\n")
    
    df = _read_csv(p)
    assert "gene" in df.columns
    assert "is_cell_surface" in df.columns