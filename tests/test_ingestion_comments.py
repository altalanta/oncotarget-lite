from pathlib import Path
from oncotarget_lite.data import _read_csv, DataPreparationError
import pandas as pd
import pytest

def test_read_csv_ignores_comment_header(tmp_path: Path):
    p = tmp_path / "x.csv"
    p.write_text("# a header line\n# another header\n" "gene,median_TPM\nA,1.0\n")
    df = _read_csv(p)
    assert list(df.columns) == ["gene", "median_TPM"]
    assert df.shape == (1, 2)

def test_read_csv_missing_required_columns(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("# meta\nid,value\n1,2\n")
    with pytest.raises(DataPreparationError) as e:
        _read_csv(p)
    assert "missing required columns" in str(e.value)