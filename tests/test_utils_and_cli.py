import subprocess, sys, os
from pathlib import Path
import pytest
from oncotarget_lite.utils import ensure_dir, save_dataframe, dataset_hash, set_seeds
import pandas as pd

def test_ensure_dir_and_save(tmp_path: Path):
    p = tmp_path / "x/y"
    ensure_dir(p)
    assert p.exists()
    df = pd.DataFrame({"a":[1,2]})
    out = tmp_path / "t.parquet"
    save_dataframe(out, df)  # Correct argument order
    assert out.exists()

def test_dataset_hash_stable(tmp_path: Path):
    features = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    labels = pd.Series([0, 1], index=["x", "y"])
    h1 = dataset_hash(features, labels)
    h2 = dataset_hash(features, labels)
    assert h1 == h2 and len(h1) >= 8

def test_main_module_help():
    out = subprocess.run([sys.executable, "-m", "oncotarget_lite", "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "oncotarget-lite" in (out.stdout + out.stderr) or "Usage" in (out.stdout + out.stderr)

def test_cli_generate_data(tmp_path: Path, monkeypatch):
    # run the new command into a temp dir
    out_dir = tmp_path / "raw"
    out = subprocess.run([sys.executable, "-m", "oncotarget_lite", "generate-data", "--out-dir", str(out_dir)],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert (out_dir / "expression.csv").exists()