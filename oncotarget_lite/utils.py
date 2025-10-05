"""Utility helpers for deterministic pipelines and IO."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


def _mlflow():
    """
    Lazy importer to avoid import-time failures in lightweight contexts (e.g., CLI --help, unit tests).
    Raise a clear message only when MLflow is actually needed.
    """
    try:
        import mlflow  # type: ignore
    except Exception as e:
        raise RuntimeError("MLflow is required for this operation but is not installed.") from e
    return mlflow


def git_commit() -> str:
    """Return the current git commit SHA (short), fallback to "unknown"."""

    try:
        import subprocess

        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return sha
    except Exception:  # pragma: no cover - git may be missing in CI
        return "unknown"

PYTHONHASHSEED = "PYTHONHASHSEED"


def ensure_dir(path: Path) -> Path:
    """Create a directory (recursively) if it is missing and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(path: Path) -> None:
    """Remove a directory tree if it exists (best-effort)."""

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int) -> None:
    """Deterministically seed Python, NumPy, and hash randomisation."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ[PYTHONHASHSEED] = "0"


@contextmanager
def timer(name: str) -> Iterator[None]:  # pragma: no cover - utility for logging
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[{name}] {elapsed:.2f}s")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_dataframe(path: Path, frame: pd.DataFrame, *, index: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=index)
    elif path.suffix == ".parquet":
        frame.to_parquet(path, index=index)
    else:
        raise ValueError(f"Unsupported extension for DataFrame export: {path.suffix}")


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported extension for DataFrame import: {path.suffix}")


def dataset_hash(features: pd.DataFrame, labels: pd.Series) -> str:
    """Compute a SHA256 fingerprint for features + labels ordering."""

    feature_bytes = features.to_numpy(dtype=np.float64).tobytes()
    label_bytes = labels.to_numpy(dtype=np.int8).tobytes()
    digest = sha256()
    digest.update(feature_bytes)
    digest.update(label_bytes)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
