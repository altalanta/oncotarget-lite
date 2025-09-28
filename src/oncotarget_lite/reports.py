"""Reporting utilities used by the CLI and documentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class ReportError(RuntimeError):
    """Raised when expected artifacts are missing or invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"expected metrics file not found: {path}"
        raise ReportError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        msg = f"expected scores file not found: {path}"
        raise ReportError(msg)
    return pd.read_parquet(path)


def summarise_run(run_dir: Path, top_k: int = 5) -> dict[str, Any]:
    """Generate a lightweight summary for CLI reporting."""

    metrics = _load_json(run_dir / "metrics.json")
    scores = _load_scores(run_dir / "scores.parquet")
    top = (
        scores.sort_values("score", ascending=False)
        .head(top_k)
        .rename(columns={"rank": "scorecard_rank"})
        .to_dict(orient="records")
    )
    return {
        "artifacts": str(run_dir),
        "test_metrics": metrics.get("test", {}),
        "bootstrap": metrics.get("bootstrap", {}),
        "top_targets": top,
    }

