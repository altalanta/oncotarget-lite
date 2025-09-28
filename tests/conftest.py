from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from oncotarget_lite.utils import ensure_dir


@pytest.fixture()
def synthetic_reports(tmp_path: Path) -> Generator[Path, None, None]:
    reports_dir = tmp_path / "reports"
    ensure_dir(reports_dir)

    rng = np.random.default_rng(42)

    def make_block(prefix: str, n: int) -> pd.DataFrame:
        labels = np.array([0, 1] * (n // 2))
        rng.shuffle(labels)
        probs = np.where(labels == 1, rng.normal(0.8, 0.05, size=n), rng.normal(0.2, 0.05, size=n))
        probs = np.clip(probs, 0.01, 0.99)
        genes = [f"{prefix}_{i:03d}" for i in range(n)]
        return pd.DataFrame({
            "gene": genes,
            "split": prefix,
            "y_prob": probs,
            "y_true": labels,
        })

    train = make_block("train", 40)
    test = make_block("test", 20)
    preds = pd.concat([train, test], ignore_index=True)
    preds.to_parquet(reports_dir / "predictions.parquet", index=False)

    yield reports_dir
