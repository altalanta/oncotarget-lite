from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROWS = 1000
FAST_ROWS = 64


def _infer_row_count(explicit_rows: int | None = None) -> int:
    if explicit_rows is not None:
        return explicit_rows

    profile = os.getenv("ONCOTARGET_LITE_PROFILE") or ""
    if profile.lower() == "ci":
        return FAST_ROWS

    fast_flag = os.getenv("ONCOTARGET_LITE_FAST", "0")
    if fast_flag.lower() in {"1", "true", "yes"}:
        return FAST_ROWS

    return DEFAULT_ROWS


def main(out_dir: str = "data/raw", rows: int | None = None) -> None:
    row_count = _infer_row_count(rows)

    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    genes = [f"GENE{i:04d}" for i in range(row_count)]
    expr = pd.DataFrame(
        {
            "gene": genes,
            "median_TPM": rng.lognormal(mean=1.5, sigma=0.5, size=len(genes)),
        }
    )
    # Prepend a comment header to simulate real-world metadata
    header = "# source: synthetic; created by generate_synthetic_data.py\n"
    (p / "expression.csv").write_text(header)
    expr.to_csv(p / "expression.csv", mode="a", index=False)
    # Minimal stubs for other raw inputs if your pipeline expects them
    pd.DataFrame({"gene": genes, "dep_score": rng.normal(0, 1, len(genes))}).to_csv(
        p / "dependencies.csv", index=False
    )
    pd.DataFrame({"gene": genes, "is_oncogene": rng.integers(0, 2, len(genes))}).to_csv(
        p / "annotations.csv", index=False
    )
    print(f"Wrote synthetic data ({row_count} rows) to {p.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic development data")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory for generated data")
    parser.add_argument("--rows", type=int, default=None, help="Number of genes to synthesise")
    args = parser.parse_args()
    main(out_dir=args.out_dir, rows=args.rows)
