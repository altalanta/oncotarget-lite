from pathlib import Path
import numpy as np
import pandas as pd

def main(out_dir: str = "data/raw"):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    genes = [f"GENE{i:04d}" for i in range(1000)]
    expr = pd.DataFrame({
        "gene": genes,
        "median_TPM": rng.lognormal(mean=1.5, sigma=0.5, size=len(genes)),
    })
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
    print(f"Wrote synthetic data to {p.resolve()}")

if __name__ == "__main__":
    main()