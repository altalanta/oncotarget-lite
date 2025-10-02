"""Generate synthetic development data for oncotarget-lite pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd


def main(out_dir: str = "data/raw") -> None:
    """Generate synthetic data matching the expected schema."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    
    # Set deterministic seed for reproducible synthetic data
    rng = np.random.default_rng(0)
    
    # Generate 1000 synthetic genes
    genes = [f"GENE{i:04d}" for i in range(1000)]
    
    # GTEx data (normal tissue expression)
    tissues = ["Brain", "Heart", "Liver", "Lung", "Kidney"]
    gtex_data = []
    for gene in genes:
        for tissue in tissues:
            gtex_data.append({
                "gene": gene,
                "tissue": tissue,
                "median_TPM": rng.lognormal(mean=1.5, sigma=0.8)
            })
    
    gtex_df = pd.DataFrame(gtex_data)
    # Ensure no duplicates
    gtex_df = gtex_df.drop_duplicates(subset=["gene", "tissue"])
    header = "# GTEx synthetic data; source: generate_synthetic_data.py\n"
    with open(p / "GTEx_subset.csv", "w") as f:
        f.write(header)
    gtex_df.to_csv(p / "GTEx_subset.csv", mode="a", index=False)
    
    # TCGA data (tumor expression)
    tumors = ["BRCA", "LUAD", "COAD", "PRAD", "THCA"]
    tcga_data = []
    for gene in genes:
        for tumor in tumors:
            # Make tumor expression slightly higher on average
            tcga_data.append({
                "gene": gene,
                "tumor": tumor,
                "median_TPM": rng.lognormal(mean=1.8, sigma=0.8)
            })
    
    tcga_df = pd.DataFrame(tcga_data)
    # Ensure no duplicates
    tcga_df = tcga_df.drop_duplicates(subset=["gene", "tumor"])
    header = "# TCGA synthetic data; source: generate_synthetic_data.py\n"
    with open(p / "TCGA_subset.csv", "w") as f:
        f.write(header)
    tcga_df.to_csv(p / "TCGA_subset.csv", mode="a", index=False)
    
    # DepMap essentiality scores (aggregate by gene)
    depmap_data = []
    for gene in genes:
        # Generate dependency scores for multiple cell lines and aggregate
        cell_line_scores = [rng.normal(0, 0.5) for _ in range(4)]
        mean_score = np.mean(cell_line_scores)
        
        depmap_data.append({
            "gene": gene,
            "dependency_score": mean_score
        })
    
    depmap_df = pd.DataFrame(depmap_data)
    header = "# DepMap synthetic data; source: generate_synthetic_data.py\n"
    with open(p / "DepMap_essentials_subset.csv", "w") as f:
        f.write(header)
    depmap_df.to_csv(p / "DepMap_essentials_subset.csv", mode="a", index=False)
    
    # UniProt annotations
    annotations_data = []
    for i, gene in enumerate(genes):
        # Create realistic distribution of cell surface proteins (~10%)
        is_cell_surface = 1 if rng.random() < 0.1 else 0
        signal_peptide = 1 if rng.random() < 0.15 else 0
        ig_like_domain = 1 if rng.random() < 0.08 else 0
        protein_length = rng.integers(100, 3000)
        
        annotations_data.append({
            "gene": gene,
            "is_cell_surface": is_cell_surface,
            "signal_peptide": signal_peptide,
            "ig_like_domain": ig_like_domain,
            "protein_length": protein_length
        })
    
    annotations_df = pd.DataFrame(annotations_data)
    header = "# UniProt synthetic annotations; source: generate_synthetic_data.py\n"
    with open(p / "uniprot_annotations.csv", "w") as f:
        f.write(header)
    annotations_df.to_csv(p / "uniprot_annotations.csv", mode="a", index=False)
    
    # PPI degree data
    ppi_data = []
    for gene in genes:
        # Power law distribution for PPI degree
        degree = int(rng.pareto(1.2) * 5) + 1
        ppi_data.append({
            "gene": gene,
            "degree": degree
        })
    
    ppi_df = pd.DataFrame(ppi_data)
    header = "# PPI synthetic data; source: generate_synthetic_data.py\n"
    with open(p / "ppi_degree_subset.csv", "w") as f:
        f.write(header)
    ppi_df.to_csv(p / "ppi_degree_subset.csv", mode="a", index=False)
    
    print(f"Generated synthetic data in {p.resolve()}")
    print(f"Files created:")
    for file in p.glob("*.csv"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()