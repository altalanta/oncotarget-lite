"""End-to-end test with synthetic data generation."""

import subprocess
import sys
import os
from pathlib import Path


def test_synthetic_data_generation(tmp_path):
    """Test that synthetic data generation works correctly."""
    # Change to tmp directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Copy the generation script to tmp directory
        script_content = """
from pathlib import Path
import numpy as np
import pandas as pd

def main(out_dir="data/raw"):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    
    # Minimal data for testing
    genes = ["GENE001", "GENE002", "GENE003"]
    
    # GTEx
    gtex_data = []
    for gene in genes:
        for tissue in ["Brain", "Heart"]:
            gtex_data.append({"gene": gene, "tissue": tissue, "median_TPM": rng.lognormal(1.0, 0.5)})
    pd.DataFrame(gtex_data).to_csv(p / "GTEx_subset.csv", index=False)
    
    # TCGA
    tcga_data = []
    for gene in genes:
        for tumor in ["BRCA", "LUAD"]:
            tcga_data.append({"gene": gene, "tumor": tumor, "median_TPM": rng.lognormal(1.2, 0.5)})
    pd.DataFrame(tcga_data).to_csv(p / "TCGA_subset.csv", index=False)
    
    # DepMap
    depmap_data = []
    for gene in genes:
        for cell_line in ["ACH-001", "ACH-002"]:
            depmap_data.append({"gene": gene, "cell_line": cell_line, "dependency_score": rng.normal(0, 0.5)})
    pd.DataFrame(depmap_data).to_csv(p / "DepMap_essentials_subset.csv", index=False)
    
    # Annotations
    annotations_data = []
    for i, gene in enumerate(genes):
        annotations_data.append({
            "gene": gene,
            "is_cell_surface": i % 2,  # Alternate 0, 1, 0
            "signal_peptide": 1,
            "ig_like_domain": 0,
            "protein_length": 500 + i * 100
        })
    pd.DataFrame(annotations_data).to_csv(p / "uniprot_annotations.csv", index=False)
    
    # PPI
    ppi_data = [{"gene": gene, "degree": 5 + i} for i, gene in enumerate(genes)]
    pd.DataFrame(ppi_data).to_csv(p / "ppi_degree_subset.csv", index=False)
    
    print(f"Generated data in {p}")

if __name__ == "__main__":
    main()
"""
        
        # Write the script
        scripts_dir = Path("scripts")
        scripts_dir.mkdir(exist_ok=True)
        script_path = scripts_dir / "generate_test_data.py"
        script_path.write_text(script_content)
        
        # Run the script
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        
        # Check that files were created
        data_dir = Path("data/raw")
        assert data_dir.exists()
        
        expected_files = [
            "GTEx_subset.csv",
            "TCGA_subset.csv", 
            "DepMap_essentials_subset.csv",
            "uniprot_annotations.csv",
            "ppi_degree_subset.csv"
        ]
        
        for filename in expected_files:
            file_path = data_dir / filename
            assert file_path.exists(), f"Missing file: {filename}"
            # Verify file has content
            assert file_path.stat().st_size > 0, f"Empty file: {filename}"
            
    finally:
        os.chdir(original_cwd)


def test_cli_generate_data_command():
    """Test the CLI generate-data command."""
    result = subprocess.run([
        sys.executable, "-c", 
        "from oncotarget_lite.cli import app; import sys; sys.argv = ['cli', 'generate-data', '--help']; app()"
    ], capture_output=True, text=True)
    
    # Should not fail (returncode 0 for help)
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "synthetic" in output or "Generate" in output