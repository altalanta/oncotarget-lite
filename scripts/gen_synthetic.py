"""Generate synthetic data with manifest support."""

from pathlib import Path
from scripts.generate_synthetic_data import main as generate_data
from scripts.fetch_data import generate_manifest


def main(output_dir: str = "data/raw"):
    """Generate synthetic data and create manifest."""
    # Generate the synthetic data
    generate_data(output_dir)
    
    # Generate manifest for all data
    data_dir = Path(output_dir).parent
    manifest = generate_manifest(data_dir)
    
    print(f"Generated synthetic data in {output_dir}")
    print(f"Created manifest with {len(manifest['files'])} files")


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    main(output_dir)