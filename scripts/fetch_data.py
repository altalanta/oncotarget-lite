"""Data fetching and manifest generation for reproducible data pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
import urllib.request
import urllib.error

from oncotarget_lite.utils import ensure_dir


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_file(url: str, output_path: Path, expected_hash: str | None = None) -> bool:
    """Download file from URL and verify hash if provided."""
    try:
        ensure_dir(output_path.parent)
        urllib.request.urlretrieve(url, output_path)
        
        if expected_hash:
            actual_hash = compute_file_hash(output_path)
            if actual_hash != expected_hash:
                output_path.unlink()  # Remove corrupted file
                raise ValueError(f"Hash mismatch for {output_path.name}: expected {expected_hash}, got {actual_hash}")
        
        return True
    except (urllib.error.URLError, ValueError) as e:
        print(f"Failed to download {url}: {e}")
        return False


def generate_manifest(data_dir: Path = Path("data"), output_file: Path = Path("data/manifest.json")) -> Dict[str, Any]:
    """Generate data manifest with file hashes and metadata."""
    
    manifest = {
        "version": "1.0",
        "generated_by": "oncotarget-lite",
        "description": "Data manifest for reproducible oncotarget-lite pipeline",
        "sources": {
            "synthetic": {
                "description": "Synthetic datasets generated for development",
                "generator": "scripts/generate_synthetic_data.py",
                "reproducible": True
            }
        },
        "files": []
    }
    
    # Process raw data files
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        for file_path in raw_dir.rglob("*.csv"):
            if file_path.is_file():
                relative_path = file_path.relative_to(data_dir)
                file_hash = compute_file_hash(file_path)
                file_size = file_path.stat().st_size
                
                file_info = {
                    "path": str(relative_path),
                    "type": "csv",
                    "source": "synthetic",
                    "sha256": file_hash,
                    "size_bytes": file_size,
                    "description": f"Synthetic {file_path.stem} data"
                }
                manifest["files"].append(file_info)
    
    # Process processed data files
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        for file_path in processed_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".parquet", ".json"]:
                relative_path = file_path.relative_to(data_dir)
                file_hash = compute_file_hash(file_path)
                file_size = file_path.stat().st_size
                
                file_info = {
                    "path": str(relative_path),
                    "type": file_path.suffix[1:],  # Remove the dot
                    "source": "processed",
                    "sha256": file_hash,
                    "size_bytes": file_size,
                    "description": f"Processed {file_path.stem} data"
                }
                manifest["files"].append(file_info)
    
    # Ensure output directory exists
    ensure_dir(output_file.parent)
    
    # Write manifest
    with open(output_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def verify_manifest(manifest_file: Path = Path("data/manifest.json"), data_dir: Path = Path("data")) -> bool:
    """Verify that all files in manifest exist and have correct hashes."""
    if not manifest_file.exists():
        print(f"Manifest file not found: {manifest_file}")
        return False
    
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    success = True
    for file_info in manifest.get("files", []):
        file_path = data_dir / file_info["path"]
        expected_hash = file_info["sha256"]
        
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            success = False
            continue
        
        actual_hash = compute_file_hash(file_path)
        if actual_hash != expected_hash:
            print(f"Hash mismatch for {file_path}: expected {expected_hash}, got {actual_hash}")
            success = False
    
    return success


def fetch_external_data(config_file: Path = Path("configs/data_sources.yaml")) -> bool:
    """Fetch external data sources if configured."""
    # This would fetch from external URLs in a real implementation
    # For now, we rely on synthetic data generation
    print("Using synthetic data generation (no external sources configured)")
    return True


def main():
    """Main function to fetch data and generate manifest."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch data and generate manifest")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing manifest")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate synthetic data first")
    
    args = parser.parse_args()
    
    if args.verify_only:
        success = verify_manifest(args.data_dir / "manifest.json", args.data_dir)
        exit(0 if success else 1)
    
    if args.regenerate:
        # Generate synthetic data first
        from generate_synthetic_data import main as gen_data
        gen_data(str(args.data_dir / "raw"))
    
    # Generate manifest
    manifest = generate_manifest(args.data_dir)
    print(f"Generated manifest with {len(manifest['files'])} files")
    
    # Verify the manifest we just created
    success = verify_manifest(args.data_dir / "manifest.json", args.data_dir)
    if success:
        print("✓ All files verified successfully")
    else:
        print("✗ Verification failed")
        exit(1)


if __name__ == "__main__":
    main()