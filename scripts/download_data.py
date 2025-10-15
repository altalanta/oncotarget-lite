"""Fetch or generate datasets declared in the manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable

from .fetch_data import compute_file_hash, generate_manifest, verify_manifest
from .generate_synthetic_data import FAST_ROWS, DEFAULT_ROWS, main as generate_synthetic


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def _needs_refresh(manifest_entries: Iterable[Dict[str, str]], data_root: Path) -> bool:
    for entry in manifest_entries:
        file_path = data_root / entry["path"]
        if not file_path.exists():
            return True
        expected_hash = entry.get("sha256")
        if expected_hash and compute_file_hash(file_path) != expected_hash:
            return True
    return False


def _ensure_synthetic_data(data_root: Path, fast: bool) -> None:
    rows = FAST_ROWS if fast else DEFAULT_ROWS
    os.environ["ONCOTARGET_LITE_FAST"] = "1" if fast else "0"
    generate_synthetic(out_dir=str(data_root / "raw"), rows=rows)


def ensure_data(manifest_path: Path, *, fast: bool = False) -> bool:
    data_root = manifest_path.parent
    manifest_exists = manifest_path.exists()

    manifest = {}
    if manifest_exists:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = manifest.get("files", []) if manifest else []
    if not manifest_exists or _needs_refresh(files, data_root):
        synthetic_entries = [entry for entry in files if entry.get("source") == "synthetic"]
        if synthetic_entries or not files:
            _ensure_synthetic_data(data_root, fast)
        else:
            for entry in files:
                url = entry.get("url")
                if not url:
                    continue
                try:
                    _download(url, data_root / entry["path"])
                except (urllib.error.URLError, urllib.error.HTTPError) as exc:  # pragma: no cover - network
                    raise RuntimeError(f"Failed to download {url}: {exc}") from exc

        # Rebuild manifest after fetching or generating data
        generate_manifest(data_dir=data_root, output_file=manifest_path)

    return verify_manifest(manifest_file=manifest_path, data_dir=data_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure the data manifest and artefacts are present")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.json"),
        help="Path to the manifest file",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that the manifest files exist and match hashes",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Generate a compact dataset suitable for CI smoke tests",
    )
    args = parser.parse_args()

    if args.verify_only:
        success = verify_manifest(manifest_file=args.manifest, data_dir=args.manifest.parent)
        return 0 if success else 1

    try:
        success = ensure_data(args.manifest, fast=args.fast)
    except Exception as exc:  # pragma: no cover - fatal path
        print(f"❌ Failed to prepare data: {exc}", file=sys.stderr)
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
