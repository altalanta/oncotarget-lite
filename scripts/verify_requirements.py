#!/usr/bin/env python3
"""
Requirements hash verification script.

This script verifies that the requirements.txt file has not been tampered with
by comparing its current hash against a known good hash stored in the repository.
"""

import hashlib
import sys
from pathlib import Path


def get_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Get hash of a file."""
    hash_func = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def main():
    """Main verification function."""
    requirements_file = Path("requirements.txt")

    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        sys.exit(1)

    current_hash = get_file_hash(requirements_file)

    # Known good hash for requirements.txt - UPDATE THIS when requirements are legitimately changed
    # This hash should be committed to the repository and updated only when requirements.txt changes
    EXPECTED_HASH = "a3a3d761dffb9712fbd7c3cd1856dc8e65ff35b6728fb549a6a3b2c83630032a"  # Current hash

    print(f"📋 Current requirements.txt hash: {current_hash}")
    print(f"📋 Expected hash: {EXPECTED_HASH}")

    if current_hash == EXPECTED_HASH:
        print("✅ Requirements file hash verification passed")
        return True
    else:
        print("❌ Requirements file hash verification failed!")
        print("❌ This indicates requirements.txt may have been tampered with or corrupted.")
        print("💡 If you intentionally modified requirements.txt, update the EXPECTED_HASH in this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()
