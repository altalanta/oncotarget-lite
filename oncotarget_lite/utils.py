"""Utility functions for data processing and reproducibility."""

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of dataset for lineage tracking."""
    # Convert to string representation and encode
    data_str = df.to_string(index=False).encode('utf-8')
    return hashlib.sha256(data_str).hexdigest()[:16]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    """Save data as JSON with proper formatting."""
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load JSON data."""
    import json
    with open(path, 'r') as f:
        return json.load(f)