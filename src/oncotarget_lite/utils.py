"""General utilities for oncotarget-lite."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - cuda optional
        torch.cuda.manual_seed_all(seed)


def select_device(preference: Literal["auto", "cpu", "cuda"]) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():  # pragma: no cover - cuda optional
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

