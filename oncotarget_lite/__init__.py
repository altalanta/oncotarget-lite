"""oncotarget-lite package with interpretability-first pipeline."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - during development the package may not be installed
    __version__ = version("oncotarget_lite")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

DEFAULT_RANDOM_STATE = 42
