"""Lean oncology target triage toolkit."""

from __future__ import annotations

from importlib import metadata

try:  # Prefer setuptools-scm generated version for in-tree use
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback when metadata missing
    try:
        __version__ = metadata.version("oncotarget-lite")
    except metadata.PackageNotFoundError:  # pragma: no cover - editable install w/out SCM
        __version__ = "0.0.0"

__all__ = ["__version__"]
