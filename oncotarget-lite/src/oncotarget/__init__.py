"""oncotarget-lite package."""

from importlib import metadata

try:
    __version__ = metadata.version("oncotarget-lite")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
