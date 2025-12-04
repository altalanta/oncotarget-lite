"""oncotarget-lite package with interpretability-first pipeline."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - during development the package may not be installed
    __version__ = version("oncotarget_lite")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

DEFAULT_RANDOM_STATE = 42

# Logging utilities - import these for consistent logging across the package
from oncotarget_lite.logging_config import (
    configure_logging,
    get_logger,
    log_timing,
    LogContext,
)

# Settings - centralized configuration
from oncotarget_lite.settings import (
    get_settings,
    Settings,
)

# Health checks
from oncotarget_lite.health import (
    get_health_checker,
    HealthChecker,
    HealthStatus,
)

# Timeouts
from oncotarget_lite.timeouts import (
    with_timeout,
    timeout_context,
    TimeoutError,
)

__all__ = [
    "__version__",
    "DEFAULT_RANDOM_STATE",
    # Logging
    "configure_logging",
    "get_logger",
    "log_timing",
    "LogContext",
    # Settings
    "get_settings",
    "Settings",
    # Health
    "get_health_checker",
    "HealthChecker",
    "HealthStatus",
    # Timeouts
    "with_timeout",
    "timeout_context",
    "TimeoutError",
]
