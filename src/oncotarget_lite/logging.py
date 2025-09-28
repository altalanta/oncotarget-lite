"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

try:
    import numpy
    import pandas
    import torch
except Exception:  # pragma: no cover - optional imports
    numpy = pandas = torch = None


@dataclass(frozen=True)
class LogContext:
    """Metadata propagated through log records."""

    run_id: str
    lineage_id: str


class JsonFormatter(logging.Formatter):
    """Render log records as structured JSON."""

    def __init__(self, context: LogContext) -> None:
        super().__init__()
        self._context = context

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting only
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": self._context.run_id,
            "lineage_id": self._context.lineage_id,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(level: str, context: LogContext, json_logs: bool = True) -> None:
    """Configure root logging handlers."""

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    if json_logs:
        handler.setFormatter(JsonFormatter(context))
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    # Avoid noisy third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def log_environment(logger: logging.Logger) -> None:
    """Emit environment diagnostics once per run."""

    logger.info(
        "runtime-environment",
        extra={
            "extra_fields": {
                "python": sys.version.split()[0],
                "pid": os.getpid(),
                "numpy": getattr(numpy, "__version__", "unknown"),
                "pandas": getattr(pandas, "__version__", "unknown"),
                "torch": getattr(torch, "__version__", "unknown"),
            }
        },
    )
