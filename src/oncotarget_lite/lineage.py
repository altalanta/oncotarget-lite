"""Lineage metadata helpers."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .logging import LogContext


def _safe_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:  # pragma: no cover - depends on environment
        return "unknown"


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "sha256": sha256(data).hexdigest(),
        "size": len(data),
    }


def build_lineage(
    inputs: Iterable[Path],
    params: dict[str, Any],
    context: LogContext,
) -> dict[str, Any]:
    """Create a lineage payload covering inputs, code, and runtime."""

    timestamp = datetime.now(timezone.utc).isoformat()
    input_records = [_file_record(path) for path in inputs if path.exists()]
    payload = {
        "run_id": context.run_id,
        "lineage_id": context.lineage_id,
        "timestamp": timestamp,
        "git_sha": _safe_git_sha(),
        "inputs": input_records,
        "params": params,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    return payload


def write_lineage(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
