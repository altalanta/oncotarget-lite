"""Comprehensive health check system for all application dependencies.

This module provides a unified health checking framework that aggregates
the status of all critical dependencies:
- ML model readiness
- Database connectivity
- Cache availability
- External service reachability
- Memory and resource constraints

Usage:
    from oncotarget_lite.health import HealthChecker, get_health_checker

    checker = get_health_checker()
    status = await checker.check_all()

    # Or check specific components
    model_status = await checker.check_model()
    db_status = await checker.check_database()
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

import psutil

from .logging_config import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class AggregatedHealth:
    """Aggregated health status across all components."""

    overall_status: HealthStatus
    components: list[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: str = field(default_factory=lambda: os.environ.get("SERVICE_VERSION", "unknown"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.overall_status.value,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "components": [c.to_dict() for c in self.components],
            "summary": {
                "healthy": sum(1 for c in self.components if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.components if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.components if c.status == HealthStatus.UNHEALTHY),
            },
        }


class HealthChecker:
    """Centralized health checker for all application dependencies."""

    def __init__(
        self,
        model_path: Path | None = None,
        db_path: Path | None = None,
        cache_dir: Path | None = None,
        memory_threshold_mb: int = 1024,
        disk_threshold_percent: int = 90,
    ):
        """
        Initialize the health checker.

        Args:
            model_path: Path to the ML model file
            db_path: Path to the monitoring database
            cache_dir: Path to the cache directory
            memory_threshold_mb: Memory usage threshold in MB
            disk_threshold_percent: Disk usage threshold percentage
        """
        self.model_path = model_path or Path("models/logreg_pipeline.pkl")
        self.db_path = db_path or Path("reports/monitoring.db")
        self.cache_dir = cache_dir or Path("data/cache")
        self.memory_threshold_mb = memory_threshold_mb
        self.disk_threshold_percent = disk_threshold_percent

        # Custom health check functions can be registered
        self._custom_checks: dict[str, Callable[[], Coroutine[Any, Any, ComponentHealth]]] = {}

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, ComponentHealth]],
    ) -> None:
        """Register a custom health check function."""
        self._custom_checks[name] = check_fn

    async def _timed_check(
        self,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, tuple[bool, str | None, dict[str, Any]]]],
    ) -> ComponentHealth:
        """Execute a health check with timing."""
        start = time.perf_counter()
        try:
            is_healthy, message, details = await check_fn()
            latency_ms = (time.perf_counter() - start) * 1000

            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                latency_ms=round(latency_ms, 2),
                message=message,
                details=details,
            )
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Health check failed for {name}", error=str(e))
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency_ms, 2),
                message=f"Check failed: {e}",
            )

    async def check_model(self) -> ComponentHealth:
        """Check ML model availability and readiness."""

        async def _check() -> tuple[bool, str | None, dict[str, Any]]:
            details: dict[str, Any] = {}

            if not self.model_path.exists():
                return False, f"Model file not found: {self.model_path}", details

            # Check file size and modification time
            stat = self.model_path.stat()
            details["file_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
            details["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # Try to load model metadata
            features_path = self.model_path.parent / "feature_list.json"
            if features_path.exists():
                import json

                with open(features_path) as f:
                    feature_info = json.load(f)
                    details["feature_count"] = len(feature_info.get("feature_order", []))

            return True, "Model is available", details

        return await self._timed_check("model", _check)

    async def check_database(self) -> ComponentHealth:
        """Check database connectivity and health."""

        async def _check() -> tuple[bool, str | None, dict[str, Any]]:
            details: dict[str, Any] = {}

            if not self.db_path.exists():
                # Database may not exist yet - that's OK for fresh installs
                return True, "Database not yet initialized", {"initialized": False}

            import sqlite3

            try:
                # Quick connectivity test
                conn = sqlite3.connect(str(self.db_path), timeout=5)
                cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master")
                table_count = cursor.fetchone()[0]
                conn.close()

                details["table_count"] = table_count
                details["file_size_mb"] = round(
                    self.db_path.stat().st_size / (1024 * 1024), 2
                )
                details["initialized"] = True

                return True, "Database is healthy", details

            except sqlite3.Error as e:
                return False, f"Database error: {e}", details

        return await self._timed_check("database", _check)

    async def check_cache(self) -> ComponentHealth:
        """Check cache directory health."""

        async def _check() -> tuple[bool, str | None, dict[str, Any]]:
            details: dict[str, Any] = {}

            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                return True, "Cache directory created", {"initialized": True}

            # Count cache files and total size
            cache_files = list(self.cache_dir.glob("**/*"))
            total_size = sum(f.stat().st_size for f in cache_files if f.is_file())

            details["file_count"] = len([f for f in cache_files if f.is_file()])
            details["total_size_mb"] = round(total_size / (1024 * 1024), 2)

            # Check if cache is writable
            test_file = self.cache_dir / ".health_check"
            try:
                test_file.write_text("health")
                test_file.unlink()
                details["writable"] = True
            except OSError:
                details["writable"] = False
                return False, "Cache directory is not writable", details

            return True, "Cache is healthy", details

        return await self._timed_check("cache", _check)

    async def check_memory(self) -> ComponentHealth:
        """Check system memory health."""

        async def _check() -> tuple[bool, str | None, dict[str, Any]]:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()

            details = {
                "system_total_mb": round(memory.total / (1024 * 1024), 2),
                "system_available_mb": round(memory.available / (1024 * 1024), 2),
                "system_percent_used": memory.percent,
                "process_rss_mb": round(process_memory.rss / (1024 * 1024), 2),
                "process_vms_mb": round(process_memory.vms / (1024 * 1024), 2),
            }

            process_mb = process_memory.rss / (1024 * 1024)

            if process_mb > self.memory_threshold_mb:
                return (
                    False,
                    f"Process memory ({process_mb:.0f}MB) exceeds threshold ({self.memory_threshold_mb}MB)",
                    details,
                )

            if memory.percent > 90:
                return False, f"System memory critically high ({memory.percent}%)", details

            if memory.percent > 80:
                return True, f"System memory elevated ({memory.percent}%)", details

            return True, "Memory usage is healthy", details

        return await self._timed_check("memory", _check)

    async def check_disk(self) -> ComponentHealth:
        """Check disk space health."""

        async def _check() -> tuple[bool, str | None, dict[str, Any]]:
            disk = psutil.disk_usage("/")

            details = {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": disk.percent,
            }

            if disk.percent > self.disk_threshold_percent:
                return (
                    False,
                    f"Disk usage ({disk.percent}%) exceeds threshold ({self.disk_threshold_percent}%)",
                    details,
                )

            return True, "Disk usage is healthy", details

        return await self._timed_check("disk", _check)

    async def check_all(self, timeout: float = 10.0) -> AggregatedHealth:
        """
        Run all health checks concurrently with timeout.

        Args:
            timeout: Maximum time to wait for all checks (seconds)

        Returns:
            Aggregated health status across all components
        """
        # Gather all standard checks
        standard_checks = [
            self.check_model(),
            self.check_database(),
            self.check_cache(),
            self.check_memory(),
            self.check_disk(),
        ]

        # Add custom checks
        custom_checks = [fn() for fn in self._custom_checks.values()]

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*standard_checks, *custom_checks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Health check timed out")
            return AggregatedHealth(
                overall_status=HealthStatus.UNKNOWN,
                components=[
                    ComponentHealth(
                        name="timeout",
                        status=HealthStatus.UNKNOWN,
                        message=f"Health check timed out after {timeout}s",
                    )
                ],
            )

        # Process results
        components: list[ComponentHealth] = []
        for result in results:
            if isinstance(result, Exception):
                components.append(
                    ComponentHealth(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {result}",
                    )
                )
            else:
                components.append(result)

        # Determine overall status
        unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)

        if unhealthy_count > 0:
            overall = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return AggregatedHealth(overall_status=overall, components=components)

    async def liveness_check(self) -> dict[str, Any]:
        """
        Quick liveness check for Kubernetes liveness probes.

        This should be fast and only check if the process is alive.
        """
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def readiness_check(self) -> dict[str, Any]:
        """
        Readiness check for Kubernetes readiness probes.

        Checks if the service is ready to accept traffic.
        """
        model_health = await self.check_model()

        if model_health.status == HealthStatus.HEALTHY:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "status": "not_ready",
                "reason": model_health.message,
                "timestamp": datetime.utcnow().isoformat(),
            }


# Singleton instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def reset_health_checker() -> None:
    """Reset the health checker (useful for testing)."""
    global _health_checker
    _health_checker = None


__all__ = [
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "AggregatedHealth",
    "get_health_checker",
    "reset_health_checker",
]




