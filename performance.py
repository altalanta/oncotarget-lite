"""Performance monitoring and optimization utilities for oncotarget-lite."""

from __future__ import annotations

import asyncio
import functools
import gc
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from memory_profiler import profile as memory_profile
except ImportError:
    # Define a no-op profile decorator if memory_profiler is not available
    def memory_profile(func):
        """No-op decorator when memory_profiler is not available."""
        return func

from .exceptions import PerformanceError
from .utils import ensure_dir

logger = logging.getLogger(__name__)

# Global performance monitoring state
_performance_monitor = None
_monitoring_enabled = False


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # CPU percentage
    available_mb: float  # Available system memory
    total_mb: float  # Total system memory


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    operation_name: str
    duration_seconds: float
    memory_peak_mb: float
    memory_increase_mb: float
    cpu_percent: float
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    start_memory: MemorySnapshot = field(default_factory=MemorySnapshot)
    end_memory: MemorySnapshot = field(default_factory=MemorySnapshot)


class PerformanceMonitor:
    """System-wide performance monitoring and optimization."""

    def __init__(self, enable_monitoring: bool = True, log_interval: int = 30):
        self.enable_monitoring = enable_monitoring
        self.log_interval = log_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.memory_threshold_mb = 8 * 1024  # 8GB default threshold
        self._stop_monitoring = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if not self.enable_monitoring:
            return

        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Performance monitoring already running")
            return

        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._background_monitor,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Started performance monitoring")

    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        if not self.enable_monitoring or not self._monitor_thread:
            return

        self._stop_monitoring.set()
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        logger.info("Stopped performance monitoring")

    def _background_monitor(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                self._check_memory_usage()
                self._stop_monitoring.wait(self.log_interval)
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                break

    def _check_memory_usage(self) -> None:
        """Check current memory usage and log warnings if needed."""
        if not PSUTIL_AVAILABLE:
            return

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        if memory_info.rss > self.memory_threshold_mb * 1024 * 1024:
            logger.warning(
                "High memory usage detected: "
                f"{memory_info.rss / 1024 / 1024:.1f}MB "
                f"({memory_percent:.1f}%)"
            )

    def get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot."""
        if not PSUTIL_AVAILABLE:
            # Return dummy values when psutil is not available
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=0.0,
                vms_mb=0.0,
                percent=0.0,
                available_mb=0.0,
                total_mb=0.0
            )

        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()

        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=process.cpu_percent(),
            available_mb=system_memory.available / 1024 / 1024,
            total_mb=system_memory.total / 1024 / 1024
        )

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        self.metrics_history.append(metrics)

        # Keep only last 1000 metrics to prevent memory growth
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        logger.info(
            f"Performance: {metrics.operation_name} - "
            f"{metrics.duration_seconds:.2f}s, "
            f"Memory: {metrics.memory_peak_mb:.1f}MB "
            f"(+{metrics.memory_increase_mb:+.1f}MB)"
        )

    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on performance history."""
        suggestions = []

        if not self.metrics_history:
            return suggestions

        # Analyze memory usage patterns
        recent_metrics = self.metrics_history[-10:]  # Last 10 operations

        avg_memory_increase = sum(m.memory_increase_mb for m in recent_metrics) / len(recent_metrics)
        max_memory_peak = max(m.memory_peak_mb for m in recent_metrics)

        if avg_memory_increase > 500:  # More than 500MB average increase
            suggestions.append(
                "High memory consumption detected. Consider using streaming processing "
                "or increasing available memory."
            )

        if max_memory_peak > self.memory_threshold_mb * 0.8:
            suggestions.append(
                "Memory usage approaching threshold. Consider enabling garbage collection "
                "or using memory-efficient data structures."
            )

        # Check for slow operations
        slow_operations = [m for m in recent_metrics if m.duration_seconds > 60]
        if slow_operations:
            suggestions.append(
                f"Found {len(slow_operations)} slow operations (>60s). "
                "Consider parallel processing or optimization."
            )

        return suggestions

    def clear_metrics(self) -> None:
        """Clear performance metrics history."""
        self.metrics_history.clear()


@contextmanager
def performance_monitor(operation_name: str, monitor: Optional[PerformanceMonitor] = None):
    """Context manager for performance monitoring."""
    if monitor is None:
        monitor = get_performance_monitor()

    start_memory = monitor.get_memory_snapshot()
    start_time = time.time()

    try:
        yield monitor
    finally:
        end_memory = monitor.get_memory_snapshot()
        end_time = time.time()

        metrics = PerformanceMetrics(
            operation_name=operation_name,
            duration_seconds=end_time - start_time,
            memory_peak_mb=end_memory.rss_mb,
            memory_increase_mb=end_memory.rss_mb - start_memory.rss_mb,
            cpu_percent=end_memory.percent,
            start_memory=start_memory,
            end_memory=end_memory
        )

        monitor.record_metrics(metrics)


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def enable_performance_monitoring(log_interval: int = 30) -> None:
    """Enable global performance monitoring."""
    global _monitoring_enabled
    if not _monitoring_enabled:
        monitor = get_performance_monitor()
        monitor.log_interval = log_interval
        monitor.start_monitoring()
        _monitoring_enabled = True
        logger.info("Performance monitoring enabled")


def disable_performance_monitoring() -> None:
    """Disable global performance monitoring."""
    global _monitoring_enabled
    if _monitoring_enabled:
        monitor = get_performance_monitor()
        monitor.stop_monitoring()
        _monitoring_enabled = False
        logger.info("Performance monitoring disabled")


def force_garbage_collection() -> Dict[str, Any]:
    """Force garbage collection and return memory stats."""
    import gc

    collected_before = gc.get_stats()

    # Force garbage collection
    gc.collect()

    collected_after = gc.get_stats()

    # Get memory info if psutil is available
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        memory_info = process.memory_info()
        current_memory_mb = memory_info.rss / 1024 / 1024
    else:
        current_memory_mb = 0.0

    return {
        "memory_freed_mb": (collected_before[-1]['collected'] - collected_after[-1]['collected']) * 28,  # Rough estimate
        "current_memory_mb": current_memory_mb,
        "objects_collected": collected_after[-1]['collected'] - collected_before[-1]['collected']
    }


def memory_optimized_function(func: Callable) -> Callable:
    """Decorator for memory optimization."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Force GC before function execution
        gc.collect()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Force GC after function execution
            gc.collect()

    return wrapper


def lazy_import(module_name: str, func_name: str = None):
    """Lazy import decorator for heavy modules."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if func_name:
                module = __import__(module_name, fromlist=[func_name])
                imported_func = getattr(module, func_name)
            else:
                module = __import__(module_name, fromlist=[''])
                imported_func = module

            # Replace the lazy import with the actual import
            globals()[func.__name__] = imported_func
            return imported_func(*args, **kwargs)
        return wrapper
    return decorator


# Optimized data structures for memory efficiency
class MemoryEfficientDict(dict):
    """Memory-efficient dictionary with automatic cleanup."""

    def __init__(self, max_size: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size
        self._access_order = []

    def __setitem__(self, key, value):
        if key not in self:
            self._access_order.append(key)

        super().__setitem__(key, value)

        # Cleanup if over max size
        if len(self) > self.max_size:
            self._cleanup()

    def _cleanup(self):
        """Remove least recently used items."""
        if len(self._access_order) > self.max_size // 2:
            # Remove oldest 25% of items
            to_remove = self._access_order[:len(self._access_order) // 4]
            for key in to_remove:
                if key in self:
                    del self[key]
            self._access_order = self._access_order[len(self._access_order) // 4:]
