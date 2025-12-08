"""Async timeout utilities for protecting API operations.

This module provides timeout decorators and context managers for:
- API endpoint protection against slow ML inference
- External service call timeouts
- Graceful timeout handling with proper logging

Usage:
    from oncotarget_lite.timeouts import with_timeout, timeout_context

    # As a decorator
    @with_timeout(seconds=10.0, fallback_value=None)
    async def slow_operation():
        ...

    # As a context manager
    async with timeout_context(seconds=5.0) as ctx:
        result = await external_api_call()
        if ctx.timed_out:
            handle_timeout()
"""

from __future__ import annotations

import asyncio
import functools
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, ParamSpec, TypeVar

from .exceptions import PredictionError
from .logging_config import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class TimeoutError(PredictionError):
    """Raised when an operation times out."""

    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Operation '{operation}' timed out after {timeout:.1f}s")


@dataclass
class TimeoutContext:
    """Context information for timeout handling."""

    timeout_seconds: float
    started_at: float = 0.0
    ended_at: float = 0.0
    timed_out: bool = False
    error: Exception | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.ended_at > 0:
            return self.ended_at - self.started_at
        return perf_counter() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        """Get remaining time before timeout."""
        return max(0, self.timeout_seconds - self.elapsed_seconds)


def with_timeout(
    seconds: float = 10.0,
    operation_name: str | None = None,
    fallback_value: T | None = None,
    raise_on_timeout: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to add timeout protection to async functions.

    Args:
        seconds: Maximum execution time in seconds
        operation_name: Name for logging (defaults to function name)
        fallback_value: Value to return on timeout if raise_on_timeout is False
        raise_on_timeout: Whether to raise TimeoutError on timeout

    Returns:
        Decorated function with timeout protection

    Example:
        @with_timeout(seconds=5.0, operation_name="model_predict")
        async def predict(features: dict) -> float:
            return await model.predict_async(features)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = perf_counter()

            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
                elapsed = perf_counter() - start

                # Log warning if close to timeout
                if elapsed > seconds * 0.8:
                    logger.warning(
                        f"{op_name}_slow",
                        elapsed_seconds=round(elapsed, 3),
                        timeout_seconds=seconds,
                        percent_of_timeout=round(elapsed / seconds * 100, 1),
                    )
                else:
                    logger.debug(
                        f"{op_name}_completed",
                        elapsed_seconds=round(elapsed, 3),
                    )

                return result

            except asyncio.TimeoutError:
                elapsed = perf_counter() - start
                logger.error(
                    f"{op_name}_timeout",
                    elapsed_seconds=round(elapsed, 3),
                    timeout_seconds=seconds,
                )

                if raise_on_timeout:
                    raise TimeoutError(op_name, seconds)

                return fallback_value  # type: ignore

        return wrapper  # type: ignore

    return decorator


def with_sync_timeout(
    seconds: float = 10.0,
    operation_name: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to add timeout protection to sync functions run in thread pool.

    This wraps a sync function to be run via asyncio.to_thread() with a timeout.

    Args:
        seconds: Maximum execution time in seconds
        operation_name: Name for logging

    Example:
        @with_sync_timeout(seconds=10.0)
        def blocking_ml_inference(features):
            return model.predict(features)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start = perf_counter()

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs),
                    timeout=seconds,
                )
                elapsed = perf_counter() - start

                logger.debug(
                    f"{op_name}_completed",
                    elapsed_seconds=round(elapsed, 3),
                )

                return result

            except asyncio.TimeoutError:
                logger.error(
                    f"{op_name}_timeout",
                    timeout_seconds=seconds,
                )
                raise TimeoutError(op_name, seconds)

        return wrapper  # type: ignore

    return decorator


@asynccontextmanager
async def timeout_context(
    seconds: float,
    operation_name: str = "operation",
    raise_on_timeout: bool = True,
):
    """
    Async context manager for timeout-protected operations.

    Args:
        seconds: Maximum execution time
        operation_name: Name for logging
        raise_on_timeout: Whether to raise on timeout

    Yields:
        TimeoutContext with status information

    Example:
        async with timeout_context(5.0, "external_api") as ctx:
            result = await fetch_external_data()

        if ctx.timed_out:
            use_cached_result()
    """
    ctx = TimeoutContext(timeout_seconds=seconds)
    ctx.started_at = perf_counter()

    try:
        async with asyncio.timeout(seconds):
            yield ctx

    except asyncio.TimeoutError:
        ctx.timed_out = True
        ctx.ended_at = perf_counter()
        ctx.error = TimeoutError(operation_name, seconds)

        logger.error(
            f"{operation_name}_timeout",
            elapsed_seconds=round(ctx.elapsed_seconds, 3),
            timeout_seconds=seconds,
        )

        if raise_on_timeout:
            raise ctx.error

    finally:
        if ctx.ended_at == 0:
            ctx.ended_at = perf_counter()


async def run_with_timeout(
    coro: Any,
    timeout: float,
    operation_name: str = "operation",
) -> Any:
    """
    Run a coroutine with timeout protection.

    Convenience function for one-off timeout protection.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        operation_name: Name for logging

    Returns:
        Result of the coroutine

    Raises:
        TimeoutError: If operation times out
    """
    start = perf_counter()

    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        elapsed = perf_counter() - start

        logger.debug(
            f"{operation_name}_completed",
            elapsed_seconds=round(elapsed, 3),
        )

        return result

    except asyncio.TimeoutError:
        logger.error(
            f"{operation_name}_timeout",
            timeout_seconds=timeout,
        )
        raise TimeoutError(operation_name, timeout)


class AdaptiveTimeout:
    """
    Adaptive timeout that adjusts based on historical latencies.

    Useful for operations with variable latency where a fixed timeout
    may be too aggressive or too lenient.
    """

    def __init__(
        self,
        initial_timeout: float = 10.0,
        min_timeout: float = 1.0,
        max_timeout: float = 60.0,
        percentile: float = 0.95,
        window_size: int = 100,
    ):
        """
        Initialize adaptive timeout.

        Args:
            initial_timeout: Starting timeout value
            min_timeout: Minimum allowed timeout
            max_timeout: Maximum allowed timeout
            percentile: Percentile of latencies to use for timeout
            window_size: Number of recent latencies to track
        """
        self.current_timeout = initial_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.percentile = percentile
        self.window_size = window_size
        self.latencies: list[float] = []

    def record_latency(self, latency: float) -> None:
        """Record an observed latency."""
        self.latencies.append(latency)
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
        self._update_timeout()

    def _update_timeout(self) -> None:
        """Update timeout based on recorded latencies."""
        if len(self.latencies) < 10:
            return  # Not enough data

        import numpy as np

        percentile_latency = float(np.percentile(self.latencies, self.percentile * 100))

        # Add 20% buffer
        new_timeout = percentile_latency * 1.2

        # Clamp to bounds
        self.current_timeout = max(
            self.min_timeout,
            min(self.max_timeout, new_timeout),
        )

    @property
    def timeout(self) -> float:
        """Get current adaptive timeout value."""
        return self.current_timeout

    def decorator(
        self,
        operation_name: str | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Create a decorator using this adaptive timeout."""

        def decorator_inner(func: Callable[P, T]) -> Callable[P, T]:
            op_name = operation_name or func.__name__

            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                start = perf_counter()

                try:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.current_timeout,
                    )
                    elapsed = perf_counter() - start
                    self.record_latency(elapsed)
                    return result

                except asyncio.TimeoutError:
                    logger.error(
                        f"{op_name}_adaptive_timeout",
                        current_timeout=self.current_timeout,
                    )
                    raise TimeoutError(op_name, self.current_timeout)

            return wrapper  # type: ignore

        return decorator_inner


__all__ = [
    "TimeoutError",
    "TimeoutContext",
    "with_timeout",
    "with_sync_timeout",
    "timeout_context",
    "run_with_timeout",
    "AdaptiveTimeout",
]




