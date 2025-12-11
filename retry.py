"""Retry utilities for external API calls and transient failures.

This module provides retry decorators with:
- Exponential backoff
- Configurable retry conditions
- Jitter to prevent thundering herd
- Circuit breaker integration

Usage:
    from oncotarget_lite.retry import retry, retry_with_backoff

    @retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
    def fetch_external_data():
        ...

    @retry_with_backoff(base_delay=1.0, max_delay=60.0)
    async def async_api_call():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Type, TypeVar

from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Default retryable exceptions
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    exceptions: tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS
    on_retry: Callable[[Exception, int], None] | None = None


def _calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
    jitter_factor: float,
) -> float:
    """Calculate delay with exponential backoff and optional jitter."""
    delay = min(base_delay * (exponential_base ** attempt), max_delay)

    if jitter:
        # Add random jitter to prevent thundering herd
        jitter_range = delay * jitter_factor
        delay = delay + random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def retry(
    max_attempts: int = 3,
    exceptions: Sequence[Type[Exception]] = DEFAULT_RETRYABLE_EXCEPTIONS,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on specified exceptions.

    Args:
        max_attempts: Maximum number of attempts
        exceptions: Tuple of exception types to retry on
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        on_retry: Optional callback on each retry

    Returns:
        Decorated function with retry logic

    Example:
        @retry(max_attempts=3, exceptions=(ConnectionError,))
        def fetch_data():
            return requests.get("https://api.example.com/data")
    """
    exceptions_tuple = tuple(exceptions)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions_tuple as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        delay = _calculate_delay(
                            attempt,
                            base_delay,
                            max_delay,
                            exponential_base,
                            jitter,
                            jitter_factor=0.1,
                        )

                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay=round(delay, 2),
                            error=str(e),
                            error_type=type(e).__name__,
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)

            # All attempts failed
            logger.error(
                "retry_exhausted",
                function=func.__name__,
                max_attempts=max_attempts,
                final_error=str(last_exception),
            )

            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry exhausted for {func.__name__}")

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions_tuple as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        delay = _calculate_delay(
                            attempt,
                            base_delay,
                            max_delay,
                            exponential_base,
                            jitter,
                            jitter_factor=0.1,
                        )

                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_attempts=max_attempts,
                            delay=round(delay, 2),
                            error=str(e),
                            error_type=type(e).__name__,
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        await asyncio.sleep(delay)

            logger.error(
                "retry_exhausted",
                function=func.__name__,
                max_attempts=max_attempts,
                final_error=str(last_exception),
            )

            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry exhausted for {func.__name__}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def retry_with_backoff(
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    max_attempts: int = 5,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Simplified retry decorator with exponential backoff.

    Convenience wrapper around retry() with sensible defaults.

    Args:
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        max_attempts: Maximum retry attempts

    Returns:
        Decorated function
    """
    return retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=True,
    )


class RetryContext:
    """
    Context manager for retry logic.

    Useful when you need more control over retry behavior.

    Usage:
        async with RetryContext(max_attempts=3) as ctx:
            while ctx.should_retry:
                try:
                    result = await risky_operation()
                    break
                except ConnectionError as e:
                    await ctx.handle_error(e)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.attempt = 0
        self.last_error: Exception | None = None

    @property
    def should_retry(self) -> bool:
        """Check if more retry attempts are available."""
        return self.attempt < self.max_attempts

    async def handle_error(self, error: Exception) -> None:
        """
        Handle an error and wait before next retry.

        Raises the error if max attempts reached or error is not retryable.
        """
        self.last_error = error

        if not isinstance(error, self.exceptions):
            raise error

        self.attempt += 1

        if self.attempt >= self.max_attempts:
            logger.error(
                "retry_context_exhausted",
                max_attempts=self.max_attempts,
                final_error=str(error),
            )
            raise error

        delay = _calculate_delay(
            self.attempt - 1,
            self.base_delay,
            self.max_delay,
            exponential_base=2.0,
            jitter=True,
            jitter_factor=0.1,
        )

        logger.warning(
            "retry_context_waiting",
            attempt=self.attempt,
            max_attempts=self.max_attempts,
            delay=round(delay, 2),
            error=str(error),
        )

        await asyncio.sleep(delay)

    def handle_error_sync(self, error: Exception) -> None:
        """Synchronous version of handle_error."""
        self.last_error = error

        if not isinstance(error, self.exceptions):
            raise error

        self.attempt += 1

        if self.attempt >= self.max_attempts:
            raise error

        delay = _calculate_delay(
            self.attempt - 1,
            self.base_delay,
            self.max_delay,
            exponential_base=2.0,
            jitter=True,
            jitter_factor=0.1,
        )

        time.sleep(delay)

    async def __aenter__(self) -> "RetryContext":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    def __enter__(self) -> "RetryContext":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    return isinstance(error, DEFAULT_RETRYABLE_EXCEPTIONS)


__all__ = [
    "retry",
    "retry_with_backoff",
    "RetryConfig",
    "RetryContext",
    "is_retryable_error",
    "DEFAULT_RETRYABLE_EXCEPTIONS",
]


