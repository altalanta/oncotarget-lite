"""Resilience framework with retries, circuit breakers, and timeouts."""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, Optional

from .exceptions import (
    ConfigurationError,
    DataPreparationError,
    ModelLoadingError,
    PredictionError,
)

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker to prevent repeated failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None

    def __enter__(self):
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise PredictionError(f"Circuit breaker '{self.name}' is open")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.last_failure_time = time.time()
                logger.warning(
                    f"Circuit breaker '{self.name}' opened due to {self.failure_count} failures"
                )
        else:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.reset()
            self.failure_count = 0

    def reset(self):
        """Reset the circuit breaker."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' has been reset to CLOSED")


class ResilienceManager:
    """Manages resilience patterns like retries and circuit breakers."""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name=name, **kwargs)
        return self.circuit_breakers[name]

    def resilient_function(
        self,
        retries: int = 3,
        backoff_factor: float = 2.0,
        circuit_breaker_name: Optional[str] = None,
    ) -> Callable:
        """Decorator for resilient function execution."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(retries):
                    try:
                        if circuit_breaker_name:
                            with self.get_circuit_breaker(circuit_breaker_name):
                                return await func(*args, **kwargs)
                        else:
                            return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        sleep_time = backoff_factor ** attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{retries} failed for {func.__name__}. "
                            f"Retrying in {sleep_time:.2f}s. Error: {e}"
                        )
                        await asyncio.sleep(sleep_time)
                raise last_exception

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(retries):
                    try:
                        if circuit_breaker_name:
                            with self.get_circuit_breaker(circuit_breaker_name):
                                return func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        sleep_time = backoff_factor ** attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{retries} failed for {func.__name__}. "
                            f"Retrying in {sleep_time:.2f}s. Error: {e}"
                        )
                        time.sleep(sleep_time)
                raise last_exception

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @contextmanager
    def graceful_degradation(self, default_value: Any = None, on_failure: Callable[[Exception], None] = None):
        """Context manager for graceful degradation."""
        try:
            yield
        except Exception as e:
            logger.error(f"Graceful degradation triggered: {e}")
            if on_failure:
                on_failure(e)
            return default_value


_resilience_manager = None


def get_resilience_manager() -> ResilienceManager:
    """Get the global resilience manager."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager
