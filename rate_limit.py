"""Rate limiting middleware for API protection.

This module provides rate limiting capabilities to protect the API from:
- Denial of service attacks
- Excessive usage from single clients
- Resource exhaustion from expensive ML inference

Usage:
    from oncotarget_lite.rate_limit import RateLimiter, add_rate_limiting

    # Add to FastAPI app
    add_rate_limiting(app, requests_per_minute=60)

    # Or use the limiter directly
    limiter = RateLimiter(requests_per_minute=60)

    @app.get("/predict")
    @limiter.limit("10/minute")
    async def predict():
        ...
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_second: int = 10
    burst_size: int = 20
    enable_headers: bool = True
    exempt_paths: list[str] = field(default_factory=lambda: ["/health", "/health/live", "/health/ready", "/metrics"])
    key_func: Optional[Callable[[Request], str]] = None


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: float
    tokens: float
    last_update: float
    refill_rate: float  # tokens per second

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.

        Returns True if tokens were consumed, False if rate limited.
        """
        now = time.monotonic()
        elapsed = now - self.last_update

        # Refill tokens
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    @property
    def retry_after(self) -> float:
        """Seconds until a token is available."""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.refill_rate


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.

    Supports:
    - Per-client rate limiting
    - Configurable limits per endpoint
    - Automatic cleanup of stale buckets
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 20,
        cleanup_interval: int = 300,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Sustained request rate limit
            burst_size: Maximum burst size allowed
            cleanup_interval: Seconds between stale bucket cleanup
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.cleanup_interval = cleanup_interval
        self.refill_rate = requests_per_minute / 60.0  # tokens per second

        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create a token bucket for a key."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=float(self.burst_size),
                tokens=float(self.burst_size),
                last_update=time.monotonic(),
                refill_rate=self.refill_rate,
            )
        return self._buckets[key]

    async def is_allowed(self, key: str, tokens: int = 1) -> tuple[bool, float]:
        """
        Check if request is allowed.

        Args:
            key: Client identifier (e.g., IP address)
            tokens: Number of tokens to consume

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        async with self._lock:
            # Periodic cleanup
            now = time.monotonic()
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_stale_buckets()
                self._last_cleanup = now

            bucket = self._get_bucket(key)
            allowed = bucket.consume(tokens)
            retry_after = 0 if allowed else bucket.retry_after

            return allowed, retry_after

    def _cleanup_stale_buckets(self) -> None:
        """Remove buckets that haven't been used recently."""
        now = time.monotonic()
        stale_threshold = self.cleanup_interval * 2

        stale_keys = [
            key for key, bucket in self._buckets.items()
            if now - bucket.last_update > stale_threshold
        ]

        for key in stale_keys:
            del self._buckets[key]

        if stale_keys:
            logger.debug("rate_limit_cleanup", removed_buckets=len(stale_keys))

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "active_buckets": len(self._buckets),
            "requests_per_minute": self.requests_per_minute,
            "burst_size": self.burst_size,
        }

    def limit(self, rate: str) -> Callable:
        """
        Decorator for endpoint-specific rate limiting.

        Args:
            rate: Rate string like "10/minute" or "100/hour"

        Returns:
            Decorator function
        """
        # Parse rate string
        count, period = rate.split("/")
        count = int(count)
        period_seconds = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400,
        }.get(period, 60)

        tokens_per_request = self.requests_per_minute * period_seconds / (60 * count)

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request from args
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                if request:
                    key = _get_client_key(request)
                    allowed, retry_after = await self.is_allowed(key, int(tokens_per_request))

                    if not allowed:
                        logger.warning(
                            "rate_limit_exceeded",
                            client=key,
                            endpoint=func.__name__,
                            retry_after=retry_after,
                        )
                        raise HTTPException(
                            status_code=429,
                            detail=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds.",
                            headers={"Retry-After": str(int(retry_after) + 1)},
                        )

                return await func(*args, **kwargs)

            return wrapper

        return decorator


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for global rate limiting.

    Applies rate limiting to all requests except exempt paths.
    """

    def __init__(
        self,
        app: FastAPI,
        config: RateLimitConfig | None = None,
    ):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.limiter = RateLimiter(
            requests_per_minute=self.config.requests_per_minute,
            burst_size=self.config.burst_size,
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Check if path is exempt
        if request.url.path in self.config.exempt_paths:
            return await call_next(request)

        # Get client key
        key = _get_client_key(request, self.config.key_func)

        # Check rate limit
        allowed, retry_after = await self.limiter.is_allowed(key)

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                client=key,
                path=request.url.path,
                method=request.method,
                retry_after=retry_after,
            )

            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": int(retry_after) + 1,
                },
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "X-RateLimit-Limit": str(self.config.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers if enabled
        if self.config.enable_headers:
            bucket = self.limiter._get_bucket(key)
            response.headers["X-RateLimit-Limit"] = str(self.config.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))

        return response


def _get_client_key(
    request: Request,
    key_func: Optional[Callable[[Request], str]] = None,
) -> str:
    """
    Extract client identifier from request.

    Priority:
    1. Custom key function
    2. X-Forwarded-For header (for proxied requests)
    3. X-Real-IP header
    4. Client host from connection

    Args:
        request: FastAPI request
        key_func: Optional custom key extraction function

    Returns:
        Client identifier string
    """
    if key_func:
        return key_func(request)

    # Check for proxy headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fall back to connection host
    if request.client:
        return request.client.host

    return "unknown"


def add_rate_limiting(
    app: FastAPI,
    requests_per_minute: int = 60,
    burst_size: int = 20,
    exempt_paths: list[str] | None = None,
) -> RateLimiter:
    """
    Add rate limiting middleware to a FastAPI app.

    Args:
        app: FastAPI application
        requests_per_minute: Sustained rate limit
        burst_size: Maximum burst size
        exempt_paths: Paths to exempt from rate limiting

    Returns:
        The RateLimiter instance for custom endpoint limiting
    """
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
        exempt_paths=exempt_paths or ["/health", "/health/live", "/health/ready", "/metrics"],
    )

    app.add_middleware(RateLimitMiddleware, config=config)

    logger.info(
        "rate_limiting_enabled",
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
    )

    return RateLimiter(requests_per_minute=requests_per_minute, burst_size=burst_size)


__all__ = [
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitMiddleware",
    "add_rate_limiting",
]

