"""FastAPI middleware for request tracing and observability.

This module provides middleware components for:
- Correlation ID injection and propagation
- Request/response logging
- Performance timing
"""

from __future__ import annotations

import time
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from oncotarget_lite.logging_config import (
    get_logger,
    set_correlation_id,
    get_correlation_id,
    LogContext,
)

logger = get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects a correlation ID into every request.

    The correlation ID is:
    1. Read from the X-Correlation-ID header if present
    2. Generated if not present
    3. Added to all log entries for the request
    4. Returned in the response X-Correlation-ID header

    This enables end-to-end request tracing across services.
    """

    HEADER_NAME = "X-Correlation-ID"

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Get or generate correlation ID
        correlation_id = request.headers.get(self.HEADER_NAME) or str(uuid4())[:8]
        set_correlation_id(correlation_id)

        # Process request with logging context
        with LogContext(
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
        ):
            response = await call_next(request)

        # Add correlation ID to response headers
        response.headers[self.HEADER_NAME] = correlation_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all incoming requests and outgoing responses.

    Logs include:
    - Request method, path, query params
    - Response status code
    - Request duration in milliseconds
    """

    # Paths to exclude from logging (health checks, metrics)
    EXCLUDED_PATHS = {"/health", "/metrics", "/favicon.ico"}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip logging for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        start_time = time.perf_counter()

        # Log incoming request
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log successful response
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log failed request
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise


def add_observability_middleware(app) -> None:
    """
    Add all observability middleware to a FastAPI app.

    Order matters: CorrelationId should be outermost so the ID is available
    for all other middleware and handlers.

    Usage:
        from oncotarget_lite.middleware import add_observability_middleware
        app = FastAPI()
        add_observability_middleware(app)
    """
    # Add in reverse order (last added = first executed)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)







