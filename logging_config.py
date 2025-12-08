"""Centralized logging configuration for oncotarget-lite.

This module provides a unified logging setup using structlog for structured,
JSON-formatted logs suitable for production observability (ELK, Datadog, etc.).

Key features:
- Structured JSON logging for production
- Human-readable colored output for development
- Correlation IDs for request tracing
- Performance timing context
- Automatic exception formatting

Usage:
    from oncotarget_lite.logging_config import get_logger, configure_logging

    # At application startup
    configure_logging(environment="production")

    # In any module
    logger = get_logger(__name__)
    logger.info("processing_started", gene_count=100, model="logreg")
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from time import perf_counter
from typing import Any, Callable, TypeVar

import structlog
from structlog.types import Processor

# Context variable for correlation ID (thread-safe for async)
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set a correlation ID in the current context. Generates one if not provided."""
    cid = correlation_id or str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid


def add_correlation_id(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor to add correlation ID to all log entries."""
    cid = get_correlation_id()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def add_service_context(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Add service-level context to all log entries."""
    event_dict["service"] = "oncotarget-lite"
    event_dict["version"] = os.environ.get("SERVICE_VERSION", "unknown")
    return event_dict


def configure_logging(
    environment: str = "development",
    log_level: str | None = None,
    json_logs: bool | None = None,
) -> None:
    """
    Configure logging for the entire application.

    Args:
        environment: One of "development", "staging", "production"
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Force JSON output (True) or console output (False)
    """
    # Determine settings based on environment
    if environment == "production":
        default_level = "INFO"
        default_json = True
    elif environment == "staging":
        default_level = "DEBUG"
        default_json = True
    else:  # development
        default_level = "DEBUG"
        default_json = False

    level = log_level or os.environ.get("LOG_LEVEL", default_level)
    use_json = json_logs if json_logs is not None else (
        os.environ.get("JSON_LOGS", str(default_json)).lower() == "true"
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Silence noisy third-party loggers
    for noisy_logger in ["httpx", "httpcore", "urllib3", "asyncio"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Build processor chain
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_correlation_id,
        add_service_context,
        structlog.processors.UnicodeDecoder(),
    ]

    if use_json:
        # Production: JSON output for log aggregation
        shared_processors.append(structlog.processors.format_exc_info)
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Development: colored, human-readable output
        shared_processors.append(structlog.dev.set_exc_info)
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the formatter for stdlib logging integration
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Apply formatter to root handler
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        A bound structlog logger
    """
    return structlog.get_logger(name)


def log_timing(operation: str | None = None) -> Callable[[F], F]:
    """
    Decorator to log function execution time.

    Usage:
        @log_timing("model_inference")
        def predict(features):
            ...
    """
    def decorator(func: F) -> F:
        op_name = operation or func.__name__
        logger = get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (perf_counter() - start) * 1000
                logger.info(
                    f"{op_name}_completed",
                    duration_ms=round(duration_ms, 2),
                    status="success",
                )
                return result
            except Exception as e:
                duration_ms = (perf_counter() - start) * 1000
                logger.error(
                    f"{op_name}_failed",
                    duration_ms=round(duration_ms, 2),
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (perf_counter() - start) * 1000
                logger.info(
                    f"{op_name}_completed",
                    duration_ms=round(duration_ms, 2),
                    status="success",
                )
                return result
            except Exception as e:
                duration_ms = (perf_counter() - start) * 1000
                logger.error(
                    f"{op_name}_failed",
                    duration_ms=round(duration_ms, 2),
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore

    return decorator


class LogContext:
    """
    Context manager for adding temporary context to logs.

    Usage:
        with LogContext(request_id="abc123", user="john"):
            logger.info("processing")  # Will include request_id and user
    """

    def __init__(self, **context: Any):
        self.context = context
        self.token: Any = None

    def __enter__(self) -> "LogContext":
        structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


# Convenience exports
__all__ = [
    "configure_logging",
    "get_logger",
    "get_correlation_id",
    "set_correlation_id",
    "log_timing",
    "LogContext",
]






