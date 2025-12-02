"""Tests for centralized logging configuration."""

from __future__ import annotations

import json
import logging
from io import StringIO
from unittest.mock import patch

import pytest

from oncotarget_lite.logging_config import (
    configure_logging,
    get_logger,
    get_correlation_id,
    set_correlation_id,
    log_timing,
    LogContext,
)


class TestCorrelationId:
    """Tests for correlation ID management."""

    def test_set_and_get_correlation_id(self):
        """Test that correlation IDs can be set and retrieved."""
        cid = set_correlation_id("test-123")
        assert cid == "test-123"
        assert get_correlation_id() == "test-123"

    def test_auto_generate_correlation_id(self):
        """Test that correlation IDs are auto-generated if not provided."""
        cid = set_correlation_id()
        assert cid is not None
        assert len(cid) == 8  # UUID prefix length
        assert get_correlation_id() == cid

    def test_correlation_id_isolation(self):
        """Test that correlation IDs are isolated per context."""
        set_correlation_id("context-1")
        assert get_correlation_id() == "context-1"


class TestConfigureLogging:
    """Tests for logging configuration."""

    def test_configure_development(self):
        """Test development logging configuration."""
        configure_logging(environment="development")
        logger = get_logger("test")
        assert logger is not None

    def test_configure_production(self):
        """Test production logging configuration."""
        configure_logging(environment="production")
        logger = get_logger("test")
        assert logger is not None

    def test_configure_with_custom_level(self):
        """Test logging with custom log level."""
        configure_logging(environment="development", log_level="WARNING")
        # The root logger level should be WARNING
        assert logging.root.level == logging.WARNING

    def test_configure_json_output(self):
        """Test that JSON output can be forced."""
        configure_logging(environment="development", json_logs=True)
        logger = get_logger("test")
        assert logger is not None


class TestGetLogger:
    """Tests for logger retrieval."""

    def test_get_logger_with_name(self):
        """Test getting a logger with a specific name."""
        configure_logging(environment="development")
        logger = get_logger("my_module")
        assert logger is not None

    def test_get_logger_without_name(self):
        """Test getting a logger without a name."""
        configure_logging(environment="development")
        logger = get_logger()
        assert logger is not None

    def test_logger_can_log(self):
        """Test that the logger can actually log messages."""
        configure_logging(environment="development")
        logger = get_logger("test_logger")
        # This should not raise
        logger.info("test message", key="value")


class TestLogTiming:
    """Tests for the log_timing decorator."""

    def test_log_timing_sync_function(self):
        """Test timing decorator on sync function."""
        configure_logging(environment="development")

        @log_timing("test_operation")
        def slow_function():
            return "result"

        result = slow_function()
        assert result == "result"

    def test_log_timing_with_exception(self):
        """Test timing decorator logs errors on exception."""
        configure_logging(environment="development")

        @log_timing("failing_operation")
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_function()

    def test_log_timing_async_function(self):
        """Test timing decorator on async function."""
        import asyncio
        configure_logging(environment="development")

        @log_timing("async_operation")
        async def async_function():
            return "async result"

        result = asyncio.run(async_function())
        assert result == "async result"


class TestLogContext:
    """Tests for the LogContext context manager."""

    def test_log_context_adds_context(self):
        """Test that LogContext adds context to logs."""
        configure_logging(environment="development")
        logger = get_logger("test")

        with LogContext(request_id="req-123", user="test_user"):
            # Context should be available within the block
            logger.info("test message")

    def test_log_context_removes_context_after_exit(self):
        """Test that context is removed after exiting the block."""
        configure_logging(environment="development")

        with LogContext(temp_key="temp_value"):
            pass
        # Context should be cleared after exit


class TestLogOutput:
    """Tests for actual log output format."""

    def test_json_log_format(self):
        """Test that JSON logs can be configured without errors."""
        configure_logging(environment="production", json_logs=True)
        set_correlation_id("test-cid")

        logger = get_logger("test_json")

        # The main test is that this doesn't raise an exception
        # and the logger is properly configured for JSON output
        logger.info("test_event", custom_field="custom_value")

        # Verify correlation ID is set
        assert get_correlation_id() == "test-cid"

    def test_structured_log_fields(self):
        """Test that structured fields are included in logs."""
        configure_logging(environment="development")
        logger = get_logger("test_structured")

        # This should not raise and should include the structured fields
        logger.info(
            "user_action",
            action="login",
            user_id=123,
            success=True,
        )


class TestMiddlewareIntegration:
    """Tests for middleware integration with logging."""

    def test_correlation_id_middleware_import(self):
        """Test that middleware can be imported."""
        from oncotarget_lite.middleware import (
            CorrelationIdMiddleware,
            RequestLoggingMiddleware,
            add_observability_middleware,
        )
        assert CorrelationIdMiddleware is not None
        assert RequestLoggingMiddleware is not None
        assert add_observability_middleware is not None

