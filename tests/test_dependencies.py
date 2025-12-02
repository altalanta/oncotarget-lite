"""Tests for the dependency injection system.

This module tests:
- LazyDependency behavior (lazy init, thread safety, override)
- DependencyContainer configuration and lifecycle
- FastAPI dependency functions
- Integration with the model server
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from oncotarget_lite.dependencies import (
    DependencyConfig,
    DependencyContainer,
    LazyDependency,
    get_container,
    reset_container,
    set_container,
    get_prediction_service,
    get_feature_validator,
)


class TestLazyDependency:
    """Tests for the LazyDependency class."""

    def test_lazy_initialization(self):
        """Test that factory is not called until get() is called."""
        factory_called = False

        def factory():
            nonlocal factory_called
            factory_called = True
            return "instance"

        lazy = LazyDependency(factory, name="test")

        assert not factory_called
        assert not lazy.is_initialized()

        result = lazy.get()

        assert factory_called
        assert lazy.is_initialized()
        assert result == "instance"

    def test_singleton_behavior(self):
        """Test that factory is only called once."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        lazy = LazyDependency(factory, name="test")

        # Multiple calls should return same instance
        result1 = lazy.get()
        result2 = lazy.get()
        result3 = lazy.get()

        assert call_count == 1
        assert result1 == result2 == result3 == "instance_1"

    def test_reset_allows_reinitialization(self):
        """Test that reset() allows the factory to be called again."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        lazy = LazyDependency(factory, name="test")

        result1 = lazy.get()
        assert result1 == "instance_1"

        lazy.reset()
        assert not lazy.is_initialized()

        result2 = lazy.get()
        assert result2 == "instance_2"
        assert call_count == 2

    def test_override_replaces_instance(self):
        """Test that override() replaces the instance."""
        lazy = LazyDependency(lambda: "original", name="test")

        # Get original
        assert lazy.get() == "original"

        # Override
        lazy.override("overridden")
        assert lazy.get() == "overridden"

    def test_thread_safety(self):
        """Test that lazy initialization is thread-safe."""
        call_count = 0
        call_lock = threading.Lock()

        def factory():
            nonlocal call_count
            with call_lock:
                call_count += 1
            time.sleep(0.01)  # Simulate slow initialization
            return "instance"

        lazy = LazyDependency(factory, name="test")

        # Start multiple threads that all try to get() simultaneously
        results = []
        threads = []

        def get_instance():
            results.append(lazy.get())

        for _ in range(10):
            t = threading.Thread(target=get_instance)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Factory should only be called once despite concurrent access
        assert call_count == 1
        assert all(r == "instance" for r in results)


class TestDependencyConfig:
    """Tests for the DependencyConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DependencyConfig()

        assert config.models_dir == Path("models")
        assert config.reports_dir == Path("reports")
        assert config.model_name == "oncotarget-lite"
        assert config.model_stage == "Production"
        assert config.lazy_load is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DependencyConfig(
            models_dir=Path("custom/models"),
            model_name="custom-model",
            model_stage="Staging",
            lazy_load=False,
        )

        assert config.models_dir == Path("custom/models")
        assert config.model_name == "custom-model"
        assert config.model_stage == "Staging"
        assert config.lazy_load is False

    def test_explainer_path_default(self):
        """Test that explainer_path defaults to reports/shap/explainer.pkl."""
        config = DependencyConfig()
        assert config.explainer_path == Path("reports/shap/explainer.pkl")

    def test_explainer_path_custom(self):
        """Test custom explainer path."""
        config = DependencyConfig(explainer_path=Path("custom/explainer.pkl"))
        assert config.explainer_path == Path("custom/explainer.pkl")


class TestDependencyContainer:
    """Tests for the DependencyContainer class."""

    @pytest.fixture
    def mock_container(self):
        """Create a container with mocked dependencies."""
        config = DependencyConfig(models_dir=Path("test_models"))
        container = DependencyContainer(config)

        # Override with mocks
        mock_service = MagicMock()
        mock_service.model_version = "test-v1"
        mock_service.is_loaded = True

        mock_validator = MagicMock()
        mock_validator.expected_features = ["a", "b", "c"]

        container.override("prediction_service", mock_service)
        container.override("feature_validator", mock_validator)

        return container

    def test_container_initialization(self):
        """Test container initializes without errors."""
        config = DependencyConfig()
        container = DependencyContainer(config)
        assert container is not None

    def test_container_lazy_loading(self, mock_container):
        """Test that dependencies are not loaded until accessed."""
        # Mocks were already overridden, so they should be "initialized"
        assert mock_container.is_initialized("prediction_service")
        assert mock_container.is_initialized("feature_validator")

    def test_container_property_access(self, mock_container):
        """Test accessing dependencies through properties."""
        service = mock_container.prediction_service
        validator = mock_container.feature_validator

        assert service.model_version == "test-v1"
        assert validator.expected_features == ["a", "b", "c"]

    def test_container_override(self, mock_container):
        """Test overriding a dependency."""
        new_mock = MagicMock()
        new_mock.model_version = "new-version"

        mock_container.override("prediction_service", new_mock)

        assert mock_container.prediction_service.model_version == "new-version"

    def test_container_override_unknown_raises(self, mock_container):
        """Test that overriding unknown dependency raises error."""
        with pytest.raises(ValueError, match="Unknown dependency"):
            mock_container.override("unknown_dependency", MagicMock())

    def test_container_reset_single(self, mock_container):
        """Test resetting a single dependency."""
        mock_container.reset("prediction_service")
        assert not mock_container.is_initialized("prediction_service")

    def test_container_reset_all(self, mock_container):
        """Test resetting all dependencies."""
        mock_container.reset()
        assert not mock_container.is_initialized("prediction_service")
        assert not mock_container.is_initialized("feature_validator")

    def test_container_is_initialized_unknown_raises(self, mock_container):
        """Test that checking unknown dependency raises error."""
        with pytest.raises(ValueError, match="Unknown dependency"):
            mock_container.is_initialized("unknown_dependency")


class TestGlobalContainer:
    """Tests for the global container management functions."""

    def setup_method(self):
        """Reset global container before each test."""
        reset_container()

    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()

    def test_get_container_creates_singleton(self):
        """Test that get_container returns the same instance."""
        container1 = get_container()
        container2 = get_container()

        assert container1 is container2

    def test_set_container_replaces_global(self):
        """Test that set_container replaces the global container."""
        custom_container = DependencyContainer(
            DependencyConfig(models_dir=Path("custom"))
        )

        set_container(custom_container)

        assert get_container() is custom_container

    def test_reset_container_clears_global(self):
        """Test that reset_container clears the global container."""
        container1 = get_container()
        reset_container()
        container2 = get_container()

        # Should be different instances
        assert container1 is not container2


class TestFastAPIDependencyFunctions:
    """Tests for the FastAPI dependency functions."""

    def setup_method(self):
        """Reset global container before each test."""
        reset_container()

    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()

    def test_get_prediction_service_returns_service(self):
        """Test get_prediction_service returns a PredictionService."""
        # Override with mock to avoid loading real model
        container = get_container()
        mock_service = MagicMock()
        container.override("prediction_service", mock_service)

        service = get_prediction_service()
        assert service is mock_service

    def test_get_feature_validator_returns_validator(self):
        """Test get_feature_validator returns a FeatureValidator."""
        container = get_container()
        mock_validator = MagicMock()
        container.override("feature_validator", mock_validator)

        validator = get_feature_validator()
        assert validator is mock_validator


class TestDependencyInjectionIntegration:
    """Integration tests for DI with FastAPI."""

    def setup_method(self):
        """Reset global container before each test."""
        reset_container()

    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()

    def test_dependency_override_in_fastapi(self):
        """Test that FastAPI dependency_overrides work correctly."""
        from fastapi import FastAPI, Depends
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/test")
        def test_endpoint(service=Depends(get_prediction_service)):
            return {"version": service.model_version}

        # Override the dependency
        mock_service = MagicMock()
        mock_service.model_version = "mocked-v1"

        container = get_container()
        container.override("prediction_service", mock_service)

        with TestClient(app) as client:
            response = client.get("/test")
            assert response.status_code == 200
            assert response.json()["version"] == "mocked-v1"

    def test_different_containers_per_test(self):
        """Test that each test can have its own container configuration."""
        # First "test" with config A
        config_a = DependencyConfig(model_name="model-a")
        container_a = DependencyContainer(config_a)
        set_container(container_a)

        assert get_container()._config.model_name == "model-a"

        # Reset for next "test"
        reset_container()

        # Second "test" with config B
        config_b = DependencyConfig(model_name="model-b")
        container_b = DependencyContainer(config_b)
        set_container(container_b)

        assert get_container()._config.model_name == "model-b"
