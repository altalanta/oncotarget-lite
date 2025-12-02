"""Dependency injection container for the oncotarget-lite application.

This module provides a centralized dependency injection system that:
- Enables lazy initialization of expensive resources (models, connections)
- Makes testing easier through dependency overrides
- Supports different configurations per environment
- Provides type-safe dependency resolution

Usage:
    # In FastAPI endpoints
    @app.get("/predict")
    async def predict(
        prediction_service: PredictionService = Depends(get_prediction_service)
    ):
        ...

    # In tests
    app.dependency_overrides[get_prediction_service] = lambda: mock_service
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Generic
import threading

from oncotarget_lite.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class LazyDependency(Generic[T]):
    """A lazily-initialized dependency with thread-safe singleton behavior.

    This class wraps a factory function and ensures the dependency is only
    created once, on first access. This is useful for expensive resources
    like ML models that shouldn't be loaded at import time.

    Example:
        model_service = LazyDependency(lambda: PredictionService())
        service = model_service.get()  # Creates instance on first call
    """

    def __init__(self, factory: Callable[[], T], name: str = "unnamed"):
        self._factory = factory
        self._instance: Optional[T] = None
        self._lock = threading.Lock()
        self._name = name

    def get(self) -> T:
        """Get the dependency instance, creating it if necessary."""
        if self._instance is None:
            with self._lock:
                # Double-check locking pattern
                if self._instance is None:
                    logger.info(f"initializing_dependency", dependency=self._name)
                    self._instance = self._factory()
                    logger.info(f"dependency_initialized", dependency=self._name)
        return self._instance

    def reset(self) -> None:
        """Reset the dependency, forcing re-initialization on next access."""
        with self._lock:
            self._instance = None
            logger.info(f"dependency_reset", dependency=self._name)

    def is_initialized(self) -> bool:
        """Check if the dependency has been initialized."""
        return self._instance is not None

    def override(self, instance: T) -> None:
        """Override the dependency with a specific instance (for testing)."""
        with self._lock:
            self._instance = instance
            logger.info(f"dependency_overridden", dependency=self._name)


@dataclass
class DependencyConfig:
    """Configuration for the dependency injection container.

    This allows customizing paths and settings without changing code.
    """

    models_dir: Path = field(default_factory=lambda: Path("models"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    explainer_path: Optional[Path] = None
    model_name: str = "oncotarget-lite"
    model_stage: str = "Production"
    lazy_load: bool = True  # If False, load immediately on container creation

    def __post_init__(self):
        if self.explainer_path is None:
            self.explainer_path = self.reports_dir / "shap" / "explainer.pkl"


class DependencyContainer:
    """Central container for all application dependencies.

    This container manages the lifecycle of all dependencies and provides
    a single point of configuration for the application.

    Example:
        # Create container with custom config
        config = DependencyConfig(models_dir=Path("custom/models"))
        container = DependencyContainer(config)

        # Get dependencies
        prediction_service = container.prediction_service
        feature_validator = container.feature_validator
    """

    def __init__(self, config: Optional[DependencyConfig] = None):
        self._config = config or DependencyConfig()
        self._dependencies: Dict[str, LazyDependency[Any]] = {}
        self._setup_dependencies()

    def _setup_dependencies(self) -> None:
        """Set up all lazy dependencies."""
        # Import here to avoid circular imports
        from oncotarget_lite.feature_validation import FeatureValidator

        # Feature validator - lightweight, can be created immediately
        self._dependencies["feature_validator"] = LazyDependency(
            factory=lambda: FeatureValidator.from_model_dir(self._config.models_dir),
            name="feature_validator",
        )

        # Model loader
        self._dependencies["model_loader"] = LazyDependency(
            factory=self._create_model_loader,
            name="model_loader",
        )

        # Prediction service - expensive, loaded lazily
        self._dependencies["prediction_service"] = LazyDependency(
            factory=self._create_prediction_service,
            name="prediction_service",
        )

        # Resilience manager
        self._dependencies["resilience_manager"] = LazyDependency(
            factory=self._create_resilience_manager,
            name="resilience_manager",
        )

    def _create_model_loader(self):
        """Factory for ModelLoader."""
        from deployment.model_loader import ModelLoader
        return ModelLoader(models_dir=self._config.models_dir)

    def _create_prediction_service(self):
        """Factory for PredictionService."""
        from deployment.prediction_service import PredictionService
        return PredictionService(
            models_dir=self._config.models_dir,
            explainer_path=self._config.explainer_path,
            model_name=self._config.model_name,
            model_stage=self._config.model_stage,
        )

    def _create_resilience_manager(self):
        """Factory for ResilienceManager."""
        from oncotarget_lite.resilience import ResilienceManager
        return ResilienceManager()

    @property
    def prediction_service(self):
        """Get the prediction service instance."""
        return self._dependencies["prediction_service"].get()

    @property
    def feature_validator(self):
        """Get the feature validator instance."""
        return self._dependencies["feature_validator"].get()

    @property
    def model_loader(self):
        """Get the model loader instance."""
        return self._dependencies["model_loader"].get()

    @property
    def resilience_manager(self):
        """Get the resilience manager instance."""
        return self._dependencies["resilience_manager"].get()

    def override(self, name: str, instance: Any) -> None:
        """Override a dependency with a specific instance."""
        if name not in self._dependencies:
            raise ValueError(f"Unknown dependency: {name}")
        self._dependencies[name].override(instance)

    def reset(self, name: Optional[str] = None) -> None:
        """Reset dependencies, forcing re-initialization.

        Args:
            name: Specific dependency to reset, or None to reset all
        """
        if name:
            if name not in self._dependencies:
                raise ValueError(f"Unknown dependency: {name}")
            self._dependencies[name].reset()
        else:
            for dep in self._dependencies.values():
                dep.reset()

    def is_initialized(self, name: str) -> bool:
        """Check if a specific dependency has been initialized."""
        if name not in self._dependencies:
            raise ValueError(f"Unknown dependency: {name}")
        return self._dependencies[name].is_initialized()

    def warmup(self) -> None:
        """Initialize all dependencies (useful for startup)."""
        logger.info("warming_up_dependencies")
        for name, dep in self._dependencies.items():
            try:
                dep.get()
            except Exception as e:
                logger.error(
                    "dependency_warmup_failed",
                    dependency=name,
                    error=str(e),
                )
        logger.info("dependency_warmup_complete")


# Global container instance (lazily created)
_container: Optional[DependencyContainer] = None
_container_lock = threading.Lock()


def get_container() -> DependencyContainer:
    """Get the global dependency container."""
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = DependencyContainer()
    return _container


def set_container(container: DependencyContainer) -> None:
    """Set the global dependency container (for testing or custom config)."""
    global _container
    with _container_lock:
        _container = container


def reset_container() -> None:
    """Reset the global container (for testing)."""
    global _container
    with _container_lock:
        _container = None


# FastAPI dependency functions
# These are used with Depends() in endpoint definitions

def get_prediction_service():
    """FastAPI dependency for PredictionService."""
    return get_container().prediction_service


def get_feature_validator():
    """FastAPI dependency for FeatureValidator."""
    return get_container().feature_validator


def get_model_loader():
    """FastAPI dependency for ModelLoader."""
    return get_container().model_loader


def get_resilience_manager():
    """FastAPI dependency for ResilienceManager."""
    return get_container().resilience_manager


__all__ = [
    "DependencyContainer",
    "DependencyConfig",
    "LazyDependency",
    "get_container",
    "set_container",
    "reset_container",
    "get_prediction_service",
    "get_feature_validator",
    "get_model_loader",
    "get_resilience_manager",
]

