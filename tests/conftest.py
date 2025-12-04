from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from oncotarget_lite.utils import ensure_dir
from tests.fixtures import TestDataFactory


# Create a module-level factory for reuse
_factory = TestDataFactory(seed=42)


@pytest.fixture()
def test_data_factory() -> TestDataFactory:
    """Get a seeded test data factory."""
    return TestDataFactory(seed=42)


@pytest.fixture()
def synthetic_reports(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary reports directory with synthetic predictions."""
    reports_dir = tmp_path / "reports"
    ensure_dir(reports_dir)

    # Use the centralized factory
    _factory.save_predictions_to_parquet(
        reports_dir / "predictions.parquet",
        n_train=40,
        n_test=20,
    )

    yield reports_dir


@pytest.fixture()
def synthetic_features(tmp_path: Path) -> Generator[tuple[pd.DataFrame, pd.Series], None, None]:
    """Create synthetic features and labels for testing."""
    factory = TestDataFactory(seed=42)

    features = factory.create_features(n_genes=100, n_features=10)
    labels = factory.create_labels(n_genes=100, positive_ratio=0.3)

    # Ensure indices match
    labels = labels.reindex(features.index)

    yield features, labels


@pytest.fixture()
def synthetic_api_request() -> dict[str, dict[str, float]]:
    """Create a synthetic API prediction request."""
    factory = TestDataFactory(seed=42)
    return factory.create_api_request(n_features=10)


@pytest.fixture()
def synthetic_predictions() -> pd.DataFrame:
    """Create synthetic predictions DataFrame."""
    factory = TestDataFactory(seed=42)
    return factory.create_train_test_predictions(n_train=40, n_test=20)


@pytest.fixture()
def synthetic_model_metrics() -> dict[str, float]:
    """Create synthetic model metrics."""
    factory = TestDataFactory(seed=42)
    return factory.create_model_metrics()
