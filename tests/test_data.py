from __future__ import annotations

import pandas as pd
import pytest

from oncotarget_lite.config import AppConfig
from oncotarget_lite.data import (
    DataContractError,
    build_feature_table,
    load_raw_data,
    split_dataset,
)
from oncotarget_lite.features import build_features

GENE_COUNT = 50
POSITIVE_THRESHOLD = 0


def test_build_feature_table(default_config: AppConfig) -> None:
    bundle = load_raw_data(default_config.data)
    merged = build_feature_table(bundle)
    assert merged.shape[0] == GENE_COUNT
    assert any(col.startswith("normal_") for col in merged.columns)
    assert "ppi_degree" in merged.columns


def test_feature_generation(default_config: AppConfig) -> None:
    bundle = load_raw_data(default_config.data)
    merged = build_feature_table(bundle)
    feature_set = build_features(merged)
    assert feature_set.features.shape[0] == GENE_COUNT
    assert feature_set.labels.sum() > POSITIVE_THRESHOLD
    assert set(feature_set.features.columns).issuperset({"mean_dependency", "ppi_degree"})


def test_split_is_deterministic(default_config: AppConfig) -> None:
    bundle = load_raw_data(default_config.data)
    merged = build_feature_table(bundle)
    feature_set = build_features(merged)
    split_a = split_dataset(feature_set.features, feature_set.labels, default_config.split)
    split_b = split_dataset(feature_set.features, feature_set.labels, default_config.split)
    pd.testing.assert_frame_equal(split_a.X_train, split_b.X_train)
    pd.testing.assert_series_equal(split_a.y_test, split_b.y_test)


def test_split_raises_on_missing_labels(default_config: AppConfig) -> None:
    bundle = load_raw_data(default_config.data)
    merged = build_feature_table(bundle)
    feature_set = build_features(merged)
    corrupted_labels = feature_set.labels.copy().astype(float)
    corrupted_labels.iloc[0] = float("nan")
    with pytest.raises(DataContractError):
        split_dataset(feature_set.features, corrupted_labels, default_config.split)
