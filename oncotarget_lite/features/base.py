"""Base classes for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors."""

    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = cache_dir
        self.feature_name = "base"

    @abstractmethod
    def extract_features(
        self,
        genes: List[str],
        **kwargs
    ) -> pd.DataFrame:
        """Extract features for the given genes."""
        pass

    @abstractmethod
    def get_required_files(self) -> List[str]:
        """Return list of required data files."""
        pass

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input data."""
        pass

    def get_cache_path(self) -> Path:
        """Get cache file path for this extractor."""
        return self.cache_dir / f"{self.feature_name}_features.parquet"

    def load_from_cache(self) -> pd.DataFrame | None:
        """Load features from cache if available."""
        cache_path = self.get_cache_path()
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None

    def save_to_cache(self, features: pd.DataFrame) -> None:
        """Save features to cache."""
        cache_path = self.get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_path)



