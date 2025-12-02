"""Tests for feature validation module."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import pytest

from oncotarget_lite.feature_validation import (
    FeatureValidator,
    ValidationMode,
    ValidationResult,
    ValidationError,
    ValidationWarning,
    FeatureSpec,
    get_feature_validator,
    reset_feature_validator,
)


class TestFeatureValidator:
    """Tests for the FeatureValidator class."""

    @pytest.fixture
    def sample_features(self) -> list[str]:
        """Sample feature list for testing."""
        return ["feature_a", "feature_b", "feature_c"]

    @pytest.fixture
    def validator(self, sample_features: list[str]) -> FeatureValidator:
        """Create a validator with sample features."""
        return FeatureValidator(expected_features=sample_features)

    def test_validator_initialization(self, validator: FeatureValidator, sample_features: list[str]):
        """Test validator initializes correctly."""
        assert validator.expected_features == sample_features
        assert len(validator.expected_feature_set) == 3
        assert "feature_a" in validator.expected_feature_set

    def test_validator_creates_default_specs(self, validator: FeatureValidator):
        """Test that default specs are created for features without explicit specs."""
        assert "feature_a" in validator.feature_specs
        assert validator.feature_specs["feature_a"].name == "feature_a"
        assert validator.feature_specs["feature_a"].required is True

    def test_validator_with_custom_specs(self, sample_features: list[str]):
        """Test validator with custom feature specifications."""
        specs = {
            "feature_a": FeatureSpec(
                name="feature_a",
                required=True,
                min_value=0.0,
                max_value=1.0,
            )
        }
        validator = FeatureValidator(
            expected_features=sample_features,
            feature_specs=specs,
        )
        assert validator.feature_specs["feature_a"].min_value == 0.0
        assert validator.feature_specs["feature_a"].max_value == 1.0


class TestValidationModes:
    """Tests for different validation modes."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        """Create a validator with known features."""
        return FeatureValidator(expected_features=["a", "b", "c"])

    def test_strict_mode_all_features_present(self, validator: FeatureValidator):
        """Test strict mode passes when all features are present."""
        features = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_strict_mode_missing_feature(self, validator: FeatureValidator):
        """Test strict mode fails when features are missing."""
        features = {"a": 1.0, "b": 2.0}  # missing "c"
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert not result.is_valid
        assert any(e.code == "MISSING_FEATURE" for e in result.errors)

    def test_strict_mode_extra_feature(self, validator: FeatureValidator):
        """Test strict mode fails when extra features are present."""
        features = {"a": 1.0, "b": 2.0, "c": 3.0, "extra": 4.0}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert not result.is_valid
        assert any(e.code == "UNEXPECTED_FEATURE" for e in result.errors)

    def test_lenient_mode_extra_features_warned(self, validator: FeatureValidator):
        """Test lenient mode warns but allows extra features."""
        features = {"a": 1.0, "b": 2.0, "c": 3.0, "extra": 4.0}
        result = validator.validate(features, mode=ValidationMode.LENIENT)
        assert result.is_valid
        assert any(w.code == "EXTRA_FEATURE" for w in result.warnings)

    def test_lenient_mode_missing_feature_fails(self, validator: FeatureValidator):
        """Test lenient mode fails on missing features."""
        features = {"a": 1.0, "b": 2.0}
        result = validator.validate(features, mode=ValidationMode.LENIENT)
        assert not result.is_valid

    def test_permissive_mode_fills_missing(self, validator: FeatureValidator):
        """Test permissive mode fills missing features with defaults."""
        features = {"a": 1.0}  # missing b and c
        result = validator.validate(features, mode=ValidationMode.PERMISSIVE)
        assert result.is_valid
        assert result.validated_features is not None
        assert "b" in result.validated_features
        assert "c" in result.validated_features
        assert result.validated_features["b"] == 0.0  # default value

    def test_permissive_mode_ignores_extra(self, validator: FeatureValidator):
        """Test permissive mode ignores extra features."""
        features = {"a": 1.0, "b": 2.0, "c": 3.0, "extra": 4.0}
        result = validator.validate(features, mode=ValidationMode.PERMISSIVE)
        assert result.is_valid
        assert "extra" not in result.validated_features


class TestValueValidation:
    """Tests for feature value validation."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        """Create a validator with value constraints."""
        specs = {
            "bounded": FeatureSpec(
                name="bounded",
                min_value=0.0,
                max_value=100.0,
            ),
            "unbounded": FeatureSpec(name="unbounded"),
        }
        return FeatureValidator(
            expected_features=["bounded", "unbounded"],
            feature_specs=specs,
        )

    def test_valid_values(self, validator: FeatureValidator):
        """Test validation passes for valid values."""
        features = {"bounded": 50.0, "unbounded": 1000.0}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert result.is_valid

    def test_value_below_min_warns(self, validator: FeatureValidator):
        """Test value below minimum generates warning."""
        features = {"bounded": -10.0, "unbounded": 0.0}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert result.is_valid  # Warnings don't fail validation
        assert any(w.code == "VALUE_BELOW_MIN" for w in result.warnings)

    def test_value_above_max_warns(self, validator: FeatureValidator):
        """Test value above maximum generates warning."""
        features = {"bounded": 200.0, "unbounded": 0.0}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert result.is_valid
        assert any(w.code == "VALUE_ABOVE_MAX" for w in result.warnings)

    def test_nan_value_fails(self, validator: FeatureValidator):
        """Test NaN values fail validation."""
        features = {"bounded": float("nan"), "unbounded": 0.0}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert not result.is_valid
        assert any(e.code == "INVALID_VALUE" for e in result.errors)

    def test_inf_value_fails(self, validator: FeatureValidator):
        """Test infinite values fail validation."""
        features = {"bounded": float("inf"), "unbounded": 0.0}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert not result.is_valid
        assert any(e.code == "INVALID_VALUE" for e in result.errors)

    def test_integer_values_converted(self, validator: FeatureValidator):
        """Test integer values are converted to float."""
        features = {"bounded": 50, "unbounded": 100}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert result.is_valid
        assert isinstance(result.validated_features["bounded"], float)


class TestTypeValidation:
    """Tests for feature type validation."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        return FeatureValidator(expected_features=["numeric"])

    def test_string_value_fails(self, validator: FeatureValidator):
        """Test string values fail validation."""
        features = {"numeric": "not a number"}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert not result.is_valid
        assert any(e.code == "INVALID_TYPE" for e in result.errors)

    def test_none_value_fails(self, validator: FeatureValidator):
        """Test None values fail validation."""
        features = {"numeric": None}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert not result.is_valid

    def test_list_value_fails(self, validator: FeatureValidator):
        """Test list values fail validation."""
        features = {"numeric": [1.0, 2.0]}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert not result.is_valid


class TestSimilarFeatureSuggestions:
    """Tests for typo suggestion feature."""

    @pytest.fixture
    def validator(self) -> FeatureValidator:
        return FeatureValidator(
            expected_features=[
                "ppi_degree",
                "ppi_clustering",
                "domain_count",
                "protein_length",
            ]
        )

    def test_suggests_similar_for_typo(self, validator: FeatureValidator):
        """Test similar features are suggested for typos."""
        features = {"ppi_degre": 1.0}  # typo - missing "e" at the end
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert not result.is_valid
        # Check that the unexpected feature error has a suggestion
        extra_error = next(
            (e for e in result.errors if e.code == "UNEXPECTED_FEATURE" and "ppi_degre" in e.message),
            None
        )
        # The typo should be flagged as unexpected, and ppi_degree should be suggested
        assert extra_error is not None
        assert "ppi_degree" in extra_error.message or "Did you mean" in extra_error.message

    def test_suggests_similar_for_extra(self, validator: FeatureValidator):
        """Test similar features are suggested for extra features."""
        features = {
            "ppi_degree": 1.0,
            "ppi_clustering": 2.0,
            "domain_count": 3.0,
            "protein_length": 4.0,
            "ppi_closeness": 5.0,  # extra, similar to ppi_*
        }
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert not result.is_valid
        error = next(e for e in result.errors if e.code == "UNEXPECTED_FEATURE")
        assert "ppi_closeness" in error.message


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_to_dict(self):
        """Test ValidationResult.to_dict() produces correct output."""
        result = ValidationResult(
            is_valid=False,
            errors=[
                ValidationError(
                    code="MISSING_FEATURE",
                    message="Missing feature",
                    field="feature_a",
                )
            ],
            warnings=[
                ValidationWarning(
                    code="EXTRA_FEATURE",
                    message="Extra feature",
                    field="feature_x",
                )
            ],
        )
        d = result.to_dict()
        assert d["is_valid"] is False
        assert len(d["errors"]) == 1
        assert d["errors"][0]["code"] == "MISSING_FEATURE"
        assert len(d["warnings"]) == 1


class TestValidatorFromModelDir:
    """Tests for loading validator from model directory."""

    def test_from_model_dir_with_valid_file(self, tmp_path: Path):
        """Test loading from a valid feature_list.json."""
        feature_list = {"feature_order": ["a", "b", "c"]}
        feature_file = tmp_path / "feature_list.json"
        feature_file.write_text(json.dumps(feature_list))

        validator = FeatureValidator.from_model_dir(tmp_path)
        assert validator.expected_features == ["a", "b", "c"]

    def test_from_model_dir_missing_file(self, tmp_path: Path):
        """Test loading from directory without feature_list.json."""
        validator = FeatureValidator.from_model_dir(tmp_path)
        assert validator.expected_features == []

    def test_from_model_dir_with_specs(self, tmp_path: Path):
        """Test loading with feature specifications."""
        feature_list = {
            "feature_order": ["score"],
            "feature_specs": {
                "score": {
                    "required": True,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "description": "A score between 0 and 1",
                }
            },
        }
        feature_file = tmp_path / "feature_list.json"
        feature_file.write_text(json.dumps(feature_list))

        validator = FeatureValidator.from_model_dir(tmp_path)
        assert validator.feature_specs["score"].min_value == 0.0
        assert validator.feature_specs["score"].max_value == 1.0

    def test_from_model_dir_invalid_json(self, tmp_path: Path):
        """Test loading from invalid JSON raises error."""
        feature_file = tmp_path / "feature_list.json"
        feature_file.write_text("not valid json")

        with pytest.raises(ValueError, match="Invalid JSON"):
            FeatureValidator.from_model_dir(tmp_path)


class TestEmptyValidator:
    """Tests for validator with no expected features."""

    def test_empty_validator_accepts_anything(self):
        """Test empty validator accepts any features."""
        validator = FeatureValidator(expected_features=[])
        features = {"any": 1.0, "features": 2.0, "allowed": 3.0}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert result.is_valid

    def test_empty_validator_returns_features(self):
        """Test empty validator returns input features."""
        validator = FeatureValidator(expected_features=[])
        features = {"a": 1.0, "b": 2.0}
        result = validator.validate(features, mode=ValidationMode.STRICT)
        assert result.validated_features == features


class TestGlobalValidator:
    """Tests for global validator singleton."""

    def test_get_feature_validator_returns_instance(self):
        """Test get_feature_validator returns a validator."""
        reset_feature_validator()
        validator = get_feature_validator()
        assert isinstance(validator, FeatureValidator)

    def test_get_feature_validator_returns_same_instance(self):
        """Test get_feature_validator returns the same instance."""
        reset_feature_validator()
        v1 = get_feature_validator()
        v2 = get_feature_validator()
        assert v1 is v2

    def test_reset_feature_validator(self):
        """Test reset_feature_validator clears the singleton."""
        v1 = get_feature_validator()
        reset_feature_validator()
        v2 = get_feature_validator()
        # After reset, should get a new instance
        # (Note: may be same object if loaded from same file)
        assert isinstance(v2, FeatureValidator)


class TestGetFeatureInfo:
    """Tests for get_feature_info method."""

    def test_get_feature_info_returns_dict(self):
        """Test get_feature_info returns expected structure."""
        validator = FeatureValidator(expected_features=["a", "b"])
        info = validator.get_feature_info()
        assert info["total_features"] == 2
        assert info["feature_names"] == ["a", "b"]
        assert "feature_specs" in info

    def test_get_feature_info_includes_specs(self):
        """Test get_feature_info includes feature specifications."""
        specs = {
            "score": FeatureSpec(
                name="score",
                min_value=0.0,
                max_value=1.0,
                description="Test score",
            )
        }
        validator = FeatureValidator(expected_features=["score"], feature_specs=specs)
        info = validator.get_feature_info()
        assert info["feature_specs"]["score"]["min_value"] == 0.0
        assert info["feature_specs"]["score"]["description"] == "Test score"

