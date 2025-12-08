"""Feature validation for API requests.

This module provides validation of input features against the trained model's
expected feature schema. It catches mismatches early to prevent silent failures
or incorrect predictions.

Key features:
- Loads expected features from the model's feature_list.json
- Three validation modes: strict, lenient, permissive
- Detailed error messages for debugging
- Feature value range validation
- Missing feature detection with suggestions

Usage:
    from oncotarget_lite.feature_validation import FeatureValidator, ValidationMode

    validator = FeatureValidator.from_model_dir(Path("models"))
    result = validator.validate(features, mode=ValidationMode.STRICT)
    if not result.is_valid:
        print(result.errors)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from oncotarget_lite.logging_config import get_logger

logger = get_logger(__name__)


class ValidationMode(str, Enum):
    """Validation strictness modes.

    STRICT: All expected features must be present, no extra features allowed
    LENIENT: All expected features must be present, extra features are warned but allowed
    PERMISSIVE: Missing features are filled with defaults, extra features ignored
    """

    STRICT = "strict"
    LENIENT = "lenient"
    PERMISSIVE = "permissive"


@dataclass
class ValidationError:
    """A single validation error."""

    code: str
    message: str
    field: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {"code": self.code, "message": self.message}
        if self.field:
            result["field"] = self.field
        if self.expected is not None:
            result["expected"] = self.expected
        if self.actual is not None:
            result["actual"] = self.actual
        return result


@dataclass
class ValidationWarning:
    """A validation warning (non-fatal)."""

    code: str
    message: str
    field: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of feature validation."""

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    validated_features: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [{"code": w.code, "message": w.message, "field": w.field} for w in self.warnings],
        }


@dataclass
class FeatureSpec:
    """Specification for a single feature."""

    name: str
    required: bool = True
    default_value: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: Optional[str] = None


class FeatureValidator:
    """Validates input features against the model's expected schema.

    This class loads the feature specification from the model directory
    and validates incoming prediction requests to ensure they match
    what the model expects.
    """

    def __init__(
        self,
        expected_features: List[str],
        feature_specs: Optional[Dict[str, FeatureSpec]] = None,
    ):
        """Initialize the validator.

        Args:
            expected_features: Ordered list of feature names the model expects
            feature_specs: Optional detailed specifications for each feature
        """
        self.expected_features = expected_features
        self.expected_feature_set = set(expected_features)
        self.feature_specs = feature_specs or {}

        # Build default specs for features without explicit specs
        for feat in expected_features:
            if feat not in self.feature_specs:
                self.feature_specs[feat] = FeatureSpec(name=feat)

        logger.info(
            "feature_validator_initialized",
            feature_count=len(expected_features),
            has_specs=len(feature_specs) if feature_specs else 0,
        )

    @classmethod
    def from_model_dir(cls, models_dir: Path = Path("models")) -> "FeatureValidator":
        """Load validator from model directory.

        Args:
            models_dir: Path to models directory containing feature_list.json

        Returns:
            Configured FeatureValidator instance
        """
        feature_list_path = models_dir / "feature_list.json"

        if not feature_list_path.exists():
            logger.warning(
                "feature_list_not_found",
                path=str(feature_list_path),
                message="Using empty feature list - validation will be permissive",
            )
            return cls(expected_features=[])

        try:
            with open(feature_list_path) as f:
                data = json.load(f)

            expected_features = data.get("feature_order", [])

            # Load feature specs if available
            feature_specs = {}
            if "feature_specs" in data:
                for name, spec_data in data["feature_specs"].items():
                    feature_specs[name] = FeatureSpec(
                        name=name,
                        required=spec_data.get("required", True),
                        default_value=spec_data.get("default_value", 0.0),
                        min_value=spec_data.get("min_value"),
                        max_value=spec_data.get("max_value"),
                        description=spec_data.get("description"),
                    )

            logger.info(
                "feature_list_loaded",
                path=str(feature_list_path),
                feature_count=len(expected_features),
            )

            return cls(expected_features=expected_features, feature_specs=feature_specs)

        except json.JSONDecodeError as e:
            logger.error(
                "feature_list_parse_error",
                path=str(feature_list_path),
                error=str(e),
            )
            raise ValueError(f"Invalid JSON in feature_list.json: {e}")

    def validate(
        self,
        features: Dict[str, Any],
        mode: ValidationMode = ValidationMode.LENIENT,
    ) -> ValidationResult:
        """Validate input features against the expected schema.

        Args:
            features: Dictionary of feature name -> value
            mode: Validation strictness mode

        Returns:
            ValidationResult with errors, warnings, and validated features
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationWarning] = []

        # If no expected features, allow anything (model not yet trained)
        if not self.expected_features:
            logger.debug("validation_skipped", reason="no_expected_features")
            return ValidationResult(
                is_valid=True,
                validated_features=dict(features),
            )

        provided_features = set(features.keys())

        # Check for missing features
        missing_features = self.expected_feature_set - provided_features
        if missing_features:
            if mode == ValidationMode.PERMISSIVE:
                # Fill with defaults
                for feat in missing_features:
                    spec = self.feature_specs.get(feat, FeatureSpec(name=feat))
                    features[feat] = spec.default_value
                    warnings.append(
                        ValidationWarning(
                            code="MISSING_FEATURE_FILLED",
                            message=f"Missing feature '{feat}' filled with default value {spec.default_value}",
                            field=feat,
                        )
                    )
            else:
                # Report as errors
                for feat in sorted(missing_features):
                    similar = self._find_similar_features(feat, provided_features)
                    suggestion = f" Did you mean: {', '.join(similar)}?" if similar else ""
                    errors.append(
                        ValidationError(
                            code="MISSING_FEATURE",
                            message=f"Missing required feature: '{feat}'.{suggestion}",
                            field=feat,
                            expected="present",
                            actual="missing",
                        )
                    )

        # Check for extra features
        extra_features = provided_features - self.expected_feature_set
        if extra_features:
            if mode == ValidationMode.STRICT:
                for feat in sorted(extra_features):
                    similar = self._find_similar_features(feat, self.expected_feature_set)
                    suggestion = f" Did you mean: {', '.join(similar)}?" if similar else ""
                    errors.append(
                        ValidationError(
                            code="UNEXPECTED_FEATURE",
                            message=f"Unexpected feature: '{feat}'.{suggestion}",
                            field=feat,
                        )
                    )
            else:
                for feat in sorted(extra_features):
                    warnings.append(
                        ValidationWarning(
                            code="EXTRA_FEATURE",
                            message=f"Extra feature '{feat}' will be ignored",
                            field=feat,
                        )
                    )

        # Validate feature values
        for feat, value in features.items():
            if feat not in self.expected_feature_set:
                continue  # Skip extra features

            # Type validation
            if not isinstance(value, (int, float)):
                errors.append(
                    ValidationError(
                        code="INVALID_TYPE",
                        message=f"Feature '{feat}' must be numeric, got {type(value).__name__}",
                        field=feat,
                        expected="float",
                        actual=type(value).__name__,
                    )
                )
                continue

            # Convert to float
            features[feat] = float(value)

            # Range validation
            spec = self.feature_specs.get(feat)
            if spec:
                if spec.min_value is not None and value < spec.min_value:
                    warnings.append(
                        ValidationWarning(
                            code="VALUE_BELOW_MIN",
                            message=f"Feature '{feat}' value {value} is below minimum {spec.min_value}",
                            field=feat,
                        )
                    )
                if spec.max_value is not None and value > spec.max_value:
                    warnings.append(
                        ValidationWarning(
                            code="VALUE_ABOVE_MAX",
                            message=f"Feature '{feat}' value {value} is above maximum {spec.max_value}",
                            field=feat,
                        )
                    )

            # Check for NaN/Inf
            import math
            if math.isnan(value) or math.isinf(value):
                errors.append(
                    ValidationError(
                        code="INVALID_VALUE",
                        message=f"Feature '{feat}' has invalid value: {value}",
                        field=feat,
                        expected="finite number",
                        actual=str(value),
                    )
                )

        # Build validated features in expected order
        validated_features = None
        if not errors or mode == ValidationMode.PERMISSIVE:
            validated_features = {}
            for feat in self.expected_features:
                if feat in features:
                    validated_features[feat] = float(features[feat])
                elif mode == ValidationMode.PERMISSIVE:
                    spec = self.feature_specs.get(feat, FeatureSpec(name=feat))
                    validated_features[feat] = spec.default_value

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(
                "feature_validation_failed",
                error_count=len(errors),
                warning_count=len(warnings),
                mode=mode.value,
            )
        elif warnings:
            logger.info(
                "feature_validation_passed_with_warnings",
                warning_count=len(warnings),
                mode=mode.value,
            )

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validated_features=validated_features,
        )

    def _find_similar_features(
        self, feature: str, candidates: Set[str], max_suggestions: int = 3
    ) -> List[str]:
        """Find features similar to the given name (for typo suggestions)."""
        if not candidates:
            return []

        # Simple similarity: common prefix or suffix, or edit distance
        feature_lower = feature.lower()
        scored = []

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Exact substring match
            if feature_lower in candidate_lower or candidate_lower in feature_lower:
                scored.append((candidate, 0))
                continue

            # Common prefix
            common_prefix = 0
            for a, b in zip(feature_lower, candidate_lower):
                if a == b:
                    common_prefix += 1
                else:
                    break

            # Common suffix
            common_suffix = 0
            for a, b in zip(reversed(feature_lower), reversed(candidate_lower)):
                if a == b:
                    common_suffix += 1
                else:
                    break

            # Score based on common characters
            score = max(common_prefix, common_suffix)
            if score >= 3:  # At least 3 characters in common
                scored.append((candidate, -score))

        # Sort by score (higher is better, so negate)
        scored.sort(key=lambda x: x[1])
        return [name for name, _ in scored[:max_suggestions]]

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about expected features for documentation."""
        return {
            "total_features": len(self.expected_features),
            "feature_names": self.expected_features,
            "feature_specs": {
                name: {
                    "required": spec.required,
                    "default_value": spec.default_value,
                    "min_value": spec.min_value,
                    "max_value": spec.max_value,
                    "description": spec.description,
                }
                for name, spec in self.feature_specs.items()
            },
        }


# Global validator instance (lazy-loaded)
_validator: Optional[FeatureValidator] = None


def get_feature_validator() -> FeatureValidator:
    """Get the global feature validator instance."""
    global _validator
    if _validator is None:
        _validator = FeatureValidator.from_model_dir()
    return _validator


def reset_feature_validator() -> None:
    """Reset the global validator (useful for testing)."""
    global _validator
    _validator = None


__all__ = [
    "FeatureValidator",
    "ValidationMode",
    "ValidationResult",
    "ValidationError",
    "ValidationWarning",
    "FeatureSpec",
    "get_feature_validator",
    "reset_feature_validator",
]






