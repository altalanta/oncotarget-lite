"""
Comprehensive API Integration and Contract Tests.

This module provides thorough testing of the FastAPI model server, including:
- Contract tests for all endpoints to prevent breaking changes.
- Error handling tests to ensure graceful failure.
- Schema validation tests using Pydantic models.
- Integration tests for the Prometheus metrics endpoint.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to sys.path so 'deployment' module can be found
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oncotarget_lite.schemas import (
    APIPredictionRequest,
    APIPredictionResponse,
    APIExplanationResponse,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def mock_prediction_service():
    """Create a mock prediction service."""
    mock_service = MagicMock()
    mock_service.predict_single.return_value = {
        "prediction": 0.85,
        "model_version": "test-v1.0.0",
    }
    mock_service.explain_single.return_value = {
        "model_version": "test-v1.0.0",
        "feature_contributions": {"feature1": 0.5, "feature2": -0.3},
    }
    mock_service.health_check.return_value = {
        "status": "ok",
        "model_status": "loaded",
        "explainer_status": "loaded",
        "last_updated": "2024-01-01T00:00:00",
    }
    return mock_service


@pytest.fixture(scope="module")
def test_client(mock_prediction_service):
    """
    Create a TestClient instance for the FastAPI app.

    We patch the PredictionService before importing the app module
    to prevent it from trying to load a real model during testing.
    """
    from fastapi.testclient import TestClient

    # Create a mock PredictionService class
    mock_service_class = MagicMock(return_value=mock_prediction_service)

    # We need to patch where PredictionService is imported, not where it's defined
    # The model_server imports it as: from deployment.prediction_service import PredictionService
    # So we need to patch it at the deployment.prediction_service module level

    # First, create a mock module for deployment.prediction_service
    mock_deployment_module = MagicMock()
    mock_deployment_module.PredictionService = mock_service_class

    # Mock the resilience manager too
    mock_resilience = MagicMock()
    mock_resilience_module = MagicMock()
    mock_resilience_module.get_resilience_manager = MagicMock(return_value=mock_resilience)

    # Inject the mock modules into sys.modules BEFORE importing model_server
    with patch.dict(sys.modules, {
        'deployment': MagicMock(),
        'deployment.prediction_service': mock_deployment_module,
    }):
        # Clear any cached import of model_server
        modules_to_clear = [k for k in list(sys.modules.keys()) if 'model_server' in k]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Also patch the resilience manager
        with patch('oncotarget_lite.resilience.get_resilience_manager', return_value=mock_resilience):
            # Now import the app - it will use our mocked modules
            from oncotarget_lite.model_server import app

            # The prediction_service global variable was created at import time
            # We need to replace it with our mock
            import oncotarget_lite.model_server as server_module
            server_module.prediction_service = mock_prediction_service

            with TestClient(app) as client:
                yield client


@pytest.fixture
def canonical_request_payload() -> dict:
    """A valid, canonical request payload for testing."""
    return {
        "features": {
            "feature1": 0.5,
            "feature2": 1.2,
            "feature3": -0.8,
        },
        "model_version": "1.0.0",
    }


# =============================================================================
# /predict ENDPOINT TESTS
# =============================================================================

class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_returns_200_on_valid_request(
        self, test_client, canonical_request_payload: dict
    ):
        """A valid request should return a 200 OK status."""
        response = test_client.post("/predict", json=canonical_request_payload)
        assert response.status_code == 200

    def test_predict_response_matches_schema(
        self, test_client, canonical_request_payload: dict
    ):
        """The response body should match the APIPredictionResponse schema."""
        response = test_client.post("/predict", json=canonical_request_payload)
        response_data = response.json()

        # This will raise a ValidationError if the schema doesn't match
        validated_response = APIPredictionResponse(**response_data)

        assert validated_response.prediction == response_data["prediction"]
        assert validated_response.model_version == response_data["model_version"]

    def test_predict_response_contains_required_fields(
        self, test_client, canonical_request_payload: dict
    ):
        """The response must contain 'prediction' and 'model_version' fields."""
        response = test_client.post("/predict", json=canonical_request_payload)
        response_data = response.json()

        assert "prediction" in response_data
        assert "model_version" in response_data

    def test_predict_prediction_is_a_float(
        self, test_client, canonical_request_payload: dict
    ):
        """The 'prediction' field must be a float."""
        response = test_client.post("/predict", json=canonical_request_payload)
        response_data = response.json()
        assert isinstance(response_data["prediction"], float)

    def test_predict_returns_422_on_missing_features(self, test_client):
        """A request without the 'features' field should return 422 Unprocessable Entity."""
        invalid_payload = {"model_version": "1.0.0"}
        response = test_client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_predict_returns_422_on_invalid_feature_type(self, test_client):
        """A request with non-float feature values should return 422."""
        invalid_payload = {
            "features": {"feature1": "not_a_number"},
        }
        response = test_client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_predict_accepts_request_without_model_version(self, test_client):
        """The 'model_version' field is optional in the request."""
        payload = {"features": {"feature1": 0.5}}
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_returns_422_on_empty_body(self, test_client):
        """An empty request body should return 422."""
        response = test_client.post("/predict", json={})
        assert response.status_code == 422


# =============================================================================
# /explain ENDPOINT TESTS
# =============================================================================

class TestExplainEndpoint:
    """Tests for the /explain endpoint."""

    def test_explain_returns_200_on_valid_request(
        self, test_client, canonical_request_payload: dict
    ):
        """A valid request should return a 200 OK status."""
        response = test_client.post("/explain", json=canonical_request_payload)
        assert response.status_code == 200

    def test_explain_response_matches_schema(
        self, test_client, canonical_request_payload: dict
    ):
        """The response body should match the APIExplanationResponse schema."""
        response = test_client.post("/explain", json=canonical_request_payload)
        response_data = response.json()

        validated_response = APIExplanationResponse(**response_data)

        assert validated_response.model_version == response_data["model_version"]
        assert (
            validated_response.feature_contributions
            == response_data["feature_contributions"]
        )

    def test_explain_response_contains_required_fields(
        self, test_client, canonical_request_payload: dict
    ):
        """The response must contain 'model_version' and 'feature_contributions'."""
        response = test_client.post("/explain", json=canonical_request_payload)
        response_data = response.json()

        assert "model_version" in response_data
        assert "feature_contributions" in response_data

    def test_explain_feature_contributions_is_a_dict(
        self, test_client, canonical_request_payload: dict
    ):
        """The 'feature_contributions' field must be a dictionary."""
        response = test_client.post("/explain", json=canonical_request_payload)
        response_data = response.json()
        assert isinstance(response_data["feature_contributions"], dict)

    def test_explain_returns_422_on_missing_features(self, test_client):
        """A request without 'features' should return 422."""
        invalid_payload = {"model_version": "1.0.0"}
        response = test_client.post("/explain", json=invalid_payload)
        assert response.status_code == 422


# =============================================================================
# /health ENDPOINT TESTS
# =============================================================================

class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, test_client):
        """The health endpoint should always return 200 OK."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_contains_status_ok(self, test_client):
        """The response should contain 'status': 'ok' when healthy."""
        response = test_client.get("/health")
        response_data = response.json()
        assert response_data.get("status") == "ok"

    def test_health_response_contains_model_status(self, test_client):
        """The response should report the model's loading status."""
        response = test_client.get("/health")
        response_data = response.json()
        assert "model_status" in response_data

    def test_health_response_contains_explainer_status(self, test_client):
        """The response should report the explainer's loading status."""
        response = test_client.get("/health")
        response_data = response.json()
        assert "explainer_status" in response_data

    def test_health_response_contains_last_updated(self, test_client):
        """The response should contain a 'last_updated' timestamp."""
        response = test_client.get("/health")
        response_data = response.json()
        assert "last_updated" in response_data


# =============================================================================
# /metrics ENDPOINT TESTS (Prometheus)
# =============================================================================

class TestMetricsEndpoint:
    """Tests for the Prometheus /metrics endpoint."""

    def test_metrics_returns_200(self, test_client):
        """The metrics endpoint should return 200 OK."""
        response = test_client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_prometheus_format(self, test_client):
        """The response should be in Prometheus text format."""
        response = test_client.get("/metrics")
        # Prometheus metrics are text/plain or a specific openmetrics type
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type or "openmetrics" in content_type

    def test_metrics_contains_http_request_metrics(self, test_client):
        """The metrics should include standard HTTP request metrics."""
        # Make a request first to generate some metrics
        test_client.get("/health")

        response = test_client.get("/metrics")
        metrics_text = response.text

        # Check for common HTTP metrics from the instrumentator
        assert "http_requests_total" in metrics_text or "http_request" in metrics_text

    def test_metrics_contains_handler_labels(self, test_client):
        """
        The metrics should include handler labels for tracking endpoint performance.
        This is useful for monitoring specific endpoint latencies.
        """
        response = test_client.get("/metrics")
        metrics_text = response.text

        # Check that handler labels are present (standard prometheus-fastapi-instrumentator behavior)
        assert "handler=" in metrics_text


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for API error handling."""

    def test_predict_handles_service_error(
        self, test_client, mock_prediction_service
    ):
        """
        If the prediction service raises an internal error, the API should
        return a 500 Internal Server Error.
        """
        from oncotarget_lite.exceptions import PredictionError

        # Temporarily make the mock raise an exception
        original_return = mock_prediction_service.predict_single.return_value
        mock_prediction_service.predict_single.side_effect = PredictionError("Model failed")

        payload = {"features": {"feature1": 0.5}}
        response = test_client.post("/predict", json=payload)

        # Reset the mock for other tests
        mock_prediction_service.predict_single.side_effect = None
        mock_prediction_service.predict_single.return_value = original_return

        assert response.status_code == 500

    def test_explain_handles_service_error(
        self, test_client, mock_prediction_service
    ):
        """
        If the explanation service raises an internal error, the API should
        return a 500 Internal Server Error.
        """
        from oncotarget_lite.exceptions import PredictionError

        original_return = mock_prediction_service.explain_single.return_value
        mock_prediction_service.explain_single.side_effect = PredictionError("Explainer failed")

        payload = {"features": {"feature1": 0.5}}
        response = test_client.post("/explain", json=payload)

        # Reset the mock
        mock_prediction_service.explain_single.side_effect = None
        mock_prediction_service.explain_single.return_value = original_return

        assert response.status_code == 500

    def test_invalid_json_returns_422(self, test_client):
        """Sending malformed JSON should return a 422 error."""
        response = test_client.post(
            "/predict",
            content="this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_nonexistent_endpoint_returns_404(self, test_client):
        """Requesting a non-existent endpoint should return 404."""
        response = test_client.get("/nonexistent")
        assert response.status_code == 404


# =============================================================================
# SCHEMA VALIDATION TESTS (Pydantic)
# =============================================================================

class TestSchemaValidation:
    """Tests for Pydantic schema validation."""

    def test_prediction_request_schema_valid(self):
        """A valid payload should pass schema validation."""
        payload = {"features": {"a": 1.0, "b": 2.0}}
        request = APIPredictionRequest(**payload)
        assert request.features == {"a": 1.0, "b": 2.0}

    def test_prediction_request_schema_optional_model_version(self):
        """The model_version field should be optional."""
        payload = {"features": {"a": 1.0}}
        request = APIPredictionRequest(**payload)
        assert request.model_version is None

    def test_prediction_response_schema_valid(self):
        """A valid response payload should pass schema validation."""
        payload = {"prediction": 0.95, "model_version": "v1"}
        response = APIPredictionResponse(**payload)
        assert response.prediction == 0.95

    def test_explanation_response_schema_valid(self):
        """A valid explanation response should pass schema validation."""
        payload = {
            "model_version": "v1",
            "feature_contributions": {"feat1": 0.1, "feat2": -0.2},
        }
        response = APIExplanationResponse(**payload)
        assert response.feature_contributions == {"feat1": 0.1, "feat2": -0.2}
