import pytest
from fastapi.testclient import TestClient
from oncotarget_lite.model_server import app
from oncotarget_lite.schemas import APIPredictionResponse

@pytest.fixture(scope="module")
def test_client():
    """Create a TestClient instance for the FastAPI app."""
    with TestClient(app) as client:
        yield client

def test_predict_contract(test_client):
    """
    Tests the contract of the /predict endpoint.
    This test ensures that the request and response schemas are not accidentally changed.
    """
    # 1. Define the canonical request payload
    # This should be a valid request that represents a typical use case.
    # We need to provide a dictionary of features. In a real scenario, you'd
    # use a realistic set of features. For this test, we'll use placeholders.
    canonical_request = {
        "features": {
            "feature1": 0.5,
            "feature2": 1.2,
            "feature3": -0.8
        },
        "model_version": "1.0.0"
    }

    # 2. Make the API call
    response = test_client.post("/predict", json=canonical_request)

    # 3. Assert the response contract
    assert response.status_code == 200
    response_data = response.json()

    # 4. Validate the response schema using the Pydantic model
    # This will raise a ValidationError if the response does not match the schema,
    # which will cause the test to fail.
    try:
        APIPredictionResponse(**response_data)
    except Exception as e:
        pytest.fail(f"Response schema validation failed: {e}\nResponse data: {response_data}")

    # 5. Optionally, assert specific fields for correctness if needed
    assert "prediction" in response_data
    assert "model_version" in response_data
    assert isinstance(response_data["prediction"], float)
    assert isinstance(response_data["model_version"], str)

def test_health_check_contract(test_client):
    """
    Tests the contract of the /health endpoint.
    """
    response = test_client.get("/health")
    assert response.status_code == 200
    response_data = response.json()

    assert "status" in response_data
    assert response_data["status"] == "ok"
    assert "model_status" in response_data
    assert "last_updated" in response_data














