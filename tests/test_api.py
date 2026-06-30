import pytest
from fastapi.testclient import TestClient
from api.main import app

def test_health_check_endpoint():
    """Verifies that the /health check endpoint returns a 200 status code and status text."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "scaler_loaded" in data

def test_predict_bad_request():
    """Verifies that making a predict request with empty history returns a 422 validation error."""
    # Invalid request body (empty history list)
    payload = {
        "machine_id": "M_001",
        "history": []
    }
    with TestClient(app) as client:
        # This should fail validation since the schema requires items, or return 400
        response = client.post("/predict", json=payload)
        # FastAPI returns 422 for unprocessable entities if validation fails, or 400 as handled by api
        assert response.status_code in [400, 422]
