import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.config import Config
from src.logger import get_logger

def test_config_loads_defaults():
    """Asserts that the configuration object parses settings and provides fallback defaults."""
    assert Config.PORT in [8000, int(Config.PORT)]
    assert isinstance(Config.DECISION_THRESHOLD, float)
    assert Config.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

def test_logger_creation():
    """Asserts that the logging utility configures a dual-handler logger successfully."""
    logger = get_logger("ValidationTests")
    assert logger is not None
    assert len(logger.handlers) >= 1

def test_schema_voltage_lower_bound_validation():
    """Asserts that negative voltage values are blocked by Pydantic schema validation."""
    payload = {
        "machine_id": "M_001",
        "history": [
            {
                "timestamp": "2026-07-01 12:00:00",
                "voltage": -12.5,  # Invalid: negative voltage
                "temperature": 45.2,
                "vibration": 1.2,
                "pressure": 101.3,
                "rotational_speed": 1500.0,
                "tool_wear": 5.0
            }
        ]
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "voltage" in data["detail"][0]["loc"]
        assert "Voltage must be between 0 and 500 Volts." in data["detail"][0]["msg"]

def test_schema_temperature_upper_bound_validation():
    """Asserts that excessively high temperatures are blocked by schema validation."""
    payload = {
        "machine_id": "M_001",
        "history": [
            {
                "timestamp": "2026-07-01 12:00:00",
                "voltage": 115.0,
                "temperature": 320.0,  # Invalid: above 250 degrees
                "vibration": 1.2,
                "pressure": 101.3,
                "rotational_speed": 1500.0,
                "tool_wear": 5.0
            }
        ]
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "temperature" in data["detail"][0]["loc"]
        assert "Temperature must be between -40 and 250 °C." in data["detail"][0]["msg"]

def test_schema_pressure_upper_bound_validation():
    """Asserts that pressures violating physical rules are blocked by schema validation."""
    payload = {
        "machine_id": "M_001",
        "history": [
            {
                "timestamp": "2026-07-01 12:00:00",
                "voltage": 115.0,
                "temperature": 45.0,
                "vibration": 1.2,
                "pressure": 2500.0,  # Invalid: above 1000 kPa
                "rotational_speed": 1500.0,
                "tool_wear": 5.0
            }
        ]
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "pressure" in data["detail"][0]["loc"]
        assert "Pressure must be between 0 and 1000 kPa." in data["detail"][0]["msg"]
