from pydantic import BaseModel, Field
from typing import List, Dict, Tuple

class TelemetryRecord(BaseModel):
    timestamp: str = Field(..., description="Timestamp of the reading in YYYY-MM-DD HH:MM:SS format")
    voltage: float = Field(..., description="Voltage reading in Volts")
    temperature: float = Field(..., description="Temperature reading in °C")
    vibration: float = Field(..., description="Vibration reading in mm/s")
    pressure: float = Field(..., description="Pressure reading in kPa")
    rotational_speed: float = Field(..., description="Rotational speed in RPM")
    tool_wear: float = Field(..., description="Cumulative tool wear metric")

class PredictionRequest(BaseModel):
    machine_id: str = Field("M_001", description="Identifier of the machine")
    history: List[TelemetryRecord] = Field(..., description="Telemetry history (minimum 24 records recommended for rolling stats)")

class PredictionResponse(BaseModel):
    machine_id: str
    timestamp: str
    failure_probability: float = Field(..., description="Probability of failure within the next 24 hours")
    risk_level: str = Field(..., description="Risk tier: Low (Green), Medium (Yellow), High (Red)")
    threshold_used: float = Field(..., description="Probability decision threshold")
    top_drivers: List[Tuple[str, float]] = Field(..., description="Top features contributing to failure risk")
    recommendation: str = Field(..., description="Suggested operator action")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    healthy_baseline_loaded: bool
