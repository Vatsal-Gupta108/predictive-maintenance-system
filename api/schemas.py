from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Tuple

class TelemetryRecord(BaseModel):
    timestamp: str = Field(..., description="Timestamp of the reading in YYYY-MM-DD HH:MM:SS format")
    voltage: float = Field(..., description="Voltage reading in Volts")
    temperature: float = Field(..., description="Temperature reading in °C")
    vibration: float = Field(..., description="Vibration reading in mm/s")
    pressure: float = Field(..., description="Pressure reading in kPa")
    rotational_speed: float = Field(..., description="Rotational speed in RPM")
    tool_wear: float = Field(..., description="Cumulative tool wear metric")

    @field_validator("voltage")
    @classmethod
    def validate_voltage(cls, v):
        if v < 0 or v > 500:
            raise ValueError("Voltage must be between 0 and 500 Volts.")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v < -40 or v > 250:
            raise ValueError("Temperature must be between -40 and 250 °C.")
        return v

    @field_validator("vibration")
    @classmethod
    def validate_vibration(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Vibration must be between 0 and 100 mm/s.")
        return v

    @field_validator("pressure")
    @classmethod
    def validate_pressure(cls, v):
        if v < 0 or v > 1000:
            raise ValueError("Pressure must be between 0 and 1000 kPa.")
        return v

    @field_validator("rotational_speed")
    @classmethod
    def validate_rotational_speed(cls, v):
        if v < 0 or v > 10000:
            raise ValueError("Rotational speed must be between 0 and 10000 RPM.")
        return v

    @field_validator("tool_wear")
    @classmethod
    def validate_tool_wear(cls, v):
        if v < 0 or v > 1000:
            raise ValueError("Tool wear must be between 0 and 1000.")
        return v

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
