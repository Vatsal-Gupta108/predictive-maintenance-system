# REST API Documentation

This document describes the API endpoints exposed by the FastAPI backend server of the Predictive Maintenance system.

---

## Base Configuration
* **Default Host**: `http://127.0.0.1:8000`
* **CORS**: Enabled for all origins (`*`) to allow Streamlit or web frontend integration.

---

## Endpoints

### 1. Health Status
Checks if the API is online and verifies that the model and preprocessing artifacts are successfully loaded into memory.

* **Method**: `GET`
* **Route**: `/health`
* **Response Model**: `HealthResponse`

#### Example Response
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "healthy_baseline_loaded": true
}
```

---

### 2. Predict Asset Failure
Performs real-time feature engineering and failure probability prediction for a specific machine using a 24-hour historical window.

* **Method**: `POST`
* **Route**: `/predict`
* **Request Model**: `PredictionRequest`
* **Response Model**: `PredictionResponse`

#### Example Request Payload
```json
{
  "machine_id": "M_001",
  "history": [
    {
      "timestamp": "2026-06-30 08:00:00",
      "voltage": 218.4,
      "temperature": 68.2,
      "vibration": 1.25,
      "pressure": 198.5,
      "rotational_speed": 1550.0,
      "tool_wear": 24.3
    },
    ...
  ]
}
```
*(Note: Minimum 24 records recommended to compute rolling 24h averages correctly).*

#### Example Response
```json
{
  "machine_id": "M_001",
  "timestamp": "2026-06-30 08:00:00",
  "failure_probability": 0.285,
  "risk_level": "High",
  "threshold_used": 0.15,
  "top_drivers": [
    ["tool_wear", 0.08982],
    ["temperature_roll_std_24h", 0.02033],
    ["vibration_roll_mean_24h", 0.0183]
  ],
  "recommendation": "CRITICAL ALERT: High probability of breakdown. Schedule immediate maintenance shutdown."
}
```

---

### 3. Batch Predict CSV File
Accepts a CSV file containing machine sensors telemetry, runs feature engineering over the entire dataset, and flags predicted risks.

* **Method**: `POST`
* **Route**: `/predict/batch`
* **Content-Type**: `multipart/form-data`
* **Request Parameter**: `file` (UploadFile)

#### Example Response
```json
{
  "total_records": 1000,
  "warnings_detected": 14,
  "high_risk_alerts": 3,
  "medium_risk_warnings": 11,
  "alerts": [
    {
      "timestamp": "2026-07-15 14:00:00",
      "machine_id": "M_001",
      "failure_probability": 0.185,
      "risk_level": "Medium"
    },
    ...
  ]
}
```
