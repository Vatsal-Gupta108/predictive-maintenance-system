import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse, TelemetryRecord
from src.feature_engineering import FeatureEngineer
from src.explain import ModelExplainer

app = FastAPI(
    title="IoT Predictive Maintenance API",
    description="API for real-time machine health monitoring and breakdown predictions.",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
model = None
scaler = None
feature_names = None
healthy_baseline = None
explainer = None
feature_engineer = None

# Optimized decision threshold for imbalanced classes
DECISION_THRESHOLD = 0.15

@app.on_event("startup")
def load_model_artifacts():
    global model, scaler, feature_names, healthy_baseline, explainer, feature_engineer
    try:
        model_path = "models/best_model.joblib"
        scaler_path = "models/scaler.joblib"
        feature_cols_path = "models/feature_cols.joblib"
        baseline_path = "models/baseline_healthy.joblib"
        
        # Verify paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing: {model_path}")
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_cols_path)
        healthy_baseline = joblib.load(baseline_path)
        
        # Instantiate helper classes
        explainer = ModelExplainer()
        explainer.load_artifacts()
        explainer.baseline_healthy = healthy_baseline
        
        feature_engineer = FeatureEngineer()
        print("API startup: Model and artifacts successfully loaded.")
        
    except Exception as e:
        print(f"CRITICAL: Failed to load model artifacts on startup: {str(e)}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns the operational status of the API and model weights."""
    return HealthResponse(
        status="healthy",
        model_loaded=(model is not None),
        scaler_loaded=(scaler is not None),
        healthy_baseline_loaded=(healthy_baseline is not None)
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_health(request: PredictionRequest):
    """
    Accepts historical sensor data for a machine, engineers rolling features
    on-the-fly, and predicts failure probability and contributors.
    """
    if model is None or scaler is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded on server.")
        
    if len(request.history) < 1:
        raise HTTPException(status_code=400, detail="Telemetry history cannot be empty.")
        
    try:
        # 1. Convert request data to DataFrame
        records = [record.dict() for record in request.history]
        df_raw = pd.DataFrame(records)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw["machine_id"] = request.machine_id
        
        # 2. Run Feature Engineering
        # To compute rolling features correctly, we need the history
        df_feat = feature_engineer.transform(df_raw)
        
        # Extract the latest record (which represents the current time step we want to predict)
        latest_record = df_feat.iloc[-1]
        latest_timestamp_str = str(latest_record["timestamp"])
        
        # 3. Format and scale features
        X_df = latest_record[feature_names].to_frame().T
        X_scaled = scaler.transform(X_df)
        
        # 4. Model Inference
        failure_prob = float(model.predict_proba(X_scaled)[0, 1])
        
        # 5. Risk and Recommendations
        if failure_prob >= 0.20:
            risk_level = "High"
            recommendation = "CRITICAL ALERT: High probability of breakdown. Schedule immediate maintenance shutdown."
        elif failure_prob >= DECISION_THRESHOLD:
            risk_level = "Medium"
            recommendation = "WARNING: Sensor anomalies detected. Schedule preventative check and maintenance."
        else:
            risk_level = "Low"
            recommendation = "HEALTHY: Machine operating normally. Continue standard monitoring schedule."
            
        # 6. Local Explainability
        latest_features_dict = latest_record[feature_names].to_dict()
        explanation = explainer.explain_instance(latest_features_dict)
        top_drivers = [(feat, float(score)) for feat, score in explanation["top_drivers"]]
        
        return PredictionResponse(
            machine_id=request.machine_id,
            timestamp=latest_timestamp_str,
            failure_probability=failure_prob,
            risk_level=risk_level,
            threshold_used=DECISION_THRESHOLD,
            top_drivers=top_drivers,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_batch_csv(file: UploadFile = File(...)):
    """
    Accepts an uploaded CSV file containing telemetry, runs full feature
    engineering, and returns failure warnings and timestamps.
    """
    if model is None or scaler is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
        
    try:
        contents = file.file.read()
        df_raw = pd.read_csv(BytesIO(contents))
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        
        # Run features
        df_feat = feature_engineer.transform(df_raw)
        
        # Separate features
        X_df = df_feat[feature_names]
        X_scaled = scaler.transform(X_df)
        
        # Predict
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= DECISION_THRESHOLD).astype(int)
        
        df_feat["failure_probability"] = probs
        df_feat["predicted_risk"] = preds
        df_feat["risk_level"] = np.where(probs >= 0.20, "High", np.where(probs >= DECISION_THRESHOLD, "Medium", "Low"))
        
        # Filter high and medium risk timestamps for reporting
        warnings_df = df_feat[df_feat["predicted_risk"] == 1][["timestamp", "machine_id", "failure_probability", "risk_level"]]
        
        summary = {
            "total_records": len(df_raw),
            "warnings_detected": len(warnings_df),
            "high_risk_alerts": int(sum(df_feat["risk_level"] == "High")),
            "medium_risk_warnings": int(sum(df_feat["risk_level"] == "Medium")),
            "alerts": warnings_df.head(100).to_dict(orient="records")
        }
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
