import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from contextlib import asynccontextmanager
from typing import Dict, Any

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.config import Config
from src.logger import get_logger
from src.exceptions import (
    PredictiveMaintenanceException,
    ModelNotLoadedError,
    DataValidationError,
    InferenceError
)
from src.feature_engineering import FeatureEngineer
from src.explain import ModelExplainer

# Setup structured logger
logger = get_logger("API")

# Global variables for model state
model = None
scaler = None
feature_names = None
healthy_baseline = None
explainer = None
feature_engineer = None

def load_model_artifacts():
    """Helper to load joblib weights on server initialization."""
    global model, scaler, feature_names, healthy_baseline, explainer, feature_engineer
    try:
        logger.info(f"Attempting to load model from: {Config.MODEL_PATH}")
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model file missing at: {Config.MODEL_PATH}")
            
        model = joblib.load(Config.MODEL_PATH)
        scaler = joblib.load(Config.SCALER_PATH)
        feature_names = joblib.load(Config.FEATURE_COLS_PATH)
        healthy_baseline = joblib.load(Config.BASELINE_PATH)
        
        # Instantiate helper classes
        explainer = ModelExplainer(
            model_path=Config.MODEL_PATH,
            scaler_path=Config.SCALER_PATH,
            feature_cols_path=Config.FEATURE_COLS_PATH
        )
        explainer.load_artifacts()
        explainer.baseline_healthy = healthy_baseline
        
        feature_engineer = FeatureEngineer()
        logger.info("Successfully loaded all machine learning model artifacts.")
        
    except Exception as e:
        logger.error(f"Critical error loading model weights: {str(e)}")
        # We don't crash the server immediately, but flags will remain None causing 503s

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager handling startup and shutdown events."""
    logger.info("Initializing API application container...")
    load_model_artifacts()
    yield
    logger.info("Shutting down API application container...")

app = FastAPI(
    title="IoT Predictive Maintenance API",
    description="Enterprise API exposing real-time predictive maintenance warnings.",
    version="1.1.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
@app.exception_handler(PredictiveMaintenanceException)
async def custom_exception_handler(request: Request, exc: PredictiveMaintenanceException):
    """Intercepts custom domain exceptions and returns structured JSON error payloads."""
    logger.error(f"Domain exception on {request.url.path}: {exc.message} (status: {exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": exc.__class__.__name__}
    )

@app.exception_handler(Exception)
async def fallback_exception_handler(request: Request, exc: Exception):
    """Catches unhandled raw runtime exceptions to prevent stack trace leaks."""
    logger.exception(f"Unhandled runtime panic on {request.url.path}:")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred.", "error_type": "RuntimeError"}
    )

# Endpoints
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Diagnostic check confirming database and model states."""
    logger.debug("Health check requested.")
    return HealthResponse(
        status="healthy",
        model_loaded=(model is not None),
        scaler_loaded=(scaler is not None),
        healthy_baseline_loaded=(healthy_baseline is not None)
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_health(request: PredictionRequest):
    """
    Ingests live sensor telemetry histories, calculates rolling metrics,
    and returns failure warnings alongside root drivers.
    """
    logger.info(f"Prediction requested for machine ID: {request.machine_id}")
    
    if model is None or scaler is None or feature_names is None:
        raise ModelNotLoadedError()
        
    if len(request.history) < 1:
        raise DataValidationError("Telemetry history input array cannot be empty.")
        
    try:
        # Convert Pydantic records to pandas
        records = [record.dict() for record in request.history]
        df_raw = pd.DataFrame(records)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw["machine_id"] = request.machine_id
        
        # Calculate features on the fly
        df_feat = feature_engineer.transform(df_raw)
        
        latest_record = df_feat.iloc[-1]
        latest_timestamp_str = str(latest_record["timestamp"])
        
        # Extract features and scale
        X_df = latest_record[feature_names].to_frame().T
        X_scaled = scaler.transform(X_df)
        
        # Predict probability
        failure_prob = float(model.predict_proba(X_scaled)[0, 1])
        
        # Risk thresholds mapping
        if failure_prob >= 0.20:
            risk_level = "High"
            recommendation = "CRITICAL ALERT: High probability of breakdown. Schedule immediate maintenance shutdown."
            logger.warning(f"CRITICAL failure warning issued for {request.machine_id} (prob: {failure_prob:.4f})")
        elif failure_prob >= Config.DECISION_THRESHOLD:
            risk_level = "Medium"
            recommendation = "WARNING: Sensor anomalies detected. Schedule preventative check and maintenance."
            logger.warning(f"Anomalous maintenance warning issued for {request.machine_id} (prob: {failure_prob:.4f})")
        else:
            risk_level = "Low"
            recommendation = "HEALTHY: Machine operating normally. Continue standard monitoring schedule."
            logger.info(f"Machine {request.machine_id} evaluated healthy (prob: {failure_prob:.4f})")
            
        # Compute local attributions
        latest_features_dict = latest_record[feature_names].to_dict()
        explanation = explainer.explain_instance(latest_features_dict)
        top_drivers = [(feat, float(score)) for feat, score in explanation["top_drivers"]]
        
        return PredictionResponse(
            machine_id=request.machine_id,
            timestamp=latest_timestamp_str,
            failure_probability=failure_prob,
            risk_level=risk_level,
            threshold_used=Config.DECISION_THRESHOLD,
            top_drivers=top_drivers,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.exception("Failed execution of inference engine:")
        raise InferenceError(f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
def predict_batch_csv(file: UploadFile = File(...)):
    """Accepts telemetry logs uploads, runs batch predictions, and summarizes risk windows."""
    logger.info(f"Batch prediction CSV uploaded: {file.filename}")
    
    if model is None or scaler is None or feature_names is None:
        raise ModelNotLoadedError()
        
    try:
        contents = file.file.read()
        if not contents:
            raise DataValidationError("Uploaded CSV file is empty.")
            
        df_raw = pd.read_csv(BytesIO(contents))
        
        # Check columns
        required_cols = ["timestamp", "machine_id", "voltage", "temperature", "vibration", "pressure", "rotational_speed", "tool_wear"]
        missing = [col for col in required_cols if col not in df_raw.columns]
        if missing:
            raise DataValidationError(f"Missing required columns in batch CSV: {missing}")
            
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        
        # Run features
        df_feat = feature_engineer.transform(df_raw)
        
        # Separate features and predict
        X_df = df_feat[feature_names]
        X_scaled = scaler.transform(X_df)
        
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= Config.DECISION_THRESHOLD).astype(int)
        
        df_feat["failure_probability"] = probs
        df_feat["predicted_risk"] = preds
        df_feat["risk_level"] = np.where(probs >= 0.20, "High", np.where(probs >= Config.DECISION_THRESHOLD, "Medium", "Low"))
        
        # Filter warnings
        warnings_df = df_feat[df_feat["predicted_risk"] == 1][["timestamp", "machine_id", "failure_probability", "risk_level"]]
        
        logger.info(f"Batch processing completed. Identified {len(warnings_df)} alerts.")
        
        return {
            "total_records": len(df_raw),
            "warnings_detected": len(warnings_df),
            "high_risk_alerts": int(sum(df_feat["risk_level"] == "High")),
            "medium_risk_warnings": int(sum(df_feat["risk_level"] == "Medium")),
            "alerts": warnings_df.head(100).to_dict(orient="records")
        }
        
    except PredictiveMaintenanceException:
        raise
    except Exception as e:
        logger.exception("Failed execution of batch inference engine:")
        raise InferenceError(f"Batch inference failed: {str(e)}")
