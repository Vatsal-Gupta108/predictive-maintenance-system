import os
from dotenv import load_dotenv

# Load settings from .env file if it exists
load_dotenv()

class Config:
    """Configuration class that loads and validates environment variables."""
    
    # API Settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DECISION_THRESHOLD: float = float(os.getenv("DECISION_THRESHOLD", "0.15"))
    
    # Model Artifact paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/best_model.joblib")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "models/scaler.joblib")
    BASELINE_PATH: str = os.getenv("BASELINE_PATH", "models/baseline_healthy.joblib")
    FEATURE_COLS_PATH: str = os.getenv("FEATURE_COLS_PATH", "models/feature_cols.joblib")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "Predictive_Maintenance_Simulation")

    @classmethod
    def print_summary(cls):
        """Prints a summary of active configurations for diagnostic use."""
        print("\n--- Active Configurations ---")
        print(f"API Target: {cls.HOST}:{cls.PORT}")
        print(f"Decision Threshold: {cls.DECISION_THRESHOLD}")
        print(f"Model Path: {cls.MODEL_PATH}")
        print(f"Log Level: {cls.LOG_LEVEL} (File: {cls.LOG_FILE})")
        print(f"MLflow Target: {cls.MLFLOW_TRACKING_URI}")
        print("-----------------------------\n")

if __name__ == "__main__":
    Config.print_summary()
