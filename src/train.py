import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.data_ingestion import DataIngestor
from src.feature_engineering import FeatureEngineer
from src.config import Config
from src.logger import get_logger

# Initialize Logger
logger = get_logger("TrainPipeline")

# Setup MLflow Tracking Configuration
mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

def train_and_evaluate_models():
    logger.info("Initializing model training pipeline...")
    
    # 1. Ingest and Engineer Features
    logger.info("Starting data ingestion and preprocessing...")
    ingestor = DataIngestor(
        telemetry_path="data/raw/sensor_telemetry.csv",
        maintenance_path="data/raw/maintenance_log.csv"
    )
    raw_df = ingestor.align_and_label(window_hours=24)
    
    engineer = FeatureEngineer()
    df = engineer.fit_transform(raw_df)
    
    # 2. Chronological Train-Val-Test Split
    logger.info("Performing chronological split to prevent time leakage...")
    unique_timestamps = sorted(df["timestamp"].unique())
    num_timestamps = len(unique_timestamps)
    
    train_end_idx = int(num_timestamps * 0.70)
    val_end_idx = int(num_timestamps * 0.85)
    
    train_cutoff = unique_timestamps[train_end_idx]
    val_cutoff = unique_timestamps[val_end_idx]
    
    train_df = df[df["timestamp"] <= train_cutoff]
    val_df = df[(df["timestamp"] > train_cutoff) & (df["timestamp"] <= val_cutoff)]
    test_df = df[df["timestamp"] > val_cutoff]
    
    logger.info(f"Train rows: {train_df.shape[0]} | Val rows: {val_df.shape[0]} | Test rows: {test_df.shape[0]}")
    
    # 3. Separate Features and Target
    target_col = "failure_within_window"
    exclude_cols = ["timestamp", "machine_id", "failure_type", "failed_now", "failure_within_window", "RUL"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # 4. Scale Features
    logger.info("Fitting feature standardizer...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler and feature columns for live API use
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, Config.SCALER_PATH)
    joblib.dump(feature_cols, Config.FEATURE_COLS_PATH)
    logger.info("Saved feature names and scaler metadata.")
    
    # 5. Define Models
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", 
            max_iter=1000, 
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            class_weight="balanced", 
            random_state=42, 
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, 
            random_state=42
        )
    }
    
    # 6. Train and Compare Models with MLflow tracking
    results = {}
    best_f1 = -1.0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        logger.info(f"Training and logging candidate: {name}")
        
        # Start MLflow run for each classifier
        with mlflow.start_run(run_name=name.replace(" ", "_")):
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_val_pred = model.predict(X_val_scaled)
            y_val_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, "predict_proba") else y_val_pred
            
            # Metrics
            accuracy = accuracy_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred, zero_division=0)
            recall = recall_score(y_val, y_val_pred, zero_division=0)
            f1 = f1_score(y_val, y_val_pred, zero_division=0)
            roc_auc = roc_auc_score(y_val, y_val_proba)
            
            # Log params to MLflow
            mlflow.log_param("classifier", name)
            if hasattr(model, "class_weight"):
                mlflow.log_param("class_weight", str(model.class_weight))
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, "max_iter"):
                mlflow.log_param("max_iter", model.max_iter)
            if hasattr(model, "learning_rate"):
                mlflow.log_param("learning_rate", model.learning_rate)
                
            # Log metrics to MLflow
            mlflow.log_metric("val_accuracy", accuracy)
            mlflow.log_metric("val_precision", precision)
            mlflow.log_metric("val_recall", recall)
            mlflow.log_metric("val_f1_score", f1)
            mlflow.log_metric("val_roc_auc", roc_auc)
            
            results[name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": roc_auc
            }
            
            logger.info(f"{name} metrics logged. Val F1: {f1:.4f} | Recall: {recall:.4f}")
            
            # Track best model based on F1-Score
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model = model
                
    logger.info(f"Champion candidate: {best_model_name} (F1: {best_f1:.4f})")
    
    # 7. Evaluate Best Model on Test Set (Final Holdout)
    logger.info(f"Evaluating Champion {best_model_name} on holdout Test set...")
    
    with mlflow.start_run(run_name=f"Champion_{best_model_name.replace(' ', '_')}"):
        y_test_pred = best_model.predict(X_test_scaled)
        y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        
        logger.info(f"Test F1-Score: {test_f1:.4f} | Recall: {test_recall:.4f}")
        
        # Save Best Model locally
        joblib.dump(best_model, Config.MODEL_PATH)
        
        # Save model details for report
        results_df = pd.DataFrame(results).T
        results_df.to_csv("models/comparison_results.csv")
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Healthy", "Risk Warning"],
                    yticklabels=["Healthy", "Risk Warning"])
        plt.title(f"Test Set Confusion Matrix - {best_model_name}", fontsize=12, pad=15)
        plt.ylabel("Actual State")
        plt.xlabel("Predicted State")
        plt.tight_layout()
        
        os.makedirs("docs/assets", exist_ok=True)
        matrix_path = "docs/assets/confusion_matrix.png"
        plt.savefig(matrix_path, dpi=150)
        plt.close()
        
        # Log holdout metrics to MLflow champion run
        mlflow.log_param("classifier", best_model_name)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        
        # Log local files as MLflow run artifacts
        mlflow.log_artifact(Config.MODEL_PATH, artifact_path="model")
        mlflow.log_artifact(matrix_path, artifact_path="plots")
        mlflow.log_artifact("models/comparison_results.csv", artifact_path="reports")
        logger.info("Logged champion artifacts, metrics, and confusion matrix plots to MLflow tracking database.")
        
    logger.info("Training pipeline execution completed successfully.")

if __name__ == "__main__":
    train_and_evaluate_models()
