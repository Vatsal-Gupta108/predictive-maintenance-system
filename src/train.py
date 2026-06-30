import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.data_ingestion import DataIngestor
from src.feature_engineering import FeatureEngineer

def train_and_evaluate_models():
    # 1. Ingest and Engineer Features
    print("Ingesting and preparing data...")
    ingestor = DataIngestor(
        telemetry_path="data/raw/sensor_telemetry.csv",
        maintenance_path="data/raw/maintenance_log.csv"
    )
    raw_df = ingestor.align_and_label(window_hours=24)
    
    engineer = FeatureEngineer()
    df = engineer.fit_transform(raw_df)
    
    # 2. Chronological Train-Val-Test Split
    # Since timestamps are sorted, we split by indexing to avoid leaking future data
    print("Splitting data chronologically...")
    unique_timestamps = sorted(df["timestamp"].unique())
    num_timestamps = len(unique_timestamps)
    
    train_end_idx = int(num_timestamps * 0.70)
    val_end_idx = int(num_timestamps * 0.85)
    
    train_cutoff = unique_timestamps[train_end_idx]
    val_cutoff = unique_timestamps[val_end_idx]
    
    train_df = df[df["timestamp"] <= train_cutoff]
    val_df = df[(df["timestamp"] > train_cutoff) & (df["timestamp"] <= val_cutoff)]
    test_df = df[df["timestamp"] > val_cutoff]
    
    print(f"Train set: {train_df.shape[0]} rows (up to {train_cutoff})")
    print(f"Val set: {val_df.shape[0]} rows ({train_cutoff} to {val_cutoff})")
    print(f"Test set: {test_df.shape[0]} rows (after {val_cutoff})")
    
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
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler and feature columns for live API use
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")
    
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
    
    # 6. Train and Compare Models
    results = {}
    best_f1 = -1.0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Predict on validation set
        y_val_pred = model.predict(X_val_scaled)
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, "predict_proba") else y_val_pred
        
        # Evaluate
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred, zero_division=0)
        recall = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_val_proba)
        
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        }
        
        print(f"{name} - Val F1: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | ROC-AUC: {roc_auc:.4f}")
        
        # Track best model based on F1-Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            
    print(f"\nBest model based on Validation F1-Score: {best_model_name} (F1: {best_f1:.4f})")
    
    # 7. Evaluate Best Model on Test Set (Final Holdout)
    print(f"Evaluating {best_model_name} on Test set...")
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    print("\n--- Final Test Set Results ---")
    print(f"Model: {best_model_name}")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_roc_auc:.4f}")
    
    # 8. Save Best Model
    model_save_path = "models/best_model.joblib"
    joblib.dump(best_model, model_save_path)
    print(f"Saved best model to: {model_save_path}")
    
    # Save model details for report
    results_df = pd.DataFrame(results).T
    results_df.to_csv("models/comparison_results.csv")
    
    # 9. Plot and Save Confusion Matrix for Test Set
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
    plt.savefig("docs/assets/confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved to: docs/assets/confusion_matrix.png")

if __name__ == "__main__":
    train_and_evaluate_models()
