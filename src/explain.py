import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModelExplainer:
    """
    Computes global feature importances and local instance-level attributions
    to explain machine failure predictions.
    """
    def __init__(self, model_path: str = "models/best_model.joblib",
                 scaler_path: str = "models/scaler.joblib",
                 feature_cols_path: str = "models/feature_cols.joblib"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_cols_path = feature_cols_path
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.baseline_healthy = None
        
    def load_artifacts(self) -> None:
        """Loads trained model, scaler, and feature names."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at: {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.feature_names = joblib.load(self.feature_cols_path)

    def calculate_global_importance(self) -> pd.DataFrame:
        """Extracts feature importances and saves a plot."""
        if self.model is None:
            self.load_artifacts()
            
        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
            
        if importances is None:
            raise ValueError("Model does not support feature_importances_ or coef_ attribute.")
            
        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        
        # Save importances as CSV
        importance_df.to_csv("models/feature_importances.csv", index=False)
        
        # Plot top 10 features
        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Importance", 
            y="Feature", 
            data=importance_df.head(10), 
            hue="Feature",
            legend=False,
            palette="viridis"
        )
        plt.title("Top 10 Most Critical Sensors & Features (Global)", fontsize=14, pad=15)
        plt.xlabel("Relative Importance Score")
        plt.ylabel("Sensor Feature")
        plt.tight_layout()
        
        os.makedirs("docs/assets", exist_ok=True)
        plt.savefig("docs/assets/global_feature_importance.png", dpi=150)
        plt.close()
        print("Global feature importance plot saved to: docs/assets/global_feature_importance.png")
        
        return importance_df

    def compute_healthy_baseline(self, df_engineered: pd.DataFrame) -> None:
        """
        Computes the average (baseline) values of all features when the machines
        are operating in a completely healthy state (outside warning window).
        """
        if self.feature_names is None:
            self.load_artifacts()
            
        # Select rows where target failure_within_window is 0 (healthy state)
        healthy_rows = df_engineered[df_engineered["failure_within_window"] == 0]
        
        # Compute mean for the features used in model training
        baseline_df = healthy_rows[self.feature_names].mean()
        self.baseline_healthy = baseline_df.to_dict()
        
        # Save baseline to models/
        joblib.dump(self.baseline_healthy, "models/baseline_healthy.joblib")
        print("Baseline healthy values saved to: models/baseline_healthy.joblib")

    def explain_instance(self, raw_instance: dict) -> dict:
        """
        Explains a single instance by calculating the weighted deviation
        of each sensor reading from the healthy baseline.
        """
        if self.model is None:
            self.load_artifacts()
            
        if self.baseline_healthy is None:
            if os.path.exists("models/baseline_healthy.joblib"):
                self.baseline_healthy = joblib.load("models/baseline_healthy.joblib")
            else:
                raise ValueError("Baseline healthy values not found. Run compute_healthy_baseline first.")
                
        # Get importances
        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
            
        importance_map = dict(zip(self.feature_names, importances))
        
        # Calculate attributions
        attributions = {}
        for feature in self.feature_names:
            val = raw_instance.get(feature, None)
            if val is None:
                attributions[feature] = 0.0
                continue
                
            baseline_val = self.baseline_healthy.get(feature, val)
            
            # Compute raw deviation
            deviation = val - baseline_val
            
            # Weighted attribution: absolute deviation * global feature importance
            # We scale this so higher positive values represent critical drivers of failure risk
            # For sensors like pressure where drops indicate failure, we flip the deviation direction
            direction = -1.0 if "pressure" in feature else 1.0
            
            # Normalized deviation (using standard deviation or min/max fallback)
            # For simplicity and visual impact on the dashboard, we multiply relative deviation * importance
            relative_deviation = abs(deviation) / (abs(baseline_val) + 1e-5)
            attributions[feature] = relative_deviation * importance_map[feature]
            
        # Sort and select top contributing features
        sorted_attr = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "attributions": attributions,
            "top_drivers": sorted_attr[:5]
        }

if __name__ == "__main__":
    from src.data_ingestion import DataIngestor
    from src.feature_engineering import FeatureEngineer
    
    # 1. Ingest and Engineer
    ingestor = DataIngestor(
        telemetry_path="data/raw/sensor_telemetry.csv",
        maintenance_path="data/raw/maintenance_log.csv"
    )
    raw_df = ingestor.align_and_label(window_hours=24)
    
    engineer = FeatureEngineer()
    df = engineer.fit_transform(raw_df)
    
    # 2. Explainer execution
    explainer = ModelExplainer()
    explainer.load_artifacts()
    explainer.compute_healthy_baseline(df)
    explainer.calculate_global_importance()
    
    # 3. Test instance explanation
    # Get a sample record where failure_within_window is 1 (high risk)
    high_risk_records = df[df["failure_within_window"] == 1]
    if len(high_risk_records) > 0:
        sample_dict = high_risk_records.iloc[0].to_dict()
        explanation = explainer.explain_instance(sample_dict)
        print("\nExplanation for High-Risk Instance:")
        print(f"Machine ID: {sample_dict['machine_id']}")
        print(f"Timestamp: {sample_dict['timestamp']}")
        print("Top 5 contributing factors to failure risk:")
        for feature, score in explanation["top_drivers"]:
            print(f" - {feature}: {score:.5f}")
