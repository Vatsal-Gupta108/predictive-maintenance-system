import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Engineers rolling statistics, lag features, sensor interactions, and
    Remaining Useful Life (RUL) for predictive maintenance.
    """
    def __init__(self, rolling_windows=[6, 24], lags=[1, 2]):
        self.rolling_windows = rolling_windows
        self.lags = lags
        self.numerical_sensors = ["voltage", "temperature", "vibration", "pressure", "rotational_speed", "tool_wear"]

    def fit(self, df: pd.DataFrame):
        """No state needs to be fitted for these transformations, but conforms to sklearn api."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the dataframe by adding engineered features."""
        # Work on a copy to prevent SettingWithCopyWarning
        feat_df = df.copy()
        
        # Sort values to ensure rolling calculations are chronological
        feat_df = feat_df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        
        # 1. Generate Rolling Features
        print("Calculating rolling averages and standard deviations...")
        for window in self.rolling_windows:
            for sensor in ["temperature", "vibration", "pressure", "voltage"]:
                # Group by machine_id to prevent rolling across different machines
                feat_df[f"{sensor}_roll_mean_{window}h"] = (
                    feat_df.groupby("machine_id")[sensor]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                feat_df[f"{sensor}_roll_std_{window}h"] = (
                    feat_df.groupby("machine_id")[sensor]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )
                # Fill NaNs from standard deviation of window size 1 with 0.0
                feat_df[f"{sensor}_roll_std_{window}h"] = feat_df[f"{sensor}_roll_std_{window}h"].fillna(0.0)

        # 2. Generate Lag Features
        print("Calculating lag features...")
        for lag in self.lags:
            for sensor in ["temperature", "vibration", "pressure"]:
                feat_df[f"{sensor}_lag_{lag}h"] = (
                    feat_df.groupby("machine_id")[sensor]
                    .shift(lag)
                )
                # Fill missing lags with the current value (bfill)
                feat_df[f"{sensor}_lag_{lag}h"] = feat_df[f"{sensor}_lag_{lag}h"].fillna(feat_df[sensor])

        # 3. Generate Sensor Interactions
        print("Calculating sensor interaction features...")
        # Thermal-Mechanical Stress: Temperature * Pressure
        feat_df["stress_temp_pressure"] = feat_df["temperature"] * feat_df["pressure"]
        # Rotational Strain: Speed * Vibration
        feat_df["stress_speed_vibration"] = feat_df["rotational_speed"] * feat_df["vibration"]
        # Mechanical Efficiency: Vibration / Voltage (avoiding divide-by-zero)
        feat_df["efficiency_vibration_voltage"] = feat_df["vibration"] / (feat_df["voltage"] + 1e-5)

        # 4. Calculate Remaining Useful Life (RUL)
        # RUL is only calculable if we have failures in the dataset
        if "failed_now" in feat_df.columns:
            print("Calculating Remaining Useful Life (RUL)...")
            feat_df["RUL"] = 9999.0  # Placeholder for rows after the last failure
            
            for machine_id, group in feat_df.groupby("machine_id"):
                failure_times = group[group["failed_now"] == 1]["timestamp"]
                indices = group.index
                
                if len(failure_times) == 0:
                    continue
                
                # Convert timestamps to nanoseconds for calculations
                group_times = group["timestamp"].astype('int64') / 1e9 / 3600  # Convert to hours
                failure_times_hours = failure_times.astype('int64') / 1e9 / 3600
                
                rul_values = np.zeros(len(group))
                
                for i, curr_time in enumerate(group_times):
                    # Find subsequent failures
                    future_failures = failure_times_hours[failure_times_hours >= curr_time]
                    if len(future_failures) > 0:
                        # Time left until next failure in hours
                        rul_values[i] = future_failures.iloc[0] - curr_time
                    else:
                        # If no future failures, set to average operating hours left or default
                        rul_values[i] = 1000.0  # Safe run duration after last repair
                        
                feat_df.loc[indices, "RUL"] = rul_values

        return feat_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits parameters and transforms the dataframe."""
        return self.fit(df).transform(df)

if __name__ == "__main__":
    from src.data_ingestion import DataIngestor
    
    ingestor = DataIngestor(
        telemetry_path="data/raw/sensor_telemetry.csv",
        maintenance_path="data/raw/maintenance_log.csv"
    )
    raw_df = ingestor.align_and_label(window_hours=24)
    
    engineer = FeatureEngineer()
    engineered_df = engineer.fit_transform(raw_df)
    
    print("\nEngineered Dataset Columns:")
    print(list(engineered_df.columns))
    print("\nSample engineered records (with RUL):")
    print(engineered_df[["timestamp", "machine_id", "temperature", "temperature_roll_mean_6h", "stress_temp_pressure", "RUL"]].head())
