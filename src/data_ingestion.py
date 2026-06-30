import pandas as pd
import numpy as np
import os

class DataIngestor:
    """
    Ingests raw predictive maintenance data, joins telemetry and maintenance logs,
    and labels data for supervised learning using a lead-time failure window.
    """
    def __init__(self, telemetry_path: str, maintenance_path: str):
        self.telemetry_path = telemetry_path
        self.maintenance_path = maintenance_path
        self.telemetry_df = None
        self.maintenance_df = None
        self.merged_df = None

    def load_data(self) -> None:
        """Loads CSV files and formats dates."""
        if not os.path.exists(self.telemetry_path):
            raise FileNotFoundError(f"Telemetry file not found: {self.telemetry_path}")
        if not os.path.exists(self.maintenance_path):
            raise FileNotFoundError(f"Maintenance log not found: {self.maintenance_path}")

        print("Loading telemetry data...")
        self.telemetry_df = pd.read_csv(self.telemetry_path)
        self.telemetry_df["timestamp"] = pd.to_datetime(self.telemetry_df["timestamp"])

        print("Loading maintenance log...")
        self.maintenance_df = pd.read_csv(self.maintenance_path)
        self.maintenance_df["timestamp"] = pd.to_datetime(self.maintenance_df["timestamp"])

    def align_and_label(self, window_hours: int = 24) -> pd.DataFrame:
        """
        Aligns sensor readings with failure logs and creates a look-ahead label
        indicating if a failure will occur within the next `window_hours`.
        """
        if self.telemetry_df is None or self.maintenance_df is None:
            self.load_data()

        # Step 1: Filter maintenance records to get failure events
        failures = self.maintenance_df[self.maintenance_df["event_type"] == "Failure"].copy()
        
        # Sort values for searchsorted logic or merge
        self.telemetry_df = self.telemetry_df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        failures = failures.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

        # Step 2: Ingest the exact failures into the telemetry data
        # We merge failure info based on matching machine and timestamp
        merged = pd.merge(
            self.telemetry_df,
            failures[["timestamp", "machine_id", "failure_type"]],
            on=["machine_id", "timestamp"],
            how="left"
        )
        
        # Fill non-failure timestamps with 'None'
        merged["failure_type"] = merged["failure_type"].fillna("None")
        merged["failed_now"] = (merged["failure_type"] != "None").astype(int)

        print(f"Aligning telemetry and generating failure look-ahead labels (window: {window_hours} hours)...")

        # Step 3: Compute the look-ahead label ('failure_within_window')
        # For each machine, we calculate if a failure occurs in the next 'window_hours'
        # To make this computationally efficient and avoid loops:
        # We find the timestamps of failures for each machine and check if the current timestamp is within [t_fail - window, t_fail]
        
        merged["failure_within_window"] = 0
        
        for machine_id, group in merged.groupby("machine_id"):
            failure_times = group[group["failed_now"] == 1]["timestamp"]
            if len(failure_times) == 0:
                continue
                
            # For each telemetry timestamp in this machine's group, check if it lies in any [t_f - window_hours, t_f]
            indices = group.index
            group_timestamps = group["timestamp"].values
            
            # Label mask
            mask = np.zeros(len(group), dtype=int)
            for f_time in failure_times:
                f_time_np = np.datetime64(f_time)
                f_time_start_np = f_time_np - np.timedelta64(window_hours, 'h')
                
                # Check if telemetry timestamp is in [f_time - window_hours, f_time]
                match = (group_timestamps > f_time_start_np) & (group_timestamps <= f_time_np)
                mask |= match
                
            merged.loc[indices, "failure_within_window"] = mask

        # Step 4: Drop offline records (where voltage, RPM, speed are zero)
        # This keeps the model from learning the trivial rule that "0 voltage means failure" 
        # (after the failure occurred and the machine is powered off for repairs)
        operational_df = merged[merged["voltage"] > 0].copy()
        
        self.merged_df = operational_df
        print(f"Data alignment complete. Operational dataset shape: {self.merged_df.shape}")
        
        return self.merged_df

if __name__ == "__main__":
    # Test script locally
    ingestor = DataIngestor(
        telemetry_path="data/raw/sensor_telemetry.csv",
        maintenance_path="data/raw/maintenance_log.csv"
    )
    df = ingestor.align_and_label(window_hours=24)
    print("Sample records with failure warning label:")
    print(df[df["failure_within_window"] == 1][["timestamp", "machine_id", "temperature", "failure_type", "failure_within_window"]].head())
