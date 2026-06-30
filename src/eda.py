import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_ingestion import DataIngestor

class EDAPipeline:
    """
    Performs Exploratory Data Analysis (EDA) on aligned predictive maintenance data
    and generates reports and visualizations for documentation.
    """
    def __init__(self, df: pd.DataFrame, output_dir: str = "docs/assets"):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="darkgrid")

    def run_full_analysis(self) -> None:
        """Executes all EDA steps and saves plots to the output directory."""
        print("Running correlation analysis...")
        self.plot_correlation_matrix()

        print("Running failure distribution analysis...")
        self.plot_failure_distribution()

        print("Plotting sensor trends leading up to failure...")
        self.plot_sensor_degradation_trends()

        print("Generating EDA report...")
        self.generate_markdown_report()

    def plot_correlation_matrix(self) -> None:
        """Generates and saves a correlation heatmap for numerical sensors."""
        sensor_cols = ["voltage", "temperature", "vibration", "pressure", "rotational_speed", "tool_wear"]
        corr = self.df[sensor_cols].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
        plt.title("Correlation Heatmap of IoT Machine Sensors", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_matrix.png"), dpi=150)
        plt.close()

    def plot_failure_distribution(self) -> None:
        """Plots the class imbalance and frequency of failure types."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Target label imbalance
        sns.countplot(x="failure_within_window", data=self.df, ax=axes[0], hue="failure_within_window", legend=False, palette="Set2")
        axes[0].set_title("Class Distribution: Warning Window (24h)", fontsize=12)
        axes[0].set_xlabel("Warning Label (0=Normal, 1=Risk Window)")
        axes[0].set_ylabel("Count")

        # Plot 2: Actual Failure modes (exclude None)
        failure_only = self.df[self.df["failure_type"] != "None"]
        if len(failure_only) > 0:
            sns.countplot(x="failure_type", data=failure_only, ax=axes[1], hue="failure_type", legend=False, palette="Accent")
            axes[1].set_title("Distribution of Failure Modes", fontsize=12)
            axes[1].set_xlabel("Failure Type")
            axes[1].set_ylabel("Count")
        else:
            axes[1].text(0.5, 0.5, "No direct failure points found", ha='center', va='center')

        plt.suptitle("Class Balance & Mechanical Failure Types", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "failure_distribution.png"), dpi=150)
        plt.close()

    def plot_sensor_degradation_trends(self) -> None:
        """Plots sensor readings leading up to a specific failure event for demonstration."""
        # Find a failure occurrence in machine 1
        m1_data = self.df[self.df["machine_id"] == "M_001"].sort_values("timestamp")
        failure_idx = m1_data[m1_data["failed_now"] == 1].index
        
        if len(failure_idx) == 0:
            # Fallback to any machine
            failure_idx = self.df[self.df["failed_now"] == 1].index
            if len(failure_idx) == 0:
                print("Warning: No failures found to plot trends.")
                return
            
        # Extract a slice of 48 hours leading up to the failure
        f_idx = failure_idx[0]
        failure_row = self.df.loc[f_idx]
        m_id = failure_row["machine_id"]
        f_time = failure_row["timestamp"]
        
        full_m_data = self.df[self.df["machine_id"] == m_id].sort_values("timestamp")
        start_time = f_time - pd.Timedelta(hours=48)
        end_time = f_time + pd.Timedelta(hours=4)  # Include a few hours after repair
        
        slice_df = full_m_data[(full_m_data["timestamp"] >= start_time) & (full_m_data["timestamp"] <= end_time)]
        
        # Plotting
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Temp plot
        axes[0].plot(slice_df["timestamp"], slice_df["temperature"], color="crimson", label="Temperature (°C)", lw=2)
        axes[0].axvline(f_time, color="black", linestyle="--", alpha=0.7, label="Failure Event")
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].legend(loc="upper left")
        
        # Vibration plot
        axes[1].plot(slice_df["timestamp"], slice_df["vibration"], color="darkblue", label="Vibration (mm/s)", lw=2)
        axes[1].axvline(f_time, color="black", linestyle="--", alpha=0.7)
        axes[1].set_ylabel("Vibration (mm/s)")
        axes[1].legend(loc="upper left")
        
        # Pressure plot
        axes[2].plot(slice_df["timestamp"], slice_df["pressure"], color="darkgreen", label="Pressure (kPa)", lw=2)
        axes[2].axvline(f_time, color="black", linestyle="--", alpha=0.7)
        axes[2].set_ylabel("Pressure (kPa)")
        axes[2].legend(loc="upper left")
        
        plt.suptitle(f"Sensor Readings Leading to Failure on Machine {m_id} (Failure: {failure_row['failure_type']})", fontsize=14)
        plt.xlabel("Time")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sensor_trends.png"), dpi=150)
        plt.close()

    def generate_markdown_report(self) -> None:
        """Writes a comprehensive EDA summary report in markdown."""
        report_path = "docs/eda_report.md"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Compute basic statistics
        total_records = len(self.df)
        num_failures = self.df[self.df["failed_now"] == 1].shape[0]
        warning_records = self.df[self.df["failure_within_window"] == 1].shape[0]
        ratio_warning = (warning_records / total_records) * 100
        
        # Manual markdown formatting for null counts
        nulls = self.df.isnull().sum()
        null_counts = "| Column | Missing Count |\n| --- | --- |\n"
        for col, val in nulls.items():
            null_counts += f"| {col} | {val} |\n"
            
        # Manual markdown formatting for summary stats
        desc = self.df[["voltage", "temperature", "vibration", "pressure", "rotational_speed", "tool_wear"]].describe()
        summary_stats = "| Metric | " + " | ".join(desc.columns) + " |\n"
        summary_stats += "| --- | " + " | ".join(["---"] * len(desc.columns)) + " |\n"
        for idx, row in desc.iterrows():
            summary_stats += f"| **{idx}** | " + " | ".join(f"{val:.3f}" for val in row) + " |\n"

        report_content = f"""# Exploratory Data Analysis (EDA) Report

This document reports the structural and statistical characteristics of the predictive maintenance dataset.

---

## 1. Dataset Dimensions & Completeness

* **Total Operational Records**: {total_records} hours of active machine telemetry.
* **Exact Failures Logged**: {num_failures} breakdowns.
* **Risk Window Records (Target Class = 1)**: {warning_records} hours ({ratio_warning:.2f}% of total data).
* **Missing Values**:
{null_counts}

---

## 2. Sensor Summary Statistics

The table below outlines the ranges, averages, and variability of the IoT sensors during operation:

{summary_stats}

---

## 3. Class Distribution & Imbalance Analysis

Predictive maintenance suffers from extreme class imbalance because machinery operates normally 95%+ of the time. Simple accuracy is a deceptive metric here. We visualize the distribution of normal vs. warning classes below:

![Class Distribution](assets/failure_distribution.png)

---

## 4. Sensor Cross-Correlation Heatmap

Understanding sensor correlations tells us if multiple sensor systems are co-dependent or redundant.
- High correlation between **Temperature** and **Vibration** is common as component friction causes both heat and oscillation.
- Negative correlation with **Pressure** represents degradation of fluid boundaries over time.

![Sensor Correlation](assets/correlation_matrix.png)

---

## 5. Physical Sensor Degradation Patterns

This chart tracks a 48-hour period leading to a mechanical failure. Observe how the signals degrade (temperature rises, vibration escalates, and pressure drops) prior to the failure boundary:

![Sensor Degradation Trends](assets/sensor_trends.png)
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Report written to: {report_path}")

if __name__ == "__main__":
    # Test pipeline
    ingestor = DataIngestor(
        telemetry_path="data/raw/sensor_telemetry.csv",
        maintenance_path="data/raw/maintenance_log.csv"
    )
    df = ingestor.align_and_label(window_hours=24)
    
    eda = EDAPipeline(df)
    eda.run_full_analysis()
