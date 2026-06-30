import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer

def test_feature_engineer_columns():
    """Verifies that the FeatureEngineer adds all expected columns."""
    # Create mock raw telemetry data
    mock_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
        "machine_id": ["M_001"] * 10,
        "voltage": np.random.uniform(210, 230, 10),
        "temperature": np.random.uniform(50, 70, 10),
        "vibration": np.random.uniform(1.0, 2.0, 10),
        "pressure": np.random.uniform(190, 210, 10),
        "rotational_speed": np.random.uniform(1500, 1600, 10),
        "tool_wear": np.arange(10) * 0.1,
        "failed_now": [0] * 9 + [1]  # Include failure at the end for RUL
    })
    
    fe = FeatureEngineer(rolling_windows=[6], lags=[1])
    feat_df = fe.fit_transform(mock_data)
    
    # Check that key columns are added
    expected_cols = [
        "temperature_roll_mean_6h", 
        "temperature_roll_std_6h", 
        "temperature_lag_1h", 
        "stress_temp_pressure", 
        "stress_speed_vibration",
        "efficiency_vibration_voltage",
        "RUL"
    ]
    for col in expected_cols:
        assert col in feat_df.columns, f"Expected column {col} was not created."

def test_rolling_calculations():
    """Verifies the mathematical correctness of the rolling averages."""
    # Constant values for temperature
    mock_data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
        "machine_id": ["M_001"] * 5,
        "voltage": [220.0] * 5,
        "temperature": [60.0] * 5,  # Constant temperature
        "vibration": [1.0] * 5,
        "pressure": [200.0] * 5,
        "rotational_speed": [1500.0] * 5,
        "tool_wear": [1.0] * 5,
        "failed_now": [0] * 5
    })
    
    fe = FeatureEngineer(rolling_windows=[3], lags=[1])
    feat_df = fe.fit_transform(mock_data)
    
    # The rolling mean of a constant should be the constant itself
    assert np.allclose(feat_df["temperature_roll_mean_3h"], 60.0)
    
    # The rolling std of a constant should be 0.0
    assert np.allclose(feat_df["temperature_roll_std_3h"], 0.0)

def test_sensor_interactions():
    """Verifies that sensor interactions evaluate to the correct equations."""
    mock_data = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01 00:00:00")],
        "machine_id": ["M_001"],
        "voltage": [200.0],
        "temperature": [50.0],
        "vibration": [2.0],
        "pressure": [150.0],
        "rotational_speed": [1000.0],
        "tool_wear": [1.0],
        "failed_now": [0]
    })
    
    fe = FeatureEngineer(rolling_windows=[6], lags=[1])
    feat_df = fe.fit_transform(mock_data)
    
    # Test Stress Temp * Pressure: 50 * 150 = 7500
    assert feat_df.loc[0, "stress_temp_pressure"] == 7500.0
    
    # Test Stress Speed * Vibration: 1000 * 2 = 2000
    assert feat_df.loc[0, "stress_speed_vibration"] == 2000.0
    
    # Test Efficiency Vibration / Voltage: 2 / 200 = 0.01
    assert np.allclose(feat_df.loc[0, "efficiency_vibration_voltage"], 0.01)
