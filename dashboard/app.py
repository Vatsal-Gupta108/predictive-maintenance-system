import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import joblib
import os
import sys

# Dynamic path resolution to import from root 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineer
from src.explain import ModelExplainer

# Set page config
st.set_page_config(
    page_title="IoT Industrial Predictive Maintenance System",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# API Endpoint
API_URL = "http://127.0.0.1:8000"

# Inject Custom CSS for Premium Design
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.875rem;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
    }
    .risk-low {
        border-left-color: #10b981 !important;
        background-color: rgba(16, 185, 129, 0.05);
    }
    .risk-medium {
        border-left-color: #f59e0b !important;
        background-color: rgba(245, 158, 11, 0.05);
    }
    .risk-high {
        border-left-color: #ef4444 !important;
        background-color: rgba(239, 68, 68, 0.05);
    }
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .status-online { background-color: #10b981; }
    .status-offline { background-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Helper function to check API health
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=1)
        if response.status_code == 200:
            return True, response.json()
    except Exception:
        pass
    return False, None

# Fallback prediction function using local model
def predict_local(machine_id, history_df):
    try:
        model = joblib.load("models/best_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        feature_names = joblib.load("models/feature_cols.joblib")
        baseline = joblib.load("models/baseline_healthy.joblib")
        
        # Calculate features
        fe = FeatureEngineer()
        df_feat = fe.transform(history_df)
        latest_record = df_feat.iloc[-1]
        
        # Prepare feature vector
        X_df = latest_record[feature_names].to_frame().T
        X_scaled = scaler.transform(X_df)
        
        # Predict probability
        failure_prob = float(model.predict_proba(X_scaled)[0, 1])
        
        # Risk classification
        if failure_prob >= 0.20:
            risk_level = "High"
            recommendation = "CRITICAL ALERT: High probability of breakdown. Schedule immediate maintenance shutdown."
        elif failure_prob >= 0.15:
            risk_level = "Medium"
            recommendation = "WARNING: Sensor anomalies detected. Schedule preventative check and maintenance."
        else:
            risk_level = "Low"
            recommendation = "HEALTHY: Machine operating normally. Continue standard monitoring schedule."
            
        # Explainer attributions
        explainer = ModelExplainer()
        explainer.model = model
        explainer.scaler = scaler
        explainer.feature_names = feature_names
        explainer.baseline_healthy = baseline
        
        latest_features_dict = latest_record[feature_names].to_dict()
        explanation = explainer.explain_instance(latest_features_dict)
        top_drivers = [(feat, float(score)) for feat, score in explanation["top_drivers"]]
        
        return {
            "timestamp": str(latest_record["timestamp"]),
            "failure_probability": failure_prob,
            "risk_level": risk_level,
            "top_drivers": top_drivers,
            "recommendation": recommendation
        }
    except Exception as e:
        st.error(f"Local failover inference failed: {str(e)}")
        return None

# Load raw telemetry data for simulation
@st.cache_data
def load_raw_telemetry():
    if os.path.exists("data/raw/sensor_telemetry.csv"):
        df = pd.read_csv("data/raw/sensor_telemetry.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    return None

# App Layout
st.title("⚙️ factory-floor IoT Predictive Maintenance dashboard")
st.markdown("---")

# Sidebar configurations
st.sidebar.header("Control Panel")

# Check API status
api_online, health_data = check_api_health()
if api_online:
    st.sidebar.markdown(f'<p><span class="status-dot status-online"></span><b>Prediction Service</b>: ONLINE</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<p><span class="status-dot status-offline"></span><b>Prediction Service</b>: OFFLINE (Failover Mode)</p>', unsafe_allow_html=True)

mode = st.sidebar.radio("Navigation", ["Real-time Monitor", "Batch CSV Upload"])

# Load simulated database
telemetry_data = load_raw_telemetry()

if mode == "Real-time Monitor":
    if telemetry_data is None:
        st.error("Raw telemetry file data/raw/sensor_telemetry.csv not found. Please run the data generator script first.")
    else:
        # Machine selector
        machines = sorted(telemetry_data["machine_id"].unique())
        selected_machine = st.sidebar.selectbox("Select Asset ID", machines)
        
        # Filter for this machine
        machine_df = telemetry_data[telemetry_data["machine_id"] == selected_machine].sort_values("timestamp")
        
        # Simulation Slider: Let user choose an hour of the year
        total_steps = len(machine_df)
        st.sidebar.subheader("Simulation Controls")
        step_idx = st.sidebar.slider(
            "Select Time Frame (Operating Hour)", 
            min_value=24, 
            max_value=total_steps - 1, 
            value=min(total_steps - 1, 385),  # default to show an interesting point
            step=1
        )
        
        # Get historical window up to the selected step
        # Feature engineering needs up to 24 records to compute rolling averages
        history_window = machine_df.iloc[step_idx - 24 : step_idx + 1].copy()
        current_record = history_window.iloc[-1]
        
        # Display timestamp
        st.subheader(f"Asset Health Report: {selected_machine} (Time: {current_record['timestamp']})")
        
        # Get predictions (API with fallback)
        prediction = None
        if api_online:
            try:
                # Prepare payload
                history_records = []
                for _, row in history_window.iterrows():
                    history_records.append({
                        "timestamp": str(row["timestamp"]),
                        "voltage": float(row["voltage"]),
                        "temperature": float(row["temperature"]),
                        "vibration": float(row["vibration"]),
                        "pressure": float(row["pressure"]),
                        "rotational_speed": float(row["rotational_speed"]),
                        "tool_wear": float(row["tool_wear"])
                    })
                
                payload = {
                    "machine_id": selected_machine,
                    "history": history_records
                }
                
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    prediction = response.json()
            except Exception:
                pass
                
        # Fallback to local prediction if API call failed
        if prediction is None:
            prediction = predict_local(selected_machine, history_window)
            
        if prediction:
            # 1. Row 1: KPI Cards
            col1, col2, col3, col4 = st.columns(4)
            
            prob = prediction["failure_probability"]
            risk = prediction["risk_level"]
            reco = prediction["recommendation"]
            
            risk_class = "risk-low" if risk == "Low" else ("risk-medium" if risk == "Medium" else "risk-high")
            
            with col1:
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <div class="metric-title">Machine Health Risk</div>
                    <div class="metric-value">{risk}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <div class="metric-title">Failure Probability</div>
                    <div class="metric-value">{prob * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                # Render current temperature
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Current Temperature</div>
                    <div class="metric-value">{current_record['temperature']:.1f} °C</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                # Render current vibration
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Current Vibration</div>
                    <div class="metric-value">{current_record['vibration']:.2f} mm/s</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Operator recommendation alert
            if risk == "High":
                st.error(reco)
            elif risk == "Medium":
                st.warning(reco)
            else:
                st.success(reco)
                
            # 2. Row 2: Explanations and Sensor Details
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.subheader("Root Cause Attribution Analysis")
                st.write("Displays which features are contributing most to the current failure probability warning.")
                
                drivers = prediction["top_drivers"]
                drivers_df = pd.DataFrame(drivers, columns=["Feature", "Contribution Score"]).sort_values("Contribution Score")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                # Map technical names to readable names
                readable_names = {row["Feature"]: row["Feature"].replace("_", " ").title() for _, row in drivers_df.iterrows()}
                drivers_df["Feature"] = drivers_df["Feature"].map(readable_names)
                
                ax.barh(drivers_df["Feature"], drivers_df["Contribution Score"], color="#3b82f6")
                ax.set_xlabel("Attribution Score")
                ax.set_title("Top Risk Contributors")
                plt.tight_layout()
                st.pyplot(fig)
                
            with col_right:
                st.subheader("Current Telemetry vs Healthy Baseline")
                st.write("Compares the current sensor inputs with the baseline average of healthy operation.")
                
                baseline = joblib.load("models/baseline_healthy.joblib")
                comparison_records = []
                for sensor in ["temperature", "vibration", "pressure", "voltage", "rotational_speed", "tool_wear"]:
                    curr_val = current_record[sensor]
                    base_val = baseline.get(sensor, curr_val)
                    diff = curr_val - base_val
                    pct_diff = (diff / base_val) * 100
                    comparison_records.append({
                        "Sensor": sensor.replace("_", " ").title(),
                        "Current Value": f"{curr_val:.2f}",
                        "Healthy Baseline": f"{base_val:.2f}",
                        "Difference": f"{diff:+.2f} ({pct_diff:+.1f}%)"
                    })
                st.table(pd.DataFrame(comparison_records))
                
            # 3. Row 3: Historic Trends
            st.subheader("Historical Trends & Alarm Thresholds")
            st.write("Last 48 hours of telemetry logs. Dashed lines represent operational boundaries.")
            
            # Extract 48h trend
            trend_df = machine_df.iloc[max(0, step_idx - 48) : step_idx + 1]
            
            fig_trends, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
            
            axes[0].plot(trend_df["timestamp"], trend_df["temperature"], color="crimson", lw=2, label="Temperature (°C)")
            axes[0].axhline(80.0, color="orange", linestyle="--", alpha=0.5, label="High Temp Warning")
            axes[0].axhline(90.0, color="red", linestyle="--", alpha=0.7, label="Critical Shutdown Limit")
            axes[0].set_ylabel("Temp (°C)")
            axes[0].legend(loc="upper left")
            
            axes[1].plot(trend_df["timestamp"], trend_df["vibration"], color="darkblue", lw=2, label="Vibration (mm/s)")
            axes[1].axhline(2.0, color="orange", linestyle="--", alpha=0.5, label="High Vibration Limit")
            axes[1].set_ylabel("Vibration (mm/s)")
            axes[1].legend(loc="upper left")
            
            axes[2].plot(trend_df["timestamp"], trend_df["pressure"], color="darkgreen", lw=2, label="Pressure (kPa)")
            axes[2].axhline(160.0, color="red", linestyle="--", alpha=0.7, label="Low Pressure Limit")
            axes[2].set_ylabel("Pressure (kPa)")
            axes[2].legend(loc="upper left")
            
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig_trends)
            
elif mode == "Batch CSV Upload":
    st.subheader("Batch Analytics Engine")
    st.write("Upload a CSV file containing machine sensors telemetry to perform batch predictions.")
    
    uploaded_file = st.file_uploader("Upload Telemetry CSV", type=["csv"])
    
    if uploaded_file is not None:
        st.info("File uploaded. Analyzing...")
        
        # Read file
        df_upload = pd.read_csv(uploaded_file)
        
        # Verify columns
        required_cols = ["timestamp", "machine_id", "voltage", "temperature", "vibration", "pressure", "rotational_speed", "tool_wear"]
        missing = [col for col in required_cols if col not in df_upload.columns]
        
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {missing}")
        else:
            # Predict
            summary = None
            if api_online:
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.read(), "text/csv")}
                    response = requests.post(f"{API_URL}/predict/batch", files=files)
                    if response.status_code == 200:
                        summary = response.json()
                except Exception as e:
                    st.warning(f"API batch call failed: {str(e)}. Falling back to local execution.")
                    
            if summary is None:
                # Fallback to local batch execution
                try:
                    fe = FeatureEngineer()
                    df_feat = fe.transform(df_upload)
                    model = joblib.load("models/best_model.joblib")
                    scaler = joblib.load("models/scaler.joblib")
                    feature_names = joblib.load("models/feature_cols.joblib")
                    
                    X_scaled = scaler.transform(df_feat[feature_names])
                    probs = model.predict_proba(X_scaled)[:, 1]
                    
                    df_feat["failure_probability"] = probs
                    df_feat["risk_level"] = np.where(probs >= 0.20, "High", np.where(probs >= 0.15, "Medium", "Low"))
                    
                    warnings_df = df_feat[df_feat["failure_probability"] >= 0.15][["timestamp", "machine_id", "failure_probability", "risk_level"]]
                    
                    summary = {
                        "total_records": len(df_upload),
                        "warnings_detected": len(warnings_df),
                        "high_risk_alerts": int(sum(df_feat["risk_level"] == "High")),
                        "medium_risk_warnings": int(sum(df_feat["risk_level"] == "Medium")),
                        "alerts": warnings_df.to_dict(orient="records")
                    }
                except Exception as e:
                    st.error(f"Local batch processing failed: {str(e)}")
                    
            if summary:
                st.success("Batch analysis complete!")
                
                # Render results summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Hours Analyzed", summary["total_records"])
                with col2:
                    st.metric("Warnings Triggered", summary["warnings_detected"])
                with col3:
                    st.metric("High Risk Alerts", summary["high_risk_alerts"])
                    
                # Display table of warnings
                if summary["warnings_detected"] > 0:
                    st.subheader("Triggered Warnings Log")
                    st.write("Filtered rows indicating potential breakdowns in the subsequent 24 hours:")
                    alerts_df = pd.DataFrame(summary["alerts"])
                    st.dataframe(alerts_df)
                    
                    # Download button
                    csv_data = alerts_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Alerts CSV",
                        data=csv_data,
                        file_name="predictive_alerts_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.balloons()
                    st.success("No anomalies or breakdown risks detected in the uploaded telemetry logs!")

# Developer Signature
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #94a3b8; font-size: 0.75rem; line-height: 1.4;'>
    Developed by <b>Vatsal Gupta</b><br>
    <a href='mailto:vatsalgupta1008@gmail.com' style='color: #3b82f6; text-decoration: none;'>vatsalgupta1008@gmail.com</a>
</div>
""", unsafe_allow_html=True)
