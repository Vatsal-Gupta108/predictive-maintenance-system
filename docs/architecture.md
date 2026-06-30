# Architecture & Data Flow

This document details the system design, components, and data flow of the IoT Predictive Maintenance System.

---

## System Architecture Diagram

```mermaid
graph TD
    subgraph IoT Data Sources (Simulated Manufacturing)
        A1[IoT Sensor Telemetry - Hourly] -->|High-Frequency continuous flow| B[Data Ingestion Class]
        A2[Maintenance Log - Discrete Events] -->|Low-Frequency repair records| B
    end

    subgraph Core ML Pipeline (Python SDK)
        B -->|Join & Lookahead Window Labeled| C[Feature Engineer Class]
        C -->|Rolling Stats, Shift Lags, Stress Interactions| D[Model Training & Evaluator]
        D -->|Validation Grid Metrics Evaluation| E[joblib Model Exporter]
    end

    subgraph Storage & Weight State
        E -->|Dump best weights & metadata| F[models/best_model.joblib]
        E -->|Dump fitted StandardScaler state| G[models/scaler.joblib]
        E -->|Dump normal operational average| H[models/baseline_healthy.joblib]
    end

    subgraph Service Delivery Layer (Web & APIs)
        F & G & H -->|Startup validation cache load| I[FastAPI REST Server]
        I -->|REST POST HTTP requests| J[Streamlit Dashboard App]
        F & G & H -->|Edge Failover Local Fallback| J
    end
```

---

## Detailed Component Flow

### 1. Ingestion & Alignment
* **Input**: Raw telemetry logs (continuous hourly voltages, temperatures, pressures) and maintenance crew failure timestamps.
* **Process**: Aligns timestamps, computes a **24-hour lookahead warning window** for supervised binary classification, and removes offline periods (preventing trivial predictions based on shutdown states).

### 2. Feature Engineering Engine
* **Input**: Aligned operational data matrix.
* **Process**: Computes rolling statistics (means and standard deviations over 6-hour and 24-hour windows), lag offsets ($t-1$, $t-2$) to capture transient rates of change, and physical interaction features (such as thermal stress = temperature $\times$ pressure, and centripetal load = speed $\times$ vibration).

### 3. Model Training & Comparison
* **Process**: Conducts a chronological split (70/15/15) to preserve sequence timeline, fits models (`LogisticRegression`, `RandomForestClassifier`, and `GradientBoostingClassifier` with balanced class weights), and compares their F1-Scores.
* **Output**: The champion model is serialized via `joblib`.

### 4. Edge Failover High-Availability Pattern
* When starting, the **Streamlit Dashboard** attempts to call the **FastAPI REST API**.
* If the API server is down or unreachable (e.g. edge connectivity loss on the factory floor), the dashboard falls back to **Local Edge Inference Mode** by loading the joblib files directly into the Streamlit process. This provides robust high-availability.
