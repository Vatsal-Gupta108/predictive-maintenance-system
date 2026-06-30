# AI-Powered Industrial Predictive Maintenance System

An end-to-end, industry-grade Machine Learning and IoT predictive maintenance system designed to predict mechanical breakdowns in manufacturing machinery before they occur.

This project is structured as a portfolio-grade repository demonstrating production software engineering practices, time-series feature engineering, REST API microservices, interactive dashboard design, automated testing, and high-availability edge failover architectures.

---

## Key Core Concepts (Teaching Modules)

This system models real manufacturing dynamics using these key engineering steps:

1. **24-Hour Lookahead Failure Horizon**: Instead of predicting a failure at the exact hour of breakdown (which is too late for operators), the system labels a 24-hour warning window preceding each failure as the target class ($1$).
2. **Chronological Splitting (Anti-Data Leakage)**: Sequential sensor records violate independent and identically distributed (i.i.d.) assumptions. We split the datasets chronologically (70/15/15) to prevent leakage of future parameters into past training boundaries.
3. **Rolling & Lag Features**: Single-point telemetry lacks historical context. We calculate 6h and 24h rolling averages/standard deviations to capture recent anomalies and gradual mechanical drift.
4. **Physical Sensor Stress Interactions**: We derive interaction variables (e.g. Temperature $\times$ Pressure for thermal-mechanical fatigue, Rotational Speed $\times$ Vibration for fatigue loads).
5. **High-Availability Edge Failover**: If the FastAPI backend server is offline, the Streamlit dashboard switches seamlessly to local execution by loading weights natively, ensuring zero uptime interruption for the factory operators.

---

## Project Structure

```text
predictive analysis/
│
├── data/                    # Telemetry datasets
│   └── generate_dataset.py  # Physics-based data simulator
│
├── src/                     # Core python modules
│   ├── __init__.py
│   ├── data_ingestion.py    # Raw data loading, cleaning & window labeling
│   ├── feature_engineering.py# Rolling statistics, lags & interactions
│   ├── train.py             # Chronological split, scale & model comparison
│   └── explain.py           # Feature importance & deviation attributions
│
├── api/                     # REST API Backend (FastAPI)
│   ├── main.py              # Route handlers and decision thresholds
│   └── schemas.py           # Pydantic schema validation
│
├── dashboard/               # Frontend Operator Control Center
│   └── app.py               # Streamlit application
│
├── tests/                   # Pytest automation test suite
│   ├── test_features.py     # Mathematical checks
│   └── test_api.py          # API route response checks
│
├── docs/                    # Architecture and report files
│   ├── assets/              # Exported plots (correlations, matrices, etc.)
│   ├── eda_report.md        # Technical EDA summary report
│   ├── architecture.md      # Data flow diagrams
│   └── api_docs.md          # Request/response documentation
│
├── requirements.txt         # Core dependencies
├── .gitignore               # Excluded caches, venv, and large binaries
└── README.md                # System documentation
```

---

## Setup & Running Guide

### 1. Initialize Virtual Environment
Create a clean virtual environment and install the required dependencies:
```bash
# Create .venv
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Generate Simulated Data
Run the physics simulator to populate telemetry and repair logs:
```bash
python -m data.generate_dataset
```

### 3. Run EDA & Visualization
Perform statistical analysis and export reports/plots:
```bash
python -m src.eda
```

### 4. Train Models
Scale features, fit models (Logistic Regression, Random Forest, Gradient Boosting), evaluate them, and serialize the champion weights:
```bash
python -m src.train
```

### 5. Generate Baselines & Global Explanations
Computes healthy sensor averages and draws importance charts:
```bash
python -m src.explain
```

### 6. Run the REST API Backend
Launch FastAPI using Uvicorn:
```bash
python -m uvicorn api.main:app --reload
```
You can view interactive Swagger docs at `http://127.0.0.1:8000/docs`.

### 7. Launch Streamlit Control Center
In a new terminal window (with venv activated), start the dashboard UI:
```bash
streamlit run dashboard/app.py
```

### 8. Run Verification Test Suite
Execute the pytest suite:
```bash
python -m pytest
```

---

## High-Availability Risk Classifications
The prediction engine maps probability outputs to operational alerts:
* **Green (Low Risk)**: Probability $< 0.10$. No actions needed.
* **Yellow (Medium Warning)**: Probability $0.10$ to $0.20$. anomalies detected; schedule inspection.
* **Red (High Critical Alert)**: Probability $\ge 0.20$. Immediate maintenance shutdown recommended.
