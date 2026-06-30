# AI-Powered Industrial Predictive Maintenance System

An end-to-end, industry-grade Machine Learning and IoT predictive maintenance system designed to predict mechanical breakdowns in manufacturing machinery before they occur.

This project is structured as a portfolio-grade repository demonstrating production software engineering practices, time-series feature engineering, REST API microservices, interactive dashboard design, automated testing, configuration management, structured logging, experiment tracking, containerization, and high-availability edge failover architectures.

---

## Key Core Concepts (Teaching Modules)

This system models real manufacturing dynamics using these key engineering steps:

1. **24-Hour Lookahead Failure Horizon**: Instead of predicting a failure at the exact hour of breakdown (which is too late for operators), the system labels a 24-hour warning window preceding each failure as the target class ($1$).
2. **Chronological Splitting (Anti-Data Leakage)**: Sequential sensor records violate independent and identically distributed (i.i.d.) assumptions. We split the datasets chronologically (70/15/15) to prevent leakage of future parameters into past training boundaries.
3. **Rolling & Lag Features**: Single-point telemetry lacks historical context. We calculate 6h and 24h rolling averages/standard deviations to capture recent anomalies and gradual mechanical drift.
4. **Physical Sensor Stress Interactions**: We derive interaction variables (e.g. Temperature $\times$ Pressure for thermal-mechanical fatigue, Rotational Speed $\times$ Vibration for fatigue loads).
5. **High-Availability Edge Failover**: If the FastAPI backend server is offline, the Streamlit dashboard switches seamlessly to local execution by loading weights natively, ensuring zero uptime interruption for the factory operators.
6. **Enterprise Configuration & Logging**: Implements standard environment variables (via `config.py` and `.env`) and structured JSON logs (per-line format in `logs/app.log`), making the system immediately ready for production log aggregators (ELK, Loki).
7. **SQLite MLflow Tracking**: Logs training parameters (hyperparameters, metrics, F1-scores, and models) to a local SQLite database (`mlflow.db`) rather than flat text folders, conforming to modern experiment standards.

---

## Project Structure

```text
predictive analysis/
│
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI/CD Pipeline
│
├── api/                     # REST API Backend (FastAPI)
│   ├── main.py              # Route handlers, lifespans & exception filters
│   └── schemas.py           # Pydantic schema validation & bounds checking
│
├── dashboard/               # Frontend Operator Control Center
│   └── app.py               # Streamlit application (with failover mode)
│
├── data/                    # Telemetry datasets
│   └── generate_dataset.py  # Physics-based data simulator
│
├── docs/                    # Architecture and report files
│   ├── assets/              # Exported plots (correlations, matrices, etc.)
│   ├── eda_report.md        # Technical EDA summary report
│   ├── architecture.md      # Data flow diagrams
│   └── api_docs.md          # Request/response documentation
│
├── src/                     # Core python modules
│   ├── __init__.py
│   ├── config.py            # Environment configuration parser
│   ├── logger.py            # Dual-handler console & JSON structured logging
│   ├── exceptions.py        # Custom business exception definitions
│   ├── data_ingestion.py    # Raw data loading, cleaning & window labeling
│   ├── feature_engineering.py# Rolling statistics, lags & interactions
│   ├── train.py             # Chronological split, scale & model comparison (MLflow)
│   └── explain.py           # Feature importance & deviation attributions
│
├── tests/                   # Pytest automation test suite
│   ├── test_features.py     # Mathematical checks
│   ├── test_api.py          # API route response checks
│   └── test_validation.py   # Pydantic validation & config loader tests
│
├── .env.example             # Configuration variables template
├── .gitignore               # Excluded caches, venv, databases, and logs
├── Dockerfile               # Consolidated container setup
├── docker-compose.yml       # Multi-container microservice orchestrator
├── render.yaml              # Cloud deployment blueprint
├── requirements.txt         # Core dependencies
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

### 2. Configure Environment Settings
Copy the environment template file:
```bash
cp .env.example .env
```
You can edit the active `.env` file to customize settings such as `DECISION_THRESHOLD` (default: `0.15`), `PORT` (default: `8000`), or `LOG_LEVEL` (default: `INFO`).

### 3. Generate Simulated Data
Run the physics simulator to populate telemetry and repair logs:
```bash
python -m data.generate_dataset
```

### 4. Train Models & Track Experiments (MLflow)
Scale features, evaluate classifiers, and write runs directly to the SQLite tracking database:
```bash
python -m src.train
```
To launch the **MLflow Dashboard** and explore hyperparameters, plots, and models:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Visit `http://localhost:5000` in your browser.

### 5. Run the REST API Backend
Launch FastAPI using Uvicorn:
```bash
python -m uvicorn api.main:app --reload
```
View the interactive Swagger docs at `http://127.0.0.1:8000/docs`. Audit logs will be printed to stdout in human-readable text and written to `logs/app.log` as structured JSON objects.

### 6. Launch Streamlit Control Center
In a new terminal window (with venv activated), start the dashboard UI:
```bash
streamlit run dashboard/app.py
```

### 7. Run Verification Test Suite
Execute the pytest suite:
```bash
python -m pytest -v
```

---

## Running with Docker (Containerized Microservices)

To spin up the entire multi-container architecture in a single command (which automatically generates the simulation dataset, trains the model, launches the API backend, and boots up the Streamlit frontend client):

```bash
# Build and run containers
docker-compose up --build
```

- **Inference API**: Running at `http://localhost:8000`
- **Operator Dashboard**: Running at `http://localhost:8501`

To tear down the containers and clean up volumes:
```bash
docker-compose down -v
```

---

## High-Availability Risk Classifications
The prediction engine maps probability outputs to operational alerts:
* **Green (Low Risk)**: Probability $< 0.10$. No actions needed.
* **Yellow (Medium Warning)**: Probability $0.10$ to $0.20$. anomalies detected; schedule inspection.
* **Red (High Critical Alert)**: Probability $\ge 0.20$. Immediate maintenance shutdown recommended.
