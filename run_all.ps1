# Automation Script to launch all microservices in parallel
Write-Host "--------------------------------------------------------" -ForegroundColor Green
Write-Host "Starting Industrial Predictive Maintenance System..." -ForegroundColor Green
Write-Host "--------------------------------------------------------" -ForegroundColor Green

# 1. Launch MLflow Server
Write-Host "1. Launching MLflow UI on http://localhost:5080..." -ForegroundColor Cyan
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Write-Host 'Starting MLflow UI server...'; & .venv/Scripts/Activate.ps1; mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1 --port 5080"

# 2. Launch FastAPI backend
Write-Host "2. Launching FastAPI backend on http://localhost:8000..." -ForegroundColor Cyan
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Write-Host 'Starting FastAPI backend server...'; & .venv/Scripts/Activate.ps1; uvicorn api.main:app --port 8000"

# 3. Launch Streamlit UI Dashboard
Write-Host "3. Launching Streamlit Operator UI on http://localhost:8501..." -ForegroundColor Cyan
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Write-Host 'Starting Streamlit dashboard...'; & .venv/Scripts/Activate.ps1; streamlit run dashboard/app.py"

Write-Host ""
Write-Host "Success: All services initialized! Check the spawned terminal windows for logs." -ForegroundColor Green
Write-Host "--------------------------------------------------------" -ForegroundColor Green
