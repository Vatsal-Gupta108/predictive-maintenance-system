# Use official slim Python runtime as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    MLFLOW_ALLOW_FILE_STORE=true

# Set working directory inside the container
WORKDIR /app

# Copy dependency requirements first to leverage Docker build cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application codebase
COPY . .

# Generate dataset and train the model during build so weights are baked into the container image
RUN python data/generate_dataset.py && python -m src.train

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Default command launches Streamlit dashboard on port 7860
CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
