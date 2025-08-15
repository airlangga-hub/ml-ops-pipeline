# ----------------------------
# Dockerfile for MLOps Pipeline
# ----------------------------

# Base image with Python
FROM python:3.11-slim

# Set workdir inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv

# Copy requirements.txt
COPY requirements.txt .

# Use uv to install from requirements.txt
RUN uv install -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Set environment variables for MLflow & DVC
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV DVC_HOME=/app/.dvc

# uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
