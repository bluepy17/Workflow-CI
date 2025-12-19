FROM python:3.12.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    mlflow==2.19.0 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    imbalanced-learn==0.12.4

# Copy application files
COPY modelling.py .
COPY MLProject .
COPY heart_preprocessing ./heart_preprocessing

# Set entrypoint
ENTRYPOINT ["python", "modelling.py"]
