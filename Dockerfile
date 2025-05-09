# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python${PYTHON_VERSION}-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/train data/val data/test logs

# Set environment variables for the application
ENV MODEL_SAVE_DIR=/app/models \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs

# Expose ports
EXPOSE 8000 8501

# Create a non-root user
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# Start the application
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run src/frontend/app.py --server.port 8501 --server.address 0.0.0.0"] 