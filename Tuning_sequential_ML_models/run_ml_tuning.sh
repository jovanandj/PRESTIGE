#!/bin/bash

# Enable error reporting
set -e

echo "=== Building and Running ML Model Tuning Docker Container ==="

# Build the Docker image
echo "Building Docker image..."
docker build -f dockerfile_ml_tuning -t ml_tuning .
echo "Docker image built successfully!"

# Try running with GPU support first
echo "Attempting to run with NVIDIA GPU support..."
if docker run --gpus all --rm nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU support confirmed! Running with GPU acceleration..."
    docker run --gpus all -it --rm \
      -v $(pwd):/app \
      ml_tuning
else
    echo "NVIDIA GPU support not available. Falling back to CPU mode..."
    # Update the tuning_ml_models.py to disable GPU
    sed -i 's/USE_GPU = True/USE_GPU = False/' tuning_ml_models.py
    
    # Run without GPU flag
    docker run -it --rm \
      -v $(pwd):/app \
      ml_tuning
    
    # Restore the setting after running
    sed -i 's/USE_GPU = False/USE_GPU = True/' tuning_ml_models.py
fi

echo "=== Process completed ==="