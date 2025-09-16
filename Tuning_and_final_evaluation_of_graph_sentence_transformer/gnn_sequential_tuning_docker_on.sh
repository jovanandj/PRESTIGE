#!/bin/bash

# This script is used to start the docker container for the GNN sequential tuning application

# Remove previous container if it exists
if docker ps -a | grep -q "sequential-gnn-tuning"; then
    echo "Removing previous container..."
    docker rm -f sequential-gnn-tuning
fi

# Build the Python application container:
echo "Building Python application container..."
docker build -t sequential-hyper-tuning-app -f Dockerfile-gnn-sequential--hyper-tuning-app .

echo "Running Python application container..."

# Check if a GPU container can run successfully
if command -v nvidia-smi &> /dev/null && docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "GPUs are available. Running with GPU acceleration..."
    docker run -it --name sequential-gnn-tuning --gpus all -v "$(pwd)":/app sequential-hyper-tuning-app
else
    echo "GPUs are not available or CUDA toolkit is not installed. Running on CPU only..."
    docker run -it --name sequential-gnn-tuning -v "$(pwd)":/app sequential-hyper-tuning-app
fi