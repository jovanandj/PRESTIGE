#!/bin/bash

# This script is used to start the docker container for the GNN final evaluation

# Build the Python application container
echo "Building Python application container..."
docker build -t final-evaluation-app -f Dockerfile_final_evaluation .

# Check if NVIDIA GPUs are available through Docker
if command -v nvidia-smi &> /dev/null && docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "GPUs are available. Running with GPU acceleration..."
    docker run -it --name final-evaluation --gpus all -v "$(pwd)":/app final-evaluation-app
else
    echo "GPUs are not available or CUDA toolkit is not installed. Running on CPU only..."
    docker run -it --name final-evaluation -v "$(pwd)":/app final-evaluation-app
fi