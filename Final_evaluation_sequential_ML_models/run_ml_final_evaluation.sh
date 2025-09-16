#!/bin/bash

# Enable error reporting
set -e

echo "=== Building and Running ML Model Final Evaluation Container ==="

# Build the Docker image
echo "Building Docker image..."
docker build -f dockerfile_ml_final_evaluation_embeddings -t ml_final_evaluation .
echo "Docker image built successfully!"

# Run the Docker container
docker run --rm \
    -v "$(pwd):/app" \
    ml_final_evaluation

echo "=== Process completed ==="