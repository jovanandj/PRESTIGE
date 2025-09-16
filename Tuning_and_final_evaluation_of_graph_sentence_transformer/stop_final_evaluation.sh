#!/bin/bash

# This script is used to stop and remove the docker containers/images for the GNN final evaluation

# Stop and remove the container
echo "Stopping container..."
docker stop final-evaluation 

echo "Removing container..."
docker rm final-evaluation 

# Uncomment if you want to remove the image as well
# echo "Removing image..."
# docker rmi final-evaluation-app