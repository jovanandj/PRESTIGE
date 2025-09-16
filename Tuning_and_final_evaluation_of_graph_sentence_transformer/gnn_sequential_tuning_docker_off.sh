#!/bin/bash

# This script is used to stop and remove the docker containers/images for the GNN tuning application

# Stop and remove the container
echo "Stopping container..."
docker stop sequential-gnn-tuning 

echo "Removing container..."
docker rm sequential-gnn-tuning 

# Remove the image
# echo "Removing image..."
# docker rmi sequential-hyper-tuning-app

