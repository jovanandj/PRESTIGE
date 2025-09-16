#!/bin/bash

echo "=== Stopping ML Model Final Evaluation Docker Container ==="

# Find and stop any running containers using the ml_tuning image
CONTAINERS=$(docker ps -q --filter ancestor=ml_final_evaluation)
if [ -n "$CONTAINERS" ]; then
    docker stop $CONTAINERS
    echo "Containers stopped."
    docker rm $CONTAINERS
    echo "Containers removed."
else
    echo "No running containers found."
fi

# Remove the Docker image
# IMAGE_ID=$(docker images -q ml_final_evaluation)
# if [ -n "$IMAGE_ID" ]; then
#     docker rmi $IMAGE_ID
#     echo "Docker image removed."
# else
#     echo "No Docker image found."
# fi

echo "=== Cleanup completed ==="