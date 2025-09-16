import torch
import logging
import os

def create_batches(nodes, batch_size):
    # Shuffle the node indices
    nodes = nodes[torch.randperm(nodes.shape[0])]

    # Split the nodes into mini-batches
    batches = torch.split(nodes, batch_size)

    # Check the size of the last batch
    if batches[-1].size(0) < batch_size:
        batches = batches[:-1]  # Drop the last batch

    return batches

def compute_class_weights(labels):
    class_counts = labels.bincount()
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    return class_weights

def find_and_delete_previous_best_models(dataset_name):
    """Delete all previous best model files in the current directory."""
    count = 0
    pattern = f'{dataset_name}_best_model_trial_'
    
    for filename in os.listdir('.'):
        if filename.startswith(pattern) and filename.endswith('.pt'):
            try:
                os.remove(filename)
                count += 1
            except Exception as e:
                logging.error(f"Failed to delete previous model file: {filename}, error: {str(e)}")
    
    return count