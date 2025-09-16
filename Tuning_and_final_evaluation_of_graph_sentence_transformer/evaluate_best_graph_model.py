import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import pandas as pd
import logging
import os
import re
import argparse
from datetime import datetime

# Import from your project modules
from data_loader import load_graph_with_masks
from model import GNNModel
from config import IN_DIM
from train import train_model, test_model
from utils import compute_class_weights
from config import DATASET_NAME # Use DATASET_NAME from config as default

# --- Logging Setup ---
def setup_logging(dataset_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{dataset_name}_evaluation_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Evaluation log started. Results will be saved with prefix: {dataset_name}")

# --- Helper Functions ---
def find_latest_best_model(dataset_name):
    """Finds the most recently saved best model file."""
    pattern = re.compile(rf'{dataset_name}_best_model_trial_\d+_epoch_\d+_(\d{{8}}_\d{{6}})\.pt')
    best_model_file = None
    latest_timestamp = ""

    logging.info(f"Searching for best model file with prefix: {dataset_name}_best_model_trial_")
    for filename in os.listdir('.'):
        match = pattern.match(filename)
        if match:
            timestamp = match.group(1)
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                best_model_file = filename

    if best_model_file:
        logging.info(f"Found latest best model file: {best_model_file}")
    else:
        logging.error(f"No best model file found for dataset '{dataset_name}'. Please ensure tuning was run and a model was saved.")
        raise FileNotFoundError(f"No best model file found for dataset '{dataset_name}'")
    return best_model_file

def load_best_model_config(model_path):
    """Loads hyperparameters, state dict, and best epoch from the saved model file."""
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logging.info(f"Loading model configuration from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu')) # Load to CPU first

    hyperparameters = checkpoint['hyperparameters']
    model_state_dict = checkpoint['model_state_dict']
    best_epoch = checkpoint['epoch']
    in_dim = checkpoint['in_dim']
    out_dim = checkpoint['out_dim']

    logging.info(f"Loaded hyperparameters: {hyperparameters}")
    logging.info(f"Model was trained for {best_epoch} epochs during tuning.")
    logging.info(f"Input dimension: {in_dim}, Output dimension: {out_dim}")

    return hyperparameters, model_state_dict, best_epoch, in_dim, out_dim

def get_mask_seeds(graph):
    """Extracts all unique seed identifiers from graph node data keys."""
    seeds = set()
    # Add default seed (represented by None or absence of suffix)
    if 'train_mask' in graph.ndata:
        seeds.add(None) # Use None to represent the default mask set

    # Find seeded masks
    pattern = re.compile(r'train_mask_(\d+)')
    for key in graph.ndata.keys():
        match = pattern.match(key)
        if match:
            seeds.add(int(match.group(1)))

    logging.info(f"Found mask sets for seeds: {seeds}")
    return list(seeds)

def get_masks_for_seed(graph, seed):
    """Retrieves train, val, and test masks for a specific seed."""
    if seed is None: # Default masks
        train_key, val_key, test_key = 'train_mask', 'val_mask', 'test_mask'
    else: # Seeded masks
        train_key, val_key, test_key = f'train_mask_{seed}', f'val_mask_{seed}', f'test_mask_{seed}'

    if train_key not in graph.ndata or val_key not in graph.ndata or test_key not in graph.ndata:
        raise KeyError(f"Masks for seed {seed} not found in the graph node data.")

    return graph.ndata[train_key], graph.ndata[val_key], graph.ndata[test_key]

# --- Main Evaluation Logic ---
def main(args):
    setup_logging(args.dataset_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    try:
        # 1. Load Best Model Configuration
        model_file = find_latest_best_model(args.dataset_name)
        hyperparams, state_dict, best_epoch_tuning, in_dim_from_checkpoint, out_dim_from_checkpoint = load_best_model_config(model_file) # Renamed for clarity

        # Use the dimensions loaded directly from the checkpoint file
        in_dim = in_dim_from_checkpoint
        out_dim = out_dim_from_checkpoint

        # Update the hyperparameters dictionary to ensure the model is created correctly
        hyperparams['in_dim'] = in_dim
        hyperparams['out_dim'] = out_dim
        logging.info(f"Using Input Dimension: {in_dim}, Output Dimension: {out_dim} from checkpoint for model instantiation.")
        # --- INSERT THIS BLOCK END ---

        # 2. Load Graph
        logging.info("Loading graph...")
        # load_graph_with_masks now only returns graph and default masks, but graph contains all masks
        graph, _, _, _ = load_graph_with_masks()
        graph = graph.to(device)
        logging.info(f"Graph loaded to {device}")

        # 3. Get Seeds
        seeds = get_mask_seeds(graph)
        if not seeds:
            logging.error("No mask sets found in the graph. Cannot proceed with evaluation.")
            return

        # 4. Evaluate for each seed
        results = []
        for seed in seeds:
            logging.info(f"--- Evaluating Seed: {seed if seed is not None else 'Default (42)'} ---")

            # Get masks for the current seed
            train_mask_orig, val_mask_orig, test_mask = get_masks_for_seed(graph, seed)
            train_mask_orig, val_mask_orig, test_mask = train_mask_orig.to(device), val_mask_orig.to(device), test_mask.to(device)

            # Combine train and validation masks for retraining
            combined_train_mask = train_mask_orig | val_mask_orig
            logging.info(f"Combined training set size: {combined_train_mask.sum().item()}")
            logging.info(f"Test set size: {test_mask.sum().item()}")
            
            # Map hyperparameter names from checkpoint to model constructor arguments if they differ
            if 'sageconv_agg' in hyperparams and 'aggregator_type' not in hyperparams:
                 hyperparams['aggregator_type'] = hyperparams.pop('sageconv_agg')
            if 'gatconv_agg' in hyperparams and 'aggregator_type' not in hyperparams: # Add similar mapping if needed for GAT
                 hyperparams['aggregator_type'] = hyperparams.pop('gatconv_agg')
            if 'gatconv_num_heads' in hyperparams and 'num_heads' not in hyperparams:
                 hyperparams['num_heads'] = hyperparams.pop('gatconv_num_heads')

            # Instantiate Model
            # Handle potential missing keys if model structure changed slightly
            model_kwargs = {k: v for k, v in hyperparams.items() if k in GNNModel.__init__.__code__.co_varnames}

            model = GNNModel(**model_kwargs).to(device)

            # Load the state dict. Now that the model architecture should match, 'strict=True' (default) might work,
            # but keeping 'strict=False' provides robustness against minor future inconsistencies.
            try:
                model.load_state_dict(state_dict, strict=True) # Try strict loading first
                logging.info("Model instantiated and state loaded (strict=True).")
            except RuntimeError as e:
                logging.warning(f"Strict state_dict loading failed: {e}. Attempting non-strict loading.")
                model.load_state_dict(state_dict, strict=False) # Fallback to non-strict
                logging.info("Model instantiated and state loaded (strict=False).")

            # Define Optimizer (use the one from best trial)
            optimizer_name = hyperparams.get('optimizer', 'Adam') # Default to Adam if not found
            learning_rate = hyperparams.get('learning_rate', 0.001)
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            else: # RMSprop
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            logging.info(f"Using optimizer: {optimizer_name} with LR: {learning_rate}")

            # Define Loss
            train_labels_combined = graph.ndata['label'][combined_train_mask]
            class_weights = compute_class_weights(train_labels_combined).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logging.info("Loss function and class weights defined.")

            # Retrain the model on combined train+val mask
            logging.info(f"Retraining model for {best_epoch_tuning} epochs...")
            l1_lambda = hyperparams.get('l1_lambda', 0) # Get L1 lambda from params
            for epoch in range(best_epoch_tuning):
                loss = train_model(model, criterion, optimizer, graph, None, l1_lambda,
                                   use_full_graph=True, train_mask=combined_train_mask)
                if (epoch + 1) % 10 == 0 or epoch == best_epoch_tuning - 1:
                     logging.info(f"Retraining Epoch {epoch+1}/{best_epoch_tuning}, Loss: {loss:.4f}")

            # Evaluate on the test mask
            logging.info("Evaluating on the test set...")
            metrics = test_model(model, graph, None, use_full_graph=True, test_mask=test_mask)
            accuracy, f1_macro, prec_macro, rec_macro, f1_w, prec_w, rec_w, mcc = metrics

            logging.info(f"Seed {seed if seed is not None else 'Default (42)'} Results:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  MCC:      {mcc:.4f}")
            logging.info(f"  F1 Macro: {f1_macro:.4f}")
            logging.info(f"  Prec Macro: {prec_macro:.4f}")
            logging.info(f"  Rec Macro: {rec_macro:.4f}")
            logging.info(f"  F1 Weighted: {f1_w:.4f}")

            results.append({
                'seed': seed if seed is not None else 42, # Use 42 for default
                'accuracy': accuracy,
                'mcc': mcc,
                'f1_macro': f1_macro,
                'precision_macro': prec_macro,
                'recall_macro': rec_macro,
                'f1_weighted': f1_w,
                'precision_weighted': prec_w,
                'recall_weighted': rec_w,
            })

            # Clean up memory
            del model, optimizer, criterion
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # 5. Aggregate and Report Results
        results_df = pd.DataFrame(results)
        logging.info("\n--- Overall Evaluation Summary ---")
        logging.info("Results per seed:")
        logging.info(results_df.to_string())

        # Calculate mean and std dev
        summary_stats = results_df.drop(columns=['seed']).agg(['mean', 'std'])
        logging.info("\nSummary Statistics (Mean +/- Std Dev):")
        for metric in summary_stats.columns:
            mean_val = summary_stats.loc['mean', metric]
            std_val = summary_stats.loc['std', metric]
            logging.info(f"  {metric.replace('_', ' ').title()}: {mean_val:.4f} +/- {std_val:.4f}")

        # 6. Save Results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_csv_path = f"{args.dataset_name}_evaluation_detailed_{timestamp}.csv"
        summary_csv_path = f"{args.dataset_name}_evaluation_summary_{timestamp}.csv"

        results_df.to_csv(detailed_csv_path, index=False)
        summary_stats.to_csv(summary_csv_path)
        logging.info(f"Detailed results saved to: {detailed_csv_path}")
        logging.info(f"Summary statistics saved to: {summary_csv_path}")

    except FileNotFoundError as e:
        logging.error(f"Evaluation failed: {e}")
    except KeyError as e:
        logging.error(f"Evaluation failed: Missing data in graph or model file - {e}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during evaluation: {e}") # Log full traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the best GNN model found during hyperparameter tuning.")
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=DATASET_NAME, # Use default from config.py
        help='Name of the dataset (used to find model and graph files, e.g., Sutherland).'
    )
    args = parser.parse_args()
    main(args)