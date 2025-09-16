import matplotlib.pyplot as plt
import logging
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from data_loader import load_graph_with_masks
from model import GNNModel
from train import train_model, validate_model
from utils import compute_class_weights, find_and_delete_previous_best_models
import multiprocessing
from datetime import datetime
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_intermediate_values, plot_slice
from config import IN_DIM, EPOCHS, N_TRIALS, DATASET_NAME, USE_SEED

# Configure logging at the top of the module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('{}_embedding_and_graphs_tuning.log'.format(DATASET_NAME))
    ]
)

# Check if GPU is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

# Set the number of threads for PyTorch to 70% of the available CPU cores
# Only relevant if using CPU
if device.type == 'cpu':
    num_cores = multiprocessing.cpu_count()
    num_threads = max(1, int(num_cores * 0.7))
    torch.set_num_threads(num_threads)
    logging.info(f'Setting PyTorch to use {num_threads} out of {num_cores} available CPU cores.')


def objective(trial, study, graph, train_mask, val_mask):
    """
    Objective function for Optuna hyperparameter optimization with layer-specific aggregators.
    """
    # Common hyperparameters for all models
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 0.1, log=True)
    num_layers = trial.suggest_int('num_layers', 2, 10)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    patience = trial.suggest_int('patience', 10, 130)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-6, 0.1, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])
    use_skip = trial.suggest_categorical('use_skip', [True, False])

    # Use a fixed, large batch size instead of tuning it
    use_full_graph = True  # Set to True for full graph training
    
    # Layer-specific parameters
    layer_type = trial.suggest_categorical('layer_type', ['GraphConv', 'SAGEConv', 'GATConv'])
    hidden_dim = trial.suggest_int('hidden_dim', 32, 1200)
    
    # SAGE-specific aggregation types
    sageconv_agg = trial.suggest_categorical('sageconv_agg', ['mean', 'pool', 'gcn'])

    # GATConv parameters
    gatconv_agg = trial.suggest_categorical('gatconv_agg', ('sum', 'mean', 'max'))
    gatconv_num_heads = trial.suggest_int('gatconv_num_heads', 1, 6)
    gatconv_hidden = trial.suggest_int('gatconv_hidden', 32, 1200)
    # Ensure hidden is divisible by num_heads
    gatconv_hidden = (gatconv_hidden // gatconv_num_heads) * gatconv_num_heads
    if gatconv_hidden == 0:  # Safeguard against zero hidden dimension
        gatconv_hidden = gatconv_num_heads
    
    # Apply conditional logic without changing the parameter space
    model_kwargs = {
        'in_dim': IN_DIM,
        'hidden_dim': hidden_dim,
        'out_dim': 3,
        'num_layers': num_layers,
        'dropout_rate': dropout_rate,
        'layer_type': layer_type,
        'activation': activation,
        'use_skip': use_skip
    }
    
    # Select the appropriate aggregator based on layer type
    if layer_type == 'SAGEConv':
        model_kwargs['aggregator_type'] = sageconv_agg
    elif layer_type == 'GATConv':
        model_kwargs['hidden_dim'] = gatconv_hidden
        model_kwargs['aggregator_type'] = gatconv_agg
        model_kwargs['num_heads'] = gatconv_num_heads

    # Create and initialize the model with the appropriate parameters
    model = GNNModel(**model_kwargs)
    
    # Move model to GPU if available
    model = model.to(device)
    
    # Compute class weights based on training data
    train_labels = graph.ndata['label'][train_mask]
    class_weights = compute_class_weights(train_labels)
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # Initialize the best validation score and the patience counter
    best_val_score = 0
    patience_counter = 0
    best_model_state = None  # Store the best model state
    best_epoch = 0

    # Track metrics for all epochs
    train_losses = []
    val_scores = []

    # Training loop
    logging.info(f'Starting training loop with {EPOCHS} epochs')
    trial_num = trial.number
    logging.info(f"Trial {trial_num}: Starting with parameters {trial.params}")

    for epoch in range(EPOCHS):
        # Use masks with full graph
        train_loss = train_model(model, criterion, optimizer, graph, None, l1_lambda, 
                                use_full_graph=use_full_graph, train_mask=train_mask)
        train_losses.append(train_loss)  # Store training loss

        # Validation loop with mask
        avg_score = validate_model(model, graph, None, 
                                  use_full_graph=use_full_graph, val_mask=val_mask)
        val_scores.append(avg_score)  # Store validation score

        # Check if the validation score has improved
        if avg_score > best_val_score:
            best_val_score = avg_score
            best_epoch = epoch
            patience_counter = 0

            # Save the best model state in memory
            best_model_state = model.state_dict().copy()

            # Save the metrics up to this point for the best model
            best_train_losses = train_losses.copy()
            best_val_scores = val_scores.copy()
        else:
            patience_counter += 1

        # Report the MCC score to the trial
        trial.report(avg_score, epoch)
            
        # Check if the trial should be pruned
        if patience_counter >= patience:
            break

     # Set the optimal number of epochs as a user attribute
    trial.set_user_attr('best_epoch', best_epoch)
    trial.set_user_attr('total_epochs', epoch + 1)  # +1 since epoch is 0-indexed
    trial.set_user_attr('train_losses', train_losses[:best_epoch+1])  # Save up to best epoch
    trial.set_user_attr('val_scores', val_scores[:best_epoch+1])      # Save up to best epoch

    # Log the best validation score and the best epoch for this trial
    logging.info(f'Trial {trial.number}: Best validation MCC score is {best_val_score}, achieved at epoch: {best_epoch}')
    logging.info(f'Trial {trial.number}: Hyperparameters are {trial.params}')

    # Check if this is the best trial so far - only save if it's better than previous best
    if trial.number > 0:  # Skip first trial comparison
        best_trial = None
        try:
            best_trial = study.best_trial
        except:
            pass  # No best trial yet
        
        if best_model_state is not None and (best_trial is None or best_val_score > best_trial.value):
            # This is the best score we've seen across all trials - save the model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Delete previous best models
            find_and_delete_previous_best_models(DATASET_NAME)
            
            # Save this model as the new best
            model_path = f'{DATASET_NAME}_best_model_trial_{trial.number}_epoch_{best_epoch}_{timestamp}.pt'
            
            torch.save({
                'trial_number': trial.number,
                'epoch': best_epoch,
                'model_state_dict': best_model_state,  # Save the best state from memory
                'val_score': best_val_score,
                'hyperparameters': trial.params,
                'timestamp': timestamp,
                'in_dim': IN_DIM,
                'out_dim': 3,
                'train_losses': best_train_losses,  # Save training losses
                'val_scores': best_val_scores,      # Save validation scores
                'seed': USE_SEED                   # Save seed information
            }, model_path)
            
            logging.info(f"New best model saved to {model_path} with score: {best_val_score:.4f}")


    # Log the best trial so far
    if trial.number > 5:
        best_trial = trial.study.best_trial
        if best_trial is not None:
            logging.info(f'Best trial so far {best_trial.number}: Best validation score is {best_trial.value}, achieved at epoch: {best_trial.user_attrs["best_epoch"]}')
            logging.info(f'Best trial so far {best_trial.number}: Hyperparameters are {best_trial.params}')
        else:
            logging.info('No trials have been completed yet')

    # Delete resources after use
    del model
    del optimizer
    
    # Clear CUDA cache if using GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return best_val_score


def main():
    logging.info('Loading graph with masks')
    # Load the single graph with masks - only use the default masks (seed 42)
    graph, train_mask, val_mask, _ = load_graph_with_masks()

    # Move to GPU once
    if device.type == 'cuda':
        graph = graph.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)

    logging.info(f'Graph loaded with {graph.num_nodes()} nodes and {graph.num_edges()} edges')
    logging.info(f'Using default seed for train/val/test masks')

    # Set to 1 job for simplicity and reliability
    n_jobs = 1  
    
    # Create a new study
    storage_name = f"{DATASET_NAME}_optuna_study.db"
    study = optuna.create_study(
        study_name="sequential-tuning", 
        direction="maximize",
        storage=f"sqlite:///{storage_name}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler()
    )
    
    # Define a wrapper function to add trial number to each log message
    def objective_wrapper(trial):
        trial_num = trial.number
        result = objective(trial, study, graph, train_mask, val_mask)
        return result
    
    study.optimize(
        objective_wrapper,
        n_jobs=n_jobs, 
        n_trials=N_TRIALS
    )

    # Print the best hyperparameters and their corresponding objective value
    logging.info('Best trial:')
    trial = study.best_trial
    logging.info(f'Value: {trial.value}')
    logging.info(f'Best epoch: {trial.user_attrs["best_epoch"]}')
    logging.info('Params:')
    for key, value in trial.params.items():
        logging.info(f'    {key}: {value}')

    # Write the best hyperparameters and their corresponding objective value to a CSV file
    with open('{}_best_trial.csv'.format(DATASET_NAME), 'w', newline='') as csvfile:
        trial = study.best_trial
        fieldnames = ['hyperparameter', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key, value in trial.params.items():
            writer.writerow({'hyperparameter': key, 'value': value})
        writer.writerow({'hyperparameter': 'objective_value', 'value': trial.value})
        writer.writerow({'hyperparameter': 'best_epoch', 'value': trial.user_attrs["best_epoch"]})
        writer.writerow({'hyperparameter': 'total_epochs', 'value': trial.user_attrs["total_epochs"]})
        writer.writerow({'hyperparameter': 'seed', 'value': USE_SEED})


    # Generate and save plots with proper handling
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots = {
        'optimization_history': optuna.visualization.plot_optimization_history,
        'param_importances': optuna.visualization.plot_param_importances,
        'intermediate_values': optuna.visualization.plot_intermediate_values,
        'slice': optuna.visualization.plot_slice
    }
    
    for name, plot_function in plots.items():
        try:
            logging.info(f'Creating {name} plot...')
            fig = plot_function(study)
            
            # Handle different figure types
            if hasattr(fig, 'write_image'):  # Plotly figure
                plot_path = f'{DATASET_NAME}_{name}_{timestamp}.png'
                fig.write_image(plot_path)
                logging.info(f'Successfully saved {name} plot to {plot_path}')
            elif hasattr(fig, 'savefig'):  # Matplotlib figure
                plot_path = f'{DATASET_NAME}_{name}_{timestamp}.png'
                fig.savefig(plot_path)
                plt.close(fig)
                logging.info(f'Successfully saved {name} plot to {plot_path}')
            else:
                logging.warning(f'Could not save {name} plot: unknown figure type {type(fig)}')
        except Exception as e:
            logging.error(f'Failed to generate {name} plot: {str(e)}')

if __name__ == "__main__":
    main()
