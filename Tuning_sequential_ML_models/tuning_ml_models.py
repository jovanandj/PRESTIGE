import pandas as pd
import optuna
import logging
import numpy as np
import json
import time
import os
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from functools import partial
import joblib
from sklearn.metrics import matthews_corrcoef
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
NUM_TRIALS = 1000  # Number of Optuna trials per model
USE_GPU = False    # Flag to control GPU usage
METADATA_FILE = 'SanBernardino_all_MiniLM_L6_v2_avg_embedding_20250408_221422_metadata.json'

def setup_logging(output_dir):
    """Set up logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'hyperparameter_tuning_{timestamp}.log')
        
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def check_gpu_availability():
    """Check for GPU availability and log information"""
    if not USE_GPU:
        logging.info("GPU usage is disabled via configuration.")
        return False
    
    try:
        # Check XGBoost GPU support
        logging.info(f"XGBoost version: {xgb.__version__}")
        
        # Create a small test matrix
        X = np.random.random((10, 10))
        y = np.random.randint(0, 2, 10)
        
        # Try to train a model with GPU
        test_model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=2)
        test_model.fit(X, y)
        logging.info("GPU is available for XGBoost! 🚀")
        return True
    
    except Exception as e:
        logging.warning(f"GPU not available or configured properly: {str(e)}")
        logging.info("Falling back to CPU for all models.")
        return False

def load_data(metadata_file):
    """Load processed data using the metadata file"""
    print(f"Loading data from metadata: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Create output directory based on metadata
    event_name = metadata.get('event_name', 'unknown_event')
    model_name = metadata.get('model_name', 'unknown_model')
    embedding_type = metadata.get('embedding_type', 'unknown_embedding')
    
    # Create folder name
    output_dir = f"output_{event_name}_{model_name}_{embedding_type}"
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Using existing output directory: {output_dir}")
    
    # Setup logging now that we have the output directory
    setup_logging(output_dir)
    logging.info(f"Loading data from metadata: {metadata_file}")
    logging.info(f"Output directory: {output_dir}")
    
    # Extract file paths
    train_file = metadata['files']['train']
    val_file = metadata['files']['validation']
    
    # Load datasets
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    logging.info(f"Loaded {len(train_df)} training samples")
    logging.info(f"Loaded {len(val_df)} validation samples")
    
    # Extract features and labels
    feature_cols = [col for col in train_df.columns if col.startswith('dim_')]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['sentiment'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['sentiment'].values
    
    # Get user IDs for later reference
    user_ids = {
        'train': train_df['user_id'].values,
        'val': val_df['user_id'].values,
    }
    
    # Custom label encoding instead of using LabelEncoder
    # Define your custom mapping
    sentiment_mapping = {
        'positive': 0,
        'neutral': 1,
        'negative': 2
    }
    
    # Apply mapping to train and validation sets
    y_train_encoded = np.array([sentiment_mapping[label] for label in y_train])
    y_val_encoded = np.array([sentiment_mapping[label] for label in y_val])
    
    # Inverse mapping (for later reference)
    class_names = ['positive', 'neutral', 'negative']  # Order matters here
    logging.info(f"Classes: {class_names} with encoding {[0, 1, 2]}")
    
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'y_train_encoded': y_train_encoded,
        'X_val': X_val,
        'y_val': y_val,
        'y_val_encoded': y_val_encoded,
        'feature_cols': feature_cols,
        'class_names': class_names,
        'user_ids': user_ids,
        'metadata': metadata,
        'sentiment_mapping': sentiment_mapping,  # Save the mapping for later use
        'use_gpu': check_gpu_availability(),  # Check if GPU is available
        'output_dir': output_dir  # Include the output directory in the data dictionary
    }
    
    return data

def optimize_model(trial, data, model_type):
    """
    Unified optimization function for all model types
    
    Args:
        trial: Optuna trial object
        data: Dictionary containing dataset
        model_type: String indicating which model to optimize
    
    Returns:
        MCC: Matthews correlation coefficient on validation set
    """
    # Get model and hyperparameters based on model type
    if model_type == 'mlp':
        # Define hyperparameters to search with expanded ranges
        hidden_layer_sizes = []
        n_layers = trial.suggest_int('n_layers', 1, 10)
        
        for i in range(n_layers):
            n_units = trial.suggest_int(f'n_units_l{i}', 32, 1024, log=True)
            hidden_layer_sizes.append(n_units)
        
        alpha = trial.suggest_float('alpha', 1e-6, 1.0, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-5, 0.5, log=True)
        batch_size = trial.suggest_categorical('batch_size', ['auto', 32, 64, 128, 256, 512, 1024, 2048])
        max_iter = trial.suggest_int('max_iter', 200, 1000)
        n_iter_no_change = trial.suggest_int('n_iter_no_change', 10, 50)
        activation = 'relu'
        solver = 'adam'
        early_stopping = True
        
        # Create model
        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            activation=activation,
            solver=solver,
            early_stopping=early_stopping,
            validation_fraction=0.1,
            max_iter=max_iter,
            n_iter_no_change=n_iter_no_change,
            verbose=0
        )
        
    elif model_type == 'xgboost':
        # Define hyperparameters to search
        n_estimators = trial.suggest_int('n_estimators', 50, 2000)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        gamma = trial.suggest_float('gamma', 0.0, 5.0)
        
        # Handle class imbalance with scale_pos_weight
        scale_pos_weight = trial.suggest_categorical('scale_pos_weight', [1, 'balanced'])
        
        if scale_pos_weight == 'balanced':
            # Calculate class distribution for the balanced weight
            neg_samples = np.sum(data['y_train_encoded'] == 2)  # negative class
            pos_samples = np.sum(data['y_train_encoded'] == 0)  # positive class
            scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1
        
        # Create model - use GPU if available
        tree_method = 'gpu_hist' if data['use_gpu'] else 'hist'
        predictor = 'gpu_predictor' if data['use_gpu'] else None
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            scale_pos_weight=scale_pos_weight,
            objective='multi:softmax',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            tree_method=tree_method,
            predictor=predictor,
            gpu_id=0 if data['use_gpu'] else None,
            n_jobs=-1,
            verbose=0
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit model - this is common for all model types
    model.fit(data['X_train'], data['y_train_encoded'])
    
    # Predict on validation set
    y_pred = model.predict(data['X_val'])
    
    # Calculate Matthews correlation coefficient
    MCC = matthews_corrcoef(data['y_val_encoded'], y_pred)
    
    return round(MCC, 4)

def run_optimization(data):
    """Run optimization for all models"""
    # Get the output directory from the data dictionary
    output_dir = data['output_dir']
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save a summary of results
    summary_file = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
    with open(summary_file, 'w') as f:
        f.write("model,best_mcc,duration_seconds,n_trials\n")
    
    # Define models to optimize
    model_types = [
        'mlp',
        'xgboost'
    ]
    
    for model_type in model_types:
        logging.info(f"\n{'='*50}")
        logging.info(f"Optimizing {model_type}")
        logging.info(f"{'='*50}")
        
        start_time = time.time()
        
        # Create study
        study = optuna.create_study(
            study_name=f"{model_type}_optimization",
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=30)
        )
        
        # Run optimization with partial to bind model_type
        study.optimize(
            partial(optimize_model, data=data, model_type=model_type),
            n_trials=NUM_TRIALS,
            show_progress_bar=False,  # Show progress bar for better visibility
            n_jobs=1  # Keep single job for GPU compatibility
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get best trial
        best_trial = study.best_trial
        
        logging.info(f"Best {model_type} trial:")
        logging.info(f"  Value (MCC): {best_trial.value:.4f}")
        logging.info(f"  Params: {best_trial.params}")
        logging.info(f"  Duration: {duration:.2f} seconds")
        
        # Train best model
        best_model = train_best_model(model_type, best_trial.params, data)
        
        # Save model with sentiment mapping
        model_file = os.path.join(output_dir, f"{model_type}_best_model_{timestamp}.pkl")
        joblib.dump({'model': best_model, 'sentiment_mapping': data['sentiment_mapping']}, model_file)
        logging.info(f"  Model saved to {model_file}")
        
        # Save study
        study_file = os.path.join(output_dir, f"{model_type}_study_{timestamp}.pkl")
        joblib.dump(study, study_file)
        
        # Save detailed results for this model to CSV
        model_results_path = os.path.join(output_dir, f"{model_type}_results_{timestamp}.csv")
        with open(model_results_path, 'w') as f:
            f.write("parameter,value\n")
            for key, value in best_trial.params.items():
                f.write(f"{key},{value}\n")
            f.write(f"best_mcc,{best_trial.value}\n")
            f.write(f"duration_seconds,{duration:.2f}\n")
            f.write(f"n_trials,{NUM_TRIALS}\n")
            f.write(f"timestamp,{timestamp}\n")
        
        # Append to summary file
        with open(summary_file, 'a') as f:
            f.write(f"{model_type},{best_trial.value:.6f},{duration:.2f},{NUM_TRIALS}\n")
        
        # Create plots
        try:
            plot_optimization_history(study, os.path.join(output_dir, f"{model_type}_{timestamp}"))
            plot_param_importances(study, os.path.join(output_dir, f"{model_type}_{timestamp}"))
        except Exception as e:
            logging.warning(f"Could not create plots for {model_type}: {e}")
    
    # Return summary file
    return summary_file

def train_best_model(model_name, params, data):
    """Train a model with optimized hyperparameters"""

    if model_name == 'mlp':
        # Create the model with the best parameters
        hidden_layer_sizes = tuple([params[f'n_units_l{i}'] for i in range(params['n_layers'])])
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=params['alpha'],
            learning_rate_init=params['learning_rate_init'],
            batch_size=params['batch_size'],
            activation='relu',  # Fixed to ReLU for stability
            solver='adam',
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=params['max_iter'],
            n_iter_no_change=params['n_iter_no_change'],
            verbose=0
        )

    elif model_name == 'xgboost':
        # Handle the scale_pos_weight parameter
        scale_pos_weight = params['scale_pos_weight']
        if scale_pos_weight == 'balanced':
            neg_samples = np.sum(data['y_train_encoded'] == 2)  # negative class
            pos_samples = np.sum(data['y_train_encoded'] == 0)  # positive class
            scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1
        
        # Use GPU if available
        tree_method = 'gpu_hist' if data['use_gpu'] else 'hist'
        predictor = 'gpu_predictor' if data['use_gpu'] else None
        
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            min_child_weight=params['min_child_weight'],
            gamma=params['gamma'],
            scale_pos_weight=scale_pos_weight,
            objective='multi:softmax',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            tree_method=tree_method,
            predictor=predictor,
            gpu_id=0 if data['use_gpu'] else None,
            n_jobs=-1,
            verbose=0
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train model
    model.fit(data['X_train'], data['y_train_encoded'])
    
    return model

def plot_optimization_history(study, filename_prefix):
    """Plot optimization history"""
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title(f"Optimization History")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_optimization_history.png")
    plt.close()

def plot_param_importances(study, filename_prefix):
    """Plot parameter importances"""
    plt.figure(figsize=(10, 6))
    try:
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title(f"Parameter Importances")
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_param_importances.png")
    except Exception as e:
        logging.warning(f"Could not plot parameter importances: {str(e)}")
    finally:
        plt.close()

def main():
    print(f"Starting hyperparameter tuning with Optuna")
    
    try:
        # Load data and set up output directory and logging
        data = load_data(METADATA_FILE)
        
        # Run optimization
        summary_file = run_optimization(data)
        
        # Log completion
        logging.info(f"\n{'='*60}")
        logging.info(f"Optimization completed. Results summary saved to: {summary_file}")
        logging.info(f"{'='*60}")
        
        # Create a simple file to indicate successful completion
        completion_file = os.path.join(data['output_dir'], f"COMPLETED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(completion_file, "w") as f:
            f.write(f"Optimization completed at {datetime.now()}\n")
            f.write(f"Total models optimized: {5}\n")
            f.write(f"Trials per model: {NUM_TRIALS}\n")
            f.write(f"Results summary: {summary_file}\n")
            f.write(f"GPU was {'used' if data['use_gpu'] else 'not used'} for XGBoost\n")
        
    except Exception as e:
        print(f"Error in hyperparameter tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()
