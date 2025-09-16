import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report
)
from datetime import datetime
import glob
import logging

METADATA_FILE = 'Sutherland_all_MiniLM_L6_v2_avg_embedding_20250402_132901_metadata.json'

# Setup logging
def setup_logging(log_file='ml_model_final_evaluation.log'):
    """Set up logging configuration"""    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_test_data():
    """Load test data from metadata file"""
    logging.info(f"Loading test data from metadata: {METADATA_FILE}")
    
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    # Extract file path for test data
    test_file = metadata['files']['test']
    
    # Load dataset
    test_df = pd.read_csv(test_file)
    
    logging.info(f"Loaded {len(test_df)} test samples")
    
    # Extract features and labels
    feature_cols = [col for col in test_df.columns if col.startswith('dim_')]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['sentiment'].values
    
    # Get user IDs for reference
    user_ids = test_df['user_id'].values
    
    # Extract sentiment mapping from metadata
    sentiment_mapping = metadata.get('sentiment_mapping', {
        'positive': 0,
        'neutral': 1,
        'negative': 2
    })
    
    # Encode labels
    y_test_encoded = np.array([sentiment_mapping[label] for label in y_test])
    
    # Get class names
    class_names = ['positive', 'neutral', 'negative']
    
    return X_test, y_test, y_test_encoded, user_ids, class_names, metadata

def find_model_files():
    """Find all model files in output directory"""
    model_files = glob.glob("*_best_model_*.pkl")
    return sorted(model_files)

def load_model(model_file):
    """Load a trained model from file"""
    logging.info(f"Loading model: {os.path.basename(model_file)}")
    model_data = joblib.load(model_file)
    
    model = model_data.get('model')
    sentiment_mapping = model_data.get('sentiment_mapping', {})
    
    # Extract model type from filename
    model_type = os.path.basename(model_file).split('_best_model_')[0]
    
    return model, sentiment_mapping, model_type

def evaluate_model(model, X_test, y_test_encoded, class_names):
    """Evaluate model on test set and return metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test_encoded, y_pred),
        
        # Macro metrics (unweighted average)
        'precision_macro': precision_score(y_test_encoded, y_pred, average='macro'),
        'recall_macro': recall_score(y_test_encoded, y_pred, average='macro'),
        'f1_macro': f1_score(y_test_encoded, y_pred, average='macro'),
        
        # Weighted metrics (accounts for class imbalance)
        'precision': precision_score(y_test_encoded, y_pred, average='weighted'),
        'recall': recall_score(y_test_encoded, y_pred, average='weighted'),
        'f1': f1_score(y_test_encoded, y_pred, average='weighted'),
        
        'mcc': matthews_corrcoef(y_test_encoded, y_pred)
    }
    
    # Get confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Get detailed classification report
    report = classification_report(y_test_encoded, y_pred, 
                                  target_names=class_names,
                                  output_dict=True)
    
    return metrics, cm, report, y_pred

def plot_confusion_matrix(cm, class_names, model_type):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_type}')
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cm_file =  f"{model_type}_confusion_matrix_{timestamp}.png"
    plt.savefig(cm_file)
    plt.close()
    
    return cm_file

def main():
    # Create output directory for evaluation results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging
    setup_logging()
    logging.info(f"Starting model evaluation on test set")
    
    # Load test data
    X_test, y_test, y_test_encoded, user_ids, class_names, metadata = load_test_data()
    
    # Get output directory from metadata
    event_name = metadata.get('event_name', 'unknown_event')
    model_name = metadata.get('model_name', 'unknown_model')
    embedding_type = metadata.get('embedding_type', 'unknown_embedding')
    
    # Find model files
    model_files = find_model_files()
    
    if not model_files:
        logging.error(f"No model files found")
        return
    
    # Prepare results storage
    all_results = []
    
    # Load and evaluate each model
    for model_file in model_files:
        try:
            # Load model
            model, model_sentiment_mapping, model_type = load_model(model_file)
            
            # Evaluate model
            metrics, cm, report, y_pred = evaluate_model(model, X_test, y_test_encoded, class_names)
            
            # Plot confusion matrix
            cm_file = plot_confusion_matrix(cm, class_names, model_type)
            
            # Store results
            result = {
                'model_type': model_type,
                'model_file': os.path.basename(model_file),
                **metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
            }
            
            all_results.append(result)
            
            # Log results
            logging.info(f"Model: {model_type}")
            logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logging.info(f"  MCC: {metrics['mcc']:.4f}")
            logging.info(f"  Weighted metrics (account for class imbalance):")
            logging.info(f"    Precision: {metrics['precision']:.4f}")
            logging.info(f"    Recall: {metrics['recall']:.4f}")
            logging.info(f"    F1: {metrics['f1']:.4f}")
            logging.info(f"  Macro metrics (unweighted):")
            logging.info(f"    Precision: {metrics['precision_macro']:.4f}")
            logging.info(f"    Recall: {metrics['recall_macro']:.4f}")
            logging.info(f"    F1: {metrics['f1_macro']:.4f}")
            logging.info(f"  Confusion matrix saved to: {cm_file}")
            
        except Exception as e:
            logging.error(f"Error evaluating model {model_file}: {str(e)}")
    
    # Save all results to JSON
    results_file = f"evaluation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary CSV
    summary_data = []
    for result in all_results:
        summary_data.append({
            'model_type': result['model_type'],
            'accuracy': result['accuracy'],
            # Macro metrics
            'precision_macro': result['precision_macro'],
            'recall_macro': result['recall_macro'],
            'f1_macro': result['f1_macro'],
            # Weighted metrics
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'mcc': result['mcc']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"evaluation_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Display summary table
    logging.info("\n" + "=" * 80)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 80)
    logging.info(f"\n{summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}')}")
    logging.info("\n" + "=" * 80)
    logging.info(f"Detailed results saved to: {results_file}")
    logging.info(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()