EPOCHS = 1000
IN_DIM = 384  # Dimension of MiniLM or BERTweet embeddings (384 for MiniLM, 768 for BERTweet, 100 for Doc2Vec)

LOAD_FROM_OPTUNA = False  # Set to True to load from Optuna, False to load from config
DATASET_NAME = "SanBernardino"  # Update to your dataset
SCALER = "StandardScaler"
EMBEDDING_TYPE = "avg_embedding" 
EMBEDDING_MODEL_NAME = "all_MiniLM_L6_v2"
MIN_SIMILARITY = 0.7
N_TRIALS = 1000  # Define the number of trials

USE_SEED = 42

# Default best parameters (these will be refined by Optuna)
BEST_PARAMS = {
    'learning_rate': 0.001,
    'num_layers': 2, 
    'hidden_dim': 128, 
    'dropout_rate': 0.1, 
    'batch_size': 512, 
    'patience': 50, 
    'l1_lambda': 0.0001, 
    'aggregator_type': 'mean'
}

BEST_EPOCH = 100  # Placeholder value
