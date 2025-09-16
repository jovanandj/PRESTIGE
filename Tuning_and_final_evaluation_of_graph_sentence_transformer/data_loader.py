import dgl
from config import DATASET_NAME, SCALER, EMBEDDING_TYPE, MIN_SIMILARITY, EMBEDDING_MODEL_NAME
import logging

def load_graph_with_masks():
    """Load a single graph with multiple train/val/test masks"""
    # Construct filename based on config parameters
    # Determine edge descriptor based on SCALER and potentially EDGE_SIMILARITY_METHOD if needed
    # Assuming 'topics' method was used with scaling as per the example filename
    scaled_str = 'scaled' # Assuming SCALE_DATA=True
    topics_scaled_str = '_topicscaled' # Assuming SCALE_TOPICS=True
    edge_descriptor = f'topic_edges_{scaled_str}{topics_scaled_str}'
    # If EDGE_SIMILARITY_METHOD was 'embeddings', adjust accordingly:
    # edge_descriptor = f'embedding_edges_{scaled_str}_no_topics_used'

    graph_filename = f"{DATASET_NAME}_full_{edge_descriptor}_{EMBEDDING_MODEL_NAME}_pruned_{MIN_SIMILARITY}_{SCALER}.bin"

    try:
        # Load graph and seeds information
        graphs, _ = dgl.load_graphs(graph_filename)
        full_graph = graphs[0]

        logging.info(f"Loaded graph with {full_graph.num_nodes()} nodes and {full_graph.num_edges()} edges")

        # Extract masks for the default seed (42)
        train_mask = full_graph.ndata['train_mask']
        val_mask = full_graph.ndata['val_mask']
        test_mask = full_graph.ndata['test_mask']

        logging.info(f"Using default masks: {train_mask.sum()} train, {val_mask.sum()} val, {test_mask.sum()} test nodes")

        return full_graph, train_mask, val_mask, test_mask

    except Exception as e:
        logging.error(f"Error loading graph from {graph_filename}: {str(e)}") # Log filename on error
        raise