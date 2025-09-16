import torch
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score
import logging

def train_model(model, criterion, optimizer, graph, train_batches, l1_lambda, use_full_graph=False, train_mask=None):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
        
    if use_full_graph:
        # Full graph training with mask
        node_features = graph.ndata['feat']
        node_labels = graph.ndata['label']
        
        # Forward pass
        outputs = model(graph, node_features)

         # Only compute loss for training nodes
        if train_mask is not None:
            outputs = outputs[train_mask]
            node_labels = node_labels[train_mask]

        loss = criterion(outputs, node_labels)
        
        # Add L1 regularization
        l1_loss = 0
        for param in model.parameters():
            l1_loss += torch.norm(param, 1)
        loss += l1_lambda * l1_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Add this line to accumulate the loss
        total_loss += loss.item()
        num_batches = 1
    
    else:
        logging.info("You chose training in batch mode, but there is no batch mode implemented yet")
    
    return round(total_loss / max(1, num_batches), 4)

def validate_model(model, graph, validation_batches=None, use_full_graph=False, val_mask=None):
    """Validate the model using Matthews Correlation Coefficient."""
    
    # Set deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.eval()
    device = graph.device  # Get the device from the graph
    
    if use_full_graph:
        with torch.no_grad():
            node_features = graph.ndata['feat']
            node_labels = graph.ndata['label']
            
            # Forward pass
            outputs = model(graph, node_features)

            # Only evaluate on validation nodes
            if val_mask is not None:
                outputs = outputs[val_mask]
                node_labels = node_labels[val_mask]
                
            pred_labels = torch.argmax(outputs, dim=1)
            
            # Process exactly like test_model
            all_labels = node_labels.cpu().tolist()
            all_predictions = pred_labels.cpu().tolist()
            
            # Calculate MCC using the same method
            mcc = matthews_corrcoef(all_labels, all_predictions)
            
        # Round to 4 decimal places
        return round(mcc, 4)
        
    else:
        logging.info("You chose validation in batch mode, but there is no batch mode implemented yet")

def test_model(model, graph, test_batches=None, use_full_graph=False, test_mask=None):
    """Evaluate the model on test data and return multiple performance metrics."""

    # Set deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model.eval()

    if use_full_graph:
        # Full graph testing
        with torch.no_grad():
            node_features = graph.ndata['feat']
            node_labels = graph.ndata['label']
            
            # Forward pass
            outputs = model(graph, node_features)

            # Only evaluate on test nodes
            if test_mask is not None:
                outputs = outputs[test_mask]
                node_labels = node_labels[test_mask]
                
            pred_labels = torch.argmax(outputs, dim=1)
            
            # Move tensors to CPU for sklearn metrics
            all_labels = node_labels.cpu().tolist()
            all_predictions = pred_labels.cpu().tolist()
    else:
        logging.info("You chose testing in batch mode, but there is no batch mode implemented yet")


    # Calculate all performance metrics and round to 4 decimal places
    accuracy = round(accuracy_score(all_labels, all_predictions), 4)
    f1_macro = round(f1_score(all_labels, all_predictions, average='macro'), 4)
    precision_macro = round(precision_score(all_labels, all_predictions, average='macro'), 4)
    recall_macro = round(recall_score(all_labels, all_predictions, average='macro'), 4)
    f1_weighted = round(f1_score(all_labels, all_predictions, average='weighted'), 4)
    precision_weighted = round(precision_score(all_labels, all_predictions, average='weighted'), 4)
    recall_weighted = round(recall_score(all_labels, all_predictions, average='weighted'), 4)
    mcc = round(matthews_corrcoef(all_labels, all_predictions), 4)

    return accuracy, f1_macro, precision_macro, recall_macro, f1_weighted, precision_weighted, recall_weighted, mcc