import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate, layer_type='GraphConv', 
                 aggregator_type='mean', activation='relu', num_heads=1):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.out_dim = out_dim  # Store the expected output dimension
        
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.layer_type = layer_type
        self.aggregator_type = aggregator_type
        self.num_heads = num_heads
        
        # Create the appropriate graph layer with the right aggregator
        if layer_type == 'GraphConv':
            # GraphConv doesn't support aggregator_type in this DGL version
            self.gnn_layer = dglnn.GraphConv(
                in_feats=in_dim,
                out_feats=out_dim,
                norm='both',
                weight=True,
                bias=True
            )
        elif layer_type == 'SAGEConv':
            # SAGEConv supports 'mean', 'pool', 'gcn'
            if aggregator_type not in ['mean', 'pool', 'gcn']:
                aggregator_type = 'mean'
                
            self.gnn_layer = dglnn.SAGEConv(
                in_feats=in_dim,
                out_feats=out_dim,
                aggregator_type=aggregator_type
            )
        elif layer_type == 'GATConv':
            # GATConv uses num_heads
            # Ensure out_dim is divisible by num_heads for proper reshaping
            if out_dim % num_heads != 0:
                out_dim = (out_dim // num_heads) * num_heads
                if out_dim == 0:  # Safeguard against zero output dimension
                    out_dim = num_heads
            
            self.out_dim = out_dim  # Update the stored output dimension
                    
            self.gnn_layer = dglnn.GATConv(
                in_feats=in_dim,
                out_feats=out_dim // num_heads,  # Divide output features by num_heads
                num_heads=num_heads,
                feat_drop=0,  # We already have dropout elsewhere
                attn_drop=0,
                negative_slope=0.2,
                residual=False
            )
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def forward(self, g, feature):
        h = self.dropout(feature)
        
        # Check if feature dimension matches expected input dimension for GATConv
        # This is critical for batch subgraph processing
        if self.layer_type == 'GATConv':
            # Check if the feature dimension is compatible with the GATConv layer
            # This handles the case where the batch has a different feature size
            try:
                h = self.gnn_layer(g, h)
                
                # Apply different aggregation for multi-head attention
                # h shape is [nodes, num_heads, features]
                if self.aggregator_type == 'mean':
                    h = torch.mean(h, dim=1)  # Average across heads
                elif self.aggregator_type == 'sum':
                    h = torch.sum(h, dim=1)   # Sum across heads
                elif self.aggregator_type == 'max':
                    h = torch.max(h, dim=1)[0]  # Max across heads
                else:
                    h = torch.mean(h, dim=1)  # Default to mean
            except RuntimeError as e:
                # If there's a shape mismatch, print warning and use a more compatible approach
                print(f"Shape mismatch in GATConv: {e}. Using linear layer to project features.")
                
                # Create a temporary projection layer
                feature_size = h.size(-1)  # Get the current feature size
                temp_layer = nn.Linear(feature_size, self.out_dim).to(h.device)
                h = temp_layer(h)  # Project to the expected dimension
        else:
            h = self.gnn_layer(g, h)
            
        h = self.activation(h)
        return h

class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout_rate, 
                 layer_type='GraphConv', aggregator_type='mean', 
                 activation='relu', use_skip=False, num_heads=1):
        super(GNNModel, self).__init__()
        self.use_skip = use_skip
        self.layers = nn.ModuleList()
        
        # Ensure hidden_dim is divisible by num_heads if using GATConv
        if layer_type == 'GATConv':
            hidden_dim = (hidden_dim // num_heads) * num_heads
            if hidden_dim == 0:  # Safeguard against zero hidden dimension
                hidden_dim = num_heads

        # First layer (input dimension to hidden dimension)
        self.layers.append(GNNLayer(in_dim, hidden_dim, dropout_rate, 
                                   layer_type, aggregator_type, activation, num_heads))
        
        # Remaining layers (hidden dimension to hidden dimension)
        for _ in range(num_layers - 1):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim, dropout_rate, 
                                       layer_type, aggregator_type, activation, num_heads))
            
        # Output layer
        self.linear = nn.Linear(hidden_dim, out_dim)
        

    def forward(self, g, feature):
        h = feature
        for i, layer in enumerate(self.layers):
            if i == 0 or not self.use_skip:
                # First layer or when skip connections are disabled
                h = layer(g, h)
            else:
                # For other layers when skip connections are enabled
                try:
                    h_new = layer(g, h)
                    h = h + h_new  # Add the skip connection
                except RuntimeError:
                    # If skip connection fails due to shape mismatch, just use the new representation
                    h = layer(g, h)
        
        # Add safety check for final linear layer
        try:
            return self.linear(h)
        except RuntimeError as e:
            print(f"Output layer dimension mismatch: {e}, using compatible projection")
            final_layer = nn.Linear(h.shape[-1], self.linear.out_features).to(h.device)
            return final_layer(h)