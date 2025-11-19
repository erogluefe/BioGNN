"""
Graph Convolutional Network (GCN) for multimodal biometric fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from typing import Optional


class GCNLayer(nn.Module):
    """Single GCN layer with batch normalization and dropout"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        use_bn: bool = True
    ):
        super().__init__()

        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MultimodalGCN(nn.Module):
    """
    GCN-based model for multimodal biometric fusion

    Architecture:
    - Multiple GCN layers to propagate information between modalities
    - Global pooling to aggregate node features
    - MLP classifier for final prediction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        num_classes: int = 2,
        dropout: float = 0.5,
        pooling: str = 'mean',
        use_residual: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension for each modality
            hidden_dims: List of hidden dimensions for GCN layers
            num_classes: Number of output classes (2 for verification)
            dropout: Dropout rate
            pooling: Global pooling method ('mean', 'max', 'concat')
            use_residual: Whether to use residual connections
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.pooling = pooling
        self.use_residual = use_residual

        # Build GCN layers
        self.convs = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            self.convs.append(
                GCNLayer(dims[i], dims[i + 1], dropout=dropout)
            )

        # Determine classifier input dimension
        if pooling == 'concat':
            classifier_input_dim = hidden_dims[-1] * 2  # mean + max
        else:
            classifier_input_dim = hidden_dims[-1]

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            logits: Class logits [batch_size, num_classes]
            embeddings: Graph embeddings [batch_size, hidden_dims[-1]]
        """
        # Store input for residual connection
        residual = x

        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)

            # Add residual connection for deeper layers
            if self.use_residual and i > 0 and x.size(-1) == residual.size(-1):
                x = x + residual
            residual = x

        # Global pooling
        if batch is None:
            # Single graph case
            if self.pooling == 'mean':
                embeddings = x.mean(dim=0, keepdim=True)
            elif self.pooling == 'max':
                embeddings = x.max(dim=0, keepdim=True)[0]
            elif self.pooling == 'concat':
                mean_pool = x.mean(dim=0, keepdim=True)
                max_pool = x.max(dim=0, keepdim=True)[0]
                embeddings = torch.cat([mean_pool, max_pool], dim=1)
        else:
            # Batch of graphs
            if self.pooling == 'mean':
                embeddings = global_mean_pool(x, batch)
            elif self.pooling == 'max':
                embeddings = global_max_pool(x, batch)
            elif self.pooling == 'concat':
                mean_pool = global_mean_pool(x, batch)
                max_pool = global_max_pool(x, batch)
                embeddings = torch.cat([mean_pool, max_pool], dim=1)

        # Classification
        logits = self.classifier(embeddings)

        return logits, embeddings

    def get_embeddings(self, x, edge_index, edge_weight=None, batch=None):
        """
        Get only the graph embeddings without classification

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            embeddings: Graph embeddings [batch_size, embedding_dim]
        """
        with torch.no_grad():
            _, embeddings = self.forward(x, edge_index, edge_weight, batch)
        return embeddings


class DeepGCN(nn.Module):
    """
    Deeper GCN with more sophisticated architecture for complex fusion
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.5,
        use_jk: bool = True  # Jumping Knowledge
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for all layers
            num_layers: Number of GCN layers
            num_classes: Number of output classes
            dropout: Dropout rate
            use_jk: Use Jumping Knowledge connections
        """
        super().__init__()

        self.num_layers = num_layers
        self.use_jk = use_jk

        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Jumping Knowledge: concatenate all layer outputs
        if use_jk:
            classifier_input = hidden_dim * num_layers
        else:
            classifier_input = hidden_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Forward pass with optional Jumping Knowledge

        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Optional edge weights
            batch: Batch assignment

        Returns:
            logits: Class logits
            embeddings: Graph embeddings
        """
        xs = []

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            xs.append(x)

        # Jumping Knowledge: concatenate all intermediate representations
        if self.use_jk:
            x = torch.cat(xs, dim=1)
        else:
            x = xs[-1]

        # Global pooling
        if batch is None:
            embeddings = x.mean(dim=0, keepdim=True)
        else:
            embeddings = global_mean_pool(x, batch)

        # Classification
        logits = self.classifier(embeddings)

        return logits, embeddings
