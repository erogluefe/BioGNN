"""
GraphSAGE for multimodal biometric fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from typing import Optional


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer with batch normalization and dropout"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean',
        dropout: float = 0.5,
        use_bn: bool = True,
        normalize: bool = True
    ):
        """
        Args:
            in_channels: Input dimension
            out_channels: Output dimension
            aggr: Aggregation method ('mean', 'max', 'lstm')
            dropout: Dropout rate
            use_bn: Use batch normalization
            normalize: L2 normalize output
        """
        super().__init__()

        self.conv = SAGEConv(
            in_channels,
            out_channels,
            aggr=aggr,
            normalize=normalize
        )

        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MultimodalGraphSAGE(nn.Module):
    """
    GraphSAGE-based model for multimodal biometric fusion

    GraphSAGE samples and aggregates features from a node's local neighborhood,
    making it suitable for learning from multimodal biometric data where
    different modalities can be seen as neighbors in a graph.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        num_classes: int = 2,
        aggr: str = 'mean',
        dropout: float = 0.5,
        pooling: str = 'mean',
        use_residual: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions
            num_classes: Number of output classes
            aggr: Aggregation method ('mean', 'max', 'lstm')
            dropout: Dropout rate
            pooling: Global pooling method
            use_residual: Use residual connections
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.pooling = pooling
        self.use_residual = use_residual

        # Build GraphSAGE layers
        self.convs = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            self.convs.append(
                GraphSAGELayer(
                    dims[i],
                    dims[i + 1],
                    aggr=aggr,
                    dropout=dropout
                )
            )

        # Residual projection layers (for dimension matching)
        if use_residual:
            self.residual_projs = nn.ModuleList()
            for i in range(len(hidden_dims)):
                if dims[i] != dims[i + 1]:
                    self.residual_projs.append(nn.Linear(dims[i], dims[i + 1]))
                else:
                    self.residual_projs.append(None)

        # Determine classifier input dimension
        if pooling == 'concat':
            classifier_input_dim = hidden_dims[-1] * 2
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

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment vector

        Returns:
            logits: Class logits
            embeddings: Graph embeddings
        """
        residual = x

        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # Add residual connection
            if self.use_residual:
                if self.residual_projs[i] is not None:
                    residual = self.residual_projs[i](residual)
                x = x + residual
                residual = x

        # Global pooling
        if batch is None:
            if self.pooling == 'mean':
                embeddings = x.mean(dim=0, keepdim=True)
            elif self.pooling == 'max':
                embeddings = x.max(dim=0, keepdim=True)[0]
            elif self.pooling == 'concat':
                mean_pool = x.mean(dim=0, keepdim=True)
                max_pool = x.max(dim=0, keepdim=True)[0]
                embeddings = torch.cat([mean_pool, max_pool], dim=1)
        else:
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

    def get_embeddings(self, x, edge_index, batch=None):
        """Get graph embeddings without classification"""
        with torch.no_grad():
            _, embeddings = self.forward(x, edge_index, batch)
        return embeddings


class AdaptiveGraphSAGE(nn.Module):
    """
    GraphSAGE with adaptive aggregation that learns which aggregation
    method to use for each layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Multiple aggregators
        self.mean_convs = nn.ModuleList()
        self.max_convs = nn.ModuleList()

        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            self.mean_convs.append(
                GraphSAGELayer(dims[i], dims[i + 1], aggr='mean', dropout=dropout)
            )
            self.max_convs.append(
                GraphSAGELayer(dims[i], dims[i + 1], aggr='max', dropout=dropout)
            )

        # Aggregation weights (learnable)
        self.aggr_weights = nn.ParameterList([
            nn.Parameter(torch.tensor([0.5, 0.5]))
            for _ in range(len(hidden_dims))
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass with adaptive aggregation

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment

        Returns:
            logits: Class logits
            embeddings: Graph embeddings
        """
        for i in range(len(self.hidden_dims)):
            # Apply both aggregators
            x_mean = self.mean_convs[i](x, edge_index)
            x_max = self.max_convs[i](x, edge_index)

            # Weighted combination
            weights = F.softmax(self.aggr_weights[i], dim=0)
            x = weights[0] * x_mean + weights[1] * x_max

        # Global pooling
        if batch is None:
            embeddings = x.mean(dim=0, keepdim=True)
        else:
            embeddings = global_mean_pool(x, batch)

        # Classification
        logits = self.classifier(embeddings)

        return logits, embeddings


class MiniBatchGraphSAGE(nn.Module):
    """
    GraphSAGE optimized for mini-batch training on large graphs
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        num_classes: int = 2,
        aggr: str = 'mean',
        dropout: float = 0.5,
        num_layers: int = 2
    ):
        super().__init__()

        self.num_layers = num_layers

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [input_dim] + hidden_dims

        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggr))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling
        if batch is None:
            embeddings = x.mean(dim=0, keepdim=True)
        else:
            embeddings = global_mean_pool(x, batch)

        logits = self.classifier(embeddings)

        return logits, embeddings
