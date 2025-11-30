"""
Graph Attention Network (GAT) for multimodal biometric fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_max_pool
from typing import Optional


class GATLayer(nn.Module):
    """Single GAT layer with batch normalization and dropout"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.5,
        use_bn: bool = True,
        concat: bool = True,
        use_v2: bool = True,
        edge_dim: int = 1
    ):
        super().__init__()

        self.heads = heads
        self.concat = concat

        # Use GATv2 (more expressive) or original GAT
        conv_class = GATv2Conv if use_v2 else GATConv

        self.conv = conv_class(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            edge_dim=edge_dim  # Support edge attributes (e.g., adaptive edge weights)
        )

        # Batch normalization
        bn_dim = out_channels * heads if concat else out_channels
        self.bn = nn.BatchNorm1d(bn_dim) if use_bn else None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        """
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            return_attention_weights: Whether to return attention weights

        Returns:
            x: Updated node features
            attention_weights: (Optional) Attention weights
        """
        if return_attention_weights:
            x, (edge_index, attention) = self.conv(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            x = self.conv(x, edge_index, edge_attr=edge_attr)
            attention = None

        if self.bn is not None:
            x = self.bn(x)

        x = F.elu(x)  # ELU works better with GAT
        x = self.dropout(x)

        if return_attention_weights:
            return x, (edge_index, attention)
        return x


class MultimodalGAT(nn.Module):
    """
    GAT-based model for multimodal biometric fusion

    Uses attention mechanism to learn which modalities are more important
    for the verification decision.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        heads: list = [4, 4],
        num_classes: int = 2,
        dropout: float = 0.5,
        pooling: str = 'mean',
        use_v2: bool = True,
        edge_dim: int = 1
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden dimensions
            heads: List of attention heads for each layer
            num_classes: Number of output classes
            dropout: Dropout rate
            pooling: Global pooling method
            use_v2: Use GATv2Conv (more expressive)
            edge_dim: Edge feature dimension (1 for scalar edge weights)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.heads = heads
        self.num_classes = num_classes
        self.pooling = pooling

        # Ensure heads list matches hidden_dims
        if len(heads) != len(hidden_dims):
            heads = [heads[0]] * len(hidden_dims)

        # Build GAT layers
        self.convs = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            # Last layer: don't concatenate heads
            concat = i < len(hidden_dims) - 1

            # Adjust input dimension for multi-head concatenation
            if i > 0:
                in_dim = dims[i] * heads[i - 1]
            else:
                in_dim = dims[i]

            self.convs.append(
                GATLayer(
                    in_dim,
                    dims[i + 1],
                    heads=heads[i],
                    dropout=dropout,
                    concat=concat,
                    use_v2=use_v2,
                    edge_dim=edge_dim
                )
            )

        # Determine final dimension
        # Last layer doesn't concatenate heads, so output is just hidden_dims[-1]
        final_dim = hidden_dims[-1]

        # Determine classifier input dimension
        if pooling == 'concat':
            classifier_input_dim = final_dim * 2
        else:
            classifier_input_dim = final_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None, return_attention=False):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge attributes
            batch: Batch assignment vector
            return_attention: Whether to return attention weights

        Returns:
            logits: Class logits
            embeddings: Graph embeddings
            attention_weights: (Optional) Attention weights from all layers
        """
        attention_weights = []

        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            if return_attention and i == len(self.convs) - 1:
                # Get attention from last layer
                x, (att_edge_index, att_weights) = conv(
                    x, edge_index, edge_attr, return_attention_weights=True
                )
                attention_weights.append((att_edge_index, att_weights))
            else:
                x = conv(x, edge_index, edge_attr)

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

        if return_attention:
            return logits, embeddings, attention_weights
        return logits, embeddings

    def get_attention_weights(self, x, edge_index, edge_attr=None):
        """
        Get attention weights for visualization

        Returns:
            attention_weights: Dictionary mapping layer index to attention weights
        """
        with torch.no_grad():
            _, _, attention_weights = self.forward(
                x, edge_index, edge_attr, return_attention=True
            )
        return attention_weights


class HierarchicalGAT(nn.Module):
    """
    Hierarchical GAT with intra-modality and inter-modality attention
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_modalities: int = 4,
        heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            input_dim: Input feature dimension per modality
            hidden_dim: Hidden dimension
            num_modalities: Number of modalities
            heads: Number of attention heads
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()

        self.num_modalities = num_modalities

        # Intra-modality attention (within each modality)
        self.intra_attn = nn.ModuleList([
            nn.MultiheadAttention(input_dim, heads, dropout=dropout)
            for _ in range(num_modalities)
        ])

        # Inter-modality attention (across modalities)
        self.inter_gat = MultimodalGAT(
            input_dim=input_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            heads=[heads, heads],
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices
            batch: Batch assignment

        Returns:
            logits: Class logits
            embeddings: Graph embeddings
        """
        # Apply intra-modality attention
        # (This is a simplified version; in practice, you'd need to
        # properly organize the data for intra-modality attention)

        # Apply inter-modality GAT
        logits, embeddings = self.inter_gat(x, edge_index, batch=batch)

        return logits, embeddings
