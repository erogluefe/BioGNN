"""
Graph construction for multimodal biometric fusion
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple
import numpy as np


class ModalityGraphBuilder:
    """
    Builds graph structures from multimodal biometric data

    Each modality is represented as a node in the graph, with edges
    representing relationships between modalities.
    """

    def __init__(
        self,
        modalities: List[str],
        edge_strategy: str = 'fully_connected',
        learnable_edges: bool = True,
        use_quality_scores: bool = False
    ):
        """
        Args:
            modalities: List of modality names
            edge_strategy: How to construct edges ('fully_connected', 'star', 'hierarchical')
            learnable_edges: Whether edge weights are learnable
            use_quality_scores: Whether to incorporate quality scores as node features
        """
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.edge_strategy = edge_strategy
        self.learnable_edges = learnable_edges
        self.use_quality_scores = use_quality_scores

        # Create modality to index mapping
        self.modality_to_idx = {mod: idx for idx, mod in enumerate(modalities)}

        # Build edge index
        self.edge_index = self._build_edge_index()

    def _build_edge_index(self) -> torch.Tensor:
        """
        Build edge index based on the edge strategy

        Returns:
            edge_index: [2, num_edges] tensor
        """
        n = self.num_modalities
        edges = []

        if self.edge_strategy == 'fully_connected':
            # Create edges between all pairs of modalities
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edges.append([i, j])

        elif self.edge_strategy == 'star':
            # Hub-and-spoke: connect all modalities to a central hub (index 0)
            # Assuming first modality (e.g., face) is the hub
            hub = 0
            for i in range(1, n):
                edges.append([hub, i])
                edges.append([i, hub])

        elif self.edge_strategy == 'hierarchical':
            # Hierarchical structure based on modality reliability
            # face -> fingerprint -> iris -> voice
            for i in range(n - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])

        else:
            raise ValueError(f"Unknown edge strategy: {self.edge_strategy}")

        if not edges:
            # If no edges, create self-loops
            edges = [[i, i] for i in range(n)]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def build_graph(
        self,
        modality_features: Dict[str, torch.Tensor],
        quality_scores: Optional[Dict[str, float]] = None,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Data:
        """
        Build a graph from modality features

        Args:
            modality_features: Dictionary mapping modality name to feature tensor [batch, feature_dim]
            quality_scores: Optional quality scores for each modality
            edge_weights: Optional edge weights [num_edges]

        Returns:
            PyG Data object representing the multimodal graph
        """
        # Stack node features in modality order
        node_features = []
        for modality in self.modalities:
            if modality not in modality_features:
                raise ValueError(f"Missing features for modality: {modality}")
            node_features.append(modality_features[modality])

        # node_features: [num_modalities, batch, feature_dim]
        x = torch.stack(node_features, dim=0)

        # Add quality scores as additional node features if available
        if self.use_quality_scores and quality_scores is not None:
            quality_tensor = torch.tensor(
                [quality_scores.get(mod, 1.0) for mod in self.modalities],
                dtype=torch.float32,
                device=x.device
            )
            # Reshape to [num_modalities, 1] and expand to batch size
            quality_tensor = quality_tensor.unsqueeze(1).unsqueeze(2)
            quality_tensor = quality_tensor.expand(-1, x.size(1), 1)
            x = torch.cat([x, quality_tensor], dim=2)

        # Create graph data
        data = Data(
            x=x,
            edge_index=self.edge_index.to(x.device),
            edge_attr=edge_weights
        )

        return data

    def build_batch_graphs(
        self,
        batch_modality_features: Dict[str, torch.Tensor],
        batch_quality_scores: Optional[List[Dict[str, float]]] = None
    ) -> List[Data]:
        """
        Build a batch of graphs

        Args:
            batch_modality_features: Dictionary of [batch_size, feature_dim] tensors
            batch_quality_scores: List of quality score dictionaries

        Returns:
            List of PyG Data objects
        """
        batch_size = next(iter(batch_modality_features.values())).size(0)
        graphs = []

        for i in range(batch_size):
            # Extract features for this sample
            sample_features = {
                mod: feat[i:i+1] for mod, feat in batch_modality_features.items()
            }

            # Extract quality scores if available
            quality = batch_quality_scores[i] if batch_quality_scores else None

            # Build graph
            graph = self.build_graph(sample_features, quality)
            graphs.append(graph)

        return graphs


class AdaptiveEdgeWeighting(nn.Module):
    """
    Learns adaptive edge weights based on modality features and quality
    """

    def __init__(
        self,
        num_modalities: int,
        feature_dim: int,
        hidden_dim: int = 128
    ):
        """
        Args:
            num_modalities: Number of modalities
            feature_dim: Feature dimension for each modality
            hidden_dim: Hidden dimension for edge weight computation
        """
        super().__init__()

        self.num_modalities = num_modalities
        self.feature_dim = feature_dim

        # MLP for computing edge weights from node features
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Edge weights in [0, 1]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive edge weights

        Args:
            node_features: [num_nodes, feature_dim]
            edge_index: [2, num_edges]

        Returns:
            edge_weights: [num_edges]
        """
        src_features = node_features[edge_index[0]]
        dst_features = node_features[edge_index[1]]

        # Concatenate source and destination features
        edge_features = torch.cat([src_features, dst_features], dim=1)

        # Compute edge weights
        edge_weights = self.edge_mlp(edge_features).squeeze(-1)

        return edge_weights


class ModalityAttention(nn.Module):
    """
    Attention mechanism to weight different modalities based on their reliability
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4
    ):
        """
        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        modality_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_features: [num_modalities, batch_size, feature_dim]
            mask: Optional mask [num_modalities] to ignore certain modalities

        Returns:
            attended_features: [num_modalities, batch_size, feature_dim]
            attention_weights: [batch_size, num_modalities, num_modalities]
        """
        num_mod, batch_size, _ = modality_features.shape

        # Reshape for multi-head attention
        # [num_modalities, batch_size, feature_dim] -> [batch_size, num_modalities, num_heads, head_dim]
        q = self.q_proj(modality_features).transpose(0, 1).reshape(
            batch_size, num_mod, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch_size, num_heads, num_modalities, head_dim]

        k = self.k_proj(modality_features).transpose(0, 1).reshape(
            batch_size, num_mod, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(modality_features).transpose(0, 1).reshape(
            batch_size, num_mod, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch_size, num_heads, num_modalities, num_modalities]

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)
        # [batch_size, num_heads, num_modalities, head_dim]

        # Reshape back
        attended = attended.transpose(1, 2).reshape(batch_size, num_mod, self.feature_dim)
        attended = self.out_proj(attended)
        attended = attended.transpose(0, 1)
        # [num_modalities, batch_size, feature_dim]

        # Average attention weights across heads
        attn_weights = attn_weights.mean(dim=1)
        # [batch_size, num_modalities, num_modalities]

        return attended, attn_weights


class QualityAwarePooling(nn.Module):
    """
    Pools modality features based on quality scores
    """

    def __init__(
        self,
        feature_dim: int,
        pooling_method: str = 'weighted_avg'
    ):
        """
        Args:
            feature_dim: Feature dimension
            pooling_method: 'weighted_avg', 'max', 'attention'
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.pooling_method = pooling_method

        if pooling_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim + 1, 128),  # +1 for quality score
                nn.Tanh(),
                nn.Linear(128, 1)
            )

    def forward(
        self,
        modality_features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            modality_features: [num_modalities, batch_size, feature_dim]
            quality_scores: [num_modalities] or [num_modalities, batch_size]

        Returns:
            pooled_features: [batch_size, feature_dim]
        """
        if self.pooling_method == 'max':
            # Max pooling across modalities
            pooled, _ = torch.max(modality_features, dim=0)
            return pooled

        elif self.pooling_method == 'weighted_avg':
            if quality_scores is None:
                # Simple average
                return torch.mean(modality_features, dim=0)
            else:
                # Weighted average by quality scores
                if quality_scores.dim() == 1:
                    quality_scores = quality_scores.unsqueeze(1).unsqueeze(2)
                else:
                    quality_scores = quality_scores.unsqueeze(2)

                # Normalize weights
                weights = torch.softmax(quality_scores, dim=0)
                pooled = torch.sum(modality_features * weights, dim=0)
                return pooled

        elif self.pooling_method == 'attention':
            # Attention-based pooling
            num_mod, batch_size, feat_dim = modality_features.shape

            # Add quality scores to features
            if quality_scores is not None:
                if quality_scores.dim() == 1:
                    quality_scores = quality_scores.unsqueeze(1).expand(-1, batch_size)
                quality_scores = quality_scores.unsqueeze(2)

                # Concatenate
                features_with_quality = torch.cat([modality_features, quality_scores], dim=2)
            else:
                # Use dummy quality scores
                dummy_quality = torch.ones(num_mod, batch_size, 1, device=modality_features.device)
                features_with_quality = torch.cat([modality_features, dummy_quality], dim=2)

            # Compute attention weights
            attn_logits = self.attention(features_with_quality)  # [num_mod, batch_size, 1]
            attn_weights = torch.softmax(attn_logits, dim=0)

            # Apply attention
            pooled = torch.sum(modality_features * attn_weights, dim=0)
            return pooled

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
