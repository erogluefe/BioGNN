"""
End-to-end multimodal fusion architecture using GNN
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data, Batch

from ..data.feature_extractors import get_feature_extractor
from ..models.gcn import MultimodalGCN
from ..models.gat import MultimodalGAT
from ..models.graphsage import MultimodalGraphSAGE
from .graph_builder import (
    ModalityGraphBuilder,
    AdaptiveEdgeWeighting,
    ModalityAttention,
    QualityAwarePooling
)


class MultimodalBiometricFusion(nn.Module):
    """
    End-to-end multimodal biometric fusion system

    Pipeline:
    1. Extract features from each modality using specialized CNNs
    2. Build a graph with modalities as nodes
    3. Apply GNN to propagate information between modalities
    4. Make verification decision
    """

    def __init__(
        self,
        modalities: List[str],
        feature_dim: int = 512,
        gnn_type: str = 'gcn',
        gnn_config: Optional[Dict] = None,
        edge_strategy: str = 'fully_connected',
        use_adaptive_edges: bool = True,
        use_quality_scores: bool = False,
        freeze_feature_extractors: bool = False
    ):
        """
        Args:
            modalities: List of modality names
            feature_dim: Feature dimension for each modality
            gnn_type: Type of GNN ('gcn', 'gat', 'graphsage')
            gnn_config: Configuration for GNN model
            edge_strategy: How to connect modality nodes
            use_adaptive_edges: Learn edge weights adaptively
            use_quality_scores: Use quality scores as node features
            freeze_feature_extractors: Freeze pretrained feature extractors
        """
        super().__init__()

        self.modalities = modalities
        self.feature_dim = feature_dim
        self.gnn_type = gnn_type
        self.use_adaptive_edges = use_adaptive_edges

        # Feature extractors for each modality
        self.feature_extractors = nn.ModuleDict({
            modality: get_feature_extractor(modality, feature_dim)
            for modality in modalities
        })

        # Optionally freeze feature extractors
        if freeze_feature_extractors:
            for extractor in self.feature_extractors.values():
                for param in extractor.parameters():
                    param.requires_grad = False

        # Graph builder
        self.graph_builder = ModalityGraphBuilder(
            modalities=modalities,
            edge_strategy=edge_strategy,
            use_quality_scores=use_quality_scores
        )

        # Adaptive edge weighting
        if use_adaptive_edges:
            self.edge_weighting = AdaptiveEdgeWeighting(
                num_modalities=len(modalities),
                feature_dim=feature_dim
            )
        else:
            self.edge_weighting = None

        # GNN model
        gnn_config = gnn_config or {}
        gnn_config.setdefault('input_dim', feature_dim)
        gnn_config.setdefault('num_classes', 2)

        if gnn_type == 'gcn':
            self.gnn = MultimodalGCN(**gnn_config)
        elif gnn_type == 'gat':
            self.gnn = MultimodalGAT(**gnn_config)
        elif gnn_type == 'graphsage':
            self.gnn = MultimodalGraphSAGE(**gnn_config)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

    def extract_features(
        self,
        modality_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from raw inputs

        Args:
            modality_inputs: Dictionary mapping modality name to raw input tensor

        Returns:
            Dictionary mapping modality name to feature tensor
        """
        features = {}
        for modality, input_data in modality_inputs.items():
            if modality not in self.feature_extractors:
                raise ValueError(f"No feature extractor for modality: {modality}")
            features[modality] = self.feature_extractors[modality](input_data)
        return features

    def build_modality_graph(
        self,
        modality_features: Dict[str, torch.Tensor],
        quality_scores: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Build graph from modality features

        Args:
            modality_features: Dictionary of features [batch, feature_dim]
            quality_scores: Optional quality scores

        Returns:
            node_features: [num_modalities, batch, feature_dim]
            edge_index: [2, num_edges]
            edge_weights: Optional [num_edges, batch]
        """
        # Stack features in modality order
        node_features = torch.stack(
            [modality_features[mod] for mod in self.modalities],
            dim=0
        )  # [num_modalities, batch, feature_dim]

        # Get edge index
        edge_index = self.graph_builder.edge_index.to(node_features.device)

        # Compute adaptive edge weights if enabled
        edge_weights = None
        if self.use_adaptive_edges and self.edge_weighting is not None:
            # Compute edge weights for each sample in batch
            batch_size = node_features.size(1)
            edge_weights_list = []

            for i in range(batch_size):
                sample_features = node_features[:, i, :]  # [num_modalities, feature_dim]
                weights = self.edge_weighting(sample_features, edge_index)
                edge_weights_list.append(weights)

            # Stack weights: [num_edges, batch]
            edge_weights = torch.stack(edge_weights_list, dim=1)

        return node_features, edge_index, edge_weights

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        quality_scores: Optional[Dict[str, float]] = None,
        extract_features: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            modality_inputs: Either raw inputs or precomputed features
            quality_scores: Optional quality scores
            extract_features: Whether to extract features or use precomputed

        Returns:
            logits: Class logits [batch, num_classes]
            embeddings: Graph embeddings [batch, embedding_dim]
        """
        # Extract features if needed
        if extract_features:
            modality_features = self.extract_features(modality_inputs)
        else:
            modality_features = modality_inputs

        # Build graph
        node_features, edge_index, edge_weights = self.build_modality_graph(
            modality_features, quality_scores
        )

        # Prepare data for GNN
        # Flatten: [num_modalities * batch, feature_dim]
        num_modalities, batch_size, feature_dim = node_features.shape
        x = node_features.transpose(0, 1).reshape(
            num_modalities * batch_size, feature_dim
        )

        # Create batch assignment for proper pooling
        batch_indices = torch.arange(batch_size, device=x.device).repeat_interleave(num_modalities)

        # Expand edge index for batch
        edge_index_batch = []
        for i in range(batch_size):
            offset = i * num_modalities
            edge_index_batch.append(edge_index + offset)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)

        # Handle edge weights
        edge_weight = None
        if edge_weights is not None:
            # Flatten edge weights
            edge_weight = edge_weights.t().reshape(-1)

        # Apply GNN
        logits, embeddings = self.gnn(x, edge_index_batch, edge_weight, batch_indices)

        return logits, embeddings

    def predict(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        quality_scores: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Make predictions

        Args:
            modality_inputs: Raw modality inputs
            quality_scores: Optional quality scores

        Returns:
            predictions: Class predictions [batch]
        """
        with torch.no_grad():
            logits, _ = self.forward(modality_inputs, quality_scores)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def get_similarity_score(
        self,
        modality_inputs1: Dict[str, torch.Tensor],
        modality_inputs2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute similarity score between two samples

        Args:
            modality_inputs1: First sample
            modality_inputs2: Second sample

        Returns:
            similarity: Similarity score
        """
        with torch.no_grad():
            _, emb1 = self.forward(modality_inputs1)
            _, emb2 = self.forward(modality_inputs2)

            # Cosine similarity
            similarity = F.cosine_similarity(emb1, emb2)

        return similarity


class EnsembleMultimodalFusion(nn.Module):
    """
    Ensemble of multiple GNN models for robust fusion
    """

    def __init__(
        self,
        modalities: List[str],
        feature_dim: int = 512,
        gnn_types: List[str] = ['gcn', 'gat', 'graphsage'],
        ensemble_method: str = 'voting'
    ):
        """
        Args:
            modalities: List of modalities
            feature_dim: Feature dimension
            gnn_types: List of GNN types to ensemble
            ensemble_method: 'voting', 'averaging', 'stacking'
        """
        super().__init__()

        self.modalities = modalities
        self.gnn_types = gnn_types
        self.ensemble_method = ensemble_method

        # Create multiple models
        self.models = nn.ModuleList([
            MultimodalBiometricFusion(
                modalities=modalities,
                feature_dim=feature_dim,
                gnn_type=gnn_type
            )
            for gnn_type in gnn_types
        ])

        # Stacking meta-learner
        if ensemble_method == 'stacking':
            self.meta_learner = nn.Sequential(
                nn.Linear(len(gnn_types) * 2, 64),  # Each model outputs 2 logits
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        quality_scores: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Ensemble forward pass

        Args:
            modality_inputs: Raw modality inputs
            quality_scores: Optional quality scores

        Returns:
            logits: Ensembled logits
        """
        # Get predictions from all models
        all_logits = []
        for model in self.models:
            logits, _ = model(modality_inputs, quality_scores)
            all_logits.append(logits)

        # Ensemble
        if self.ensemble_method == 'voting':
            # Majority voting
            predictions = torch.stack([torch.argmax(l, dim=1) for l in all_logits], dim=0)
            final_pred = torch.mode(predictions, dim=0)[0]
            # Convert to one-hot logits
            logits = torch.zeros_like(all_logits[0])
            logits.scatter_(1, final_pred.unsqueeze(1), 1.0)

        elif self.ensemble_method == 'averaging':
            # Average logits
            logits = torch.stack(all_logits, dim=0).mean(dim=0)

        elif self.ensemble_method == 'stacking':
            # Stack and learn combination
            stacked = torch.cat(all_logits, dim=1)
            logits = self.meta_learner(stacked)

        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        return logits


class HybridFusion(nn.Module):
    """
    Hybrid fusion combining early, late, and GNN-based fusion
    """

    def __init__(
        self,
        modalities: List[str],
        feature_dim: int = 512,
        fusion_type: str = 'all'
    ):
        """
        Args:
            modalities: List of modalities
            feature_dim: Feature dimension
            fusion_type: 'early', 'late', 'gnn', 'all'
        """
        super().__init__()

        self.fusion_type = fusion_type

        # Feature extractors
        self.feature_extractors = nn.ModuleDict({
            mod: get_feature_extractor(mod, feature_dim)
            for mod in modalities
        })

        # Early fusion (concatenate features)
        if fusion_type in ['early', 'all']:
            self.early_fusion = nn.Sequential(
                nn.Linear(feature_dim * len(modalities), 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

        # Late fusion (separate classifiers)
        if fusion_type in ['late', 'all']:
            self.late_fusion = nn.ModuleDict({
                mod: nn.Sequential(
                    nn.Linear(feature_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2)
                )
                for mod in modalities
            })

        # GNN fusion
        if fusion_type in ['gnn', 'all']:
            self.gnn_fusion = MultimodalBiometricFusion(
                modalities=modalities,
                feature_dim=feature_dim,
                gnn_type='gat'
            )

        # Final combiner if using all
        if fusion_type == 'all':
            num_inputs = 2 + 2 * len(modalities) + 2  # early + late + gnn
            self.combiner = nn.Sequential(
                nn.Linear(num_inputs, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )

    def forward(self, modality_inputs: Dict[str, torch.Tensor]):
        outputs = []

        # Extract features
        features = {
            mod: self.feature_extractors[mod](inp)
            for mod, inp in modality_inputs.items()
        }

        # Early fusion
        if self.fusion_type in ['early', 'all']:
            concat_features = torch.cat(list(features.values()), dim=1)
            early_out = self.early_fusion(concat_features)
            if self.fusion_type == 'early':
                return early_out
            outputs.append(early_out)

        # Late fusion
        if self.fusion_type in ['late', 'all']:
            late_outs = []
            for mod, feat in features.items():
                late_outs.append(self.late_fusion[mod](feat))
            late_avg = torch.stack(late_outs, dim=0).mean(dim=0)
            if self.fusion_type == 'late':
                return late_avg
            outputs.append(late_avg)
            outputs.extend(late_outs)

        # GNN fusion
        if self.fusion_type in ['gnn', 'all']:
            gnn_out, _ = self.gnn_fusion(features, extract_features=False)
            if self.fusion_type == 'gnn':
                return gnn_out
            outputs.append(gnn_out)

        # Combine all
        if self.fusion_type == 'all':
            combined = torch.cat(outputs, dim=1)
            return self.combiner(combined)
