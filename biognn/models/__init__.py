"""
GNN models for multimodal biometric fusion
"""

from .gcn import MultimodalGCN, DeepGCN
from .gat import MultimodalGAT, HierarchicalGAT
from .graphsage import (
    MultimodalGraphSAGE,
    AdaptiveGraphSAGE,
    MiniBatchGraphSAGE
)
from .multitask import (
    MultiTaskBiometricModel,
    AdaptiveTaskWeighting,
    GradientBalancing
)

__all__ = [
    # GCN models
    'MultimodalGCN',
    'DeepGCN',

    # GAT models
    'MultimodalGAT',
    'HierarchicalGAT',

    # GraphSAGE models
    'MultimodalGraphSAGE',
    'AdaptiveGraphSAGE',
    'MiniBatchGraphSAGE',

    # Multi-task models
    'MultiTaskBiometricModel',
    'AdaptiveTaskWeighting',
    'GradientBalancing',
]
