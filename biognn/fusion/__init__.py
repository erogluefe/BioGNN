"""
Multimodal fusion modules
"""

from .graph_builder import (
    ModalityGraphBuilder,
    AdaptiveEdgeWeighting,
    ModalityAttention,
    QualityAwarePooling
)

__all__ = [
    'ModalityGraphBuilder',
    'AdaptiveEdgeWeighting',
    'ModalityAttention',
    'QualityAwarePooling',
]
from .multimodal_fusion import (
    MultimodalBiometricFusion,
    EnsembleMultimodalFusion,
    HybridFusion
)

__all__ = [
    'ModalityGraphBuilder',
    'AdaptiveEdgeWeighting',
    'ModalityAttention',
    'QualityAwarePooling',
    'MultimodalBiometricFusion',
    'EnsembleMultimodalFusion',
    'HybridFusion',
]
