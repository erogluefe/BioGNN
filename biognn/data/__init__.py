"""
Data loading and preprocessing modules for multimodal biometric data
"""

from .base_dataset import (
    BiometricModality,
    BiometricSample,
    MultimodalBiometricDataset,
    VerificationPairDataset
)
from .transforms import (
    FaceTransform,
    FingerprintTransform,
    IrisTransform,
    VoiceTransform,
    get_default_transforms
)
from .feature_extractors import (
    FaceFeatureExtractor,
    FingerprintFeatureExtractor,
    IrisFeatureExtractor,
    VoiceFeatureExtractor,
    get_feature_extractor
)

__all__ = [
    # Base classes
    'BiometricModality',
    'BiometricSample',
    'MultimodalBiometricDataset',
    'VerificationPairDataset',

    # Transforms
    'FaceTransform',
    'FingerprintTransform',
    'IrisTransform',
    'VoiceTransform',
    'get_default_transforms',

    # Feature extractors
    'FaceFeatureExtractor',
    'FingerprintFeatureExtractor',
    'IrisFeatureExtractor',
    'VoiceFeatureExtractor',
    'get_feature_extractor',
]
