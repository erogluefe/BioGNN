"""
Data loading and preprocessing modules for multimodal biometric data
"""

from .base_dataset import (
    BiometricModality,
    BiometricSample,
    MultimodalBiometricDataset,
    VerificationPairDataset,
    biometric_sample_collate_fn,
    biometric_collate_fn
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
from .example_dataset import (
    ExampleMultimodalDataset,
    SyntheticMultimodalDataset,
    create_example_dataset
)

# Dataset downloaders (lazy import to avoid dependencies)
def get_downloader(dataset_name, root='./datasets'):
    """Get a dataset downloader instance."""
    from .downloaders import get_downloader as _get_downloader
    return _get_downloader(dataset_name, root)

__all__ = [
    # Base classes
    'BiometricModality',
    'BiometricSample',
    'MultimodalBiometricDataset',
    'VerificationPairDataset',
    'biometric_sample_collate_fn',
    'biometric_collate_fn',

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

    # Example datasets
    'ExampleMultimodalDataset',
    'SyntheticMultimodalDataset',
    'create_example_dataset',

    # Downloaders
    'get_downloader',
]
