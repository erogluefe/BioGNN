"""
Dataset downloaders for multimodal biometric datasets.

This module provides automatic downloaders for various public biometric datasets
including face, fingerprint, iris, and voice datasets.
"""

from .base import BaseDatasetDownloader
from .face_datasets import LFWDownloader, CelebADownloader, CASIAWebFaceDownloader
from .fingerprint_datasets import SOCOFingDownloader, FVC2004Downloader
from .iris_datasets import UBIRISDownloader, CASIAIrisDownloader, MMUDownloader
from .voice_datasets import LibriSpeechDownloader, VoxCelebDownloader, CommonVoiceDownloader

__all__ = [
    'BaseDatasetDownloader',
    # Face datasets
    'LFWDownloader',
    'CelebADownloader',
    'CASIAWebFaceDownloader',
    # Fingerprint datasets
    'SOCOFingDownloader',
    'FVC2004Downloader',
    # Iris datasets
    'UBIRISDownloader',
    'CASIAIrisDownloader',
    'MMUDownloader',
    # Voice datasets
    'LibriSpeechDownloader',
    'VoxCelebDownloader',
    'CommonVoiceDownloader',
]

# Registry of available downloaders
DATASET_REGISTRY = {
    # Face datasets
    'lfw': LFWDownloader,
    'celeba': CelebADownloader,
    'casia-webface': CASIAWebFaceDownloader,
    # Fingerprint datasets
    'socofing': SOCOFingDownloader,
    'fvc2004': FVC2004Downloader,
    # Iris datasets
    'ubiris': UBIRISDownloader,
    'casia-iris': CASIAIrisDownloader,
    'mmu': MMUDownloader,
    # Voice datasets
    'librispeech': LibriSpeechDownloader,
    'voxceleb': VoxCelebDownloader,
    'commonvoice': CommonVoiceDownloader,
}


def get_downloader(dataset_name: str, root: str = './datasets'):
    """
    Get a downloader instance for the specified dataset.

    Args:
        dataset_name: Name of the dataset. Available datasets:
            Face: 'lfw', 'celeba', 'casia-webface'
            Fingerprint: 'socofing', 'fvc2004'
            Iris: 'ubiris', 'casia-iris', 'mmu'
            Voice: 'librispeech', 'voxceleb', 'commonvoice'
        root: Root directory to download the dataset to

    Returns:
        Downloader instance

    Example:
        >>> downloader = get_downloader('lfw', './datasets')
        >>> downloader.download()
    """
    if dataset_name.lower() not in DATASET_REGISTRY:
        available = ', '.join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    downloader_class = DATASET_REGISTRY[dataset_name.lower()]
    return downloader_class(root=root)
