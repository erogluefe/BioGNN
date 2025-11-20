"""
Dataset downloaders for multimodal biometric datasets.

This module provides automatic downloaders for various public biometric datasets
including face, fingerprint, iris, and voice datasets.
"""

from .base import BaseDatasetDownloader
from .face_datasets import LFWDownloader, CelebADownloader
from .fingerprint_datasets import SOCOFingDownloader
from .iris_datasets import UBIRISDownloader
from .voice_datasets import LibriSpeechDownloader

__all__ = [
    'BaseDatasetDownloader',
    'LFWDownloader',
    'CelebADownloader',
    'SOCOFingDownloader',
    'UBIRISDownloader',
    'LibriSpeechDownloader',
]

# Registry of available downloaders
DATASET_REGISTRY = {
    'lfw': LFWDownloader,
    'celeba': CelebADownloader,
    'socofing': SOCOFingDownloader,
    'ubiris': UBIRISDownloader,
    'librispeech': LibriSpeechDownloader,
}


def get_downloader(dataset_name: str, root: str = './datasets'):
    """
    Get a downloader instance for the specified dataset.

    Args:
        dataset_name: Name of the dataset ('lfw', 'celeba', 'socofing', 'ubiris', 'librispeech')
        root: Root directory to download the dataset to

    Returns:
        Downloader instance

    Example:
        >>> downloader = get_downloader('lfw', './datasets')
        >>> downloader.download()
    """
    if dataset_name.lower() not in DATASET_REGISTRY:
        available = ', '.join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    downloader_class = DATASET_REGISTRY[dataset_name.lower()]
    return downloader_class(root=root)
