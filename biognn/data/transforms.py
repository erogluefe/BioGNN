"""
Transform functions for different biometric modalities
"""

import torch
import torchvision.transforms as T
import torchaudio.transforms as AT
import numpy as np
from typing import Optional, Tuple, Callable


class FaceTransform:
    """Transforms for face images"""

    def __init__(
        self,
        img_size: Tuple[int, int] = (112, 112),
        augment: bool = True,
        normalize: bool = True
    ):
        transforms = []

        # Resize
        transforms.append(T.Resize(img_size))

        # Data augmentation for training
        if augment:
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])

        # Convert to tensor
        transforms.append(T.ToTensor())

        # Normalize
        if normalize:
            transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.transform = T.Compose(transforms)

    def __call__(self, img):
        return self.transform(img)


class FingerprintTransform:
    """Transforms for fingerprint images"""

    def __init__(
        self,
        img_size: Tuple[int, int] = (96, 96),
        augment: bool = True,
        normalize: bool = True
    ):
        transforms = []

        # Resize
        transforms.append(T.Resize(img_size))

        # Data augmentation
        if augment:
            transforms.extend([
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
            ])

        # Convert to tensor (fingerprints are typically grayscale)
        transforms.append(T.ToTensor())

        # Normalize
        if normalize:
            transforms.append(T.Normalize(mean=[0.5], std=[0.5]))

        self.transform = T.Compose(transforms)

    def __call__(self, img):
        return self.transform(img)


class IrisTransform:
    """Transforms for iris images"""

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 256),  # Typical iris unwrapped size
        augment: bool = True,
        normalize: bool = True
    ):
        transforms = []

        # Resize
        transforms.append(T.Resize(img_size))

        # Light augmentation (iris images are sensitive)
        if augment:
            transforms.extend([
                T.RandomHorizontalFlip(p=0.3),
                T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            ])

        # Convert to tensor
        transforms.append(T.ToTensor())

        # Normalize
        if normalize:
            transforms.append(T.Normalize(mean=[0.5], std=[0.5]))

        self.transform = T.Compose(transforms)

    def __call__(self, img):
        return self.transform(img)


class VoiceTransform:
    """Transforms for voice/audio data"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 512,
        hop_length: int = 160,
        augment: bool = True,
        max_length: Optional[int] = None
    ):
        """
        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT size
            hop_length: Hop length for STFT
            augment: Whether to apply data augmentation
            max_length: Maximum length of audio in samples
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment
        self.max_length = max_length

        # MFCC transform
        self.mfcc_transform = AT.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': 64,
            }
        )

        # Augmentation transforms
        if augment:
            self.time_stretch = AT.TimeStretch(n_freq=n_fft // 2 + 1)
            self.freq_mask = AT.FrequencyMasking(freq_mask_param=15)
            self.time_mask = AT.TimeMasking(time_mask_param=35)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Audio waveform tensor [channels, samples]

        Returns:
            MFCC features [n_mfcc, time_frames]
        """
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Trim or pad to max_length
        if self.max_length is not None:
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            elif waveform.shape[1] < self.max_length:
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Extract MFCC features
        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0)  # Remove channel dimension

        # Data augmentation
        if self.augment:
            # Add noise
            if np.random.random() < 0.5:
                noise = torch.randn_like(mfcc) * 0.01
                mfcc = mfcc + noise

            # Frequency masking
            if np.random.random() < 0.3:
                mfcc = self.freq_mask(mfcc)

            # Time masking
            if np.random.random() < 0.3:
                mfcc = self.time_mask(mfcc)

        return mfcc


def get_default_transforms(modality: str, augment: bool = True) -> Callable:
    """
    Get default transform for a modality

    Args:
        modality: Modality name ('face', 'fingerprint', 'iris', 'voice')
        augment: Whether to apply data augmentation

    Returns:
        Transform function
    """
    transforms_map = {
        'face': FaceTransform(augment=augment),
        'fingerprint': FingerprintTransform(augment=augment),
        'iris': IrisTransform(augment=augment),
        'voice': VoiceTransform(augment=augment),
    }

    if modality not in transforms_map:
        raise ValueError(f"Unknown modality: {modality}")

    return transforms_map[modality]
