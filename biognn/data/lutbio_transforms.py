"""
Data transforms for LUTBio dataset

Provides preprocessing and augmentation for:
- Face images
- Fingerprint images
- Voice audio
"""

import torch
import torchvision.transforms as T
import torchaudio.transforms as AT
from typing import Dict, Optional


def get_lutbio_transforms(
    split: str = 'train',
    image_size: int = 112,
    fingerprint_size: int = 96,
    spectrogram_size: tuple = (40, 100),
    augmentation: bool = True
) -> Dict:
    """
    Get transforms for LUTBio dataset

    Args:
        split: 'train', 'val', or 'test'
        image_size: Face image size (square)
        fingerprint_size: Fingerprint image size (square)
        spectrogram_size: Spectrogram size (n_mels, time_frames)
        augmentation: Whether to apply augmentation (only for training)

    Returns:
        Dictionary mapping modality to transform
    """
    is_train = (split == 'train') and augmentation

    transforms = {
        'face': get_face_transform(image_size, is_train),
        'finger': get_fingerprint_transform(fingerprint_size, is_train),
        'voice': get_voice_transform(spectrogram_size, is_train)
    }

    return transforms


def get_face_transform(image_size: int = 112, is_train: bool = True):
    """
    Transform for face images

    Args:
        image_size: Target image size (square)
        is_train: Whether to apply training augmentations

    Returns:
        Composed transform
    """
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            T.RandomRotation(degrees=5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_fingerprint_transform(image_size: int = 96, is_train: bool = True):
    """
    Transform for fingerprint images

    Args:
        image_size: Target image size (square)
        is_train: Whether to apply training augmentations

    Returns:
        Composed transform
    """
    if is_train:
        return T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
            # Add random noise
            AddGaussianNoise(mean=0, std=0.02)
        ])
    else:
        return T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])


def get_voice_transform(
    spectrogram_size: tuple = (40, 100),
    is_train: bool = True,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160
):
    """
    Transform for voice audio

    Converts waveform to mel-spectrogram

    Args:
        spectrogram_size: (n_mels, time_frames)
        is_train: Whether to apply training augmentations
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT

    Returns:
        Transform function
    """
    n_mels, target_length = spectrogram_size

    def transform(waveform):
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed (assuming input is 16kHz or similar)
        # Note: LUTBio voice files are lossless WAV, usually 16kHz or higher

        # Create mel spectrogram
        mel_spec = AT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )(waveform)

        # Convert to log scale
        mel_spec = AT.AmplitudeToDB()(mel_spec)

        # Resize to fixed length
        mel_spec = torch.nn.functional.interpolate(
            mel_spec.unsqueeze(0),
            size=(n_mels, target_length),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        # Training augmentation
        if is_train:
            # Time masking
            if torch.rand(1) < 0.3:
                mel_spec = time_mask(mel_spec, max_mask_size=10)

            # Frequency masking
            if torch.rand(1) < 0.3:
                mel_spec = freq_mask(mel_spec, max_mask_size=5)

        return mel_spec

    return transform


class AddGaussianNoise:
    """Add Gaussian noise to tensor"""

    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if self.std > 0:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def time_mask(spectrogram: torch.Tensor, max_mask_size: int = 10) -> torch.Tensor:
    """
    Apply time masking to spectrogram (SpecAugment)

    Args:
        spectrogram: Input spectrogram [C, freq, time]
        max_mask_size: Maximum mask size in time dimension

    Returns:
        Masked spectrogram
    """
    time_size = spectrogram.shape[-1]
    mask_size = torch.randint(1, max_mask_size + 1, (1,)).item()
    mask_start = torch.randint(0, max(1, time_size - mask_size), (1,)).item()

    spectrogram_masked = spectrogram.clone()
    spectrogram_masked[..., mask_start:mask_start + mask_size] = 0

    return spectrogram_masked


def freq_mask(spectrogram: torch.Tensor, max_mask_size: int = 5) -> torch.Tensor:
    """
    Apply frequency masking to spectrogram (SpecAugment)

    Args:
        spectrogram: Input spectrogram [C, freq, time]
        max_mask_size: Maximum mask size in frequency dimension

    Returns:
        Masked spectrogram
    """
    freq_size = spectrogram.shape[-2]
    mask_size = torch.randint(1, max_mask_size + 1, (1,)).item()
    mask_start = torch.randint(0, max(1, freq_size - mask_size), (1,)).item()

    spectrogram_masked = spectrogram.clone()
    spectrogram_masked[..., mask_start:mask_start + mask_size, :] = 0

    return spectrogram_masked


# Convenience functions for specific modalities
def face_train_transform(image_size: int = 112):
    """Get training transform for face"""
    return get_face_transform(image_size, is_train=True)


def face_val_transform(image_size: int = 112):
    """Get validation transform for face"""
    return get_face_transform(image_size, is_train=False)


def fingerprint_train_transform(image_size: int = 96):
    """Get training transform for fingerprint"""
    return get_fingerprint_transform(image_size, is_train=True)


def fingerprint_val_transform(image_size: int = 96):
    """Get validation transform for fingerprint"""
    return get_fingerprint_transform(image_size, is_train=False)


def voice_train_transform(spectrogram_size: tuple = (40, 100)):
    """Get training transform for voice"""
    return get_voice_transform(spectrogram_size, is_train=True)


def voice_val_transform(spectrogram_size: tuple = (40, 100)):
    """Get validation transform for voice"""
    return get_voice_transform(spectrogram_size, is_train=False)
