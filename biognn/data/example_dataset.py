"""
Example dataset implementation for BioGNN

This is a template/example showing how to create your own multimodal dataset.
Adapt this to your specific data format and structure.
"""

import os
import numpy as np
from PIL import Image
import soundfile as sf
import torch
from pathlib import Path
from typing import Dict, List, Optional

from biognn.data import MultimodalBiometricDataset, BiometricSample
from biognn.data import get_default_transforms


class ExampleMultimodalDataset(MultimodalBiometricDataset):
    """
    Example implementation of multimodal biometric dataset

    Expected directory structure:

    root/
    ├── train/
    │   ├── subject_001/
    │   │   ├── face_001.jpg
    │   │   ├── face_002.jpg
    │   │   ├── fingerprint_001.png
    │   │   ├── iris_001.png
    │   │   └── voice_001.wav
    │   ├── subject_002/
    │   └── ...
    ├── val/
    └── test/
    """

    def __init__(
        self,
        root: str,
        modalities: List[str] = ['face', 'fingerprint', 'iris', 'voice'],
        split: str = 'train',
        transform: Optional[Dict] = None,
        download: bool = False
    ):
        """
        Args:
            root: Root directory of dataset
            modalities: List of modalities to load
            split: 'train', 'val', or 'test'
            transform: Dictionary of transforms per modality
            download: Whether to download (not implemented in example)
        """
        super().__init__(root, modalities, split, transform, download)

    def _load_data(self):
        """
        Load dataset from disk

        This is where you implement your data loading logic
        """
        self.data_dir = Path(self.root) / self.split

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please organize your data or see documentation for dataset setup."
            )

        # Find all subject directories
        subject_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])

        self.samples = []
        self.subject_ids = []

        for subject_dir in subject_dirs:
            subject_id = int(subject_dir.name.split('_')[-1])

            # Find all samples for this subject
            # Group files by sample index
            sample_files = self._group_files_by_sample(subject_dir)

            for sample_idx, files in sample_files.items():
                sample_info = {
                    'subject_id': subject_id,
                    'files': files,
                    'is_genuine': True  # Default, adjust based on your needs
                }
                self.samples.append(sample_info)
                self.subject_ids.append(subject_id)

        print(f"Loaded {len(self.samples)} samples from {len(set(self.subject_ids))} subjects")

    def _group_files_by_sample(self, subject_dir: Path) -> Dict[int, Dict[str, Path]]:
        """
        Group files by sample index

        Example: face_001.jpg, fingerprint_001.png -> sample 001
        """
        samples = {}

        for modality in self.modalities:
            # Find files for this modality
            pattern = f"{modality}_*.{'jpg' if modality == 'face' else 'png' if modality in ['fingerprint', 'iris'] else 'wav'}"
            files = list(subject_dir.glob(pattern))

            for file in files:
                # Extract sample index from filename
                # e.g., face_001.jpg -> 001
                sample_idx = int(file.stem.split('_')[-1])

                if sample_idx not in samples:
                    samples[sample_idx] = {}

                samples[sample_idx][modality] = file

        return samples

    def download(self):
        """
        Download dataset (implement if needed)
        """
        raise NotImplementedError(
            "Dataset download not implemented. Please manually download and organize your data.\n"
            "See README.md for supported datasets and organization structure."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BiometricSample:
        """
        Get a single sample

        Returns:
            BiometricSample with all available modalities
        """
        sample_info = self.samples[idx]
        subject_id = sample_info['subject_id']
        files = sample_info['files']

        # Load each modality
        modalities = {}

        for modality_name in self.modalities:
            if modality_name not in files:
                continue  # Skip if this modality not available

            file_path = files[modality_name]

            # Load based on modality type
            if modality_name == 'face':
                data = self._load_image(file_path)
            elif modality_name == 'fingerprint':
                data = self._load_image(file_path, grayscale=True)
            elif modality_name == 'iris':
                data = self._load_image(file_path, grayscale=True)
            elif modality_name == 'voice':
                data = self._load_audio(file_path)
            else:
                raise ValueError(f"Unknown modality: {modality_name}")

            # Apply transform if available
            if self.transform and modality_name in self.transform:
                data = self.transform[modality_name](data)

            modalities[modality_name] = data

        # Create BiometricSample
        sample = BiometricSample(
            subject_id=subject_id,
            modalities=modalities,
            is_genuine=sample_info['is_genuine']
        )

        return sample

    def _load_image(self, path: Path, grayscale: bool = False) -> Image.Image:
        """Load image file"""
        img = Image.open(path)
        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        return img

    def _load_audio(self, path: Path, target_sr: int = 16000) -> torch.Tensor:
        """Load audio file"""
        # Load audio
        waveform, sr = sf.read(path)

        # Convert to torch tensor
        waveform = torch.FloatTensor(waveform)

        # Ensure correct shape [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample if needed (simplified, use torchaudio.transforms.Resample in production)
        # For now, just assume correct sample rate

        return waveform


def create_example_dataset(
    root: str = './datasets',
    split: str = 'train',
    modalities: List[str] = None,
    augment: bool = True
) -> ExampleMultimodalDataset:
    """
    Create dataset with default transforms

    Args:
        root: Dataset root directory
        split: 'train', 'val', or 'test'
        modalities: List of modalities (None for all)
        augment: Whether to apply data augmentation

    Returns:
        Dataset instance
    """
    if modalities is None:
        modalities = ['face', 'fingerprint', 'iris', 'voice']

    # Get default transforms
    transforms = {}
    for modality in modalities:
        transforms[modality] = get_default_transforms(modality, augment=augment)

    dataset = ExampleMultimodalDataset(
        root=root,
        modalities=modalities,
        split=split,
        transform=transforms
    )

    return dataset


# ===== ALTERNATIVE: Simple synthetic dataset for testing =====

class SyntheticMultimodalDataset(MultimodalBiometricDataset):
    """
    Synthetic dataset for testing/demo purposes

    Generates random data - useful for testing pipeline without real data
    """

    def __init__(
        self,
        num_subjects: int = 100,
        samples_per_subject: int = 5,
        modalities: List[str] = ['face', 'fingerprint', 'iris', 'voice'],
        split: str = 'train',
        seed: int = 42
    ):
        self.num_subjects = num_subjects
        self.samples_per_subject = samples_per_subject
        self.seed = seed

        # Don't call super().__init__() as we don't need file loading
        self.modalities = modalities
        self.split = split
        self.transform = None

        self._load_data()

    def _load_data(self):
        """Generate synthetic data"""
        np.random.seed(self.seed)

        self.samples = []
        for subject_id in range(self.num_subjects):
            for sample_idx in range(self.samples_per_subject):
                self.samples.append({
                    'subject_id': subject_id,
                    'sample_idx': sample_idx
                })

    def download(self):
        """Not needed for synthetic data"""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> BiometricSample:
        sample_info = self.samples[idx]
        subject_id = sample_info['subject_id']

        # Generate synthetic data for each modality
        modalities = {}

        np.random.seed(self.seed + idx)  # Deterministic but different per sample

        if 'face' in self.modalities:
            # Synthetic face image [3, 112, 112]
            face = torch.randn(3, 112, 112) * 0.5 + 0.5
            modalities['face'] = torch.clamp(face, 0, 1)

        if 'fingerprint' in self.modalities:
            # Synthetic fingerprint [1, 96, 96]
            fingerprint = torch.randn(1, 96, 96) * 0.3 + 0.5
            modalities['fingerprint'] = torch.clamp(fingerprint, 0, 1)

        if 'iris' in self.modalities:
            # Synthetic iris [1, 64, 256]
            iris = torch.randn(1, 64, 256) * 0.3 + 0.5
            modalities['iris'] = torch.clamp(iris, 0, 1)

        if 'voice' in self.modalities:
            # Synthetic MFCC features [40, 100]
            voice = torch.randn(40, 100) * 2.0
            modalities['voice'] = voice

        return BiometricSample(
            subject_id=subject_id,
            modalities=modalities,
            is_genuine=True
        )
