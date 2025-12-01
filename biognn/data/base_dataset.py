"""
Base dataset classes for multimodal biometric data
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


class BiometricModality:
    """Enum-like class for biometric modalities"""
    FACE = "face"
    FINGERPRINT = "fingerprint"
    FINGER = "finger"  # Alias for fingerprint (used in some datasets)
    VOICE = "voice"
    IRIS = "iris"

    @classmethod
    def all_modalities(cls) -> List[str]:
        return [cls.FACE, cls.FINGERPRINT, cls.FINGER, cls.VOICE, cls.IRIS]


class BiometricSample:
    """Container for a single biometric sample across multiple modalities"""

    def __init__(
        self,
        subject_id: int,
        modalities: Dict[str, torch.Tensor],
        labels: Optional[Dict[str, Any]] = None,
        is_genuine: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            subject_id: Unique identifier for the subject
            modalities: Dictionary mapping modality name to feature tensor
            labels: Optional labels for classification/verification
            is_genuine: Whether this is a genuine or spoofed sample
            metadata: Additional metadata (quality scores, timestamps, etc.)
        """
        self.subject_id = subject_id
        self.modalities = modalities
        self.labels = labels or {}
        self.is_genuine = is_genuine
        self.metadata = metadata or {}

    def get_available_modalities(self) -> List[str]:
        """Get list of available modalities in this sample"""
        return list(self.modalities.keys())

    def has_modality(self, modality: str) -> bool:
        """Check if a modality is available"""
        return modality in self.modalities

    def to(self, device: torch.device) -> 'BiometricSample':
        """Move all tensors to specified device"""
        modalities = {k: v.to(device) for k, v in self.modalities.items()}
        return BiometricSample(
            self.subject_id,
            modalities,
            self.labels,
            self.is_genuine,
            self.metadata
        )


class MultimodalBiometricDataset(Dataset, ABC):
    """Base class for multimodal biometric datasets"""

    def __init__(
        self,
        root: str,
        modalities: List[str],
        split: str = "train",
        transform: Optional[Dict[str, Any]] = None,
        download: bool = False
    ):
        """
        Args:
            root: Root directory of dataset
            modalities: List of modalities to load
            split: Dataset split ('train', 'val', 'test')
            transform: Dictionary of transforms for each modality
            download: Whether to download the dataset if not found
        """
        self.root = Path(root)
        self.modalities = modalities
        self.split = split
        self.transform = transform or {}

        # Validate modalities
        for mod in modalities:
            if mod not in BiometricModality.all_modalities():
                raise ValueError(f"Unknown modality: {mod}")

        if download:
            self.download()

        self._load_data()

    @abstractmethod
    def _load_data(self):
        """Load dataset from disk - to be implemented by subclasses"""
        pass

    @abstractmethod
    def download(self):
        """Download dataset - to be implemented by subclasses"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> BiometricSample:
        """Get a single sample"""
        pass

    def get_num_subjects(self) -> int:
        """Get number of unique subjects in dataset"""
        return len(set([self[i].subject_id for i in range(len(self))]))

    def get_subject_samples(self, subject_id: int) -> List[BiometricSample]:
        """Get all samples for a specific subject"""
        return [self[i] for i in range(len(self)) if self[i].subject_id == subject_id]


class VerificationPairDataset(Dataset):
    """Dataset for verification tasks (genuine vs impostor pairs)"""

    def __init__(
        self,
        base_dataset: MultimodalBiometricDataset,
        num_pairs: int = 10000,
        genuine_ratio: float = 0.5,
        seed: int = 42
    ):
        """
        Args:
            base_dataset: Underlying multimodal dataset
            num_pairs: Number of pairs to generate
            genuine_ratio: Ratio of genuine pairs
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.num_pairs = num_pairs
        self.genuine_ratio = genuine_ratio

        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate pairs
        self.pairs = self._generate_pairs()

    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Generate pairs of samples
        Returns:
            List of (idx1, idx2, label) where label=1 for genuine, 0 for impostor
        """
        pairs = []
        num_genuine = int(self.num_pairs * self.genuine_ratio)
        num_impostor = self.num_pairs - num_genuine

        # Get subject IDs
        subject_ids = {}
        for idx in range(len(self.base_dataset)):
            sid = self.base_dataset[idx].subject_id
            if sid not in subject_ids:
                subject_ids[sid] = []
            subject_ids[sid].append(idx)

        subjects = list(subject_ids.keys())

        # Generate genuine pairs
        for _ in range(num_genuine):
            # Select a random subject with at least 2 samples
            valid_subjects = [s for s in subjects if len(subject_ids[s]) >= 2]
            if not valid_subjects:
                break
            subject = np.random.choice(valid_subjects)
            idx1, idx2 = np.random.choice(subject_ids[subject], 2, replace=False)
            pairs.append((idx1, idx2, 1))

        # Generate impostor pairs
        for _ in range(num_impostor):
            if len(subjects) < 2:
                break
            subj1, subj2 = np.random.choice(subjects, 2, replace=False)
            idx1 = np.random.choice(subject_ids[subj1])
            idx2 = np.random.choice(subject_ids[subj2])
            pairs.append((idx1, idx2, 0))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[BiometricSample, BiometricSample, int]:
        """
        Returns:
            (sample1, sample2, label) where label=1 for genuine, 0 for impostor
        """
        idx1, idx2, label = self.pairs[idx]
        sample1 = self.base_dataset[idx1]
        sample2 = self.base_dataset[idx2]
        return sample1, sample2, label


def biometric_sample_collate_fn(batch: List[BiometricSample]) -> BiometricSample:
    """
    Custom collate function for batching BiometricSample objects.

    Args:
        batch: List of BiometricSample objects

    Returns:
        A single BiometricSample with batched tensors
    """
    # Get all modalities present in the batch
    all_modalities = set()
    for sample in batch:
        all_modalities.update(sample.modalities.keys())

    # Stack tensors for each modality
    batched_modalities = {}
    for modality in all_modalities:
        # Get tensors for this modality from all samples
        tensors = [s.modalities[modality] for s in batch if modality in s.modalities]
        if tensors:
            batched_modalities[modality] = torch.stack(tensors, dim=0)

    # Collect subject IDs
    subject_ids = [s.subject_id for s in batch]

    # Create batched sample
    return BiometricSample(
        subject_id=subject_ids,  # Store as list for batch
        modalities=batched_modalities,
        labels={},
        is_genuine=batch[0].is_genuine,
        metadata={'batch_size': len(batch)}
    )


def biometric_collate_fn(batch: List[Tuple[BiometricSample, BiometricSample, int]]) -> Tuple[BiometricSample, BiometricSample, torch.Tensor]:
    """
    Custom collate function for batching BiometricSample pairs (for verification tasks).

    Args:
        batch: List of (sample1, sample2, label) tuples

    Returns:
        Tuple of (batched_sample1, batched_sample2, labels_tensor)
    """
    samples1, samples2, labels = zip(*batch)

    # Collate labels
    labels = torch.tensor(labels, dtype=torch.long)

    # Collate BiometricSamples
    batched_sample1 = biometric_sample_collate_fn(samples1)
    batched_sample2 = biometric_sample_collate_fn(samples2)

    return batched_sample1, batched_sample2, labels
