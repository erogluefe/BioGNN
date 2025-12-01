"""
LUTBio Multimodal Biometric Database Loader

Dataset: https://data.mendeley.com/datasets/jszw485f8j/6
Paper: LUTBio - A Multimodal Biometric Database

File structure:
    LUTBIO sample data/
    ├── 001/
    │   ├── face/      (6 JPG images)
    │   ├── finger/    (10 BMP images)
    │   └── voice/     (3 WAV files)
    ├── 063/
    ...

File naming: {subject_id}_{gender}_{age}_{modality}_{sample}.{ext}
Example: 001_male_56_face_01.jpg
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import torch
import numpy as np
from PIL import Image
import torchaudio

from .base_dataset import MultimodalBiometricDataset, BiometricSample


class LUTBioDataset(MultimodalBiometricDataset):
    """
    LUTBio Multimodal Biometric Database

    Supports 3 modalities: face, fingerprint (finger), voice

    Args:
        root: Root directory containing subject folders
        modalities: List of modalities to load ['face', 'finger', 'voice']
        split: 'train', 'val', or 'test'
        mode: 'verification' or 'identification'
        transform: Transform to apply to samples
        train_subjects: List of subject IDs for training (default: first 4)
        val_subjects: List of subject IDs for validation (default: 5th subject)
        test_subjects: List of subject IDs for testing (default: 6th subject)
        pairs_per_subject: Number of pairs per subject for verification mode
    """

    # Modality folder names in dataset
    MODALITY_FOLDERS = {
        'face': 'face',
        'finger': 'finger',  # fingerprint
        'voice': 'voice'
    }

    # File extensions
    MODALITY_EXTENSIONS = {
        'face': '.jpg',
        'finger': '.bmp',
        'voice': '.wav'
    }

    def __init__(
        self,
        root: str,
        modalities: List[str] = ['face', 'finger', 'voice'],
        split: str = 'train',
        mode: str = 'verification',
        transform: Optional[Dict] = None,
        train_subjects: Optional[List[str]] = None,
        val_subjects: Optional[List[str]] = None,
        test_subjects: Optional[List[str]] = None,
        pairs_per_subject: int = 10,
        seed: int = 42
    ):
        self.root = Path(root)
        self.modalities = modalities
        self.split = split
        self.mode = mode
        self.pairs_per_subject = pairs_per_subject
        self.seed = seed

        # Validate modalities
        for mod in modalities:
            if mod not in self.MODALITY_FOLDERS:
                raise ValueError(f"Unsupported modality: {mod}. Choose from {list(self.MODALITY_FOLDERS.keys())}")

        # Scan dataset
        self.subjects_data = self._scan_dataset()

        if len(self.subjects_data) == 0:
            raise ValueError(f"No subjects found in {root}")

        # Split subjects
        all_subjects = sorted(self.subjects_data.keys())

        # Create subject ID mapping (string -> int)
        self.subject_id_map = {subj_str: idx for idx, subj_str in enumerate(all_subjects)}
        self.reverse_subject_id_map = {idx: subj_str for subj_str, idx in self.subject_id_map.items()}

        if train_subjects is None or val_subjects is None or test_subjects is None:
            # Default split: 4/1/1
            if len(all_subjects) < 3:
                raise ValueError(f"Need at least 3 subjects, found {len(all_subjects)}")

            train_subjects = all_subjects[:4] if len(all_subjects) >= 4 else all_subjects[:1]
            val_subjects = all_subjects[4:5] if len(all_subjects) >= 5 else all_subjects[:1]
            test_subjects = all_subjects[5:6] if len(all_subjects) >= 6 else all_subjects[:1]

        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.test_subjects = test_subjects

        # Select subjects for current split
        if split == 'train':
            self.active_subjects = self.train_subjects
        elif split == 'val':
            self.active_subjects = self.val_subjects
        elif split == 'test':
            self.active_subjects = self.test_subjects
        else:
            raise ValueError(f"Invalid split: {split}")

        print(f"\n{'='*60}")
        print(f"LUTBio Dataset - {split.upper()} split")
        print(f"{'='*60}")
        print(f"Total subjects: {len(all_subjects)}")
        print(f"Train subjects: {len(self.train_subjects)} - {self.train_subjects}")
        print(f"Val subjects: {len(self.val_subjects)} - {self.val_subjects}")
        print(f"Test subjects: {len(self.test_subjects)} - {self.test_subjects}")
        print(f"Active subjects ({split}): {len(self.active_subjects)} - {self.active_subjects}")
        print(f"Modalities: {modalities}")
        print(f"Mode: {mode}")
        print(f"{'='*60}\n")

        # Initialize parent class
        super().__init__(
            root=str(root),
            modalities=modalities,
            transform=transform
        )

        # Generate pairs/samples based on mode
        if mode == 'verification':
            self.pairs = self._generate_verification_pairs()
        elif mode == 'identification':
            self.gallery, self.probes = self._generate_identification_sets()
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _scan_dataset(self) -> Dict:
        """
        Scan dataset directory and collect all subject data

        Returns:
            Dictionary: {subject_id: {'metadata': {...}, 'files': {...}}}
        """
        subjects_data = {}

        # Iterate through all directories in root
        for subject_dir in sorted(self.root.iterdir()):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            # Parse first file to get metadata
            metadata = None
            files = {}

            for modality in self.modalities:
                folder_name = self.MODALITY_FOLDERS[modality]
                modality_dir = subject_dir / folder_name

                if not modality_dir.exists():
                    warnings.warn(f"Modality {modality} not found for subject {subject_id}")
                    continue

                # Get all files for this modality
                ext = self.MODALITY_EXTENSIONS[modality]
                modality_files = sorted(list(modality_dir.glob(f'*{ext}')))

                if len(modality_files) == 0:
                    warnings.warn(f"No {modality} files found for subject {subject_id}")
                    continue

                files[modality] = modality_files

                # Parse metadata from first filename
                if metadata is None:
                    metadata = self._parse_filename(modality_files[0].name)

            if metadata and len(files) > 0:
                subjects_data[subject_id] = {
                    'metadata': metadata,
                    'files': files
                }

        return subjects_data

    def _parse_filename(self, filename: str) -> Dict:
        """
        Parse metadata from filename

        Format: {subject_id}_{gender}_{age}_{modality}_{sample}.{ext}
        Example: 001_male_56_face_01.jpg

        Returns:
            Dictionary with metadata
        """
        # Remove extension
        name = os.path.splitext(filename)[0]

        # Try to parse
        # Pattern: subject_gender_age_modality_sample
        parts = name.split('_')

        if len(parts) >= 3:
            return {
                'subject_id': parts[0],
                'gender': parts[1] if len(parts) > 1 else 'unknown',
                'age': int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
            }

        return {'subject_id': parts[0], 'gender': 'unknown', 'age': -1}

    def _generate_verification_pairs(self) -> List[Dict]:
        """
        Generate genuine and imposter pairs for verification

        Returns:
            List of pair dictionaries
        """
        pairs = []
        np.random.seed(self.seed)

        for subject_id in self.active_subjects:
            subject_data = self.subjects_data[subject_id]

            # Generate genuine pairs (same subject, different samples)
            for _ in range(self.pairs_per_subject // 2):
                pair = {
                    'subject_id': subject_id,
                    'is_genuine': True,
                    'modality_samples_1': {},
                    'modality_samples_2': {}
                }

                for modality in self.modalities:
                    if modality not in subject_data['files']:
                        continue

                    files = subject_data['files'][modality]
                    if len(files) >= 2:
                        # Pick two different samples
                        idx1, idx2 = np.random.choice(len(files), 2, replace=False)
                        pair['modality_samples_1'][modality] = files[idx1]
                        pair['modality_samples_2'][modality] = files[idx2]

                if len(pair['modality_samples_1']) > 0:
                    pairs.append(pair)

            # Generate imposter pairs (different subjects)
            for _ in range(self.pairs_per_subject // 2):
                # Pick random different subject
                other_subjects = [s for s in self.active_subjects if s != subject_id]
                if len(other_subjects) == 0:
                    continue

                other_subject = np.random.choice(other_subjects)
                other_data = self.subjects_data[other_subject]

                pair = {
                    'subject_id': subject_id,
                    'other_subject_id': other_subject,
                    'is_genuine': False,
                    'modality_samples_1': {},
                    'modality_samples_2': {}
                }

                for modality in self.modalities:
                    if modality in subject_data['files'] and modality in other_data['files']:
                        files1 = subject_data['files'][modality]
                        files2 = other_data['files'][modality]

                        idx1 = np.random.choice(len(files1))
                        idx2 = np.random.choice(len(files2))

                        pair['modality_samples_1'][modality] = files1[idx1]
                        pair['modality_samples_2'][modality] = files2[idx2]

                if len(pair['modality_samples_1']) > 0:
                    pairs.append(pair)

        print(f"Generated {len(pairs)} verification pairs ({sum(1 for p in pairs if p['is_genuine'])} genuine, {sum(1 for p in pairs if not p['is_genuine'])} imposter)")

        return pairs

    def _generate_identification_sets(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate gallery and probe sets for identification

        Returns:
            (gallery_samples, probe_samples)
        """
        gallery = []
        probes = []

        for subject_id in self.active_subjects:
            subject_data = self.subjects_data[subject_id]

            # Use first samples for gallery
            gallery_sample = {
                'subject_id': subject_id,
                'modality_samples': {}
            }

            for modality in self.modalities:
                if modality in subject_data['files']:
                    files = subject_data['files'][modality]
                    if len(files) > 0:
                        gallery_sample['modality_samples'][modality] = files[0]

            if len(gallery_sample['modality_samples']) > 0:
                gallery.append(gallery_sample)

            # Use remaining samples as probes
            for modality in self.modalities:
                if modality not in subject_data['files']:
                    continue

                files = subject_data['files'][modality]
                for file_path in files[1:]:  # Skip first (used in gallery)
                    probe_sample = {
                        'subject_id': subject_id,
                        'modality_samples': {modality: file_path}
                    }
                    probes.append(probe_sample)

        print(f"Generated {len(gallery)} gallery samples and {len(probes)} probe samples")

        return gallery, probes

    def _load_data(self):
        """Required by parent class - we handle loading differently"""
        pass

    def download(self):
        """
        Download LUTBio dataset

        Note: LUTBio requires manual application and approval.
        Visit: https://data.mendeley.com/datasets/jszw485f8j/6
        Email the application form to: rykeryang@163.com
        """
        raise NotImplementedError(
            "LUTBio dataset requires manual download.\n"
            "Please visit: https://data.mendeley.com/datasets/jszw485f8j/6\n"
            "Fill the application form and email to: rykeryang@163.com\n"
            "After approval, download and extract to the specified root directory."
        )

    def __len__(self) -> int:
        if self.mode == 'verification':
            return len(self.pairs)
        else:  # identification
            return len(self.probes)

    def __getitem__(self, idx: int) -> BiometricSample:
        if self.mode == 'verification':
            return self._get_verification_pair(idx)
        else:
            return self._get_identification_sample(idx)

    def _get_verification_pair(self, idx: int) -> BiometricSample:
        """Get a verification pair"""
        pair = self.pairs[idx]

        # Load modality data for both samples
        modalities_1 = self._load_modality_samples(pair['modality_samples_1'])
        modalities_2 = self._load_modality_samples(pair['modality_samples_2'])

        # Return as BiometricSample
        return BiometricSample(
            subject_id=self.subject_id_map[pair['subject_id']],
            modalities=modalities_1,
            is_genuine=pair['is_genuine'],
            metadata={
                'modalities_2': modalities_2,
                'is_verification': True
            }
        )

    def _get_identification_sample(self, idx: int) -> BiometricSample:
        """Get an identification probe sample"""
        probe = self.probes[idx]

        modalities = self._load_modality_samples(probe['modality_samples'])

        return BiometricSample(
            subject_id=self.subject_id_map[probe['subject_id']],
            modalities=modalities,
            is_genuine=True,
            metadata={'is_identification': True}
        )

    def _load_modality_samples(self, modality_samples: Dict[str, Path]) -> Dict:
        """
        Load modality data from file paths

        Args:
            modality_samples: Dict mapping modality to file path

        Returns:
            Dict mapping modality to loaded tensor
        """
        loaded_data = {}

        for modality, file_path in modality_samples.items():
            if modality == 'face' or modality == 'finger':
                # Load image
                img = Image.open(file_path).convert('RGB')

                # Apply transform if available
                if self.transform and modality in self.transform:
                    img = self.transform[modality](img)
                else:
                    # Default: convert to tensor
                    import torchvision.transforms as T
                    img = T.ToTensor()(img)

                loaded_data[modality] = img

            elif modality == 'voice':
                # Load audio
                waveform, sample_rate = torchaudio.load(file_path)

                # Apply transform if available
                if self.transform and modality in self.transform:
                    waveform = self.transform[modality](waveform)

                loaded_data[modality] = waveform

        return loaded_data

    def get_subject_info(self, subject_id: str) -> Dict:
        """Get metadata for a subject"""
        if subject_id in self.subjects_data:
            return self.subjects_data[subject_id]['metadata']
        return None

    def get_dataset_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total_subjects': len(self.subjects_data),
            'active_subjects': len(self.active_subjects),
            'modalities': self.modalities,
            'samples_per_modality': {}
        }

        for modality in self.modalities:
            total_samples = sum(
                len(subject_data['files'].get(modality, []))
                for subject_data in self.subjects_data.values()
            )
            stats['samples_per_modality'][modality] = total_samples

        # Gender distribution
        genders = [data['metadata']['gender'] for data in self.subjects_data.values()]
        stats['gender_distribution'] = {
            'male': genders.count('male'),
            'female': genders.count('female')
        }

        # Age statistics
        ages = [data['metadata']['age'] for data in self.subjects_data.values() if data['metadata']['age'] > 0]
        if ages:
            stats['age_stats'] = {
                'min': min(ages),
                'max': max(ages),
                'mean': np.mean(ages),
                'median': np.median(ages)
            }

        return stats
