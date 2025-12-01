#!/usr/bin/env python3
"""
LUTBio Dataset Visualization

Visualizes LUTBio multimodal biometric data including:
- Sample grid with real images
- Dataset statistics
- Distribution analysis

Usage:
    python examples/visualize_lutbio.py --root datasets/lutbio
    python examples/visualize_lutbio.py --root datasets/lutbio --output outputs/lutbio
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from biognn.data.lutbio_dataset import LUTBioDataset
from biognn.data.lutbio_transforms import get_lutbio_transforms
from biognn.visualization import DatasetVisualizer, plot_data_distribution_dashboard


def main():
    parser = argparse.ArgumentParser(description='Visualize LUTBio Dataset')
    parser.add_argument('--root', type=str, default='datasets/lutbio',
                       help='Root directory of LUTBio dataset')
    parser.add_argument('--output', type=str, default='outputs/lutbio',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to visualize')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LUTBIO DATASET VISUALIZATION")
    print("="*80)

    # Get transforms (no augmentation, no normalization for visualization)
    # Use simple transforms: just resize and convert to tensor
    from torchvision import transforms as T
    import torchaudio.transforms as AT

    def get_unnormalized_voice_transform(
        spectrogram_size=(40, 100),
        sample_rate=16000,
        n_fft=400,
        hop_length=160
    ):
        """Voice transform without normalization for visualization"""
        n_mels, target_length = spectrogram_size

        def transform(waveform):
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Create mel spectrogram
            mel_spec = AT.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )(waveform)

            # Convert to log scale (dB)
            mel_spec = AT.AmplitudeToDB()(mel_spec)

            # Resize to fixed length
            mel_spec = torch.nn.functional.interpolate(
                mel_spec.unsqueeze(0),
                size=(n_mels, target_length),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # No normalization for visualization
            return mel_spec

        return transform

    transforms = {
        'face': T.Compose([
            T.Resize((112, 112)),
            T.ToTensor()
        ]),
        'finger': T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((96, 96)),
            T.ToTensor()
        ]),
        'voice': get_unnormalized_voice_transform()
    }

    # Create dataset
    print("\nLoading dataset...")
    dataset = LUTBioDataset(
        root=args.root,
        modalities=['face', 'finger', 'voice'],
        split='train',
        mode='verification',
        transform=transforms,
        pairs_per_subject=10
    )

    # Get dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    stats = dataset.get_dataset_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Visualization 1: Sample Grid
    print("\n[1] Creating sample grid...")

    viz = DatasetVisualizer(['face', 'finger', 'voice'])

    # Get a few samples
    num_samples = min(args.num_samples, len(dataset))
    samples = []
    subject_ids = []

    for i in range(num_samples):
        sample = dataset[i]
        samples.append(sample)
        subject_ids.append(sample.subject_id)

    # Prepare modality tensors
    modality_tensors = {
        'face': torch.stack([s.modalities['face'] for s in samples if 'face' in s.modalities]),
        'finger': torch.stack([s.modalities['finger'] for s in samples if 'finger' in s.modalities]),
        'voice': torch.stack([s.modalities['voice'] for s in samples if 'voice' in s.modalities])
    }

    # Plot sample grid
    fig = viz.plot_sample_grid(
        samples=modality_tensors,
        num_samples=num_samples,
        subject_ids=subject_ids,
        title='LUTBio Multimodal Samples',
        save_path=output_dir / 'lutbio_sample_grid.png'
    )
    plt.close()
    print(f"✓ Saved: {output_dir / 'lutbio_sample_grid.png'}")

    # Visualization 2: Data Distribution Dashboard
    print("\n[2] Creating data distribution dashboard...")

    # Prepare stats for dashboard
    dashboard_stats = {
        'samples_per_modality': stats['samples_per_modality'],
        'train_val_test_split': {
            'train': len(dataset.train_subjects),
            'val': len(dataset.val_subjects),
            'test': len(dataset.test_subjects)
        },
        'class_balance': {
            'genuine': sum(1 for p in dataset.pairs if p['is_genuine']),
            'imposter': sum(1 for p in dataset.pairs if not p['is_genuine'])
        }
    }

    # Add age and gender if available
    if 'gender_distribution' in stats:
        # Ages per subject
        ages = []
        for subject_id in dataset.subjects_data.keys():
            age = dataset.subjects_data[subject_id]['metadata']['age']
            if age > 0:
                ages.append(age)

        if ages:
            dashboard_stats['samples_per_subject'] = [10] * len(ages)  # Approximate

    fig = plot_data_distribution_dashboard(
        dataset_stats=dashboard_stats,
        save_path=output_dir / 'lutbio_distribution.png'
    )
    plt.close()
    print(f"✓ Saved: {output_dir / 'lutbio_distribution.png'}")

    # Visualization 3: Genuine vs Imposter Comparison
    print("\n[3] Creating genuine vs imposter comparison...")

    # Find genuine and imposter pairs
    genuine_idx = next(i for i, p in enumerate(dataset.pairs) if p['is_genuine'])
    imposter_idx = next(i for i, p in enumerate(dataset.pairs) if not p['is_genuine'])

    genuine_sample = dataset[genuine_idx]
    imposter_sample = dataset[imposter_idx]

    # Get both samples from metadata
    genuine_modalities_1 = genuine_sample.modalities
    genuine_modalities_2 = genuine_sample.metadata['modalities_2']

    imposter_modalities_1 = imposter_sample.modalities
    imposter_modalities_2 = imposter_sample.metadata['modalities_2']

    # Prepare for visualization (stack to create batch)
    genuine_samples = {
        mod: torch.stack([genuine_modalities_1[mod].unsqueeze(0), genuine_modalities_2[mod].unsqueeze(0)]).squeeze(1)
        for mod in ['face', 'finger'] if mod in genuine_modalities_1
    }

    imposter_samples = {
        mod: torch.stack([imposter_modalities_1[mod].unsqueeze(0), imposter_modalities_2[mod].unsqueeze(0)]).squeeze(1)
        for mod in ['face', 'finger'] if mod in imposter_modalities_1
    }

    fig = viz.plot_genuine_vs_imposter(
        genuine_samples=genuine_samples,
        imposter_samples=imposter_samples,
        num_pairs=1,
        save_path=output_dir / 'lutbio_genuine_vs_imposter.png'
    )
    plt.close()
    print(f"✓ Saved: {output_dir / 'lutbio_genuine_vs_imposter.png'}")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETED")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - lutbio_sample_grid.png")
    print(f"  - lutbio_distribution.png")
    print(f"  - lutbio_genuine_vs_imposter.png")


if __name__ == '__main__':
    main()
