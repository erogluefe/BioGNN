#!/usr/bin/env python3
"""
Test script for LUTBio dataset

Usage:
    python scripts/test_lutbio.py --checkpoint experiments/lutbio/checkpoints/best_model.pth
    python scripts/test_lutbio.py --checkpoint best_model.pth --split test
"""

import argparse
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from biognn.data.lutbio_dataset import LUTBioDataset
from biognn.data.lutbio_transforms import get_lutbio_transforms
from biognn.fusion import MultimodalBiometricFusion
from biognn.metrics import (
    calculate_verification_metrics,
    print_metrics_summary,
    plot_roc_curve,
    plot_det_curve
)


def load_config(config_path: str) -> dict:
    """Load configuration from checkpoint or config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def biometric_collate_fn(batch):
    """Custom collate function for BiometricSample objects"""
    modalities = batch[0].get_available_modalities()

    batched_modalities = {}
    for modality in modalities:
        modality_tensors = [sample.modalities[modality] for sample in batch]
        batched_modalities[modality] = torch.stack(modality_tensors, dim=0)

    labels = torch.tensor([sample.labels.get('is_genuine', sample.is_genuine) for sample in batch], dtype=torch.long)
    subject_ids = torch.tensor([sample.subject_id for sample in batch], dtype=torch.long)

    return {
        'modalities': batched_modalities,
        'labels': labels,
        'subject_ids': subject_ids,
        'is_genuine': labels
    }


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load model and config from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        (model, config)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint['config']

    # Build model
    gnn_config = config['model'].get('gnn_config', {}).copy()
    gnn_config['num_classes'] = config['model'].get('num_classes', 2)

    model = MultimodalBiometricFusion(
        modalities=config['dataset']['modalities'],
        feature_dim=config['model']['feature_dim'],
        gnn_type=config['model']['gnn_type'],
        gnn_config=gnn_config,
        edge_strategy=config['model']['graph']['edge_strategy'],
        use_adaptive_edges=config['model']['graph']['use_adaptive_edges'],
        use_quality_scores=config['model']['graph']['use_quality_scores']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    return model, config


def build_test_dataloader(config: dict, split: str = 'test'):
    """Build test dataloader"""

    # Get transforms (no augmentation for test)
    transforms = get_lutbio_transforms(
        split='val',  # Use validation transforms (no augmentation)
        image_size=config['dataset']['face_size'],
        fingerprint_size=config['dataset']['fingerprint_size'],
        spectrogram_size=tuple(config['dataset']['spectrogram_size']),
        augmentation=False
    )

    # Create dataset
    dataset = LUTBioDataset(
        root=config['dataset']['root'],
        modalities=config['dataset']['modalities'],
        split=split,
        mode='verification',
        transform=transforms,
        pairs_per_subject=config['dataset']['pairs_per_subject'],
        seed=config['experiment']['seed']
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=biometric_collate_fn
    )

    return dataloader, dataset


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict
) -> dict:
    """
    Evaluate model and collect scores

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device
        config: Configuration

    Returns:
        Dictionary with scores and labels
    """
    model.eval()

    all_scores = []
    all_labels = []
    all_logits = []
    all_subject_ids = []

    print("\nEvaluating model...")
    for batch in tqdm(dataloader, desc="Evaluation"):
        # Get modality inputs
        modality_inputs = {}
        for modality in config['dataset']['modalities']:
            if modality in batch['modalities']:
                modality_inputs[modality] = batch['modalities'][modality].to(device)

        # Get labels
        labels = batch['labels'].float().to(device)

        # Forward pass
        logits, attention_weights = model(modality_inputs)

        # Convert logits to scores (probabilities)
        scores = torch.sigmoid(logits.squeeze())

        # Store results
        all_logits.extend(logits.squeeze().cpu().numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_subject_ids.extend(batch['subject_ids'].cpu().numpy().tolist())

    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    all_subject_ids = np.array(all_subject_ids)

    return {
        'scores': all_scores,
        'labels': all_labels,
        'logits': all_logits,
        'subject_ids': all_subject_ids
    }


def main():
    parser = argparse.ArgumentParser(description='Test LUTBio model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: checkpoint dir)')
    args = parser.parse_args()

    # Get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    model, config = load_checkpoint(str(checkpoint_path), device)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")

    # Build test dataloader
    print(f"\nBuilding {args.split} dataloader...")
    test_loader, test_dataset = build_test_dataloader(config, args.split)

    print(f"Test set size: {len(test_dataset)} pairs")

    # Evaluate model
    results = evaluate_model(model, test_loader, device, config)

    # Separate genuine and impostor scores
    genuine_mask = results['labels'] == 1
    impostor_mask = results['labels'] == 0

    genuine_scores = results['scores'][genuine_mask]
    impostor_scores = results['scores'][impostor_mask]

    print(f"\nGenuine pairs: {len(genuine_scores)}")
    print(f"Impostor pairs: {len(impostor_scores)}")

    # Check for imbalanced data
    if len(impostor_scores) == 0:
        print("\n⚠️  WARNING: No impostor pairs in test set!")
        print("⚠️  Metrics (EER, FAR, ROC) will not be meaningful.")
        print("⚠️  Consider using a larger test set or cross-validation.\n")
    elif len(genuine_scores) == 0:
        print("\n⚠️  WARNING: No genuine pairs in test set!")
        print("⚠️  Metrics will not be meaningful.\n")

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_verification_metrics(genuine_scores, impostor_scores)

    # Print metrics
    print_metrics_summary(metrics)

    # Plot ROC curve
    print("\nGenerating ROC curve...")
    roc_fig = plot_roc_curve(
        genuine_scores,
        impostor_scores,
        save_path=output_dir / f'roc_curve_{args.split}.png',
        title=f'ROC Curve - {args.split.upper()} Set'
    )
    print(f"Saved: {output_dir / f'roc_curve_{args.split}.png'}")

    # Plot DET curve
    print("Generating DET curve...")
    det_fig = plot_det_curve(
        genuine_scores,
        impostor_scores,
        save_path=output_dir / f'det_curve_{args.split}.png',
        title=f'DET Curve - {args.split.upper()} Set'
    )
    print(f"Saved: {output_dir / f'det_curve_{args.split}.png'}")

    # Plot score distribution
    print("Generating score distribution...")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', color='green', density=True)
    ax.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', color='red', density=True)
    ax.axvline(metrics['eer_threshold'], color='black', linestyle='--', linewidth=2,
               label=f'EER Threshold = {metrics["eer_threshold"]:.3f}')
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Score Distribution - {args.split.upper()} Set', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'score_distribution_{args.split}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / f'score_distribution_{args.split}.png'}")

    # Save metrics to file
    metrics_file = output_dir / f'metrics_{args.split}.txt'
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"VERIFICATION METRICS - {args.split.upper()} SET\n")
        f.write("="*60 + "\n\n")
        f.write(f"EER:                {metrics['eer']*100:.2f}%\n")
        f.write(f"FAR:                {metrics['far']*100:.2f}%\n")
        f.write(f"FRR:                {metrics['frr']*100:.2f}%\n")
        f.write(f"Accuracy:           {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision:          {metrics['precision']*100:.2f}%\n")
        f.write(f"Recall:             {metrics['recall']*100:.2f}%\n")
        f.write(f"F1 Score:           {metrics['f1_score']*100:.2f}%\n")
        f.write(f"AUC:                {metrics['auc']:.4f}\n")
        f.write(f"\nEER Threshold:      {metrics['eer_threshold']:.4f}\n")
        f.write(f"\nGenuine pairs:      {len(genuine_scores)}\n")
        f.write(f"Impostor pairs:     {len(impostor_scores)}\n")

    print(f"\nMetrics saved to: {metrics_file}")

    # Save raw results (for further analysis)
    results_file = output_dir / f'results_{args.split}.npz'
    np.savez(
        results_file,
        scores=results['scores'],
        labels=results['labels'],
        logits=results['logits'],
        subject_ids=results['subject_ids'],
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores
    )
    print(f"Raw results saved to: {results_file}")

    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)
    print(f"\nKey Metrics:")
    print(f"  EER:      {metrics['eer']*100:.2f}%")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
