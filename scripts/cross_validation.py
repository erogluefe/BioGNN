#!/usr/bin/env python3
"""
Cross-validation script for LUTBio dataset

Implements Leave-One-Subject-Out (LOSO) cross-validation
Perfect for small datasets like LUTBio (6 subjects)

Usage:
    python scripts/cross_validation.py --config configs/lutbio_config.yaml --num_epochs 50
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
import random

from biognn.data.lutbio_dataset import LUTBioDataset
from biognn.data.lutbio_transforms import get_lutbio_transforms
from biognn.fusion import MultimodalBiometricFusion
from biognn.metrics import calculate_verification_metrics, print_metrics_summary


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def biometric_collate_fn(batch):
    """Custom collate function"""
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


def build_model(config: dict) -> nn.Module:
    """Build model"""
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

    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: dict
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        modality_inputs = {}
        for modality in config['dataset']['modalities']:
            if modality in batch['modalities']:
                modality_inputs[modality] = batch['modalities'][modality].to(device)

        labels = batch['labels'].float().to(device)

        logits, _ = model(modality_inputs)
        loss = criterion(logits.squeeze(), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict
) -> tuple:
    """Evaluate model"""
    model.eval()

    all_scores = []
    all_labels = []

    for batch in dataloader:
        modality_inputs = {}
        for modality in config['dataset']['modalities']:
            if modality in batch['modalities']:
                modality_inputs[modality] = batch['modalities'][modality].to(device)

        labels = batch['labels'].float().to(device)

        logits, _ = model(modality_inputs)
        scores = torch.sigmoid(logits.squeeze())

        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    genuine_mask = all_labels == 1
    impostor_mask = all_labels == 0

    genuine_scores = all_scores[genuine_mask]
    impostor_scores = all_scores[impostor_mask]

    return genuine_scores, impostor_scores


def run_fold(
    config: dict,
    test_subject: str,
    all_subjects: list,
    device: torch.device,
    num_epochs: int,
    fold_idx: int
) -> dict:
    """
    Run one fold of cross-validation

    Args:
        config: Configuration
        test_subject: Subject to use for testing
        all_subjects: List of all subjects
        device: Device
        num_epochs: Number of training epochs
        fold_idx: Fold index

    Returns:
        Dictionary of results
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx+1} - Test Subject: {test_subject}")
    print(f"{'='*70}\n")

    # Create train/val split from remaining subjects
    train_subjects = [s for s in all_subjects if s != test_subject]

    # Use last training subject for validation
    if len(train_subjects) > 1:
        val_subject = [train_subjects[-1]]
        train_subjects = train_subjects[:-1]
    else:
        val_subject = train_subjects
        # If only one training subject, use it for both

    print(f"Train subjects: {train_subjects}")
    print(f"Val subject: {val_subject}")
    print(f"Test subject: {test_subject}\n")

    # Build datasets
    train_transforms = get_lutbio_transforms(
        split='train',
        image_size=config['dataset']['face_size'],
        fingerprint_size=config['dataset']['fingerprint_size'],
        spectrogram_size=tuple(config['dataset']['spectrogram_size']),
        augmentation=config['dataset']['augmentation']
    )

    val_transforms = get_lutbio_transforms(
        split='val',
        image_size=config['dataset']['face_size'],
        fingerprint_size=config['dataset']['fingerprint_size'],
        spectrogram_size=tuple(config['dataset']['spectrogram_size']),
        augmentation=False
    )

    # Create datasets with specific subject splits
    train_dataset = LUTBioDataset(
        root=config['dataset']['root'],
        modalities=config['dataset']['modalities'],
        split='train',
        mode='verification',
        transform=train_transforms,
        train_subjects=train_subjects,
        val_subjects=val_subject,
        test_subjects=[test_subject],
        pairs_per_subject=config['dataset']['pairs_per_subject'],
        seed=config['experiment']['seed'] + fold_idx  # Different seed per fold
    )

    val_dataset = LUTBioDataset(
        root=config['dataset']['root'],
        modalities=config['dataset']['modalities'],
        split='val',
        mode='verification',
        transform=val_transforms,
        train_subjects=train_subjects,
        val_subjects=val_subject,
        test_subjects=[test_subject],
        pairs_per_subject=config['dataset']['pairs_per_subject'],
        seed=config['experiment']['seed'] + fold_idx
    )

    test_dataset = LUTBioDataset(
        root=config['dataset']['root'],
        modalities=config['dataset']['modalities'],
        split='test',
        mode='verification',
        transform=val_transforms,
        train_subjects=train_subjects,
        val_subjects=val_subject,
        test_subjects=[test_subject],
        pairs_per_subject=config['dataset']['pairs_per_subject'],
        seed=config['experiment']['seed'] + fold_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=biometric_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=biometric_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=biometric_collate_fn
    )

    print(f"Train: {len(train_dataset)} pairs")
    print(f"Val:   {len(val_dataset)} pairs")
    print(f"Test:  {len(test_dataset)} pairs\n")

    # Build model
    model = build_model(config)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_val_eer = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)

        # Validate
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_genuine, val_impostor = evaluate(model, val_loader, device, config)
            val_metrics = calculate_verification_metrics(val_genuine, val_impostor)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {train_loss:.4f} - "
                  f"Val EER: {val_metrics['eer']*100:.2f}% - "
                  f"Val Acc: {val_metrics['accuracy']*100:.2f}%")

            # Early stopping
            if val_metrics['eer'] < best_val_eer:
                best_val_eer = val_metrics['eer']
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    print(f"\nEvaluating on test set...")
    test_genuine, test_impostor = evaluate(model, test_loader, device, config)

    if len(test_genuine) == 0 or len(test_impostor) == 0:
        print(f"Warning: Fold {fold_idx+1} has insufficient test samples")
        return None

    test_metrics = calculate_verification_metrics(test_genuine, test_impostor)

    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx+1} RESULTS")
    print(f"{'='*70}")
    print(f"Test Subject: {test_subject}")
    print(f"EER:          {test_metrics['eer']*100:.2f}%")
    print(f"Accuracy:     {test_metrics['accuracy']*100:.2f}%")
    print(f"AUC:          {test_metrics['auc']:.4f}")
    print(f"{'='*70}\n")

    return {
        'test_subject': test_subject,
        'metrics': test_metrics,
        'genuine_scores': test_genuine,
        'impostor_scores': test_impostor
    }


def main():
    parser = argparse.ArgumentParser(description='Cross-validation for LUTBio')
    parser.add_argument('--config', type=str, default='configs/lutbio_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs per fold')
    parser.add_argument('--output_dir', type=str, default='experiments/cross_validation',
                       help='Output directory')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config['experiment']['seed'])

    # Get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"\nUsing device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all subjects
    root = Path(config['dataset']['root'])
    all_subjects = sorted([d.name for d in root.iterdir() if d.is_dir()])

    print(f"\nAll subjects: {all_subjects}")
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Cross-validation: {len(all_subjects)}-fold LOSO\n")

    # Run cross-validation
    fold_results = []

    for fold_idx, test_subject in enumerate(all_subjects):
        result = run_fold(
            config=config,
            test_subject=test_subject,
            all_subjects=all_subjects,
            device=device,
            num_epochs=args.num_epochs,
            fold_idx=fold_idx
        )

        if result is not None:
            fold_results.append(result)

    # Aggregate results
    if not fold_results:
        print("Error: No valid fold results")
        return

    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70 + "\n")

    # Per-fold results
    print("Per-Fold Results:")
    print("-" * 70)
    for i, result in enumerate(fold_results):
        metrics = result['metrics']
        print(f"Fold {i+1} ({result['test_subject']}): "
              f"EER={metrics['eer']*100:.2f}%, "
              f"Acc={metrics['accuracy']*100:.2f}%, "
              f"AUC={metrics['auc']:.4f}")

    # Mean and std
    eers = [r['metrics']['eer'] for r in fold_results]
    accs = [r['metrics']['accuracy'] for r in fold_results]
    aucs = [r['metrics']['auc'] for r in fold_results]

    print("\n" + "-" * 70)
    print("Mean ± Std:")
    print("-" * 70)
    print(f"EER:      {np.mean(eers)*100:.2f}% ± {np.std(eers)*100:.2f}%")
    print(f"Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print("="*70 + "\n")

    # Save results
    results_file = output_dir / 'cv_results.txt'
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CROSS-VALIDATION RESULTS\n")
        f.write("="*70 + "\n\n")

        f.write("Per-Fold Results:\n")
        f.write("-" * 70 + "\n")
        for i, result in enumerate(fold_results):
            metrics = result['metrics']
            f.write(f"Fold {i+1} ({result['test_subject']}): ")
            f.write(f"EER={metrics['eer']*100:.2f}%, ")
            f.write(f"Acc={metrics['accuracy']*100:.2f}%, ")
            f.write(f"AUC={metrics['auc']:.4f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("Mean ± Std:\n")
        f.write("-" * 70 + "\n")
        f.write(f"EER:      {np.mean(eers)*100:.2f}% ± {np.std(eers)*100:.2f}%\n")
        f.write(f"Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%\n")
        f.write(f"AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\n")

    print(f"Results saved to: {results_file}")

    # Save detailed results
    np.savez(
        output_dir / 'cv_detailed_results.npz',
        fold_eers=eers,
        fold_accs=accs,
        fold_aucs=aucs,
        subjects=all_subjects
    )

    print(f"Detailed results saved to: {output_dir / 'cv_detailed_results.npz'}")


if __name__ == '__main__':
    main()
