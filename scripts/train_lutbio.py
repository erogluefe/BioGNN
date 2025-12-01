#!/usr/bin/env python3
"""
Training script for LUTBio dataset

Usage:
    python scripts/train_lutbio.py --config configs/lutbio_config.yaml
    python scripts/train_lutbio.py --config configs/lutbio_config.yaml --resume experiments/lutbio/checkpoints/last.pth
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
import random
from tqdm import tqdm

from biognn.data.lutbio_dataset import LUTBioDataset
from biognn.data.lutbio_transforms import get_lutbio_transforms
from biognn.fusion import MultimodalBiometricFusion
from biognn.visualization import TrainingMonitor, create_training_dashboard

# Set random seed
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device(config: dict) -> torch.device:
    device_type = config['device']['type']

    if device_type == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_type)

    print(f"Using device: {device}")
    return device


def biometric_collate_fn(batch):
    """
    Custom collate function for BiometricSample objects

    Args:
        batch: List of BiometricSample objects

    Returns:
        Dictionary with batched tensors
    """
    # Get all available modalities from first sample
    modalities = batch[0].get_available_modalities()

    # Batch modalities
    batched_modalities = {}
    for modality in modalities:
        modality_tensors = [sample.modalities[modality] for sample in batch]
        batched_modalities[modality] = torch.stack(modality_tensors, dim=0)

    # Batch labels
    labels = torch.tensor([sample.labels.get('is_genuine', sample.is_genuine) for sample in batch], dtype=torch.long)

    # Batch subject IDs (now properly mapped to integers in dataset)
    subject_ids = torch.tensor([sample.subject_id for sample in batch], dtype=torch.long)

    return {
        'modalities': batched_modalities,
        'labels': labels,
        'subject_ids': subject_ids,
        'is_genuine': labels  # alias for compatibility
    }


def build_dataloaders(config: dict):
    """Build train and validation dataloaders"""

    # Get transforms
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

    # Create datasets
    train_dataset = LUTBioDataset(
        root=config['dataset']['root'],
        modalities=config['dataset']['modalities'],
        split='train',
        mode='verification',  # Start with verification
        transform=train_transforms,
        pairs_per_subject=config['dataset']['pairs_per_subject'],
        seed=config['experiment']['seed']
    )

    val_dataset = LUTBioDataset(
        root=config['dataset']['root'],
        modalities=config['dataset']['modalities'],
        split='val',
        mode='verification',
        transform=val_transforms,
        pairs_per_subject=config['dataset']['pairs_per_subject'],
        seed=config['experiment']['seed']
    )

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for Mac compatibility
        pin_memory=False,
        collate_fn=biometric_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=biometric_collate_fn
    )

    return train_loader, val_loader, train_dataset, val_dataset


def build_model(config: dict) -> nn.Module:
    """Build multimodal fusion model"""

    model = MultimodalBiometricFusion(
        modalities=config['dataset']['modalities'],
        feature_dim=config['model']['feature_dim'],
        gnn_type=config['model']['gnn_type'],
        gnn_config=config['model'].get('gnn_config'),
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
    epoch: int,
    config: dict
) -> dict:
    """Train for one epoch"""

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Get modality inputs - batch is now a dict from collate_fn
        modality_inputs = {}
        for modality in config['dataset']['modalities']:
            if modality in batch['modalities']:
                modality_inputs[modality] = batch['modalities'][modality].to(device)

        # Get labels
        labels = batch['labels'].float().to(device)

        # Forward pass
        logits, _ = model(modality_inputs)

        # Compute loss
        loss = criterion(logits.squeeze(), labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    metrics = {
        'train_loss': total_loss / len(dataloader),
        'train_accuracy': 100. * correct / total
    }

    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict
) -> dict:
    """Validate model"""

    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_scores = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Validation"):
        # Get modality inputs - batch is now a dict from collate_fn
        modality_inputs = {}
        for modality in config['dataset']['modalities']:
            if modality in batch['modalities']:
                modality_inputs[modality] = batch['modalities'][modality].to(device)

        # Get labels
        labels = batch['labels'].float().to(device)

        # Forward pass
        logits, _ = model(modality_inputs)

        # Compute loss
        loss = criterion(logits.squeeze(), labels)

        # Statistics
        total_loss += loss.item()
        scores = torch.sigmoid(logits.squeeze())
        predictions = (scores > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Store for EER calculation
        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    # Calculate EER (Equal Error Rate)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Simple EER calculation
    thresholds = np.linspace(0, 1, 100)
    far_list = []
    frr_list = []

    for threshold in thresholds:
        predictions = (all_scores > threshold).astype(int)

        # FAR: false accepts / total imposters
        imposters = (all_labels == 0)
        if imposters.sum() > 0:
            far = (predictions[imposters] == 1).sum() / imposters.sum()
        else:
            far = 0

        # FRR: false rejects / total genuines
        genuines = (all_labels == 1)
        if genuines.sum() > 0:
            frr = (predictions[genuines] == 0).sum() / genuines.sum()
        else:
            frr = 0

        far_list.append(far)
        frr_list.append(frr)

    far_list = np.array(far_list)
    frr_list = np.array(frr_list)

    # EER is where FAR = FRR
    eer_idx = np.argmin(np.abs(far_list - frr_list))
    eer = (far_list[eer_idx] + frr_list[eer_idx]) / 2

    metrics = {
        'val_loss': total_loss / len(dataloader),
        'val_accuracy': 100. * correct / total,
        'val_eer': eer * 100,  # Convert to percentage
        'val_far': far_list[eer_idx] * 100,
        'val_frr': frr_list[eer_idx] * 100
    }

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    filename: str = 'checkpoint.pth'
):
    """Save training checkpoint"""

    checkpoint_dir = Path(config['experiment']['output_dir']) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }

    torch.save(checkpoint, checkpoint_dir / filename)
    print(f"Checkpoint saved: {checkpoint_dir / filename}")


def main():
    parser = argparse.ArgumentParser(description='Train LUTBio model')
    parser.add_argument('--config', type=str, default='configs/lutbio_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded configuration from {args.config}")

    # Set seed
    set_seed(config['experiment']['seed'])

    # Get device
    device = get_device(config)

    # Create output directory
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataloaders
    print("\nBuilding dataloaders...")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(config)

    # Print dataset statistics
    print("\nDataset Statistics:")
    stats = train_dataset.get_dataset_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=tuple(config['training']['optimizer']['betas'])
    )

    # Learning rate scheduler
    if config['training']['scheduler']['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['scheduler']['min_lr']
        )
    else:
        scheduler = None

    # Training monitor
    monitor = TrainingMonitor(
        modalities=config['dataset']['modalities'],
        save_dir=output_dir / 'visualizations'
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    best_eer = float('inf')

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, config)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)

        # Combine metrics
        metrics = {**train_metrics, **val_metrics}

        # Add learning rate
        metrics['learning_rate'] = optimizer.param_groups[0]['lr']

        # Log metrics
        monitor.log_metrics(metrics, epoch)

        # Print metrics
        print(f"\nTrain Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']:.2f}%")
        print(f"Val EER: {val_metrics['val_eer']:.2f}% | FAR: {val_metrics['val_far']:.2f}% | FRR: {val_metrics['val_frr']:.2f}%")

        # Save checkpoints
        if epoch % config['experiment']['save_checkpoint_every'] == 0:
            save_checkpoint(model, optimizer, epoch, metrics, config, f'checkpoint_epoch_{epoch}.pth')

        # Save best model
        if val_metrics['val_eer'] < best_eer:
            best_eer = val_metrics['val_eer']
            save_checkpoint(model, optimizer, epoch, metrics, config, 'best_model.pth')
            print(f"✓ New best EER: {best_eer:.2f}%")

        # Save last checkpoint
        save_checkpoint(model, optimizer, epoch, metrics, config, 'last.pth')

        # Learning rate step
        if scheduler:
            scheduler.step()

        # Early stopping
        if config['training']['early_stopping']['enabled']:
            # Simple implementation: check if no improvement for N epochs
            # You can implement more sophisticated early stopping here
            pass

    # Final visualization
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)

    print("\nGenerating final visualizations...")
    monitor.plot_training_curves(save_path=output_dir / 'visualizations' / 'training_curves.png')
    create_training_dashboard(monitor, save_path=output_dir / 'visualizations' / 'training_dashboard.png')

    print(f"\nBest validation EER: {best_eer:.2f}%")
    print(f"Model and logs saved to: {output_dir}")


if __name__ == '__main__':
    main()
