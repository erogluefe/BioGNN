#!/usr/bin/env python3
"""
Training script for BioGNN multimodal biometric authentication

Usage:
    python train.py --config configs/default_config.yaml
    python train.py --config configs/gcn_config.yaml --gpu 0
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import random

from biognn.fusion import MultimodalBiometricFusion, EnsembleMultimodalFusion
from biognn.data import VerificationPairDataset, get_default_transforms
from biognn.utils import Trainer
from biognn.evaluation import BiometricEvaluator


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict) -> nn.Module:
    """Build model from configuration"""
    model_config = config['model']

    if model_config['type'] == 'multimodal_fusion':
        model = MultimodalBiometricFusion(
            modalities=config['dataset']['modalities'],
            feature_dim=model_config['feature_dim'],
            gnn_type=model_config['gnn_type'],
            gnn_config=model_config.get('gnn_config'),
            edge_strategy=model_config['graph']['edge_strategy'],
            use_adaptive_edges=model_config['graph']['use_adaptive_edges'],
            use_quality_scores=model_config['graph']['use_quality_scores']
        )

    elif model_config['type'] == 'ensemble':
        model = EnsembleMultimodalFusion(
            modalities=config['dataset']['modalities'],
            feature_dim=model_config['feature_dim'],
            gnn_types=model_config['gnn_types'],
            ensemble_method=model_config['ensemble_method']
        )

    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")

    return model


def build_dataloaders(config: dict):
    """Build data loaders"""
    # Note: This is a template. You need to implement your actual dataset class
    # that inherits from MultimodalBiometricDataset

    print("WARNING: Using dummy dataset. Please implement your actual dataset loader.")
    print("See biognn/data/base_dataset.py for the base class.")

    # For now, return None to indicate dataset needs to be implemented
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Train BioGNN model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    print(yaml.dump(config, default_flow_style=False))

    # Set seed
    set_seed(config['experiment']['seed'])
    print(f"Set random seed to {config['experiment']['seed']}")

    # Set device
    if config['experiment']['device'] == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    print(f"Model: {config['model']['type']} with {config['model']['gnn_type']} GNN")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Build dataloaders
    print("\nBuilding dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(config)

    if train_loader is None:
        print("\n" + "="*70)
        print("ERROR: Dataset not implemented!")
        print("="*70)
        print("\nTo use this training script, you need to:")
        print("1. Implement a dataset class that inherits from MultimodalBiometricDataset")
        print("2. Update the build_dataloaders() function in this script")
        print("\nSee the following files for reference:")
        print("  - biognn/data/base_dataset.py (base classes)")
        print("  - biognn/data/transforms.py (data transforms)")
        print("  - biognn/data/feature_extractors.py (feature extraction)")
        print("\nExample dataset structure:")
        print("  datasets/")
        print("    ├── train/")
        print("    │   ├── face/")
        print("    │   ├── fingerprint/")
        print("    │   ├── iris/")
        print("    │   └── voice/")
        print("    ├── val/")
        print("    └── test/")
        print("="*70 + "\n")
        return

    # Build optimizer
    if config['training']['optimizer']['type'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['optimizer']['learning_rate'],
            weight_decay=config['training']['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']['type']}")

    # Build loss function
    if config['training']['loss']['type'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {config['training']['loss']['type']}")

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=config['experiment']['output_dir'],
        experiment_name=config['experiment']['name'],
        use_amp=config['training']['use_amp'],
        log_interval=config['experiment']['log_interval']
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    early_stopping_patience = None
    if config['training']['early_stopping']['enabled']:
        early_stopping_patience = config['training']['early_stopping']['patience']

    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_best=config['training']['save_best'],
        early_stopping_patience=early_stopping_patience
    )

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
