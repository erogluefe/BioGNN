#!/usr/bin/env python3
"""
Quick start example for BioGNN

This script demonstrates:
1. Creating a synthetic dataset (for testing without real data)
2. Building a model
3. Training the model
4. Evaluating performance

For real usage, replace SyntheticMultimodalDataset with your own dataset.
"""

import torch
from torch.utils.data import DataLoader

from biognn.data.example_dataset import SyntheticMultimodalDataset
from biognn.data import VerificationPairDataset, biometric_collate_fn
from biognn.fusion import MultimodalBiometricFusion
from biognn.evaluation import BiometricEvaluator
from biognn.utils import Trainer
import numpy as np


def main():
    print("="*70)
    print("BioGNN Quick Start Example")
    print("="*70)

    # ===== 1. Create Datasets =====
    print("\n1. Creating synthetic datasets...")

    # Training set
    train_base = SyntheticMultimodalDataset(
        num_subjects=50,
        samples_per_subject=5,
        modalities=['face', 'fingerprint', 'iris', 'voice'],
        split='train'
    )

    train_dataset = VerificationPairDataset(
        base_dataset=train_base,
        num_pairs=1000,
        genuine_ratio=0.5
    )

    # Validation set
    val_base = SyntheticMultimodalDataset(
        num_subjects=20,
        samples_per_subject=5,
        split='val',
        seed=43
    )

    val_dataset = VerificationPairDataset(
        base_dataset=val_base,
        num_pairs=200,
        genuine_ratio=0.5
    )

    print(f"  Training pairs: {len(train_dataset)}")
    print(f"  Validation pairs: {len(val_dataset)}")

    # ===== 2. Create DataLoaders =====
    print("\n2. Creating data loaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Use 0 for demo, increase for real training
        collate_fn=biometric_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=biometric_collate_fn
    )

    # ===== 3. Build Model =====
    print("\n3. Building model...")

    model = MultimodalBiometricFusion(
        modalities=['face', 'fingerprint', 'iris', 'voice'],
        feature_dim=512,
        gnn_type='gat',  # Try 'gcn', 'gat', or 'graphsage'
        gnn_config={
            'hidden_dims': [256, 128],
            'heads': [4, 4],  # For GAT
            'dropout': 0.5
        },
        edge_strategy='fully_connected',
        use_adaptive_edges=True
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # ===== 4. Setup Training =====
    print("\n4. Setting up training...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir='./experiments',
        experiment_name='quickstart_demo',
        use_amp=False,  # Disable for demo
        log_interval=5
    )

    # ===== 5. Train Model =====
    print("\n5. Training model (demo: 3 epochs)...")
    print("  Note: This is a quick demo. Real training needs more epochs.")

    trainer.train(
        num_epochs=3,
        save_best=True,
        early_stopping_patience=None
    )

    # ===== 6. Evaluate =====
    print("\n6. Evaluating model...")

    model.eval()
    evaluator = BiometricEvaluator()

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for sample1, sample2, labels in val_loader:
            # Prepare inputs
            modality_inputs = {}
            for modality, tensor in sample1.modalities.items():
                modality_inputs[modality] = tensor.to(device)

            # Forward pass
            logits, _ = model(modality_inputs)
            probs = torch.softmax(logits, dim=1)

            all_labels.extend(labels.numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())

    # Compute metrics
    results = evaluator.evaluate(
        np.array(all_labels),
        np.array(all_scores)
    )

    evaluator.print_summary()

    # ===== 7. Save Visualizations =====
    print("\n7. Saving visualizations...")

    output_dir = './experiments/quickstart_demo'

    evaluator.plot_roc_curve(
        np.array(all_labels),
        np.array(all_scores),
        save_path=f'{output_dir}/roc_curve.png'
    )
    print(f"  ✓ ROC curve saved to {output_dir}/roc_curve.png")

    evaluator.plot_det_curve(
        np.array(all_labels),
        np.array(all_scores),
        save_path=f'{output_dir}/det_curve.png'
    )
    print(f"  ✓ DET curve saved to {output_dir}/det_curve.png")

    # ===== Done =====
    print("\n" + "="*70)
    print("Quick start completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Replace SyntheticMultimodalDataset with your real dataset")
    print("2. Adjust hyperparameters in configs/")
    print("3. Train for more epochs")
    print("4. Try different GNN architectures")
    print("5. Run ablation studies: python scripts/ablation_study.py")
    print("\nSee README.md for detailed documentation.")


if __name__ == '__main__':
    main()
