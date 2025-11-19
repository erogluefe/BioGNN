#!/usr/bin/env python3
"""
Ablation study for multimodal biometric fusion

Tests:
1. Modality ablation: Remove each modality and measure impact
2. Architecture ablation: Test different GNN configurations
3. Component ablation: Test impact of different fusion components
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from itertools import combinations
import json
import pandas as pd

from biognn.fusion import MultimodalBiometricFusion
from biognn.evaluation import BiometricEvaluator


def modality_ablation_study(
    model,
    test_loader,
    modalities,
    device='cuda'
):
    """
    Test performance with different modality combinations

    Args:
        model: Trained multimodal fusion model
        test_loader: Test data loader
        modalities: List of all modalities
        device: Device to run on

    Returns:
        results: Dictionary with results for each combination
    """
    print("\n" + "="*70)
    print("MODALITY ABLATION STUDY")
    print("="*70)

    results = {}

    # Test all possible combinations
    for r in range(1, len(modalities) + 1):
        for combo in combinations(modalities, r):
            combo_name = "+".join(combo)
            print(f"\nTesting: {combo_name}")

            # Evaluate with only these modalities
            evaluator = BiometricEvaluator()
            all_labels = []
            all_scores = []

            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    sample1, sample2, labels = batch

                    # Filter modalities
                    modality_inputs = {
                        mod: sample1.modalities[mod].to(device)
                        for mod in combo
                        if mod in sample1.modalities
                    }

                    if len(modality_inputs) != len(combo):
                        continue  # Skip if not all modalities available

                    # Forward pass
                    logits, _ = model(modality_inputs, extract_features=True)
                    probs = torch.softmax(logits, dim=1)

                    all_labels.extend(labels.numpy())
                    all_scores.extend(probs[:, 1].cpu().numpy())

            if not all_labels:
                continue

            # Evaluate
            metrics = evaluator.evaluate(
                np.array(all_labels),
                np.array(all_scores)
            )

            results[combo_name] = {
                'num_modalities': len(combo),
                'modalities': list(combo),
                'eer': metrics['eer'],
                'auc': metrics['auc'],
                'accuracy': metrics['accuracy'],
                'far': metrics['far'],
                'frr': metrics['frr']
            }

            print(f"  EER: {metrics['eer']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")

    return results


def architecture_ablation_study(
    modalities,
    train_loader,
    val_loader,
    device='cuda'
):
    """
    Test different GNN architectures and configurations

    Tests:
    - Different GNN types (GCN, GAT, GraphSAGE)
    - Different number of layers
    - Different hidden dimensions
    - Different attention heads (for GAT)
    """
    print("\n" + "="*70)
    print("ARCHITECTURE ABLATION STUDY")
    print("="*70)

    results = {}

    # Test different GNN types
    gnn_types = ['gcn', 'gat', 'graphsage']
    for gnn_type in gnn_types:
        print(f"\nTesting GNN type: {gnn_type}")

        config = {
            'input_dim': 512,
            'hidden_dims': [256, 128],
            'dropout': 0.5
        }

        if gnn_type == 'gat':
            config['heads'] = [4, 4]

        model = MultimodalBiometricFusion(
            modalities=modalities,
            feature_dim=512,
            gnn_type=gnn_type,
            gnn_config=config
        ).to(device)

        # Quick training (simplified)
        # In practice, you would do full training
        print(f"  Training {gnn_type}... (simplified)")
        # ... training code ...

        results[gnn_type] = {
            'type': gnn_type,
            'config': config,
            # Add metrics after training
        }

    # Test different layer depths
    layer_configs = [
        [256],
        [256, 128],
        [512, 256, 128],
        [512, 256, 128, 64]
    ]

    for i, hidden_dims in enumerate(layer_configs):
        print(f"\nTesting {len(hidden_dims)} layers: {hidden_dims}")
        # Similar evaluation as above
        # ... code ...

    return results


def component_ablation_study():
    """
    Test impact of different fusion components

    Tests:
    - With/without adaptive edge weighting
    - With/without attention mechanism
    - Different pooling strategies
    """
    print("\n" + "="*70)
    print("COMPONENT ABLATION STUDY")
    print("="*70)

    components = [
        ('adaptive_edges', [True, False]),
        ('pooling', ['mean', 'max', 'concat']),
        ('edge_strategy', ['fully_connected', 'star', 'hierarchical'])
    ]

    results = {}

    for component_name, values in components:
        print(f"\nTesting {component_name}:")
        for value in values:
            print(f"  {component_name}={value}")
            # Build and test model with this configuration
            # ... code ...

    return results


def save_results(results, output_path):
    """Save ablation results to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Save as CSV for easy analysis
    if isinstance(results, dict):
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(output_path.with_suffix('.csv'))

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Ablation study for BioGNN')
    parser.add_argument('--study', type=str, default='modality',
                       choices=['modality', 'architecture', 'component', 'all'],
                       help='Type of ablation study')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='ablation_results',
                       help='Output file prefix')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()

    print(f"Running {args.study} ablation study...")

    # Load model and data
    # ... (implementation depends on your data pipeline)

    results = {}

    if args.study in ['modality', 'all']:
        modality_results = modality_ablation_study(
            model=None,  # Load your model here
            test_loader=None,  # Load your test data here
            modalities=['face', 'fingerprint', 'iris', 'voice'],
            device=args.device
        )
        results['modality'] = modality_results
        save_results(modality_results, f"{args.output}_modality")

    if args.study in ['architecture', 'all']:
        arch_results = architecture_ablation_study(
            modalities=['face', 'fingerprint', 'iris', 'voice'],
            train_loader=None,
            val_loader=None,
            device=args.device
        )
        results['architecture'] = arch_results
        save_results(arch_results, f"{args.output}_architecture")

    if args.study in ['component', 'all']:
        component_results = component_ablation_study()
        results['component'] = component_results
        save_results(component_results, f"{args.output}_component")

    print("\nAblation study completed!")


if __name__ == '__main__':
    main()
