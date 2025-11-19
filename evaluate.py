#!/usr/bin/env python3
"""
Evaluation script for BioGNN multimodal biometric authentication

Usage:
    python evaluate.py --checkpoint experiments/biognn_default/checkpoints/best_model.pth
    python evaluate.py --checkpoint path/to/model.pth --config configs/default_config.yaml
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from biognn.fusion import MultimodalBiometricFusion, EnsembleMultimodalFusion
from biognn.evaluation import BiometricEvaluator
from biognn.attacks import RobustnessEvaluator, SpoofingType


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, device: str) -> torch.nn.Module:
    """Load model from checkpoint"""
    # Build model architecture
    model_config = config['model']

    if model_config['type'] == 'multimodal_fusion':
        model = MultimodalBiometricFusion(
            modalities=config['dataset']['modalities'],
            feature_dim=model_config['feature_dim'],
            gnn_type=model_config['gnn_type'],
            gnn_config=model_config.get('gnn_config'),
            edge_strategy=model_config['graph']['edge_strategy'],
            use_adaptive_edges=model_config['graph']['use_adaptive_edges']
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    return model


def evaluate_model(model, test_loader, device, config):
    """Evaluate model on test set"""
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)

    evaluator = BiometricEvaluator()

    all_labels = []
    all_scores = []
    all_predictions = []

    with torch.no_grad():
        for sample1, sample2, labels in tqdm(test_loader, desc="Evaluating"):
            labels = labels.to(device)

            # Prepare modality inputs
            modality_inputs = {}
            for modality, tensor in sample1.modalities.items():
                modality_inputs[modality] = tensor.to(device)

            # Forward pass
            logits, _ = model(modality_inputs)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            # Collect results
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Convert to numpy
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_predictions = np.array(all_predictions)

    # Evaluate
    results = evaluator.evaluate(all_labels, all_scores, all_predictions)
    evaluator.print_summary()

    # Plot results if configured
    output_dir = Path(config['experiment']['output_dir']) / config['experiment']['name']

    if config['evaluation']['plot_roc']:
        evaluator.plot_roc_curve(
            all_labels, all_scores,
            save_path=str(output_dir / 'roc_curve.png')
        )
        print("✓ Saved ROC curve")

    if config['evaluation']['plot_det']:
        evaluator.plot_det_curve(
            all_labels, all_scores,
            save_path=str(output_dir / 'det_curve.png')
        )
        print("✓ Saved DET curve")

    if config['evaluation']['plot_confusion_matrix']:
        evaluator.plot_confusion_matrix(
            all_labels, all_predictions,
            save_path=str(output_dir / 'confusion_matrix.png')
        )
        print("✓ Saved confusion matrix")

    if config['evaluation']['plot_score_distribution']:
        evaluator.plot_score_distribution(
            all_labels, all_scores,
            save_path=str(output_dir / 'score_distribution.png')
        )
        print("✓ Saved score distribution")

    return results


def evaluate_spoofing_robustness(model, test_loader, device, config):
    """Evaluate robustness against spoofing attacks"""
    if not config['spoofing']['enabled']:
        return None

    print("\n" + "="*70)
    print("EVALUATING ROBUSTNESS AGAINST SPOOFING ATTACKS")
    print("="*70)

    robustness_evaluator = RobustnessEvaluator(model)

    # Get a sample for testing
    sample1, _, _ = next(iter(test_loader))

    # Prepare genuine data
    genuine_data = {}
    for modality, tensor in sample1.modalities.items():
        genuine_data[modality] = tensor[0:1].to(device)  # Take first sample

    # Convert attack type names to enum
    attack_types = []
    for attack_name in config['spoofing']['attack_types']:
        try:
            attack_types.append(SpoofingType[attack_name.upper()])
        except KeyError:
            print(f"Warning: Unknown attack type '{attack_name}', skipping...")

    # Evaluate
    results = robustness_evaluator.evaluate_attack_robustness(
        genuine_data=genuine_data,
        attack_types=attack_types,
        num_trials=config['spoofing']['num_trials']
    )

    robustness_evaluator.print_robustness_report(results)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate BioGNN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set device
    if config['experiment']['device'] == 'cuda' and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = 'cpu'
        print("Using CPU")

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load test data
    print("\nLoading test data...")
    # Note: You need to implement your dataset loading
    test_loader = None  # Implement this

    if test_loader is None:
        print("\nERROR: Test dataset not implemented!")
        print("Please implement the test dataset loader.")
        return

    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, device, config)

    # Evaluate spoofing robustness
    if config['spoofing']['enabled']:
        spoofing_results = evaluate_spoofing_robustness(model, test_loader, device, config)

    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
