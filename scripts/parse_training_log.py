#!/usr/bin/env python3
"""
Training log parser - Egitim ciktilarini JSON formatina donusturur.

Kullanim:
    python scripts/parse_training_log.py training.log
    python scripts/parse_training_log.py --stdin < training.log
"""

import re
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def parse_training_log(log_content: str) -> Dict:
    """Parse training log output and extract metrics."""

    epochs_data = []
    current_epoch = None

    # Regex patterns
    epoch_header_pattern = r"Epoch (\d+)/(\d+)"
    train_metrics_pattern = r"Train Loss: ([\d.]+) \| Train Acc: ([\d.]+)%"
    val_metrics_pattern = r"Val Loss: ([\d.]+) \| Val Acc: ([\d.]+)%"
    val_eer_pattern = r"Val EER: ([\d.]+)% \| FAR: ([\d.]+)% \| FRR: ([\d.]+)%"
    checkpoint_pattern = r"Checkpoint saved: (.+\.pth)"
    best_eer_pattern = r"Best validation EER: ([\d.]+)%"

    lines = log_content.strip().split('\n')
    best_eer = None
    total_epochs = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Epoch header
        epoch_match = re.search(epoch_header_pattern, line)
        if epoch_match:
            if current_epoch is not None:
                epochs_data.append(current_epoch)

            current_epoch = {
                'epoch': int(epoch_match.group(1)),
                'total_epochs': int(epoch_match.group(2)),
                'train_loss': None,
                'train_acc': None,
                'val_loss': None,
                'val_acc': None,
                'val_eer': None,
                'val_far': None,
                'val_frr': None,
                'checkpoints': []
            }
            total_epochs = int(epoch_match.group(2))

        # Train metrics
        train_match = re.search(train_metrics_pattern, line)
        if train_match and current_epoch:
            current_epoch['train_loss'] = float(train_match.group(1))
            current_epoch['train_acc'] = float(train_match.group(2)) / 100.0

        # Val metrics
        val_match = re.search(val_metrics_pattern, line)
        if val_match and current_epoch:
            current_epoch['val_loss'] = float(val_match.group(1))
            current_epoch['val_acc'] = float(val_match.group(2)) / 100.0

        # Val EER/FAR/FRR
        eer_match = re.search(val_eer_pattern, line)
        if eer_match and current_epoch:
            current_epoch['val_eer'] = float(eer_match.group(1)) / 100.0
            current_epoch['val_far'] = float(eer_match.group(2)) / 100.0
            current_epoch['val_frr'] = float(eer_match.group(3)) / 100.0

        # Checkpoint
        ckpt_match = re.search(checkpoint_pattern, line)
        if ckpt_match and current_epoch:
            current_epoch['checkpoints'].append(ckpt_match.group(1))

        # Best EER
        best_match = re.search(best_eer_pattern, line)
        if best_match:
            best_eer = float(best_match.group(1)) / 100.0

        i += 1

    # Append last epoch
    if current_epoch is not None:
        epochs_data.append(current_epoch)

    # Build training history
    training_history = {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_eer': [],
        'val_far': [],
        'val_frr': []
    }

    for epoch_data in epochs_data:
        training_history['epochs'].append(epoch_data['epoch'])
        training_history['train_loss'].append(epoch_data['train_loss'])
        training_history['train_accuracy'].append(epoch_data['train_acc'])
        training_history['val_loss'].append(epoch_data['val_loss'])
        training_history['val_accuracy'].append(epoch_data['val_acc'])
        training_history['val_eer'].append(epoch_data['val_eer'])
        training_history['val_far'].append(epoch_data['val_far'])
        training_history['val_frr'].append(epoch_data['val_frr'])

    # Final metrics (from last epoch)
    final_metrics = {}
    if epochs_data:
        last = epochs_data[-1]
        final_metrics = {
            'accuracy': last['val_acc'],
            'eer': last['val_eer'] if last['val_eer'] is not None else 0.0,
            'far': last['val_far'] if last['val_far'] is not None else 0.0,
            'frr': last['val_frr'] if last['val_frr'] is not None else 0.0,
            'val_loss': last['val_loss'],
            'train_loss': last['train_loss'],
            'best_eer': best_eer if best_eer is not None else (last['val_eer'] if last['val_eer'] else 0.0)
        }

    return {
        'training_history': training_history,
        'final_metrics': final_metrics,
        'total_epochs': total_epochs,
        'num_logged_epochs': len(epochs_data),
        'epochs_data': epochs_data
    }


def save_training_history(parsed_data: Dict, output_dir: str = "experiments/lutbio"):
    """Save parsed training data to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training history
    history_file = output_path / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(parsed_data['training_history'], f, indent=2)
    print(f"Training history saved to: {history_file}")

    # Save final metrics
    metrics_file = output_path / "final_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(parsed_data['final_metrics'], f, indent=2)
    print(f"Final metrics saved to: {metrics_file}")

    # Save full data
    full_file = output_path / "training_log_parsed.json"
    with open(full_file, 'w') as f:
        json.dump(parsed_data, f, indent=2)
    print(f"Full parsed data saved to: {full_file}")

    return history_file, metrics_file


def main():
    parser = argparse.ArgumentParser(description='Parse training log to JSON')
    parser.add_argument('log_file', nargs='?', help='Training log file path')
    parser.add_argument('--stdin', action='store_true', help='Read from stdin')
    parser.add_argument('--output', '-o', default='experiments/lutbio', help='Output directory')
    args = parser.parse_args()

    if args.stdin:
        log_content = sys.stdin.read()
    elif args.log_file:
        with open(args.log_file, 'r') as f:
            log_content = f.read()
    else:
        print("Error: Provide log file path or use --stdin")
        sys.exit(1)

    parsed = parse_training_log(log_content)
    save_training_history(parsed, args.output)

    # Print summary
    print(f"\nSummary:")
    print(f"  Total epochs: {parsed['total_epochs']}")
    print(f"  Logged epochs: {parsed['num_logged_epochs']}")
    if parsed['final_metrics']:
        print(f"  Final Accuracy: {parsed['final_metrics']['accuracy']*100:.2f}%")
        print(f"  Final EER: {parsed['final_metrics']['eer']*100:.2f}%")


if __name__ == '__main__':
    main()
