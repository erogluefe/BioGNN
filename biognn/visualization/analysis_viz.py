"""
Analysis and Evaluation Visualization Tools

Visualizes:
- Error analysis (hard negatives, misclassifications)
- Spoofing attack visualization
- Augmentation comparison
- Per-subject performance
- Quality impact analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    sample_ids: Optional[List] = None,
    top_k: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze and visualize prediction errors

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities
        sample_ids: Optional sample identifiers
        top_k: Number of top errors to show
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Find errors
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]

    if len(error_indices) == 0:
        print("No errors found!")
        return None

    # Compute error confidence (how confident was the wrong prediction)
    error_scores = y_scores[error_indices]
    error_true = y_true[error_indices]
    error_pred = y_pred[error_indices]

    # Sort by confidence (most confident errors first)
    sorted_indices = np.argsort(error_scores)[::-1][:top_k]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Error distribution by confidence
    ax1 = axes[0, 0]
    ax1.hist(error_scores, bins=30, edgecolor='black', alpha=0.7, color='red')
    ax1.axvline(np.mean(error_scores), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(error_scores):.3f}')
    ax1.set_xlabel('Prediction Confidence', fontsize=11)
    ax1.set_ylabel('Number of Errors', fontsize=11)
    ax1.set_title('Error Distribution by Confidence', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. False Positives vs False Negatives
    ax2 = axes[0, 1]
    false_positives = np.sum((error_pred == 1) & (error_true == 0))
    false_negatives = np.sum((error_pred == 0) & (error_true == 1))

    bars = ax2.bar(['False Positives\n(Imposters accepted)', 'False Negatives\n(Genuine rejected)'],
                   [false_positives, false_negatives],
                   color=['red', 'orange'],
                   edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Error Type Breakdown', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Top errors table
    ax3 = axes[1, 0]
    ax3.axis('tight')
    ax3.axis('off')

    table_data = []
    for i, idx in enumerate(sorted_indices):
        orig_idx = error_indices[idx]
        sample_id = sample_ids[orig_idx] if sample_ids else orig_idx
        table_data.append([
            str(sample_id)[:10],
            f'{error_true[idx]}',
            f'{error_pred[idx]}',
            f'{error_scores[idx]:.3f}'
        ])

    table = ax3.table(
        cellText=table_data,
        colLabels=['Sample ID', 'True', 'Pred', 'Confidence'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.2, 0.2, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax3.set_title(f'Top {top_k} Most Confident Errors', fontsize=12, fontweight='bold', pad=20)

    # 4. Error rate by confidence bins
    ax4 = axes[1, 1]
    bins = np.linspace(0, 1, 11)
    bin_errors = []
    bin_totals = []
    bin_centers = []

    for i in range(len(bins) - 1):
        mask = (y_scores >= bins[i]) & (y_scores < bins[i+1])
        if np.sum(mask) > 0:
            bin_errors.append(np.sum(errors[mask]))
            bin_totals.append(np.sum(mask))
            bin_centers.append((bins[i] + bins[i+1]) / 2)

    error_rates = [e / t * 100 if t > 0 else 0 for e, t in zip(bin_errors, bin_totals)]

    ax4.bar(bin_centers, error_rates, width=0.08, edgecolor='black', alpha=0.7, color='coral')
    ax4.set_xlabel('Confidence Bin', fontsize=11)
    ax4.set_ylabel('Error Rate (%)', fontsize=11)
    ax4.set_title('Error Rate by Confidence', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Error Analysis - Total Errors: {len(error_indices)} ({len(error_indices)/len(y_true)*100:.2f}%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_hard_negative_pairs(
    samples: Dict[str, Union[torch.Tensor, np.ndarray]],
    scores: np.ndarray,
    labels: np.ndarray,
    modalities: List[str],
    num_pairs: int = 5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize hard negative pairs (imposters with high similarity scores)

    Args:
        samples: Dictionary mapping modality to sample images
        scores: Similarity scores
        labels: True labels (0=imposter, 1=genuine)
        modalities: List of modalities
        num_pairs: Number of hard pairs to show
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Find hard negatives (imposters with high scores)
    imposter_mask = labels == 0
    imposter_scores = scores[imposter_mask]
    imposter_indices = np.where(imposter_mask)[0]

    # Sort by score (highest first)
    sorted_idx = np.argsort(imposter_scores)[::-1][:num_pairs]

    fig, axes = plt.subplots(num_pairs, len(modalities), figsize=(3 * len(modalities), 3 * num_pairs))

    if num_pairs == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(sorted_idx):
        orig_idx = imposter_indices[idx]
        score = imposter_scores[idx]

        for col, modality in enumerate(modalities):
            ax = axes[row, col]

            if modality in samples:
                sample = samples[modality][orig_idx]
                if isinstance(sample, torch.Tensor):
                    sample = sample.cpu().numpy()

                if sample.ndim == 3 and sample.shape[0] in [1, 3]:
                    sample = np.transpose(sample, (1, 2, 0)).squeeze()

                if sample.ndim == 2:
                    ax.imshow(sample, cmap='gray', aspect='auto')

                if row == 0:
                    ax.set_title(modality.capitalize(), fontsize=11, fontweight='bold')

                if col == 0:
                    ax.set_ylabel(f'Score: {score:.3f}', fontsize=10, fontweight='bold', color='red')

            ax.set_xticks([])
            ax.set_yticks([])

            # Red border for hard negatives
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)

    plt.suptitle('Hard Negative Pairs (High-Score Imposters)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_per_subject_performance(
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze performance per subject

    Args:
        subject_ids: Subject IDs for each sample
        y_true: True labels
        y_pred: Predicted labels
        top_k: Number of subjects to show
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    unique_subjects = np.unique(subject_ids)

    subject_accuracy = []
    subject_samples = []

    for subject in unique_subjects:
        mask = subject_ids == subject
        acc = np.mean(y_true[mask] == y_pred[mask])
        subject_accuracy.append(acc)
        subject_samples.append(np.sum(mask))

    subject_accuracy = np.array(subject_accuracy)
    subject_samples = np.array(subject_samples)

    # Sort by accuracy
    sorted_idx = np.argsort(subject_accuracy)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy distribution
    ax1 = axes[0, 0]
    ax1.hist(subject_accuracy * 100, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(np.mean(subject_accuracy) * 100, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(subject_accuracy)*100:.1f}%')
    ax1.set_xlabel('Accuracy (%)', fontsize=11)
    ax1.set_ylabel('Number of Subjects', fontsize=11)
    ax1.set_title('Per-Subject Accuracy Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Worst performing subjects
    ax2 = axes[0, 1]
    worst_subjects = sorted_idx[:top_k]
    y_pos = np.arange(len(worst_subjects))

    bars = ax2.barh(y_pos, subject_accuracy[worst_subjects] * 100, color='coral', edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'Subject {unique_subjects[i]}' for i in worst_subjects], fontsize=8)
    ax2.set_xlabel('Accuracy (%)', fontsize=11)
    ax2.set_title(f'Bottom {top_k} Subjects', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # 3. Best performing subjects
    ax3 = axes[1, 0]
    best_subjects = sorted_idx[-top_k:][::-1]
    y_pos = np.arange(len(best_subjects))

    bars = ax3.barh(y_pos, subject_accuracy[best_subjects] * 100, color='lightgreen', edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f'Subject {unique_subjects[i]}' for i in best_subjects], fontsize=8)
    ax3.set_xlabel('Accuracy (%)', fontsize=11)
    ax3.set_title(f'Top {top_k} Subjects', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()

    # 4. Accuracy vs number of samples
    ax4 = axes[1, 1]
    scatter = ax4.scatter(subject_samples, subject_accuracy * 100, alpha=0.6, s=50, c=subject_accuracy, cmap='RdYlGn')
    ax4.set_xlabel('Number of Samples', fontsize=11)
    ax4.set_ylabel('Accuracy (%)', fontsize=11)
    ax4.set_title('Accuracy vs Sample Count', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Accuracy')

    plt.suptitle(f'Per-Subject Performance Analysis ({len(unique_subjects)} subjects)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_spoofing_attack_comparison(
    genuine_samples: Dict[str, Union[torch.Tensor, np.ndarray]],
    spoofed_samples: Dict[str, Union[torch.Tensor, np.ndarray]],
    attack_types: List[str],
    detection_scores: Optional[np.ndarray] = None,
    modalities: List[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare genuine vs spoofed samples with different attack types

    Args:
        genuine_samples: Genuine samples per modality
        spoofed_samples: Spoofed samples per modality
        attack_types: List of attack type names
        detection_scores: Optional detection confidence scores
        modalities: List of modalities to show
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if modalities is None:
        modalities = list(genuine_samples.keys())

    num_attacks = len(attack_types)

    fig, axes = plt.subplots(num_attacks + 1, len(modalities), figsize=(3 * len(modalities), 3 * (num_attacks + 1)))

    if num_attacks == 0:
        axes = axes.reshape(1, -1)

    # First row: genuine samples
    for col, modality in enumerate(modalities):
        ax = axes[0, col]

        if modality in genuine_samples:
            sample = genuine_samples[modality][0]
            if isinstance(sample, torch.Tensor):
                sample = sample.cpu().numpy()

            if sample.ndim == 3 and sample.shape[0] in [1, 3]:
                sample = np.transpose(sample, (1, 2, 0)).squeeze()

            if sample.ndim == 2:
                ax.imshow(sample, cmap='gray', aspect='auto')

        ax.set_title(modality.capitalize(), fontsize=11, fontweight='bold')

        if col == 0:
            ax.set_ylabel('GENUINE', fontsize=11, fontweight='bold', color='green')

        ax.set_xticks([])
        ax.set_yticks([])

        # Green border
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)

    # Subsequent rows: attack types
    for row, attack_type in enumerate(attack_types, start=1):
        for col, modality in enumerate(modalities):
            ax = axes[row, col]

            if modality in spoofed_samples:
                sample = spoofed_samples[modality][row - 1]
                if isinstance(sample, torch.Tensor):
                    sample = sample.cpu().numpy()

                if sample.ndim == 3 and sample.shape[0] in [1, 3]:
                    sample = np.transpose(sample, (1, 2, 0)).squeeze()

                if sample.ndim == 2:
                    ax.imshow(sample, cmap='gray', aspect='auto')

            if col == 0:
                label = attack_type.replace('_', ' ').title()
                if detection_scores is not None:
                    score = detection_scores[row - 1]
                    label += f'\n(Detect: {score:.2f})'
                ax.set_ylabel(label, fontsize=10, fontweight='bold', color='red')

            ax.set_xticks([])
            ax.set_yticks([])

            # Red border
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

    plt.suptitle('Spoofing Attack Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_augmentation_comparison(
    original_samples: Dict[str, Union[torch.Tensor, np.ndarray]],
    augmented_samples_list: List[Dict[str, Union[torch.Tensor, np.ndarray]]],
    augmentation_names: List[str],
    modalities: List[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare original vs augmented samples

    Args:
        original_samples: Original samples per modality
        augmented_samples_list: List of augmented sample dictionaries
        augmentation_names: Names of augmentations
        modalities: List of modalities to show
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if modalities is None:
        modalities = list(original_samples.keys())

    num_augs = len(augmentation_names)

    fig, axes = plt.subplots(num_augs + 1, len(modalities), figsize=(3 * len(modalities), 3 * (num_augs + 1)))

    if num_augs == 0:
        axes = axes.reshape(1, -1)

    # First row: original samples
    for col, modality in enumerate(modalities):
        ax = axes[0, col]

        if modality in original_samples:
            sample = original_samples[modality][0]
            if isinstance(sample, torch.Tensor):
                sample = sample.cpu().numpy()

            if sample.ndim == 3 and sample.shape[0] in [1, 3]:
                sample = np.transpose(sample, (1, 2, 0)).squeeze()

            if sample.ndim == 2:
                ax.imshow(sample, cmap='gray', aspect='auto')

        ax.set_title(modality.capitalize(), fontsize=11, fontweight='bold')

        if col == 0:
            ax.set_ylabel('ORIGINAL', fontsize=11, fontweight='bold', color='blue')

        ax.set_xticks([])
        ax.set_yticks([])

        # Blue border
        for spine in ax.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(2)

    # Subsequent rows: augmentations
    for row, (aug_samples, aug_name) in enumerate(zip(augmented_samples_list, augmentation_names), start=1):
        for col, modality in enumerate(modalities):
            ax = axes[row, col]

            if modality in aug_samples:
                sample = aug_samples[modality][0]
                if isinstance(sample, torch.Tensor):
                    sample = sample.cpu().numpy()

                if sample.ndim == 3 and sample.shape[0] in [1, 3]:
                    sample = np.transpose(sample, (1, 2, 0)).squeeze()

                if sample.ndim == 2:
                    ax.imshow(sample, cmap='gray', aspect='auto')

            if col == 0:
                ax.set_ylabel(aug_name.replace('_', ' ').title(), fontsize=10, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle('Data Augmentation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_quality_impact_analysis(
    quality_scores: np.ndarray,
    accuracies: np.ndarray,
    modality_name: str = 'All',
    num_bins: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze impact of quality scores on performance

    Args:
        quality_scores: Quality scores for samples
        accuracies: Accuracy for each sample (0 or 1)
        modality_name: Name of modality
        num_bins: Number of quality bins
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Accuracy vs quality (binned)
    ax1 = axes[0]
    bins = np.linspace(quality_scores.min(), quality_scores.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = []

    for i in range(len(bins) - 1):
        mask = (quality_scores >= bins[i]) & (quality_scores < bins[i+1])
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(accuracies[mask]) * 100)
        else:
            bin_accs.append(0)

    ax1.bar(bin_centers, bin_accs, width=(bins[1]-bins[0])*0.8, edgecolor='black', alpha=0.7, color='teal')
    ax1.set_xlabel('Quality Score', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title(f'{modality_name} - Accuracy vs Quality', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Scatter plot
    ax2 = axes[1]
    jitter = np.random.normal(0, 0.01, size=len(accuracies))
    colors = ['green' if a == 1 else 'red' for a in accuracies]
    ax2.scatter(quality_scores, accuracies + jitter, alpha=0.4, s=20, c=colors)
    ax2.set_xlabel('Quality Score', fontsize=11)
    ax2.set_ylabel('Accuracy (jittered)', fontsize=11)
    ax2.set_title(f'{modality_name} - Quality vs Outcome', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(quality_scores, accuracies, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(quality_scores.min(), quality_scores.max(), 100)
    ax2.plot(x_trend, p(x_trend), 'b--', linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax2.legend()

    plt.suptitle('Quality Score Impact Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig
