"""
Biometric evaluation metrics

Provides comprehensive metrics for biometric verification and identification:
- EER (Equal Error Rate)
- FAR (False Accept Rate)
- FRR (False Reject Rate)
- ROC (Receiver Operating Characteristic)
- DET (Detection Error Tradeoff)
- CMC (Cumulative Match Characteristic)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    num_thresholds: int = 1000
) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER) and optimal threshold

    EER is where FAR = FRR

    Args:
        genuine_scores: Scores for genuine pairs (higher = more similar)
        impostor_scores: Scores for impostor pairs
        num_thresholds: Number of thresholds to evaluate

    Returns:
        (eer, optimal_threshold)
    """
    # Combine scores and create labels
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])

    # Calculate thresholds
    min_score = scores.min()
    max_score = scores.max()
    thresholds = np.linspace(min_score, max_score, num_thresholds)

    far_list = []
    frr_list = []

    for threshold in thresholds:
        # Predictions: score >= threshold → genuine
        predictions = (scores >= threshold).astype(int)

        # False Accept: impostor accepted (predicted 1, actual 0)
        false_accepts = np.sum((predictions == 1) & (labels == 0))
        total_impostors = np.sum(labels == 0)
        far = false_accepts / max(total_impostors, 1)

        # False Reject: genuine rejected (predicted 0, actual 1)
        false_rejects = np.sum((predictions == 0) & (labels == 1))
        total_genuines = np.sum(labels == 1)
        frr = false_rejects / max(total_genuines, 1)

        far_list.append(far)
        frr_list.append(frr)

    far_array = np.array(far_list)
    frr_array = np.array(frr_list)

    # EER is where FAR ≈ FRR
    eer_idx = np.argmin(np.abs(far_array - frr_array))
    eer = (far_array[eer_idx] + frr_array[eer_idx]) / 2
    optimal_threshold = thresholds[eer_idx]

    return eer, optimal_threshold


def calculate_far_frr_at_threshold(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: float
) -> Tuple[float, float]:
    """
    Calculate FAR and FRR at a specific threshold

    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        threshold: Decision threshold

    Returns:
        (far, frr)
    """
    # FAR: fraction of impostors accepted
    far = np.mean(impostor_scores >= threshold)

    # FRR: fraction of genuines rejected
    frr = np.mean(genuine_scores < threshold)

    return far, frr


def calculate_roc_auc(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate ROC curve and AUC

    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs

    Returns:
        (fpr, tpr, auc_score)
    """
    # Combine scores and create labels
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score


def calculate_det_curve(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    num_thresholds: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate DET (Detection Error Tradeoff) curve

    DET plots FAR vs FRR on a normal deviate scale

    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        num_thresholds: Number of thresholds

    Returns:
        (far_array, frr_array)
    """
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])

    min_score = scores.min()
    max_score = scores.max()
    thresholds = np.linspace(min_score, max_score, num_thresholds)

    far_list = []
    frr_list = []

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        false_accepts = np.sum((predictions == 1) & (labels == 0))
        total_impostors = np.sum(labels == 0)
        far = false_accepts / max(total_impostors, 1)

        false_rejects = np.sum((predictions == 0) & (labels == 1))
        total_genuines = np.sum(labels == 1)
        frr = false_rejects / max(total_genuines, 1)

        far_list.append(far)
        frr_list.append(frr)

    return np.array(far_list), np.array(frr_list)


def calculate_verification_metrics(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive verification metrics

    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        threshold: Optional decision threshold (if None, uses EER threshold)

    Returns:
        Dictionary of metrics
    """
    # Calculate EER
    eer, optimal_threshold = calculate_eer(genuine_scores, impostor_scores)

    # Use provided threshold or optimal
    if threshold is None:
        threshold = optimal_threshold

    # Calculate FAR and FRR at threshold
    far, frr = calculate_far_frr_at_threshold(
        genuine_scores, impostor_scores, threshold
    )

    # Calculate ROC AUC
    _, _, auc_score = calculate_roc_auc(genuine_scores, impostor_scores)

    # Calculate accuracy
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    predictions = (scores >= threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'eer': eer,
        'eer_threshold': optimal_threshold,
        'far': far,
        'frr': frr,
        'auc': auc_score,
        'accuracy': accuracy,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_genuine': len(genuine_scores),
        'total_impostor': len(impostor_scores)
    }


def calculate_cmc_curve(
    probe_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    probe_labels: np.ndarray,
    gallery_labels: np.ndarray,
    max_rank: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate CMC (Cumulative Match Characteristic) curve for identification

    Args:
        probe_embeddings: Feature embeddings for probe samples [N, D]
        gallery_embeddings: Feature embeddings for gallery samples [M, D]
        probe_labels: Subject IDs for probe samples [N]
        gallery_labels: Subject IDs for gallery samples [M]
        max_rank: Maximum rank to compute

    Returns:
        (ranks, recognition_rates)
    """
    num_probes = len(probe_embeddings)
    ranks = np.arange(1, max_rank + 1)
    correct_at_rank = np.zeros(max_rank)

    for i, probe_emb in enumerate(probe_embeddings):
        probe_label = probe_labels[i]

        # Calculate similarity scores (cosine similarity)
        similarities = np.dot(gallery_embeddings, probe_emb)
        similarities /= (np.linalg.norm(gallery_embeddings, axis=1) * np.linalg.norm(probe_emb) + 1e-8)

        # Sort gallery by similarity (descending)
        sorted_indices = np.argsort(-similarities)
        sorted_labels = gallery_labels[sorted_indices]

        # Check if correct match is in top-k
        for rank in range(max_rank):
            if probe_label in sorted_labels[:rank + 1]:
                correct_at_rank[rank] += 1

    # Calculate cumulative recognition rates
    recognition_rates = correct_at_rank / num_probes

    return ranks, recognition_rates


def plot_roc_curve(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve"
) -> plt.Figure:
    """
    Plot ROC curve

    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fpr, tpr, auc_score = calculate_roc_auc(genuine_scores, impostor_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate (FAR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (1 - FRR)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_det_curve(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "DET Curve"
) -> plt.Figure:
    """
    Plot DET curve

    Args:
        genuine_scores: Scores for genuine pairs
        impostor_scores: Scores for impostor pairs
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    far, frr = calculate_det_curve(genuine_scores, impostor_scores)

    # Calculate EER point
    eer, _ = calculate_eer(genuine_scores, impostor_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(far * 100, frr * 100, 'b-', linewidth=2)
    ax.plot([eer * 100], [eer * 100], 'ro', markersize=10, label=f'EER = {eer*100:.2f}%')
    ax.set_xlabel('False Accept Rate (%)', fontsize=12)
    ax.set_ylabel('False Reject Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_cmc_curve(
    ranks: np.ndarray,
    recognition_rates: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "CMC Curve"
) -> plt.Figure:
    """
    Plot CMC curve

    Args:
        ranks: Rank values [1, 2, 3, ...]
        recognition_rates: Recognition rate at each rank
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ranks, recognition_rates * 100, 'b-', linewidth=2, marker='o')
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Recognition Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, ranks[-1]])
    ax.set_ylim([0, 105])

    # Add rank-1 accuracy annotation
    rank1_acc = recognition_rates[0] * 100
    ax.axhline(y=rank1_acc, color='r', linestyle='--', alpha=0.5)
    ax.text(ranks[-1] * 0.6, rank1_acc + 5, f'Rank-1: {rank1_acc:.1f}%',
            fontsize=10, color='r')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """
    Print formatted metrics summary

    Args:
        metrics: Dictionary of metrics from calculate_verification_metrics
    """
    print("\n" + "="*60)
    print("VERIFICATION METRICS SUMMARY")
    print("="*60)

    print(f"\n📊 Error Rates:")
    print(f"  EER (Equal Error Rate):        {metrics['eer']*100:6.2f}%")
    print(f"  FAR (False Accept Rate):       {metrics['far']*100:6.2f}%")
    print(f"  FRR (False Reject Rate):       {metrics['frr']*100:6.2f}%")

    print(f"\n🎯 Classification Metrics:")
    print(f"  Accuracy:                      {metrics['accuracy']*100:6.2f}%")
    print(f"  Precision:                     {metrics['precision']*100:6.2f}%")
    print(f"  Recall:                        {metrics['recall']*100:6.2f}%")
    print(f"  F1 Score:                      {metrics['f1_score']*100:6.2f}%")
    print(f"  AUC:                           {metrics['auc']:6.4f}")

    print(f"\n🔢 Confusion Matrix:")
    print(f"  True Positives:                {metrics['true_positives']:6d}")
    print(f"  True Negatives:                {metrics['true_negatives']:6d}")
    print(f"  False Positives:               {metrics['false_positives']:6d}")
    print(f"  False Negatives:               {metrics['false_negatives']:6d}")

    print(f"\n⚙️  Thresholds:")
    print(f"  EER Threshold:                 {metrics['eer_threshold']:6.4f}")
    print(f"  Operating Threshold:           {metrics['threshold']:6.4f}")

    print(f"\n📈 Dataset Info:")
    print(f"  Total Genuine Pairs:           {metrics['total_genuine']:6d}")
    print(f"  Total Impostor Pairs:          {metrics['total_impostor']:6d}")

    print("="*60 + "\n")
