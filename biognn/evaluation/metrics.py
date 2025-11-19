"""
Evaluation metrics for biometric authentication systems
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve
)
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def compute_far_frr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Tuple[float, float]:
    """
    Compute False Accept Rate (FAR) and False Reject Rate (FRR)

    Args:
        y_true: True labels (1 for genuine, 0 for impostor)
        y_scores: Prediction scores
        threshold: Decision threshold

    Returns:
        FAR: False Accept Rate
        FRR: False Reject Rate
    """
    predictions = (y_scores >= threshold).astype(int)

    # True Positives: genuine accepted
    # False Positives: impostor accepted (False Accept)
    # True Negatives: impostor rejected
    # False Negatives: genuine rejected (False Reject)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    # FAR: proportion of impostor attempts that are incorrectly accepted
    if (fp + tn) > 0:
        far = fp / (fp + tn)
    else:
        far = 0.0

    # FRR: proportion of genuine attempts that are incorrectly rejected
    if (fn + tp) > 0:
        frr = fn / (fn + tp)
    else:
        frr = 0.0

    return far, frr


def compute_eer(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and corresponding threshold

    EER is the point where FAR = FRR

    Args:
        y_true: True labels (1 for genuine, 0 for impostor)
        y_scores: Prediction scores

    Returns:
        eer: Equal Error Rate
        threshold: Threshold at EER
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    # FRR = 1 - TPR (False Negative Rate)
    frr = 1 - tpr

    # FAR = FPR (False Positive Rate)
    far = fpr

    # Find point where FAR = FRR
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)

    eer = (far[min_index] + frr[min_index]) / 2
    eer_threshold = thresholds[min_index]

    return eer, eer_threshold


def compute_far_at_frr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_frr: float = 0.01
) -> Tuple[float, float]:
    """
    Compute FAR at a specific FRR

    Args:
        y_true: True labels
        y_scores: Prediction scores
        target_frr: Target FRR (e.g., 0.01 for 1% FRR)

    Returns:
        far: FAR at target FRR
        threshold: Corresponding threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    frr = 1 - tpr

    # Find closest FRR to target
    idx = np.argmin(np.abs(frr - target_frr))

    return fpr[idx], thresholds[idx]


def compute_frr_at_far(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_far: float = 0.01
) -> Tuple[float, float]:
    """
    Compute FRR at a specific FAR

    Args:
        y_true: True labels
        y_scores: Prediction scores
        target_far: Target FAR (e.g., 0.001 for 0.1% FAR)

    Returns:
        frr: FRR at target FAR
        threshold: Corresponding threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    frr = 1 - tpr

    # Find closest FAR to target
    idx = np.argmin(np.abs(fpr - target_far))

    return frr[idx], thresholds[idx]


def compute_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> float:
    """
    Compute Area Under ROC Curve (AUC)

    Args:
        y_true: True labels
        y_scores: Prediction scores

    Returns:
        auc_score: AUC value
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    auc_score = auc(fpr, tpr)
    return auc_score


def compute_det_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Detection Error Tradeoff (DET) curve

    Args:
        y_true: True labels
        y_scores: Prediction scores

    Returns:
        far: False Accept Rates
        frr: False Reject Rates
        thresholds: Thresholds
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    far = fpr
    frr = 1 - tpr

    return far, frr, thresholds


class BiometricEvaluator:
    """
    Comprehensive evaluator for biometric authentication systems
    """

    def __init__(self):
        self.results = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics

        Args:
            y_true: True labels (1 for genuine, 0 for impostor)
            y_scores: Prediction scores or probabilities
            y_pred: Optional predictions (if not provided, computed from scores)
            threshold: Optional threshold (if not provided, uses EER threshold)

        Returns:
            Dictionary of metrics
        """
        # Ensure numpy arrays
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_scores):
            y_scores = y_scores.cpu().numpy()
        if y_pred is not None and torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()

        # Compute EER
        eer, eer_threshold = compute_eer(y_true, y_scores)

        # Use EER threshold if none provided
        if threshold is None:
            threshold = eer_threshold

        # Compute FAR and FRR at threshold
        far, frr = compute_far_frr(y_true, y_scores, threshold)

        # Compute predictions if not provided
        if y_pred is None:
            y_pred = (y_scores >= threshold).astype(int)

        # Compute standard classification metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Compute AUC
        auc_score = compute_roc_auc(y_true, y_scores)

        # Compute FAR at different FRR values
        far_at_1_frr, _ = compute_far_at_frr(y_true, y_scores, target_frr=0.01)
        far_at_01_frr, _ = compute_far_at_frr(y_true, y_scores, target_frr=0.001)

        # Compute FRR at different FAR values
        frr_at_1_far, _ = compute_frr_at_far(y_true, y_scores, target_far=0.01)
        frr_at_01_far, _ = compute_frr_at_far(y_true, y_scores, target_far=0.001)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Genuine Accept Rate (GAR) = TPR
        gar = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Compute precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results = {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'threshold': threshold,
            'far': far,
            'frr': frr,
            'gar': gar,
            'accuracy': accuracy,
            'auc': auc_score,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'far_at_1%_frr': far_at_1_frr,
            'far_at_0.1%_frr': far_at_01_frr,
            'frr_at_1%_far': frr_at_1_far,
            'frr_at_0.1%_far': frr_at_01_far,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
        }

        self.results = results
        return results

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve

        Args:
            y_true: True labels
            y_scores: Prediction scores
            save_path: Optional path to save figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        auc_score = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate (FAR)', fontsize=12)
        plt.ylabel('True Positive Rate (GAR)', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_det_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot DET (Detection Error Tradeoff) curve

        Args:
            y_true: True labels
            y_scores: Prediction scores
            save_path: Optional path to save figure
        """
        far, frr, _ = compute_det_curve(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(far, frr, 'b-', linewidth=2, label='DET Curve')

        # Mark EER point
        eer, _ = compute_eer(y_true, y_scores)
        plt.plot([eer], [eer], 'ro', markersize=8, label=f'EER = {eer:.4f}')

        plt.xlabel('False Accept Rate (FAR)', fontsize=12)
        plt.ylabel('False Reject Rate (FRR)', fontsize=12)
        plt.title('Detection Error Tradeoff (DET) Curve', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Impostor', 'Genuine'],
            yticklabels=['Impostor', 'Genuine'],
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_score_distribution(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot score distribution for genuine vs impostor

        Args:
            y_true: True labels
            y_scores: Prediction scores
            save_path: Optional path to save figure
        """
        genuine_scores = y_scores[y_true == 1]
        impostor_scores = y_scores[y_true == 0]

        plt.figure(figsize=(10, 6))
        plt.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', color='red')
        plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', color='blue')

        # Mark EER threshold
        eer, eer_threshold = compute_eer(y_true, y_scores)
        plt.axvline(eer_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'EER Threshold = {eer_threshold:.4f}')

        plt.xlabel('Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Score Distribution: Genuine vs Impostor', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def print_summary(self):
        """Print summary of evaluation results"""
        if not self.results:
            print("No evaluation results available. Run evaluate() first.")
            return

        print("\n" + "="*60)
        print("BIOMETRIC AUTHENTICATION EVALUATION RESULTS")
        print("="*60)
        print(f"\nKey Metrics:")
        print(f"  Equal Error Rate (EER):        {self.results['eer']:.4f} ({self.results['eer']*100:.2f}%)")
        print(f"  EER Threshold:                 {self.results['eer_threshold']:.4f}")
        print(f"  AUC:                          {self.results['auc']:.4f}")
        print(f"  Accuracy:                     {self.results['accuracy']:.4f} ({self.results['accuracy']*100:.2f}%)")
        print(f"\nAt Decision Threshold ({self.results['threshold']:.4f}):")
        print(f"  False Accept Rate (FAR):      {self.results['far']:.4f} ({self.results['far']*100:.2f}%)")
        print(f"  False Reject Rate (FRR):      {self.results['frr']:.4f} ({self.results['frr']*100:.2f}%)")
        print(f"  Genuine Accept Rate (GAR):    {self.results['gar']:.4f} ({self.results['gar']*100:.2f}%)")
        print(f"\nClassification Metrics:")
        print(f"  Precision:                    {self.results['precision']:.4f}")
        print(f"  Recall:                       {self.results['recall']:.4f}")
        print(f"  F1 Score:                     {self.results['f1']:.4f}")
        print(f"\nOperating Points:")
        print(f"  FAR at 1% FRR:                {self.results['far_at_1%_frr']:.4f}")
        print(f"  FAR at 0.1% FRR:              {self.results['far_at_0.1%_frr']:.4f}")
        print(f"  FRR at 1% FAR:                {self.results['frr_at_1%_far']:.4f}")
        print(f"  FRR at 0.1% FAR:              {self.results['frr_at_0.1%_far']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:               {self.results['true_positives']}")
        print(f"  False Positives:              {self.results['false_positives']}")
        print(f"  True Negatives:               {self.results['true_negatives']}")
        print(f"  False Negatives:              {self.results['false_negatives']}")
        print("="*60 + "\n")
