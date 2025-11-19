"""
Adaptive threshold mechanisms for biometric authentication

Provides user-specific and modality-specific adaptive thresholding
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import pickle
from pathlib import Path


class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds for biometric verification

    Supports:
    - User-specific thresholds
    - Modality-specific thresholds
    - Dynamic threshold updates based on user history
    """

    def __init__(
        self,
        initial_threshold: float = 0.5,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        window_size: int = 50
    ):
        """
        Args:
            initial_threshold: Initial threshold value
            adaptation_rate: Learning rate for threshold updates
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            window_size: Number of recent samples to consider
        """
        self.initial_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.window_size = window_size

        # User-specific thresholds
        self.user_thresholds = defaultdict(lambda: initial_threshold)

        # Modality-specific thresholds
        self.modality_thresholds = defaultdict(lambda: initial_threshold)

        # History for each user
        self.user_history = defaultdict(list)

        # Global statistics
        self.global_threshold = initial_threshold
        self.global_far = 0.0
        self.global_frr = 0.0

    def get_threshold(
        self,
        user_id: Optional[int] = None,
        modality: Optional[str] = None,
        use_adaptive: bool = True
    ) -> float:
        """
        Get threshold for verification

        Args:
            user_id: User identifier (None for global)
            modality: Modality name (None for all modalities)
            use_adaptive: Whether to use adaptive threshold

        Returns:
            threshold: Threshold value
        """
        if not use_adaptive:
            return self.initial_threshold

        if user_id is not None:
            # User-specific threshold
            return self.user_thresholds[user_id]
        elif modality is not None:
            # Modality-specific threshold
            return self.modality_thresholds[modality]
        else:
            # Global threshold
            return self.global_threshold

    def update_threshold(
        self,
        score: float,
        is_genuine: bool,
        user_id: Optional[int] = None,
        modality: Optional[str] = None
    ):
        """
        Update threshold based on verification result

        Args:
            score: Verification score
            is_genuine: Whether the attempt was genuine
            user_id: User identifier
            modality: Modality name
        """
        # Update user history
        if user_id is not None:
            self.user_history[user_id].append((score, is_genuine))

            # Keep only recent history
            if len(self.user_history[user_id]) > self.window_size:
                self.user_history[user_id] = self.user_history[user_id][-self.window_size:]

            # Recompute user threshold
            self._update_user_threshold(user_id)

        # Update modality threshold
        if modality is not None:
            self._update_modality_threshold(modality, score, is_genuine)

        # Update global threshold
        self._update_global_threshold(score, is_genuine)

    def _update_user_threshold(self, user_id: int):
        """
        Update user-specific threshold based on history

        Uses EER-based optimization on user's historical data
        """
        history = self.user_history[user_id]
        if len(history) < 10:  # Need minimum samples
            return

        genuine_scores = [s for s, g in history if g]
        impostor_scores = [s for s, g in history if not g]

        if not genuine_scores or not impostor_scores:
            return

        # Find threshold that minimizes error
        all_scores = sorted(set(genuine_scores + impostor_scores))

        best_threshold = self.user_thresholds[user_id]
        min_error = float('inf')

        for threshold in all_scores:
            # Compute FAR and FRR
            far = sum(1 for s in impostor_scores if s >= threshold) / len(impostor_scores)
            frr = sum(1 for s in genuine_scores if s < threshold) / len(genuine_scores)
            error = abs(far - frr)  # Minimize difference (EER)

            if error < min_error:
                min_error = error
                best_threshold = threshold

        # Update with learning rate
        current = self.user_thresholds[user_id]
        new_threshold = (1 - self.adaptation_rate) * current + self.adaptation_rate * best_threshold

        # Clip to valid range
        self.user_thresholds[user_id] = np.clip(
            new_threshold,
            self.min_threshold,
            self.max_threshold
        )

    def _update_modality_threshold(
        self,
        modality: str,
        score: float,
        is_genuine: bool
    ):
        """
        Update modality-specific threshold

        Uses exponential moving average
        """
        current = self.modality_thresholds[modality]

        # Simple rule: if error occurs, adjust threshold
        if is_genuine and score < current:
            # False rejection: lower threshold
            new_threshold = current - self.adaptation_rate * 0.1
        elif not is_genuine and score >= current:
            # False acceptance: raise threshold
            new_threshold = current + self.adaptation_rate * 0.1
        else:
            # Correct decision: no change
            new_threshold = current

        self.modality_thresholds[modality] = np.clip(
            new_threshold,
            self.min_threshold,
            self.max_threshold
        )

    def _update_global_threshold(self, score: float, is_genuine: bool):
        """Update global threshold"""
        # Similar to modality threshold update
        if is_genuine and score < self.global_threshold:
            self.global_threshold -= self.adaptation_rate * 0.05
        elif not is_genuine and score >= self.global_threshold:
            self.global_threshold += self.adaptation_rate * 0.05

        self.global_threshold = np.clip(
            self.global_threshold,
            self.min_threshold,
            self.max_threshold
        )

    def verify(
        self,
        score: float,
        user_id: Optional[int] = None,
        modality: Optional[str] = None
    ) -> bool:
        """
        Make verification decision using adaptive threshold

        Args:
            score: Verification score
            user_id: User identifier
            modality: Modality name

        Returns:
            decision: True if accepted, False if rejected
        """
        threshold = self.get_threshold(user_id, modality)
        return score >= threshold

    def save(self, path: str):
        """Save threshold manager state"""
        state = {
            'user_thresholds': dict(self.user_thresholds),
            'modality_thresholds': dict(self.modality_thresholds),
            'user_history': dict(self.user_history),
            'global_threshold': self.global_threshold,
            'global_far': self.global_far,
            'global_frr': self.global_frr,
            'config': {
                'initial_threshold': self.initial_threshold,
                'adaptation_rate': self.adaptation_rate,
                'min_threshold': self.min_threshold,
                'max_threshold': self.max_threshold,
                'window_size': self.window_size
            }
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """Load threshold manager state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.user_thresholds = defaultdict(lambda: self.initial_threshold, state['user_thresholds'])
        self.modality_thresholds = defaultdict(lambda: self.initial_threshold, state['modality_thresholds'])
        self.user_history = defaultdict(list, state['user_history'])
        self.global_threshold = state['global_threshold']
        self.global_far = state['global_far']
        self.global_frr = state['global_frr']

        config = state['config']
        self.initial_threshold = config['initial_threshold']
        self.adaptation_rate = config['adaptation_rate']
        self.min_threshold = config['min_threshold']
        self.max_threshold = config['max_threshold']
        self.window_size = config['window_size']

    def get_statistics(self) -> Dict[str, any]:
        """Get current statistics"""
        return {
            'global_threshold': self.global_threshold,
            'num_users': len(self.user_thresholds),
            'num_modalities': len(self.modality_thresholds),
            'user_thresholds': dict(self.user_thresholds),
            'modality_thresholds': dict(self.modality_thresholds),
            'avg_user_threshold': np.mean(list(self.user_thresholds.values())) if self.user_thresholds else self.global_threshold,
        }


class QualityBasedThreshold:
    """
    Adjusts threshold based on biometric sample quality

    High quality → lower threshold (more lenient)
    Low quality → higher threshold (more strict)
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        quality_weight: float = 0.2,
        min_quality: float = 0.0,
        max_quality: float = 1.0
    ):
        """
        Args:
            base_threshold: Base threshold value
            quality_weight: Weight for quality adjustment
            min_quality: Minimum quality score
            max_quality: Maximum quality score
        """
        self.base_threshold = base_threshold
        self.quality_weight = quality_weight
        self.min_quality = min_quality
        self.max_quality = max_quality

    def get_threshold(
        self,
        quality_scores: Dict[str, float]
    ) -> float:
        """
        Compute quality-adjusted threshold

        Args:
            quality_scores: Dictionary of quality scores per modality

        Returns:
            threshold: Adjusted threshold
        """
        if not quality_scores:
            return self.base_threshold

        # Average quality across modalities
        avg_quality = np.mean(list(quality_scores.values()))

        # Normalize quality to [0, 1]
        normalized_quality = (avg_quality - self.min_quality) / (self.max_quality - self.min_quality)
        normalized_quality = np.clip(normalized_quality, 0, 1)

        # High quality → lower threshold
        # Low quality → higher threshold
        adjustment = self.quality_weight * (1 - normalized_quality)
        threshold = self.base_threshold + adjustment

        return np.clip(threshold, 0.1, 0.9)


class MultiModalAdaptiveThreshold:
    """
    Adaptive threshold that considers all modalities

    Adjusts based on:
    - Per-modality confidence
    - Quality scores
    - Historical performance
    """

    def __init__(
        self,
        modalities: List[str],
        base_threshold: float = 0.5
    ):
        self.modalities = modalities
        self.base_threshold = base_threshold

        # Per-modality managers
        self.modality_managers = {
            mod: AdaptiveThresholdManager(initial_threshold=base_threshold)
            for mod in modalities
        }

        # Quality-based threshold
        self.quality_threshold = QualityBasedThreshold(base_threshold=base_threshold)

    def get_threshold(
        self,
        modality_scores: Dict[str, float],
        quality_scores: Optional[Dict[str, float]] = None,
        user_id: Optional[int] = None
    ) -> float:
        """
        Get adaptive threshold considering all factors

        Args:
            modality_scores: Verification scores per modality
            quality_scores: Quality scores per modality
            user_id: User identifier

        Returns:
            threshold: Final adaptive threshold
        """
        # Get per-modality thresholds
        modality_thresholds = []
        for modality, score in modality_scores.items():
            if modality in self.modality_managers:
                threshold = self.modality_managers[modality].get_threshold(
                    user_id=user_id,
                    modality=modality
                )
                modality_thresholds.append(threshold)

        # Average modality thresholds
        if modality_thresholds:
            avg_threshold = np.mean(modality_thresholds)
        else:
            avg_threshold = self.base_threshold

        # Adjust for quality if available
        if quality_scores:
            quality_adjustment = self.quality_threshold.get_threshold(quality_scores)
            final_threshold = 0.7 * avg_threshold + 0.3 * quality_adjustment
        else:
            final_threshold = avg_threshold

        return final_threshold

    def update(
        self,
        modality_scores: Dict[str, float],
        is_genuine: bool,
        user_id: Optional[int] = None
    ):
        """Update all modality managers"""
        for modality, score in modality_scores.items():
            if modality in self.modality_managers:
                self.modality_managers[modality].update_threshold(
                    score=score,
                    is_genuine=is_genuine,
                    user_id=user_id,
                    modality=modality
                )
