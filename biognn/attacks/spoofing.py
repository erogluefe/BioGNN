"""
Spoofing attack simulation and detection for biometric systems
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SpoofingType(Enum):
    """Types of spoofing attacks"""
    PRINT_ATTACK = "print_attack"  # Printed photo (face)
    REPLAY_ATTACK = "replay_attack"  # Video replay (face)
    MASK_ATTACK = "mask_attack"  # 3D mask (face)
    SYNTHETIC_FINGERPRINT = "synthetic_fingerprint"
    FAKE_IRIS = "fake_iris"
    VOICE_SYNTHESIS = "voice_synthesis"
    DEEPFAKE = "deepfake"
    ADVERSARIAL = "adversarial"  # Adversarial perturbation


class SpoofingAttackSimulator:
    """
    Simulates various spoofing attacks on biometric data
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def add_noise(
        self,
        data: torch.Tensor,
        noise_level: float = 0.1
    ) -> torch.Tensor:
        """
        Add Gaussian noise to simulate low-quality spoofing

        Args:
            data: Input data
            noise_level: Standard deviation of noise

        Returns:
            Noisy data
        """
        noise = torch.randn_like(data) * noise_level
        return data + noise

    def simulate_print_attack(
        self,
        face_image: torch.Tensor,
        degradation: float = 0.3
    ) -> torch.Tensor:
        """
        Simulate print attack on face images

        Args:
            face_image: Original face image [C, H, W]
            degradation: Amount of degradation (0-1)

        Returns:
            Spoofed face image
        """
        # Reduce color saturation
        if face_image.size(0) == 3:
            # Convert to grayscale and mix
            gray = face_image.mean(dim=0, keepdim=True)
            spoofed = face_image * (1 - degradation) + gray.expand_as(face_image) * degradation
        else:
            spoofed = face_image.clone()

        # Add noise (print artifacts)
        spoofed = self.add_noise(spoofed, noise_level=0.05)

        # Reduce contrast slightly
        mean_val = spoofed.mean()
        spoofed = (spoofed - mean_val) * (1 - degradation * 0.5) + mean_val

        return spoofed

    def simulate_replay_attack(
        self,
        face_image: torch.Tensor,
        blur_kernel: int = 3
    ) -> torch.Tensor:
        """
        Simulate replay attack (video replay)

        Args:
            face_image: Original face image
            blur_kernel: Blur kernel size

        Returns:
            Spoofed face image
        """
        # Apply slight blur (screen effect)
        if blur_kernel > 1:
            padding = blur_kernel // 2
            kernel = torch.ones(1, 1, blur_kernel, blur_kernel) / (blur_kernel ** 2)
            kernel = kernel.to(face_image.device)

            spoofed = []
            for c in range(face_image.size(0)):
                channel = face_image[c:c+1].unsqueeze(0)
                blurred = nn.functional.conv2d(channel, kernel, padding=padding)
                spoofed.append(blurred.squeeze(0))
            spoofed = torch.cat(spoofed, dim=0)
        else:
            spoofed = face_image.clone()

        # Add screen reflection noise
        spoofed = self.add_noise(spoofed, noise_level=0.03)

        return spoofed

    def simulate_mask_attack(
        self,
        face_image: torch.Tensor,
        mask_quality: float = 0.8
    ) -> torch.Tensor:
        """
        Simulate 3D mask attack

        Args:
            face_image: Original face image
            mask_quality: Quality of mask (0-1, higher is better)

        Returns:
            Spoofed face image
        """
        # High-quality masks are harder to detect
        # Lower quality: more artifacts

        # Reduce texture details
        degradation = 1 - mask_quality
        spoofed = self.simulate_print_attack(face_image, degradation)

        # Add subtle geometric distortions
        # (In practice, this would require more sophisticated image warping)
        spoofed = self.add_noise(spoofed, noise_level=0.02 * degradation)

        return spoofed

    def simulate_synthetic_fingerprint(
        self,
        fingerprint: torch.Tensor,
        method: str = 'basic'
    ) -> torch.Tensor:
        """
        Simulate synthetic fingerprint attack

        Args:
            fingerprint: Original fingerprint image
            method: Synthesis method ('basic', 'advanced')

        Returns:
            Spoofed fingerprint
        """
        spoofed = fingerprint.clone()

        if method == 'basic':
            # Basic: add noise and slight blur
            spoofed = self.add_noise(spoofed, noise_level=0.08)

            # Slight blur
            kernel_size = 3
            padding = kernel_size // 2
            kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
            kernel = kernel.to(spoofed.device)

            spoofed = spoofed.unsqueeze(0).unsqueeze(0)
            spoofed = nn.functional.conv2d(spoofed, kernel, padding=padding)
            spoofed = spoofed.squeeze(0).squeeze(0)

        elif method == 'advanced':
            # Advanced: minimal degradation (harder to detect)
            spoofed = self.add_noise(spoofed, noise_level=0.02)

        return spoofed

    def simulate_voice_synthesis(
        self,
        voice_features: torch.Tensor,
        synthesis_quality: float = 0.9
    ) -> torch.Tensor:
        """
        Simulate voice synthesis/replay attack

        Args:
            voice_features: Original voice features (e.g., MFCC)
            synthesis_quality: Quality of synthesis (0-1)

        Returns:
            Spoofed voice features
        """
        # Add artifacts typical of synthesized voice
        degradation = 1 - synthesis_quality

        spoofed = voice_features.clone()

        # Add noise
        spoofed = self.add_noise(spoofed, noise_level=0.05 * degradation)

        # Modify certain frequency bands (artifacts from vocoder)
        if torch.rand(1).item() < 0.5:
            # Randomly attenuate some frequency bins
            mask = torch.rand_like(spoofed) > (0.1 * degradation)
            spoofed = spoofed * mask.float()

        return spoofed

    def simulate_adversarial_attack(
        self,
        data: torch.Tensor,
        epsilon: float = 0.01,
        targeted: bool = False
    ) -> torch.Tensor:
        """
        Simulate adversarial perturbation attack

        Args:
            data: Input data
            epsilon: Perturbation magnitude
            targeted: Whether attack is targeted

        Returns:
            Adversarially perturbed data
        """
        # Simple FGSM-like perturbation
        perturbation = torch.randn_like(data) * epsilon

        if targeted:
            # Targeted: add perturbation in specific direction
            spoofed = data + perturbation
        else:
            # Untargeted: random perturbation
            spoofed = data + torch.sign(perturbation) * epsilon

        # Clip to valid range
        spoofed = torch.clamp(spoofed, data.min(), data.max())

        return spoofed


class SpoofingDetector(nn.Module):
    """
    Liveness/spoofing detection module

    Can be integrated with the main biometric system
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2  # genuine vs spoofed
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes (2 for binary detection)
        """
        super().__init__()

        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features

        Returns:
            logits: Classification logits [batch, num_classes]
        """
        return self.detector(features)

    def is_genuine(self, features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Determine if samples are genuine

        Args:
            features: Input features
            threshold: Decision threshold

        Returns:
            Binary predictions (1 for genuine, 0 for spoofed)
        """
        with torch.no_grad():
            logits = self.forward(features)
            probs = torch.softmax(logits, dim=1)
            # Class 1 is genuine
            is_genuine = (probs[:, 1] > threshold).long()
        return is_genuine


class RobustnessEvaluator:
    """
    Evaluates system robustness against spoofing attacks
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Biometric authentication model to evaluate
        """
        self.model = model
        self.simulator = SpoofingAttackSimulator()

    def evaluate_attack_robustness(
        self,
        genuine_data: Dict[str, torch.Tensor],
        attack_types: List[SpoofingType],
        num_trials: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model robustness against various attacks

        Args:
            genuine_data: Genuine biometric data
            attack_types: List of attack types to test
            num_trials: Number of attack trials per type

        Returns:
            Dictionary of attack success rates and metrics
        """
        results = {}

        self.model.eval()

        for attack_type in attack_types:
            attack_name = attack_type.value
            success_count = 0
            confidence_scores = []

            for _ in range(num_trials):
                # Generate spoofed sample
                spoofed_data = self._generate_spoofed_sample(
                    genuine_data, attack_type
                )

                # Test if attack succeeds
                with torch.no_grad():
                    logits, _ = self.model(spoofed_data, extract_features=True)
                    probs = torch.softmax(logits, dim=1)

                    # Check if spoofed sample is accepted as genuine
                    pred = torch.argmax(probs, dim=1)
                    confidence = probs[0, pred].item()

                    if pred.item() == 1:  # Accepted as genuine
                        success_count += 1

                    confidence_scores.append(confidence)

            # Compute metrics
            success_rate = success_count / num_trials
            avg_confidence = np.mean(confidence_scores)
            std_confidence = np.std(confidence_scores)

            results[attack_name] = {
                'success_rate': success_rate,
                'avg_confidence': avg_confidence,
                'std_confidence': std_confidence,
                'num_trials': num_trials
            }

        return results

    def _generate_spoofed_sample(
        self,
        genuine_data: Dict[str, torch.Tensor],
        attack_type: SpoofingType
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a spoofed sample based on attack type

        Args:
            genuine_data: Original genuine data
            attack_type: Type of spoofing attack

        Returns:
            Spoofed data
        """
        spoofed_data = {}

        for modality, data in genuine_data.items():
            if attack_type == SpoofingType.PRINT_ATTACK and modality == 'face':
                spoofed_data[modality] = self.simulator.simulate_print_attack(data)

            elif attack_type == SpoofingType.REPLAY_ATTACK and modality == 'face':
                spoofed_data[modality] = self.simulator.simulate_replay_attack(data)

            elif attack_type == SpoofingType.MASK_ATTACK and modality == 'face':
                spoofed_data[modality] = self.simulator.simulate_mask_attack(data)

            elif attack_type == SpoofingType.SYNTHETIC_FINGERPRINT and modality == 'fingerprint':
                spoofed_data[modality] = self.simulator.simulate_synthetic_fingerprint(data)

            elif attack_type == SpoofingType.VOICE_SYNTHESIS and modality == 'voice':
                spoofed_data[modality] = self.simulator.simulate_voice_synthesis(data)

            elif attack_type == SpoofingType.ADVERSARIAL:
                spoofed_data[modality] = self.simulator.simulate_adversarial_attack(data)

            else:
                # No specific attack for this modality, use original
                spoofed_data[modality] = data.clone()

        return spoofed_data

    def print_robustness_report(self, results: Dict[str, Dict[str, float]]):
        """
        Print formatted robustness evaluation report

        Args:
            results: Results from evaluate_attack_robustness
        """
        print("\n" + "="*70)
        print("SPOOFING ATTACK ROBUSTNESS EVALUATION")
        print("="*70)

        for attack_type, metrics in results.items():
            print(f"\n{attack_type.upper().replace('_', ' ')}:")
            print(f"  Attack Success Rate:    {metrics['success_rate']:.2%}")
            print(f"  Avg Confidence:         {metrics['avg_confidence']:.4f}")
            print(f"  Std Confidence:         {metrics['std_confidence']:.4f}")
            print(f"  Number of Trials:       {metrics['num_trials']}")

        # Overall robustness score
        avg_success_rate = np.mean([m['success_rate'] for m in results.values()])
        robustness_score = 1 - avg_success_rate

        print(f"\n{'='*70}")
        print(f"OVERALL ROBUSTNESS SCORE: {robustness_score:.2%}")
        print(f"  (Higher is better - indicates resistance to attacks)")
        print("="*70 + "\n")
