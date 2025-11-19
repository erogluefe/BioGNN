"""
Multi-task learning model for biometric authentication

Combines:
1. Main task: Identity verification
2. Auxiliary task 1: Modality quality estimation
3. Auxiliary task 2: Spoofing detection (liveness)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from ..fusion import MultimodalBiometricFusion
from ..attacks import SpoofingDetector


class MultiTaskBiometricModel(nn.Module):
    """
    Multi-task learning model for biometric authentication

    Tasks:
    1. Verification: Binary classification (genuine vs impostor)
    2. Quality estimation: Regression for each modality quality score
    3. Liveness detection: Binary classification (genuine vs spoofed)
    """

    def __init__(
        self,
        modalities: List[str],
        feature_dim: int = 512,
        gnn_type: str = 'gat',
        gnn_config: Optional[Dict] = None,
        use_quality_task: bool = True,
        use_liveness_task: bool = True,
        shared_layers: int = 2
    ):
        """
        Args:
            modalities: List of modality names
            feature_dim: Feature dimension
            gnn_type: Type of GNN ('gcn', 'gat', 'graphsage')
            gnn_config: Configuration for GNN
            use_quality_task: Enable quality estimation task
            use_liveness_task: Enable liveness detection task
            shared_layers: Number of shared representation layers
        """
        super().__init__()

        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.use_quality_task = use_quality_task
        self.use_liveness_task = use_liveness_task

        # Main fusion model (shared backbone)
        self.fusion_model = MultimodalBiometricFusion(
            modalities=modalities,
            feature_dim=feature_dim,
            gnn_type=gnn_type,
            gnn_config=gnn_config
        )

        # Get embedding dimension from fusion model
        # Assuming the embeddings from GNN have specific dimension
        if gnn_config and 'hidden_dims' in gnn_config:
            embedding_dim = gnn_config['hidden_dims'][-1]
        else:
            embedding_dim = 128  # Default

        # Shared representation layers
        shared_dims = [embedding_dim] + [256] * shared_layers
        shared_layers_list = []
        for i in range(shared_layers):
            shared_layers_list.extend([
                nn.Linear(shared_dims[i], shared_dims[i + 1]),
                nn.BatchNorm1d(shared_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
        self.shared_representation = nn.Sequential(*shared_layers_list)

        shared_output_dim = shared_dims[-1]

        # Task 1: Verification head
        self.verification_head = nn.Sequential(
            nn.Linear(shared_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary: genuine vs impostor
        )

        # Task 2: Quality estimation head (per modality)
        if use_quality_task:
            self.quality_heads = nn.ModuleDict()
            for modality in modalities:
                self.quality_heads[modality] = nn.Sequential(
                    nn.Linear(shared_output_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),  # Regression: quality score [0, 1]
                    nn.Sigmoid()
                )

        # Task 3: Liveness detection head (per modality)
        if use_liveness_task:
            self.liveness_heads = nn.ModuleDict()
            for modality in modalities:
                self.liveness_heads[modality] = SpoofingDetector(
                    input_dim=shared_output_dim,
                    hidden_dim=128,
                    num_classes=2  # genuine vs spoofed
                )

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        return_all_tasks: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all tasks

        Args:
            modality_inputs: Dictionary of modality inputs
            return_all_tasks: Whether to return all task outputs

        Returns:
            Dictionary with task outputs:
                - 'verification_logits': [batch, 2]
                - 'embeddings': [batch, embedding_dim]
                - 'quality_scores': Dict[modality, [batch, 1]] (if enabled)
                - 'liveness_logits': Dict[modality, [batch, 2]] (if enabled)
        """
        # Main fusion forward pass
        logits, embeddings = self.fusion_model(modality_inputs)

        # Shared representation
        shared_repr = self.shared_representation(embeddings)

        # Task 1: Verification
        verification_logits = self.verification_head(shared_repr)

        outputs = {
            'verification_logits': verification_logits,
            'embeddings': embeddings,
            'shared_representation': shared_repr
        }

        if return_all_tasks:
            # Task 2: Quality estimation
            if self.use_quality_task:
                quality_scores = {}
                for modality in self.modalities:
                    quality_scores[modality] = self.quality_heads[modality](shared_repr)
                outputs['quality_scores'] = quality_scores

            # Task 3: Liveness detection
            if self.use_liveness_task:
                liveness_logits = {}
                for modality in self.modalities:
                    liveness_logits[modality] = self.liveness_heads[modality](shared_repr)
                outputs['liveness_logits'] = liveness_logits

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        verification_labels: torch.Tensor,
        quality_labels: Optional[Dict[str, torch.Tensor]] = None,
        liveness_labels: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss

        Args:
            outputs: Model outputs from forward()
            verification_labels: Verification ground truth [batch]
            quality_labels: Quality scores ground truth (optional)
            liveness_labels: Liveness labels ground truth (optional)
            task_weights: Weights for each task

        Returns:
            total_loss: Weighted sum of all task losses
            loss_dict: Dictionary with individual task losses
        """
        if task_weights is None:
            task_weights = {
                'verification': 1.0,
                'quality': 0.3,
                'liveness': 0.5
            }

        loss_dict = {}

        # Task 1: Verification loss (CrossEntropy)
        verification_loss = F.cross_entropy(
            outputs['verification_logits'],
            verification_labels
        )
        loss_dict['verification_loss'] = verification_loss.item()
        total_loss = task_weights['verification'] * verification_loss

        # Task 2: Quality estimation loss (MSE)
        if self.use_quality_task and quality_labels is not None:
            quality_loss = 0.0
            for modality in self.modalities:
                if modality in quality_labels:
                    pred_quality = outputs['quality_scores'][modality]
                    true_quality = quality_labels[modality]
                    quality_loss += F.mse_loss(pred_quality, true_quality)

            quality_loss /= len(self.modalities)
            loss_dict['quality_loss'] = quality_loss.item()
            total_loss = total_loss + task_weights['quality'] * quality_loss

        # Task 3: Liveness detection loss (CrossEntropy)
        if self.use_liveness_task and liveness_labels is not None:
            liveness_loss = 0.0
            for modality in self.modalities:
                if modality in liveness_labels:
                    pred_liveness = outputs['liveness_logits'][modality]
                    true_liveness = liveness_labels[modality]
                    liveness_loss += F.cross_entropy(pred_liveness, true_liveness)

            liveness_loss /= len(self.modalities)
            loss_dict['liveness_loss'] = liveness_loss.item()
            total_loss = total_loss + task_weights['liveness'] * liveness_loss

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def predict_verification(
        self,
        modality_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Predict verification only (for inference)

        Args:
            modality_inputs: Dictionary of modality inputs

        Returns:
            predictions: Verification predictions [batch]
        """
        with torch.no_grad():
            outputs = self.forward(modality_inputs, return_all_tasks=False)
            predictions = torch.argmax(outputs['verification_logits'], dim=1)
        return predictions

    def estimate_quality(
        self,
        modality_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate quality scores for each modality

        Args:
            modality_inputs: Dictionary of modality inputs

        Returns:
            quality_scores: Dictionary of quality scores per modality
        """
        if not self.use_quality_task:
            raise ValueError("Quality estimation task is not enabled")

        with torch.no_grad():
            outputs = self.forward(modality_inputs, return_all_tasks=True)
            quality_scores = outputs['quality_scores']

        return quality_scores

    def detect_liveness(
        self,
        modality_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Detect liveness (spoofing) for each modality

        Args:
            modality_inputs: Dictionary of modality inputs

        Returns:
            liveness_predictions: Dictionary of liveness predictions per modality
        """
        if not self.use_liveness_task:
            raise ValueError("Liveness detection task is not enabled")

        with torch.no_grad():
            outputs = self.forward(modality_inputs, return_all_tasks=True)
            liveness_logits = outputs['liveness_logits']

            liveness_predictions = {}
            for modality, logits in liveness_logits.items():
                liveness_predictions[modality] = torch.argmax(logits, dim=1)

        return liveness_predictions


class AdaptiveTaskWeighting(nn.Module):
    """
    Learnable task weighting using uncertainty

    Reference: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """

    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Log variance for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        task_losses: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss using learned uncertainties

        Args:
            task_losses: List of task losses

        Returns:
            total_loss: Weighted total loss
            weights: Dictionary with task weights
        """
        total_loss = 0.0
        weights = {}

        for i, loss in enumerate(task_losses):
            # Weight = 1 / (2 * sigma^2), where sigma^2 = exp(log_var)
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
            weights[f'task_{i}_weight'] = precision.item()

        return total_loss, weights


class GradientBalancing(nn.Module):
    """
    Gradient-based task balancing

    Balances task gradients to prevent one task from dominating
    """

    def __init__(self, num_tasks: int = 3, alpha: float = 0.12):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.task_weights = nn.Parameter(torch.ones(num_tasks))

    def forward(
        self,
        task_losses: List[torch.Tensor],
        shared_params: nn.ParameterList
    ) -> torch.Tensor:
        """
        Compute balanced loss using gradient information

        Args:
            task_losses: List of task losses
            shared_params: Shared parameters to compute gradients

        Returns:
            balanced_loss: Gradient-balanced total loss
        """
        # Compute gradients for each task
        task_grads = []
        for loss in task_losses:
            grads = torch.autograd.grad(
                loss,
                shared_params,
                retain_graph=True,
                create_graph=True
            )
            # Compute gradient norm
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
            task_grads.append(grad_norm)

        # Normalize weights
        weights = F.softmax(self.task_weights, dim=0)

        # Compute weighted loss
        balanced_loss = sum(w * l for w, l in zip(weights, task_losses))

        return balanced_loss
