"""
Contrastive learning loss functions for biometric authentication

Includes:
- TripletLoss: For learning discriminative embeddings
- ContrastiveLoss: For pair-wise similarity learning
- HardNegativeTripletLoss: With hard negative mining
- AngularLoss: For angular-based metric learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class TripletLoss(nn.Module):
    """
    Triplet loss for contrastive learning

    L = max(d(a,p) - d(a,n) + margin, 0)

    where:
    - a: anchor sample
    - p: positive sample (same identity)
    - n: negative sample (different identity)
    - d: distance metric
    """

    def __init__(
        self,
        margin: float = 0.3,
        distance_metric: str = 'euclidean',
        reduction: str = 'mean'
    ):
        """
        Args:
            margin: Margin for triplet loss
            distance_metric: 'euclidean' or 'cosine'
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings [batch, dim]
            positive: Positive embeddings [batch, dim]
            negative: Negative embeddings [batch, dim]

        Returns:
            loss: Triplet loss
        """
        if self.distance_metric == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HardNegativeTripletLoss(nn.Module):
    """
    Triplet loss with hard negative mining

    Automatically selects hardest negatives within batch
    """

    def __init__(
        self,
        margin: float = 0.3,
        distance_metric: str = 'euclidean',
        mining_strategy: str = 'hard',  # 'hard', 'semi-hard', 'all'
    ):
        """
        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric to use
            mining_strategy: Strategy for selecting negatives
                - 'hard': hardest negative
                - 'semi-hard': semi-hard negatives
                - 'all': all negatives
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.mining_strategy = mining_strategy

    def _pairwise_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances"""
        if self.distance_metric == 'euclidean':
            # ||a-b||^2 = ||a||^2 - 2*<a,b> + ||b||^2
            dot_product = torch.mm(x, x.t())
            square_norm = dot_product.diag()
            distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
            distances = F.relu(distances)  # Numerical stability
            return torch.sqrt(distances + 1e-16)
        elif self.distance_metric == 'cosine':
            # Cosine distance
            x_normalized = F.normalize(x, p=2, dim=1)
            cosine_sim = torch.mm(x_normalized, x_normalized.t())
            return 1 - cosine_sim

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            embeddings: Embeddings [batch, dim]
            labels: Labels [batch]

        Returns:
            loss: Triplet loss
            stats: Dictionary with mining statistics
        """
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)

        batch_size = embeddings.size(0)

        # Create mask for positive and negative pairs
        labels = labels.unsqueeze(1)
        mask_pos = labels == labels.t()  # Same identity
        mask_neg = ~mask_pos  # Different identity

        # Remove diagonal (self-pairs)
        mask_pos.fill_diagonal_(False)

        # For each anchor, find hardest positive and negative
        losses = []
        num_valid_triplets = 0
        num_hard_triplets = 0

        for i in range(batch_size):
            # Get positive distances for this anchor
            pos_dists = pairwise_dist[i][mask_pos[i]]
            if len(pos_dists) == 0:
                continue

            # Get negative distances for this anchor
            neg_dists = pairwise_dist[i][mask_neg[i]]
            if len(neg_dists) == 0:
                continue

            # Mining strategy
            if self.mining_strategy == 'hard':
                # Hardest positive (farthest positive)
                hardest_pos_dist = pos_dists.max()
                # Hardest negative (closest negative)
                hardest_neg_dist = neg_dists.min()

                loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
                losses.append(loss)

                if loss > 0:
                    num_hard_triplets += 1

            elif self.mining_strategy == 'semi-hard':
                # Semi-hard negatives: d(a,p) < d(a,n) < d(a,p) + margin
                hardest_pos_dist = pos_dists.max()

                # Find semi-hard negatives
                semi_hard_mask = (neg_dists > hardest_pos_dist) & (neg_dists < hardest_pos_dist + self.margin)
                semi_hard_negs = neg_dists[semi_hard_mask]

                if len(semi_hard_negs) > 0:
                    hardest_semi_hard = semi_hard_negs.min()
                else:
                    # Fall back to hardest negative
                    hardest_semi_hard = neg_dists.min()

                loss = F.relu(hardest_pos_dist - hardest_semi_hard + self.margin)
                losses.append(loss)

            elif self.mining_strategy == 'all':
                # All combinations
                for pos_dist in pos_dists:
                    for neg_dist in neg_dists:
                        loss = F.relu(pos_dist - neg_dist + self.margin)
                        if loss > 0:
                            losses.append(loss)
                            num_hard_triplets += 1

            num_valid_triplets += 1

        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True), {}

        total_loss = torch.stack(losses).mean()

        stats = {
            'num_valid_triplets': num_valid_triplets,
            'num_hard_triplets': num_hard_triplets,
            'fraction_hard': num_hard_triplets / max(num_valid_triplets, 1)
        }

        return total_loss, stats


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for pair-wise similarity learning

    L = (1-y) * 0.5 * d^2 + y * 0.5 * max(margin - d, 0)^2

    where y=0 for similar pairs, y=1 for dissimilar pairs
    """

    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: str = 'euclidean'
    ):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(
        self,
        output1: torch.Tensor,
        output2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            output1: First embeddings [batch, dim]
            output2: Second embeddings [batch, dim]
            label: 0 for similar, 1 for dissimilar [batch]

        Returns:
            loss: Contrastive loss
        """
        if self.distance_metric == 'euclidean':
            distance = F.pairwise_distance(output1, output2)
        elif self.distance_metric == 'cosine':
            distance = 1 - F.cosine_similarity(output1, output2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # Loss for similar pairs (label=0)
        loss_similar = (1 - label) * torch.pow(distance, 2)

        # Loss for dissimilar pairs (label=1)
        loss_dissimilar = label * torch.pow(F.relu(self.margin - distance), 2)

        loss = 0.5 * (loss_similar + loss_dissimilar)

        return loss.mean()


class AngularLoss(nn.Module):
    """
    Angular loss for metric learning

    Encourages features from the same class to be closer in angular space
    """

    def __init__(self, alpha: float = 45.0):
        """
        Args:
            alpha: Angle threshold in degrees
        """
        super().__init__()
        self.alpha = alpha
        self.alpha_rad = np.deg2rad(alpha)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings [batch, dim]
            positive: Positive embeddings [batch, dim]
            negative: Negative embeddings [batch, dim]

        Returns:
            loss: Angular loss
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Center anchor at origin
        # x_p = positive - anchor
        # x_n = negative - anchor

        # Compute angles
        # cos(angle) = dot(x_p, x_n) / (||x_p|| * ||x_n||)

        # For normalized vectors, we can compute angles directly
        # angle(anchor, positive)
        cos_ap = (anchor * positive).sum(dim=1)
        # angle(anchor, negative)
        cos_an = (anchor * negative).sum(dim=1)

        # We want: angle(a,p) + alpha < angle(a,n)
        # In cosine space: cos(angle(a,p)) > cos(angle(a,n) + alpha)

        # tan(alpha) approximation for small angles
        tan_alpha = np.tan(self.alpha_rad)

        # Angular loss formulation
        # L = log(1 + exp(f_apn))
        # where f_apn = 4*tan(alpha)^2 * (x_a + x_p)^T * x_n - 2*(1+tan(alpha)^2)*x_a^T*x_p

        sq_tan_alpha = tan_alpha ** 2

        # Simplified version using cosine similarities
        f_apn = 4 * sq_tan_alpha * cos_an - 2 * (1 + sq_tan_alpha) * cos_ap

        loss = torch.log(1 + torch.exp(f_apn))

        return loss.mean()


class CenterLoss(nn.Module):
    """
    Center loss for learning discriminative features

    Minimizes intra-class variation by penalizing distance to class centers
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        lambda_c: float = 0.003
    ):
        """
        Args:
            num_classes: Number of identity classes
            feature_dim: Feature dimension
            lambda_c: Weight for center loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c

        # Initialize class centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: Feature embeddings [batch, dim]
            labels: Class labels [batch]

        Returns:
            loss: Center loss
        """
        batch_size = features.size(0)

        # Get centers for each sample
        centers_batch = self.centers[labels]

        # Compute distances to centers
        loss = F.mse_loss(features, centers_batch, reduction='sum') / batch_size

        return self.lambda_c * loss


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss for deep metric learning

    Reference: Wang et al. "Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning"
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 50.0,
        base: float = 0.5,
        epsilon: float = 0.1
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.epsilon = epsilon

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Embeddings [batch, dim]
            labels: Labels [batch]

        Returns:
            loss: Multi-similarity loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_mat = torch.mm(embeddings, embeddings.t())

        batch_size = embeddings.size(0)

        # Create masks
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.t()).float()
        mask_neg = (labels != labels.t()).float()

        # Remove diagonal
        mask_pos.fill_diagonal_(0)

        # Positive loss
        pos_sim = sim_mat * mask_pos
        pos_loss = (1.0 / self.alpha) * torch.log(
            1 + torch.sum(torch.exp(-self.alpha * (pos_sim - self.base)) * mask_pos, dim=1)
        )

        # Negative loss
        neg_sim = sim_mat * mask_neg
        neg_loss = (1.0 / self.beta) * torch.log(
            1 + torch.sum(torch.exp(self.beta * (neg_sim - self.base)) * mask_neg, dim=1)
        )

        loss = pos_loss + neg_loss

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for biometric authentication

    Combines classification loss with metric learning loss
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        use_triplet: bool = True,
        use_center: bool = True,
        triplet_margin: float = 0.3,
        center_lambda: float = 0.003,
        triplet_weight: float = 1.0,
        center_weight: float = 0.5
    ):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()

        self.use_triplet = use_triplet
        self.use_center = use_center

        if use_triplet:
            self.triplet_loss = HardNegativeTripletLoss(
                margin=triplet_margin,
                mining_strategy='hard'
            )
            self.triplet_weight = triplet_weight

        if use_center:
            self.center_loss = CenterLoss(
                num_classes=num_classes,
                feature_dim=feature_dim,
                lambda_c=center_lambda
            )
            self.center_weight = center_weight

    def forward(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            logits: Classification logits [batch, num_classes]
            embeddings: Feature embeddings [batch, dim]
            labels: Ground truth labels [batch]

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Classification loss
        ce_loss = self.ce_loss(logits, labels)

        total_loss = ce_loss
        loss_dict = {'ce_loss': ce_loss.item()}

        # Triplet loss
        if self.use_triplet:
            triplet_loss, stats = self.triplet_loss(embeddings, labels)
            total_loss = total_loss + self.triplet_weight * triplet_loss
            loss_dict['triplet_loss'] = triplet_loss.item()
            loss_dict.update(stats)

        # Center loss
        if self.use_center:
            center_loss = self.center_loss(embeddings, labels)
            total_loss = total_loss + self.center_weight * center_loss
            loss_dict['center_loss'] = center_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
