"""
Cumulative Match Characteristic (CMC) curve for biometric identification

CMC shows identification accuracy at different ranks
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from sklearn.metrics import pairwise_distances


def compute_cmc(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    max_rank: int = 20,
    distance_metric: str = 'euclidean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Cumulative Match Characteristic (CMC) curve

    Args:
        query_features: Query feature vectors [num_queries, feature_dim]
        gallery_features: Gallery feature vectors [num_gallery, feature_dim]
        query_labels: Query identity labels [num_queries]
        gallery_labels: Gallery identity labels [num_gallery]
        max_rank: Maximum rank to compute
        distance_metric: Distance metric ('euclidean', 'cosine')

    Returns:
        ranks: Rank values [max_rank]
        cmc: Cumulative match scores [max_rank]
    """
    num_queries = query_features.shape[0]

    # Compute pairwise distances
    if distance_metric == 'euclidean':
        distances = pairwise_distances(query_features, gallery_features, metric='euclidean')
    elif distance_metric == 'cosine':
        distances = pairwise_distances(query_features, gallery_features, metric='cosine')
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # For each query, rank gallery samples by distance
    matches = np.zeros((num_queries, max_rank))

    for i in range(num_queries):
        # Sort gallery by distance to query
        sorted_indices = np.argsort(distances[i])

        # Find where the correct match appears
        query_label = query_labels[i]

        for rank in range(max_rank):
            # Check if any of the top-rank matches is correct
            top_k_labels = gallery_labels[sorted_indices[:rank + 1]]
            if query_label in top_k_labels:
                matches[i, rank] = 1
                # Once matched, all higher ranks also match
                matches[i, rank:] = 1
                break

    # CMC: percentage of queries where correct match appears in top-k
    cmc = np.mean(matches, axis=0)
    ranks = np.arange(1, max_rank + 1)

    return ranks, cmc


def plot_cmc_curve(
    ranks: np.ndarray,
    cmc: np.ndarray,
    save_path: Optional[str] = None,
    label: str = 'CMC',
    title: str = 'Cumulative Match Characteristic (CMC) Curve'
):
    """
    Plot CMC curve

    Args:
        ranks: Rank values
        cmc: CMC scores
        save_path: Path to save figure
        label: Label for the curve
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, cmc * 100, 'b-', linewidth=2, marker='o', markersize=4, label=label)

    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Identification Rate (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)

    # Add text annotations for key ranks
    key_ranks = [1, 5, 10]
    for rank in key_ranks:
        if rank <= len(ranks):
            idx = rank - 1
            plt.annotate(
                f'Rank-{rank}: {cmc[idx]*100:.2f}%',
                xy=(ranks[idx], cmc[idx]*100),
                xytext=(10, -10),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_multiple_cmc_curves(
    cmc_results: List[Tuple[np.ndarray, np.ndarray, str]],
    save_path: Optional[str] = None,
    title: str = 'CMC Comparison'
):
    """
    Plot multiple CMC curves for comparison

    Args:
        cmc_results: List of (ranks, cmc, label) tuples
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))

    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown']

    for idx, (ranks, cmc, label) in enumerate(cmc_results):
        color = colors[idx % len(colors)]
        plt.plot(ranks, cmc * 100, color=color, linewidth=2,
                marker='o', markersize=3, label=label)

    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Identification Rate (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


class CMCEvaluator:
    """
    Comprehensive CMC evaluation for biometric identification
    """

    def __init__(self, max_rank: int = 20):
        self.max_rank = max_rank
        self.results = {}

    def evaluate(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        query_labels: np.ndarray,
        gallery_labels: np.ndarray,
        distance_metric: str = 'euclidean'
    ) -> dict:
        """
        Evaluate and store CMC results

        Returns:
            Dictionary with CMC metrics
        """
        ranks, cmc = compute_cmc(
            query_features,
            gallery_features,
            query_labels,
            gallery_labels,
            max_rank=self.max_rank,
            distance_metric=distance_metric
        )

        self.results = {
            'ranks': ranks,
            'cmc': cmc,
            'rank1_accuracy': cmc[0],
            'rank5_accuracy': cmc[4] if len(cmc) >= 5 else cmc[-1],
            'rank10_accuracy': cmc[9] if len(cmc) >= 10 else cmc[-1],
            'rank20_accuracy': cmc[19] if len(cmc) >= 20 else cmc[-1],
        }

        return self.results

    def print_summary(self):
        """Print CMC summary"""
        if not self.results:
            print("No results available. Run evaluate() first.")
            return

        print("\n" + "="*60)
        print("CUMULATIVE MATCH CHARACTERISTIC (CMC) RESULTS")
        print("="*60)
        print(f"Rank-1 Accuracy:  {self.results['rank1_accuracy']*100:.2f}%")
        print(f"Rank-5 Accuracy:  {self.results['rank5_accuracy']*100:.2f}%")
        print(f"Rank-10 Accuracy: {self.results['rank10_accuracy']*100:.2f}%")
        print(f"Rank-20 Accuracy: {self.results['rank20_accuracy']*100:.2f}%")
        print("="*60 + "\n")

    def plot(self, save_path: Optional[str] = None):
        """Plot CMC curve"""
        if not self.results:
            print("No results available. Run evaluate() first.")
            return

        plot_cmc_curve(
            self.results['ranks'],
            self.results['cmc'],
            save_path=save_path
        )
