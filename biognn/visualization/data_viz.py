"""
Dataset Visualization Tools

Visualizes:
- Multimodal sample grids
- Data distribution analysis
- Feature space (t-SNE/UMAP)
- Quality score distributions
- Class balance and statistics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")


class DatasetVisualizer:
    """
    Comprehensive dataset visualization for multimodal biometric data
    """

    def __init__(self, modalities: List[str]):
        """
        Args:
            modalities: List of modality names
        """
        self.modalities = modalities
        self.num_modalities = len(modalities)

        # Modality display settings
        self.modality_settings = {
            'face': {'cmap': 'gray', 'aspect': 'equal'},
            'fingerprint': {'cmap': 'gray', 'aspect': 'equal'},
            'iris': {'cmap': 'gray', 'aspect': 'auto'},
            'voice': {'cmap': 'viridis', 'aspect': 'auto'},
            'palm': {'cmap': 'gray', 'aspect': 'equal'},
            'gait': {'cmap': 'viridis', 'aspect': 'auto'},
            'signature': {'cmap': 'gray', 'aspect': 'equal'}
        }

    def plot_sample_grid(
        self,
        samples: Dict[str, Union[torch.Tensor, np.ndarray]],
        num_samples: int = 5,
        subject_ids: Optional[List[int]] = None,
        quality_scores: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None,
        title: str = 'Multimodal Biometric Samples'
    ) -> plt.Figure:
        """
        Plot grid of multimodal samples

        Args:
            samples: Dictionary mapping modality to samples [batch, C, H, W] or [batch, features]
            num_samples: Number of samples to display
            subject_ids: Optional subject IDs for each sample
            quality_scores: Optional quality scores per modality
            save_path: Path to save figure
            title: Figure title

        Returns:
            Matplotlib figure
        """
        num_samples = min(num_samples, next(iter(samples.values())).shape[0])

        fig, axes = plt.subplots(
            num_samples,
            self.num_modalities,
            figsize=(3 * self.num_modalities, 3 * num_samples)
        )

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for row in range(num_samples):
            for col, modality in enumerate(self.modalities):
                ax = axes[row, col]

                if modality not in samples:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                    ax.axis('off')
                    continue

                sample = samples[modality][row]

                # Convert to numpy if needed
                if isinstance(sample, torch.Tensor):
                    sample = sample.cpu().numpy()

                # Handle different data types
                settings = self.modality_settings.get(modality, {'cmap': 'gray', 'aspect': 'equal'})

                if modality == 'voice' and sample.ndim >= 2:
                    # Voice: show spectrogram
                    im = ax.imshow(
                        sample if sample.ndim == 2 else sample[0],
                        cmap=settings['cmap'],
                        aspect=settings['aspect'],
                        origin='lower'
                    )
                    ax.set_ylabel('Frequency' if row == 0 else '', fontsize=10)
                    ax.set_xlabel('Time' if row == num_samples - 1 else '', fontsize=10)
                elif sample.ndim == 3:
                    # Image with channels
                    if sample.shape[0] in [1, 3]:  # [C, H, W]
                        sample = np.transpose(sample, (1, 2, 0))
                    if sample.shape[2] == 1:
                        sample = sample.squeeze(2)

                    # Check dimensionality after squeeze
                    if sample.ndim == 3 and sample.shape[2] == 3:
                        im = ax.imshow(sample)
                    else:
                        im = ax.imshow(sample, cmap=settings['cmap'], aspect=settings['aspect'])
                elif sample.ndim == 2:
                    # 2D image or spectrogram
                    im = ax.imshow(sample, cmap=settings['cmap'], aspect=settings['aspect'])
                else:
                    # 1D signal or features
                    ax.plot(sample)
                    ax.set_xlim(0, len(sample))

                # Title for first row
                if row == 0:
                    ax.set_title(modality.capitalize(), fontsize=12, fontweight='bold')

                # Add quality score if provided
                if quality_scores and modality in quality_scores:
                    score = quality_scores[modality][row]
                    color = 'green' if score > 0.7 else 'orange' if score > 0.4 else 'red'
                    ax.text(
                        0.05, 0.95, f'Q: {score:.2f}',
                        transform=ax.transAxes,
                        fontsize=9,
                        va='top',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3)
                    )

                # Add subject ID for first column
                if col == 0 and subject_ids:
                    ax.set_ylabel(f'ID: {subject_ids[row]}', fontsize=10, fontweight='bold')

                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        return fig

    def plot_genuine_vs_imposter(
        self,
        genuine_samples: Dict[str, Union[torch.Tensor, np.ndarray]],
        imposter_samples: Dict[str, Union[torch.Tensor, np.ndarray]],
        num_pairs: int = 3,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare genuine vs imposter pairs side by side

        Args:
            genuine_samples: Genuine samples per modality
            imposter_samples: Imposter samples per modality
            num_pairs: Number of pairs to show
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(
            num_pairs * 2,
            self.num_modalities,
            figsize=(3 * self.num_modalities, 6 * num_pairs)
        )

        for pair in range(num_pairs):
            for col, modality in enumerate(self.modalities):
                # Genuine sample
                ax_genuine = axes[pair * 2, col]
                if modality in genuine_samples:
                    sample = genuine_samples[modality][pair]
                    if isinstance(sample, torch.Tensor):
                        sample = sample.cpu().numpy()

                    settings = self.modality_settings.get(modality, {'cmap': 'gray', 'aspect': 'equal'})

                    if sample.ndim == 3 and sample.shape[0] in [1, 3]:
                        sample = np.transpose(sample, (1, 2, 0)).squeeze()

                    if sample.ndim == 2:
                        ax_genuine.imshow(sample, cmap=settings['cmap'], aspect=settings['aspect'])

                    if pair == 0:
                        ax_genuine.set_title(modality.capitalize(), fontsize=12, fontweight='bold')

                    if col == 0:
                        ax_genuine.set_ylabel('Genuine', fontsize=11, fontweight='bold', color='green')

                ax_genuine.set_xticks([])
                ax_genuine.set_yticks([])
                ax_genuine.spines['top'].set_color('green')
                ax_genuine.spines['bottom'].set_color('green')
                ax_genuine.spines['left'].set_color('green')
                ax_genuine.spines['right'].set_color('green')
                ax_genuine.spines['top'].set_linewidth(2)
                ax_genuine.spines['bottom'].set_linewidth(2)
                ax_genuine.spines['left'].set_linewidth(2)
                ax_genuine.spines['right'].set_linewidth(2)

                # Imposter sample
                ax_imposter = axes[pair * 2 + 1, col]
                if modality in imposter_samples:
                    sample = imposter_samples[modality][pair]
                    if isinstance(sample, torch.Tensor):
                        sample = sample.cpu().numpy()

                    if sample.ndim == 3 and sample.shape[0] in [1, 3]:
                        sample = np.transpose(sample, (1, 2, 0)).squeeze()

                    if sample.ndim == 2:
                        ax_imposter.imshow(sample, cmap=settings['cmap'], aspect=settings['aspect'])

                    if col == 0:
                        ax_imposter.set_ylabel('Imposter', fontsize=11, fontweight='bold', color='red')

                ax_imposter.set_xticks([])
                ax_imposter.set_yticks([])
                ax_imposter.spines['top'].set_color('red')
                ax_imposter.spines['bottom'].set_color('red')
                ax_imposter.spines['left'].set_color('red')
                ax_imposter.spines['right'].set_color('red')
                ax_imposter.spines['top'].set_linewidth(2)
                ax_imposter.spines['bottom'].set_linewidth(2)
                ax_imposter.spines['left'].set_linewidth(2)
                ax_imposter.spines['right'].set_linewidth(2)

        plt.suptitle('Genuine vs Imposter Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        return fig


def plot_data_distribution_dashboard(
    dataset_stats: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive data distribution dashboard

    Args:
        dataset_stats: Dictionary with dataset statistics:
            - 'samples_per_modality': Dict[str, int]
            - 'samples_per_subject': List[int]
            - 'train_val_test_split': Dict[str, int]
            - 'quality_scores': Dict[str, List[float]]
            - 'class_balance': Dict[str, int] (genuine/imposter)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Samples per modality (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'samples_per_modality' in dataset_stats:
        modalities = list(dataset_stats['samples_per_modality'].keys())
        counts = list(dataset_stats['samples_per_modality'].values())
        bars = ax1.bar(modalities, counts, color='skyblue', edgecolor='black')
        ax1.set_title('Samples per Modality', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # 2. Train/Val/Test split (pie chart)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train_val_test_split' in dataset_stats:
        splits = dataset_stats['train_val_test_split']
        labels = list(splits.keys())
        sizes = list(splits.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Train/Val/Test Split', fontsize=12, fontweight='bold')

    # 3. Class balance (pie chart)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'class_balance' in dataset_stats:
        balance = dataset_stats['class_balance']
        labels = list(balance.keys())
        sizes = list(balance.values())
        colors = ['#90EE90', '#FFB6C6']
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Class Balance (Genuine/Imposter)', fontsize=12, fontweight='bold')

    # 4. Samples per subject (histogram)
    ax4 = fig.add_subplot(gs[1, :])
    if 'samples_per_subject' in dataset_stats:
        samples = dataset_stats['samples_per_subject']
        ax4.hist(samples, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax4.axvline(np.mean(samples), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(samples):.1f}')
        ax4.axvline(np.median(samples), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(samples):.1f}')
        ax4.set_title('Distribution of Samples per Subject', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Number of Samples', fontsize=10)
        ax4.set_ylabel('Number of Subjects', fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Quality score distributions (violin plot)
    ax5 = fig.add_subplot(gs[2, :])
    if 'quality_scores' in dataset_stats:
        quality_data = dataset_stats['quality_scores']
        modalities = list(quality_data.keys())
        scores = [quality_data[mod] for mod in modalities]

        parts = ax5.violinplot(scores, positions=range(len(modalities)), showmeans=True, showmedians=True)
        ax5.set_xticks(range(len(modalities)))
        ax5.set_xticklabels(modalities, rotation=45)
        ax5.set_title('Quality Score Distributions by Modality', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Quality Score', fontsize=10)
        ax5.set_ylim(0, 1)
        ax5.grid(True, alpha=0.3, axis='y')

        # Color the violin plots
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

    plt.suptitle('Dataset Distribution Dashboard', fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_feature_space(
    embeddings: Union[torch.Tensor, np.ndarray],
    labels: np.ndarray,
    modality_labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    n_components: int = 2,
    perplexity: int = 30,
    title: str = 'Feature Space Visualization',
    save_path: Optional[str] = None,
    show_legend: bool = True
) -> plt.Figure:
    """
    Visualize feature space using dimensionality reduction

    Args:
        embeddings: Feature embeddings [num_samples, feature_dim]
        labels: Sample labels (e.g., subject IDs)
        modality_labels: Optional modality labels for coloring
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        n_components: Number of components (2 or 3)
        perplexity: Perplexity for t-SNE
        title: Plot title
        save_path: Path to save figure
        show_legend: Show legend

    Returns:
        Matplotlib figure
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(embeddings)
        var_explained = reducer.explained_variance_ratio_
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create figure
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))

        if modality_labels is not None:
            # Color by modality
            unique_modalities = np.unique(modality_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_modalities)))

            for i, mod in enumerate(unique_modalities):
                mask = modality_labels == mod
                ax.scatter(
                    reduced[mask, 0], reduced[mask, 1],
                    c=[colors[i]], label=mod, alpha=0.6, s=50
                )
        else:
            # Color by label
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1],
                c=labels, cmap='tab20', alpha=0.6, s=50
            )
            if show_legend:
                plt.colorbar(scatter, ax=ax, label='Subject ID')

        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)

        if method == 'pca':
            ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=12)
            ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=12)

    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        if modality_labels is not None:
            unique_modalities = np.unique(modality_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_modalities)))

            for i, mod in enumerate(unique_modalities):
                mask = modality_labels == mod
                ax.scatter(
                    reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
                    c=[colors[i]], label=mod, alpha=0.6, s=50
                )
        else:
            scatter = ax.scatter(
                reduced[:, 0], reduced[:, 1], reduced[:, 2],
                c=labels, cmap='tab20', alpha=0.6, s=50
            )

        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=10)
        ax.set_zlabel(f'{method.upper()} Component 3', fontsize=10)

    if show_legend and modality_labels is not None:
        ax.legend(loc='best', fontsize=10)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_before_after_fusion(
    embeddings_before: Dict[str, Union[torch.Tensor, np.ndarray]],
    embeddings_after: Union[torch.Tensor, np.ndarray],
    labels: np.ndarray,
    method: str = 'tsne',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare embeddings before and after GNN fusion

    Args:
        embeddings_before: Dictionary mapping modality to embeddings
        embeddings_after: Fused embeddings
        labels: Sample labels
        method: Dimensionality reduction method
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    num_modalities = len(embeddings_before)

    fig, axes = plt.subplots(2, (num_modalities + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    # Plot before fusion (per modality)
    for idx, (modality, emb) in enumerate(embeddings_before.items()):
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().detach().numpy()

        # Reduce dimensions
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            reduced = reducer.fit_transform(emb)
        else:
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(emb)

        ax = axes[idx]
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', alpha=0.6, s=30)
        ax.set_title(f'{modality.capitalize()} (Before Fusion)', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)

    # Plot after fusion
    if isinstance(embeddings_after, torch.Tensor):
        embeddings_after = embeddings_after.cpu().detach().numpy()

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced_after = reducer.fit_transform(embeddings_after)
    else:
        reducer = PCA(n_components=2)
        reduced_after = reducer.fit_transform(embeddings_after)

    ax = axes[-1]
    scatter = ax.scatter(reduced_after[:, 0], reduced_after[:, 1], c=labels, cmap='tab20', alpha=0.6, s=30)
    ax.set_title('After GNN Fusion', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_modalities + 1, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Feature Space: Before vs After Fusion ({method.upper()})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig
