"""
Training Visualization Tools

Visualizes:
- Training/validation curves (loss, accuracy, metrics)
- Learning rate schedules
- Attention weights evolution
- Per-modality performance
- Gradient flow analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict


class TrainingMonitor:
    """
    Real-time training monitoring and visualization
    """

    def __init__(self, modalities: List[str], save_dir: Optional[str] = None):
        """
        Args:
            modalities: List of modality names
            save_dir: Directory to save plots
        """
        self.modalities = modalities
        self.save_dir = Path(save_dir) if save_dir else None

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for metrics
        self.history = defaultdict(list)
        self.epoch_history = defaultdict(list)

    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        """
        Log metrics for an epoch

        Args:
            metrics: Dictionary of metric name -> value
            epoch: Current epoch number
        """
        self.epoch_history['epoch'].append(epoch)

        for name, value in metrics.items():
            self.epoch_history[name].append(value)

    def plot_training_curves(
        self,
        metrics_to_plot: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show_grid: bool = True
    ) -> plt.Figure:
        """
        Plot training curves

        Args:
            metrics_to_plot: List of metric names to plot (default: all)
            save_path: Path to save figure
            show_grid: Show grid

        Returns:
            Matplotlib figure
        """
        if not self.epoch_history:
            raise ValueError("No training history available. Call log_metrics() first.")

        epochs = self.epoch_history['epoch']

        # Determine which metrics to plot
        all_metrics = [k for k in self.epoch_history.keys() if k != 'epoch']
        if metrics_to_plot is None:
            metrics_to_plot = all_metrics

        # Separate train and val metrics
        train_metrics = [m for m in metrics_to_plot if 'train' in m.lower()]
        val_metrics = [m for m in metrics_to_plot if 'val' in m.lower()]
        other_metrics = [m for m in metrics_to_plot if m not in train_metrics and m not in val_metrics]

        # Create subplots
        n_plots = len(set([m.replace('train_', '').replace('val_', '') for m in metrics_to_plot]))
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]

        plot_idx = 0
        plotted_metrics = set()

        # Plot paired train/val metrics
        for train_metric in train_metrics:
            base_name = train_metric.replace('train_', '')
            if base_name in plotted_metrics:
                continue

            val_metric = f'val_{base_name}'

            ax = axes[plot_idx]
            ax.plot(epochs, self.epoch_history[train_metric], 'b-', label='Train', linewidth=2)

            if val_metric in self.epoch_history:
                ax.plot(epochs, self.epoch_history[val_metric], 'r-', label='Validation', linewidth=2)

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(base_name.replace('_', ' ').title(), fontsize=11)
            ax.set_title(base_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(show_grid, alpha=0.3)

            plotted_metrics.add(base_name)
            plot_idx += 1

        # Plot other metrics
        for metric in other_metrics:
            if metric in plotted_metrics:
                continue

            ax = axes[plot_idx]
            ax.plot(epochs, self.epoch_history[metric], 'g-', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(show_grid, alpha=0.3)

            plotted_metrics.add(metric)
            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Training Curves', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        elif self.save_dir:
            plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')

        return fig

    def plot_per_modality_performance(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot performance metrics per modality

        Returns:
            Matplotlib figure
        """
        # Find modality-specific metrics
        modality_metrics = {}
        for modality in self.modalities:
            modality_metrics[modality] = {}
            for metric_name in self.epoch_history.keys():
                if modality.lower() in metric_name.lower():
                    modality_metrics[modality][metric_name] = self.epoch_history[metric_name]

        if not any(modality_metrics.values()):
            print("No per-modality metrics found")
            return None

        epochs = self.epoch_history['epoch']

        fig, axes = plt.subplots(1, len(self.modalities), figsize=(5 * len(self.modalities), 4))
        if len(self.modalities) == 1:
            axes = [axes]

        for idx, modality in enumerate(self.modalities):
            ax = axes[idx]

            for metric_name, values in modality_metrics[modality].items():
                label = metric_name.replace(modality.lower(), '').replace('_', ' ').strip().title()
                ax.plot(epochs, values, linewidth=2, label=label, marker='o', markersize=3)

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Metric Value', fontsize=11)
            ax.set_title(f'{modality.capitalize()} Performance', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Per-Modality Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        return fig


def plot_learning_rate_schedule(
    learning_rates: List[float],
    epochs: Optional[List[int]] = None,
    title: str = 'Learning Rate Schedule',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning rate schedule

    Args:
        learning_rates: List of learning rates
        epochs: Optional list of epoch numbers
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if epochs is None:
        epochs = list(range(len(learning_rates)))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, learning_rates, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_attention_evolution(
    attention_weights_history: List[Union[torch.Tensor, np.ndarray]],
    modalities: List[str],
    epochs: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot how attention weights evolve during training

    Args:
        attention_weights_history: List of attention matrices over epochs
        modalities: List of modality names
        epochs: Optional list of epoch numbers
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if epochs is None:
        epochs = list(range(len(attention_weights_history)))

    # Convert to numpy and compute mean across batch dimension if needed
    attention_history = []
    for attn in attention_weights_history:
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().detach().numpy()
        if attn.ndim == 3:
            attn = attn.mean(axis=0)
        attention_history.append(attn)

    n_modalities = len(modalities)
    n_snapshots = min(6, len(attention_history))
    snapshot_indices = np.linspace(0, len(attention_history) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, snap_idx in enumerate(snapshot_indices):
        ax = axes[idx]
        attn = attention_history[snap_idx]

        im = ax.imshow(attn, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(n_modalities))
        ax.set_yticks(range(n_modalities))
        ax.set_xticklabels([m.capitalize() for m in modalities], rotation=45, ha='right')
        ax.set_yticklabels([m.capitalize() for m in modalities])
        ax.set_title(f'Epoch {epochs[snap_idx]}', fontsize=11, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add values
        for i in range(n_modalities):
            for j in range(n_modalities):
                text = ax.text(j, i, f'{attn[i, j]:.2f}',
                             ha='center', va='center', color='black', fontsize=8)

    plt.suptitle('Attention Weights Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_gradient_flow(
    named_parameters,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot gradient flow through the network

    Args:
        named_parameters: Model's named parameters
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color='c', label='max gradient')
    ax.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color='b', label='mean gradient')

    ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color='k')
    ax.set_xticks(range(0, len(ave_grads), 1))
    ax.set_xticklabels(layers, rotation=90, fontsize=8)
    ax.set_xlim(left=0, right=len(ave_grads))
    ax.set_ylim(bottom=-0.001)
    ax.set_xlabel('Layers', fontsize=12)
    ax.set_ylabel('Average Gradient', fontsize=12)
    ax.set_title('Gradient Flow', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_confusion_matrix_evolution(
    confusion_matrices: List[np.ndarray],
    class_names: List[str],
    epochs: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot how confusion matrix evolves during training

    Args:
        confusion_matrices: List of confusion matrices over epochs
        class_names: List of class names
        epochs: Optional list of epoch numbers
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if epochs is None:
        epochs = list(range(len(confusion_matrices)))

    n_snapshots = min(6, len(confusion_matrices))
    snapshot_indices = np.linspace(0, len(confusion_matrices) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, snap_idx in enumerate(snapshot_indices):
        ax = axes[idx]
        cm = confusion_matrices[snap_idx]

        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        im = sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=True,
            square=True
        )

        ax.set_title(f'Epoch {epochs[snap_idx]}', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)

    plt.suptitle('Confusion Matrix Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def create_training_dashboard(
    monitor: TrainingMonitor,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive training dashboard with all metrics

    Args:
        monitor: TrainingMonitor instance with logged metrics
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    epochs = monitor.epoch_history['epoch']

    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    if 'train_loss' in monitor.epoch_history:
        ax1.plot(epochs, monitor.epoch_history['train_loss'], 'b-', label='Train', linewidth=2)
    if 'val_loss' in monitor.epoch_history:
        ax1.plot(epochs, monitor.epoch_history['val_loss'], 'r-', label='Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train_accuracy' in monitor.epoch_history:
        ax2.plot(epochs, monitor.epoch_history['train_accuracy'], 'b-', label='Train', linewidth=2)
    if 'val_accuracy' in monitor.epoch_history:
        ax2.plot(epochs, monitor.epoch_history['val_accuracy'], 'r-', label='Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. EER
    ax3 = fig.add_subplot(gs[0, 2])
    if 'val_eer' in monitor.epoch_history:
        ax3.plot(epochs, monitor.epoch_history['val_eer'], 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('EER (%)')
        ax3.set_title('Equal Error Rate', fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # 4. Learning Rate
    ax4 = fig.add_subplot(gs[1, 0])
    if 'learning_rate' in monitor.epoch_history:
        ax4.plot(epochs, monitor.epoch_history['learning_rate'], 'purple', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule', fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

    # 5. FAR/FRR
    ax5 = fig.add_subplot(gs[1, 1])
    if 'val_far' in monitor.epoch_history and 'val_frr' in monitor.epoch_history:
        ax5.plot(epochs, monitor.epoch_history['val_far'], 'r-', label='FAR', linewidth=2)
        ax5.plot(epochs, monitor.epoch_history['val_frr'], 'b-', label='FRR', linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Error Rate (%)')
        ax5.set_title('FAR/FRR Curves', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. AUC
    ax6 = fig.add_subplot(gs[1, 2])
    if 'val_auc' in monitor.epoch_history:
        ax6.plot(epochs, monitor.epoch_history['val_auc'], 'orange', linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('AUC')
        ax6.set_title('ROC AUC Score', fontweight='bold')
        ax6.grid(True, alpha=0.3)

    # 7-9. Per-modality metrics (if available)
    modality_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    for idx, modality in enumerate(monitor.modalities[:3]):
        ax = modality_axes[idx]
        found_metric = False

        for metric_name in monitor.epoch_history.keys():
            if modality.lower() in metric_name.lower() and 'loss' in metric_name:
                ax.plot(epochs, monitor.epoch_history[metric_name], linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{modality.capitalize()} Loss', fontweight='bold')
                ax.grid(True, alpha=0.3)
                found_metric = True
                break

        if not found_metric:
            ax.axis('off')

    plt.suptitle('Training Dashboard', fontsize=18, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig
