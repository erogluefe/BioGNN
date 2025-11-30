"""
Graph Neural Network Visualization Tools

Visualizes:
- Graph structure (nodes and edges)
- Edge strategies (fully_connected, star, hierarchical)
- Attention weights
- Node embeddings
- Message passing flows
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. Install with: pip install networkx")


class GraphVisualizer:
    """
    Comprehensive graph visualization for GNN-based multimodal fusion
    """

    def __init__(
        self,
        modalities: List[str],
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        style: str = 'default'
    ):
        """
        Args:
            modalities: List of modality names
            edge_index: Edge connections [2, num_edges]
            edge_weights: Optional edge weights [num_edges]
            style: Visualization style ('default', 'publication', 'presentation')
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required. Install with: pip install networkx")

        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.edge_index = edge_index.cpu().numpy() if isinstance(edge_index, torch.Tensor) else edge_index
        self.edge_weights = edge_weights.cpu().numpy() if isinstance(edge_weights, torch.Tensor) else edge_weights
        self.style = style

        # Build networkx graph
        self.graph = self._build_networkx_graph()

        # Style configurations
        self.styles = {
            'default': {
                'node_size': 3000,
                'font_size': 12,
                'edge_width_scale': 5,
                'figsize': (10, 8)
            },
            'publication': {
                'node_size': 2000,
                'font_size': 10,
                'edge_width_scale': 3,
                'figsize': (8, 6)
            },
            'presentation': {
                'node_size': 4000,
                'font_size': 14,
                'edge_width_scale': 7,
                'figsize': (12, 10)
            }
        }

        # Modality colors (intuitive colors for each modality)
        self.modality_colors = {
            'face': '#FF6B6B',      # Red
            'fingerprint': '#4ECDC4',  # Teal
            'iris': '#45B7D1',      # Blue
            'voice': '#FFA07A',     # Orange
            'palm': '#98D8C8',      # Green
            'gait': '#C7CEEA',      # Purple
            'signature': '#FFD93D'   # Yellow
        }

    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from edge_index"""
        G = nx.DiGraph()

        # Add nodes
        for i, modality in enumerate(self.modalities):
            G.add_node(i, modality=modality, label=modality)

        # Add edges
        num_edges = self.edge_index.shape[1]
        for i in range(num_edges):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            weight = self.edge_weights[i] if self.edge_weights is not None else 1.0
            G.add_edge(src, dst, weight=float(weight))

        return G

    def plot(
        self,
        layout: str = 'circular',
        show_edge_weights: bool = True,
        show_edge_labels: bool = False,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Plot graph structure

        Args:
            layout: Node layout ('circular', 'spring', 'kamada_kawai', 'spectral')
            show_edge_weights: Show edge weights as edge thickness
            show_edge_labels: Show edge weight values as labels
            save_path: Path to save figure
            title: Plot title
            ax: Matplotlib axes to plot on

        Returns:
            Matplotlib figure
        """
        style_config = self.styles[self.style]

        if ax is None:
            fig, ax = plt.subplots(figsize=style_config['figsize'])
        else:
            fig = ax.get_figure()

        # Get layout positions
        pos = self._get_layout(layout)

        # Get node colors
        node_colors = [
            self.modality_colors.get(self.modalities[i], '#95A5A6')
            for i in range(self.num_modalities)
        ]

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=style_config['node_size'],
            alpha=0.9,
            ax=ax,
            edgecolors='black',
            linewidths=2
        )

        # Draw node labels
        labels = {i: self.modalities[i].capitalize() for i in range(self.num_modalities)}
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels,
            font_size=style_config['font_size'],
            font_weight='bold',
            font_color='white',
            ax=ax
        )

        # Draw edges with weights
        if show_edge_weights and self.edge_weights is not None:
            # Normalize edge weights for visualization
            weights = np.array([self.graph[u][v]['weight'] for u, v in self.graph.edges()])
            weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
            edge_widths = 1 + weights_norm * style_config['edge_width_scale']

            # Edge colors based on weight
            edge_colors = plt.cm.viridis(weights_norm)

            nx.draw_networkx_edges(
                self.graph,
                pos,
                width=edge_widths,
                edge_color=edge_colors,
                alpha=0.6,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
        else:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                width=2,
                alpha=0.5,
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )

        # Draw edge labels
        if show_edge_labels and self.edge_weights is not None:
            edge_labels = {
                (u, v): f'{self.graph[u][v]["weight"]:.2f}'
                for u, v in self.graph.edges()
            }
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels,
                font_size=8,
                ax=ax
            )

        # Add colorbar for edge weights
        if show_edge_weights and self.edge_weights is not None:
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.viridis,
                norm=plt.Normalize(vmin=weights.min(), vmax=weights.max())
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Edge Weight', rotation=270, labelpad=20)

        # Set title
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        return fig

    def _get_layout(self, layout: str) -> Dict:
        """Get node positions based on layout algorithm"""
        if layout == 'circular':
            return nx.circular_layout(self.graph)
        elif layout == 'spring':
            return nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'kamada_kawai':
            return nx.kamada_kawai_layout(self.graph)
        elif layout == 'spectral':
            return nx.spectral_layout(self.graph)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def get_graph_statistics(self) -> Dict:
        """
        Compute graph statistics

        Returns:
            Dictionary with graph metrics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
        }

        # Degree statistics
        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]

        stats['avg_in_degree'] = np.mean(in_degrees)
        stats['avg_out_degree'] = np.mean(out_degrees)
        stats['max_in_degree'] = np.max(in_degrees)
        stats['max_out_degree'] = np.max(out_degrees)

        return stats

    def print_statistics(self):
        """Print graph statistics"""
        stats = self.get_graph_statistics()

        print("\n" + "="*60)
        print("GRAPH STRUCTURE STATISTICS")
        print("="*60)
        print(f"Number of nodes (modalities): {stats['num_nodes']}")
        print(f"Number of edges: {stats['num_edges']}")
        print(f"Graph density: {stats['density']:.4f}")
        print(f"Is connected: {stats['is_connected']}")
        print(f"\nAverage in-degree: {stats['avg_in_degree']:.2f}")
        print(f"Average out-degree: {stats['avg_out_degree']:.2f}")
        print(f"Max in-degree: {stats['max_in_degree']}")
        print(f"Max out-degree: {stats['max_out_degree']}")
        print("="*60 + "\n")


def plot_graph_structure(
    modalities: List[str],
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
    edge_strategy: str = 'fully_connected',
    layout: str = 'circular',
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Quick plot of graph structure

    Args:
        modalities: List of modality names
        edge_index: Edge connections [2, num_edges]
        edge_weights: Optional edge weights [num_edges]
        edge_strategy: Edge strategy name (for title)
        layout: Node layout
        save_path: Path to save figure
        title: Custom title

    Returns:
        Matplotlib figure
    """
    viz = GraphVisualizer(modalities, edge_index, edge_weights)

    if title is None:
        title = f"Graph Structure: {edge_strategy.replace('_', ' ').title()}"

    fig = viz.plot(
        layout=layout,
        show_edge_weights=(edge_weights is not None),
        title=title,
        save_path=save_path
    )

    return fig


def plot_edge_strategies_comparison(
    modalities: List[str],
    edge_strategies: List[str] = ['fully_connected', 'star', 'hierarchical'],
    edge_weights_dict: Optional[Dict[str, torch.Tensor]] = None,
    save_path: Optional[str] = None,
    builder_class = None
) -> plt.Figure:
    """
    Compare different edge strategies side-by-side

    Args:
        modalities: List of modality names
        edge_strategies: List of edge strategy names
        edge_weights_dict: Optional dictionary mapping strategy to edge weights
        save_path: Path to save figure
        builder_class: Optional ModalityGraphBuilder class (for testing)

    Returns:
        Matplotlib figure
    """
    if builder_class is None:
        from ..fusion.graph_builder import ModalityGraphBuilder
    else:
        ModalityGraphBuilder = builder_class

    n_strategies = len(edge_strategies)
    fig, axes = plt.subplots(1, n_strategies, figsize=(6 * n_strategies, 6))

    if n_strategies == 1:
        axes = [axes]

    for idx, strategy in enumerate(edge_strategies):
        # Build graph with this strategy
        builder = ModalityGraphBuilder(modalities, edge_strategy=strategy)
        edge_index = builder.edge_index

        # Get edge weights if provided
        edge_weights = edge_weights_dict.get(strategy) if edge_weights_dict else None

        # Create visualizer
        viz = GraphVisualizer(modalities, edge_index, edge_weights)

        # Plot on subplot
        viz.plot(
            layout='circular',
            show_edge_weights=(edge_weights is not None),
            title=strategy.replace('_', ' ').title(),
            ax=axes[idx]
        )

    plt.suptitle('Edge Strategy Comparison', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_attention_weights(
    attention_matrix: Union[torch.Tensor, np.ndarray],
    modalities: List[str],
    title: str = 'Attention Weights Heatmap',
    save_path: Optional[str] = None,
    cmap: str = 'YlOrRd'
) -> plt.Figure:
    """
    Plot attention weights as a heatmap

    Args:
        attention_matrix: Attention weights [num_modalities, num_modalities]
        modalities: List of modality names
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.cpu().detach().numpy()

    # If batch dimension exists, take mean across batch
    if attention_matrix.ndim == 3:
        attention_matrix = attention_matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        attention_matrix,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        xticklabels=[m.capitalize() for m in modalities],
        yticklabels=[m.capitalize() for m in modalities],
        square=True,
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Target Modality', fontsize=12, fontweight='bold')
    ax.set_ylabel('Source Modality', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_modality_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray],
    modalities: List[str],
    labels: Optional[np.ndarray] = None,
    method: str = 'tsne',
    title: str = 'Modality Embeddings',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot modality embeddings in 2D using dimensionality reduction

    Args:
        embeddings: Embeddings [num_samples, embedding_dim]
        modalities: List of modality names
        labels: Optional sample labels
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()

    # Dimensionality reduction
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not available, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot embeddings
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=f'Class {label}',
                alpha=0.6,
                s=50
            )
        ax.legend()
    else:
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6,
            s=50,
            c=range(len(embeddings_2d)),
            cmap='viridis'
        )

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig
