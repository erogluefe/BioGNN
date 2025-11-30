"""
Visualization tools for BioGNN

Includes:
- Graph structure visualization
- Attention weights heatmaps
- Dataset visualization (samples, distributions)
- Feature space embeddings
- Training monitoring
- Error analysis
- Spoofing attack visualization
- Augmentation comparison
"""

# Graph visualization
from .graph_viz import (
    GraphVisualizer,
    plot_graph_structure,
    plot_edge_strategies_comparison,
    plot_attention_weights,
    plot_modality_embeddings
)

# Dataset visualization
from .data_viz import (
    DatasetVisualizer,
    plot_data_distribution_dashboard,
    plot_feature_space,
    plot_before_after_fusion
)

# Training visualization
from .training_viz import (
    TrainingMonitor,
    plot_learning_rate_schedule,
    plot_attention_evolution,
    plot_gradient_flow,
    plot_confusion_matrix_evolution,
    create_training_dashboard
)

# Analysis visualization
from .analysis_viz import (
    plot_error_analysis,
    plot_hard_negative_pairs,
    plot_per_subject_performance,
    plot_spoofing_attack_comparison,
    plot_augmentation_comparison,
    plot_quality_impact_analysis
)

__all__ = [
    # Graph visualization
    'GraphVisualizer',
    'plot_graph_structure',
    'plot_edge_strategies_comparison',
    'plot_attention_weights',
    'plot_modality_embeddings',

    # Dataset visualization
    'DatasetVisualizer',
    'plot_data_distribution_dashboard',
    'plot_feature_space',
    'plot_before_after_fusion',

    # Training visualization
    'TrainingMonitor',
    'plot_learning_rate_schedule',
    'plot_attention_evolution',
    'plot_gradient_flow',
    'plot_confusion_matrix_evolution',
    'create_training_dashboard',

    # Analysis visualization
    'plot_error_analysis',
    'plot_hard_negative_pairs',
    'plot_per_subject_performance',
    'plot_spoofing_attack_comparison',
    'plot_augmentation_comparison',
    'plot_quality_impact_analysis',
]
