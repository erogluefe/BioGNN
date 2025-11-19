"""
Evaluation metrics and tools for biometric authentication
"""

from .metrics import (
    compute_far_frr,
    compute_eer,
    compute_far_at_frr,
    compute_frr_at_far,
    compute_roc_auc,
    compute_det_curve,
    BiometricEvaluator
)
from .cmc import (
    compute_cmc,
    plot_cmc_curve,
    plot_multiple_cmc_curves,
    CMCEvaluator
)

__all__ = [
    'compute_far_frr',
    'compute_eer',
    'compute_far_at_frr',
    'compute_frr_at_far',
    'compute_roc_auc',
    'compute_det_curve',
    'BiometricEvaluator',
    'compute_cmc',
    'plot_cmc_curve',
    'plot_multiple_cmc_curves',
    'CMCEvaluator',
]
