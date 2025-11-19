"""
Spoofing attack simulation and detection
"""

from .spoofing import (
    SpoofingType,
    SpoofingAttackSimulator,
    SpoofingDetector,
    RobustnessEvaluator
)

__all__ = [
    'SpoofingType',
    'SpoofingAttackSimulator',
    'SpoofingDetector',
    'RobustnessEvaluator',
]
