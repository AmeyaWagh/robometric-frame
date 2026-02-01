"""Safety and Robustness metrics for VLA model evaluation.

This module provides metrics for evaluating the safety and robustness of VLA models,
including collision detection, obstacle proximity, and risk assessment.

All safety metrics use user-defined distance functions to evaluate trajectories against
environment constraints.
"""

from robometric_frame.safety.collision_rate import CollisionRate
from robometric_frame.safety.obstacle_proximity import ObstacleProximity
from robometric_frame.safety.risk_factor import RiskFactor

__all__ = [
    "CollisionRate",
    "ObstacleProximity",
    "RiskFactor",
]
