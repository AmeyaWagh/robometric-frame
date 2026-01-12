"""Safety and Robustness metrics for VLA model evaluation.

This module provides metrics for evaluating the safety and robustness of VLA models,
including collision detection, obstacle proximity, and risk assessment.

All safety metrics use user-defined functions to evaluate trajectories against
environment constraints, following the control barrier function paradigm.
"""

from vla_metrics.safety.collision_rate import CollisionRate
from vla_metrics.safety.obstacle_proximity import ObstacleProximity
from vla_metrics.safety.risk_factor import RiskFactor

__all__ = [
    "CollisionRate",
    "ObstacleProximity",
    "RiskFactor",
]
