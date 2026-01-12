"""VLA Metrics: TorchMetrics-based evaluation metrics for Vision-Language-Action models.

This library provides comprehensive evaluation metrics for VLA models in robotics,
including task performance, trajectory quality, vision-language alignment, safety,
and efficiency metrics.
"""

from importlib.metadata import version

try:
    __version__ = version("vla-metrics")
except Exception:
    # Fallback for development/editable installs
    __version__ = "0.1.0"

from vla_metrics.efficiency import InferenceLatency, MemoryUsage
from vla_metrics.safety import CollisionRate, ObstacleProximity, RiskFactor
from vla_metrics.task_performance import ActionAccuracy, SuccessRate, TaskCompletionRate
from vla_metrics.trajectory_quality import (
    AbsoluteTrajectoryError,
    CurvatureChange,
    PathLength,
    PathSmoothness,
    RelativeTrajectoryError,
)

__all__ = [
    "AbsoluteTrajectoryError",
    "ActionAccuracy",
    "CollisionRate",
    "CurvatureChange",
    "InferenceLatency",
    "MemoryUsage",
    "ObstacleProximity",
    "PathLength",
    "PathSmoothness",
    "RelativeTrajectoryError",
    "RiskFactor",
    "SuccessRate",
    "TaskCompletionRate",
]
