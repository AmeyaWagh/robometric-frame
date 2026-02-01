"""FRAME: Framework for Robotic Action and Motion Evaluation.

TorchMetrics-based evaluation metrics for Vision-Language-Action models.
This library provides comprehensive evaluation metrics for VLA models in robotics,
including task performance, trajectory quality, vision-language alignment, safety,
and efficiency metrics.
"""

from importlib.metadata import version

try:
    __version__ = version("robometric-frame")
except Exception:
    # Fallback for development/editable installs
    __version__ = "0.1.0"

from robometric_frame.efficiency import InferenceLatency, MemoryUsage
from robometric_frame.safety import CollisionRate, ObstacleProximity, RiskFactor
from robometric_frame.task_performance import ActionAccuracy, SuccessRate, TaskCompletionRate
from robometric_frame.trajectory_quality import (
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
