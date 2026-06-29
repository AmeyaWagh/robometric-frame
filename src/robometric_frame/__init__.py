"""FRAME: Framework for Robotic Action and Motion Evaluation.

TorchMetrics-based evaluation metrics for robotics policies.
This library provides comprehensive evaluation metrics for robot learning,
including task performance, trajectory quality, safety, and efficiency metrics.
"""

from importlib.metadata import version

try:
    __version__ = version("robometric-frame")
except Exception:  # pragma: no cover
    __version__ = "0.2.0"

from robometric_frame.efficiency import InferenceLatency, MemoryUsage
from robometric_frame.safety import CollisionRate, ObstacleProximity, RiskFactor
from robometric_frame.task_performance import ActionAccuracy, SuccessRate, TaskCompletionRate
from robometric_frame.trajectory_quality import (
    AbsoluteTrajectoryError,
    CurvatureChange,
    DTWBase,
    DTWDistance,
    NormalizedDTW,
    PathLength,
    PathSmoothness,
    RelativeTrajectoryError,
    SuccessWeightedDTW,
)

__all__ = [
    "AbsoluteTrajectoryError",
    "ActionAccuracy",
    "CollisionRate",
    "CurvatureChange",
    "DTWBase",
    "DTWDistance",
    "InferenceLatency",
    "MemoryUsage",
    "NormalizedDTW",
    "ObstacleProximity",
    "PathLength",
    "PathSmoothness",
    "RelativeTrajectoryError",
    "RiskFactor",
    "SuccessRate",
    "SuccessWeightedDTW",
    "TaskCompletionRate",
]
