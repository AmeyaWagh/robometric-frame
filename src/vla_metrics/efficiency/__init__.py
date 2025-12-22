"""Efficiency metrics for VLA model evaluation.

This module provides metrics for evaluating the computational efficiency of VLA models,
including inference latency, computation time, and memory usage.
"""

from vla_metrics.efficiency.inference_latency import InferenceLatency

__all__ = [
    "InferenceLatency",
]
