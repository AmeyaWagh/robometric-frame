"""Inference Latency metric for robotics policy evaluation.

Inference Latency measures the time required to generate actions from visual
observations and language instructions. This metric is crucial for real-time
applications where responsive behavior is essential for effective human-robot
interaction.

Reference:
    A. Brohan et al., "RT-1: Robotics transformer for real-world control at
    scale," arXiv:2212.06817, 2022.
"""

import time
from typing import Any, Optional

import torch

from robometric_frame.efficiency.base import EfficiencyMetric


class InferenceLatency(EfficiencyMetric):
    r"""Compute Inference Latency for robotics policy evaluation.

    Inference Latency is calculated as:
        IL = t_infer,end - t_infer,start

    This metric tracks the time elapsed during model inference operations,
    which is critical for real-time robotics applications. It accumulates
    timing measurements across multiple inference calls and provides statistics
    including mean, minimum, maximum, total latency, and configurable percentiles.

    The metric is designed to be used in two ways:
    1. Manual timing: Call start() before inference and stop() after
    2. Direct update: Call update() with pre-measured latency values

    Args:
        percentiles: List of percentile values to compute (e.g., [0.5, 0.95, 0.99]).
            Default: [0.5, 0.95, 0.99] for median, 95th, and 99th percentiles.
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from robometric_frame.efficiency import InferenceLatency
        >>> import torch
        >>> import time
        >>> metric = InferenceLatency()
        >>> # Manual timing
        >>> metric.start()
        >>> # ... model inference ...
        >>> time.sleep(0.1)  # Simulate inference
        >>> metric.stop()
        >>> result = metric.compute()
        >>> result['mean'] > 0
        tensor(True)

    Example (direct update):
        >>> # Direct update with measured latency
        >>> metric = InferenceLatency()
        >>> latencies = torch.tensor([0.1, 0.15, 0.12, 0.11])  # seconds
        >>> metric.update(latencies)
        >>> result = metric.compute()
        >>> result['mean'].item()
        0.12

    Example (batched):
        >>> # Multiple inference measurements
        >>> metric = InferenceLatency()
        >>> for _ in range(10):
        ...     metric.start()
        ...     time.sleep(0.01)  # Simulate inference
        ...     metric.stop()
        >>> result = metric.compute()
        >>> result['count']
        tensor(10)

    Example (distributed):
        >>> # In distributed training, metrics are automatically synced
        >>> metric = InferenceLatency()
        >>> # On GPU 0
        >>> metric.update(torch.tensor([0.1, 0.12]))
        >>> # On GPU 1
        >>> metric.update(torch.tensor([0.11, 0.13]))
        >>> # Final result aggregates across all GPUs
        >>> result = metric.compute()
        >>> result['mean'].item()
        0.115

    Example (custom percentiles):
        >>> # Track specific percentiles for robustness analysis
        >>> metric = InferenceLatency(percentiles=[0.5, 0.9, 0.95, 0.99])
        >>> latencies = torch.tensor([0.1, 0.12, 0.15, 0.11, 0.13, 0.2, 0.25, 0.3])
        >>> metric.update(latencies)
        >>> result = metric.compute()
        >>> result['p50']  # median
        tensor(0.1350)
        >>> result['p95']  # 95th percentile
        tensor(0.2875)
    """

    _start_time: Optional[float]

    def __init__(
        self,
        percentiles: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the InferenceLatency metric."""
        super().__init__(percentiles=percentiles, **kwargs)
        self._start_time = None

    def _on_start(self) -> None:
        """Record the start time for latency measurement."""
        # Use CUDA events for GPU timing if available
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self._start_time = time.perf_counter()

    def _on_stop(self) -> float:
        """Calculate and return the elapsed time since start.

        Returns:
            Elapsed time in seconds.
        """
        # Synchronize GPU if needed
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        end_time = time.perf_counter()
        latency = end_time - self._start_time  # type: ignore[operator]
        self._start_time = None
        return latency

    def _get_measurement_unit(self) -> str:
        """Return empty string as latency is measured in seconds (base unit)."""
        return ""

    def reset(self) -> None:
        """Reset the metric state."""
        super().reset()
        self._start_time = None
