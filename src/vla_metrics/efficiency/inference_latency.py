"""Inference Latency metric for VLA model evaluation.

Inference Latency measures the time required to generate actions from visual
observations and language instructions. This metric is crucial for real-time
applications where responsive behavior is essential for effective human-robot
interaction.

Reference:
    A. Brohan et al., "RT-1: Robotics transformer for real-world control at
    scale," arXiv:2212.06817, 2022.
"""

from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class InferenceLatency(Metric):
    r"""Compute Inference Latency for VLA model evaluation.

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
        >>> from vla_metrics.efficiency import InferenceLatency
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

    # Metric states that persist across updates
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = False

    # Dynamically added by add_state() in __init__
    total_latency: Tensor
    min_latency: Tensor
    max_latency: Tensor
    count: Tensor
    latencies: list[Tensor]
    _start_time: Optional[float]

    def __init__(
        self,
        percentiles: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the InferenceLatency metric."""
        super().__init__(**kwargs)

        # Store percentiles to compute
        if percentiles is None:
            percentiles = [0.5, 0.95, 0.99]  # median, 95th, 99th percentiles
        self.percentiles = percentiles

        # Validate percentiles
        for p in self.percentiles:
            if not 0 <= p <= 1:
                raise ValueError(f"Percentiles must be between 0 and 1, got {p}")

        # Add metric states for distributed computation
        self.add_state("total_latency", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("min_latency", default=torch.tensor(float("inf")), dist_reduce_fx="min")
        self.add_state("max_latency", default=torch.tensor(0.0), dist_reduce_fx="max")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        # Store all latencies for percentile calculation
        self.add_state("latencies", default=[], dist_reduce_fx="cat")

        # Internal state for manual timing (not synced across devices)
        self._start_time = None

    def start(self) -> None:
        """Start timing an inference operation.

        This method records the current time to be used when stop() is called.
        Use this for manual timing of inference operations.

        Raises:
            RuntimeError: If start() is called before a previous stop().

        Example:
            >>> metric = InferenceLatency()
            >>> metric.start()
            >>> # ... perform inference ...
            >>> metric.stop()
        """
        if self._start_time is not None:
            raise RuntimeError(
                "Timer already started. Call stop() before starting a new timing measurement."
            )

        # Use CUDA events for GPU timing if available, otherwise use time.perf_counter
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        import time

        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop timing and record the elapsed time.

        This method calculates the elapsed time since start() was called
        and updates the metric state.

        Raises:
            RuntimeError: If stop() is called without a preceding start().

        Example:
            >>> metric = InferenceLatency()
            >>> metric.start()
            >>> # ... perform inference ...
            >>> metric.stop()
        """
        if self._start_time is None:
            raise RuntimeError("Timer not started. Call start() before stop().")

        # Synchronize GPU if needed
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        import time

        end_time = time.perf_counter()
        latency = end_time - self._start_time
        self._start_time = None

        # Update metric with measured latency
        self.update(torch.tensor(latency, device=self.device))

    def update(self, latency: Tensor) -> None:  # pylint: disable=arguments-differ
        """Update metric state with latency measurements.

        Args:
            latency: Latency measurements in seconds. Can be:
                - Scalar tensor: Single latency measurement
                - 1D tensor: Batch of latency measurements

                All values must be non-negative.

        Raises:
            ValueError: If latency contains negative values.

        Example:
            >>> metric = InferenceLatency()
            >>> metric.update(torch.tensor(0.1))  # Single measurement
            >>> metric.update(torch.tensor([0.11, 0.12]))  # Batch
        """
        # Flatten to 1D if needed
        latency = latency.flatten().float()

        # Validate inputs
        if (latency < 0).any():
            raise ValueError("Latency values must be non-negative")

        # Update states
        self.total_latency += latency.sum()  # pylint: disable=no-member
        self.min_latency = torch.min(self.min_latency, latency.min())  # pylint: disable=no-member
        self.max_latency = torch.max(self.max_latency, latency.max())  # pylint: disable=no-member
        self.count += latency.numel()  # pylint: disable=no-member
        # Store latencies for percentile calculation
        self.latencies.append(latency)  # pylint: disable=no-member

    def compute(self) -> dict[str, Tensor]:
        """Compute latency statistics including percentiles.

        Returns:
            Dictionary containing:
                - 'mean': Mean latency in seconds
                - 'min': Minimum latency in seconds
                - 'max': Maximum latency in seconds
                - 'total': Total accumulated latency in seconds
                - 'count': Number of measurements
                - 'p{X}': Xth percentile (e.g., 'p50' for median, 'p95' for 95th percentile)

        Raises:
            RuntimeError: If no measurements have been recorded.

        Example:
            >>> metric = InferenceLatency()
            >>> metric.update(torch.tensor([0.1, 0.2, 0.15]))
            >>> result = metric.compute()
            >>> result['mean'].item()
            0.15
            >>> result['min'].item()
            0.1
            >>> result['max'].item()
            0.2
            >>> result['p50'].item()  # median
            0.15
        """
        if self.count == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute latency: no measurements have been recorded. "
                "Call update() or start()/stop() before compute()."
            )

        # Base statistics
        stats = {
            "mean": self.total_latency / self.count,  # pylint: disable=no-member
            "min": self.min_latency,  # pylint: disable=no-member
            "max": self.max_latency,  # pylint: disable=no-member
            "total": self.total_latency,  # pylint: disable=no-member
            "count": self.count.float(),  # pylint: disable=no-member
        }

        # Compute percentiles if latencies are stored
        if self.latencies:  # pylint: disable=no-member
            all_latencies = torch.cat(self.latencies)  # pylint: disable=no-member
            for p in self.percentiles:
                # Format percentile key (e.g., 0.95 -> 'p95', 0.5 -> 'p50')
                key = f"p{int(p * 100)}"
                stats[key] = torch.quantile(all_latencies, p)

        return stats

    def reset(self) -> None:
        """Reset the metric state.

        This method resets all metric states to their default values and clears
        any internal timing state.

        Example:
            >>> metric = InferenceLatency()
            >>> metric.update(torch.tensor([0.1, 0.2]))
            >>> metric.reset()
            >>> # Metric is now ready for new measurements
        """
        super().reset()
        self._start_time = None
