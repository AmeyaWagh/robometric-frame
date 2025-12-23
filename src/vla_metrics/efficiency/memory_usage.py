"""Memory Usage metric for VLA model evaluation.

Memory Usage assesses resource consumption during operation, particularly relevant
for compact VLA models designed for consumer hardware deployment. Efficient memory
usage enables broader accessibility and real-world deployment scenarios.

Reference:
    X.-H. Sun and D. Wang, "APC: A novel memory metric and measurement methodology
    for modern memory systems," IEEE Trans. Comput., 2012.
"""

from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class MemoryUsage(Metric):
    r"""Compute Memory Usage for VLA model evaluation.

    Memory Usage is calculated as:
        MU = max_t(RAM_t + VRAM_t)

    This metric tracks the peak memory consumption (RAM + VRAM) during model
    operations, which is critical for deployment on resource-constrained devices.
    It provides statistics including peak, mean, current, and configurable percentiles.

    The metric can be used in two ways:
    1. Manual tracking: Call start() to begin tracking and stop() to end
    2. Direct update: Call update() with pre-measured memory values

    Args:
        track_ram: Whether to track RAM usage. Default: True.
        track_vram: Whether to track VRAM (GPU memory) usage. Default: True if CUDA available.
        percentiles: List of percentile values to compute (e.g., [0.5, 0.95, 0.99]).
            Default: [0.5, 0.95, 0.99] for median, 95th, and 99th percentiles.
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from vla_metrics.efficiency import MemoryUsage
        >>> import torch
        >>> # Manual tracking
        >>> metric = MemoryUsage()
        >>> metric.start()
        >>> # ... model operations ...
        >>> _ = torch.randn(1000, 1000)  # Allocate some memory
        >>> metric.stop()
        >>> result = metric.compute()
        >>> result['peak_mb'] > 0
        tensor(True)

    Example (direct update):
        >>> # Direct update with measured memory (in MB)
        >>> metric = MemoryUsage()
        >>> memory_readings = torch.tensor([100.0, 150.0, 120.0, 180.0])  # MB
        >>> metric.update(memory_readings)
        >>> result = metric.compute()
        >>> result['peak_mb'].item()
        180.0

    Example (batched tracking):
        >>> # Multiple memory measurements
        >>> metric = MemoryUsage()
        >>> for _ in range(10):
        ...     metric.start()
        ...     _ = torch.randn(500, 500)  # Some operation
        ...     metric.stop()
        >>> result = metric.compute()
        >>> result['count']
        tensor(10.)

    Example (custom percentiles):
        >>> # Track specific percentiles
        >>> metric = MemoryUsage(percentiles=[0.5, 0.9, 0.95, 0.99])
        >>> memory = torch.tensor([100.0, 120.0, 150.0, 180.0, 200.0])
        >>> metric.update(memory)
        >>> result = metric.compute()
        >>> result['p95_mb']  # 95th percentile
        tensor(191.)

    Example (RAM only):
        >>> # Track only RAM, not VRAM
        >>> metric = MemoryUsage(track_ram=True, track_vram=False)
        >>> metric.start()
        >>> _ = [i for i in range(100000)]  # CPU memory allocation
        >>> metric.stop()
        >>> result = metric.compute()
    """

    # Metric states that persist across updates
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = False

    # Dynamically added by add_state() in __init__
    total_memory: Tensor
    peak_memory: Tensor
    count: Tensor
    memory_readings: list[Tensor]
    _tracking: bool

    def __init__(
        self,
        track_ram: bool = True,
        track_vram: Optional[bool] = None,
        percentiles: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MemoryUsage metric."""
        super().__init__(**kwargs)

        self.track_ram = track_ram
        # Auto-detect VRAM tracking if not specified
        if track_vram is None:
            track_vram = torch.cuda.is_available()
        self.track_vram = track_vram

        # Store percentiles to compute
        if percentiles is None:
            percentiles = [0.5, 0.95, 0.99]  # median, 95th, 99th percentiles
        self.percentiles = percentiles

        # Validate percentiles
        for p in self.percentiles:
            if not 0 <= p <= 1:
                raise ValueError(f"Percentiles must be between 0 and 1, got {p}")

        # Add metric states for distributed computation
        self.add_state("total_memory", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("peak_memory", default=torch.tensor(0.0), dist_reduce_fx="max")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        # Store all readings for percentile calculation
        self.add_state("memory_readings", default=[], dist_reduce_fx="cat")

        # Internal state for tracking
        self._tracking = False

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Current memory usage (RAM + VRAM) in megabytes.
        """
        total_mb = 0.0

        if self.track_ram:
            try:
                import psutil

                process = psutil.Process()
                ram_bytes = process.memory_info().rss
                total_mb += ram_bytes / (1024 * 1024)  # Convert to MB
            except ImportError:
                pass  # psutil not available, skip RAM tracking

        if self.track_vram and torch.cuda.is_available():
            # Sum across all CUDA devices
            for device_id in range(torch.cuda.device_count()):
                vram_bytes = torch.cuda.memory_allocated(device_id)
                total_mb += vram_bytes / (1024 * 1024)  # Convert to MB

        return total_mb

    def start(self) -> None:
        """Start tracking memory usage.

        This method begins memory tracking. Call stop() to record the peak
        memory usage during the tracked period.

        Raises:
            RuntimeError: If start() is called while already tracking.

        Example:
            >>> metric = MemoryUsage()
            >>> metric.start()
            >>> # ... perform memory-intensive operations ...
            >>> metric.stop()
        """
        if self._tracking:
            raise RuntimeError(
                "Already tracking memory usage. Call stop() before starting a new measurement."
            )

        self._tracking = True

    def stop(self) -> None:
        """Stop tracking and record the current memory usage.

        This method stops memory tracking and records the current memory
        consumption.

        Raises:
            RuntimeError: If stop() is called without a preceding start().

        Example:
            >>> metric = MemoryUsage()
            >>> metric.start()
            >>> # ... perform operations ...
            >>> metric.stop()
        """
        if not self._tracking:
            raise RuntimeError("Not currently tracking. Call start() before stop().")

        self._tracking = False

        # Get current memory and update metric
        current_memory = self._get_current_memory()
        self.update(torch.tensor(current_memory, device=self.device))

    def update(self, memory_mb: Tensor) -> None:  # pylint: disable=arguments-differ
        """Update metric state with memory measurements.

        Args:
            memory_mb: Memory measurements in megabytes (MB). Can be:
                - Scalar tensor: Single memory measurement
                - 1D tensor: Batch of memory measurements

                All values must be non-negative.

        Raises:
            ValueError: If memory contains negative values.

        Example:
            >>> metric = MemoryUsage()
            >>> metric.update(torch.tensor(128.5))  # Single measurement
            >>> metric.update(torch.tensor([100.0, 150.0, 200.0]))  # Batch
        """
        # Flatten to 1D if needed
        memory_mb = memory_mb.flatten().float()

        # Validate inputs
        if (memory_mb < 0).any():
            raise ValueError("Memory values must be non-negative")

        # Update states
        self.total_memory += memory_mb.sum()  # pylint: disable=no-member
        self.peak_memory = torch.max(self.peak_memory, memory_mb.max())  # pylint: disable=no-member
        self.count += memory_mb.numel()  # pylint: disable=no-member
        # Store readings for percentile calculation
        self.memory_readings.append(memory_mb)  # pylint: disable=no-member

    def compute(self) -> dict[str, Tensor]:
        """Compute memory usage statistics including percentiles.

        Returns:
            Dictionary containing:
                - 'mean_mb': Mean memory usage in megabytes
                - 'peak_mb': Peak memory usage in megabytes
                - 'total_mb': Total accumulated memory in megabytes
                - 'count': Number of measurements
                - 'p{X}_mb': Xth percentile in MB (e.g., 'p50_mb' for median)

        Raises:
            RuntimeError: If no measurements have been recorded.

        Example:
            >>> metric = MemoryUsage()
            >>> metric.update(torch.tensor([100.0, 200.0, 150.0]))
            >>> result = metric.compute()
            >>> result['mean_mb'].item()
            150.0
            >>> result['peak_mb'].item()
            200.0
            >>> result['p50_mb'].item()  # median
            150.0
        """
        if self.count == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute memory usage: no measurements have been recorded. "
                "Call update() or start()/stop() before compute()."
            )

        # Base statistics
        stats = {
            "mean_mb": self.total_memory / self.count,  # pylint: disable=no-member
            "peak_mb": self.peak_memory,  # pylint: disable=no-member
            "total_mb": self.total_memory,  # pylint: disable=no-member
            "count": self.count.float(),  # pylint: disable=no-member
        }

        # Compute percentiles if readings are stored
        if self.memory_readings:  # pylint: disable=no-member
            all_readings = torch.cat(self.memory_readings)  # pylint: disable=no-member
            for p in self.percentiles:
                # Format percentile key (e.g., 0.95 -> 'p95_mb', 0.5 -> 'p50_mb')
                key = f"p{int(p * 100)}_mb"
                stats[key] = torch.quantile(all_readings, p)

        return stats

    def reset(self) -> None:
        """Reset the metric state.

        This method resets all metric states to their default values and clears
        any internal tracking state.

        Example:
            >>> metric = MemoryUsage()
            >>> metric.update(torch.tensor([100.0, 200.0]))
            >>> metric.reset()
            >>> # Metric is now ready for new measurements
        """
        super().reset()
        self._tracking = False
