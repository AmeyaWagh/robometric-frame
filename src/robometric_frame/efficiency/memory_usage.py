"""Memory Usage metric for robotics policy evaluation.

Memory Usage assesses resource consumption during operation, particularly relevant
for compact robotics policies designed for consumer hardware deployment. Efficient memory
usage enables broader accessibility and real-world deployment scenarios.

Reference:
    X.-H. Sun and D. Wang, "APC: A novel memory metric and measurement methodology
    for modern memory systems," IEEE Trans. Comput., 2012.
"""

from typing import Any, Optional

import torch
from torch import Tensor

from robometric_frame.efficiency.base import EfficiencyMetric


class MemoryUsage(EfficiencyMetric):
    r"""Compute Memory Usage for robotics policy evaluation.

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
        >>> from robometric_frame.efficiency import MemoryUsage
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

    def __init__(
        self,
        track_ram: bool = True,
        track_vram: Optional[bool] = None,
        percentiles: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the MemoryUsage metric."""
        super().__init__(percentiles=percentiles, **kwargs)

        self.track_ram = track_ram
        # Auto-detect VRAM tracking if not specified
        if track_vram is None:
            track_vram = torch.cuda.is_available()
        self.track_vram = track_vram

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

    def _on_start(self) -> None:
        """Called when memory tracking starts (no-op for memory)."""
        pass  # Memory is sampled at stop time

    def _on_stop(self) -> float:
        """Get current memory usage.

        Returns:
            Current memory usage in MB.
        """
        return self._get_current_memory()

    def _get_measurement_unit(self) -> str:
        """Return '_mb' suffix for megabytes."""
        return "_mb"

    def compute(self) -> dict[str, Tensor]:
        """Compute memory usage statistics including percentiles.

        Returns:
            Dictionary containing:
                - 'mean_mb': Mean memory usage in megabytes
                - 'peak_mb': Peak memory usage in megabytes (alias for max_mb)
                - 'min_mb': Minimum memory usage in megabytes
                - 'max_mb': Maximum memory usage in megabytes
                - 'total_mb': Total accumulated memory in megabytes
                - 'count': Number of measurements
                - 'p{X}_mb': Xth percentile in MB (e.g., 'p50_mb' for median)

        Raises:
            RuntimeError: If no measurements have been recorded.
        """
        stats = super().compute()
        # Add peak_mb as alias for max_mb for backwards compatibility
        stats["peak_mb"] = stats["max_mb"]
        return stats
