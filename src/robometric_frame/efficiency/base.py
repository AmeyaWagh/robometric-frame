"""Base class for efficiency metrics with start/stop interface.

This module provides an abstract base class for efficiency metrics that track
resource usage over time intervals using a start/stop pattern.
"""

from abc import abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class EfficiencyMetric(Metric):
    """Abstract base class for efficiency metrics with start/stop interface.

    This base class provides common functionality for metrics that track
    resource usage (time, memory, etc.) over intervals. It implements:
    - start()/stop() interface for interval-based measurements
    - Percentile computation support
    - Common state management

    Subclasses must implement:
    - _on_start(): Called when measurement starts
    - _on_stop(): Called when measurement stops, should return measured value
    - _get_measurement_unit(): Returns the unit suffix for computed statistics

    Args:
        percentiles: List of percentile values to compute (e.g., [0.5, 0.95, 0.99]).
            Default: [0.5, 0.95, 0.99] for median, 95th, and 99th percentiles.
        **kwargs: Additional keyword arguments passed to the base Metric class.
    """

    # Metric attributes
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = False

    # State attributes (dynamically added by add_state)
    total_value: Tensor
    min_value: Tensor
    max_value: Tensor
    count: Tensor
    values: list[Tensor]

    def __init__(
        self,
        percentiles: Optional[list[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the efficiency metric."""
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
        self.add_state("total_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("min_value", default=torch.tensor(float("inf")), dist_reduce_fx="min")
        self.add_state("max_value", default=torch.tensor(0.0), dist_reduce_fx="max")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        # Store all values for percentile calculation
        self.add_state("values", default=[], dist_reduce_fx="cat")

        # Internal tracking state
        self._is_tracking = False

    @abstractmethod
    def _on_start(self) -> None:
        """Called when measurement starts.

        Subclasses should implement any setup needed when tracking begins
        (e.g., recording start time, initializing memory tracking).
        """

    @abstractmethod
    def _on_stop(self) -> float:
        """Called when measurement stops.

        Subclasses should implement the actual measurement logic and return
        the measured value.

        Returns:
            The measured value (e.g., elapsed time, memory usage).
        """

    @abstractmethod
    def _get_measurement_unit(self) -> str:
        """Return the unit suffix for computed statistics.

        Returns:
            Unit suffix string (e.g., '' for seconds, '_mb' for megabytes).
        """

    def start(self) -> None:
        """Start a measurement interval.

        This method begins tracking. Call stop() to end the interval and
        record the measurement.

        Raises:
            RuntimeError: If start() is called while already tracking.

        Example:
            >>> metric.start()
            >>> # ... perform operations ...
            >>> metric.stop()
        """
        if self._is_tracking:
            raise RuntimeError("Already tracking. Call stop() before starting a new measurement.")

        self._is_tracking = True
        self._on_start()

    def stop(self) -> None:
        """Stop tracking and record the measurement.

        This method ends the tracking interval and records the measured value.

        Raises:
            RuntimeError: If stop() is called without a preceding start().

        Example:
            >>> metric.start()
            >>> # ... perform operations ...
            >>> metric.stop()
        """
        if not self._is_tracking:
            raise RuntimeError("Not currently tracking. Call start() before stop().")

        self._is_tracking = False
        value = self._on_stop()

        # Update metric with measured value
        self.update(torch.tensor(value, device=self.device))

    def update(self, value: Tensor) -> None:
        """Update metric state with measurements.

        Args:
            value: Measurement values. Can be:
                - Scalar tensor: Single measurement
                - 1D tensor: Batch of measurements

                All values must be non-negative.

        Raises:
            ValueError: If value contains negative values.

        Example:
            >>> metric.update(torch.tensor(0.1))  # Single measurement
            >>> metric.update(torch.tensor([0.11, 0.12]))  # Batch
        """
        # Flatten to 1D if needed
        value = value.flatten().float()

        # Validate inputs
        if (value < 0).any():
            raise ValueError("Values must be non-negative")

        # Update states
        self.total_value += value.sum()
        self.min_value = torch.min(self.min_value, value.min())
        self.max_value = torch.max(self.max_value, value.max())
        self.count += value.numel()
        # Store values for percentile calculation
        self.values.append(value)

    def compute(self) -> dict[str, Tensor]:
        """Compute statistics including percentiles.

        Returns:
            Dictionary containing:
                - 'mean{unit}': Mean value
                - 'min{unit}': Minimum value
                - 'max{unit}': Maximum value
                - 'total{unit}': Total accumulated value
                - 'count': Number of measurements
                - 'p{X}{unit}': Xth percentile (e.g., 'p50' for median)

        Raises:
            RuntimeError: If no measurements have been recorded.
        """
        if self.count == 0:
            raise RuntimeError(
                "Cannot compute: no measurements have been recorded. "
                "Call update() or start()/stop() before compute()."
            )

        unit = self._get_measurement_unit()

        # Base statistics
        stats = {
            f"mean{unit}": self.total_value / self.count,
            f"min{unit}": self.min_value,
            f"max{unit}": self.max_value,
            f"total{unit}": self.total_value,
            "count": self.count.float(),
        }

        # Compute percentiles if values are stored
        if self.values:
            all_values = torch.cat(self.values)
            for p in self.percentiles:
                # Format percentile key (e.g., 0.95 -> 'p95', 0.5 -> 'p50')
                key = f"p{int(p * 100)}{unit}"
                stats[key] = torch.quantile(all_values, p)

        return stats

    def reset(self) -> None:
        """Reset the metric state.

        This method resets all metric states to their default values and clears
        any internal tracking state.
        """
        super().reset()
        self._is_tracking = False
