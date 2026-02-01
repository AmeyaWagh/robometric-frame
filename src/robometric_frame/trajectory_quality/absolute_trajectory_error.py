"""Absolute Trajectory Error (ATE) metric for robotics policy trajectory evaluation.

ATE measures the global consistency between predicted and reference trajectories
by computing the average point-to-point Euclidean distance.

Reference:
    J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, "A benchmark
    for the evaluation of RGB-D SLAM systems," in 2012 IEEE/RSJ International
    Conference on Intelligent Robots and Systems, IEEE, Oct. 2012.

    F. Endres, J. Hess, N. Engelhard, J. Sturm, D. Cremers, and W. Burgard,
    "An evaluation of the RGB-D SLAM system," in 2012 IEEE International
    Conference on Robotics and Automation, IEEE, May 2012.
"""

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class AbsoluteTrajectoryError(Metric):
    r"""Compute Absolute Trajectory Error (ATE) for robotics policy trajectory evaluation.

    ATE is calculated as:
        ATE = (1/L) * Σ(i=1 to L) \|p_i - p_i*\|_2

    where p_i are predicted trajectory points, p_i* are reference (ground truth)
    trajectory points, and L is the trajectory length. ATE evaluates global
    consistency by measuring the average Euclidean distance between corresponding
    points in predicted and reference trajectories.

    This metric is critical for navigation and manipulation tasks requiring precise
    positioning. Lower ATE values indicate better trajectory tracking performance.

    This metric accumulates errors across multiple trajectory pairs and returns
    the average ATE when compute() is called.

    Args:
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from robometric_frame.trajectory_quality import AbsoluteTrajectoryError
        >>> import torch
        >>> metric = AbsoluteTrajectoryError()
        >>> # Perfect prediction (zero error)
        >>> predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(predicted, reference)
        >>> metric.compute()
        tensor(0.0000)

    Example (with error):
        >>> # Prediction with constant offset
        >>> metric = AbsoluteTrajectoryError()
        >>> predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> reference = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        >>> metric.update(predicted, reference)
        >>> metric.compute()
        tensor(1.0000)

    Example (batched):
        >>> # Batch of trajectory pairs - shape (B, L, D)
        >>> metric = AbsoluteTrajectoryError()
        >>> predicted_batch = torch.tensor([
        ...     [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ...     [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
        ... ])
        >>> reference_batch = torch.tensor([
        ...     [[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]],
        ...     [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
        ... ])
        >>> metric.update(predicted_batch, reference_batch)
        >>> result = metric.compute()

    Example (3D trajectories):
        >>> # 3D trajectory comparison
        >>> metric = AbsoluteTrajectoryError()
        >>> predicted = torch.tensor([
        ...     [0.0, 0.0, 0.0],
        ...     [1.0, 0.0, 0.0],
        ...     [1.0, 1.0, 0.0]
        ... ])
        >>> reference = torch.tensor([
        ...     [0.0, 0.0, 0.0],
        ...     [1.0, 0.0, 0.0],
        ...     [1.0, 1.0, 1.0]
        ... ])
        >>> metric.update(predicted, reference)
        >>> result = metric.compute()

    Example (distributed):
        >>> # In distributed training, metrics are automatically synced
        >>> metric = AbsoluteTrajectoryError()
        >>> # On GPU 0
        >>> pred_gpu0 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        >>> ref_gpu0 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        >>> metric.update(pred_gpu0, ref_gpu0)
        >>> # On GPU 1
        >>> pred_gpu1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        >>> ref_gpu1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        >>> metric.update(pred_gpu1, ref_gpu1)
        >>> # Final result aggregates across all GPUs
        >>> result = metric.compute()
    """

    # Metric states that persist across updates
    full_state_update: bool = False

    # Dynamically added by add_state() in __init__
    total_error: Tensor
    num_trajectories: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the AbsoluteTrajectoryError metric."""
        super().__init__(**kwargs)

        # Add metric states for distributed computation
        self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_trajectories", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # pylint: disable=arguments-differ
        self, predicted: Tensor, reference: Tensor
    ) -> None:
        """Update metric state with new predicted and reference trajectory pair(s).

        Args:
            predicted: Predicted trajectory tensor of shape (..., L, D) where:
                - ... represents any number of batch dimensions (can be empty)
                - L is the number of points (must be >= 1)
                - D is the spatial dimensionality (e.g., 2 for 2D, 3 for 3D)

                Examples of valid shapes:
                - (L, D): Single trajectory
                - (B, L, D): Batch of B trajectories
                - (B, T, L, D): Batch of B sequences with T slices each

                Points should be ordered chronologically along the L dimension.

            reference: Reference (ground truth) trajectory tensor with the same
                shape as predicted.

        Raises:
            ValueError: If trajectories have invalid shape, mismatched shapes,
                or insufficient points.
        """
        if predicted.ndim < 2:
            raise ValueError(
                f"Trajectories must have at least 2 dimensions (..., L, D), "
                f"got {predicted.ndim}D tensor with shape {predicted.shape}"
            )

        if predicted.shape != reference.shape:
            raise ValueError(
                f"Predicted and reference trajectories must have the same shape, "
                f"got predicted: {predicted.shape}, reference: {reference.shape}"
            )

        num_points = predicted.shape[-2]  # L is the second-to-last dimension
        if num_points < 1:
            raise ValueError(
                f"Trajectories must have at least 1 point along dimension -2, "
                f"got {num_points} point(s)"
            )

        # Convert to float for numerical operations
        predicted = predicted.float()
        reference = reference.float()

        # Calculate point-to-point differences
        # Shape: (..., L, D)
        differences = predicted - reference

        # Calculate Euclidean distances (L2 norm) along the D dimension
        # Shape: (..., L)
        point_errors = torch.norm(differences, p=2, dim=-1)

        # Average along the L dimension to get ATE for each trajectory
        # Shape: (...)
        ate_values = point_errors.mean(dim=-1)

        # Count total number of trajectories (product of all batch dimensions)
        num_trajectories = ate_values.numel()

        # Update states
        self.total_error += ate_values.sum()  # pylint: disable=no-member
        self.num_trajectories += num_trajectories  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the average Absolute Trajectory Error across all trajectory pairs.

        Returns:
            Average ATE as a scalar tensor. Lower values indicate better
            trajectory tracking performance.

        Raises:
            RuntimeError: If no trajectories have been recorded.
        """
        if self.num_trajectories == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute ATE: no trajectories have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        return self.total_error / self.num_trajectories  # pylint: disable=no-member
