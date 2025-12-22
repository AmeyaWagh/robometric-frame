"""Relative Trajectory Error (RTE) metric for VLA trajectory evaluation.

RTE measures the local accuracy between predicted and reference trajectories by
comparing relative motion (displacement vectors) over a specified time window.

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


class RelativeTrajectoryError(Metric):
    r"""Compute Relative Trajectory Error (RTE) for VLA trajectory evaluation.

    RTE is calculated as:
        RTE = (1/(L-Δ)) * Σ(i=1 to L-Δ) \|(p_{i+Δ} - p_i) - (p_{i+Δ}* - p_i*)\|_2

    where p_i are predicted trajectory points, p_i* are reference (ground truth)
    trajectory points, L is the trajectory length, and Δ (delta) is the step size
    for computing relative motion.

    RTE assesses local accuracy by comparing displacement vectors between the
    predicted and reference trajectories. Unlike ATE which measures global
    consistency, RTE focuses on the correctness of relative motion, making it
    particularly useful for evaluating drift and local tracking performance.

    This metric accumulates errors across multiple trajectory pairs and returns
    the average RTE when compute() is called.

    Args:
        delta: Step size for computing relative motion. Must be >= 1.
            Larger values assess consistency over longer time windows.
            Default: 1 (consecutive points).
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from vla_metrics.trajectory_quality import RelativeTrajectoryError
        >>> import torch
        >>> metric = RelativeTrajectoryError(delta=1)
        >>> # Perfect prediction (zero error)
        >>> predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(predicted, reference)
        >>> metric.compute()
        tensor(0.0000)

    Example (with drift):
        >>> # Prediction with constant drift in motion
        >>> metric = RelativeTrajectoryError(delta=1)
        >>> predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]])
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(predicted, reference)
        >>> result = metric.compute()

    Example (larger delta):
        >>> # Using delta=2 to check motion over 2-step windows
        >>> metric = RelativeTrajectoryError(delta=2)
        >>> predicted = torch.tensor([
        ...     [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]
        ... ])
        >>> reference = torch.tensor([
        ...     [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]
        ... ])
        >>> metric.update(predicted, reference)
        >>> metric.compute()
        tensor(0.0000)

    Example (batched):
        >>> # Batch of trajectory pairs - shape (B, L, D)
        >>> metric = RelativeTrajectoryError(delta=1)
        >>> predicted_batch = torch.tensor([
        ...     [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ...     [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
        ... ])
        >>> reference_batch = torch.tensor([
        ...     [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ...     [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
        ... ])
        >>> metric.update(predicted_batch, reference_batch)
        >>> metric.compute()
        tensor(0.0000)

    Example (3D trajectories):
        >>> # 3D trajectory comparison
        >>> metric = RelativeTrajectoryError(delta=1)
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
        >>> metric = RelativeTrajectoryError(delta=1)
        >>> # On GPU 0
        >>> pred_gpu0 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> ref_gpu0 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(pred_gpu0, ref_gpu0)
        >>> # On GPU 1
        >>> pred_gpu1 = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        >>> ref_gpu1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(pred_gpu1, ref_gpu1)
        >>> # Final result aggregates across all GPUs
        >>> result = metric.compute()
    """

    # Metric states that persist across updates
    full_state_update: bool = False

    # Dynamically added by add_state() in __init__
    total_error: Tensor
    num_trajectories: Tensor
    delta: int

    def __init__(
        self,
        delta: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize the RelativeTrajectoryError metric.

        Args:
            delta: Step size for computing relative motion. Must be >= 1.
            **kwargs: Additional keyword arguments passed to the base Metric class.

        Raises:
            ValueError: If delta is less than 1.
        """
        super().__init__(**kwargs)

        if delta < 1:
            raise ValueError(f"Delta must be >= 1, got {delta}")

        self.delta = delta

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
                - L is the number of points (must be > delta)
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
        if num_points <= self.delta:
            raise ValueError(
                f"Trajectories must have more than delta={self.delta} points "
                f"along dimension -2, got {num_points} point(s)"
            )

        # Convert to float for numerical operations
        predicted = predicted.float()
        reference = reference.float()

        # Calculate displacement vectors for predicted trajectory
        # p_{i+delta} - p_i for i=1 to L-delta
        # Shape: (..., L-delta, D)
        pred_displacements = predicted[..., self.delta :, :] - predicted[..., : -self.delta, :]

        # Calculate displacement vectors for reference trajectory
        # p_{i+delta}* - p_i* for i=1 to L-delta
        # Shape: (..., L-delta, D)
        ref_displacements = reference[..., self.delta :, :] - reference[..., : -self.delta, :]

        # Calculate differences between displacement vectors
        # Shape: (..., L-delta, D)
        displacement_errors = pred_displacements - ref_displacements

        # Calculate Euclidean distances (L2 norm) along the D dimension
        # Shape: (..., L-delta)
        relative_errors = torch.norm(displacement_errors, p=2, dim=-1)

        # Average along the L-delta dimension to get RTE for each trajectory
        # Shape: (...)
        rte_values = relative_errors.mean(dim=-1)

        # Count total number of trajectories (product of all batch dimensions)
        num_trajectories = rte_values.numel()

        # Update states
        self.total_error += rte_values.sum()  # pylint: disable=no-member
        self.num_trajectories += num_trajectories  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the average Relative Trajectory Error across all trajectory pairs.

        Returns:
            Average RTE as a scalar tensor. Lower values indicate better
            local tracking performance and less drift.

        Raises:
            RuntimeError: If no trajectories have been recorded.
        """
        if self.num_trajectories == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute RTE: no trajectories have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        return self.total_error / self.num_trajectories  # pylint: disable=no-member
