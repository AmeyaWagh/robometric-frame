"""Curvature Change metric for robotics policy trajectory evaluation.

Curvature Change measures trajectory smoothness while accounting for robot
orientation, particularly valuable for car-like mobile robots where curvature
relates to turning radius constraints.

Reference:
    J.-H. Hwang, R. C. Arkin, and D.-S. Kwon, "Mobile robots at your fingertip:
    Bezier curve on-line trajectory generation for supervisory control," in
    Proceedings 2003 IEEE/RSJ International Conference on Intelligent Robots and
    Systems (IROS 2003), IEEE, 2004.
"""

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class CurvatureChange(Metric):
    r"""Compute Curvature Change for robotics policy trajectory evaluation.

    Curvature Change is calculated as:

    .. math::

        CC = \frac{1}{L-2} \sum_{i=1}^{L-2} |\kappa_{i+1} - \kappa_i|, \quad
        \kappa_i = \frac{\theta_{i+1} - \theta_i}{\|\mathbf{p}_{i+1} - \mathbf{p}_i\|_2}

    where :math:`\mathbf{p}_i` are trajectory positions, :math:`\theta_i` are
    orientations (heading angles), and :math:`\kappa_i` is the curvature at
    segment :math:`i`. Unlike path smoothness, this metric incorporates angular
    velocity and is particularly useful for evaluating car-like mobile robots
    where curvature relates to turning radius constraints.

    This metric accumulates curvature change values across multiple trajectories
    and returns the average when compute() is called.

    Args:
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from robometric_frame.trajectory_quality import CurvatureChange
        >>> import torch
        >>> metric = CurvatureChange()
        >>> # Straight line motion (constant orientation)
        >>> positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        >>> orientations = torch.tensor([0.0, 0.0, 0.0, 0.0])
        >>> metric.update(positions, orientations)
        >>> metric.compute()
        tensor(0.0000)

    Example (with turn):
        >>> # Path with a turn
        >>> metric = CurvatureChange()
        >>> positions = torch.tensor([
        ...     [0.0, 0.0],
        ...     [1.0, 0.0],
        ...     [2.0, 0.0],
        ...     [3.0, 1.0]
        ... ])
        >>> # Orientations change from 0 to π/4 radians
        >>> orientations = torch.tensor([0.0, 0.0, 0.0, 0.785])
        >>> metric.update(positions, orientations)
        >>> result = metric.compute()

    Example (batched):
        >>> # Batch of trajectory pairs - shape (B, L, D)
        >>> metric = CurvatureChange()
        >>> positions_batch = torch.tensor([
        ...     [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        ...     [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]
        ... ])
        >>> orientations_batch = torch.tensor([
        ...     [0.0, 0.0, 0.0, 0.0],
        ...     [1.57, 1.57, 1.57, 1.57]
        ... ])
        >>> metric.update(positions_batch, orientations_batch)
        >>> result = metric.compute()

    Example (circular path):
        >>> # Circular motion with constant curvature
        >>> metric = CurvatureChange()
        >>> import math
        >>> angles = torch.linspace(0, math.pi/2, 10)
        >>> positions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        >>> orientations = angles + math.pi/2  # Tangent direction
        >>> metric.update(positions, orientations)
        >>> result = metric.compute()  # Should be small for smooth circular motion

    Example (distributed):
        >>> # In distributed training, metrics are automatically synced
        >>> metric = CurvatureChange()
        >>> # On GPU 0
        >>> pos_gpu0 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> ori_gpu0 = torch.tensor([0.0, 0.0, 0.0])
        >>> metric.update(pos_gpu0, ori_gpu0)
        >>> # On GPU 1
        >>> pos_gpu1 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        >>> ori_gpu1 = torch.tensor([1.57, 1.57, 1.57])
        >>> metric.update(pos_gpu1, ori_gpu1)
        >>> # Final result aggregates across all GPUs
        >>> result = metric.compute()
    """

    # Metric states that persist across updates
    full_state_update: bool = False

    # Dynamically added by add_state() in __init__
    total_curvature_change: Tensor
    num_trajectories: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the CurvatureChange metric."""
        super().__init__(**kwargs)

        # Add metric states for distributed computation
        self.add_state("total_curvature_change", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_trajectories", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # pylint: disable=arguments-differ
        self, positions: Tensor, orientations: Tensor
    ) -> None:
        """Update metric state with new trajectory or batch of trajectories.

        Args:
            positions: Position trajectory tensor of shape (..., L, D) where:
                - ... represents any number of batch dimensions (can be empty)
                - L is the number of points (must be >= 3)
                - D is the spatial dimensionality (typically 2 for mobile robots)

                Examples of valid shapes:
                - (L, D): Single trajectory
                - (B, L, D): Batch of B trajectories
                - (B, T, L, D): Batch of B sequences with T slices each

                Points should be ordered chronologically along the L dimension.

            orientations: Orientation (heading angle) tensor of shape (..., L) where:
                - ... represents the same batch dimensions as positions
                - L is the number of points (must match positions)
                - Values are heading angles in radians

                Must have the same batch dimensions and L as positions, but without
                the spatial dimension D.

        Raises:
            ValueError: If trajectories have invalid shape, mismatched shapes,
                or insufficient points.
        """
        if positions.ndim < 2:
            raise ValueError(
                f"Positions must have at least 2 dimensions (..., L, D), "
                f"got {positions.ndim}D tensor with shape {positions.shape}"
            )

        num_points = positions.shape[-2]  # L is the second-to-last dimension
        if num_points < 3:
            raise ValueError(
                f"Trajectory must have at least 3 points along dimension -2 "
                f"to compute curvature change, got {num_points} point(s)"
            )

        # Check that orientations shape matches positions (without the D dimension)
        expected_ori_shape = positions.shape[:-1]  # (..., L)
        if orientations.shape != expected_ori_shape:
            raise ValueError(
                f"Orientations shape must match positions shape without the last "
                f"dimension. Expected {expected_ori_shape}, got {orientations.shape}"
            )

        # Convert to float for numerical operations
        positions = positions.float()
        orientations = orientations.float()

        # Calculate displacement vectors between consecutive points
        # Shape: (..., L-1, D)
        displacements = positions[..., 1:, :] - positions[..., :-1, :]

        # Calculate segment lengths (Euclidean distances)
        # Shape: (..., L-1)
        segment_lengths = torch.norm(displacements, p=2, dim=-1)

        # Calculate orientation changes between consecutive points
        # Shape: (..., L-1)
        orientation_changes = orientations[..., 1:] - orientations[..., :-1]

        # Calculate curvature κ_i = (θ_{i+1} - θ_i) / ||p_{i+1} - p_i||_2
        # Add epsilon to avoid division by zero for degenerate trajectories
        eps = torch.finfo(positions.dtype).eps
        # Shape: (..., L-1)
        curvatures = orientation_changes / (segment_lengths + eps)

        # Calculate absolute differences in curvature between consecutive segments
        # Shape: (..., L-2)
        curvature_changes = torch.abs(curvatures[..., 1:] - curvatures[..., :-1])

        # Average along the L-2 dimension to get CC for each trajectory
        # Shape: (...)
        cc_values = curvature_changes.mean(dim=-1)

        # Count total number of trajectories (product of all batch dimensions)
        num_trajectories = cc_values.numel()

        # Update states
        self.total_curvature_change += cc_values.sum()  # pylint: disable=no-member
        self.num_trajectories += num_trajectories  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the average Curvature Change across all trajectories.

        Returns:
            Average curvature change as a scalar tensor. Lower values indicate
            smoother trajectories with more consistent turning behavior.

        Raises:
            RuntimeError: If no trajectories have been recorded.
        """
        if self.num_trajectories == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute curvature change: no trajectories have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        return self.total_curvature_change / self.num_trajectories  # pylint: disable=no-member
