"""Path Smoothness metric for robotics policy trajectory evaluation.

Path Smoothness evaluates the rate of change in trajectory direction, detecting
oscillations that may arise from velocity changes or directional adjustments.

Reference:
    M. Dobiš, M. Dekan, P. Beňo, F. Duchoň, and A. Babinec, "Evaluation criteria
    for trajectories of robotic arms," Robotics, vol. 11, p. 29, Feb. 2022.

    S. Guillén Ruiz, L. V. Calderita, A. Hidalgo-Paniagua, and J. P. Bandera Rubio,
    "Measuring smoothness as a factor for efficient and socially accepted robot
    motion," Sensors (Basel), vol. 20, p. 6822, Nov. 2020.
"""

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class PathSmoothness(Metric):
    r"""Compute Path Smoothness for robotics policy trajectory evaluation.

    Path Smoothness is calculated as:
        PS = (1/PL) * Σ(i=1 to L-2) \|(p_{i+2} - p_{i+1}) - (p_{i+1} - p_i)\|_2

    where p_i are trajectory points in D-dimensional space, L is the length of
    the trajectory, and PL is the path length. This metric measures the rate of
    change in trajectory direction, with lower values indicating smoother paths.

    The metric calculates the difference between consecutive displacement vectors,
    effectively measuring the second derivative (acceleration) of the path. It is
    normalized by the total path length to make it scale-invariant.

    This metric accumulates smoothness values across multiple trajectories and
    returns the average path smoothness when compute() is called.

    Args:
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from robometric_frame.trajectory_quality import PathSmoothness
        >>> import torch
        >>> metric = PathSmoothness()
        >>> # Smooth straight line (perfect smoothness = 0)
        >>> trajectory = torch.tensor([
        ...     [0.0, 0.0],
        ...     [1.0, 0.0],
        ...     [2.0, 0.0],
        ...     [3.0, 0.0]
        ... ])
        >>> metric.update(trajectory)
        >>> metric.compute()
        tensor(0.0000)

    Example (with direction change):
        >>> # Path with a turn (higher smoothness value)
        >>> metric = PathSmoothness()
        >>> trajectory = torch.tensor([
        ...     [0.0, 0.0],
        ...     [1.0, 0.0],
        ...     [2.0, 0.0],
        ...     [2.0, 1.0]
        ... ])
        >>> metric.update(trajectory)
        >>> result = metric.compute()
        >>> result > 0  # Non-zero smoothness due to direction change
        tensor(True)

    Example (batched):
        >>> # Batch of trajectories - shape (B, L, D)
        >>> metric = PathSmoothness()
        >>> batch = torch.tensor([
        ...     [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],  # smooth
        ...     [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]   # has turn
        ... ])
        >>> metric.update(batch)
        >>> result = metric.compute()  # Average smoothness

    Example (3D trajectories):
        >>> # 3D trajectory
        >>> metric = PathSmoothness()
        >>> trajectory_3d = torch.tensor([
        ...     [0.0, 0.0, 0.0],
        ...     [1.0, 0.0, 0.0],
        ...     [2.0, 0.0, 0.0],
        ...     [3.0, 0.0, 0.0]
        ... ])
        >>> metric.update(trajectory_3d)
        >>> metric.compute()  # Perfect smoothness for straight line
        tensor(0.0000)

    Example (distributed):
        >>> # In distributed training, metrics are automatically synced
        >>> metric = PathSmoothness()
        >>> # On GPU 0
        >>> traj_gpu0 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(traj_gpu0)
        >>> # On GPU 1
        >>> traj_gpu1 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        >>> metric.update(traj_gpu1)
        >>> # Final result aggregates across all GPUs
        >>> result = metric.compute()  # Returns aggregated average smoothness
    """

    # Metric states that persist across updates
    full_state_update: bool = False

    # Dynamically added by add_state() in __init__
    total_smoothness: Tensor
    num_trajectories: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the PathSmoothness metric."""
        super().__init__(**kwargs)

        # Add metric states for distributed computation
        self.add_state("total_smoothness", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_trajectories", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, trajectory: Tensor) -> None:  # pylint: disable=arguments-differ
        """Update metric state with new trajectory or batch of trajectories.

        Args:
            trajectory: Tensor of shape (..., L, D) where:
                - ... represents any number of batch dimensions (can be empty)
                - L is the number of points (must be >= 3)
                - D is the spatial dimensionality (e.g., 2 for 2D, 3 for 3D)

                Examples of valid shapes:
                - (L, D): Single trajectory
                - (B, L, D): Batch of B trajectories
                - (B, T, L, D): Batch of B sequences with T slices each

                Points should be ordered chronologically along the L dimension.

        Raises:
            ValueError: If trajectory has invalid shape or insufficient points.
        """
        if trajectory.ndim < 2:
            raise ValueError(
                f"Trajectory must have at least 2 dimensions (..., L, D), "
                f"got {trajectory.ndim}D tensor with shape {trajectory.shape}"
            )

        num_points = trajectory.shape[-2]  # L is the second-to-last dimension
        if num_points < 3:
            raise ValueError(
                f"Trajectory must have at least 3 points along dimension -2 "
                f"to compute smoothness, got {num_points} point(s)"
            )

        # Convert to float for numerical operations
        trajectory = trajectory.float()

        # Calculate displacement vectors between consecutive points
        # Shape: (..., L-1, D)
        displacements = trajectory[..., 1:, :] - trajectory[..., :-1, :]

        # Calculate the difference between consecutive displacement vectors
        # This measures the change in direction (second derivative)
        # Shape: (..., L-2, D)
        direction_changes = displacements[..., 1:, :] - displacements[..., :-1, :]

        # Calculate Euclidean distances (L2 norm) along the D dimension
        # Shape: (..., L-2)
        change_magnitudes = torch.norm(direction_changes, p=2, dim=-1)

        # Sum along the L-2 dimension to get total direction change
        # Shape: (...)
        total_change = change_magnitudes.sum(dim=-1)

        # Calculate path lengths for normalization
        # Shape: (..., L-1)
        segment_lengths = torch.norm(displacements, p=2, dim=-1)
        # Shape: (...)
        path_lengths = segment_lengths.sum(dim=-1)

        # Normalize by path length to get smoothness metric
        # Add epsilon to avoid division by zero for degenerate trajectories
        eps = torch.finfo(trajectory.dtype).eps
        smoothness_values = total_change / (path_lengths + eps)

        # Count total number of trajectories (product of all batch dimensions)
        num_trajectories = smoothness_values.numel()

        # Update states
        self.total_smoothness += smoothness_values.sum()  # pylint: disable=no-member
        self.num_trajectories += num_trajectories  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the average Path Smoothness across all trajectories.

        Returns:
            Average path smoothness as a scalar tensor. Lower values indicate
            smoother trajectories with less direction changes.

        Raises:
            RuntimeError: If no trajectories have been recorded.
        """
        if self.num_trajectories == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute path smoothness: no trajectories have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        return self.total_smoothness / self.num_trajectories  # pylint: disable=no-member
