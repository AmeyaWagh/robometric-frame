"""Path Length metric for VLA trajectory evaluation.

Path Length quantifies the total distance traveled to complete a task, serving as
a crucial metric for efficiency evaluation.

Reference:
    P. Fankhauser et al., "Kinect v2 for mobile robot navigation: Evaluation and
    modeling," in 2015 International Conference on Advanced Robotics (ICAR), IEEE,
    July 2015.
"""

from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


class PathLength(Metric):
    """Compute Path Length for VLA trajectory evaluation.

    Path Length is calculated as:
        PL = Σ(i=1 to L-1) ||p_{i+1} - p_i||_2

    where p_i are trajectory points in D-dimensional space and L is the length
    of the trajectory. Shorter paths generally indicate more efficient task
    execution.

    This metric accumulates path lengths across multiple trajectories and returns
    the average path length when compute() is called.

    Args:
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from vla_metrics.trajectory_quality import PathLength
        >>> import torch
        >>> metric = PathLength()
        >>> # 2D trajectory with 5 points
        >>> trajectory = torch.tensor([
        ...     [0.0, 0.0],
        ...     [1.0, 0.0],
        ...     [1.0, 1.0],
        ...     [2.0, 1.0],
        ...     [2.0, 2.0]
        ... ])
        >>> metric.update(trajectory)
        >>> metric.compute()
        tensor(4.0000)

    Example (batched):
        >>> # Batch of trajectories - shape (B, L, D)
        >>> metric = PathLength()
        >>> batch = torch.tensor([
        ...     [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],  # trajectory 1
        ...     [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]   # trajectory 2
        ... ])
        >>> metric.update(batch)
        >>> metric.compute()  # Average of 2.0 and 2.0
        tensor(2.0000)

    Example (3D trajectories):
        >>> # 3D trajectory
        >>> metric = PathLength()
        >>> trajectory_3d = torch.tensor([
        ...     [0.0, 0.0, 0.0],
        ...     [1.0, 0.0, 0.0],
        ...     [1.0, 1.0, 0.0],
        ...     [1.0, 1.0, 1.0]
        ... ])
        >>> metric.update(trajectory_3d)
        >>> metric.compute()
        tensor(3.0000)

    Example (distributed):
        >>> # In distributed training, metrics are automatically synced
        >>> metric = PathLength()
        >>> # On GPU 0
        >>> traj_gpu0 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        >>> metric.update(traj_gpu0)
        >>> # On GPU 1
        >>> traj_gpu1 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        >>> metric.update(traj_gpu1)
        >>> # Final result aggregates across all GPUs
        >>> result = metric.compute()  # Returns aggregated average path length
    """

    # Metric states that persist across updates
    full_state_update: bool = False

    # Dynamically added by add_state() in __init__
    total_path_length: Tensor
    num_trajectories: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the PathLength metric."""
        super().__init__(**kwargs)

        # Add metric states for distributed computation
        self.add_state("total_path_length", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_trajectories", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, trajectory: Tensor) -> None:  # pylint: disable=arguments-differ
        """Update metric state with new trajectory or batch of trajectories.

        Args:
            trajectory: Tensor of shape (..., L, D) where:
                - ... represents any number of batch dimensions (can be empty)
                - L is the number of points (must be >= 2)
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
        if num_points < 2:
            raise ValueError(
                f"Trajectory must have at least 2 points along dimension -2, "
                f"got {num_points} point(s)"
            )

        # Convert to float for numerical operations
        trajectory = trajectory.float()

        # Calculate differences between consecutive points along the L dimension
        # Shape: (..., L-1, D)
        deltas = trajectory[..., 1:, :] - trajectory[..., :-1, :]

        # Calculate Euclidean distances (L2 norm) along the D dimension
        # Shape: (..., L-1)
        distances = torch.norm(deltas, p=2, dim=-1)

        # Sum along the L-1 dimension to get path lengths for each trajectory
        # Shape: (...)
        path_lengths = distances.sum(dim=-1)

        # Count total number of trajectories (product of all batch dimensions)
        num_trajectories = path_lengths.numel()

        # Update states
        self.total_path_length += path_lengths.sum()  # pylint: disable=no-member
        self.num_trajectories += num_trajectories  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the average Path Length across all trajectories.

        Returns:
            Average path length as a scalar tensor.

        Raises:
            RuntimeError: If no trajectories have been recorded.
        """
        if self.num_trajectories == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute path length: no trajectories have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        return self.total_path_length / self.num_trajectories  # pylint: disable=no-member
