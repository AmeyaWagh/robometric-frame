"""Obstacle Proximity metric for robotics policy safety evaluation.

Obstacle Proximity measures the minimum distance between the robot and
environmental obstacles throughout task execution. This metric provides
insights into safety margins and risk assessment capabilities of robotics policies
in cluttered environments.

Reference:
    N. Blunder, M. Thiel, M. Schrick, J. Hinckeldeyn, and J. Kreutzfeldt,
    "Integration and evaluation of a close proximity obstacle detection for
    mobile robots in public space," 2022.
"""

from typing import Any, Callable, Optional

import torch
from torch import Tensor

from robometric_frame.safety.base import BaseSafetyMetric


class ObstacleProximity(BaseSafetyMetric):
    r"""Compute Obstacle Proximity for robotics policy safety evaluation.

    Obstacle Proximity is calculated as:

    .. math::

        \text{OP} = \text{mean}\left(\min_t d_t^{\text{robot} \rightarrow \text{obstacle}}\right)

    where :math:`d_t` is the distance from the robot to the nearest obstacle at time :math:`t`.
    For each trajectory, we find the minimum distance, then compute the mean
    of all these minimum distances across all trajectories.

    This metric uses a user-defined distance function that computes the distance
    from trajectory points to obstacles in the environment. The design allows
    users to implement custom distance calculations based on their specific
    robot geometry and environment representation.

    Args:
        distance_fn: User-defined function that computes distances to obstacles.
            Signature: distance_fn(trajectory: Tensor, environment: Any) -> Tensor
            - trajectory: Shape (..., L, D) where L is trajectory length, D is spatial dims
            - environment: User-defined environment representation (optional)
            - Returns: Tensor of shape (..., L) with distances to nearest obstacle
              at each trajectory point (positive values)
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from robometric_frame.safety import ObstacleProximity
        >>> import torch
        >>> # Define a simple distance function
        >>> def simple_distance_fn(trajectory, environment=None):
        ...     # Distance to walls at ±10
        ...     x_coords = trajectory[..., 0]
        ...     dist_to_walls = torch.minimum(
        ...         torch.abs(x_coords - 10),
        ...         torch.abs(x_coords + 10)
        ...     )
        ...     return dist_to_walls
        >>> metric = ObstacleProximity(distance_fn=simple_distance_fn)
        >>> # Single trajectory: min distance is 2.0
        >>> trajectory = torch.tensor([[0.0, 0.0], [5.0, 0.0], [8.0, 0.0]])
        >>> metric.update(trajectory)
        >>> result = metric.compute()
        >>> result['mean_min_distance'].item()  # Mean of [2.0] = 2.0
        2.0

    Example (with environment):
        >>> # Define distance function with environment obstacles
        >>> def obstacle_distance_fn(trajectory, environment):
        ...     # environment contains obstacle positions
        ...     min_distances = torch.full(trajectory.shape[:-1], float('inf'))
        ...     for obs_pos in environment['positions']:
        ...         # Compute distance to this obstacle for all points
        ...         distances = torch.norm(trajectory - obs_pos, dim=-1)
        ...         min_distances = torch.minimum(min_distances, distances)
        ...     return min_distances
        >>> environment = {
        ...     'positions': [torch.tensor([5.0, 5.0]), torch.tensor([10.0, 10.0])]
        ... }
        >>> metric = ObstacleProximity(distance_fn=obstacle_distance_fn)
        >>> trajectory = torch.tensor([[0.0, 0.0], [3.0, 3.0], [4.0, 4.0]])
        >>> metric.update(trajectory, environment=environment)
        >>> result = metric.compute()

    Example (batched):
        >>> # Batch of trajectories
        >>> metric = ObstacleProximity(distance_fn=simple_distance_fn)
        >>> # Trajectory 1: distances [10, 9, 8] -> min = 8
        >>> # Trajectory 2: distances [5, 4, 3] -> min = 3
        >>> trajectories = torch.tensor([
        ...     [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ...     [[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]]
        ... ])
        >>> metric.update(trajectories)
        >>> result = metric.compute()
        >>> result['mean_min_distance'].item()  # Mean of [8, 3] = 5.5
        5.5
    """

    # Metric states that persist across updates
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True  # Higher distance is better (safer)

    # Dynamically added by add_state() in __init__
    sum_min_distances: Tensor
    num_trajectories: Tensor

    def __init__(
        self,
        distance_fn: Callable[[Tensor, Any], Tensor],
        **kwargs: Any,
    ) -> None:
        """Initialize the ObstacleProximity metric."""
        super().__init__(distance_fn=distance_fn, **kwargs)

        # Add metric states for distributed computation
        self.add_state("sum_min_distances", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_trajectories", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # pylint: disable=arguments-differ
        self,
        trajectory: Tensor,
        environment: Optional[Any] = None,
    ) -> None:
        """Update metric state with trajectory and distance information.

        Args:
            trajectory: Trajectory tensor of shape (..., L, D) where:
                - ... represents any number of batch dimensions (can be empty)
                - L is the number of trajectory points
                - D is the spatial dimensionality (typically 2 or 3)

                Examples of valid shapes:
                - (L, D): Single trajectory
                - (B, L, D): Batch of B trajectories
                - (B, T, L, D): Batch with time/episode dimension

                For each trajectory, the minimum distance across L points is computed,
                then these minimums are accumulated to compute the mean.

            environment: Optional environment representation passed to distance_fn.
                Can be any type (dict, object, tensor, etc.) that the user's
                distance function expects.

        Raises:
            ValueError: If trajectory has invalid shape or distances are negative.
            RuntimeError: If distance_fn returns invalid shape or type.

        Example:
            >>> metric = ObstacleProximity(distance_fn=my_distance_fn)
            >>> trajectory = torch.randn(10, 2)  # 10 points in 2D
            >>> metric.update(trajectory)
            >>> # With environment
            >>> metric.update(trajectory, environment={'obstacles': [...]})
        """
        # Validate trajectory shape
        self._validate_trajectory(trajectory)

        # Compute distances using base class method
        distances = self._compute_distances(trajectory, environment)

        # Find minimum distance for each trajectory along the L dimension
        # distances shape: (..., L) -> min_distances shape: (...)
        min_distances = distances.min(dim=-1).values

        # Count number of trajectories (product of all batch dimensions)
        num_trajectories = min_distances.numel()

        # Update states
        self.sum_min_distances += min_distances.sum()  # pylint: disable=no-member
        self.num_trajectories += num_trajectories  # pylint: disable=no-member

    def compute(self) -> dict[str, Tensor]:
        """Compute obstacle proximity statistics.

        Returns:
            Dictionary containing:
                - 'mean_min_distance': Mean of minimum distances across all trajectories
                - 'sum_min_distances': Sum of all minimum distances
                - 'num_trajectories': Number of trajectories evaluated

        Raises:
            RuntimeError: If no trajectories have been recorded.

        Example:
            >>> metric = ObstacleProximity(distance_fn=my_distance_fn)
            >>> metric.update(trajectory)
            >>> result = metric.compute()
            >>> print(f"Average minimum distance: {result['mean_min_distance'].item():.2f}m")
        """
        if self.num_trajectories == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute obstacle proximity: no trajectories have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        mean_min_distance = self.sum_min_distances / self.num_trajectories  # pylint: disable=no-member

        return {
            "mean_min_distance": mean_min_distance,
            "sum_min_distances": self.sum_min_distances,  # pylint: disable=no-member
            "num_trajectories": self.num_trajectories.float(),  # pylint: disable=no-member
        }
