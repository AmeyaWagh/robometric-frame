"""Collision Rate metric for robotics policy safety evaluation.

Collision Rate quantifies the frequency of collisions during task execution,
serving as a primary safety indicator. This metric is particularly critical
for mobile robots and humanoids operating in human environments where safety
is paramount.

Reference:
    M. Hoy, A. S. Matveev, and A. V. Savkin, "Algorithms for collision-free
    navigation of mobile robots in complex cluttered environments: a survey,"
    Robotica, vol. 33, pp. 463–497, Mar. 2015.
"""

from typing import Any, Callable, Optional

import torch
from torch import Tensor

from robometric_frame.safety.base import BaseSafetyMetric


class CollisionRate(BaseSafetyMetric):
    r"""Compute Collision Rate for robotics policy safety evaluation.

    Collision Rate is calculated as:

    .. math::

        \text{CR} = \frac{N_{\text{collisions}}}{T_{\text{steps}}}

    where :math:`N_{\text{collisions}}` is the total number of collision occurrences and
    :math:`T_{\text{steps}}` is the total number of trajectory steps.

    This metric uses a user-defined distance function to compute distances
    to obstacles, then applies a threshold to detect collisions. A collision
    is detected when the distance is less than or equal to the threshold.

    Args:
        distance_fn: User-defined function that computes distances to obstacles.
            Signature: distance_fn(trajectory: Tensor, environment: Any) -> Tensor
            - trajectory: Shape (..., L, D) where L is trajectory length, D is spatial dims
            - environment: User-defined environment representation (optional)
            - Returns: Tensor of shape (..., L) with distances to nearest obstacle
              at each trajectory point (positive values)
        collision_threshold: Distance threshold for collision detection. Distances
            less than or equal to this value are considered collisions. Default: 0.0
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from robometric_frame.safety import CollisionRate
        >>> import torch
        >>> # Define a distance function
        >>> def simple_distance_fn(trajectory, environment=None):
        ...     # Distance to walls at ±5
        ...     x_coords = trajectory[..., 0]
        ...     dist_to_walls = torch.minimum(
        ...         torch.abs(x_coords - 5),
        ...         torch.abs(x_coords + 5)
        ...     )
        ...     return dist_to_walls
        >>> metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=0.5)
        >>> # Trajectory with some points close to walls
        >>> trajectory = torch.tensor([[0.0, 0.0], [3.0, 0.0], [4.8, 0.0], [1.0, 0.0]])
        >>> metric.update(trajectory)
        >>> result = metric.compute()
        >>> result['collision_rate'].item()  # One point at distance 0.2 <= 0.5
        0.25

    Example (with environment):
        >>> # Define distance function with environment obstacles
        >>> def obstacle_distance_fn(trajectory, environment):
        ...     # environment is a dict with obstacle positions and radii
        ...     min_distances = torch.full(trajectory.shape[:-1], float('inf'))
        ...     for obs_pos, obs_radius in zip(environment['positions'], environment['radii']):
        ...         # Compute distance to obstacle surface
        ...         distances = torch.norm(trajectory - obs_pos, dim=-1) - obs_radius
        ...         min_distances = torch.minimum(min_distances, distances)
        ...     return torch.clamp(min_distances, min=0.0)  # Ensure non-negative
        >>> environment = {
        ...     'positions': [torch.tensor([2.0, 2.0]), torch.tensor([5.0, 5.0])],
        ...     'radii': [0.5, 0.5]
        ... }
        >>> metric = CollisionRate(distance_fn=obstacle_distance_fn)
        >>> trajectory = torch.tensor([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        >>> metric.update(trajectory, environment=environment)
        >>> result = metric.compute()

    Example (batched):
        >>> # Batch of trajectories
        >>> metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=0.5)
        >>> trajectories = torch.tensor([
        ...     [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ...     [[4.7, 0.0], [4.9, 0.0], [5.0, 0.0]]
        ... ])
        >>> metric.update(trajectories)
        >>> result = metric.compute()
    """

    # Metric states that persist across updates
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = False

    # Dynamically added by add_state() in __init__
    total_collisions: Tensor
    total_steps: Tensor

    def __init__(
        self,
        distance_fn: Callable[[Tensor, Any], Tensor],
        collision_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the CollisionRate metric."""
        super().__init__(distance_fn=distance_fn, **kwargs)

        if collision_threshold < 0:
            raise ValueError(f"collision_threshold must be non-negative, got {collision_threshold}")

        self.collision_threshold = collision_threshold

        # Add metric states for distributed computation
        self.add_state("total_collisions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_steps", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # pylint: disable=arguments-differ
        self,
        trajectory: Tensor,
        environment: Optional[Any] = None,
    ) -> None:
        """Update metric state with trajectory and collision information.

        Args:
            trajectory: Trajectory tensor of shape (..., L, D) where:
                - ... represents any number of batch dimensions (can be empty)
                - L is the number of trajectory points
                - D is the spatial dimensionality (typically 2 or 3)

                Examples of valid shapes:
                - (L, D): Single trajectory
                - (B, L, D): Batch of B trajectories
                - (B, T, L, D): Batch with time/episode dimension

            environment: Optional environment representation passed to distance_fn.
                Can be any type (dict, object, tensor, etc.) that the user's
                distance function expects.

        Raises:
            ValueError: If trajectory has invalid shape or distances are negative.
            RuntimeError: If distance_fn returns invalid shape or type.

        Example:
            >>> metric = CollisionRate(distance_fn=my_distance_fn)
            >>> trajectory = torch.randn(10, 2)  # 10 points in 2D
            >>> metric.update(trajectory)
            >>> # With environment
            >>> metric.update(trajectory, environment={'obstacles': [...]})
        """
        # Validate trajectory shape
        self._validate_trajectory(trajectory)

        # Compute distances using base class method
        distances = self._compute_distances(trajectory, environment)

        # Detect collisions based on threshold
        collisions = self._detect_collisions(distances, self.collision_threshold)

        # Count collisions and steps
        num_collisions = collisions.sum().long()
        num_steps = collisions.numel()

        # Update states
        self.total_collisions += num_collisions  # pylint: disable=no-member
        self.total_steps += num_steps  # pylint: disable=no-member

    def compute(self) -> dict[str, Tensor]:
        """Compute collision rate statistics.

        Returns:
            Dictionary containing:
                - 'collision_rate': Ratio of collision steps to total steps
                - 'total_collisions': Total number of collision occurrences
                - 'total_steps': Total number of trajectory steps
                - 'collision_percentage': Collision rate as percentage (0-100)

        Raises:
            RuntimeError: If no trajectories have been recorded.

        Example:
            >>> metric = CollisionRate(distance_fn=my_distance_fn)
            >>> metric.update(trajectory)
            >>> result = metric.compute()
            >>> print(f"Collision rate: {result['collision_rate'].item():.2%}")
            >>> print(f"Total collisions: {result['total_collisions'].item()}")
        """
        if self.total_steps == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute collision rate: no trajectories have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        collision_rate = self.total_collisions.float() / self.total_steps  # pylint: disable=no-member

        return {
            "collision_rate": collision_rate,
            "total_collisions": self.total_collisions.float(),  # pylint: disable=no-member
            "total_steps": self.total_steps.float(),  # pylint: disable=no-member
            "collision_percentage": collision_rate * 100,
        }
