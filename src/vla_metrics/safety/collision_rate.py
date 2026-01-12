"""Collision Rate metric for VLA safety evaluation.

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
from torchmetrics import Metric


class CollisionRate(Metric):
    r"""Compute Collision Rate for VLA safety evaluation.

    Collision Rate is calculated as:
        CR = N_collisions / T_steps

    where N_collisions is the total number of collision occurrences and
    T_steps is the total number of trajectory steps.

    This metric uses a user-defined collision detection function that evaluates
    whether trajectory points result in collisions with the environment. This
    design follows the control barrier function paradigm where users define
    safety constraints based on their specific environment representation.

    Args:
        collision_fn: User-defined function that detects collisions.
            Signature: collision_fn(trajectory: Tensor, environment: Any) -> Tensor
            - trajectory: Shape (..., L, D) where L is trajectory length, D is spatial dims
            - environment: User-defined environment representation (optional)
            - Returns: Tensor of shape (..., L) with collision indicators
              (True/1 = collision, False/0 = no collision)
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from vla_metrics.safety import CollisionRate
        >>> import torch
        >>> # Define a simple collision detection function
        >>> def simple_collision_fn(trajectory, environment=None):
        ...     # Check if any point exceeds bounds [-5, 5]
        ...     return ((trajectory.abs() > 5).any(dim=-1)).float()
        >>> metric = CollisionRate(collision_fn=simple_collision_fn)
        >>> # Trajectory with some points outside bounds
        >>> trajectory = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 2.0], [1.0, 1.0]])
        >>> metric.update(trajectory)
        >>> result = metric.compute()
        >>> result['collision_rate'].item()
        0.25

    Example (with environment):
        >>> # Define collision function with environment obstacles
        >>> def obstacle_collision_fn(trajectory, environment):
        ...     # environment is a dict with obstacle positions and radii
        ...     collisions = torch.zeros(trajectory.shape[:-1], dtype=torch.bool)
        ...     for obs_pos, obs_radius in zip(environment['positions'], environment['radii']):
        ...         # Check distance to obstacle
        ...         distances = torch.norm(trajectory - obs_pos, dim=-1)
        ...         collisions |= (distances < obs_radius)
        ...     return collisions.float()
        >>> environment = {
        ...     'positions': [torch.tensor([2.0, 2.0]), torch.tensor([5.0, 5.0])],
        ...     'radii': [0.5, 0.5]
        ... }
        >>> metric = CollisionRate(collision_fn=obstacle_collision_fn)
        >>> trajectory = torch.tensor([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        >>> metric.update(trajectory, environment=environment)
        >>> result = metric.compute()

    Example (batched):
        >>> # Batch of trajectories
        >>> metric = CollisionRate(collision_fn=simple_collision_fn)
        >>> trajectories = torch.tensor([
        ...     [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        ...     [[0.0, 0.0], [6.0, 6.0], [7.0, 7.0]]
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
        collision_fn: Callable[[Tensor, Any], Tensor],
        **kwargs: Any,
    ) -> None:
        """Initialize the CollisionRate metric."""
        super().__init__(**kwargs)

        if not callable(collision_fn):
            raise TypeError("collision_fn must be a callable function")

        self.collision_fn = collision_fn

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

            environment: Optional environment representation passed to collision_fn.
                Can be any type (dict, object, tensor, etc.) that the user's
                collision function expects.

        Raises:
            ValueError: If trajectory has invalid shape.
            RuntimeError: If collision_fn returns invalid shape or type.

        Example:
            >>> metric = CollisionRate(collision_fn=my_collision_fn)
            >>> trajectory = torch.randn(10, 2)  # 10 points in 2D
            >>> metric.update(trajectory)
            >>> # With environment
            >>> metric.update(trajectory, environment={'obstacles': [...]})
        """
        if trajectory.ndim < 2:
            raise ValueError(
                f"Trajectory must have at least 2 dimensions (..., L, D), "
                f"got {trajectory.ndim}D tensor with shape {trajectory.shape}"
            )

        # Call user's collision detection function
        try:
            collisions = self.collision_fn(trajectory, environment)
        except Exception as e:
            raise RuntimeError(
                f"User-provided collision_fn raised an exception: {e}. "
                f"Ensure collision_fn accepts (trajectory, environment) and returns a tensor."
            ) from e

        # Validate collision output
        if not isinstance(collisions, Tensor):
            raise RuntimeError(
                f"collision_fn must return a Tensor, got {type(collisions)}. "
                f"Expected shape (..., L) where L is trajectory length."
            )

        # Expected shape is trajectory shape without the last dimension (D)
        expected_shape = trajectory.shape[:-1]  # (..., L)
        if collisions.shape != expected_shape:
            raise RuntimeError(
                f"collision_fn returned tensor with shape {collisions.shape}, "
                f"expected {expected_shape} (trajectory shape without spatial dimension)"
            )

        # Convert to binary (handle both boolean and numeric)
        collisions = (collisions != 0).long()

        # Count collisions and steps
        num_collisions = collisions.sum()
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
            >>> metric = CollisionRate(collision_fn=my_collision_fn)
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
