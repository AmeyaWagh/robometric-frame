"""Risk Factor metric for VLA safety evaluation.

Risk Factor offers comprehensive safety evaluation by integrating proximity
measurements throughout the route. Calculated as the average of the reciprocal
distances from obstacles, this metric provides a holistic assessment of
safety-conscious behavior.

Reference:
    A. Majumdar and M. Pavone, "How should a robot assess risk? towards an
    axiomatic theory of risk in robotics," in Springer Proceedings in Advanced
    Robotics, pp. 75–84, Cham: Springer International Publishing, 2020.
"""

from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class RiskFactor(Metric):
    r"""Compute Risk Factor for VLA safety evaluation.

    Risk Factor is calculated as:
        RF = (1/T) * Σ(t=1 to T) 1/d_t

    where d_t is the distance from the robot to the nearest obstacle at time t,
    and T is the total number of trajectory steps. This metric provides a
    comprehensive safety assessment by penalizing trajectories that stay close
    to obstacles, with risk increasing as the robot approaches obstacles.

    This metric uses a user-defined distance function that computes the distance
    from trajectory points to obstacles in the environment. The reciprocal of
    these distances creates a risk measure where smaller distances result in
    higher risk values.

    Args:
        distance_fn: User-defined function that computes distances to obstacles.
            Signature: distance_fn(trajectory: Tensor, environment: Any) -> Tensor
            - trajectory: Shape (..., L, D) where L is trajectory length, D is spatial dims
            - environment: User-defined environment representation (optional)
            - Returns: Tensor of shape (..., L) with distances to nearest obstacle
              at each trajectory point (positive values)
        epsilon: Small value added to distances to avoid division by zero.
            Default: 1e-6
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from vla_metrics.safety import RiskFactor
        >>> import torch
        >>> # Define a simple distance function
        >>> def simple_distance_fn(trajectory, environment=None):
        ...     # Distance to a wall at x=10
        ...     x_coords = trajectory[..., 0]
        ...     return torch.abs(10 - x_coords)
        >>> metric = RiskFactor(distance_fn=simple_distance_fn)
        >>> # Trajectory getting closer to wall: distances [5, 2, 1]
        >>> trajectory = torch.tensor([[5.0, 0.0], [8.0, 0.0], [9.0, 0.0]])
        >>> metric.update(trajectory)
        >>> result = metric.compute()
        >>> # RF = mean([1/5, 1/2, 1/1]) = mean([0.2, 0.5, 1.0]) = 0.567
        >>> result['risk_factor'].item()
        0.5666...

    Example (with environment):
        >>> # Define distance function with environment obstacles
        >>> def obstacle_distance_fn(trajectory, environment):
        ...     # environment contains obstacle positions
        ...     min_distances = torch.full(trajectory.shape[:-1], float('inf'))
        ...     for obs_pos in environment['positions']:
        ...         distances = torch.norm(trajectory - obs_pos, dim=-1)
        ...         min_distances = torch.minimum(min_distances, distances)
        ...     return min_distances
        >>> environment = {
        ...     'positions': [torch.tensor([5.0, 5.0])]
        ... }
        >>> metric = RiskFactor(distance_fn=obstacle_distance_fn)
        >>> trajectory = torch.tensor([[0.0, 0.0], [3.0, 3.0], [4.5, 4.5]])
        >>> metric.update(trajectory, environment=environment)
        >>> result = metric.compute()

    Example (batched):
        >>> # Batch of trajectories
        >>> metric = RiskFactor(distance_fn=simple_distance_fn)
        >>> # Trajectory 1: distances [5, 4, 3]
        >>> # Trajectory 2: distances [2, 1, 0.5]
        >>> trajectories = torch.tensor([
        ...     [[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]],
        ...     [[8.0, 0.0], [9.0, 0.0], [9.5, 0.0]]
        ... ])
        >>> metric.update(trajectories)
        >>> result = metric.compute()
        >>> # Higher risk for trajectory 2 (closer to wall)
    """

    # Metric states that persist across updates
    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = False  # Lower risk is better (safer)

    # Dynamically added by add_state() in __init__
    total_risk: Tensor
    total_steps: Tensor

    def __init__(
        self,
        distance_fn: Callable[[Tensor, Any], Tensor],
        epsilon: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        """Initialize the RiskFactor metric."""
        super().__init__(**kwargs)

        if not callable(distance_fn):
            raise TypeError("distance_fn must be a callable function")

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.distance_fn = distance_fn
        self.epsilon = epsilon

        # Add metric states for distributed computation
        self.add_state("total_risk", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_steps", default=torch.tensor(0), dist_reduce_fx="sum")

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

            environment: Optional environment representation passed to distance_fn.
                Can be any type (dict, object, tensor, etc.) that the user's
                distance function expects.

        Raises:
            ValueError: If trajectory has invalid shape or distances are negative.
            RuntimeError: If distance_fn returns invalid shape or type.

        Example:
            >>> metric = RiskFactor(distance_fn=my_distance_fn)
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

        # Call user's distance function
        try:
            distances = self.distance_fn(trajectory, environment)
        except Exception as e:
            raise RuntimeError(
                f"User-provided distance_fn raised an exception: {e}. "
                f"Ensure distance_fn accepts (trajectory, environment) and returns a tensor."
            ) from e

        # Validate distance output
        if not isinstance(distances, Tensor):
            raise RuntimeError(
                f"distance_fn must return a Tensor, got {type(distances)}. "
                f"Expected shape (..., L) where L is trajectory length."
            )

        # Expected shape is trajectory shape without the last dimension (D)
        expected_shape = trajectory.shape[:-1]  # (..., L)
        if distances.shape != expected_shape:
            raise RuntimeError(
                f"distance_fn returned tensor with shape {distances.shape}, "
                f"expected {expected_shape} (trajectory shape without spatial dimension)"
            )

        # Convert to float and validate
        distances = distances.float()
        if (distances < 0).any():
            raise ValueError(
                "distance_fn returned negative distances. " "Distances must be non-negative values."
            )

        # Compute reciprocal distances (risk at each point)
        # Add epsilon to avoid division by zero
        reciprocal_distances = 1.0 / (distances + self.epsilon)

        # Update states
        self.total_risk += reciprocal_distances.sum()  # pylint: disable=no-member
        self.total_steps += reciprocal_distances.numel()  # pylint: disable=no-member

    def compute(self) -> dict[str, Tensor]:
        """Compute risk factor statistics.

        Returns:
            Dictionary containing:
                - 'risk_factor': Average risk across all trajectory points
                - 'total_risk': Total accumulated risk
                - 'total_steps': Total number of trajectory points

        Raises:
            RuntimeError: If no trajectories have been recorded.

        Example:
            >>> metric = RiskFactor(distance_fn=my_distance_fn)
            >>> metric.update(trajectory)
            >>> result = metric.compute()
            >>> print(f"Risk factor: {result['risk_factor'].item():.4f}")
            >>> # Lower values indicate safer trajectories
        """
        if self.total_steps == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute risk factor: no trajectories have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        risk_factor = self.total_risk / self.total_steps  # pylint: disable=no-member

        return {
            "risk_factor": risk_factor,
            "total_risk": self.total_risk,  # pylint: disable=no-member
            "total_steps": self.total_steps.float(),  # pylint: disable=no-member
        }
