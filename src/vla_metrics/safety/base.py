"""Base class for VLA safety metrics.

Provides common functionality for trajectory validation and distance function
calling patterns shared across all safety metrics.
"""

from abc import abstractmethod
from typing import Any, Callable, Optional

from torch import Tensor
from torchmetrics import Metric


class BaseSafetyMetric(Metric):
    """Base class for safety metrics with trajectory validation and distance function calling.

    This base class provides common functionality for safety metrics that use
    user-defined distance functions to evaluate trajectories against environments:

    - Trajectory validation (shape and dimensionality checks)
    - Safe calling of user-provided distance functions with error handling
    - Output validation (type, shape, and non-negativity constraints)
    - Collision detection based on distance thresholds
    - Standardized signatures for distance functions

    All safety metrics use distance functions as the fundamental building block.
    Collision information can be derived from distances using threshold-based
    detection via `_detect_collisions()`.

    Subclasses should:
    1. Implement metric-specific logic in `update()` and `compute()`
    2. Call `_compute_distances()` to get validated distance values
    3. Use `_detect_collisions()` for threshold-based collision detection

    Args:
        distance_fn: User-defined function that computes distances to obstacles.
            Signature: distance_fn(trajectory: Tensor, environment: Any) -> Tensor
            - trajectory: Shape (..., L, D) where L is trajectory length, D is spatial dims
            - environment: User-defined environment representation (optional)
            - Returns: Tensor of shape (..., L) with distances to nearest obstacle
              at each trajectory point (positive values)
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Raises:
        TypeError: If distance_fn is not callable.
    """

    def __init__(
        self,
        distance_fn: Callable[[Tensor, Any], Tensor],
        **kwargs: Any,
    ) -> None:
        """Initialize the base safety metric."""
        super().__init__(**kwargs)

        if not callable(distance_fn):
            raise TypeError("distance_fn must be a callable function")

        self.distance_fn = distance_fn

    def _validate_trajectory(self, trajectory: Tensor) -> None:
        """Validate trajectory shape and dimensionality.

        Args:
            trajectory: Trajectory tensor to validate.

        Raises:
            ValueError: If trajectory has invalid shape (must be at least 2D).

        Example:
            >>> self._validate_trajectory(torch.randn(10, 2))  # Valid: (L, D)
            >>> self._validate_trajectory(torch.randn(5, 10, 2))  # Valid: (B, L, D)
            >>> self._validate_trajectory(torch.randn(10))  # Invalid: 1D
            Traceback (most recent call last):
                ...
            ValueError: Trajectory must have at least 2 dimensions ...
        """
        if trajectory.ndim < 2:
            raise ValueError(
                f"Trajectory must have at least 2 dimensions (..., L, D), "
                f"got {trajectory.ndim}D tensor with shape {trajectory.shape}"
            )

    def _compute_distances(
        self,
        trajectory: Tensor,
        environment: Optional[Any] = None,
    ) -> Tensor:
        """Compute and validate distances from user's distance function.

        This method:
        1. Calls the user-provided distance function
        2. Validates the return type is a Tensor
        3. Validates the output shape matches trajectory shape (minus last dim)
        4. Validates distances are non-negative
        5. Converts output to float dtype

        Args:
            trajectory: Trajectory tensor of shape (..., L, D).
            environment: Optional environment representation.

        Returns:
            Validated distance tensor of shape (..., L) with non-negative float values.

        Raises:
            RuntimeError: If distance_fn raises an exception, returns wrong type,
                or returns wrong shape.
            ValueError: If distance_fn returns negative values.

        Example:
            >>> trajectory = torch.randn(10, 2)
            >>> distances = self._compute_distances(trajectory, env)
            >>> distances.shape
            torch.Size([10])

            >>> # Batched trajectories
            >>> trajectory = torch.randn(5, 10, 3)
            >>> distances = self._compute_distances(trajectory, env)
            >>> distances.shape  # Shape is (5, 10) - batch and trajectory length
            torch.Size([5, 10])
        """
        # Call user's distance function with exception handling
        try:
            distances = self.distance_fn(trajectory, environment)
        except Exception as e:
            raise RuntimeError(
                f"User-provided distance_fn raised an exception: {e}. "
                f"Ensure distance_fn accepts (trajectory, environment) and returns a tensor."
            ) from e

        # Validate output is a Tensor
        if not isinstance(distances, Tensor):
            raise RuntimeError(
                f"distance_fn must return a Tensor, got {type(distances)}. "
                f"Expected shape (..., L) where L is trajectory length."
            )

        # Validate output shape matches trajectory shape (without spatial dimension)
        expected_shape = trajectory.shape[:-1]  # (..., L)
        if distances.shape != expected_shape:
            raise RuntimeError(
                f"distance_fn returned tensor with shape {distances.shape}, "
                f"expected {expected_shape} (trajectory shape without spatial dimension)"
            )

        # Convert to float for computation
        distances = distances.float()

        # Validate non-negativity
        if (distances < 0).any():
            raise ValueError(
                "distance_fn returned negative distances. Distances must be non-negative values."
            )

        return distances

    def _detect_collisions(
        self,
        distances: Tensor,
        threshold: float = 0.0,
    ) -> Tensor:
        """Detect collisions based on distance threshold.

        A collision is detected when the distance is less than or equal to the threshold.
        By default, threshold=0.0 means any contact (distance <= 0) is a collision.

        Args:
            distances: Distance tensor of shape (..., L) from `_compute_distances()`.
            threshold: Distance threshold for collision detection. Distances less than
                or equal to this value are considered collisions. Default: 0.0

        Returns:
            Binary collision tensor of shape (..., L) where 1 indicates collision
            and 0 indicates no collision.

        Example:
            >>> distances = torch.tensor([2.0, 0.5, 0.0, -0.1, 1.0])
            >>> collisions = self._detect_collisions(distances, threshold=0.5)
            >>> collisions  # [0, 1, 1, 1, 0] - distances <= 0.5
            tensor([0., 1., 1., 1., 0.])

            >>> # Batched
            >>> distances = torch.tensor([[1.0, 0.0], [0.3, 0.8]])
            >>> collisions = self._detect_collisions(distances, threshold=0.5)
            >>> collisions
            tensor([[0., 1.], [1., 0.]])
        """
        return (distances <= threshold).float()

    @abstractmethod
    def update(
        self,
        trajectory: Tensor,
        environment: Optional[Any] = None,
    ) -> None:
        """Update metric state with new trajectory data.

        Subclasses must implement this method to define how to update
        their specific metric states.

        Args:
            trajectory: Trajectory tensor of shape (..., L, D).
            environment: Optional environment representation.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict[str, Tensor]:
        """Compute final metric value(s).

        Subclasses must implement this method to define how to compute
        their final metric values from accumulated states.

        Returns:
            Dictionary mapping metric names to their computed values.
        """
        raise NotImplementedError
