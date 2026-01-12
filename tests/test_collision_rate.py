"""Tests for CollisionRate metric."""

import pytest
import torch

from vla_metrics.safety import CollisionRate


# Simple distance functions for testing
def simple_distance_fn(trajectory, environment=None):
    """Compute distance to bounds at ±5."""
    # Distance to nearest wall at ±5
    x_coords = trajectory[..., 0]
    y_coords = trajectory[..., 1]
    dist_x = torch.minimum(torch.abs(x_coords - 5), torch.abs(x_coords + 5))
    dist_y = torch.minimum(torch.abs(y_coords - 5), torch.abs(y_coords + 5))
    # Minimum distance to any wall
    return torch.minimum(dist_x, dist_y)


def obstacle_distance_fn(trajectory, environment):
    """Compute distance to circular obstacles."""
    min_distances = torch.full(trajectory.shape[:-1], float("inf"))
    for obs_pos, obs_radius in zip(environment["positions"], environment["radii"]):
        # Distance to obstacle surface
        distances = torch.norm(trajectory - obs_pos, dim=-1) - obs_radius
        min_distances = torch.minimum(min_distances, distances)
    # Clamp to ensure non-negative
    return torch.clamp(min_distances, min=0.0)


class TestCollisionRate:
    """Test suite for CollisionRate metric."""

    def test_basic_no_collisions(self) -> None:
        """Test trajectory with no collisions."""
        metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=0.0)
        # All points within bounds
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        metric.update(trajectory)

        result = metric.compute()
        assert result["collision_rate"] == 0.0
        assert result["total_collisions"] == 0.0
        assert result["total_steps"] == 3.0

    def test_basic_with_collisions(self) -> None:
        """Test trajectory with some collisions."""
        metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=0.0)
        # Points at exactly the boundary (distance = 0) should be collisions
        # Point at (5, 2) has distance 0 to x=5 wall
        trajectory = torch.tensor([[0.0, 0.0], [3.0, 4.0], [5.0, 2.0], [1.0, 1.0]])
        metric.update(trajectory)

        result = metric.compute()
        assert torch.isclose(result["collision_rate"], torch.tensor(0.25))
        assert result["total_collisions"] == 1.0
        assert result["total_steps"] == 4.0
        assert torch.isclose(result["collision_percentage"], torch.tensor(25.0))

    def test_with_environment(self) -> None:
        """Test collision detection with environment obstacles."""
        environment = {
            "positions": [torch.tensor([2.0, 2.0])],
            "radii": [0.5],
        }
        metric = CollisionRate(distance_fn=obstacle_distance_fn, collision_threshold=0.0)
        # Point at (2.0, 2.0) is at obstacle center, distance = -0.5, clamped to 0
        trajectory = torch.tensor([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
        metric.update(trajectory, environment=environment)

        result = metric.compute()
        assert result["total_collisions"] == 1.0

    def test_batched_trajectories(self) -> None:
        """Test with batch of trajectories."""
        metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=0.0)
        # Batch of 2 trajectories, 3 points each
        # Traj 1: all safe
        # Traj 2: 2 collisions (at x=5 and x=-5 boundaries)
        trajectories = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                [[5.0, 0.0], [-5.0, 0.0], [1.0, 1.0]],
            ]
        )
        metric.update(trajectories)

        result = metric.compute()
        # 2 collisions out of 6 total points
        assert torch.isclose(result["collision_rate"], torch.tensor(2.0 / 6.0))
        assert result["total_collisions"] == 2.0
        assert result["total_steps"] == 6.0

    def test_multi_batch_updates(self) -> None:
        """Test metric accumulation across multiple batches."""
        metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=0.0)

        # First batch: 1/3 collision (at x=5)
        metric.update(torch.tensor([[0.0, 0.0], [5.0, 0.0], [1.0, 1.0]]))
        # Second batch: 0/2 collisions
        metric.update(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))

        result = metric.compute()
        # 1 collision out of 5 total points
        assert torch.isclose(result["collision_rate"], torch.tensor(0.2))
        assert result["total_collisions"] == 1.0
        assert result["total_steps"] == 5.0

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=0.0)
        metric.update(torch.tensor([[5.0, 0.0], [-5.0, 0.0]]))

        # Reset and update with new values
        metric.reset()
        metric.update(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))

        result = metric.compute()
        assert result["collision_rate"] == 0.0
        assert result["total_steps"] == 2.0

    def test_invalid_trajectory_shape_error(self) -> None:
        """Test that invalid trajectory shape raises an error."""
        metric = CollisionRate(distance_fn=simple_distance_fn)

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([1.0, 2.0]))  # 1D tensor

    def test_compute_before_update_error(self) -> None:
        """Test that compute() raises error when called before update."""
        metric = CollisionRate(distance_fn=simple_distance_fn)

        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_non_callable_distance_fn_error(self) -> None:
        """Test that non-callable distance_fn raises an error."""
        with pytest.raises(TypeError, match="must be a callable function"):
            CollisionRate(distance_fn="not a function")

    def test_distance_fn_exception_handling(self) -> None:
        """Test proper error handling when distance_fn raises exception."""

        def bad_distance_fn(trajectory, environment=None):
            raise ValueError("Test error")

        metric = CollisionRate(distance_fn=bad_distance_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(RuntimeError, match="User-provided distance_fn raised an exception"):
            metric.update(trajectory)

    def test_distance_fn_wrong_return_type(self) -> None:
        """Test error when distance_fn returns wrong type."""

        def bad_return_fn(trajectory, environment=None):
            return [1.0, 0.5, 1.2]  # List instead of tensor

        metric = CollisionRate(distance_fn=bad_return_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(RuntimeError, match="must return a Tensor"):
            metric.update(trajectory)

    def test_distance_fn_wrong_shape(self) -> None:
        """Test error when distance_fn returns wrong shape."""

        def wrong_shape_fn(trajectory, environment=None):
            return torch.zeros(2, 3, 2)  # Wrong shape

        metric = CollisionRate(distance_fn=wrong_shape_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(RuntimeError, match="returned tensor with shape"):
            metric.update(trajectory)

    def test_negative_distance_error(self) -> None:
        """Test error when distance_fn returns negative distances."""

        def negative_distance_fn(trajectory, environment=None):
            return torch.full(trajectory.shape[:-1], -1.0)

        metric = CollisionRate(distance_fn=negative_distance_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match="negative distances"):
            metric.update(trajectory)

    def test_collision_threshold(self) -> None:
        """Test collision detection with custom threshold."""
        metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=1.0)
        # Point at (4.0, 0) has distance 1.0 to x=5 wall
        # Point at (3.5, 0) has distance 1.5 to x=5 wall
        trajectory = torch.tensor([[0.0, 0.0], [4.0, 0.0], [3.5, 0.0]])
        metric.update(trajectory)

        result = metric.compute()
        # 1 collision (distance <= 1.0)
        assert torch.isclose(result["collision_rate"], torch.tensor(1.0 / 3.0))

    def test_all_collisions(self) -> None:
        """Test trajectory where all points collide."""
        metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=0.0)
        # All points at boundaries
        trajectory = torch.tensor([[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0]])
        metric.update(trajectory)

        result = metric.compute()
        assert result["collision_rate"] == 1.0
        assert result["collision_percentage"] == 100.0

    def test_higher_is_better_false(self) -> None:
        """Test that higher_is_better is set to False."""
        metric = CollisionRate(distance_fn=simple_distance_fn)
        assert metric.higher_is_better is False

    def test_is_differentiable_false(self) -> None:
        """Test that is_differentiable is set to False."""
        metric = CollisionRate(distance_fn=simple_distance_fn)
        assert metric.is_differentiable is False

    def test_invalid_collision_threshold_error(self) -> None:
        """Test that negative collision_threshold raises an error."""
        with pytest.raises(ValueError, match="collision_threshold must be non-negative"):
            CollisionRate(distance_fn=simple_distance_fn, collision_threshold=-1.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self) -> None:
        """Test metric on GPU."""
        metric = CollisionRate(distance_fn=simple_distance_fn, collision_threshold=0.0).to("cuda")
        trajectory = torch.tensor([[0.0, 0.0], [5.0, 0.0], [1.0, 1.0]], device="cuda")
        metric.update(trajectory)

        result = metric.compute()
        assert result["collision_rate"].device.type == "cuda"
