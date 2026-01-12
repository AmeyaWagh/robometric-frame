"""Tests for CollisionRate metric."""

import pytest
import torch

from vla_metrics.safety import CollisionRate


# Simple collision detection functions for testing
def simple_collision_fn(trajectory, environment=None):
    """Check if any point exceeds bounds [-5, 5]."""
    return ((trajectory.abs() > 5).any(dim=-1)).float()


def obstacle_collision_fn(trajectory, environment):
    """Check collisions with circular obstacles."""
    collisions = torch.zeros(trajectory.shape[:-1], dtype=torch.bool)
    for obs_pos, obs_radius in zip(environment["positions"], environment["radii"]):
        distances = torch.norm(trajectory - obs_pos, dim=-1)
        collisions |= distances < obs_radius
    return collisions.float()


class TestCollisionRate:
    """Test suite for CollisionRate metric."""

    def test_basic_no_collisions(self) -> None:
        """Test trajectory with no collisions."""
        metric = CollisionRate(collision_fn=simple_collision_fn)
        # All points within bounds
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        metric.update(trajectory)

        result = metric.compute()
        assert result["collision_rate"] == 0.0
        assert result["total_collisions"] == 0.0
        assert result["total_steps"] == 3.0

    def test_basic_with_collisions(self) -> None:
        """Test trajectory with some collisions."""
        metric = CollisionRate(collision_fn=simple_collision_fn)
        # Points: [no, no, yes (6>5), no]
        trajectory = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 2.0], [1.0, 1.0]])
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
        metric = CollisionRate(collision_fn=obstacle_collision_fn)
        # Point at (2.0, 2.0) collides with obstacle
        trajectory = torch.tensor([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
        metric.update(trajectory, environment=environment)

        result = metric.compute()
        assert result["total_collisions"] == 1.0

    def test_batched_trajectories(self) -> None:
        """Test with batch of trajectories."""
        metric = CollisionRate(collision_fn=simple_collision_fn)
        # Batch of 2 trajectories, 3 points each
        # Traj 1: all safe
        # Traj 2: 2 collisions
        trajectories = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                [[6.0, 6.0], [7.0, 7.0], [1.0, 1.0]],
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
        metric = CollisionRate(collision_fn=simple_collision_fn)

        # First batch: 1/3 collision
        metric.update(torch.tensor([[0.0, 0.0], [6.0, 0.0], [1.0, 1.0]]))
        # Second batch: 0/2 collisions
        metric.update(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))

        result = metric.compute()
        # 1 collision out of 5 total points
        assert torch.isclose(result["collision_rate"], torch.tensor(0.2))
        assert result["total_collisions"] == 1.0
        assert result["total_steps"] == 5.0

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = CollisionRate(collision_fn=simple_collision_fn)
        metric.update(torch.tensor([[6.0, 0.0], [7.0, 0.0]]))

        # Reset and update with new values
        metric.reset()
        metric.update(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))

        result = metric.compute()
        assert result["collision_rate"] == 0.0
        assert result["total_steps"] == 2.0

    def test_invalid_trajectory_shape_error(self) -> None:
        """Test that invalid trajectory shape raises an error."""
        metric = CollisionRate(collision_fn=simple_collision_fn)

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([1.0, 2.0]))  # 1D tensor

    def test_compute_before_update_error(self) -> None:
        """Test that compute() raises error when called before update."""
        metric = CollisionRate(collision_fn=simple_collision_fn)

        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_non_callable_collision_fn_error(self) -> None:
        """Test that non-callable collision_fn raises an error."""
        with pytest.raises(TypeError, match="must be a callable function"):
            CollisionRate(collision_fn="not a function")

    def test_collision_fn_exception_handling(self) -> None:
        """Test proper error handling when collision_fn raises exception."""

        def bad_collision_fn(trajectory, environment=None):
            raise ValueError("Test error")

        metric = CollisionRate(collision_fn=bad_collision_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(RuntimeError, match="User-provided collision_fn raised an exception"):
            metric.update(trajectory)

    def test_collision_fn_wrong_return_type(self) -> None:
        """Test error when collision_fn returns wrong type."""

        def bad_return_fn(trajectory, environment=None):
            return [1, 0, 1]  # List instead of tensor

        metric = CollisionRate(collision_fn=bad_return_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(RuntimeError, match="must return a Tensor"):
            metric.update(trajectory)

    def test_collision_fn_wrong_shape(self) -> None:
        """Test error when collision_fn returns wrong shape."""

        def wrong_shape_fn(trajectory, environment=None):
            return torch.zeros(2, 3, 2)  # Wrong shape

        metric = CollisionRate(collision_fn=wrong_shape_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(RuntimeError, match="returned tensor with shape"):
            metric.update(trajectory)

    def test_boolean_collision_output(self) -> None:
        """Test that boolean collision output is handled correctly."""

        def bool_collision_fn(trajectory, environment=None):
            return (trajectory[..., 0] > 5).bool()

        metric = CollisionRate(collision_fn=bool_collision_fn)
        trajectory = torch.tensor([[0.0, 0.0], [6.0, 0.0], [7.0, 0.0]])
        metric.update(trajectory)

        result = metric.compute()
        assert torch.isclose(result["collision_rate"], torch.tensor(2.0 / 3.0))

    def test_all_collisions(self) -> None:
        """Test trajectory where all points collide."""
        metric = CollisionRate(collision_fn=simple_collision_fn)
        trajectory = torch.tensor([[10.0, 0.0], [15.0, 0.0], [20.0, 0.0]])
        metric.update(trajectory)

        result = metric.compute()
        assert result["collision_rate"] == 1.0
        assert result["collision_percentage"] == 100.0

    def test_higher_is_better_false(self) -> None:
        """Test that higher_is_better is set to False."""
        metric = CollisionRate(collision_fn=simple_collision_fn)
        assert metric.higher_is_better is False

    def test_is_differentiable_false(self) -> None:
        """Test that is_differentiable is set to False."""
        metric = CollisionRate(collision_fn=simple_collision_fn)
        assert metric.is_differentiable is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self) -> None:
        """Test metric on GPU."""
        metric = CollisionRate(collision_fn=simple_collision_fn).to("cuda")
        trajectory = torch.tensor([[0.0, 0.0], [6.0, 0.0], [1.0, 1.0]], device="cuda")
        metric.update(trajectory)

        result = metric.compute()
        assert result["collision_rate"].device.type == "cuda"
