"""Tests for ObstacleProximity metric."""

import pytest
import torch

from robometric_frame.safety import ObstacleProximity


# Simple distance functions for testing
def simple_distance_fn(trajectory, environment=None):
    """Distance to a wall at x=10."""
    x_coords = trajectory[..., 0]
    return torch.abs(10 - x_coords)


def multi_obstacle_distance_fn(trajectory, environment):
    """Distance to nearest of multiple circular obstacles."""
    min_distances = torch.full(trajectory.shape[:-1], float("inf"))
    for obs_pos, obs_radius in zip(environment["positions"], environment["radii"]):
        distances = torch.norm(trajectory - obs_pos, dim=-1)
        # Distance to obstacle surface (not center)
        distances = torch.clamp(distances - obs_radius, min=0.0)
        min_distances = torch.minimum(min_distances, distances)
    return min_distances


class TestObstacleProximity:
    """Test suite for ObstacleProximity metric."""

    def test_basic_single_trajectory(self) -> None:
        """Test with a single trajectory."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)
        # Distances from wall: [5, 3, 1]
        # Min distance: 1.0
        trajectory = torch.tensor([[5.0, 0.0], [7.0, 0.0], [9.0, 0.0]])
        metric.update(trajectory)

        result = metric.compute()
        assert torch.isclose(result["mean_min_distance"], torch.tensor(1.0))
        assert result["num_trajectories"] == 1.0

    def test_multiple_trajectories_batch(self) -> None:
        """Test with batch of trajectories."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)
        # Traj 1 distances: [5, 3, 1] -> min: 1
        # Traj 2 distances: [7, 5, 3] -> min: 3
        # Mean of mins: (1 + 3) / 2 = 2.0
        trajectories = torch.tensor(
            [
                [[5.0, 0.0], [7.0, 0.0], [9.0, 0.0]],
                [[3.0, 0.0], [5.0, 0.0], [7.0, 0.0]],
            ]
        )
        metric.update(trajectories)

        result = metric.compute()
        assert torch.isclose(result["mean_min_distance"], torch.tensor(2.0))
        assert result["num_trajectories"] == 2.0

    def test_with_environment(self) -> None:
        """Test distance computation with environment obstacles."""
        environment = {
            "positions": [torch.tensor([5.0, 5.0])],
            "radii": [1.0],
        }
        metric = ObstacleProximity(distance_fn=multi_obstacle_distance_fn)
        # Point distances from (5,5) with radius 1:
        # (0,0): sqrt(50) - 1 ≈ 6.07
        # (4,4): sqrt(2) - 1 ≈ 0.41
        # (5,5): 0 - 1 = 0 (clamped to 0)
        # Min: 0.0
        trajectory = torch.tensor([[0.0, 0.0], [4.0, 4.0], [5.0, 5.0]])
        metric.update(trajectory, environment=environment)

        result = metric.compute()
        assert torch.isclose(result["mean_min_distance"], torch.tensor(0.0), atol=1e-5)

    def test_multi_batch_updates(self) -> None:
        """Test metric accumulation across multiple batches."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)

        # First batch: single trajectory, distances [2, 1, 5], min = 1
        metric.update(torch.tensor([[8.0, 0.0], [9.0, 0.0], [5.0, 0.0]]))
        # Second batch: 2 trajectories, min distances = 1, 3
        metric.update(
            torch.tensor(
                [
                    [[9.0, 0.0], [8.0, 0.0]],
                    [[7.0, 0.0], [6.0, 0.0]],
                ]
            )
        )

        result = metric.compute()
        # Mean of [1, 1, 3] = 5/3 = 1.6667
        assert torch.isclose(result["mean_min_distance"], torch.tensor(5.0 / 3.0))
        assert result["num_trajectories"] == 3.0

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)
        metric.update(torch.tensor([[5.0, 0.0], [9.0, 0.0]]))

        # Reset and update with new values
        metric.reset()
        metric.update(torch.tensor([[8.0, 0.0], [7.0, 0.0]]))

        result = metric.compute()
        assert torch.isclose(result["mean_min_distance"], torch.tensor(2.0))
        assert result["num_trajectories"] == 1.0

    def test_invalid_trajectory_shape_error(self) -> None:
        """Test that invalid trajectory shape raises an error."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([1.0, 2.0]))  # 1D tensor

    def test_compute_before_update_error(self) -> None:
        """Test that compute() raises error when called before update."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)

        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_non_callable_distance_fn_error(self) -> None:
        """Test that non-callable distance_fn raises an error."""
        with pytest.raises(TypeError, match="must be a callable function"):
            ObstacleProximity(distance_fn="not a function")

    def test_distance_fn_exception_handling(self) -> None:
        """Test proper error handling when distance_fn raises exception."""

        def bad_distance_fn(trajectory, environment=None):
            raise ValueError("Test error")

        metric = ObstacleProximity(distance_fn=bad_distance_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(RuntimeError, match="User-provided distance_fn raised an exception"):
            metric.update(trajectory)

    def test_distance_fn_wrong_return_type(self) -> None:
        """Test error when distance_fn returns wrong type."""

        def bad_return_fn(trajectory, environment=None):
            return [1.0, 2.0, 3.0]  # List instead of tensor

        metric = ObstacleProximity(distance_fn=bad_return_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(RuntimeError, match="must return a Tensor"):
            metric.update(trajectory)

    def test_distance_fn_wrong_shape(self) -> None:
        """Test error when distance_fn returns wrong shape."""

        def wrong_shape_fn(trajectory, environment=None):
            return torch.zeros(2, 3, 2)  # Wrong shape

        metric = ObstacleProximity(distance_fn=wrong_shape_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(RuntimeError, match="returned tensor with shape"):
            metric.update(trajectory)

    def test_negative_distance_error(self) -> None:
        """Test error when distance_fn returns negative distances."""

        def negative_distance_fn(trajectory, environment=None):
            return -torch.ones(trajectory.shape[:-1])

        metric = ObstacleProximity(distance_fn=negative_distance_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match="negative distances"):
            metric.update(trajectory)

    def test_zero_distances(self) -> None:
        """Test with zero distances (touching obstacles)."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)
        # All points at x=10 (zero distance to wall)
        trajectory = torch.tensor([[10.0, 0.0], [10.0, 1.0], [10.0, 2.0]])
        metric.update(trajectory)

        result = metric.compute()
        assert torch.isclose(result["mean_min_distance"], torch.tensor(0.0))

    def test_large_distances(self) -> None:
        """Test with very large distances."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)
        # Very far from wall at x=10
        trajectory = torch.tensor([[-1000.0, 0.0], [-500.0, 0.0]])
        metric.update(trajectory)

        result = metric.compute()
        assert torch.isclose(result["mean_min_distance"], torch.tensor(510.0))

    def test_higher_is_better_true(self) -> None:
        """Test that higher_is_better is set to True."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)
        assert metric.higher_is_better is True

    def test_is_differentiable_false(self) -> None:
        """Test that is_differentiable is set to False."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)
        assert metric.is_differentiable is False

    def test_multiple_obstacles(self) -> None:
        """Test with multiple obstacles in environment."""
        environment = {
            "positions": [
                torch.tensor([2.0, 2.0]),
                torch.tensor([8.0, 8.0]),
            ],
            "radii": [0.5, 0.5],
        }
        metric = ObstacleProximity(distance_fn=multi_obstacle_distance_fn)
        # Point (5,5) is equidistant from both obstacles
        trajectory = torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        metric.update(trajectory, environment=environment)

        result = metric.compute()
        # Should compute min distance to nearest obstacle at each point
        assert result["mean_min_distance"] > 0.0

    def test_batched_with_extra_dimensions(self) -> None:
        """Test with extra batch dimensions."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn)
        # Shape: (2, 3, 4, 2) - 2 batches, 3 sub-batches, 4 points, 2D space
        trajectories = torch.randn(2, 3, 4, 2)
        trajectories[..., 0] = torch.clamp(trajectories[..., 0], -5, 15)  # Keep near wall

        metric.update(trajectories)
        result = metric.compute()

        # Should have 2*3 = 6 trajectories
        assert result["num_trajectories"] == 6.0
        assert result["mean_min_distance"] >= 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self) -> None:
        """Test metric on GPU."""
        metric = ObstacleProximity(distance_fn=simple_distance_fn).to("cuda")
        trajectory = torch.tensor([[5.0, 0.0], [9.0, 0.0], [7.0, 0.0]], device="cuda")
        metric.update(trajectory)

        result = metric.compute()
        assert result["mean_min_distance"].device.type == "cuda"
