"""Tests for RiskFactor metric."""

import pytest
import torch

from robometric_frame.safety import RiskFactor


# Simple distance functions for testing
def simple_distance_fn(trajectory, environment=None):
    """Distance to a wall at x=10."""
    x_coords = trajectory[..., 0]
    return torch.abs(10 - x_coords)


def circular_obstacle_distance_fn(trajectory, environment):
    """Distance to circular obstacles."""
    min_distances = torch.full(trajectory.shape[:-1], float("inf"))
    for obs_pos, obs_radius in zip(environment["positions"], environment["radii"]):
        distances = torch.norm(trajectory - obs_pos, dim=-1)
        # Distance to obstacle surface
        distances = torch.clamp(distances - obs_radius, min=0.0)
        min_distances = torch.minimum(min_distances, distances)
    return min_distances


class TestRiskFactor:
    """Test suite for RiskFactor metric."""

    def test_basic_single_trajectory(self) -> None:
        """Test with a single trajectory."""
        metric = RiskFactor(distance_fn=simple_distance_fn, epsilon=1e-6)
        # Distances: [5, 2, 1]
        # Risk: [1/5, 1/2, 1/1] = [0.2, 0.5, 1.0]
        # Mean risk: (0.2 + 0.5 + 1.0) / 3 = 0.5667
        trajectory = torch.tensor([[5.0, 0.0], [8.0, 0.0], [9.0, 0.0]])
        metric.update(trajectory)

        result = metric.compute()
        expected_risk = (1.0 / 5.0 + 1.0 / 2.0 + 1.0 / 1.0) / 3.0
        assert torch.isclose(result["risk_factor"], torch.tensor(expected_risk), atol=1e-4)
        assert result["total_steps"] == 3.0

    def test_closer_trajectory_higher_risk(self) -> None:
        """Test that closer trajectories have higher risk."""
        metric1 = RiskFactor(distance_fn=simple_distance_fn)
        metric2 = RiskFactor(distance_fn=simple_distance_fn)

        # Far trajectory
        far_trajectory = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        metric1.update(far_trajectory)

        # Close trajectory
        close_trajectory = torch.tensor([[8.0, 0.0], [9.0, 0.0], [9.5, 0.0]])
        metric2.update(close_trajectory)

        far_risk = metric1.compute()["risk_factor"]
        close_risk = metric2.compute()["risk_factor"]

        assert close_risk > far_risk

    def test_with_environment(self) -> None:
        """Test risk computation with environment obstacles."""
        environment = {
            "positions": [torch.tensor([5.0, 5.0])],
            "radii": [1.0],
        }
        metric = RiskFactor(distance_fn=circular_obstacle_distance_fn)
        # Trajectory approaching obstacle
        trajectory = torch.tensor([[0.0, 0.0], [3.0, 3.0], [4.5, 4.5]])
        metric.update(trajectory, environment=environment)

        result = metric.compute()
        assert result["risk_factor"] > 0.0
        assert result["total_steps"] == 3.0

    def test_batched_trajectories(self) -> None:
        """Test with batch of trajectories."""
        metric = RiskFactor(distance_fn=simple_distance_fn)
        # Traj 1 distances: [5, 3, 1] -> risks: [0.2, 0.333, 1.0]
        # Traj 2 distances: [7, 5, 3] -> risks: [0.143, 0.2, 0.333]
        trajectories = torch.tensor(
            [
                [[5.0, 0.0], [7.0, 0.0], [9.0, 0.0]],
                [[3.0, 0.0], [5.0, 0.0], [7.0, 0.0]],
            ]
        )
        metric.update(trajectories)

        result = metric.compute()
        # Total 6 points
        assert result["total_steps"] == 6.0
        assert result["risk_factor"] > 0.0

    def test_multi_batch_updates(self) -> None:
        """Test metric accumulation across multiple batches."""
        metric = RiskFactor(distance_fn=simple_distance_fn)

        # First batch: distances [5, 2]
        metric.update(torch.tensor([[5.0, 0.0], [8.0, 0.0]]))
        # Second batch: distances [1, 3]
        metric.update(torch.tensor([[9.0, 0.0], [7.0, 0.0]]))

        result = metric.compute()
        # Risks: [1/5, 1/2, 1/1, 1/3] = [0.2, 0.5, 1.0, 0.333]
        # Mean: (0.2 + 0.5 + 1.0 + 0.333) / 4 = 0.508
        expected_risk = (1.0 / 5.0 + 1.0 / 2.0 + 1.0 / 1.0 + 1.0 / 3.0) / 4.0
        assert torch.isclose(result["risk_factor"], torch.tensor(expected_risk), atol=1e-3)
        assert result["total_steps"] == 4.0

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = RiskFactor(distance_fn=simple_distance_fn)
        metric.update(torch.tensor([[9.0, 0.0], [9.5, 0.0]]))

        # Reset and update with new values
        metric.reset()
        metric.update(torch.tensor([[5.0, 0.0], [5.0, 0.0]]))

        result = metric.compute()
        expected_risk = 1.0 / 5.0
        assert torch.isclose(result["risk_factor"], torch.tensor(expected_risk), atol=1e-5)
        assert result["total_steps"] == 2.0

    def test_epsilon_prevents_division_by_zero(self) -> None:
        """Test that epsilon prevents division by zero."""
        epsilon = 1e-3
        metric = RiskFactor(distance_fn=simple_distance_fn, epsilon=epsilon)
        # Point exactly at wall (distance = 0)
        trajectory = torch.tensor([[10.0, 0.0], [10.0, 1.0]])
        metric.update(trajectory)

        result = metric.compute()
        # Risk at wall: 1 / epsilon
        expected_risk = 1.0 / epsilon
        assert torch.isclose(result["risk_factor"], torch.tensor(expected_risk), atol=1.0)

    def test_custom_epsilon(self) -> None:
        """Test custom epsilon value."""
        epsilon = 0.1
        metric = RiskFactor(distance_fn=simple_distance_fn, epsilon=epsilon)
        # Distance = 0.05 (very close to wall)
        trajectory = torch.tensor([[9.95, 0.0]])
        metric.update(trajectory)

        result = metric.compute()
        # Risk = 1 / (0.05 + 0.1) = 1 / 0.15 = 6.667
        expected_risk = 1.0 / (0.05 + epsilon)
        assert torch.isclose(result["risk_factor"], torch.tensor(expected_risk), atol=1e-3)

    def test_invalid_trajectory_shape_error(self) -> None:
        """Test that invalid trajectory shape raises an error."""
        metric = RiskFactor(distance_fn=simple_distance_fn)

        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([1.0, 2.0]))  # 1D tensor

    def test_compute_before_update_error(self) -> None:
        """Test that compute() raises error when called before update."""
        metric = RiskFactor(distance_fn=simple_distance_fn)

        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_non_callable_distance_fn_error(self) -> None:
        """Test that non-callable distance_fn raises an error."""
        with pytest.raises(TypeError, match="must be a callable function"):
            RiskFactor(distance_fn="not a function")

    def test_invalid_epsilon_error(self) -> None:
        """Test that invalid epsilon raises an error."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            RiskFactor(distance_fn=simple_distance_fn, epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            RiskFactor(distance_fn=simple_distance_fn, epsilon=-0.1)

    def test_distance_fn_exception_handling(self) -> None:
        """Test proper error handling when distance_fn raises exception."""

        def bad_distance_fn(trajectory, environment=None):
            raise ValueError("Test error")

        metric = RiskFactor(distance_fn=bad_distance_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(RuntimeError, match="User-provided distance_fn raised an exception"):
            metric.update(trajectory)

    def test_distance_fn_wrong_return_type(self) -> None:
        """Test error when distance_fn returns wrong type."""

        def bad_return_fn(trajectory, environment=None):
            return [1.0, 2.0, 3.0]  # List instead of tensor

        metric = RiskFactor(distance_fn=bad_return_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(RuntimeError, match="must return a Tensor"):
            metric.update(trajectory)

    def test_distance_fn_wrong_shape(self) -> None:
        """Test error when distance_fn returns wrong shape."""

        def wrong_shape_fn(trajectory, environment=None):
            return torch.zeros(2, 3, 2)  # Wrong shape

        metric = RiskFactor(distance_fn=wrong_shape_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(RuntimeError, match="returned tensor with shape"):
            metric.update(trajectory)

    def test_negative_distance_error(self) -> None:
        """Test error when distance_fn returns negative distances."""

        def negative_distance_fn(trajectory, environment=None):
            return -torch.ones(trajectory.shape[:-1])

        metric = RiskFactor(distance_fn=negative_distance_fn)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match="negative distances"):
            metric.update(trajectory)

    def test_uniform_distances(self) -> None:
        """Test with uniform distances."""
        metric = RiskFactor(distance_fn=simple_distance_fn)
        # All points at x=5, distance = 5 from wall
        trajectory = torch.tensor([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])
        metric.update(trajectory)

        result = metric.compute()
        expected_risk = 1.0 / 5.0
        assert torch.isclose(result["risk_factor"], torch.tensor(expected_risk), atol=1e-5)

    def test_higher_is_better_false(self) -> None:
        """Test that higher_is_better is set to False."""
        metric = RiskFactor(distance_fn=simple_distance_fn)
        assert metric.higher_is_better is False

    def test_is_differentiable_false(self) -> None:
        """Test that is_differentiable is set to False."""
        metric = RiskFactor(distance_fn=simple_distance_fn)
        assert metric.is_differentiable is False

    def test_very_large_distances(self) -> None:
        """Test with very large distances (low risk)."""
        metric = RiskFactor(distance_fn=simple_distance_fn)
        # Very far from wall
        trajectory = torch.tensor([[-1000.0, 0.0], [-500.0, 0.0]])
        metric.update(trajectory)

        result = metric.compute()
        # Risk should be very small
        assert result["risk_factor"] < 0.01

    def test_mixed_distances(self) -> None:
        """Test with mix of close and far distances."""
        metric = RiskFactor(distance_fn=simple_distance_fn)
        # Distances: [0.1, 10, 0.5]
        # Risks: [10, 0.1, 2]
        # Mean: 4.033
        trajectory = torch.tensor([[9.9, 0.0], [0.0, 0.0], [9.5, 0.0]])
        metric.update(trajectory)

        result = metric.compute()
        expected_risk = (1.0 / 0.1 + 1.0 / 10.0 + 1.0 / 0.5) / 3.0
        assert torch.isclose(result["risk_factor"], torch.tensor(expected_risk), atol=1e-2)

    def test_multiple_obstacles(self) -> None:
        """Test with multiple obstacles in environment."""
        environment = {
            "positions": [
                torch.tensor([2.0, 2.0]),
                torch.tensor([8.0, 8.0]),
            ],
            "radii": [0.5, 0.5],
        }
        metric = RiskFactor(distance_fn=circular_obstacle_distance_fn)
        trajectory = torch.tensor([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        metric.update(trajectory, environment=environment)

        result = metric.compute()
        assert result["risk_factor"] > 0.0
        assert result["total_steps"] == 3.0

    def test_batched_with_extra_dimensions(self) -> None:
        """Test with extra batch dimensions."""
        metric = RiskFactor(distance_fn=simple_distance_fn)
        # Shape: (2, 3, 4, 2) - 2 batches, 3 sub-batches, 4 points, 2D space
        trajectories = torch.randn(2, 3, 4, 2)
        trajectories[..., 0] = torch.clamp(trajectories[..., 0], -5, 15)  # Keep near wall

        metric.update(trajectories)
        result = metric.compute()

        # Should have 2*3*4 = 24 points
        assert result["total_steps"] == 24.0
        assert result["risk_factor"] > 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self) -> None:
        """Test metric on GPU."""
        metric = RiskFactor(distance_fn=simple_distance_fn).to("cuda")
        trajectory = torch.tensor([[5.0, 0.0], [8.0, 0.0], [9.0, 0.0]], device="cuda")
        metric.update(trajectory)

        result = metric.compute()
        assert result["risk_factor"].device.type == "cuda"
