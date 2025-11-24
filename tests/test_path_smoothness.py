"""Tests for PathSmoothness metric."""

import pytest
import torch

from vla_metrics.trajectory_quality import PathSmoothness


class TestPathSmoothness:
    """Test suite for PathSmoothness metric."""

    def test_perfect_straight_line_2d(self) -> None:
        """Test path smoothness for a perfect straight line (should be 0)."""
        metric = PathSmoothness()
        # Perfect straight line - no direction changes
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        metric.update(trajectory)
        result = metric.compute()
        # Perfect smoothness = 0 (no direction changes)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_perfect_straight_line_3d(self) -> None:
        """Test path smoothness for a 3D straight line."""
        metric = PathSmoothness()
        trajectory = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_single_turn_90_degrees(self) -> None:
        """Test path smoothness for a single 90-degree turn."""
        metric = PathSmoothness()
        # Path: (0,0) -> (1,0) -> (2,0) -> (2,1)
        # Turn occurs at (2,0)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
        metric.update(trajectory)
        result = metric.compute()
        # Path length = 3.0, direction change magnitude = sqrt(2)
        # Smoothness = sqrt(2) / 3.0
        expected = torch.sqrt(torch.tensor(2.0)) / 3.0
        assert torch.isclose(result, expected, rtol=1e-5)

    def test_zigzag_path(self) -> None:
        """Test path smoothness for a zigzag pattern."""
        metric = PathSmoothness()
        # Zigzag: (0,0) -> (1,1) -> (2,0) -> (3,1)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        metric.update(trajectory)
        result = metric.compute()
        # Should have high smoothness due to multiple direction changes
        assert result > 0.5  # Non-trivial smoothness value

    def test_smooth_curve_approximation(self) -> None:
        """Test path smoothness for a smooth curve approximation."""
        metric = PathSmoothness()
        # Quarter circle approximation with many points
        num_points = 20
        angles = torch.linspace(0, torch.pi / 2, num_points)
        trajectory = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        metric.update(trajectory)
        result = metric.compute()
        # Smooth curve should have relatively low smoothness
        assert result < 1.0

    def test_u_turn(self) -> None:
        """Test path smoothness for a U-turn."""
        metric = PathSmoothness()
        # U-turn: forward, turn 180 degrees, return
        trajectory = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
        )
        metric.update(trajectory)
        result = metric.compute()
        # U-turn should have significant direction changes
        assert result > 0

    def test_minimum_points(self) -> None:
        """Test with minimum required points (3 points)."""
        metric = PathSmoothness()
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(trajectory)
        result = metric.compute()
        # Straight line with 3 points = perfect smoothness
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_higher_dimensional_trajectory(self) -> None:
        """Test path smoothness for higher-dimensional trajectory."""
        metric = PathSmoothness()
        # 5D trajectory - straight line
        trajectory = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]]
        )
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_duplicate_points(self) -> None:
        """Test trajectory with duplicate points (zero-length segments)."""
        metric = PathSmoothness()
        # Path with duplicate point in the middle
        # Displacements: [1,0], [0,0], [1,0]
        # Direction changes: [-1,0], [1,0] with magnitudes 1.0 each
        # Total change: 2.0, Path length: 2.0, Smoothness: 1.0
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(trajectory)
        result = metric.compute()
        # Duplicate point creates stop-and-go pattern (direction change)
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_multiple_updates(self) -> None:
        """Test metric with multiple trajectory updates."""
        metric = PathSmoothness()
        # First trajectory: perfect straight line
        traj1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(traj1)
        # Second trajectory: also straight line
        traj2 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        metric.update(traj2)
        result = metric.compute()
        # Average of 0.0 and 0.0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_multiple_updates_different_smoothness(self) -> None:
        """Test metric with trajectories of different smoothness."""
        metric = PathSmoothness()
        # First trajectory: perfect straight line (smoothness = 0)
        traj1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        metric.update(traj1)
        smooth1 = metric.compute()
        metric.reset()

        # Second trajectory: with turn (smoothness > 0)
        traj2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
        metric.update(traj2)
        smooth2 = metric.compute()

        # Verify the turn has higher smoothness
        assert smooth2 > smooth1

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = PathSmoothness()
        traj1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
        metric.update(traj1)
        result1 = metric.compute()
        assert result1 > 0

        # Reset and compute with straight line
        metric.reset()
        traj2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(traj2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(0.0), atol=1e-6)

    def test_two_points_raises(self) -> None:
        """Test that two-point trajectory raises ValueError."""
        metric = PathSmoothness()
        with pytest.raises(ValueError, match="at least 3 points"):
            metric.update(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))

    def test_single_point_raises(self) -> None:
        """Test that single-point trajectory raises ValueError."""
        metric = PathSmoothness()
        with pytest.raises(ValueError, match="at least 3 points"):
            metric.update(torch.tensor([[0.0, 0.0]]))

    def test_empty_trajectory_raises(self) -> None:
        """Test that empty trajectory raises ValueError."""
        metric = PathSmoothness()
        with pytest.raises(ValueError, match="at least 3 points"):
            metric.update(torch.tensor([]).reshape(0, 2))

    def test_invalid_shape_raises(self) -> None:
        """Test that invalid tensor shape raises ValueError."""
        metric = PathSmoothness()
        # 1D tensor (invalid - need at least 2 dimensions)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([0.0, 1.0, 2.0]))

    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = PathSmoothness()
        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_different_dtypes(self) -> None:
        """Test with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            metric = PathSmoothness()
            trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=dtype)
            metric.update(trajectory)
            result = metric.compute()
            assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_int_dtype(self) -> None:
        """Test with integer dtype (should work with automatic conversion)."""
        metric = PathSmoothness()
        trajectory = torch.tensor([[0, 0], [1, 0], [2, 0], [3, 0]])
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_large_trajectory(self) -> None:
        """Test with large trajectory to verify numerical stability."""
        metric = PathSmoothness()
        # Create a long straight line
        num_points = 1000
        trajectory = torch.zeros((num_points, 2))
        trajectory[:, 0] = torch.arange(num_points, dtype=torch.float32)
        metric.update(trajectory)
        result = metric.compute()
        # Straight line should have near-zero smoothness
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_gpu_if_available(self) -> None:
        """Test metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = PathSmoothness().to("cuda")
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], device="cuda")
        metric.update(trajectory)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.0, device="cuda"), atol=1e-6)

    def test_mixed_positive_negative_coordinates(self) -> None:
        """Test with trajectories that include negative coordinates."""
        metric = PathSmoothness()
        # Straight diagonal line through origin
        trajectory = torch.tensor([[-2.0, -2.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        metric.update(trajectory)
        result = metric.compute()
        # Should be smooth (straight line)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_very_short_segments(self) -> None:
        """Test numerical stability with very short segments."""
        metric = PathSmoothness()
        # Create trajectory with very small steps
        trajectory = torch.tensor(
            [[0.0, 0.0], [1e-6, 0.0], [2e-6, 0.0], [3e-6, 0.0]], dtype=torch.float64
        )
        metric.update(trajectory)
        result = metric.compute()
        # Straight line should have zero smoothness
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_square_path(self) -> None:
        """Test with a square path (4 right-angle turns)."""
        metric = PathSmoothness()
        # Square path: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        metric.update(trajectory)
        result = metric.compute()
        # Should have significant smoothness due to 90-degree turns
        assert result > 0.2


class TestPathSmoothnessBatched:
    """Test suite for batched PathSmoothness metric."""

    def test_batched_2d_simple(self) -> None:
        """Test simple batch of 2D trajectories - shape (B, L, D)."""
        metric = PathSmoothness()
        # Batch of 2 straight trajectories
        batch = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        metric.update(batch)
        result = metric.compute()
        # Both are straight lines - average smoothness = 0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_mixed_smoothness(self) -> None:
        """Test batch with trajectories of different smoothness."""
        metric = PathSmoothness()
        batch = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],  # straight
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]],  # with turn
            ]
        )
        metric.update(batch)
        result = metric.compute()
        # Average should be > 0 due to the turn in second trajectory
        assert result > 0

    def test_batched_3d(self) -> None:
        """Test batch of 3D trajectories."""
        metric = PathSmoothness()
        batch = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        )
        metric.update(batch)
        result = metric.compute()
        # Both straight lines
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_temporal_batch(self) -> None:
        """Test temporal batch - shape (B, T, L, D)."""
        metric = PathSmoothness()
        # 2 batches, 3 timesteps, 4 points, 2D
        temporal_batch = torch.randn(2, 3, 4, 2)
        # Ensure we can process it
        metric.update(temporal_batch)
        result = metric.compute()
        # Should return a scalar (average across all 2*3=6 trajectories)
        assert result.ndim == 0

    def test_higher_dimensional_batch(self) -> None:
        """Test higher dimensional batch - shape (B1, B2, B3, L, D)."""
        metric = PathSmoothness()
        # Create 5D batch
        batch = torch.randn(2, 3, 2, 5, 3)  # (2, 3, 2, L=5, D=3)
        metric.update(batch)
        result = metric.compute()
        # Should return scalar (average across all 2*3*2=12 trajectories)
        assert result.ndim == 0

    def test_batched_with_known_values(self) -> None:
        """Test batched processing with known smoothness values."""
        metric = PathSmoothness()
        # Create batch of straight lines
        batch = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            ]
        )
        metric.update(batch)
        result = metric.compute()
        # All straight lines - average smoothness = 0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_multiple_updates(self) -> None:
        """Test multiple updates with batched trajectories."""
        metric = PathSmoothness()
        # First batch - straight lines
        batch1 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        metric.update(batch1)
        # Second batch - also straight lines
        batch2 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        metric.update(batch2)
        result = metric.compute()
        # All straight lines
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_mixed_with_single(self) -> None:
        """Test mixing batched and single trajectory updates."""
        metric = PathSmoothness()
        # Single trajectory - straight line
        single = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(single)
        # Batched trajectories - also straight
        batch = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        metric.update(batch)
        result = metric.compute()
        # All straight lines
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_reset(self) -> None:
        """Test reset functionality with batched trajectories."""
        metric = PathSmoothness()
        batch1 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]],  # with turn
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]],  # with turn
            ]
        )
        metric.update(batch1)
        result1 = metric.compute()
        assert result1 > 0

        metric.reset()
        batch2 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],  # straight
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],  # straight
            ]
        )
        metric.update(batch2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(0.0), atol=1e-6)

    def test_batched_invalid_shape_raises(self) -> None:
        """Test that invalid batched shapes raise errors."""
        metric = PathSmoothness()
        # 1D tensor
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([1.0, 2.0, 3.0]))

    def test_batched_insufficient_points_raises(self) -> None:
        """Test that batched tensors with insufficient points raise errors."""
        metric = PathSmoothness()
        # Batch where L=2 (only 2 points per trajectory, need 3)
        batch = torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]]])
        with pytest.raises(ValueError, match="at least 3 points"):
            metric.update(batch)

    def test_large_batch(self) -> None:
        """Test with large batch to verify numerical stability."""
        metric = PathSmoothness()
        # Create large batch of random trajectories
        batch = torch.randn(100, 10, 3)  # 100 trajectories, 10 points each, 3D
        metric.update(batch)
        result = metric.compute()
        # Just verify it computes without error and returns scalar
        assert result.ndim == 0
        assert result >= 0  # Smoothness should be non-negative

    def test_batched_gpu_if_available(self) -> None:
        """Test batched metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = PathSmoothness().to("cuda")
        batch = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ],
            device="cuda",
        )
        metric.update(batch)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.0, device="cuda"), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
