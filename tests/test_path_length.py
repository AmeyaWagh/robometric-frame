"""Tests for PathLength metric."""

import pytest
import torch

from vla_metrics.trajectory_quality import PathLength


class TestPathLength:
    """Test suite for PathLength metric."""

    def test_simple_2d_straight_line(self) -> None:
        """Test path length for a simple 2D straight line."""
        metric = PathLength()
        # Straight line from (0,0) to (3,0)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(3.0))

    def test_simple_2d_path(self) -> None:
        """Test path length for a simple 2D path."""
        metric = PathLength()
        # Path: (0,0) -> (1,0) -> (1,1) -> (2,1) -> (2,2)
        # Distances: 1 + 1 + 1 + 1 = 4
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]])
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(4.0))

    def test_diagonal_path(self) -> None:
        """Test path length for a diagonal path."""
        metric = PathLength()
        # Diagonal from (0,0) to (1,1) to (2,2)
        # Each segment has length sqrt(2)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        metric.update(trajectory)
        result = metric.compute()
        expected = 2 * torch.sqrt(torch.tensor(2.0))
        assert torch.isclose(result, expected)

    def test_3d_trajectory(self) -> None:
        """Test path length for a 3D trajectory."""
        metric = PathLength()
        # 3D path: (0,0,0) -> (1,0,0) -> (1,1,0) -> (1,1,1)
        trajectory = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        )
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(3.0))

    def test_higher_dimensional_trajectory(self) -> None:
        """Test path length for higher-dimensional trajectory."""
        metric = PathLength()
        # 5D trajectory
        trajectory = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]]
        )
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(2.0))

    def test_zero_length_segments(self) -> None:
        """Test trajectory with zero-length segments (duplicate points)."""
        metric = PathLength()
        # Path with a duplicate point
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(2.0))

    def test_multiple_updates(self) -> None:
        """Test metric with multiple trajectory updates."""
        metric = PathLength()
        # First trajectory: length 2.0
        traj1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(traj1)
        # Second trajectory: length 2.0
        traj2 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        metric.update(traj2)
        result = metric.compute()
        # Average of 2.0 and 2.0
        assert torch.isclose(result, torch.tensor(2.0))

    def test_multiple_updates_different_lengths(self) -> None:
        """Test metric with trajectories of different path lengths."""
        metric = PathLength()
        # First trajectory: length 3.0
        traj1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        metric.update(traj1)
        # Second trajectory: length 1.0
        traj2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        metric.update(traj2)
        result = metric.compute()
        # Average of 3.0 and 1.0
        assert torch.isclose(result, torch.tensor(2.0))

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = PathLength()
        traj1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        metric.update(traj1)
        result1 = metric.compute()
        assert torch.isclose(result1, torch.tensor(3.0))

        # Reset and compute again
        metric.reset()
        traj2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        metric.update(traj2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(1.0))

    def test_single_point_raises(self) -> None:
        """Test that single-point trajectory raises ValueError."""
        metric = PathLength()
        with pytest.raises(ValueError, match="at least 2 points"):
            metric.update(torch.tensor([[0.0, 0.0]]))

    def test_empty_trajectory_raises(self) -> None:
        """Test that empty trajectory raises ValueError."""
        metric = PathLength()
        with pytest.raises(ValueError, match="at least 2 points"):
            metric.update(torch.tensor([]).reshape(0, 2))

    def test_invalid_shape_raises(self) -> None:
        """Test that invalid tensor shape raises ValueError."""
        metric = PathLength()
        # 1D tensor (invalid - need at least 2 dimensions)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([0.0, 1.0, 2.0]))

    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = PathLength()
        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_different_dtypes(self) -> None:
        """Test with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            metric = PathLength()
            trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=dtype)
            metric.update(trajectory)
            result = metric.compute()
            # Result is converted to float32 internally
            assert torch.isclose(result, torch.tensor(2.0))

    def test_int_dtype(self) -> None:
        """Test with integer dtype (should work with automatic conversion)."""
        metric = PathLength()
        trajectory = torch.tensor([[0, 0], [1, 0], [2, 0]])
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(2.0))

    def test_large_trajectory(self) -> None:
        """Test with large trajectory to verify numerical stability."""
        metric = PathLength()
        # Create a long straight line
        num_points = 1000
        trajectory = torch.zeros((num_points, 2))
        trajectory[:, 0] = torch.arange(num_points, dtype=torch.float32)
        metric.update(trajectory)
        result = metric.compute()
        expected = float(num_points - 1)
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_complex_path_with_known_length(self) -> None:
        """Test with a more complex path where length can be verified."""
        metric = PathLength()
        # Square path: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        metric.update(trajectory)
        result = metric.compute()
        # Perimeter of unit square = 4
        assert torch.isclose(result, torch.tensor(4.0))

    def test_circular_approximation(self) -> None:
        """Test with points approximating a circle."""
        metric = PathLength()
        # Create points on a unit circle (8 points)
        angles = torch.linspace(0, 2 * torch.pi, 9)  # 9 points to close the circle
        trajectory = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        metric.update(trajectory)
        result = metric.compute()
        # Should approximate 2*pi (circumference of unit circle)
        expected = torch.tensor(2 * torch.pi)
        # Allow more tolerance since we're approximating with 8 segments
        assert torch.isclose(result, expected, rtol=0.05)

    def test_gpu_if_available(self) -> None:
        """Test metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = PathLength().to("cuda")
        trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], device="cuda")
        metric.update(trajectory)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(2.0, device="cuda"))

    def test_mixed_positive_negative_coordinates(self) -> None:
        """Test with trajectories that include negative coordinates."""
        metric = PathLength()
        trajectory = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        metric.update(trajectory)
        result = metric.compute()
        # Each segment has length sqrt(2)
        expected = 2 * torch.sqrt(torch.tensor(2.0))
        assert torch.isclose(result, expected)

    def test_very_short_segments(self) -> None:
        """Test numerical stability with very short segments."""
        metric = PathLength()
        # Create trajectory with very small steps
        trajectory = torch.tensor(
            [[0.0, 0.0], [1e-6, 0.0], [2e-6, 0.0], [3e-6, 0.0]], dtype=torch.float64
        )
        metric.update(trajectory)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(3e-6), rtol=1e-5)


class TestPathLengthBatched:
    """Test suite for batched PathLength metric."""

    def test_batched_2d_simple(self) -> None:
        """Test simple batch of 2D trajectories - shape (B, L, D)."""
        metric = PathLength()
        # Batch of 2 trajectories, each with 3 points in 2D
        batch = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],  # length 2.0
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],  # length 2.0
            ]
        )
        metric.update(batch)
        result = metric.compute()
        # Average of 2.0 and 2.0
        assert torch.isclose(result, torch.tensor(2.0))

    def test_batched_2d_different_lengths(self) -> None:
        """Test batch with trajectories of different path lengths."""
        metric = PathLength()
        batch = torch.tensor(
            [
                [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]],  # L-shaped, length 7.0
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],  # straight, length 2.0
            ]
        )
        metric.update(batch)
        result = metric.compute()
        # Average of 7.0 and 2.0 = 4.5
        assert torch.isclose(result, torch.tensor(4.5))

    def test_batched_3d(self) -> None:
        """Test batch of 3D trajectories."""
        metric = PathLength()
        batch = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],  # length 2.0
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],  # length 2.0
            ]
        )
        metric.update(batch)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(2.0))

    def test_temporal_batch(self) -> None:
        """Test temporal batch - shape (B, T, L, D)."""
        metric = PathLength()
        # 2 batches, 3 timesteps, 4 points, 2D
        temporal_batch = torch.randn(2, 3, 4, 2)
        # Ensure we can process it
        metric.update(temporal_batch)
        result = metric.compute()
        # Should return a scalar (average across all 2*3=6 trajectories)
        assert result.ndim == 0

    def test_higher_dimensional_batch(self) -> None:
        """Test higher dimensional batch - shape (B1, B2, B3, L, D)."""
        metric = PathLength()
        # Create 5D batch
        batch = torch.randn(2, 3, 2, 5, 3)  # (2, 3, 2, L=5, D=3)
        metric.update(batch)
        result = metric.compute()
        # Should return scalar (average across all 2*3*2=12 trajectories)
        assert result.ndim == 0

    def test_batched_with_known_values(self) -> None:
        """Test batched processing with known path lengths."""
        metric = PathLength()
        # Create batch where we know exact path lengths
        batch = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],  # length 1.0
                [[0.0, 0.0], [0.0, 2.0]],  # length 2.0
                [[0.0, 0.0], [3.0, 0.0]],  # length 3.0
                [[0.0, 0.0], [0.0, 4.0]],  # length 4.0
            ]
        )
        metric.update(batch)
        result = metric.compute()
        # Average of 1, 2, 3, 4 = 2.5
        assert torch.isclose(result, torch.tensor(2.5))

    def test_batched_multiple_updates(self) -> None:
        """Test multiple updates with batched trajectories."""
        metric = PathLength()
        # First batch
        batch1 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],  # length 1.0
                [[0.0, 0.0], [2.0, 0.0]],  # length 2.0
            ]
        )
        metric.update(batch1)
        # Second batch
        batch2 = torch.tensor(
            [
                [[0.0, 0.0], [3.0, 0.0]],  # length 3.0
                [[0.0, 0.0], [4.0, 0.0]],  # length 4.0
            ]
        )
        metric.update(batch2)
        result = metric.compute()
        # Average of 1, 2, 3, 4 = 2.5
        assert torch.isclose(result, torch.tensor(2.5))

    def test_batched_mixed_with_single(self) -> None:
        """Test mixing batched and single trajectory updates."""
        metric = PathLength()
        # Single trajectory
        single = torch.tensor([[0.0, 0.0], [1.0, 0.0]])  # length 1.0
        metric.update(single)
        # Batched trajectories
        batch = torch.tensor(
            [
                [[0.0, 0.0], [2.0, 0.0]],  # length 2.0
                [[0.0, 0.0], [3.0, 0.0]],  # length 3.0
            ]
        )
        metric.update(batch)
        result = metric.compute()
        # Average of 1, 2, 3 = 2.0
        assert torch.isclose(result, torch.tensor(2.0))

    def test_batched_reset(self) -> None:
        """Test reset functionality with batched trajectories."""
        metric = PathLength()
        batch1 = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [2.0, 0.0]]])
        metric.update(batch1)
        result1 = metric.compute()
        assert torch.isclose(result1, torch.tensor(1.5))

        metric.reset()
        batch2 = torch.tensor([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [4.0, 0.0]]])
        metric.update(batch2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(3.5))

    def test_batched_invalid_shape_raises(self) -> None:
        """Test that invalid batched shapes raise errors."""
        metric = PathLength()
        # 1D tensor
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([1.0, 2.0, 3.0]))

    def test_batched_insufficient_points_raises(self) -> None:
        """Test that batched tensors with insufficient points raise errors."""
        metric = PathLength()
        # Batch where L=1 (only 1 point per trajectory)
        batch = torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]]])  # Shape (2, 1, 2)
        with pytest.raises(ValueError, match="at least 2 points"):
            metric.update(batch)

    def test_large_batch(self) -> None:
        """Test with large batch to verify numerical stability."""
        metric = PathLength()
        # Create large batch of random trajectories
        batch = torch.randn(100, 10, 3)  # 100 trajectories, 10 points each, 3D
        metric.update(batch)
        result = metric.compute()
        # Just verify it computes without error and returns scalar
        assert result.ndim == 0
        assert result > 0

    def test_batched_gpu_if_available(self) -> None:
        """Test batched metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = PathLength().to("cuda")
        batch = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ],
            device="cuda",
        )
        metric.update(batch)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(1.0, device="cuda"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
