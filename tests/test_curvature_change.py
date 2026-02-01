"""Tests for CurvatureChange metric."""

import math

import pytest
import torch

from robometric_frame.trajectory_quality import CurvatureChange


class TestCurvatureChange:
    """Test suite for CurvatureChange metric."""

    def test_straight_line_constant_orientation(self) -> None:
        """Test CC for straight line with constant orientation (should be 0)."""
        metric = CurvatureChange()
        # Straight line moving in +x direction
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        orientations = torch.tensor([0.0, 0.0, 0.0, 0.0])  # Constant orientation
        metric.update(positions, orientations)
        result = metric.compute()
        # No curvature change for constant curvature (zero)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_straight_line_varying_orientation(self) -> None:
        """Test CC for straight line with linearly changing orientation."""
        metric = CurvatureChange()
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        # Linear orientation change: 0, 0.1, 0.2, 0.3
        orientations = torch.tensor([0.0, 0.1, 0.2, 0.3])
        metric.update(positions, orientations)
        result = metric.compute()
        # Curvature is constant (0.1 per unit distance), so curvature change = 0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_constant_curvature_circular_motion(self) -> None:
        """Test CC for circular motion (constant curvature)."""
        metric = CurvatureChange()
        # Approximate circular motion with small arc
        num_points = 10
        angles = torch.linspace(0, math.pi / 4, num_points)
        positions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        # Orientation is perpendicular to radius (tangent direction)
        orientations = angles + math.pi / 2
        metric.update(positions, orientations)
        result = metric.compute()
        # Circular motion has constant curvature, so change should be small
        assert result < 0.1

    def test_sharp_turn(self) -> None:
        """Test CC for trajectory with a sharp turn."""
        metric = CurvatureChange()
        # Path with sharp turn: straight then sudden direction change
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ]
        )
        # Orientation changes sharply at the turn
        orientations = torch.tensor([0.0, 0.0, 0.0, math.pi / 2])
        metric.update(positions, orientations)
        result = metric.compute()
        # Should have high curvature change due to sharp turn
        assert result > 0.5

    def test_s_curve(self) -> None:
        """Test CC for S-curve (changing curvature)."""
        metric = CurvatureChange()
        # S-curve: turn one way then the other
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.5],
                [3.0, 1.0],
                [4.0, 0.0],
            ]
        )
        # Orientation changes from +45° to 0° to -45°
        orientations = torch.tensor(
            [
                math.pi / 4,
                math.pi / 8,
                0.0,
                -math.pi / 8,
                -math.pi / 4,
            ]
        )
        metric.update(positions, orientations)
        result = metric.compute()
        # S-curve should have curvature changes
        assert result > 0

    def test_minimum_points(self) -> None:
        """Test with minimum required points (3 points)."""
        metric = CurvatureChange()
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        orientations = torch.tensor([0.0, 0.0, 0.0])
        metric.update(positions, orientations)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_3d_positions(self) -> None:
        """Test with 3D positions (orientation is still scalar)."""
        metric = CurvatureChange()
        # 3D trajectory, but orientation is still 1D heading
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        orientations = torch.tensor([0.0, 0.0, 0.0])
        metric.update(positions, orientations)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_multiple_updates(self) -> None:
        """Test metric with multiple trajectory updates."""
        metric = CurvatureChange()
        # First trajectory: constant curvature
        pos1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        ori1 = torch.tensor([0.0, 0.0, 0.0])
        metric.update(pos1, ori1)
        # Second trajectory: constant curvature
        pos2 = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        ori2 = torch.tensor([math.pi / 2, math.pi / 2, math.pi / 2])
        metric.update(pos2, ori2)
        result = metric.compute()
        # Both have zero curvature change
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = CurvatureChange()
        pos1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
        ori1 = torch.tensor([0.0, 0.0, 0.0, math.pi / 2])
        metric.update(pos1, ori1)
        result1 = metric.compute()
        assert result1 > 0

        # Reset and compute with constant curvature
        metric.reset()
        pos2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        ori2 = torch.tensor([0.0, 0.0, 0.0])
        metric.update(pos2, ori2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(0.0), atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        """Test that mismatched shapes raise ValueError."""
        metric = CurvatureChange()
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        orientations = torch.tensor([0.0, 0.0])  # Wrong length
        with pytest.raises(ValueError, match="shape must match"):
            metric.update(positions, orientations)

    def test_insufficient_points_raises(self) -> None:
        """Test that fewer than 3 points raises ValueError."""
        metric = CurvatureChange()
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        orientations = torch.tensor([0.0, 0.0])
        with pytest.raises(ValueError, match="at least 3 points"):
            metric.update(positions, orientations)

    def test_invalid_shape_raises(self) -> None:
        """Test that invalid tensor shape raises ValueError."""
        metric = CurvatureChange()
        # 1D positions tensor (invalid - need at least 2 dimensions)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.0, 0.0, 0.0]))

    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = CurvatureChange()
        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_different_dtypes(self) -> None:
        """Test with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            metric = CurvatureChange()
            positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=dtype)
            orientations = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
            metric.update(positions, orientations)
            result = metric.compute()
            assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_int_dtype(self) -> None:
        """Test with integer dtype (should work with automatic conversion)."""
        metric = CurvatureChange()
        positions = torch.tensor([[0, 0], [1, 0], [2, 0]])
        orientations = torch.tensor([0, 0, 0])
        metric.update(positions, orientations)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_large_trajectory(self) -> None:
        """Test with large trajectory to verify numerical stability."""
        metric = CurvatureChange()
        num_points = 1000
        positions = torch.zeros((num_points, 2))
        positions[:, 0] = torch.arange(num_points, dtype=torch.float32)
        orientations = torch.zeros(num_points)
        metric.update(positions, orientations)
        result = metric.compute()
        # Straight line with constant orientation
        assert torch.isclose(result, torch.tensor(0.0), rtol=1e-5)

    def test_gpu_if_available(self) -> None:
        """Test metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = CurvatureChange().to("cuda")
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], device="cuda")
        orientations = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        metric.update(positions, orientations)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.0, device="cuda"), atol=1e-6)

    def test_mixed_positive_negative_coordinates(self) -> None:
        """Test with trajectories that include negative coordinates."""
        metric = CurvatureChange()
        positions = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        # Orientation along diagonal
        orientations = torch.tensor([math.pi / 4, math.pi / 4, math.pi / 4])
        metric.update(positions, orientations)
        result = metric.compute()
        # Constant orientation along straight line
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_zero_length_segment(self) -> None:
        """Test trajectory with zero-length segment (duplicate points)."""
        metric = CurvatureChange()
        # Duplicate point in the middle
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        orientations = torch.tensor([0.0, 0.0, 0.0, 0.0])
        metric.update(positions, orientations)
        result = metric.compute()
        # Should handle gracefully with epsilon
        assert result >= 0  # Non-negative

    def test_orientation_wrapping(self) -> None:
        """Test with orientations that wrap around (e.g., -π to π)."""
        metric = CurvatureChange()
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        # Orientations near ±π boundary
        orientations = torch.tensor([3.0, 3.1, 3.14, -3.1])
        metric.update(positions, orientations)
        result = metric.compute()
        # Should compute (note: this doesn't handle wrapping, just tests it runs)
        assert result >= 0


class TestCurvatureChangeBatched:
    """Test suite for batched CurvatureChange metric."""

    def test_batched_2d_simple(self) -> None:
        """Test simple batch of 2D trajectories - shape (B, L, D)."""
        metric = CurvatureChange()
        positions = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        orientations = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [math.pi / 2, math.pi / 2, math.pi / 2],
            ]
        )
        metric.update(positions, orientations)
        result = metric.compute()
        # Both have constant curvature (zero change)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_mixed_curvatures(self) -> None:
        """Test batch with trajectories of different curvature changes."""
        metric = CurvatureChange()
        positions = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],  # straight
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]],  # with turn
            ]
        )
        orientations = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, math.pi / 2],
            ]
        )
        metric.update(positions, orientations)
        result = metric.compute()
        # Second has curvature change, so average > 0
        assert result > 0

    def test_batched_3d_positions(self) -> None:
        """Test batch of 3D position trajectories."""
        metric = CurvatureChange()
        positions = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        )
        orientations = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [math.pi / 2, math.pi / 2, math.pi / 2],
            ]
        )
        metric.update(positions, orientations)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_temporal_batch(self) -> None:
        """Test temporal batch - shape (B, T, L, D)."""
        metric = CurvatureChange()
        # 2 batches, 3 timesteps, 4 points, 2D
        positions = torch.randn(2, 3, 4, 2)
        orientations = torch.randn(2, 3, 4)
        # Ensure we can process it
        metric.update(positions, orientations)
        result = metric.compute()
        # Should return a scalar
        assert result.ndim == 0

    def test_higher_dimensional_batch(self) -> None:
        """Test higher dimensional batch - shape (B1, B2, B3, L, D)."""
        metric = CurvatureChange()
        # Create 5D batch
        positions = torch.randn(2, 3, 2, 5, 2)
        orientations = torch.randn(2, 3, 2, 5)
        metric.update(positions, orientations)
        result = metric.compute()
        # Should return scalar
        assert result.ndim == 0

    def test_batched_multiple_updates(self) -> None:
        """Test multiple updates with batched trajectories."""
        metric = CurvatureChange()
        # First batch - constant curvature
        pos1 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        ori1 = torch.tensor([[0.0, 0.0, 0.0], [math.pi / 2, math.pi / 2, math.pi / 2]])
        metric.update(pos1, ori1)
        # Second batch - constant curvature
        pos2 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        ori2 = torch.tensor([[0.0, 0.0, 0.0], [math.pi / 2, math.pi / 2, math.pi / 2]])
        metric.update(pos2, ori2)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_reset(self) -> None:
        """Test reset functionality with batched trajectories."""
        metric = CurvatureChange()
        pos1 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]],
            ]
        )
        ori1 = torch.tensor(
            [
                [0.0, 0.0, 0.0, math.pi / 2],
                [0.0, 0.0, 0.0, math.pi / 2],
            ]
        )
        metric.update(pos1, ori1)
        result1 = metric.compute()
        assert result1 > 0

        metric.reset()
        pos2 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        ori2 = torch.tensor([[0.0, 0.0, 0.0], [math.pi / 2, math.pi / 2, math.pi / 2]])
        metric.update(pos2, ori2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(0.0), atol=1e-6)

    def test_batched_shape_mismatch_raises(self) -> None:
        """Test that mismatched batched shapes raise errors."""
        metric = CurvatureChange()
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        orientations = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="shape must match"):
            metric.update(positions, orientations)

    def test_batched_insufficient_points_raises(self) -> None:
        """Test that batched tensors with insufficient points raise errors."""
        metric = CurvatureChange()
        # Only 2 points, need at least 3
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]])
        orientations = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        with pytest.raises(ValueError, match="at least 3 points"):
            metric.update(positions, orientations)

    def test_large_batch(self) -> None:
        """Test with large batch to verify numerical stability."""
        metric = CurvatureChange()
        # Create large batch of trajectories
        positions = torch.randn(100, 10, 2)
        orientations = torch.randn(100, 10)
        metric.update(positions, orientations)
        result = metric.compute()
        # Just verify it computes without error and returns scalar
        assert result.ndim == 0
        assert result >= 0  # Curvature change should be non-negative

    def test_batched_gpu_if_available(self) -> None:
        """Test batched metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = CurvatureChange().to("cuda")
        positions = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ],
            device="cuda",
        )
        orientations = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [math.pi / 2, math.pi / 2, math.pi / 2],
            ],
            device="cuda",
        )
        metric.update(positions, orientations)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.0, device="cuda"), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
