"""Tests for RelativeTrajectoryError metric."""

import pytest
import torch

from robometric_frame.trajectory_quality import RelativeTrajectoryError


class TestRelativeTrajectoryError:
    """Test suite for RelativeTrajectoryError metric."""

    def test_perfect_match_delta1(self) -> None:
        """Test RTE for perfect prediction with delta=1."""
        metric = RelativeTrajectoryError(delta=1)
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_perfect_match_delta2(self) -> None:
        """Test RTE for perfect prediction with delta=2."""
        metric = RelativeTrajectoryError(delta=2)
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_constant_drift_delta1(self) -> None:
        """Test RTE with constant drift in motion (delta=1)."""
        metric = RelativeTrajectoryError(delta=1)
        # Predicted moves [1,0], [1,0.5]
        # Reference moves [1,0], [1,0]
        # Difference in second move: [0,0.5]
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Errors: 0.0, 0.5 -> average = 0.25
        assert torch.isclose(result, torch.tensor(0.25), atol=1e-5)

    def test_global_offset_delta1(self) -> None:
        """Test that global offset doesn't affect RTE (delta=1)."""
        metric = RelativeTrajectoryError(delta=1)
        # Both have same relative motions, just different starting points
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[5.0, 5.0], [6.0, 5.0], [7.0, 5.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Same relative motion, error should be 0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_different_motion_patterns_delta1(self) -> None:
        """Test RTE with completely different motion patterns (delta=1)."""
        metric = RelativeTrajectoryError(delta=1)
        # Predicted: moves [1,0], [0,1]
        # Reference: moves [0,1], [1,0]
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        reference = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Should have non-zero error
        assert result > 0.5

    def test_delta2_motion(self) -> None:
        """Test RTE with delta=2 (checking motion over 2 steps)."""
        metric = RelativeTrajectoryError(delta=2)
        # Predicted: displacement from point 0 to 2 is [2,0]
        # Reference: displacement from point 0 to 2 is [2,0]
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_delta2_with_error(self) -> None:
        """Test RTE with delta=2 and motion error."""
        metric = RelativeTrajectoryError(delta=2)
        # Predicted: [0,0]->[2,0] and [1,0]->[3,0] (displacements of [2,0])
        # Reference: [0,0]->[2,1] and [1,0]->[3,1] (displacements of [2,1])
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Displacement difference: ||[2,0] - [2,1]|| = 1.0
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_3d_trajectory(self) -> None:
        """Test RTE for 3D trajectories."""
        metric = RelativeTrajectoryError(delta=1)
        predicted = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_invalid_delta_raises(self) -> None:
        """Test that delta < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Delta must be >= 1"):
            RelativeTrajectoryError(delta=0)

    def test_insufficient_points_raises(self) -> None:
        """Test that L <= delta raises ValueError."""
        metric = RelativeTrajectoryError(delta=2)
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]])  # Only 2 points
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="more than delta"):
            metric.update(predicted, reference)

    def test_multiple_updates(self) -> None:
        """Test metric with multiple trajectory updates."""
        metric = RelativeTrajectoryError(delta=1)
        # First trajectory: zero error
        pred1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        ref1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(pred1, ref1)
        # Second trajectory: constant drift
        pred2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
        ref2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(pred2, ref2)
        result = metric.compute()
        # First has error 0, second has average error 0.5, combined average = 0.25
        assert torch.isclose(result, torch.tensor(0.25), atol=1e-5)

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = RelativeTrajectoryError(delta=1)
        pred1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
        ref1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(pred1, ref1)
        result1 = metric.compute()
        assert result1 > 0

        # Reset and compute with perfect match
        metric.reset()
        pred2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        ref2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(pred2, ref2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(0.0), atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        """Test that mismatched shapes raise ValueError."""
        metric = RelativeTrajectoryError(delta=1)
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        with pytest.raises(ValueError, match="same shape"):
            metric.update(predicted, reference)

    def test_invalid_shape_raises(self) -> None:
        """Test that invalid tensor shape raises ValueError."""
        metric = RelativeTrajectoryError(delta=1)
        # 1D tensor (invalid - need at least 2 dimensions)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = RelativeTrajectoryError(delta=1)
        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_different_dtypes(self) -> None:
        """Test with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            metric = RelativeTrajectoryError(delta=1)
            predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=dtype)
            reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=dtype)
            metric.update(predicted, reference)
            result = metric.compute()
            assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_int_dtype(self) -> None:
        """Test with integer dtype (should work with automatic conversion)."""
        metric = RelativeTrajectoryError(delta=1)
        predicted = torch.tensor([[0, 0], [1, 0], [2, 0]])
        reference = torch.tensor([[0, 0], [1, 0], [2, 0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_large_trajectory(self) -> None:
        """Test with large trajectory to verify numerical stability."""
        metric = RelativeTrajectoryError(delta=1)
        num_points = 1000
        predicted = torch.zeros((num_points, 2))
        predicted[:, 0] = torch.arange(num_points, dtype=torch.float32)
        reference = predicted.clone()
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), rtol=1e-5)

    def test_gpu_if_available(self) -> None:
        """Test metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = RelativeTrajectoryError(delta=1).to("cuda")
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], device="cuda")
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], device="cuda")
        metric.update(predicted, reference)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.0, device="cuda"), atol=1e-6)

    def test_mixed_positive_negative_coordinates(self) -> None:
        """Test with trajectories that include negative coordinates."""
        metric = RelativeTrajectoryError(delta=1)
        predicted = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        reference = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_higher_dimensional_trajectory(self) -> None:
        """Test RTE for higher-dimensional trajectory."""
        metric = RelativeTrajectoryError(delta=1)
        # 5D trajectory
        predicted = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]]
        )
        reference = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_large_delta(self) -> None:
        """Test with large delta value."""
        metric = RelativeTrajectoryError(delta=5)
        predicted = torch.zeros((10, 2))
        predicted[:, 0] = torch.arange(10, dtype=torch.float32)
        reference = predicted.clone()
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)


class TestRelativeTrajectoryErrorBatched:
    """Test suite for batched RelativeTrajectoryError metric."""

    def test_batched_2d_simple(self) -> None:
        """Test simple batch of 2D trajectories - shape (B, L, D)."""
        metric = RelativeTrajectoryError(delta=1)
        predicted = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        reference = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        # Both perfect matches, average = 0.0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_mixed_errors(self) -> None:
        """Test batch with trajectories of different errors."""
        metric = RelativeTrajectoryError(delta=1)
        predicted = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],  # perfect match
                [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]],  # has drift
            ]
        )
        reference = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        # First has error 0, second has average error 0.5, combined average = 0.25
        assert torch.isclose(result, torch.tensor(0.25), atol=1e-5)

    def test_batched_3d(self) -> None:
        """Test batch of 3D trajectories."""
        metric = RelativeTrajectoryError(delta=1)
        predicted = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        )
        reference = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_temporal_batch(self) -> None:
        """Test temporal batch - shape (B, T, L, D)."""
        metric = RelativeTrajectoryError(delta=1)
        # 2 batches, 3 timesteps, 4 points, 2D
        predicted = torch.randn(2, 3, 4, 2)
        reference = predicted.clone()
        metric.update(predicted, reference)
        result = metric.compute()
        # Perfect match, should be ~0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_higher_dimensional_batch(self) -> None:
        """Test higher dimensional batch - shape (B1, B2, B3, L, D)."""
        metric = RelativeTrajectoryError(delta=1)
        # Create 5D batch
        predicted = torch.randn(2, 3, 2, 5, 3)
        reference = predicted.clone()
        metric.update(predicted, reference)
        result = metric.compute()
        # Perfect match
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_delta2(self) -> None:
        """Test batched trajectories with delta=2."""
        metric = RelativeTrajectoryError(delta=2)
        predicted = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
            ]
        )
        reference = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_multiple_updates(self) -> None:
        """Test multiple updates with batched trajectories."""
        metric = RelativeTrajectoryError(delta=1)
        # First batch - perfect matches
        pred1 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        ref1 = pred1.clone()
        metric.update(pred1, ref1)
        # Second batch - perfect matches
        pred2 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        ref2 = pred2.clone()
        metric.update(pred2, ref2)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_reset(self) -> None:
        """Test reset functionality with batched trajectories."""
        metric = RelativeTrajectoryError(delta=1)
        pred1 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]],
            ]
        )
        ref1 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            ]
        )
        metric.update(pred1, ref1)
        result1 = metric.compute()
        assert result1 > 0

        metric.reset()
        pred2 = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ]
        )
        ref2 = pred2.clone()
        metric.update(pred2, ref2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(0.0), atol=1e-6)

    def test_batched_shape_mismatch_raises(self) -> None:
        """Test that mismatched batched shapes raise errors."""
        metric = RelativeTrajectoryError(delta=1)
        predicted = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        reference = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            ]
        )
        with pytest.raises(ValueError, match="same shape"):
            metric.update(predicted, reference)

    def test_batched_insufficient_points_raises(self) -> None:
        """Test that batched tensors with insufficient points raise errors."""
        metric = RelativeTrajectoryError(delta=2)
        # Only 2 points, but delta=2 requires more than 2
        predicted = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]])
        reference = predicted.clone()
        with pytest.raises(ValueError, match="more than delta"):
            metric.update(predicted, reference)

    def test_large_batch(self) -> None:
        """Test with large batch to verify numerical stability."""
        metric = RelativeTrajectoryError(delta=1)
        # Create large batch of random trajectories
        predicted = torch.randn(100, 10, 3)
        reference = predicted.clone()
        metric.update(predicted, reference)
        result = metric.compute()
        # Perfect match
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_batched_gpu_if_available(self) -> None:
        """Test batched metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = RelativeTrajectoryError(delta=1).to("cuda")
        predicted = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
            ],
            device="cuda",
        )
        reference = predicted.clone()
        metric.update(predicted, reference)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.0, device="cuda"), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
