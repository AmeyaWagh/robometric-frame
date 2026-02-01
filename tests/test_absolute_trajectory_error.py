"""Tests for AbsoluteTrajectoryError metric."""

import pytest
import torch

from robometric_frame.trajectory_quality import AbsoluteTrajectoryError


class TestAbsoluteTrajectoryError:
    """Test suite for AbsoluteTrajectoryError metric."""

    def test_perfect_match(self) -> None:
        """Test ATE for perfect prediction (zero error)."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_constant_offset(self) -> None:
        """Test ATE with constant offset in all points."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Each point has error of 1.0, average is 1.0
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_varying_errors(self) -> None:
        """Test ATE with varying errors at different points."""
        metric = AbsoluteTrajectoryError()
        # Errors: 0.0, 1.0, 2.0
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Average error: (0.0 + 1.0 + 2.0) / 3 = 1.0
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_3d_trajectory(self) -> None:
        """Test ATE for 3D trajectories."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Errors: 0.0, 0.0, 1.0 -> average = 1/3
        expected = 1.0 / 3.0
        assert torch.isclose(result, torch.tensor(expected), rtol=1e-5)

    def test_diagonal_error(self) -> None:
        """Test ATE with diagonal errors."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Errors: 0.0, 1.0 -> average = 0.5
        assert torch.isclose(result, torch.tensor(0.5), atol=1e-5)

    def test_single_point(self) -> None:
        """Test ATE with single point trajectory."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[1.0, 2.0]])
        reference = torch.tensor([[1.0, 3.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Error is 1.0
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_multiple_updates(self) -> None:
        """Test metric with multiple trajectory updates."""
        metric = AbsoluteTrajectoryError()
        # First trajectory: zero error
        pred1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ref1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        metric.update(pred1, ref1)
        # Second trajectory: error of 1.0
        pred2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ref2 = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        metric.update(pred2, ref2)
        result = metric.compute()
        # Average of 0.0 and 1.0 = 0.5
        assert torch.isclose(result, torch.tensor(0.5), atol=1e-5)

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = AbsoluteTrajectoryError()
        pred1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ref1 = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        metric.update(pred1, ref1)
        result1 = metric.compute()
        assert torch.isclose(result1, torch.tensor(1.0))

        # Reset and compute with perfect match
        metric.reset()
        pred2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ref2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        metric.update(pred2, ref2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(0.0), atol=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        """Test that mismatched shapes raise ValueError."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        with pytest.raises(ValueError, match="same shape"):
            metric.update(predicted, reference)

    def test_invalid_shape_raises(self) -> None:
        """Test that invalid tensor shape raises ValueError."""
        metric = AbsoluteTrajectoryError()
        # 1D tensor (invalid - need at least 2 dimensions)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            metric.update(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))

    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = AbsoluteTrajectoryError()
        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_different_dtypes(self) -> None:
        """Test with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            metric = AbsoluteTrajectoryError()
            predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
            reference = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
            metric.update(predicted, reference)
            result = metric.compute()
            assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_int_dtype(self) -> None:
        """Test with integer dtype (should work with automatic conversion)."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[0, 0], [1, 0]])
        reference = torch.tensor([[0, 1], [1, 1]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(1.0))

    def test_large_trajectory(self) -> None:
        """Test with large trajectory to verify numerical stability."""
        metric = AbsoluteTrajectoryError()
        num_points = 1000
        predicted = torch.zeros((num_points, 2))
        predicted[:, 0] = torch.arange(num_points, dtype=torch.float32)
        reference = predicted.clone()
        reference[:, 1] = 1.0  # Constant offset of 1.0 in y
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(1.0), rtol=1e-5)

    def test_gpu_if_available(self) -> None:
        """Test metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = AbsoluteTrajectoryError().to("cuda")
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device="cuda")
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device="cuda")
        metric.update(predicted, reference)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.0, device="cuda"), atol=1e-6)

    def test_mixed_positive_negative_coordinates(self) -> None:
        """Test with trajectories that include negative coordinates."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        reference = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_very_small_errors(self) -> None:
        """Test numerical stability with very small errors."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        reference = torch.tensor([[0.0, 0.0], [1.0, 1e-8]], dtype=torch.float64)
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(5e-9), atol=1e-8)

    def test_higher_dimensional_trajectory(self) -> None:
        """Test ATE for higher-dimensional trajectory."""
        metric = AbsoluteTrajectoryError()
        # 5D trajectory
        predicted = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Error at second point is 1.0, average is 0.5
        assert torch.isclose(result, torch.tensor(0.5), atol=1e-5)


class TestAbsoluteTrajectoryErrorBatched:
    """Test suite for batched AbsoluteTrajectoryError metric."""

    def test_batched_2d_simple(self) -> None:
        """Test simple batch of 2D trajectories - shape (B, L, D)."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )
        reference = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        # Both perfect matches, average = 0.0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_mixed_errors(self) -> None:
        """Test batch with trajectories of different errors."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],  # perfect match
                [[0.0, 0.0], [1.0, 0.0]],  # constant error of 1.0
            ]
        )
        reference = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        # Average of 0.0 and 1.0 = 0.5
        assert torch.isclose(result, torch.tensor(0.5), atol=1e-5)

    def test_batched_3d(self) -> None:
        """Test batch of 3D trajectories."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        reference = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_temporal_batch(self) -> None:
        """Test temporal batch - shape (B, T, L, D)."""
        metric = AbsoluteTrajectoryError()
        # 2 batches, 3 timesteps, 4 points, 2D
        predicted = torch.randn(2, 3, 4, 2)
        reference = predicted.clone()
        metric.update(predicted, reference)
        result = metric.compute()
        # Perfect match, should be ~0
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_higher_dimensional_batch(self) -> None:
        """Test higher dimensional batch - shape (B1, B2, B3, L, D)."""
        metric = AbsoluteTrajectoryError()
        # Create 5D batch
        predicted = torch.randn(2, 3, 2, 5, 3)
        reference = predicted.clone()
        metric.update(predicted, reference)
        result = metric.compute()
        # Perfect match
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_batched_with_known_values(self) -> None:
        """Test batched processing with known error values."""
        metric = AbsoluteTrajectoryError()
        # Create batch where we know exact errors
        predicted = torch.tensor(
            [
                [[0.0, 0.0]],  # ATE = 0.0
                [[0.0, 0.0]],  # ATE = 1.0
                [[0.0, 0.0]],  # ATE = 2.0
            ]
        )
        reference = torch.tensor(
            [
                [[0.0, 0.0]],
                [[0.0, 1.0]],
                [[0.0, 2.0]],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        # Average of 0, 1, 2 = 1.0
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

    def test_batched_multiple_updates(self) -> None:
        """Test multiple updates with batched trajectories."""
        metric = AbsoluteTrajectoryError()
        # First batch
        pred1 = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
        ref1 = torch.tensor([[[0.0, 0.0]], [[0.0, 1.0]]])
        metric.update(pred1, ref1)
        # Second batch
        pred2 = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
        ref2 = torch.tensor([[[0.0, 2.0]], [[0.0, 3.0]]])
        metric.update(pred2, ref2)
        result = metric.compute()
        # Average of 0, 1, 2, 3 = 1.5
        assert torch.isclose(result, torch.tensor(1.5), atol=1e-5)

    def test_batched_reset(self) -> None:
        """Test reset functionality with batched trajectories."""
        metric = AbsoluteTrajectoryError()
        pred1 = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
        ref1 = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]])
        metric.update(pred1, ref1)
        result1 = metric.compute()
        assert torch.isclose(result1, torch.tensor(1.0))

        metric.reset()
        pred2 = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
        ref2 = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
        metric.update(pred2, ref2)
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(0.0), atol=1e-6)

    def test_batched_shape_mismatch_raises(self) -> None:
        """Test that mismatched batched shapes raise errors."""
        metric = AbsoluteTrajectoryError()
        predicted = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
        reference = torch.tensor([[[0.0, 0.0]]])
        with pytest.raises(ValueError, match="same shape"):
            metric.update(predicted, reference)

    def test_large_batch(self) -> None:
        """Test with large batch to verify numerical stability."""
        metric = AbsoluteTrajectoryError()
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

        metric = AbsoluteTrajectoryError().to("cuda")
        predicted = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]], device="cuda")
        reference = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]], device="cuda")
        metric.update(predicted, reference)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.0, device="cuda"), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
