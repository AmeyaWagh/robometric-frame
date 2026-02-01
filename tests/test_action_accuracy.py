"""Tests for ActionAccuracy metric."""

import pytest
import torch

from robometric_frame.task_performance.action_accuracy import ActionAccuracy


class TestActionAccuracy:
    """Test suite for ActionAccuracy metric."""

    def test_perfect_predictions(self) -> None:
        """Test MSE with perfect predictions (should be 0)."""
        metric = ActionAccuracy()
        predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        targets = predictions.clone()

        metric.update(predictions, targets)
        results = metric.compute()

        assert torch.isclose(results["mse"], torch.tensor(0.0))
        assert torch.isclose(results["amse"], torch.tensor(0.0))

    def test_known_mse(self) -> None:
        """Test MSE with known values."""
        metric = ActionAccuracy()

        # Simple case: predictions are off by 1 in each dimension
        predictions = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        targets = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        metric.update(predictions, targets)
        results = metric.compute()

        # MSE = mean of squared L2 norms
        # Timestep 1: ||[1,1] - [0,0]||^2 = 1^2 + 1^2 = 2
        # Timestep 2: ||[2,2] - [1,1]||^2 = 1^2 + 1^2 = 2
        # MSE = (2 + 2) / 2 = 2.0
        assert torch.isclose(results["mse"], torch.tensor(2.0))
        assert torch.isclose(results["amse"], torch.tensor(2.0))

    def test_multiple_trajectories_amse(self) -> None:
        """Test AMSE computation across multiple trajectories."""
        metric = ActionAccuracy()

        # Trajectory 1: MSE = 1.0
        pred1 = torch.tensor([[1.0], [1.0]])
        target1 = torch.tensor([[0.0], [0.0]])
        metric.update(pred1, target1)  # MSE = (1 + 1) / 2 = 1.0

        # Trajectory 2: MSE = 4.0
        pred2 = torch.tensor([[2.0], [2.0]])
        target2 = torch.tensor([[0.0], [0.0]])
        metric.update(pred2, target2)  # MSE = (4 + 4) / 2 = 4.0

        results = metric.compute()

        # AMSE = (1.0 + 4.0) / 2 = 2.5
        assert torch.isclose(results["amse"], torch.tensor(2.5))

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = ActionAccuracy()

        # First trajectory
        metric.update(torch.ones(5, 3), torch.zeros(5, 3))
        metric.reset()

        # Second trajectory after reset
        predictions = torch.zeros(5, 3)
        targets = torch.zeros(5, 3)
        metric.update(predictions, targets)

        results = metric.compute()
        assert torch.isclose(results["mse"], torch.tensor(0.0))

    def test_normalization_with_provided_variance(self) -> None:
        """Test NAMSE computation with provided action variance."""
        action_var = 2.0
        metric = ActionAccuracy(normalize=True, action_variance=action_var)

        # Create simple case where AMSE = 4.0
        predictions = torch.tensor([[2.0], [2.0]])
        targets = torch.tensor([[0.0], [0.0]])
        metric.update(predictions, targets)  # MSE = 4.0

        results = metric.compute()

        assert "namse" in results
        # NAMSE = AMSE / variance = 4.0 / 2.0 = 2.0
        assert torch.isclose(results["namse"], torch.tensor(2.0))

    def test_normalization_computed_variance(self) -> None:
        """Test NAMSE computation with variance computed from data."""
        metric = ActionAccuracy(normalize=True)

        # Use targets with known variance
        # targets = [0, 1, 0, 1], mean = 0.5, variance = 0.25
        targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        predictions = torch.tensor([[0.5], [1.5], [0.5], [1.5]])

        metric.update(predictions, targets)
        results = metric.compute()

        assert "namse" in results
        # Variance of targets = 0.25
        # AMSE / 0.25 should give NAMSE
        expected_namse = results["amse"] / 0.25
        assert torch.isclose(results["namse"], expected_namse)

    def test_shape_mismatch_raises(self) -> None:
        """Test that shape mismatch raises ValueError."""
        metric = ActionAccuracy()

        predictions = torch.randn(10, 4)
        targets = torch.randn(10, 5)  # Different action dimension

        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.update(predictions, targets)

    def test_empty_tensor_raises(self) -> None:
        """Test that empty tensors raise ValueError."""
        metric = ActionAccuracy()

        predictions = torch.empty(0, 4)
        targets = torch.empty(0, 4)

        with pytest.raises(ValueError, match="empty"):
            metric.update(predictions, targets)

    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = ActionAccuracy()

        with pytest.raises(RuntimeError, match="no trajectories have been recorded"):
            metric.compute()

    def test_different_trajectory_lengths(self) -> None:
        """Test handling trajectories of different lengths."""
        metric = ActionAccuracy()

        # Short trajectory
        metric.update(torch.zeros(5, 2), torch.zeros(5, 2))

        # Long trajectory
        metric.update(torch.zeros(20, 2), torch.zeros(20, 2))

        results = metric.compute()
        assert torch.isclose(results["amse"], torch.tensor(0.0))

    def test_multidimensional_actions(self) -> None:
        """Test with different action dimensions."""
        for action_dim in [1, 4, 7, 10]:
            metric = ActionAccuracy()

            predictions = torch.randn(10, action_dim)
            targets = torch.randn(10, action_dim)

            metric.update(predictions, targets)
            results = metric.compute()

            assert results["mse"] >= 0  # MSE should always be non-negative
            assert results["amse"] >= 0

    def test_normalization_without_variance_raises(self) -> None:
        """Test that NAMSE with zero variance raises RuntimeError."""
        metric = ActionAccuracy(normalize=True)

        # All targets are the same (zero variance)
        predictions = torch.tensor([[1.0], [1.0], [1.0]])
        targets = torch.tensor([[0.0], [0.0], [0.0]])

        metric.update(predictions, targets)

        with pytest.raises(RuntimeError, match="variance is zero or negative"):
            metric.compute()

    def test_different_dtypes(self) -> None:
        """Test metric with different tensor dtypes."""
        metric = ActionAccuracy()

        # Test with float32
        metric.update(
            torch.randn(10, 4, dtype=torch.float32), torch.randn(10, 4, dtype=torch.float32)
        )
        results = metric.compute()
        # Metric states maintain their initialization dtype (float32)
        assert results["mse"].dtype == torch.float32
        assert torch.isfinite(results["mse"])

        metric.reset()

        # Test with float64
        metric.update(
            torch.randn(10, 4, dtype=torch.float64), torch.randn(10, 4, dtype=torch.float64)
        )
        results = metric.compute()
        # States remain float32 regardless of input dtype
        assert results["mse"].dtype == torch.float32
        assert torch.isfinite(results["mse"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_if_available(self) -> None:
        """Test metric on GPU if CUDA is available."""
        metric = ActionAccuracy().cuda()

        predictions = torch.randn(10, 4, device="cuda")
        targets = torch.randn(10, 4, device="cuda")

        metric.update(predictions, targets)
        results = metric.compute()

        assert results["mse"].device.type == "cuda"
        assert results["amse"].device.type == "cuda"

    def test_large_batch(self) -> None:
        """Test metric with large trajectory."""
        metric = ActionAccuracy()

        # Large trajectory
        predictions = torch.randn(1000, 10)
        targets = torch.randn(1000, 10)

        metric.update(predictions, targets)
        results = metric.compute()

        assert results["mse"] >= 0
        assert torch.isfinite(results["mse"])

    def test_incremental_updates(self) -> None:
        """Test incremental trajectory updates."""
        metric = ActionAccuracy()

        # Add 5 trajectories incrementally
        for _ in range(5):
            predictions = torch.randn(10, 4)
            targets = torch.randn(10, 4)
            metric.update(predictions, targets)

        results = metric.compute()

        # Should have averaged across all 5 trajectories
        assert results["amse"] >= 0
        assert torch.isfinite(results["amse"])

    def test_normalized_vs_unnormalized(self) -> None:
        """Test that unnormalized metric doesn't include namse."""
        # Without normalization
        metric_unnorm = ActionAccuracy(normalize=False)
        predictions = torch.randn(10, 4)
        targets = torch.randn(10, 4)
        metric_unnorm.update(predictions, targets)
        results_unnorm = metric_unnorm.compute()

        assert "namse" not in results_unnorm
        assert "mse" in results_unnorm
        assert "amse" in results_unnorm

        # With normalization
        metric_norm = ActionAccuracy(normalize=True)
        metric_norm.update(predictions, targets)
        results_norm = metric_norm.compute()

        assert "namse" in results_norm
        assert "mse" in results_norm
        assert "amse" in results_norm
