"""Tests for SuccessRate metric."""

import pytest
import torch

from vla_metrics.task_performance import SuccessRate


class TestSuccessRate:
    """Test suite for SuccessRate metric."""

    def test_binary_success_perfect(self) -> None:
        """Test perfect success rate (100%)."""
        metric = SuccessRate()
        success = torch.tensor([1, 1, 1, 1])
        metric.update(success)
        result = metric.compute()
        assert result == 1.0

    def test_binary_success_zero(self) -> None:
        """Test zero success rate (0%)."""
        metric = SuccessRate()
        success = torch.tensor([0, 0, 0, 0])
        metric.update(success)
        result = metric.compute()
        assert result == 0.0

    def test_binary_success_partial(self) -> None:
        """Test partial success rate."""
        metric = SuccessRate()
        success = torch.tensor([1, 1, 0, 1, 0, 0, 1])
        metric.update(success)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(4.0 / 7.0))

    def test_continuous_scores_with_threshold(self) -> None:
        """Test success rate with continuous scores and threshold."""
        metric = SuccessRate(threshold=0.8)
        scores = torch.tensor([0.9, 0.7, 0.85, 0.6, 0.95, 0.82])
        metric.update(scores)
        result = metric.compute()
        # Scores >= 0.8: 0.9, 0.85, 0.95, 0.82 = 4 out of 6
        assert torch.isclose(result, torch.tensor(4.0 / 6.0))

    def test_multiple_updates(self) -> None:
        """Test metric with multiple update calls."""
        metric = SuccessRate()
        # First batch
        success1 = torch.tensor([1, 1, 0])
        metric.update(success1)
        # Second batch
        success2 = torch.tensor([1, 0, 1])
        metric.update(success2)
        result = metric.compute()
        # Total: 4 successes out of 6 tasks
        assert torch.isclose(result, torch.tensor(4.0 / 6.0))

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = SuccessRate()
        success1 = torch.tensor([1, 1, 0, 1])
        metric.update(success1)
        result1 = metric.compute()
        assert result1 == 0.75

        # Reset and compute again
        metric.reset()
        success2 = torch.tensor([1, 0])
        metric.update(success2)
        result2 = metric.compute()
        assert result2 == 0.5

    def test_ignore_index(self) -> None:
        """Test ignoring specific values."""
        metric = SuccessRate(ignore_index=-1)
        success = torch.tensor([1, -1, 0, 1, -1, 0])
        metric.update(success)
        result = metric.compute()
        # Ignoring -1 values: [1, 0, 1, 0] = 2 out of 4
        assert result == 0.5

    def test_empty_tensor_raises(self) -> None:
        """Test that empty tensor raises ValueError."""
        metric = SuccessRate()
        with pytest.raises(ValueError, match="Input tensor is empty"):
            metric.update(torch.tensor([]))

    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = SuccessRate()
        with pytest.raises(RuntimeError, match="no tasks have been recorded"):
            metric.compute()

    def test_non_binary_without_threshold_raises(self) -> None:
        """Test that non-binary values without threshold raise ValueError."""
        metric = SuccessRate()
        success = torch.tensor([0.5, 0.7, 0.9])
        with pytest.raises(ValueError, match="must be binary"):
            metric.update(success)

    def test_bool_input(self) -> None:
        """Test with boolean input."""
        metric = SuccessRate()
        success = torch.tensor([True, True, False, True, False])
        metric.update(success)
        result = metric.compute()
        assert result == 0.6

    def test_float_binary_input(self) -> None:
        """Test with float binary input."""
        metric = SuccessRate()
        success = torch.tensor([1.0, 1.0, 0.0, 1.0])
        metric.update(success)
        result = metric.compute()
        assert result == 0.75

    def test_large_batch(self) -> None:
        """Test with large batch to verify numerical stability."""
        metric = SuccessRate()
        success = torch.randint(0, 2, (10000,))
        metric.update(success)
        result = metric.compute()
        expected = success.float().mean()
        assert torch.isclose(result, expected, rtol=1e-5)

    def test_all_ignored(self) -> None:
        """Test when all values are ignored."""
        metric = SuccessRate(ignore_index=0)
        success = torch.tensor([0, 0, 0])
        metric.update(success)
        # After ignoring all, no tasks recorded
        with pytest.raises(RuntimeError, match="no tasks have been recorded"):
            metric.compute()

    def test_different_dtypes(self) -> None:
        """Test with different tensor dtypes."""
        for dtype in [torch.int32, torch.int64, torch.float32, torch.float64]:
            metric = SuccessRate()
            success = torch.tensor([1, 0, 1, 1], dtype=dtype)
            metric.update(success)
            result = metric.compute()
            assert result == 0.75

    def test_gpu_if_available(self) -> None:
        """Test metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = SuccessRate().to("cuda")
        success = torch.tensor([1, 1, 0, 1], device="cuda")
        metric.update(success)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert result == 0.75


class TestTaskSuccessRate:
    """Test suite for TaskSuccessRate alias."""

    def test_alias_functionality(self) -> None:
        """Test that TaskSuccessRate works as an alias."""
        from vla_metrics.task_performance.success_rate import TaskSuccessRate

        metric = TaskSuccessRate()
        success = torch.tensor([1, 1, 0, 1])
        metric.update(success)
        result = metric.compute()
        assert result == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
