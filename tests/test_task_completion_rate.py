"""Tests for TaskCompletionRate metric."""

import pytest
import torch

from vla_metrics.task_performance.task_completion_rate import TaskCompletionRate


class TestTaskCompletionRate:
    """Test suite for TaskCompletionRate metric."""

    def test_binary_completion_perfect(self) -> None:
        """Test perfect completion rate (100%)."""
        metric = TaskCompletionRate()
        completion = torch.tensor([1, 1, 1, 1])
        metric.update(completion)
        result = metric.compute()
        assert result == 1.0

    def test_binary_completion_zero(self) -> None:
        """Test zero completion rate (0%)."""
        metric = TaskCompletionRate()
        completion = torch.tensor([0, 0, 0, 0])
        metric.update(completion)
        result = metric.compute()
        assert result == 0.0

    def test_binary_completion_partial(self) -> None:
        """Test partial completion rate (60%)."""
        metric = TaskCompletionRate()
        completion = torch.tensor([1, 0, 1, 1, 0])
        metric.update(completion)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.6))

    def test_continuous_scores_with_threshold(self) -> None:
        """Test completion rate with continuous scores and threshold."""
        metric = TaskCompletionRate(threshold=0.8)
        scores = torch.tensor([0.9, 0.7, 0.85, 0.6, 0.95])
        metric.update(scores)
        result = metric.compute()
        # Expected: 3 scores >= 0.8 (0.9, 0.85, 0.95) out of 5 = 0.6
        assert torch.isclose(result, torch.tensor(0.6))

    def test_multiple_updates(self) -> None:
        """Test metric with multiple batch updates."""
        metric = TaskCompletionRate()
        # First batch: 2 out of 3 completed
        metric.update(torch.tensor([1, 0, 1]))
        # Second batch: 1 out of 2 completed
        metric.update(torch.tensor([0, 1]))
        result = metric.compute()
        # Expected: 3 completed out of 5 total = 0.6
        assert torch.isclose(result, torch.tensor(0.6))

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = TaskCompletionRate()
        metric.update(torch.tensor([1, 1, 0]))
        metric.reset()
        metric.update(torch.tensor([1, 1]))
        result = metric.compute()
        assert result == 1.0  # Only counts chains after reset

    def test_ignore_index(self) -> None:
        """Test ignore_index parameter."""
        metric = TaskCompletionRate(ignore_index=-1)
        completion = torch.tensor([1, -1, 0, 1, -1])
        metric.update(completion)
        result = metric.compute()
        # Expected: 2 out of 3 (ignoring -1 values) = 0.6667
        assert torch.isclose(result, torch.tensor(2.0 / 3.0))

    def test_empty_tensor_raises(self) -> None:
        """Test that empty tensor raises ValueError."""
        metric = TaskCompletionRate()
        empty = torch.tensor([])
        with pytest.raises(ValueError, match="empty"):
            metric.update(empty)

    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = TaskCompletionRate()
        with pytest.raises(RuntimeError, match="no task chains have been recorded"):
            metric.compute()

    def test_non_binary_without_threshold_raises(self) -> None:
        """Test that non-binary values without threshold raise ValueError."""
        metric = TaskCompletionRate()
        non_binary = torch.tensor([0.5, 0.8, 0.3])
        with pytest.raises(ValueError, match="binary"):
            metric.update(non_binary)

    def test_bool_input(self) -> None:
        """Test metric with boolean input."""
        metric = TaskCompletionRate()
        completion = torch.tensor([True, False, True, True])
        metric.update(completion)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.75))

    def test_float_binary_input(self) -> None:
        """Test metric with float binary input."""
        metric = TaskCompletionRate()
        completion = torch.tensor([1.0, 0.0, 1.0, 0.0])
        metric.update(completion)
        result = metric.compute()
        assert result == 0.5

    def test_large_batch(self) -> None:
        """Test metric with large batch size."""
        metric = TaskCompletionRate()
        large_batch = torch.randint(0, 2, (1000,))
        metric.update(large_batch)
        result = metric.compute()
        expected = large_batch.float().mean()
        assert torch.isclose(result, expected, atol=1e-6)

    def test_all_ignored(self) -> None:
        """Test that all ignored values doesn't raise error."""
        metric = TaskCompletionRate(ignore_index=-1)
        all_ignored = torch.tensor([-1, -1, -1])
        metric.update(all_ignored)  # Should not raise
        # Compute should raise since no valid data
        with pytest.raises(RuntimeError):
            metric.compute()

    def test_different_dtypes(self) -> None:
        """Test metric with different tensor dtypes."""
        metric = TaskCompletionRate()

        # Test with int
        metric.update(torch.tensor([1, 0, 1], dtype=torch.int32))
        metric.reset()

        # Test with long
        metric.update(torch.tensor([1, 0, 1], dtype=torch.int64))
        metric.reset()

        # Test with float
        metric.update(torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32))
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(2.0 / 3.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_if_available(self) -> None:
        """Test metric on GPU if CUDA is available."""
        metric = TaskCompletionRate().cuda()
        completion = torch.tensor([1, 0, 1, 1], device="cuda")
        metric.update(completion)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.75, device="cuda"))

    def test_sequential_task_chains(self) -> None:
        """Test realistic scenario with sequential task chain evaluation."""
        metric = TaskCompletionRate()

        # Simulate 3 episodes with different task chain lengths
        # Episode 1: pick, move, place - all succeeded
        metric.update(torch.tensor([1]))

        # Episode 2: pick, move, place - failed at move
        metric.update(torch.tensor([0]))

        # Episode 3: pick, move, place - all succeeded
        metric.update(torch.tensor([1]))

        # Episode 4: pick, move, place - failed at place
        metric.update(torch.tensor([0]))

        result = metric.compute()
        # 2 out of 4 task chains completed = 0.5
        assert result == 0.5
