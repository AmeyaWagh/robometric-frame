"""Tests for InferenceLatency metric."""

import time

import pytest
import torch

from vla_metrics.efficiency import InferenceLatency


class TestInferenceLatency:
    """Test suite for InferenceLatency metric."""

    def test_basic_update(self) -> None:
        """Test basic latency measurement with direct update."""
        metric = InferenceLatency()
        latencies = torch.tensor([0.1, 0.2, 0.15])
        metric.update(latencies)

        result = metric.compute()
        assert torch.isclose(result["mean"], torch.tensor(0.15), atol=1e-6)
        assert torch.isclose(result["min"], torch.tensor(0.1), atol=1e-6)
        assert torch.isclose(result["max"], torch.tensor(0.2), atol=1e-6)
        assert torch.isclose(result["total"], torch.tensor(0.45), atol=1e-6)
        assert result["count"] == 3

    def test_scalar_update(self) -> None:
        """Test update with scalar tensor."""
        metric = InferenceLatency()
        metric.update(torch.tensor(0.1))

        result = metric.compute()
        assert torch.isclose(result["mean"], torch.tensor(0.1), atol=1e-6)
        assert result["count"] == 1

    def test_manual_timing(self) -> None:
        """Test manual timing with start/stop."""
        metric = InferenceLatency()

        metric.start()
        time.sleep(0.01)  # Sleep for ~10ms
        metric.stop()

        result = metric.compute()
        # Check that latency is reasonable (between 5ms and 50ms to account for variance)
        assert 0.005 < result["mean"].item() < 0.05
        assert result["count"] == 1

    def test_multiple_manual_timings(self) -> None:
        """Test multiple manual timing measurements."""
        metric = InferenceLatency()

        for _ in range(3):
            metric.start()
            time.sleep(0.01)
            metric.stop()

        result = metric.compute()
        assert result["count"] == 3
        # Mean should be around 10ms
        assert 0.005 < result["mean"].item() < 0.05

    def test_multi_batch_updates(self) -> None:
        """Test metric accumulation across multiple batches."""
        metric = InferenceLatency()

        metric.update(torch.tensor([0.1, 0.2]))
        metric.update(torch.tensor([0.15, 0.25]))

        result = metric.compute()
        assert torch.isclose(result["mean"], torch.tensor(0.175), atol=1e-6)
        assert torch.isclose(result["min"], torch.tensor(0.1), atol=1e-6)
        assert torch.isclose(result["max"], torch.tensor(0.25), atol=1e-6)
        assert result["count"] == 4

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = InferenceLatency()
        metric.update(torch.tensor([0.1, 0.2]))

        # Reset and update with new values
        metric.reset()
        metric.update(torch.tensor([0.3, 0.4]))

        result = metric.compute()
        assert torch.isclose(result["mean"], torch.tensor(0.35), atol=1e-6)
        assert torch.isclose(result["min"], torch.tensor(0.3), atol=1e-6)
        assert torch.isclose(result["max"], torch.tensor(0.4), atol=1e-6)
        assert result["count"] == 2

    def test_reset_clears_internal_state(self) -> None:
        """Test that reset clears internal timing state."""
        metric = InferenceLatency()

        # Start timing but don't stop
        metric.start()
        assert metric._start_time is not None

        # Reset should clear the internal state
        metric.reset()
        assert metric._start_time is None

        # Should be able to start a new measurement after reset
        metric.start()
        time.sleep(0.01)
        metric.stop()

        result = metric.compute()
        assert result["count"] == 1

    def test_negative_latency_error(self) -> None:
        """Test that negative latency values raise an error."""
        metric = InferenceLatency()

        with pytest.raises(ValueError, match="Latency values must be non-negative"):
            metric.update(torch.tensor([-0.1, 0.2]))

    def test_compute_before_update_error(self) -> None:
        """Test that compute() raises error when called before update."""
        metric = InferenceLatency()

        with pytest.raises(RuntimeError, match="no measurements have been recorded"):
            metric.compute()

    def test_start_without_stop_error(self) -> None:
        """Test that calling start() twice without stop() raises an error."""
        metric = InferenceLatency()
        metric.start()

        with pytest.raises(RuntimeError, match="Timer already started"):
            metric.start()

    def test_stop_without_start_error(self) -> None:
        """Test that calling stop() without start() raises an error."""
        metric = InferenceLatency()

        with pytest.raises(RuntimeError, match="Timer not started"):
            metric.stop()

    def test_mixed_timing_and_update(self) -> None:
        """Test combining manual timing with direct updates."""
        metric = InferenceLatency()

        # Manual timing
        metric.start()
        time.sleep(0.01)
        metric.stop()

        # Direct update
        metric.update(torch.tensor(0.1))

        result = metric.compute()
        assert result["count"] == 2
        # Min should be the smaller of the two
        assert result["min"].item() <= 0.1

    def test_zero_latency(self) -> None:
        """Test handling of zero latency values."""
        metric = InferenceLatency()
        metric.update(torch.tensor([0.0, 0.1, 0.0]))

        result = metric.compute()
        assert torch.isclose(result["mean"], torch.tensor(0.1 / 3), atol=1e-6)
        assert result["min"] == 0.0
        assert result["max"] == 0.1

    def test_large_batch(self) -> None:
        """Test with large batch of measurements."""
        metric = InferenceLatency()
        latencies = torch.rand(1000) * 0.5  # Random latencies between 0 and 0.5
        metric.update(latencies)

        result = metric.compute()
        assert result["count"] == 1000
        assert 0.0 <= result["min"].item() <= result["mean"].item() <= result["max"].item() <= 0.5

    def test_float_dtypes(self) -> None:
        """Test that metric works with different float dtypes."""
        for dtype in [torch.float32, torch.float64]:
            metric = InferenceLatency()
            latencies = torch.tensor([0.1, 0.2], dtype=dtype)
            metric.update(latencies)

            result = metric.compute()
            assert torch.isclose(result["mean"], torch.tensor(0.15), atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self) -> None:
        """Test metric on GPU."""
        metric = InferenceLatency().to("cuda")
        latencies = torch.tensor([0.1, 0.2, 0.15], device="cuda")
        metric.update(latencies)

        result = metric.compute()
        assert result["mean"].device.type == "cuda"
        assert torch.isclose(result["mean"], torch.tensor(0.15, device="cuda"), atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_manual_timing(self) -> None:
        """Test manual timing on GPU with CUDA synchronization."""
        metric = InferenceLatency().to("cuda")

        metric.start()
        # Simulate some GPU work
        _ = torch.randn(1000, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")
        metric.stop()

        result = metric.compute()
        assert result["mean"].device.type == "cuda"
        assert result["mean"].item() > 0
        assert result["count"] == 1

    def test_different_batch_shapes(self) -> None:
        """Test with different input shapes (all get flattened)."""
        metric = InferenceLatency()

        # 1D tensor
        metric.update(torch.tensor([0.1, 0.2]))
        # 2D tensor (gets flattened)
        metric.update(torch.tensor([[0.15, 0.25], [0.3, 0.4]]))

        result = metric.compute()
        assert result["count"] == 6  # 2 + 4 measurements

    def test_statistics_consistency(self) -> None:
        """Test that statistics are consistent."""
        metric = InferenceLatency()
        latencies = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        metric.update(latencies)

        result = metric.compute()
        # Mean should be between min and max
        assert result["min"] <= result["mean"] <= result["max"]
        # Total should equal mean * count
        assert torch.isclose(result["total"], result["mean"] * result["count"], atol=1e-6)

    def test_very_small_latencies(self) -> None:
        """Test with very small (microsecond) latencies."""
        metric = InferenceLatency()
        # Simulate microsecond-level latencies
        latencies = torch.tensor([1e-6, 2e-6, 3e-6])
        metric.update(latencies)

        result = metric.compute()
        assert torch.isclose(result["mean"], torch.tensor(2e-6), atol=1e-9)
        assert result["min"] == 1e-6
        assert result["max"] == 3e-6

    def test_higher_is_better_false(self) -> None:
        """Test that higher_is_better is set to False."""
        metric = InferenceLatency()
        assert metric.higher_is_better is False

    def test_is_differentiable_false(self) -> None:
        """Test that is_differentiable is set to False."""
        metric = InferenceLatency()
        assert metric.is_differentiable is False

    def test_default_percentiles(self) -> None:
        """Test that default percentiles (50th, 95th, 99th) are computed."""
        metric = InferenceLatency()
        latencies = torch.linspace(0.1, 1.0, 100)  # 100 evenly spaced values
        metric.update(latencies)

        result = metric.compute()
        assert "p50" in result  # median
        assert "p95" in result  # 95th percentile
        assert "p99" in result  # 99th percentile

        # Check percentiles are in reasonable range
        assert result["min"] <= result["p50"] <= result["max"]
        assert result["p50"] <= result["p95"] <= result["p99"]

    def test_custom_percentiles(self) -> None:
        """Test custom percentile specification."""
        metric = InferenceLatency(percentiles=[0.25, 0.5, 0.75])
        latencies = torch.tensor([0.1, 0.2, 0.3, 0.4])
        metric.update(latencies)

        result = metric.compute()
        assert "p25" in result
        assert "p50" in result
        assert "p75" in result
        # Should not have default percentiles
        assert "p95" not in result
        assert "p99" not in result

    def test_single_percentile(self) -> None:
        """Test with a single percentile."""
        metric = InferenceLatency(percentiles=[0.95])
        latencies = torch.linspace(0.0, 1.0, 100)
        metric.update(latencies)

        result = metric.compute()
        assert "p95" in result
        # 95th percentile of values from 0 to 1 should be around 0.95
        assert 0.94 <= result["p95"].item() <= 0.96

    def test_percentile_accuracy(self) -> None:
        """Test that percentile calculations are accurate."""
        metric = InferenceLatency(percentiles=[0.5, 0.9])
        # Known distribution: 0.1 to 1.0 in steps of 0.1
        latencies = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        metric.update(latencies)

        result = metric.compute()
        # Median of 10 values should be average of 5th and 6th values
        assert torch.isclose(result["p50"], torch.tensor(0.55), atol=0.01)
        # 90th percentile (torch.quantile uses linear interpolation)
        assert result["p90"].item() >= 0.9  # Should be between 0.9 and 1.0

    def test_invalid_percentile_error(self) -> None:
        """Test that invalid percentile values raise an error."""
        with pytest.raises(ValueError, match="Percentiles must be between 0 and 1"):
            InferenceLatency(percentiles=[0.5, 1.5])

        with pytest.raises(ValueError, match="Percentiles must be between 0 and 1"):
            InferenceLatency(percentiles=[-0.1, 0.5])

    def test_percentiles_with_batched_updates(self) -> None:
        """Test percentiles work correctly with multiple batch updates."""
        metric = InferenceLatency(percentiles=[0.5])
        metric.update(torch.tensor([0.1, 0.2, 0.3]))
        metric.update(torch.tensor([0.4, 0.5]))

        result = metric.compute()
        # Median of [0.1, 0.2, 0.3, 0.4, 0.5] should be 0.3
        assert torch.isclose(result["p50"], torch.tensor(0.3), atol=0.01)

    def test_extreme_percentiles(self) -> None:
        """Test extreme percentiles (0 and 100)."""
        metric = InferenceLatency(percentiles=[0.0, 1.0])
        latencies = torch.tensor([0.1, 0.5, 0.9])
        metric.update(latencies)

        result = metric.compute()
        # 0th percentile should equal min
        assert torch.isclose(result["p0"], result["min"], atol=1e-6)
        # 100th percentile should equal max
        assert torch.isclose(result["p100"], result["max"], atol=1e-6)

    def test_percentiles_reset(self) -> None:
        """Test that percentiles are recalculated after reset."""
        metric = InferenceLatency(percentiles=[0.5])
        metric.update(torch.tensor([0.1, 0.2, 0.3]))

        result1 = metric.compute()
        first_median = result1["p50"]

        metric.reset()
        metric.update(torch.tensor([0.5, 0.6, 0.7]))

        result2 = metric.compute()
        second_median = result2["p50"]

        # Medians should be different
        assert not torch.isclose(first_median, second_median, atol=0.01)
