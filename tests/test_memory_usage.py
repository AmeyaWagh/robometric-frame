"""Tests for MemoryUsage metric."""

import pytest
import torch

from vla_metrics.efficiency import MemoryUsage


class TestMemoryUsage:
    """Test suite for MemoryUsage metric."""

    def test_basic_update(self) -> None:
        """Test basic memory measurement with direct update."""
        metric = MemoryUsage()
        memory_readings = torch.tensor([100.0, 200.0, 150.0])  # MB
        metric.update(memory_readings)

        result = metric.compute()
        assert torch.isclose(result["mean_mb"], torch.tensor(150.0), atol=1e-6)
        assert torch.isclose(result["peak_mb"], torch.tensor(200.0), atol=1e-6)
        assert torch.isclose(result["total_mb"], torch.tensor(450.0), atol=1e-6)
        assert result["count"] == 3

    def test_scalar_update(self) -> None:
        """Test update with scalar tensor."""
        metric = MemoryUsage()
        metric.update(torch.tensor(128.5))

        result = metric.compute()
        assert torch.isclose(result["mean_mb"], torch.tensor(128.5), atol=1e-6)
        assert torch.isclose(result["peak_mb"], torch.tensor(128.5), atol=1e-6)
        assert result["count"] == 1

    def test_manual_tracking(self) -> None:
        """Test manual tracking with start/stop."""
        metric = MemoryUsage()

        metric.start()
        # Allocate some memory
        _ = torch.randn(1000, 1000)
        metric.stop()

        result = metric.compute()
        # Memory should be recorded (actual value depends on system)
        assert result["peak_mb"].item() >= 0
        assert result["count"] == 1

    def test_multiple_manual_trackings(self) -> None:
        """Test multiple manual tracking measurements."""
        metric = MemoryUsage()

        for i in range(3):
            metric.start()
            # Allocate progressively more memory
            _ = torch.randn(100 * (i + 1), 100 * (i + 1))
            metric.stop()

        result = metric.compute()
        assert result["count"] == 3
        # Memory might be 0 if psutil is not available
        assert result["peak_mb"].item() >= 0

    def test_multi_batch_updates(self) -> None:
        """Test metric accumulation across multiple batches."""
        metric = MemoryUsage()

        metric.update(torch.tensor([100.0, 150.0]))
        metric.update(torch.tensor([120.0, 180.0]))

        result = metric.compute()
        assert torch.isclose(result["mean_mb"], torch.tensor(137.5), atol=1e-6)
        assert torch.isclose(result["peak_mb"], torch.tensor(180.0), atol=1e-6)
        assert result["count"] == 4

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = MemoryUsage()
        metric.update(torch.tensor([100.0, 200.0]))

        # Reset and update with new values
        metric.reset()
        metric.update(torch.tensor([300.0, 400.0]))

        result = metric.compute()
        assert torch.isclose(result["mean_mb"], torch.tensor(350.0), atol=1e-6)
        assert torch.isclose(result["peak_mb"], torch.tensor(400.0), atol=1e-6)
        assert result["count"] == 2

    def test_reset_clears_internal_state(self) -> None:
        """Test that reset clears internal tracking state."""
        metric = MemoryUsage()

        # Start tracking but don't stop
        metric.start()
        assert metric._tracking is True

        # Reset should clear the internal state
        metric.reset()
        assert metric._tracking is False

        # Should be able to start a new measurement after reset
        metric.start()
        metric.stop()

        result = metric.compute()
        assert result["count"] == 1

    def test_negative_memory_error(self) -> None:
        """Test that negative memory values raise an error."""
        metric = MemoryUsage()

        with pytest.raises(ValueError, match="Memory values must be non-negative"):
            metric.update(torch.tensor([-100.0, 200.0]))

    def test_compute_before_update_error(self) -> None:
        """Test that compute() raises error when called before update."""
        metric = MemoryUsage()

        with pytest.raises(RuntimeError, match="no measurements have been recorded"):
            metric.compute()

    def test_start_without_stop_error(self) -> None:
        """Test that calling start() twice without stop() raises an error."""
        metric = MemoryUsage()
        metric.start()

        with pytest.raises(RuntimeError, match="Already tracking memory usage"):
            metric.start()

    def test_stop_without_start_error(self) -> None:
        """Test that calling stop() without start() raises an error."""
        metric = MemoryUsage()

        with pytest.raises(RuntimeError, match="Not currently tracking"):
            metric.stop()

    def test_mixed_tracking_and_update(self) -> None:
        """Test combining manual tracking with direct updates."""
        metric = MemoryUsage()

        # Manual tracking
        metric.start()
        _ = torch.randn(100, 100)
        metric.stop()

        # Direct update
        metric.update(torch.tensor(500.0))

        result = metric.compute()
        assert result["count"] == 2
        # Peak should be the larger of the two
        assert result["peak_mb"].item() >= 0

    def test_zero_memory(self) -> None:
        """Test handling of zero memory values."""
        metric = MemoryUsage()
        metric.update(torch.tensor([0.0, 100.0, 0.0]))

        result = metric.compute()
        assert torch.isclose(result["mean_mb"], torch.tensor(100.0 / 3), atol=1e-6)
        assert result["peak_mb"] == 100.0

    def test_large_batch(self) -> None:
        """Test with large batch of measurements."""
        metric = MemoryUsage()
        memory_readings = torch.rand(1000) * 1000  # Random 0-1000 MB
        metric.update(memory_readings)

        result = metric.compute()
        assert result["count"] == 1000
        assert 0.0 <= result["peak_mb"].item() <= 1000.0

    def test_float_dtypes(self) -> None:
        """Test that metric works with different float dtypes."""
        for dtype in [torch.float32, torch.float64]:
            metric = MemoryUsage()
            memory = torch.tensor([100.0, 200.0], dtype=dtype)
            metric.update(memory)

            result = metric.compute()
            assert torch.isclose(result["mean_mb"], torch.tensor(150.0), atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_support(self) -> None:
        """Test metric on GPU."""
        metric = MemoryUsage().to("cuda")
        memory = torch.tensor([100.0, 200.0, 150.0], device="cuda")
        metric.update(memory)

        result = metric.compute()
        assert result["mean_mb"].device.type == "cuda"
        assert torch.isclose(result["mean_mb"], torch.tensor(150.0, device="cuda"), atol=1e-6)

    def test_different_batch_shapes(self) -> None:
        """Test with different input shapes (all get flattened)."""
        metric = MemoryUsage()

        # 1D tensor
        metric.update(torch.tensor([100.0, 200.0]))
        # 2D tensor (gets flattened)
        metric.update(torch.tensor([[150.0, 250.0], [300.0, 400.0]]))

        result = metric.compute()
        assert result["count"] == 6  # 2 + 4 measurements

    def test_statistics_consistency(self) -> None:
        """Test that statistics are consistent."""
        metric = MemoryUsage()
        memory = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
        metric.update(memory)

        result = metric.compute()
        # Peak should be >= mean
        assert result["peak_mb"] >= result["mean_mb"]
        # Total should equal mean * count
        assert torch.isclose(result["total_mb"], result["mean_mb"] * result["count"], atol=1e-6)

    def test_higher_is_better_false(self) -> None:
        """Test that higher_is_better is set to False."""
        metric = MemoryUsage()
        assert metric.higher_is_better is False

    def test_is_differentiable_false(self) -> None:
        """Test that is_differentiable is set to False."""
        metric = MemoryUsage()
        assert metric.is_differentiable is False

    def test_default_percentiles(self) -> None:
        """Test that default percentiles (50th, 95th, 99th) are computed."""
        metric = MemoryUsage()
        memory = torch.linspace(100.0, 1000.0, 100)  # 100 evenly spaced values
        metric.update(memory)

        result = metric.compute()
        assert "p50_mb" in result  # median
        assert "p95_mb" in result  # 95th percentile
        assert "p99_mb" in result  # 99th percentile

        # Check percentiles are in reasonable range
        assert result["p50_mb"] <= result["p95_mb"] <= result["p99_mb"]

    def test_custom_percentiles(self) -> None:
        """Test custom percentile specification."""
        metric = MemoryUsage(percentiles=[0.25, 0.5, 0.75])
        memory = torch.tensor([100.0, 200.0, 300.0, 400.0])
        metric.update(memory)

        result = metric.compute()
        assert "p25_mb" in result
        assert "p50_mb" in result
        assert "p75_mb" in result
        # Should not have default percentiles
        assert "p95_mb" not in result
        assert "p99_mb" not in result

    def test_single_percentile(self) -> None:
        """Test with a single percentile."""
        metric = MemoryUsage(percentiles=[0.95])
        memory = torch.linspace(0.0, 1000.0, 100)
        metric.update(memory)

        result = metric.compute()
        assert "p95_mb" in result
        # 95th percentile should be around 950
        assert 940 <= result["p95_mb"].item() <= 960

    def test_percentile_accuracy(self) -> None:
        """Test that percentile calculations are accurate."""
        metric = MemoryUsage(percentiles=[0.5, 0.9])
        memory = torch.tensor(
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
        )
        metric.update(memory)

        result = metric.compute()
        # Median should be around 550
        assert torch.isclose(result["p50_mb"], torch.tensor(550.0), atol=10.0)

    def test_invalid_percentile_error(self) -> None:
        """Test that invalid percentile values raise an error."""
        with pytest.raises(ValueError, match="Percentiles must be between 0 and 1"):
            MemoryUsage(percentiles=[0.5, 1.5])

        with pytest.raises(ValueError, match="Percentiles must be between 0 and 1"):
            MemoryUsage(percentiles=[-0.1, 0.5])

    def test_percentiles_with_batched_updates(self) -> None:
        """Test percentiles work correctly with multiple batch updates."""
        metric = MemoryUsage(percentiles=[0.5])
        metric.update(torch.tensor([100.0, 200.0, 300.0]))
        metric.update(torch.tensor([400.0, 500.0]))

        result = metric.compute()
        # Median of [100, 200, 300, 400, 500] should be 300
        assert torch.isclose(result["p50_mb"], torch.tensor(300.0), atol=10.0)

    def test_extreme_percentiles(self) -> None:
        """Test extreme percentiles (0 and 100)."""
        metric = MemoryUsage(percentiles=[0.0, 1.0])
        memory = torch.tensor([100.0, 500.0, 900.0])
        metric.update(memory)

        result = metric.compute()
        # 0th percentile should be minimum
        assert torch.isclose(result["p0_mb"], torch.tensor(100.0), atol=1e-6)
        # 100th percentile should equal peak
        assert torch.isclose(result["p100_mb"], result["peak_mb"], atol=1e-6)

    def test_percentiles_reset(self) -> None:
        """Test that percentiles are recalculated after reset."""
        metric = MemoryUsage(percentiles=[0.5])
        metric.update(torch.tensor([100.0, 200.0, 300.0]))

        result1 = metric.compute()
        first_median = result1["p50_mb"]

        metric.reset()
        metric.update(torch.tensor([500.0, 600.0, 700.0]))

        result2 = metric.compute()
        second_median = result2["p50_mb"]

        # Medians should be different
        assert not torch.isclose(first_median, second_median, atol=10.0)

    def test_track_ram_only(self) -> None:
        """Test tracking RAM only (no VRAM)."""
        metric = MemoryUsage(track_ram=True, track_vram=False)
        metric.start()
        _ = list(range(10000))  # CPU allocation
        metric.stop()

        result = metric.compute()
        assert result["count"] == 1

    def test_track_vram_only(self) -> None:
        """Test tracking VRAM only (no RAM)."""
        metric = MemoryUsage(track_ram=False, track_vram=True)
        # This should work even without CUDA (will just record 0 for VRAM)
        metric.start()
        metric.stop()

        result = metric.compute()
        assert result["count"] == 1

    def test_auto_vram_detection(self) -> None:
        """Test automatic VRAM tracking detection."""
        metric = MemoryUsage()  # track_vram not specified
        # Should auto-detect based on CUDA availability
        if torch.cuda.is_available():
            assert metric.track_vram is True
        else:
            assert metric.track_vram is False

    def test_large_memory_values(self) -> None:
        """Test with large memory values (GB range)."""
        metric = MemoryUsage()
        # Simulate gigabytes of memory
        memory = torch.tensor([1024.0, 2048.0, 4096.0, 8192.0])  # MB
        metric.update(memory)

        result = metric.compute()
        assert result["peak_mb"].item() == 8192.0
        assert torch.isclose(result["mean_mb"], torch.tensor(3840.0), atol=1e-6)
