"""Tests for metric normalization, Pareto analysis, and plotting."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robometric_frame.visualization import (  # noqa: E402
    normalize_metrics,
    pareto_chart,
    pareto_front,
    pareto_hypervolume,
    radar_chart,
)


class TestNormalizeMetrics:
    """Tests for normalize_metrics."""

    def test_mixed_directions_and_shape(self) -> None:
        """Normalize multiple policies and orient every metric upward."""
        values = [[1.0, 10.0], [3.0, 5.0], [2.0, 7.5]]
        result = normalize_metrics(values, [True, False])
        np.testing.assert_allclose(result, [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        assert result.shape == (3, 2)

    def test_one_dimensional_explicit_bounds(self) -> None:
        """Preserve a one-dimensional input shape when bounds are supplied."""
        result = normalize_metrics([0.75, 25.0], [True, False], bounds=[[0, 1], [0, 100]])
        np.testing.assert_allclose(result, [0.75, 0.75])
        assert result.shape == (2,)

    def test_missing_values_are_preserved(self) -> None:
        """Do not convert missing metrics to zero."""
        result = normalize_metrics([[1.0, np.nan], [2.0, np.nan]], [True, False])
        np.testing.assert_allclose(result[:, 0], [0.0, 1.0])
        assert np.isnan(result[:, 1]).all()

    def test_clipping_explicit_bounds(self) -> None:
        """Optionally clip values falling outside explicit bounds."""
        result = normalize_metrics([2.0, -1.0], [True, True], bounds=[[0, 1], [0, 1]], clip=True)
        np.testing.assert_allclose(result, [1.0, 0.0])

    @pytest.mark.parametrize(
        ("values", "message"),
        [
            (np.zeros((1, 1, 1)), "one- or two-dimensional"),
            (np.empty((1, 0)), "at least one metric"),
            ([[np.inf]], "infinite"),
        ],
    )
    def test_invalid_metric_arrays(self, values: np.ndarray, message: str) -> None:
        """Reject arrays that cannot represent finite metric rows."""
        with pytest.raises(ValueError, match=message):
            normalize_metrics(values, [True])

    def test_direction_count_and_type_are_validated(self) -> None:
        """Require exactly one boolean direction flag per metric."""
        with pytest.raises(ValueError, match="one flag per metric"):
            normalize_metrics([[1.0, 2.0], [3.0, 4.0]], [True])
        with pytest.raises(TypeError, match="boolean"):
            normalize_metrics([[1.0], [2.0]], [1])

    def test_constant_column_requires_bounds(self) -> None:
        """Reject data-derived normalization of a constant metric."""
        with pytest.raises(ValueError, match="column 0 is constant"):
            normalize_metrics([[1.0], [1.0]], [True])

    @pytest.mark.parametrize(
        ("bounds", "message"),
        [
            ([[0.0, 1.0]], "shape"),
            ([[0.0, np.inf], [0.0, 1.0]], "finite"),
            ([[1.0, 1.0], [0.0, 1.0]], "less than"),
        ],
    )
    def test_invalid_bounds(self, bounds: list[list[float]], message: str) -> None:
        """Reject incompatible, non-finite, and non-increasing bounds."""
        with pytest.raises(ValueError, match=message):
            normalize_metrics([[1.0, 2.0]], [True, True], bounds=bounds)


class TestParetoAnalysis:
    """Tests for Pareto-front membership and exact hypervolume."""

    def test_front_with_mixed_directions_duplicates_and_missing(self) -> None:
        """Retain trade-offs and duplicates while excluding incomplete rows."""
        values = [[0.9, 10.0], [0.8, 8.0], [0.7, 12.0], [0.9, 10.0], [np.nan, 1.0]]
        result = pareto_front(values, [True, False])
        np.testing.assert_array_equal(result, [True, True, False, True, False])

    def test_two_dimensional_hypervolume(self) -> None:
        """Compute the exact union area of overlapping dominated rectangles."""
        result = pareto_hypervolume([[2.0, 1.0], [1.0, 2.0]], [0.0, 0.0], [True, True])
        assert result == pytest.approx(3.0)

    def test_hypervolume_with_mixed_directions(self) -> None:
        """Orient minimizing and maximizing objectives before integration."""
        result = pareto_hypervolume([[0.9, 10.0], [0.8, 8.0]], [0.5, 20.0], [True, False])
        assert result == pytest.approx(4.6)

    def test_three_dimensional_hypervolume(self) -> None:
        """Support more objectives than the two-dimensional plotter."""
        result = pareto_hypervolume([[2.0, 3.0, 4.0]], [0.0, 0.0, 0.0], [True] * 3)
        assert result == pytest.approx(24.0)

    def test_only_missing_rows_have_zero_volume(self) -> None:
        """Ignore incomplete rows without treating missing values as zero."""
        result = pareto_hypervolume([[np.nan, 1.0]], [0.0, 0.0], [True, True])
        assert result == 0.0

    @pytest.mark.parametrize(
        ("reference", "message"),
        [
            ([0.0], "one value per metric"),
            ([0.0, np.inf], "finite"),
            ([3.0, 0.0], "no better"),
        ],
    )
    def test_invalid_reference_points(self, reference: list[float], message: str) -> None:
        """Require a finite reference that bounds every complete candidate."""
        with pytest.raises(ValueError, match=message):
            pareto_hypervolume([[2.0, 1.0]], reference, [True, True])


class TestRadarChart:
    """Tests for the radar-chart renderer."""

    def teardown_method(self) -> None:
        """Close figures created by each plotting test."""
        plt.close("all")

    def test_creates_polar_chart_with_default_labels(self) -> None:
        """Create a closed line and filled area for a complete series."""
        figure, axes = radar_chart([0.2, 0.5, 0.8], ["A", "B", "C"])
        assert axes.name == "polar"
        assert axes.figure is figure
        assert len(axes.lines) == 1
        assert len(axes.patches) == 1
        assert axes.get_legend().get_texts()[0].get_text() == "Series 1"

    def test_uses_existing_axes_and_preserves_missing_gap(self) -> None:
        """Draw on supplied polar axes and skip fill for incomplete rows."""
        figure, axes = plt.subplots(subplot_kw={"projection": "polar"})
        result_figure, result_axes = radar_chart(
            [[0.2, np.nan, 0.8], [0.5, 0.6, 0.7]],
            ["A", "B", "C"],
            series_names=["Incomplete", "Complete"],
            ax=axes,
            fill_alpha=0.0,
        )
        assert result_figure is figure
        assert result_axes is axes
        assert len(axes.lines) == 2
        assert len(axes.patches) == 0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"metric_names": ["A"]}, "one label per metric"),
            ({"metric_names": ["A", "B"], "series_names": ["one", "two"]}, "one label per row"),
            ({"metric_names": ["A", "B"], "fill_alpha": 2.0}, "between zero and one"),
        ],
    )
    def test_invalid_chart_configuration(self, kwargs: dict[str, object], message: str) -> None:
        """Validate labels and fill opacity."""
        with pytest.raises(ValueError, match=message):
            radar_chart([0.2, 0.8], **kwargs)

    def test_rejects_unnormalized_values_and_cartesian_axes(self) -> None:
        """Require normalized data and polar axes."""
        with pytest.raises(ValueError, match="normalized"):
            radar_chart([0.0, 1.1], ["A", "B"])
        _, axes = plt.subplots()
        with pytest.raises(ValueError, match="polar"):
            radar_chart([0.0, 1.0], ["A", "B"], ax=axes)


class TestParetoChart:
    """Tests for the two-dimensional Pareto renderer."""

    def teardown_method(self) -> None:
        """Close figures created by each plotting test."""
        plt.close("all")

    def test_plots_front_hypervolume_reference_and_annotations(self) -> None:
        """Render all comparison layers without displaying the figure."""
        figure, axes = pareto_chart(
            [[2.0, 1.0], [1.0, 2.0], [1.0, 1.0], [np.nan, 3.0]],
            ["Reward", "Safety"],
            [True, True],
            reference_point=[0.0, 0.0],
            labels=["A", "B", "C", "Missing"],
        )
        assert axes.figure is figure
        assert axes.get_title() == "Dominated hypervolume: 3"
        assert len(axes.collections) == 3
        assert len(axes.patches) == 2
        assert [text.get_text() for text in axes.texts] == ["A", "B", "C"]
        assert "higher is better" in axes.get_xlabel()

    def test_uses_existing_axes_without_reference(self) -> None:
        """Allow callers to compose a Pareto chart into an existing figure."""
        figure, axes = plt.subplots()
        result_figure, result_axes = pareto_chart(
            [[1.0, 2.0], [2.0, 1.0]], ["Cost", "Latency"], [False, False], ax=axes
        )
        assert result_figure is figure
        assert result_axes is axes
        assert "lower is better" in axes.get_ylabel()

    def test_all_missing_data_creates_empty_chart(self) -> None:
        """Create labeled axes without inventing points for missing rows."""
        _, axes = pareto_chart([[np.nan, 1.0]], ["A", "B"], [True, True])
        assert not axes.collections
        assert axes.get_legend() is None

    @pytest.mark.parametrize(
        ("values", "names", "labels", "message"),
        [
            ([[1.0, 2.0, 3.0]], ["A", "B"], None, "exactly two objectives"),
            ([[1.0, 2.0]], ["A"], None, "exactly two labels"),
            ([[1.0, 2.0]], ["A", "B"], ["one", "two"], "one value per row"),
        ],
    )
    def test_invalid_chart_configuration(
        self,
        values: list[list[float]],
        names: list[str],
        labels: list[str] | None,
        message: str,
    ) -> None:
        """Require a valid two-objective plot configuration."""
        with pytest.raises(ValueError, match=message):
            pareto_chart(values, names, [True] * len(values[0]), labels=labels)
