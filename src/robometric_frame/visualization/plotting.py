"""Matplotlib plots for robotics metric comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from robometric_frame.visualization._validation import as_directions, as_metric_matrix
from robometric_frame.visualization.pareto import pareto_front, pareto_hypervolume


def radar_chart(
    values: ArrayLike,
    metric_names: Sequence[str],
    *,
    series_names: Sequence[str] | None = None,
    ax: Any | None = None,
    fill_alpha: float = 0.1,
) -> tuple[Any, Any]:
    """Plot normalized metric rows on a radar chart.

    Args:
        values: Normalized, higher-is-better values with shape ``(M,)`` or
            ``(N, M)``. Missing values are left as gaps.
        metric_names: Labels for the radar axes.
        series_names: Optional label for each row.
        ax: Optional Matplotlib polar axes on which to draw.
        fill_alpha: Opacity of the area under complete series. Set to zero to
            disable filling.

    Returns:
        The Matplotlib ``(figure, axes)`` pair. The function never calls
        ``show()``.
    """
    matrix, _ = as_metric_matrix(values)
    if len(metric_names) != matrix.shape[1]:
        raise ValueError("metric_names must contain one label per metric")
    finite = matrix[np.isfinite(matrix)]
    if finite.size and (np.any(finite < 0.0) or np.any(finite > 1.0)):
        raise ValueError("radar chart values must be normalized between zero and one")
    if not 0.0 <= fill_alpha <= 1.0:
        raise ValueError("fill_alpha must be between zero and one")

    labels = _series_labels(series_names, matrix.shape[0])
    pyplot = _import_pyplot()
    if ax is None:
        figure, ax = pyplot.subplots(subplot_kw={"projection": "polar"})
    else:
        if getattr(ax, "name", None) != "polar":
            raise ValueError("ax must use a polar projection")
        figure = ax.figure

    angles = np.linspace(0.0, 2.0 * np.pi, matrix.shape[1], endpoint=False)
    closed_angles = np.concatenate((angles, angles[:1]))
    for row, label in zip(matrix, labels):
        closed_values = np.concatenate((row, row[:1]))
        line = ax.plot(closed_angles, closed_values, label=label)[0]
        if fill_alpha > 0.0 and np.isfinite(row).all():
            ax.fill(closed_angles, closed_values, alpha=fill_alpha, color=line.get_color())

    ax.set_xticks(angles)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.0)
    if labels:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    return figure, ax


def pareto_chart(
    values: ArrayLike,
    objective_names: Sequence[str],
    higher_is_better: Sequence[bool],
    *,
    reference_point: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot a two-objective Pareto front and optional dominated hypervolume.

    Args:
        values: Candidate metric rows with shape ``(N, 2)``.
        objective_names: Labels for the x and y objectives.
        higher_is_better: Direction flag for each objective.
        reference_point: Optional point bounding the dominated hypervolume.
        labels: Optional annotation for each candidate.
        ax: Optional Matplotlib axes on which to draw.

    Returns:
        The Matplotlib ``(figure, axes)`` pair.
    """
    matrix, _ = as_metric_matrix(values)
    if matrix.shape[1] != 2:
        raise ValueError("pareto_chart requires exactly two objectives")
    if len(objective_names) != 2:
        raise ValueError("objective_names must contain exactly two labels")
    directions = as_directions(higher_is_better, 2)
    annotations = _optional_labels(labels, matrix.shape[0])

    pyplot = _import_pyplot()
    if ax is None:
        figure, ax = pyplot.subplots()
    else:
        figure = ax.figure

    complete = ~np.isnan(matrix).any(axis=1)
    front = pareto_front(matrix, higher_is_better)
    dominated = complete & ~front
    if dominated.any():
        ax.scatter(matrix[dominated, 0], matrix[dominated, 1], label="Dominated", alpha=0.6)
    if front.any():
        ax.scatter(matrix[front, 0], matrix[front, 1], label="Pareto front", zorder=3)
        order = np.argsort(matrix[front, 0])
        front_values = matrix[front][order]
        ax.plot(front_values[:, 0], front_values[:, 1], alpha=0.6)

    for row, label in zip(matrix, annotations):
        if label and np.isfinite(row).all():
            ax.annotate(label, (row[0], row[1]))

    if reference_point is not None:
        reference = np.asarray(reference_point, dtype=np.float64)
        volume = pareto_hypervolume(matrix, reference, higher_is_better)
        for point in matrix[front]:
            origin = np.minimum(point, reference)
            size = np.abs(point - reference)
            rectangle = pyplot.Rectangle(origin, size[0], size[1], alpha=0.08, zorder=0)
            ax.add_patch(rectangle)
        ax.scatter(reference[0], reference[1], marker="x", label="Reference", zorder=3)
        ax.set_title(f"Dominated hypervolume: {volume:.4g}")

    direction_labels = np.where(directions, "higher is better", "lower is better")
    ax.set_xlabel(f"{objective_names[0]} ({direction_labels[0]})")
    ax.set_ylabel(f"{objective_names[1]} ({direction_labels[1]})")
    if complete.any() or reference_point is not None:
        ax.legend()
    return figure, ax


def _series_labels(labels: Sequence[str] | None, expected: int) -> list[str]:
    """Create or validate radar-series labels."""
    if labels is None:
        return [f"Series {index + 1}" for index in range(expected)]
    if len(labels) != expected:
        raise ValueError("series_names must contain one label per row")
    return list(labels)


def _optional_labels(labels: Sequence[str] | None, expected: int) -> list[str | None]:
    """Validate optional point annotations."""
    if labels is None:
        return [None] * expected
    if len(labels) != expected:
        raise ValueError("labels must contain one value per row")
    return list(labels)


def _import_pyplot() -> Any:
    """Import Matplotlib lazily so analysis utilities remain dependency-light."""
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Plotting requires Matplotlib; install robometric-frame[visualization]"
        ) from error
    return pyplot
