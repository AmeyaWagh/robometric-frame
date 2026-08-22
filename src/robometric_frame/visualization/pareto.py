"""Pareto-front and hypervolume analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robometric_frame.visualization._validation import as_directions, as_metric_matrix


def pareto_front(values: ArrayLike, higher_is_better: Sequence[bool]) -> NDArray[np.bool_]:
    """Return a mask selecting non-dominated metric rows.

    Rows containing missing values are excluded because dominance cannot be
    established without every objective. Duplicate non-dominated rows are all
    retained.

    Args:
        values: Candidate metric rows with shape ``(N, M)``.
        higher_is_better: One boolean per metric describing its direction.

    Returns:
        A boolean mask with one entry per candidate row.
    """
    matrix, _ = as_metric_matrix(values)
    directions = as_directions(higher_is_better, matrix.shape[1])
    complete = ~np.isnan(matrix).any(axis=1)
    oriented = matrix * np.where(directions, 1.0, -1.0)
    result = np.zeros(matrix.shape[0], dtype=np.bool_)

    for index in np.flatnonzero(complete):
        candidates = oriented[complete]
        weakly_better = np.all(candidates >= oriented[index], axis=1)
        strictly_better = np.any(candidates > oriented[index], axis=1)
        result[index] = not np.any(weakly_better & strictly_better)
    return result


def pareto_hypervolume(
    values: ArrayLike,
    reference_point: ArrayLike,
    higher_is_better: Sequence[bool],
) -> float:
    """Calculate exact dominated hypervolume relative to a reference point.

    The reference point must be no better than every complete candidate in
    every objective. Rows containing ``NaN`` are ignored rather than replacing
    their missing metrics with zero.

    Args:
        values: Candidate metric rows with shape ``(N, M)``.
        reference_point: A worse point bounding the dominated region, shape
            ``(M,)``.
        higher_is_better: One boolean per metric describing its direction.

    Returns:
        The union volume dominated by the non-dominated candidates.

    Raises:
        ValueError: If the reference point has an invalid shape, is non-finite,
            or is better than a complete candidate on any objective.
    """
    matrix, _ = as_metric_matrix(values)
    directions = as_directions(higher_is_better, matrix.shape[1])
    reference = np.asarray(reference_point, dtype=np.float64)
    if reference.shape != (matrix.shape[1],):
        raise ValueError("reference_point must contain one value per metric")
    if not np.isfinite(reference).all():
        raise ValueError("reference_point must contain only finite values")

    complete_matrix = matrix[~np.isnan(matrix).any(axis=1)]
    if complete_matrix.size == 0:
        return 0.0

    orientation = np.where(directions, 1.0, -1.0)
    gains = complete_matrix * orientation - reference * orientation
    if np.any(gains < 0.0):
        raise ValueError("reference_point must be no better than every complete candidate")

    front_mask = pareto_front(complete_matrix, higher_is_better)
    return float(_union_box_volume(gains[front_mask]))


def _union_box_volume(boxes: NDArray[np.float64]) -> float:
    """Calculate the union of origin-anchored axis-aligned boxes."""
    if boxes.shape[1] == 1:
        return float(np.max(boxes[:, 0]))

    origin = np.asarray([0.0], dtype=np.float64)
    boundaries: NDArray[np.float64] = np.unique(np.concatenate((origin, boxes[:, 0])))
    volume = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        active = boxes[boxes[:, 0] >= upper, 1:]
        volume += (upper - lower) * _union_box_volume(active)
    return volume
