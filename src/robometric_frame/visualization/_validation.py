"""Shared input validation for visualization utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_metric_matrix(values: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    """Convert one or more metric rows to a two-dimensional float array."""
    matrix = np.asarray(values, dtype=np.float64)
    was_one_dimensional = matrix.ndim == 1
    if was_one_dimensional:
        matrix = matrix[np.newaxis, :]
    elif matrix.ndim != 2:
        raise ValueError("values must be a one- or two-dimensional array")

    if matrix.shape[1] == 0:
        raise ValueError("values must contain at least one metric")
    if np.isinf(matrix).any():
        raise ValueError("values cannot contain infinite values")
    return matrix, was_one_dimensional


def as_directions(higher_is_better: Sequence[bool], number_of_metrics: int) -> NDArray[np.bool_]:
    """Validate and convert metric direction flags."""
    directions = np.asarray(higher_is_better)
    if directions.ndim != 1 or len(directions) != number_of_metrics:
        raise ValueError("higher_is_better must contain one flag per metric")
    if directions.dtype.kind != "b":
        raise TypeError("higher_is_better must contain boolean values")
    return directions.astype(np.bool_)
