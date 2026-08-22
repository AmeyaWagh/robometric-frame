"""Metric normalization utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robometric_frame.visualization._validation import as_directions, as_metric_matrix


def normalize_metrics(
    values: ArrayLike,
    higher_is_better: Sequence[bool],
    *,
    bounds: ArrayLike | None = None,
    clip: bool = False,
) -> NDArray[np.float64]:
    """Normalize heterogeneous metrics to a common higher-is-better scale.

    Each output value is on a nominal zero-to-one scale, where one is best.
    Missing values (``NaN``) remain missing. When ``bounds`` is omitted, the
    finite minimum and maximum of each column are used. Constant columns raise
    an error because they cannot be meaningfully normalized.

    Args:
        values: One metric row with shape ``(M,)`` or multiple rows with shape
            ``(N, M)``.
        higher_is_better: One boolean per metric describing its direction.
        bounds: Optional raw-value lower and upper bounds with shape ``(M, 2)``.
            Supplying bounds makes comparisons stable across separate runs.
        clip: Whether to clip values outside explicit bounds to zero and one.

    Returns:
        A float array with the same shape as ``values``.

    Raises:
        ValueError: If inputs have incompatible shapes, invalid bounds, or a
            finite metric column has no range.
        TypeError: If metric directions are not booleans.
    """
    matrix, was_one_dimensional = as_metric_matrix(values)
    directions = as_directions(higher_is_better, matrix.shape[1])

    if bounds is None:
        lower = np.full(matrix.shape[1], np.nan, dtype=np.float64)
        upper = np.full(matrix.shape[1], np.nan, dtype=np.float64)
        for index in range(matrix.shape[1]):
            finite_values = matrix[np.isfinite(matrix[:, index]), index]
            if finite_values.size == 0:
                continue
            lower[index] = finite_values.min()
            upper[index] = finite_values.max()
            if lower[index] == upper[index]:
                raise ValueError(
                    f"metric column {index} is constant; provide explicit bounds to normalize it"
                )
    else:
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if bounds_array.shape != (matrix.shape[1], 2):
            raise ValueError("bounds must have shape (number_of_metrics, 2)")
        if not np.isfinite(bounds_array).all():
            raise ValueError("bounds must contain only finite values")
        lower = bounds_array[:, 0]
        upper = bounds_array[:, 1]
        if np.any(lower >= upper):
            raise ValueError("each lower bound must be less than its upper bound")

    scale = upper - lower
    normalized = (matrix - lower) / scale
    normalized[:, ~directions] = 1.0 - normalized[:, ~directions]
    if clip:
        normalized = np.asarray(np.clip(normalized, 0.0, 1.0), dtype=np.float64)

    if was_one_dimensional:
        return np.asarray(normalized[0], dtype=np.float64)
    return normalized
