"""Dynamic Time Warping (DTW) metrics for robotics policy trajectory evaluation.

DTW-based metrics measure trajectory similarity while allowing for temporal misalignment.
Unlike MSE which requires point-to-point correspondence, DTW finds the optimal warping
between sequences of different lengths or timing.

Reference:
    G. Ilharco, V. Jain, A. Ku, E. Ie, and J. Baldridge, "General Evaluation for
    Instruction Conditioned Navigation using Dynamic Time Warping," arXiv:1907.05446,
    NeurIPS ViGIL Workshop, 2019.
"""

from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class DTWBase(Metric):
    """Base class for DTW-based trajectory metrics.

    Provides shared validation, DTW computation, path length computation,
    and normalization factor logic used by DTWDistance, NormalizedDTW, and
    SuccessWeightedDTW.

    Subclasses must define their own metric states and implement update()/compute().
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    # Dynamically added by add_state() in __init__
    num_trajectory_pairs: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("num_trajectory_pairs", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _validate_trajectories(predicted: Tensor, reference: Tensor) -> None:
        """Validate predicted and reference trajectory shapes and compatibility.

        Args:
            predicted: Predicted trajectory tensor.
            reference: Reference trajectory tensor.

        Raises:
            ValueError: If trajectories have invalid shape, mismatched dimensionality,
                or insufficient points.
        """
        if predicted.ndim != 2:
            raise ValueError(
                f"Predicted trajectory must have 2 dimensions (T, D), "
                f"got {predicted.ndim}D tensor with shape {predicted.shape}"
            )

        if reference.ndim != 2:
            raise ValueError(
                f"Reference trajectory must have 2 dimensions (T, D), "
                f"got {reference.ndim}D tensor with shape {reference.shape}"
            )

        if predicted.shape[-1] != reference.shape[-1]:
            raise ValueError(
                f"Predicted and reference trajectories must have the same dimensionality D, "
                f"got predicted D={predicted.shape[-1]}, reference D={reference.shape[-1]}"
            )

        if predicted.shape[0] < 1:
            raise ValueError(
                f"Predicted trajectory must have at least 1 point, "
                f"got {predicted.shape[0]} point(s)"
            )

        if reference.shape[0] < 1:
            raise ValueError(
                f"Reference trajectory must have at least 1 point, "
                f"got {reference.shape[0]} point(s)"
            )

    @staticmethod
    def _compute_dtw(predicted: Tensor, reference: Tensor) -> Tensor:
        """Compute DTW distance between two trajectories using dynamic programming.

        Args:
            predicted: Predicted trajectory tensor of shape (T_pred, D).
            reference: Reference trajectory tensor of shape (T_ref, D).

        Returns:
            DTW distance as a scalar tensor.
        """
        cost_matrix = torch.cdist(predicted.unsqueeze(0), reference.unsqueeze(0), p=2.0).squeeze(0)

        t_pred, t_ref = cost_matrix.shape

        accumulated = torch.zeros_like(cost_matrix)
        accumulated[:, 0] = torch.cumsum(cost_matrix[:, 0], dim=0)
        accumulated[0, :] = torch.cumsum(cost_matrix[0, :], dim=0)

        for k in range(2, t_pred + t_ref - 1):
            i_start = max(1, k - t_ref + 1)
            i_end = min(t_pred - 1, k - 1)
            i_idx = torch.arange(i_start, i_end + 1, device=cost_matrix.device)
            j_idx = k - i_idx

            accumulated[i_idx, j_idx] = cost_matrix[i_idx, j_idx] + torch.minimum(
                torch.minimum(accumulated[i_idx - 1, j_idx], accumulated[i_idx, j_idx - 1]),
                accumulated[i_idx - 1, j_idx - 1],
            )

        return accumulated[t_pred - 1, t_ref - 1]

    @staticmethod
    def _compute_path_length(trajectory: Tensor) -> Tensor:
        """Compute the total path length of a trajectory.

        Args:
            trajectory: Trajectory tensor of shape (L, D).

        Returns:
            Path length as a scalar tensor.
        """
        if trajectory.shape[0] < 2:
            return torch.tensor(0.0, device=trajectory.device, dtype=trajectory.dtype)

        deltas = trajectory[1:, :] - trajectory[:-1, :]
        distances = torch.norm(deltas, p=2, dim=-1)
        return distances.sum()

    @staticmethod
    def _compute_normalization_factor(
        predicted: Tensor,
        reference: Tensor,
        normalization_factor: Optional[float],
    ) -> float:
        """Compute the normalization constant d for nDTW/SDTW.

        Args:
            predicted: Predicted trajectory tensor of shape (T_pred, D).
            reference: Reference trajectory tensor of shape (T_ref, D).
            normalization_factor: User-specified value, or None for auto-computation.

        Returns:
            Normalization constant d.
        """
        if normalization_factor is not None:
            return normalization_factor

        ref_length = reference.shape[0]
        if ref_length == 1:
            d = float(torch.norm(predicted[0] - reference[0], p=2).item())
            return d if d != 0 else 1.0

        path_length = DTWBase._compute_path_length(reference)
        d = float((path_length / (ref_length - 1)).item())
        return d if d != 0 else 1.0


class DTWDistance(DTWBase):
    r"""Compute Dynamic Time Warping (DTW) distance for trajectory evaluation.

    DTW distance measures the minimum-cost temporal alignment between predicted
    and reference trajectories. Unlike MSE which compares trajectories timestep-by-
    timestep, DTW finds the optimal warping to align sequences that may differ in
    length or timing.

    DTW is calculated by building an accumulated cost matrix D where:
        D[0,0] = C[0,0]
        D[i,0] = D[i-1,0] + C[i,0] for i > 0
        D[0,j] = D[0,j-1] + C[0,j] for j > 0
        D[i,j] = C[i,j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1]) for i,j > 0

    where C[i,j] is the Euclidean distance between predicted[i] and reference[j].
    The final DTW distance is D[n-1, m-1].

    This metric is particularly useful for evaluating VLA models and policies using
    action chunking (e.g., ACT, Diffusion Policy) where predicted trajectories may
    be temporally misaligned with demonstrations.

    This metric accumulates DTW distances across multiple trajectory pairs and returns
    the average DTW distance when compute() is called.

    Args:
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Attributes:
        higher_is_better: False - lower DTW distance indicates better similarity.
        is_differentiable: False - DTW computation is not differentiable.
        full_state_update: False - incremental state updates.

    Note:
        Memory complexity is O(T_pred * T_ref) for the cost matrices.
        For very long trajectories, this may require significant memory.

    Example:
        >>> from robometric_frame.trajectory_quality import DTWDistance
        >>> import torch
        >>> metric = DTWDistance()
        >>> # Identical trajectories (zero distance)
        >>> predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(predicted, reference)
        >>> metric.compute()
        tensor(0.0000)

    Example (different lengths):
        >>> # Trajectories of different lengths (the core use case)
        >>> metric = DTWDistance()
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        >>> predicted = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0],
        ...                           [2.0, 0.0], [2.5, 0.0], [3.0, 0.0]])
        >>> metric.update(predicted, reference)
        >>> result = metric.compute()  # Small value (same path, different density)

    Example (temporal shift):
        >>> # Hesitation at start (same actions, different timing)
        >>> metric = DTWDistance()
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> predicted = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        ...                           [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(predicted, reference)
        >>> result = metric.compute()  # Small value (DTW tolerates hesitation)
    """

    higher_is_better: bool = False

    # Dynamically added by add_state() in __init__
    total_dtw_distance: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the DTWDistance metric."""
        super().__init__(**kwargs)
        self.add_state("total_dtw_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(  # pylint: disable=arguments-differ
        self, predicted: Tensor, reference: Tensor
    ) -> None:
        """Update metric state with new predicted and reference trajectory pair.

        Args:
            predicted: Predicted trajectory tensor of shape (T_pred, D) where:
                - T_pred is the number of timesteps (can differ from T_ref)
                - D is the spatial dimensionality (e.g., 2 for 2D, 3 for 3D, 7 for 7-DoF)

                Points should be ordered chronologically.

            reference: Reference (ground truth) trajectory tensor of shape (T_ref, D).
                T_ref can differ from T_pred - this is the core advantage of DTW.
                D must match predicted trajectory dimensionality.

        Raises:
            ValueError: If trajectories have invalid shape (< 2 dimensions),
                mismatched dimensionality, or insufficient points.
        """
        self._validate_trajectories(predicted, reference)
        predicted = predicted.float()
        reference = reference.float()

        dtw_distance = self._compute_dtw(predicted, reference)

        self.total_dtw_distance += dtw_distance  # pylint: disable=no-member
        self.num_trajectory_pairs += 1  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the average DTW distance across all trajectory pairs.

        Returns:
            Average DTW distance as a scalar tensor. Lower values indicate
            better trajectory similarity.

        Raises:
            RuntimeError: If no trajectory pairs have been recorded.
        """
        if self.num_trajectory_pairs == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute DTW distance: no trajectory pairs have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        return self.total_dtw_distance / self.num_trajectory_pairs  # pylint: disable=no-member


class NormalizedDTW(DTWBase):
    r"""Compute Normalized DTW (nDTW) for trajectory evaluation.

    nDTW normalizes the raw DTW distance and maps it to a [0, 1] score:
        nDTW = exp(-DTW / (|R| * d))

    where:
        - DTW is the raw DTW distance
        - |R| is the length of the reference trajectory (number of points)
        - d is a normalization constant (average step distance of the reference,
          or a user-specified value)

    Higher nDTW scores indicate better trajectory similarity (1.0 = perfect match,
    approaches 0.0 for very dissimilar trajectories).

    This metric is particularly useful for evaluating VLA models and policies using
    action chunking where predicted trajectories may be temporally misaligned.

    Args:
        normalization_factor: Optional user-specified normalization constant d.
            If None (default), automatically computed as the mean step distance
            of the reference trajectory: PathLength(reference) / (len(reference) - 1).
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Attributes:
        higher_is_better: True - higher nDTW indicates better similarity.
        is_differentiable: False - DTW computation is not differentiable.
        full_state_update: False - incremental state updates.

    Note:
        Memory complexity is O(T_pred * T_ref) for the cost matrices.
        For very long trajectories, this may require significant memory.

    Example:
        >>> from robometric_frame.trajectory_quality import NormalizedDTW
        >>> import torch
        >>> metric = NormalizedDTW()
        >>> # Identical trajectories (perfect score)
        >>> predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(predicted, reference)
        >>> metric.compute()
        tensor(1.0000)

    Example (different lengths):
        >>> # Trajectories of different lengths
        >>> metric = NormalizedDTW()
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        >>> predicted = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0],
        ...                           [2.0, 0.0], [2.5, 0.0], [3.0, 0.0]])
        >>> metric.update(predicted, reference)
        >>> result = metric.compute()  # High value (same path)

    Example (custom normalization):
        >>> # Use custom normalization factor
        >>> metric = NormalizedDTW(normalization_factor=0.5)
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> predicted = torch.tensor([[0.0, 0.1], [1.0, 0.1], [2.0, 0.1]])
        >>> metric.update(predicted, reference)
        >>> result = metric.compute()
    """

    higher_is_better: bool = True

    # Dynamically added by add_state() in __init__
    total_ndtw: Tensor

    def __init__(
        self,
        normalization_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the NormalizedDTW metric.

        Args:
            normalization_factor: Optional user-specified normalization constant d.
                If None, automatically computed as the mean step distance of the
                reference trajectory.
            **kwargs: Additional keyword arguments passed to the base Metric class.
        """
        super().__init__(**kwargs)

        if normalization_factor is not None and normalization_factor <= 0:
            raise ValueError(f"normalization_factor must be positive, got {normalization_factor}")

        self.normalization_factor = normalization_factor

        self.add_state("total_ndtw", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(  # pylint: disable=arguments-differ
        self, predicted: Tensor, reference: Tensor
    ) -> None:
        """Update metric state with new predicted and reference trajectory pair.

        Args:
            predicted: Predicted trajectory tensor of shape (T_pred, D) where:
                - T_pred is the number of timesteps (can differ from T_ref)
                - D is the spatial dimensionality

                Points should be ordered chronologically.

            reference: Reference (ground truth) trajectory tensor of shape (T_ref, D).
                T_ref can differ from T_pred. D must match predicted dimensionality.

        Raises:
            ValueError: If trajectories have invalid shape, mismatched dimensionality,
                or insufficient points.
        """
        self._validate_trajectories(predicted, reference)
        predicted = predicted.float()
        reference = reference.float()

        dtw_distance = self._compute_dtw(predicted, reference)
        ref_length = reference.shape[0]
        d = self._compute_normalization_factor(predicted, reference, self.normalization_factor)

        ndtw = torch.exp(-dtw_distance / (ref_length * d))

        self.total_ndtw += ndtw  # pylint: disable=no-member
        self.num_trajectory_pairs += 1  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the average nDTW score across all trajectory pairs.

        Returns:
            Average nDTW score as a scalar tensor in range [0, 1].
            Higher values indicate better trajectory similarity.

        Raises:
            RuntimeError: If no trajectory pairs have been recorded.
        """
        if self.num_trajectory_pairs == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute nDTW: no trajectory pairs have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        return self.total_ndtw / self.num_trajectory_pairs  # pylint: disable=no-member


class SuccessWeightedDTW(DTWBase):
    r"""Compute Success-weighted DTW (SDTW) for trajectory evaluation.

    SDTW combines trajectory fidelity with task success:
        SDTW = nDTW * Success

    where:
        - nDTW is the normalized DTW score (see NormalizedDTW)
        - Success is a binary indicator (1 if task succeeded, 0 if not)

    If the task failed, SDTW = 0 regardless of trajectory similarity. This captures
    both "did you succeed?" and "did you follow the right path?"

    This metric is particularly useful for benchmarking policies where both task
    completion and trajectory quality matter.

    Args:
        normalization_factor: Optional user-specified normalization constant d.
            If None (default), automatically computed as the mean step distance
            of the reference trajectory.
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Attributes:
        higher_is_better: True - higher SDTW indicates better performance.
        is_differentiable: False - DTW computation is not differentiable.
        full_state_update: False - incremental state updates.

    Note:
        Memory complexity is O(T_pred * T_ref) for the cost matrices.
        For very long trajectories, this may require significant memory.

    Example:
        >>> from robometric_frame.trajectory_quality import SuccessWeightedDTW
        >>> import torch
        >>> metric = SuccessWeightedDTW()
        >>> # Successful task with good trajectory
        >>> predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(predicted, reference, success=torch.tensor(True))
        >>> metric.compute()
        tensor(1.0000)

    Example (failed task):
        >>> # Failed task (SDTW = 0 regardless of trajectory quality)
        >>> metric = SuccessWeightedDTW()
        >>> predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(predicted, reference, success=torch.tensor(False))
        >>> metric.compute()
        tensor(0.0000)

    Example (multiple updates):
        >>> # Mix of successes and failures
        >>> metric = SuccessWeightedDTW()
        >>> ref = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> pred = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        >>> metric.update(pred, ref, success=torch.tensor(True))   # SDTW = 1.0
        >>> metric.update(pred, ref, success=torch.tensor(False))  # SDTW = 0.0
        >>> metric.compute()  # Average: 0.5
        tensor(0.5000)
    """

    higher_is_better: bool = True

    # Dynamically added by add_state() in __init__
    total_sdtw: Tensor

    def __init__(
        self,
        normalization_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SuccessWeightedDTW metric.

        Args:
            normalization_factor: Optional user-specified normalization constant d.
                If None, automatically computed as the mean step distance of the
                reference trajectory.
            **kwargs: Additional keyword arguments passed to the base Metric class.
        """
        super().__init__(**kwargs)

        if normalization_factor is not None and normalization_factor <= 0:
            raise ValueError(f"normalization_factor must be positive, got {normalization_factor}")

        self.normalization_factor = normalization_factor

        self.add_state("total_sdtw", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(  # pylint: disable=arguments-differ
        self, predicted: Tensor, reference: Tensor, success: Tensor
    ) -> None:
        """Update metric state with new trajectory pair and success indicator.

        Args:
            predicted: Predicted trajectory tensor of shape (T_pred, D) where:
                - T_pred is the number of timesteps (can differ from T_ref)
                - D is the spatial dimensionality

                Points should be ordered chronologically.

            reference: Reference (ground truth) trajectory tensor of shape (T_ref, D).
                T_ref can differ from T_pred. D must match predicted dimensionality.

            success: Boolean or 0/1 tensor indicating task success.
                If False/0, SDTW will be 0 regardless of trajectory similarity.

        Raises:
            ValueError: If trajectories have invalid shape, mismatched dimensionality,
                or insufficient points.
        """
        self._validate_trajectories(predicted, reference)
        predicted = predicted.float()
        reference = reference.float()

        success_value = success.float() if isinstance(success, Tensor) else float(success)
        if isinstance(success_value, Tensor):
            success_value = success_value.item()

        if success_value == 0:
            sdtw = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
        else:
            dtw_distance = self._compute_dtw(predicted, reference)
            ref_length = reference.shape[0]
            d = self._compute_normalization_factor(predicted, reference, self.normalization_factor)

            ndtw = torch.exp(-dtw_distance / (ref_length * d))
            sdtw = ndtw * success_value

        self.total_sdtw += sdtw  # pylint: disable=no-member
        self.num_trajectory_pairs += 1  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the average SDTW score across all trajectory pairs.

        Returns:
            Average SDTW score as a scalar tensor in range [0, 1].
            Higher values indicate better overall performance (trajectory
            quality weighted by task success).

        Raises:
            RuntimeError: If no trajectory pairs have been recorded.
        """
        if self.num_trajectory_pairs == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute SDTW: no trajectory pairs have been recorded. "
                "Call update() with trajectory data before compute()."
            )

        return self.total_sdtw / self.num_trajectory_pairs  # pylint: disable=no-member
