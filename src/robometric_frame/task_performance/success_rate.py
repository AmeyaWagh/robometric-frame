"""Success Rate metric for robotics policy evaluation.

Success Rate (SR) is a fundamental metric measuring the percentage of successfully
completed tasks in robotics policy evaluation.

Reference:
    A. Brohan et al., "RT-1: Robotics transformer for real-world control at scale,"
    arXiv preprint arXiv:2212.06817, 2022.
"""

from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class SuccessRate(Metric):
    """Compute Success Rate for robotics policy task evaluation.

    Success Rate is calculated as:
        SR = N_success / N_total

    where N_success is the number of successfully completed tasks and N_total is
    the total number of tasks attempted.

    This metric supports both binary success indicators and continuous success scores
    with an optional threshold.

    Args:
        threshold: Threshold for binary classification when using continuous scores.
            If None, assumes binary inputs (0 or 1). Default: None.
        ignore_index: Value to ignore in the success tensor. Default: None.
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from robometric_frame import SuccessRate
        >>> metric = SuccessRate()
        >>> # Binary success indicators
        >>> success = torch.tensor([1, 1, 0, 1, 0, 0, 1])
        >>> metric(success)
        tensor(0.5714)

        >>> # With continuous scores and threshold
        >>> metric = SuccessRate(threshold=0.8)
        >>> scores = torch.tensor([0.9, 0.7, 0.85, 0.6, 0.95])
        >>> metric(scores)
        tensor(0.6000)

    Example (distributed):
        >>> # In distributed training, metrics are automatically synced
        >>> metric = SuccessRate()
        >>> # On GPU 0
        >>> success_gpu0 = torch.tensor([1, 1, 0])
        >>> metric(success_gpu0)
        >>> # On GPU 1
        >>> success_gpu1 = torch.tensor([1, 0, 1])
        >>> metric(success_gpu1)
        >>> # Final result aggregates across all GPUs
        >>> result = metric.compute()  # Returns aggregated success rate
    """

    # Metric states that persist across updates
    full_state_update: bool = False

    # Dynamically added by add_state() in __init__
    total_success: Tensor
    total_tasks: Tensor

    def __init__(
        self,
        threshold: Optional[float] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SuccessRate metric."""
        super().__init__(**kwargs)

        self.threshold = threshold
        self.ignore_index = ignore_index

        # Add metric states for distributed computation
        self.add_state("total_success", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_tasks", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, success: Tensor) -> None:  # pylint: disable=arguments-differ
        """Update metric state with new success indicators.

        Args:
            success: Tensor of shape (N,) containing binary success indicators (0 or 1)
                or continuous success scores if threshold is set. Values can be int,
                float, or bool.

        Raises:
            ValueError: If success tensor is empty or contains invalid values.
        """
        if success.numel() == 0:
            raise ValueError("Input tensor is empty")

        # Handle ignore_index
        if self.ignore_index is not None:
            mask = success != self.ignore_index
            success = success[mask]

            if success.numel() == 0:
                return  # All values were ignored

        # Apply threshold if provided (for continuous scores)
        if self.threshold is not None:
            success = (success >= self.threshold).float()
        else:
            # Ensure binary values for non-thresholded input
            success = success.float()
            if not torch.all((success == 0) | (success == 1)):
                raise ValueError(
                    "Success indicators must be binary (0 or 1) when threshold is not set. "
                    "Set threshold parameter for continuous scores."
                )

        # Update states
        self.total_success += success.sum()  # pylint: disable=no-member
        self.total_tasks += success.numel()  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the final Success Rate.

        Returns:
            Success rate as a scalar tensor in range [0, 1].

        Raises:
            RuntimeError: If no tasks have been recorded (total_tasks == 0).
        """
        if self.total_tasks == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute success rate: no tasks have been recorded. "
                "Call update() with success indicators before compute()."
            )

        return self.total_success.float() / self.total_tasks  # pylint: disable=no-member


class TaskSuccessRate(SuccessRate):
    """Alias for SuccessRate metric.

    This class provides a more descriptive name for the same functionality.
    """
