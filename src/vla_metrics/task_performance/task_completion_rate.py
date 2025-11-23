"""Task Completion Rate metric for VLA model evaluation.

Task Completion Rate (TCR) evaluates the ability to execute multi-step task sequences,
revealing critical limitations in current VLA models when handling complex natural
language instructions that require multiple sequential actions.

Reference:
    A. Brohan et al., "RT-1: Robotics transformer for real-world control at scale,"
    arXiv preprint arXiv:2212.06817, 2022.
"""

from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class TaskCompletionRate(Metric):
    """Compute Task Completion Rate for VLA task chain evaluation.

    Task Completion Rate is calculated as:
        TCR = N_completed_tasks / N_task_chains

    where N_completed_tasks is the number of successfully completed task chains and
    N_task_chains is the total number of task chains attempted.

    This metric evaluates multi-step task sequences, measuring success rates across
    sequential steps. Research shows that success rates drop significantly between
    sequential steps, indicating challenges in complex instruction following.

    Args:
        threshold: Threshold for binary classification when using continuous scores.
            If None, assumes binary inputs (0 or 1). Default: None.
        ignore_index: Value to ignore in the completion tensor. Default: None.
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from vla_metrics import TaskCompletionRate
        >>> metric = TaskCompletionRate()
        >>> # Binary completion indicators for task chains
        >>> completion = torch.tensor([1, 0, 1, 1, 0])
        >>> metric(completion)
        tensor(0.6000)

        >>> # With continuous scores and threshold
        >>> metric = TaskCompletionRate(threshold=0.8)
        >>> scores = torch.tensor([0.9, 0.7, 0.85, 0.95])
        >>> metric(scores)
        tensor(0.7500)

    Example (multi-step evaluation):
        >>> # Evaluate task chains over multiple batches
        >>> metric = TaskCompletionRate()
        >>> # First batch: 3 task chains, 2 completed
        >>> batch1 = torch.tensor([1, 0, 1])
        >>> metric.update(batch1)
        >>> # Second batch: 2 task chains, 1 completed
        >>> batch2 = torch.tensor([0, 1])
        >>> metric.update(batch2)
        >>> # Overall completion rate
        >>> metric.compute()
        tensor(0.6000)
    """

    # Metric states that persist across updates
    full_state_update: bool = False

    # Dynamically added by add_state() in __init__
    total_completed: Tensor
    total_chains: Tensor

    def __init__(
        self,
        threshold: Optional[float] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the TaskCompletionRate metric."""
        super().__init__(**kwargs)

        self.threshold = threshold
        self.ignore_index = ignore_index

        # Add metric states for distributed computation
        self.add_state("total_completed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_chains", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, completion: Tensor) -> None:  # pylint: disable=arguments-differ
        """Update metric state with new task chain completion indicators.

        Args:
            completion: Tensor of shape (N,) containing binary completion indicators
                (0 or 1) or continuous completion scores if threshold is set. Values
                can be int, float, or bool.

        Raises:
            ValueError: If completion tensor is empty or contains invalid values.
        """
        if completion.numel() == 0:
            raise ValueError("Input tensor is empty")

        # Handle ignore_index
        if self.ignore_index is not None:
            mask = completion != self.ignore_index
            completion = completion[mask]

            if completion.numel() == 0:
                return  # All values were ignored

        # Apply threshold if provided (for continuous scores)
        if self.threshold is not None:
            completion = (completion >= self.threshold).float()
        else:
            # Ensure binary values for non-thresholded input
            completion = completion.float()
            if not torch.all((completion == 0) | (completion == 1)):
                raise ValueError(
                    "Completion indicators must be binary (0 or 1) when threshold is not set. "
                    "Set threshold parameter for continuous scores."
                )

        # Update states
        self.total_completed += completion.sum()  # pylint: disable=no-member
        self.total_chains += completion.numel()  # pylint: disable=no-member

    def compute(self) -> Tensor:
        """Compute the final Task Completion Rate.

        Returns:
            Task completion rate as a scalar tensor in range [0, 1].

        Raises:
            RuntimeError: If no task chains have been recorded (total_chains == 0).
        """
        if self.total_chains == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute task completion rate: no task chains have been recorded. "
                "Call update() with completion indicators before compute()."
            )

        return self.total_completed.float() / self.total_chains  # pylint: disable=no-member
