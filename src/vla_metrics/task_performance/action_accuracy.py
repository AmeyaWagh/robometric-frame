"""Action Accuracy metrics for VLA model evaluation.

Action Accuracy measures the precision of predicted actions against ground truth
trajectories using Mean Squared Error (MSE) and its variations. This provides
direct assessment of model performance in offline evaluation scenarios.

References:
    [1] M. Dobiš et al., "Evaluation criteria for trajectories of robotic arms,"
        Robotics, vol. 11, p. 29, 2022.
    [2] K. K. A. Farag et al., "Mobile robot obstacle avoidance based on neural
        network with a standardization technique," J. Robot., vol. 2021, 2021.
"""

from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class ActionAccuracy(Metric):
    r"""Compute Action Accuracy metrics (MSE, AMSE, NAMSE) for VLA evaluation.

    This metric computes three related measures of action prediction accuracy:
    - MSE: Mean Squared Error per trajectory
    - AMSE: Average MSE across multiple trajectories
    - NAMSE: Normalized AMSE (scaled by action variance)

    Formulas:
        MSE = (1/T) * sum_{t=1}^{T} \|a_t - â_t\|_2^2
        AMSE = (1/K) * sum_{k=1}^{K} MSE_k
        NAMSE = AMSE / σ²_action

    where:
        - a_t is the ground truth action at timestep t
        - â_t is the predicted action at timestep t
        - T is the number of timesteps in a trajectory
        - K is the number of trajectories
        - σ²_action is the variance of ground truth actions

    Args:
        normalize: Whether to compute NAMSE. If True, action variance is computed
            from the data. If False, only MSE and AMSE are computed. Default: False.
        action_variance: Pre-computed action variance for normalization. If provided,
            this value is used instead of computing from data. Default: None.
        **kwargs: Additional keyword arguments passed to the base Metric class.

    Example:
        >>> from vla_metrics import ActionAccuracy
        >>> import torch
        >>> metric = ActionAccuracy()
        >>>
        >>> # Single trajectory
        >>> predictions = torch.randn(10, 4)  # 10 timesteps, 4-dim actions
        >>> targets = torch.randn(10, 4)
        >>> metric.update(predictions, targets)
        >>> results = metric.compute()
        >>> print(f"MSE: {results['mse']:.4f}, AMSE: {results['amse']:.4f}")
        >>>
        >>> # With normalization
        >>> metric = ActionAccuracy(normalize=True)
        >>> metric.update(predictions, targets)
        >>> results = metric.compute()
        >>> print(f"NAMSE: {results['namse']:.4f}")

    Example (multiple trajectories):
        >>> metric = ActionAccuracy()
        >>> # Trajectory 1
        >>> metric.update(torch.randn(10, 4), torch.randn(10, 4))
        >>> # Trajectory 2
        >>> metric.update(torch.randn(15, 4), torch.randn(15, 4))
        >>> results = metric.compute()
        >>> # AMSE is averaged across both trajectories
    """

    full_state_update: bool = False

    # Dynamically added by add_state() in __init__
    total_mse: Tensor
    total_trajectories: Tensor
    total_squared_actions: Tensor
    total_actions: Tensor
    total_action_count: Tensor

    def __init__(
        self,
        normalize: bool = False,
        action_variance: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ActionAccuracy metric."""
        super().__init__(**kwargs)

        self.normalize = normalize
        self.action_variance = action_variance

        # States for MSE and AMSE computation
        self.add_state("total_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_trajectories", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # States for action variance computation (needed for NAMSE)
        if normalize and action_variance is None:
            self.add_state("total_squared_actions", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_actions", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total_action_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions: Tensor, targets: Tensor) -> None:  # pylint: disable=arguments-differ
        """Update metric state with predicted and target actions.

        Args:
            predictions: Predicted actions of shape (T, D) where T is the number of
                timesteps and D is the action dimension.
            targets: Ground truth actions of shape (T, D).

        Raises:
            ValueError: If predictions and targets have different shapes or are empty.
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        if predictions.numel() == 0:
            raise ValueError("Input tensors are empty")

        # Compute MSE for this trajectory: mean of squared L2 norms
        squared_errors = torch.sum((predictions - targets) ** 2, dim=-1)  # (T,)
        mse = squared_errors.mean()

        # Update MSE accumulator
        self.total_mse += mse  # pylint: disable=no-member
        self.total_trajectories += 1.0  # pylint: disable=no-member

        # Update action statistics for variance computation (if needed for NAMSE)
        if self.normalize and self.action_variance is None:
            # Flatten actions to compute overall statistics
            targets_flat = targets.reshape(-1)
            self.total_squared_actions += (targets_flat**2).sum()  # pylint: disable=no-member
            self.total_actions += targets_flat.sum()  # pylint: disable=no-member
            self.total_action_count += targets_flat.numel()  # pylint: disable=no-member

    def compute(self) -> dict[str, Tensor]:
        """Compute the final Action Accuracy metrics.

        Returns:
            Dictionary containing:
                - 'mse': Mean Squared Error of the last trajectory
                - 'amse': Average MSE across all trajectories
                - 'namse': Normalized AMSE (only if normalize=True)

        Raises:
            RuntimeError: If no trajectories have been recorded.
        """
        if self.total_trajectories == 0:  # pylint: disable=no-member
            raise RuntimeError(
                "Cannot compute action accuracy: no trajectories have been recorded. "
                "Call update() with predictions and targets before compute()."
            )

        # Compute AMSE (average of MSEs across trajectories)
        amse = self.total_mse / self.total_trajectories  # pylint: disable=no-member

        results = {
            "mse": self.total_mse / self.total_trajectories,  # pylint: disable=no-member
            "amse": amse,
        }

        # Compute NAMSE if normalization is enabled
        if self.normalize:
            if self.action_variance is not None:
                # Use provided variance
                variance = torch.tensor(self.action_variance, dtype=amse.dtype, device=amse.device)
            else:
                # Compute variance from accumulated statistics
                # Var(X) = E[X²] - E[X]²
                mean_squared = (
                    self.total_squared_actions / self.total_action_count  # pylint: disable=no-member
                )
                mean = self.total_actions / self.total_action_count  # pylint: disable=no-member
                variance = mean_squared - mean**2

            if variance <= 0:
                raise RuntimeError(
                    "Action variance is zero or negative. Cannot compute NAMSE. "
                    "Ensure target actions have non-zero variance."
                )

            results["namse"] = amse / variance

        return results
