"""Example usage of ActionAccuracy metric for VLA evaluation.

This example demonstrates how to use the ActionAccuracy metric to evaluate
action prediction accuracy using MSE, AMSE, and NAMSE in Vision-Language-Action models.
"""
# pylint: skip-file

import torch

from vla_metrics import ActionAccuracy


def main() -> None:
    """Run ActionAccuracy examples."""
    print("=" * 80)
    print("ActionAccuracy Metric Examples")
    print("=" * 80)

    # Example 1: Basic MSE computation for a single trajectory
    print("\n1. Basic MSE Computation (Single Trajectory)")
    print("-" * 80)
    metric = ActionAccuracy()

    # Simulate a trajectory with 10 timesteps and 4-dimensional actions
    predictions = torch.tensor(
        [
            [1.0, 2.0, 0.5, 1.5],
            [1.2, 2.1, 0.6, 1.4],
            [0.9, 1.9, 0.4, 1.6],
        ]
    )
    targets = torch.tensor(
        [
            [1.1, 2.0, 0.5, 1.5],
            [1.0, 2.0, 0.5, 1.5],
            [1.0, 2.0, 0.5, 1.5],
        ]
    )

    metric.update(predictions, targets)
    results = metric.compute()

    print(f"Trajectory shape: {predictions.shape} (timesteps, action_dim)")
    print(f"MSE: {results['mse']:.4f}")
    print(f"AMSE: {results['amse']:.4f}")

    # Example 2: Multiple trajectories with AMSE
    print("\n2. Multiple Trajectories (AMSE Computation)")
    print("-" * 80)
    metric = ActionAccuracy()

    # Simulate 3 trajectories with different lengths
    trajectories = [
        (torch.randn(5, 4), torch.randn(5, 4)),  # Trajectory 1: 5 timesteps
        (torch.randn(8, 4), torch.randn(8, 4)),  # Trajectory 2: 8 timesteps
        (torch.randn(10, 4), torch.randn(10, 4)),  # Trajectory 3: 10 timesteps
    ]

    trajectory_mses = []
    for i, (pred, target) in enumerate(trajectories, 1):
        # Calculate individual trajectory MSE
        traj_metric = ActionAccuracy()
        traj_metric.update(pred, target)
        traj_results = traj_metric.compute()
        trajectory_mses.append(traj_results["mse"].item())

        # Add to overall metric
        metric.update(pred, target)
        print(f"Trajectory {i}: {pred.shape[0]} timesteps, MSE = {traj_results['mse']:.4f}")

    results = metric.compute()
    print(f"\nAverage MSE across trajectories (AMSE): {results['amse']:.4f}")
    print(f"Manual calculation: {sum(trajectory_mses) / len(trajectory_mses):.4f}")

    # Example 3: Normalized AMSE with provided variance
    print("\n3. Normalized AMSE (NAMSE) with Provided Variance")
    print("-" * 80)
    action_variance = 2.0
    metric = ActionAccuracy(normalize=True, action_variance=action_variance)

    # Create trajectories with known variance relationship
    predictions = torch.randn(20, 4)
    targets = torch.randn(20, 4)

    metric.update(predictions, targets)
    results = metric.compute()

    print(f"Action variance (provided): {action_variance}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"AMSE: {results['amse']:.4f}")
    print(f"NAMSE: {results['namse']:.4f}")
    print(f"Verification: NAMSE = AMSE / variance = {results['amse'].item() / action_variance:.4f}")

    # Example 4: Normalized AMSE with computed variance
    print("\n4. Normalized AMSE (NAMSE) with Computed Variance")
    print("-" * 80)
    metric = ActionAccuracy(normalize=True)  # No variance provided

    # Generate targets with known properties
    targets = torch.randn(50, 4)
    predictions = targets + torch.randn(50, 4) * 0.5  # Add some noise

    metric.update(predictions, targets)
    results = metric.compute()

    # Compute variance manually for verification
    computed_variance = targets.var(unbiased=False).item()

    print("Computing variance from target actions...")
    print(f"Computed variance: {computed_variance:.4f}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"AMSE: {results['amse']:.4f}")
    print(f"NAMSE: {results['namse']:.4f}")

    # Example 5: Realistic VLA evaluation scenario
    print("\n5. Realistic VLA Evaluation Across Different Tasks")
    print("-" * 80)

    # Define different robotic tasks with varying action dimensions
    task_scenarios = {
        "Reaching (3-DoF)": (3, 15),  # (action_dim, num_trajectories)
        "Grasping (4-DoF)": (4, 12),
        "Manipulation (7-DoF)": (7, 10),
    }

    overall_metric = ActionAccuracy()

    print("\nTask-wise action accuracy:")
    for task_name, (action_dim, num_trajs) in task_scenarios.items():
        task_metric = ActionAccuracy()

        # Simulate multiple trajectories for this task
        for _ in range(num_trajs):
            traj_len = int(torch.randint(10, 30, (1,)).item())  # Random length 10-30
            pred = torch.randn(traj_len, action_dim)
            target = pred + torch.randn(traj_len, action_dim) * 0.3  # Add noise

            task_metric.update(pred, target)
            overall_metric.update(pred, target)

        task_results = task_metric.compute()
        print(f"  {task_name:25s}: AMSE = {task_results['amse']:.4f}")

    overall_results = overall_metric.compute()
    print(f"\nOverall AMSE across all tasks: {overall_results['amse']:.4f}")

    # Example 6: Per-epoch training progress
    print("\n6. Training Progress: Per-Epoch Action Accuracy")
    print("-" * 80)

    num_epochs = 5
    print(f"Simulating {num_epochs} training epochs with improving accuracy...\n")

    for epoch in range(1, num_epochs + 1):
        epoch_metric = ActionAccuracy()

        # Simulate improving prediction accuracy over epochs
        num_trajectories = 10
        noise_scale = max(1.0 - epoch * 0.15, 0.1)  # Decreasing noise

        for _ in range(num_trajectories):
            traj_len = 15
            target = torch.randn(traj_len, 4)
            pred = target + torch.randn(traj_len, 4) * noise_scale

            epoch_metric.update(pred, target)

        epoch_results = epoch_metric.compute()
        print(f"Epoch {epoch}: AMSE = {epoch_results['amse']:.4f} (noise scale: {noise_scale:.2f})")

    # Example 7: Comparison of different action dimensions
    print("\n7. Effect of Action Dimensionality on MSE")
    print("-" * 80)

    action_dims = [2, 4, 7, 10]
    print("\nAction dimensionality comparison:")

    for dim in action_dims:
        metric = ActionAccuracy()

        # Fixed number of trajectories, varying action dimension
        for _ in range(5):
            pred = torch.randn(20, dim)
            target = pred + torch.randn(20, dim) * 0.5

            metric.update(pred, target)

        results = metric.compute()
        print(f"  Action dim {dim:2d}: AMSE = {results['amse']:.4f}")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
