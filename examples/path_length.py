"""Basic example of using PathLength metric for trajectory evaluation."""

import torch

from vla_metrics import PathLength


def main() -> None:
    """Demonstrate basic PathLength usage."""
    print("=" * 60)
    print("VLA Metrics - Path Length Example")
    print("=" * 60)

    # Example 1: Simple 2D trajectory
    print("\nExample 1: Simple 2D Trajectory")
    print("-" * 60)
    metric = PathLength()

    # Straight path from (0,0) to (3,0)
    trajectory = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    print(f"Trajectory points: {trajectory.tolist()}")

    metric.update(trajectory)
    path_length = metric.compute()
    print(f"Path Length: {path_length:.4f} units")

    # Example 2: L-shaped path vs straight diagonal
    print("\nExample 2: Comparing Path Efficiency")
    print("-" * 60)

    # L-shaped path (inefficient)
    metric_l = PathLength()
    path_l = torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    metric_l.update(path_l)
    length_l = metric_l.compute()
    print(f"L-shaped path: {length_l:.4f} units")

    # Diagonal path (efficient)
    metric_diag = PathLength()
    path_diag = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    metric_diag.update(path_diag)
    length_diag = metric_diag.compute()
    print(f"Direct diagonal: {length_diag:.4f} units")
    print(f"Efficiency gain: {((length_l - length_diag) / length_l * 100):.1f}% shorter")

    # Example 2b: Batched trajectories (B, L, D)
    print("\nExample 2b: Batched Processing - Shape (B, L, D)")
    print("-" * 60)
    metric_batch = PathLength()

    # Process multiple trajectories at once
    batch = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],  # Trajectory 1: straight, length 2.0
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],  # Trajectory 2: straight, length 2.0
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],  # Trajectory 3: diagonal, length ~2.83
        ]
    )

    print(
        f"Batch shape: {batch.shape} (B={batch.shape[0]}, L={batch.shape[1]}, D={batch.shape[2]})"
    )
    metric_batch.update(batch)
    avg_length = metric_batch.compute()
    print(f"Average path length across batch: {avg_length:.4f} units")

    # Example 2c: Temporal batched trajectories (B, T, L, D)
    print("\nExample 2c: Temporal Batched Processing - Shape (B, T, L, D)")
    print("-" * 60)
    metric_temporal = PathLength()

    # Batch of temporal sequences (e.g., multiple rollouts with temporal steps)
    # B=2 robots, T=3 timesteps, L=4 waypoints each, D=2 dimensions
    temporal_batch = torch.randn(2, 3, 4, 2).abs()  # Random positive trajectories

    print(
        f"Temporal batch shape: {temporal_batch.shape} "
        f"(B={temporal_batch.shape[0]}, T={temporal_batch.shape[1]}, "
        f"L={temporal_batch.shape[2]}, D={temporal_batch.shape[3]})"
    )
    print(f"Total trajectories: {temporal_batch.shape[0] * temporal_batch.shape[1]}")
    metric_temporal.update(temporal_batch)
    avg_temporal_length = metric_temporal.compute()
    print(f"Average path length: {avg_temporal_length:.4f} units")

    # Example 3: 3D robotic arm trajectory
    print("\nExample 3: 3D Robotic Arm Trajectory")
    print("-" * 60)
    metric_3d = PathLength()

    # Simulated end-effector trajectory in 3D space (x, y, z in meters)
    arm_trajectory = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Start position
            [0.1, 0.0, 0.0],  # Move along x
            [0.1, 0.1, 0.0],  # Move along y
            [0.1, 0.1, 0.1],  # Move along z
            [0.2, 0.1, 0.1],  # Final grasp position
        ]
    )

    print("3D End-effector trajectory:")
    for i, point in enumerate(arm_trajectory):
        print(f"  Point {i}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")

    metric_3d.update(arm_trajectory)
    length_3d = metric_3d.compute()
    print(f"Total arm movement: {length_3d:.4f} meters")

    # Example 4: Multiple trajectories (averaging)
    print("\nExample 4: Averaging Multiple Trajectories")
    print("-" * 60)
    metric_multi = PathLength()

    # Different attempts at the same task
    attempt1 = torch.tensor([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]])  # Curved path
    attempt2 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # Straight path
    attempt3 = torch.tensor([[0.0, 0.0], [1.0, 0.8], [2.0, 0.0]])  # More curved

    print("Attempt 1 (slightly curved):")
    metric_multi.update(attempt1)

    print("Attempt 2 (straight):")
    metric_multi.update(attempt2)

    print("Attempt 3 (more curved):")
    metric_multi.update(attempt3)

    avg_length = metric_multi.compute()
    print(f"\nAverage path length across attempts: {avg_length:.4f} units")

    # Example 5: Real-world scenario - Mobile robot navigation
    print("\nExample 5: Real-world Mobile Robot Navigation")
    print("-" * 60)

    # Simulated robot navigation paths (in meters)
    scenarios = {
        "obstacle_free": torch.tensor(
            [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]  # Direct path
        ),
        "obstacle_avoidance": torch.tensor(
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [5.0, 2.0],  # Detour around obstacle
                [7.0, 2.0],
                [10.0, 0.0],
            ]
        ),
        "narrow_corridor": torch.tensor(
            [
                [0.0, 0.0],
                [2.0, 0.5],  # Slight adjustments
                [4.0, 0.3],
                [6.0, 0.6],
                [8.0, 0.4],
                [10.0, 0.0],
            ]
        ),
    }

    print("Navigation scenario comparison:")
    for scenario_name, path in scenarios.items():
        metric_scenario = PathLength()
        metric_scenario.update(path)
        length = metric_scenario.compute()
        print(f"  {scenario_name:20s}: {length:.4f} meters")

    # Example 6: Batch processing with reset
    print("\nExample 6: Batch Processing with Reset")
    print("-" * 60)
    metric_batch = PathLength()

    # First batch of trajectories
    print("First batch (training set):")
    for _ in range(3):
        traj = torch.randn(5, 2) * 2 + torch.arange(5).unsqueeze(1)
        metric_batch.update(traj)

    batch1_avg = metric_batch.compute()
    print(f"  Average path length: {batch1_avg:.4f} units")

    # Reset and process second batch
    metric_batch.reset()
    print("\nSecond batch (validation set):")
    for _ in range(3):
        traj = torch.randn(5, 2) * 1.5 + torch.arange(5).unsqueeze(1)
        metric_batch.update(traj)

    batch2_avg = metric_batch.compute()
    print(f"  Average path length: {batch2_avg:.4f} units")

    # Example 7: Circular path approximation
    print("\nExample 7: Circular Path")
    print("-" * 60)
    metric_circle = PathLength()

    # Create points approximating a circle
    num_points = 16
    angles = torch.linspace(0, 2 * torch.pi, num_points + 1)
    radius = 1.0
    circle_path = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)

    print(f"Circle with radius {radius:.1f} m, {num_points} segments")
    metric_circle.update(circle_path)
    circle_length = metric_circle.compute()
    theoretical_circumference = 2 * torch.pi * radius

    print(f"Measured path length: {circle_length:.4f} meters")
    print(f"Theoretical circumference: {theoretical_circumference:.4f} meters")
    print(f"Approximation error: {abs(circle_length - theoretical_circumference):.4f} meters")

    print("\n" + "=" * 60)
    print("Tips for using PathLength:")
    print("  - Shorter paths generally indicate more efficient execution")
    print("  - Compare paths for the same task to evaluate optimization")
    print("  - Works with any dimensionality (2D, 3D, higher)")
    print("  - Supports batched processing: (..., L, D)")
    print("    * (L, D): Single trajectory")
    print("    * (B, L, D): Batch of trajectories")
    print("    * (B, T, L, D): Temporal batches")
    print("  - Use reset() between different evaluation phases")
    print("  - Average multiple attempts to account for variability")
    print("=" * 60)


if __name__ == "__main__":
    main()
