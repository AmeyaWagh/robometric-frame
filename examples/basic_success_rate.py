"""Basic example of using SuccessRate metric."""

import torch

from vla_metrics import SuccessRate


def main() -> None:
    """Demonstrate basic SuccessRate usage."""
    print("=" * 60)
    print("VLA Metrics - Success Rate Example")
    print("=" * 60)

    # Example 1: Binary success indicators
    print("\nExample 1: Binary Success Indicators")
    print("-" * 60)
    metric = SuccessRate()

    # Simulate task outcomes (1 = success, 0 = failure)
    task_results = torch.tensor([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    print(f"Task results: {task_results.tolist()}")

    metric.update(task_results)
    success_rate = metric.compute()
    print(f"Success Rate: {success_rate:.2%}")

    # Example 2: Multiple batches (e.g., from different episodes)
    print("\nExample 2: Multiple Batches")
    print("-" * 60)
    metric.reset()

    episode1 = torch.tensor([1, 1, 0, 1])
    episode2 = torch.tensor([0, 1, 1])
    episode3 = torch.tensor([1, 0, 1, 1, 0])

    print(f"Episode 1: {episode1.tolist()}")
    metric.update(episode1)

    print(f"Episode 2: {episode2.tolist()}")
    metric.update(episode2)

    print(f"Episode 3: {episode3.tolist()}")
    metric.update(episode3)

    total_success_rate = metric.compute()
    print(f"Overall Success Rate: {total_success_rate:.2%}")

    # Example 3: Continuous scores with threshold
    print("\nExample 3: Continuous Scores with Threshold")
    print("-" * 60)
    metric_threshold = SuccessRate(threshold=0.8)

    # Simulate task completion scores (0.0 to 1.0)
    task_scores = torch.tensor([0.95, 0.75, 0.85, 0.60, 0.92, 0.78, 0.88])
    print(f"Task scores: {task_scores.tolist()}")
    print("Threshold: 0.8")

    metric_threshold.update(task_scores)
    success_rate = metric_threshold.compute()
    print(f"Success Rate (score >= 0.8): {success_rate:.2%}")

    # Example 4: Real-world scenario - Pick-up tasks
    print("\nExample 4: Real-world Scenario - Pick-up Tasks")
    print("-" * 60)
    metric_pickup = SuccessRate()

    # Simulate results from RT-1 paper benchmarks
    # Pick-up tasks across different objects
    pickup_results = {
        "easy_objects": torch.tensor([1, 1, 1, 1, 0, 1, 1, 1]),
        "medium_objects": torch.tensor([1, 0, 1, 1, 0, 1, 0]),
        "hard_objects": torch.tensor([0, 1, 0, 0, 1, 0]),
    }

    for difficulty, results in pickup_results.items():
        print(f"{difficulty}: {results.tolist()}")
        metric_pickup.update(results)

    overall_sr = metric_pickup.compute()
    print(f"\nOverall Pick-up Success Rate: {overall_sr:.2%}")
    print("(Compared to RT-1-400k: 34.4% on pick-up tasks)")


if __name__ == "__main__":
    main()
