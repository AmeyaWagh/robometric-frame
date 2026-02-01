"""Example usage of TaskCompletionRate metric for robotics policy evaluation.

This example demonstrates how to use the TaskCompletionRate metric to evaluate
multi-step task chain completion in robotics policies.
"""
# pylint: skip-file

import torch

from robometric_frame import TaskCompletionRate


def main() -> None:
    """Run TaskCompletionRate examples."""
    print("=" * 80)
    print("TaskCompletionRate Metric Examples")
    print("=" * 80)

    # Example 1: Basic binary completion
    print("\n1. Basic Binary Task Chain Completion")
    print("-" * 80)
    metric = TaskCompletionRate()

    # Simulate 5 task chains (e.g., pick-move-place sequences)
    # 1 = chain completed successfully, 0 = chain failed at some step
    completion = torch.tensor([1, 0, 1, 1, 0])

    metric.update(completion)
    tcr = metric.compute()

    print(f"Task chains: {completion.tolist()}")
    print(f"Completed: {completion.sum().item()}/{len(completion)}")
    print(f"Task Completion Rate: {tcr:.2%}")

    # Example 2: Multi-step evaluation with batches
    print("\n2. Multi-Step Evaluation (Multiple Batches)")
    print("-" * 80)
    metric = TaskCompletionRate()

    # Simulate evaluation over multiple episodes
    episodes = [
        torch.tensor([1, 0, 1]),  # Episode 1: 3 task chains, 2 completed
        torch.tensor([1, 1]),  # Episode 2: 2 task chains, 2 completed
        torch.tensor([0, 1, 0, 1]),  # Episode 3: 4 task chains, 2 completed
    ]

    for i, episode in enumerate(episodes, 1):
        metric.update(episode)
        completed = episode.sum().item()
        total = len(episode)
        print(f"Episode {i}: {completed}/{total} completed ({completed / total:.1%})")

    overall_tcr = metric.compute()
    print(f"\nOverall Task Completion Rate: {overall_tcr:.2%}")

    # Example 3: Continuous scores with threshold
    print("\n3. Continuous Completion Scores with Threshold")
    print("-" * 80)
    metric = TaskCompletionRate(threshold=0.8)

    # Continuous scores representing partial completion
    # (e.g., 0.9 = 90% of steps completed successfully)
    scores = torch.tensor([0.95, 0.70, 0.85, 0.60, 0.90, 0.75])

    metric.update(scores)
    tcr = metric.compute()

    print(f"Completion scores: {scores.tolist()}")
    print("Threshold: 0.8")
    print(f"Chains above threshold: {(scores >= 0.8).sum().item()}/{len(scores)}")
    print(f"Task Completion Rate: {tcr:.2%}")

    # Example 4: Realistic robotics policy evaluation scenario
    print("\n4. Realistic robotics Task Chain Evaluation")
    print("-" * 80)
    metric = TaskCompletionRate()

    # Simulate different task types with varying difficulty
    task_scenarios = {
        "Pick-and-Place": torch.tensor([1, 1, 0, 1, 1, 0, 1, 1]),  # 75% completion
        "Multi-Step Assembly": torch.tensor([0, 1, 0, 0, 1, 0]),  # 33% completion
        "Navigation-Manipulation": torch.tensor([1, 0, 1, 1]),  # 75% completion
    }

    print("\nTask-wise completion rates:")
    for task_name, results in task_scenarios.items():
        task_metric = TaskCompletionRate()
        task_metric.update(results)
        task_tcr = task_metric.compute()
        completed = results.sum().item()
        total = len(results)
        print(f"  {task_name:25s}: {completed:2d}/{total} ({task_tcr:.1%})")

        # Add to overall metric
        metric.update(results)

    overall_tcr = metric.compute()
    print(f"\nOverall Task Completion Rate: {overall_tcr:.2%}")

    # Example 5: Ignoring invalid chains
    print("\n5. Handling Invalid Task Chains (ignore_index)")
    print("-" * 80)
    metric = TaskCompletionRate(ignore_index=-1)

    # -1 indicates invalid/skipped task chains
    completion = torch.tensor([1, -1, 0, 1, -1, 1, 0])

    metric.update(completion)
    tcr = metric.compute()

    valid_chains = completion[completion != -1]
    print(f"Task chains: {completion.tolist()}")
    print(f"Valid chains: {valid_chains.tolist()}")
    print(f"Completed: {valid_chains.sum().item()}/{len(valid_chains)}")
    print(f"Task Completion Rate: {tcr:.2%}")

    # Example 6: Per-epoch evaluation
    print("\n6. Training Progress: Per-Epoch Evaluation")
    print("-" * 80)

    num_epochs = 5
    print(f"Simulating {num_epochs} training epochs...\n")

    for epoch in range(1, num_epochs + 1):
        epoch_metric = TaskCompletionRate()

        # Simulate improving performance over epochs
        num_chains = 10
        success_rate = min(0.3 + epoch * 0.15, 1.0)  # Gradually improving
        completions = torch.bernoulli(torch.full((num_chains,), success_rate)).long()

        epoch_metric.update(completions)
        epoch_tcr = epoch_metric.compute()

        completed = completions.sum().item()
        print(f"Epoch {epoch}: {completed:2d}/{num_chains} completed (TCR: {epoch_tcr:.2%})")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
