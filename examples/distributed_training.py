"""Example of using SuccessRate in distributed training context."""

import torch

from vla_metrics import SuccessRate


def simulate_distributed_evaluation() -> None:
    """Simulate distributed evaluation across multiple processes/GPUs.

    In real distributed training, each process would compute metrics on its
    subset of data, and torchmetrics automatically syncs across processes.
    """
    print("=" * 60)
    print("VLA Metrics - Distributed Evaluation Simulation")
    print("=" * 60)

    print("\nSimulating evaluation across 4 GPUs...")
    print("-" * 60)

    # Create metric instances (one per process in real scenario)
    metrics = [SuccessRate() for _ in range(4)]

    # Simulate different task results on each GPU
    gpu_results = [
        torch.tensor([1, 1, 0, 1, 0]),  # GPU 0: 3/5 = 60%
        torch.tensor([1, 0, 1, 1]),  # GPU 1: 3/4 = 75%
        torch.tensor([0, 1, 1, 0, 1, 1]),  # GPU 2: 4/6 = 66.7%
        torch.tensor([1, 1, 1]),  # GPU 3: 3/3 = 100%
    ]

    # Update each metric with GPU-specific results
    local_rates = []
    for i, (metric, results) in enumerate(zip(metrics, gpu_results)):
        metric.update(results)
        local_rate = metric.compute()
        local_rates.append(local_rate)
        print(f"GPU {i} results: {results.tolist()}")
        print(f"GPU {i} local success rate: {local_rate:.2%}")

    # In real distributed setting, torchmetrics would sync automatically
    # Here we manually compute the global metric
    all_results = torch.cat(gpu_results)
    global_metric = SuccessRate()
    global_metric.update(all_results)
    global_rate = global_metric.compute()

    print(f"\nGlobal Success Rate (all GPUs): {global_rate:.2%}")
    print(f"Total tasks: {len(all_results)}")
    print(f"Total successes: {all_results.sum().item()}")


def batch_evaluation_example() -> None:
    """Example of batched evaluation in training loop."""
    print("\n" + "=" * 60)
    print("Batch Evaluation in Training Loop")
    print("=" * 60)

    metric = SuccessRate()
    num_epochs = 3
    batches_per_epoch = 4

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        epoch_metric = SuccessRate()

        for batch_idx in range(batches_per_epoch):
            # Simulate batch of task results
            batch_size = int(torch.randint(3, 8, (1,)).item())
            batch_results = torch.randint(0, 2, (batch_size,))

            # Update both running metric and epoch metric
            metric.update(batch_results)
            epoch_metric.update(batch_results)

            batch_sr = batch_results.float().mean()
            print(f"  Batch {batch_idx + 1}: {batch_results.tolist()} -> SR: {batch_sr:.2%}")

        # Compute epoch success rate
        epoch_sr = epoch_metric.compute()
        print(f"Epoch {epoch + 1} Success Rate: {epoch_sr:.2%}")

    # Compute overall success rate across all epochs
    overall_sr = metric.compute()
    print(f"\nOverall Training Success Rate: {overall_sr:.2%}")


if __name__ == "__main__":
    simulate_distributed_evaluation()
    batch_evaluation_example()
