# VLA Metrics

[![CI](https://github.com/ameyawagh/vla-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/ameyawagh/vla-metrics/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TorchMetrics-based evaluation metrics for Vision-Language-Action (VLA) models in robotics.

## Overview

`vla-metrics` provides a comprehensive suite of evaluation metrics specifically designed for Vision-Language-Action models in robotics. Built on top of [TorchMetrics](https://torchmetrics.readthedocs.io/), it offers:

- **Easy Integration**: Drop-in compatibility with PyTorch, PyTorch Lightning, and Hugging Face
- **Distributed Training**: Native support for multi-GPU/multi-node training
- **Type Safety**: Full type annotations for better IDE support
- **Well Tested**: Comprehensive test coverage
- **Extensible**: Easy to extend with custom metrics

## Installation

```bash
# Install from source
git clone https://github.com/ameyawagh/vla-metrics.git
cd vla-metrics

# Using uv (recommended - faster)
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
uv pip install -e .

# Or using pip
pip install -e .
```

## Quick Start

```python
import torch
from vla_metrics import SuccessRate

# Initialize metric
metric = SuccessRate()

# Evaluate task outcomes (1 = success, 0 = failure)
task_results = torch.tensor([1, 1, 0, 1, 0, 0, 1])
metric.update(task_results)

# Compute success rate
success_rate = metric.compute()
print(f"Success Rate: {success_rate:.2%}")  # Success Rate: 57.14%
```

## Metrics

### Task Performance Metrics

#### Success Rate (SR)

Measures the percentage of successfully completed tasks:

```
SR = N_success / N_total
```

**Usage:**

```python
from vla_metrics import SuccessRate

# Binary indicators
metric = SuccessRate()
success = torch.tensor([1, 1, 0, 1, 0])
metric.update(success)
print(metric.compute())  # tensor(0.6000)

# Continuous scores with threshold
metric = SuccessRate(threshold=0.8)
scores = torch.tensor([0.9, 0.7, 0.85, 0.6])
metric.update(scores)
print(metric.compute())  # tensor(0.5000)
```

**Parameters:**
- `threshold` (float, optional): Threshold for continuous scores. Default: None
- `ignore_index` (int, optional): Value to ignore. Default: None

**Reference:** Brohan et al., "RT-1: Robotics transformer for real-world control at scale," arXiv:2212.06817, 2022.

## Features

### Distributed Training Support

All metrics support distributed training out of the box:

```python
import torch.distributed as dist
from vla_metrics import SuccessRate

# Automatically syncs across all processes
metric = SuccessRate()

# Each process updates with its local data
local_results = torch.tensor([1, 0, 1])
metric.update(local_results)

# Compute aggregates results from all processes
global_success_rate = metric.compute()
```

### Multi-Batch Updates

Metrics can be updated incrementally:

```python
metric = SuccessRate()

# Update with multiple batches
for batch in dataloader:
    results = evaluate_batch(batch)
    metric.update(results)

# Compute overall success rate
overall_sr = metric.compute()

# Reset for next epoch
metric.reset()
```

### GPU Support

Metrics work seamlessly on GPU:

```python
metric = SuccessRate().to("cuda")
success = torch.tensor([1, 1, 0, 1], device="cuda")
metric.update(success)
result = metric.compute()  # Result is on GPU
```

## Integration Examples

### PyTorch Training Loop

```python
from vla_metrics import SuccessRate

success_metric = SuccessRate()

for epoch in range(num_epochs):
    for batch in dataloader:
        predictions = model(batch)
        success = evaluate_tasks(predictions, batch.targets)
        success_metric.update(success)

    epoch_sr = success_metric.compute()
    print(f"Epoch {epoch} SR: {epoch_sr:.2%}")
    success_metric.reset()
```

### PyTorch Lightning

```python
import pytorch_lightning as pl
from vla_metrics import SuccessRate

class VLAModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.val_success_rate = SuccessRate()

    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        success = self.evaluate(predictions, batch)
        self.val_success_rate.update(success)

    def on_validation_epoch_end(self):
        sr = self.val_success_rate.compute()
        self.log("val_sr", sr)
```

### Hugging Face Transformers

```python
from transformers import Trainer
from vla_metrics import SuccessRate

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    metric = SuccessRate()
    metric.update(torch.tensor(predictions))
    return {"success_rate": metric.compute().item()}

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/ameyawagh/vla-metrics.git
cd vla-metrics

# Using uv (recommended - faster)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

This installs all development dependencies and configures git hooks for automatic code quality checks on commit.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vla_metrics --cov-report=html

# Run specific test file
pytest tests/test_success_rate.py -v
```

### Code Quality

Pre-commit hooks automatically run code quality checks before each commit:

```bash
# Run all pre-commit hooks manually
pre-commit run --all-files

# Run specific hooks
pre-commit run ruff --all-files         # Lint code
pre-commit run ruff-format --all-files  # Format code
pre-commit run mypy --all-files         # Type checking

# Or run individual tools directly
ruff check src/ tests/ examples/   # Lint
ruff format src/ tests/ examples/  # Format
mypy src/                          # Type check
```

**What runs on commit:**
- Code formatting (Ruff)
- Linting (Ruff)
- Type checking (Mypy)
- Import sorting (Ruff)
- YAML/TOML validation
- Trailing whitespace removal

## Project Structure

```
vla-metrics/
├── src/vla_metrics/          # Source code
│   ├── __init__.py
│   └── task_performance/     # Task performance metrics
│       ├── __init__.py
│       └── success_rate.py
├── tests/                    # Test suite
│   ├── __init__.py
│   └── test_success_rate.py
├── examples/                 # Usage examples
│   ├── basic_success_rate.py
│   ├── distributed_training.py
│   └── README.md
├── docs/                     # Documentation
│   └── metrics.md
├── .github/workflows/        # CI/CD
│   └── ci.yml
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Roadmap

### Upcoming Metrics

- **Task Performance**: Task Completion Rate, Action Accuracy (MSE/AMSE/NAMSE)
- **Trajectory Quality**: Path Length, Path Smoothness, Curvature Change, Trajectory Errors
- **Vision-Language Alignment**: BLEU, CIDEr, METEOR, IoU, CLIP Score
- **Safety & Robustness**: Collision Rate, Obstacle Proximity, Risk Factor
- **Efficiency**: Inference Latency, Computation Time, Memory Usage

See [docs/metrics.md](docs/metrics.md) for detailed metric descriptions and formulations.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up your development environment
- Branching strategy
- Testing requirements
- Submitting pull requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{vla_metrics,
  title = {VLA Metrics: Evaluation Metrics for Vision-Language-Action Models},
  author = {Wagh, Ameya},
  year = {2025},
  url = {https://github.com/ameyawagh/vla-metrics}
}
```

## References

See [docs/metrics.md](docs/metrics.md) for comprehensive references to research papers and methodologies.

## Acknowledgments

- Built on [TorchMetrics](https://torchmetrics.readthedocs.io/)
- Inspired by VLA research including RT-1, RT-2, and other robotics transformers
