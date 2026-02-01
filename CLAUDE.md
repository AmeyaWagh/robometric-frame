# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository provides a TorchMetrics-based library for evaluating robotics policies and robot learning models. It includes both comprehensive metric documentation and a Python implementation compatible with PyTorch, PyTorch Lightning, and Hugging Face Transformers.

## Repository Structure

```
robometric-frame/
├── .vscode/                      # VS Code configuration
│   ├── settings.json            # Editor settings, formatters, linters
│   ├── extensions.json          # Recommended extensions
│   ├── launch.json              # Debug configurations
│   └── tasks.json               # Build and test tasks
├── src/robometric_frame/              # Source code (src layout)
│   ├── __init__.py              # Package initialization, exports
│   └── task_performance/         # Task performance metrics module
│       ├── __init__.py
│       └── success_rate.py      # SuccessRate metric implementation
├── tests/                        # Test suite
│   ├── __init__.py
│   └── test_success_rate.py     # SuccessRate tests
├── examples/                     # Usage examples
│   ├── basic_success_rate.py    # Basic usage examples
│   ├── distributed_training.py  # Distributed training examples
│   └── README.md                # Examples documentation
├── docs/                         # Documentation
│   └── metrics.md               # Comprehensive robotics metrics reference
├── .pre-commit-config.yaml      # Pre-commit hooks configuration
├── pyproject.toml               # Project configuration and dependencies
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
└── CLAUDE.md                    # This file
```

## Installation & Setup

### Development Installation with uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install project with development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Alternative: Traditional pip Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

### Dependencies

All dependencies are defined in `pyproject.toml`:

- **Core**: torch>=1.10.0, torchmetrics>=0.11.0, numpy>=1.20.0
- **Dev**: pytest>=7.0.0, pytest-cov>=4.0.0, ruff>=0.1.0, mypy>=1.0.0, pre-commit>=3.0.0, interrogate>=1.5.0

### VS Code Setup

The project includes VS Code configuration for a consistent development experience:

**Recommended Extensions:**
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Ruff (charliermarsh.ruff)
- Mypy Type Checker (ms-python.mypy-type-checker)

**Features:**
- Auto-formatting with Ruff on save (100 char line length)
- Linting with Ruff and type checking with Mypy
- Integrated testing with pytest
- Debug configurations for examples and tests
- Pre-configured tasks (Cmd/Ctrl+Shift+P → "Tasks: Run Task")

After opening the project in VS Code:
1. Install recommended extensions when prompted
2. Select the Python interpreter: `.venv/bin/python`
3. Run tests from the Testing panel or use debug configurations

## Development Commands

**Note**: Make sure to activate your virtual environment before running these commands:
```bash
source .venv/bin/activate
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=robometric_frame --cov-report=term-missing

# Run with HTML coverage report
pytest --cov=robometric_frame --cov-report=html

# Run specific test file
pytest tests/test_success_rate.py -v

# Run specific test
pytest tests/test_success_rate.py::TestSuccessRate::test_binary_success_perfect -v
```

### Code Quality

**Git Hooks**: Pre-commit hooks automatically run before each commit.

```bash
# Run all pre-commit hooks manually
pre-commit run --all-files

# Run specific hooks
pre-commit run ruff --all-files         # Lint
pre-commit run ruff-format --all-files  # Format
pre-commit run mypy --all-files         # Type check

# Run individual tools directly
ruff check src/ tests/ examples/  # Lint code
ruff format src/ tests/ examples/ # Format code
mypy src/                         # Type checking
interrogate src/                  # Docstring coverage

# Update pre-commit hooks
pre-commit autoupdate
```

### Running Examples

```bash
# Basic success rate examples
python examples/basic_success_rate.py

# Distributed training examples
python examples/distributed_training.py
```

## Architecture

### Metrics Design Pattern

All metrics follow the TorchMetrics pattern:

1. **Inherit from `torchmetrics.Metric`**: Base class provides distributed training support
2. **Initialize states in `__init__`**: Use `add_state()` for distributed sync
3. **Update states in `update()`**: Accumulate metric values
4. **Compute final result in `compute()`**: Calculate metric from accumulated states
5. **Reset states in `reset()`**: Clear states for next epoch (inherited)

### SuccessRate Implementation (src/robometric_frame/task_performance/success_rate.py)

```python
class SuccessRate(Metric):
    """Success Rate metric following TorchMetrics pattern."""

    def __init__(self, threshold=None, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        # States automatically synced in distributed training
        self.add_state("total_success", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_tasks", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, success: Tensor) -> None:
        """Accumulate success indicators."""
        # Apply threshold, handle ignore_index, update states

    def compute(self) -> Tensor:
        """Calculate final success rate."""
        return self.total_success.float() / self.total_tasks
```

### Key Features

- **Binary & Continuous Support**: Accepts binary (0/1) or continuous scores with threshold
- **Distributed Training**: States automatically synced across processes
- **GPU Support**: Works seamlessly on CUDA devices
- **Type Safety**: Full type annotations (Python 3.8+)
- **Error Handling**: Validates inputs and provides clear error messages

## Metric Categories (from docs/metrics.md)

### 1. Task Performance Metrics

- **Success Rate (SR)**: `SR = N_success / N_total` (✅ Implemented)
- **Task Completion Rate (TCR)**: Multi-step task sequences (🔜 Planned)
- **Action Accuracy**: MSE, AMSE, NAMSE (🔜 Planned)

### 2. Trajectory Quality Metrics (🔜 Planned)

- Path Length (PL), Path Smoothness (PS), Curvature Change (CC), Trajectory Errors (ATE/RTE)

### 3. Vision-Language Alignment Metrics (🔜 Planned)

- BLEU, CIDEr, METEOR, IoU, CLIP Score

### 4. Safety and Robustness Metrics (🔜 Planned)

- Collision Rate (CR), Obstacle Proximity (OP), Risk Factor (RF)

### 5. Efficiency Metrics (🔜 Planned)

- Inference Latency (IL), Computation Time (CT), Memory Usage (MU)

## Testing Guidelines

### Test Structure

Tests are organized by metric in `tests/`:
- `test_success_rate.py`: Comprehensive tests for SuccessRate

### Test Coverage Areas

1. **Basic Functionality**: Perfect/zero/partial success rates
2. **Input Variations**: Binary, continuous, bool, different dtypes
3. **Edge Cases**: Empty tensors, all ignored values, large batches
4. **Multi-batch Updates**: Incremental updates, reset functionality
5. **Special Features**: Threshold, ignore_index
6. **Error Handling**: Invalid inputs, compute before update
7. **Hardware**: GPU support (if CUDA available)

### Writing New Tests

Follow the existing pattern in `test_success_rate.py`:
```python
class TestNewMetric:
    def test_basic_case(self) -> None:
        """Test basic functionality."""
        metric = NewMetric()
        # ... test logic
        assert result == expected
```

## Code Style

- **Line Length**: 100 characters (configured in pyproject.toml)
- **Type Hints**: Required for all public functions
- **Docstrings**: Google-style docstrings with examples
- **Imports**: Organized (stdlib, third-party, local)
- **Formatting**: Black (automatic)
- **Linting**: Ruff (E, F, I, N, W, B, C4, UP rules)

## Integration Patterns

### PyTorch Training Loop
```python
metric = SuccessRate()
for batch in dataloader:
    success = evaluate(model(batch))
    metric.update(success)
epoch_sr = metric.compute()
metric.reset()  # For next epoch
```

### PyTorch Lightning
```python
class RobotPolicyModel(pl.LightningModule):
    def __init__(self):
        self.val_sr = SuccessRate()

    def validation_step(self, batch, batch_idx):
        self.val_sr.update(evaluate(self(batch)))

    def on_validation_epoch_end(self):
        self.log("val_sr", self.val_sr.compute())
```

## Adding New Metrics

When implementing a new metric:

1. **Create metric file**: `src/robometric_frame/{category}/{metric_name}.py`
2. **Implement class**: Inherit from `torchmetrics.Metric`
3. **Add states**: Use `add_state()` for distributed support
4. **Implement methods**: `__init__`, `update`, `compute`
5. **Add exports**: Update `__init__.py` files
6. **Write tests**: Create `tests/test_{metric_name}.py`
7. **Add examples**: Create usage examples
8. **Document**: Update README.md and docs/

## References

All metrics are based on peer-reviewed research. See `docs/metrics.md` for comprehensive citations and mathematical formulations.

Key reference for Success Rate:
- Brohan et al., "RT-1: Robotics transformer for real-world control at scale," arXiv:2212.06817, 2022.
