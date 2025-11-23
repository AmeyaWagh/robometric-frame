# Git Hooks Setup Guide

This document explains how to set up and use git hooks for automated code quality checks in the vla-metrics project.

## Overview

The project uses [pre-commit](https://pre-commit.com/) to manage git hooks that automatically check code quality before commits. The hooks include:

- **Ruff**: Fast Python linting and code formatting
- **MyPy**: Static type checking
- **Interrogate**: Docstring coverage checking
- **Standard checks**: Trailing whitespace, file endings, YAML/TOML validation, etc.

## Installation

### 1. Install Development Dependencies

First, install the package with development dependencies:

```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

This installs `pre-commit`, `ruff`, `mypy`, `interrogate`, and other dev tools.

### 2. Install Git Hooks

Install the pre-commit hooks into your git repository:

```bash
pre-commit install
```

This creates a `.git/hooks/pre-commit` script that runs automatically before each commit.

### 3. (Optional) Install Commit Message Hook

To also check commit messages:

```bash
pre-commit install --hook-type commit-msg
```

## Usage

### Automatic Running on Commit

Once installed, the hooks run automatically when you commit:

```bash
git add src/vla_metrics/new_file.py
git commit -m "feat: add new metric"
```

The hooks will:
1. Check for common issues (trailing whitespace, file endings, etc.)
2. Lint code with Ruff (and auto-fix issues)
3. Format code with Ruff
4. Type-check with MyPy
5. Check docstring coverage with Interrogate

If any hook fails, the commit is aborted and you'll see error messages.

### Manual Running

Run hooks manually on all files:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook on all files
pre-commit run ruff --all-files         # Lint
pre-commit run ruff-format --all-files  # Format
pre-commit run mypy --all-files         # Type check

# Run hooks on specific files
pre-commit run --files src/vla_metrics/task_performance/success_rate.py
```

### Skipping Hooks (Not Recommended)

In rare cases, you may need to skip hooks:

```bash
# Skip all hooks (use sparingly!)
git commit --no-verify -m "message"

# Set SKIP environment variable to skip specific hooks
SKIP=mypy git commit -m "message"
```

**Note**: Only skip hooks when absolutely necessary. Code should pass all checks before merging.

## Hook Details

### Ruff - Linter and Formatter

**What it does**: Fast Python linter and code formatter that replaces Black, Flake8, isort, and more. Checks for code quality issues, common bugs, and automatically formats code to ensure consistent style.

**Configuration**: `pyproject.toml` → `[tool.ruff]` and `[tool.ruff.format]`
- Line length: 100 characters
- Enabled rules: E (errors), F (pyflakes), I (imports), N (naming), W (warnings), B (bugbear), C4 (comprehensions), UP (pyupgrade)
- Format style: Double quotes, space indentation

**Example**:
```bash
# Lint and auto-fix issues
ruff check src/ tests/ examples/
ruff check --fix src/

# Format code
ruff format src/ tests/ examples/

# Check formatting without modifying
ruff format --check src/
```

### MyPy - Type Checker

**What it does**: Verifies type hints and catches type-related bugs.

**Configuration**: `pyproject.toml` → `[tool.mypy]`
- Target: Python 3.9+
- Requires type hints for all functions

**Example**:
```bash
# Run manually
mypy src/

# Run on specific module
mypy src/vla_metrics/task_performance/
```

### Interrogate - Docstring Coverage

**What it does**: Checks that code has adequate docstring coverage.

**Configuration**: `pyproject.toml` → `[tool.interrogate]`
- Minimum coverage: 80%
- Excludes: tests/, examples/

**Example**:
```bash
# Run manually
interrogate src/

# Detailed report
interrogate -v src/
```

## Updating Hooks

Pre-commit hooks are pinned to specific versions in `.pre-commit-config.yaml`. To update:

```bash
# Update to latest versions
pre-commit autoupdate

# Test updated hooks
pre-commit run --all-files
```

## Troubleshooting

### Hooks Not Running

If hooks don't run on commit:

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Verify installation
pre-commit run --all-files
```

### Hook Failures

If a hook fails:

1. **Read the error message**: It usually tells you exactly what's wrong
2. **Run the hook manually**: `pre-commit run <hook-name> --all-files`
3. **Fix the issue**: Make the required changes
4. **Re-run hooks**: `pre-commit run --all-files` to verify
5. **Commit again**: Try your commit again

### Slow Hook Performance

MyPy can be slow on large changes:

- **Run incrementally**: Commit smaller changes more frequently
- **Cache**: Pre-commit caches results; subsequent runs are faster
- **Skip temporarily**: Use `SKIP=mypy git commit -m "message"` for work-in-progress commits, but fix before pushing

### Type Checking Errors

If mypy reports errors about missing imports:

```bash
# Install type stubs
pip install types-setuptools types-requests

# Or ignore specific imports in pyproject.toml
# [tool.mypy]
# ignore_missing_imports = true  # Not recommended for production
```

## CI/CD Integration

The same hooks run in CI/CD pipelines to ensure code quality. Local hooks help catch issues before pushing:

```yaml
# Example GitHub Actions workflow
- name: Run pre-commit hooks
  run: |
    pip install pre-commit
    pre-commit run --all-files
```

## Configuration Files

- **`.pre-commit-config.yaml`**: Hook configuration and versions
- **`pyproject.toml`**: Tool-specific configurations (ruff, mypy, interrogate)
- **`.git/hooks/pre-commit`**: Auto-generated hook script (don't edit manually)

## Best Practices

1. **Install hooks immediately** after cloning the repository
2. **Run `pre-commit run --all-files`** periodically to catch issues
3. **Don't skip hooks** unless absolutely necessary
4. **Keep hooks updated** with `pre-commit autoupdate`
5. **Commit frequently** to minimize hook run time
6. **Fix issues immediately** rather than accumulating technical debt

## Additional Resources

- [Pre-commit documentation](https://pre-commit.com/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [MyPy documentation](https://mypy.readthedocs.io/)
- [Interrogate documentation](https://interrogate.readthedocs.io/)
