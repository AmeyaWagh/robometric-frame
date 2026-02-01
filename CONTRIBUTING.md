# Contributing to FRAME

## Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/robometric-frame.git
   cd robometric-frame
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   pre-commit install
   ```

3. **Verify setup:**
   ```bash
   pytest
   pre-commit run --all-files
   ```

## Branching Strategy

Create branches from `main` using the following naming convention:

```
<type>/<short-description>
```

**Branch types:**
- `feature/` - New metrics or features (e.g., `feature/collision-rate-metric`)
- `bugfix/` - Bug fixes (e.g., `bugfix/success-rate-type-error`)
- `hotfix/` - Critical production fixes (e.g., `hotfix/division-by-zero`)
- `docs/` - Documentation updates (e.g., `docs/update-installation`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-states`)

**Workflow:**
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# Make changes, commit regularly
git add .
git commit -m "Descriptive message"

# Keep branch updated
git checkout main
git pull origin main
git checkout feature/your-feature-name
git rebase main

# Push and create PR
git push origin feature/your-feature-name
```

## Testing

All code must pass tests and maintain 100% coverage:

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=robometric_frame --cov-report=term-missing

# Run pre-commit hooks
pre-commit run --all-files
```

**Test requirements:**
- All tests pass
- 100% code coverage for new code
- Test basic functionality, edge cases, and error handling
- Include GPU tests if applicable (can be skipped if CUDA unavailable)

## Pull Requests

**Before submitting:**
- [ ] All tests pass
- [ ] Pre-commit hooks pass
- [ ] Code is documented with docstrings
- [ ] Branch is up to date with main

**PR title format:**
```
<type>: Brief description

Examples:
feat: Add Collision Rate metric
fix: Correct SuccessRate with ignore_index
docs: Update README installation steps
```

**PR requirements:**
- At least one approval from maintainers
- All CI checks passing
- All review comments addressed

That's it! Thank you for contributing! 🤖
