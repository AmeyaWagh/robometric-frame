#!/bin/bash
# Development environment setup script for FRAME (robometric-frame)

set -e  # Exit on error

echo "=========================================="
echo "FRAME - Development Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Install package with dev dependencies
echo "Installing robometric-frame with dev dependencies..."
pip install -e ".[dev]"
echo ""

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install
echo ""

# Run pre-commit on all files to verify setup
echo "Running pre-commit checks on all files..."
echo "(This may take a few minutes on first run)"
pre-commit run --all-files || {
    echo ""
    echo "⚠️  Some pre-commit checks failed."
    echo "This is normal for initial setup."
    echo "Please review the errors above and fix them."
    echo ""
}

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Git hooks are now installed. They will run automatically before each commit."
echo ""
echo "Useful commands:"
echo "  - Run all tests:          pytest"
echo "  - Run with coverage:      pytest --cov=robometric_frame"
echo "  - Format code:            black src/ tests/ examples/"
echo "  - Lint code:              ruff src/ tests/ examples/"
echo "  - Type check:             mypy src/"
echo "  - Run all hooks:          pre-commit run --all-files"
echo "  - Update hooks:           pre-commit autoupdate"
echo ""
echo "For more information, see docs/SETUP_HOOKS.md"
echo ""
