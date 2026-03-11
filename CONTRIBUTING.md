# Contributing to SeapoPym

Thank you for your interest in contributing to SeapoPym!

## Prerequisites

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Ash12H/SeapoPym-JAX.git
cd SeapoPym-JAX

# Install all dependencies (including dev extras)
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

## Running Tests

```bash
uv run pytest
```

## Linting & Formatting

```bash
# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Check formatting
uv run ruff format --check .

# Apply formatting
uv run ruff format .
```

## Type Checking

```bash
uv run pyright
```

## Code Conventions

- **Style**: snake_case, 120-character line limit
- **Docstrings**: Google style
- **Type hints**: required on all public functions
- **Tests**: pytest, placed in `tests/` mirroring the package structure

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes
3. Ensure all tests pass (`uv run pytest`)
4. Ensure linting passes (`uv run ruff check .`)
5. Ensure type checking passes (`uv run pyright`)
6. Open a pull request against `main`
