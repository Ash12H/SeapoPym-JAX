# Installation

## Requirements

- Python **>= 3.12**
- A working JAX installation (CPU or GPU)

## Quick Install

Install SeapoPym directly from GitHub:

```bash
pip install "seapopym @ git+https://github.com/Ash12H/SeapoPym-JAX.git"
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install "seapopym @ git+https://github.com/Ash12H/SeapoPym-JAX.git"
```

## Optional Extras

SeapoPym provides several optional dependency groups:

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `gpu` | JAX with CUDA 12 support | `pip install "seapopym[gpu] @ git+..."` |
| `optimization` | Evolutionary strategies (evosax) | `pip install "seapopym[optimization] @ git+..."` |
| `viz` | Plotting (matplotlib, seaborn, tqdm) | `pip install "seapopym[viz] @ git+..."` |
| `docs` | Documentation build tools | `pip install "seapopym[docs] @ git+..."` |
| `dev` | Testing and linting (pytest, ruff, pyright) | `pip install "seapopym[dev] @ git+..."` |

Combine extras with commas:

```bash
pip install "seapopym[optimization,viz] @ git+https://github.com/Ash12H/SeapoPym-JAX.git"
```

## GPU Support

SeapoPym runs on CPU by default. For GPU acceleration:

```bash
pip install "seapopym[gpu] @ git+https://github.com/Ash12H/SeapoPym-JAX.git"
```

!!! note
    GPU support requires CUDA 12 and a compatible NVIDIA driver. See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details.

## Development Setup

Clone the repository and install all dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Ash12H/SeapoPym-JAX.git
cd SeapoPym-JAX
uv sync --all-extras
```

This installs all optional groups (dev, docs, optimization, viz) in a virtual environment.

### Verify Installation

```bash
# Run the test suite
uv run pytest

# Check linting
uv run ruff check .

# Check types
uv run pyright

# Build documentation locally
uv run mkdocs serve
```
