# SeapoPym

![Tests](https://github.com/Ash12H/SeapoPym-JAX/actions/workflows/ci.yml/badge.svg)
![Docs](https://github.com/Ash12H/SeapoPym-JAX/actions/workflows/docs.yml/badge.svg)
[![Coverage](https://codecov.io/gh/Ash12H/SeapoPym-JAX/branch/main/graph/badge.svg)](https://codecov.io/gh/Ash12H/SeapoPym-JAX)
![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![JAX](https://img.shields.io/badge/powered_by-JAX-red)

**SeapoPym** is a JAX-accelerated framework for Eulerian population dynamics on N-dimensional grids. Models are declared as YAML blueprints (DAG of processes), compiled into optimized JAX graphs, and executed on CPU or GPU.

- **For scientists** — Explicit numerical schemes, YAML model declaration, strict unit validation (Pint), visual process DAG.
- **For ML engineers** — Pure JAX backend, end-to-end differentiable, GPU/TPU support, built-in optimization (Optax, CMA-ES, GA).

## Installation

```bash
pip install git+https://github.com/Ash12H/SeapoPym-JAX.git
```

With GPU support:
```bash
pip install "git+https://github.com/Ash12H/SeapoPym-JAX.git[gpu]"
```

For development:
```bash
git clone https://github.com/Ash12H/SeapoPym-JAX.git
cd SeapoPym-JAX
uv sync --all-extras
```

## Simulation pipeline

A model is declared as a YAML blueprint and configured with concrete data. The compiler validates units and shapes, then produces a `CompiledModel` ready for execution:

```mermaid
graph LR
    A[Blueprint YAML] --> C[compile_model]
    B[Config] --> C
    C --> D[CompiledModel]
    D --> E[simulate / run]
    E --> F[Outputs]

    style A fill:#1b4965,stroke:#0d2b3e,color:#fff
    style B fill:#1b4965,stroke:#0d2b3e,color:#fff
    style D fill:#1b4965,stroke:#0d2b3e,color:#fff
    style F fill:#1b4965,stroke:#0d2b3e,color:#fff
    style C fill:#e8833a,stroke:#c06a2a,color:#fff
    style E fill:#e8833a,stroke:#c06a2a,color:#fff
```

At each timestep, the process DAG (solid arrows) computes tendencies from state, parameters and forcings. An explicit Euler solver (dashed arrows) then integrates the tendencies to advance the state:

```mermaid
graph LR
    S[State] --> F[Process Function]
    P[Parameters] --> F
    Fo[Forcings] --> F
    F --> T[Tendency]
    T -.-> E[Euler Solver]
    S -.->|t| E
    E -.->|t+1| S

    style S fill:#1b4965,stroke:#0d2b3e,color:#fff
    style P fill:#1b4965,stroke:#0d2b3e,color:#fff
    style Fo fill:#1b4965,stroke:#0d2b3e,color:#fff
    style T fill:#1b4965,stroke:#0d2b3e,color:#fff
    style F fill:#e8833a,stroke:#c06a2a,color:#fff
    style E fill:#e8833a,stroke:#c06a2a,color:#fff
```

## Optimization

Parameter calibration builds on the same Blueprint + Config base. Two additional components are needed: **Objectives** (observed data + loss metric) and an **Optimizer** (calibration strategy):

```mermaid
graph LR
    A[Blueprint] --> C[compile_model]
    B[Config] --> C
    C --> M[CompiledModel]
    Obj[Objectives] --> Opt[Optimizer]
    M --> Opt
    Opt --> Res[Optimized Parameters]

    style A fill:#1b4965,stroke:#0d2b3e,color:#fff
    style B fill:#1b4965,stroke:#0d2b3e,color:#fff
    style M fill:#1b4965,stroke:#0d2b3e,color:#fff
    style Obj fill:#1b4965,stroke:#0d2b3e,color:#fff
    style Res fill:#1b4965,stroke:#0d2b3e,color:#fff
    style C fill:#e8833a,stroke:#c06a2a,color:#fff
    style Opt fill:#e8833a,stroke:#c06a2a,color:#fff
```

Three methods are available: **Gradient descent** (Optax), **Genetic Algorithm** and **CMA-ES** (evosax). Gradient-based optimization leverages JAX's automatic differentiation; evolutionary methods work without gradients.

## Documentation

Full documentation, conceptual guides, and runnable examples are available at:

**[https://ash12h.github.io/SeapoPym-JAX/](https://ash12h.github.io/SeapoPym-JAX/)**

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[MIT](LICENSE) — Jules Lehodey, 2024
