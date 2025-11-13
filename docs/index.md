# SEAPOPYM-Message Documentation

**Distributed Population Dynamics Simulation using Ray and JAX**

Welcome to the SEAPOPYM-Message documentation!

## What is SEAPOPYM-Message?

SEAPOPYM-Message is a modern, scalable framework for simulating spatially-explicit population dynamics with:

- **Event-Driven Architecture**: Asynchronous scheduler with priority queue
- **Composable Kernels**: Define models as lists of modular units
- **Distributed Computing**: Ray-based workers with intelligent message passing
- **High Performance**: JAX JIT compilation for GPU/TPU acceleration
- **2D Spatial Grids**: Native lat/lon support with halo exchange

## Key Features

### 🎯 Modular Design

Define your model by composing units:

```python
kernel = Kernel([
    compute_recruitment,
    compute_mortality,
    compute_growth,
    transport_diffusion
])
```

### ⚡ High Performance

- JAX JIT compilation
- GPU/TPU support
- Distributed computing with Ray
- Efficient halo exchange

### 🌍 Spatial Grids

Native support for 2D latitude/longitude grids with:

- Automatic domain decomposition
- Neighbor communication
- Boundary conditions
- Spherical geometry

### 📊 Scientific Rigor

- Type-checked with Mypy
- Comprehensive test suite
- Validated numerical schemes
- Reproducible results

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Architecture Overview](architecture/overview.md)
- [API Reference](api/core.md)
- [Tutorials](tutorials/0d-model.md)

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/seapopym-message/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/seapopym-message/discussions)

## License

SEAPOPYM-Message is released under the [MIT License](about/license.md).
