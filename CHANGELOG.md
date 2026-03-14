# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-11

### Added
- Blueprint/Config declarative architecture with YAML support
- Compiler pipeline: validates, infers shapes, builds CompiledModel
- Engine with `simulate()` and `run()` functional API using `jax.lax.scan`
- Automatic `vmap` dispatch over non-core dimensions (canonical order: E, T, F, C, Z, Y, X)
- ForcingStore with lazy loading and temporal interpolation
- Output writers: WriterRaw (JAX-traceable), MemoryWriter (xr.Dataset), DiskWriter (Zarr)
- `@functional` decorator for registering physics functions with units and dimension metadata
- Strict unit validation via Pint
- LMTL ecosystem model (day length, temperature, recruitment, mortality, NPP, aging)
- First-order upwind finite-volume transport (advection/diffusion)
- Optimization module: Gradient (Optax), CMA-ES, GA, IPOP-CMA-ES (evosax)
- Prior distributions for parameter calibration
- GPU/TPU support via JAX

### Changed
- Complete rewrite using JAX instead of NumPy/Dask
- Functional API (`run()`, `simulate()`) replaces OOP Runner class
- Pydantic 2 for all schema validation
