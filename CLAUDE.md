# SeapoPym

Marine ecosystem simulation framework — JAX-native, functional, strictly validated.

## Pipeline

```
Blueprint (YAML) + Config → compile_model() → CompiledModel → simulate()/run() → Outputs
```

## Modules

- **`blueprint/`** : Model declaration (Pydantic). Blueprint (topology, variables, process steps, tendencies) + Config (params, forcings, initial state, execution). `@functional` decorator registers physics functions with units/dims metadata. Strict Pint-based unit validation.
- **`compiler/`** : `compile_model(blueprint, config)` → CompiledModel. Validates, builds TimeGrid, infers shapes, prepares ForcingStore (lazy xarray → JAX), transposes to canonical dim order.
- **`engine/`** : `simulate()` (high-level) and `run()` (pure loop via `jax.lax.scan`). `build_step_fn()` composes compute nodes → Euler tendencies → mask → diagnostics. Writers: WriterRaw (JAX-traceable), MemoryWriter (xr.Dataset), DiskWriter (Zarr). Auto-vmap on non-core dims.
- **`functions/`** : Physics implementations. `lmtl.py` (day length, temperature, recruitment, mortality, NPP, aging). `transport.py` (Zalesak advection/diffusion).
- **`models/`** : Pre-defined YAML blueprints (`seapodym_lmtl_no_transport.yaml`, `seapodym_lmtl.yaml`).
- **`optimization/`** : Gradient (Optax), CMA-ES, GA, IPOP-CMA-ES (evosax). `build_loss_fn` calls `run()` directly. Loss: RMSE, NRMSE, MSE. Priors: Uniform, Normal, LogNormal, etc.
- **`types.py`** : Type aliases (Array, State, Params, Forcings, Outputs).
- **`dims.py`** : Canonical dimension order `(E, T, F, C, Z, Y, X)`.

## Conventions

- Python >=3.12, JAX, xarray, Pydantic 2, Pint
- Build: Hatchling + uv
- snake_case, type hints, 120 chars, Google docstrings
- Tests: pytest (markers: slow, integration, unit, gpu)
- Linting: Ruff, Pyright (basic)
- Don't mention Claude in commit messages
- Use `uv` for environment tasks
- Functional composition over OO patterns
- Strict validation: no NaN, strict xr.DataArray typing
