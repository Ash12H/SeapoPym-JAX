# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MUSCL advection with the van Leer limiter (`phys:transport_tendency_muscl`): every face flux is computed once, so
  mass is conserved to rounding on latitude-longitude grids, and it is positive with Euler or RK2 up to a Courant
  number of 0.5.
- Face velocities for MUSCL (`phys:transport_tendency_muscl_faces`), for currents given on the faces of a C-grid.
- An exterior field for open edges (`phys:transport_tendency_muscl_faces_exterior`): what flows in through an OPEN
  edge carries the value given outside the edge cell instead of the edge value, as limited-area models prescribe
  external data where the flow enters. With the edge value, an edge cell walled on its other sides can feed itself
  where the flow converges; with an exterior, what flows in no longer depends on what the cell holds. Everywhere
  else the tendencies are those of `phys:transport_tendency_muscl_faces` to the bit.
- A relaxation tendency for sponge layers along open boundaries (`phys:relaxation_tendency`).

### Changed
- Day length follows the CBM model of Forsythe et al. (1995) with p = 0, as the original SeapoPym, instead of Brock's
  declination.
- Cohort ageing (`lmtl:aging_flow`, `lmtl:recruitment_flow`): production is advected along the age axis by MUSCL with
  the van Leer limiter, and recruitment splits what leaves the cohorts around the recruitment age so that the mean
  delay equals it whatever the cohort widths; the last cohort recruits everything it holds. The one-day sigmoid is
  removed. **The cohort blueprints give different results**: the recruitment delay used to be lengthened by about one
  and a half cohort widths, shortened by up to a day with sub-daily cohorts, and production reaching the last cohort
  before its recruitment age stayed there. With a step shorter than the cohorts, the first-order outflow also spread
  the delay (a chain of compartments); the reconstruction keeps most of that spread away.
- Linear forcing interpolation (`ForcingStore`, `interp_method="linear"`) runs on the device: each chunk sends only the
  source frames it needs, with the bracket and weight of every step, and the steps are built where they are used.
  The host used to interpolate every field to every step with xarray, check it for NaN and transfer it, which took
  83 % of a GPU run on a 1/36 degree grid. Results change at float32 rounding (about 1e-6 relative after a month),
  since the interpolation is now done in float32 on the device instead of float64 on the host; a step on a source time
  gets that frame exactly, so a model stepping on its forcing's own times is unchanged to the bit. The NaN check reads
  the frames the brackets use, and repeated or unsorted source times are an error. nearest and ffill are unchanged.

### Fixed
- The gradient through cohort ageing is no longer NaN for small production. The van Leer limiter of
  `lmtl:aging_flow` and `lmtl:recruitment_flow` formed the squares of the raw age-axis differences in the reverse
  pass; production of 1e-6 g/m^2 in daily cohorts gives differences near 1e-18 per second, whose squares fall below
  the smallest float32. The forward pass was right, but a calibration on 400 cells got a NaN gradient for the energy
  transfer coefficients. The limiter is now evaluated on the two differences divided by the larger of them and scaled
  back, which leaves its value and its derivative unchanged.

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
