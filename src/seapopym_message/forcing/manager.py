"""ForcingManager: Central orchestrator for environmental forcings.

The ForcingManager handles interpolation and distribution of environmental
forcing data (temperature, currents, primary production, etc.) to distributed workers.

Key features:
- Temporal interpolation using xarray.interp()
- Caching in Ray object store for zero-copy sharing
- Support for derived forcings (computed from base forcings)
- Support for N-dimensional forcings (e.g., 3D temperature fields)

Note:
    ForcingManager expects pre-loaded xarray Datasets wrapped in ForcingSource objects.
"""

from typing import Any

import jax.numpy as jnp
import ray

from seapopym_message.forcing.derived import resolve_dependencies
from seapopym_message.forcing.source import ForcingSource


class ForcingManager:
    """Central manager for environmental forcing data.

    The ForcingManager performs temporal interpolation and distributes data
    to workers via Ray object store.

    Architecture:
    - Level 1: Base forcings (ForcingSource objects)
    - Level 2: Derived forcings (computed from base forcings)
    - Distribution via Ray object store (zero-copy)

    Args:
        forcings: List of ForcingSource objects.
        derived_forcings: Optional dict of DerivedForcing instances.

    Example:
        >>> import xarray as xr
        >>> temp_ds = xr.open_zarr("data/temp.zarr")
        >>> temp_source = ForcingSource(temp_ds, name="temperature")
        >>> manager = ForcingManager(forcings=[temp_source])
        >>> forcings_at_t = manager.prepare_timestep(time=3600.0)
    """

    def __init__(
        self,
        forcings: list[ForcingSource],
        derived_forcings: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ForcingManager with forcing sources.

        Args:
            forcings: List of ForcingSource objects.
            derived_forcings: Optional dict of DerivedForcing instances.
        """
        self.forcings = {source.name: source for source in forcings}

        # Cache for interpolated forcings
        # Key: (time,) -> Ray ObjectRef
        self._cache: dict[tuple[float], ray.ObjectRef] = {}

        # Registry for derived forcings
        self.derived_forcings = derived_forcings if derived_forcings is not None else {}

    def prepare_timestep_xarray(
        self, time: float, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Prepare all forcings as xarray DataArrays (preserves metadata).

        Loads and interpolates all base forcings to the specified time.
        Then computes derived forcings if registered.
        Returns xarray DataArrays with dimension metadata preserved.

        Args:
            time: Simulation time.
            params: Parameters for derived forcings (optional).

        Returns:
            Dictionary mapping forcing names to xarray DataArrays.
            DataArrays preserve dimension names, coordinates, and attributes.
            Includes both base and derived forcings.

        Example:
            >>> forcings = manager.prepare_timestep_xarray(time=3600.0)
            >>> forcings["temperature"].dims  # ('depth', 'lat', 'lon')
            >>> forcings["temperature"].sel(depth=0)  # Select by name!
        """
        if params is None:
            params = {}

        import xarray as xr

        forcings_xr: dict[str, Any] = {}

        # Load base forcings from sources (keep as xarray)
        for name, source in self.forcings.items():
            # Interpolate to time (returns DataArray without time dim)
            forcings_xr[name] = source.interpolate(time)

        # Compute derived forcings in dependency order
        if self.derived_forcings:
            derived_order = resolve_dependencies(self.derived_forcings)

            for name in derived_order:
                derived = self.derived_forcings[name]
                # Pass xarray objects directly to derived forcing computation
                result = derived.compute(forcings_xr, params)

                # Ensure result is stored as DataArray
                if isinstance(result, xr.DataArray):
                    forcings_xr[name] = result
                else:
                    # Wrap result back as DataArray (inherit coords from first input)
                    # This supports legacy functions returning numpy/jax arrays
                    if derived.inputs:
                        first_input = forcings_xr[derived.inputs[0]]
                        forcings_xr[name] = xr.DataArray(
                            result, coords=first_input.coords, dims=first_input.dims
                        )
                    else:
                        forcings_xr[name] = xr.DataArray(result)

        return forcings_xr

    def prepare_timestep(
        self, time: float, params: dict[str, Any] | None = None
    ) -> dict[str, jnp.ndarray]:
        """Prepare all forcings for a given timestep (converts to JAX arrays).

        Loads and interpolates all base forcings to the specified time.
        Then computes derived forcings if registered.
        Results are converted to JAX arrays for use in JIT-compiled kernels.

        Args:
            time: Simulation time.
            params: Parameters for derived forcings (optional).

        Returns:
            Dictionary mapping forcing names to JAX arrays.
            Arrays have shape matching forcing dimensions (excluding time).
            Includes both base and derived forcings.

        Example:
            >>> forcings = manager.prepare_timestep(time=3600.0)
            >>> forcings["temperature"].shape  # (depth, lat, lon)
            (10, 100, 100)
        """
        # Get xarray forcings and convert to numpy
        forcings_xr = self.prepare_timestep_xarray(time, params)
        return {k: jnp.array(v.values) for k, v in forcings_xr.items()}

    def prepare_timestep_distributed(
        self, time: float, params: dict[str, Any] | None = None
    ) -> ray.ObjectRef:
        """Prepare forcings and put in Ray object store for distributed access.

        This method is optimized for distributed simulations. It loads forcings
        once and shares them across all workers using Ray's object store (zero-copy).

        Args:
            time: Simulation time.
            params: Parameters for derived forcings (optional).

        Returns:
            Ray ObjectRef pointing to forcings dict in object store.
        """
        # Check cache (include params in key if they affect result)
        # Note: We assume params don't change often or are None for base forcings
        cache_key = (time,)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Prepare forcings
        forcings = self.prepare_timestep(time, params)

        # Put in Ray object store
        forcings_ref = ray.put(forcings)

        # Cache the reference
        self._cache[cache_key] = forcings_ref

        return forcings_ref

    def register_derived(self, derived_forcing: Any) -> None:
        """Register a derived forcing function.

        Args:
            derived_forcing: DerivedForcing instance (created by decorator).
        """
        self.derived_forcings[derived_forcing.name] = derived_forcing

    def __repr__(self) -> str:
        """String representation."""
        num_base = len(self.forcings)
        num_derived = len(self.derived_forcings)
        return f"ForcingManager(base_forcings={num_base}, derived_forcings={num_derived})"
