"""ForcingManager: Central orchestrator for environmental forcings.

The ForcingManager handles interpolation and distribution of environmental
forcing data (temperature, currents, primary production, etc.) to distributed workers.

Key features:
- Temporal interpolation using xarray.interp()
- Caching in Ray object store for zero-copy sharing
- Support for derived forcings (computed from base forcings)

Note:
    ForcingManager expects pre-loaded xarray Datasets. Loading from files
    is the user's responsibility, allowing full control over I/O (chunking,
    caching, parallel loading, etc.).

Metadata:
    Datasets should store metadata in their .attrs dictionary:
    - 'units': Physical units (e.g., '°C', 'm/s')
    - 'interpolation_method': Temporal interpolation method ('linear', 'nearest')
"""

from typing import Any

import jax.numpy as jnp
import ray
import xarray as xr

from seapopym_message.forcing.derived import resolve_dependencies


class ForcingManager:
    """Central manager for environmental forcing data.

    The ForcingManager performs temporal interpolation and distributes data
    to workers via Ray object store.

    Architecture:
    - Level 1: Base forcings (pre-loaded xarray Datasets)
    - Level 2: Derived forcings (computed from base forcings)
    - Distribution via Ray object store (zero-copy)

    Args:
        datasets: Dictionary mapping forcing names to pre-loaded xarray Datasets.
                 Datasets should include metadata in .attrs:
                 - 'interpolation_method': 'linear' or 'nearest' (default: 'linear')
                 - 'units': Physical units (optional, for documentation)
        derived_forcings: Optional dict of DerivedForcing instances.

    Example:
        >>> import xarray as xr
        >>> # User controls I/O
        >>> temp_ds = xr.open_zarr("data/temp.zarr", chunks={'time': 10})
        >>> temp_ds.attrs['units'] = '°C'
        >>> temp_ds.attrs['interpolation_method'] = 'linear'
        >>>
        >>> manager = ForcingManager(datasets={'temperature': temp_ds})
        >>> forcings_at_t = manager.prepare_timestep(time=3600.0)
    """

    def __init__(
        self,
        datasets: dict[str, xr.Dataset],
        derived_forcings: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ForcingManager with pre-loaded datasets.

        Args:
            datasets: Dict mapping forcing names to xarray Datasets.
            derived_forcings: Optional dict of DerivedForcing instances.
        """
        self.datasets = datasets

        # Cache for interpolated forcings
        # Key: (time,) -> Ray ObjectRef
        self._cache: dict[tuple[float], ray.ObjectRef] = {}

        # Registry for derived forcings
        self.derived_forcings = derived_forcings if derived_forcings is not None else {}

    def _interpolate_time(
        self, dataset: xr.Dataset, time: float, method: str | None = None
    ) -> xr.Dataset:
        """Interpolate dataset to a specific time.

        Args:
            dataset: xarray Dataset to interpolate.
            time: Target time (in same units as dataset's time coordinate).
            method: Interpolation method ("linear", "nearest").
                   If None, reads from dataset.attrs['interpolation_method'],
                   defaulting to 'linear'.

        Returns:
            Interpolated dataset at the given time.

        Raises:
            ValueError: If time is outside the dataset's time range (no extrapolation).
        """
        time_coord = dataset.coords["time"]
        time_min = float(time_coord.min())
        time_max = float(time_coord.max())

        if time < time_min or time > time_max:
            msg = (
                f"Time {time} is outside the forcing data range "
                f"[{time_min}, {time_max}]. Extrapolation is not supported."
            )
            raise ValueError(msg)

        # Get interpolation method from attrs if not provided
        if method is None:
            method = dataset.attrs.get("interpolation_method", "linear")

        # Interpolate using xarray
        interpolated = dataset.interp(time=time, method=method)

        return interpolated

    def prepare_timestep(
        self, time: float, params: dict[str, Any] | None = None
    ) -> dict[str, jnp.ndarray]:
        """Prepare all forcings for a given timestep.

        Loads and interpolates all base forcings to the specified time.
        Then computes derived forcings if registered.
        Results are converted to JAX arrays for use in kernels.

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
            >>> forcings["primary_production"].shape  # (lat, lon)
            (100, 100)
        """
        if params is None:
            params = {}

        forcings: dict[str, jnp.ndarray] = {}

        # Load base forcings from datasets
        for name, dataset in self.datasets.items():
            # Interpolate to time (method read from dataset.attrs)
            interpolated = self._interpolate_time(dataset, time)

            # Convert to JAX array
            # Assuming single data variable in dataset, or using name as variable
            if name in interpolated.data_vars:
                data_array = interpolated[name]
            else:
                # Use first data variable if name doesn't match
                data_array = interpolated[list(interpolated.data_vars)[0]]

            forcings[name] = jnp.array(data_array.values)

        # Compute derived forcings in dependency order
        if self.derived_forcings:
            derived_order = resolve_dependencies(self.derived_forcings)

            for name in derived_order:
                derived = self.derived_forcings[name]
                forcings[name] = derived.compute(forcings, params)

        return forcings

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

        Example:
            >>> forcings_ref = manager.prepare_timestep_distributed(time=3600.0)
            >>> # Workers can access with ray.get(forcings_ref)
        """
        # Check cache (include params in key if they affect result)
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

        Derived forcings are computed from base forcings and/or other derived forcings.
        They are created using the @derived_forcing decorator.

        Args:
            derived_forcing: DerivedForcing instance (created by decorator).

        Example:
            >>> @derived_forcing(
            ...     name="recruitment",
            ...     inputs=["primary_production"],
            ...     params=["transfer_coefficient"],
            ... )
            >>> def compute_recruitment(primary_production, transfer_coefficient):
            ...     return primary_production * transfer_coefficient
            ...
            >>> manager.register_derived(compute_recruitment)
        """
        self.derived_forcings[derived_forcing.name] = derived_forcing

    def __repr__(self) -> str:
        """String representation."""
        num_base = len(self.datasets)
        num_derived = len(self.derived_forcings)
        return f"ForcingManager(base_forcings={num_base}, " f"derived_forcings={num_derived})"
