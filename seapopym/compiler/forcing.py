"""ForcingStore: lazy forcing loading with xarray-based interpolation.

Encapsulates all forcings and provides chunked access with on-the-fly
temporal interpolation using xr.DataArray.interp() / .reindex().

Key features:
- Lazy loading: xr.DataArray forcings are NOT materialized at compile time
- xarray interpolation: uses scipy (linear) or pandas (nearest/ffill) under the hood
- Unified chunking API: get_chunk() provides chunked access for all runners
- Static/dynamic separation: forcings without dim T are static, with dim T are dynamic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

from seapopym.types import Array


def _check_nan(name: str, arr: np.ndarray) -> None:
    """Raise ValueError if array contains NaN values."""
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        raise ValueError(f"Forcing '{name}' contains NaN values. Handle NaN upstream.")


@dataclass
class ForcingStore:
    """Encapsulates all forcings with lazy loading and chunked access.

    Forcings are separated into:
    - Static (no T dim): constants across time, stored as xr.DataArray
    - Dynamic (with T dim): time-varying, stored as xr.DataArray

    The store exposes a unified API for chunking and interpolation.

    Attributes:
        _static: Static forcing data (no time dimension).
        _dynamic: Dynamic forcing data (with time dimension).
        n_timesteps: Total number of simulation timesteps.
        interp_method: Interpolation method ("constant", "nearest", "linear", "ffill").
        _time_coords: Target datetime coordinates for interpolation (from TimeGrid).
    """

    _static: dict[str, xr.DataArray] = field(default_factory=dict)
    _dynamic: dict[str, xr.DataArray] = field(default_factory=dict)
    n_timesteps: int = 1
    interp_method: str = "constant"
    _time_coords: np.ndarray | None = None

    def get_chunk(self, start: int, end: int) -> dict[str, Array]:
        """Load and interpolate a temporal chunk [start, end).

        Returns only dynamic forcings (with dim T). Static forcings
        must be accessed separately via get_statics().

        Args:
            start: Start timestep (inclusive).
            end: End timestep (exclusive).

        Returns:
            Dict of dynamic forcing arrays with shape (chunk_len, ...).
        """
        result: dict[str, Array] = {}

        for name, data in self._dynamic.items():
            arr = self._load_dynamic(name, data, start, end)
            _check_nan(name, np.asarray(arr))
            result[name] = jnp.asarray(arr)

        return result

    def get_statics(self) -> dict[str, Array]:
        """Convert all static forcings to JAX arrays.

        Called once by the Runner to capture statics in closure.

        Returns:
            Dict of static forcing JAX arrays (no time dimension).
        """
        result = {}
        for name, da in self._static.items():
            values = da.values
            _check_nan(name, values)
            result[name] = jnp.asarray(values)
        return result

    def get_all_dynamic(self) -> dict[str, Array]:
        """Materialize all dynamic forcings for the full simulation.

        Returns only dynamic forcings (with dim T), like get_chunk().
        Static forcings must be accessed via get_statics().

        Returns:
            Dict of dynamic forcing arrays with full time dimension.
        """
        return self.get_chunk(0, self.n_timesteps)

    def get(self, name: str, default: Any = None) -> Any:
        """Access a single forcing, materializing if lazy.

        Args:
            name: Forcing name.
            default: Default value if not found.

        Returns:
            Forcing array or default.
        """
        if name in self._static:
            return jnp.asarray(self._static[name].values)
        if name in self._dynamic:
            return jnp.asarray(self._dynamic[name].values)
        return default

    @classmethod
    def from_config(
        cls,
        forcings: dict[str, xr.DataArray],
        blueprint_dims: dict[str, list[str] | None],
        n_timesteps: int,
        interp_method: str = "constant",
        time_coords: np.ndarray | None = None,
    ) -> ForcingStore:
        """Build a ForcingStore from config forcings.

        Separates forcings into static (no T dim) and dynamic (with T dim)
        based on blueprint dimension declarations.

        Args:
            forcings: Forcing data as xr.DataArray (already transposed/mapped).
            blueprint_dims: Dimension declarations from blueprint (keyed by "forcings.<name>").
            n_timesteps: Total number of simulation timesteps.
            interp_method: Interpolation method.
            time_coords: Target datetime coordinates for interpolation.

        Returns:
            ForcingStore with separated static/dynamic forcings.
        """
        static: dict[str, xr.DataArray] = {}
        dynamic: dict[str, xr.DataArray] = {}

        for name, da in forcings.items():
            bp_dims = blueprint_dims.get(f"forcings.{name}")
            is_dynamic = bp_dims is not None and "T" in bp_dims

            if is_dynamic:
                dynamic[name] = da
            else:
                static[name] = da

        return cls(
            _static=static,
            _dynamic=dynamic,
            n_timesteps=n_timesteps,
            interp_method=interp_method,
            _time_coords=time_coords,
        )

    def __contains__(self, name: str) -> bool:
        return name in self._static or name in self._dynamic

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def _load_dynamic(
        self,
        name: str,
        data: xr.DataArray,
        start: int,
        end: int,
    ) -> Array:
        """Load a dynamic forcing for the given chunk range.

        Args:
            name: Forcing name.
            data: Dynamic forcing as xr.DataArray (lazy or loaded, with dim T).
            start: Start timestep (inclusive).
            end: End timestep (exclusive).

        Returns:
            Array sliced/interpolated for the chunk.
        """
        source_len = data.sizes["T"]

        if source_len == self.n_timesteps or self.interp_method == "constant":
            return data.isel(T=slice(start, end)).values

        # Interpolation needed
        if self._time_coords is None:
            raise ValueError(
                f"Forcing '{name}' requires interpolation but _time_coords is None. "
                f"Provide _time_coords when constructing ForcingStore."
            )
        target_times = self._time_coords[start:end]
        return self._xarray_interpolate(data, target_times)

    def _xarray_interpolate(self, data: xr.DataArray, target_times: np.ndarray) -> np.ndarray:
        """Interpolate a lazy DataArray to target times via xarray."""
        data = self._compute_source_window(data, target_times)
        if self.interp_method == "linear":
            return data.interp(T=target_times, method="linear",
                               kwargs={"fill_value": "extrapolate"}).values
        if self.interp_method in ("nearest", "ffill"):
            return data.reindex(T=target_times, method=self.interp_method).values
        raise ValueError(f"Unknown interp_method: {self.interp_method}")

    def _compute_source_window(self, data: xr.DataArray, target_times: np.ndarray) -> xr.DataArray:
        """Slice source DataArray to minimal temporal window for interpolation."""
        if "T" not in data.dims or len(target_times) == 0:
            return data

        t_index = data.indexes["T"]
        source_len = len(t_index)

        # Bracket: 1 point before first target, 1 point after last target
        i_start = max(0, t_index.searchsorted(target_times[0], side="right") - 1)
        i_end = min(source_len, t_index.searchsorted(target_times[-1], side="left") + 2)

        if i_end - i_start >= source_len:
            return data

        return data.isel(T=slice(i_start, i_end))

