"""ForcingStore: lazy forcing loading with xarray-based interpolation.

Encapsulates all forcings and provides chunked access with on-the-fly
temporal interpolation using xr.DataArray.interp() / .reindex().

Key features:
- Lazy loading: xr.DataArray forcings are NOT materialized at compile time
- xarray interpolation: uses scipy (linear) or pandas (nearest/ffill) under the hood
- Unified chunking API: get_chunk() replaces _slice_forcings() in all runners
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
import xarray as xr

from seapopym.types import Array


@dataclass
class ForcingStore:
    """Encapsulates all forcings with lazy loading and chunked access.

    Forcings can be either:
    - xr.DataArray (lazy): loaded from file, materialized only when accessed
    - Array (in-memory): already materialized JAX arrays

    The store exposes a unified API for chunking and interpolation,
    replacing the duplicated _slice_forcings() logic in all runners.

    Attributes:
        _forcings: Raw forcing data (lazy xarray or in-memory arrays).
        n_timesteps: Total number of simulation timesteps.
        interp_method: Interpolation method ("constant", "nearest", "linear", "ffill").
        fill_nan: Value to replace NaN with.
        _dynamic_forcings: Set of forcing names that have a time dimension.
        _time_coords: Target datetime coordinates for interpolation (from TimeGrid).
    """

    _forcings: dict[str, xr.DataArray | Array] = field(default_factory=dict)
    n_timesteps: int = 1
    interp_method: str = "constant"
    fill_nan: float = 0.0
    _dynamic_forcings: set[str] = field(default_factory=set)
    _time_coords: np.ndarray | None = None

    def get_chunk(self, start: int, end: int) -> dict[str, Array]:
        """Load and interpolate a temporal chunk [start, end).

        For each forcing:
        - Static (no time dim): broadcast to chunk length
        - Dynamic, aligned (shape[0] == n_timesteps): slice directly
        - Dynamic, needs interpolation: load source window, interpolate

        Args:
            start: Start timestep (inclusive).
            end: End timestep (exclusive).

        Returns:
            Dict of forcing arrays, all with shape (chunk_len, ...).
        """
        chunk_len = end - start
        result: dict[str, Array] = {}

        for name, data in self._forcings.items():
            is_dynamic = name in self._dynamic_forcings
            arr = self._load_single(name, data, start, end, is_dynamic)
            arr = jnp.asarray(arr)

            if not is_dynamic:
                arr = jnp.broadcast_to(arr, (chunk_len,) + arr.shape)

            result[name] = arr

        return result

    def get_all(self) -> dict[str, Array]:
        """Materialize all forcings for the full simulation.

        Used by GradientRunner where the full time series must be
        in memory for autodiff.

        Returns:
            Dict of forcing arrays with full time dimension.
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
        if name not in self._forcings:
            return default

        data = self._forcings[name]
        if isinstance(data, xr.DataArray):
            return jnp.asarray(data.values)
        return data

    def __contains__(self, name: str) -> bool:
        return name in self._forcings

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def _load_single(
        self,
        name: str,
        data: xr.DataArray | Array,
        start: int,
        end: int,
        is_dynamic: bool,
    ) -> Array:
        """Load a single forcing for the given chunk range.

        Args:
            name: Forcing name.
            data: Raw forcing (lazy xr.DataArray or in-memory array).
            start: Start timestep (inclusive).
            end: End timestep (exclusive).
            is_dynamic: Whether this forcing has a time dimension.

        Returns:
            Array sliced/interpolated for the chunk.
        """
        # --- In-memory array ---
        if not isinstance(data, xr.DataArray):
            arr = np.asarray(data)
            return arr[start:end] if is_dynamic else arr

        # --- Lazy xr.DataArray — static ---
        if not is_dynamic:
            return self._materialize_with_nan(data.values)

        # --- Lazy xr.DataArray — dynamic ---
        source_len = data.sizes["T"]

        if source_len == self.n_timesteps or self.interp_method == "constant":
            chunk = data.isel(T=slice(start, end)).values
            return self._materialize_with_nan(chunk)

        # Interpolation needed
        target_times = self._time_coords[start:end]
        chunk = self._xarray_interpolate(data, target_times)
        return self._materialize_with_nan(chunk)

    def _xarray_interpolate(self, data: xr.DataArray, target_times: np.ndarray) -> np.ndarray:
        """Interpolate a lazy DataArray to target times via xarray."""
        if self.interp_method == "linear":
            return data.interp(T=target_times, method="linear",
                               kwargs={"fill_value": "extrapolate"}).values
        if self.interp_method in ("nearest", "ffill"):
            return data.reindex(T=target_times, method=self.interp_method).values
        raise ValueError(f"Unknown interp_method: {self.interp_method}")

    def _materialize_with_nan(self, arr: Array) -> Array:
        """Materialize array and replace NaN with fill_nan."""
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                arr = np.where(nan_mask, self.fill_nan, arr)
        return arr
