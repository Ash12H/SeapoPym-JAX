"""ForcingStore: lazy forcing loading with JAX-native interpolation.

Encapsulates all forcings and provides chunked access with on-the-fly
temporal interpolation using jax.vmap(jnp.interp) instead of scipy.

Key features:
- Lazy loading: xr.DataArray forcings are NOT materialized at compile time
- JAX-native interpolation: differentiable, JIT-compilable, no scipy dependency
- Unified chunking API: get_chunk() replaces _slice_forcings() in all runners
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import xarray as xr

from seapopym.types import Array


def compute_source_window(
    source_len: int,
    target_len: int,
    start: int,
    end: int,
    method: str,
) -> tuple[int, int]:
    """Compute the source index window needed to interpolate a target chunk.

    Works in index space: target_indices = linspace(0, source_len-1, target_len).
    For the target chunk [start, end), determines which source indices are needed.

    Args:
        source_len: Number of source timesteps.
        target_len: Number of target timesteps.
        start: Start of target chunk (inclusive).
        end: End of target chunk (exclusive).
        method: Interpolation method ("nearest", "linear", "ffill").

    Returns:
        Tuple of (src_start, src_end) — inclusive start, exclusive end.
    """
    if source_len <= 1 or target_len <= 1:
        return 0, source_len

    target_indices = np.linspace(0, source_len - 1, target_len)
    chunk_indices = target_indices[start:end]

    min_idx = chunk_indices[0]
    max_idx = chunk_indices[-1]

    if method == "nearest":
        src_start = int(np.clip(np.round(min_idx), 0, source_len - 1))
        src_end = int(np.clip(np.round(max_idx), 0, source_len - 1)) + 1
    elif method == "linear":
        src_start = int(np.clip(np.floor(min_idx), 0, source_len - 1))
        src_end = int(np.clip(np.ceil(max_idx), 0, source_len - 1)) + 1
    elif method == "ffill":
        src_start = int(np.clip(np.floor(min_idx), 0, source_len - 1))
        src_end = int(np.clip(np.floor(max_idx), 0, source_len - 1)) + 1
    else:
        src_start = 0
        src_end = source_len

    return src_start, src_end


def interpolate_chunk(
    source_chunk: Array,
    source_indices: Array,
    target_indices: Array,
    method: str,
    backend: str = "jax",
) -> Array:
    """Interpolate a source chunk to target indices.

    Replaces scipy.interpolate.interp1d with backend-native operations.
    For JAX: uses jax.vmap(jnp.interp), which is differentiable and JIT-compilable.
    For NumPy: uses np.interp with vectorization.

    Args:
        source_chunk: Source data, shape (S, ...).
        source_indices: Source positions, shape (S,).
        target_indices: Target positions for this chunk, shape (C,).
        method: Interpolation method ("nearest", "linear", "ffill").
        backend: "jax" or "numpy".

    Returns:
        Interpolated data, shape (C, ...).
    """
    if method == "nearest":
        nearest_idx = np.round(target_indices).astype(int)
        offset = int(source_indices[0])
        nearest_idx = np.clip(nearest_idx - offset, 0, len(source_indices) - 1)
        return source_chunk[nearest_idx]

    if method == "ffill":
        ffill_idx = np.floor(target_indices).astype(int)
        offset = int(source_indices[0])
        ffill_idx = np.clip(ffill_idx - offset, 0, len(source_indices) - 1)
        return source_chunk[ffill_idx]

    # method == "linear"
    if backend == "jax":
        import jax
        import jax.numpy as jnp

        src_idx = jnp.asarray(source_indices)
        tgt_idx = jnp.asarray(target_indices)
        chunk = jnp.asarray(source_chunk)

        original_shape = chunk.shape[1:]
        flat = chunk.reshape(chunk.shape[0], -1)  # (S, N)

        result_flat = jax.vmap(
            lambda fp: jnp.interp(tgt_idx, src_idx, fp),
            in_axes=1,
            out_axes=1,
        )(flat)  # (C, N)

        return result_flat.reshape((len(target_indices),) + original_shape)
    else:
        # NumPy: vectorize np.interp over spatial dims
        src_idx = np.asarray(source_indices)
        tgt_idx = np.asarray(target_indices)
        chunk = np.asarray(source_chunk)

        original_shape = chunk.shape[1:]
        flat = chunk.reshape(chunk.shape[0], -1)  # (S, N)

        n_spatial = flat.shape[1]
        result_flat = np.empty((len(tgt_idx), n_spatial), dtype=flat.dtype)
        for i in range(n_spatial):
            result_flat[:, i] = np.interp(tgt_idx, src_idx, flat[:, i])

        return result_flat.reshape((len(tgt_idx),) + original_shape)


def _interpolate_full(
    source: Array,
    source_len: int,
    target_len: int,
    method: str,
    backend: str = "jax",
) -> Array:
    """Interpolate an entire forcing from source_len to target_len timesteps.

    Convenience wrapper around interpolate_chunk for eager (compile-time) use.

    Args:
        source: Source array, shape (source_len, ...).
        source_len: Number of source timesteps.
        target_len: Number of target timesteps.
        method: Interpolation method.
        backend: "jax" or "numpy".

    Returns:
        Interpolated array, shape (target_len, ...).
    """
    source_indices = np.arange(source_len, dtype=np.float64)
    target_indices = np.linspace(0, source_len - 1, target_len)
    return interpolate_chunk(source, source_indices, target_indices, method, backend)


@dataclass
class ForcingStore:
    """Encapsulates all forcings with lazy loading and chunked access.

    Forcings can be either:
    - xr.DataArray (lazy): loaded from file, materialized only when accessed
    - Array (in-memory): already materialized numpy/jax arrays

    The store exposes a unified API for chunking and interpolation,
    replacing the duplicated _slice_forcings() logic in all runners.

    Attributes:
        _forcings: Raw forcing data (lazy xarray or in-memory arrays).
        n_timesteps: Total number of simulation timesteps.
        interp_method: Interpolation method ("constant", "nearest", "linear", "ffill").
        backend: Target backend ("jax" or "numpy").
        fill_nan: Value to replace NaN with.
        _dynamic_forcings: Set of forcing names that have a time dimension.
    """

    _forcings: dict[str, xr.DataArray | Array] = field(default_factory=dict)
    n_timesteps: int = 1
    interp_method: str = "constant"
    backend: Literal["jax", "numpy"] = "jax"
    fill_nan: float = 0.0
    _dynamic_forcings: set[str] = field(default_factory=set)

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
            arr = self._to_backend(arr)

            if not is_dynamic:
                arr = self._broadcast_static(arr, chunk_len)

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
            return self._to_backend(data.values)
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
        return self._interpolate_lazy(name, data, source_len, start, end)

    def _interpolate_lazy(
        self,
        name: str,
        data: xr.DataArray,
        source_len: int,
        start: int,
        end: int,
    ) -> Array:
        """Load source window from lazy DataArray and interpolate."""
        # Compute source window
        src_start, src_end = compute_source_window(
            source_len, self.n_timesteps, start, end, self.interp_method
        )

        # Load only the needed window
        chunk_data = data.isel(T=slice(src_start, src_end)).values

        # Preprocess NaN
        nan_mask = np.isnan(chunk_data)
        if nan_mask.any():
            chunk_data = np.where(nan_mask, self.fill_nan, chunk_data)

        # Compute index mappings
        source_indices = np.arange(src_start, src_end, dtype=np.float64)
        all_target_indices = np.linspace(0, source_len - 1, self.n_timesteps)
        target_indices = all_target_indices[start:end]

        return interpolate_chunk(
            chunk_data, source_indices, target_indices, self.interp_method, self.backend
        )

    def _materialize_with_nan(self, arr: Array) -> Array:
        """Materialize array and replace NaN with fill_nan."""
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                arr = np.where(nan_mask, self.fill_nan, arr)
        return arr

    def _to_backend(self, arr: Array) -> Array:
        """Convert array to target backend."""
        if self.backend == "jax":
            import jax.numpy as jnp

            return jnp.asarray(arr)
        return np.asarray(arr)

    def _broadcast_static(self, arr: Array, chunk_len: int) -> Array:
        """Broadcast a static forcing to chunk length."""
        if self.backend == "jax":
            import jax.numpy as jnp

            return jnp.broadcast_to(arr, (chunk_len,) + arr.shape)
        return np.broadcast_to(arr, (chunk_len,) + arr.shape)
