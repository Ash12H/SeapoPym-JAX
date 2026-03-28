"""I/O for streaming output.

Provides writers for simulation outputs:
- WriterRaw: JAX-traceable writer returning raw arrays (optimization, vmap, grad).
- MemoryWriter: In-memory writer returning xarray Dataset.
- DiskWriter: Synchronous writer to Zarr stores.
- build_writer: Factory function to create and initialize the appropriate writer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import jax
import numpy as np

from seapopym.types import Array

from .exceptions import EngineIOError

if TYPE_CHECKING:
    import xarray as xr

    from seapopym.blueprint.nodes import DataNode
    from seapopym.compiler import CompiledModel

logger = logging.getLogger(__name__)


def resolve_var_dims(
    data_nodes: dict[str, DataNode],
    variables: list[str],
) -> dict[str, tuple[str, ...]]:
    """Resolve dimensions for variables using data_nodes.

    Args:
        data_nodes: Mapping of node names to DataNode objects.
        variables: List of variable names to resolve.

    Returns:
        Mapping from variable name to its dimension tuple.
    """
    resolved: dict[str, tuple[str, ...]] = {}

    for node in data_nodes.values():
        if node.dims is None:
            continue

        node_short_name = node.name.split(".")[-1] if "." in node.name else node.name

        if node.name in variables:
            resolved[node.name] = tuple(node.dims)
        elif node_short_name in variables:
            resolved[node_short_name] = tuple(node.dims)

    return resolved


class OutputWriter(Protocol):
    """Interface for simulation output writers."""

    def initialize(
        self,
        shapes: dict[str, int],
        variables: list[str],
        coords: dict[str, Array] | None = None,
        var_dims: dict[str, tuple[str, ...]] | None = None,
        time_dim: str = "T",
    ) -> None:
        """Initialize the writer.

        Args:
            shapes: Dimension sizes.
            variables: List of variable names to write/keep.
            coords: Coordinate arrays for dimensions (includes real accumulated timestamps).
            var_dims: Mapping from variable name to its dimension names (e.g. {"biomass": ("Y", "X")}).
        """
        ...

    def append(self, data: dict[str, Array]) -> None:
        """Append a chunk of data.

        Args:
            data: Dictionary of arrays for the chunk.
        """
        ...

    def finalize(self) -> Any:
        """Finalize writing and return result (if any)."""
        ...

    def close(self) -> None:
        """Close resources."""
        ...


class WriterRaw:
    """JAX-traceable writer that accumulates raw arrays.

    Stores chunks in a Python list and concatenates on finalize.
    Because list.append is a trace-time operation, this writer is
    compatible with ``jax.vmap`` and ``jax.grad``.

    No ``initialize()`` call is needed — append/finalize/close suffice.
    """

    def __init__(self) -> None:
        self._chunks: list[dict[str, Array]] = []

    def initialize(
        self,
        shapes: dict[str, int],
        variables: list[str],
        coords: dict[str, Array] | None = None,
        var_dims: dict[str, tuple[str, ...]] | None = None,
        time_dim: str = "T",
    ) -> None:
        """No-op (kept for Protocol compatibility)."""

    def append(self, data: dict[str, Array]) -> None:
        """Store a chunk of arrays."""
        self._chunks.append(data)

    def finalize(self) -> dict[str, Array]:
        """Concatenate all chunks along axis 0."""
        import jax.numpy as jnp

        if not self._chunks:
            return {}
        if len(self._chunks) == 1:
            return self._chunks[0]
        return jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *self._chunks)

    def close(self) -> None:
        """No-op."""


class DiskWriter:
    """Synchronous writer for simulation outputs to a Zarr store.

    Example:
        >>> writer = DiskWriter(output_path="/results/sim/")
        >>> writer.append({"biomass": arr})
        >>> writer.append({"biomass": arr})
        >>> writer.finalize()
    """

    def __init__(self, output_path: str | Path) -> None:
        """Initialize writer.

        Args:
            output_path: Base path for output files.
        """
        self.output_path = Path(output_path)
        self._initialized = False
        self.store: Any = None  # zarr.Group at runtime
        self._time_dim: str = "T"

    def initialize(
        self,
        shapes: dict[str, int],
        variables: list[str],
        coords: dict[str, Array] | None = None,
        var_dims: dict[str, tuple[str, ...]] | None = None,
        time_dim: str = "T",
    ) -> None:
        """Initialize output storage structure.

        Args:
            shapes: Dimension sizes.
            variables: List of variable names to write.
            coords: Coordinate arrays for dimensions.
            var_dims: Mapping from variable name to its dimension names.
        """
        self._time_dim = time_dim
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._init_zarr(shapes, variables, coords=coords, var_dims=var_dims)
        self._initialized = True

    def _init_zarr(
        self,
        shapes: dict[str, int],
        variables: list[str],
        coords: dict[str, Array] | None = None,
        var_dims: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Initialize Zarr store with coordinate metadata."""
        import zarr

        self.store = zarr.open(str(self.output_path), mode="w")

        # Write coordinate arrays
        if coords is not None:
            for dim_name, coord_arr in coords.items():
                coord_np = np.asarray(coord_arr)
                ds = self.store.create_array(
                    dim_name,
                    shape=coord_np.shape,
                    dtype=coord_np.dtype,
                )
                ds[:] = coord_np

        # Create arrays for each variable
        # Time dimension is unlimited (append along axis 0)
        for var_name in variables:
            if not var_dims or var_name not in var_dims:
                continue
            dims = var_dims[var_name]
            # Non-time dims only (time is the append axis)
            spatial_dims = tuple(d for d in dims if d != self._time_dim)

            var_shape = (0,) + tuple(shapes.get(d, 1) for d in spatial_dims)
            chunks = (1,) + var_shape[1:]

            ds = self.store.create_array(
                var_name,
                shape=var_shape,
                chunks=chunks,
                dtype=np.float32,
            )
            # xarray/zarr convention: store dimension names as attribute
            ds.attrs["_ARRAY_DIMENSIONS"] = [self._time_dim, *spatial_dims]

    def append(
        self,
        data: dict[str, Array],
    ) -> None:
        """Write a chunk of data synchronously.

        Args:
            data: Dict mapping variable names to arrays.
        """
        if not self._initialized:
            raise EngineIOError(str(self.output_path), "Writer not initialized")

        # Transfer JAX arrays to host (numpy)
        data_np = {k: jax.device_get(v) for k, v in data.items()}
        self._write_zarr_chunk(data_np)

    def _write_zarr_chunk(self, data: dict[str, np.ndarray]) -> None:
        """Write chunk to Zarr store."""
        import zarr

        if self.store is None:
            raise EngineIOError(str(self.output_path), "Store not initialized")

        for var_name, arr in data.items():
            if var_name in self.store and isinstance(existing := self.store[var_name], zarr.Array):
                # Append along time axis
                new_shape = (existing.shape[0] + arr.shape[0],) + existing.shape[1:]
                existing.resize(new_shape)
                existing[-arr.shape[0] :] = arr

    def finalize(self) -> None:
        """Finalize writing (no-op, kept for Protocol compatibility)."""

    def close(self) -> None:
        """Close the writer (no-op, kept for Protocol compatibility)."""

    def __enter__(self) -> DiskWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


class MemoryWriter:
    """In-memory writer that builds an xarray Dataset from chunks."""

    def __init__(self) -> None:
        """Initialize memory writer."""
        self.variables: list[str] = []
        self._accumulator: dict[str, list[Array]] = {}
        self._coords: dict[str, Array] = {}
        self._var_dims: dict[str, tuple[str, ...]] = {}
        self._time_dim: str = "T"

    def initialize(
        self,
        shapes: dict[str, int],
        variables: list[str],
        coords: dict[str, Array] | None = None,
        var_dims: dict[str, tuple[str, ...]] | None = None,
        time_dim: str = "T",
    ) -> None:
        """Initialize accumulator.

        Args:
            shapes: Dimension sizes (unused, kept for protocol compatibility).
            variables: List of variable names to accumulate.
            coords: Coordinate arrays for dimensions (includes accumulated timestamps).
            var_dims: Mapping from variable name to its dims.
            time_dim: Name of the time dimension.
        """
        del shapes  # Unused, kept for protocol compatibility
        self._time_dim = time_dim
        self.variables = variables
        for var in variables:
            self._accumulator[var] = []

        if var_dims is not None:
            self._var_dims = var_dims

        if coords is not None:
            self._coords = {k: np.asarray(v) for k, v in coords.items()}

    def append(self, data: dict[str, Array]) -> None:
        """Append chunk to memory (keeps raw JAX arrays until finalize)."""
        for var_name in self.variables:
            if var_name in data:
                self._accumulator[var_name].append(data[var_name])

    def finalize(self) -> xr.Dataset | None:
        """Construct and return the xarray Dataset."""
        if not self.variables:
            return None

        import jax.numpy as jnp
        import xarray as xr

        from seapopym.dims import get_canonical_order

        # 1. Concatenate arrays along time axis, then convert to numpy once
        merged_data = {}
        for var_name, chunks in self._accumulator.items():
            if not chunks:
                continue
            if len(chunks) == 1:
                merged_data[var_name] = np.asarray(chunks[0])
            else:
                merged_data[var_name] = np.asarray(jnp.concatenate(chunks, axis=0))

        # 2. Use stored var_dims
        var_dims = self._var_dims

        # 3. Use stored coords (includes real accumulated timestamps)
        coords = self._coords

        data_vars = {}
        for var_name, data in merged_data.items():
            dims = var_dims.get(var_name, None)

            # Fallback if dims not found
            if dims is None:
                continue

            # Add time dimension if needed (accumulated data has extra dimension)
            if len(dims) + 1 == data.ndim:
                dims = (self._time_dim,) + dims
            elif len(dims) != data.ndim:
                # Dimension mismatch - skip this variable
                continue

            # Apply canonical order (non-canonical dims preserved at end)
            dims = get_canonical_order(dims)

            data_vars[var_name] = (dims, data)

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def close(self) -> None:
        """Release resources (no-op for memory writer)."""
        pass


def build_writer(
    model: CompiledModel,
    output_path: str | Path | None,
    export_variables: list[str],
) -> OutputWriter:
    """Build and initialize the appropriate output writer.

    Args:
        model: Compiled model (provides shapes, coords, data_nodes, time_grid).
        output_path: If not None, creates a DiskWriter; otherwise MemoryWriter.
        export_variables: Variables to export.

    Returns:
        Initialized OutputWriter ready for append/finalize/close.
    """
    time_dim = model.time_dim
    n_timesteps = model.n_timesteps
    writer_coords = dict(model.coords)
    if model.time_grid is not None:
        writer_coords[time_dim] = model.time_grid.coords[:n_timesteps]
    var_dims = resolve_var_dims(model.data_nodes, export_variables)

    writer: OutputWriter
    writer = DiskWriter(output_path) if output_path is not None else MemoryWriter()
    writer.initialize(model.shapes, export_variables, coords=writer_coords, var_dims=var_dims, time_dim=time_dim)
    return writer
