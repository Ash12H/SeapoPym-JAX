"""I/O for streaming output.

Provides DiskWriter for writing simulation outputs synchronously to Zarr stores.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from seapopym.compiler import CompiledModel


from .exceptions import EngineIOError

logger = logging.getLogger(__name__)

from seapopym.types import Array


class OutputWriter(Protocol):
    """Interface for simulation output writers."""

    def initialize(
        self,
        shapes: dict[str, int],
        variables: list[str],
        coords: dict[str, Array] | None = None,
        var_dims: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Initialize the writer.

        Args:
            shapes: Dimension sizes.
            variables: List of variable names to write/keep.
            coords: Coordinate arrays for dimensions (includes real accumulated timestamps).
            var_dims: Mapping from variable name to its dimension names (e.g. {"biomass": ("Y", "X")}).
        """
        ...

    def append(self, data: dict[str, Array], chunk_index: int) -> None:
        """Append a chunk of data.

        Args:
            data: Dictionary of arrays for the chunk.
            chunk_index: Index of the current chunk.
        """
        ...

    def finalize(self) -> Any:
        """Finalize writing and return result (if any)."""
        ...

    def close(self) -> None:
        """Close resources."""
        ...


class DiskWriter:
    """Synchronous writer for simulation outputs to a Zarr store.

    Example:
        >>> writer = DiskWriter(output_path="/results/sim/")
        >>> writer.append({"biomass": arr}, chunk_index=0)
        >>> writer.append({"biomass": arr}, chunk_index=1)
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

    def initialize(
        self,
        shapes: dict[str, int],
        variables: list[str],
        coords: dict[str, Array] | None = None,
        var_dims: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Initialize output storage structure.

        Args:
            shapes: Dimension sizes.
            variables: List of variable names to write.
            coords: Coordinate arrays for dimensions.
            var_dims: Mapping from variable name to its dimension names.
        """
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
                ds = self.store.create_dataset(
                    dim_name,
                    shape=coord_np.shape,
                    dtype=coord_np.dtype,
                )
                ds[:] = coord_np

        # Create arrays for each variable
        # Time dimension is unlimited (append along axis 0)
        for var_name in variables:
            # Use per-variable dims if provided, otherwise fall back to spatial dims
            if var_dims and var_name in var_dims:
                dims = var_dims[var_name]
                # Spatial dims only (exclude T — it's the append axis)
                spatial_dims = tuple(d for d in dims if d != "T")
            else:
                spatial_dims = tuple(d for d in ["Y", "X"] if d in shapes)

            var_shape = (0,) + tuple(shapes.get(d, 1) for d in spatial_dims)
            chunks = (1,) + var_shape[1:]

            ds = self.store.create_dataset(
                var_name,
                shape=var_shape,
                chunks=chunks,
                dtype=np.float32,
            )
            # xarray/zarr convention: store dimension names as attribute
            ds.attrs["_ARRAY_DIMENSIONS"] = ["T", *spatial_dims]

    def append(
        self,
        data: dict[str, Array],
        chunk_index: int,
    ) -> None:
        """Write a chunk of data synchronously.

        Args:
            data: Dict mapping variable names to arrays.
            chunk_index: Chunk index (for logging).
        """
        if not self._initialized:
            raise EngineIOError(str(self.output_path), "Writer not initialized")

        # Convert JAX arrays to numpy
        data_np = {k: np.asarray(v) for k, v in data.items()}
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

    def __init__(self, model: CompiledModel) -> None:
        """Initialize memory writer.

        Args:
            model: Compiled model (needed for metadata: graph, coords, etc.)
        """
        self.model = model
        self.variables: list[str] = []
        self._accumulator: dict[str, list[np.ndarray]] = {}
        self._coords: dict[str, Array] = {}

    def initialize(
        self,
        shapes: dict[str, int],
        variables: list[str],
        coords: dict[str, Array] | None = None,
        var_dims: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Initialize accumulator.

        Args:
            shapes: Dimension sizes (unused, kept for protocol compatibility).
            variables: List of variable names to accumulate.
            coords: Coordinate arrays for dimensions (includes accumulated timestamps).
            var_dims: Mapping from variable name to its dims (unused, kept for protocol compatibility).
        """
        del shapes, var_dims  # Unused, kept for protocol compatibility
        self.variables = variables
        for var in variables:
            self._accumulator[var] = []

        # Store coords (fallback to model coords if not provided)
        if coords is not None:
            self._coords = {k: np.asarray(v) for k, v in coords.items()}
        else:
            self._coords = {k: np.asarray(v) for k, v in self.model.coords.items()}

    def append(self, data: dict[str, Array], chunk_index: int) -> None:  # noqa: ARG002
        """Append chunk to memory."""
        # We accumulate specific variables
        for var_name in self.variables:
            if var_name in data:
                # Ensure numpy array
                arr = np.asarray(data[var_name])
                self._accumulator[var_name].append(arr)

    def finalize(self) -> xr.Dataset | None:
        """Construct and return the xarray Dataset."""
        if not self.variables:
            return None

        import xarray as xr

        from seapopym.dims import get_canonical_order

        # 1. Concatenate arrays along time axis
        merged_data = {}
        for var_name, chunks in self._accumulator.items():
            if not chunks:
                continue
            merged_data[var_name] = np.concatenate(chunks, axis=0)

        # 2. Resolve dimensions from graph
        var_dims = self._resolve_variable_dims()

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
                dims = ("T",) + dims
            elif len(dims) != data.ndim:
                # Dimension mismatch - skip this variable
                continue

            # Apply canonical order (preserve non-canonical dims)
            canonical = get_canonical_order(dims)
            extras = tuple(d for d in dims if d not in canonical)
            dims = canonical + extras

            data_vars[var_name] = (dims, data)

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def close(self) -> None:
        """Release resources (no-op for memory writer)."""
        pass

    def _resolve_variable_dims(self) -> dict[str, tuple[str, ...]]:
        """Resolve dimensions for all requested variables using data_nodes."""
        resolved = {}

        for node in self.model.data_nodes.values():
            if node.dims is None:
                continue

            # Check if node name matches any requested variable
            # Handle both short names ("biomass") and fully qualified names ("state.biomass")
            node_short_name = node.name.split(".")[-1] if "." in node.name else node.name

            if node.name in self.variables:
                resolved[node.name] = tuple(d for d in node.dims)
            elif node_short_name in self.variables:
                resolved[node_short_name] = tuple(d for d in node.dims)

        return resolved
