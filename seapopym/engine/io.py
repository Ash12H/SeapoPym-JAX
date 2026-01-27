"""Asynchronous I/O for streaming output.

Provides AsyncWriter for writing simulation outputs in parallel with computation.
Uses ThreadPoolExecutor since GIL is released during JAX execution.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from seapopym.compiler import CompiledModel


from .exceptions import EngineIOError

logger = logging.getLogger(__name__)

# Type alias
Array = Any  # np.ndarray | jax.Array


class OutputWriter(Protocol):
    """Interface for simulation output writers."""

    def initialize(self, shapes: dict[str, int], variables: list[str]) -> None:
        """Initialize the writer.

        Args:
            shapes: Dimension sizes.
            variables: List of variable names to write/keep.
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
    """Asynchronous writer for simulation outputs.

    Uses a thread pool to write chunks in parallel with computation.
    The GIL is released during JAX execution, allowing true parallelism.

    Example:
        >>> writer = DiskWriter(output_path="/results/sim/", max_workers=2)
        >>> writer.append({"biomass": arr}, chunk_index=0)
        >>> writer.append({"biomass": arr}, chunk_index=1)
        >>> writer.finalize()  # Wait for all writes to complete
    """

    def __init__(
        self,
        output_path: str | Path,
        max_workers: int = 2,
        format: str = "zarr",
    ) -> None:
        """Initialize async writer.

        Args:
            output_path: Base path for output files.
            max_workers: Number of writer threads.
            format: Output format ("zarr" or "netcdf").
        """
        self.output_path = Path(output_path)
        self.format = format
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures: list[Future[None]] = []
        self._initialized = False
        self.store: Any = None  # zarr.Group at runtime
        self._write_lock = threading.Lock()  # Thread safety for zarr writes

    def initialize(self, shapes: dict[str, int], variables: list[str]) -> None:
        """Initialize output storage structure.

        Args:
            shapes: Dimension sizes.
            variables: List of variable names to write.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.format == "zarr":
            self._init_zarr(shapes, variables)
        elif self.format == "netcdf":
            self._init_netcdf(shapes, variables)
        else:
            raise EngineIOError(str(self.output_path), f"Unknown format: {self.format}")

        self._initialized = True

    def _init_zarr(self, shapes: dict[str, int], variables: list[str]) -> None:
        """Initialize Zarr store."""
        import zarr

        self.store = zarr.open(str(self.output_path), mode="w")

        # Create arrays for each variable
        # Time dimension is unlimited (append along axis 0)
        for var_name in variables:
            # Determine shape based on variable (simplified)
            # In practice, would use blueprint declarations
            var_shape = (0,) + tuple(shapes.get(d, 1) for d in ["Y", "X"] if d in shapes)
            chunks = (1,) + var_shape[1:]

            self.store.create_dataset(
                var_name,
                shape=var_shape,
                chunks=chunks,
                dtype=np.float32,
            )

    def _init_netcdf(self, _shapes: dict[str, int], _variables: list[str]) -> None:
        """Initialize NetCDF file with unlimited time dimension."""
        # For simplicity, we'll use zarr format primarily
        # NetCDF support can be added later
        raise EngineIOError(str(self.output_path), "NetCDF format not yet implemented")

    def append(
        self,
        data: dict[str, Array],
        chunk_index: int,
    ) -> None:
        """Submit a chunk for asynchronous writing.

        Args:
            data: Dict mapping variable names to arrays.
            chunk_index: Chunk index (for ordering).
        """
        if not self._initialized:
            raise EngineIOError(str(self.output_path), "Writer not initialized")

        # Convert JAX arrays to numpy before submitting
        data_np = {k: np.asarray(v) for k, v in data.items()}

        future = self.executor.submit(self._write_chunk, data_np, chunk_index)
        self.futures.append(future)

    def _write_chunk(self, data: dict[str, np.ndarray], chunk_id: int) -> None:
        """Write a chunk to storage (runs in thread).

        Args:
            data: Dict mapping variable names to numpy arrays.
            chunk_id: Chunk index.
        """
        try:
            if self.format == "zarr":
                self._write_zarr_chunk(data, chunk_id)
            else:
                raise EngineIOError(str(self.output_path), f"Unknown format: {self.format}")
        except Exception as e:
            logger.error(f"Failed to write chunk {chunk_id}: {e}")
            raise EngineIOError(str(self.output_path), str(e)) from e

    def _write_zarr_chunk(self, data: dict[str, np.ndarray], _chunk_id: int) -> None:
        """Write chunk to Zarr store."""
        import zarr

        if self.store is None:
            raise EngineIOError(str(self.output_path), "Store not initialized")

        # Use lock to ensure thread safety when resizing and writing
        with self._write_lock:
            for var_name, arr in data.items():
                if var_name in self.store and isinstance(existing := self.store[var_name], zarr.Array):
                    # Append along time axis
                    new_shape = (existing.shape[0] + arr.shape[0],) + existing.shape[1:]
                    existing.resize(new_shape)
                    existing[-arr.shape[0] :] = arr

    def finalize(self) -> None:
        """Wait for all pending writes to complete."""
        errors = []
        for future in self.futures:
            try:
                future.result()
            except Exception as e:
                errors.append(e)

        self.futures.clear()

        if errors:
            raise EngineIOError(
                str(self.output_path),
                f"{len(errors)} write(s) failed. First error: {errors[0]}",
            )

    def close(self) -> None:
        """Close the writer and release resources."""
        self.finalize()
        self.executor.shutdown(wait=True)

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

    def initialize(self, shapes: dict[str, int], variables: list[str]) -> None:  # noqa: ARG002
        """Initialize accumulator."""
        self.variables = variables
        for var in variables:
            self._accumulator[var] = []

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

        # 1. Contatenate arrays along time axis
        merged_data = {}
        for var_name, chunks in self._accumulator.items():
            if not chunks:
                continue
            merged_data[var_name] = np.concatenate(chunks, axis=0)

        # 2. Resolve coordinates and dimensions
        coords = {k: np.asarray(v) for k, v in self.model.coords.items()}

        # We need to map variable names to their dimensions
        # This requires searching the graph for the DataNode
        var_dims = self._resolve_variable_dims()

        data_vars = {}
        for var_name, data in merged_data.items():
            dims = var_dims.get(var_name, None)

            # Fallback if dims not found or rank mismatch
            if dims is None or len(dims) != data.ndim:
                # Try to guess standard dims based on rank if possible,
                # otherwise use default names dim_0, dim_1...
                # But typically valid variables should be in graph.
                # Standard JAX output is (T, ...)
                pass

            data_vars[var_name] = (dims, data)

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def close(self) -> None:
        """Release resources (no-op for memory writer)."""
        pass

    def _resolve_variable_dims(self) -> dict[str, tuple[str, ...]]:
        """Resolve dimensions for all requested variables using the graph."""
        from seapopym.blueprint.nodes import DataNode

        # Map var_name -> dims tuple
        resolved = {}

        # We scan graph nodes. This is fast enough for initialization.
        for node in self.model.graph.nodes:
            if isinstance(node, DataNode) and node.name in self.variables and node.dims is not None:
                resolved[node.name] = tuple(d for d in node.dims)

        return resolved


# Alias for backward compatibility
AsyncWriter = DiskWriter
