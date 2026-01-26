"""Asynchronous I/O for streaming output.

Provides AsyncWriter for writing simulation outputs in parallel with computation.
Uses ThreadPoolExecutor since GIL is released during JAX execution.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions import EngineIOError

logger = logging.getLogger(__name__)

# Type alias
Array = Any  # np.ndarray | jax.Array


class AsyncWriter:
    """Asynchronous writer for simulation outputs.

    Uses a thread pool to write chunks in parallel with computation.
    The GIL is released during JAX execution, allowing true parallelism.

    Example:
        >>> writer = AsyncWriter(output_path="/results/sim/", max_workers=2)
        >>> writer.write_async({"biomass": arr}, chunk_id=0)
        >>> writer.write_async({"biomass": arr}, chunk_id=1)
        >>> writer.flush()  # Wait for all writes to complete
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

    def write_async(
        self,
        data: dict[str, Array],
        chunk_id: int,
    ) -> None:
        """Submit a chunk for asynchronous writing.

        Args:
            data: Dict mapping variable names to arrays.
            chunk_id: Chunk index (for ordering).
        """
        if not self._initialized:
            raise EngineIOError(str(self.output_path), "Writer not initialized")

        # Convert JAX arrays to numpy before submitting
        data_np = {k: np.asarray(v) for k, v in data.items()}

        future = self.executor.submit(self._write_chunk, data_np, chunk_id)
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
                if var_name in self.store:
                    # Append along time axis
                    existing = self.store[var_name]
                    if isinstance(existing, zarr.Array):
                        new_shape = (existing.shape[0] + arr.shape[0],) + existing.shape[1:]
                        existing.resize(new_shape)
                        existing[-arr.shape[0] :] = arr

    def flush(self) -> None:
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
        self.flush()
        self.executor.shutdown(wait=True)

    def __enter__(self) -> AsyncWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
