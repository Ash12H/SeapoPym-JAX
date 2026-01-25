"""Input/Output management for Seapopym."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    pass


class BaseOutputWriter(ABC):
    """Abstract base class for output writers."""

    def __init__(
        self,
        variables: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the writer.

        Args:
            variables: List of variable names to save. If None, save all variables.
            metadata: Dictionary of metadata to add to the output dataset attributes.
        """
        self.variables = variables
        self.metadata = metadata or {}
        self._is_initialized = False

    @abstractmethod
    def _initialize(self, state: xr.Dataset) -> None:
        """Initialize the storage structure based on the first state."""
        ...

    @abstractmethod
    def _write_step(self, state: xr.Dataset) -> None:
        """Write a single timestep to storage."""
        ...

    @abstractmethod
    def finalize(self) -> xr.Dataset:
        """Finalize writing and return the complete dataset."""
        ...

    def get_append_task(self, state: xr.Dataset, time: Any | None = None) -> Any:
        """Create a task to append the state to the output.

        The returned task can be:
        - A dask.delayed object (for ZarrWriter with lazy data)
        - A callable (for MemoryWriter)
        - None (if nothing to do)

        This allows the Controller/Backend to decide WHEN and WHERE to execute the task.
        """
        # Common logic (variable selection, time coord)
        if self.variables is not None:
            state = state[self.variables]

        if time is not None:
            from seapopym.standard.coordinates import Coordinates

            time_coord = Coordinates.T.value
            if time_coord in state.dims:
                state = state.assign_coords({time_coord: [time]})
            else:
                state = state.assign_coords({time_coord: time})

        # Return a callable that executes the write
        # Subclasses can override to return a Delayed object
        def _task() -> None:
            if not self._is_initialized:
                self._initialize(state)
                self._is_initialized = True
            self._write_step(state)

        return _task

    def append(self, state: xr.Dataset, time: Any | None = None) -> None:
        """Append a simulation state to the output (Synchronous).

        Automatically initializes the writer on the first call.
        """
        task = self.get_append_task(state, time)
        if hasattr(task, "compute"):
            task.compute()
        elif callable(task):
            task()


class MemoryWriter(BaseOutputWriter):
    """Stores simulation results in memory (list of Datasets)."""

    def __init__(
        self,
        variables: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize the memory writer.

        Args:
            variables: List of variable names to save. If None, save all variables.
            metadata: Dictionary of metadata to add to the output dataset attributes.
        """
        super().__init__(variables, metadata)
        self._buffer: list[xr.Dataset] = []

    def _initialize(self, state: xr.Dataset) -> None:
        """Nothing specific to initialize for memory storage."""
        pass

    def _write_step(self, state: xr.Dataset) -> None:
        """Store the state in the buffer.

        We make a deep copy to ensure subsequent modifications to 'state'
        in the simulation don't affect stored history.
        """
        self._buffer.append(state.copy(deep=True))

    def finalize(self) -> xr.Dataset:
        """Concatenate all buffered states into a single Dataset."""
        if not self._buffer:
            return xr.Dataset(attrs=self.metadata)

        # Concatenate along time dimension
        # We assume the states have a scalar time coordinate (Coordinates.T)
        # xr.concat will stack them along that coordinate.
        from seapopym.standard.coordinates import Coordinates

        try:
            ds = xr.concat(self._buffer, dim=Coordinates.T.value)
        except Exception:
            # Fallback if Coordinates enum is not used or T is missing
            # We try to find a common time dimension or just 'time'
            ds = xr.concat(self._buffer, dim="time")

        # Update metadata
        ds.attrs.update(self.metadata)
        return ds


class ZarrWriter(BaseOutputWriter):
    """Stores simulation results to a Zarr store on disk."""

    @staticmethod
    def _sanitize_var_names(ds: xr.Dataset) -> tuple[xr.Dataset, dict[str, str]]:
        """Replace slashes in variable names with double underscores for Zarr compatibility.

        Zarr interprets slashes as group hierarchy, which breaks append_dim functionality.
        Returns: (sanitized_dataset, name_mapping dict from sanitized_name -> original_name)
        """
        name_mapping = {}
        rename_dict = {}

        for var_name in ds.data_vars:
            if "/" in var_name:
                sanitized_name = var_name.replace("/", "__")
                name_mapping[sanitized_name] = var_name
                rename_dict[var_name] = sanitized_name

        return ds.rename(rename_dict) if rename_dict else ds, name_mapping

    @staticmethod
    def _restore_var_names(ds: xr.Dataset, name_mapping: dict[str, str]) -> xr.Dataset:
        """Restore original variable names with slashes from sanitized names."""
        return ds.rename(name_mapping) if name_mapping else ds

    def __init__(
        self,
        path: str | Path,
        variables: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        chunks: dict[str, int] | None = None,
    ):
        """Initialize the Zarr writer.

        Args:
            path: Output path for the Zarr store.
            variables: List of variable names to save. If None, save all variables.
            metadata: Dictionary of metadata to add to the output dataset attributes.
            chunks: Chunking strategy for Zarr storage. Defaults to {'time': 1}.
        """
        super().__init__(variables, metadata)
        self.path = Path(path)
        self.chunks = chunks or {"time": 1}
        self._name_mapping: dict[str, str] = {}  # Maps sanitized -> original variable names

    def _initialize(self, state: xr.Dataset, compute: bool = True) -> Any:
        """Create the Zarr store structure."""
        if self.path.exists():
            raise FileExistsError(f"Output path '{self.path}' already exists.")

        self.path.mkdir(parents=True, exist_ok=True)

        from seapopym.standard.coordinates import Coordinates

        # Check for time coordinate
        time_dim = Coordinates.T.value
        if time_dim not in state.coords:
            raise ValueError(f"State is missing time coordinate '{time_dim}'")

        # Ensure time is a dimension
        ds_to_init = state.expand_dims(dim=time_dim) if time_dim not in state.dims else state

        # Sanitize variable names (replace slashes with underscores for Zarr compatibility)
        ds_to_init, self._name_mapping = self._sanitize_var_names(ds_to_init)

        # Initialize Zarr store
        ds_to_init.attrs.update(self.metadata)
        return ds_to_init.to_zarr(self.path, mode="w", safe_chunks=False, compute=compute)

    def _write_step(self, state: xr.Dataset, compute: bool = True) -> Any:
        """Write timestep to Zarr."""
        from seapopym.standard.coordinates import Coordinates

        time_dim = Coordinates.T.value
        ds_to_write = state.expand_dims(dim=time_dim) if time_dim not in state.dims else state

        # Sanitize variable names (must match those used during initialization)
        ds_to_write, _ = self._sanitize_var_names(ds_to_write)

        # When using append_dim, xarray/Zarr automatically handles:
        # - Appending data variables along the append dimension
        # - Validating that static coordinates match those in the existing store
        return ds_to_write.to_zarr(
            self.path, append_dim=time_dim, safe_chunks=False, compute=compute
        )

    def get_append_task(self, state: xr.Dataset, time: Any | None = None) -> Any:
        """Get the dask.delayed task for writing without executing it."""
        # Pre-process state (filtering, coordinates)
        if self.variables is not None:
            state = state[self.variables]

        if time is not None:
            from seapopym.standard.coordinates import Coordinates

            time_coord = Coordinates.T.value
            if time_coord in state.dims:
                state = state.assign_coords({time_coord: [time]})
            else:
                state = state.assign_coords({time_coord: time})

        # Zarr appending does not work well with delayed tasks because:
        # - Multiple delayed append tasks created before execution cause ContainsArrayError
        # - Zarr's append logic expects sequential execution on a single store
        # Solution: Execute all Zarr writes synchronously

        if not self._is_initialized:
            self._is_initialized = True
            self._initialize(state, compute=True)  # Synchronous initialization
        else:
            self._write_step(state, compute=True)  # Synchronous append

        # Return a no-op delayed task since we already executed the write
        import dask

        return dask.delayed(lambda: None)()

    def append(self, state: xr.Dataset, time: Any | None = None) -> None:
        """Sync append override."""
        # Use the base implementation which calls get_append_task().compute()
        # But we need to make sure _initialize/_write_step are called with compute=False internally
        # if called via get_append_task.
        # Actually, base implementation calls get_append_task then .compute().
        # So ZarrWriter.get_append_task returns a Delayed.
        # Base.append calls Delayed.compute(). It works.
        super().append(state, time)

    def finalize(self) -> xr.Dataset:
        """Return the opened Zarr dataset with original variable names restored."""
        ds = xr.open_zarr(self.path)
        # Restore original variable names (with slashes)
        return self._restore_var_names(ds, self._name_mapping)
