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

    def append(self, state: xr.Dataset) -> None:
        """Append a simulation state to the output.

        Automatically initializes the writer on the first call.
        Filters variables if a list was provided.
        """
        # Filter variables
        if self.variables is not None:
            # We only keep variables that exist in the state
            # (to avoid errors if a requested variable is missing, or we could raise)
            # Let's be strict: if user asked for it, it should be there.
            # But state might contain coords that are not in data_vars.
            # We select data_vars + coords that are in the list.

            # Actually, simple selection works well in xarray
            # It selects data_vars and keeps associated coords.
            # If a variable in 'variables' is not in state, it raises KeyError.
            state = state[self.variables]

        if not self._is_initialized:
            self._initialize(state)
            self._is_initialized = True

        self._write_step(state)


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

    def _initialize(self, state: xr.Dataset) -> None:
        """Create the Zarr store structure."""
        if self.path.exists():
            raise FileExistsError(f"Output path '{self.path}' already exists.")

        self.path.mkdir(parents=True, exist_ok=True)

        # We need to prepare the dataset for Zarr writing:
        # 1. Expand dims to include time (if scalar) so Zarr knows it's a dimension
        # 2. Set up encoding/chunks

        from seapopym.standard.coordinates import Coordinates

        # Check for time coordinate
        time_dim = Coordinates.T.value
        if time_dim not in state.coords:
            # If missing, we can't really initialize properly for time series
            raise ValueError(f"State is missing time coordinate '{time_dim}'")

        # Ensure time is a dimension (it might be a scalar coord)
        ds_to_init = state.expand_dims(dim=time_dim) if time_dim not in state.dims else state

        # Initialize Zarr store
        # We use compute=False to just write metadata/structure?
        # No, to_zarr with compute=False returns a Delayed object.
        # We want to write the first chunk or initialize empty?
        # Actually, we can just write the first step, and subsequent steps will append.
        # But 'append' mode in Zarr requires the store to exist.

        # Strategy:
        # 1. Write the first timestep (mode='w')
        # 2. Subsequent steps use mode='a' (append)

        # Apply chunking
        # We can pass encoding to to_zarr
        # encoding = {var: {'chunks': ...} for var in ds_to_init.data_vars}

        # Write first step
        ds_to_init.attrs.update(self.metadata)
        ds_to_init.to_zarr(self.path, mode="w", safe_chunks=False)

    def _write_step(self, state: xr.Dataset) -> None:
        """Write timestep to Zarr."""
        from seapopym.standard.coordinates import Coordinates

        time_dim = Coordinates.T.value

        # Ensure time dimension
        ds_to_write = state.expand_dims(dim=time_dim) if time_dim not in state.dims else state

        # Append to existing store
        ds_to_write.to_zarr(self.path, append_dim=time_dim, safe_chunks=False)

    def append(self, state: xr.Dataset) -> None:
        """Overridden to handle the specific init/write logic for Zarr."""
        # Filter variables
        if self.variables is not None:
            state = state[self.variables]

        if not self._is_initialized:
            # For Zarr, initialization IS writing the first step
            self._initialize(state)
            self._is_initialized = True
        else:
            self._write_step(state)

    def finalize(self) -> xr.Dataset:
        """Return the opened Zarr dataset."""
        return xr.open_zarr(self.path)
