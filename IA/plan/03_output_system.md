# Implementation Plan - Flexible Output System

This plan outlines the implementation of a flexible output system for Seapopym, supporting both in-memory storage (for notebooks/dev) and disk storage (Zarr, for production).

## 1. Create IO Module Structure
- **Directory**: `seapopym/io`
- **File**: `seapopym/io/__init__.py`
- **File**: `seapopym/io/writer.py`

## 2. Implement Writer Classes
- **Location**: `seapopym/io/writer.py`
- **Abstract Base Class**: `BaseOutputWriter`
    - `__init__(variables: list[str] | None = None, metadata: dict | None = None)`
        - `variables`: List of variable names to save (None = all variables)
        - `metadata`: Dictionary to add as Dataset attributes
    - `_initialize(state: xr.Dataset)`: Private method called automatically on first `append()`
        - Infers structure from the first state (dimensions, coordinates, dtypes)
        - Creates storage structure (buffer for Memory, Zarr store for Disk)
        - Adds `metadata` to Dataset attributes
    - `append(state: xr.Dataset)`: Filters state based on `variables`, then writes
        - Note: `state` contains coordinate `Coordinates.T` (scalar), NOT a time dimension
        - The writer adds the time dimension internally and accumulates timesteps
        - Calls `_initialize()` automatically on first call
    - `finalize() -> xr.Dataset`: Returns the complete dataset with time dimension
- **Implementation 1**: `MemoryWriter`
    - Stores filtered states in a list `self._buffer`
    - `_initialize()` stores the first state structure for reference
    - `append()` adds filtered state to buffer
    - `finalize()` concatenates buffer along a new time dimension, returns Dataset
- **Implementation 2**: `ZarrWriter`
    - `__init__(path, variables=None, metadata=None, chunks=None)`
        - `path`: Output path for Zarr store
        - `chunks`: Chunking strategy (default: `{'time': 1}` + auto for spatial dims)
    - `_initialize()`:
        - Raises error if `path` already exists (prevents overwriting)
        - Creates parent directories if they don't exist
        - Creates Zarr store structure with unlimited time dimension
        - Sets up chunking strategy for efficient I/O
    - `append()` writes the filtered timestep to Zarr along time axis
    - `finalize()` returns `xr.open_zarr(path)` (lazy-loaded Dataset)
    - If simulation crashes, partial Zarr files are left for debugging

## 3. Integrate into SimulationController
- **Location**: `seapopym/controller/core.py`
- **Changes**:
    - Update `setup()`:
        - Add `output_path: str | Path | None = None`
        - Add `output_variables: list[str] | None = None`
        - Add `output_metadata: dict | None = None`
        - If `output_path` is None -> Instantiate `MemoryWriter(variables, metadata)`
        - If `output_path` is provided -> Instantiate `ZarrWriter(path, variables, metadata)`
    - Update `run()`:
        - Call `self.writer.append(state)` inside the time loop
        - No explicit initialization needed (automatic on first append)
    - Add property `results`:
        ```python
        @property
        def results(self) -> xr.Dataset:
            return self.writer.finalize()
        ```
        - Note: Not cached to avoid memory leaks with Dask distributed runs
        - MemoryWriter: returns same object each time (no cost)
        - ZarrWriter: returns lazy-loaded Dataset (minimal memory footprint)

## 4. Verification
- **Test**: `tests/test_io_writer.py`
- **Scenario 1 (Memory)**: Run short sim, check `controller.results` is a Dataset with time dimension
- **Scenario 2 (Disk)**: Run short sim with path, verify:
    - Zarr store created at specified path
    - Error raised if path already exists
    - `controller.results` loads Dataset lazily
- **Scenario 3 (Variable filtering)**: Verify only requested variables are saved
- **Scenario 4 (Metadata)**: Verify metadata dict appears in `ds.attrs`
