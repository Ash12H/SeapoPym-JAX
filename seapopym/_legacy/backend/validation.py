"""Validation utilities for backend configuration."""

import logging
import warnings

import xarray as xr

from seapopym.backend.exceptions import BackendConfigurationError

logger = logging.getLogger(__name__)


def has_chunked_arrays(dataset: xr.Dataset) -> bool:
    """Check if dataset contains any Dask arrays with chunks.

    Args:
        dataset: xarray Dataset to check

    Returns:
        True if at least one variable has chunked Dask arrays, False otherwise
    """
    for var_name in dataset.data_vars:
        var = dataset[var_name]
        # Check if it's a Dask array with chunks
        if hasattr(var.data, "chunks") and var.data.chunks is not None:
            return True
    return False


def get_chunked_variables(dataset: xr.Dataset) -> dict[str, tuple]:
    """Get all variables with chunks and their chunk structure.

    Args:
        dataset: xarray Dataset to analyze

    Returns:
        Dictionary mapping variable names to their chunk structure
        Example: {"production": ((1920,), (4320,), (1, 1, 1, ...))}
    """
    chunked_vars = {}
    for var_name in dataset.data_vars:
        var = dataset[var_name]
        if hasattr(var.data, "chunks") and var.data.chunks is not None:
            chunked_vars[var_name] = var.data.chunks
    return chunked_vars


def validate_no_chunks(dataset: xr.Dataset, backend_name: str) -> None:
    """Raise error if dataset contains chunked Dask arrays.

    This validation is used by TaskParallelBackend which is incompatible
    with chunked data due to the use of dask.delayed.

    Args:
        dataset: xarray Dataset to validate
        backend_name: Name of the backend for error message

    Raises:
        BackendConfigurationError: If chunked data is detected
    """
    chunked_vars = get_chunked_variables(dataset)

    if chunked_vars:
        # Format chunk info for error message
        chunks_info = "\n".join(
            [f"  - '{var}': chunks={chunks}" for var, chunks in chunked_vars.items()]
        )

        raise BackendConfigurationError(
            f"{backend_name} detected chunked Dask arrays, which are incompatible with task parallelism.\n"
            f"Chunked variables:\n{chunks_info}\n\n"
            f"Solution: Use backend='data_parallel' for data chunking parallelism.\n"
            f"Example:\n"
            f"  controller = SimulationController(config, backend='data_parallel')\n"
            f"  controller.setup(..., chunks={{'cohort': 1}})"
        )


def validate_has_chunks(dataset: xr.Dataset, backend_name: str) -> None:
    """Warn if dataset has no chunked arrays.

    This validation is used by DataParallelBackend which is optimized for
    chunked data. If no chunks are detected, it suggests using a different backend.

    Args:
        dataset: xarray Dataset to validate
        backend_name: Name of the backend for warning message
    """
    if not has_chunked_arrays(dataset):
        warnings.warn(
            f"{backend_name} selected but no chunked Dask arrays found in state.\n"
            f"{backend_name} is optimized for chunked data parallelism.\n"
            f"Consider:\n"
            f"  1. Adding chunks to your data: setup(..., chunks={{'cohort': 1}})\n"
            f"  2. Using backend='sequential' for eager computation\n"
            f"  3. Using backend='task_parallel' for task-level parallelism",
            UserWarning,
            stacklevel=3,
        )
    else:
        # Log chunk info for debugging
        chunked_vars = get_chunked_variables(dataset)
        logger.info(
            f"{backend_name}: Found {len(chunked_vars)} chunked variable(s): "
            f"{list(chunked_vars.keys())}"
        )
