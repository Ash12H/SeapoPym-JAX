"""Helper functions for backend tests.

These functions are defined at module level to ensure they can be
serialized by Ray (cloudpickle) when running distributed tests.
"""

import xarray as xr


def add_one(x: xr.DataArray) -> dict[str, xr.DataArray]:
    """Add 1 to input."""
    return {"result": x + 1.0}


def multiply_two(x: xr.DataArray) -> dict[str, xr.DataArray]:
    """Multiply input by 2."""
    return {"result": x * 2.0}


def group1_func(x: xr.DataArray) -> dict[str, xr.DataArray]:
    """Group 1 function: add 1."""
    return {"out": x + 1.0}


def group2_func(y: xr.DataArray) -> dict[str, xr.DataArray]:
    """Group 2 function: multiply by 2."""
    return {"out": y * 2.0}


def create_local(x: xr.DataArray) -> dict[str, xr.DataArray]:
    """Create local variable."""
    return {"var": x + 10.0}


def use_local(var: xr.DataArray) -> dict[str, xr.DataArray]:
    """Use local variable."""
    return {"result": var * 2.0}


def dummy(x: xr.DataArray) -> dict[str, xr.DataArray]:
    """Dummy function for testing."""
    return {"result": x}


def buggy_func(x: xr.DataArray) -> dict[str, xr.DataArray]:
    """Function that raises an error."""
    raise ValueError("Intentional error")


def bad_func(x: xr.DataArray) -> xr.DataArray:
    """Function with wrong return type."""
    return x  # Should return dict
