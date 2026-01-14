"""Stability checking utilities for transport schemes.

This module provides functions to verify numerical stability constraints
for explicit time integration schemes used in advection and diffusion.

References:
    - Diffusion stability: IA/Diffusion-euler-explicite-description.md (line 41)
    - CFL condition: IA/transport/advection.py
"""

import logging
from typing import Any

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def check_diffusion_stability(
    D: float | xr.DataArray,
    dx: xr.DataArray | float,
    dy: xr.DataArray | float,
    dt: float | xr.DataArray,
) -> dict[str, Any]:
    """Check stability criterion for explicit diffusion scheme.

    The explicit Euler scheme for diffusion is stable if:
        dt ≤ min(dx², dy²) / (4 × D)

    For spherical grids, dx decreases toward poles, so the minimum dx
    (at the poles) limits the timestep most severely.

    This function:
    1. Computes the maximum stable timestep dt_max
    2. Checks if the provided dt satisfies the stability criterion
    3. Computes the diffusion CFL number (should be ≤ 0.25)
    4. Logs warnings if the scheme is unstable

    Args:
        D: Diffusion coefficient [m²/s]
           Can be scalar or DataArray (spatially varying)
        dx: Grid spacing in X direction [m]
            Can be scalar or DataArray (varies with latitude for spherical grids)
        dy: Grid spacing in Y direction [m]
            Can be scalar or DataArray
        dt: Time step [s]

    Returns:
        Dictionary with:
        - is_stable: bool, whether dt satisfies stability criterion
        - dt_max: Maximum stable dt [s]
        - dx_min: Minimum grid spacing in x [m]
        - dy_min: Minimum grid spacing in y [m]
        - D_max: Maximum diffusion coefficient [m²/s]
        - cfl_diffusion: Actual CFL number (should be ≤ 0.25)
        - margin: Safety margin (dt_max / dt), should be > 1

    Example:
        >>> dx = compute_spherical_dx(lats, lons)
        >>> dy = compute_spherical_dy(lats, lons)
        >>> D = 1000.0  # m²/s
        >>> dt = 3600.0  # 1 hour
        >>> stability = check_diffusion_stability(D, dx, dy, dt)
        >>> if not stability["is_stable"]:
        ...     print(f"Unstable! Reduce dt to {stability['dt_max']:.1f} s")

    Reference:
        IA/Diffusion-euler-explicite-description.md, line 41:
        "Pour la stabilité de ce schéma explicite, il faut que:
         dt ≤ min(dx², dy²) / (4·D)"
    """
    # Extract scalar values from DataArrays if needed
    # Skip NaN values to handle masked data
    D_max = float(D.max(skipna=True).values) if isinstance(D, xr.DataArray) else float(D)

    dx_min = float(dx.min(skipna=True).values) if isinstance(dx, xr.DataArray) else float(dx)

    dy_min = float(dy.min(skipna=True).values) if isinstance(dy, xr.DataArray) else float(dy)

    # Extract scalar value from dt if it's a DataArray
    dt_val = float(dt.values) if isinstance(dt, xr.DataArray) else float(dt)

    # Handle D=0 case (no diffusion, always stable)
    if D_max == 0.0:
        return {
            "is_stable": True,
            "dt_max": float("inf"),
            "dx_min": dx_min,
            "dy_min": dy_min,
            "D_max": D_max,
            "cfl_diffusion": 0.0,
            "margin": float("inf"),
        }

    # Stability criterion: dt ≤ min(dx², dy²) / (4D)
    min_spacing_sq = min(dx_min**2, dy_min**2)
    dt_max = min_spacing_sq / (4 * D_max)

    # CFL number for diffusion: (D × dt) / dx²
    # Should be ≤ 0.25 for stability
    cfl = (D_max * dt_val) / min_spacing_sq

    # Check stability
    is_stable = dt_val <= dt_max

    # Safety margin
    margin = dt_max / dt_val if dt_val > 0 else float("inf")

    # Log warnings if unstable
    if not is_stable:
        logger.warning(
            f"Diffusion scheme is UNSTABLE!\n"
            f"  Current dt = {dt_val:.2f} s\n"
            f"  Maximum stable dt = {dt_max:.2f} s\n"
            f"  CFL number = {cfl:.4f} (should be ≤ 0.25)\n"
            f"  D = {D_max:.2e} m²/s\n"
            f"  min(dx) = {dx_min:.2f} m\n"
            f"  min(dy) = {dy_min:.2f} m\n"
            f"Recommendation: Reduce timestep by factor of {1 / margin:.2f}"
        )
    elif cfl > 0.2:
        # Warn if CFL is close to limit (within 80% of max)
        logger.warning(
            f"Diffusion scheme is near stability limit:\n"
            f"  CFL number = {cfl:.4f} (limit = 0.25)\n"
            f"  Safety margin = {margin:.2f}x\n"
            f"  Consider reducing timestep for better accuracy"
        )

    return {
        "is_stable": is_stable,
        "dt_max": dt_max,
        "dx_min": dx_min,
        "dy_min": dy_min,
        "D_max": D_max,
        "cfl_diffusion": float(cfl),
        "margin": margin,
    }


def compute_advection_cfl(
    u: xr.DataArray | float,
    v: xr.DataArray | float,
    dx: xr.DataArray | float,
    dy: xr.DataArray | float,
    dt: float | xr.DataArray,
) -> dict[str, Any]:
    """Compute CFL number for advection scheme.

    The CFL (Courant-Friedrichs-Lewy) condition for advection is:
        CFL = (|u| × dt / dx) + (|v| × dt / dy) ≤ 1

    For upwind schemes, CFL ≤ 1 ensures stability.
    For better accuracy, CFL ≤ 0.5 is often recommended.

    Args:
        u: Zonal velocity [m/s]
        v: Meridional velocity [m/s]
        dx: Grid spacing in X direction [m]
        dy: Grid spacing in Y direction [m]
        dt: Time step [s]

    Returns:
        Dictionary with:
        - cfl_max: Maximum CFL number across grid
        - cfl_x_max: Maximum CFL in x direction
        - cfl_y_max: Maximum CFL in y direction
        - is_stable: Whether CFL ≤ 1
        - u_max: Maximum |u| [m/s]
        - v_max: Maximum |v| [m/s]

    Example:
        >>> cfl_info = compute_advection_cfl(u, v, dx, dy, dt)
        >>> if not cfl_info["is_stable"]:
        ...     print(f"CFL = {cfl_info['cfl_max']:.2f} > 1, reduce timestep!")
    """
    # Extract arrays and compute absolute values
    u_abs = np.abs(u.values) if isinstance(u, xr.DataArray) else np.abs(u)

    v_abs = np.abs(v.values) if isinstance(v, xr.DataArray) else np.abs(v)

    dx_vals = dx.values if isinstance(dx, xr.DataArray) else dx

    dy_vals = dy.values if isinstance(dy, xr.DataArray) else dy

    # Extract scalar value from dt if it's a DataArray
    dt_val = float(dt.values) if isinstance(dt, xr.DataArray) else float(dt)

    # CFL components
    cfl_x = u_abs * dt_val / dx_vals
    cfl_y = v_abs * dt_val / dy_vals

    # Maximum values (skip NaN values to handle masked data)
    cfl_x_max = float(np.nanmax(cfl_x))
    cfl_y_max = float(np.nanmax(cfl_y))
    cfl_max = cfl_x_max + cfl_y_max

    is_stable = cfl_max <= 1.0

    u_max = float(np.nanmax(u_abs))
    v_max = float(np.nanmax(v_abs))

    # Log warnings
    if not is_stable:
        logger.warning(
            f"Advection CFL condition violated!\n"
            f"  CFL = {cfl_max:.3f} > 1.0\n"
            f"  CFL_x = {cfl_x_max:.3f}, CFL_y = {cfl_y_max:.3f}\n"
            f"  max|u| = {u_max:.3f} m/s, max|v| = {v_max:.3f} m/s\n"
            f"  dt = {dt_val:.2f} s\n"
            f"Recommendation: Reduce timestep by factor of {1 / cfl_max:.2f}"
        )
    elif cfl_max > 0.8:
        logger.warning(
            f"Advection CFL near limit:\n"
            f"  CFL = {cfl_max:.3f} (limit = 1.0)\n"
            f"  Consider reducing timestep for better accuracy"
        )

    return {
        "cfl_max": cfl_max,
        "cfl_x_max": cfl_x_max,
        "cfl_y_max": cfl_y_max,
        "is_stable": is_stable,
        "u_max": u_max,
        "v_max": v_max,
    }
