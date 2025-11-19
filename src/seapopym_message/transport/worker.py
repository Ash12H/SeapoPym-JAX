"""TransportWorker: Centralized transport computation for distributed simulation.

This module implements a Ray remote actor that handles all transport operations
(advection + diffusion) on the global domain. The worker receives biomass fields
from distributed CellWorkers, applies transport, and returns the updated state.

Architecture:
    EventScheduler
        ↓
    CellWorker2D (biology) + TransportWorker (transport)
        ↓
    GPU-optimized JAX transport

Workflow per timestep:
    1. Biology phase: Parallel computation on each CellWorker
    2. Collect global biomass from all workers
    3. Transport phase: TransportWorker applies advection + diffusion globally
    4. Redistribute updated biomass to CellWorkers

Implementation:
    - Uses flux-based upwind advection (volumes finis)
    - Uses explicit Euler diffusion on spherical grid
    - Supports land masking and configurable boundary conditions
    - References: IA/TRANSPORT_ANALYSIS.md, IA/TRANSPORT_IMPLEMENTATION_PLAN.md
"""

from typing import Any

import jax.numpy as jnp
import ray

from seapopym_message.transport.advection import (
    advection_upwind_flux,
    compute_advection_diagnostics,
)
from seapopym_message.transport.boundary import BoundaryConditions, BoundaryType
from seapopym_message.transport.diffusion import (
    check_diffusion_stability,
    diffusion_explicit_spherical,
)
from seapopym_message.transport.grid import Grid, PlaneGrid, SphericalGrid


@ray.remote
class TransportWorker:
    """Ray actor for centralized transport computation.

    This worker implements physics-based advection-diffusion transport using:
    - Flux-based upwind scheme for advection (volumes finis)
    - Explicit Euler scheme for diffusion on spherical grid

    The grid is configured once at initialization and reused for all timesteps.

    Args:
        grid_type: Type of grid ("spherical" or "plane")
        lat_min, lat_max: Latitude bounds [degrees] (spherical only)
        lon_min, lon_max: Longitude bounds [degrees] (spherical only)
        nlat, nlon: Number of grid cells
        dx, dy: Grid spacing [m] (plane grid only)
        R: Earth radius [m] (spherical only, default 6371e3)
        lat_bc: North/South boundary type ("closed", "periodic", "open")
        lon_bc: East/West boundary type ("closed", "periodic", "open")

    Example:
        >>> import ray
        >>> ray.init()
        >>> worker = TransportWorker.remote(
        ...     grid_type="spherical",
        ...     lat_min=-60.0, lat_max=60.0,
        ...     lon_min=0.0, lon_max=360.0,
        ...     nlat=120, nlon=360,
        ...     lat_bc="closed", lon_bc="periodic"
        ... )
        >>> result = ray.get(worker.transport_step.remote(
        ...     biomass=biomass_array,
        ...     u=u_velocity, v=v_velocity, D=1000.0,
        ...     dt=3600.0, mask=ocean_mask
        ... ))
    """

    def __init__(
        self,
        grid_type: str = "spherical",
        lat_min: float = -90.0,
        lat_max: float = 90.0,
        lon_min: float = 0.0,
        lon_max: float = 360.0,
        nlat: int = 180,
        nlon: int = 360,
        dx: float | None = None,
        dy: float | None = None,
        R: float = 6371e3,
        lat_bc: str = "closed",
        lon_bc: str = "periodic",
        mask: Any | None = None,
    ) -> None:
        """Initialize TransportWorker with grid and boundary conditions.

        The grid geometry is computed once and cached for efficiency.

        Args:
            grid_type: Type of grid ("spherical" or "plane")
            lat_min: Minimum latitude [degrees] (spherical only)
            lat_max: Maximum latitude [degrees] (spherical only)
            lon_min: Minimum longitude [degrees] (spherical only)
            lon_max: Maximum longitude [degrees] (spherical only)
            nlat: Number of latitude grid cells
            nlon: Number of longitude grid cells
            dx: Grid spacing in x direction [m] (plane grid only)
            dy: Grid spacing in y direction [m] (plane grid only)
            R: Earth radius [m] (spherical only, default 6371e3)
            lat_bc: North/South boundary type ("closed", "periodic", "open")
            lon_bc: East/West boundary type ("closed", "periodic", "open")
            mask: Ocean/land mask as xarray.DataArray (optional)
                  - 2D: coords=["lat", "lon"] or ["Y", "X"], shape (nlat, nlon)
                  - 3D: coords=["depth", "lat", "lon"] for bathymetry
                  - Values: 1.0 = ocean, 0.0 = land
                  - dtype: float32
        """
        # Create grid based on type
        self.grid: Grid
        if grid_type == "spherical":
            self.grid = SphericalGrid(
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                nlat=nlat,
                nlon=nlon,
                R=R,
                mask=mask,
            )
        elif grid_type == "plane":
            if dx is None or dy is None:
                raise ValueError("Plane grid requires dx and dy parameters")
            self.grid = PlaneGrid(dx=dx, dy=dy, nlat=nlat, nlon=nlon, mask=mask)
        else:
            raise ValueError(f"Unknown grid_type: {grid_type}")

        # Parse boundary condition strings
        lat_bc_enum = BoundaryType(lat_bc)
        lon_bc_enum = BoundaryType(lon_bc)

        # Create boundary conditions (N, S, E, W)
        self.boundary = BoundaryConditions(
            north=lat_bc_enum,
            south=lat_bc_enum,
            east=lon_bc_enum,
            west=lon_bc_enum,
        )

        self.grid_type = grid_type

    def transport_step(
        self,
        biomass: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        D: float | jnp.ndarray,
        dt: float,
        _dx: float | None = None,  # Legacy parameter, ignored if grid initialized
        _dy: float | None = None,  # Legacy parameter, ignored if grid initialized
        mask: jnp.ndarray | None = None,  # Deprecated: use grid.mask instead
    ) -> dict[str, Any]:
        """Execute one transport step with advection + diffusion.

        Args:
            biomass: Biomass concentration field [kg/m²], shape (nlat, nlon)
            u: Zonal velocity field [m/s], shape (nlat, nlon)
            v: Meridional velocity field [m/s], shape (nlat, nlon)
            D: Horizontal diffusivity [m²/s], scalar or array (nlat, nlon)
            dt: Time step [s]
            _dx: Legacy parameter (ignored, grid spacing from self.grid)
            _dy: Legacy parameter (ignored, grid spacing from self.grid)
            mask: Deprecated parameter (use grid.mask instead)
                  Ocean mask (1=ocean, 0=land, NaN=land), shape (nlat, nlon)
                  If provided, overrides grid.mask for backward compatibility

        Returns:
            Dictionary containing:
                - 'biomass': Updated biomass field after transport
                - 'diagnostics': Conservation and performance metrics
        """
        import time

        t_start = time.perf_counter()

        # Use grid mask if available, otherwise fall back to parameter mask
        # Grid mask takes precedence (new architecture)
        effective_mask = self.grid.get_mask() if self.grid.get_mask() is not None else mask

        # Convert D to scalar if array (take mean for stability check)
        D_scalar = float(jnp.mean(D)) if isinstance(D, jnp.ndarray) else D

        # Check stability (diffusion is more restrictive)
        stability = check_diffusion_stability(dt, D_scalar, self.grid)
        if not stability["is_stable"]:
            import warnings

            warnings.warn(
                f"Diffusion timestep may be unstable! "
                f"dt={dt:.1f}s > dt_max={stability['dt_max']:.1f}s. "
                f"CFL={stability['cfl_diffusion']:.3f} > 0.25",
                stacklevel=2,
            )

        # Step 1: Advection
        t_adv_start = time.perf_counter()
        biomass_advected = advection_upwind_flux(
            biomass=biomass,
            u=u,
            v=v,
            dt=dt,
            grid=self.grid,
            boundary=self.boundary,
            mask=effective_mask,
        )
        t_adv_end = time.perf_counter()

        adv_diagnostics = compute_advection_diagnostics(
            biomass, biomass_advected, u, v, dt, self.grid, effective_mask
        )

        # Step 2: Diffusion
        t_diff_start = time.perf_counter()
        biomass_final = diffusion_explicit_spherical(
            biomass=biomass_advected,
            D=D_scalar,
            dt=dt,
            grid=self.grid,
            boundary=self.boundary,
            mask=effective_mask,
        )
        t_diff_end = time.perf_counter()

        # Final diagnostics
        cell_areas = self.grid.cell_areas()
        if effective_mask is not None:
            ocean_mask = jnp.where(jnp.isnan(effective_mask), 0.0, effective_mask)
        else:
            ocean_mask = jnp.ones_like(biomass)

        mass_before = float(jnp.sum(biomass * cell_areas * ocean_mask))
        mass_after = float(jnp.sum(biomass_final * cell_areas * ocean_mask))
        mass_error = abs(mass_after - mass_before)
        conservation_fraction = mass_after / (mass_before + 1e-10)

        t_end = time.perf_counter()

        diagnostics = {
            "mass_before": mass_before,
            "mass_after": mass_after,
            "mass_error_total": mass_error,
            "conservation_fraction": conservation_fraction,
            "mass_error_advection": adv_diagnostics["mass_change"],
            "conservation_advection": adv_diagnostics["conservation_fraction"],
            "max_velocity": adv_diagnostics["max_velocity"],
            "cfl_advection": adv_diagnostics["cfl_number"],
            "cfl_diffusion": stability["cfl_diffusion"],
            "stability_ok": stability["is_stable"],
            "dt_max_diffusion": stability["dt_max"],
            "compute_time_s": t_end - t_start,
            "compute_time_advection_s": t_adv_end - t_adv_start,
            "compute_time_diffusion_s": t_diff_end - t_diff_start,
            "mode": "physics",
            "grid_type": self.grid_type,
        }

        return {"biomass": biomass_final, "diagnostics": diagnostics}
