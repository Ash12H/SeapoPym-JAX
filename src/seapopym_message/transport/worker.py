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

Current implementation:
    - PASSTHROUGH mode: Returns biomass unchanged
    - This validates the infrastructure before implementing actual transport
    - Transport will be implemented using JAX-Fluids
"""

from typing import Any

import jax.numpy as jnp
import ray


@ray.remote
class TransportWorker:
    """Ray actor for centralized transport computation.

    CURRENT STATUS: PASSTHROUGH MODE

    This worker is currently in passthrough mode, returning biomass unchanged.
    This validates the Ray communication infrastructure before implementing
    the actual transport physics.

    Next step: Implement advection-diffusion using JAX-Fluids.

    Args:
        None (simplified for passthrough mode)

    Example:
        >>> import ray
        >>> ray.init()
        >>> worker = TransportWorker.remote()
        >>> result = ray.get(worker.transport_step.remote(
        ...     biomass=biomass_array,
        ...     u=u_velocity, v=v_velocity, D=diffusivity,
        ...     dt=3600.0, dx=20000.0, dy=20000.0,
        ...     mask=ocean_mask
        ... ))
        >>> # result['biomass'] == biomass_array (unchanged in passthrough mode)
    """

    def __init__(self) -> None:
        """Initialize TransportWorker in passthrough mode.

        No transport is applied - this validates the infrastructure.
        """
        pass  # Passthrough mode - no initialization needed

    def transport_step(
        self,
        biomass: jnp.ndarray,
        _u: jnp.ndarray,
        _v: jnp.ndarray,
        _D: float | jnp.ndarray,
        _dt: float,
        _dx: float,
        _dy: float,
        _mask: jnp.ndarray | None = None,
    ) -> dict[str, Any]:
        """Execute one transport step (PASSTHROUGH MODE).

        Currently returns biomass unchanged to validate infrastructure.

        Args:
            biomass: Biomass concentration field (nlat, nlon).
            u: Zonal velocity field (m/s), shape (nlat, nlon).
            v: Meridional velocity field (m/s), shape (nlat, nlon).
            D: Diffusivity (m²/s), scalar or array (nlat, nlon).
            dt: Time step (s).
            dx: Grid spacing in x-direction (m).
            dy: Grid spacing in y-direction (m).
            mask: Ocean mask (True=ocean, False=land), shape (nlat, nlon).

        Returns:
            Dictionary containing:
                - 'biomass': Unchanged biomass field (passthrough)
                - 'diagnostics': Dictionary with perfect conservation metrics

        Note:
            PASSTHROUGH MODE: No transport applied.
            This validates the Ray communication infrastructure.
        """
        import time

        t_start = time.perf_counter()

        # PASSTHROUGH: Return biomass unchanged
        biomass_final = biomass

        # Compute diagnostics (perfect conservation in passthrough)
        mass_before = float(jnp.sum(biomass))
        mass_after = float(jnp.sum(biomass_final))

        t_end = time.perf_counter()
        compute_time = t_end - t_start

        diagnostics = {
            "mass_before": mass_before,
            "mass_after": mass_after,
            "mass_error_total": 0.0,  # Perfect conservation
            "mass_error_advection": 0.0,
            "mass_error_diffusion": 0.0,
            "compute_time_s": compute_time,
            "mode": "passthrough",
            "message": "Infrastructure validation mode - no transport applied",
        }

        return {"biomass": biomass_final, "diagnostics": diagnostics}
