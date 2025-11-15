"""Transport module: Centralized advection and diffusion for distributed simulation.

This module provides physics-based transport using:
- Flux-based upwind advection (volumes finis method)
- Explicit Euler diffusion on spherical grids
- Land masking and configurable boundary conditions

Main components:
- TransportWorker: Ray remote actor for centralized transport computation
- Grid classes: SphericalGrid, PlaneGrid for domain geometry
- BoundaryConditions: Configurable boundary types (CLOSED, PERIODIC, OPEN)
- Transport schemes: advection_upwind_flux, diffusion_explicit_spherical

References:
    - IA/TRANSPORT_ANALYSIS.md: Technical analysis and comparison
    - IA/TRANSPORT_IMPLEMENTATION_PLAN.md: Implementation roadmap
    - IA/Advection-upwind-description.md: Upwind method theory
    - IA/Diffusion-euler-explicite-description.md: Diffusion theory
"""

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
from seapopym_message.transport.worker import TransportWorker

__all__ = [
    # Worker (main interface)
    "TransportWorker",
    # Grid classes
    "Grid",
    "SphericalGrid",
    "PlaneGrid",
    # Boundary conditions
    "BoundaryConditions",
    "BoundaryType",
    # Transport schemes
    "advection_upwind_flux",
    "diffusion_explicit_spherical",
    # Diagnostics and utilities
    "compute_advection_diagnostics",
    "check_diffusion_stability",
]
