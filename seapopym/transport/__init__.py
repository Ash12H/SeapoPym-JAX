"""Transport module for Seapopym.

Provides advection and diffusion processes as Blueprint-compatible units.
"""

from .boundary import BoundaryConditions, BoundaryType, get_neighbors_with_bc
from .core import compute_advection_numba, compute_advection_tendency, compute_diffusion_tendency
from .grid import (
    compute_spherical_cell_areas,
    compute_spherical_dx,
    compute_spherical_dy,
    compute_spherical_face_areas_ew,
    compute_spherical_face_areas_ns,
)
from .stability import check_diffusion_stability, compute_advection_cfl

__all__ = [
    # Core transport functions
    "compute_advection_tendency",
    "compute_advection_numba",
    "compute_diffusion_tendency",
    # Boundary conditions
    "BoundaryType",
    "BoundaryConditions",
    "get_neighbors_with_bc",
    # Grid geometry
    "compute_spherical_cell_areas",
    "compute_spherical_face_areas_ew",
    "compute_spherical_face_areas_ns",
    "compute_spherical_dx",
    "compute_spherical_dy",
    # Stability checking
    "check_diffusion_stability",
    "compute_advection_cfl",
]
