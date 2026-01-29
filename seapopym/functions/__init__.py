"""Function library for SeapoPym models.

This package contains @functional-decorated functions organized by domain:
- lmtl: LMTL ecosystem model (temperature, mortality, cohort dynamics)
- transport: Advection and diffusion
"""

from .lmtl import (
    aging_flow,
    day_length,
    gillooly_temperature,
    layer_weighted_mean,
    mortality_tendency,
    npp_injection,
    recruitment_age,
    recruitment_flow,
    threshold_temperature,
)
from .transport import BoundaryType, transport_tendency

__all__ = [
    # LMTL functions - Environment
    "day_length",
    "layer_weighted_mean",
    # LMTL functions - Temperature
    "threshold_temperature",
    "gillooly_temperature",
    # LMTL functions - Dynamics
    "recruitment_age",
    "mortality_tendency",
    "npp_injection",
    "aging_flow",
    "recruitment_flow",
    # Transport functions
    "transport_tendency",
    "BoundaryType",
]
