"""Function library for SeapoPym models.

This package contains @functional-decorated functions organized by domain:
- biology: Growth, predation, mortality, aging
- physics: Transport, diffusion, advection
- utils: Helper functions
"""

from .biology import simple_growth
from .transport import BoundaryType, transport_tendency

__all__ = [
    "simple_growth",
    "transport_tendency",
    "BoundaryType",
]
