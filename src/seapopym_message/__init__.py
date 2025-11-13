"""SEAPOPYM-Message: Distributed Population Dynamics Simulation.

A modern, scalable framework for simulating spatially-explicit population dynamics
with event-driven architecture and composable computational kernels.
"""

__version__ = "0.1.0"

# Import core classes for convenient access
from seapopym_message.core.kernel import Kernel
from seapopym_message.core.unit import Unit, unit

__all__ = [
    "__version__",
    "Kernel",
    "Unit",
    "unit",
]
