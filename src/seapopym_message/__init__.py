"""SEAPOPYM-Message: Distributed Population Dynamics Simulation.

A modern, scalable framework for simulating spatially-explicit population dynamics
with event-driven architecture and composable computational kernels.
"""

__version__ = "0.1.0"

# Import core classes for convenient access
from seapopym_message.core.kernel import Kernel
from seapopym_message.core.unit import Unit, unit
from seapopym_message.simulation import (
    create_distributed_simulation,
    get_global_state,
    initialize_workers,
    run_simulation,
    setup_and_run,
)
from seapopym_message.visualization import (
    create_animation,
    plot_convergence,
    plot_diagnostics_timeseries,
    plot_multi_variable,
    plot_spatial_field,
)

__all__ = [
    "__version__",
    "Kernel",
    "Unit",
    "unit",
    "create_distributed_simulation",
    "initialize_workers",
    "run_simulation",
    "get_global_state",
    "setup_and_run",
    "plot_diagnostics_timeseries",
    "plot_spatial_field",
    "plot_convergence",
    "plot_multi_variable",
    "create_animation",
]
