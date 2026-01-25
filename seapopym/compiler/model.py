"""CompiledModel dataclass for storing compiled blueprint data.

The CompiledModel contains all data structures ready for JAX execution:
- State variables (evolve each timestep)
- Forcings (input data, read-only)
- Parameters (constants)
- Metadata (shapes, coordinates, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

    from seapopym.blueprint import Blueprint

# Type alias for arrays (JAX or NumPy)
Array = Any  # jax.Array | np.ndarray


# Canonical dimension order as per SPEC_02 §4.1
CANONICAL_DIMS: tuple[str, ...] = ("E", "T", "F", "C", "Z", "Y", "X")


@dataclass
class CompiledModel:
    """Compiled model ready for JAX execution.

    This dataclass contains all the data structures produced by the Compiler,
    organized as pytrees for efficient JAX operations.

    Attributes:
        blueprint: Original blueprint definition.
        graph: Dependency graph from validation (NetworkX DiGraph).
        state: State variables that evolve each timestep.
        forcings: Input data (temperature, currents, etc.). Includes mask.
        parameters: Model constants (growth_rate, mortality, etc.).
        shapes: Resolved dimension sizes {"Y": 180, "X": 360, ...}.
        coords: Coordinate arrays for each dimension.
        dt: Timestep in seconds.
        backend: Target backend ("jax" or "numpy").
        trainable_params: List of parameter names that can be optimized.
    """

    # Source
    blueprint: Blueprint

    # Dependency graph
    graph: nx.DiGraph

    # Data pytrees
    state: dict[str, Array] = field(default_factory=dict)
    forcings: dict[str, Array] = field(default_factory=dict)
    parameters: dict[str, Array] = field(default_factory=dict)

    # Metadata
    shapes: dict[str, int] = field(default_factory=dict)
    coords: dict[str, Array] = field(default_factory=dict)
    dt: float = 86400.0  # Default: 1 day in seconds

    # Configuration
    backend: Literal["jax", "numpy"] = "jax"
    trainable_params: list[str] = field(default_factory=list)

    @property
    def mask(self) -> Array | None:
        """Shortcut to access the mask from forcings."""
        return self.forcings.get("mask")

    @property
    def n_timesteps(self) -> int:
        """Return the number of timesteps from the time dimension."""
        return self.shapes.get("T", 1)

    def get_state_shape(self, var_name: str) -> tuple[int, ...]:
        """Get the shape of a state variable."""
        if var_name not in self.state:
            raise KeyError(f"State variable '{var_name}' not found")
        return self.state[var_name].shape

    def to_numpy(self) -> CompiledModel:
        """Convert all arrays to NumPy (useful for debugging)."""

        def convert(arr: Array) -> np.ndarray:
            if hasattr(arr, "numpy"):
                return arr.numpy()
            return np.asarray(arr)

        return CompiledModel(
            blueprint=self.blueprint,
            graph=self.graph,
            state={k: convert(v) for k, v in self.state.items()},
            forcings={k: convert(v) for k, v in self.forcings.items()},
            parameters={k: convert(v) for k, v in self.parameters.items()},
            shapes=self.shapes,
            coords={k: convert(v) for k, v in self.coords.items()},
            dt=self.dt,
            backend="numpy",
            trainable_params=self.trainable_params,
        )
