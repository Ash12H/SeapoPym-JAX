"""CompiledModel dataclass for storing compiled blueprint data.

The CompiledModel contains all data structures ready for JAX execution:
- State variables (evolve each timestep)
- Forcings (input data, read-only)
- Parameters (constants)
- Metadata (shapes, coordinates, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seapopym.blueprint import Blueprint
    from seapopym.blueprint.schema import TendencySource
    from seapopym.compiler.forcing import ForcingStore
    from seapopym.compiler.time_grid import TimeGrid

from seapopym.blueprint.nodes import ComputeNode, DataNode
from seapopym.types import Array


@dataclass
class CompiledModel:
    """Compiled model ready for JAX execution.

    This dataclass contains all the data structures produced by the Compiler,
    organized as pytrees for efficient JAX operations.

    Attributes:
        blueprint: Original blueprint definition.
        compute_nodes: Ordered list of compute nodes (process steps).
        data_nodes: Dict mapping variable paths to DataNode metadata.
        tendency_map: Mapping from state variable names to tendency sources.
        state: State variables that evolve each timestep.
        forcings: Input data (temperature, currents, etc.). Includes mask.
        parameters: Model constants (growth_rate, mortality, etc.).
        shapes: Resolved dimension sizes {"Y": 180, "X": 360, ...}.
        coords: Coordinate arrays for each dimension.
        dt: Timestep in seconds.
        time_grid: Temporal grid (start, end, n_timesteps, coords). None if not using calendar.
    """

    # Source
    blueprint: Blueprint

    # Computation structure (replaces graph)
    compute_nodes: list[ComputeNode] = field(default_factory=list)
    data_nodes: dict[str, DataNode] = field(default_factory=dict)
    tendency_map: dict[str, list[TendencySource]] = field(default_factory=dict)

    # Data pytrees
    state: dict[str, Array] = field(default_factory=dict)
    forcings: ForcingStore = field(default_factory=lambda: _default_forcing_store())
    parameters: dict[str, Array] = field(default_factory=dict)

    # Metadata
    shapes: dict[str, int] = field(default_factory=dict)
    coords: dict[str, Array] = field(default_factory=dict)
    dt: float = 86400.0  # Default: 1 day in seconds

    # Temporal configuration
    time_grid: TimeGrid | None = None

    # Time-indexed parameters (params with time dim, passed as scan xs)
    time_indexed_params: set[str] = field(default_factory=set)

    # Clamping bounds per state variable (from blueprint declarations)
    clamp_map: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    # Name of the time dimension (default "T", configurable via ExecutionParams)
    time_dim: str = "T"

    @property
    def n_timesteps(self) -> int:
        """Return the number of timesteps from the time dimension."""
        return self.shapes.get(self.time_dim, 1)

    def get_state_shape(self, var_name: str) -> tuple[int, ...]:
        """Get the shape of a state variable."""
        if var_name not in self.state:
            raise KeyError(f"State variable '{var_name}' not found")
        return self.state[var_name].shape


def _default_forcing_store() -> ForcingStore:
    """Create a default empty ForcingStore (avoids circular import at module level)."""
    from seapopym.compiler.forcing import ForcingStore

    return ForcingStore()
