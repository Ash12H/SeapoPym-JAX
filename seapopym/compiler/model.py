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

import numpy as np

if TYPE_CHECKING:
    from seapopym.blueprint import Blueprint
    from seapopym.blueprint.schema import TendencySource
    from seapopym.compiler.forcing import ForcingStore
    from seapopym.compiler.time_grid import TimeGrid

from seapopym.blueprint.nodes import ComputeNode, DataNode
from seapopym.types import Array, Params


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
        chunk_size: Number of timesteps per temporal chunk. None = process all at once.
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
    chunk_size: int | None = None

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

    def run_with_params(
        self,
        params: Params,
        initial_state: dict[str, Array] | None = None,
        forcings: dict[str, Array] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Run the model with specified parameters.

        Args:
            params: Parameter values to use for this run.
            initial_state: Initial state. If None, uses model's initial state.
            forcings: Forcing data. If None, uses model's forcings.

        Returns:
            Tuple of (final_state, outputs) where outputs contains all
            timesteps stacked along axis 0.
        """
        import jax.lax as lax

        from seapopym.engine.step import build_step_fn

        step_fn = build_step_fn(self)

        if initial_state is None:
            initial_state = self.state
        if forcings is None:
            forcings = self.forcings.get_all()

        init_carry = (initial_state, params)
        (final_state, _), outputs = lax.scan(step_fn, init_carry, forcings)

        return final_state, outputs

    def to_numpy(self) -> CompiledModel:
        """Convert all arrays to NumPy (useful for debugging)."""
        from seapopym.compiler.forcing import ForcingStore

        def convert(arr: Array) -> np.ndarray:
            if hasattr(arr, "numpy"):
                return arr.numpy()
            return np.asarray(arr)

        # Materialize all forcings and convert to numpy
        all_forcings = self.forcings.get_all()
        numpy_forcings_dict = {k: convert(v) for k, v in all_forcings.items()}
        numpy_store = ForcingStore(
            _forcings=numpy_forcings_dict,
            n_timesteps=self.n_timesteps,
            interp_method=self.forcings.interp_method,
            fill_nan=self.forcings.fill_nan,
            _dynamic_forcings=set(self.forcings._dynamic_forcings),
            _time_coords=self.forcings._time_coords,
        )

        return CompiledModel(
            blueprint=self.blueprint,
            compute_nodes=self.compute_nodes,
            data_nodes=self.data_nodes,
            tendency_map=self.tendency_map,
            state={k: convert(v) for k, v in self.state.items()},
            forcings=numpy_store,
            parameters={k: convert(v) for k, v in self.parameters.items()},
            shapes=self.shapes,
            coords={k: convert(v) for k, v in self.coords.items()},
            dt=self.dt,
        )


def _default_forcing_store() -> ForcingStore:
    """Create a default empty ForcingStore (avoids circular import at module level)."""
    from seapopym.compiler.forcing import ForcingStore

    return ForcingStore()
