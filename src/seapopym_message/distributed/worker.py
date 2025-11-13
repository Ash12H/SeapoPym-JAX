"""CellWorker2D: Ray actor for distributed spatial simulation.

A CellWorker manages a rectangular patch of the 2D spatial domain and executes
a Kernel locally. Workers communicate via halo exchange for global operations.
"""

from typing import Any

import jax.numpy as jnp
import ray

from seapopym_message.core.kernel import Kernel
from seapopym_message.utils.grid import GridInfo


@ray.remote
class CellWorker2D:
    """Ray actor managing a 2D spatial patch.

    Each worker:
    - Owns a rectangular patch (lat_slice × lon_slice)
    - Executes a Kernel on its local state
    - Exchanges halo data with neighbors for global operations

    Args:
        worker_id: Unique integer identifier.
        grid_info: Global grid information.
        lat_start: Starting latitude index (inclusive).
        lat_end: Ending latitude index (exclusive).
        lon_start: Starting longitude index (inclusive).
        lon_end: Ending longitude index (exclusive).
        kernel: Computational kernel to execute.
        params: Model parameters.

    Example:
        >>> import ray
        >>> ray.init()
        >>> grid = GridInfo(0, 10, 0, 20, nlat=20, nlon=40)
        >>> kernel = Kernel([compute_growth])
        >>> worker = CellWorker2D.remote(
        ...     worker_id=0,
        ...     grid_info=grid,
        ...     lat_start=0, lat_end=10,
        ...     lon_start=0, lon_end=20,
        ...     kernel=kernel,
        ...     params={'R': 5.0, 'lambda': 0.1}
        ... )
    """

    def __init__(
        self,
        worker_id: int,
        grid_info: GridInfo,
        lat_start: int,
        lat_end: int,
        lon_start: int,
        lon_end: int,
        kernel: Kernel,
        params: dict[str, Any],
    ) -> None:
        """Initialize worker with patch and kernel."""
        self.worker_id = worker_id
        self.grid_info = grid_info
        self.lat_start = lat_start
        self.lat_end = lat_end
        self.lon_start = lon_start
        self.lon_end = lon_end
        self.kernel = kernel
        self.params = params

        # Local patch dimensions
        self.nlat = lat_end - lat_start
        self.nlon = lon_end - lon_start

        # Neighbors (will be set later)
        self.neighbors: dict[str, ray.ObjectRef | None] = {
            "north": None,
            "south": None,
            "east": None,
            "west": None,
        }

        # Current simulation time
        self.t = 0.0

        # State (will be initialized later)
        self.state: dict[str, jnp.ndarray] = {}

    def set_neighbors(self, neighbors: dict[str, ray.ObjectRef | None]) -> None:
        """Set references to neighboring workers.

        Args:
            neighbors: Dictionary with keys 'north', 'south', 'east', 'west'.
                      Values are Ray ObjectRefs to neighbor workers, or None at boundaries.
        """
        self.neighbors = neighbors

    def set_initial_state(self, state: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        """Set initial state for this worker's patch.

        Args:
            state: Dictionary of state variables (e.g., {'biomass': array}).
                  Arrays should have shape (nlat, nlon).

        Returns:
            The state that was set (for confirmation).
        """
        self.state = {k: jnp.array(v) for k, v in state.items()}
        return self.state

    def get_state(self) -> dict[str, jnp.ndarray]:
        """Get current state.

        Returns:
            Dictionary of state variables.
        """
        return self.state

    def get_time(self) -> float:
        """Get current simulation time.

        Returns:
            Current time.
        """
        return self.t

    def get_boundary_north(self) -> dict[str, jnp.ndarray]:
        """Get northern boundary (first row) for halo exchange.

        Returns:
            Dictionary with state variables at northern boundary.
            Each array has shape (nlon,).
        """
        return {key: val[0, :] for key, val in self.state.items()}

    def get_boundary_south(self) -> dict[str, jnp.ndarray]:
        """Get southern boundary (last row) for halo exchange.

        Returns:
            Dictionary with state variables at southern boundary.
            Each array has shape (nlon,).
        """
        return {key: val[-1, :] for key, val in self.state.items()}

    def get_boundary_east(self) -> dict[str, jnp.ndarray]:
        """Get eastern boundary (last column) for halo exchange.

        Returns:
            Dictionary with state variables at eastern boundary.
            Each array has shape (nlat,).
        """
        return {key: val[:, -1] for key, val in self.state.items()}

    def get_boundary_west(self) -> dict[str, jnp.ndarray]:
        """Get western boundary (first column) for halo exchange.

        Returns:
            Dictionary with state variables at western boundary.
            Each array has shape (nlat,).
        """
        return {key: val[:, 0] for key, val in self.state.items()}

    async def step(self, dt: float) -> dict[str, Any]:
        """Execute one timestep.

        Workflow:
        1. Execute local phase (embarrassingly parallel)
        2. If kernel has global units:
           a. Request halo data from neighbors
           b. Wait for all halos to arrive
           c. Execute global phase with halo data
        3. Update time
        4. Return diagnostics

        Args:
            dt: Time step size.

        Returns:
            Dictionary with diagnostics (time, mean values, etc.).
        """
        # Phase 1: Local computation
        self.state = self.kernel.execute_local_phase(
            self.state, dt=dt, params=self.params, grid_shape=(self.nlat, self.nlon)
        )

        # Phase 2: Global computation (if needed)
        if self.kernel.has_global_units():
            # Request boundary data from neighbors
            halo_futures = self._request_halos()

            # Gather halo data
            halo_data = await self._gather_halos(halo_futures)

            # Execute global phase
            self.state = self.kernel.execute_global_phase(
                self.state, dt=dt, params=self.params, neighbor_data=halo_data
            )

        # Update time
        self.t += dt

        # Return diagnostics
        return self._compute_diagnostics()

    def _request_halos(self) -> dict[str, ray.ObjectRef | None]:
        """Request boundary data from neighbors.

        Returns:
            Dictionary with futures for each neighbor's boundary data.
        """
        halo_futures = {}

        if self.neighbors["north"] is not None:
            halo_futures["north"] = self.neighbors["north"].get_boundary_south.remote()
        else:
            halo_futures["north"] = None

        if self.neighbors["south"] is not None:
            halo_futures["south"] = self.neighbors["south"].get_boundary_north.remote()
        else:
            halo_futures["south"] = None

        if self.neighbors["east"] is not None:
            halo_futures["east"] = self.neighbors["east"].get_boundary_west.remote()
        else:
            halo_futures["east"] = None

        if self.neighbors["west"] is not None:
            halo_futures["west"] = self.neighbors["west"].get_boundary_east.remote()
        else:
            halo_futures["west"] = None

        return halo_futures

    async def _gather_halos(
        self, halo_futures: dict[str, ray.ObjectRef | None]
    ) -> dict[str, dict[str, jnp.ndarray] | None]:
        """Wait for and gather halo data from neighbors.

        Args:
            halo_futures: Dictionary with futures from _request_halos.

        Returns:
            Dictionary with actual halo data for each direction.
            Format: {'halo_north': {...}, 'halo_south': {...}, ...}
        """
        halo_data = {}

        # Build halo data dictionary by awaiting each future
        for direction in ["north", "south", "east", "west"]:
            future = halo_futures[direction]
            if future is not None:
                halo_data[f"halo_{direction}"] = await future
            else:
                halo_data[f"halo_{direction}"] = None

        return halo_data

    def _compute_diagnostics(self) -> dict[str, Any]:
        """Compute diagnostic information.

        Returns:
            Dictionary with worker_id, time, and mean state values.
        """
        diagnostics: dict[str, Any] = {
            "worker_id": self.worker_id,
            "t": self.t,
        }

        # Add mean values for all state variables
        for key, val in self.state.items():
            diagnostics[f"{key}_mean"] = float(jnp.mean(val))

        return diagnostics
