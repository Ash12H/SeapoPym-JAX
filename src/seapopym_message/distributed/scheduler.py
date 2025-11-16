"""EventScheduler: Orchestrates distributed simulation with priority queue.

The EventScheduler manages the temporal evolution of the simulation by:
- Maintaining a priority queue of events (timesteps)
- Coordinating parallel execution of CellWorker2D actors
- Synchronizing simulation time across all workers
- Collecting and aggregating diagnostics
- Optionally coordinating centralized transport via TransportWorker
"""

import heapq
from typing import Any

import jax.numpy as jnp
import ray


class EventScheduler:
    """Event-driven scheduler for distributed spatial simulation.

    Uses a priority queue to manage simulation events and coordinate
    worker execution. All workers are synchronized at each timestep.

    Args:
        workers: List of CellWorker2D Ray actor references.
        dt: Timestep size (constant for now).
        t_max: Maximum simulation time.
        forcing_manager: Optional ForcingManager for environmental forcings.
        forcing_params: Optional parameters for derived forcings.
        transport_worker: Optional TransportWorker Ray actor for centralized transport.
        transport_enabled: If True, use centralized transport via TransportWorker.
        global_nlat: Global grid latitude dimension (required if transport_enabled).
        global_nlon: Global grid longitude dimension (required if transport_enabled).

    Example:
        >>> import ray
        >>> ray.init()
        >>> workers = [CellWorker2D.remote(...) for _ in range(4)]
        >>> scheduler = EventScheduler(workers, dt=0.1, t_max=10.0)
        >>> diagnostics = scheduler.run()
        >>> len(diagnostics)  # 100 timesteps
        100

        With transport:
        >>> transport = TransportWorker.remote(grid_type="spherical", ...)
        >>> scheduler = EventScheduler(
        ...     workers, dt=3600.0, t_max=86400.0,
        ...     transport_worker=transport, transport_enabled=True,
        ...     global_nlat=120, global_nlon=360
        ... )
    """

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        dt: float,
        t_max: float,
        forcing_manager: Any = None,
        forcing_params: dict[str, Any] | None = None,
        transport_worker: ray.actor.ActorHandle | None = None,
        transport_enabled: bool = False,
        global_nlat: int | None = None,
        global_nlon: int | None = None,
    ) -> None:
        """Initialize scheduler with workers and time parameters."""
        self.workers = workers
        self.dt = dt
        self.t_max = t_max
        self.t_current = 0.0
        self.forcing_manager = forcing_manager
        self.forcing_params = forcing_params if forcing_params is not None else {}

        # Transport configuration
        self.transport_worker = transport_worker
        self.transport_enabled = transport_enabled and transport_worker is not None

        # Global grid dimensions (for transport)
        self.global_nlat = global_nlat
        self.global_nlon = global_nlon

        if self.transport_enabled:
            if self.global_nlat is None or self.global_nlon is None:
                raise ValueError("global_nlat and global_nlon required when transport_enabled=True")
            # Build worker topology by querying workers
            self._build_worker_topology()
        else:
            self.worker_topology: list[dict[str, Any]] = []

        # Priority queue: stores (time, event_type, data)
        self.event_queue: list[tuple[float, str, dict[str, Any]]] = []

        # Schedule first event
        self._schedule_event(time=0.0, event_type="step", data={})

    def _schedule_event(self, time: float, event_type: str, data: dict[str, Any]) -> None:
        """Add event to priority queue.

        Args:
            time: Event time.
            event_type: Type of event ("step", "output", etc.).
            data: Event-specific data.
        """
        heapq.heappush(self.event_queue, (time, event_type, data))

    def _pop_event(self) -> tuple[float, str, dict[str, Any]] | None:
        """Remove and return earliest event from queue.

        Returns:
            Tuple of (time, event_type, data) or None if queue is empty.
        """
        if self.event_queue:
            return heapq.heappop(self.event_queue)
        return None

    def _build_worker_topology(self) -> None:
        """Build worker topology by querying worker patch boundaries.

        Queries each worker to get its lat_start, lat_end, lon_start, lon_end
        and stores this information for later use in collect/redistribute operations.
        """
        # Query all workers for their topology information in parallel
        futures = [worker.get_topology.remote() for worker in self.workers]
        self.worker_topology = ray.get(futures)

    def _collect_global_biomass(self) -> jnp.ndarray:
        """Assemble global biomass field from all workers.

        Collects biomass patches from each worker and assembles them
        into a global grid according to worker topology.

        Returns:
            Global biomass array with shape (global_nlat, global_nlon).

        Raises:
            ValueError: If transport not enabled or topology not available.
        """
        if not self.transport_enabled:
            raise ValueError("Cannot collect biomass: transport not enabled")

        # Collect biomass from all workers in parallel
        futures = [worker.get_biomass.remote() for worker in self.workers]
        patches = ray.get(futures)

        # Initialize global biomass grid
        biomass_global = jnp.zeros((self.global_nlat, self.global_nlon), dtype=jnp.float32)

        # Assemble patches into global grid
        for patch, topology in zip(patches, self.worker_topology, strict=True):
            lat_start = topology["lat_start"]
            lat_end = topology["lat_end"]
            lon_start = topology["lon_start"]
            lon_end = topology["lon_end"]

            # Insert patch into global grid
            biomass_global = biomass_global.at[lat_start:lat_end, lon_start:lon_end].set(patch)

        return biomass_global

    def _redistribute_biomass(self, biomass_global: jnp.ndarray) -> None:
        """Redistribute global biomass field to all workers.

        Extracts patches from the global biomass grid and distributes
        them to the corresponding workers according to topology.

        Args:
            biomass_global: Global biomass array with shape (global_nlat, global_nlon).

        Raises:
            ValueError: If transport not enabled or topology not available.
        """
        if not self.transport_enabled:
            raise ValueError("Cannot redistribute biomass: transport not enabled")

        # Extract and send patches to workers
        futures = []
        for worker, topology in zip(self.workers, self.worker_topology, strict=True):
            lat_start = topology["lat_start"]
            lat_end = topology["lat_end"]
            lon_start = topology["lon_start"]
            lon_end = topology["lon_end"]

            # Extract patch from global grid
            patch = biomass_global[lat_start:lat_end, lon_start:lon_end]

            # Send to worker (non-blocking)
            futures.append(worker.set_biomass.remote(patch))

        # Wait for all workers to receive their biomass
        ray.get(futures)

    def _get_transport_forcings(self, forcings_ref: ray.ObjectRef | None) -> dict[str, jnp.ndarray]:
        """Extract transport forcings from global forcings.

        Args:
            forcings_ref: Ray ObjectRef to global forcings dict.

        Returns:
            Dictionary with:
                - 'u': Zonal velocity [m/s], shape (global_nlat, global_nlon)
                - 'v': Meridional velocity [m/s], shape (global_nlat, global_nlon)
                - 'D': Horizontal diffusivity [m²/s], scalar or array
                - 'mask': Ocean mask (1=ocean, 0=land), optional

        Note:
            This is a placeholder implementation. In Phase 3.6, this will be
            improved to properly interface with ForcingManager.

            For now, it attempts to extract u, v, D, mask from forcings_ref.
            If not available, it returns default values (zero velocity, default D).
        """
        # Get forcings if available
        forcings: dict[str, Any] = ray.get(forcings_ref) if forcings_ref is not None else {}

        # Ensure global_nlat and global_nlon are set (required for transport)
        assert self.global_nlat is not None and self.global_nlon is not None

        # Extract transport forcings
        u = forcings.get("u", jnp.zeros((self.global_nlat, self.global_nlon), dtype=jnp.float32))
        v = forcings.get("v", jnp.zeros((self.global_nlat, self.global_nlon), dtype=jnp.float32))

        # Diffusivity: try to get from forcings, params, or use default
        D = forcings.get("D")
        if D is None:
            D = self.forcing_params.get("horizontal_diffusivity", 1000.0)

        # Ocean mask (optional)
        mask = forcings.get("ocean_mask")

        return {"u": u, "v": v, "D": D, "mask": mask}

    def step(self) -> dict[str, Any]:
        """Execute one synchronized timestep across all workers.

        Workflow (without transport):
        1. Prepare forcings for current timestep (if forcing_manager provided)
        2. Launch step() on all workers in parallel (non-blocking)
        3. Wait for all workers to complete
        4. Collect and aggregate diagnostics

        Workflow (with transport, 5 phases):
        1. PHASE BIOLOGIE: Launch biology_step() on all workers in parallel
        2. COLLECTE BIOMASSE: Assemble global biomass grid from workers
        3. PHASE TRANSPORT: Execute transport_step() on TransportWorker
        4. REDISTRIBUTION: Distribute updated biomass to workers
        5. AGRÉGATION: Aggregate biology + transport diagnostics

        Returns:
            Dictionary with aggregated diagnostics:
            - 't': Current time
            - 'num_workers': Number of workers
            - 'diagnostics': List of per-worker diagnostics
            - Aggregated statistics (mean, min, max) for state variables
            - If transport enabled: transport diagnostics
        """
        # Prepare forcings for this timestep
        forcings_ref = None
        if self.forcing_manager is not None:
            forcings_ref = self.forcing_manager.prepare_timestep_distributed(
                time=self.t_current, params=self.forcing_params
            )

        # PHASE 1: Biology (parallel)
        if self.transport_enabled:
            # Use biology_step when transport is external
            futures = [worker.biology_step.remote(self.dt, forcings_ref) for worker in self.workers]
        else:
            # Use regular step when transport is local (legacy mode)
            futures = [worker.step.remote(self.dt, forcings_ref) for worker in self.workers]

        bio_diagnostics = ray.get(futures)

        # PHASE 2-4: Transport (if enabled)
        transport_diag = None
        if self.transport_enabled:
            assert self.transport_worker is not None  # For type checking

            # PHASE 2: Collect global biomass
            biomass_global = self._collect_global_biomass()

            # PHASE 3: Transport step
            # Get transport forcings (u, v, D, mask)
            transport_forcings = self._get_transport_forcings(forcings_ref)

            # Execute transport on global grid
            transport_result = ray.get(
                self.transport_worker.transport_step.remote(
                    biomass=biomass_global,
                    u=transport_forcings["u"],
                    v=transport_forcings["v"],
                    D=transport_forcings["D"],
                    dt=self.dt,
                    mask=transport_forcings.get("mask"),
                )
            )

            biomass_global = transport_result["biomass"]
            transport_diag = transport_result["diagnostics"]

            # PHASE 4: Redistribute biomass
            self._redistribute_biomass(biomass_global)

        # Update current time
        self.t_current += self.dt

        # PHASE 5: Aggregate diagnostics
        aggregated = self._aggregate_diagnostics(bio_diagnostics)

        # Add transport diagnostics if available
        if transport_diag is not None:
            aggregated["transport"] = transport_diag

        return aggregated

    def _aggregate_diagnostics(self, worker_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate diagnostics from all workers.

        Args:
            worker_diagnostics: List of diagnostics from each worker.

        Returns:
            Dictionary with:
            - 't': Current simulation time
            - 'num_workers': Number of workers
            - 'diagnostics': Original per-worker diagnostics
            - For each state variable 'X':
                - 'X_mean': Mean across all workers
                - 'X_min': Minimum across all workers
                - 'X_max': Maximum across all workers
        """
        result: dict[str, Any] = {
            "t": self.t_current,
            "num_workers": len(worker_diagnostics),
            "diagnostics": worker_diagnostics,
        }

        # Identify all state variables (e.g., "biomass_mean")
        # by looking at first worker's diagnostics
        if worker_diagnostics:
            first_diag = worker_diagnostics[0]
            state_vars = [key.replace("_mean", "") for key in first_diag if key.endswith("_mean")]

            # Aggregate each state variable
            for var in state_vars:
                key = f"{var}_mean"
                values = [diag[key] for diag in worker_diagnostics if key in diag]

                if values:
                    result[f"{var}_global_mean"] = sum(values) / len(values)
                    result[f"{var}_global_min"] = min(values)
                    result[f"{var}_global_max"] = max(values)

        return result

    def run(self) -> list[dict[str, Any]]:
        """Run simulation from t=0 to t=t_max.

        Executes timesteps using the event queue until reaching t_max.

        Returns:
            List of aggregated diagnostics for each timestep.
        """
        all_diagnostics: list[dict[str, Any]] = []

        # Use small epsilon for floating point comparison
        eps = 1e-10

        while self.t_current < self.t_max - eps:
            # Execute one step
            diagnostics = self.step()
            all_diagnostics.append(diagnostics)

            # Schedule next step (if needed)
            if self.t_current < self.t_max - eps:
                self._schedule_event(time=self.t_current + self.dt, event_type="step", data={})

        return all_diagnostics

    def get_current_time(self) -> float:
        """Get current simulation time.

        Returns:
            Current time.
        """
        return self.t_current

    def get_worker_states(self) -> list[dict[str, Any]]:
        """Retrieve current state from all workers.

        Returns:
            List of state dictionaries, one per worker.
        """
        futures = [worker.get_state.remote() for worker in self.workers]
        states: list[dict[str, Any]] = ray.get(futures)
        return states

    def __repr__(self) -> str:
        """String representation of scheduler."""
        return (
            f"EventScheduler(workers={len(self.workers)}, "
            f"dt={self.dt}, t_max={self.t_max}, t_current={self.t_current:.2f})"
        )
