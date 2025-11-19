"""EventScheduler: Orchestrates distributed simulation with priority queue.

The EventScheduler manages the temporal evolution of the simulation by:
- Maintaining a priority queue of events (timesteps)
- Coordinating parallel execution of CellWorker2D actors
- Synchronizing simulation time across all workers
- Collecting and aggregating diagnostics
- Optionally coordinating centralized transport via TransportWorker

Note:
    The scheduler is domain-agnostic. It uses TransportConfig to know which
    fields to transport, making it extensible to different simulation types.
"""

import heapq
import itertools
from typing import Any

import jax.numpy as jnp
import ray

from seapopym_message.distributed.transport_config import TransportConfig


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
        transport_config: TransportConfig | None = None,
        global_nlat: int | None = None,
        global_nlon: int | None = None,
    ) -> None:
        """Initialize scheduler with workers and time parameters.

        Args:
            workers: List of CellWorker2D actors.
            dt: Timestep size.
            t_max: Maximum simulation time.
            forcing_manager: Optional ForcingManager for environmental forcings.
            forcing_params: Optional parameters for derived forcings and field dimensions.
            transport_worker: Optional TransportWorker actor for centralized transport.
            transport_config: Optional TransportConfig specifying fields to transport.
                             If provided with transport_worker, enables centralized transport.
            global_nlat: Global grid latitude dimension.
            global_nlon: Global grid longitude dimension.
        """
        self.workers = workers
        self.dt = dt
        self.t_max = t_max
        self.t_current = 0.0
        self.forcing_manager = forcing_manager
        self.forcing_params = forcing_params if forcing_params is not None else {}

        # Transport configuration
        self.transport_worker = transport_worker
        self.transport_config = transport_config
        self.transport_enabled = transport_worker is not None and transport_config is not None

        # Global grid dimensions (for transport)
        self.global_nlat = global_nlat
        self.global_nlon = global_nlon

        if self.transport_enabled:
            if self.global_nlat is None or self.global_nlon is None:
                raise ValueError("global_nlat and global_nlon required when transport is enabled")
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

    def _collect_global_field(self, field_name: str, field_shape: tuple[int, ...]) -> jnp.ndarray:
        """Assemble a global field from all workers.

        Generic method that collects a field from all workers and assembles
        the patches into a global grid according to worker topology.

        Args:
            field_name: Name of the field to collect.
            field_shape: Expected global shape of the field (including extra dimensions).
                        Example: (nlat, nlon) for 2D, (n_ages, nlat, nlon) for 3D.

        Returns:
            Global field array with the specified shape.

        Raises:
            ValueError: If field patches have incorrect shapes.
        """
        if not self.transport_enabled:
            raise ValueError(f"Cannot collect field '{field_name}': transport not enabled")

        # Collect field from all workers in parallel
        futures = [worker.get_field.remote(field_name) for worker in self.workers]
        patches = ray.get(futures)

        # Initialize global field
        field_global = jnp.zeros(field_shape, dtype=jnp.float32)

        # Assemble patches into global grid
        for patch, topology in zip(patches, self.worker_topology, strict=True):
            lat_start = topology["lat_start"]
            lat_end = topology["lat_end"]
            lon_start = topology["lon_start"]
            lon_end = topology["lon_end"]

            # Check if patch is empty (field might not exist in worker state)
            if patch.size == 0:
                raise ValueError(
                    f"Worker returned empty patch for field '{field_name}'. "
                    f"Field may not exist in worker state."
                )

            # Insert patch into global grid
            # Handle different dimensionalities
            if len(field_shape) == 2:
                # 2D field (nlat, nlon)
                field_global = field_global.at[lat_start:lat_end, lon_start:lon_end].set(patch)
            else:
                # Multi-dimensional field (extra_dims..., nlat, nlon)
                # Use ellipsis to handle arbitrary number of leading dimensions
                field_global = field_global.at[..., lat_start:lat_end, lon_start:lon_end].set(patch)

        return field_global

    def _redistribute_field(self, field_name: str, field_global: jnp.ndarray) -> None:
        """Redistribute a global field to all workers.

        Generic method that extracts patches from a global field and distributes
        them to workers according to topology.

        Args:
            field_name: Name of the field to redistribute.
            field_global: Global field array.

        Raises:
            ValueError: If transport not enabled.
        """
        if not self.transport_enabled:
            raise ValueError(f"Cannot redistribute field '{field_name}': transport not enabled")

        # Extract and send patches to workers
        futures = []
        for worker, topology in zip(self.workers, self.worker_topology, strict=True):
            lat_start = topology["lat_start"]
            lat_end = topology["lat_end"]
            lon_start = topology["lon_start"]
            lon_end = topology["lon_end"]

            # Extract patch from global grid
            # Handle different dimensionalities
            if field_global.ndim == 2:
                # 2D field
                patch = field_global[lat_start:lat_end, lon_start:lon_end]
            else:
                # Multi-dimensional field (extra_dims..., nlat, nlon)
                patch = field_global[..., lat_start:lat_end, lon_start:lon_end]

            # Send to worker (non-blocking)
            futures.append(worker.set_field.remote(field_name, patch))

        # Wait for all workers to receive the field
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

        # Ocean mask (optional): try forcings first, then forcing_params
        mask = forcings.get("ocean_mask")
        if mask is None:
            mask = self.forcing_params.get("mask")

        return {"u": u, "v": v, "D": D, "mask": mask}

    def step(self) -> dict[str, Any]:
        """Execute one synchronized timestep across all workers.

        Workflow (without transport):
        1. Prepare forcings for current timestep (if forcing_manager provided)
        2. Launch step() on all workers in parallel (non-blocking)
        3. Wait for all workers to complete
        4. Collect and aggregate diagnostics

        Workflow (with transport, generic for any fields):
        1. PHASE LOCAL: Launch step() on all workers in parallel
        2. PHASE COLLECT: For each field in transport_config, assemble global grid
        3. PHASE TRANSPORT: For each field, execute transport:
           - If 2D field: transport directly
           - If N-D field: loop over all non-spatial dimensions and transport each slice
        4. PHASE REDISTRIBUTE: Distribute updated fields back to workers
        5. PHASE AGGREGATE: Aggregate diagnostics

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

        # PHASE 1: Local computation on all workers (parallel)
        futures = [worker.step.remote(self.dt, forcings_ref) for worker in self.workers]
        bio_diagnostics = ray.get(futures)

        # PHASE 2-4: Transport (if enabled and configured)
        transport_diag = None
        if self.transport_enabled and self.transport_config is not None:
            assert self.transport_worker is not None  # For type checking

            # Get transport forcings (u, v, D, mask)
            transport_forcings = self._get_transport_forcings(forcings_ref)

            # Process each field in transport configuration
            for field_name in self.transport_config.get_field_names():
                # PHASE 2: Compute field shape and collect global field
                field_dims = self.transport_config.get_field_dims(field_name)
                field_shape = self._compute_field_shape(field_name, field_dims)

                # Collect global field from all workers
                field_global = self._collect_global_field(field_name, field_shape)

                # PHASE 3: Transport field
                field_global, transport_diag = self._transport_field(
                    field_name=field_name,
                    field_global=field_global,
                    field_dims=field_dims,
                    transport_forcings=transport_forcings,
                    last_diag=transport_diag,
                )

                # PHASE 4: Redistribute field
                self._redistribute_field(field_name, field_global)

        # Update current time
        self.t_current += self.dt

        # PHASE 5: Aggregate diagnostics
        aggregated = self._aggregate_diagnostics(bio_diagnostics)

        # Add transport diagnostics if available
        if transport_diag is not None:
            aggregated["transport"] = transport_diag

        return aggregated

    def _compute_field_shape(self, field_name: str, field_dims: list[str]) -> tuple[int, ...]:
        """Compute global shape for a field based on its dimensions.

        Args:
            field_name: Name of the field.
            field_dims: List of dimension names for this field.

        Returns:
            Tuple representing the global shape.

        Example:
            >>> # Field with dims ['age', 'Y', 'X'] and n_ages=11
            >>> shape = self._compute_field_shape('production', ['age', 'Y', 'X'])
            >>> shape
            (11, 120, 360)  # (n_ages, nlat, nlon)
        """
        shape = []
        for dim in field_dims:
            if dim == "Y":
                assert self.global_nlat is not None
                shape.append(self.global_nlat)
            elif dim == "X":
                assert self.global_nlon is not None
                shape.append(self.global_nlon)
            else:
                # Non-spatial dimension: get size from forcing_params
                dim_size = self.forcing_params.get(dim)
                if dim_size is None:
                    raise ValueError(
                        f"Size for dimension '{dim}' of field '{field_name}' not found in forcing_params. "
                        f"Please provide '{dim}' in forcing_params."
                    )
                shape.append(dim_size)

        return tuple(shape)

    def _transport_field(
        self,
        field_name: str,
        field_global: jnp.ndarray,
        field_dims: list[str],
        transport_forcings: dict[str, jnp.ndarray],
        last_diag: dict[str, Any] | None,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        """Transport a field using the transport worker.

        Handles both 2D and N-dimensional fields by looping over non-spatial dimensions.

        Args:
            field_name: Name of the field being transported.
            field_global: Global field array.
            field_dims: List of dimension names for this field.
            transport_forcings: Dictionary with u, v, D, mask for transport.
            last_diag: Previous transport diagnostics (to keep last one).

        Returns:
            Tuple of (transported_field, diagnostics).
        """
        assert self.transport_config is not None

        # Get non-spatial dimensions
        non_spatial_dims = self.transport_config.get_non_spatial_dims(field_name)

        # Case 1: Pure 2D field (no extra dimensions)
        if len(non_spatial_dims) == 0:
            assert self.transport_worker is not None  # For mypy
            result = ray.get(
                self.transport_worker.transport_step.remote(
                    biomass=field_global,
                    u=transport_forcings["u"],
                    v=transport_forcings["v"],
                    D=transport_forcings["D"],
                    dt=self.dt,
                    mask=transport_forcings.get("mask"),
                )
            )
            return result["biomass"], result["diagnostics"]

        # Case 2: N-dimensional field (has extra dimensions)
        # Build list of dimension sizes for iteration
        dim_sizes = []
        for dim in non_spatial_dims:
            dim_idx = field_dims.index(dim)
            dim_sizes.append(field_global.shape[dim_idx])

        # Create all combinations of indices using itertools.product
        assert self.transport_worker is not None  # For mypy
        transported = jnp.zeros_like(field_global)
        diagnostics: dict[str, Any] = last_diag if last_diag is not None else {}

        for indices in itertools.product(*[range(size) for size in dim_sizes]):
            # Build slice for extracting 2D slice
            # We need to place indices in correct positions
            full_slice = [slice(None)] * len(field_global.shape)

            for i, dim in enumerate(non_spatial_dims):
                dim_idx = field_dims.index(dim)
                full_slice[dim_idx] = indices[i]

            # Extract 2D slice
            slice_2d = field_global[tuple(full_slice)]

            # Transport this slice
            result = ray.get(
                self.transport_worker.transport_step.remote(
                    biomass=slice_2d,
                    u=transport_forcings["u"],
                    v=transport_forcings["v"],
                    D=transport_forcings["D"],
                    dt=self.dt,
                    mask=transport_forcings.get("mask"),
                )
            )

            # Store transported slice back
            transported = transported.at[tuple(full_slice)].set(result["biomass"])
            diagnostics = result["diagnostics"]  # Update diagnostics

        return transported, diagnostics

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
