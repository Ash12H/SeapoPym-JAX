"""EventScheduler: Orchestrates distributed simulation with priority queue.

The EventScheduler manages the temporal evolution of the simulation by:
- Maintaining a priority queue of events (timesteps)
- Coordinating parallel execution of CellWorker2D actors
- Synchronizing simulation time across all workers
- Collecting and aggregating diagnostics
"""

import heapq
from typing import Any

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

    Example:
        >>> import ray
        >>> ray.init()
        >>> workers = [CellWorker2D.remote(...) for _ in range(4)]
        >>> scheduler = EventScheduler(workers, dt=0.1, t_max=10.0)
        >>> diagnostics = scheduler.run()
        >>> len(diagnostics)  # 100 timesteps
        100
    """

    def __init__(
        self,
        workers: list[ray.actor.ActorHandle],
        dt: float,
        t_max: float,
        forcing_manager: Any = None,
        forcing_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize scheduler with workers and time parameters."""
        self.workers = workers
        self.dt = dt
        self.t_max = t_max
        self.t_current = 0.0
        self.forcing_manager = forcing_manager
        self.forcing_params = forcing_params if forcing_params is not None else {}

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

    def step(self) -> dict[str, Any]:
        """Execute one synchronized timestep across all workers.

        Workflow:
        1. Prepare forcings for current timestep (if forcing_manager provided)
        2. Launch step() on all workers in parallel (non-blocking)
        3. Wait for all workers to complete
        4. Collect and aggregate diagnostics

        Returns:
            Dictionary with aggregated diagnostics:
            - 't': Current time
            - 'num_workers': Number of workers
            - 'diagnostics': List of per-worker diagnostics
            - Aggregated statistics (mean, min, max) for state variables
        """
        # Prepare forcings for this timestep
        forcings_ref = None
        if self.forcing_manager is not None:
            forcings_ref = self.forcing_manager.prepare_timestep_distributed(
                time=self.t_current, params=self.forcing_params
            )

        # Launch all workers in parallel
        futures = [worker.step.remote(self.dt, forcings_ref) for worker in self.workers]

        # Wait for all to complete
        worker_diagnostics = ray.get(futures)

        # Update current time
        self.t_current += self.dt

        # Aggregate diagnostics
        aggregated = self._aggregate_diagnostics(worker_diagnostics)

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
