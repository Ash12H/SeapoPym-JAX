"""High-level simulation setup and execution utilities.

This module provides convenient functions to set up and run complete
distributed simulations with minimal boilerplate.
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import ray

from seapopym_message.core.kernel import Kernel
from seapopym_message.distributed.scheduler import EventScheduler
from seapopym_message.distributed.worker import CellWorker2D
from seapopym_message.utils.domain import split_domain_2d, split_domain_2d_periodic_lon
from seapopym_message.utils.grid import GridInfo


def create_distributed_simulation(
    grid: GridInfo,
    kernel: Kernel,
    params: dict[str, Any],
    num_workers_lat: int = 2,
    num_workers_lon: int = 2,
    periodic_lon: bool = False,
) -> tuple[list[ray.actor.ActorHandle], list[dict[str, Any]]]:
    """Create a distributed simulation with workers and domain decomposition.

    Args:
        grid: Global grid information.
        kernel: Computational kernel to execute on each worker.
        params: Model parameters.
        num_workers_lat: Number of workers in latitude direction.
        num_workers_lon: Number of workers in longitude direction.
        periodic_lon: If True, use periodic boundary conditions in longitude.

    Returns:
        Tuple of (workers, patches):
        - workers: List of CellWorker2D Ray actor handles.
        - patches: List of patch information dictionaries.

    Example:
        >>> import ray
        >>> ray.init()
        >>> from seapopym_message.core.kernel import Kernel
        >>> from seapopym_message.core.blueprint import Blueprint
        >>> from seapopym_message.kernels.biology import compute_growth
        >>> from seapopym_message.utils.grid import GridInfo
        >>>
        >>> grid = GridInfo(0, 10, 0, 20, nlat=20, nlon=40)
        >>>
        >>> # Build Kernel via Blueprint
        >>> bp = Blueprint()
        >>> bp.add_unit(compute_growth)
        >>> kernel = Kernel(bp.build())
        >>>
        >>> params = {"R": 10.0, "lambda": 0.1}
        >>>
        >>> workers, patches = create_distributed_simulation(
        ...     grid, kernel, params, num_workers_lat=2, num_workers_lon=2
        ... )
        >>> len(workers)
        4
    """
    # Split domain
    if periodic_lon:
        patches = split_domain_2d_periodic_lon(
            nlat_global=grid.nlat,
            nlon_global=grid.nlon,
            num_workers_lat=num_workers_lat,
            num_workers_lon=num_workers_lon,
        )
    else:
        patches = split_domain_2d(
            nlat_global=grid.nlat,
            nlon_global=grid.nlon,
            num_workers_lat=num_workers_lat,
            num_workers_lon=num_workers_lon,
        )

    # Create workers
    workers = []
    for patch in patches:
        worker = CellWorker2D.remote(  # type: ignore[attr-defined]
            worker_id=patch["worker_id"],
            grid_info=grid,
            lat_start=patch["lat_start"],
            lat_end=patch["lat_end"],
            lon_start=patch["lon_start"],
            lon_end=patch["lon_end"],
            kernel=kernel,
            params=params,
        )
        workers.append(worker)

    # Set up neighbor connections
    for i, patch in enumerate(patches):
        neighbors = {}
        for direction in ["north", "south", "east", "west"]:
            neighbor_id = patch["neighbors"][direction]
            if neighbor_id is not None:
                neighbors[direction] = workers[neighbor_id]
            else:
                neighbors[direction] = None

        workers[i].set_neighbors.remote(neighbors)

    return workers, patches


def initialize_workers(
    workers: list[ray.actor.ActorHandle],
    patches: list[dict[str, Any]],
    initial_state_fn: Callable[[int, int, int, int], dict[str, jnp.ndarray]],
) -> None:
    """Initialize state for all workers.

    Args:
        workers: List of CellWorker2D actors.
        patches: List of patch information.
        initial_state_fn: Function that takes (lat_start, lat_end, lon_start, lon_end)
                         and returns initial state dictionary.

    Example:
        >>> def uniform_biomass(lat_start, lat_end, lon_start, lon_end):
        ...     nlat = lat_end - lat_start
        ...     nlon = lon_end - lon_start
        ...     return {"biomass": jnp.ones((nlat, nlon)) * 10.0}
        >>>
        >>> initialize_workers(workers, patches, uniform_biomass)
    """
    futures = []
    for worker, patch in zip(workers, patches, strict=True):
        initial_state = initial_state_fn(
            patch["lat_start"],
            patch["lat_end"],
            patch["lon_start"],
            patch["lon_end"],
        )
        future = worker.set_initial_state.remote(initial_state)
        futures.append(future)

    # Wait for all initializations to complete
    ray.get(futures)


def run_simulation(
    workers: list[ray.actor.ActorHandle],
    dt: float,
    t_max: float,
) -> list[dict[str, Any]]:
    """Run simulation until t_max.

    Args:
        workers: List of CellWorker2D actors.
        dt: Timestep size.
        t_max: Maximum simulation time.

    Returns:
        List of aggregated diagnostics for each timestep.

    Example:
        >>> diagnostics = run_simulation(workers, dt=0.1, t_max=10.0)
        >>> len(diagnostics)  # 100 timesteps
        100
    """
    scheduler = EventScheduler(workers=workers, dt=dt, t_max=t_max)
    return scheduler.run()


def get_global_state(
    workers: list[ray.actor.ActorHandle],
    patches: list[dict[str, Any]],
) -> dict[str, jnp.ndarray]:
    """Assemble global state from all workers.

    Args:
        workers: List of CellWorker2D actors.
        patches: List of patch information.

    Returns:
        Dictionary with global arrays for each state variable.

    Example:
        >>> global_state = get_global_state(workers, patches)
        >>> global_state["biomass"].shape
        (20, 40)  # Global grid shape
    """
    # Get states from all workers
    futures = [worker.get_state.remote() for worker in workers]
    worker_states = ray.get(futures)

    # Determine global grid size from patches
    nlat_global = max(p["lat_end"] for p in patches)
    nlon_global = max(p["lon_end"] for p in patches)

    # Get state variable names from first worker
    state_vars = list(worker_states[0].keys())

    # Initialize global arrays
    global_state = {}
    for var in state_vars:
        global_state[var] = jnp.zeros((nlat_global, nlon_global))

    # Fill in each patch
    for worker_state, patch in zip(worker_states, patches, strict=True):
        lat_start = patch["lat_start"]
        lat_end = patch["lat_end"]
        lon_start = patch["lon_start"]
        lon_end = patch["lon_end"]

        for var in state_vars:
            global_state[var] = (
                global_state[var].at[lat_start:lat_end, lon_start:lon_end].set(worker_state[var])
            )

    return global_state


def setup_and_run(
    grid: GridInfo,
    kernel: Kernel,
    params: dict[str, Any],
    initial_state_fn: Callable[[int, int, int, int], dict[str, jnp.ndarray]],
    dt: float,
    t_max: float,
    num_workers_lat: int = 2,
    num_workers_lon: int = 2,
    periodic_lon: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, jnp.ndarray]]:
    """Complete end-to-end simulation setup and execution.

    This is a convenience function that combines all steps:
    1. Create distributed simulation
    2. Initialize workers
    3. Run simulation
    4. Get final global state

    Args:
        grid: Global grid information.
        kernel: Computational kernel.
        params: Model parameters.
        initial_state_fn: Function to generate initial state for each patch.
        dt: Timestep size.
        t_max: Maximum simulation time.
        num_workers_lat: Number of workers in latitude.
        num_workers_lon: Number of workers in longitude.
        periodic_lon: Use periodic longitude boundaries.

    Returns:
        Tuple of (diagnostics, final_state):
        - diagnostics: List of aggregated diagnostics.
        - final_state: Global state at t_max.

    Example:
        >>> grid = GridInfo(0, 10, 0, 20, nlat=20, nlon=40)
        >>> kernel = Kernel([compute_growth])
        >>> params = {"R": 10.0, "lambda": 0.1}
        >>> initial_fn = lambda *args: {"biomass": jnp.zeros((args[1]-args[0], args[3]-args[2]))}
        >>>
        >>> diagnostics, final_state = setup_and_run(
        ...     grid, kernel, params, initial_fn, dt=0.1, t_max=1.0
        ... )
    """
    # Create simulation
    workers, patches = create_distributed_simulation(
        grid=grid,
        kernel=kernel,
        params=params,
        num_workers_lat=num_workers_lat,
        num_workers_lon=num_workers_lon,
        periodic_lon=periodic_lon,
    )

    # Initialize
    initialize_workers(workers, patches, initial_state_fn)

    # Run
    diagnostics = run_simulation(workers, dt=dt, t_max=t_max)

    # Get final state
    final_state = get_global_state(workers, patches)

    return diagnostics, final_state
