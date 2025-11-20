"""Integration tests for forcing system with distributed workers."""

import jax.numpy as jnp
import numpy as np
import pytest
import ray
import xarray as xr

from seapopym_message import create_distributed_simulation, initialize_workers
from seapopym_message.core.kernel import Kernel
from seapopym_message.core.unit import unit
from seapopym_message.distributed.scheduler import EventScheduler
from seapopym_message.forcing import ForcingManager, derived_forcing
from seapopym_message.utils.grid import SphericalGridInfo


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for tests."""
    if not ray.is_initialized():
        ray.init(num_cpus=4)
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_forcing_with_unit(ray_context):
    """Test that Units can access forcings correctly."""

    # Create a Unit that uses forcings
    @unit(
        name="growth_with_forcing",
        inputs=["biomass"],
        outputs=["biomass"],
        scope="local",
        compiled=True,
        forcings=["recruitment"],
    )
    def compute_growth_with_forcing(biomass, dt, params, forcings):
        """Growth with recruitment from forcings."""
        R = forcings["recruitment"]
        lambda_val = params["lambda"]
        return biomass + (R - lambda_val * biomass) * dt

    # Create synthetic forcings
    times = np.array([0.0, 100.0])
    lats = np.linspace(-5, 5, 10)
    lons = np.linspace(-5, 5, 10)

    recruitment_data = np.ones((len(times), len(lats), len(lons))) * 5.0

    recruitment_ds = xr.Dataset(
        {"recruitment": (["time", "lat", "lon"], recruitment_data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    # Create ForcingManager
    recruitment_ds.attrs["interpolation_method"] = "linear"
    forcing_manager = ForcingManager(datasets={"recruitment": recruitment_ds})

    # Create grid and workers
    grid = SphericalGridInfo(lat_min=-5, lat_max=5, lon_min=-5, lon_max=5, nlat=10, nlon=10)

    kernel = Kernel([compute_growth_with_forcing])
    params = {"lambda": 0.1}

    workers, patches = create_distributed_simulation(
        grid=grid,
        kernel=kernel,
        params=params,
        num_workers_lat=2,
        num_workers_lon=2,
    )

    # Initialize state
    def initial_state(lat_start, lat_end, lon_start, lon_end):
        nlat = lat_end - lat_start
        nlon = lon_end - lon_start
        return {"biomass": jnp.ones((nlat, nlon)) * 10.0}

    initialize_workers(workers, patches, initial_state)

    # Create scheduler with forcing_manager
    scheduler = EventScheduler(
        workers=workers,
        dt=10.0,
        t_max=50.0,
        forcing_manager=forcing_manager,
    )

    # Run simulation
    diagnostics = scheduler.run()

    assert len(diagnostics) == 5  # 5 timesteps
    assert diagnostics[0]["t"] == 10.0
    assert diagnostics[-1]["t"] == 50.0

    # Biomass should have grown due to recruitment
    initial_biomass = 10.0
    final_biomass = diagnostics[-1]["biomass_global_mean"]

    # With R=5, lambda=0.1, equilibrium is R/lambda = 50
    # Should be moving towards 50 or reach it
    assert final_biomass > initial_biomass
    assert final_biomass <= 50.0  # At or approaching equilibrium


def test_derived_forcing_with_workers(ray_context):
    """Test derived forcings in distributed simulation."""

    # Base forcing
    times = np.array([0.0, 100.0])
    lats = np.linspace(-5, 5, 10)
    lons = np.linspace(-5, 5, 10)

    pp_data = np.ones((len(times), len(lats), len(lons))) * 100.0

    pp_ds = xr.Dataset(
        {"primary_production": (["time", "lat", "lon"], pp_data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    # Derived forcing
    @derived_forcing(
        name="recruitment",
        inputs=["primary_production"],
        params=["transfer_coefficient"],
    )
    def compute_recruitment(primary_production, transfer_coefficient):
        """Compute recruitment from PP."""
        return primary_production * transfer_coefficient

    # Unit that uses derived forcing
    @unit(
        name="growth_with_recruitment",
        inputs=["biomass"],
        outputs=["biomass"],
        scope="local",
        compiled=True,
        forcings=["recruitment"],
    )
    def compute_growth(biomass, dt, params, forcings):
        """Growth with recruitment."""
        R = forcings["recruitment"]
        lambda_val = params["lambda"]
        return biomass + (R - lambda_val * biomass) * dt

    # Setup forcing manager with derived forcing
    pp_ds.attrs["interpolation_method"] = "linear"
    forcing_manager = ForcingManager(datasets={"primary_production": pp_ds})
    forcing_manager.register_derived(compute_recruitment)

    # Create simulation
    grid = SphericalGridInfo(lat_min=-5, lat_max=5, lon_min=-5, lon_max=5, nlat=10, nlon=10)

    kernel = Kernel([compute_growth])
    params = {"lambda": 0.1}

    workers, patches = create_distributed_simulation(
        grid=grid,
        kernel=kernel,
        params=params,
        num_workers_lat=2,
        num_workers_lon=2,
    )

    # Initialize state
    def initial_state(lat_start, lat_end, lon_start, lon_end):
        nlat = lat_end - lat_start
        nlon = lon_end - lon_start
        return {"biomass": jnp.ones((nlat, nlon)) * 10.0}

    initialize_workers(workers, patches, initial_state)

    # Create scheduler with forcing params
    scheduler = EventScheduler(
        workers=workers,
        dt=10.0,
        t_max=50.0,
        forcing_manager=forcing_manager,
        forcing_params={"transfer_coefficient": 0.15},  # PP * 0.15 = 15
    )

    # Run simulation
    diagnostics = scheduler.run()

    assert len(diagnostics) == 5

    # Equilibrium should be R/lambda = 15/0.1 = 150
    final_biomass = diagnostics[-1]["biomass_global_mean"]

    # Should be growing towards equilibrium or reach it
    assert final_biomass > 10.0
    assert final_biomass <= 150.0


def test_forcing_without_forcing_manager(ray_context):
    """Test that simulation works without forcing_manager (backward compatibility)."""

    @unit(
        name="simple_growth",
        inputs=["biomass"],
        outputs=["biomass"],
        scope="local",
        compiled=True,
    )
    def compute_simple_growth(biomass, dt, params):
        """Simple growth without forcings."""
        R = params["R"]
        lambda_val = params["lambda"]
        return biomass + (R - lambda_val * biomass) * dt

    grid = SphericalGridInfo(lat_min=-5, lat_max=5, lon_min=-5, lon_max=5, nlat=10, nlon=10)

    kernel = Kernel([compute_simple_growth])
    params = {"R": 5.0, "lambda": 0.1}

    workers, patches = create_distributed_simulation(
        grid=grid,
        kernel=kernel,
        params=params,
        num_workers_lat=2,
        num_workers_lon=2,
    )

    def initial_state(lat_start, lat_end, lon_start, lon_end):
        nlat = lat_end - lat_start
        nlon = lon_end - lon_start
        return {"biomass": jnp.ones((nlat, nlon)) * 10.0}

    initialize_workers(workers, patches, initial_state)

    # Scheduler WITHOUT forcing_manager
    scheduler = EventScheduler(
        workers=workers,
        dt=10.0,
        t_max=50.0,
        # No forcing_manager
    )

    diagnostics = scheduler.run()

    assert len(diagnostics) == 5
    # Should still work as before
    final_biomass = diagnostics[-1]["biomass_global_mean"]
    assert final_biomass > 10.0
