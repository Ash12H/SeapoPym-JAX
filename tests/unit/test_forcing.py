"""Unit tests for forcing module."""

import jax.numpy as jnp
import numpy as np
import pytest
import ray
import xarray as xr

from seapopym_message.forcing import (
    DerivedForcing,
    ForcingManager,
    derived_forcing,
)
from seapopym_message.forcing.derived import resolve_dependencies


@pytest.fixture
def sample_temperature_dataset():
    """Create a sample temperature dataset for testing."""
    # Create synthetic data: (time, depth, lat, lon)
    times = np.array([0.0, 3600.0, 7200.0])  # 0h, 1h, 2h in seconds
    depths = np.array([0, 150, 400])  # meters
    lats = np.linspace(-5, 5, 20)
    lons = np.linspace(-5, 5, 20)

    # Create temperature data (time, depth, lat, lon)
    temperature = np.random.rand(len(times), len(depths), len(lats), len(lons)) * 30  # 0-30°C

    ds = xr.Dataset(
        {
            "temperature": (["time", "depth", "lat", "lon"], temperature),
        },
        coords={
            "time": times,
            "depth": depths,
            "lat": lats,
            "lon": lons,
        },
    )

    return ds


@pytest.fixture
def sample_pp_dataset():
    """Create a sample primary production dataset for testing."""
    times = np.array([0.0, 3600.0, 7200.0])
    lats = np.linspace(-5, 5, 20)
    lons = np.linspace(-5, 5, 20)

    # Create PP data (time, lat, lon)
    pp = np.random.rand(len(times), len(lats), len(lons)) * 100  # 0-100 mg C/m³/day

    ds = xr.Dataset(
        {
            "primary_production": (["time", "lat", "lon"], pp),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )

    return ds


def test_forcing_manager_initialization(sample_temperature_dataset):
    """Test ForcingManager initialization with xarray Dataset."""
    # Set metadata in attrs
    sample_temperature_dataset.attrs["units"] = "°C"
    sample_temperature_dataset.attrs["interpolation_method"] = "linear"

    manager = ForcingManager(datasets={"temperature": sample_temperature_dataset})

    assert "temperature" in manager.datasets
    assert isinstance(manager.datasets["temperature"], xr.Dataset)
    assert manager.datasets["temperature"].attrs["units"] == "°C"


def test_interpolate_time(sample_temperature_dataset):
    """Test temporal interpolation."""
    sample_temperature_dataset.attrs["interpolation_method"] = "linear"

    manager = ForcingManager(datasets={"temperature": sample_temperature_dataset})

    # Interpolate at t=1800s (halfway between 0 and 3600)
    forcings = manager.prepare_timestep(time=1800.0)

    assert "temperature" in forcings
    assert forcings["temperature"].shape == (3, 20, 20)  # (depth, lat, lon)
    assert isinstance(forcings["temperature"], jnp.ndarray)


def test_interpolate_time_extrapolation_error(sample_temperature_dataset):
    """Test that extrapolation raises an error."""
    manager = ForcingManager(datasets={"temperature": sample_temperature_dataset})

    # Try to extrapolate beyond time range
    with pytest.raises(ValueError, match="outside the forcing data range"):
        manager.prepare_timestep(time=10000.0)  # Beyond [0, 7200]


def test_derived_forcing_decorator():
    """Test @derived_forcing decorator."""

    @derived_forcing(
        name="recruitment",
        inputs=["primary_production"],
        params=["transfer_coefficient"],
    )
    def compute_recruitment(primary_production, transfer_coefficient):
        """Compute recruitment from primary production."""
        return primary_production * transfer_coefficient

    assert isinstance(compute_recruitment, DerivedForcing)
    assert compute_recruitment.name == "recruitment"
    assert compute_recruitment.inputs == ["primary_production"]
    assert compute_recruitment.params == ["transfer_coefficient"]


def test_derived_forcing_compute():
    """Test DerivedForcing.compute()."""

    @derived_forcing(
        name="recruitment",
        inputs=["primary_production"],
        params=["transfer_coefficient"],
    )
    def compute_recruitment(primary_production, transfer_coefficient):
        """Compute recruitment."""
        return primary_production * transfer_coefficient

    # Prepare inputs
    pp = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    forcings = {"primary_production": pp}
    params = {"transfer_coefficient": 0.1}

    # Compute
    result = compute_recruitment.compute(forcings, params)

    expected = pp * 0.1
    assert jnp.allclose(result, expected)


def test_forcing_manager_with_derived(sample_pp_dataset):
    """Test ForcingManager with derived forcings."""
    # Create manager with base forcing
    manager = ForcingManager(datasets={"primary_production": sample_pp_dataset})

    # Register derived forcing
    @derived_forcing(
        name="recruitment",
        inputs=["primary_production"],
        params=["transfer_coefficient"],
    )
    def compute_recruitment(primary_production, transfer_coefficient):
        """Compute recruitment."""
        return primary_production * transfer_coefficient

    manager.register_derived(compute_recruitment)

    # Prepare forcings with params
    forcings = manager.prepare_timestep(time=1800.0, params={"transfer_coefficient": 0.15})

    assert "primary_production" in forcings
    assert "recruitment" in forcings
    assert forcings["recruitment"].shape == forcings["primary_production"].shape

    # Check computation
    expected_recruitment = forcings["primary_production"] * 0.15
    assert jnp.allclose(forcings["recruitment"], expected_recruitment)


def test_resolve_dependencies_simple():
    """Test dependency resolution with simple chain."""

    @derived_forcing(name="A", inputs=[], params=[])
    def compute_a():
        """Compute A."""
        return jnp.array([1.0])

    @derived_forcing(name="B", inputs=["A"], params=[])
    def compute_b(A):  # noqa: N803
        """Compute B from A."""
        return A * 2

    @derived_forcing(name="C", inputs=["B"], params=[])
    def compute_c(B):  # noqa: N803
        """Compute C from B."""
        return B * 3

    forcings = {"A": compute_a, "B": compute_b, "C": compute_c}

    order = resolve_dependencies(forcings)

    # A must come before B, B before C
    assert order.index("A") < order.index("B")
    assert order.index("B") < order.index("C")


def test_resolve_dependencies_complex():
    """Test dependency resolution with complex graph."""

    @derived_forcing(name="A", inputs=[], params=[])
    def compute_a():
        """Compute A."""
        return jnp.array([1.0])

    @derived_forcing(name="B", inputs=[], params=[])
    def compute_b():
        """Compute B."""
        return jnp.array([2.0])

    @derived_forcing(name="C", inputs=["A", "B"], params=[])
    def compute_c(A, B):  # noqa: N803
        """Compute C from A and B."""
        return A + B

    @derived_forcing(name="D", inputs=["C"], params=[])
    def compute_d(C):  # noqa: N803
        """Compute D from C."""
        return C * 2

    forcings = {"A": compute_a, "B": compute_b, "C": compute_c, "D": compute_d}

    order = resolve_dependencies(forcings)

    # A and B must come before C
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("C")
    # C must come before D
    assert order.index("C") < order.index("D")


def test_resolve_dependencies_circular():
    """Test that circular dependencies raise an error."""

    @derived_forcing(name="A", inputs=["B"], params=[])
    def compute_a(B):  # noqa: N803
        """Compute A from B."""
        return B * 2

    @derived_forcing(name="B", inputs=["A"], params=[])
    def compute_b(A):  # noqa: N803
        """Compute B from A."""
        return A * 2

    forcings = {"A": compute_a, "B": compute_b}

    with pytest.raises(ValueError, match="Circular dependency"):
        resolve_dependencies(forcings)


@pytest.mark.skipif(not ray.is_initialized(), reason="Ray not initialized")
def test_prepare_timestep_distributed(sample_temperature_dataset):
    """Test distributed forcing preparation with Ray object store."""
    if not ray.is_initialized():
        ray.init(num_cpus=2)

    try:
        manager = ForcingManager(datasets={"temperature": sample_temperature_dataset})

        # Prepare forcings and put in object store
        forcings_ref = manager.prepare_timestep_distributed(time=1800.0)

        # Retrieve from object store
        forcings = ray.get(forcings_ref)

        assert "temperature" in forcings
        assert isinstance(forcings["temperature"], jnp.ndarray)
    finally:
        if ray.is_initialized():
            ray.shutdown()
