"""Pytest configuration and shared fixtures."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="session")
def jax_config() -> None:
    """Configure JAX for testing."""
    # Use CPU for tests
    jax.config.update("jax_platform_name", "cpu")
    # Enable 64-bit precision for numerical accuracy
    jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_params() -> dict:
    """Simple model parameters for testing."""
    return {
        "R": 10.0,  # Recruitment (g/m²/day)
        "lambda_0": 0.01,  # Base mortality rate (day⁻¹)
        "k": 0.05,  # Temperature sensitivity
        "D": 100.0,  # Diffusion coefficient (m²/day)
        "dx": 1000.0,  # Spatial resolution (m)
        "dy": 1000.0,
    }


@pytest.fixture
def grid_1d() -> dict:
    """Simple 1D grid for testing."""
    n = 10
    return {
        "biomass": jnp.ones(n) * 50.0,
        "temperature": jnp.ones(n) * 24.0,
    }


@pytest.fixture
def grid_2d() -> dict:
    """Simple 2D grid for testing."""
    nlat, nlon = 10, 15
    return {
        "biomass": jnp.ones((nlat, nlon)) * 50.0,
        "temperature": jnp.ones((nlat, nlon)) * 24.0,
    }


@pytest.fixture
def mock_forcing_data() -> dict:
    """Mock environmental forcing data."""
    nt, nlat, nlon = 12, 10, 15  # 12 months, 10x15 grid
    return {
        "temperature": np.random.randn(nt, nlat, nlon) * 2 + 24.0,  # ~24°C
        "velocity_u": np.random.randn(nt, nlat, nlon) * 0.5,  # m/s
        "velocity_v": np.random.randn(nt, nlat, nlon) * 0.5,
    }
