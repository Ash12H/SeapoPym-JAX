"""Benchmark tests for zooplankton kernel functions.

This module benchmarks the three core zooplankton units:
- age_production: Ages production with NPP source and recruitment absorption
- compute_recruitment: Calculates recruitment from production
- update_biomass: Updates adult biomass with recruitment and mortality

Key considerations for JAX benchmarking:
1. Use .block_until_ready() to ensure asynchronous execution completes
2. Warm-up phase to trigger JIT compilation before timing
3. Multiple iterations to get stable measurements
"""

import jax.numpy as jnp
import pytest

from seapopym_message.kernels.zooplankton import (
    age_production,
    compute_recruitment,
    update_biomass,
)


@pytest.fixture
def zooplankton_params():
    """Standard zooplankton parameters."""
    return {
        "n_ages": 11,
        "E": 0.1668,  # Transfer efficiency NPP -> production
    }


@pytest.fixture
def forcing_data_small():
    """Small grid forcings (10x10) for quick benchmarks."""
    nlat, nlon = 10, 10
    return {
        "npp": jnp.ones((nlat, nlon)) * 5.0,
        "tau_r": jnp.ones((nlat, nlon)) * 3.45,
        "mortality": jnp.ones((nlat, nlon)) * 0.01,
    }


@pytest.fixture
def forcing_data_medium():
    """Medium grid forcings (100x100) for realistic benchmarks."""
    nlat, nlon = 100, 100
    return {
        "npp": jnp.ones((nlat, nlon)) * 5.0,
        "tau_r": jnp.ones((nlat, nlon)) * 3.45,
        "mortality": jnp.ones((nlat, nlon)) * 0.01,
    }


@pytest.fixture
def forcing_data_large():
    """Large grid forcings (360x720) for performance testing."""
    nlat, nlon = 360, 720
    return {
        "npp": jnp.ones((nlat, nlon)) * 5.0,
        "tau_r": jnp.ones((nlat, nlon)) * 3.45,
        "mortality": jnp.ones((nlat, nlon)) * 0.01,
    }


class TestZooplanktonBenchmarks:
    """Benchmark suite for zooplankton kernel functions."""

    # =========================================================================
    # age_production benchmarks
    # =========================================================================

    def test_benchmark_age_production_small(
        self, benchmark, zooplankton_params, forcing_data_small
    ):
        """Benchmark age_production on small grid (10x10)."""
        nlat, nlon = 10, 10
        n_ages = zooplankton_params["n_ages"]
        production = jnp.zeros((n_ages, nlat, nlon))

        # Access the underlying function from the Unit object
        age_production_func = age_production.func

        # Warm-up: trigger JIT compilation
        _ = age_production_func(production, 1.0, zooplankton_params, forcing_data_small)

        # Benchmark with proper JAX synchronization
        def run():
            result = age_production_func(production, 1.0, zooplankton_params, forcing_data_small)
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (n_ages, nlat, nlon)

    def test_benchmark_age_production_medium(
        self, benchmark, zooplankton_params, forcing_data_medium
    ):
        """Benchmark age_production on medium grid (100x100)."""
        nlat, nlon = 100, 100
        n_ages = zooplankton_params["n_ages"]
        production = jnp.zeros((n_ages, nlat, nlon))

        age_production_func = age_production.func

        # Warm-up
        _ = age_production_func(production, 1.0, zooplankton_params, forcing_data_medium)

        def run():
            result = age_production_func(production, 1.0, zooplankton_params, forcing_data_medium)
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (n_ages, nlat, nlon)

    def test_benchmark_age_production_large(
        self, benchmark, zooplankton_params, forcing_data_large
    ):
        """Benchmark age_production on large grid (360x720)."""
        nlat, nlon = 360, 720
        n_ages = zooplankton_params["n_ages"]
        production = jnp.zeros((n_ages, nlat, nlon))

        age_production_func = age_production.func

        # Warm-up
        _ = age_production_func(production, 1.0, zooplankton_params, forcing_data_large)

        def run():
            result = age_production_func(production, 1.0, zooplankton_params, forcing_data_large)
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (n_ages, nlat, nlon)

    # =========================================================================
    # compute_recruitment benchmarks
    # =========================================================================

    def test_benchmark_compute_recruitment_small(
        self, benchmark, zooplankton_params, forcing_data_small
    ):
        """Benchmark compute_recruitment on small grid (10x10)."""
        nlat, nlon = 10, 10
        n_ages = zooplankton_params["n_ages"]
        production = jnp.ones((n_ages, nlat, nlon)) * 2.0

        compute_recruitment_func = compute_recruitment.func

        # Warm-up
        _ = compute_recruitment_func(production, 1.0, zooplankton_params, forcing_data_small)

        def run():
            result = compute_recruitment_func(
                production, 1.0, zooplankton_params, forcing_data_small
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_compute_recruitment_medium(
        self, benchmark, zooplankton_params, forcing_data_medium
    ):
        """Benchmark compute_recruitment on medium grid (100x100)."""
        nlat, nlon = 100, 100
        n_ages = zooplankton_params["n_ages"]
        production = jnp.ones((n_ages, nlat, nlon)) * 2.0

        compute_recruitment_func = compute_recruitment.func

        # Warm-up
        _ = compute_recruitment_func(production, 1.0, zooplankton_params, forcing_data_medium)

        def run():
            result = compute_recruitment_func(
                production, 1.0, zooplankton_params, forcing_data_medium
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_compute_recruitment_large(
        self, benchmark, zooplankton_params, forcing_data_large
    ):
        """Benchmark compute_recruitment on large grid (360x720)."""
        nlat, nlon = 360, 720
        n_ages = zooplankton_params["n_ages"]
        production = jnp.ones((n_ages, nlat, nlon)) * 2.0

        compute_recruitment_func = compute_recruitment.func

        # Warm-up
        _ = compute_recruitment_func(production, 1.0, zooplankton_params, forcing_data_large)

        def run():
            result = compute_recruitment_func(
                production, 1.0, zooplankton_params, forcing_data_large
            )
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    # =========================================================================
    # update_biomass benchmarks
    # =========================================================================

    def test_benchmark_update_biomass_small(self, benchmark, forcing_data_small):
        """Benchmark update_biomass on small grid (10x10)."""
        nlat, nlon = 10, 10
        biomass = jnp.ones((nlat, nlon)) * 100.0
        recruitment = jnp.ones((nlat, nlon)) * 5.0

        update_biomass_func = update_biomass.func

        # Warm-up
        _ = update_biomass_func(biomass, recruitment, 1.0, {}, forcing_data_small)

        def run():
            result = update_biomass_func(biomass, recruitment, 1.0, {}, forcing_data_small)
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_update_biomass_medium(self, benchmark, forcing_data_medium):
        """Benchmark update_biomass on medium grid (100x100)."""
        nlat, nlon = 100, 100
        biomass = jnp.ones((nlat, nlon)) * 100.0
        recruitment = jnp.ones((nlat, nlon)) * 5.0

        update_biomass_func = update_biomass.func

        # Warm-up
        _ = update_biomass_func(biomass, recruitment, 1.0, {}, forcing_data_medium)

        def run():
            result = update_biomass_func(biomass, recruitment, 1.0, {}, forcing_data_medium)
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)

    def test_benchmark_update_biomass_large(self, benchmark, forcing_data_large):
        """Benchmark update_biomass on large grid (360x720)."""
        nlat, nlon = 360, 720
        biomass = jnp.ones((nlat, nlon)) * 100.0
        recruitment = jnp.ones((nlat, nlon)) * 5.0

        update_biomass_func = update_biomass.func

        # Warm-up
        _ = update_biomass_func(biomass, recruitment, 1.0, {}, forcing_data_large)

        def run():
            result = update_biomass_func(biomass, recruitment, 1.0, {}, forcing_data_large)
            result.block_until_ready()
            return result

        result = benchmark(run)
        assert result.shape == (nlat, nlon)
