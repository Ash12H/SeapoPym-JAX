"""Tests for IPOP-CMA-ES."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from seapopym.optimization.ipop import IPOPResult, run_ipop_cmaes


class TestIPOPCMAES:
    """Test IPOP-CMA-ES on simple functions."""

    def test_unimodal_finds_minimum(self):
        """Single quadratic: should find the minimum."""

        def loss_fn(params):
            return (params["x"] - 3.0) ** 2

        result = run_ipop_cmaes(
            loss_fn,
            initial_params={"x": jnp.array(0.0)},
            bounds={"x": (-10.0, 10.0)},
            n_restarts=2,
            initial_popsize=16,
            n_generations=50,
        )
        assert isinstance(result, IPOPResult)
        assert len(result.modes) >= 1
        assert float(result.modes[0].params["x"]) == pytest.approx(3.0, abs=0.5)

    def test_bimodal_finds_two_modes(self):
        """Two wells: should find both modes with enough restarts."""

        def loss_fn(params):
            x = params["x"]
            # Two minima at x=-3 and x=3
            return (x**2 - 9.0) ** 2

        result = run_ipop_cmaes(
            loss_fn,
            initial_params={"x": jnp.array(0.0)},
            bounds={"x": (-10.0, 10.0)},
            n_restarts=6,
            initial_popsize=16,
            n_generations=80,
            distance_threshold=1.0,
            seed=42,
        )
        # Should find at least 2 distinct modes
        assert len(result.modes) >= 2
        # The two modes should be near -3 and 3
        mode_xs = sorted([float(m.params["x"]) for m in result.modes[:2]])
        assert mode_xs[0] == pytest.approx(-3.0, abs=0.5)
        assert mode_xs[1] == pytest.approx(3.0, abs=0.5)

    def test_population_doubles(self):
        """All restarts should run (population doubles)."""

        def loss_fn(params):
            return params["x"] ** 2

        result = run_ipop_cmaes(
            loss_fn,
            initial_params={"x": jnp.array(1.0)},
            bounds={"x": (-5.0, 5.0)},
            n_restarts=4,
            initial_popsize=8,
            n_generations=20,
        )
        assert result.n_restarts == 4
        assert len(result.all_results) == 4

    def test_modes_sorted_by_loss(self):
        """Modes should be sorted best-first."""

        def loss_fn(params):
            return (params["x"] - 1.0) ** 2

        result = run_ipop_cmaes(
            loss_fn,
            initial_params={"x": jnp.array(0.0)},
            bounds={"x": (-10.0, 10.0)},
            n_restarts=3,
            initial_popsize=16,
            n_generations=50,
        )
        for i in range(len(result.modes) - 1):
            assert result.modes[i].loss <= result.modes[i + 1].loss

    def test_2d_params(self):
        """Works with multiple parameters."""

        def loss_fn(params):
            return (params["x"] - 1.0) ** 2 + (params["y"] - 2.0) ** 2

        result = run_ipop_cmaes(
            loss_fn,
            initial_params={"x": jnp.array(0.0), "y": jnp.array(0.0)},
            bounds={"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
            n_restarts=2,
            initial_popsize=16,
            n_generations=50,
        )
        best = result.modes[0]
        assert float(best.params["x"]) == pytest.approx(1.0, abs=0.5)
        assert float(best.params["y"]) == pytest.approx(2.0, abs=0.5)
