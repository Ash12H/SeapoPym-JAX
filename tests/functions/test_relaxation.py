"""Tests for the relaxation used by sponge layers (relaxation_tendency)."""

import jax.numpy as jnp
import numpy as np

from seapopym.functions.transport import relaxation_tendency


def test_a_zero_rate_gives_exactly_zero():
    state = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    target = jnp.array([[0.5, 5.0], [3.0, 0.0]])
    assert np.array_equal(np.asarray(relaxation_tendency(state, target, jnp.zeros((2, 2)))), np.zeros((2, 2)))


def test_the_field_moves_towards_the_target_at_the_rate():
    state = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([2.0, 2.0, 1.0])
    rate = jnp.array([0.5, 0.5, 0.25])
    assert np.allclose(np.asarray(relaxation_tendency(state, target, rate)), [0.5, 0.0, -0.5])


def test_an_explicit_step_approaches_the_target_without_overshoot():
    state, target, rate, dt = jnp.array(4.0), jnp.array(1.0), jnp.array(1.0 / 86400.0), 2400.0
    after = state + dt * relaxation_tendency(state, target, rate)
    assert 1.0 < float(after) < 4.0
