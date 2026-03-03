"""Tests for seapopym.optimization._common helpers."""

import jax.numpy as jnp
import pytest

from seapopym.optimization._common import (
    build_bounds_arrays,
    build_loss_fn,
    denormalize,
    flatten_params,
    normalize,
    resolve_metric,
    unflatten_params,
)


# ---------------------------------------------------------------------------
# flatten_params / unflatten_params roundtrip
# ---------------------------------------------------------------------------


class TestFlattenUnflatten:
    def test_scalar_roundtrip(self):
        params = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
        keys, flat = flatten_params(params)
        shapes = {k: jnp.atleast_1d(params[k]).shape for k in keys}
        restored = unflatten_params(keys, flat, shapes, params)
        for k in params:
            assert float(restored[k]) == pytest.approx(float(params[k]))

    def test_1d_roundtrip(self):
        params = {"v": jnp.array([1.0, 2.0, 3.0])}
        keys, flat = flatten_params(params)
        shapes = {k: jnp.atleast_1d(params[k]).shape for k in keys}
        restored = unflatten_params(keys, flat, shapes, params)
        assert jnp.allclose(restored["v"], params["v"])

    def test_2d_roundtrip(self):
        params = {"m": jnp.ones((2, 3))}
        keys, flat = flatten_params(params)
        shapes = {k: jnp.atleast_1d(params[k]).shape for k in keys}
        restored = unflatten_params(keys, flat, shapes, params)
        assert restored["m"].shape == (2, 3)
        assert jnp.allclose(restored["m"], params["m"])

    def test_mixed_shapes(self):
        params = {"a": jnp.array(5.0), "b": jnp.array([1.0, 2.0]), "c": jnp.ones((2, 2))}
        keys, flat = flatten_params(params)
        assert flat.shape == (7,)  # 1 + 2 + 4
        shapes = {k: jnp.atleast_1d(params[k]).shape for k in keys}
        restored = unflatten_params(keys, flat, shapes, params)
        assert float(restored["a"]) == pytest.approx(5.0)
        assert jnp.allclose(restored["b"], params["b"])
        assert jnp.allclose(restored["c"], params["c"])

    def test_keys_sorted(self):
        params = {"z": jnp.array(1.0), "a": jnp.array(2.0)}
        keys, _ = flatten_params(params)
        assert keys == ["a", "z"]


# ---------------------------------------------------------------------------
# build_bounds_arrays
# ---------------------------------------------------------------------------


class TestBuildBoundsArrays:
    def test_valid_bounds(self):
        params = {"x": jnp.array(1.0), "y": jnp.array(2.0)}
        bounds = {"x": (0.0, 10.0), "y": (-1.0, 1.0)}
        keys = sorted(params.keys())
        lower, upper = build_bounds_arrays(keys, params, bounds)
        assert float(lower[0]) == pytest.approx(0.0)
        assert float(upper[0]) == pytest.approx(10.0)
        assert float(lower[1]) == pytest.approx(-1.0)
        assert float(upper[1]) == pytest.approx(1.0)

    def test_inverted_bounds_raises(self):
        params = {"x": jnp.array(1.0)}
        bounds = {"x": (10.0, 0.0)}
        keys = sorted(params.keys())
        with pytest.raises(ValueError, match="Invalid bounds for 'x'"):
            build_bounds_arrays(keys, params, bounds)

    def test_equal_bounds_raises(self):
        params = {"x": jnp.array(1.0)}
        bounds = {"x": (5.0, 5.0)}
        keys = sorted(params.keys())
        with pytest.raises(ValueError, match="Invalid bounds for 'x'"):
            build_bounds_arrays(keys, params, bounds)

    def test_missing_key_defaults(self):
        params = {"x": jnp.array(0.5), "y": jnp.array(0.5)}
        bounds = {"x": (0.0, 10.0)}
        keys = sorted(params.keys())
        lower, upper = build_bounds_arrays(keys, params, bounds)
        # y has no bounds → defaults to [0, 1]
        assert float(lower[1]) == pytest.approx(0.0)
        assert float(upper[1]) == pytest.approx(1.0)

    def test_array_param_bounds_repeated(self):
        params = {"v": jnp.array([1.0, 2.0, 3.0])}
        bounds = {"v": (0.0, 5.0)}
        keys = sorted(params.keys())
        lower, upper = build_bounds_arrays(keys, params, bounds)
        assert lower.shape == (3,)
        assert upper.shape == (3,)
        assert jnp.allclose(lower, jnp.array([0.0, 0.0, 0.0]))
        assert jnp.allclose(upper, jnp.array([5.0, 5.0, 5.0]))


# ---------------------------------------------------------------------------
# normalize / denormalize roundtrip
# ---------------------------------------------------------------------------


class TestNormalizeDenormalize:
    def test_roundtrip(self):
        flat = jnp.array([2.0, 5.0, 8.0])
        lower = jnp.array([0.0, 0.0, 0.0])
        upper = jnp.array([10.0, 10.0, 10.0])
        norm = normalize(flat, lower, upper)
        assert jnp.allclose(norm, jnp.array([0.2, 0.5, 0.8]))
        restored = denormalize(norm, lower, upper)
        assert jnp.allclose(restored, flat)

    def test_boundary_values(self):
        lower = jnp.array([0.0])
        upper = jnp.array([10.0])
        assert float(normalize(lower, lower, upper)[0]) == pytest.approx(0.0)
        assert float(normalize(upper, lower, upper)[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# resolve_metric
# ---------------------------------------------------------------------------


class TestResolveMetric:
    def test_known_names(self):
        for name in ("mse", "rmse", "nrmse", "nrmse_std", "nrmse_mean", "nrmse_minmax"):
            fn = resolve_metric(name)
            assert callable(fn)

    def test_custom_callable(self):
        def my_metric(p, o):
            return jnp.mean((p - o) ** 2)

        assert resolve_metric(my_metric) is my_metric

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown metric 'bad'"):
            resolve_metric("bad")


# ---------------------------------------------------------------------------
# build_loss_fn
# ---------------------------------------------------------------------------


class TestBuildLossFn:
    def test_returns_callable_producing_scalar(self):
        from seapopym.optimization.objective import Objective

        class _FakeRunner:
            def __call__(self, model, free_params):
                return {"out": free_params["x"] * jnp.ones(3)}

        class _FakeModel:
            parameters = {"x": jnp.array(1.0)}
            coords = {}

        obs = jnp.array([1.0, 1.0, 1.0])
        obj = Objective(observations=obs, transform=lambda o: o["out"])
        from seapopym.optimization._common import setup_objectives

        prepared = setup_objectives([(obj, "mse", 1.0)], {})
        loss_fn = build_loss_fn(_FakeRunner(), _FakeModel(), prepared, None)
        result = loss_fn({"x": jnp.array(2.0)})
        assert jnp.ndim(result) == 0
        assert float(result) == pytest.approx(1.0)  # mse of [2,2,2] vs [1,1,1]
