"""Tests for Objective and PreparedObjective."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from seapopym.optimization.objective import Objective, PreparedObjective

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _model_coords() -> dict[str, np.ndarray]:
    """Fake model coordinate grid (T=4, Y=3, X=5)."""
    return {
        "T": np.arange(4),
        "Y": np.array([10.0, 20.0, 30.0]),
        "X": np.array([100.0, 110.0, 120.0, 130.0, 140.0]),
    }


def _model_outputs() -> dict[str, jnp.ndarray]:
    """Fake model outputs matching _model_coords shape."""
    return {"biomass": jnp.arange(60.0).reshape(4, 3, 5)}


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


class TestObjectiveInit:
    def test_target_mode(self):
        obs = xr.DataArray(np.zeros((2, 2)), dims=["Y", "X"])
        obj = Objective(observations=obs, target="biomass")
        assert obj.target == "biomass"
        assert obj.transform is None

    def test_transform_mode(self):
        obs = jnp.zeros(3)

        def fn(o):
            return o["biomass"].sum(axis=0).ravel()[:3]

        obj = Objective(observations=obs, transform=fn)
        assert obj.target is None
        assert obj.transform is fn

    def test_neither_target_nor_transform_raises(self):
        with pytest.raises(ValueError, match="requires either"):
            Objective(observations=jnp.zeros(3))

    def test_both_target_and_transform_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            Objective(
                observations=jnp.zeros(3),
                target="biomass",
                transform=lambda o: o["biomass"],
            )

    def test_repr_target(self):
        obj = Objective(observations=jnp.zeros(1), target="biomass")
        assert "target='biomass'" in repr(obj)

    def test_repr_transform(self):
        obj = Objective(observations=jnp.zeros(1), transform=lambda o: o["x"])
        assert "transform=<fn>" in repr(obj)


# ---------------------------------------------------------------------------
# Target mode — xarray
# ---------------------------------------------------------------------------


class TestSetupXarray:
    def test_subgrid_extraction(self):
        """xarray observations on a coordinate subset extracts correct values."""
        coords = _model_coords()
        outputs = _model_outputs()

        # Observe T=[0,2], Y=[10,30], X=[100,120]
        obs_data = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        obs_xr = xr.DataArray(
            obs_data,
            dims=["T", "Y", "X"],
            coords={"T": [0, 2], "Y": [10.0, 30.0], "X": [100.0, 120.0]},
        )

        obj = Objective(observations=obs_xr, target="biomass")
        prepared = obj.setup(coords)

        assert isinstance(prepared, PreparedObjective)
        assert prepared.obs_array.shape == (2, 2, 2)

        pred = prepared.extract_fn(outputs)
        # outputs["biomass"] shape (4,3,5)
        # T=[0,2] → idx [0,2], Y=[10,30] → idx [0,2], X=[100,120] → idx [0,2]
        # jnp.ix_([0,2],[0,2],[0,2]) extracts a (2,2,2) subgrid
        expected = outputs["biomass"][jnp.ix_(jnp.array([0, 2]), jnp.array([0, 2]), jnp.array([0, 2]))]
        np.testing.assert_array_equal(np.asarray(pred), np.asarray(expected))

    def test_single_dim_xarray(self):
        """xarray with only Y dimension."""
        coords = _model_coords()
        outputs = {"profile": jnp.array([10.0, 20.0, 30.0])}  # shape (3,)

        obs_xr = xr.DataArray(
            np.array([10.0, 30.0]),
            dims=["Y"],
            coords={"Y": [10.0, 30.0]},
        )

        obj = Objective(observations=obs_xr, target="profile")
        prepared = obj.setup(coords)

        pred = prepared.extract_fn(outputs)
        # Y=[10,30] → idx [0,2]
        expected = jnp.array([10.0, 30.0])
        np.testing.assert_array_almost_equal(np.asarray(pred), np.asarray(expected))


# ---------------------------------------------------------------------------
# Target mode — DataFrame
# ---------------------------------------------------------------------------


class TestSetupDataFrame:
    def test_sparse_point_extraction(self):
        """DataFrame observations extract sparse points correctly."""
        coords = _model_coords()
        outputs = _model_outputs()

        df = pd.DataFrame(
            {
                "T": [0, 2],
                "Y": [10.0, 30.0],
                "X": [100.0, 120.0],
                "biomass": [99.0, 88.0],
            }
        )

        obj = Objective(observations=df, target="biomass")
        prepared = obj.setup(coords)

        assert prepared.obs_array.shape == (2,)
        np.testing.assert_array_almost_equal(np.asarray(prepared.obs_array), [99.0, 88.0])

        pred = prepared.extract_fn(outputs)
        # Point (T=0,Y=0,X=0) → outputs["biomass"][0,0,0] = 0.0
        # Point (T=2,Y=2,X=2) → outputs["biomass"][2,2,2] = 2*15+2*5+2 = 42.0
        expected = jnp.array([0.0, 42.0])
        np.testing.assert_array_almost_equal(np.asarray(pred), np.asarray(expected))

    def test_missing_target_column_raises(self):
        """DataFrame without target column raises ValueError."""
        coords = _model_coords()
        df = pd.DataFrame({"T": [0], "Y": [10.0], "X": [100.0]})

        obj = Objective(observations=df, target="biomass")
        with pytest.raises(ValueError, match="target column"):
            obj.setup(coords)

    def test_unsupported_type_raises(self):
        """Non-xarray/DataFrame observations in target mode raises TypeError."""
        coords = _model_coords()
        obj = Objective(observations=[1, 2, 3], target="biomass")
        with pytest.raises(TypeError, match="requires xarray.DataArray or pandas.DataFrame"):
            obj.setup(coords)


# ---------------------------------------------------------------------------
# Transform mode
# ---------------------------------------------------------------------------


class TestSetupTransform:
    def test_transform_wraps_callable(self):
        """Transform mode wraps the user callable as extract_fn."""
        obs = jnp.array([1.0, 2.0, 3.0])

        def fn(o):
            return o["biomass"][:3, 0, 0]

        obj = Objective(observations=obs, transform=fn)
        prepared = obj.setup(model_coords={})

        assert prepared.obs_array.shape == (3,)
        # extract_fn should be the transform itself
        outputs = _model_outputs()
        pred = prepared.extract_fn(outputs)
        expected = outputs["biomass"][:3, 0, 0]
        np.testing.assert_array_almost_equal(np.asarray(pred), np.asarray(expected))

    def test_shape_validation_pass(self):
        """Shape validation passes when transform output matches obs."""
        obs = jnp.zeros((4, 3))

        def fn(o):
            return o["biomass"][:, :, 0]  # (4, 3)

        dummy = {"biomass": jnp.zeros((4, 3, 5))}
        obj = Objective(observations=obs, transform=fn)
        prepared = obj.setup(model_coords={}, dummy_outputs=dummy)
        assert prepared.obs_array.shape == (4, 3)

    def test_shape_validation_mismatch(self):
        """Shape validation fails when transform output doesn't match obs."""
        obs = jnp.zeros((4, 5))  # Wrong shape

        def fn(o):
            return o["biomass"][:, :, 0]  # (4, 3)

        dummy = {"biomass": jnp.zeros((4, 3, 5))}
        obj = Objective(observations=obs, transform=fn)
        with pytest.raises(ValueError, match="transform output shape"):
            obj.setup(model_coords={}, dummy_outputs=dummy)
