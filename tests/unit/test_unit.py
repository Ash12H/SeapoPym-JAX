"""Tests for the Unit class and @unit decorator."""

import jax.numpy as jnp
import pytest

from seapopym_message.core.unit import Unit, unit


@pytest.mark.unit
class TestUnit:
    """Test the Unit class."""

    def test_unit_creation(self) -> None:
        """Test creating a Unit instance."""

        def simple_func(x: float) -> float:
            return x * 2

        u = Unit(
            name="double",
            func=simple_func,
            inputs=["x"],
            outputs=["y"],
            scope="local",
            compiled=False,
        )

        assert u.name == "double"
        assert u.inputs == ["x"]
        assert u.outputs == ["y"]
        assert u.scope == "local"
        assert u.compiled is False

    def test_can_execute_with_available_inputs(self) -> None:
        """Test can_execute returns True when inputs are available."""

        def dummy_func(a: float, b: float) -> float:
            return a + b

        u = Unit(name="add", func=dummy_func, inputs=["a", "b"], outputs=["c"], scope="local")

        available = {"a", "b", "c", "d"}
        assert u.can_execute(available) is True

    def test_can_execute_with_missing_inputs(self) -> None:
        """Test can_execute returns False when inputs are missing."""

        def dummy_func(a: float, b: float) -> float:
            return a + b

        u = Unit(name="add", func=dummy_func, inputs=["a", "b"], outputs=["c"], scope="local")

        available = {"a", "d"}  # missing 'b'
        assert u.can_execute(available) is False

    def test_execute_with_single_output(self) -> None:
        """Test execute with a single output."""

        def multiply(x: jnp.ndarray, factor: float) -> jnp.ndarray:
            return x * factor

        u = Unit(
            name="multiply",
            func=multiply,
            inputs=["x"],
            outputs=["y"],
            scope="local",
            compiled=False,
        )

        state = {"x": jnp.array([1.0, 2.0, 3.0])}
        result = u.execute(state, factor=2.0)

        assert "y" in result
        assert jnp.allclose(result["y"], jnp.array([2.0, 4.0, 6.0]))

    def test_execute_with_multiple_outputs_tuple(self) -> None:
        """Test execute with multiple outputs returned as tuple."""

        def split_values(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return x * 2, x * 3

        u = Unit(
            name="split",
            func=split_values,
            inputs=["x"],
            outputs=["y", "z"],
            scope="local",
            compiled=False,
        )

        state = {"x": jnp.array([1.0, 2.0])}
        result = u.execute(state)

        assert "y" in result
        assert "z" in result
        assert jnp.allclose(result["y"], jnp.array([2.0, 4.0]))
        assert jnp.allclose(result["z"], jnp.array([3.0, 6.0]))

    def test_execute_with_multiple_outputs_dict(self) -> None:
        """Test execute with multiple outputs returned as dict."""

        def compute_stats(x: jnp.ndarray) -> dict[str, jnp.ndarray]:
            return {"mean": jnp.mean(x), "sum": jnp.sum(x)}

        u = Unit(
            name="stats",
            func=compute_stats,
            inputs=["x"],
            outputs=["mean", "sum"],
            scope="local",
            compiled=False,
        )

        state = {"x": jnp.array([1.0, 2.0, 3.0, 4.0])}
        result = u.execute(state)

        assert "mean" in result
        assert "sum" in result
        assert jnp.allclose(result["mean"], 2.5)
        assert jnp.allclose(result["sum"], 10.0)

    def test_execute_raises_on_missing_input(self) -> None:
        """Test execute raises ValueError when required input is missing."""

        def dummy_func(a: float, b: float) -> float:
            return a + b

        u = Unit(name="add", func=dummy_func, inputs=["a", "b"], outputs=["c"], scope="local")

        state = {"a": 1.0}  # missing 'b'

        with pytest.raises(ValueError, match="missing required inputs"):
            u.execute(state)

    def test_execute_raises_on_wrong_output_count(self) -> None:
        """Test execute raises when function returns wrong number of outputs."""

        def bad_func(x: float) -> tuple[float, float, float]:
            return x, x * 2, x * 3  # returns 3 values

        u = Unit(
            name="bad",
            func=bad_func,
            inputs=["x"],
            outputs=["y", "z"],  # declares 2 outputs
            scope="local",
            compiled=False,
        )

        state = {"x": 1.0}

        with pytest.raises(ValueError, match="returned 3 values but declared 2 outputs"):
            u.execute(state)

    def test_execute_raises_on_missing_declared_output(self) -> None:
        """Test execute raises when function doesn't produce declared output."""

        def bad_func(x: float) -> dict[str, float]:
            return {"wrong_key": x * 2}  # doesn't produce 'y'

        u = Unit(
            name="bad",
            func=bad_func,
            inputs=["x"],
            outputs=["y"],
            scope="local",
            compiled=False,
        )

        state = {"x": 1.0}

        with pytest.raises(ValueError, match="did not produce declared output"):
            u.execute(state)

    def test_compiled_flag(self) -> None:
        """Test that compiled flag triggers JAX JIT compilation."""

        def jax_func(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        u = Unit(
            name="jitted",
            func=jax_func,
            inputs=["x"],
            outputs=["y"],
            scope="local",
            compiled=True,
        )

        # Compiled function should be wrapped
        assert u._compiled_func is not None
        assert u._compiled_func != jax_func

        # Should still execute correctly
        state = {"x": jnp.array([1.0, 2.0, 3.0])}
        result = u.execute(state)

        assert jnp.allclose(result["y"], jnp.array([2.0, 4.0, 6.0]))


@pytest.mark.unit
class TestUnitDecorator:
    """Test the @unit decorator."""

    def test_decorator_creates_unit(self) -> None:
        """Test that @unit decorator creates a Unit instance."""

        @unit(name="add", inputs=["a", "b"], outputs=["c"], scope="local", compiled=False)
        def add_func(a: float, b: float) -> float:
            return a + b

        assert isinstance(add_func, Unit)
        assert add_func.name == "add"
        assert add_func.inputs == ["a", "b"]
        assert add_func.outputs == ["c"]
        assert add_func.scope == "local"
        assert add_func.compiled is False

    def test_decorator_with_jax_compilation(self) -> None:
        """Test decorator with compiled=True."""

        @unit(name="multiply", inputs=["x"], outputs=["y"], scope="local", compiled=True)
        def multiply_func(x: jnp.ndarray, factor: float) -> jnp.ndarray:
            return x * factor

        assert isinstance(multiply_func, Unit)
        assert multiply_func.compiled is True

        # Should execute correctly
        state = {"x": jnp.array([1.0, 2.0, 3.0])}
        result = multiply_func.execute(state, factor=3.0)

        assert jnp.allclose(result["y"], jnp.array([3.0, 6.0, 9.0]))

    def test_decorator_with_global_scope(self) -> None:
        """Test decorator with scope='global'."""

        @unit(name="diffusion", inputs=["biomass"], outputs=["biomass"], scope="global")
        def diffusion_func(biomass: jnp.ndarray, dt: float) -> jnp.ndarray:
            return biomass  # dummy implementation

        assert isinstance(diffusion_func, Unit)
        assert diffusion_func.scope == "global"

    def test_decorator_default_values(self) -> None:
        """Test decorator uses correct default values."""

        @unit(name="simple", inputs=["x"], outputs=["y"])
        def simple_func(x: float) -> float:
            return x * 2

        assert simple_func.scope == "local"
        assert simple_func.compiled is False


@pytest.mark.unit
class TestUnitIntegration:
    """Integration tests for Unit with realistic scenarios."""

    def test_biology_unit_realistic(self) -> None:
        """Test a realistic biology unit (mortality)."""

        @unit(name="mortality", inputs=["biomass"], outputs=["mortality_rate"], compiled=True)
        def compute_mortality(biomass: jnp.ndarray, params: dict) -> jnp.ndarray:
            return params["lambda"] * biomass

        state = {"biomass": jnp.array([[10.0, 20.0], [30.0, 40.0]])}
        params = {"lambda": 0.1}

        result = compute_mortality.execute(state, params=params)

        expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.allclose(result["mortality_rate"], expected)

    def test_chained_units(self) -> None:
        """Test multiple units executing in sequence."""

        @unit(name="recruitment", inputs=[], outputs=["R"], compiled=True)
        def compute_recruitment(params: dict) -> jnp.ndarray:
            return jnp.full((2, 2), params["R"])

        @unit(name="mortality", inputs=["biomass"], outputs=["M"], compiled=True)
        def compute_mortality(biomass: jnp.ndarray, params: dict) -> jnp.ndarray:
            return params["lambda"] * biomass

        @unit(name="growth", inputs=["biomass", "R", "M"], outputs=["biomass"], compiled=True)
        def compute_growth(
            biomass: jnp.ndarray, R: jnp.ndarray, M: jnp.ndarray, dt: float
        ) -> jnp.ndarray:
            return biomass + (R - M) * dt

        # Initial state
        state = {"biomass": jnp.array([[10.0, 20.0], [30.0, 40.0]])}
        params = {"R": 5.0, "lambda": 0.1}
        dt = 0.1

        # Execute chain
        result = compute_recruitment.execute(state, params=params)
        state.update(result)

        result = compute_mortality.execute(state, params=params)
        state.update(result)

        result = compute_growth.execute(state, dt=dt)
        state.update(result)

        # Biomass should increase: B + (R - λB) * dt
        # For B=10: 10 + (5 - 1) * 0.1 = 10.4
        expected = jnp.array([[10.4, 20.3], [30.2, 40.1]])
        assert jnp.allclose(state["biomass"], expected)

    def test_unit_with_kwargs_forwarding(self) -> None:
        """Test that execute correctly forwards kwargs to function."""

        @unit(name="complex", inputs=["x"], outputs=["y"], compiled=False)
        def complex_func(
            x: jnp.ndarray, dt: float, params: dict, extra: str = "default"
        ) -> jnp.ndarray:
            # Use all parameters to ensure they're forwarded
            factor = params.get("factor", 1.0)
            return x * factor * dt if extra == "multiply" else x + factor * dt

        state = {"x": jnp.array([1.0, 2.0, 3.0])}
        params = {"factor": 2.0}

        result = complex_func.execute(state, dt=0.5, params=params, extra="multiply")

        expected = jnp.array([1.0, 2.0, 3.0]) * 2.0 * 0.5
        assert jnp.allclose(result["y"], expected)
