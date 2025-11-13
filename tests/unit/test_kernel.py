"""Tests for the Kernel class."""

import jax.numpy as jnp
import pytest

from seapopym_message.core.kernel import Kernel
from seapopym_message.core.unit import unit


@pytest.mark.unit
class TestKernel:
    """Test the Kernel class."""

    def test_kernel_creation_empty(self) -> None:
        """Test creating a Kernel with no Units."""
        kernel = Kernel([])

        assert kernel.local_units == []
        assert kernel.global_units == []
        assert kernel.has_global_units() is False

    def test_kernel_separates_local_global(self) -> None:
        """Test that Kernel correctly separates local and global Units."""

        @unit(name="local1", inputs=["x"], outputs=["y"], scope="local")
        def local_func(x: float) -> float:
            return x * 2

        @unit(name="global1", inputs=["y"], outputs=["z"], scope="global")
        def global_func(y: float) -> float:
            return y + 1

        kernel = Kernel([local_func, global_func])

        assert len(kernel.local_units) == 1
        assert len(kernel.global_units) == 1
        assert kernel.local_units[0].name == "local1"
        assert kernel.global_units[0].name == "global1"
        assert kernel.has_global_units() is True

    def test_execute_local_phase_single_unit(self) -> None:
        """Test executing local phase with a single Unit."""

        @unit(name="multiply", inputs=["x"], outputs=["y"], scope="local", compiled=True)
        def multiply_func(x: jnp.ndarray, params: dict) -> jnp.ndarray:
            return x * params["factor"]

        kernel = Kernel([multiply_func])

        state = {"x": jnp.array([1.0, 2.0, 3.0])}
        params = {"factor": 2.0}

        result = kernel.execute_local_phase(state, dt=0.1, params=params)

        assert "y" in result
        assert jnp.allclose(result["y"], jnp.array([2.0, 4.0, 6.0]))

    def test_execute_local_phase_chain(self) -> None:
        """Test executing local phase with chained Units."""

        @unit(name="step1", inputs=["x"], outputs=["y"], scope="local")
        def step1(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        @unit(name="step2", inputs=["y"], outputs=["z"], scope="local")
        def step2(y: jnp.ndarray) -> jnp.ndarray:
            return y + 1

        @unit(name="step3", inputs=["z"], outputs=["w"], scope="local")
        def step3(z: jnp.ndarray) -> jnp.ndarray:
            return z * 3

        kernel = Kernel([step1, step2, step3])

        state = {"x": jnp.array([1.0, 2.0])}
        result = kernel.execute_local_phase(state, dt=0.1, params={})

        # x=1 -> y=2 -> z=3 -> w=9
        # x=2 -> y=4 -> z=5 -> w=15
        assert jnp.allclose(result["w"], jnp.array([9.0, 15.0]))

    def test_execute_global_phase_with_neighbor_data(self) -> None:
        """Test executing global phase with neighbor data."""

        @unit(name="diffusion", inputs=["biomass"], outputs=["biomass"], scope="global")
        def simple_diffusion(
            biomass: jnp.ndarray,
            dt: float,
            params: dict,
            halo_north: dict | None = None,
            halo_south: dict | None = None,
        ) -> jnp.ndarray:
            # Dummy diffusion: just add neighbor values
            result = biomass.copy()
            if halo_north:
                result = result + halo_north["biomass"] * 0.1
            if halo_south:
                result = result + halo_south["biomass"] * 0.1
            return result

        kernel = Kernel([simple_diffusion])

        state = {"biomass": jnp.array([10.0, 20.0])}
        neighbor_data = {
            "halo_north": {"biomass": jnp.array([5.0, 5.0])},
            "halo_south": {"biomass": jnp.array([3.0, 3.0])},
        }

        result = kernel.execute_global_phase(state, dt=0.1, params={}, neighbor_data=neighbor_data)

        # biomass + 0.1*5 + 0.1*3 = biomass + 0.8
        expected = jnp.array([10.8, 20.8])
        assert jnp.allclose(result["biomass"], expected)

    def test_topological_sort_simple(self) -> None:
        """Test topological sorting with simple dependency chain."""

        @unit(name="a", inputs=[], outputs=["x"], scope="local")
        def unit_a() -> float:
            return 1.0

        @unit(name="b", inputs=["x"], outputs=["y"], scope="local")
        def unit_b(x: float) -> float:
            return x * 2

        @unit(name="c", inputs=["y"], outputs=["z"], scope="local")
        def unit_c(y: float) -> float:
            return y + 1

        # Create in wrong order
        kernel = Kernel([unit_c, unit_a, unit_b])

        # Should be sorted: a -> b -> c
        assert kernel.local_units[0].name == "a"
        assert kernel.local_units[1].name == "b"
        assert kernel.local_units[2].name == "c"

    def test_topological_sort_diamond(self) -> None:
        """Test topological sorting with diamond dependency pattern."""

        @unit(name="a", inputs=[], outputs=["x"], scope="local")
        def unit_a() -> float:
            return 1.0

        @unit(name="b", inputs=["x"], outputs=["y1"], scope="local")
        def unit_b(x: float) -> float:
            return x * 2

        @unit(name="c", inputs=["x"], outputs=["y2"], scope="local")
        def unit_c(x: float) -> float:
            return x * 3

        @unit(name="d", inputs=["y1", "y2"], outputs=["z"], scope="local")
        def unit_d(y1: float, y2: float) -> float:
            return y1 + y2

        # Create in random order
        kernel = Kernel([unit_d, unit_b, unit_a, unit_c])

        # 'a' must come first, 'd' must come last
        assert kernel.local_units[0].name == "a"
        assert kernel.local_units[-1].name == "d"
        # 'b' and 'c' can be in either order (both depend on 'a', both feed into 'd')

    def test_topological_sort_detects_cycle(self) -> None:
        """Test that cyclic dependencies are detected."""

        @unit(name="a", inputs=["z"], outputs=["x"], scope="local")
        def unit_a(z: float) -> float:
            return z * 2

        @unit(name="b", inputs=["x"], outputs=["y"], scope="local")
        def unit_b(x: float) -> float:
            return x + 1

        @unit(name="c", inputs=["y"], outputs=["z"], scope="local")
        def unit_c(y: float) -> float:
            return y * 3

        # Cycle: a -> b -> c -> a
        with pytest.raises(ValueError, match="Cyclic dependency detected"):
            Kernel([unit_a, unit_b, unit_c])

    def test_kernel_with_external_inputs(self) -> None:
        """Test Kernel with Units that require external inputs."""

        @unit(name="growth", inputs=["biomass"], outputs=["biomass"], scope="local")
        def growth(biomass: jnp.ndarray, dt: float, params: dict) -> jnp.ndarray:
            return biomass + params["growth_rate"] * dt

        kernel = Kernel([growth])

        state = {"biomass": jnp.array([10.0, 20.0, 30.0])}
        params = {"growth_rate": 5.0}

        result = kernel.execute_local_phase(state, dt=0.1, params=params)

        expected = jnp.array([10.5, 20.5, 30.5])
        assert jnp.allclose(result["biomass"], expected)

    def test_kernel_modifies_same_variable(self) -> None:
        """Test multiple Units modifying the same state variable."""

        @unit(name="step1", inputs=["x"], outputs=["x"], scope="local")
        def step1(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        @unit(name="step2", inputs=["x"], outputs=["x"], scope="local")
        def step2(x: jnp.ndarray) -> jnp.ndarray:
            return x + 1

        kernel = Kernel([step1, step2])

        state = {"x": jnp.array([1.0, 2.0])}
        result = kernel.execute_local_phase(state, dt=0.1, params={})

        # step1: x -> x*2
        # step2: x -> x+1
        # Total: (x*2)+1
        expected = jnp.array([3.0, 5.0])
        assert jnp.allclose(result["x"], expected)


@pytest.mark.unit
class TestKernelRealistic:
    """Realistic integration tests for Kernel."""

    def test_biology_kernel(self) -> None:
        """Test a realistic biology kernel (recruitment + mortality + growth)."""

        @unit(name="recruitment", inputs=[], outputs=["R"], scope="local", compiled=True)
        def compute_recruitment(params: dict) -> jnp.ndarray:
            return jnp.full((2, 2), params["R"])

        @unit(name="mortality", inputs=["biomass"], outputs=["M"], scope="local", compiled=True)
        def compute_mortality(biomass: jnp.ndarray, params: dict) -> jnp.ndarray:
            return params["lambda"] * biomass

        @unit(
            name="growth",
            inputs=["biomass", "R", "M"],
            outputs=["biomass"],
            scope="local",
            compiled=True,
        )
        def compute_growth(
            biomass: jnp.ndarray, R: jnp.ndarray, M: jnp.ndarray, dt: float
        ) -> jnp.ndarray:
            return biomass + (R - M) * dt

        kernel = Kernel([compute_recruitment, compute_mortality, compute_growth])

        # Verify separation
        assert len(kernel.local_units) == 3
        assert len(kernel.global_units) == 0

        # Initial state
        state = {"biomass": jnp.array([[10.0, 20.0], [30.0, 40.0]])}
        params = {"R": 5.0, "lambda": 0.1}
        dt = 0.1

        # Execute one timestep
        state = kernel.execute_local_phase(state, dt=dt, params=params)

        # B_new = B + (R - λB) * dt
        # For B=10: 10 + (5 - 1) * 0.1 = 10.4
        expected = jnp.array([[10.4, 20.3], [30.2, 40.1]])
        assert jnp.allclose(state["biomass"], expected, atol=1e-6)

    def test_biology_with_transport_kernel(self) -> None:
        """Test a kernel combining biology (local) and transport (global)."""

        @unit(name="recruitment", inputs=[], outputs=["R"], scope="local")
        def compute_recruitment(params: dict) -> jnp.ndarray:
            return jnp.full((2, 2), params["R"])

        @unit(name="mortality", inputs=["biomass"], outputs=["M"], scope="local")
        def compute_mortality(biomass: jnp.ndarray, params: dict) -> jnp.ndarray:
            return params["lambda"] * biomass

        @unit(name="growth", inputs=["biomass", "R", "M"], outputs=["biomass"], scope="local")
        def compute_growth(
            biomass: jnp.ndarray, R: jnp.ndarray, M: jnp.ndarray, dt: float
        ) -> jnp.ndarray:
            return biomass + (R - M) * dt

        @unit(name="diffusion", inputs=["biomass"], outputs=["biomass"], scope="global")
        def compute_diffusion(biomass: jnp.ndarray, dt: float, params: dict) -> jnp.ndarray:
            # Dummy diffusion: smooth slightly
            return biomass * 0.9 + jnp.mean(biomass) * 0.1

        kernel = Kernel([compute_recruitment, compute_mortality, compute_growth, compute_diffusion])

        # Verify separation
        assert len(kernel.local_units) == 3
        assert len(kernel.global_units) == 1
        assert kernel.has_global_units() is True

        # Initial state
        state = {"biomass": jnp.array([[10.0, 20.0], [30.0, 40.0]])}
        params = {"R": 5.0, "lambda": 0.1}
        dt = 0.1

        # Execute local phase
        state = kernel.execute_local_phase(state, dt=dt, params=params)

        # Execute global phase
        state = kernel.execute_global_phase(state, dt=dt, params=params)

        # Result should be smoothed
        assert "biomass" in state
        assert state["biomass"].shape == (2, 2)
        # Mean should be preserved
        original_mean = jnp.mean(jnp.array([[10.4, 20.3], [30.2, 40.1]]))
        result_mean = jnp.mean(state["biomass"])
        assert jnp.allclose(result_mean, original_mean, atol=1e-5)

    def test_multi_timestep_simulation(self) -> None:
        """Test running multiple timesteps with a Kernel."""

        @unit(name="growth", inputs=["biomass"], outputs=["biomass"], scope="local", compiled=True)
        def simple_growth(biomass: jnp.ndarray, dt: float, params: dict) -> jnp.ndarray:
            return biomass * (1 + params["rate"] * dt)

        kernel = Kernel([simple_growth])

        state = {"biomass": jnp.array([1.0, 2.0, 3.0])}
        params = {"rate": 0.1}
        dt = 0.1
        num_steps = 10

        # Run simulation
        for _ in range(num_steps):
            state = kernel.execute_local_phase(state, dt=dt, params=params)

        # After 10 steps: B * (1 + 0.1*0.1)^10 = B * 1.01^10 ≈ B * 1.104622
        expected = jnp.array([1.0, 2.0, 3.0]) * (1.01**10)
        assert jnp.allclose(state["biomass"], expected, atol=1e-5)
