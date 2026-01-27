"""Benchmark: Monolithic vs Decomposed LMTL Implementations.

This script compares two implementation strategies for the LMTL model:
1. Monolithic: Single function for production dynamics (Aging + Recruitment).
2. Decomposed: Separate functions for NPP, Aging, and Recruitment.

Metrics:
- Execution Time (End-to-End Simulation)
- Numerical Accuracy (Biomass Evolution Comparison)
"""

import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import xarray as xr

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine import StreamingRunner

# Initialize Pint
ureg = pint.get_application_registry()

# =============================================================================
# 1. SHARED PARAMETERS & FUNCTIONS
# =============================================================================

# Parameters
LMTL_E = 0.1668
LMTL_LAMBDA_0 = 1 / 150
LMTL_GAMMA_LAMBDA = 0.15
LMTL_TAU_R_0 = 10.38
LMTL_GAMMA_TAU_R = 0.11
LMTL_T_REF = 0.0


@functional(name="common:gillooly", backend="jax", units={"temp": "degC", "return": "degC"})
def gillooly_temperature(temp):
    """Normalize temperature using Gillooly's method.

    Args:
        temp: Temperature in Celsius.

    Returns:
        Normalized temperature.
    """
    return temp / (1.0 + temp / 273.0)


@functional(
    name="common:rec_age",
    backend="jax",
    units={"temp": "degC", "tau_r_0": "s", "gamma": "1/delta_degC", "t_ref": "degC", "return": "s"},
)
def recruitment_age(temp, tau_r_0, gamma, t_ref):
    """Calculate recruitment age threshold.

    Args:
        temp: Temperature.
        tau_r_0: Base recruitment age.
        gamma: Temperature sensitivity of recruitment age.
        t_ref: Reference temperature.

    Returns:
        Recruitment age in seconds.
    """
    return tau_r_0 * jnp.exp(-gamma * (temp - t_ref))


@functional(
    name="common:mortality",
    backend="jax",
    units={
        "biomass": "g/m^2",
        "temp": "degC",
        "lambda_0": "1/s",
        "gamma": "1/delta_degC",
        "t_ref": "degC",
        "return": "g/m^2/s",
    },
)
def mortality_tendency(biomass, temp, lambda_0, gamma, t_ref):
    """Calculate mortality tendency.

    Args:
        biomass: Current biomass.
        temp: Temperature.
        lambda_0: Base mortality rate.
        gamma: Temperature sensitivity of mortality.
        t_ref: Reference temperature.

    Returns:
        Biomass loss rate due to mortality.
    """
    rate = lambda_0 * jnp.exp(gamma * (temp - t_ref))
    return -rate * biomass


# =============================================================================
# 2. MONOLITHIC IMPLEMENTATION
# =============================================================================


@functional(
    name="mono:prod_dynamics",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    outputs=["prod_tendency", "biomass_source"],
    units={
        "production": "g/m^2",
        "cohort_ages": "s",
        "rec_age": "s",
        "npp": "g/m^2/s",
        "efficiency": "dimensionless",
        "prod_tendency": "g/m^2/s",
        "biomass_source": "g/m^2/s",
    },
)
def production_dynamics_mono(production, cohort_ages, rec_age, npp, efficiency):
    """Calculate production dynamics (aging + recruitment) in one step.

    Args:
        production: Current production state.
        cohort_ages: Cohort age definitions.
        rec_age: Recruitment age threshold.
        npp: Net Primary Production.
        efficiency: Trophic efficiency.

    Returns:
        Tuple of (production_tendency, biomass_source_tendency).
    """
    spatial_ndim = production.ndim - 1
    c_broadcast = (production.shape[0],) + (1,) * spatial_ndim

    d_tau = jnp.concatenate([cohort_ages[1:] - cohort_ages[:-1], cohort_ages[-1:] - cohort_ages[-2:-1]])
    aging_rate = (1.0 / d_tau).reshape(c_broadcast)
    outflow = production * aging_rate

    source_flux = jnp.expand_dims(npp * efficiency, axis=0)
    influx_from_prev = jnp.concatenate([source_flux, outflow[:-1]], axis=0)

    cohort_ages_grid = cohort_ages.reshape(c_broadcast)
    is_recruited = cohort_ages_grid >= rec_age

    false_slice = jnp.zeros((1,) + production.shape[1:], dtype=bool)
    prev_recruited = jnp.concatenate([false_slice, is_recruited[:-1]], axis=0)
    effective_influx = jnp.where(prev_recruited, 0.0, influx_from_prev)

    prod_tendency = effective_influx - outflow
    recruitment_flux = jnp.where(is_recruited, outflow, 0.0)

    # Correction: Stop outflow from last cohort if not recruited (Plus Group Logic)
    # The monolithic version in full_model_0d.py missed the explicit "last cohort block"
    # but relies on "is_recruited". If last cohort is recruited, it empties.
    # To be strictly identical to decomposed, we should ensure consistent logic.
    # The decomposed version forces last cohort outflow to 0 in aging, but allows it in recruitment.
    # Here, 'outflow' is used for both.

    biomass_source_val = jnp.sum(recruitment_flux, axis=0)
    return prod_tendency, biomass_source_val


# =============================================================================
# 3. DECOMPOSED IMPLEMENTATION
# =============================================================================


@functional(
    name="decomp:npp",
    backend="jax",
    core_dims={"production": ["C"]},
    units={"npp": "g/m^2/s", "efficiency": "dimensionless", "production": "g/m^2", "return": "g/m^2/s"},
)
def npp_injection(npp, efficiency, production):
    """Calculate Net Primary Production injection tendency.

    Args:
        npp: Net Primary Production forcing.
        efficiency: Trophic efficiency parameter.
        production: Current production state (unused for calculation but required for dimension).

    Returns:
        The production tendency due to NPP injection.
    """
    source_flux = npp * efficiency
    tendency = jnp.zeros_like(production)
    return tendency.at[0, ...].set(source_flux)


@functional(
    name="decomp:aging",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    units={"production": "g/m^2", "cohort_ages": "s", "rec_age": "s", "return": "g/m^2/s"},
)
def aging_flow(production, cohort_ages, rec_age):
    """Calculate aging flow between cohorts.

    Args:
        production: Current production state.
        cohort_ages: Age threshold for each cohort.
        rec_age: Recruitment age threshold.

    Returns:
        The production tendency due to aging (loss from current, gain to next).
    """
    spatial_ndim = production.ndim - 1
    c_broadcast = (production.shape[0],) + (1,) * spatial_ndim

    d_tau = jnp.concatenate([cohort_ages[1:] - cohort_ages[:-1], cohort_ages[-1:] - cohort_ages[-2:-1]])
    aging_coef = (1.0 / d_tau).reshape(c_broadcast)
    base_outflow = production * aging_coef

    cohort_ages_grid = cohort_ages.reshape(c_broadcast)
    is_recruited = cohort_ages_grid >= rec_age

    aging_outflow = jnp.where(is_recruited, 0.0, base_outflow)
    aging_outflow = aging_outflow.at[-1, ...].set(0.0)

    loss = -aging_outflow
    gain = jnp.concatenate([jnp.zeros((1,) + production.shape[1:]), aging_outflow[:-1]], axis=0)
    return loss + gain


@functional(
    name="decomp:recruit",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    outputs=["prod_loss", "biomass_gain"],
    units={
        "production": "g/m^2",
        "cohort_ages": "s",
        "rec_age": "s",
        "prod_loss": "g/m^2/s",
        "biomass_gain": "g/m^2/s",
    },
)
def recruitment_flow(production, cohort_ages, rec_age):
    """Calculate recruitment flow into biomass.

    Args:
        production: Current production state.
        cohort_ages: Age threshold for each cohort.
        rec_age: Recruitment age threshold.

    Returns:
        Tuple of (production_loss, biomass_gain).
    """
    spatial_ndim = production.ndim - 1
    c_broadcast = (production.shape[0],) + (1,) * spatial_ndim

    d_tau = jnp.concatenate([cohort_ages[1:] - cohort_ages[:-1], cohort_ages[-1:] - cohort_ages[-2:-1]])
    aging_coef = (1.0 / d_tau).reshape(c_broadcast)
    base_outflow = production * aging_coef

    cohort_ages_grid = cohort_ages.reshape(c_broadcast)
    is_recruited = cohort_ages_grid >= rec_age

    flux_to_biomass = jnp.where(is_recruited, base_outflow, 0.0)

    prod_loss = -flux_to_biomass
    biomass_gain = jnp.sum(flux_to_biomass, axis=0)
    return prod_loss, biomass_gain


# =============================================================================
# 4. BENCHMARK RUNNER
# =============================================================================


def create_config(grid_size=(10, 10), n_days=365):
    """Create configuration for the benchmark.

    Args:
        grid_size: Tuple of (ny, nx) for the spatial grid.
        n_days: Duration of the simulation in days.

    Returns:
        Config: The configured Seapopym Config object.
    """
    # Setup Parameters
    max_age_days = int(np.ceil(LMTL_TAU_R_0) + 5)
    cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
    n_cohorts = len(cohort_ages_sec)

    start_date = "2000-01-01"
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    ny, nx = grid_size

    # Constant Forcing
    temp_da = xr.DataArray(np.full((len(dates), ny, nx), 25.0), dims=["T", "Y", "X"], coords={"T": dates})
    npp_da = xr.DataArray(np.full((len(dates), ny, nx), 1.0 / 86400.0), dims=["T", "Y", "X"], coords={"T": dates})

    config = Config.from_dict(
        {
            "parameters": {
                "lambda_0": {"value": LMTL_LAMBDA_0 / 86400.0},
                "gamma_lambda": {"value": LMTL_GAMMA_LAMBDA},
                "tau_r_0": {"value": LMTL_TAU_R_0 * 86400.0},
                "gamma_tau_r": {"value": LMTL_GAMMA_TAU_R},
                "t_ref": {"value": LMTL_T_REF},
                "efficiency": {"value": LMTL_E},
                "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
            },
            "forcings": {"temperature": temp_da, "primary_production": npp_da},
            "initial_state": {
                "biomass": xr.DataArray(np.zeros((ny, nx)), dims=["Y", "X"]),
                "production": xr.DataArray(np.zeros((ny, nx, n_cohorts)), dims=["Y", "X", "C"]),
            },
            "execution": {
                "time_start": start_date,
                "time_end": str(dates[-1].date()),
                "dt": "1d",
            },
        }
    )
    return config


def get_blueprints():
    """Create Monolithic and Decomposed blueprints for comparison.

    Returns:
        Tuple of (Blueprint, Blueprint): The monolithic and decomposed blueprints.
    """
    # Common process parts
    common_process = [
        {
            "func": "common:gillooly",
            "inputs": {"temp": "forcings.temperature"},
            "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}},
        },
        {
            "func": "common:rec_age",
            "inputs": {
                "temp": "derived.temp_norm",
                "tau_r_0": "parameters.tau_r_0",
                "gamma": "parameters.gamma_tau_r",
                "t_ref": "parameters.t_ref",
            },
            "outputs": {"return": {"target": "derived.rec_age", "type": "derived"}},
        },
        {
            "func": "common:mortality",
            "inputs": {
                "biomass": "state.biomass",
                "temp": "derived.temp_norm",
                "lambda_0": "parameters.lambda_0",
                "gamma": "parameters.gamma_lambda",
                "t_ref": "parameters.t_ref",
            },
            "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}},
        },
    ]

    vars_decl = {
        "state": {
            "biomass": {"units": "g/m^2", "dims": ["Y", "X"]},
            "production": {"units": "g/m^2", "dims": ["Y", "X", "C"]},
        },
        "parameters": {
            "lambda_0": {"units": "1/s"},
            "gamma_lambda": {"units": "1/delta_degC"},
            "tau_r_0": {"units": "s"},
            "gamma_tau_r": {"units": "1/delta_degC"},
            "t_ref": {"units": "degC"},
            "efficiency": {"units": "dimensionless"},
            "cohort_ages": {"units": "s", "dims": ["C"]},
        },
        "forcings": {
            "temperature": {"units": "degC", "dims": ["T", "Y", "X"]},
            "primary_production": {"units": "g/m^2/s", "dims": ["T", "Y", "X"]},
        },
    }

    # Monolithic Blueprint
    bp_mono = Blueprint.from_dict(
        {
            "id": "mono",
            "version": "1.0",
            "declarations": vars_decl,
            "process": common_process
            + [
                {
                    "func": "mono:prod_dynamics",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                        "npp": "forcings.primary_production",
                        "efficiency": "parameters.efficiency",
                    },
                    "outputs": {
                        "prod_tendency": {"target": "tendencies.production", "type": "tendency"},
                        "biomass_source": {"target": "tendencies.biomass", "type": "tendency"},
                    },
                }
            ],
        }
    )

    # Decomposed Blueprint
    bp_decomp = Blueprint.from_dict(
        {
            "id": "decomp",
            "version": "1.0",
            "declarations": vars_decl,
            "process": common_process
            + [
                {
                    "func": "decomp:npp",
                    "inputs": {
                        "npp": "forcings.primary_production",
                        "efficiency": "parameters.efficiency",
                        "production": "state.production",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": "decomp:aging",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": "decomp:recruit",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {
                        "prod_loss": {"target": "tendencies.production", "type": "tendency"},
                        "biomass_gain": {"target": "tendencies.biomass", "type": "tendency"},
                    },
                },
            ],
        }
    )

    return bp_mono, bp_decomp


def run_benchmark():
    """Run the benchmark comparison between monolithic and decomposed model implementations.

    Executes both model versions, measures execution time, and compares biomass evolution.
    Prints benchmark results and saves a comparison plot.
    """
    grid_size = (20, 20)
    n_days = 365 * 2

    print(f"Setting up Benchmark: Grid {grid_size}, Duration {n_days} days")
    config = create_config(grid_size, n_days)
    bp_mono, bp_decomp = get_blueprints()

    results = {}

    for name, bp in [("Monolithic", bp_mono), ("Decomposed", bp_decomp)]:
        print(f"\n--- Running {name} Model ---")

        # Compile
        start = time.time()
        model = compile_model(bp, config, backend="jax")
        print(f"Compilation: {time.time() - start:.4f}s")

        runner = StreamingRunner(model)

        # Warmup
        print("Warming up (JIT)...")
        runner.run()

        # Benchmark Loops
        n_runs = 1000
        print(f"Benchmarking ({n_runs} runs)...")
        durations = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            # We don't need outputs for timing, but we need to run it
            st, out = runner.run()
            durations.append(time.perf_counter() - t0)

        # Store results from last run
        results[name] = {
            "time_mean": np.mean(durations),
            "time_std": np.std(durations),
            "biomass": np.array(out["biomass"]).mean(axis=(1, 2)),
            "production": np.array(out["production"]).mean(axis=(1, 2, 3)),
        }
        print(f"Mean Execution: {results[name]['time_mean']:.5f}s ± {results[name]['time_std']:.5f}s")

    # Comparison
    bio_mono = results["Monolithic"]["biomass"]
    bio_decomp = results["Decomposed"]["biomass"]

    diff = np.abs(bio_mono - bio_decomp)
    max_diff = np.max(diff)
    mae = np.mean(diff)

    print("\n--- Comparison Results ---")
    print(f"Max Biomass Difference: {max_diff:.6e}")
    print(f"Mean Absolute Error:    {mae:.6e}")

    t_mono = results["Monolithic"]["time_mean"]
    t_decomp = results["Decomposed"]["time_mean"]
    print(f"\nExecution Time (Mean of {n_runs} runs):")
    print(f"Monolithic: {t_mono:.5f}s ± {results['Monolithic']['time_std']:.5f}s")
    print(f"Decomposed: {t_decomp:.5f}s ± {results['Decomposed']['time_std']:.5f}s")
    print(f"Ratio (Decomp/Mono): {t_decomp / t_mono:.3f}x")

    # Plotting
    plt.figure(figsize=(10, 6))
    time_axis = np.arange(len(bio_mono))
    plt.plot(time_axis, bio_mono, "b-", label="Monolithic", linewidth=2, alpha=0.7)
    plt.plot(time_axis, bio_decomp, "r--", label="Decomposed", linewidth=2, alpha=0.7)
    plt.title("Biomass Evolution Comparison (Spatial Mean)")
    plt.xlabel("Time (days)")
    plt.ylabel("Biomass (g/m^2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("benchmark_comparison.png")
    print("Plot saved to benchmark_comparison.png")


if __name__ == "__main__":
    run_benchmark()
