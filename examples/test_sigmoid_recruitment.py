"""Compare hard-threshold vs sigmoid recruitment on LMTL 0D.

Runs two simulations side by side:
- Original: is_recruited = cohort_ages >= rec_age (step function, gradient = 0)
- Sigmoid:  fraction = sigmoid(k * (cohort_ages - rec_age))  (smooth, gradient != 0)

The sigmoid has half-saturation at rec_age and transitions from ~0 to ~1
over ±1 day (k = ln(99) / 86400 ≈ 5.32e-5 s⁻¹).

Produces a 4-panel comparison plot:
1. Biomass: step vs sigmoid
2. Recruitment fraction for each cohort over time
3. Gradient magnitudes for all parameters (step vs sigmoid)
4. Sigmoid shape illustration
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine.step import build_step_fn

jax.config.update("jax_default_device", jax.devices("cpu")[0])

# =============================================================================
# CONFIGURATION
# =============================================================================

TRUE_PARAMS = {
    "lambda_0": 1 / 150 / 86400,
    "gamma_lambda": 0.15,
    "tau_r_0": 10.38 * 86400,
    "gamma_tau_r": 0.11,
    "efficiency": 0.1668,
    "t_ref": 0.0,
}
PARAM_NAMES = ["lambda_0", "gamma_lambda", "tau_r_0", "gamma_tau_r", "efficiency"]
FIXED_PARAMS = {"t_ref": TRUE_PARAMS["t_ref"]}

SPINUP_YEARS = 1
SIM_YEARS = 2
DT = "1d"

# Sigmoid steepness: transition from ~1% to ~99% over ±1 day
TRANSITION_DAYS = 1.0
K_SIGMOID = float(jnp.log(99.0)) / (TRANSITION_DAYS * 86400.0)

# =============================================================================
# SIGMOID RECRUITMENT FUNCTIONS (registered with different names)
# =============================================================================


@functional(
    name="lmtl:aging_flow_sigmoid",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    out_dims=["C"],
    units={
        "production": "g/m^2",
        "cohort_ages": "s",
        "rec_age": "s",
        "return": "g/m^2/s",
    },
)
def aging_flow_sigmoid(
    production: jnp.ndarray,
    cohort_ages: jnp.ndarray,
    rec_age: jnp.ndarray,
) -> jnp.ndarray:
    """Aging flux with sigmoid recruitment (smooth, differentiable)."""
    d_tau_raw = cohort_ages[1:] - cohort_ages[:-1]
    last_d_tau = d_tau_raw[-1:]
    d_tau = jnp.concatenate([d_tau_raw, last_d_tau])

    aging_coef = 1.0 / d_tau
    base_outflow = production * aging_coef

    # Smooth recruitment fraction: 0 (young) -> 1 (old enough)
    recruit_fraction = jax.nn.sigmoid(K_SIGMOID * (cohort_ages - rec_age))

    # Aging gets the non-recruited fraction
    aging_outflow = (1.0 - recruit_fraction) * base_outflow

    # Last cohort: no outflow (plus group)
    aging_outflow = aging_outflow.at[-1].set(0.0)

    loss = -aging_outflow
    gain = jnp.concatenate([jnp.zeros(1), aging_outflow[:-1]])
    return loss + gain


@functional(
    name="lmtl:recruitment_flow_sigmoid",
    backend="jax",
    core_dims={"production": ["C"], "cohort_ages": ["C"]},
    out_dims=["C"],
    outputs=["prod_loss", "biomass_gain"],
    units={
        "production": "g/m^2",
        "cohort_ages": "s",
        "rec_age": "s",
        "prod_loss": "g/m^2/s",
        "biomass_gain": "g/m^2/s",
    },
)
def recruitment_flow_sigmoid(
    production: jnp.ndarray,
    cohort_ages: jnp.ndarray,
    rec_age: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recruitment flux with sigmoid (smooth, differentiable)."""
    d_tau_raw = cohort_ages[1:] - cohort_ages[:-1]
    last_d_tau = d_tau_raw[-1:]
    d_tau = jnp.concatenate([d_tau_raw, last_d_tau])

    aging_coef = 1.0 / d_tau
    base_outflow = production * aging_coef

    # Smooth recruitment fraction
    recruit_fraction = jax.nn.sigmoid(K_SIGMOID * (cohort_ages - rec_age))
    flux_to_biomass = recruit_fraction * base_outflow

    prod_loss = -flux_to_biomass
    biomass_gain = jnp.sum(flux_to_biomass)
    return prod_loss, biomass_gain


# =============================================================================
# BLUEPRINT BUILDER (parameterized by function names)
# =============================================================================


def make_blueprint(aging_fn: str, recruitment_fn: str, blueprint_id: str) -> Blueprint:
    """Build a LMTL 0D blueprint with specified aging/recruitment functions."""
    return Blueprint.from_dict(
        {
            "id": blueprint_id,
            "version": "1.0",
            "declarations": {
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
            },
            "process": [
                {
                    "func": "lmtl:gillooly_temperature",
                    "inputs": {"temp": "forcings.temperature"},
                    "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}},
                },
                {
                    "func": "lmtl:recruitment_age",
                    "inputs": {
                        "temp": "derived.temp_norm",
                        "tau_r_0": "parameters.tau_r_0",
                        "gamma": "parameters.gamma_tau_r",
                        "t_ref": "parameters.t_ref",
                    },
                    "outputs": {"return": {"target": "derived.rec_age", "type": "derived"}},
                },
                {
                    "func": "lmtl:npp_injection",
                    "inputs": {
                        "npp": "forcings.primary_production",
                        "efficiency": "parameters.efficiency",
                        "production": "state.production",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": aging_fn,
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": recruitment_fn,
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
                {
                    "func": "lmtl:mortality",
                    "inputs": {
                        "biomass": "state.biomass",
                        "temp": "derived.temp_norm",
                        "lambda_0": "parameters.lambda_0",
                        "gamma": "parameters.gamma_lambda",
                        "t_ref": "parameters.t_ref",
                    },
                    "outputs": {"return": {"target": "tendencies.biomass", "type": "tendency"}},
                },
            ],
        }
    )


# =============================================================================
# SHARED FORCINGS & CONFIG
# =============================================================================

max_age_days = int(np.ceil(TRUE_PARAMS["tau_r_0"] / 86400))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

total_years = SPINUP_YEARS + SIM_YEARS
start_date = "2000-01-01"
end_date = str((pd.Timestamp(start_date) + pd.DateOffset(years=total_years)).date())

dates = pd.date_range(
    start=pd.to_datetime(start_date), periods=(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 5, freq="D"
)
ny, nx = 1, 1
lat, lon = np.arange(ny), np.arange(nx)

day_of_year = dates.dayofyear.values
temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_da = xr.DataArray(
    np.broadcast_to(temp_c[:, None, None], (len(dates), ny, nx)),
    dims=["T", "Y", "X"],
    coords={"T": dates, "Y": lat, "X": lon},
)
npp_sec = (1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)) / 86400.0
npp_da = xr.DataArray(
    np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx)),
    dims=["T", "Y", "X"],
    coords={"T": dates, "Y": lat, "X": lon},
)

config_dict = {
    "parameters": {
        "lambda_0": {"value": TRUE_PARAMS["lambda_0"]},
        "gamma_lambda": {"value": TRUE_PARAMS["gamma_lambda"]},
        "tau_r_0": {"value": TRUE_PARAMS["tau_r_0"]},
        "gamma_tau_r": {"value": TRUE_PARAMS["gamma_tau_r"]},
        "t_ref": {"value": TRUE_PARAMS["t_ref"]},
        "efficiency": {"value": TRUE_PARAMS["efficiency"]},
        "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
    },
    "forcings": {"temperature": temp_da, "primary_production": npp_da},
    "initial_state": {
        "biomass": xr.DataArray(np.zeros((ny, nx)), dims=["Y", "X"], coords={"Y": lat, "X": lon}),
        "production": xr.DataArray(np.zeros((ny, nx, n_cohorts)), dims=["Y", "X", "C"], coords={"Y": lat, "X": lon}),
    },
    "execution": {
        "time_start": start_date,
        "time_end": end_date,
        "dt": DT,
        "forcing_interpolation": "linear",
        "batch_size": 1000,
    },
}


# =============================================================================
# BUILD SIMULATION FUNCTION
# =============================================================================


def build_simulator(blueprint: Blueprint):
    """Compile model and return a simulation function."""
    config = Config.from_dict(config_dict)
    model = compile_model(blueprint, config, backend="jax")
    step_fn = build_step_fn(model, params_as_argument=True)
    n_timesteps = model.n_timesteps
    initial_state = model.state
    forcings_stacked = model.forcings.get_all()

    def run_simulation(params: dict) -> jnp.ndarray:
        full_params = {**params, **{k: jnp.array(v) for k, v in FIXED_PARAMS.items()}}
        full_params["cohort_ages"] = model.parameters["cohort_ages"]

        def scan_body(carry, t):
            state, p = carry
            forcings_t = {}
            for name, arr in forcings_stacked.items():
                if arr.ndim > 0 and arr.shape[0] == n_timesteps:
                    forcings_t[name] = arr[t]
                else:
                    forcings_t[name] = arr
            new_carry, outputs = step_fn((state, p), forcings_t)
            new_state, _ = new_carry
            return (new_state, p), jnp.mean(outputs["biomass"])

        _, biomass = jax.lax.scan(scan_body, (initial_state, full_params), jnp.arange(n_timesteps))
        return biomass

    return run_simulation, n_timesteps


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Sigmoid steepness: k = {K_SIGMOID:.6e} s⁻¹ (transition ±{TRANSITION_DAYS} day)")
    print()

    # ----- Build both models -----
    print("Compiling STEP model...")
    bp_step = make_blueprint("lmtl:aging_flow", "lmtl:recruitment_flow", "step")
    sim_step, n_timesteps = build_simulator(bp_step)

    print("Compiling SIGMOID model...")
    bp_sigmoid = make_blueprint("lmtl:aging_flow_sigmoid", "lmtl:recruitment_flow_sigmoid", "sigmoid")
    sim_sigmoid, _ = build_simulator(bp_sigmoid)

    spinup_steps = int(SPINUP_YEARS / total_years * n_timesteps)

    # ----- Run simulations at true params -----
    true_p = {k: jnp.array(TRUE_PARAMS[k]) for k in PARAM_NAMES}

    print("Running STEP simulation...")
    bio_step = sim_step(true_p)
    print("Running SIGMOID simulation...")
    bio_sigmoid = sim_sigmoid(true_p)

    # ----- Compute gradients at +20% offset -----
    # Build a simple loss for gradient comparison
    obs_indices = jnp.arange(spinup_steps, n_timesteps, 10)  # every 10 steps
    obs_step = bio_step[obs_indices]
    obs_std = float(jnp.std(obs_step))

    def make_loss(sim_fn):
        def loss_fn(params):
            bio = sim_fn(params)
            pred = bio[obs_indices]
            rmse = jnp.sqrt(jnp.mean((pred - obs_step) ** 2))
            return rmse / obs_std

        return loss_fn

    loss_step = make_loss(sim_step)
    loss_sigmoid = make_loss(sim_sigmoid)

    grad_step_fn = jax.grad(loss_step)
    grad_sigmoid_fn = jax.grad(loss_sigmoid)

    offset_p = {k: jnp.array(TRUE_PARAMS[k] * 1.2) for k in PARAM_NAMES}

    print("\nComputing gradients at +20% offset...")
    grads_step = grad_step_fn(offset_p)
    grads_sigmoid = grad_sigmoid_fn(offset_p)

    print(f"\n{'Param':<14} {'|grad×val| STEP':>16} {'|grad×val| SIGMOID':>18}")
    print("-" * 52)
    for k in PARAM_NAMES:
        val = float(offset_p[k])
        gs = abs(float(grads_step[k]) * val)
        gsig = abs(float(grads_sigmoid[k]) * val)
        marker = " ← NEW!" if gs == 0 and gsig > 0 else ""
        print(f"{k:<14} {gs:>16.4g} {gsig:>18.4g}{marker}")

    # ----- Visualization -----
    time_days = np.arange(n_timesteps)
    bio_step_np = np.array(bio_step)
    bio_sigmoid_np = np.array(bio_sigmoid)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Biomass comparison
    ax = axes[0, 0]
    ax.plot(time_days, bio_step_np, label="Step (original)", linewidth=1.5, color="tab:blue")
    ax.plot(time_days, bio_sigmoid_np, label="Sigmoid", linewidth=1.5, color="tab:orange", linestyle="--")
    ax.axvline(x=spinup_steps, color="gray", linestyle=":", alpha=0.5, label="End spin-up")
    ax.set_xlabel("Day")
    ax.set_ylabel("Biomass (g/m²)")
    ax.set_title("Biomass: Step vs Sigmoid recruitment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Difference
    ax = axes[0, 1]
    diff = bio_sigmoid_np - bio_step_np
    ax.plot(time_days, diff, color="tab:red", linewidth=1)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=spinup_steps, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Day")
    ax.set_ylabel("Biomass difference (g/m²)")
    ax.set_title("Sigmoid − Step")
    ax.grid(True, alpha=0.3)

    # Panel 3: Gradient comparison (bar chart)
    ax = axes[1, 0]
    x_pos = np.arange(len(PARAM_NAMES))
    sens_step = [abs(float(grads_step[k]) * float(offset_p[k])) for k in PARAM_NAMES]
    sens_sigmoid = [abs(float(grads_sigmoid[k]) * float(offset_p[k])) for k in PARAM_NAMES]
    bar_width = 0.35
    ax.bar(x_pos - bar_width / 2, sens_step, bar_width, label="Step", color="tab:blue", alpha=0.7)
    ax.bar(x_pos + bar_width / 2, sens_sigmoid, bar_width, label="Sigmoid", color="tab:orange", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(PARAM_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("|∂loss/∂p × p|  (normalized sensitivity)")
    ax.set_title("Gradient sensitivity: Step vs Sigmoid (+20% offset)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Sigmoid shape illustration
    ax = axes[1, 1]
    rec_age_example = TRUE_PARAMS["tau_r_0"]  # ~10.38 days in seconds
    ages_plot = np.linspace(0, (max_age_days + 1) * 86400, 500)
    # Step function
    step_vals = (ages_plot >= rec_age_example).astype(float)
    # Sigmoid function
    sigmoid_vals = 1.0 / (1.0 + np.exp(-K_SIGMOID * (ages_plot - rec_age_example)))

    ages_days_plot = ages_plot / 86400.0
    rec_age_days = rec_age_example / 86400.0
    ax.plot(ages_days_plot, step_vals, label="Step (original)", linewidth=2, color="tab:blue")
    ax.plot(ages_days_plot, sigmoid_vals, label="Sigmoid", linewidth=2, color="tab:orange", linestyle="--")
    ax.axvline(x=rec_age_days, color="gray", linestyle=":", alpha=0.7, label=f"τ_r = {rec_age_days:.1f} d")
    ax.axvline(x=rec_age_days - TRANSITION_DAYS, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=rec_age_days + TRANSITION_DAYS, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Cohort age (days)")
    ax.set_ylabel("Recruitment fraction")
    ax.set_title(f"Recruitment function (transition ±{TRANSITION_DAYS} day)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Step vs Sigmoid Recruitment — LMTL 0D Comparison", fontsize=14)
    fig.tight_layout()
    plt.savefig("examples/test_sigmoid_recruitment_results.png", dpi=150)
    print("\nPlot saved to examples/test_sigmoid_recruitment_results.png")
