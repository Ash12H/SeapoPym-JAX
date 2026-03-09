# %% [markdown]
# # LMTL Model — Floating-Point Precision Comparison (no transport)
#
# Compares float64, float32, float16, and float8 precision on a 2D grid
# simulation using the LMTL_NO_TRANSPORT blueprint.
#
# Only the **second year** is displayed (year 1 = spin-up).

# %%
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_dtypes
import numpy as np
import pandas as pd
import xarray as xr

import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import simulate
from seapopym.models import LMTL_NO_TRANSPORT

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])
print(f"JAX device: {jax.devices('cpu')[0]}")

# %% [markdown]
# ## Parameters

# %%
LMTL_E = 0.1668
LMTL_LAMBDA_0 = 1 / 150  # 1/day
LMTL_GAMMA_LAMBDA = 0.15  # 1/degC
LMTL_TAU_R_0 = 10.38  # days
LMTL_GAMMA_TAU_R = 0.11  # 1/degC
LMTL_T_REF = 0.0  # degC

DTYPES = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "float8": ml_dtypes.float8_e5m2,
}

PLOT_FILE = "examples/images/01_lmtl_no_transport.png"

# %% [markdown]
# ## Common Setup

# %%
blueprint = LMTL_NO_TRANSPORT

max_age_days = int(np.ceil(LMTL_TAU_R_0))
cohort_ages_sec = np.arange(0, max_age_days + 1) * 86400.0
n_cohorts = len(cohort_ages_sec)

start_date = "2000-01-01"
end_date = "2002-01-01"  # 2 years (year 1 = spin-up)
dt = "3h"

start_pd = pd.to_datetime(start_date)
end_pd = pd.to_datetime(end_date)
n_days = (end_pd - start_pd).days + 5
dates = pd.date_range(start=start_pd, periods=n_days, freq="D")

grid_size = (1, 1)
ny, nx = grid_size
lat = np.arange(ny)
lon = np.arange(nx)

day_of_year = dates.dayofyear.values

# Base forcing arrays (float64)
doy_float = day_of_year.astype(np.float64)

temp_c = 15.0 + 5.0 * np.sin(2 * np.pi * day_of_year / 365.0)
temp_4d = np.broadcast_to(temp_c[:, None, None, None], (len(dates), 1, ny, nx))

npp_day = 1.0 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.0)
npp_sec = npp_day / 86400.0
npp_3d = np.broadcast_to(npp_sec[:, None, None], (len(dates), ny, nx))

# Parameter values (float64)
param_vals = {
    "lambda_0": LMTL_LAMBDA_0 / 86400.0,
    "gamma_lambda": LMTL_GAMMA_LAMBDA,
    "tau_r_0": LMTL_TAU_R_0 * 86400.0,
    "gamma_tau_r": LMTL_GAMMA_TAU_R,
    "t_ref": LMTL_T_REF,
    "efficiency": LMTL_E,
}

# %% [markdown]
# ## Run for each precision

# %%
results = {}

for name, dtype in DTYPES.items():
    info = jnp.finfo(dtype)
    print(f"\n{'=' * 50}")
    print(f"Running with {name} (eps={float(info.eps):.2e}, max={float(info.max):.2e})")
    print(f"{'=' * 50}")

    cast = lambda a: np.array(a, dtype=dtype)  # noqa: E731

    config = Config(
        parameters={
            "lambda_0": xr.DataArray([cast(param_vals["lambda_0"])], dims=["F"]),
            "gamma_lambda": xr.DataArray([cast(param_vals["gamma_lambda"])], dims=["F"]),
            "tau_r_0": xr.DataArray([cast(param_vals["tau_r_0"])], dims=["F"]),
            "gamma_tau_r": xr.DataArray([cast(param_vals["gamma_tau_r"])], dims=["F"]),
            "t_ref": xr.DataArray(cast(param_vals["t_ref"])),
            "efficiency": xr.DataArray([cast(param_vals["efficiency"])], dims=["F"]),
            "cohort_ages": xr.DataArray(cast(cohort_ages_sec), dims=["C"]),
            "day_layer": xr.DataArray([0], dims=["F"]),
            "night_layer": xr.DataArray([0], dims=["F"]),
        },
        forcings={
            "latitude": xr.DataArray(cast(np.array([30.0])), dims=["Y"], coords={"Y": lat}),
            "temperature": xr.DataArray(
                cast(temp_4d), dims=["T", "Z", "Y", "X"],
                coords={"T": dates, "Z": np.arange(1), "Y": lat, "X": lon},
            ),
            "primary_production": xr.DataArray(
                cast(npp_3d), dims=["T", "Y", "X"],
                coords={"T": dates, "Y": lat, "X": lon},
            ),
            "day_of_year": xr.DataArray(
                cast(doy_float), dims=["T"],
                coords={"T": dates},
            ),
        },
        initial_state={
            "biomass": xr.DataArray(
                np.zeros((1, ny, nx)), dims=["F", "Y", "X"], coords={"Y": lat, "X": lon}
            ),
            "production": xr.DataArray(
                np.zeros((1, n_cohorts, ny, nx)), dims=["F", "C", "Y", "X"],
                coords={"Y": lat, "X": lon},
            ),
        },
        execution={
            "time_start": start_date,
            "time_end": end_date,
            "dt": dt,
            "forcing_interpolation": "linear",
        },
    )

    try:
        model = compile_model(blueprint, config)
        t0 = time.time()
        state, outputs = simulate(model, chunk_size=800, export_variables=["biomass"])
        elapsed = time.time() - t0

        biomass = outputs["biomass"]  # xarray
        biomass_mean = biomass.mean(dim=("Y", "X"))
        plot_dates = biomass_mean.coords["T"].values

        # Filter to year 2 only
        year2_mask = plot_dates >= np.datetime64("2001-01-01")
        results[name] = {
            "dates": plot_dates[year2_mask],
            "biomass": biomass_mean.values[year2_mask],
            "elapsed": elapsed,
        }
        bmin = float(np.nanmin(results[name]["biomass"]))
        bmax = float(np.nanmax(results[name]["biomass"]))
        print(f"  Done in {elapsed:.2f}s — biomass range: [{bmin:.6f}, {bmax:.6f}]")
    except Exception as e:
        print(f"  FAILED: {e}")
        results[name] = None

# %% [markdown]
# ## Visualization

# %%
dtype_colors = {"float64": "black", "float32": "tab:blue", "float16": "tab:orange", "float8": "tab:red"}
dtype_names = list(DTYPES.keys())
ref = results.get("float64")
ref_bio = ref["biomass"].astype(np.float64) if ref is not None else None

n_dtypes = len(dtype_names)
fig, axes = plt.subplots(n_dtypes, 1, figsize=(12, 3 * n_dtypes), sharex=True)

for idx, name in enumerate(dtype_names):
    ax = axes[idx]
    res = results[name]
    color = dtype_colors[name]

    # Always draw float64 reference in light grey
    if ref is not None:
        ax.plot(ref["dates"], ref_bio, color="lightgrey", linewidth=4, label="float64 ref", zorder=1)

    if res is None:
        ax.text(0.5, 0.5, "FAILED", transform=ax.transAxes, ha="center", va="center",
                fontsize=20, color=color, fontweight="bold", alpha=0.6)
        status = "failed"
    else:
        bio = res["biomass"].astype(np.float64)
        has_nan = np.any(np.isnan(bio))
        if has_nan:
            ax.text(0.5, 0.5, "NaN (overflow)", transform=ax.transAxes, ha="center", va="center",
                    fontsize=16, color=color, fontweight="bold", alpha=0.6)
            status = "NaN"
        else:
            ax.plot(res["dates"], bio, color=color, linewidth=1.5, label=name, zorder=2)
            if ref_bio is not None and name != "float64":
                max_err = float(np.max(np.abs(bio - ref_bio) / (np.abs(ref_bio) + 1e-30)))
                status = f"max rel. error = {max_err:.2e}"
                ax.text(0.98, 0.95, status, transform=ax.transAxes, ha="right", va="top",
                        fontsize=9, color=color, fontstyle="italic",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                status = "reference"

    info = jnp.finfo(DTYPES[name])
    elapsed_str = f" — {res['elapsed']:.2f}s" if res is not None else ""
    ax.set_title(f"{name}  (eps={float(info.eps):.1e}, max={float(info.max):.1e}{elapsed_str})",
                 loc="left", fontsize=10, color=color, fontweight="bold")
    ax.set_ylabel("Biomass (g/m²)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

axes[-1].set_xlabel("Date (year 2)")
fig.suptitle("LMTL 0D — Floating-Point Precision Comparison", fontsize=13, fontweight="bold")
fig.tight_layout()
Path(PLOT_FILE).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(PLOT_FILE, dpi=150)
print(f"\nPlot saved to {PLOT_FILE}")
