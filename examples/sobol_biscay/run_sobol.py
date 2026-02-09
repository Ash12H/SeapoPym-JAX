"""Sobol sensitivity analysis on Bay of Biscay real SEAPODYM data.

Runs a full Sobol analysis on a 20x20 grid with 30 cohorts and 60
monthly timesteps (2000-2005), using the LMTL+Transport model with
checkpoint support for pause/resume.

Domain: 20x20 degrees around Bay of Biscay (45.5N, 4W)
Temperate NE Atlantic waters: SST ~10 to 25C.

Parameters varied:
  - efficiency: [0, 1]
  - tau_r_0: [0, 30 days] in seconds
  - gamma_tau_r: [0, 0.5] 1/delta_degC
  - lambda_0: [1/1000, 1/4] per day in 1/s
  - gamma_lambda: [0, 0.5] 1/delta_degC

Usage:
    uv run python examples/sobol_biscay/run_sobol.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import seapopym.functions.lmtl  # noqa: F401 — register LMTL functions
import seapopym.functions.transport  # noqa: F401 — register transport functions
from seapopym.blueprint import Blueprint, Config
from seapopym.compiler import compile_model
from seapopym.sensitivity.sobol import SobolAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
FORCINGS_PATH = Path(__file__).parent / "forcings.zarr"
CHECKPOINT_PATH = Path(__file__).parent / "checkpoint.parquet"
RESULTS_PATH = Path(__file__).parent / "sobol_results.csv"

N_COHORTS = 30
N_SAMPLES = 2**14
BATCH_SIZE = 2**14 * 2
CHUNK_SIZE = 500

# Extraction points (Y, X indices within 20x20 grid)
EXTRACTION_POINTS = [
    (9, 9),    # Biscay center (~45N, 4W)
    (4, 9),    # South (~40N, 4W)
    (14, 9),   # North (~50N, 4W)
    (9, 4),    # West (~45N, 9W)
    (9, 14),   # East (~45N, 1E)
]

PARAM_BOUNDS = {
    "efficiency": (0.0, 1.0),
    "tau_r_0": (0.0, 30.0 * 86400.0),
    "gamma_tau_r": (0.0, 0.5),
    "lambda_0": (1.0 / 1000.0 / 86400.0, 1.0 / 4.0 / 86400.0),
    "gamma_lambda": (0.0, 0.5),
}

NOMINAL_PARAMS = {
    "lambda_0": 1.0 / 150.0 / 86400.0,
    "gamma_lambda": 0.15,
    "tau_r_0": 10.38 * 86400.0,
    "gamma_tau_r": 0.11,
    "t_ref": 0.0,
    "efficiency": 0.1668,
}


def create_blueprint() -> Blueprint:
    """Create LMTL + Transport blueprint."""
    return Blueprint.from_dict(
        {
            "id": "sobol-biscay",
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
                    "u": {"units": "m/s", "dims": ["Y", "X"]},
                    "v": {"units": "m/s", "dims": ["Y", "X"]},
                    "D": {"units": "m^2/s", "dims": ["Y", "X"]},
                    "dx": {"units": "m", "dims": ["Y", "X"]},
                    "dy": {"units": "m", "dims": ["Y", "X"]},
                    "face_height": {"units": "m", "dims": ["Y", "X"]},
                    "face_width": {"units": "m", "dims": ["Y", "X"]},
                    "cell_area": {"units": "m^2", "dims": ["Y", "X"]},
                    "mask": {"units": "dimensionless", "dims": ["Y", "X"]},
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
                    "func": "lmtl:aging_flow",
                    "inputs": {
                        "production": "state.production",
                        "cohort_ages": "parameters.cohort_ages",
                        "rec_age": "derived.rec_age",
                    },
                    "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
                },
                {
                    "func": "lmtl:recruitment_flow",
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
                {
                    "func": "phys:transport_tendency",
                    "inputs": {
                        "state": "state.biomass",
                        "u": "forcings.u",
                        "v": "forcings.v",
                        "D": "forcings.D",
                        "dx": "forcings.dx",
                        "dy": "forcings.dy",
                        "face_height": "forcings.face_height",
                        "face_width": "forcings.face_width",
                        "cell_area": "forcings.cell_area",
                        "mask": "forcings.mask",
                    },
                    "outputs": {
                        "advection_rate": {"target": "tendencies.biomass", "type": "tendency"},
                        "diffusion_rate": {"target": "tendencies.biomass", "type": "tendency"},
                    },
                },
                {
                    "func": "phys:transport_tendency",
                    "inputs": {
                        "state": "state.production",
                        "u": "forcings.u",
                        "v": "forcings.v",
                        "D": "forcings.D",
                        "dx": "forcings.dx",
                        "dy": "forcings.dy",
                        "face_height": "forcings.face_height",
                        "face_width": "forcings.face_width",
                        "cell_area": "forcings.cell_area",
                        "mask": "forcings.mask",
                    },
                    "outputs": {
                        "advection_rate": {"target": "tendencies.production", "type": "tendency"},
                        "diffusion_rate": {"target": "tendencies.production", "type": "tendency"},
                    },
                },
            ],
        }
    )


def build_config() -> Config:
    """Build Config by loading forcings from Zarr."""
    if not FORCINGS_PATH.exists():
        raise FileNotFoundError(f"Forcings not found at {FORCINGS_PATH}. Run generate_forcings.py first.")

    ds = xr.open_zarr(FORCINGS_PATH)
    NY = len(ds.Y)
    NX = len(ds.X)

    cohort_ages = np.arange(N_COHORTS) * 86400.0

    forcings = {
        "temperature": ds["temperature"],
        "primary_production": ds["primary_production"],
        "u": ds["u"],
        "v": ds["v"],
        "D": ds["D"],
        "dx": ds["dx"],
        "dy": ds["dy"],
        "face_height": ds["face_height"],
        "face_width": ds["face_width"],
        "cell_area": ds["cell_area"],
        "mask": ds["mask"],
    }

    lat = ds.Y.values
    lon = ds.X.values

    return Config.from_dict(
        {
            "parameters": {
                **{k: {"value": v} for k, v in NOMINAL_PARAMS.items()},
                "cohort_ages": xr.DataArray(cohort_ages, dims=["C"]),
            },
            "forcings": forcings,
            "initial_state": {
                "biomass": xr.DataArray(
                    np.zeros((NY, NX), dtype=np.float32),
                    dims=["Y", "X"],
                    coords={"Y": lat, "X": lon},
                ),
                "production": xr.DataArray(
                    np.zeros((NY, NX, N_COHORTS), dtype=np.float32),
                    dims=["Y", "X", "C"],
                    coords={"Y": lat, "X": lon},
                ),
            },
            "execution": {
                "time_start": str(ds.T.values[0])[:10],
                "time_end": str(ds.T.values[-1])[:10],
                "dt": "1d",
                "forcing_interpolation": "linear",
            },
        }
    )


def main():
    logger.info("=" * 70)
    logger.info("SOBOL SENSITIVITY — Bay of Biscay (real data, 20x20, 30 cohorts)")
    logger.info("=" * 70)

    logger.info("Building blueprint and config...")
    blueprint = create_blueprint()
    config = build_config()

    logger.info("Compiling model (JAX backend)...")
    model = compile_model(blueprint, config, backend="jax")

    n_evals = N_SAMPLES * (len(PARAM_BOUNDS) + 2)
    logger.info(
        f"Sobol setup: N={N_SAMPLES}, D={len(PARAM_BOUNDS)}, "
        f"total evaluations={n_evals}, "
        f"batch_size={BATCH_SIZE}, chunk_size={CHUNK_SIZE}"
    )

    analyzer = SobolAnalyzer(model)
    result = analyzer.analyze(
        param_bounds=PARAM_BOUNDS,
        extraction_points=EXTRACTION_POINTS,
        output_variable="biomass",
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        chunk_size=CHUNK_SIZE,
        checkpoint_path=str(CHECKPOINT_PATH),
        qoi=["mean", "var", "argmax", "median"],
    )

    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)

    print("\n--- First-order Sobol indices (S1) ---")
    print(result.S1.to_string(float_format="%.4f"))

    print("\n--- Total-order Sobol indices (ST) ---")
    print(result.ST.to_string(float_format="%.4f"))

    print("\n--- S1 confidence intervals ---")
    print(result.S1_conf.to_string(float_format="%.4f"))

    s1_csv = result.S1.copy()
    s1_csv.columns = [f"S1_{c}" for c in s1_csv.columns]
    st_csv = result.ST.copy()
    st_csv.columns = [f"ST_{c}" for c in st_csv.columns]
    combined = pd.concat([s1_csv, st_csv], axis=1)
    combined.to_csv(RESULTS_PATH)
    logger.info(f"Results saved to {RESULTS_PATH}")

    tps = result.time_per_sim
    print(f"\n--- Timing ---")
    print(f"  Average time per simulation: {tps:.3f} s/sim ({1 / tps:.1f} sim/s)")
    for target_n in [1_000, 10_000, 100_000, 1_000_000]:
        hours = target_n * tps / 3600
        if hours < 1:
            print(f"  Estimated for {target_n:>10,} sims: {hours * 60:.1f} min")
        else:
            print(f"  Estimated for {target_n:>10,} sims: {hours:.1f} h")

    print("\n--- Parameter ranking (mean |ST| across all QoI/points) ---")
    mean_st = result.ST.abs().mean(axis=0).sort_values(ascending=False)
    for param, val in mean_st.items():
        print(f"  {param:20s}: {val:.4f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
