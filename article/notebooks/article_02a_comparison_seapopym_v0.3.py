"""Article 02A: Comparaison SeapoPym JAX vs SeapoPym v0.3.

Valide que la nouvelle architecture JAX reproduit fidèlement les résultats
de SeapoPym v0.3 en configuration sans transport (0D biologie sur grille 2D).

Cette comparaison correspond à la section 1.2 des Résultats de l'article :
> "Comparaison avec SeapoPym v0.3 (Sans Transport)"
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Import LMTL functions (registers them with @functional decorator)
import seapopym.functions.lmtl  # noqa: F401
from seapopym.blueprint import Blueprint, Config
from seapopym.compiler import compile_model
from seapopym.engine import Runner

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
FIGURES_DIR = BASE_DIR.parent / "figures"
SUMMARY_DIR = BASE_DIR.parent / "summary"
FIGURES_DIR.mkdir(exist_ok=True)
SUMMARY_DIR.mkdir(exist_ok=True)

# External Data Paths
DATA_EXTERNAL_BASE = Path("/Users/adm-lehodey/Documents/Workspace/Projects/phd_optimization/notebooks/Article_1/data")
ZARR_FORCINGS = DATA_EXTERNAL_BASE / "1_global/post_processed_light_global_multiyear_bgc_001_033.zarr"
ZARR_V03 = DATA_EXTERNAL_BASE / "2_global_simulation/biomass_global.zarr"

# LMTL Biological Parameters (IDENTICAL to v0.3)
LMTL_E = 0.1668  # Transfer efficiency [dimensionless]
LMTL_LAMBDA_0 = 1 / 150  # Base mortality rate [1/day]
LMTL_GAMMA_LAMBDA = 0.15  # Thermal sensitivity of mortality [1/°C]
LMTL_TAU_R_0 = 10.38  # Base recruitment age [days]
LMTL_GAMMA_TAU_R = 0.11  # Thermal sensitivity of recruitment [1/°C]
LMTL_T_REF = 0.0  # Reference temperature [°C]
LMTL_DAY_LAYER = 0  # Daytime depth layer index
LMTL_NIGHT_LAYER = 0  # Nighttime depth layer index

# Time Configuration
START_DATE = "1998-01-02"
END_DATE = "2019-12-31"
TIMESTEP_DAYS = 1

# Comparison Period (exclude spin-up)
COMPARISON_START = "2000"
COMPARISON_END = "2019"

# Validation Thresholds
THRESHOLD_CORRELATION = 0.99
THRESHOLD_BIAS = 0.01  # g/m²
THRESHOLD_L2_ERROR = 5.0  # %
THRESHOLD_NRMSE = 0.5

# Figure Settings
FIGURE_PREFIX = "fig_02a_comparison_v03"
FIGURE_DPI = 300

print("=" * 80)
print("COMPARAISON SeapoPym JAX vs SeapoPym v0.3")
print("=" * 80)
print(f"Données externes : {DATA_EXTERNAL_BASE}")
print(f"Période          : {START_DATE} → {END_DATE}")
print(f"Comparaison      : {COMPARISON_START} → {COMPARISON_END}")
print("=" * 80)

# =============================================================================
# LOAD FORCING DATA
# =============================================================================

print(f"\nChargement des forçages depuis : {ZARR_FORCINGS}")
ds_raw = xr.open_zarr(ZARR_FORCINGS)

# Rename dimensions to SeapoPym standard
ds = ds_raw.rename({"T": "time", "Z": "z", "Y": "y", "X": "x"})

# Select time range and variables
forcings_raw = ds.sel(time=slice("1998-01-01", "2020-01-01"))
forcings_raw = forcings_raw[["primary_production", "temperature"]].load()

print(f"  Variables : {list(forcings_raw.data_vars)}")
print(f"  Période   : {forcings_raw.time.values[0]} → {forcings_raw.time.values[-1]}")
print(f"  Grille    : {len(forcings_raw.y)} × {len(forcings_raw.x)}")
print(f"  Couches Z : {len(forcings_raw.z)}")

# =============================================================================
# PREPARE FORCINGS FOR BLUEPRINT
# =============================================================================

# Extract coordinates
times = forcings_raw.time
lats = forcings_raw.y
lons = forcings_raw.x
depths = forcings_raw.z

# Day of year for day_length calculation
day_of_year = xr.DataArray(
    pd.to_datetime(times.values).dayofyear,
    dims=["T"],
    coords={"T": times.values},
)

# Latitude broadcast to (T, Y, X) for day_length
latitude_broadcast = xr.DataArray(
    np.broadcast_to(lats.values[None, :, None], (len(times), len(lats), len(lons))),
    dims=["T", "Y", "X"],
    coords={"T": times.values, "Y": lats.values, "X": lons.values},
)

# Day of year broadcast to (T, Y, X)
doy_broadcast = xr.DataArray(
    np.broadcast_to(day_of_year.values[:, None, None], (len(times), len(lats), len(lons))),
    dims=["T", "Y", "X"],
    coords={"T": times.values, "Y": lats.values, "X": lons.values},
)

# Temperature with Z dimension -> (T, Z, Y, X)
temperature = forcings_raw["temperature"].rename({"time": "T", "z": "Z", "y": "Y", "x": "X"})

# NPP: convert from mg/m²/day to g/m²/s
npp_raw = forcings_raw["primary_production"].rename({"time": "T", "y": "Y", "x": "X"})
npp_si = npp_raw * 1e-3 / 86400.0  # mg/m²/day -> g/m²/s

# Cohort setup
max_age_days = int(np.ceil(LMTL_TAU_R_0))
cohort_ages_days = np.arange(0, max_age_days + 1)
cohort_ages_sec = cohort_ages_days * 86400.0
n_cohorts = len(cohort_ages_sec)

print(f"  Cohortes  : {n_cohorts}")

# =============================================================================
# BLUEPRINT DEFINITION
# =============================================================================

blueprint = Blueprint.from_dict(
    {
        "id": "lmtl-comparison-v03",
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
                "day_layer": {"units": "dimensionless"},
                "night_layer": {"units": "dimensionless"},
            },
            "forcings": {
                "temperature": {"units": "degC", "dims": ["T", "Z", "Y", "X"]},
                "primary_production": {"units": "g/m^2/s", "dims": ["T", "Y", "X"]},
                "latitude": {"units": "degrees", "dims": ["T", "Y", "X"]},
                "day_of_year": {"units": "dimensionless", "dims": ["T", "Y", "X"]},
            },
        },
        "process": [
            # 1. Day length calculation
            {
                "func": "lmtl:day_length",
                "inputs": {
                    "latitude": "forcings.latitude",
                    "day_of_year": "forcings.day_of_year",
                },
                "outputs": {"return": {"target": "derived.day_length", "type": "derived"}},
            },
            # 2. Layer-weighted mean temperature (DVM)
            {
                "func": "lmtl:layer_weighted_mean",
                "inputs": {
                    "forcing": "forcings.temperature",
                    "day_length": "derived.day_length",
                    "day_layer": "parameters.day_layer",
                    "night_layer": "parameters.night_layer",
                },
                "outputs": {"return": {"target": "derived.mean_temperature", "type": "derived"}},
            },
            # 3. Temperature threshold
            {
                "func": "lmtl:threshold_temperature",
                "inputs": {
                    "temp": "derived.mean_temperature",
                    "min_temp": "parameters.t_ref",
                },
                "outputs": {"return": {"target": "derived.thresh_temp", "type": "derived"}},
            },
            # 4. Gillooly normalization
            {
                "func": "lmtl:gillooly_temperature",
                "inputs": {"temp": "derived.thresh_temp"},
                "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}},
            },
            # 5. Recruitment age
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
            # 6. NPP injection
            {
                "func": "lmtl:npp_injection",
                "inputs": {
                    "npp": "forcings.primary_production",
                    "efficiency": "parameters.efficiency",
                    "production": "state.production",
                },
                "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
            },
            # 7. Aging flow
            {
                "func": "lmtl:aging_flow",
                "inputs": {
                    "production": "state.production",
                    "cohort_ages": "parameters.cohort_ages",
                    "rec_age": "derived.rec_age",
                },
                "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
            },
            # 8. Recruitment flow
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
            # 9. Mortality
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
# CONFIG
# =============================================================================

config = Config.from_dict(
    {
        "parameters": {
            "lambda_0": {"value": LMTL_LAMBDA_0 / 86400.0},  # 1/day -> 1/s
            "gamma_lambda": {"value": LMTL_GAMMA_LAMBDA},
            "tau_r_0": {"value": LMTL_TAU_R_0 * 86400.0},  # days -> seconds
            "gamma_tau_r": {"value": LMTL_GAMMA_TAU_R},
            "t_ref": {"value": LMTL_T_REF},
            "efficiency": {"value": LMTL_E},
            "cohort_ages": xr.DataArray(cohort_ages_sec, dims=["C"]),
            "day_layer": {"value": LMTL_DAY_LAYER},
            "night_layer": {"value": LMTL_NIGHT_LAYER},
        },
        "forcings": {
            "temperature": temperature,
            "primary_production": npp_si,
            "latitude": latitude_broadcast,
            "day_of_year": doy_broadcast,
        },
        "initial_state": {
            "biomass": xr.DataArray(
                np.zeros((len(lats), len(lons))),
                dims=["Y", "X"],
                coords={"Y": lats.values, "X": lons.values},
            ),
            "production": xr.DataArray(
                np.zeros((len(lats), len(lons), n_cohorts)),
                dims=["Y", "X", "C"],
                coords={"Y": lats.values, "X": lons.values},
            ),
        },
        "execution": {
            "time_start": START_DATE,
            "time_end": END_DATE,
            "dt": f"{TIMESTEP_DAYS}d",
            "forcing_interpolation": "nearest",
            "batch_size": 365,
        },
    }
)

# =============================================================================
# RUN SIMULATION
# =============================================================================

print("\nCompilation du modèle...")
model = compile_model(blueprint, config)
print(f"  Timesteps: {model.n_timesteps}")

print("\nExécution de la simulation...")
runner = Runner.simulation(chunk_size=365)
state, outputs = runner.run(model, export_variables=["biomass"])

# Extract results
results_jax = outputs["biomass"].rename("biomass_jax")
print(f"  Résultats JAX : {results_jax.shape}")

# =============================================================================
# LOAD V0.3 REFERENCE
# =============================================================================

print(f"\nChargement des résultats v0.3 depuis : {ZARR_V03}")
seapopym_v03 = xr.open_zarr(ZARR_V03)
seapopym_v03 = seapopym_v03["biomass"].squeeze().rename({"T": "time", "X": "x", "Y": "y"}).load()

# Convert units: v0.3 is in kg/m², we need g/m²
seapopym_v03 = seapopym_v03 * 1000.0  # kg/m² -> g/m²
print(f"  Résultats v0.3 : {seapopym_v03.shape}")
print("  Unités converties : kg/m² -> g/m²")

# =============================================================================
# ALIGN DATA FOR COMPARISON
# =============================================================================

print("\nAlignement des données...")

# Select comparison period
comparison_slice = slice(COMPARISON_START, COMPARISON_END)
v03_aligned = seapopym_v03.sel(time=comparison_slice)
jax_aligned = results_jax.sel(T=comparison_slice).rename({"T": "time", "Y": "y", "X": "x"})

# Normalize timestamps to daily
v03_aligned = v03_aligned.assign_coords(time=pd.to_datetime(v03_aligned.time.values).floor("D"))
jax_aligned = jax_aligned.assign_coords(time=pd.to_datetime(jax_aligned.time.values).floor("D"))

# Align
v03_aligned, jax_aligned = xr.align(v03_aligned, jax_aligned, join="inner")

print(f"  v0.3 shape: {v03_aligned.shape}")
print(f"  JAX shape:  {jax_aligned.shape}")

# =============================================================================
# COMPUTE METRICS
# =============================================================================


def compute_metrics(ref, test):
    """Compute comparison metrics."""
    diff = test - ref
    rmse = float(np.sqrt((diff**2).mean().values))
    corr = float(xr.corr(ref, test).values)
    bias = float(diff.mean().values)
    l2_error = float(np.sqrt((diff**2).sum() / (ref**2).sum()).values) * 100
    std_ref = float(ref.std().values)
    nrmse = rmse / std_ref if std_ref > 0 else np.nan

    return {
        "RMSE (g/m²)": rmse,
        "Corrélation": corr,
        "Biais moyen (g/m²)": bias,
        "Erreur L2 (%)": l2_error,
        "NRMSE": nrmse,
    }


metrics = compute_metrics(v03_aligned, jax_aligned)

print("\n" + "=" * 60)
print("MÉTRIQUES DE VALIDATION")
print("=" * 60)
for key, value in metrics.items():
    print(f"  {key}: {value:.6f}")
print("=" * 60)

# Validation status
all_valid = all(
    [
        metrics["Corrélation"] > THRESHOLD_CORRELATION,
        abs(metrics["Biais moyen (g/m²)"]) < THRESHOLD_BIAS,
        metrics["Erreur L2 (%)"] < THRESHOLD_L2_ERROR,
        metrics["NRMSE"] < THRESHOLD_NRMSE,
    ]
)

if all_valid:
    print("\n✅ VALIDATION RÉUSSIE - Tous les critères sont satisfaits")
else:
    print("\n⚠️ ATTENTION - Certains critères ne sont pas satisfaits")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\nGénération des figures...")

# Figure 1: Mean biomass maps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
v03_aligned.mean("time").plot(ax=axes[0], cmap="viridis", vmin=0)
axes[0].set_title("SeapoPym v0.3 - Biomasse moyenne")
jax_aligned.mean("time").plot(ax=axes[1], cmap="viridis", vmin=0)
axes[1].set_title("SeapoPym JAX - Biomasse moyenne")
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_maps.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURE_PREFIX}_maps.png")

# Figure 2: Bias map
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
diff_mean = (jax_aligned - v03_aligned).mean("time")
vmax_val = max(abs(float(diff_mean.min())), abs(float(diff_mean.max())))
diff_mean.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-vmax_val,
    vmax=vmax_val,
    cbar_kwargs={"label": "Biais (g/m²)", "shrink": 0.8},
)
ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=1)
ax.coastlines(resolution="110m", linewidth=0.2, zorder=2)
ax.set_title("Biais (SeapoPym JAX - v0.3)")
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_bias.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURE_PREFIX}_bias.png")

# Figure 3: Time series
fig, ax = plt.subplots(figsize=(12, 4))
biomass_v03 = v03_aligned.sum(["x", "y"])
biomass_jax = jax_aligned.sum(["x", "y"])
biomass_v03.plot(ax=ax, label="SeapoPym v0.3", alpha=0.8)
biomass_jax.plot(ax=ax, label="SeapoPym JAX", alpha=0.8, linestyle="--")
ax.legend()
ax.set_title("Biomasse totale globale")
ax.set_ylabel("Biomasse totale (g/m²)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_timeseries.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURE_PREFIX}_timeseries.png")

# Figure 4: Scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
sample_v03 = v03_aligned.values.flatten()[::100]
sample_jax = jax_aligned.values.flatten()[::100]
mask = ~np.isnan(sample_v03) & ~np.isnan(sample_jax)
ax.scatter(sample_v03[mask], sample_jax[mask], alpha=0.1, s=1)
ax.plot([0, sample_v03[mask].max()], [0, sample_v03[mask].max()], "r--", label="1:1")
ax.set_xlabel("SeapoPym v0.3 (g/m²)")
ax.set_ylabel("SeapoPym JAX (g/m²)")
ax.set_title(f"Scatter v0.3 vs JAX (R² = {metrics['Corrélation'] ** 2:.4f})")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / f"{FIGURE_PREFIX}_scatter.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close()
print(f"  Saved: {FIGURE_PREFIX}_scatter.png")

# =============================================================================
# SUMMARY
# =============================================================================

summary_filename = f"{FIGURE_PREFIX.replace('fig_', 'notebook_')}_summary.txt"
summary_path = SUMMARY_DIR / summary_filename

results_table = pd.DataFrame(
    {
        "Métrique": list(metrics.keys()),
        "Valeur": [f"{v:.6f}" for v in metrics.values()],
        "Seuil": [
            "—",
            f"> {THRESHOLD_CORRELATION}",
            f"< {THRESHOLD_BIAS}",
            f"< {THRESHOLD_L2_ERROR}%",
            f"< {THRESHOLD_NRMSE}",
        ],
        "Validation": [
            "—",
            "✓" if metrics["Corrélation"] > THRESHOLD_CORRELATION else "✗",
            "✓" if abs(metrics["Biais moyen (g/m²)"]) < THRESHOLD_BIAS else "✗",
            "✓" if metrics["Erreur L2 (%)"] < THRESHOLD_L2_ERROR else "✗",
            "✓" if metrics["NRMSE"] < THRESHOLD_NRMSE else "✗",
        ],
    }
)

with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("COMPARAISON SeapoPym JAX vs SeapoPym v0.3\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("CONFIGURATION:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Période simulation : {START_DATE} → {END_DATE}\n")
    f.write(f"Période comparaison: {COMPARISON_START} → {COMPARISON_END}\n")
    f.write(f"Grille             : {len(lats)} × {len(lons)}\n")
    f.write(f"Cohortes           : {n_cohorts}\n\n")

    f.write("PARAMÈTRES LMTL:\n")
    f.write("-" * 80 + "\n")
    f.write(f"E            : {LMTL_E}\n")
    f.write(f"lambda_0     : {LMTL_LAMBDA_0:.6f} day^-1\n")
    f.write(f"gamma_lambda : {LMTL_GAMMA_LAMBDA} °C^-1\n")
    f.write(f"tau_r_0      : {LMTL_TAU_R_0} days\n")
    f.write(f"gamma_tau_r  : {LMTL_GAMMA_TAU_R} °C^-1\n")
    f.write(f"T_ref        : {LMTL_T_REF} °C\n\n")

    f.write("MÉTRIQUES:\n")
    f.write("-" * 80 + "\n")
    f.write(results_table.to_string(index=False) + "\n\n")

    f.write("VALIDATION:\n")
    f.write("-" * 80 + "\n")
    if all_valid:
        f.write("✅ VALIDATION RÉUSSIE\n")
    else:
        f.write("⚠️ ATTENTION - Certains critères ne sont pas satisfaits\n")

print(f"\nRésumé sauvegardé : {summary_path}")
