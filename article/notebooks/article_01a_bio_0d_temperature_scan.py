"""Article 01A: Validation Biologie 0D - Scan en Température.

Objectif: Valider la convergence asymptotique des processus biologiques
sur une plage continue de températures (0°C à 35°C).

Approche Adaptative:
La dynamique biologique dépend fortement de la température :
- À basse température, la mortalité (λ) est faible → convergence lente → simulation longue.
- À haute température, la mortalité est élevée → convergence rapide → simulation courte.

Nous adaptons dynamiquement la durée de simulation et le pas de temps pour chaque
température afin d'optimiser le coût de calcul tout en garantissant la précision.
"""

from pathlib import Path

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

# LMTL Biological Parameters
LMTL_E = 0.1668  # Transfer efficiency [dimensionless]
LMTL_LAMBDA_0 = 1 / 150  # Base mortality rate [1/day]
LMTL_GAMMA_LAMBDA = 0.15  # Thermal sensitivity of mortality [1/°C]
LMTL_TAU_R_0 = 10.38  # Base recruitment age [days]
LMTL_GAMMA_TAU_R = 0.11  # Thermal sensitivity of recruitment [1/°C]
LMTL_T_REF = 0.0  # Reference temperature [°C]

# NPP Forcing
NPP_MG_M2_DAY = 300.0  # Net Primary Production [mg/m²/day]
NPP_G_M2_S = (NPP_MG_M2_DAY * 1e-3) / 86400.0  # Convert to SI

# Temperature Scan
TEMP_MIN = 0  # Minimum temperature [°C]
TEMP_MAX = 35  # Maximum temperature [°C]
TEMP_STEP = 1  # Temperature step [°C]
TEMPERATURES = np.arange(TEMP_MIN, TEMP_MAX + 1, TEMP_STEP)

# Adaptive Simulation Criteria
DURATION_FACTOR = 15.0  # Duration = FACTOR × (1/λ) for > 99.9% convergence
DT_FACTOR = 100.0  # dt = (1/λ) / FACTOR for fine resolution
DT_MIN_SECONDS = 1.0  # Minimum timestep [s]
DT_MAX_SECONDS = 12 * 3600.0  # Maximum timestep [s] (12 hours)

# Figure Settings
FIGURE_PREFIX = "fig_01a_temperature_scan"
FIGURE_FORMATS = ["png"]

print("=" * 80)
print("VALIDATION BIOLOGIE 0D - SCAN EN TEMPÉRATURE")
print("=" * 80)
print(f"E              : {LMTL_E}")
print(f"λ₀             : {LMTL_LAMBDA_0:.6f} day⁻¹")
print(f"γ_λ            : {LMTL_GAMMA_LAMBDA} °C⁻¹")
print(f"τ_r,0          : {LMTL_TAU_R_0} days")
print(f"γ_τr           : {LMTL_GAMMA_TAU_R} °C⁻¹")
print(f"T_ref          : {LMTL_T_REF} °C")
print(f"NPP            : {NPP_G_M2_S:.2e} g/m²/s ({NPP_MG_M2_DAY} mg/m²/day)")
print(f"Températures   : {TEMP_MIN}°C à {TEMP_MAX}°C (pas de {TEMP_STEP}°C)")
print(f"Simulations    : {len(TEMPERATURES)}")
print("=" * 80)

# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "blue": "#0077BB",
    "orange": "#EE7733",
    "green": "#009988",
    "red": "#CC3311",
    "purple": "#AA3377",
}


def save_figure(fig, name, formats=FIGURE_FORMATS):
    """Save figure in specified formats."""
    for fmt in formats:
        filepath = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filepath}")


# =============================================================================
# THEORETICAL CALCULATIONS
# =============================================================================


def compute_theory_and_sim_params(T, NPP, params):
    """Compute theoretical values and adaptive simulation parameters."""
    # Gillooly temperature normalization
    T_thresh = max(T, params["t_ref"])
    T_gillooly = T_thresh / (1 + T_thresh / 273.0)

    # Mortality rate
    lambda_T = params["lambda_0"] * np.exp(params["gamma_lambda"] * (T_gillooly - params["t_ref"]))

    # Recruitment age
    tau_r = params["tau_r_0"] * np.exp(-params["gamma_tau_r"] * (T_gillooly - params["t_ref"]))

    # Equilibrium biomass: B_eq = R / λ where R = E × NPP
    R = params["E"] * NPP
    B_eq = R / lambda_T

    # Adaptive simulation parameters
    time_scale = 1.0 / lambda_T
    duration_seconds = DURATION_FACTOR * time_scale
    dt_seconds = time_scale / DT_FACTOR
    dt_seconds = np.clip(dt_seconds, DT_MIN_SECONDS, DT_MAX_SECONDS)

    return {
        "B_eq": B_eq,
        "lambda": lambda_T,
        "tau_r": tau_r,
        "time_scale_days": time_scale / 86400.0,
        "duration_days": duration_seconds / 86400.0,
        "dt_seconds": dt_seconds,
    }


# =============================================================================
# BLUEPRINT DEFINITION
# =============================================================================

# Cohort setup
max_age_days = int(np.ceil(LMTL_TAU_R_0))
cohort_ages_days = np.arange(0, max_age_days + 1)
cohort_ages_sec = cohort_ages_days * 86400.0
n_cohorts = len(cohort_ages_sec)

blueprint = Blueprint.from_dict(
    {
        "id": "lmtl-0d-validation",
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
            # Temperature normalization
            {
                "func": "lmtl:gillooly_temperature",
                "inputs": {"temp": "forcings.temperature"},
                "outputs": {"return": {"target": "derived.temp_norm", "type": "derived"}},
            },
            # Recruitment age
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
            # NPP injection
            {
                "func": "lmtl:npp_injection",
                "inputs": {
                    "npp": "forcings.primary_production",
                    "efficiency": "parameters.efficiency",
                    "production": "state.production",
                },
                "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
            },
            # Aging flow
            {
                "func": "lmtl:aging_flow",
                "inputs": {
                    "production": "state.production",
                    "cohort_ages": "parameters.cohort_ages",
                    "rec_age": "derived.rec_age",
                },
                "outputs": {"return": {"target": "tendencies.production", "type": "tendency"}},
            },
            # Recruitment flow
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
            # Mortality
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
# SIMULATION LOOP
# =============================================================================

# Parameters dict for theoretical calculations
params_dict = {
    "E": LMTL_E,
    "lambda_0": LMTL_LAMBDA_0 / 86400.0,  # Convert to 1/s
    "gamma_lambda": LMTL_GAMMA_LAMBDA,
    "tau_r_0": LMTL_TAU_R_0 * 86400.0,  # Convert to seconds
    "gamma_tau_r": LMTL_GAMMA_TAU_R,
    "t_ref": LMTL_T_REF,
}

# Pre-compute theoretical values
print("\nPré-calcul des valeurs théoriques...")
print(f"{'T [°C]':<8} {'λ [1/d]':<12} {'1/λ [d]':<12} {'Durée [d]':<12} {'dt [s]':<12} {'B_eq':<10}")
print("-" * 70)

scan_config = {}
for T in TEMPERATURES:
    cfg = compute_theory_and_sim_params(T, NPP_G_M2_S, params_dict)
    scan_config[T] = cfg
    if T % 5 == 0:
        print(
            f"{T:<8} {cfg['lambda'] * 86400:<12.4f} {cfg['time_scale_days']:<12.1f} "
            f"{cfg['duration_days']:<12.1f} {cfg['dt_seconds']:<12.0f} {cfg['B_eq']:<10.4f}"
        )

# Run simulations
print("\nDémarrage du scan en température...")
results_data = []

for i, T in enumerate(TEMPERATURES):
    cfg = scan_config[T]
    dt_val = cfg["dt_seconds"]
    duration_days = cfg["duration_days"]

    # Time configuration
    start_date = "2000-01-01"
    start_pd = pd.to_datetime(start_date)
    n_steps = int(np.ceil(duration_days * 86400.0 / dt_val))
    # Ensure dt is an integer for exact alignment
    dt_int = int(dt_val)
    end_pd = start_pd + pd.Timedelta(seconds=n_steps * dt_int)

    # Create forcing data (constant temperature and NPP)
    # We need at least 2 time points for interpolation
    dates = pd.date_range(start=start_pd, end=end_pd + pd.Timedelta(days=1), periods=2)

    temp_da = xr.DataArray(
        np.full((2, 1, 1), T),
        dims=["T", "Y", "X"],
        coords={"T": dates, "Y": [0], "X": [0]},
    )
    npp_da = xr.DataArray(
        np.full((2, 1, 1), NPP_G_M2_S),
        dims=["T", "Y", "X"],
        coords={"T": dates, "Y": [0], "X": [0]},
    )

    # Config
    config = Config.from_dict(
        {
            "parameters": {
                "lambda_0": {"value": params_dict["lambda_0"]},
                "gamma_lambda": {"value": params_dict["gamma_lambda"]},
                "tau_r_0": {"value": params_dict["tau_r_0"]},
                "gamma_tau_r": {"value": params_dict["gamma_tau_r"]},
                "t_ref": {"value": params_dict["t_ref"]},
                "efficiency": {"value": LMTL_E},
                "cohort_ages": {"value": cohort_ages_sec.tolist()},
            },
            "forcings": {"temperature": temp_da, "primary_production": npp_da},
            "initial_state": {
                "biomass": xr.DataArray(np.zeros((1, 1)), dims=["Y", "X"], coords={"Y": [0], "X": [0]}),
                "production": xr.DataArray(
                    np.zeros((1, 1, n_cohorts)), dims=["Y", "X", "C"], coords={"Y": [0], "X": [0]}
                ),
            },
            "execution": {
                "time_start": start_date,
                "time_end": end_pd.isoformat(),
                "dt": f"{dt_int}s",
                "forcing_interpolation": "linear",
            },
        }
    )

    # Compile and run
    model = compile_model(blueprint, config)
    runner = Runner.simulation()

    if i % 10 == 0:
        print(f"  T={T}°C (dt={dt_int}s, Durée={duration_days:.1f}d, Steps={n_steps})...")

    state, outputs = runner.run(model, export_variables=["biomass"])

    # Extract final biomass
    biomass_final = float(outputs["biomass"].values[-1, 0, 0])
    B_theory = cfg["B_eq"]
    error = 100 * abs(biomass_final - B_theory) / B_theory if B_theory > 0 else 0.0

    results_data.append(
        {
            "Temperature": T,
            "Simulated": biomass_final,
            "Theoretical": B_theory,
            "Error_Percent": error,
            "dt": dt_val,
            "duration": duration_days,
        }
    )

df_results = pd.DataFrame(results_data)
print("Scan terminé.")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\nGénération des figures...")

# Figure: Comparison Simulation vs Theory
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.9, 7), sharex=True)

# Panel A: Absolute biomass
ax1.plot(df_results["Temperature"], df_results["Theoretical"], "k--", label="Théorie", linewidth=1.5)
ax1.plot(
    df_results["Temperature"],
    df_results["Simulated"],
    "o",
    color=COLORS["blue"],
    markersize=4,
    label="Simulation",
    alpha=0.8,
)
ax1.set_ylabel("Biomasse à l'équilibre [g/m²]")
ax1.set_yscale("log")
ax1.set_title("A. Biomasse d'Équilibre vs Température")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel B: Relative Error
ax2.bar(
    df_results["Temperature"],
    df_results["Error_Percent"],
    color=COLORS["red"],
    alpha=0.7,
    width=0.8,
)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_xlabel("Température [°C]")
ax2.set_ylabel("Erreur Relative [%]")
ax2.set_title("B. Précision de la Simulation")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, FIGURE_PREFIX)
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\nStatistiques d'erreur :")
print(f"  Max Error: {df_results['Error_Percent'].max():.4f}%")
print(f"  Mean Error: {df_results['Error_Percent'].mean():.4f}%")
print(f"  Température Max Error: {df_results.loc[df_results['Error_Percent'].idxmax(), 'Temperature']}°C")

if df_results["Error_Percent"].max() < 0.1:
    print("\n✅ VALIDATION RÉUSSIE : L'erreur est négligeable sur toute la plage.")
elif df_results["Error_Percent"].max() < 1.0:
    print("\n✅ VALIDATION RÉUSSIE : L'erreur reste inférieure à 1%.")
else:
    print("\n⚠️ ATTENTION : Certaines erreurs dépassent 1%.")

# Generate summary file
rmse = np.sqrt(((df_results["Simulated"] - df_results["Theoretical"]) ** 2).mean())
nrmse_val = rmse / df_results["Theoretical"].mean()

lambda_values = np.array([scan_config[T]["lambda"] for T in TEMPERATURES])
beq_values = df_results["Theoretical"].values
time_scale_values = np.array([scan_config[T]["time_scale_days"] for T in TEMPERATURES])

summary_filename = f"{FIGURE_PREFIX.replace('fig_', 'notebook_')}_summary.txt"
summary_path = SUMMARY_DIR / summary_filename

with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("ARTICLE 01A: VALIDATION BIOLOGIE 0D - SCAN EN TEMPÉRATURE\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"DATE: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("OBJECTIF:\n")
    f.write("-" * 80 + "\n")
    f.write("Valider la convergence asymptotique des processus biologiques sur une plage\n")
    f.write(f"de températures ({TEMP_MIN}°C à {TEMP_MAX}°C) avec pas de temps adaptatif.\n\n")

    f.write("PARAMÈTRES DU MODÈLE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"E                : {LMTL_E}\n")
    f.write(f"lambda_0         : {LMTL_LAMBDA_0:.6f} day^-1\n")
    f.write(f"gamma_lambda     : {LMTL_GAMMA_LAMBDA} °C^-1\n")
    f.write(f"tau_r_0          : {LMTL_TAU_R_0} days\n")
    f.write(f"gamma_tau_r      : {LMTL_GAMMA_TAU_R} °C^-1\n")
    f.write(f"T_ref            : {LMTL_T_REF} °C\n")
    f.write(f"NPP              : {NPP_G_M2_S:.2e} g/m²/s ({NPP_MG_M2_DAY} mg/m²/day)\n\n")

    f.write("VARIABILITÉ DU SYSTÈME:\n")
    f.write("-" * 80 + "\n")
    f.write(f"λ(T) range       : {lambda_values.min() * 86400:.4f} - {lambda_values.max() * 86400:.4f} day^-1\n")
    f.write(f"1/λ(T) range     : {time_scale_values.min():.2f} - {time_scale_values.max():.1f} days\n")
    f.write(f"B_eq(T) range    : {beq_values.min():.4f} - {beq_values.max():.4f} g/m²\n\n")

    f.write("RÉSULTATS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"NRMSE            : {nrmse_val:.6f}\n")
    f.write(f"Erreur maximale  : {df_results['Error_Percent'].max():.4f}%\n")
    f.write(f"Erreur moyenne   : {df_results['Error_Percent'].mean():.4f}%\n\n")

    f.write("VALIDATION:\n")
    f.write("-" * 80 + "\n")
    if df_results["Error_Percent"].max() < 1.0:
        f.write("✅ VALIDATION RÉUSSIE\n")
    else:
        f.write("⚠️ ATTENTION : Certaines erreurs dépassent 1%\n")

    f.write("\nFICHIERS GÉNÉRÉS:\n")
    f.write("-" * 80 + "\n")
    for fmt in FIGURE_FORMATS:
        f.write(f"- {FIGURE_PREFIX}.{fmt}\n")
    f.write(f"- {summary_filename}\n")

print(f"\nRésumé sauvegardé : {summary_path}")
