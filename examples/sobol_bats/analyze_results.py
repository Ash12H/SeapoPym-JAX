"""Analyze Sobol results for Station BATS and produce summary figure.

Reads the checkpoint parquet, recomputes Sobol indices (S1, ST) with
confidence intervals via SALib, averages across extraction points,
and produces a bar chart comparing S1 and ST per parameter.

Usage:
    uv run python examples/sobol_bats/analyze_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze

# =============================================================================
# Paths
# =============================================================================
CHECKPOINT_PATH = Path(__file__).parent / "checkpoint.parquet"
FIGURE_PATH = Path(__file__).parent / "sobol_indices.png"

# Must match run_sobol.py
PARAM_NAMES = ["efficiency", "tau_r_0", "gamma_tau_r", "lambda_0", "gamma_lambda"]
PARAM_LABELS = {
    "efficiency": r"$\varepsilon$",
    "tau_r_0": r"$\tau_{r,0}$",
    "gamma_tau_r": r"$\gamma_{\tau_r}$",
    "lambda_0": r"$\lambda_0$",
    "gamma_lambda": r"$\gamma_\lambda$",
}
N_SAMPLES = 2**14
N_POINTS = 5
POINT_LABELS = [
    "BATS center",
    "South",
    "North",
    "West",
    "East",
]


def load_and_compute():
    """Load checkpoint, compute Sobol indices per QoI and point."""
    df = pd.read_parquet(CHECKPOINT_PATH)

    problem = {
        "num_vars": len(PARAM_NAMES),
        "names": PARAM_NAMES,
        "bounds": [
            [0.0, 1.0],
            [0.0, 30.0 * 86400.0],
            [0.0, 0.5],
            [1.0 / 1000.0 / 86400.0, 1.0 / 4.0 / 86400.0],
            [0.0, 0.5],
        ],
    }

    qoi_names = []
    for qoi in ["mean", "var", "argmax", "median"]:
        col = f"qoi_{qoi}_point_0"
        if col in df.columns and df[col].notna().all():
            qoi_names.append(qoi)

    print(f"Valid QoI: {qoi_names}")

    results = {}
    for qoi in qoi_names:
        s1_list, st_list, s1_conf_list, st_conf_list = [], [], [], []
        for pt in range(N_POINTS):
            col = f"qoi_{qoi}_point_{pt}"
            y = df[col].values

            si = sobol_analyze.analyze(problem, y, calc_second_order=False)
            s1_list.append(si["S1"])
            st_list.append(si["ST"])
            s1_conf_list.append(si["S1_conf"])
            st_conf_list.append(si["ST_conf"])

        results[qoi] = {
            "S1": np.array(s1_list),
            "ST": np.array(st_list),
            "S1_conf": np.array(s1_conf_list),
            "ST_conf": np.array(st_conf_list),
        }

    return results, qoi_names


def plot_mean_indices(results, qoi_names):
    """Plot mean S1 and ST across points with confidence bars."""
    n_qoi = len(qoi_names)
    fig, axes = plt.subplots(1, n_qoi, figsize=(5 * n_qoi, 5), squeeze=False)

    for i, qoi in enumerate(qoi_names):
        ax = axes[0, i]
        data = results[qoi]

        s1_mean = data["S1"].mean(axis=0)
        st_mean = data["ST"].mean(axis=0)
        s1_conf_mean = data["S1_conf"].mean(axis=0)
        st_conf_mean = data["ST_conf"].mean(axis=0)

        x = np.arange(len(PARAM_NAMES))
        width = 0.35

        ax.bar(
            x - width / 2, s1_mean, width,
            yerr=s1_conf_mean, capsize=4,
            label="S1 (first-order)", color="#4C72B0", alpha=0.85,
        )
        ax.bar(
            x + width / 2, st_mean, width,
            yerr=st_conf_mean, capsize=4,
            label="ST (total-order)", color="#DD8452", alpha=0.85,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([PARAM_LABELS[p] for p in PARAM_NAMES], fontsize=12)
        ax.set_ylabel("Sobol index", fontsize=11)
        ax.set_title(f"QoI: {qoi}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0] - 0.02))
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Sobol Sensitivity — BATS (N={N_SAMPLES}, mean over {N_POINTS} points)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {FIGURE_PATH}")


def print_summary(results, qoi_names):
    """Print a text summary of mean indices."""
    for qoi in qoi_names:
        data = results[qoi]
        s1_mean = data["S1"].mean(axis=0)
        st_mean = data["ST"].mean(axis=0)
        s1_conf_mean = data["S1_conf"].mean(axis=0)
        st_conf_mean = data["ST_conf"].mean(axis=0)

        print(f"\n{'='*60}")
        print(f"QoI: {qoi} — mean over {N_POINTS} extraction points")
        print(f"{'='*60}")
        print(f"{'Parameter':>20s}   {'S1':>8s}  {'S1_conf':>8s}   {'ST':>8s}  {'ST_conf':>8s}")
        print("-" * 60)

        order = np.argsort(-st_mean)
        for j in order:
            print(
                f"{PARAM_NAMES[j]:>20s}   {s1_mean[j]:8.4f}  {s1_conf_mean[j]:8.4f}   "
                f"{st_mean[j]:8.4f}  {st_conf_mean[j]:8.4f}"
            )


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}. Run run_sobol.py first.")

    print("Analyzing Sobol results for Station BATS...")
    results, qoi_names = load_and_compute()
    print_summary(results, qoi_names)
    plot_mean_indices(results, qoi_names)


if __name__ == "__main__":
    main()
