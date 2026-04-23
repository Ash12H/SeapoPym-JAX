# %% [markdown]
# # Zalesak Slotted Disk — Spatial × Temporal scheme comparison
#
# Full 2×2 comparison of spatial schemes (upwind, QUICK) crossed with
# temporal integrators (Heun RK2, classical RK4). This shows the
# independent contributions of spatial accuracy (order 1 vs 3) and
# temporal accuracy (order 2 vs 4) on the same benchmark.

# %%
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from seapopym.functions.transport import (
    BoundaryType,
    transport_tendency,
    transport_tendency_quick,
)

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Configuration

# %%
DOMAIN_SIZE = 1.0
GRID_RESOLUTIONS = [32, 64, 128, 256]

DISK_CENTER_X = 0.50
DISK_CENTER_Y = 0.75
DISK_RADIUS = 0.15
SLOT_WIDTH = 0.05
SLOT_HEIGHT = 0.25
DISK_VALUE = 1.0

ROTATION_CENTER_X = 0.5
ROTATION_CENTER_Y = 0.5
ROTATION_PERIOD = 1.0
N_REVOLUTIONS = 1

CFL_TARGET = 0.4
D_DIFFUSION = 0.0

omega = 2 * np.pi / ROTATION_PERIOD

SPATIAL_SCHEMES = {
    "upwind": transport_tendency,
    "quick": transport_tendency_quick,
}

INTEGRATORS = ["rk2", "rk4"]

# %% [markdown]
# ## Helpers


# %%
def create_slotted_disk(nx, ny):
    x = (jnp.arange(nx) + 0.5) / nx
    y = (jnp.arange(ny) + 0.5) / ny
    X, Y = jnp.meshgrid(x, y)
    distance = jnp.sqrt((X - DISK_CENTER_X) ** 2 + (Y - DISK_CENTER_Y) ** 2)
    disk = distance <= DISK_RADIUS
    slot_left = DISK_CENTER_X - SLOT_WIDTH / 2
    slot_right = DISK_CENTER_X + SLOT_WIDTH / 2
    slot_bottom = DISK_CENTER_Y - SLOT_HEIGHT
    slot_top = DISK_CENTER_Y
    slot = (slot_left <= X) & (slot_right >= X) & (slot_bottom <= Y) & (slot_top >= Y)
    return jnp.where(disk & ~slot, DISK_VALUE, 0.0)


def create_rotation_velocity(nx, ny):
    x = (jnp.arange(nx) + 0.5) / nx
    y = (jnp.arange(ny) + 0.5) / ny
    X, Y = jnp.meshgrid(x, y)
    u = -omega * (Y - ROTATION_CENTER_Y)
    v = omega * (X - ROTATION_CENTER_X)
    return u, v


def run_simulation(scheme_fn, integrator, n_cells, label):
    nx = ny = n_cells
    dx = dy = DOMAIN_SIZE / n_cells

    state = create_slotted_disk(nx, ny)
    initial_mass = jnp.sum(state) * dx * dy

    u, v = create_rotation_velocity(nx, ny)

    dx_arr = jnp.full((ny, nx), dx)
    dy_arr = jnp.full((ny, nx), dy)
    face_height = dy_arr
    face_width = dx_arr
    cell_area = dx_arr * dy_arr
    mask = jnp.ones((ny, nx))

    v_max = omega * 0.5
    dt = CFL_TARGET * dx / v_max
    n_steps = int((ROTATION_PERIOD * N_REVOLUTIONS) / dt)
    dt = (ROTATION_PERIOD * N_REVOLUTIONS) / n_steps

    def tendency(s):
        adv, diff = scheme_fn(
            s,
            u,
            v,
            D_DIFFUSION,
            dx_arr,
            dy_arr,
            face_height,
            face_width,
            cell_area,
            mask,
            bc_north=BoundaryType.CLOSED,
            bc_south=BoundaryType.CLOSED,
            bc_east=BoundaryType.CLOSED,
            bc_west=BoundaryType.CLOSED,
        )
        return adv + diff

    if integrator == "rk2":

        @jax.jit
        def step(state):
            k1 = tendency(state)
            k2 = tendency(state + dt * k1)
            return state + 0.5 * dt * (k1 + k2)

    elif integrator == "rk4":

        @jax.jit
        def step(state):
            k1 = tendency(state)
            k2 = tendency(state + 0.5 * dt * k1)
            k3 = tendency(state + 0.5 * dt * k2)
            k4 = tendency(state + dt * k3)
            return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Warm-up JIT
    _ = step(state)

    start = time.time()
    for _ in tqdm(range(n_steps), desc=f"{label} {n_cells}x{n_cells}", leave=False):
        state = step(state)
    elapsed = time.time() - start

    state_init = create_slotted_disk(nx, ny)
    final_mass = jnp.sum(state) * dx * dy
    mass_error_pct = 100 * abs(float(final_mass - initial_mass)) / float(initial_mass)

    error = state - state_init
    l2_error = jnp.sqrt(jnp.sum(error**2) * dx * dy)
    l2_norm_init = jnp.sqrt(jnp.sum(state_init**2) * dx * dy)
    nrmse = float(l2_error / l2_norm_init)

    max_init = float(jnp.max(state_init))
    max_final = float(jnp.max(state))
    max_preservation = max_final / max_init

    min_final = float(jnp.min(state))

    return {
        "label": label,
        "n_cells": n_cells,
        "n_steps": n_steps,
        "dt": dt,
        "elapsed_s": elapsed,
        "mass_error_pct": mass_error_pct,
        "nrmse": nrmse,
        "max_preservation": max_preservation,
        "min_value": min_final,
        "state_init": np.array(state_init),
        "state_final": np.array(state),
    }


# %% [markdown]
# ## Run all 4 combinations at each resolution

# %%
print("=" * 100)
print("ZALESAK SLOTTED DISK — SPATIAL (upwind / QUICK) × TEMPORAL (RK2 / RK4)")
print("=" * 100)

COMBOS = [(s, t) for s in SPATIAL_SCHEMES for t in INTEGRATORS]
combo_labels = [f"{s}+{t}" for s, t in COMBOS]
results = {label: [] for label in combo_labels}

for n_cells in GRID_RESOLUTIONS:
    print(f"\n--- Resolution: {n_cells}x{n_cells} ---")
    for spatial_name, integrator_name in COMBOS:
        label = f"{spatial_name}+{integrator_name}"
        r = run_simulation(SPATIAL_SCHEMES[spatial_name], integrator_name, n_cells, label)
        results[label].append(r)
        print(
            f"  {label:14s}  steps={r['n_steps']:5d}  t={r['elapsed_s']:6.2f}s  "
            f"mass_err={r['mass_error_pct']:.2e}%  "
            f"NRMSE={r['nrmse']:.4f}  "
            f"max={r['max_preservation']:.3f}  "
            f"min={r['min_value']:+.3f}"
        )

# %% [markdown]
# ## Summary table

# %%
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"{'Combo':<16} {'N':<6} {'NRMSE':<10} {'Mass err %':<12} {'Max pres':<10} {'Min val':<10} {'Time [s]':<10}")
print("-" * 100)
for label in combo_labels:
    for r in results[label]:
        print(
            f"{label:<16} {r['n_cells']:<6} {r['nrmse']:<10.4f} "
            f"{r['mass_error_pct']:<12.2e} {r['max_preservation']:<10.4f} "
            f"{r['min_value']:<+10.4f} {r['elapsed_s']:<10.2f}"
        )
    print()

# %% [markdown]
# ## Convergence order

# %%
slopes = {}
for label in combo_labels:
    res = np.array([r["n_cells"] for r in results[label]])
    nrmse = np.array([r["nrmse"] for r in results[label]])
    dx_vals = DOMAIN_SIZE / res
    slope, _ = np.polyfit(np.log10(dx_vals), np.log10(nrmse), 1)
    slopes[label] = slope
    print(f"{label:14s} convergence slope (NRMSE vs dx): {slope:.2f}")

# %% [markdown]
# ## Visualization — final states (4 columns × N resolutions)

# %%
print("\n--- Generating figures ---")
n_rows = len(GRID_RESOLUTIONS)
n_cols = 1 + len(COMBOS)  # initial + 4 combos

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

for i, n_cells in enumerate(GRID_RESOLUTIONS):
    # Initial state (column 0)
    ref = results[combo_labels[0]][i]
    axes[i, 0].imshow(ref["state_init"], origin="lower", cmap="viridis", vmin=-0.2, vmax=1.1)
    axes[i, 0].set_title(f"Initial — {n_cells}×{n_cells}", fontsize=9)
    axes[i, 0].axis("off")

    # One column per combo
    for j, label in enumerate(combo_labels):
        r = results[label][i]
        axes[i, j + 1].imshow(r["state_final"], origin="lower", cmap="viridis", vmin=-0.2, vmax=1.1)
        axes[i, j + 1].set_title(
            f"{label}\nNRMSE={r['nrmse']:.3f}  max={r['max_preservation']:.2f}",
            fontsize=8,
        )
        axes[i, j + 1].axis("off")

fig.suptitle("Zalesak — 1 revolution — Spatial × Temporal comparison", fontsize=13)
fig.tight_layout()
Path("examples/images").mkdir(parents=True, exist_ok=True)
fields_file = "examples/images/02c_transport_zalesak_rk_comparison_fields.png"
fig.savefig(fields_file, dpi=150)
print(f"  Saved: {fields_file}")

# %% [markdown]
# ## Convergence plot — all 4 combos

# %%
fig, ax = plt.subplots(figsize=(8, 5))
resolutions = np.array(GRID_RESOLUTIONS)

colors = {
    "upwind+rk2": "tab:orange",
    "upwind+rk4": "tab:red",
    "quick+rk2": "tab:blue",
    "quick+rk4": "tab:green",
}
markers = {"upwind+rk2": "s", "upwind+rk4": "D", "quick+rk2": "o", "quick+rk4": "^"}

for label in combo_labels:
    nrmse_vals = np.array([r["nrmse"] for r in results[label]])
    ax.loglog(
        resolutions,
        nrmse_vals,
        f"{markers[label]}-",
        color=colors[label],
        linewidth=2,
        markersize=7,
        label=f"{label} (slope={slopes[label]:.2f})",
    )
    for n, e in zip(resolutions, nrmse_vals, strict=True):
        ax.annotate(f"{e:.3f}", (n, e), textcoords="offset points", xytext=(6, 4), fontsize=7)

# Reference slopes
nrmse_ref = results["upwind+rk2"][0]["nrmse"]
ax.loglog(
    resolutions,
    nrmse_ref * (resolutions[0] / resolutions),
    "--",
    color="gray",
    alpha=0.3,
    label="Order 1 ref",
)
ax.loglog(
    resolutions,
    nrmse_ref * (resolutions[0] / resolutions) ** 2,
    ":",
    color="gray",
    alpha=0.3,
    label="Order 2 ref",
)

ax.set_xlabel("Grid resolution N")
ax.set_ylabel("NRMSE")
ax.set_title("Zalesak — Convergence: Spatial × Temporal")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
convergence_file = "examples/images/02c_transport_zalesak_rk_comparison_convergence.png"
fig.savefig(convergence_file, dpi=150)
print(f"  Saved: {convergence_file}")

# %%
print("\nDone.")
