# %% [markdown]
# # Zalesak Slotted Disk — Upwind vs QUICK
#
# Side-by-side comparison of the 1st-order upwind scheme (`transport_tendency`)
# and the 3rd-order QUICK scheme (`transport_tendency_quick`) on the Zalesak
# slotted-disk benchmark. Same setup as `02_transport_zalesak_jax.py`:
# solid-body rotation, one full revolution, analytical solution = initial
# condition.
#
# Metrics collected per resolution:
# - **Mass error (%)** — conservation check
# - **NRMSE** — bulk error vs initial state
# - **Max preservation** — how well the 1.0 plateau survives (diffusion proxy)
# - **Min value** — reveals overshoot/undershoot (QUICK is not monotonic)
# - **Convergence order** — log-log slope of NRMSE vs resolution
#
# Expectation: QUICK should drastically reduce NRMSE and preserve the peak
# much better, at the cost of small undershoots (expected ~-0.05 to -0.15)
# that are a known signature of linear high-order schemes on discontinuous
# fields.
#
# Time integration note: both schemes are stepped with **Heun RK2**, not
# explicit Euler. Plain Euler + QUICK is unconditionally unstable (von
# Neumann analysis gives |g| > 1 for any CFL > 0). RK2 stabilizes QUICK up
# to CFL ~0.87 in 1D. Using the same integrator for both schemes gives a
# fair accuracy comparison — any remaining difference is due to spatial
# discretization.

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

# CFL for Heun RK2: QUICK is stable up to CFL ~0.5 in 2D with RK2,
# upwind is stable to CFL ~1.0 with Euler but RK2 gives it a bit more
# margin as well. 0.4 is safe for both.
CFL_TARGET = 0.4
D_DIFFUSION = 0.0

omega = 2 * np.pi / ROTATION_PERIOD

SCHEMES = {
    "upwind": transport_tendency,
    "quick": transport_tendency_quick,
}

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


def run_simulation(scheme_fn, n_cells, label):
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

    @jax.jit
    def step(state):
        # Heun's RK2: y_n+1 = y_n + dt/2 * (k1 + k2), k1 = f(y_n), k2 = f(y_n + dt*k1)
        # Required for stability with QUICK (explicit Euler + QUICK is
        # unconditionally unstable). Applied to upwind too for fair comparison.
        k1 = tendency(state)
        k2 = tendency(state + dt * k1)
        return state + 0.5 * dt * (k1 + k2)

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
        "scheme": label,
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
# ## Run both schemes at each resolution

# %%
print("=" * 84)
print("ZALESAK SLOTTED DISK — UPWIND vs QUICK")
print("=" * 84)

results = {label: [] for label in SCHEMES}
for n_cells in GRID_RESOLUTIONS:
    print(f"\n--- Resolution: {n_cells}x{n_cells} ---")
    for label, fn in SCHEMES.items():
        r = run_simulation(fn, n_cells, label)
        results[label].append(r)
        print(
            f"  {label:7s}  steps={r['n_steps']:5d}  t={r['elapsed_s']:6.2f}s  "
            f"mass_err={r['mass_error_pct']:.2e}%  "
            f"NRMSE={r['nrmse']:.4f}  "
            f"max={r['max_preservation']:.3f}  "
            f"min={r['min_value']:+.3f}"
        )

# %% [markdown]
# ## Summary table

# %%
print("\n" + "=" * 92)
print("SUMMARY")
print("=" * 92)
print(f"{'Scheme':<8} {'N':<6} {'NRMSE':<10} {'Mass err %':<12} {'Max pres':<10} {'Min val':<10} {'Time [s]':<10}")
print("-" * 92)
for label in SCHEMES:
    for r in results[label]:
        print(
            f"{label:<8} {r['n_cells']:<6} {r['nrmse']:<10.4f} "
            f"{r['mass_error_pct']:<12.2e} {r['max_preservation']:<10.4f} "
            f"{r['min_value']:<+10.4f} {r['elapsed_s']:<10.2f}"
        )
    print()

# %% [markdown]
# ## Convergence order

# %%
slopes = {}
for label in SCHEMES:
    res = np.array([r["n_cells"] for r in results[label]])
    nrmse = np.array([r["nrmse"] for r in results[label]])
    dx_vals = DOMAIN_SIZE / res
    slope, _ = np.polyfit(np.log10(dx_vals), np.log10(nrmse), 1)
    slopes[label] = slope
    print(f"{label:7s} convergence slope (NRMSE vs dx): {slope:.2f}")

# %% [markdown]
# ## Visualization — final states per scheme

# %%
print("\n--- Generating figures ---")
n_rows = len(GRID_RESOLUTIONS)
n_cols = 3  # initial, upwind, quick

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

for i, n_cells in enumerate(GRID_RESOLUTIONS):
    r_up = results["upwind"][i]
    r_qk = results["quick"][i]

    im0 = axes[i, 0].imshow(r_up["state_init"], origin="lower", cmap="viridis", vmin=-0.2, vmax=1.1)
    axes[i, 0].set_title(f"Initial — {n_cells}×{n_cells}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(r_up["state_final"], origin="lower", cmap="viridis", vmin=-0.2, vmax=1.1)
    axes[i, 1].set_title(
        f"Upwind — NRMSE={r_up['nrmse']:.3f}\nmax={r_up['max_preservation']:.2f}  min={r_up['min_value']:+.2f}"
    )
    axes[i, 1].axis("off")

    axes[i, 2].imshow(r_qk["state_final"], origin="lower", cmap="viridis", vmin=-0.2, vmax=1.1)
    axes[i, 2].set_title(
        f"QUICK — NRMSE={r_qk['nrmse']:.3f}\nmax={r_qk['max_preservation']:.2f}  min={r_qk['min_value']:+.2f}"
    )
    axes[i, 2].axis("off")

fig.suptitle("Zalesak — 1 revolution — Upwind (1st order) vs QUICK (3rd order)", fontsize=14)
fig.tight_layout()
Path("examples/images").mkdir(parents=True, exist_ok=True)
fields_file = "examples/images/02b_transport_zalesak_quick_fields.png"
fig.savefig(fields_file, dpi=150)
print(f"  Saved: {fields_file}")

# %% [markdown]
# ## Convergence plot

# %%
fig, ax = plt.subplots(figsize=(7, 5))
resolutions = np.array(GRID_RESOLUTIONS)

for label, color in [("upwind", "tab:orange"), ("quick", "tab:blue")]:
    nrmse_vals = np.array([r["nrmse"] for r in results[label]])
    ax.loglog(
        resolutions,
        nrmse_vals,
        "o-",
        color=color,
        linewidth=2,
        markersize=8,
        label=f"{label} (slope={slopes[label]:.2f})",
    )
    for n, e in zip(resolutions, nrmse_vals, strict=True):
        ax.annotate(f"{e:.3f}", (n, e), textcoords="offset points", xytext=(6, 4), fontsize=8)

# Reference slopes
nrmse_up_0 = results["upwind"][0]["nrmse"]
ax.loglog(
    resolutions,
    nrmse_up_0 * (resolutions[0] / resolutions),
    "--",
    color="gray",
    alpha=0.4,
    label="Order 1 ref",
)
ax.loglog(
    resolutions,
    nrmse_up_0 * (resolutions[0] / resolutions) ** 2,
    ":",
    color="gray",
    alpha=0.4,
    label="Order 2 ref",
)

ax.set_xlabel("Grid resolution N")
ax.set_ylabel("NRMSE")
ax.set_title("Zalesak — Convergence (upwind vs QUICK)")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
convergence_file = "examples/images/02b_transport_zalesak_quick_convergence.png"
fig.savefig(convergence_file, dpi=150)
print(f"  Saved: {convergence_file}")

# %%
print("\nDone.")
