# %% [markdown]
# # Zalesak Slotted Disk Test — JAX Transport
#
# Classical benchmark for advection schemes (Zalesak, 1979). A **slotted disk**
# (binary 0/1 field with a rectangular slot cut out) is placed off-center in the
# domain and advected by a **solid-body rotation** field centered at the domain
# midpoint.
#
# Because solid-body rotation has a uniform angular velocity everywhere, every
# point in the field rotates at the same rate — the disk undergoes **no
# deformation**. After exactly one full revolution (`T = 2pi/omega`), the
# analytical solution is identical to the initial condition. Any difference
# measured at `t = T` is therefore purely **numerical error** from the transport
# scheme (diffusion, dispersion, mass loss).
#
# **Convergence study**: the time step `dt` is derived from a fixed CFL number
# (`dt = CFL * dx / v_max`), so refining the grid also refines `dt` while
# keeping the CFL constant across resolutions. This is standard practice for
# advection schemes whose spatial and temporal errors are coupled through the
# CFL number — it ensures the scheme operates in the same regime at every
# resolution. A final adjustment rounds `dt` so that `n_steps * dt` lands
# exactly at `t = T` (one full revolution).
#
# This test validates:
# 1. Mass conservation
# 2. Numerical diffusion (profile spreading)
# 3. Shape preservation
# 4. JAX differentiability

# %%
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from seapopym.functions.transport import BoundaryType, transport_tendency

# Enable 64-bit precision for fair comparison with Numba (float64)
jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Configuration — Zalesak (1979) Slotted Disk

# %%
DOMAIN_SIZE = 1.0  # Normalized domain [0, 1]
GRID_RESOLUTIONS = [32, 64, 128, 256, 512, 1024]  # Test resolutions

# Slotted Disk parameters
DISK_CENTER_X = 0.50
DISK_CENTER_Y = 0.75
DISK_RADIUS = 0.15
SLOT_WIDTH = 0.05
SLOT_HEIGHT = 0.25
DISK_VALUE = 1.0

# Rotation
ROTATION_CENTER_X = 0.5
ROTATION_CENTER_Y = 0.5
ROTATION_PERIOD = 1.0
N_REVOLUTIONS = 1

# Numerical
CFL_TARGET = 0.5
D_DIFFUSION = 0.0  # No physical diffusion

omega = 2 * np.pi / ROTATION_PERIOD

# %% [markdown]
# ## Helper Functions

# %%
def create_slotted_disk(nx, ny):
    """Create a slotted disk (Zalesak test case)."""
    x = (jnp.arange(nx) + 0.5) / nx
    y = (jnp.arange(ny) + 0.5) / ny
    X, Y = jnp.meshgrid(x, y)

    # Disk
    distance = jnp.sqrt((X - DISK_CENTER_X) ** 2 + (Y - DISK_CENTER_Y) ** 2)
    disk = distance <= DISK_RADIUS

    # Slot
    slot_left = DISK_CENTER_X - SLOT_WIDTH / 2
    slot_right = DISK_CENTER_X + SLOT_WIDTH / 2
    slot_bottom = DISK_CENTER_Y - SLOT_HEIGHT
    slot_top = DISK_CENTER_Y

    slot = (slot_left <= X) & (slot_right >= X) & (slot_bottom <= Y) & (slot_top >= Y)

    # Disk minus slot
    slotted_disk = jnp.where(disk & ~slot, DISK_VALUE, 0.0)
    return slotted_disk


def create_rotation_velocity(nx, ny):
    """Create solid body rotation velocity field."""
    x = (jnp.arange(nx) + 0.5) / nx
    y = (jnp.arange(ny) + 0.5) / ny
    X, Y = jnp.meshgrid(x, y)

    u = -omega * (Y - ROTATION_CENTER_Y)
    v = omega * (X - ROTATION_CENTER_X)
    return u, v


def run_simulation_jax(n_cells, use_jit=True):
    """Run transport simulation using JAX."""
    nx = ny = n_cells
    dx = dy = DOMAIN_SIZE / n_cells

    # Initial condition
    state = create_slotted_disk(nx, ny)
    initial_mass = jnp.sum(state) * dx * dy

    # Velocity field
    u, v = create_rotation_velocity(nx, ny)

    # Grid parameters (uniform)
    dx_arr = jnp.full((ny, nx), dx)
    dy_arr = jnp.full((ny, nx), dy)
    face_height = dy_arr  # For simple grid
    face_width = dx_arr
    cell_area = dx_arr * dy_arr
    D_arr = jnp.full((ny, nx), D_DIFFUSION)
    mask = jnp.ones((ny, nx))

    # Time stepping
    v_max = omega * 0.5
    dt = CFL_TARGET * dx / v_max
    n_steps = int((ROTATION_PERIOD * N_REVOLUTIONS) / dt)
    dt = (ROTATION_PERIOD * N_REVOLUTIONS) / n_steps

    # Define step function
    def step(state):
        adv, diff = transport_tendency(
            state,
            u,
            v,
            D_arr,
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
        return state + dt * (adv + diff)

    # JIT compile if requested
    if use_jit:
        step = jax.jit(step)
        # Warm-up JIT
        _ = step(state)

    # Run simulation
    start_time = time.time()
    for _ in tqdm(range(n_steps), desc=f"JAX {n_cells}x{n_cells}", leave=False):
        state = step(state)
    elapsed = time.time() - start_time

    # Compute metrics
    final_mass = jnp.sum(state) * dx * dy
    mass_error_pct = 100 * abs(float(final_mass - initial_mass)) / float(initial_mass)

    # L2 error vs initial
    state_init = create_slotted_disk(nx, ny)
    error = state - state_init
    l2_error = jnp.sqrt(jnp.sum(error**2) * dx * dy)
    l2_norm_init = jnp.sqrt(jnp.sum(state_init**2) * dx * dy)
    nrmse = float(l2_error / l2_norm_init)

    # Peak preservation
    max_init = float(jnp.max(state_init))
    max_final = float(jnp.max(state))
    max_preservation = max_final / max_init

    return {
        "n_cells": n_cells,
        "n_steps": n_steps,
        "dt": dt,
        "elapsed_s": elapsed,
        "mass_error_pct": mass_error_pct,
        "nrmse": nrmse,
        "max_preservation": max_preservation,
        "state_init": np.array(state_init),
        "state_final": np.array(state),
    }

# %% [markdown]
# ## Run Simulations

# %%
print("=" * 80)
print("ZALESAK SLOTTED DISK TEST - JAX TRANSPORT")
print("=" * 80)

results = []

for n_cells in GRID_RESOLUTIONS:
    print(f"\n--- Resolution: {n_cells}x{n_cells} ---")
    res = run_simulation_jax(n_cells, use_jit=True)
    results.append(res)

    print(f"  Steps: {res['n_steps']}, Time: {res['elapsed_s']:.2f}s")
    print(f"  Mass error: {res['mass_error_pct']:.2e}%")
    print(f"  NRMSE: {res['nrmse']:.4f}")
    print(f"  Max preservation: {res['max_preservation']:.4f}")

# %% [markdown]
# ## Summary & Convergence

# %%
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"{'Resolution':<12} {'Steps':<8} {'Time [s]':<10} {'Mass Err %':<12} {'NRMSE':<10} {'Max Pres':<10}")
print("-" * 80)
for r in results:
    print(
        f"{r['n_cells']}x{r['n_cells']:<6} {r['n_steps']:<8} {r['elapsed_s']:<10.2f} "
        f"{r['mass_error_pct']:<12.2e} {r['nrmse']:<10.4f} {r['max_preservation']:<10.4f}"
    )

# Convergence analysis
dx_values = np.array([DOMAIN_SIZE / r["n_cells"] for r in results])
nrmse_values = np.array([r["nrmse"] for r in results])

log_dx = np.log10(dx_values)
log_nrmse = np.log10(nrmse_values)
slope, intercept = np.polyfit(log_dx, log_nrmse, 1)

print(f"\nConvergence order (slope): {slope:.2f}")

# %% [markdown]
# ## Differentiability Test

# %%
print("\n--- Testing JAX Differentiability ---")
n_test = 32
state = create_slotted_disk(n_test, n_test)
u, v = create_rotation_velocity(n_test, n_test)
dx = dy = DOMAIN_SIZE / n_test
dx_arr = jnp.full((n_test, n_test), dx)
dy_arr = jnp.full((n_test, n_test), dy)
D_arr = jnp.zeros((n_test, n_test))
mask = jnp.ones((n_test, n_test))
cell_area = dx_arr * dy_arr

def loss_fn(state):
    """Compute sum of squared transport tendencies for gradient testing."""
    adv, diff = transport_tendency(
        state,
        u,
        v,
        D_arr,
        dx_arr,
        dy_arr,
        dy_arr,
        dx_arr,
        cell_area,
        mask,
        bc_north=0,
        bc_south=0,
        bc_east=0,
        bc_west=0,
    )
    return jnp.sum((adv + diff) ** 2)

grad = jax.grad(loss_fn)(state)
print(f"  Gradient shape: {grad.shape}")
print(f"  Gradient has NaN: {bool(jnp.any(jnp.isnan(grad)))}")
print(f"  Gradient non-zero: {bool(jnp.any(grad != 0))}")

# %% [markdown]
# ## Visualization

# %%
# Plot comparison (3 columns x 4 rows: 6 pairs initial/final)
print("\n--- Generating Figure ---")
n_cols = 3
n_rows = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

for i, r in enumerate(results):
    col = i % n_cols
    row_init = (i // n_cols) * 2
    row_final = row_init + 1

    axes[row_init, col].imshow(r["state_init"], origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[row_init, col].set_title(f"Initial {r['n_cells']}x{r['n_cells']}")
    axes[row_init, col].axis("off")

    axes[row_final, col].imshow(r["state_final"], origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[row_final, col].set_title(f"Final (NRMSE={r['nrmse']:.3f})")
    axes[row_final, col].axis("off")

plt.suptitle("Zalesak Test - JAX Transport (1 revolution)", fontsize=14)
plt.tight_layout()
Path("examples/images").mkdir(parents=True, exist_ok=True)
fields_file = "examples/images/02_transport_zalesak_jax_fields.png"
plt.savefig(fields_file, dpi=150)
print(f"  Saved: {fields_file}")

# %%
# Convergence plot (NRMSE vs resolution)
fig, ax = plt.subplots(figsize=(6, 4))
resolutions = np.array([r["n_cells"] for r in results])
nrmse_values = np.array([r["nrmse"] for r in results])

ax.loglog(resolutions, nrmse_values, "o-", color="tab:blue", linewidth=2, markersize=7)
ax.loglog(resolutions, nrmse_values[0] * (resolutions[0] / resolutions), "--",
          color="gray", alpha=0.5, label=f"Order 1 (slope={slope:.2f})")
for i, r in enumerate(results):
    ax.annotate(f"{r['nrmse']:.3f}", (resolutions[i], nrmse_values[i]),
                textcoords="offset points", xytext=(5, 8), fontsize=8)
ax.set_xlabel("Grid resolution (N)")
ax.set_ylabel("NRMSE")
ax.set_title(f"Convergence (slope={slope:.2f})")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
convergence_file = "examples/images/02_transport_zalesak_jax_convergence.png"
plt.savefig(convergence_file, dpi=150)
print(f"  Saved: {convergence_file}")
