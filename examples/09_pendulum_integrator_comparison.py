# %% [markdown]
# # Pendulum Energy Conservation — Euler vs RK2 vs RK4
#
# A free pendulum is a Hamiltonian system: total energy (kinetic + potential)
# should be conserved exactly. Forward Euler **injects energy** at every step,
# causing the pendulum to gain amplitude and eventually spin. RK2 and RK4
# have much smaller energy drift, producing physically correct oscillations.
#
# This example runs the CartPole physics through the SeapoPym engine with
# `force = 0` and a very large cart mass (fixed cart), comparing the three
# integrators available via `ExecutionParams.integrator`.

# %%
import base64
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter

from seapopym.blueprint import Blueprint, Config, functional
from seapopym.compiler import compile_model
from seapopym.engine.run import run
from seapopym.engine.step import build_step_fn

PALETTE = ["#1B4965", "#62B6CB", "#E8833A", "#5FA8D3", "#2D9B4E"]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3})

# %% [markdown]
# ## Configuration

# %%
DT = 0.02  # seconds — large enough for Euler energy drift to be visible
DURATION = 15.0  # seconds — ~10 pendulum periods
N_STEPS = int(DURATION / DT)
THETA_0 = np.pi / 2  # initial angle: 90 degrees (horizontal)
MASS_CART = 1e6  # effectively infinite — cart doesn't move
MASS_POLE = 0.1  # kg
HALF_LENGTH = 0.5  # m (pole length = 1m)
GRAVITY = 9.81  # m/s^2

INTEGRATORS = ["euler", "rk2", "rk4"]
COLORS = {"euler": PALETTE[2], "rk2": PALETTE[1], "rk4": PALETTE[4]}
LABELS = {"euler": "Euler (order 1)", "rk2": "RK2 Heun (order 2)", "rk4": "RK4 (order 4)"}

# %% [markdown]
# ## Physics — reuse CartPole functions with zero force


# %%
@functional(
    name="cp:velocity",
    units={
        "x_dot": "m/s",
        "theta_dot": "rad/s",
        "dx_dt": "m/s",
        "dtheta_dt": "rad/s",
    },
    outputs=("dx_dt", "dtheta_dt"),
)
def velocity(x_dot, theta_dot):
    """Identity: velocity is the time derivative of position."""
    return x_dot, theta_dot


@functional(
    name="cp:angular_accel",
    units={
        "theta": "rad",
        "theta_dot": "rad/s",
        "force": "N",
        "gravity": "m/s^2",
        "mass_cart": "kg",
        "mass_pole": "kg",
        "half_length": "m",
        "return": "rad/s^2",
    },
)
def angular_accel(theta, theta_dot, force, gravity, mass_cart, mass_pole, half_length):
    """Pendulum angular acceleration from Barto et al. (1983)."""
    total_mass = mass_cart + mass_pole
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    return (
        gravity * sin_theta + cos_theta * ((-force - mass_pole * half_length * theta_dot**2 * sin_theta) / total_mass)
    ) / (half_length * (4.0 / 3.0 - (mass_pole * cos_theta**2) / total_mass))


@functional(
    name="cp:linear_accel",
    units={
        "theta": "rad",
        "theta_dot": "rad/s",
        "theta_ddot": "rad/s^2",
        "force": "N",
        "mass_cart": "kg",
        "mass_pole": "kg",
        "half_length": "m",
        "return": "m/s^2",
    },
)
def linear_accel(theta, theta_dot, theta_ddot, force, mass_cart, mass_pole, half_length):
    """Cart linear acceleration — depends on angular acceleration."""
    total_mass = mass_cart + mass_pole
    return (
        force + mass_pole * half_length * (theta_dot**2 * jnp.sin(theta) - theta_ddot * jnp.cos(theta))
    ) / total_mass


# %% [markdown]
# ## Blueprint

# %%
blueprint = Blueprint.from_dict(
    {
        "id": "pendulum-integrator-comparison",
        "version": "1.0",
        "declarations": {
            "state": {
                "x": {"units": "m", "dims": ["Y", "X"]},
                "x_dot": {"units": "m/s", "dims": ["Y", "X"]},
                "theta": {"units": "rad", "dims": ["Y", "X"]},
                "theta_dot": {"units": "rad/s", "dims": ["Y", "X"]},
            },
            "parameters": {
                "gravity": {"units": "m/s^2"},
                "mass_cart": {"units": "kg"},
                "mass_pole": {"units": "kg"},
                "half_length": {"units": "m"},
                "force": {"units": "N", "dims": ["T"]},
            },
            "forcings": {},
            "derived": {
                "dx_dt": {"units": "m/s"},
                "dtheta_dt": {"units": "rad/s"},
                "theta_ddot": {"units": "rad/s^2"},
                "x_ddot": {"units": "m/s^2"},
            },
        },
        "process": [
            {
                "func": "cp:velocity",
                "inputs": {"x_dot": "state.x_dot", "theta_dot": "state.theta_dot"},
                "outputs": {"dx_dt": "derived.dx_dt", "dtheta_dt": "derived.dtheta_dt"},
            },
            {
                "func": "cp:angular_accel",
                "inputs": {
                    "theta": "state.theta",
                    "theta_dot": "state.theta_dot",
                    "force": "parameters.force",
                    "gravity": "parameters.gravity",
                    "mass_cart": "parameters.mass_cart",
                    "mass_pole": "parameters.mass_pole",
                    "half_length": "parameters.half_length",
                },
                "outputs": {"return": "derived.theta_ddot"},
            },
            {
                "func": "cp:linear_accel",
                "inputs": {
                    "theta": "state.theta",
                    "theta_dot": "state.theta_dot",
                    "theta_ddot": "derived.theta_ddot",
                    "force": "parameters.force",
                    "mass_cart": "parameters.mass_cart",
                    "mass_pole": "parameters.mass_pole",
                    "half_length": "parameters.half_length",
                },
                "outputs": {"return": "derived.x_ddot"},
            },
        ],
        "tendencies": {
            "x": [{"source": "derived.dx_dt"}],
            "x_dot": [{"source": "derived.x_ddot"}],
            "theta": [{"source": "derived.dtheta_dt"}],
            "theta_dot": [{"source": "derived.theta_ddot"}],
        },
    }
)

# %% [markdown]
# ## Run with each integrator

# %%
results = {}

for integrator in INTEGRATORS:
    config = Config(
        parameters={
            "gravity": xr.DataArray(GRAVITY),
            "mass_cart": xr.DataArray(MASS_CART),
            "mass_pole": xr.DataArray(MASS_POLE),
            "half_length": xr.DataArray(HALF_LENGTH),
            "force": xr.DataArray(np.zeros(N_STEPS), dims=["T"]),
        },
        forcings={},
        initial_state={
            "x": xr.DataArray(np.array([[0.0]]), dims=["Y", "X"]),
            "x_dot": xr.DataArray(np.array([[0.0]]), dims=["Y", "X"]),
            "theta": xr.DataArray(np.array([[THETA_0]]), dims=["Y", "X"]),
            "theta_dot": xr.DataArray(np.array([[0.0]]), dims=["Y", "X"]),
        },
        execution={
            "time_start": "2000-01-01",
            "time_end": f"2000-01-01T00:00:{int(DURATION):02d}",
            "dt": f"{DT}s",
            "integrator": integrator,
        },
    )

    model = compile_model(blueprint, config)
    step_fn = build_step_fn(model, export_variables=["theta", "theta_dot"])
    _, outputs = run(step_fn, model, dict(model.state), dict(model.parameters))

    theta = np.asarray(outputs["theta"][:, 0, 0])
    theta_dot = np.asarray(outputs["theta_dot"][:, 0, 0])

    # Energy: E = 0.5 * I_eff * theta_dot^2 + m*g*L_cm*cos(theta)
    # CartPole convention: theta=0 = upright, so potential = m*g*L_cm*cos(theta)
    # For uniform rod on fixed pivot: I_eff = (1/3)*m*(2*half_length)^2
    pole_length = 2 * HALF_LENGTH
    I_eff = (1 / 3) * MASS_POLE * pole_length**2
    E_kinetic = 0.5 * I_eff * theta_dot**2
    E_potential = MASS_POLE * GRAVITY * HALF_LENGTH * np.cos(theta)
    E_total = E_kinetic + E_potential
    E_initial = MASS_POLE * GRAVITY * HALF_LENGTH * np.cos(THETA_0)  # = 0 for pi/2

    # Normalize energy drift by m*g*L_cm (E_initial=0 for theta_0=pi/2)
    E_scale = MASS_POLE * GRAVITY * HALF_LENGTH
    results[integrator] = {
        "theta": theta,
        "theta_dot": theta_dot,
        "energy": E_total,
        "energy_drift": (E_total - E_initial) / E_scale,
    }

    print(
        f"  {integrator:5s}  final_theta={theta[-1]:+8.2f} rad  "
        f"energy_drift={(E_total[-1] - E_initial) / E_scale:+.3f} mgl  "
        f"max_|theta|={np.max(np.abs(theta)):.2f} rad ({np.degrees(np.max(np.abs(theta))):.0f} deg)"
    )

# %% [markdown]
# ## Plot 1 — Angle and energy over time

# %%
time_axis = np.arange(N_STEPS) * DT

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Panel 1: theta(t)
for integrator in INTEGRATORS:
    theta_deg = np.degrees(results[integrator]["theta"])
    axes[0].plot(time_axis, theta_deg, color=COLORS[integrator], linewidth=1.5, label=LABELS[integrator], alpha=0.85)

axes[0].axhline(0, color="gray", linestyle="--", alpha=0.4)
axes[0].axhline(180, color="gray", linestyle=":", alpha=0.3)
axes[0].axhline(-180, color="gray", linestyle=":", alpha=0.3)
axes[0].set_ylabel("Pole angle (deg)")
axes[0].set_title(
    f"Free Pendulum — Euler injects energy, RK2/RK4 conserve it  (dt = {DT}s)",
    fontsize=13,
    fontweight="bold",
    color=PALETTE[0],
)
axes[0].legend(loc="upper left", fontsize=10)

# Panel 2: energy drift (E(t) - E(0)) / mgl
for integrator in INTEGRATORS:
    axes[1].plot(
        time_axis,
        results[integrator]["energy_drift"],
        color=COLORS[integrator],
        linewidth=1.5,
        label=LABELS[integrator],
        alpha=0.85,
    )

axes[1].axhline(0.0, color="gray", linestyle="--", alpha=0.4, label="Exact (no drift)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("(E(t) - E(0)) / mgl")
axes[1].set_title("Energy drift — Euler injects energy, RK conserves it", fontweight="bold", color=PALETTE[0])
axes[1].legend(loc="upper left", fontsize=10)

fig.tight_layout()
Path("examples/images").mkdir(parents=True, exist_ok=True)
fig.savefig("examples/images/09_pendulum_integrator_comparison.png", dpi=150)
print(f"  Saved: examples/images/09_pendulum_integrator_comparison.png")

# %% [markdown]
# ## Plot 2 — Animation GIF (3 panels side by side)

# %%
POLE_LEN = 2 * HALF_LENGTH
SIM_FPS = 25
GIF_FPS = 25
frame_step = max(1, int(1.0 / (SIM_FPS * DT)))
frame_indices = np.arange(0, N_STEPS, frame_step)
n_frames = len(frame_indices)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("white")

elements = []
for ax, integrator in zip(axes, INTEGRATORS, strict=True):
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(LABELS[integrator], fontsize=12, fontweight="bold", color=PALETTE[0])
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors="#718096", labelsize=8)

    # Pivot point
    ax.plot(0, 0, "o", color="#1B4965", markersize=8, zorder=5)

    # Pole
    (pole_line,) = ax.plot([], [], color=COLORS[integrator], linewidth=4, solid_capstyle="round", zorder=3)
    (pole_tip,) = ax.plot([], [], "o", color=COLORS[integrator], markersize=10, zorder=4)

    # Trail (faded path of the tip)
    (trail,) = ax.plot([], [], color=COLORS[integrator], linewidth=0.5, alpha=0.3, zorder=1)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, color=PALETTE[0], fontweight="bold")
    energy_text = ax.text(0.02, 0.88, "", transform=ax.transAxes, fontsize=9, color="#718096")

    theta_arr = results[integrator]["theta"]
    energy_drift = results[integrator]["energy_drift"]
    elements.append((pole_line, pole_tip, trail, time_text, energy_text, theta_arr, energy_drift))

plt.tight_layout()


def animate(frame_idx):
    artists = []
    i = frame_indices[frame_idx]
    t = i * DT

    for pole_line, pole_tip, trail, time_text, energy_text, theta_arr, energy_drift in elements:
        th = float(theta_arr[i])
        # CartPole convention: theta=0 = upright, theta=pi = hanging
        # Tip position: (L*sin(theta), L*cos(theta))
        tip_x = POLE_LEN * np.sin(th)
        tip_y = POLE_LEN * np.cos(th)

        pole_line.set_data([0, tip_x], [0, tip_y])
        pole_tip.set_data([tip_x], [tip_y])

        # Trail: show last ~2 seconds of tip positions
        trail_start = max(0, i - int(2.0 / DT))
        trail_thetas = theta_arr[trail_start : i + 1 : max(1, frame_step // 2)]
        trail_x = POLE_LEN * np.sin(trail_thetas)
        trail_y = POLE_LEN * np.cos(trail_thetas)
        trail.set_data(trail_x, trail_y)

        time_text.set_text(f"t = {t:.1f}s")
        energy_text.set_text(f"dE = {energy_drift[i]:+.2f} mgl")
        artists.extend([pole_line, pole_tip, trail, time_text, energy_text])

    return artists


gif_path = "examples/images/09_pendulum_integrator_comparison.gif"
anim = FuncAnimation(fig, animate, frames=n_frames, interval=1000 / GIF_FPS, blit=True)
anim.save(gif_path, writer=PillowWriter(fps=GIF_FPS))
plt.close()
print(f"  Saved: {gif_path} ({n_frames} frames at {GIF_FPS} fps)")

# %%
print("\nDone.")
