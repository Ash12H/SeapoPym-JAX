#!/usr/bin/env python
"""Generate the hero figure for the documentation landing page.

Usage:
    uv run python docs/assets/generate_hero.py

Requires: seapopym[viz,optimization]

Outputs: docs/assets/hero.png
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import optax
import xarray as xr

import seapopym.functions.lotka_volterra  # noqa: F401 — register LV functions
from seapopym.blueprint import Config
from seapopym.compiler import compile_model
from seapopym.engine import build_step_fn, run, simulate
from seapopym.models import LOTKA_VOLTERRA

OUTPUT_DIR = Path(__file__).parent
PALETTE = ["#1B4965", "#62B6CB", "#E8833A", "#5FA8D3"]
DAY = 86400.0

TRUE_PARAMS = {
    "alpha": 0.04 / DAY,
    "beta": 0.005 / DAY,
    "delta": 0.5,
    "gamma": 0.1 / DAY,
}

INITIAL_STATE = {
    "prey": xr.DataArray(np.array([[42.0]]), dims=["Y", "X"]),
    "predator": xr.DataArray(np.array([[7.0]]), dims=["Y", "X"]),
}


def run_simulation():
    """3-year simulation for time series panel."""
    config = Config.from_dict(
        {
            "parameters": {k: xr.DataArray(v) for k, v in TRUE_PARAMS.items()},
            "forcings": {},
            "initial_state": INITIAL_STATE,
            "execution": {"time_start": "2000-01-01", "time_end": "2002-12-31", "dt": "1d"},
        }
    )
    model = compile_model(LOTKA_VOLTERRA, config)
    _, outputs = simulate(model)
    return outputs["prey"].values[:, 0, 0], outputs["predator"].values[:, 0, 0]


def run_gradient_optimization():
    """Twin experiment + Adam optimization for landscape and convergence panels."""
    config = Config.from_dict(
        {
            "parameters": {k: xr.DataArray(v) for k, v in TRUE_PARAMS.items()},
            "forcings": {},
            "initial_state": INITIAL_STATE,
            "execution": {"time_start": "2000-01-01", "time_end": "2000-06-30", "dt": "1d"},
        }
    )
    model = compile_model(LOTKA_VOLTERRA, config)
    step_fn = build_step_fn(model, export_variables=["prey", "predator"])
    _, truth = run(step_fn, model, model.state, model.parameters, chunk_size=None)

    key = jax.random.PRNGKey(42)
    obs_prey = truth["prey"] + 0.05 * truth["prey"] * jax.random.normal(key, truth["prey"].shape)

    def loss_fn(params):
        _, outputs = run(step_fn, model, model.state, params, chunk_size=None)
        return jnp.mean((outputs["prey"] - obs_prey) ** 2)

    # Loss landscape (alpha x gamma)
    n_grid = 30
    alpha_range = np.linspace(0.02 / DAY, 0.06 / DAY, n_grid)
    gamma_range = np.linspace(0.05 / DAY, 0.15 / DAY, n_grid)
    loss_grid = np.zeros((n_grid, n_grid))
    for i, a in enumerate(alpha_range):
        for j, g in enumerate(gamma_range):
            params = {**model.parameters, "alpha": jnp.array(a), "gamma": jnp.array(g)}
            loss_grid[j, i] = float(loss_fn(params))

    # Adam optimization trajectory — optimize alpha and gamma only
    opt_keys = ["alpha", "gamma"]
    bounds = {k: (TRUE_PARAMS[k] * 0.5, TRUE_PARAMS[k] * 2.0) for k in opt_keys}

    def to_real(norm):
        return {
            **model.parameters,
            **{k: jnp.array(bounds[k][0] + norm[k] * (bounds[k][1] - bounds[k][0])) for k in norm},
        }

    def to_norm(real):
        return {k: jnp.array((real[k] - bounds[k][0]) / (bounds[k][1] - bounds[k][0])) for k in opt_keys}

    def loss_norm(norm_params):
        return loss_fn(to_real(norm_params))

    vg = jax.jit(jax.value_and_grad(loss_norm))
    init_guess = {k: jnp.array(TRUE_PARAMS[k] * 1.3) for k in opt_keys}
    norm_params = to_norm(init_guess)
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(norm_params)

    hist_alpha, hist_gamma, hist_loss = [], [], []
    for _ in range(200):
        loss_val, grads = vg(norm_params)
        updates, opt_state = optimizer.update(grads, opt_state, norm_params)
        norm_params = optax.apply_updates(norm_params, updates)
        norm_params = {k: jnp.clip(v, 0.01, 0.99) for k, v in norm_params.items()}
        real = to_real(norm_params)
        hist_alpha.append(float(real["alpha"]))
        hist_gamma.append(float(real["gamma"]))
        hist_loss.append(float(loss_val))

    return alpha_range, gamma_range, loss_grid, hist_alpha, hist_gamma, hist_loss


def generate_hero(prey_ts, pred_ts, alpha_range, gamma_range, loss_grid, hist_alpha, hist_gamma, hist_loss):
    """Compose the 3-panel hero figure."""
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.35)

    # Panel 1: Simulation
    ax1 = fig.add_subplot(gs[0])
    days = np.arange(len(prey_ts))
    ax1.plot(days, prey_ts, color=PALETTE[0], linewidth=2, label="Prey")
    ax1.plot(days, pred_ts, color=PALETTE[2], linewidth=2, label="Predator")
    ax1.set_xlabel("Day", color="#718096", fontsize=10)
    ax1.set_ylabel("Population density", color="#718096", fontsize=10)
    ax1.set_title("Simulation", color="#1B4965", fontsize=13, fontweight="bold")
    ax1.legend(framealpha=0.9, fontsize=9)
    ax1.tick_params(colors="#718096")
    ax1.grid(True, alpha=0.2)

    # Panel 2: Loss landscape + trajectory
    ax2 = fig.add_subplot(gs[1])
    ad = alpha_range * DAY
    gd = gamma_range * DAY
    ax2.contourf(ad, gd, np.log10(loss_grid + 1), levels=20, cmap="Blues")
    ax2.contour(ad, gd, np.log10(loss_grid + 1), levels=8, colors="#718096", linewidths=0.4, alpha=0.4)
    ta = np.array(hist_alpha) * DAY
    tg = np.array(hist_gamma) * DAY
    ax2.plot(ta, tg, "-", color=PALETTE[2], linewidth=2, alpha=0.9)
    ax2.plot(ta[0], tg[0], "o", color=PALETTE[2], markersize=8, zorder=5)
    ax2.plot(TRUE_PARAMS["alpha"] * DAY, TRUE_PARAMS["gamma"] * DAY, "*", color="white", markersize=14, zorder=5)
    ax2.set_xlabel("\u03b1 (day\u207b\u00b9)", color="#718096", fontsize=10)
    ax2.set_ylabel("\u03b3 (day\u207b\u00b9)", color="#718096", fontsize=10)
    ax2.set_title("Gradient Descent", color="#1B4965", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="#718096")

    # Panel 3: Convergence
    ax3 = fig.add_subplot(gs[2])
    ax3.semilogy(hist_loss, color=PALETTE[0], linewidth=2)
    ax3.set_xlabel("Step", color="#718096", fontsize=10)
    ax3.set_ylabel("MSE Loss", color="#718096", fontsize=10)
    ax3.set_title("Convergence", color="#1B4965", fontsize=13, fontweight="bold")
    ax3.tick_params(colors="#718096")
    ax3.grid(True, alpha=0.2)

    path = OUTPUT_DIR / "hero.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Generated {path}")


if __name__ == "__main__":
    print("Running simulation...")
    prey_ts, pred_ts = run_simulation()
    print("Running gradient optimization...")
    alpha_range, gamma_range, loss_grid, hist_alpha, hist_gamma, hist_loss = run_gradient_optimization()
    print("Generating hero figure...")
    generate_hero(prey_ts, pred_ts, alpha_range, gamma_range, loss_grid, hist_alpha, hist_gamma, hist_loss)
