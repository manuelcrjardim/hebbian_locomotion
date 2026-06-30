"""
eval_leglock.py — Run a trained policy with an optional mid-episode leg-lock
and log everything (rewards, velocity, distance, posture, plastic weight change)
to TensorBoard, plus publication-ready trajectory figures.

No training, no ES updates — pure evaluation.

What you set:
    * CHECKPOINT_PATH   : which trained policy to load
    * LOCK_STEP         : the control step the leg-lock engages (None = no lock)
    * LEG_PATTERNS      : which leg(s) to lock (from inspect_robot.py names)
    * NET_CLASS/KWARGS  : match the architecture you trained (HebbianNet/HANNet/D2)

Run inside the Apptainer container:
    python eval_leglock.py
Then:
    tensorboard --logdir <LOG_ROOT>

Figures are written as vector PDF (for LaTeX) + high-DPI PNG (for preview) to
logs/positions, and mirrored into TensorBoard under Plots/.
"""

import os
import pickle
from datetime import datetime

# ── Launch Isaac Sim (must precede isaaclab.* imports) ──
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from hebbian_locomotion.envs.GeckoEnv import (
    GeckoEnvCfg,
    forward_velocity_x,
    upright_posture,
)

# ── Pick the architecture that matches your checkpoint ─────────────────
# Default: the plastic HebbianNet baseline. For the boxcar / EMA variants,
# swap NET_CLASS and fill NET_KWARGS with the SAME hyperparameters you
# trained with (the script prints the checkpoint's attributes below to help
# you read off M / tau_hebb / gamma).
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet
NET_CLASS = HebbianNet
NET_KWARGS = dict(norm_mode="max")
# Example for the boxcar variant:
# from hebbian_locomotion.networks.han_net import HANNet
# NET_CLASS = HANNet
# NET_KWARGS = dict(norm_mode="max", window_size=10, tau_hebb=...)

# ── Leg-lock module (drop perturbation.py on the path / same dir) ──────
from perturbation import LegLock

# ======================================================================
# CONFIG
# ======================================================================
ROOT = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion"

CHECKPOINT_PATH = f"{ROOT}/checkpoints/Gecko_hebbian_es_checkpoint_06:10-19:04_499_BENCHMARK.pickle"
WHICH_PARAMS    = "best_mu"          # "mu" (ES mean) or "best_mu" (best seen)
NUM_ENVS        = 1             # >1 averages over reset-noise seeds; weight diag uses env 0
STEPS           = 300           # control steps; <=1000 stays inside one 20 s episode

# --- the perturbation you control ---
LOCK_STEP       = 499           # control step the lock engages; None disables it
LEG_PATTERNS    = [".*lf.*"]    # fill from inspect_robot.py; e.g. [".*LF.*", ".*RF.*"]
MASK_LOCKED_ACTIONS = False     # also zero the policy's commands on locked joints

# --- match training dynamics ---
USE_ACTION_FILTER = True        # EMA filter: 0.1*new + 0.9*prev (as in run_es_dynamic)

# --- figure options ---
FIG_WIDTH_IN = 3.5              # IEEE single-column width; use ~7.0 for full width
SHOW_TITLES  = False            # publication figures rely on the caption, not a title

LOG_ROOT = f"{ROOT}/logs/eval"

# Okabe-Ito colorblind-safe palette.
C_X      = "#0072B2"  # blue
C_Y      = "#E69F00"  # orange
C_START  = "#009E73"  # green
C_END    = "#D55E00"  # vermillion
C_LOCK   = "#CC79A7"  # reddish purple


# ======================================================================
# PUBLICATION STYLE
# ======================================================================
def setup_pub_style():
    """Centralized matplotlib settings for publication-quality output.

    Serif typography to sit cleanly next to LaTeX body text, inward ticks,
    thin spines, embedded TrueType fonts (no Type-3, which many publishers
    reject), and vector-friendly defaults.
    """
    plt.rcParams.update({
        # Typography — serif to match an IEEE/LaTeX layout; DejaVu Serif is
        # always available as a fallback if Times is not installed.
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Nimbus Roman",
                             "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size":        9,
        "axes.titlesize":   9,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  8,
        # Lines and spines
        "axes.linewidth":   0.8,
        "lines.linewidth":  1.5,
        "grid.linewidth":   0.5,
        "grid.alpha":       0.3,
        # Ticks
        "xtick.direction":  "in",
        "ytick.direction":  "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top":        True,
        "ytick.right":      True,
        # Legend
        "legend.frameon":   False,
        "legend.handlelength": 1.6,
        # Saving — vector first, embed fonts as TrueType (Type 42).
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "figure.dpi":       150,
    })


def save_pub_fig(fig, path_base, writer, tb_tag):
    """Save a figure as vector PDF + high-DPI PNG and mirror into TensorBoard."""
    fig.savefig(f"{path_base}_{WHICH_PARAMS}.pdf")
    fig.savefig(f"{path_base}_{WHICH_PARAMS}.png", dpi=300)
    writer.add_figure(tb_tag, fig, 0)
    print(f"[INFO] Saved {path_base}.pdf / .png")


# ======================================================================
# HELPERS
# ======================================================================
def get_params(solver, which: str):
    """Pull the parameter vector to evaluate from the ES solver."""
    if which == "best_mu" and hasattr(solver, "best_mu") and solver.best_mu is not None:
        return solver.best_mu
    if which == "best_mu":
        print("[WARN] solver.best_mu unavailable; falling back to solver.mu")
    return solver.mu


def per_layer_weights_env0(model):
    """Return a list of flat plastic-weight tensors for env 0, or None.

    Used to track Sum_k ||W_t^k - W_{t-1}^k||_2 (the paper's plasticity metric).
    Defensive: returns None if the model doesn't expose weights as expected.
    """
    getter = getattr(model, "get_weight_snapshot", None) or getattr(model, "get_weights", None)
    if getter is None:
        return None
    try:
        w = getter()
        if torch.is_tensor(w):
            w = [w]
        return [t[0].reshape(-1).detach().clone() for t in w]
    except Exception as e:
        print(f"[WARN] weight-change tracking disabled: {e}")
        return None


def plot_xy_trajectory(xs, ys, times, lock_time, lock_tag):
    """Bird's-eye path, coloured by time, with start/end and lock markers."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    times = np.asarray(times)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN + 0.6, FIG_WIDTH_IN * 0.85))

    # Time-coloured path via a LineCollection (matches the papers' viridis-by-time).
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis",
                        norm=plt.Normalize(times.min(), times.max()))
    lc.set_array(times[:-1])
    lc.set_linewidth(1.8)
    ax.add_collection(lc)

    cbar = fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Time (s)")
    cbar.outline.set_linewidth(0.8)

    ax.plot(xs[0], ys[0], "o", color=C_START, markersize=6,
            markeredgecolor="black", markeredgewidth=0.5, label="Start", zorder=5)
    ax.plot(xs[-1], ys[-1], "s", color=C_END, markersize=6,
            markeredgecolor="black", markeredgewidth=0.5, label="End", zorder=5)
    if lock_time is not None and lock_time <= times.max():
        i = int(np.searchsorted(times, lock_time))
        i = min(i, len(xs) - 1)
        ax.plot(xs[i], ys[i], "X", color=C_LOCK, markersize=9,
                markeredgecolor="black", markeredgewidth=0.5,
                label="Leg-lock", zorder=6)

    # Equal aspect with a small data margin.
    mx = 0.05 * (xs.max() - xs.min() + 1e-6)
    my = 0.05 * (ys.max() - ys.min() + 1e-6)
    ax.set_xlim(xs.min() - mx, xs.max() + mx)
    ax.set_ylim(ys.min() - my, ys.max() + my)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    if SHOW_TITLES:
        ax.set_title(lock_tag)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_xy_over_time(xs, ys, times, lock_time, lock_tag):
    """X and Y world position against time, with the lock onset marked."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.7))
    ax.plot(times, xs, color=C_X, label="$x$")
    ax.plot(times, ys, color=C_Y, label="$y$")
    if lock_time is not None and lock_time <= times[-1]:
        ax.axvline(lock_time, color=C_LOCK, linestyle="--", linewidth=1.2,
                   label="Leg-lock")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.set_xlim(times[0], times[-1])
    if SHOW_TITLES:
        ax.set_title(lock_tag)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ======================================================================
# MAIN
# ======================================================================
def main():
    setup_pub_style()

    # ── Load checkpoint ──
    print(f"[INFO] Loading checkpoint: {CHECKPOINT_PATH}")
    with open(CHECKPOINT_PATH, "rb") as f:
        solver, models_ckpt, _, _ = pickle.load(f)

    hp = {k: v for k, v in vars(models_ckpt).items()
          if isinstance(v, (int, float, str, bool))}
    print(f"[INFO] checkpoint architecture: {models_ckpt.architecture}")
    print(f"[INFO] checkpoint scalar attrs (match these in NET_KWARGS): {hp}")

    params = get_params(solver, WHICH_PARAMS)

    # ── Build env ──
    env_cfg = GeckoEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env = ManagerBasedRLEnv(cfg=env_cfg)
    step_dt = env_cfg.decimation * env_cfg.sim.dt   # seconds per control step

    # ── Build network, load trained coefficients, broadcast to all envs ──
    model = NET_CLASS(popsize=NUM_ENVS, sizes=models_ckpt.architecture, **NET_KWARGS)
    model.set_a_model_params(params)
    model.reset_weights()

    # ── Leg-lock ──
    obs, _ = env.reset()
    robot = env.scene["robot"]
    leg_lock = LegLock(lock_step=LOCK_STEP, patterns=LEG_PATTERNS,
                       hold_position="default")
    leg_lock.resolve(robot)

    # ── TensorBoard ──
    stamp = datetime.now().strftime("%m-%d_%H-%M")
    ckpt_tag = os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0]
    lock_tag = f"lock{LOCK_STEP}_{'_'.join(LEG_PATTERNS)}" if LOCK_STEP is not None else "nolock"
    log_dir = f"{LOG_ROOT}/{stamp}_{ckpt_tag}_{WHICH_PARAMS}_{lock_tag}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[INFO] Logging to {log_dir}")

    robot_cfg = SceneEntityCfg("robot")
    initial_xy = robot.data.root_pos_w[:, :2].clone()

    cumulative_reward = torch.zeros(NUM_ENVS, device=env.device)
    prev_actions = None
    prev_weights = per_layer_weights_env0(model)
    track_weights = prev_weights is not None

    # Position trace (mean across envs) for the trajectory figures.
    xs, ys = [], []

    # ── Rollout ──
    for step in range(STEPS):
        actions = model.forward(obs["policy"])

        if MASK_LOCKED_ACTIONS:
            actions = leg_lock.mask_actions(actions, step)

        if USE_ACTION_FILTER and prev_actions is not None:
            actions = 0.1 * actions + 0.9 * prev_actions

        obs, rewards, terminates, truncates, extras = env.step(actions)

        # Re-assert the kinematic pin AFTER the physics step.
        leg_lock.maybe_apply(robot, step)

        prev_actions = actions
        cumulative_reward += rewards

        mean_xy = robot.data.root_pos_w[:, :2].mean(dim=0).cpu().numpy()
        xs.append(float(mean_xy[0]))
        ys.append(float(mean_xy[1]))

        # ── Scalar logging (means across envs) ──
        writer.add_scalar("Reward/Total_Step", rewards.mean().item(), step)
        writer.add_scalar("Reward/Cumulative", cumulative_reward.mean().item(), step)

        v_term = forward_velocity_x(env, robot_cfg) * 2.0
        u_term = upright_posture(env, robot_cfg) * 0.5
        writer.add_scalar("RewardTerms/V_forward_vel", v_term.mean().item(), step)
        writer.add_scalar("RewardTerms/U_upright", u_term.mean().item(), step)

        fwd_vel = robot.data.root_lin_vel_b[:, 0]
        body_z = robot.data.root_pos_w[:, 2]
        upright_proj = -robot.data.projected_gravity_b[:, 2]
        dist = torch.norm(robot.data.root_pos_w[:, :2] - initial_xy, dim=1)
        writer.add_scalar("Velocity/Forward_X", fwd_vel.mean().item(), step)
        writer.add_scalar("Position/Distance_From_Start", dist.mean().item(), step)
        writer.add_scalar("Position/Body_Height_Z", body_z.mean().item(), step)
        writer.add_scalar("Posture/Upright_Proj", upright_proj.mean().item(), step)

        writer.add_scalar("Perturbation/Lock_Active",
                          1.0 if leg_lock.is_active(step) else 0.0, step)

        if track_weights:
            cur = per_layer_weights_env0(model)
            dW = sum((c - p).norm().item() for c, p in zip(cur, prev_weights))
            writer.add_scalar("Plasticity/Sum_L2_Weight_Change", dW, step)
            prev_weights = cur

        if step % 50 == 0:
            tag = " [LOCKED]" if leg_lock.is_active(step) else ""
            print(f"  step {step:4d} | rew {rewards.mean().item():6.3f} | "
                  f"vx {fwd_vel.mean().item():6.3f} | dist {dist.mean().item():5.2f}{tag}")

    final_dist = torch.norm(robot.data.root_pos_w[:, :2] - initial_xy, dim=1).mean().item()
    avg_speed = final_dist / (STEPS * step_dt)
    print(f"\n[INFO] Done. Final distance {final_dist:.3f} m | avg speed {avg_speed:.3f} m/s")
    writer.add_scalar("Summary/Final_Distance", final_dist, 0)
    writer.add_scalar("Summary/Avg_Speed", avg_speed, 0)

    # ── Publication figures ──
    plot_dir = f"{ROOT}/logs/positions"
    os.makedirs(plot_dir, exist_ok=True)
    times = np.arange(len(xs)) * step_dt
    lock_time = LOCK_STEP * step_dt if LOCK_STEP is not None else None
    base = f"{plot_dir}/{stamp}_{ckpt_tag}_{lock_tag}"

    fig1 = plot_xy_trajectory(xs, ys, times, lock_time, lock_tag)
    save_pub_fig(fig1, f"{base}_xy", writer, "Plots/XY_Trajectory")
    plt.close(fig1)

    fig2 = plot_xy_over_time(xs, ys, times, lock_time, lock_tag)
    save_pub_fig(fig2, f"{base}_xy_over_time", writer, "Plots/XY_Over_Time")
    plt.close(fig2)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()