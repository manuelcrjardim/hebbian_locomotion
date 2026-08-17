"""
Weight-dynamics analysis for a single evolved Hebbian policy.

Loads an ES checkpoint, selects one individual, rolls it out for N steps, and
records the full plastic weight trajectory, per-step reward, and body state.
Produces publication-ready figures (PCA embedding, summed L2 weight change,
sample weight traces, reward, state) plus an .npz of the raw arrays so the
10-seed aggregation can be done later without re-running the simulation.

Edit the CONFIG block below, then:

    python analyse_weight_dynamics.py

Note on FREEZE_STEP = 0
-----------------------
With norm_mode='max', weight normalisation happens *inside* hebbian_update.
Freezing from t=0 without intervention would leave the weights at their
uniform(-init_noise, init_noise) initialisation (magnitude ~1e-2), so every
tanh saturates near zero and the robot stands still. That is not a freeze
control, it is a broken network. This script therefore applies one
normalisation pass before the rollout when FREEZE_STEP = 0.

FREEZE_STEP = 250 (mid-episode) follows the protocol in Dittrich et al., who
interrupt the Hebbian update partway through the lifetime and measure the
resulting drop in step reward. That is the more meaningful control: it asks
whether the *converged* weights sustain the gait.
"""

from datetime import datetime
current_time = datetime.now().strftime("%m:%d-%H:%M")

# ===========================================================================
# CONFIG -- edit everything here
# ===========================================================================
CKPT = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:11-16:45_GO1_FINAL_HAN_M_160_1566955798_benchmark.pickle"  # ES checkpoint pickle

OUTDIR = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/analysis"            # figures + .npz land here


STEPS = 1000                    # rollout length, control steps
FREEZE_STEP = -1               # -1 = never freeze; 0 = from start; N = at step N
PARAM_SOURCE = "mu"            # "mu" or "best_mu"
ACTION_FILTER_ALPHA = 0.1      # a_t = alpha*a_raw + (1-alpha)*a_{t-1}
RHO = 0.9                      # convergence-criterion threshold
EARLY_FRAC = 0.05              # fraction of rollout counted as "early"
N_SAMPLE_WEIGHTS = 10          # traces in the sample-weights figure
PLOT_SEED = 0                  # which weights get sampled for that figure
HEADLESS = True                # False to watch the rollout
MODEL = 'HAN_M_20'

TAG = f"{current_time}_{MODEL}_{PARAM_SOURCE}"   

OUTDIR = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/analysis/PCA_{TAG}"
# ===========================================================================

import json
import os
import pickle

# ---------------------------------------------------------------------------
# Launch Isaac Sim (must precede all isaaclab imports)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

# TODO: point this at your Go1 environment config.
from hebbian_locomotion.envs.Go1_env import Go1EnvCfg
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet  # noqa: E402
from hebbian_locomotion.networks.han_net import HANNet  # noqa: E402

# ===========================================================================
# Figure style
# ===========================================================================
OKABE_ITO = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "verm":   "#D55E00",
    "purple": "#CC79A7",
}


# NeurIPS 2026: text block is 5.5 in x 9 in, single column, 10 pt Times.
# Generate each figure at the width it will be inserted at, and never scale it
# in LaTeX -- a width mismatch rescales the fonts and breaks the match with
# body text.
FIG_W_FULL  = 5.5    # \includegraphics[width=\linewidth]{...}
FIG_W_HALF  = 2.65   # two side by side: width=0.48\linewidth
FIG_W_THIRD = 1.72   # three side by side: width=0.32\linewidth


def set_pub_style():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name):
    """Write both vector PDF and 300-DPI PNG."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[FIG] {name}.pdf / .png")


# ===========================================================================
# Helpers
# ===========================================================================
def load_checkpoint(path):
    """Load a 4- or 5-element ES checkpoint."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    solver, models_ckpt = data[0], data[1]
    run_cfg = data[4] if len(data) >= 5 else None
    return solver, models_ckpt, run_cfg


def flat_weights(net):
    """Concatenate individual 0's plastic weights across layers -> (D,) numpy."""
    return torch.cat([w[0].flatten() for w in net.weights]).cpu().numpy()


def freeze_plasticity(net):
    """Disable Hebbian updates in place.

    forward() calls self.hebbian_update(i, W, pre, post, A, B, C, D, lr).
    Shadowing it on the instance intercepts that call without touching the
    class definition; the bound-method 'self' is not passed, hence the
    signature below.
    """
    net.hebbian_update = lambda layer_idx, weights, *a, **k: weights


def normalise_weights_once(net):
    """Apply one normalisation pass (needed when freezing from t=0)."""
    for i in range(net.n_layers):
        net.weights[i] = net.WeightStand(net.weights[i])


# ===========================================================================
# Rollout
# ===========================================================================
def rollout(env, net):
    """Run one episode, recording weights, reward and body state each step."""
    obs, _ = env.reset()
    net.reset_weights()

    if FREEZE_STEP == 0:
        normalise_weights_once(net)
        freeze_plasticity(net)
        print("[INFO] Plasticity frozen at t=0 (weights normalised once first).")

    robot = env.scene["robot"]

    W_traj, rewards, vx, z = [], [], [], []
    prev_actions = None

    for step in range(STEPS):
        if FREEZE_STEP > 0 and step == FREEZE_STEP:
            freeze_plasticity(net)
            print(f"[INFO] Plasticity frozen at step {step}.")

        # Record weights *before* the update this step produces
        W_traj.append(flat_weights(net))

        raw = net.forward(obs["policy"])

        # EMA action filter -- the policy was evolved with this in the loop
        if prev_actions is None:
            actions = raw
        else:
            actions = (ACTION_FILTER_ALPHA * raw
                       + (1.0 - ACTION_FILTER_ALPHA) * prev_actions)
        prev_actions = actions

        obs, rew, _, _, _ = env.step(actions)

        rewards.append(rew[0].item())
        vx.append(robot.data.root_lin_vel_b[0, 0].item())
        z.append(robot.data.root_pos_w[0, 2].item())

    return (np.asarray(W_traj), np.asarray(rewards),
            np.asarray(vx), np.asarray(z))


# ===========================================================================
# Analysis
# ===========================================================================
def weight_change_l2(W_traj):
    """Summed L2 norm of the per-step weight change (Dittrich et al., row 2)."""
    return np.linalg.norm(np.diff(W_traj, axis=0), axis=1)


def convergence_ratio(dW):
    """Return (ratio, early_mean, late_mean) for the fixed-point criterion.

    Converged (fixed-point) if late < RHO * early.
    """
    n_early = max(1, int(EARLY_FRAC * len(dW)))
    early = dW[:n_early].mean()
    late = dW[n_early:].mean()
    return (late / early if early > 0 else np.nan), early, late


# ===========================================================================
# Figures
# ===========================================================================
def fig_pca(pcs):
    """PC1 vs PC2, coloured by time. Cross = start, star = end (Leung et al.)."""
    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.72))

    pts = pcs[:, None, :2]
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", linewidth=0.8)
    lc.set_array(np.arange(len(segs)))
    ax.add_collection(lc)

    ax.plot(pcs[0, 0], pcs[0, 1], "x", color=OKABE_ITO["black"],
            markersize=5, mew=1.2, label="start", zorder=5)
    ax.plot(pcs[-1, 0], pcs[-1, 1], "*", color=OKABE_ITO["verm"],
            markersize=8, label="end", zorder=5)

    if 0 < FREEZE_STEP < len(pcs):
        ax.plot(pcs[FREEZE_STEP, 0], pcs[FREEZE_STEP, 1], "o",
                color=OKABE_ITO["blue"], markersize=4, label="freeze", zorder=5)

    ax.autoscale()
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best")

    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label("step")
    cbar.outline.set_linewidth(0.6)

    save_fig(fig, f"pca")

# Paste these two functions into pca.py, directly after fig_pca().
# Requires: from mpl_toolkits.mplot3d.art3d import Line3DCollection

def fig_pca_3d(pcs, evr, elev=22, azim=-58):
    """PC1-PC2 trajectory unrolled along a time axis.

    The 2D phase portrait cannot distinguish a converging spiral from a
    persistent orbit once the late loops overlap the early ones. Lifting the
    trajectory onto a time axis separates them by construction: convergence
    reads as a cone narrowing to a line, a limit cycle as a uniform tube.
    """
    fig = plt.figure(figsize=(FIG_W_FULL, FIG_W_FULL * 0.78))
    ax = fig.add_subplot(111, projection="3d")

    steps = np.arange(len(pcs))
    pts = np.stack([pcs[:, 0], pcs[:, 1], steps], axis=1)[:, None, :]
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap="viridis", linewidth=0.7)
    lc.set_array(steps[:-1])
    ax.add_collection3d(lc)

    # Shadow: the same trajectory projected onto the PC1-PC2 floor, so the 3D
    # figure still carries the information of the 2D one.
    floor = steps.min() - 0.04 * (steps.max() - steps.min())
    sh = np.stack([pcs[:, 0], pcs[:, 1], np.full(len(pcs), floor)], axis=1)[:, None, :]
    ssegs = np.concatenate([sh[:-1], sh[1:]], axis=1)
    lcs = Line3DCollection(ssegs, colors="0.75", linewidth=0.4, alpha=0.6)
    ax.add_collection3d(lcs)

    ax.plot([pcs[0, 0]], [pcs[0, 1]], [steps[0]], "x",
            color=OKABE_ITO["black"], markersize=5, mew=1.2, label="start")
    ax.plot([pcs[-1, 0]], [pcs[-1, 1]], [steps[-1]], "*",
            color=OKABE_ITO["verm"], markersize=8, label="end")
    if 0 < FREEZE_STEP < len(pcs):
        ax.plot([pcs[FREEZE_STEP, 0]], [pcs[FREEZE_STEP, 1]], [FREEZE_STEP], "o",
                color=OKABE_ITO["blue"], markersize=4, label="freeze")

    pad = 0.05
    for lim, dat in ((ax.set_xlim, pcs[:, 0]), (ax.set_ylim, pcs[:, 1])):
        lo, hi = dat.min(), dat.max()
        m = pad * (hi - lo) if hi > lo else 1.0
        lim(lo - m, hi + m)
    ax.set_zlim(floor, steps.max())

    ax.set_xlabel(f"PC 1 ({100 * evr[0]:.0f}%)", labelpad=1)
    ax.set_ylabel(f"PC 2 ({100 * evr[1]:.0f}%)", labelpad=1)
    # The native z label is placed outside the axes and gets cropped by
    # bbox_inches='tight'; a 2D annotation in axes coordinates is stable.
    ax.set_zlabel("")
    ax.text2D(1.055, 0.52, "step", transform=ax.transAxes, rotation=90,
              va="center", ha="left", fontsize=matplotlib.rcParams["axes.labelsize"])
    ax.view_init(elev=elev, azim=azim)

    # Lighten the default 3D furniture: panes and heavy grid dominate at
    # 5.5 in and read as chartjunk in print.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("0.85")
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update(color="0.90", linewidth=0.5)
    ax.tick_params(pad=1)
    # No colour bar: the z axis already encodes step, so a bar would label the
    # same variable twice. Colour is kept only to make trace order readable
    # where loops overlap.
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.92), handletextpad=0.4)

    save_fig(fig, f"pca_3d")


def fig_pca_time(pcs, dW):
    """PC1(t), PC2(t) and plasticity on a shared time axis.

    This is the quantitative companion to the 3D view: convergence time reads
    off directly and lines up with the weight-change trace, which the phase
    portrait cannot show.
    """
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W_FULL, FIG_W_FULL * 0.72),
                             sharex=True)
    axes[0].plot(pcs[:, 0], color=OKABE_ITO["blue"], lw=0.8)
    axes[0].set_ylabel("PC 1")
    axes[1].plot(pcs[:, 1], color=OKABE_ITO["green"], lw=0.8)
    axes[1].set_ylabel("PC 2")
    axes[2].plot(dW, color=OKABE_ITO["verm"], lw=0.8)
    axes[2].set_ylabel(r"$\sum_k \Vert \Delta W_t^{(k)} \Vert_2$")
    axes[2].set_xlabel("step")
    for a in axes:
        if FREEZE_STEP > 0:
            a.axvline(FREEZE_STEP, color=OKABE_ITO["black"], ls="--", lw=0.8)
        a.set_xlim(0, len(pcs))
    fig.subplots_adjust(hspace=0.10)
    save_fig(fig, f"pca_time")

def fig_weight_change(dW):
    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.40))
    ax.plot(dW, color=OKABE_ITO["blue"])
    if FREEZE_STEP > 0:
        ax.axvline(FREEZE_STEP, color=OKABE_ITO["verm"], ls="--", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\sum_k \Vert \Delta W_t^{(k)} \Vert_2$")
    ax.set_xlim(0, len(dW))
    save_fig(fig, f"weight_change")


def fig_sample_weights(W_traj):
    """Randomly chosen plastic weights over time (Dittrich et al., row 1)."""
    rng = np.random.default_rng(PLOT_SEED)
    n = min(N_SAMPLE_WEIGHTS, W_traj.shape[1])
    idx = rng.choice(W_traj.shape[1], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.40))
    for i in idx:
        ax.plot(W_traj[:, i], lw=0.6, alpha=0.85)
    if FREEZE_STEP > 0:
        ax.axvline(FREEZE_STEP, color=OKABE_ITO["verm"], ls="--", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$w_{ij}(t)$")
    ax.set_xlim(0, len(W_traj))
    save_fig(fig, f"sample_weights")


def fig_reward(rewards):
    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.40))
    ax.plot(rewards, color=OKABE_ITO["green"])
    if FREEZE_STEP > 0:
        ax.axvline(FREEZE_STEP, color=OKABE_ITO["verm"], ls="--", lw=0.8,
                   label="freeze")
        ax.legend(loc="best")
    ax.set_xlabel("step")
    ax.set_ylabel(r"step reward $r_t$")
    ax.set_xlim(0, len(rewards))
    save_fig(fig, f"reward")


def fig_state(vx, z):
    """Forward velocity and body height -- the gait discriminator."""
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_FULL, FIG_W_FULL * 0.62),
                             sharex=True)
    axes[0].plot(vx, color=OKABE_ITO["orange"])
    axes[0].set_ylabel(r"$v_x$ (m s$^{-1}$)")
    axes[1].plot(z, color=OKABE_ITO["purple"])
    axes[1].set_ylabel(r"$z$ (m)")
    axes[1].set_xlabel("step")
    for a in axes:
        if FREEZE_STEP > 0:
            a.axvline(FREEZE_STEP, color=OKABE_ITO["verm"], ls="--", lw=0.8)
        a.set_xlim(0, len(vx))
    fig.subplots_adjust(hspace=0.12)
    save_fig(fig, f"state")


# ===========================================================================
# Main
# ===========================================================================
def main():
    set_pub_style()
    os.makedirs(OUTDIR, exist_ok=True)

    # --- checkpoint -------------------------------------------------------
    solver, models_ckpt, run_cfg = load_checkpoint(CKPT)
    params = solver.mu if PARAM_SOURCE == "mu" else solver.best_mu
    sizes = list(models_ckpt.architecture)
    print(f"[INFO] Architecture from checkpoint: {sizes}")
    print(f"[INFO] Parameter source: {PARAM_SOURCE}")

    # --- environment ------------------------------------------------------
    env_cfg = Go1EnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # net = HebbianNet(popsize=1, sizes=sizes, norm_mode="max")
    # net.set_a_model_params(np.asarray(params))

    M = 10
    TAU_HEBB = 1

    net = HANNet(popsize=1, sizes=sizes, norm_mode="max", M=M, tau_hebb=TAU_HEBB)
    net.set_a_model_params(np.asarray(params))
    if run_cfg is not None:
        print(f"[INFO] run_cfg: {run_cfg}")   # cross-check M against the sidecar

    # --- rollout ----------------------------------------------------------
    W_traj, rewards, vx, z = rollout(env, net)
    env.close()
    print(f"[INFO] Weight trajectory: {W_traj.shape} (steps x plastic weights)")

    # --- analysis ---------------------------------------------------------
    dW = weight_change_l2(W_traj)
    ratio, early, late = convergence_ratio(dW)
    converged = bool(ratio < RHO) if np.isfinite(ratio) else False

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(W_traj)
    evr = pca.explained_variance_ratio_

    summary = {
        "tag": TAG,
        "checkpoint": os.path.abspath(CKPT),
        "param_source": PARAM_SOURCE,
        "steps": STEPS,
        "freeze_step": FREEZE_STEP,
        "rho": RHO,
        "convergence_ratio": float(ratio),
        "dW_early_mean": float(early),
        "dW_late_mean": float(late),
        "converged_fixed_point": converged,
        "explained_variance_pc1": float(evr[0]),
        "explained_variance_pc2": float(evr[1]),
        "mean_reward": float(rewards.mean()),
        "total_reward": float(rewards.sum()),
        "mean_vx": float(vx.mean()),
        "z_range": float(z.max() - z.min()),
    }

    print("\n--- summary " + "-" * 45)
    for k, v in summary.items():
        print(f"  {k:26s} {v}")
    print("-" * 57 + "\n")

    # --- outputs ----------------------------------------------------------
    np.savez_compressed(
        os.path.join(OUTDIR, f"data.npz"),
        W_traj=W_traj.astype(np.float32),
        pcs=pcs, dW=dW, rewards=rewards, vx=vx, z=z,
        explained_variance_ratio=evr,
    )
    with open(os.path.join(OUTDIR, f"summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig_pca(pcs)
    fig_pca_3d(pcs, evr)
    fig_pca_time(pcs, dW)
    fig_weight_change(dW)
    fig_sample_weights(W_traj)
    fig_reward(rewards)
    fig_state(vx, z)

    print(f"[DONE] Outputs written to {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
    simulation_app.close()