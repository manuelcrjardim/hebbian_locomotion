"""
Weight-dynamics analysis for a SET of evolved Hebbian policies.

For each checkpoint in CONDITIONS: load it, select one individual, roll it out
for N steps, and record the full plastic weight trajectory, per-step reward and
body state. Produces the same publication-ready figures as the single-network
version (PCA embedding, 3D unrolled PCA, PCA-vs-time, summed L2 weight change,
sample weight traces, reward, state) plus an .npz of the raw arrays, written
into a PER-NETWORK directory:

    OUT_ROOT/PCA_<timestamp>/<label>/{pca,pca_3d,pca_time,weight_change,
                                      sample_weights,reward,state}.{pdf,png}
                                     summary.json

Every .npz is ALSO collected flat in one place:

    OUT_ROOT/PCA_<timestamp>/npz/<label>_data.npz

so gait_spectrum.py can pick them all up with a single glob

    NETWORK_GLOBS = [("M=1", f"{RUN_DIR}/npz/M1_*_data.npz"), ...]

rather than needing one path per network. The label is recoverable from the
filename (strip "_data"), which is exactly what that script's glob branch does.

and one combined index at the top level:

    OUT_ROOT/PCA_<timestamp>/index.json      every network's summary in one file

The .npz CONTENTS are unchanged, so gait_spectrum.py and any downstream
aggregation read these exactly as before.

Isaac Sim is launched ONCE and the environment is reused across networks; only
the network object is rebuilt per checkpoint. Rebuilding the env per network
would dominate the runtime for no benefit.

Edit the CONFIG block below, then:

    python pca.py

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
ROOT = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
        "workspace/hebbian_locomotion")

# Networks to analyse. Same shape as the leg-lock script, so entries copy over
# directly. "label" names the output subdirectory and must be unique.
#
# M is read from each checkpoint's run_cfg sidecar by default. Add an explicit
# "M" key ONLY to override that -- see the note in build_net() for why the
# sidecar is trusted over the filename.
CONDITIONS = [
    {"group": "M=1",   "label": "860896728",
     "ckpt": f"{ROOT}/checkpoints/08:04-11:52_GO1_FINAL_HAN_499_860896728_M_1_benchmark.pickle"},
     {"group": "M=1",   "label": "1396289849",
     "ckpt": f"{ROOT}/checkpoints/08:04-11:52_GO1_FINAL_HAN_499_1396289849_M_1_benchmark.pickle"},
     {"group": "M=1",   "label": "2113178504",
     "ckpt": f"{ROOT}/checkpoints/08:26-19:37_GO1_FINAL_HAN_M_1_2113178504_benchmark.pickle"}, # RAN OUT OF INPUT ERROR
     {"group": "M=1",   "label": "1295751242",
     "ckpt": f"{ROOT}/checkpoints/08:18-14:01_GO1_FINAL_HAN_M_1_1295751242_benchmark.pickle"},
     {"group": "M=1",   "label": "489225843",
     "ckpt": f"{ROOT}/checkpoints/08:18-14:05_GO1_FINAL_HAN_M_1_489225843_benchmark.pickle"},


    {"group": "M=10",  "label": "617764346",
     "ckpt": f"{ROOT}/checkpoints/08:21-16:22_GO1_FINAL_HAN_M_10_617764346_benchmark.pickle"},
    {"group": "M=10",  "label": "689362957",
     "ckpt": f"{ROOT}/checkpoints/08:26-19:50_GO1_FINAL_HAN_M_10_689362957_benchmark.pickle"}, # RAN OUT OF INPUT ERROR
     {"group": "M=10",  "label": "1146876823",
     "ckpt": f"{ROOT}/checkpoints/08:23-16:38_GO1_FINAL_HAN_M_10_1146876823_benchmark.pickle"}, # m = 20
     {"group": "M=10",  "label": "1331919846",
     "ckpt": f"{ROOT}/checkpoints/08:23-16:39_GO1_FINAL_HAN_M_10_1331919846_benchmark.pickle"}, # m =20
     {"group": "M=10",  "label": "27294405", 
     "ckpt": f"{ROOT}/checkpoints/08:15-11:09_GO1_FINAL_HAN_M_10_27294405_benchmark.pickle"}, # ONLY GOOD ONE


    {"group": "M=20",  "label": "1853697842",
     "ckpt": f"{ROOT}/checkpoints/08:04-18:00_GO1_FINAL_HAN_499_1853697842_M_20_benchamrk.pickle"},
     {"group": "M=20",  "label": "685059270",
     "ckpt": f"{ROOT}/checkpoints/08:11-10:43_GO1_FINAL_HAN_M_20_685059270_M_20_benchmark.pickle"},
     {"group": "M=20",  "label": "1919632181",
     "ckpt": f"{ROOT}/checkpoints/08:11-10:41_GO1_FINAL_HAN_M_20_1919632181_benchmark.pickle"},
     {"group": "M=20",  "label": "641049713",
     "ckpt": f"{ROOT}/checkpoints/08:11-10:40_GO1_FINAL_HAN_M_20_641049713_benchmark.pickle"},
     {"group": "M=20",  "label": "1298619277",
     "ckpt": f"{ROOT}/checkpoints/08:26-16:50_GO1_FINAL_HAN_M_20_1298619277_benchmark.pickle"}, # RAN OUT OF INPUT


    {"group": "M=160", "label": "576310242",
     "ckpt": f"{ROOT}/checkpoints/08:04-00:56_GO1_FINAL_HAN_499_576310242_M_160_benchmark.pickle"},
     {"group": "M=160", "label": "756800567",
     "ckpt": f"{ROOT}/checkpoints/08:14-14:35_GO1_FINAL_HAN_M_160_756800567_benchmark.pickle"},
     {"group": "M=160", "label": "1042351053",
     "ckpt": f"{ROOT}/checkpoints/08:14-14:35_GO1_FINAL_HAN_M_160_1042351053_behcnmark.pickle"},
     {"group": "M=160", "label": "2126766829",
     "ckpt": f"{ROOT}/checkpoints/08:14-14:37_GO1_FINAL_HAN_M_160_2126766829_benchmark.pickle"},
    {"group": "M=160", "label": "602647351",
     "ckpt": f"{ROOT}/checkpoints/08:15-11:04_GO1_FINAL_HAN_M_160_602647351_benchmark.pickle"},
]

CONDITIONS2 = [

    {"group": "M=160", "label": "576310242",
     "ckpt": f"{ROOT}/checkpoints/08:04-00:56_GO1_FINAL_HAN_499_576310242_M_160_benchmark.pickle"},
     {"group": "M=160", "label": "756800567",
     "ckpt": f"{ROOT}/checkpoints/08:14-14:35_GO1_FINAL_HAN_M_160_756800567_benchmark.pickle"},
     {"group": "M=160", "label": "1042351053",
     "ckpt": f"{ROOT}/checkpoints/08:14-14:35_GO1_FINAL_HAN_M_160_1042351053_behcnmark.pickle"},
     {"group": "M=160", "label": "2126766829",
     "ckpt": f"{ROOT}/checkpoints/08:14-14:37_GO1_FINAL_HAN_M_160_2126766829_benchmark.pickle"},
    {"group": "M=160", "label": "602647351",
     "ckpt": f"{ROOT}/checkpoints/08:15-11:04_GO1_FINAL_HAN_M_160_602647351_benchmark.pickle"},
]

OUT_ROOT = f"{ROOT}/analysis"   # per-network directories are created under here

STEPS = 1000                    # rollout length, control steps
FREEZE_STEP = -1               # -1 = never freeze; 0 = from start; N = at step N
PARAM_SOURCE = "mu"            # "mu" or "best_mu"
ACTION_FILTER_ALPHA = 0.1      # a_t = alpha*a_raw + (1-alpha)*a_{t-1}
RHO = 0.9                      # convergence-criterion threshold
EARLY_FRAC = 0.05              # fraction of rollout counted as "early"
N_SAMPLE_WEIGHTS = 10          # traces in the sample-weights figure
PLOT_SEED = 0                  # which weights get sampled for that figure
HEADLESS = True                # False to watch the rollout

TAU_HEBB = 1                   # Hebbian update period, matched across conditions
DEFAULT_M = None               # fallback M when a checkpoint has no run_cfg and
                               # no explicit "M" key. None = hard error instead
                               # of silently analysing the wrong dynamics.
SKIP_MISSING = True            # skip absent checkpoints rather than aborting
CONTINUE_ON_ERROR = True       # one bad network must not lose the whole batch

NPZ_SUBDIR = "npz"             # flat collection of every <label>_data.npz,
                               # so gait_spectrum.py can glob them in one line

TAG = f"{current_time}_{PARAM_SOURCE}"
RUN_DIR = f"{OUT_ROOT}/PCA_{TAG}"
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
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
try:
    from cmcrameri import cm as cmc
    _ROMA = cmc.roma
except ImportError:
    _ROMA = plt.get_cmap("Spectral")

# Canvas
BG   = "#F6F5F1"
FG   = "#22252A"
GRID = "#E4E2DC"

# Unordered categorical (e.g. freeze x lock cells, distinct signals)
CAT = ["#1A5E63", "#D9A441", "#B4553F", "#6E4A7E"]


def ramp(n, start=0.22, stop=0.92):
    """n ORDERED categorical colours (e.g. M values), sampled from roma.

    Starts ~22% along so the lightest group stays legible on the off-white
    canvas, and stops short of the far end so a two-colour ramp does not
    reach the extremes.
    """
    if n == 1:
        return [_ROMA(start)]
    return [_ROMA(start + (stop - start) * i / (n - 1)) for i in range(n)]


def tints(base_frac, k, spread=0.10):
    """k dark-biased tints around a ramp position, for seeds within a group."""
    return [_ROMA(min(0.98, max(0.02, base_frac + spread * (i / max(1, k - 1)))))
            for i in range(k)]


def traj_cmap():
    """Continuous colormap for time-coloured trajectories.

    roma trimmed to 0.03-0.97 so it never starts near-white and washes out
    against the off-white canvas.
    """
    return LinearSegmentedColormap.from_list(
        "roma_trim", [_ROMA(x) for x in np.linspace(0.03, 0.97, 256)])


# ICLR 2027: text block is 5.5 in x 9 in, single column, 10 pt Times.
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
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": FG,
        "axes.edgecolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # Do NOT use savefig.bbox="tight". It crops the canvas to the drawn
        # content, so a figure built at FIG_W_FULL is written narrower than
        # FIG_W_FULL; \includegraphics[width=\linewidth] then rescales it and
        # every font size above becomes wrong. constrained_layout packs the
        # axes inward instead, leaving the canvas at exactly the width asked
        # for, so the point sizes here are the true printed point sizes.
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, name, outdir):
    """Write both vector PDF and 300-DPI PNG into this network's directory."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"    [FIG] {name}.pdf / .png")


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
def fig_pca(pcs, outdir):
    """PC1 vs PC2, coloured by time. Cross = start, star = end (Leung et al.)."""
    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.72))

    pts = pcs[:, None, :2]
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=traj_cmap(), linewidth=0.8)
    lc.set_array(np.arange(len(segs)))
    ax.add_collection(lc)

    ax.plot(pcs[0, 0], pcs[0, 1], "x", color=FG,
            markersize=5, mew=1.2, label="start", zorder=5)
    ax.plot(pcs[-1, 0], pcs[-1, 1], "*", color=CAT[2],
            markersize=8, label="end", zorder=5)

    if 0 < FREEZE_STEP < len(pcs):
        ax.plot(pcs[FREEZE_STEP, 0], pcs[FREEZE_STEP, 1], "o",
                color=CAT[0], markersize=4, label="freeze", zorder=5)

    ax.autoscale()
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="best")

    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label("step")
    cbar.outline.set_linewidth(0.6)

    save_fig(fig, "pca", outdir)

# Paste these two functions into pca.py, directly after fig_pca().
# Requires: from mpl_toolkits.mplot3d.art3d import Line3DCollection

def fig_pca_3d(pcs, evr, outdir, elev=22, azim=-58):
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
    lc = Line3DCollection(segs, cmap=traj_cmap(), linewidth=0.7)
    lc.set_array(steps[:-1])
    ax.add_collection3d(lc)

    # Shadow: the same trajectory projected onto the PC1-PC2 floor, so the 3D
    # figure still carries the information of the 2D one.
    floor = steps.min() - 0.04 * (steps.max() - steps.min())
    sh = np.stack([pcs[:, 0], pcs[:, 1], np.full(len(pcs), floor)], axis=1)[:, None, :]
    ssegs = np.concatenate([sh[:-1], sh[1:]], axis=1)
    lcs = Line3DCollection(ssegs, colors=GRID, linewidth=0.4, alpha=0.9)
    ax.add_collection3d(lcs)

    ax.plot([pcs[0, 0]], [pcs[0, 1]], [steps[0]], "x",
            color=FG, markersize=5, mew=1.2, label="start")
    ax.plot([pcs[-1, 0]], [pcs[-1, 1]], [steps[-1]], "*",
            color=CAT[2], markersize=8, label="end")
    if 0 < FREEZE_STEP < len(pcs):
        ax.plot([pcs[FREEZE_STEP, 0]], [pcs[FREEZE_STEP, 1]], [FREEZE_STEP], "o",
                color=CAT[0], markersize=4, label="freeze")

    pad = 0.05
    for lim, dat in ((ax.set_xlim, pcs[:, 0]), (ax.set_ylim, pcs[:, 1])):
        lo, hi = dat.min(), dat.max()
        m = pad * (hi - lo) if hi > lo else 1.0
        lim(lo - m, hi + m)
    ax.set_zlim(floor, steps.max())

    ax.set_xlabel(f"PC 1 ({100 * evr[0]:.0f}%)", labelpad=1)
    ax.set_ylabel(f"PC 2 ({100 * evr[1]:.0f}%)", labelpad=1)
    # The native z label is placed outside the axes and is unreliable under
    # any layout engine; a 2D annotation in axes coordinates is stable.
    ax.set_zlabel("")
    ax.text2D(1.055, 0.52, "step", transform=ax.transAxes, rotation=90,
              va="center", ha="left", fontsize=matplotlib.rcParams["axes.labelsize"])
    ax.view_init(elev=elev, azim=azim)

    # Lighten the default 3D furniture: panes and heavy grid dominate at
    # 5.5 in and read as chartjunk in print.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(BG)
        axis.pane.set_edgecolor(GRID)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
    ax.tick_params(pad=1)
    # No colour bar: the z axis already encodes step, so a bar would label the
    # same variable twice. Colour is kept only to make trace order readable
    # where loops overlap.
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.92), handletextpad=0.4)

    save_fig(fig, "pca_3d", outdir)


def fig_pca_time(pcs, dW, outdir):
    """PC1(t), PC2(t) and plasticity on a shared time axis.

    This is the quantitative companion to the 3D view: convergence time reads
    off directly and lines up with the weight-change trace, which the phase
    portrait cannot show.
    """
    fig, axes = plt.subplots(3, 1, figsize=(FIG_W_FULL, FIG_W_FULL * 0.72),
                             sharex=True)
    axes[0].plot(pcs[:, 0], color=CAT[0], lw=0.8)
    axes[0].set_ylabel("PC 1")
    axes[1].plot(pcs[:, 1], color=CAT[3], lw=0.8)
    axes[1].set_ylabel("PC 2")
    axes[2].plot(dW, color=CAT[2], lw=0.8)
    axes[2].set_ylabel(r"$\sum_k \Vert \Delta W_t^{(k)} \Vert_2$")
    axes[2].set_xlabel("step")
    for a in axes:
        if FREEZE_STEP > 0:
            a.axvline(FREEZE_STEP, color=FG, ls="--", lw=0.8)
        a.set_xlim(0, len(pcs))
    save_fig(fig, "pca_time", outdir)

def fig_weight_change(dW, outdir):
    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.40))
    ax.plot(dW, color=CAT[0])
    if FREEZE_STEP > 0:
        ax.axvline(FREEZE_STEP, color=FG, ls="--", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\sum_k \Vert \Delta W_t^{(k)} \Vert_2$")
    ax.set_xlim(0, len(dW))
    save_fig(fig, "weight_change", outdir)


def fig_sample_weights(W_traj, outdir):
    """Randomly chosen plastic weights over time (Dittrich et al., row 1)."""
    rng = np.random.default_rng(PLOT_SEED)
    n = min(N_SAMPLE_WEIGHTS, W_traj.shape[1])
    idx = rng.choice(W_traj.shape[1], size=n, replace=False)

    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.40))
    for i in idx:
        ax.plot(W_traj[:, i], lw=0.6, alpha=0.85)
    if FREEZE_STEP > 0:
        ax.axvline(FREEZE_STEP, color=FG, ls="--", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$w_{ij}(t)$")
    ax.set_xlim(0, len(W_traj))
    save_fig(fig, "sample_weights", outdir)


def fig_reward(rewards, outdir):
    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.40))
    ax.plot(rewards, color=CAT[3])
    if FREEZE_STEP > 0:
        ax.axvline(FREEZE_STEP, color=FG, ls="--", lw=0.8,
                   label="freeze")
        ax.legend(loc="best")
    ax.set_xlabel("step")
    ax.set_ylabel(r"step reward $r_t$")
    ax.set_xlim(0, len(rewards))
    save_fig(fig, "reward", outdir)


def fig_state(vx, z, outdir):
    """Forward velocity and body height -- the gait discriminator."""
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W_FULL, FIG_W_FULL * 0.62),
                             sharex=True)
    axes[0].plot(vx, color=CAT[1])
    axes[0].set_ylabel(r"$v_x$ (m s$^{-1}$)")
    axes[1].plot(z, color=CAT[3])
    axes[1].set_ylabel(r"$z$ (m)")
    axes[1].set_xlabel("step")
    for a in axes:
        if FREEZE_STEP > 0:
            a.axvline(FREEZE_STEP, color=FG, ls="--", lw=0.8)
        a.set_xlim(0, len(vx))
    save_fig(fig, "state", outdir)


# ===========================================================================
# Per-network driver
# ===========================================================================
def resolve_M(cond, run_cfg, models_ckpt):
    """Decide the boxcar window M for one checkpoint, and say where it came from.

    Precedence: explicit "M" in the condition dict > run_cfg sidecar >
    the live model object > DEFAULT_M.

    The FILENAME is deliberately never consulted. run_es_go1.py builds the run
    name and checkpoint path from a script constant while instantiating HANNet
    with a separately written M, so the two can disagree -- a file named
    "..._M_160_..." may hold an M=20 network. run_cfg["network"]["M"] is read
    off the live model at save time and is the only trustworthy record.
    """
    if "M" in cond and cond["M"] is not None:
        return int(cond["M"]), "condition"

    if run_cfg is not None:
        m = (run_cfg.get("network") or {}).get("M")
        if m is not None:
            return int(m), "run_cfg"

    m = getattr(models_ckpt, "M", None)
    if m is not None:
        return int(m), "model_object"

    if DEFAULT_M is not None:
        return int(DEFAULT_M), "DEFAULT_M"

    raise ValueError(
        "No M for this checkpoint: no \"M\" key in the condition, no run_cfg "
        "sidecar, no M on the pickled model, and DEFAULT_M is None. Analysing "
        "with the wrong M silently produces the wrong dynamics, so this is a "
        "hard error rather than a guess."
    )


def analyse_one(env, cond, outdir, npz_dir):
    """Load, roll out and analyse a single network. Returns its summary dict.

    Figures and summary.json go to outdir (this network's own directory); the
    .npz goes to npz_dir, shared across all networks, so the whole set can be
    globbed by gait_spectrum.py without listing paths.
    """
    label, ckpt, group = cond["label"], cond["ckpt"], cond['group']

    solver, models_ckpt, run_cfg = load_checkpoint(ckpt)
    params = solver.mu if PARAM_SOURCE == "mu" else solver.best_mu
    sizes = list(models_ckpt.architecture)
    M, m_source = resolve_M(cond, run_cfg, models_ckpt)

    print(f"    arch {sizes}   M={M} (from {m_source})   source={PARAM_SOURCE}")
    if m_source == "run_cfg" and "_M_" in os.path.basename(ckpt):
        # Cheap consistency check on the known filename/instantiation mismatch.
        try:
            stem = os.path.basename(ckpt)
            fname_M = int(stem.split("_M_")[1].split("_")[0])
            if fname_M != M:
                print(f"    [WARN] filename says M={fname_M} but run_cfg says "
                      f"M={M}; using run_cfg. Check the training script.")
        except (IndexError, ValueError):
            pass

    net = HANNet(popsize=1, sizes=sizes, norm_mode="max", M=M, tau_hebb=TAU_HEBB)
    net.set_a_model_params(np.asarray(params))

    # --- rollout ----------------------------------------------------------
    W_traj, rewards, vx, z = rollout(env, net)
    print(f"    weight trajectory {W_traj.shape} (steps x plastic weights)")

    # --- analysis ---------------------------------------------------------
    dW = weight_change_l2(W_traj)
    ratio, early, late = convergence_ratio(dW)
    converged = bool(ratio < RHO) if np.isfinite(ratio) else False

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(W_traj)
    evr = pca.explained_variance_ratio_

    summary = {
        "tag": TAG,
        "group": cond.get("group"),
        "label": label,
        "checkpoint": os.path.abspath(ckpt),
        "M": M,
        "M_source": m_source,
        "tau_hebb": TAU_HEBB,
        "architecture": sizes,
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

    # --- outputs ----------------------------------------------------------
    npz_path = os.path.join(npz_dir, f"{group}_{label}_data.npz")
    np.savez_compressed(
        npz_path,
        W_traj=W_traj.astype(np.float32),
        pcs=pcs, dW=dW, rewards=rewards, vx=vx, z=z,
        explained_variance_ratio=evr,
    )
    summary["npz"] = os.path.abspath(npz_path)
    print(f"    [NPZ] {os.path.basename(npz_path)}")

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig_pca(pcs, outdir)
    fig_pca_3d(pcs, evr, outdir)
    fig_pca_time(pcs, dW, outdir)
    fig_weight_change(dW, outdir)
    fig_sample_weights(W_traj, outdir)
    fig_reward(rewards, outdir)
    fig_state(vx, z, outdir)

    return summary


# ===========================================================================
# Main
# ===========================================================================
def main():
    set_pub_style()
    os.makedirs(RUN_DIR, exist_ok=True)
    npz_dir = os.path.join(RUN_DIR, NPZ_SUBDIR)
    os.makedirs(npz_dir, exist_ok=True)

    # --- resolve the work list before touching the simulator --------------
    todo, seen = [], set()
    for cond in CONDITIONS:
        label = cond["label"]
        if label in seen:
            print(f"[WARN] duplicate label {label!r}; skipping the repeat.")
            continue
        if not os.path.exists(cond["ckpt"]):
            msg = f"[WARN] missing checkpoint for {label}: {cond['ckpt']}"
            if SKIP_MISSING:
                print(msg + "  -> skipped")
                continue
            raise FileNotFoundError(msg)
        seen.add(label)
        todo.append(cond)

    if not todo:
        raise SystemExit("No usable checkpoints in CONDITIONS.")
    print(f"[INFO] {len(todo)} network(s) to analyse -> {RUN_DIR}\n")

    # --- environment: built ONCE and reused --------------------------------
    # Isaac Sim startup dominates the runtime; the env does not depend on which
    # checkpoint is loaded, so rebuilding it per network buys nothing.
    env_cfg = Go1EnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    results, failures = [], []
    try:
        for k, cond in enumerate(todo, 1):
            label = cond["label"]
            group = cond['group']
            outdir = os.path.join(RUN_DIR, f'{group}_{label}')
            os.makedirs(outdir, exist_ok=True)
            print(f"[{k}/{len(todo)}] {label}  [{cond.get('group')}]")
            try:
                results.append(analyse_one(env, cond, outdir, npz_dir))
            except Exception as exc:                      # noqa: BLE001
                failures.append({"label": label, "error": repr(exc)})
                print(f"    [FAIL] {exc!r}")
                if not CONTINUE_ON_ERROR:
                    raise
    finally:
        env.close()

    # --- combined index ----------------------------------------------------
    index = {
        "tag": TAG,
        "run_dir": os.path.abspath(RUN_DIR),
        "npz_dir": os.path.abspath(npz_dir),
        "param_source": PARAM_SOURCE,
        "steps": STEPS,
        "freeze_step": FREEZE_STEP,
        "tau_hebb": TAU_HEBB,
        "n_requested": len(CONDITIONS),
        "n_analysed": len(results),
        "failures": failures,
        "networks": results,
    }
    with open(os.path.join(RUN_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # --- console summary table --------------------------------------------
    print("\n--- summary " + "-" * 62)
    print(f"  {'label':22s} {'group':8s} {'M':>4s} {'ratio':>7s} "
          f"{'conv':>5s} {'PC1+PC2':>8s} {'mean_vx':>8s}")
    for r in sorted(results, key=lambda x: (str(x["group"]), x["label"])):
        evr2 = r["explained_variance_pc1"] + r["explained_variance_pc2"]
        print(f"  {r['label']:22s} {str(r['group']):8s} {r['M']:>4d} "
              f"{r['convergence_ratio']:>7.3f} "
              f"{str(r['converged_fixed_point']):>5s} {evr2:>7.1%} "
              f"{r['mean_vx']:>8.3f}")
    if failures:
        print(f"\n  {len(failures)} failure(s):")
        for f_ in failures:
            print(f"    {f_['label']:22s} {f_['error']}")
    print("-" * 74 + "\n")

    print(f"[DONE] {len(results)}/{len(todo)} analysed. "
          f"Outputs under {os.path.abspath(RUN_DIR)}")

    # Emit a manifest and a ready-made NETWORKS block for gait_spectrum.py, so
    # the next step needs no hand-written paths.
    #
    # Deliberately NOT auto-generated globs: a glob prefix derived from the
    # labels is fragile. os.path.commonprefix over seeds that happen to share a
    # digit yields something like "M160_s5*", which silently drops any future
    # seed not starting with 5, and a prefix like "M1_*" would also swallow the
    # M10 and M160 groups. The group of each network is known exactly here, so
    # record it rather than trying to re-derive it from filenames later.
    if results:
        manifest = {
            "tag": TAG,
            "npz_dir": os.path.abspath(npz_dir),
            "networks": [
                {"group": r["group"], "label": r["label"],
                 "npz": os.path.basename(r["npz"])}
                for r in sorted(results, key=lambda x: (str(x["group"]), x["label"]))
            ],
        }
        with open(os.path.join(npz_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  [MANIFEST] {os.path.join(os.path.abspath(npz_dir), 'manifest.json')}")

        print("\n  Paste into gait_spectrum.py:\n")
        print(f'    BASE = "{os.path.abspath(npz_dir)}"')
        print("    NETWORKS = [")
        for r in sorted(results, key=lambda x: (str(x["group"]), x["label"])):
            print(f'        ("{r["group"]}", "{r["label"]}", '
                  f'f"{{BASE}}/{os.path.basename(r["npz"])}"),')
        print("    ]\n")


if __name__ == "__main__":
    main()
    simulation_app.close()