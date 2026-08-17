"""
eval_leglock_go1.py -- The full 2x2 freeze x perturbation design, all conditions,
one launch.

Loads each trained policy in CONDITIONS, runs the four cells of the 2x2 for each,
and reports two scale-free measures per cell. Isaac Lab is started once and the
environment is reused across every run, so a full sweep costs one app launch.

MEASURES
--------
Both are proportions of the rollout, not fixed step counts, so they are
comparable across policies whose gait periods differ.

  1. velocity retention
         mean v_x over the last WIN_FRAC of the episode
       / mean v_x over the WIN_FRAC immediately preceding the event step
     The behavioural outcome: did the robot keep walking. Defined identically
     in the undamaged cells (using the same nominal event step), so those give
     the reference level rather than a missing value.

  2. plasticity convergence ratio  (Dittrich et al., re-anchored at the event)
         mean ||dW|| over the LATE portion after the event
       / mean ||dW|| over the EARLY portion after the event
     A rollout is classified CONVERGED when this ratio < RHO, exactly the
     criterion of the HAN paper, applied with the perturbation as the origin
     rather than the start of the rollout. Reported per cell as the fraction
     of repeats meeting the criterion.

Everything else -- transient ratios, re-convergence times, settling windows --
has been dropped. Those presuppose a response shape (quiet, spike, decay,
settle) that has not been observed, and under max-normalisation ||dW|| does not
decay to zero, so a tolerance-band re-convergence measure has little dynamic
range. The qualitative story lives in the trace figures instead.

Repeats are collapsed to a MEDIAN, giving one number per policy per cell. The
unit of analysis for all downstream statistics is the seed, never the repeat.

GROUPS
------
Each entry in CONDITIONS carries a `group` (the M-set it belongs to) and a
unique `label` (the individual network). Several networks may share a group;
this is how the results are shown to generalise beyond a single trained policy.

Outputs, per group G:
  {RUN_TAG}_traces_{G}    one row per network in G, all cells overlaid
  {RUN_TAG}_summary_{G}   grouped bars, one x position per network in G
and once across all groups:
  {RUN_TAG}_groups_joint  one x position per M-set; bar = mean across the
                          networks of that set, whiskers = full min-max range,
                          individual networks overlaid as open markers

The two error-bar semantics are deliberately different and must be labelled as
such in captions. Within a group figure the interval is the IQR across the
REPEATS of one network (within-policy rollout noise). In the joint figure it is
the range across NETWORKS (between-policy spread), computed from one number per
network so that unequal repeat counts cannot weight one policy over another.

GAIT-PHASE STRATIFICATION
-------------------------
Where in the gait cycle the damage lands is a nuisance variable, and it is not
the same nuisance for every network: measured gait periods across these
policies span roughly a factor of two (35-70 control steps). A single global
EVENT_STEP would therefore ablate each network at an arbitrary and DIFFERENT
phase of its own cycle, and the resulting per-network numbers would be phase-
conditional in a way that varies systematically with the group -- sitting
directly on top of the retention effect being measured.

This script marginalises over phase instead of fixing it. Each network carries
its own measured gait period T_gait (in control steps, from gait_spectrum.py),
and each cell is run N_ABLATIONS times at evenly spaced phases of THAT
network's cycle:

    t_k = EVENT_STEP_BASE + round(k * T_gait / N_ABLATIONS),   k = 0..N-1

so every network's ablations span exactly one of its own cycles at 0, 360/N,
2*360/N ... degrees. The spacing is T/N and not T/(N-1): with T/(N-1) the last
stratum lands back on phase 0, duplicating the first and overweighting that
phase in the mean.

The per-network scalar is then the median over ALL repeats of ALL strata, i.e.
retention marginalised over gait phase, which is a stronger and more honest
claim than a phase-conditional one. Per-stratum values are also retained so
phase dependence can be inspected -- treat that as exploratory: N_ABLATIONS
points at 360/N resolution is a weak test of a smooth curve.

One rollout is run per stratum rather than assigning different event steps to
different envs within a rollout. Both freeze and leg-lock act on the whole
batch -- freeze_network() shadows a method on the model and LegLock patches the
actuator -- so per-env event steps would require changing both. The cost is
N_ABLATIONS x more rollouts; the benefit is that the damage model stays exactly
as validated.

Traces are ALIGNED to each stratum's own event step before pooling, so the
x axis of the trace figure is steps since ablation and the pooled median is
meaningful across strata.
"""

from datetime import datetime


current_time = datetime.now().strftime("%m:%d-%H:%M")

# ===========================================================================
# CONFIG -- edit everything here
# ===========================================================================
ROOT = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
        "workspace/hebbian_locomotion")

# --- the policies to sweep ---
#   group : the M-set a network belongs to. Defines the per-group figures and
#           the aggregation in the joint figure.
#   label : must be UNIQUE across the whole sweep; it keys the results dict,
#           the npz payload and the figure filenames.
#   T_gait: this network's OWN gait period in CONTROL STEPS, measured by
#           gait_spectrum.py (field "T_gait_steps_used"). The ablation phases
#           are derived from it, so it must be the period of the same policy
#           under the same undamaged conditions. Copy them from
#           <TAG>_ablation_offsets.json rather than retyping.
CONDITIONS = [
    # T_gait in CONTROL STEPS from gait_spectrum.py. None -> falls back to
    # DEFAULT_T_GAIT, and the fallback is recorded in the JSON.
    {"group": "M=1",   "label": "M1_s860896728",
     "ckpt": f"{ROOT}/checkpoints/08:04-11:52_GO1_FINAL_HAN_499_860896728_M_1_benchmark.pickle",
     "T_gait": 37.5},
     {"group": "M=1",   "label": "M1_s1396289849",
     "ckpt": f"{ROOT}/checkpoints/08:04-11:52_GO1_FINAL_HAN_499_1396289849_M_1_benchmark.pickle",
     "T_gait": 56.25},
     {"group": "M=1",   "label": "M1_s1534702257",
     "ckpt": f"{ROOT}/checkpoints/08:04-11:51_GO1_FINAL_HAN_499_1534702257_M_1_benchmark.pickle",
     "T_gait": 47.36},

    {"group": "M=10",  "label": "M10_s46838495",
     "ckpt": f"{ROOT}/checkpoints/08:06-13:33_GO1_FINAL_HAN_499_46838495_M_10_benchmark.pickle",
     "T_gait": 40.909},
    {"group": "M=10",  "label": "M10_s1315706754",
     "ckpt": f"{ROOT}/checkpoints/08:07-14:17_GO1_FINAL_HAN_499_1315706754_M_10_benchmark.pickle",
     "T_gait": 47.368},
     {"group": "M=10",  "label": "M_10_1302491059",
     "ckpt": f"{ROOT}/checkpoints/08:08-15:23_GO1_FINAL_HAN_M_10_1302491059_M_10_benchmark.pickle",
     "T_gait": 44.999},


    {"group": "M=20",  "label": "M20_s1853697842",
     "ckpt": f"{ROOT}/checkpoints/08:04-18:00_GO1_FINAL_HAN_499_1853697842_M_20_benchamrk.pickle",
     "T_gait": 34.615},
     {"group": "M=20",  "label": "M_20_685059270",
     "ckpt": f"{ROOT}/checkpoints/08:11-10:43_GO1_FINAL_HAN_M_20_685059270_M_20_benchmark.pickle",
     "T_gait": 52.941},
     {"group": "M=20",  "label": "M_20_1919632181",
     "ckpt": f"{ROOT}/checkpoints/08:11-10:41_GO1_FINAL_HAN_M_20_1919632181_benchmark.pickle",
     "T_gait": 69.230},
     {"group": "M=20",  "label": "M_20_641049713",
     "ckpt": f"{ROOT}/checkpoints/08:11-10:40_GO1_FINAL_HAN_M_20_641049713_benchmark.pickle",
     "T_gait": 47.368},

    {"group": "M=160", "label": "M160_s576310242",
     "ckpt": f"{ROOT}/checkpoints/08:04-00:56_GO1_FINAL_HAN_499_576310242_M_160_benchmark.pickle",
     "T_gait": 36.0},
]

RUN_TAG = "leglock_sweep_RL_hip_only"                 # prefix for every output file
OUTDIR = f"{ROOT}/analysis/leglock_{current_time}_{RUN_TAG}"

PARAM_SOURCE = "mu"          # "mu" or "best_mu"; mu is the converged solution
NUM_REPEATS = 5             # parallel copies of the SAME network PER STRATUM;
                             # total repeats per cell = NUM_REPEATS * N_ABLATIONS
STEPS = 1000                 # control steps; 1000 = 20 s at 50 Hz, as in HAN

# --- gait-phase stratification ---
EVENT_STEP_BASE = 500        # phase-0 stratum. Damage engages here; also the
                             # window origin in the UNDAMAGED cells, so all four
                             # cells are measured over identical spans.
N_ABLATIONS = 5              # ablations per gait cycle. Offsets are T_gait/N
                             # apart -- NOT T_gait/(N-1), which would put the
                             # last stratum back on phase 0.
DEFAULT_T_GAIT = None        # steps, used when a condition has "T_gait": None.
                             # None -> hard error. Guessing a period silently
                             # produces offsets that cover the wrong fraction of
                             # a cycle, with no visible symptom in any figure.

# --- the 2x2; set False to skip a cell ---
CELLS = {
    "baseline":    {"freeze": False, "lock": False},
    "freeze_only": {"freeze": True,  "lock": False},
    "lock_only":   {"freeze": False, "lock": True},
    "both":        {"freeze": True,  "lock": True},
}

# --- damage model; fix once, identical across EVERY condition and cell ---
LOCK_MODE = "stuck"          # "stuck" (Leung: leg stops following commands) or "limp"
LEG_PATTERNS = ["RL_hip_joint"]     # confirm names with inspect_robot.py
HOLD_POSITION = "default"    # "current" = no step change in target at damage onset
STUCK_STIFFNESS = 8.0       # ignored if perturbation_go1 overrides the target
STUCK_DAMPING = 1.0
LIMP_DAMPING = 1.0
MASK_LOCKED_ACTIONS = False  # the actuator override already decouples the joint

# --- must match training ---
ACTION_FILTER_ALPHA = 0.1    # a_t = 0.1*a_raw + 0.9*a_{t-1}
NORM_MODE = "max"            # NEVER change: 'var' gives ~14.5x different weights

# --- network overrides; None -> read from each checkpoint ---
NET_M = None
NET_TAU_HEBB = None

# --- measurement (pre-registered; do not tune after seeing perturbation data) ---
WIN_FRAC = 0.25              # retention windows, as a fraction of STEPS
EARLY_FRAC = 0.05            # "early" portion after the event, as a fraction of
                             # the post-event span (HAN uses 5% of the rollout)
RHO = 0.90                   # HAN convergence threshold: late < RHO * early

ROLLOUT_SEED = 0
HEADLESS = True
# ===========================================================================

import json
import os
import pickle

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from hebbian_locomotion.envs.Go1_env import Go1EnvCfg  # noqa: E402
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet  # noqa: E402
from hebbian_locomotion.networks.han_net import HANNet  # noqa: E402
from perturbation_go1 import LegLock  # noqa: E402


# ===========================================================================
# Figure style -- ICLR 2027 (5.5 in text block, 10 pt Times), roma palette
# ===========================================================================
# A palette is THREE things, and conflating them misleads:
#   RAMP -- ordered categorical (M = 1, 10, 20, 160 is a progression)
#   CAT  -- unordered categorical (the 2x2 cells have no natural order, so
#           colouring them from the ramp would imply one that does not exist)
#   CMAP -- continuous (gait phase, 0 to 360 deg)
# The ramp and cmap are sampled from Crameri's `roma`, which is perceptually
# uniform and colour-vision-deficiency safe by construction. Hardcoded rather
# than imported so the script has no cmcrameri dependency on the cluster.
ROMA = ["#7e1700", "#903e0d", "#9e5c19", "#ac7726", "#ba9437", "#c8b455",
        "#d2d484", "#cce7b2", "#b1e9ce", "#89dad7", "#5dc1d3", "#3ea4c9",
        "#2d88be", "#236eb3", "#1951a6", "#033198"]

# Off-white ground: softer in print than pure white, and it lets pale fills
# read as tinted rather than as grey.
BG, FG, GRID = "#F6F5F1", "#22252A", "#E4E2DC"

# Unordered categorical, for the four cells of the 2x2.
CELL_COLOUR = {
    "baseline": "#1A5E63", "freeze_only": "#D9A441",
    "lock_only": "#B4553F", "both": "#6E4A7E",
}
CELL_LABEL = {
    "baseline": "baseline", "freeze_only": "freeze",
    "lock_only": "lock", "both": "freeze + lock",
}
FIG_W_FULL = 5.5


def _lerp_hex(a, b, t):
    a = tuple(int(a[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(b[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(a[k] + (b[k] - a[k]) * t))
                                   for k in range(3))


def roma_at(x):
    """Sample the roma ramp at x in [0, 1]."""
    x = min(max(float(x), 0.0), 1.0) * (len(ROMA) - 1)
    i = int(x)
    return ROMA[i] if i >= len(ROMA) - 1 else _lerp_hex(ROMA[i], ROMA[i + 1], x - i)


def ramp_colors(n, lo=0.05, hi=0.95):
    """n colours along roma, for an ORDERED variable.

    Sampling stops short of both ends: the extremes of roma are very dark and
    very light respectively, and the light end is unreadable as a thin line on
    an off-white ground.
    """
    if n == 1:
        return [roma_at(0.5)]
    return [roma_at(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def roma_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "roma", [roma_at(x) for x in np.linspace(0.03, 0.97, 16)])

# Joint-figure x axis. GROUP_TICK maps a group name to a short tick label;
# groups not listed fall back to their own name.
GROUP_AXIS_LABEL = r"averaging window $M$"
GROUP_TICK = {"M=1": "1", "M=10": "10", "M=20": "20",
              "M=110": "110", "M=160": "160"}


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
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "lines.solid_capstyle": "round",
        # Off-white canvas and a subordinate horizontal grid. Dropping the top
        # and right spines is most of what makes a 5.5 in figure feel airy --
        # the full box competes with the data at this size.
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG,
        "xtick.color": FG, "ytick.color": FG,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # Do NOT use savefig.bbox="tight". It crops the canvas to the drawn
        # content, so a figure built at FIG_W_FULL is written narrower than
        # FIG_W_FULL; \includegraphics[width=\linewidth] then rescales it and
        # every font size below becomes wrong. constrained_layout packs the
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


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[FIG] {name}.pdf / .png")


# ===========================================================================
# Grouping
# ===========================================================================
def group_of(cond):
    """The M-set a condition belongs to; falls back to its own label."""
    return cond.get("group", cond["label"])


def group_order():
    """Groups in order of first appearance in CONDITIONS."""
    seen = []
    for c in CONDITIONS:
        g = group_of(c)
        if g not in seen:
            seen.append(g)
    return seen


def conds_in_group(g):
    return [c for c in CONDITIONS if group_of(c) == g]


def slug(s):
    """Filename-safe version of a group name."""
    return "".join(ch if (ch.isalnum() or ch == "-") else "_"
                   for ch in str(s)).strip("_")


def t_gait_of(cond):
    """This network's gait period in control steps."""
    t = cond.get("T_gait")
    if t is None:
        t = DEFAULT_T_GAIT
    if t is None:
        raise ValueError(
            f"{cond['label']}: no T_gait and DEFAULT_T_GAIT is None. Take it "
            f"from gait_spectrum.py's <TAG>_ablation_offsets.json "
            f"(\"T_gait_steps\"). Guessing the period would silently make the "
            f"ablations cover the wrong fraction of a gait cycle."
        )
    if not np.isfinite(t) or t <= 0:
        raise ValueError(f"{cond['label']}: T_gait must be positive, got {t!r}")
    return float(t)


def ablation_steps(cond):
    """Event steps for one network: N_ABLATIONS evenly spaced phases of ITS
    OWN cycle.

    Spacing is T/N, not T/(N-1). With T/(N-1) the final stratum lands back on
    phase 0, duplicating the first and overweighting that phase in the mean.
    """
    T = t_gait_of(cond)
    k = np.arange(N_ABLATIONS)
    offs = np.rint(k * T / N_ABLATIONS).astype(int)
    steps = EVENT_STEP_BASE + offs
    phases = (360.0 * offs / T)
    return steps.tolist(), phases.round(1).tolist(), T


def net_scalar(results, label, cell, which):
    """One number per NETWORK per cell -- repeats AND strata already collapsed.

    This is the seed-level unit of analysis. The joint figure and the group
    table aggregate these, never the individual repeats, so a policy evaluated
    with more repeats cannot outvote one evaluated with fewer.

    Pooling across strata is what makes this a phase-MARGINAL quantity: the
    median is taken over every repeat at every gait phase, so the number means
    "retention averaged over where in the cycle the damage lands" rather than
    "retention at one arbitrary phase".
    """
    key = f"{label}__{cell}"
    if key not in results:
        return float("nan")
    r = results[key]["per_repeat"]
    if which == "retention":
        return float(np.nanmedian(r["retention"]))
    if which == "converged":
        return float(np.nanmean(r["converged"]))
    if which == "conv_ratio":
        return float(np.nanmedian(r["conv_ratio"]))
    raise KeyError(which)


# ===========================================================================
# Helpers
# ===========================================================================
def load_checkpoint(path):
    """Load a 4- or 5-element ES checkpoint without assuming its length."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    run_cfg = data[4] if len(data) >= 5 else None
    return data[0], data[1], run_cfg


def build_network(models_ckpt, popsize):
    """Instantiate the right class with hyperparameters read from the checkpoint.

    Avoids the manual NET_CLASS / NET_KWARGS pattern, which silently evaluates
    an M=160 checkpoint as M=1 if the constants are not kept in sync.
    """
    sizes = list(models_ckpt.architecture)
    M = NET_M if NET_M is not None else getattr(models_ckpt, "M", None)
    tau = NET_TAU_HEBB if NET_TAU_HEBB is not None else getattr(models_ckpt, "tau_hebb", None)

    if M is None:
        print('\n ############################################################## \n M is none \n ##############################################################  ')
        

    M, tau = int(M), int(tau if tau is not None else 1)
    return HANNet(popsize, sizes=sizes, norm_mode=NORM_MODE, M=M, tau_hebb=tau), M, tau


def freeze_network(model):
    """Stop Hebbian updates. Uses HANNet.freeze() when available."""
    if hasattr(model, "freeze"):
        model.freeze()
    else:
        # HebbianNet has no freeze(); shadow hebbian_update on the instance.
        model.hebbian_update = lambda layer_idx, weights, *a, **k: weights


def per_env_weights(model):
    """Flat plastic weights for every env: (num_envs, D) on GPU."""
    return torch.cat([w.reshape(w.shape[0], -1) for w in model.weights], dim=1)


def safe_ratio(num, den, eps=1e-12):
    return float(num / den) if abs(den) > eps else float("nan")


# ===========================================================================
# The two measures
# ===========================================================================
def compute_measures(vx, dW, event_step):
    """Per-repeat measures. vx and dW are both (STEPS, num_envs).

    All windows are fractions of the rollout, so they span the same number of
    gait cycles regardless of a policy's gait period. The undamaged cells use
    the same nominal event_step, which makes every cell directly comparable.
    """
    n_steps, n_env = vx.shape
    win = int(round(WIN_FRAC * n_steps))

    pre_lo, pre_hi = event_step - win, event_step
    post_lo, post_hi = n_steps - win, n_steps

    n_post = n_steps - event_step
    early_hi = event_step + max(1, int(round(EARLY_FRAC * n_post)))

    if pre_lo < 0 or early_hi >= n_steps or post_lo <= pre_hi:
        raise ValueError(
            f"[measures] windows do not fit: STEPS={n_steps}, event_step="
            f"{event_step}, WIN_FRAC={WIN_FRAC}. Need event_step >= {win} and "
            f"event_step <= {n_steps - win}. Note the LAST stratum sits at "
            f"EVENT_STEP_BASE + ~T_gait, so budget for that too."
        )

    out = {}
    out["v_pre"] = vx[pre_lo:pre_hi].mean(axis=0)
    out["v_post"] = vx[post_lo:post_hi].mean(axis=0)
    out["dW_early"] = dW[event_step:early_hi].mean(axis=0)
    out["dW_late"] = dW[early_hi:].mean(axis=0)

    out["retention"] = np.array(
        [safe_ratio(out["v_post"][e], out["v_pre"][e]) for e in range(n_env)])
    out["conv_ratio"] = np.array(
        [safe_ratio(out["dW_late"][e], out["dW_early"][e]) for e in range(n_env)])
    out["converged"] = (out["conv_ratio"] < RHO).astype(float)

    return out


# ===========================================================================
# One rollout
# ===========================================================================
def run_cell(env, robot, ckpt_path, do_freeze, do_lock, step_dt, verbose_lock,
             event_step):
    """Run ONE stratum of one cell: all envs share this stratum's event_step.

    One rollout per stratum, rather than per-env event steps within a single
    rollout. Both interventions act on the whole batch -- freeze_network()
    shadows a method on the model object and LegLock patches the shared
    actuator -- so staggering them per env would mean changing both. Running
    them separately keeps the damage model exactly as validated.
    """
    solver, models_ckpt, _ = load_checkpoint(ckpt_path)
    params = np.asarray(solver.mu if PARAM_SOURCE == "mu" else solver.best_mu)

    # Rebuilt every cell: freeze_network() shadows a method on HebbianNet
    # instances, and a stale activation buffer would leak across cells.
    model, M, tau = build_network(models_ckpt, NUM_REPEATS)
    model.set_a_model_params(params)

    obs, _ = env.reset()
    model.reset_weights()

    leg_lock = LegLock(
        lock_step=(event_step if do_lock else None),
        patterns=LEG_PATTERNS, mode=LOCK_MODE,
        stuck_stiffness=STUCK_STIFFNESS, stuck_damping=STUCK_DAMPING,
        limp_damping=LIMP_DAMPING, hold_position=HOLD_POSITION,
        verbose=verbose_lock,
    )
    leg_lock.resolve(robot)

    vx_l, z_l, rew_l, dW_l, up_l = [], [], [], [], []
    prev_actions = None
    prev_W = per_env_weights(model).clone()
    frozen = False

    for step in range(STEPS):
        if do_freeze and step == event_step and not frozen:
            freeze_network(model)
            frozen = True

        actions = model.forward(obs["policy"])
        if MASK_LOCKED_ACTIONS and do_lock:
            actions = leg_lock.mask_actions(actions, step)
        if prev_actions is not None:
            actions = (ACTION_FILTER_ALPHA * actions
                       + (1.0 - ACTION_FILTER_ALPHA) * prev_actions)
        prev_actions = actions

        # BEFORE env.step: the damage must be in place for this step's physics.
        leg_lock.maybe_apply(robot, step)

        obs, rewards, _, _, _ = env.step(actions)

        cur_W = per_env_weights(model)
        dW_l.append((cur_W - prev_W).norm(dim=1).cpu().numpy())
        prev_W = cur_W.clone()

        vx_l.append(robot.data.root_lin_vel_b[:, 0].cpu().numpy())
        z_l.append(robot.data.root_pos_w[:, 2].cpu().numpy())
        up_l.append((-robot.data.projected_gravity_b[:, 2]).cpu().numpy())
        rew_l.append(rewards.cpu().numpy())

    # Undo the actuator patch before the next cell reuses this env, otherwise
    # damage leaks and silently contaminates every subsequent rollout.
    leg_lock.restore(robot)

    traces = {k: np.asarray(v) for k, v in
              (("vx", vx_l), ("z", z_l), ("rew", rew_l),
               ("up", up_l), ("dW", dW_l))}
    return traces, compute_measures(traces["vx"], traces["dW"], event_step), M, tau


# ===========================================================================
# Pooling across strata
# ===========================================================================
def align_window(event_steps):
    """Common (pre, post) window, in steps, around each stratum's own event.

    Strata have different event steps, so raw traces cannot be averaged
    elementwise -- step 520 is 20 steps post-damage for one stratum and 20
    steps PRE-damage for another. Cropping each to [event-pre, event+post)
    puts them on a shared "steps since ablation" axis where pooling is
    meaningful, and where the trace figure reads directly as a response.
    """
    pre = int(min(event_steps))
    post = int(STEPS - max(event_steps))
    if post <= 0:
        raise ValueError(
            f"No post-event span left: last stratum at {max(event_steps)} of "
            f"STEPS={STEPS}. Raise STEPS or lower EVENT_STEP_BASE.")
    return pre, post


def align_trace(arr, event_step, pre, post):
    """Crop one stratum's trace to [event-pre, event+post) along axis 0."""
    return arr[event_step - pre: event_step + post]


def pool_strata(strata, pre, post):
    """Merge per-stratum results into pooled traces and pooled measures.

    Traces are aligned to the event first, then concatenated along the ENV
    axis, so a pooled median is a median over (repeats x strata) at a fixed
    lag from the damage. Measures need no alignment -- each was already
    computed against its own stratum's event step -- so they are simply
    concatenated.
    """
    traces = {}
    for name in strata[0]["traces"]:
        traces[name] = np.concatenate(
            [align_trace(s["traces"][name], s["event_step"], pre, post)
             for s in strata], axis=1)

    per_repeat = {}
    for name in strata[0]["measures"]:
        per_repeat[name] = np.concatenate(
            [np.atleast_1d(s["measures"][name]) for s in strata])

    return traces, per_repeat


# ===========================================================================
# Figures
# ===========================================================================
def fig_traces_group(results, g, labels):
    """One row per NETWORK in group g; v_x and ||dW|| with all cells overlaid."""
    nrow = len(labels)

    # Height scales with row count: a fixed aspect would squash a 4-row figure
    # into the same height as a 2-row one. ~0.30 of the width per row, plus a
    # little headroom for the shared x-label.
    aspect = 0.30 * nrow + 0.10

    fig, axes = plt.subplots(nrow, 2,
                             figsize=(FIG_W_FULL, FIG_W_FULL * aspect),
                             sharex=True, sharey="col", squeeze=False)

    for r, lab in enumerate(labels):
        pre = post = None
        for cell in CELLS:
            key = f"{lab}__{cell}"
            if key not in results:
                continue
            tr = results[key]["traces"]
            pre, post = results[key]["pre"], results[key]["post"]
            col, name = CELL_COLOUR[cell], CELL_LABEL[cell]
            # x is steps SINCE ABLATION: traces from different strata were
            # aligned to their own event before pooling, so a common absolute
            # step axis would be meaningless here.
            t = np.arange(-pre, post)
            for c, arr in enumerate((tr["vx"], tr["dW"])):
                med = np.median(arr, axis=1)
                lo, hi = np.nanpercentile(arr, [25, 75], axis=1)
                axes[r, c].fill_between(t, lo, hi, color=col, alpha=0.13, lw=0)
                axes[r, c].plot(t, med, color=col, lw=1.0,
                                label=name if (r == 0 and c == 0) else None)

        for c in range(2):
            axes[r, c].axvline(0, color=FG, ls="--", lw=0.8, alpha=0.55, zorder=0)
            if pre is not None:
                axes[r, c].set_xlim(-pre, post)
        axes[r, 0].set_ylabel(rf"{lab}" "\n" r"$v_x$ (m s$^{-1}$)")

    # The dW label is long; with sharey="col" one instance on the middle row is
    # enough, and repeating it per row makes it overlap the left column.
    axes[nrow // 2, 1].set_ylabel(r"$\Vert \Delta W_t \Vert_2$")
    axes[-1, 0].set_xlabel("steps since ablation")
    axes[-1, 1].set_xlabel("steps since ablation")

    # Legend above the figure: with several cells overlaid there is no free
    # space inside the axes that does not cover a trace. "outside upper center"
    # reserves the space in the layout instead of drawing over the top row.
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center",
               ncol=len(handles), handlelength=1.6, columnspacing=1.4)

    save_fig(fig, f"{RUN_TAG}_traces_{slug(g)}")


def fig_summary_group(results, g):
    """Grouped bars, one x position per NETWORK in group g.

    Error bars are the interquartile range ACROSS REPEATS of a single network:
    within-policy rollout variability. The between-policy spread lives in the
    joint figure instead. The two must not be conflated in the caption.
    """
    conds = conds_in_group(g)
    labels = [c["label"] for c in conds]
    cells = list(CELLS.keys())
    x = np.arange(len(labels))
    w = 0.8 / len(cells)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_FULL, 2.3))
    for k, cell in enumerate(cells):
        off = (k - (len(cells) - 1) / 2) * w
        ret_med, ret_err, conv = [], [[], []], []
        for lab in labels:
            key = f"{lab}__{cell}"
            if key not in results:
                ret_med.append(np.nan); conv.append(np.nan)
                ret_err[0].append(0.0); ret_err[1].append(0.0)
                continue
            r = results[key]["per_repeat"]["retention"]
            m = np.nanmedian(r)
            lo, hi = np.nanpercentile(r, [25, 75])
            ret_med.append(m)
            ret_err[0].append(max(0.0, m - lo)); ret_err[1].append(max(0.0, hi - m))
            conv.append(np.nanmean(results[key]["per_repeat"]["converged"]))
        axes[0].bar(x + off, ret_med, w, yerr=np.array(ret_err), capsize=1.5,
                    color=CELL_COLOUR[cell], label=CELL_LABEL[cell],
                    edgecolor="none", error_kw={"lw": 0.7, "ecolor": FG})
        axes[1].bar(x + off, conv, w, color=CELL_COLOUR[cell], edgecolor="none")

    axes[0].axhline(1.0, color=FG, ls=":", lw=0.8, alpha=0.5)
    axes[0].set_ylabel("velocity retention")
    axes[1].set_ylabel("fraction converged")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.grid(axis="x", visible=False)     # vertical lines fight the bars

    # Legend above the figure, not inside axes[0] -- with 4 cells there is no
    # free space inside the axes that doesn't sit on a bar or a tick label.
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center",
               ncol=len(handles), handlelength=1.2, columnspacing=0.9)
    save_fig(fig, f"{RUN_TAG}_summary_{slug(g)}")


# ===========================================================================
# Figures -- joint across groups
# ===========================================================================
def _panel_spec():
    return (("retention", "velocity retention", None),
            ("converged", "fraction converged", (0, 1.05)))


def fig_groups_joint(results):
    """One x position per M-set. Bar = MEAN across networks of that set of the
    per-network median; whiskers = full RANGE (min to max) across networks.

    Individual networks are overlaid as open markers. With n of order 2-3 the
    range is the honest interval -- a standard error would imply a sampling
    distribution the sample size cannot support -- and showing every point
    makes the aggregate auditable.
    """
    groups = group_order()
    cells = list(CELLS.keys())
    x = np.arange(len(groups))
    w = 0.8 / len(cells)
    panels = _panel_spec()

    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_W_FULL, 2.4))
    for k, cell in enumerate(cells):
        off = (k - (len(cells) - 1) / 2) * w
        for p, (which, _ylab, _ylim) in enumerate(panels):
            means, lo_err, hi_err = [], [], []
            for gi, g in enumerate(groups):
                vals = np.array([net_scalar(results, c["label"], cell, which)
                                 for c in conds_in_group(g)], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    means.append(np.nan); lo_err.append(0.0); hi_err.append(0.0)
                    continue
                m = float(vals.mean())
                means.append(m)
                lo_err.append(max(0.0, m - vals.min()))
                hi_err.append(max(0.0, vals.max() - m))
                jit = (np.linspace(-0.18, 0.18, vals.size) * w
                       if vals.size > 1 else np.zeros(1))
                axes[p].plot(x[gi] + off + jit, vals, ls="none", marker="o",
                             ms=2.2, mfc=BG, mec=FG, mew=0.5, zorder=3)
            axes[p].bar(x + off, means, w,
                        yerr=np.array([lo_err, hi_err]), capsize=1.5,
                        color=CELL_COLOUR[cell], edgecolor="none",
                        label=CELL_LABEL[cell] if p == 0 else None,
                        error_kw={"lw": 0.7, "zorder": 2.5, "ecolor": FG},
                        zorder=2)

    axes[0].axhline(1.0, color=FG, ls=":", lw=0.8, alpha=0.5, zorder=1)
    for p, (_which, ylab, ylim) in enumerate(panels):
        axes[p].set_ylabel(ylab)
        if ylim is not None:
            axes[p].set_ylim(*ylim)
        axes[p].set_xticks(x)
        axes[p].set_xticklabels(
            [f"{GROUP_TICK.get(g, g)}\n($n$={len(conds_in_group(g))})"
             for g in groups])
        axes[p].set_xlabel(GROUP_AXIS_LABEL)
        axes[p].set_xlim(-0.5, len(groups) - 0.5)
        axes[p].grid(axis="x", visible=False)

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center",
               ncol=len(handles), handlelength=1.2, columnspacing=0.9)
    save_fig(fig, f"{RUN_TAG}_groups_joint")


def fig_phase(results):
    """Retention against the gait phase at which the damage landed.

    EXPLORATORY. N_ABLATIONS points at 360/N resolution is a weak test of a
    smooth curve, so read a flat line as "no evidence of phase dependence"
    rather than as "phase-invariant". Its real job is a check on the design:
    if retention swings wildly with phase, the marginal means in the other
    figures are averaging over a large effect and the spread should be
    reported alongside them.
    """
    groups = group_order()
    gcol = dict(zip(groups, ramp_colors(len(groups))))
    cells = [c for c in CELLS if CELLS[c]["lock"] or CELLS[c]["freeze"]]
    if not cells:
        return

    fig, axes = plt.subplots(1, len(cells), figsize=(FIG_W_FULL, 2.0),
                             sharey=True, squeeze=False)
    for p, cell in enumerate(cells):
        ax = axes[0, p]
        for g in groups:
            for cond in conds_in_group(g):
                key = f"{cond['label']}__{cell}"
                if key not in results:
                    continue
                st = results[key]["strata"]
                ph = [s["phase_deg"] for s in st]
                rv = [float(np.nanmedian(s["measures"]["retention"])) for s in st]
                o = np.argsort(ph)
                ax.plot(np.array(ph)[o], np.array(rv)[o], color=gcol[g],
                        lw=0.9, marker="o", ms=2.4, alpha=0.85,
                        label=g if cond is conds_in_group(g)[0] else None)
        ax.axhline(1.0, color=FG, ls=":", lw=0.8, alpha=0.5)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 120, 240, 360])
        ax.set_title(CELL_LABEL[cell], fontsize=9)
    axes[0, 0].set_ylabel("velocity retention")
    # One shared x label: three copies of this string collide at 5.5 in.
    fig.supxlabel("gait phase at ablation (deg)", fontsize=9)

    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=len(names),
               handlelength=1.2, columnspacing=0.9)
    save_fig(fig, f"{RUN_TAG}_phase_dependence")


def group_rows_table(results, active):
    """Mean / min / max across networks per (group, cell), plus every network's
    own value so the aggregate can be reconstructed from the JSON alone."""
    out = []
    for g in group_order():
        labs = [c["label"] for c in conds_in_group(g)]
        for cell in active:
            row = {"group": g, "cell": cell, "n_networks": len(labs)}
            for which in ("retention", "converged", "conv_ratio"):
                v = np.array([net_scalar(results, l, cell, which) for l in labs],
                             dtype=float)
                vf = v[np.isfinite(v)]
                row[f"{which}_mean"] = float(vf.mean()) if vf.size else float("nan")
                row[f"{which}_min"] = float(vf.min()) if vf.size else float("nan")
                row[f"{which}_max"] = float(vf.max()) if vf.size else float("nan")
                row[f"{which}_per_network"] = {
                    l: net_scalar(results, l, cell, which) for l in labs}
            out.append(row)
    return out


# ===========================================================================
# Main
# ===========================================================================
def main():
    set_pub_style()
    os.makedirs(OUTDIR, exist_ok=True)

    # Labels key the results dict, the npz payload and the figure filenames; a
    # duplicate would silently overwrite a network's results with another's.
    _labels = [c["label"] for c in CONDITIONS]
    if len(set(_labels)) != len(_labels):
        raise ValueError("CONDITIONS labels must be unique. Got: " + str(_labels))

    # Resolve every network's ablation schedule BEFORE launching anything: a
    # missing T_gait should fail in a second, not after an hour of rollouts.
    schedule = {}
    for c in CONDITIONS:
        steps_k, phases_k, T = ablation_steps(c)
        schedule[c["label"]] = {"steps": steps_k, "phases_deg": phases_k,
                                "T_gait": T,
                                "t_gait_source": ("condition"
                                                  if c.get("T_gait") is not None
                                                  else "DEFAULT_T_GAIT")}
    win = int(round(WIN_FRAC * STEPS))
    last = max(max(v["steps"]) for v in schedule.values())
    if last > STEPS - win:
        raise ValueError(
            f"Last stratum at step {last} leaves under WIN_FRAC of rollout "
            f"after it (STEPS={STEPS}, need <= {STEPS - win}). Lower "
            f"EVENT_STEP_BASE or raise STEPS.")

    print(f"[INFO] ablation schedule ({N_ABLATIONS} strata per cell):")
    for c in CONDITIONS:
        v = schedule[c["label"]]
        print(f"  {c['label']:20s} T_gait {v['T_gait']:6.1f}  "
              f"steps {v['steps']}  phases {[f'{p:.0f}' for p in v['phases_deg']]}")

    torch.manual_seed(ROLLOUT_SEED)
    torch.cuda.manual_seed_all(ROLLOUT_SEED)
    np.random.seed(ROLLOUT_SEED)

    env_cfg = Go1EnvCfg()
    env_cfg.scene.num_envs = NUM_REPEATS
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    step_dt = env_cfg.decimation * env_cfg.sim.dt

    # An auto-reset mid-rollout teleports the robot to spawn while the plastic
    # weights and activation buffers carry over -- an off-manifold state the
    # network never saw in training. Catch it here rather than in the results.
    max_steps = int(round(env_cfg.episode_length_s / step_dt))
    if STEPS > max_steps:
        print(f"[WARN] STEPS={STEPS} exceeds one episode ({max_steps} steps at "
              f"episode_length_s={env_cfg.episode_length_s}). The env will "
              f"auto-reset mid-rollout. Reduce STEPS to {max_steps} or fewer.")

    active = [k for k, v in CELLS.items() if v is not None]
    total = len(CONDITIONS) * len(active)
    n_roll = total * N_ABLATIONS
    print(f"[INFO] sweep: {len(CONDITIONS)} conditions x {len(active)} cells "
          f"x {N_ABLATIONS} strata = {n_roll} rollouts of {STEPS} steps "
          f"({STEPS * step_dt:.1f} s each), {NUM_REPEATS} envs each "
          f"-> {NUM_REPEATS * N_ABLATIONS} repeats per cell")

    results, rows, first_lock = {}, [], True
    for ci, cond in enumerate(CONDITIONS):
        sch = schedule[cond["label"]]
        pre, post = align_window(sch["steps"])

        for cell in active:
            key = f"{cond['label']}__{cell}"
            n = len(results) + 1
            print(f"\n[{n}/{total}] {key}  ({N_ABLATIONS} strata)")

            strata = []
            M = tau = None
            for k, (ev, ph) in enumerate(zip(sch["steps"], sch["phases_deg"])):
                traces_k, meas_k, M, tau = run_cell(
                    env, robot, cond["ckpt"],
                    do_freeze=CELLS[cell]["freeze"], do_lock=CELLS[cell]["lock"],
                    step_dt=step_dt,
                    verbose_lock=(first_lock and CELLS[cell]["lock"]),
                    event_step=ev,
                )
                if CELLS[cell]["lock"]:
                    first_lock = False
                strata.append({"k": k, "event_step": int(ev),
                               "phase_deg": float(ph),
                               "traces": traces_k, "measures": meas_k})
                print(f"      stratum {k}  step {ev:4d}  phase {ph:5.1f} deg  "
                      f"retention {float(np.nanmedian(meas_k['retention'])):6.3f}")

            traces, meas = pool_strata(strata, pre, post)

            results[key] = {"traces": traces, "per_repeat": meas,
                            "strata": strata, "pre": pre, "post": post,
                            "M": M, "tau_hebb": tau}

            row = {
                "condition": cond["label"], "group": group_of(cond),
                "cell": cell,
                "M": M, "tau_hebb": tau,
                "T_gait_steps": sch["T_gait"],
                "t_gait_source": sch["t_gait_source"],
                "n_strata": N_ABLATIONS,
                "event_steps": sch["steps"],
                "phases_deg": sch["phases_deg"],
                "retention_per_phase": [
                    float(np.nanmedian(s["measures"]["retention"])) for s in strata],
                "retention_phase_range": float(
                    np.nanmax([np.nanmedian(s["measures"]["retention"]) for s in strata])
                    - np.nanmin([np.nanmedian(s["measures"]["retention"]) for s in strata])),
                "velocity_retention": float(np.nanmedian(meas["retention"])),
                "convergence_ratio": float(np.nanmedian(meas["conv_ratio"])),
                "fraction_converged": float(np.nanmean(meas["converged"])),
                "v_pre": float(np.nanmedian(meas["v_pre"])),
                "v_post": float(np.nanmedian(meas["v_post"])),
                "dW_early": float(np.nanmedian(meas["dW_early"])),
                "dW_late": float(np.nanmedian(meas["dW_late"])),
                "mean_reward": float(np.median(traces["rew"].mean(axis=0))),
                "final_upright": float(np.median(traces["up"][-1])),
                "fallen_fraction": float(np.mean(traces["up"][-1] < 0.5)),
            }
            rows.append(row)
            print(f"      retention {row['velocity_retention']:6.3f} | "
                  f"conv ratio {row['convergence_ratio']:6.3f} | "
                  f"converged {row['fraction_converged']:5.2f} | "
                  f"fallen {row['fallen_fraction']:5.2f}")

    env.close()

    # ---- table ----
    print("\n" + "=" * 74)
    print(f"{'condition':>10} {'cell':>12} {'retention':>10} {'conv ratio':>11} "
          f"{'converged':>10} {'fallen':>8}")
    print("-" * 74)
    for r in rows:
        print(f"{r['condition']:>10} {r['cell']:>12} "
              f"{r['velocity_retention']:>10.3f} {r['convergence_ratio']:>11.3f} "
              f"{r['fraction_converged']:>10.2f} {r['fallen_fraction']:>8.2f}")
    print("=" * 74 + "\n")

    # ---- group table: one line per (M-set, cell), aggregated over networks ----
    group_rows = group_rows_table(results, active)
    print("=" * 78)
    print(f"{'group':>8} {'cell':>12} {'n':>3} {'retention':>10} "
          f"{'[min':>8} {'max]':>8} {'converged':>10}")
    print("-" * 78)
    for r in group_rows:
        print(f"{r['group']:>8} {r['cell']:>12} {r['n_networks']:>3} "
              f"{r['retention_mean']:>10.3f} {r['retention_min']:>8.3f} "
              f"{r['retention_max']:>8.3f} {r['converged_mean']:>10.2f}")
    print("=" * 78 + "\n")

    # ---- persist ----
    meta = {
        "run_tag": RUN_TAG,
        "conditions": [{"group": group_of(c), "label": c["label"],
                        "checkpoint": os.path.abspath(c["ckpt"]),
                        "T_gait_steps": schedule[c["label"]]["T_gait"],
                        "t_gait_source": schedule[c["label"]]["t_gait_source"]}
                       for c in CONDITIONS],
        "cells": CELLS, "param_source": PARAM_SOURCE,
        "num_repeats_per_stratum": NUM_REPEATS,
        "n_ablations": N_ABLATIONS,
        "repeats_per_cell": NUM_REPEATS * N_ABLATIONS,
        "steps": STEPS, "event_step_base": EVENT_STEP_BASE,
        "ablation_schedule": schedule,
        "episode_seconds": STEPS * step_dt,
        "lock_mode": LOCK_MODE, "leg_patterns": LEG_PATTERNS,
        "hold_position": HOLD_POSITION, "mask_locked_actions": MASK_LOCKED_ACTIONS,
        "norm_mode": NORM_MODE,
        "measures": {"win_frac": WIN_FRAC, "early_frac": EARLY_FRAC, "rho": RHO},
        "results": rows,
        "group_results": group_rows,
    }
    with open(os.path.join(OUTDIR, f"{RUN_TAG}_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    payload = {}
    for key, r in results.items():
        # Pooled, event-aligned traces: axis 0 is steps since ablation over
        # [-pre, +post), axis 1 is repeats x strata.
        for name, arr in r["traces"].items():
            payload[f"{key}__{name}"] = arr
        for name, arr in r["per_repeat"].items():
            payload[f"{key}__per_repeat_{name}"] = arr
        payload[f"{key}__align"] = np.array([r["pre"], r["post"]])
        # Per-stratum measures kept separately so phase dependence can be
        # re-analysed without re-running the sweep.
        for st in r["strata"]:
            for name, arr in st["measures"].items():
                payload[f"{key}__stratum{st['k']}_{name}"] = np.atleast_1d(arr)
        payload[f"{key}__stratum_event_steps"] = np.array(
            [st["event_step"] for st in r["strata"]])
        payload[f"{key}__stratum_phases_deg"] = np.array(
            [st["phase_deg"] for st in r["strata"]])
    np.savez_compressed(os.path.join(OUTDIR, f"{RUN_TAG}_data.npz"), **payload)

    for g in group_order():
        fig_traces_group(results, g, [c["label"] for c in conds_in_group(g)])
        fig_summary_group(results, g)
    fig_groups_joint(results)
    fig_phase(results)
    print(f"[DONE] Outputs written to {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
    simulation_app.close()