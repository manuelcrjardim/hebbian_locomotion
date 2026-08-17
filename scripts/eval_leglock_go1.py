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
"""

from datetime import datetime

current_time = datetime.now().strftime("%m:%d-%H:%M")

# ===========================================================================
# CONFIG -- edit everything here
# ===========================================================================
ROOT = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
        "workspace/hebbian_locomotion")

# --- the policies to sweep; label is used in filenames and figures ---
CONDITIONS = [
    {"label": "M1",   "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:04-11:52_GO1_FINAL_HAN_499_860896728_M_1_benchmark.pickle"},
    {"label": "M10 1",   "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:06-13:33_GO1_FINAL_HAN_499_46838495_M_10_benchmark.pickle"},
    {"label": "M10 2",   "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:07-14:17_GO1_FINAL_HAN_499_1315706754_M_10_benchmark.pickle"},
    {"label": "M20",  "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:04-18:00_GO1_FINAL_HAN_499_1853697842_M_20_benchamrk.pickle"},
    {"label": "M160", "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:04-00:56_GO1_FINAL_HAN_499_576310242_M_160_benchmark.pickle"},
]

RUN_TAG = "leglock_sweep_RL_hip_only"                 # prefix for every output file
OUTDIR = f"{ROOT}/analysis/leglock_{current_time}_{RUN_TAG}"

PARAM_SOURCE = "mu"          # "mu" or "best_mu"; mu is the converged solution
NUM_REPEATS = 11             # parallel copies of the SAME network; odd -> true median
STEPS = 1000                 # control steps; 1000 = 20 s at 50 Hz, as in HAN
EVENT_STEP = 500             # freeze and/or lock engage here; also the window
                             # origin in the UNDAMAGED cells, so all four cells
                             # are measured over identical spans

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
# Figure style -- NeurIPS 2026 (5.5 in text block, 10 pt Times)
# ===========================================================================
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "verm": "#D55E00", "purple": "#CC79A7",
}
CELL_COLOUR = {
    "baseline": OKABE_ITO["black"], "freeze_only": OKABE_ITO["sky"],
    "lock_only": OKABE_ITO["orange"], "both": OKABE_ITO["verm"],
}
CELL_LABEL = {
    "baseline": "baseline", "freeze_only": "freeze",
    "lock_only": "lock", "both": "freeze + lock",
}
FIG_W_FULL = 5.5


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
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[FIG] {name}.pdf / .png")


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
            f"[measures] windows do not fit: STEPS={n_steps}, EVENT_STEP="
            f"{event_step}, WIN_FRAC={WIN_FRAC}. Need EVENT_STEP >= "
            f"{win} and EVENT_STEP <= {n_steps - win}."
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
def run_cell(env, robot, ckpt_path, do_freeze, do_lock, step_dt, verbose_lock):
    """Run a single cell of the 2x2 and return traces plus per-repeat measures."""
    solver, models_ckpt, _ = load_checkpoint(ckpt_path)
    params = np.asarray(solver.mu if PARAM_SOURCE == "mu" else solver.best_mu)

    # Rebuilt every cell: freeze_network() shadows a method on HebbianNet
    # instances, and a stale activation buffer would leak across cells.
    model, M, tau = build_network(models_ckpt, NUM_REPEATS)
    model.set_a_model_params(params)

    obs, _ = env.reset()
    model.reset_weights()

    leg_lock = LegLock(
        lock_step=(EVENT_STEP if do_lock else None),
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
        if do_freeze and step == EVENT_STEP and not frozen:
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
    return traces, compute_measures(traces["vx"], traces["dW"], EVENT_STEP), M, tau


# ===========================================================================
# Figures
# ===========================================================================
def fig_traces(results):
    """One row per condition; v_x and ||dW|| with all four cells overlaid."""
    labels = [c["label"] for c in CONDITIONS]
    fig, axes = plt.subplots(len(labels), 2, sharex=True, sharey="col",
                             figsize=(FIG_W_FULL, 1.35 * len(labels) + 0.6))
    axes = np.atleast_2d(axes)

    for r, lab in enumerate(labels):
        for cell in CELLS:
            key = f"{lab}__{cell}"
            if key not in results:
                continue
            tr = results[key]["traces"]
            col, name = CELL_COLOUR[cell], CELL_LABEL[cell]
            for c, arr in enumerate((tr["vx"], tr["dW"])):
                med = np.median(arr, axis=1)
                axes[r, c].plot(np.arange(len(med)), med, color=col, lw=1.0,
                                label=name if (r == 0 and c == 0) else None)

        for c in range(2):
            axes[r, c].axvline(EVENT_STEP, color="0.55", ls="--", lw=0.8, zorder=0)
            axes[r, c].set_xlim(0, STEPS)
        axes[r, 0].set_ylabel(rf"{lab}" "\n" r"$v_x$ (m s$^{-1}$)")
        axes[r, 1].set_ylabel(r"$\sum_k \Vert \Delta W_t^{(k)} \Vert_2$")

    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")

    # Legend above the figure: with several cells overlaid there is no free
    # space inside the axes that does not cover a trace.
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.99),
               ncol=len(handles), handlelength=1.6, columnspacing=1.4,
               borderaxespad=0.0)
    fig.subplots_adjust(hspace=0.15, wspace=0.32)
    save_fig(fig, f"{RUN_TAG}_traces")


def fig_summary(results):
    """Grouped bars: velocity retention and convergence fraction per cell."""
    labels = [c["label"] for c in CONDITIONS]
    cells = list(CELLS.keys())
    x = np.arange(len(labels))
    w = 0.8 / len(cells)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W_FULL, 2.1))
    for k, cell in enumerate(cells):
        off = (k - (len(cells) - 1) / 2) * w
        ret_med, ret_err, conv = [], [[], []], []
        for lab in labels:
            key = f"{lab}__{cell}"
            if key not in results:
                ret_med.append(np.nan); conv.append(np.nan)
                ret_err[0].append(0); ret_err[1].append(0)
                continue
            r = results[key]["per_repeat"]["retention"]
            m = np.nanmedian(r)
            lo, hi = np.nanpercentile(r, [25, 75])
            ret_med.append(m)
            ret_err[0].append(max(0, m - lo)); ret_err[1].append(max(0, hi - m))
            conv.append(np.nanmean(results[key]["per_repeat"]["converged"]))
        axes[0].bar(x + off, ret_med, w, yerr=np.array(ret_err), capsize=1.5,
                    color=CELL_COLOUR[cell], label=CELL_LABEL[cell],
                    error_kw={"lw": 0.7})
        axes[1].bar(x + off, conv, w, color=CELL_COLOUR[cell])

    axes[0].axhline(1.0, color="0.55", ls=":", lw=0.8)
    axes[0].set_ylabel("velocity retention")
    axes[1].set_ylabel("fraction converged")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels)

    # Legend above the figure, not inside axes[0] -- with 4 cells there is no
    # free space inside the axes that doesn't sit on top of a bar or tick label.
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(handles), handlelength=1.2, columnspacing=0.9,
               borderaxespad=0.0)
    fig.subplots_adjust(wspace=0.34, top=0.78)
    save_fig(fig, f"{RUN_TAG}_summary")


# ===========================================================================
# Main
# ===========================================================================
def main():
    set_pub_style()
    os.makedirs(OUTDIR, exist_ok=True)

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
    print(f"[INFO] sweep: {len(CONDITIONS)} conditions x {len(active)} cells "
          f"= {total} rollouts of {STEPS} steps ({STEPS * step_dt:.1f} s each)")

    results, rows, first_lock = {}, [], True
    for ci, cond in enumerate(CONDITIONS):
        for cell in active:
            key = f"{cond['label']}__{cell}"
            n = len(results) + 1
            print(f"\n[{n}/{total}] {key}")
            traces, meas, M, tau = run_cell(
                env, robot, cond["ckpt"],
                do_freeze=CELLS[cell]["freeze"], do_lock=CELLS[cell]["lock"],
                step_dt=step_dt,
                verbose_lock=(first_lock and CELLS[cell]["lock"]),
            )
            if CELLS[cell]["lock"]:
                first_lock = False

            results[key] = {"traces": traces, "per_repeat": meas,
                            "M": M, "tau_hebb": tau}

            row = {
                "condition": cond["label"], "cell": cell,
                "M": M, "tau_hebb": tau,
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

    # ---- persist ----
    meta = {
        "run_tag": RUN_TAG,
        "conditions": [{"label": c["label"], "checkpoint": os.path.abspath(c["ckpt"])}
                       for c in CONDITIONS],
        "cells": CELLS, "param_source": PARAM_SOURCE,
        "num_repeats": NUM_REPEATS, "steps": STEPS, "event_step": EVENT_STEP,
        "episode_seconds": STEPS * step_dt,
        "lock_mode": LOCK_MODE, "leg_patterns": LEG_PATTERNS,
        "hold_position": HOLD_POSITION, "mask_locked_actions": MASK_LOCKED_ACTIONS,
        "norm_mode": NORM_MODE,
        "measures": {"win_frac": WIN_FRAC, "early_frac": EARLY_FRAC, "rho": RHO},
        "results": rows,
    }
    with open(os.path.join(OUTDIR, f"{RUN_TAG}_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    payload = {}
    for key, r in results.items():
        for name, arr in r["traces"].items():
            payload[f"{key}__{name}"] = arr
        for name, arr in r["per_repeat"].items():
            payload[f"{key}__per_repeat_{name}"] = arr
    np.savez_compressed(os.path.join(OUTDIR, f"{RUN_TAG}_data.npz"), **payload)

    fig_traces(results)
    fig_summary(results)
    print(f"[DONE] Outputs written to {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
    simulation_app.close()