"""
test_es_go1.py — evaluate a saved Go1 ES checkpoint and inspect the walking gait.

Checkpoint layout (from run_es_go1.py):
    (solver, models, pop_mean_curve, best_sol_curve)

"Best policy" is ambiguous for OpenES, so this script evaluates each requested
candidate parameter vector back-to-back in the SAME environment and reports the
ACHIEVED reward + distance, rather than trusting any single stored "best":

    - "mu"   : solver.mu           -> centre of the search distribution
                                      (robust to fitness noise; usually the deploy choice)
    - "best" : solver.best_param() -> solver.best_mu
                                      (with forget_best=True this is the best of the
                                       FINAL generation only, a single noisy sample)

Gait-focused: alongside the X-Y path it plots forward velocity vs time, which is
the clearest signal of gait quality — a steady plateau near v_target is a real
gait; sawtooth spikes are the lunge-and-recover failure mode.
"""

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# ── Launch Isaac Sim (headless=False so you can watch in the viewport) ──
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from hebbian_locomotion.envs.Go1_env import Go1EnvCfg

from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet
from hebbian_locomotion.networks.han_net import HANNet
# from hebbian_locomotion.networks.d2_net import D2Net   # for D2 checkpoints


# ── Config ──────────────────────────────────────────────────────────────────
EPOCH = 499
STEPS = 1000                 # full training rollout length (50 Hz -> 20 s)
V_TARGET = 1.5               # reference line on the velocity plot (match RewardsCfg)
ACTION_DIM = 12              # Go1: 12 joints
# Which candidate parameter vectors to evaluate and overlay. Order = plot order.
WHICH_PARAMS = ["mu", "best"]          # subset of {"mu", "best", "curr_best"}

CHECKPOINT_PATH = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/07:30-17:11_GO1_FINAL_HEB_499_1529172083.pickle")

current_time = datetime.now().strftime("%m-%d_%H-%M")
OUT_DIR = (
    "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
    "workspace/hebbian_locomotion/logs/positions"
)
STEM_XY   = f"{OUT_DIR}/go1_traj_xy_{STEPS}steps_{current_time}_epoch_{EPOCH}"
STEM_TIME = f"{OUT_DIR}/go1_traj_time_{STEPS}steps_{current_time}_epoch_{EPOCH}"
STEM_VEL  = f"{OUT_DIR}/go1_velocity_{STEPS}steps_{current_time}_epoch_{EPOCH}"

# Colours for each candidate (IEEE-friendly, colourblind-safe / Okabe-Ito)
COLOURS = {"mu": "#0072B2", "best": "#D55E00", "curr_best": "#009E73"}
LABELS  = {"mu": r"$\mu$ (dist. centre)",
           "best": r"best$\_\mu$ (final gen)",
           "curr_best": r"curr best$\_\mu$"}


# ── Publication-ready matplotlib styling (IEEE single-column) ────────────────
def set_pub_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,   # editable/embeddable text in vector PDF
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

IEEE_COL_W = 3.5  # inches, single-column width


def save_fig(fig, stem):
    """Vector PDF + 300-DPI PNG, both publication-ready."""
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    print(f"  saved {stem}.pdf  and  {stem}.png")


def get_params(solver, which):
    if which == "mu":
        return solver.mu
    if which == "best":
        return solver.best_param()      # -> solver.best_mu
    if which == "curr_best":
        return solver.curr_best_mu
    raise ValueError(f"unknown param source: {which}")


def main():
    set_pub_style()

    # ── Load checkpoint ──
    with open(CHECKPOINT_PATH, "rb") as f:
        solver, models_ckpt, pop_mean_curve, best_sol_curve = pickle.load(f)

    # ── Diagnostics: is THIS checkpoint your best epoch? ──
    bsc = np.asarray(best_sol_curve)
    nz = np.nonzero(bsc)[0]
    if nz.size:
        last = nz[-1]
        peak_epoch = int(np.argmax(bsc[: last + 1]))
        print("\n── checkpoint diagnostics ─────────────────────────────")
        print(f"  epochs recorded so far : {last + 1}")
        print(f"  peak best-fitness epoch: {peak_epoch}  "
              f"(value {bsc[peak_epoch]:.2f})")
        print(f"  final recorded epoch   : {last}  (value {bsc[last]:.2f})")
        print(f"  solver.best_reward     : {getattr(solver, 'best_reward', float('nan')):.2f}")
        print(f"  solver.sigma           : "
              f"{np.mean(getattr(solver, 'sigma', np.nan)):.4f}")
        if peak_epoch < last:
            print("  NOTE: peak was BEFORE this checkpoint's final epoch — the")
            print("        single best generation may live in an earlier checkpoint.")
        print("───────────────────────────────────────────────────────\n")

    # ── Env (single agent) ──
    env_cfg = Go1EnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    # ── Build one network; we'll reload params per candidate ──
    # This checkpoint is a Hebbian run -> HebbianNet with max-norm.
    # For HAN / D2 / LSTM checkpoints, swap the class (architecture comes from
    # the checkpoint, so only the class name changes).
    model = HebbianNet(
        popsize=1,
        sizes=models_ckpt.architecture,
        norm_mode="max",
    )
    # model = HANNet(popsize=1, sizes=models_ckpt.architecture, norm_mode="max", M=10, tau_hebb=4)

    results = {}  # which -> dict(xs, ys, zs, vxs, reward, distance)

    for which in WHICH_PARAMS:
        print(f"[eval] rolling out '{which}' ...")
        model.set_a_model_params(get_params(solver, which))
        model.reset_weights()

        obs, _ = env.reset()
        xs, ys, zs, vxs = [], [], [], []
        raw_return = 0.0
        init_xy = robot.data.root_pos_w[:, :2].clone()

        # CRITICAL: replicate the training-time low-pass action filter
        # (run_es_go1.py):  a_t = 0.1*a_raw + 0.9*a_{t-1}
        # The policy was evolved to drive this smoothed signal; feeding raw
        # actions gives jerky control the network never optimised for.
        prev_actions = None
        # Action buffers the augmented reward's action_rate term reads from.
        env.prev_action_buf = torch.zeros(1, ACTION_DIM, device=env.device)
        env.prev_prev_action_buf = torch.zeros(1, ACTION_DIM, device=env.device)

        for _ in range(STEPS):
            actions = model.forward(obs["policy"])
            if prev_actions is not None:
                actions = 0.1 * actions + 0.9 * prev_actions

            # Stash filtered actions so reward terms (and any obs feedback) match
            # training: shift current -> prev before overwriting.
            env.prev_prev_action_buf = env.prev_action_buf
            env.prev_action_buf = actions.detach()

            obs, rewards, _, _, _ = env.step(actions)
            prev_actions = actions

            raw_return += float(rewards.mean().item())
            p = robot.data.root_pos_w.mean(dim=0).cpu().numpy()
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
            # Body-frame forward velocity — the gait-quality signal.
            vxs.append(float(robot.data.root_lin_vel_b[:, 0].mean().item()))

        final_xy = robot.data.root_pos_w[:, :2]
        distance = float(torch.norm(final_xy - init_xy, dim=1).mean().item())
        # Training-equivalent fitness: total_rewards / length * 100
        train_units = raw_return / STEPS * 100.0
        results[which] = dict(xs=xs, ys=ys, zs=zs, vxs=vxs, raw_return=raw_return,
                              fitness=train_units, distance=distance)
        mean_vx = float(np.mean(vxs))
        print(f"        raw_return={raw_return:8.2f}   "
              f"fitness(train units)={train_units:7.3f}   "
              f"distance={distance:6.3f} m   mean vx={mean_vx:5.2f} m/s")

    env.close()

    # ── Verdict ── (compare in TRAINING units against pop_mean, not best_sol)
    best = max(results, key=lambda k: results[k]["fitness"])
    pmc = np.asarray(pop_mean_curve)
    nz_p = np.nonzero(pmc)[0]
    print("\n── achieved performance (re-evaluated, noise included) ──")
    print("   reward shown in TRAINING units (raw_return / length * 100)")
    for which in WHICH_PARAMS:
        tag = "  <-- best here" if which == best else ""
        print(f"  {which:>10s}:  fitness {results[which]['fitness']:7.3f}   "
              f"distance {results[which]['distance']:6.3f} m{tag}")
    if nz_p.size:
        print(f"\n  FAIR training baseline = pop_mean_curve (avg over popsize),")
        print(f"  NOT best_sol_curve (max over popsize, optimistically biased).")
        print(f"  final-epoch pop mean : {pmc[nz_p[-1]]:7.3f}  (training units)")
    print("─────────────────────────────────────────────────────────\n")

    # ── Figure 1: X–Y trajectory (overlaid candidates) ──
    fig1, ax1 = plt.subplots(figsize=(IEEE_COL_W, IEEE_COL_W * 0.85))
    for which in WHICH_PARAMS:
        r = results[which]; c = COLOURS.get(which, None)
        ax1.plot(r["xs"], r["ys"], color=c, label=LABELS.get(which, which))
        ax1.plot(r["xs"][0], r["ys"][0], "o", color=c, ms=4, mfc="white", mew=0.8)
        ax1.plot(r["xs"][-1], r["ys"][-1], "s", color=c, ms=4)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, lw=0.3, alpha=0.5)
    ax1.legend(loc="best")
    fig1.tight_layout()
    save_fig(fig1, STEM_XY)

    # ── Figure 2: forward position (x) vs time (overlaid candidates) ──
    fig2, ax2 = plt.subplots(figsize=(IEEE_COL_W, IEEE_COL_W * 0.7))
    t = np.arange(STEPS)
    for which in WHICH_PARAMS:
        r = results[which]; c = COLOURS.get(which, None)
        ax2.plot(t, r["xs"], color=c, label=LABELS.get(which, which))
    ax2.set_xlabel("time step")
    ax2.set_ylabel("forward position x (m)")
    ax2.grid(True, lw=0.3, alpha=0.5)
    ax2.legend(loc="best")
    fig2.tight_layout()
    save_fig(fig2, STEM_TIME)

    # ── Figure 3: forward velocity vs time — the gait-quality diagnostic ──
    fig3, ax3 = plt.subplots(figsize=(IEEE_COL_W, IEEE_COL_W * 0.7))
    for which in WHICH_PARAMS:
        r = results[which]; c = COLOURS.get(which, None)
        ax3.plot(t, r["vxs"], color=c, label=LABELS.get(which, which))
    ax3.axhline(V_TARGET, color="0.4", lw=0.8, ls="--", label=r"$v_{\mathrm{target}}$")
    ax3.set_xlabel("time step")
    ax3.set_ylabel(r"forward velocity $v_x$ (m/s)")
    ax3.grid(True, lw=0.3, alpha=0.5)
    ax3.legend(loc="best")
    fig3.tight_layout()
    save_fig(fig3, STEM_VEL)


if __name__ == "__main__":
    main()
    simulation_app.close()