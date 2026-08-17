"""
Fitness-noise diagnostic for the Go1 HAN / Hebbian ES pipeline.

Question: when OpenES estimates its gradient from a generation, how much of
that estimate reflects the evolved ABCD coefficients, and how much reflects
the random plastic weight draw in reset_weights() plus environment variability?

Method. Four parallel rollouts, each with N_ENVS environments:

  A  same params (mu), independent weight draws, independent env states
     -> Var(A) = Var(weight init) + Var(env)

  B  same params (mu), SHARED weight draw,      independent env states
     -> Var(B) = Var(env)

  C1 N individuals sampled ANTITHETICALLY from mu + sigma*eps
  C2 the same N individuals, re-evaluated with fresh draws
     -> Var(C) = Var(signal) + Var(weight init) + Var(env)

Hence:
     Var(env)         = Var(B)
     Var(weight init) = Var(A) - Var(B)
     Var(signal)      = Var(C) - Var(A)

HEADLINE STATISTIC. With rank_fitness=False, OpenES.tell() computes

    z = (f - mean(f)) / std(f)
    change_mu = 1/(popsize*sigma) * eps.T @ z

so the update is LINEAR in raw fitness -- ranks are never formed. Because C1
and C2 share the same eps, both gradient estimates are directly computable and
their cosine similarity IS the quantity that matters, not a proxy for it:

    cos(g1, g2) ~ 1   consecutive generations pull the same way
    cos(g1, g2) ~ 0   the gradient is noise

Pearson r is reported as a consistency check (for random eps with N << d,
cos(g1,g2) ~= r). Spearman rho is kept as a secondary diagnostic: with bimodal
fitness a large r-rho gap means ES resolves "walks vs doesn't" but not the
ordering within either cluster.

Edit the CONFIG block, then:

    python test_fitness_noise.py
"""


from datetime import datetime
current_time = datetime.now().strftime("%m:%d-%H:%M")

# ===========================================================================
# CONFIG -- edit everything here
# ===========================================================================
CKPT = ('/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/07:29-17:39_GO1_NEW_WITH_HEALTHY_BONUS_4096_SPEED_1_NO_RANK_FITNESS_UPDATED_FOOT_CONTACT_HEALTHY_BONUS_0.1_HEB_499_418040111.pickle')
OUTDIR = (
    "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
    "workspace/hebbian_locomotion/analysis/noise"
)

N_ENVS = 64                    # samples per condition; MUST be even (antithetic)
EPISODE_STEPS = 2000            # = EPISODE_LENGTH_TRAIN in run_es_go1.py
ACTION_FILTER_ALPHA = 0.1      # a_t = 0.1*a_raw + 0.9*a_{t-1}
FITNESS_SCALE = 100.0          # training does total / current_length * 100

# Network overrides. None -> read from the checkpointed model object.
NET_M = None                   # moving-average window
NET_TAU_HEBB = None            # Hebbian update period
NET_NORM_MODE = "max"

SIGMA = None                   # None -> read solver.sigma
N_BOOTSTRAP = 5000             # resamples for the CI on r / cos(g1,g2)
SEED = 0
HEADLESS = True

MODEL = 'HEB'

TAG = f"{current_time}_{MODEL}"   
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
from scipy.stats import spearmanr, pearsonr  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from hebbian_locomotion.envs.Go1_env import Go1EnvCfg  # noqa: E402
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
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[FIG] {name}.pdf / .png")


# ===========================================================================
# Helpers
# ===========================================================================
def load_checkpoint(path):
    """Return (solver, models, run_cfg) from a 4- or 5-element pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    run_cfg = data[4] if len(data) >= 5 else None
    return data[0], data[1], run_cfg


def share_weight_init(net):
    """Overwrite every individual's plastic weights with individual 0's draw."""
    for i in range(net.n_layers):
        w0 = net.weights[i][0:1]
        net.weights[i] = w0.expand_as(net.weights[i]).contiguous()


def rollout_fitness(env, net, action_dim, shared_init=False):
    """One parallel episode; returns (N_ENVS,) fitness matching run_es_go1.py.

    Mirrors the training loop: unfiltered first action, EMA thereafter,
    prev_action_buf bookkeeping, unmasked reward accumulation (done_mask is
    never updated in training and terminations are EmptyManagerCfg), and the
    final /steps*100 scaling.
    """
    obs, _ = env.reset()
    net.reset_weights()
    if shared_init:
        share_weight_init(net)

    total = torch.zeros(N_ENVS, device=env.device)
    prev_actions = None

    env.prev_action_buf = torch.zeros(N_ENVS, action_dim, device=env.device)
    env.prev_prev_action_buf = torch.zeros(N_ENVS, action_dim, device=env.device)

    for _ in range(EPISODE_STEPS):
        actions = net.forward(obs["policy"])
        if prev_actions is not None:
            actions = ACTION_FILTER_ALPHA * actions + (1.0 - ACTION_FILTER_ALPHA) * prev_actions

        env.prev_prev_action_buf = env.prev_action_buf
        env.prev_action_buf = actions.detach()

        obs, rewards, _, _, _ = env.step(actions)
        prev_actions = actions
        total += rewards

    return (total / EPISODE_STEPS * FITNESS_SCALE).cpu().numpy()


def zscore(x):
    return (x - x.mean()) / x.std()


def grad_cosine(eps, f1, f2):
    """Cosine similarity of the two OpenES gradient estimates (rank_fitness=False)."""
    g1 = eps.T @ zscore(f1)
    g2 = eps.T @ zscore(f2)
    return float(g1 @ g2 / (np.linalg.norm(g1) * np.linalg.norm(g2)))


def bootstrap_ci(eps, f1, f2, n_boot, rng):
    """Percentile CIs for Pearson r and the gradient cosine."""
    n = len(f1)
    rs, cs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(f1[idx]) == 0 or np.std(f2[idx]) == 0:
            continue
        rs.append(pearsonr(f1[idx], f2[idx])[0])
        cs.append(grad_cosine(eps[idx], f1[idx], f2[idx]))
    return (np.percentile(rs, [2.5, 97.5]).tolist(),
            np.percentile(cs, [2.5, 97.5]).tolist())


def safe_var(x):
    return float(np.var(x, ddof=1))


# ===========================================================================
# Figures
# ===========================================================================
def fig_distributions(fA, fB, fC):
    fig, ax = plt.subplots(figsize=	(FIG_W_FULL, FIG_W_FULL * 0.52))

    data = [fB, fA, fC]
    labels = ["env only\n(B)", "env + init\n(A)", "env + init\n+ params (C)"]
    colours = [OKABE_ITO["sky"], OKABE_ITO["orange"], OKABE_ITO["green"]]

    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color=OKABE_ITO["black"], lw=0.9),
                    boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                    capprops=dict(lw=0.6))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)

    rng = np.random.default_rng(SEED)
    for i, (d, c) in enumerate(zip(data, colours), start=1):
        ax.plot(rng.normal(i, 0.055, size=len(d)), d, ".",
                color=c, markersize=2.5, alpha=0.7)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("fitness")
    save_fig(fig, f"{TAG}_distributions")


def fig_variance_decomposition(v_env, v_init, v_signal):
    total = v_env + v_init + v_signal
    if total <= 0:
        print("[WARN] Non-positive total variance; skipping decomposition figure.")
        return
    parts = np.array([v_env, v_init, v_signal]) / total * 100.0

    fig, ax = plt.subplots(figsize=(FIG_W_FULL, FIG_W_FULL * 0.28))
    left = 0.0
    for p, c, lab in zip(parts,
                         [OKABE_ITO["sky"], OKABE_ITO["orange"], OKABE_ITO["green"]],
                         ["environment", "weight init", "parameters"]):
        ax.barh(0, p, left=left, color=c, edgecolor="none", label=lab)
        if p > 6:
            ax.text(left + p / 2, 0, f"{p:.0f}%", ha="center", va="center",
                    fontsize=8, color="white")
        left += p

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("share of fitness variance (%)")   # was "(\\%)"
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3)
    save_fig(fig, f"{TAG}_variance_decomposition")


def fig_reliability(fC1, fC2, r, g_cos):
    fig, ax = plt.subplots(figsize=	(FIG_W_FULL, FIG_W_FULL * 0.72))
    ax.plot(fC1, fC2, "o", color=OKABE_ITO["purple"], markersize=3,
            alpha=0.8, mew=0)

    lo, hi = min(fC1.min(), fC2.min()), max(fC1.max(), fC2.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--",
            color=OKABE_ITO["black"], lw=0.6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("fitness, evaluation 1")
    ax.set_ylabel("fitness, evaluation 2")
    ax.set_title(rf"$r = {r:.2f}$,  $\cos(\hat{{g}}_1,\hat{{g}}_2) = {g_cos:.2f}$")
    save_fig(fig, f"{TAG}_reliability")


# ===========================================================================
# Main
# ===========================================================================
def main():
    assert N_ENVS % 2 == 0, "N_ENVS must be even for antithetic sampling."

    set_pub_style()
    os.makedirs(OUTDIR, exist_ok=True)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    # --- checkpoint, with defensive network configuration -----------------
    solver, models_ckpt, run_cfg = load_checkpoint(CKPT)
    mu = np.asarray(solver.mu)
    sizes = list(models_ckpt.architecture)
    M = NET_M if NET_M is not None else getattr(models_ckpt, "M", 1)
    tau_hebb = NET_TAU_HEBB if NET_TAU_HEBB is not None else getattr(models_ckpt, "tau_hebb", 1)
    sigma = float(SIGMA if SIGMA is not None else solver.sigma)

    print(f"[INFO] architecture {sizes}  n_params {mu.size}")
    print(f"[INFO] M {M}  tau_hebb {tau_hebb}  sigma {sigma:.4f}")
    print(f"[INFO] rank_fitness {getattr(solver, 'rank_fitness', None)}  "
          f"antithetic {getattr(solver, 'antithetic', None)}")

    obs_dim, action_dim = sizes[0], sizes[-1]

    # --- environment ------------------------------------------------------
    env_cfg = Go1EnvCfg()
    env_cfg.scene.num_envs = N_ENVS
    env = ManagerBasedRLEnv(cfg=env_cfg)

    net = HANNet(N_ENVS, sizes=sizes, norm_mode=NET_NORM_MODE,
                 M=M, tau_hebb=tau_hebb)

    # --- A: same params, independent weight draws -------------------------
    net.set_a_model_params(mu)
    print("[RUN] A  same params, independent inits")
    fA = rollout_fitness(env, net, action_dim, shared_init=False)

    # --- B: same params, shared weight draw -------------------------------
    net.set_a_model_params(mu)
    print("[RUN] B  same params, shared init")
    fB = rollout_fitness(env, net, action_dim, shared_init=True)

    # --- C: an antithetic generation, evaluated twice ---------------------
    half = N_ENVS // 2
    eps_half = np.random.randn(half, mu.size)
    eps = np.concatenate([eps_half, -eps_half])
    pop = (mu[None, :] + sigma * eps).astype(np.float32)

    net.set_models_params(pop)
    print("[RUN] C1 sampled population, evaluation 1")
    fC1 = rollout_fitness(env, net, action_dim, shared_init=False)

    net.set_models_params(pop)
    print("[RUN] C2 sampled population, evaluation 2")
    fC2 = rollout_fitness(env, net, action_dim, shared_init=False)

    env.close()

    # --- statistics -------------------------------------------------------
    fC = np.concatenate([fC1, fC2])
    v_A, v_B, v_C = safe_var(fA), safe_var(fB), safe_var(fC)

    v_env = v_B
    v_init = max(v_A - v_B, 0.0)
    v_signal = max(v_C - v_A, 0.0)
    if v_A - v_B < 0 or v_C - v_A < 0:
        print("[WARN] A variance component was clamped to zero: the components "
              "are not separable at this N_ENVS. Increase N_ENVS before "
              "interpreting the decomposition.")

    g_cos = grad_cosine(eps, fC1, fC2)
    r, _ = pearsonr(fC1, fC2)
    rho, rho_p = spearmanr(fC1, fC2)
    r_ci, g_ci = bootstrap_ci(eps, fC1, fC2, N_BOOTSTRAP, rng)

    summary = {
        "tag": TAG,
        "checkpoint": os.path.abspath(CKPT),
        "n_envs": N_ENVS,
        "episode_steps": EPISODE_STEPS,
        "M": int(M),
        "tau_hebb": int(tau_hebb),
        "sigma": sigma,
        "mean_fitness_mu": float(fA.mean()),
        "var_A_env_plus_init": v_A,
        "var_B_env": v_B,
        "var_C_total": v_C,
        "var_environment": v_env,
        "var_weight_init": v_init,
        "var_parameters": v_signal,
        "sd_environment": float(np.sqrt(v_env)),
        "sd_weight_init": float(np.sqrt(v_init)),
        "sd_parameters": float(np.sqrt(v_signal)),
        "gradient_cosine": g_cos,
        "gradient_cosine_ci95": g_ci,
        "pearson_r": float(r),
        "pearson_r_ci95": r_ci,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
    }

    print("\n--- summary " + "-" * 45)
    for k, v in summary.items():
        print(f"  {k:24s} {v}")
    print("-" * 57)

    print("\nInterpretation")
    print(f"  cos(g1, g2) = {g_cos:.3f}  95% CI [{g_ci[0]:.3f}, {g_ci[1]:.3f}]")
    if g_cos < 0.3:
        print("  -> Consecutive gradient estimates are nearly unrelated. "
              "Variance reduction (shared init, repeated evaluations) should "
              "help substantially.")
    elif g_cos < 0.6:
        print("  -> The gradient is noisy but informative. Shared init and "
              "2-4 repeats per evaluation are worth adding.")
    else:
        print("  -> The gradient is reliable. Look elsewhere for the cause of "
              "run-to-run failure (reward landscape, sigma schedule).")

    if r - rho > 0.25:
        print(f"  Note: r ({r:.2f}) >> rho ({rho:.2f}). ES separates walkers "
              "from non-walkers but barely orders within each cluster.")

    dominant = max(("environment", v_env), ("weight init", v_init),
                   ("parameters", v_signal), key=lambda kv: kv[1])[0]
    print(f"  Dominant variance source: {dominant}\n")

    np.savez_compressed(
        os.path.join(OUTDIR, f"{TAG}_data.npz"),
        fA=fA, fB=fB, fC1=fC1, fC2=fC2, eps=eps.astype(np.float32),
    )
    with open(os.path.join(OUTDIR, f"{TAG}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig_distributions(fA, fB, fC)
    fig_variance_decomposition(v_env, v_init, v_signal)
    fig_reliability(fC1, fC2, r, g_cos)

    print(f"[DONE] Outputs written to {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
    simulation_app.close()