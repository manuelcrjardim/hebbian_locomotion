import os
import json
import torch
import numpy as np
import pickle
import copy
import random
import cv2
import matplotlib.pyplot as plt

# 1. Launch Isaac Sim (Must be first!)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from isaaclab.envs import ManagerBasedRLEnv

# Import your custom modules
from hebbian_locomotion.envs.AnymalEnv import AnymalEnvCfg
from hebbian_locomotion.envs.Go1_env import Go1EnvCfg
from hebbian_locomotion.envs.GeckoEnv import GeckoEnvCfg
from isaaclab.managers import SceneEntityCfg
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet
from hebbian_locomotion.networks.han_net import HANNet
from hebbian_locomotion.networks.LSTM import LSTMNet
from hebbian_locomotion.networks.ES_classes import OpenES


current_time = datetime.now().strftime("%m:%d-%H:%M")

# --- Seeding -----------------------------------------------------------
# Draws a fresh seed from OS entropy on every launch. Set RUN_SEED only
# when you deliberately want to re-run a specific configuration.
_env_seed = os.environ.get("RUN_SEED")
SEED = int(_env_seed) if _env_seed is not None else random.SystemRandom().randint(0, 2**31 - 1)

random.seed(SEED)
torch.manual_seed(SEED)          # LSTMNet init + reset_weights()
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)             # OpenES.ask() -> np.random.randn

print(f"[INFO] Seed: {SEED} ({'explicit' if _env_seed else 'auto'})")
# -----------------------------------------------------------------------


# ======================================================================
# Config capture
# ----------------------------------------------------------------------
# The pickled (solver, models, curves) tuple preserves the ES config and the
# network *shape* implicitly, but the reward terms, target speed, and hand-set
# loop constants live only as script variables and are lost with the script.
# These helpers snapshot all of it into a fifth pickle element + JSON sidecar,
# reading the reward terms off the LIVE env cfg (post __post_init__) so the
# recorded values are the ones actually trained on, not the pre-override intent.
# ======================================================================
def _term_to_dict(term):
    """Introspect a single RewTerm: function name, weight, and scalar params only."""
    func = getattr(term, "func", None)
    params = getattr(term, "params", {}) or {}
    return {
        "func": getattr(func, "__name__", str(func)),
        "weight": getattr(term, "weight", None),
        # keep only JSON-friendly scalar params (target_speed, std, ...);
        # drop SceneEntityCfg and other non-scalar objects.
        "params": {
            k: v for k, v in params.items()
            if isinstance(v, (int, float, str, bool))
        },
    }


def _rewards_to_dict(rewards_cfg):
    """Walk a reward configclass, capturing every RewTerm as effectively set."""
    terms = {}
    for name, term in vars(rewards_cfg).items():
        if name.startswith("_"):
            continue
        if hasattr(term, "func"):          # it's a RewTerm
            terms[name] = _term_to_dict(term)
    return terms


def build_run_cfg(env_cfg, models, solver, *,
                  label, robot, model,
                  epochs, save_every,
                  episode_length_train, initial_episode_length,
                  fall_z_threshold, growth_factor,
                  action_filter_alpha, reward_scaling,
                  obs_dim, action_dim, timestamp, seed=SEED):
    """Assemble a plain, JSON-serialisable record of everything that shaped a run:
    reward terms (off the live env cfg), ES hyperparameters (off the solver),
    network architecture / plasticity params (off the model), and the hand-set
    loop constants that live on no object."""

    # --- reward terms, as effectively set (post __post_init__) ---
    reward_terms = _rewards_to_dict(env_cfg.rewards)

    # --- ES hyperparameters (read back off the solver object) ---
    es = {
        "popsize": getattr(solver, "popsize", None),
        "learning_rate": getattr(solver, "learning_rate", None),
        "learning_rate_decay": getattr(solver, "learning_rate_decay", None),
        "sigma_init": getattr(solver, "sigma_init", None),
        "sigma_decay": getattr(solver, "sigma_decay", None),
        "sigma_current": getattr(solver, "sigma", None),
        "rank_fitness": getattr(solver, "rank_fitness", None),
        "antithetic": getattr(solver, "antithetic", None),
    }

    # --- network architecture / plasticity hyperparameters ---
    net = {
        "class": type(models).__name__,
        "sizes": getattr(models, "architecture", None),
        "norm_mode": getattr(models, "norm_mode", None),
        "init_noise": getattr(models, "init_noise", None),
        "M": getattr(models, "M", None),                 # HANNet boxcar window
        "tau_hebb": getattr(models, "tau_hebb", None),   # HANNet dual-timescale
        "gamma_logit": None,                             # D2Net evolved EMA decay
        "state_init": getattr(models, "state_init", None),  # LSTM h/c init scale
    }
    # gamma_logit is a per-individual tensor on D2Net; store the first indiv's scalar.
    gl = getattr(models, "gamma_logit", None)
    if gl is not None:
        try:
            net["gamma_logit"] = float(gl.flatten()[0].item())
        except Exception:
            net["gamma_logit"] = str(gl)

    # --- env-level parameters that impact training ---
    env = {
        "target_speed": getattr(env_cfg, "target_speed", None),
        "decimation": getattr(env_cfg, "decimation", None),
        "sim_dt": getattr(getattr(env_cfg, "sim", None), "dt", None),
        "episode_length_s": getattr(env_cfg, "episode_length_s", None),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }

    return {
        "robot": robot,
        "model": model,
        "label": label,
        "timestamp": timestamp,
        "seed": seed,
        "epochs": epochs,
        "save_every": save_every,
        "episode_length_train": episode_length_train,
        "initial_episode_length": initial_episode_length,
        "fall_z_threshold": fall_z_threshold,
        "growth_factor": growth_factor,
        "action_filter_alpha": action_filter_alpha,
        "reward_scaling": reward_scaling,
        "reward_terms": reward_terms,
        "es": es,
        "network": net,
        "env": env,
    }


def main():

    #run_name = f"han_run_lr_{current_time}"

    base_dir = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/tensorboard"
    #custom_log_dir = os.path.join(base_dir, run_name)

    #writer = SummaryWriter(log_dir=custom_log_dir)
    # --- 1. Hyperparameters ---
    EPOCHS = 500                    # Total generations
    EPISODE_LENGTH_TRAIN = 500      # MAX episode length (curriculum cap)
    INITIAL_EPISODE_LENGTH = 500     # Starting episode length
    FALL_Z_THRESHOLD = 0.1          # Robot considered fallen when base z < this
    GROWTH_FACTOR = 1.5             # next_length = last_fall_step * GROWTH_FACTOR
    POPSIZE = 4096                   # Population size (paper uses 1024)
    SAVE_EVERY = 500

    ROBOT = 'GO1'
    MODEL = 'LSTM'
    REWARD = f'FINAL'
    # ES parameters (paper: sigma_init=0.1, decay=0.999, lr=0.1, lr_decay=0.999)
    LEARNING_RATE = 0.1
    LEARNING_RATE_DECAY = 0.999
    LEARNING_RATE_LIMIT = 0.001
    SIGMA_INIT = 0.2
    SIGMA_DECAY = 0.995
    SIGMA_LIMIT = 0.01
    RANK_FITNESS = False
    ANTITHETIC = True
    WEIGHT_DECAY = 0.0

    # Size HIDDEN so the LSTM's evolved-param count matches the plain HebbianNet
    # ([33,64,32,12] -> ~22,720 evolved params). For the 3-stack SeqLSTMs at
    # obs=33/act=12, H=25 gives ~22.2k (H=26 ~23.3k); granularity is ~1.1k/unit,
    # so 25 is the closest match. CONFIRM against the printed n_params below.
    HIDDEN = 25                                  # Size of hidden layer in LSTM network

    print("[INFO] Initializing Environment...")
    env_cfg = Go1EnvCfg()
    env_cfg.seed = SEED
    env_cfg.scene.num_envs = POPSIZE
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # --- 2. Network & Optimizer Setup ---
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = 12

    run_name = f"{current_time}_{ROBOT}_{REWARD}_{MODEL}_{SEED}"

    base_dir = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/tensorboard"
    custom_log_dir = os.path.join(base_dir, run_name)

    writer = SummaryWriter(log_dir=custom_log_dir)

    print(f"\n[INFO] Initializing LSTM Network (obs={obs_dim}, hidden={HIDDEN}, act={action_dim})...")

    # Recurrence baseline: 3-stacked-block SeqLSTMs (aliased LSTMNet), exposing the
    # same ES interface as HebbianNet. The evolved weight matrices are fixed within
    # a rollout; the hidden/cell state is the within-lifetime variable, reset each
    # generation by reset_weights() (interface-compatible name — it resets h/c).
    models = LSTMNet(
        popsize=POPSIZE,
        sizes=[obs_dim, HIDDEN, action_dim],
    )
    n_params = models.get_n_params_a_model()
    print(f"[INFO] Evolvable parameters per individual: {n_params}")

    print("\n[INFO] Initializing OpenES Optimizer...")
    solver = OpenES(
        n_params,
        popsize=POPSIZE,
        rank_fitness=RANK_FITNESS,
        antithetic=ANTITHETIC,
        learning_rate=LEARNING_RATE,
        learning_rate_decay=LEARNING_RATE_DECAY,
        learning_rate_limit=LEARNING_RATE_LIMIT,
        sigma_init=SIGMA_INIT,
        sigma_decay=SIGMA_DECAY,
        sigma_limit=SIGMA_LIMIT,
        weight_decay=WEIGHT_DECAY
    )
    solver.set_mu(models.get_a_model_params())

    # --- Capture the full run configuration (reward terms, ES/net/env params,
    #     hand-set loop constants). Built once, appended to every checkpoint. ---
    run_cfg = build_run_cfg(
        env_cfg, models, solver,
        label=REWARD, robot=ROBOT, model=MODEL,
        epochs=EPOCHS, save_every=SAVE_EVERY,
        episode_length_train=EPISODE_LENGTH_TRAIN,
        initial_episode_length=INITIAL_EPISODE_LENGTH,
        fall_z_threshold=FALL_Z_THRESHOLD,
        growth_factor=GROWTH_FACTOR,
        action_filter_alpha=0.1,                       # load-bearing at eval
        reward_scaling="total / current_length * 100",
        obs_dim=obs_dim, action_dim=action_dim,
        timestamp=current_time,
        seed=SEED,
    )

    # Readable sidecar you can `cat` on the cluster without unpickling / Isaac.
    with open(os.path.join(custom_log_dir, "run_cfg.json"), "w") as f:
        json.dump(run_cfg, f, indent=2, default=str)

    # Logging arrays
    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)

    # --- 3. Main Training Loop ---
    print("\n[INFO] Starting Evolution Strategy Training...")

    current_length = INITIAL_EPISODE_LENGTH  # dynamic episode length

    reward_list = []

    for epoch in range(EPOCHS):

        PLOT_OUTPUT = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/logs/rewards/_{current_time}_rewards_epoch_{epoch}.png"
        # A. Ask: get new population of Hebbian coefficients from ES
        solutions = solver.ask()

        # B. Distribute: load evolved coefficients into the parallel networks
        models.set_models_params(solutions)

        # C. Reset the LSTM hidden/cell state for this rollout.
        #    reset_weights() is the interface-compatible name; for the LSTM it
        #    re-initialises h/c to small random values, so the recurrent state
        #    starts fresh each generation rather than carrying over.
        models.reset_weights()

        # D. Reset the environment
        obs, _ = env.reset()

        # Per-env fall tracking (step index at which base z first dropped below threshold)
        robot = env.scene["robot"]
        initial_xy = robot.data.root_pos_w[:, :2].clone()
        fall_step = torch.full((POPSIZE,), current_length, dtype=torch.long, device=env.device)
        fallen_mask = torch.zeros(POPSIZE, dtype=torch.bool, device=env.device)

        # Track rewards and termination per individual
        total_rewards = torch.zeros(POPSIZE, device=env.device)
        done_mask = torch.ones(POPSIZE, device=env.device, dtype=torch.bool)

        # --- NEW: Initialize tracking metrics for this epoch ---
        survival_steps = torch.zeros(POPSIZE, device=env.device)
        total_velocity = torch.zeros(POPSIZE, device=env.device)
        action_saturation_count = 0
        total_action_elements = 0

        # Per-timestep raw physical quantities (accumulated over the rollout).
        # These replace the old gecko reward-term sums; they feed the RawState/*
        # diagnostics (Vx / YawRate / Z / Z_range) used to separate walking from
        # the ballistic/porpoising regime in the (Vx_mean, Z_range_mean) plane.
        vx_sum  = torch.zeros(POPSIZE, device=env.device)   # forward velocity v_x
        yaw_sum = torch.zeros(POPSIZE, device=env.device)   # yaw rate w_z
        z_sum   = torch.zeros(POPSIZE, device=env.device)   # base height z
        z_min   = torch.full((POPSIZE,), float("inf"),  device=env.device)
        z_max   = torch.full((POPSIZE,), float("-inf"), device=env.device)

        # Save the old mean parameter vector to calculate step size later[cite: 5]
        old_mu = solver.mu.copy()

        tracked_step_rewards = torch.zeros(current_length)

        prev_actions = None
        # Initialize the previous-action buffer the observation reads from.
        env.prev_action_buf = torch.zeros(POPSIZE, action_dim, device=env.device)
        env.prev_prev_action_buf = torch.zeros(POPSIZE, action_dim, device=env.device)

        # E. Rollout: evaluate the population
        for step in range(current_length):
            policy_obs = obs["policy"]

            # Forward pass (advances the LSTM hidden/cell state internally)
            actions = models.forward(policy_obs)

            # Low-pass action filter (paper: 0.1*new + 0.9*prev)
            if prev_actions is not None:
                actions = 0.1 * actions + 0.9 * prev_actions

            # Stash the FILTERED action so the next observation can feed it back
            # (the obs built inside env.step() will read env.prev_action_buf).
            env.prev_prev_action_buf = getattr(
                env, "prev_action_buf",
                torch.zeros(POPSIZE, action_dim, device=env.device),
            )
            env.prev_action_buf = actions.detach()

            # Environment step
            obs, rewards, terminates, truncates, extras = env.step(actions)

            prev_actions = actions

            tracked_step_rewards[step] = rewards[0].item()

            survival_steps += done_mask.float()
            
            # Track first-fall step per env (z below threshold)
            z = robot.data.root_pos_w[:, 2]
            just_fell = (z < FALL_Z_THRESHOLD) & ~fallen_mask
            fall_step[just_fell] = step
            fallen_mask = fallen_mask | just_fell

            # 2. Track real forward X-velocity[cite: 4]
            # We only record velocity for agents that are currently "alive" (done_mask)
            robot = env.scene["robot"]
            total_velocity += robot.data.root_lin_vel_b[:, 0] * done_mask.float()
            
            # 3. Track action saturation (what percentage of outputs are pegged at the extremes)
            saturated = (torch.abs(actions) > 0.95).sum().item()
            action_saturation_count += saturated
            total_action_elements += actions.numel()

            # Accumulate raw physical quantities this timestep.
            # z = robot.data.root_pos_w[:, 2] is already computed above.
            vx_sum  += robot.data.root_lin_vel_b[:, 0]
            yaw_sum += robot.data.root_ang_vel_b[:, 2]
            z_sum   += z
            z_min = torch.minimum(z_min, z)
            z_max = torch.maximum(z_max, z)

            # Only accumulate rewards for individuals that haven't terminated.
            # IsaacLab auto-resets terminated envs, which would mix rewards
            # from separate episodes and corrupt the fitness signal.
            total_rewards += rewards * done_mask.float()
            # done_mask = done_mask & ~terminates & ~truncates

        # F. Scale rewards to match the original es_train.py convention
        total_rewards = total_rewards / current_length * 100

        # Update episode length for next epoch: last_fall * 1.5, monotonically non-decreasing, capped at max
        last_fall = fall_step.max().item()
        new_length = min(EPISODE_LENGTH_TRAIN, int(last_fall * GROWTH_FACTOR))
        current_length = max(current_length, new_length)


        # G. Tell: pass fitness back to the ES optimizer
        total_rewards_cpu = total_rewards.cpu().numpy()
        fitlist = list(total_rewards_cpu)
        solver.tell(fitlist)

        # --- 4. Logging & Saving ---
        fit_arr = np.array(fitlist)
        mean_reward = fit_arr.mean()
        best_reward = fit_arr.max()

        pop_mean_curve[epoch] = mean_reward
        best_sol_curve[epoch] = best_reward

        # --- NEW: Calculate Final Metrics ---
        # Physical
        avg_survival = survival_steps.mean().item()
        avg_velocity = (total_velocity / (survival_steps + 1e-5)).mean().item() 
        
        # Internals
        saturation_pct = (action_saturation_count / total_action_elements) * 100.0

        # LSTM recurrent-state health (analogue of the Hebbian dynamic-weight
        # internals). get_states() returns (h, c), each (popsize, 3*hidden), as
        # left at the end of the rollout — watch these for blow-up / saturation.
        h_state, c_state = models.get_states()

        # ES Optimizer Health[cite: 5]
        reward_std = fit_arr.std()
        step_size = np.linalg.norm(solver.mu - old_mu)

        # --- NEW: Log everything to TensorBoard ---
        writer.add_scalar("Fitness/Avg_Survival_Steps", avg_survival, epoch)
        writer.add_scalar("Fitness/Avg_Forward_Velocity_m_s", avg_velocity, epoch)
        writer.add_scalar("Fitness/Action_Saturation_Pct", saturation_pct, epoch)

        writer.add_scalar("Internals/Max_Abs_Hidden_State",  h_state.abs().max().item(),  epoch)
        writer.add_scalar("Internals/Max_Abs_Cell_State",    c_state.abs().max().item(),  epoch)
        writer.add_scalar("Internals/Mean_Abs_Hidden_State", h_state.abs().mean().item(), epoch)
        
        writer.add_scalar("Rewards/Reward_Std_Dev", reward_std, epoch)
        writer.add_scalar("Rewards/Mu_Step_Size", step_size, epoch)
        writer.add_scalar("Rewards/Mean_Reward", mean_reward, epoch)
        writer.add_scalar("Rewards/Best_Reward", best_reward, epoch)
        # Log the full array as a histogram to see the distribution curve
        writer.add_histogram("Rewards/Reward_Distribution", fit_arr, epoch)
        # It's also helpful to track the ES sigma (exploration variance)
        writer.add_scalar("Parameters/Sigma", solver.sigma, epoch)
        writer.add_scalar("Parameters/Learning_Rate", solver.learning_rate, epoch)
        writer.add_scalar("Curriculum/Episode_Length", current_length, epoch)
        writer.add_scalar("Curriculum/Last_Fall_Step", last_fall, epoch)

        # --- Distance travelled (XY displacement) ---
        final_xy = robot.data.root_pos_w[:, :2]
        distance = torch.norm(final_xy - initial_xy, dim=1).cpu().numpy()
        writer.add_histogram("Position/Distance_Distribution", distance, epoch)
        writer.add_scalar("Position/Avg_Distance", distance.mean(), epoch)

        # --- Raw physical-quantity breakdown (per-timestep averages over the episode) ---
        # These are the diagnostics for the (Vx_mean, Z_range_mean) plane: a walk sits
        # at high Vx / low Z_range (~0.05-0.10); the porpoising bounce at high Z_range (~0.25).
        vx_avg  = vx_sum  / current_length          # mean v_x       [m/s]
        yaw_avg = yaw_sum / current_length          # mean w_z       [rad/s]
        z_avg   = z_sum   / current_length          # mean height z  [m]
        z_range = z_max - z_min                     # vertical excursion per env [m]

        writer.add_scalar("RawState/Vx_mean",      vx_avg.mean().item(),  epoch)
        writer.add_scalar("RawState/YawRate_mean", yaw_avg.mean().item(), epoch)
        writer.add_scalar("RawState/Z_mean",       z_avg.mean().item(),   epoch)
        writer.add_scalar("RawState/Z_range_mean", z_range.mean().item(), epoch)

        writer.add_histogram("RawState/Vx_dist",      vx_avg.cpu().numpy(),  epoch)
        writer.add_histogram("RawState/YawRate_dist", yaw_avg.cpu().numpy(), epoch)
        writer.add_histogram("RawState/Z_range_dist", z_range.cpu().numpy(), epoch)

        print(f"Epoch {epoch:03d} | Mean: {mean_reward:>8.2f} | Best: {best_reward:>8.2f} | Sigma: {solver.sigma:.4f} | LR: {solver.learning_rate:.6f}")

        if (epoch + 1) % SAVE_EVERY == 0 and avg_velocity > 0.55:
            save_path = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/{run_name}_{epoch}.pickle"
            print(f"  -> Saving checkpoint to {save_path}")
            with open(save_path, 'wb') as f:
                pickle.dump((
                    solver,
                    copy.deepcopy(models),
                    pop_mean_curve,
                    best_sol_curve,
                    run_cfg,
                ), f)
    
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()