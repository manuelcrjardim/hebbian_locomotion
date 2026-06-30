"""
run_es_dynamic_lstm.py — LSTM-baseline sibling of run_es_dynamic.py.

Identical to run_es_dynamic.py (same env, reward, ES optimiser, EMA action
filter, dynamic-episode curriculum, checkpointing, and all generic TensorBoard
logs) EXCEPT the controller is an ES-trained LSTM and the Hebbian-specific
internals logs are replaced with LSTM-specific ones.

Logging changes vs. run_es_dynamic.py (everything else is unchanged):
  REMOVED (HNN-specific, would crash on the LSTM):
    - Fitness/Max_Dynamic_Weight        (max plastic Hebbian weight)
    - Internals/Values of A,B,C,D        (evolved Hebbian coefficients)
    - Internals/Learning_Rates           (evolved per-synapse lr)
  ADDED (LSTM-specific):
    - Internals/Max_Abs_Hidden_State, Max_Abs_Cell_State, Mean_Abs_Hidden_State
        (recurrent-state magnitude monitors — the analog of Max_Dynamic_Weight)
    - Internals/W_ih, W_hh, Gate_Biases, W_out, b_out   (evolved LSTM params)
    - Internals/Hidden_State_h, Cell_State_c            (plastic recurrent state)

Set HIDDEN to match the evolved-parameter count of the net you are comparing
against (the constructor prints the count). For I~23, O=16: H=60 -> ~21k
(near plain ABCD), H=64-66 -> ~24k (brackets the eligibility-trace ABCD variant).
"""

import os
import torch
import numpy as np
import pickle
import copy
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
from hebbian_locomotion.envs.GeckoEnv import GeckoEnvCfg
from hebbian_locomotion.networks.LSTM import LSTMNet
from hebbian_locomotion.networks.ES_classes import OpenES


current_time = datetime.now().strftime("%m:%d-%H:%M")

def main():

    run_name = f"lstm_run_lr_{current_time}"

    base_dir = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/tensorboard"
    custom_log_dir = os.path.join(base_dir, run_name)

    writer = SummaryWriter(log_dir=custom_log_dir)
    # --- 1. Hyperparameters ---
    EPOCHS = 500                    # Total generations
    EPISODE_LENGTH_TRAIN = 500      # MAX episode length (curriculum cap)
    INITIAL_EPISODE_LENGTH = 500    # Starting episode length
    FALL_Z_THRESHOLD = 0.1          # Robot considered fallen when base z < this
    GROWTH_FACTOR = 1.5             # next_length = last_fall_step * GROWTH_FACTOR
    POPSIZE = 1024                  # Population size (paper uses 1024)
    SAVE_EVERY = 50

    HIDDEN = 30      # was 64; the 3-stack net is ~12·h² in params, so h≈30 matches the ABCD nets

    ROBOT = 'Gecko'

    # ES parameters (paper: sigma_init=0.1, decay=0.999, lr=0.1, lr_decay=0.999)
    LEARNING_RATE = 0.1
    LEARNING_RATE_DECAY = 0.999
    SIGMA_INIT = 0.1
    SIGMA_DECAY = 0.999

    print("[INFO] Initializing Environment...")
    env_cfg = GeckoEnvCfg()
    env_cfg.scene.num_envs = POPSIZE
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # --- 2. Network & Optimizer Setup ---
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = 16

    print(f"\n[INFO] Initializing LSTM Network (obs={obs_dim}, hidden={HIDDEN}, act={action_dim})...")
    models = LSTMNet(popsize=POPSIZE, sizes=[obs_dim, HIDDEN, action_dim])

    models

    n_params = models.get_n_params_a_model()
    print(f"[INFO] Evolvable parameters per individual: {n_params}")

    print("\n[INFO] Initializing OpenES Optimizer...")
    solver = OpenES(
        n_params,
        popsize=POPSIZE,
        rank_fitness=True,
        antithetic=True,
        learning_rate=LEARNING_RATE,
        learning_rate_decay=LEARNING_RATE_DECAY,
        sigma_init=SIGMA_INIT,
        sigma_decay=SIGMA_DECAY
    )
    solver.set_mu(models.get_a_model_params())

    # Logging arrays
    pop_mean_curve = np.zeros(EPOCHS)
    best_sol_curve = np.zeros(EPOCHS)

    # --- 3. Main Training Loop ---
    print("\n[INFO] Starting Evolution Strategy Training...")

    current_length = INITIAL_EPISODE_LENGTH  # dynamic episode length

    reward_list = []

    for epoch in range(EPOCHS):

        PLOT_OUTPUT = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/logs/rewards/_{current_time}_rewards_epoch_{epoch}.png"
        # A. Ask: get new population of LSTM weights from ES
        solutions = solver.ask()

        # B. Distribute: load evolved LSTM weights into the parallel networks
        models.set_models_params(solutions)

        # C. Reset the LSTM hidden/cell state for this rollout.
        #    Analogous to resetting the Hebbian connection weights: the recurrent
        #    state must start fresh each generation, not carry over.
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

        # --- Initialize tracking metrics for this epoch ---
        survival_steps = torch.zeros(POPSIZE, device=env.device)
        total_velocity = torch.zeros(POPSIZE, device=env.device)
        action_saturation_count = 0
        total_action_elements = 0

        # Save the old mean parameter vector to calculate step size later
        old_mu = solver.mu.copy()

        tracked_step_rewards = torch.zeros(current_length)

        prev_actions = None

        # E. Rollout: evaluate the population
        for step in range(current_length):
            policy_obs = obs["policy"]

            # Forward pass (advances LSTM hidden/cell state internally)
            actions = models.forward(policy_obs)

            if prev_actions != None:
                actions = 0.1*actions + 0.9 * prev_actions

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

            # Track real forward X-velocity (body frame)
            # We only record velocity for agents that are currently "alive" (done_mask)
            robot = env.scene["robot"]
            total_velocity += robot.data.root_lin_vel_b[:, 0] * done_mask.float()

            # Track action saturation (what percentage of outputs are pegged at the extremes)
            saturated = (torch.abs(actions) > 0.95).sum().item()
            action_saturation_count += saturated
            total_action_elements += actions.numel()

            # Only accumulate rewards for individuals that haven't terminated.
            # IsaacLab auto-resets terminated envs, which would mix rewards
            # from separate episodes and corrupt the fitness signal.
            total_rewards += rewards * done_mask.float()
            done_mask = done_mask & ~terminates & ~truncates

        # F. Scale rewards to match the original es_train.py convention
        total_rewards = total_rewards / current_length * 100

        # Update episode length for next epoch: last_fall * 1.5, monotonically non-decreasing, capped at max
        last_fall = fall_step.max().item()
        new_length = min(EPISODE_LENGTH_TRAIN, int(last_fall * GROWTH_FACTOR))
        current_length = max(current_length, new_length)

        # Matplotlib plotting logic
        if epoch % 25 == 0:
            plt.figure(figsize=(8, 6))
            plt.plot(tracked_step_rewards.cpu().numpy(), label="Agent 0 Reward")
            plt.title(f"Step Rewards - Epoch {epoch}")
            plt.xlabel("Simulation Step")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(PLOT_OUTPUT, dpi=150)
            plt.close()
            print(f"Saved plot to {PLOT_OUTPUT}")

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

        # Physical
        avg_survival = survival_steps.mean().item()
        avg_velocity = (total_velocity / (survival_steps + 1e-5)).mean().item()

        # Internals (generic)
        saturation_pct = (action_saturation_count / total_action_elements) * 100.0


        # ES Optimizer Health
        reward_std = fit_arr.std()
        step_size = np.linalg.norm(solver.mu - old_mu)

        # --- Log everything to TensorBoard ---
        writer.add_scalar("Fitness/Avg_Survival_Steps", avg_survival, epoch)
        writer.add_scalar("Fitness/Avg_Forward_Velocity_m_s", avg_velocity, epoch)
        writer.add_scalar("Fitness/Action_Saturation_Pct", saturation_pct, epoch)

         # --- LSTM-specific internals (3-stack SeqLSTMs) ---
        h_state, c_state = models.get_states()          # (pop, 3*hidden)
        writer.add_scalar("Internals/Max_Abs_Hidden_State", h_state.abs().max().item(), epoch)
        writer.add_scalar("Internals/Max_Abs_Cell_State",  c_state.abs().max().item(), epoch)
        writer.add_scalar("Internals/Mean_Abs_Hidden_State", h_state.abs().mean().item(), epoch)

        for bi, blk in enumerate(models.get_weights(), start=1):
            gate_w = torch.cat([blk["Wf"].flatten(), blk["Wi"].flatten(),
                                blk["Wc"].flatten(), blk["Wo"].flatten()])
            writer.add_histogram(f"Internals/Block{bi}_GateWeights", gate_w.cpu().numpy(), epoch)
            writer.add_histogram(f"Internals/Block{bi}_Wout", blk["Wout"].flatten().cpu().numpy(), epoch)
        writer.add_histogram("Internals/Hidden_State_h", h_state.flatten().cpu().numpy(), epoch)
        writer.add_histogram("Internals/Cell_State_c", c_state.flatten().cpu().numpy(), epoch)

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

        print(f"Epoch {epoch:03d} | Mean: {mean_reward:>8.2f} | Best: {best_reward:>8.2f} | Sigma: {solver.sigma:.4f} | LR: {solver.learning_rate:.6f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            save_path = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/{ROBOT}_lstm_es_checkpoint_{current_time}_{epoch}.pickle"
            print(f"  -> Saving checkpoint to {save_path}")
            with open(save_path, 'wb') as f:
                pickle.dump((
                    solver,
                    copy.deepcopy(models),
                    pop_mean_curve,
                    best_sol_curve,
                ), f)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()