import os
import torch
import numpy as np
import pickle
import copy
import cv2

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
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet
from hebbian_locomotion.networks.ES_classes import OpenES


current_time = datetime.now().strftime("%m\%d-%H:%M")

def main():

    run_name = f"airl_run_lr_{current_time}"

    base_dir = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/tensorboard"
    custom_log_dir = os.path.join(base_dir, run_name)

    writer = SummaryWriter(log_dir=custom_log_dir)
    # --- 1. Hyperparameters ---
    EPOCHS = 500                    # Total generations
    EPISODE_LENGTH_TRAIN = 500      # Simulation steps per rollout
    POPSIZE = 1024                    # Population size (paper uses 1024)
    SAVE_EVERY = 5

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

    print(f"\n[INFO] Initializing Hebbian Network (obs={obs_dim}, act={action_dim})...")
    models = HebbianNet(
        popsize=POPSIZE,
        sizes=[obs_dim, 64, 32, action_dim],
        norm_mode='max'
    )

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

    for epoch in range(EPOCHS):
        # A. Ask: get new population of Hebbian coefficients from ES
        solutions = solver.ask()

        # B. Distribute: load evolved coefficients into the parallel networks
        models.set_models_params(solutions)

        # C. Reset connection weights to fresh random values for this rollout.
        #    This is critical — the Hebbian rules must transform a new set of
        #    weights each generation, not carry over from the previous one.
        models.reset_weights()

        # D. Reset the environment
        obs, _ = env.reset()

        # Track rewards and termination per individual
        total_rewards = torch.zeros(POPSIZE, device=env.device)
        done_mask = torch.ones(POPSIZE, device=env.device, dtype=torch.bool)

        # --- NEW: Initialize tracking metrics for this epoch ---
        survival_steps = torch.zeros(POPSIZE, device=env.device)
        total_velocity = torch.zeros(POPSIZE, device=env.device)
        action_saturation_count = 0
        total_action_elements = 0
        
        # Save the old mean parameter vector to calculate step size later[cite: 5]
        old_mu = solver.mu.copy()

        # E. Rollout: evaluate the population
        for step in range(EPISODE_LENGTH_TRAIN):
            policy_obs = obs["policy"]

            # Forward pass (updates Hebbian weights internally)
            actions = models.forward(policy_obs)

            # Environment step
            obs, rewards, terminates, truncates, extras = env.step(actions)

            # --- NEW: Track Physical & Network Metrics ---
            # 1. Track survival (only add a step if the agent hasn't terminated)
            survival_steps += done_mask.float()
            
            # 2. Track real forward X-velocity[cite: 4]
            # We only record velocity for agents that are currently "alive" (done_mask)
            robot = env.scene["robot"]
            total_velocity += robot.data.root_lin_vel_w[:, 0] * done_mask.float()
            
            # 3. Track action saturation (what percentage of outputs are pegged at the extremes)
            saturated = (torch.abs(actions) > 0.95).sum().item()
            action_saturation_count += saturated
            total_action_elements += actions.numel()

            # Only accumulate rewards for individuals that haven't terminated.
            # IsaacLab auto-resets terminated envs, which would mix rewards
            # from separate episodes and corrupt the fitness signal.
            total_rewards += rewards * done_mask.float()
            done_mask = done_mask & ~terminates & ~truncates

        # F. Scale rewards to match the original es_train.py convention
        total_rewards = total_rewards / EPISODE_LENGTH_TRAIN * 100

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
        # Avoid division by zero by adding 1e-5
        avg_velocity = (total_velocity / (survival_steps + 1e-5)).mean().item() 
        
        # Internals
        saturation_pct = (action_saturation_count / total_action_elements) * 100.0
        max_weight = max([torch.max(torch.abs(w)).item() for w in models.get_weights()])
        
        # Flatten Hebbian coefficients to log their distributions[cite: 3]
        A_vals = torch.cat([a.flatten() for a in models.A]).cpu().numpy()
        lr_vals = torch.cat([lr.flatten() for lr in models.lr]).cpu().numpy()
        
        # ES Optimizer Health[cite: 5]
        reward_std = fit_arr.std()
        step_size = np.linalg.norm(solver.mu - old_mu)

        # --- NEW: Log everything to TensorBoard ---
        writer.add_scalar("Hebbian/Avg_Survival_Steps", avg_survival, epoch)
        writer.add_scalar("Hebbian/Avg_Forward_Velocity_m_s", avg_velocity, epoch)
        
        writer.add_scalar("Hebbian/Max_Dynamic_Weight", max_weight, epoch)
        writer.add_scalar("Hebbian/Action_Saturation_Pct", saturation_pct, epoch)
        
        writer.add_histogram("Hebbian/A_Correlation_Term", A_vals, epoch)
        writer.add_histogram("Hebbian/Learning_Rates", lr_vals, epoch)
        
        writer.add_scalar("Hebbian/Reward_Std_Dev", reward_std, epoch)
        writer.add_scalar("Hebbian/Mu_Step_Size", step_size, epoch)

        writer.add_scalar("Hebbian/Mean_Reward", mean_reward, epoch)
        writer.add_scalar("Hebbian/Best_Reward", best_reward, epoch)
        # Log the full array as a histogram to see the distribution curve
        writer.add_histogram("Hebbian/Reward_Distribution", fit_arr, epoch)
        # It's also helpful to track the ES sigma (exploration variance)
        writer.add_scalar("Hebbian/Sigma", solver.sigma, epoch)
        writer.add_scalar("Hebbian/Learning_Rate", solver.learning_rate, epoch)

        print(f"Epoch {epoch:03d} | Mean: {mean_reward:>8.2f} | Best: {best_reward:>8.2f} | Sigma: {solver.sigma:.4f} | LR: {solver.learning_rate:.6f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            save_path = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/{ROBOT}_hebbian_es_checkpoint_{current_time}_{epoch}.pickle"
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