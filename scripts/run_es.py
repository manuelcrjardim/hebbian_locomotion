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
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet
from hebbian_locomotion.networks.ES_classes import OpenES


current_time = datetime.now().strftime("%m:%d-%H:%M")

def main():

    run_name = f"hebbian_run_lr_{current_time}"

    base_dir = "/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/tensorboard"
    custom_log_dir = os.path.join(base_dir, run_name)

    writer = SummaryWriter(log_dir=custom_log_dir)
    # --- 1. Hyperparameters ---
    EPOCHS = 500                    # Total generations
    EPISODE_LENGTH_TRAIN = 500      # Simulation steps per rollout
    POPSIZE = 1024                   # Population size (paper uses 1024)
    SAVE_EVERY = 25

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

    reward_list = []

    for epoch in range(EPOCHS):

        PLOT_OUTPUT = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/logs/rewards/_{current_time}_rewards_epoch_{epoch}.png"
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

        # Capture starting XY position for distance-travelled logging
        robot = env.scene["robot"]
        initial_xy = robot.data.root_pos_w[:, :2].clone()

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

        tracked_step_rewards = torch.zeros(EPISODE_LENGTH_TRAIN)

        # E. Rollout: evaluate the population
        for step in range(EPISODE_LENGTH_TRAIN):
            policy_obs = obs["policy"]

            # Forward pass (updates Hebbian weights internally)
            actions = models.forward(policy_obs)

            # Environment step
            obs, rewards, terminates, truncates, extras = env.step(actions)

            tracked_step_rewards[step] = rewards[0].item()

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

        # FIX 3: Correct Matplotlib plotting logic
        if epoch % 5 == 0:
            plt.figure(figsize=(8, 6))
            # Ensure tensor is moved to CPU and converted to numpy for plotting
            plt.plot(tracked_step_rewards.cpu().numpy(), label="Agent 0 Reward")
            plt.title(f"Step Rewards - Epoch {epoch}")
            plt.xlabel("Simulation Step")
            plt.ylabel("Reward")
            plt.legend()
            # plt.axis("equal") <- REMOVED: This forces X and Y axes to share the same scale, 
            # which ruins line graphs where X is 500 steps and Y is small reward numbers.
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(PLOT_OUTPUT, dpi=150)
            plt.close() # <-- ADDED: Crucial to prevent a memory leak by closing the figure!
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

        # --- NEW: Calculate Final Metrics ---
        # Physical
        avg_survival = survival_steps.mean().item()
        avg_velocity = (total_velocity / (survival_steps + 1e-5)).mean().item() 
        
        # Internals
        saturation_pct = (action_saturation_count / total_action_elements) * 100.0
        max_weight = max([torch.max(torch.abs(w)).item() for w in models.get_weights()])
 
        A_vals = torch.cat([a.flatten() for a in models.A]).cpu().numpy()
        B_vals = torch.cat([b.flatten() for b in models.B]).cpu().numpy()
        C_vals = torch.cat([c.flatten() for c in models.C]).cpu().numpy()
        D_vals = torch.cat([d.flatten() for d in models.D]).cpu().numpy()
        lr_vals = torch.cat([lr.flatten() for lr in models.lr]).cpu().numpy()
        
        # ES Optimizer Health[cite: 5]
        reward_std = fit_arr.std()
        step_size = np.linalg.norm(solver.mu - old_mu)

        # --- NEW: Log everything to TensorBoard ---
        writer.add_scalar("Fitness/Avg_Survival_Steps", avg_survival, epoch)
        writer.add_scalar("Fitness/Avg_Forward_Velocity_m_s", avg_velocity, epoch)
        writer.add_scalar("Fitness/Max_Dynamic_Weight", max_weight, epoch)
        writer.add_scalar("Fitness/Action_Saturation_Pct", saturation_pct, epoch)
        
        writer.add_histogram("Internals/Values of A", A_vals, epoch)
        writer.add_histogram("Internals/Values of B", B_vals, epoch)
        writer.add_histogram("Internals/Values of C", C_vals, epoch)
        writer.add_histogram("Internals/Values of D", D_vals, epoch)
        writer.add_histogram("Internals/Learning_Rates", lr_vals, epoch)
        
        writer.add_scalar("Rewards/Reward_Std_Dev", reward_std, epoch)
        writer.add_scalar("Rewards/Mu_Step_Size", step_size, epoch)
        writer.add_scalar("Rewards/Mean_Reward", mean_reward, epoch)
        writer.add_scalar("Rewards/Best_Reward", best_reward, epoch)
        # Log the full array as a histogram to see the distribution curve
        writer.add_histogram("Rewards/Reward_Distribution", fit_arr, epoch)
        # It's also helpful to track the ES sigma (exploration variance)
        writer.add_scalar("Parameters/Sigma", solver.sigma, epoch)
        writer.add_scalar("Parameters/Learning_Rate", solver.learning_rate, epoch)

        # --- Distance travelled (XY displacement) ---
        final_xy = robot.data.root_pos_w[:, :2]
        distance = torch.norm(final_xy - initial_xy, dim=1).cpu().numpy()
        writer.add_histogram("Position/Distance_Distribution", distance, epoch)
        writer.add_scalar("Position/Avg_Distance", distance.mean(), epoch)

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