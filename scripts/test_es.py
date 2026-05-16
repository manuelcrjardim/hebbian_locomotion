import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Launch Isaac Sim ──
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from hebbian_locomotion.envs.GeckoEnv import GeckoEnvCfg
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet
from datetime import datetime


# ── Config ──
EPOCH = 55
STEPS = 500
current_time = datetime.now().strftime("%m:%d_%H:%M")
CHECKPOINT_PATH = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/Gecko_hebbian_es_checkpoint_05:16-17:20_224.pickle"
PLOT_OUTPUT = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/logs/positions/trajectory_{STEPS}_steps_{current_time}_epoch_{EPOCH}.png"
PLOT_OUTPUT2 = f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/logs/positions/trajectory_{STEPS}_steps_over_time_{current_time}_epoch_{EPOCH}.png"

def main():
    # Load checkpoint
    with open(CHECKPOINT_PATH, "rb") as f:
        solver, models_ckpt, _, _ = pickle.load(f)

    # Create env (single agent)
    env_cfg = GeckoEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Build network and load best params
    model = HebbianNet(
        popsize=1,
        sizes=models_ckpt.architecture,
        norm_mode="max",
    )
    model.set_a_model_params(solver.mu)
    model.reset_weights()

    # Run rollout, record position
    obs, _ = env.reset()
    xs, ys, zs = [], [], []
    robot = env.scene["robot"]

    for step in range(STEPS):
        actions = model.forward(obs["policy"])
        obs, rewards, _, _, _ = env.step(actions)

        pos = robot.data.root_pos_w[0].cpu().numpy()
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])

    env.close()

    # Plot 1: X-Y Trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, linewidth=2)
    plt.plot(xs[0], ys[0], "go", markersize=10, label="Start")
    plt.plot(xs[-1], ys[-1], "rs", markersize=10, label="End")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Robot Trajectory")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150)
    print(f"Saved plot to {PLOT_OUTPUT}")

    # Plot 2: Position Over Time
    plt.figure(figsize=(8, 6))
    time_steps = range(STEPS) # Create an array representing time steps
    
    # Plot X, Y and Z over time
    plt.plot(time_steps, xs, label="X Position", linewidth=2)
    plt.plot(time_steps, ys, label="Y Position", linewidth=2)
    plt.plot(time_steps, zs, label="Z Position", linewidth=2)
    
    # Mark Start and End points
    plt.plot(time_steps[0], xs[0], "go", markersize=8, label="Start X")
    plt.plot(time_steps[-1], xs[-1], "rs", markersize=8, label="End X")
    
    plt.xlabel("Time Step")
    plt.ylabel("Position (m)")
    plt.title("Robot Position Over Time")
    plt.legend()
    # Removed plt.axis("equal") so time and position scales are independent
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT2, dpi=150)
    print(f"Saved plot to {PLOT_OUTPUT2}")

if __name__ == "__main__":
    main()
    simulation_app.close()