"""
test_standing.py — Spawn the robot with zero actions and record its Z coordinate.

Reveals whether the actuators can hold the body up against gravity.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from hebbian_locomotion.envs.GeckoEnv import GeckoEnvCfg

# ── Config ──
STEPS = 300
PLOT_OUTPUT = "standing_test.png"


def main():
    env_cfg = GeckoEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    action_dim = 16
    zero_actions = torch.zeros(1, action_dim, device=env.device)

    obs, _ = env.reset()
    zs = []

    for step in range(STEPS):
        obs, _, _, _, _ = env.step(zero_actions)
        zs.append(robot.data.root_pos_w[0, 2].item())

    env.close()

    zs = np.array(zs)
    settled = zs[-50:].mean()

    print(f"\n  Initial z:        {zs[0]:.4f} m")
    print(f"  Minimum z:        {zs.min():.4f} m  (step {zs.argmin()})")
    print(f"  Final z:          {zs[-1]:.4f} m")
    print(f"  Settled z (last 50 steps avg): {settled:.4f} m\n")

    plt.figure(figsize=(8, 5))
    plt.plot(zs, linewidth=2)
    plt.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Base z (m)")
    plt.title("Robot Base Height with Zero Actions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=150)
    print(f"Saved plot to {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()
    simulation_app.close()