import pickle
import torch

# 1. Launch Isaac Sim (Must be first, headless=False to see the simulation)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras = True)
simulation_app = app_launcher.app

import os
from PIL import Image
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import omni.replicator.core as rep
from isaaclab.envs import ManagerBasedRLEnv
from hebbian_locomotion.envs.AnymalEnv import AnymalEnvCfg 
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet


file_path = '/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/hebbian_es_checkpoint_100.pickle'

with open(file_path, 'rb') as file:
    solver, trained_models, pop_mean_curve, best_sol_curve = pickle.load(file)

best_model = solver.best_mu

def main():
    print("[INFO] Initializing Environment for 1 Agent...")
    env_cfg = AnymalEnvCfg()
    env_cfg.scene.num_envs = 1  # We only need 1 environment for testing

    env_cfg.scene.camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/GlobalCamera",
    update_period=0,       # 0 means capture every simulation step
    height=480,            # Image height in pixels
    width=640,             # Image width in pixels
    data_types=["rgb"],    # Request RGB data
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, 
        focus_distance=400.0, 
        horizontal_aperture=20.955, 
        clipping_range=(0.1, 1.0e5)
    ),
    # Position the camera 2 meters behind and 1.5 meters above the origin
    offset=CameraCfg.OffsetCfg(pos=(-2.0, 0.0, 1.5), rot=(0.9238, 0.0, 0.3826, 0.0), convention="ros"),)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # --- 3. Initialize Model and Apply Weights ---
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = 12
    
    test_model = HebbianNet(
        popsize=1, # Match the test environment size
        sizes=[obs_dim, 64, 32, action_dim], 
        norm_mode='max' 
    )
    
    # Apply the best parameters to our test model
    test_model.set_a_model_params(best_model)
    
    # --- 4. Test Rollout ---
    print("\n[INFO] Starting Test Rollout...")
    obs, _ = env.reset()
    
    # Run a test loop (e.g., 1000 steps)
    for step in range(1000):

        if step%50 == 0:

            print(f'currently on step {step}')

        policy_obs = obs["policy"]
        
        
        # Forward pass using the loaded best weights
        actions = test_model.forward(policy_obs)
        
        # Step the environment
        obs, rewards, terminates, truncates, extras = env.step(actions)

    print("[INFO] Test complete.")
    env.close()

if __name__ == "__main__":
    # Point this to your saved pickle file
    main()
    simulation_app.close()