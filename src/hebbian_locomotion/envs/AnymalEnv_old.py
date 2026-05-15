import torch
import hydra
from omegaconf import DictConfig
import os
from PIL import Image

# Isaac Lab Imports
from isaaclab.app import AppLauncher
#app_launcher = AppLauncher(headless=True, livestream=1)
#simulation_app = app_launcher.app

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip

# Import your custom assets and network
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # add terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # add robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=900.0,
            texture_file=f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
            visible_in_primary_ray=False,
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


def constant_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        
            Need to implement foot contact into observations 
        """

        # observation terms (order preserved)
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        #base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        #velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # actions = ObsTerm(func=mdp.last_action)
        '''height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )'''

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EmptyManagerCfg:
    """Empty manager specifications for the environment."""

    pass

# 1. Forward Velocity in World X-axis
def forward_velocity_x(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # Extract the x-axis component of the world-frame linear velocity
    return asset.data.root_lin_vel_w[:, 0]

# 2. Upright Posture Reward
def upright_posture(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # The projection of local Z onto world Z is the negative of the Z-component 
    # of the projected gravity vector.
    z_proj = -asset.data.projected_gravity_b[:, 2]
    
    # Return 0.0 if z_proj > 0.93, else -0.5
    return torch.where(z_proj > 0.93, 0.0, -0.5)

# 3. Heading Yaw Reward
def heading_yaw(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # Convert the world-frame root quaternion into Euler angles
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    
    # Return 0.0 if abs(yaw) < 0.45, else -0.5
    return torch.where(torch.abs(yaw) < 0.45, 0.0, -0.5)

@configclass
class RewardsCfg:
    """Reward specifications matching the Hebbian Locomotion paper."""
    
    # Weight coefficients: kv=2.0, ku=0.5, ky=0.5
    V_t = RewTerm(
        func=forward_velocity_x, 
        weight=2.0, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    U_t = RewTerm(
        func=upright_posture, 
        weight=0.5, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    Yaw_t = RewTerm(
        func=heading_yaw, 
        weight=0.5, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

@configclass
class AnymalEnvCfg(ManagerBasedRLEnvCfg):
    """The Master Configuration"""
    # 1. Scene Setup
    scene = MySceneCfg(num_envs=64, env_spacing=2.5)

    # 2. Managers
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: EmptyManagerCfg = EmptyManagerCfg()
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

def main():
    print("[INFO] Initializing Environment for Streaming...")
    env_cfg = AnymalEnvCfg()
    
    # 2. REMOVE "rgb_array". Standard mode is fine for streaming.
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # ... (Your network setup code remains the same) ...
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = 12
    net = HebbianNet(popsize=env.num_envs, sizes=[obs_dim, 64, 32, action_dim], norm_mode='max')
    
    obs, _ = env.reset() 
    count = 0


    # 3. INFINITE LOOP (Press Ctrl+C in terminal to stop)
    while simulation_app.is_running():  
        policy_obs = obs["policy"]      
        action = net.forward(policy_obs)
        obs, _, _, _, _  = env.step(action)
        
        # We don't need env.render() here, the extension handles it automatically
        # But keeping it doesn't hurt.
        
        count += 1
        if count % 1000 == 0:
             print(f"[INFO] Simulating step {count}")

    env.close()
