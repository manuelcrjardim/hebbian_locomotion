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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.actuators import ImplicitActuatorCfg

# Import your custom assets and network
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet

# ------------------------------------------------------------------
# Custom Robot Asset Configuration
# ------------------------------------------------------------------
GECKO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/robots/slalom_fixedbody_16dof.usd", # <-- UPDATE THIS TO YOUR ACTUAL PATH
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Adjust Z-height so it doesn't spawn inside the floor
        joint_pos={".*": 0.0}, # Initializes all 16 joints to 0
    ),
        actuators={
            "all_motors": ImplicitActuatorCfg(
                joint_names_expr=[".*"], 
                stiffness=1.0,     # Mapped from set_drive
                damping=0.0,       # Mapped from set_drive
                effort_limit=4.1,  # Mapped from set_drive max_force
            ),
        },

)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene configuration with terrain, robot, and sensors."""

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
    robot: ArticulationCfg = GECKO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*foot.*", 
        update_period=0.0,      
        history_length=1,       
        debug_vis=False,
    )

    # height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robot_base", # Using robot_base based on slalom.py joint paths
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

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
    )


def constant_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


# ------------------------------------------------------------------
# Custom observation: binary foot contact
# ------------------------------------------------------------------

def body_euler_angles(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return body orientation as (roll, pitch, yaw) Euler angles in [-pi, pi].

    Matches the paper's Table I: body orientation in radians.
    Yaw is in the world frame, so it gives the network access to absolute
    heading — critical for learning to move in +X direction.
    """
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    # Wrap to [-pi, pi] (euler_xyz_from_quat returns values in [0, 2*pi])
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.stack([roll, pitch, yaw], dim=-1)

def forward_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body-frame linear velocity (vx, vy, vz)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def up_projection(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Upright projection (1 = upright, 0 = sideways). Code obs index 54."""
    asset = env.scene[asset_cfg.name]
    return (-asset.data.projected_gravity_b[:, 2]).unsqueeze(-1)


def prev_action_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the previous (filtered) action fed back as observation.

    The paper feeds back self.actions (the filtered, pre-scale [-1, 1] policy
    output). Because the low-pass filter lives in the training loop, we can't
    use mdp.last_action (which would return the scaled effort). Instead the
    training loop stashes the filtered action on env.prev_action_buf each step,
    and this term reads it back.

    Falls back to zeros on the first step / before the buffer exists.
    """
    buf = getattr(env, "prev_action_buf", None)
    if buf is None:
        return torch.zeros(env.num_envs, 16, device=env.device)
    return buf

def foot_contact_binary(env: ManagerBasedRLEnv,
                        sensor_cfg: SceneEntityCfg,
                        threshold: float = 1.0) -> torch.Tensor:
    """Return binary foot contact signals (0 or 1) for each foot.

    Matches the paper's foot contact observation (Table I):
        0 = no contact, 1 = leg is in contact.

    The contact sensor reports net normal forces in world frame with shape
    (num_envs, num_bodies, 3). We take the norm across the xyz force
    components and threshold it.

    Args:
        env:        The environment instance.
        sensor_cfg: SceneEntityCfg pointing to the contact sensor.
        threshold:  Force magnitude (N) above which contact is registered.

    Returns:
        Tensor of shape (num_envs, num_feet) with values 0.0 or 1.0.
        For ANYmal C this is (num_envs, 4).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w has shape (num_envs, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w
    # Compute force magnitude per foot: (num_envs, num_bodies)
    force_magnitude = torch.norm(net_forces, dim=-1)
    # Threshold to binary
    binary_contact = (force_magnitude > threshold).float()
    return binary_contact

# ------------------------------------------------------------------
# Reward functions (HAN-style Gaussian tracking)
# ------------------------------------------------------------------

def track_lin_vel_gaussian(
    env,
    asset_cfg: SceneEntityCfg,
    target_speed: float = 1.0,
    std: float = 0.5,
) -> torch.Tensor:
    """Gaussian forward-velocity tracking reward (HAN Fig. 10, left).

    r = exp(-(v_x - v_target)^2 / std^2), peaked at 1.0 when v_x == target.
    v_x is body-frame forward velocity. `std` is the tracking bandwidth
    (MuJoCo Playground Joystick default: std^2 = 0.25 -> std = 0.5).
    """
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_b[:, 0]
    err = vx - target_speed
    return torch.exp(-torch.square(err) / (std ** 2))


def track_zero_yaw_rate_gaussian(
    env,
    asset_cfg: SceneEntityCfg,
    std: float = 0.5,
) -> torch.Tensor:
    """Gaussian zero-yaw-rate tracking reward (HAN Fig. 10, right).

    r = exp(-(w_z - 0)^2 / std^2). Uses body-frame yaw angular velocity,
    matching the paper's 'yaw rate (rad/s)' axis rather than yaw angle.
    """
    asset = env.scene[asset_cfg.name]
    wz = asset.data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(wz) / (std ** 2))

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.

        Matches the ACTUAL slalom code observation (locomotion_simple_rew.py),
        NOT the paper's Table I. Layout (55 dims), in code order:
            - joint_pos:    16   code idx 0-15
            - joint_vel:    16   code idx 16-31
            - roll/pitch/yaw: 3  code idx 32-34
            - last_action:  16   code idx 35-50  (previous FILTERED action)
            - body_lin_vel:  3   code idx 51-53
            - up_proj:       1   code idx 54
        No foot contacts (the code's active obs omits them).
        """

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        body_orientation = ObsTerm(
            func=body_euler_angles,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        last_action = ObsTerm(
            func=prev_action_obs,
        )
        body_lin_vel = ObsTerm(
            func=forward_velocity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        up_proj = ObsTerm(
            func=up_projection,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EmptyManagerCfg:
    """Empty manager specifications for the environment."""

    pass


# ------------------------------------------------------------------
# Reward functions
# ------------------------------------------------------------------

def forward_velocity_x(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Forward velocity in robot-frame X-axis."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0]


def upright_posture(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Upright posture reward (Eq. 6 in the paper)."""
    asset = env.scene[asset_cfg.name]
    z_proj = -asset.data.projected_gravity_b[:, 2]
    return torch.where(z_proj > 0.93, 0.0, -0.5)


def heading_yaw(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Heading yaw reward (Eq. 7 in the paper)."""
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.where(torch.abs(yaw) < 0.45, 0.0, -0.5)

def yaw_gaussian(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    target = 0
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return

def speed_gaussian(env, asset_cfg: SceneEntityCfg, target=1) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    pass

@configclass
class RewardsCfg2:
    """Gaussian tracking rewards matching the HAN paper (Fig. 10)."""

    track_vel = RewTerm(
        func=track_lin_vel_gaussian,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_speed": 1.0,   # <-- HAN target: sweep {1.0, 1.5, 2.0}
            "std": 0.5,
        },
    )
    track_yaw = RewTerm(
        func=track_zero_yaw_rate_gaussian,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.5},
    )

@configclass
class RewardsCfg:
    """Reward specifications matching the Hebbian Locomotion paper (Eq. 5)."""

    V_t = RewTerm(
        func=forward_velocity_x,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    U_t = RewTerm(
        func=upright_posture,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    Yaw_t = RewTerm(
        func=heading_yaw,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class GeckoEnvCfg(ManagerBasedRLEnvCfg):
    """The Master Configuration."""

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
        self.target_speed = self.rewards.track_vel.params["target_speed"]
        # simulation settings
        self.sim.dt = 0.005
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt