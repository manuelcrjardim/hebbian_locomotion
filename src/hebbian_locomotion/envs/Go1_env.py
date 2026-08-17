import torch

# Isaac Lab imports
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.math import euler_xyz_from_quat

# Pre-configured Unitree Go1 asset (correct meshes, inertias, actuators, standing pose).
# Canonical location; also re-exported as `from isaaclab_assets import UNITREE_GO1_CFG`.
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG


# ------------------------------------------------------------------
# Scene
# ------------------------------------------------------------------
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Flat-plane scene with the Go1 and a foot contact sensor."""

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

    # Robot: keep the pre-set init_state (proper standing pose + spawn height).
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Foot contacts (not in the reward, handy for logging / later leg-lock work).
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.0,
        history_length=3,          # was 1; a small history stabilises edge detection
        track_air_time=True,       # exposes data.last_air_time / current_air_time
        debug_vis=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=900.0,
            texture_file=f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
            visible_in_primary_ray=False,
        ),
    )


# ------------------------------------------------------------------
# Actions
# ------------------------------------------------------------------
@configclass
class ActionsCfg:
    """Desired joint angles (12 DOF), offset around the default standing pose.

    use_default_offset=True is essential for Go1: the network commands a *delta*
    around the standing configuration, so a near-zero output holds a stable stance
    rather than collapsing the legs. HAN tracks joint targets with a PD controller;
    the Go1 asset's actuators serve that role. scale is tunable (0.25-0.5 typical).
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


# ------------------------------------------------------------------
# Observations  (HAN Go1: trunk lin/ang vel, joint pos/vel, gravity vector -> 33 dims)
# ------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """Matches the HAN paper's Go1 observation space (Section IV-D).

    Layout (33 dims), in order:
        - base_lin_vel:      3   trunk linear velocity (body frame)
        - base_ang_vel:      3   trunk angular velocity (body frame)
        - joint_pos:        12   joint angles (relative to default)
        - joint_vel:        12   joint velocities
        - projected_gravity: 3   gravity vector in body frame

    NOTE: construct your HebbianNet with input_dim=33, output_dim=12.
    No last-action feedback (the paper's Go1 obs omits it); the training-loop
    low-pass filter still applies, it just isn't fed back as an observation.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EmptyManagerCfg:
    pass

def forward_velocity_x(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Forward velocity in robot-frame X-axis."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0]

def healthy_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Alive/upright bonus: 1.0 when sufficiently upright, else 0.0."""
    asset = env.scene[asset_cfg.name]
    up_proj = -asset.data.projected_gravity_b[:, 2]
    return (up_proj > 0.9).float()

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

def lin_vel_z_penalty(env, asset_cfg, std: float = 1.0):
    """Cost for vertical velocity — kills the ballistic/porpoising phase."""
    asset = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_b[:, 2]
    return -(1.0 - torch.exp(-torch.square(vz) / (std ** 2)))

def track_lin_vel_gaussian(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_speed: float = 1.0,
    std: float = 1.0,
) -> torch.Tensor:
    """r = exp(-(v_x - v_target)^2 / std^2), in [0, 1].

    std widened to 1.0 (from 0.5) so the reward has meaningful slope from
    v_x=0: exp(-1)=0.37 at standstill vs 1.0 at target, instead of the
    exp(-4)=0.018 dead tail. This is what lets ES bootstrap from standing.
    """
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_b[:, 0]
    return torch.exp(-torch.square(vx - target_speed) / (std ** 2))


def yaw_rate_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float = 1.0,
) -> torch.Tensor:
    """Cost for nonzero yaw rate (HAN Fig. 10, 'Yaw penalty').

    Returns 0 when w_z=0 and -> -1 as |w_z| grows. Crucially this is 0 (not
    +1) for a stationary robot, so standing still no longer pays a free
    reward — removing the degenerate attractor ES was collapsing onto.
    """
    asset = env.scene[asset_cfg.name]
    wz = asset.data.root_ang_vel_b[:, 2]
    return -(1.0 - torch.exp(-torch.square(wz) / (std ** 2)))

def feet_air_time_targeted(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    t_target: float = 0.20,   # target swing duration [s] — Go1 trot ~0.15-0.25
    sigma_t: float = 0.10,    # tolerance around the target
    vx_min: float = 0.10,     # only credit stepping that produces forward motion
) -> torch.Tensor:
    """Reward a stepping cadence near t_target, paid at touchdown.

    Tent on air time: a shuffle (short) and a bound (long flight) both score
    below a clean step. Gated on forward v_x so a vertical hop collects nothing.
    Uses the contact sensor's native air-time buffers (reset-safe edge detection).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    shaped = torch.exp(-torch.square(last_air_time - t_target) / (sigma_t ** 2))
    reward = torch.sum(shaped * first_contact.float(), dim=1) / first_contact.shape[1]

    vx = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    return reward * torch.sigmoid((vx - vx_min) / 0.05)

@configclass
class RewardsCfg:
    """Green's config + the walk-vs-bounce pincer (air-time reward × v_z cost)."""

    track_vel = RewTerm(
        func=track_lin_vel_gaussian, weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_speed": 1.0, "std": 0.8},
    )
    Healthy_t = RewTerm(
        func=healthy_bonus, weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    yaw_pen = RewTerm(
        func=yaw_rate_penalty, weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.0},
    )
    air_time = RewTerm(
        func=feet_air_time_targeted, weight=1.0,   # strong enough to forbid standstill
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
            "t_target": 0.20, "sigma_t": 0.10, "vx_min": 0.10,
        },
    )
    vz_pen = RewTerm(
        func=lin_vel_z_penalty, weight=0.2,        # small — forbids the launch, doesn't crush CoM motion
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.0},
    )

# ------------------------------------------------------------------
# Events
# ------------------------------------------------------------------
@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.2, 0.2)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )


# ------------------------------------------------------------------
# Master config
# ------------------------------------------------------------------
@configclass
class Go1EnvCfg(ManagerBasedRLEnvCfg):
    """HAN-faithful Go1 environment. Substitute for GeckoEnvCfg in the ES loop."""

    # One env per ES individual (match POPSIZE); overridden by the training script.
    scene = MySceneCfg(num_envs=1024, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()

    # No early termination: HAN states rollouts are not terminated early;
    # all episode steps contribute to fitness. This is what lets the bare
    # speed+yaw reward work on a robot that would otherwise fall out early.
    terminations: EmptyManagerCfg = EmptyManagerCfg()

    # Single knob for the tracking target (sweep {1.0, 1.5, 2.0} as in the paper).
    target_speed: float = 1.0

    def __post_init__(self):
        # 50 Hz control (fNN in the paper): 200 Hz physics / decimation 4.
        self.decimation = 4
        self.episode_length_s = 20.0   # 1000 steps at 50 Hz
        self.sim.dt = 0.005
        # Propagate the single target-speed knob into the reward term.
        self.rewards.track_vel.params["target_speed"] = self.target_speed