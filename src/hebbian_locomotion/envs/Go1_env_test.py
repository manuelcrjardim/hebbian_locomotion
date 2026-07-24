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
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

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
        history_length=1,
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
    rather than collapsing the legs. scale is tunable (0.25-0.5 typical).
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
    """Matches the HAN paper's Go1 observation space.

    Layout (33 dims), in order:
        - base_lin_vel:      3

        - base_ang_vel:      3
        - joint_pos:        12
        - joint_vel:        12
        - projected_gravity: 3

    NOTE: construct your HebbianNet with input_dim=33, output_dim=12.
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


def capped_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    v_target: float = 1.5,
) -> torch.Tensor:
    """Forward velocity capped at v_target: monotonic up to target, flat above.

    min(v_x, v_target) rewards progress without paying for lunging past the
    target, so there's no incentive to over-accelerate then recover.
    """
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_b[:, 0]
    return torch.clamp(vx, max=v_target)


def healthy_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Alive/upright bonus: 1.0 when sufficiently upright, else 0.0."""
    asset = env.scene[asset_cfg.name]
    up_proj = -asset.data.projected_gravity_b[:, 2]
    return (up_proj > 0.9).float()


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared change in filtered action: -||a_t - a_{t-1}||^2.

    Reads the current filtered action (env.prev_action_buf, set by the training
    loop) and the previous one (env.prev_prev_action_buf). The loop must stash
    the prior buffer before overwriting it each step (see loop change below).
    Returns 0 on the first step, before both buffers exist.
    """
    cur = getattr(env, "prev_action_buf", None)
    prev = getattr(env, "prev_prev_action_buf", None)
    if cur is None or prev is None:
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.sum(torch.square(cur - prev), dim=-1)

# ------------------------------------------------------------------
# Reward  (ORIGINAL gecko reward: monotonic forward velocity + upright + heading)
# Eq. 5 in the Leung/gecko paper. Known-good, ES-bootstrappable signal.
# ------------------------------------------------------------------
def forward_velocity_x(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body-frame forward velocity (unbounded, monotonic reward)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 0]


def upright_posture(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """0.0 if sufficiently upright, else -0.5 (Eq. 6)."""
    asset = env.scene[asset_cfg.name]
    z_proj = -asset.data.projected_gravity_b[:, 2]
    return torch.where(z_proj > 0.93, 0.0, -0.5)


def heading_yaw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """0.0 if heading within +/-0.45 rad of +X, else -0.5 (Eq. 7)."""
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.where(torch.abs(yaw) < 0.45, 0.0, -0.5)


@configclass
class RewardsCfg2:
    """Gecko paper reward (Eq. 5): kv=2.0, ku=0.5, ky=0.5."""

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
class RewardsCfg:
    """Augmented Gym-style locomotion reward: velocity + healthy - action_rate."""

    V_t = RewTerm(
        func=capped_forward_velocity,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "v_target": 1.5},
    )
    Healthy_t = RewTerm(
        func=healthy_bonus,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ActionRate_t = RewTerm(
        func=action_rate_penalty,
        weight=0.01,
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
    """Go1 environment with the ORIGINAL gecko reward. Drop-in for the ES loop."""

    # One env per ES individual (match POPSIZE); overridden by the training script.
    scene = MySceneCfg(num_envs=1024, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()

    # No early termination: all steps contribute to fitness (matches your loop's
    # done_mask handling and the paper's no-early-termination rollouts).
    terminations: EmptyManagerCfg = EmptyManagerCfg()

    def __post_init__(self):
        # 50 Hz control: 200 Hz physics / decimation 4.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005