"""
check_leglock_visual.py — Eyeball the leg damage before trusting any numbers.

One robot, viewport on, no statistics, no files written. The only job is to
answer three questions:

  1. Does the damaged leg pass through the ground?
  2. Does the base get kicked at the moment damage engages?
  3. Does the robot keep moving afterwards, or collapse immediately?

The old kinematic pin (write_joint_state_to_sim) failed all three: it gave the
joint infinite stiffness, so when the held pose intersected the floor PhysX had
no compliant degree of freedom in the leg and resolved the penetration by
launching the base. The compliant PD hold should fail none of them.

Watch for, in the printed output:

  * dz_base spiking at the lock step        -> stiffness too high, base kicked
  * foot_z going negative                   -> leg through the floor
  * leg_err growing large in "stuck" mode   -> stiffness too low, leg sagging

Tune STUCK_STIFFNESS until dz_base stays quiet and leg_err stays small. Then
fix that value and use it in every condition for the rest of the project.

Set POLICY = None to test the damage on a robot that simply holds its default
stance -- the cleanest check, since nothing else is moving.
"""

# ===========================================================================
# CONFIG
# ===========================================================================
ROOT = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
        "workspace/hebbian_locomotion")

POLICY = '/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:04-18:00_GO1_FINAL_HAN_499_1853697842.pickle'                # None = hold default stance; or a checkpoint path
PARAM_SOURCE = "mu"

STEPS = 3000
LOCK_STEP = 250

LOCK_MODE = "stuck"          # "stuck" or "limp"
LEG_PATTERNS =  ["RR_.*"]
STUCK_STIFFNESS = 8.0     # <- the number you are tuning
STUCK_DAMPING = 1.0
LIMP_DAMPING = 1

ACTION_FILTER_ALPHA = 0.1
NORM_MODE = "max"
PRINT_EVERY = 10
ROLLOUT_SEED = 0
# ===========================================================================

import pickle

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)     # viewport ON -- the whole point
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from hebbian_locomotion.envs.Go1_env import Go1EnvCfg  # noqa: E402
from hebbian_locomotion.networks.hebbian_neural_net import HebbianNet  # noqa: E402
from hebbian_locomotion.networks.han_net import HANNet  # noqa: E402
from perturbation_go1 import LegLock  # noqa: E402


def build_policy(num_envs):
    """Load the checkpointed network, or return None to hold the default stance."""
    if POLICY is None:
        print("[INFO] no policy -- robot holds its default stance (zero action).")
        return None

    with open(POLICY, "rb") as f:
        data = pickle.load(f)
    solver, models_ckpt = data[0], data[1]
    params = np.asarray(solver.mu if PARAM_SOURCE == "mu" else solver.best_mu)

    sizes = list(models_ckpt.architecture)
    M = getattr(models_ckpt, "M", None)
    if M is None:
        model = HebbianNet(popsize=num_envs, sizes=sizes, norm_mode=NORM_MODE)
        print("[INFO] HebbianNet (instantaneous ABCD)")
    else:
        tau = int(getattr(models_ckpt, "tau_hebb", 1))
        model = HANNet(num_envs, sizes=sizes, norm_mode=NORM_MODE,
                       M=int(M), tau_hebb=tau)
        print(f"[INFO] HANNet  M={int(M)}  tau_hebb={tau}")

    model.set_a_model_params(params)
    return model


def main():
    torch.manual_seed(ROLLOUT_SEED)
    torch.cuda.manual_seed_all(ROLLOUT_SEED)
    np.random.seed(ROLLOUT_SEED)

    env_cfg = Go1EnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    model = build_policy(1)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    if model is not None:
        model.reset_weights()

    leg_lock = LegLock(
        lock_step=LOCK_STEP, patterns=LEG_PATTERNS, mode=LOCK_MODE,
        stuck_stiffness=STUCK_STIFFNESS, stuck_damping=STUCK_DAMPING,
        limp_damping=LIMP_DAMPING, hold_position='default'
    )
    leg_lock.resolve(robot)

    locked_ids = leg_lock.locked_ids
    hold_pos = leg_lock._hold_pos

    # Foot body of the damaged leg, e.g. "RL_.*" -> "RL_foot".
    prefix = LEG_PATTERNS[0].split("_")[0]
    try:
        foot_ids, foot_names = robot.find_bodies(f"{prefix}_foot")
        foot_id = foot_ids[0]
        print(f"[INFO] tracking foot body {foot_names[0]} (id {foot_id})")
    except Exception as e:
        foot_id = None
        print(f"[WARN] could not resolve the foot body ({e}); "
              "foot height will not be reported.")

    n_act = robot.data.default_joint_pos.shape[1]
    prev_actions = None
    prev_z = float(robot.data.root_pos_w[0, 2])

    print(f"\n{'step':>5} {'base_z':>8} {'dz_base':>9} {'foot_z':>8} "
          f"{'leg_err':>8} {'vx':>7} {'up':>6}")
    print("-" * 60)

    for step in range(STEPS):
        if model is not None:
            actions = model.forward(obs["policy"])
            actions = leg_lock.mask_actions(actions, step)
            if prev_actions is not None:
                actions = (ACTION_FILTER_ALPHA * actions
                           + (1.0 - ACTION_FILTER_ALPHA) * prev_actions)
            prev_actions = actions
        else:
            # Zero action = default stance, since use_default_offset=True.
            actions = torch.zeros(1, n_act, device=env.device)

        leg_lock.maybe_apply(robot, step)          # BEFORE env.step
        obs, _, _, _, _ = env.step(actions)

        base_z = float(robot.data.root_pos_w[0, 2])
        dz = base_z - prev_z
        prev_z = base_z

        foot_z = (float(robot.data.body_pos_w[0, foot_id, 2])
                  if foot_id is not None else float("nan"))
        leg_err = float(
            (robot.data.joint_pos[0, locked_ids] - hold_pos[0]).abs().max()
        )
        vx = float(robot.data.root_lin_vel_b[0, 0])
        up = float(-robot.data.projected_gravity_b[0, 2])

        at_lock = step in (LOCK_STEP - 1, LOCK_STEP, LOCK_STEP + 1)
        if step % PRINT_EVERY == 0 or at_lock:
            mark = "  <== DAMAGE" if step == LOCK_STEP else ""
            print(f"{step:5d} {base_z:8.4f} {dz:9.5f} {foot_z:8.4f} "
                  f"{leg_err:8.4f} {vx:7.3f} {up:6.3f}{mark}")



    leg_lock.restore(robot)
    env.close()

    print("\nChecklist")
    print("  dz_base quiet through the lock step?   -> no base kick")
    print("  foot_z stayed above ~0?                -> no ground penetration")
    print("  leg_err small in 'stuck' mode?         -> stiffness sufficient")
    print("  robot still upright afterwards?        -> graded, not catastrophic")


if __name__ == "__main__":
    main()
    simulation_app.close()