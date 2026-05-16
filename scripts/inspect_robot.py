"""
inspect_robot.py — Print joint names, body names, and contact sensor bodies.
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from hebbian_locomotion.envs.GeckoEnv import GeckoEnvCfg


def main():
    env_cfg = GeckoEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    contact_sensor = env.scene["contact_sensor"]

    print("\n" + "=" * 60)
    print("JOINT NAMES")
    print("=" * 60)
    for i, name in enumerate(robot.data.joint_names):
        print(f"  [{i:2d}] {name}")

    print("\n" + "=" * 60)
    print("BODY / LINK NAMES")
    print("=" * 60)
    for i, name in enumerate(robot.data.body_names):
        print(f"  [{i:2d}] {name}")

    print("\n" + "=" * 60)
    print("CONTACT SENSOR — matched bodies")
    print("=" * 60)
    print(f"  prim_path filter: {env_cfg.scene.contact_sensor.prim_path}")
    print(f"  matched body names:")
    for i, name in enumerate(contact_sensor.body_names):
        print(f"    [{i:2d}] {name}")
    print(f"  net_forces_w shape: {contact_sensor.data.net_forces_w.shape}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()