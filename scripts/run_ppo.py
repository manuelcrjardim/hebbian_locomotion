import argparse
import os
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Launch Isaac Sim FIRST (before importing anything that touches the sim).
# -----------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO baseline for the Gecko task.")
parser.add_argument("--num_envs", type=int, default=1024,
                    help="Parallel envs (ES uses 1024).")
parser.add_argument("--arch", type=str, default="default", choices=["tiny", "default"],
                    help="'default' = actor/critic (64,64); 'tiny' = actor (16), critic (64,64).")
parser.add_argument("--max_iterations", type=int, default=3000,
                    help="PPO update iterations (ignored if --target_env_steps is set).")
parser.add_argument("--target_env_steps", type=int, default=None,
                    help="If set, overrides --max_iterations to hit this total env-step budget "
                         "(HAN reference: 80_000_000; Ant-damage experiment: 1_000_000_000).")
parser.add_argument("--num_steps_per_env", type=int, default=24,
                    help="Rollout length per env per update (rsl_rl convention).")
parser.add_argument("--learning_epochs", type=int, default=5,
                    help="PPO epochs per update (HAN/SB3 used 10).")
parser.add_argument("--action_filter_alpha", type=float, default=0.1,
                    help="EMA action filter: a_t = alpha*a_raw + (1-alpha)*a_{t-1}. "
                         "Matches run_es_dynamic.py (0.1). Set 1.0 to disable.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--log_root", type=str,
                    default="/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/"
                            "isaac_lab_sandbox/workspace/hebbian_locomotion/tensorboard_ppo",
                    help="Directory for TensorBoard logs and checkpoints.")
# adds --headless, --device, --livestream, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# 2. Imports that require the sim app to exist.
# -----------------------------------------------------------------------------
import torch

from isaaclab.envs import ManagerBasedRLEnv

# rsl_rl integration lives in `isaaclab_rl` on Isaac Lab 2.x; fall back to the
# older `omni.isaac.lab_rl` namespace if needed.
try:
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )
except ImportError:  # older Isaac Lab builds
    from omni.isaac.lab_rl.rsl_rl import (  # type: ignore
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )

from rsl_rl.runners import OnPolicyRunner

# Same env + reward as the ES runs — imported, not re-defined.
from hebbian_locomotion.envs.GeckoEnv import GeckoEnvCfg


# -----------------------------------------------------------------------------
# 3. Env subclass that applies the ES rollout's low-pass action filter.
#
#    In run_es_dynamic.py the EMA is applied in the rollout loop, *outside* the
#    env cfg:  actions = 0.1*actions + 0.9*prev_actions, with prev seeded at the
#    raw first action and reset each epoch. Replicating it here keeps the closed-
#    loop dynamics PPO sees identical to what the Hebbian controllers see. The
#    filter is applied to the raw (pre-scale) action, before the ActionManager
#    scales it — exactly as in ES.
# -----------------------------------------------------------------------------
class FilteredGeckoEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, ema_alpha: float = 0.1, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self._ema_alpha = float(ema_alpha)
        self._prev_action = None

    def step(self, action: torch.Tensor):
        if self._ema_alpha < 1.0:
            if self._prev_action is None:
                self._prev_action = torch.zeros_like(action)
            action = self._ema_alpha * action + (1.0 - self._ema_alpha) * self._prev_action
            self._prev_action = action.detach().clone()

        obs, rew, terminated, truncated, info = super().step(action)

        # Reset filter memory for envs that auto-reset this step, so the smoothing
        # does not bleed across episode boundaries.
        if self._prev_action is not None:
            done = (terminated | truncated)
            if done.any():
                self._prev_action[done] = 0.0
        return obs, rew, terminated, truncated, info


def main():
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    run_name = f"gecko_ppo_{args_cli.arch}_seed{args_cli.seed}_{timestamp}"
    log_dir = os.path.join(args_cli.log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # --- Environment: identical cfg to the ES runs ---------------------------
    env_cfg = GeckoEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = FilteredGeckoEnv(cfg=env_cfg, ema_alpha=args_cli.action_filter_alpha)

    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    act_dim = env.action_manager.total_action_dim
    print(f"[INFO] Gecko PPO baseline | obs={obs_dim}  act={act_dim}  "
          f"num_envs={args_cli.num_envs}  arch={args_cli.arch}  "
          f"action_filter_alpha={args_cli.action_filter_alpha}")

    # --- Resolve iteration budget -------------------------------------------
    max_iterations = args_cli.max_iterations
    if args_cli.target_env_steps is not None:
        steps_per_iter = args_cli.num_envs * args_cli.num_steps_per_env
        max_iterations = max(1, args_cli.target_env_steps // steps_per_iter)
        print(f"[INFO] target_env_steps={args_cli.target_env_steps:,} -> "
              f"{max_iterations} iterations ({steps_per_iter:,} env-steps/iter)")
    total_steps = max_iterations * args_cli.num_envs * args_cli.num_steps_per_env
    print(f"[INFO] total environment-step budget ~= {total_steps:,}")

    # --- Actor-critic architecture ------------------------------------------
    # tanh matches both the ES/Hebbian nets and SB3's default PPO MLP.
    actor_hidden = [16] if args_cli.arch == "tiny" else [64, 64]
    critic_hidden = [64, 64]
    policy_cfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,              # SB3 log_std init 0 -> std 1.0
        actor_hidden_dims=actor_hidden,
        critic_hidden_dims=critic_hidden,
        activation="tanh",
    )

    # --- PPO algorithm (HAN / SB3 defaults where they map) -------------------
    algorithm_cfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=args_cli.learning_epochs,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",                # HAN used fixed LR (target_kl=None)
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,                 # unused while schedule="fixed"
        max_grad_norm=0.5,
    )

    runner_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=args_cli.num_steps_per_env,
        max_iterations=max_iterations,
        save_interval=50,
        experiment_name="gecko_ppo",
        run_name=run_name,
        empirical_normalization=False,   # SB3 PPO default: no obs normalisation
        policy=policy_cfg,
        algorithm=algorithm_cfg,
        seed=args_cli.seed,
    )

    # --- Wrap for rsl_rl and train ------------------------------------------
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(
        env,
        runner_cfg.to_dict(),
        log_dir=log_dir,
        device=env.unwrapped.device,
    )

    print(f"[INFO] Logging to {log_dir}")
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()