"""
Rollout of the evolved LSTM baselines, for gait-period measurement only.

For each checkpoint in CONDITIONS: load it, select one individual, roll it out
for STEPS control steps, and record the per-step reward and body state. Writes
one .npz per network into

    OUT_ROOT/LSTM_<timestamp>/npz/<group>_<label>_data.npz

with the same filename convention and the same vx / z / rewards keys that
pca_sequence.py writes, so gait_spectrum.py picks these up unchanged.

There is no weight-space analysis here. An LSTM's evolved parameters are fixed
for the whole rollout, so there is no weight trajectory to embed, no summed
weight change, and no fixed-point criterion to apply. The within-episode state
is the hidden and cell state, which is not what the gait spectrum needs.

Isaac Sim is launched ONCE and the environment is reused across networks.

Edit the CONFIG block below, then:

    python rollout_lstm.py
"""

from datetime import datetime
current_time = datetime.now().strftime("%m:%d-%H:%M")

# ===========================================================================
# CONFIG -- edit everything here
# ===========================================================================
ROOT = ("/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/"
        "workspace/hebbian_locomotion")

# Same shape as pca_sequence.py, so entries copy over directly.
# "label" names the output and must be unique.
CONDITIONS = [
    {"group": "LSTM", "label": "SEED1",
     "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:28-19:54_GO1_FINAL_LSTM_687358595_499_benchmark.pickle"},
    {"group": "LSTM", "label": "SEED2",
     "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:29-12:50_GO1_FINAL_LSTM_36833173_499_benchmark.pickle"},
    {"group": "LSTM", "label": "SEED3",
     "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:29-14:33_GO1_FINAL_LSTM_1521446440_499_benchmark.pickle"},
    {"group": "LSTM", "label": "SEED4",
     "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:29-17:26_GO1_FINAL_LSTM_1550689036_499_benchmark.pickle"},
    {"group": "LSTM", "label": "SEED5",
    "ckpt": f"/cs/student/project_msc/2025/rai/mdecastr/Isaac_Lab/isaac_lab_sandbox/workspace/hebbian_locomotion/checkpoints/08:31-15:33_GO1_FINAL_LSTM_1091802232_499_benchmark.pickle"},
]

OUT_ROOT = f"{ROOT}/analysis"

STEPS = 500                    # rollout length, control steps (training used 500)
PARAM_SOURCE = "mu"            # "mu" or "best_mu"
ACTION_FILTER_ALPHA = 0.1      # a_t = alpha*a_raw + (1-alpha)*a_{t-1}
HEADLESS = True                # False to watch the rollout

SKIP_MISSING = True            # skip absent checkpoints rather than aborting
CONTINUE_ON_ERROR = True       # one bad network must not lose the whole batch

NPZ_SUBDIR = "npz"             # flat collection of every <label>_data.npz

TAG = f"{current_time}_{PARAM_SOURCE}"
RUN_DIR = f"{OUT_ROOT}/LSTM_{TAG}"
# ===========================================================================

import json
import os
import pickle

# ---------------------------------------------------------------------------
# Launch Isaac Sim (must precede all isaaclab imports)
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=HEADLESS)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from hebbian_locomotion.envs.Go1_env import Go1EnvCfg  # noqa: E402
from hebbian_locomotion.networks.LSTM import LSTMNet  # noqa: E402


# ===========================================================================
# Checkpoint
# ===========================================================================
def load_checkpoint(path):
    """Load a 4- or 5-element ES checkpoint."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    solver, models_ckpt = data[0], data[1]
    run_cfg = data[4] if len(data) >= 5 else None
    return solver, models_ckpt, run_cfg


# ===========================================================================
# Rollout
# ===========================================================================
def rollout(env, net):
    """Run one episode, recording reward and body state each step."""
    obs, _ = env.reset()
    net.reset_weights()          # resets hidden/cell states of all three blocks

    robot = env.scene["robot"]

    rewards, vx, z = [], [], []
    prev_actions = None

    for _ in range(STEPS):
        raw = net.forward(obs["policy"])

        # EMA action filter -- the policy was evolved with this in the loop
        if prev_actions is None:
            actions = raw
        else:
            actions = (ACTION_FILTER_ALPHA * raw
                       + (1.0 - ACTION_FILTER_ALPHA) * prev_actions)
        prev_actions = actions

        obs, rew, _, _, _ = env.step(actions)

        rewards.append(rew[0].item())
        vx.append(robot.data.root_lin_vel_b[0, 0].item())
        z.append(robot.data.root_pos_w[0, 2].item())

    return np.asarray(rewards), np.asarray(vx), np.asarray(z)


# ===========================================================================
# Per-network
# ===========================================================================
def analyse_one(env, cond, npz_dir):
    """Load, roll out and record a single network. Returns its summary dict."""
    label, ckpt, group = cond["label"], cond["ckpt"], cond["group"]

    solver, models_ckpt, run_cfg = load_checkpoint(ckpt)
    params = solver.mu if PARAM_SOURCE == "mu" else solver.best_mu
    sizes = list(models_ckpt.architecture)

    print(f"    arch {sizes}   source={PARAM_SOURCE}")

    net = LSTMNet(popsize=1, sizes=sizes)
    net.set_a_model_params(np.asarray(params))

    rewards, vx, z = rollout(env, net)

    summary = {
        "tag": TAG,
        "group": group,
        "label": label,
        "checkpoint": os.path.abspath(ckpt),
        "architecture": sizes,
        "param_source": PARAM_SOURCE,
        "steps": STEPS,
        "mean_reward": float(rewards.mean()),
        "total_reward": float(rewards.sum()),
        "mean_vx": float(vx.mean()),
        "z_range": float(z.max() - z.min()),
    }

    npz_path = os.path.join(npz_dir, f"{group}_{label}_data.npz")
    np.savez_compressed(npz_path, rewards=rewards, vx=vx, z=z)
    summary["npz"] = os.path.abspath(npz_path)
    print(f"    [NPZ] {os.path.basename(npz_path)}")

    return summary


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    npz_dir = os.path.join(RUN_DIR, NPZ_SUBDIR)
    os.makedirs(npz_dir, exist_ok=True)

    # --- resolve the work list before touching the simulator --------------
    todo, seen = [], set()
    for cond in CONDITIONS:
        label = cond["label"]
        if label in seen:
            print(f"[WARN] duplicate label {label!r}; skipping the repeat.")
            continue
        if not os.path.exists(cond["ckpt"]):
            msg = f"[WARN] missing checkpoint for {label}: {cond['ckpt']}"
            if SKIP_MISSING:
                print(msg + "  -> skipped")
                continue
            raise FileNotFoundError(msg)
        seen.add(label)
        todo.append(cond)

    if not todo:
        raise SystemExit("No usable checkpoints in CONDITIONS.")
    print(f"[INFO] {len(todo)} network(s) to roll out -> {RUN_DIR}\n")

    # --- environment: built ONCE and reused --------------------------------
    env_cfg = Go1EnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    results, failures = [], []
    try:
        for k, cond in enumerate(todo, 1):
            label = cond["label"]
            print(f"[{k}/{len(todo)}] {label}  [{cond['group']}]")
            try:
                results.append(analyse_one(env, cond, npz_dir))
            except Exception as exc:                      # noqa: BLE001
                failures.append({"label": label, "error": repr(exc)})
                print(f"    [FAIL] {exc!r}")
                if not CONTINUE_ON_ERROR:
                    raise
    finally:
        env.close()

    # --- combined index ----------------------------------------------------
    index = {
        "tag": TAG,
        "run_dir": os.path.abspath(RUN_DIR),
        "npz_dir": os.path.abspath(npz_dir),
        "param_source": PARAM_SOURCE,
        "steps": STEPS,
        "n_requested": len(CONDITIONS),
        "n_analysed": len(results),
        "failures": failures,
        "networks": results,
    }
    with open(os.path.join(RUN_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # --- console summary table --------------------------------------------
    print("\n--- summary " + "-" * 50)
    print(f"  {'label':22s} {'group':8s} {'mean_vx':>8s} {'z_range':>8s}")
    for r in sorted(results, key=lambda x: (str(x["group"]), x["label"])):
        print(f"  {r['label']:22s} {str(r['group']):8s} "
              f"{r['mean_vx']:>8.3f} {r['z_range']:>8.3f}")
    if failures:
        print(f"\n  {len(failures)} failure(s):")
        for f_ in failures:
            print(f"    {f_['label']:22s} {f_['error']}")
    print("-" * 62 + "\n")

    print(f"[DONE] {len(results)}/{len(todo)} rolled out. "
          f"Outputs under {os.path.abspath(RUN_DIR)}")

    if results:
        manifest = {
            "tag": TAG,
            "npz_dir": os.path.abspath(npz_dir),
            "networks": [
                {"group": r["group"], "label": r["label"],
                 "npz": os.path.basename(r["npz"])}
                for r in sorted(results, key=lambda x: (str(x["group"]), x["label"]))
            ],
        }
        with open(os.path.join(npz_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  [MANIFEST] {os.path.join(os.path.abspath(npz_dir), 'manifest.json')}")

        print("\n  Paste into gait_spectrum.py:\n")
        print(f'    BASE = "{os.path.abspath(npz_dir)}"')
        print("    NETWORKS = [")
        for r in sorted(results, key=lambda x: (str(x["group"]), x["label"])):
            print(f'        ("{r["group"]}", "{r["label"]}", '
                  f'f"{{BASE}}/{os.path.basename(r["npz"])}"),')
        print("    ]\n")


if __name__ == "__main__":
    main()