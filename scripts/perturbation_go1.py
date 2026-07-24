"""
perturbation.py — Mid-episode kinematic leg-lock for the Unitree Go1 rollout.

Robot-agnostic: the pin works on any articulation. Only the leg *patterns*
change per robot. Go1 uses the Unitree naming convention FL/FR/RL/RR
(front-left = FL), so the front-left leg is matched by ``"FL_.*"`` — NOT the
gecko's ``".*LF.*"``. Confirm exact names with inspect_robot.py before a run.

Semantics (matching Leung et al. "fixed in default position"):
    From a user-set trigger step onward, all joints of the targeted leg(s)
    are pinned to their default position with zero velocity, every control
    step. The policy keeps running and keeps *observing* those joints (so it
    feels the damage through proprioception), but the locked joints no longer
    follow its commands.

Why a kinematic pin (write_joint_state_to_sim) rather than an action override:
    It is control-mode-agnostic. Whether the live ActionsCfg is
    JointEffortActionCfg or JointPositionActionCfg, the pin overrides the
    resulting joint state directly, so the lock behaves identically either way.
    This sidesteps the effort-vs-position action-config discrepancy entirely.

Usage (in run_es_dynamic.py — see INTEGRATION block at the bottom):

    from perturbation import LegLock

    leg_lock = LegLock(
        lock_step=150,           # <- the timestep YOU set; None = never lock
        patterns=["FL_.*"],      # <- Go1 front-left leg (FL_hip/thigh/calf_joint)
    )
    leg_lock.resolve(robot)      # once, after env.reset() / robot is available

    # ... inside the rollout loop, AFTER env.step(actions):
    leg_lock.maybe_apply(robot, step)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


class LegLock:
    """Pin one or more legs to their default joint positions from a set step.

    Parameters
    ----------
    lock_step:
        Control-loop step index at which the lock engages. The lock is active
        for every step ``s >= lock_step``. Pass ``None`` to disable (this is
        the perturbation-absent arm of the 2x2 freeze x perturbation design).
    patterns:
        Regex pattern(s) passed to ``robot.find_joints``. Each pattern should
        match all three joints of one Go1 leg, e.g. ``"FL_.*"`` catches
        ``FL_hip_joint``, ``FL_thigh_joint``, ``FL_calf_joint``. Provide several
        to lock multiple legs at once (e.g. an LF+RF-analogous front-pair
        condition: ``["FL_.*", "FR_.*"]``). Go1 legs are FL / FR / RL / RR.
        Resolve these against the real joint names printed by
        ``inspect_robot.py`` — do not guess.
    hold_position:
        Where to pin the locked joints. ``"default"`` uses
        ``robot.data.default_joint_pos`` (the constant-default-position
        semantics of Leung et al.). For the Go1 with
        ``use_default_offset=True`` this default is the nominal standing pose,
        so the locked leg freezes in a natural stance. A float tensor of shape
        ``(num_envs, n_locked)`` may be passed instead for a custom pose.
    verbose:
        If True, print the resolved joint ids/names once at resolve time.
    """

    def __init__(self, lock_step: Optional[int], patterns: Sequence[str], hold_position: str = "default", verbose: bool = True,) -> None:
        if isinstance(patterns, str):
            patterns = [patterns]
        self.lock_step = lock_step
        self.patterns = list(patterns)
        self.hold_position = hold_position
        self.verbose = verbose

        # filled by resolve()
        self.locked_ids: Optional[torch.Tensor] = None
        self.locked_names: list[str] = []
        self._hold_pos: Optional[torch.Tensor] = None
        self._zero_vel: Optional[torch.Tensor] = None
        self._resolved = False

    # ------------------------------------------------------------------
    def resolve(self, robot) -> None:
        """Resolve regex patterns to joint indices against the live asset.

        Call once, after the robot articulation exists (post env.reset()).
        Caches the locked joint ids, the hold pose, and a zero-velocity buffer
        so the per-step path allocates nothing.
        """
        if self.lock_step is None:
            # Perturbation-absent arm: nothing to resolve, maybe_apply is a no-op.
            self._resolved = True
            if self.verbose:
                print("[LegLock] lock_step=None -> perturbation DISABLED.")
            return

        all_ids: list[int] = []
        for pat in self.patterns:
            ids, names = robot.find_joints(pat)
            if len(ids) == 0:
                raise ValueError(
                    f"[LegLock] pattern {pat!r} matched no joints. "
                    f"Available joints: {list(robot.data.joint_names)}"
                )
            all_ids.extend(ids)
            self.locked_names.extend(names)

        # De-duplicate while preserving order (patterns may overlap).
        seen = set()
        dedup = [j for j in all_ids if not (j in seen or seen.add(j))]

        device = robot.data.default_joint_pos.device
        self.locked_ids = torch.tensor(dedup, dtype=torch.long, device=device)

        num_envs = robot.data.default_joint_pos.shape[0]
        if self.hold_position == "default":
            self._hold_pos = robot.data.default_joint_pos[:, self.locked_ids].clone()
        elif torch.is_tensor(self.hold_position):
            self._hold_pos = self.hold_position.to(device)
        else:
            raise ValueError(
                f"[LegLock] hold_position must be 'default' or a tensor, "
                f"got {self.hold_position!r}"
            )
        self._zero_vel = torch.zeros(
            (num_envs, self.locked_ids.numel()), device=device
        )

        self._resolved = True
        if self.verbose:
            print(
                f"[LegLock] lock_step={self.lock_step} | "
                f"locking {self.locked_ids.numel()} joints "
                f"{self.locked_names} (ids={dedup})"
            )

    # ------------------------------------------------------------------
    def is_active(self, step: int) -> bool:
        """True iff the lock is engaged at this step."""
        return self.lock_step is not None and step >= self.lock_step

    # ------------------------------------------------------------------
    def maybe_apply(self, robot, step: int) -> None:
        """Pin the locked joints to the hold pose if the lock is active.

        Call once per control step, AFTER ``env.step(actions)`` so the snap to
        default overrides the physics result of the step just taken. The lock
        is re-asserted every step (each decimation window is ~0.02 s, so the
        leg stays effectively fixed with only sub-window drift).
        """
        if not self._resolved:
            raise RuntimeError("[LegLock] call resolve(robot) before maybe_apply().")
        if not self.is_active(step):
            return
        robot.write_joint_state_to_sim(
            position=self._hold_pos,
            velocity=self._zero_vel,
            joint_ids=self.locked_ids,
        )

    # ------------------------------------------------------------------
    def mask_actions(self, actions: torch.Tensor, step: int) -> torch.Tensor:
        """OPTIONAL: zero the policy's commands on locked joints once active.

        Not required for the pin to work, but keeps the action signal (and the
        EMA filter / any prev-action observation) from carrying commands that
        the locked joints can't follow. Returns actions unchanged when the lock
        is inactive. Operates out-of-place.
        """
        if not self.is_active(step):
            return actions
        actions = actions.clone()
        actions[:, self.locked_ids] = 0.0
        return actions