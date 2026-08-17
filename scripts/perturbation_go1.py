"""
perturbation_go1.py — Mid-episode compliant leg damage for the Unitree Go1.

Robot-agnostic: the mechanism works on any articulation with PD actuators.
Only the leg *patterns* change per robot. Go1 uses the Unitree naming
convention FL/FR/RL/RR (front-left = FL), so the front-left leg is matched by
``"FL_.*"``. Confirm exact names with inspect_robot.py before a run.

Why a compliant PD hold rather than a kinematic pin
---------------------------------------------------
The earlier implementation used ``write_joint_state_to_sim``, which teleports
the joint each step and gives it effectively infinite stiffness. Such a joint
cannot yield to contact: when the held pose intersects the ground, PhysX has
no compliant degree of freedom in the leg to resolve the penetration, so the
only thing that can move is the robot's base — which gets launched. That is
the instant-fall artefact seen in visualisation.

Overriding the actuator gains instead keeps the joint inside the physics
solver. It still tracks the damaged pose, but it deflects under load, so
contacts resolve normally and the base is never kicked.

Two severity levels
-------------------
``"stuck"``  — high stiffness, holds the DEFAULT pose. This is the damage
    model of Leung et al. ("fixed in default position"): the leg stops
    following commands but still bears weight, so the robot degrades from
    four working limbs to three working plus one rigid prop. Graded
    degradation, which is what the between-condition tests need.

``"limp"``   — zero stiffness, damping only. The dead-motor model: the leg
    hangs and swings freely with joint friction. More severe, because the
    robot drops to three support limbs and must find a tripod gait. Risks a
    floor effect where every condition fails equally.

Running both is a better design than either alone: a dose-response result
("plasticity helps at mild damage, all conditions collapse at severe") is far
more informative than a single ambiguous data point, and costs only one extra
evaluation pass over policies that are already trained.

Gains are restored on ``reset()`` because ``write_joint_stiffness_to_sim``
mutates persistent actuator state — without this, damage leaks across
episodes and silently contaminates every subsequent rollout.

Usage
-----
    from perturbation_go1 import LegLock

    leg_lock = LegLock(lock_step=150, patterns=["FL_.*"], mode="stuck")
    leg_lock.resolve(robot)          # once, after env.reset()

    # inside the rollout loop, BEFORE env.step(actions):
    leg_lock.maybe_apply(robot, step)

    # after the episode, before the next reset:
    leg_lock.restore(robot)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


class LegLock:
    """Damage one or more legs from a set step via compliant actuator override.

    Parameters
    ----------
    lock_step:
        Control-loop step index at which damage engages; active for every step
        ``s >= lock_step``. Pass ``None`` to disable — this is the
        perturbation-absent arm of the 2x2 freeze x perturbation design.
    patterns:
        Regex pattern(s) for ``robot.find_joints``. ``"FL_.*"`` catches
        ``FL_hip_joint``, ``FL_thigh_joint``, ``FL_calf_joint``. Pass several
        for a multi-leg condition, e.g. ``["FL_.*", "FR_.*"]``. Resolve
        against the names printed by ``inspect_robot.py`` — do not guess.
    mode:
        ``"stuck"`` (default) holds the default pose with high stiffness;
        ``"limp"`` sets zero stiffness so the leg hangs with damping only.
    stuck_stiffness, stuck_damping:
        PD gains for ``"stuck"``. High enough to hold the pose against body
        weight, finite enough to deflect on contact rather than launching the
        base. Go1 nominal joint gains are k_p = 25, k_d = 0.5.
    limp_damping:
        Viscous damping for ``"limp"``. tau = -k_d * qdot, i.e. joint friction.
        This is a free parameter that affects results, so fix it once and use
        the identical value across every condition.
    hold_position:
        ``"default"`` uses ``robot.data.default_joint_pos``. A float tensor of
        shape ``(num_envs, n_locked)`` may be passed for a custom pose. Ignored
        in ``"limp"`` mode.
    verbose:
        Print resolved joint ids/names once at resolve time.
    """

    VALID_MODES = ("stuck", "limp")

    def __init__(
        self,
        lock_step: Optional[int],
        patterns: Sequence[str],
        mode: str = "stuck",
        stuck_stiffness: float = 500.0,
        stuck_damping: float = 20.0,
        limp_damping: float = 2.0,
        hold_position: str = "default",
        verbose: bool = True,
    ) -> None:
        if isinstance(patterns, str):
            patterns = [patterns]
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"[LegLock] mode must be one of {self.VALID_MODES}, got {mode!r}"
            )

        self.lock_step = lock_step
        self.patterns = list(patterns)
        self.mode = mode
        self.stuck_stiffness = float(stuck_stiffness)
        self.stuck_damping = float(stuck_damping)
        self.limp_damping = float(limp_damping)
        self.hold_position = hold_position
        self.verbose = verbose
        self._patched: list = []
        # filled by resolve()
        self.locked_ids: Optional[torch.Tensor] = None
        self.locked_names: list[str] = []
        self._hold_pos: Optional[torch.Tensor] = None
        self._orig_stiffness: Optional[torch.Tensor] = None
        self._orig_damping: Optional[torch.Tensor] = None
        self._resolved = False
        self._applied = False      # gains overridden this episode?

    # ------------------------------------------------------------------
    def resolve(self, robot) -> None:
        """Resolve patterns to joint indices and cache the original gains.

        Call once, after the robot articulation exists (post env.reset()).
        Caches the locked ids, the hold pose, and the pre-damage PD gains so
        ``restore()`` can put them back.
        """
        if self.lock_step is None:
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

        if self.hold_position == "default":
            self._hold_pos = robot.data.default_joint_pos[:, self.locked_ids].clone()
        elif torch.is_tensor(self.hold_position):
            self._hold_pos = self.hold_position.to(device)
        else:
            raise ValueError(
                f"[LegLock] hold_position must be 'default' or a tensor, "
                f"got {self.hold_position!r}"
            )

        # Cache the undamaged gains so restore() can undo the override.
        self._orig_stiffness = robot.data.joint_stiffness[:, self.locked_ids].clone()
        self._orig_damping = robot.data.joint_damping[:, self.locked_ids].clone()

        self._resolved = True
        if self.verbose:
            kp = self._orig_stiffness[0].tolist()
            kd = self._orig_damping[0].tolist()
            print(
                f"[LegLock] lock_step={self.lock_step} mode={self.mode!r} | "
                f"{self.locked_ids.numel()} joints {self.locked_names} "
                f"(ids={dedup})"
            )
            print(f"[LegLock] original gains  k_p={kp}  k_d={kd}")
            if self.mode == "stuck":
                print(f"[LegLock] damaged gains  k_p={self.stuck_stiffness} "
                      f"k_d={self.stuck_damping}  (holds default pose)")
            else:
                print(f"[LegLock] damaged gains  k_p=0.0 "
                      f"k_d={self.limp_damping}  (hangs, damping only)")

    # ------------------------------------------------------------------
    def is_active(self, step: int) -> bool:
        """True iff damage is engaged at this step."""
        return self.lock_step is not None and step >= self.lock_step

    def _patch_actuator(self, robot):
        """Override computed efforts on damaged joints.

        ActuatorNetMLP ignores stiffness/damping, so gain writes have no effect.
        Wrapping compute() is the only reliable way to damage the joint while
        keeping it inside the physics solver (i.e. still compliant to contact).
        """
        for name, act in robot.actuators.items():
            idx = act.joint_indices
            glob = (torch.arange(robot.num_joints, device=self.locked_ids.device)[idx]
                    if not isinstance(idx, slice) or idx != slice(None)
                    else torch.arange(robot.num_joints, device=self.locked_ids.device))
            # positions of our locked joints within THIS actuator's slice
            local = torch.tensor(
                [i for i, g in enumerate(glob.tolist()) if g in set(self.locked_ids.tolist())],
                dtype=torch.long, device=self.locked_ids.device)
            if local.numel() == 0:
                continue

            orig = act.compute
            hold = self._hold_pos
            mode, kp, kd, ld = self.mode, self.stuck_stiffness, self.stuck_damping, self.limp_damping

            def patched(control_action, joint_pos, joint_vel, _o=orig, _l=local):
                out = _o(control_action, joint_pos, joint_vel)
                if mode == "limp":
                    out.joint_efforts[:, _l] = -ld * joint_vel[:, _l]
                else:
                    out.joint_efforts[:, _l] = (kp * (hold - joint_pos[:, _l])
                                                - kd * joint_vel[:, _l])
                return out

            act.compute = patched
            self._patched.append((act, orig))

    # ------------------------------------------------------------------
    def maybe_apply(self, robot, step: int) -> None:
        """Engage damage if active. Call once per control step, BEFORE env.step().

        The gain override is written once, on the first active step. Thereafter
        only the position target is re-asserted, because the action manager
        rewrites targets for all joints every step and would otherwise let the
        policy drive the damaged leg again.
        """
        if not self._resolved:
            raise RuntimeError("[LegLock] call resolve(robot) before maybe_apply().")
        if not self.is_active(step):
            return

        if not self._applied:
            self._apply_damage(robot)
            self._applied = True
            if self.verbose:
                print(f"[LegLock] damage engaged at step {step} ({self.mode}).")

    # ------------------------------------------------------------------
    def _apply_damage(self, robot) -> None:
            """Damage the joints by overriding the actuator's computed efforts.

            ActuatorNetMLP ignores stiffness/damping, so write_joint_stiffness_to_sim
            is a no-op on this robot. Wrapping compute() is the only reliable route,
            and it keeps the joint inside the solver so contacts still resolve.
            """
            if self.mode == "stuck" and self.hold_position == "current":
                # Capture the pose at the damage instant: no step change in target.
                self._hold_pos = robot.data.joint_pos[:, self.locked_ids].clone()

            self._patch_actuator(robot)
    # ------------------------------------------------------------------

    def restore(self, robot) -> None:
            """Undo the actuator patch. Call before reusing the env."""
            if not self._resolved or self.lock_step is None or not self._applied:
                return
            for act, orig in self._patched:
                act.compute = orig
            self._patched.clear()
            self._applied = False
            if self.verbose:
                print("[LegLock] actuator restored.")

    # ------------------------------------------------------------------
    def mask_actions(self, actions: torch.Tensor, step: int) -> torch.Tensor:
        """OPTIONAL: zero the policy's commands on damaged joints once active.

        Not required — the gain override already decouples the joint from the
        policy — but it keeps the EMA action filter from integrating commands
        that the damaged joints cannot follow. Out-of-place. If used, apply it
        identically in every condition.
        """
        if not self.is_active(step):
            return actions
        actions = actions.clone()
        actions[:, self.locked_ids] = 0.0
        return actions