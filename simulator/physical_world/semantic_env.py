# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SemanticEnv -- the online, per-command environment shell for the RL loop.

A thin scaffold that changes NO behavior. ``SemanticEnv`` owns the cross-command scheduler state
(``semantic_executor.SchedState``) and exposes ``reset()`` / ``run_goal()`` / ``step()``. ``run_goal``
runs ONE already-compiled goal through the shared producer (via ``semantic_executor.step_one``) and
ACCUMULATES its frames on ``self.motion``; a caller renders ``self.motion`` at the end (batched), so the
frame stream is byte-identical to ``execute(all goals)``.

Deliberately NOT done here:
  * per-frame FPV observation returned from step()
  * per-command physics compliance (settle / nudge) after each command
  * relocating the whole-program door pre-pass into walk
  * online command-string grammar semantics (assistant emits every [Walk] explicitly, no auto-insert)

``run_goal`` takes an ALREADY-COMPILED ``Goal`` so the equivalence (SemanticEnv frames == execute
frames) is independent of any grammar-compile ambiguity. ``step`` is a thin command-string
convenience that uses the standard grammar.
"""

from __future__ import annotations

import command_grammar
import humanoid_motion as wpp
import isaac_goal
import semantic_executor as se
from goal_schema import Goal, Primitive, PrimitiveError
from scene_state import Role, SceneState


class SemanticEnv:
    """Owns (rig, gait, scene, sd_world) + the persistent SchedState; drives one command at a time.

    Mirrors what ``semantic_executor.execute`` does per goal, but as a stateful object so the SAME loop
    serves both a pre-authored BEHAVIOR command stream and a live RL policy (one feeds fixed commands, the
    other feeds live commands)."""

    def __init__(self, rig, g, scene: SceneState, sd_world=None) -> None:
        self.rig = rig
        self.g = g
        self.scene = scene
        self.sd_world = sd_world
        self.st: se.SchedState | None = None
        self.motion: wpp.Motion = wpp.Motion()
        self.reports: list[tuple[Goal, PrimitiveError]] = []
        # WHY each verdict came out that way, keyed by ``id(goal)`` to mirror ``_record``'s identity
        # keying. Parallel to ``reports`` rather than folded into it because external code appends
        # 2-tuples to ``reports`` directly and unpacks them positionally; widening the tuple would break
        # those. ``reports`` keeps a strong reference for the env's lifetime, so the id cannot be recycled
        # out from under an entry. Read defensively by ``episode_runtime._episode_extra``.
        self.report_detail: dict[int, str] = {}
        self.report_frame_span: dict[int, tuple[int, int]] = {}
        self.report_state: dict[int, dict[str, object]] = {}

    def reset(self, start_xy: tuple[float, float] | None = None) -> dict:
        """Initialize the cross-command state. Mirrors execute()'s make_state + natural_arm setup so a
        run driven through run_goal() is byte-identical to execute(all goals)."""
        if getattr(self.scene, "_episode_baseline", None) is not None:
            self.scene.restore_episode_baseline()
        self.st = se.make_state(self.rig, self.scene, start_xy)
        self.motion = wpp.Motion()
        self.reports = []
        self.report_detail = {}
        self.report_frame_span = {}
        self.report_state = {}
        if getattr(self.scene, "natural_arm", False):
            self.rig.natural_arm = True
        return {"n_frames": 0}

    def run_goal(self, goal: Goal) -> tuple[None, bool, dict]:
        """Run ONE already-compiled goal; accumulate its frames; return (obs, done, info).

        obs is always None (no per-frame FPV observation); info carries the per-command report + running
        frame count so an RL/eval loop can read success/failure per command."""
        if self.st is None:
            raise RuntimeError("SemanticEnv.run_goal called before reset()")
        implicit_open = self._implicit_lid_open(goal)
        if implicit_open is not None:
            _, _, open_info = self._run_goal_once(implicit_open)
            if not open_info["ok"]:
                frame_start = len(self.motion.frames)
                detail = (
                    f"automatic lid open for {implicit_open.obj_key!r} failed: "
                    f"{open_info['detail'] or open_info['reason'].value}"
                )
                self._record(goal, open_info["reason"], detail)
                self.report_frame_span[id(goal)] = (frame_start, frame_start)
                self.report_state[id(goal)] = {
                    "before": self._capture_goal_state(goal),
                    "after": self._capture_goal_state(goal),
                }
                return (
                    None,
                    False,
                    {
                        "primitive": goal.prim.value,
                        "reason": open_info["reason"],
                        "detail": detail,
                        "n_new_frames": 0,
                        "n_frames": len(self.motion.frames),
                        "ok": False,
                    },
                )
        return self._run_goal_once(goal)

    def _implicit_lid_open(self, goal: Goal) -> Goal | None:
        if goal.prim not in (Primitive.PLACE, Primitive.PLACE_INSIDE):
            return None
        target_key = str((goal.meta or {}).get("surface") or "")
        if not target_key or target_key in self.scene.open_objs:
            return None
        target = self.scene.target_view(target_key, Role.CONTAINER)
        if target is None:
            return None
        meta = target.meta or {}
        if not meta.get("auto_open_before_place") or not target.leaf_paths:
            return None
        return Goal(
            Primitive.OPEN,
            obj_key=target_key,
            meta={
                "_implicit_action": "auto_open_for_placement",
                "_trigger_primitive": goal.prim.value,
            },
        )

    def _run_goal_once(self, goal: Goal) -> tuple[None, bool, dict]:
        """Execute exactly one goal without inserting simulator-side prerequisite actions."""
        frame_start = len(self.motion.frames)
        goal.meta["_camera_frame_offset"] = frame_start
        state_before = self._capture_goal_state(goal)
        frames, reason, detail = se.step_one(
            self.rig, self.g, self.scene, self.st, goal, sd_world=self.sd_world
        )
        if frames:
            self.motion.frames.extend(frames)
        self._record(goal, reason, detail)
        self.report_frame_span[id(goal)] = (frame_start, len(self.motion.frames))
        self.report_state[id(goal)] = {
            "before": state_before,
            "after": self._capture_goal_state(goal),
        }
        info = {
            "primitive": goal.prim.value,
            "reason": reason,
            "detail": detail,
            "n_new_frames": len(frames),
            "n_frames": len(self.motion.frames),
            "ok": reason is PrimitiveError.OK,
        }
        return None, False, info

    def _capture_goal_state(self, goal: Goal) -> dict[str, object] | None:
        """Capture the event-time state needed to prove an attach error was repaired."""
        child = goal.meta.get("child") or goal.obj_key or goal.target_key
        if self.scene is None or child is None:
            return None
        ent = self.scene.entities.get(str(child))
        child = (
            ((ent.meta or {}).get("state_child") or child) if ent is not None else child
        )
        attachment = None
        if hasattr(self.scene, "attachment_status"):
            attachment = self.scene.attachment_status(str(child))
        held_key = (
            None
            if self.st is None or self.st.held is None
            else self.st.held.get("obj_key")
        )
        state: dict[str, object] = {
            "held_key": held_key,
            "attachment": attachment,
        }
        task = getattr(self.scene, "behavior_task", None) or {}
        carcass = list(task.get("carcass") or [])
        if carcass:
            relations = self.scene.relations(source="physics")
            met = sum(
                1
                for part, parent in carcass
                if isaac_goal.attached(part, parent).as_tuple() in relations
            )
            state["task_progress"] = {
                "met": met,
                "total": len(carcass),
                "fraction": round(met / len(carcass), 6),
            }
        return state

    def _record(self, goal: Goal, reason: PrimitiveError, detail: str = "") -> None:
        """Record ONE verdict per Goal: re-running the SAME Goal SUPERSEDES its earlier verdict.

        Callers retry a failed command in place -- walk to a reach-capable stand, then re-run the SAME
        ``Goal`` object (``episode_runtime.run_render``'s reach-renav, ``IsaacEnv._drive_goal``). ``reports``
        therefore holds each command's FINAL outcome, never one row per attempt: its only consumers count
        failures over it, and a command that eventually succeeded is not a failure.

        Keyed on object IDENTITY, never equality. ``Goal`` is a plain dataclass, so two independent
        commands (a program that says ``[Grab] <can>`` twice) compare EQUAL while being distinct commands
        that each deserve their own verdict -- for the same reason ``in`` / ``list.index`` must not be used
        here, as both fall back to ``==``."""
        for i, (prev_goal, prev_reason) in enumerate(self.reports):
            if prev_goal is goal:
                self.reports[i] = (goal, reason)
                self.report_detail[id(goal)] = detail
                print(
                    f"[env] retry {goal.prim.value}: "
                    f"{prev_reason.value} -> {reason.value} "
                    "(verdict superseded; the failure count reads the final outcome)"
                )
                return
        self.reports.append((goal, reason))
        self.report_detail[id(goal)] = detail

    def step(self, command_str: str) -> list[tuple[None, bool, dict]]:
        """Compile ONE command string (EXISTING grammar) and run its goal(s); returns per-goal results.

        Uses ``command_grammar.compile_program`` unchanged, so a command may expand to an
        auto-inserted navigate + the action. The online 'assistant controls every walk explicitly'
        semantics is deliberately not applied here."""
        goals = command_grammar.compile_program(command_str, self.scene)
        return [self.run_goal(goal) for goal in goals]


def _selftest() -> int:
    """Offline (no Isaac, no scene) proof of the ONE-verdict-per-Goal bookkeeping in ``_record``."""
    from unittest import mock

    from goal_schema import Primitive
    from scene_state import Role

    ok = True

    def _chk(cond, msg) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    def _n_fail(env: SemanticEnv) -> int:
        return sum(1 for _, r in env.reports if r is not PrimitiveError.OK)

    env = SemanticEnv(None, None, None)

    walk = Goal(Primitive.NAVIGATE_TO, target_key="shelf")
    attach = Goal(Primitive.ATTACH, meta={"parent": "back"})
    env._record(walk, PrimitiveError.OK)
    env._record(attach, PrimitiveError.OUT_OF_REACH)
    _chk(_n_fail(env) == 1, "a failed command counts once")

    # the reach-renav shape: an inserted walk, then a re-run of the SAME attach Goal object
    env._record(Goal(Primitive.NAVIGATE_TO, target_xy=(1.0, 2.0)), PrimitiveError.OK)
    env._record(attach, PrimitiveError.OK)
    _chk(
        _n_fail(env) == 0,
        "a SUCCESSFUL retry supersedes the failure (no phantom n_fail)",
    )
    _chk(len(env.reports) == 3, "the retry replaces its row instead of appending a 4th")
    _chk(
        env.reports[1][0] is attach and env.reports[1][1] is PrimitiveError.OK,
        "the superseded row keeps its issue-order position and holds the FINAL verdict",
    )

    env2 = SemanticEnv(None, None, None)
    again = Goal(Primitive.ATTACH, meta={"parent": "back"})
    env2._record(again, PrimitiveError.OUT_OF_REACH, "seat_err=812mm")
    env2._record(again, PrimitiveError.PLACE_UNSTABLE, "EJECTED |v|=1.8 m/s")
    _chk(_n_fail(env2) == 1, "a retry that ALSO fails counts ONCE, not twice")
    _chk(
        env2.reports[-1][1] is PrimitiveError.PLACE_UNSTABLE,
        "a re-failed retry reports its LAST reason",
    )
    # The reason is a CATEGORY shared by several unrelated emitters; the detail is the only thing that
    # says which one fired. It must survive a supersede in lockstep with the reason, or the manifest
    # would pair a fresh verdict with a stale explanation -- worse than reporting none.
    _chk(
        env2.report_detail[id(again)] == "EJECTED |v|=1.8 m/s",
        "the detail is superseded in lockstep with the reason (never a stale WHY)",
    )

    # Goal is a plain dataclass: distinct commands compare EQUAL, so identity is the only sound key.
    env3 = SemanticEnv(None, None, None)
    grab_a = Goal(Primitive.PICK, obj_key="can1")
    grab_b = Goal(Primitive.PICK, obj_key="can1")
    _chk(
        grab_a == grab_b and grab_a is not grab_b, "two distinct commands compare EQUAL"
    )
    env3._record(grab_a, PrimitiveError.GRASP_FAILED)
    env3._record(grab_b, PrimitiveError.GRASP_FAILED)
    _chk(
        len(env3.reports) == 2 and _n_fail(env3) == 2,
        "two EQUAL-but-distinct commands keep two verdicts (identity, not equality)",
    )

    # externally appended rows (episode_runtime / IsaacEnv record PATH_BLOCKED without running the goal)
    env4 = SemanticEnv(None, None, None)
    blocked = Goal(Primitive.NAVIGATE_TO, target_key="fridge")
    env4.reports.append((blocked, PrimitiveError.PATH_BLOCKED))
    env4._record(Goal(Primitive.PICK, obj_key="can1"), PrimitiveError.OK)
    _chk(_n_fail(env4) == 1, "a caller-appended PATH_BLOCKED row still counts")

    reset_scene = SceneState()
    reset_scene.register("tray", Role.GRASPABLE, xy=(1.0, 2.0), z=0.5)
    reset_scene.register("nested", Role.GRASPABLE, xy=(1.0, 2.0), z=0.55)
    reset_scene.declare_inside("nested", "tray")
    reset_scene.capture_episode_baseline()
    reset_scene.declare_pick("nested")
    reset_scene.set_object_rest("tray", (4.0, 5.0, 0.5))
    env5 = SemanticEnv(None, None, reset_scene)
    with mock.patch.object(se, "make_state", return_value=object()):
        result = env5.reset()
    _chk(
        result == {"n_frames": 0}
        and reset_scene.inside == {"nested": "tray"}
        and reset_scene.entities["tray"].xy == (1.0, 2.0),
        "reset restores the episode's movable-object pose and containment baseline",
    )

    print(f"[semantic_env] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())
