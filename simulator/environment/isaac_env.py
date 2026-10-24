"""Persistent Isaac environment for frozen InteractiveWorld TaskPacks.

The environment boots one SimulationApp, supports deterministic reset and scene
switching, executes one semantic command at a time, renders egocentric observations,
and reports physical predicates and score state.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import importlib
import logging
import math
import os
import statistics
import time
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, ExitStack
from typing import TYPE_CHECKING

_MAX_DENSE_RENDER_CACHE_BYTES = 128 * 1024 * 1024

from . import reset as _reset
from .interaction_placement import settle_or_lock_taskpack_placement
from .runtime_builder_spec import register_runtime_builder_spec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scene_state import SceneState

logger: logging.Logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _LastDenseRenderDelivery:
    scene_identity: int
    frame_identity: int
    frame_index: int | None
    world_epoch: int
    rgb: object
    eye_pose: tuple | None
    segment: str
    delivered_pose: object


# Dump every frame handed to the VLM into this directory (empty = off). Diagnostic only.
_FRAME_DUMP = os.getenv("ISAAC_FRAME_DUMP", "")
if _FRAME_DUMP:
    os.makedirs(_FRAME_DUMP, exist_ok=True)

# ---- visible-set MEASUREMENT (ISAAC_VISCHECK) ----------------------------------------------------
# Beyond this the object stops being a usable navigation landmark: at 8 m a 30 cm object subtends ~2 deg,
# and every scene in the benchmark fits inside it (the largest BEHAVIOR room diagonal is ~12 m, but a
# landmark across two rooms is wall-occluded anyway).
_VIS_MAX_DIST_M = 8.0
# Apparent-size floor, in pixels of the 960x760 FPV. A Qwen3-VL patch is 16 px, so anything below half a
# patch cannot survive patch embedding as a distinguishable object -- calling a 3-px speck "seen" would
# inflate the visible set with exactly the far-wall windows the measurement is trying to count out.
_VIS_MIN_APPARENT_PX = 8.0
# Both ENDS of an occlusion ray are skipped by this much. A window, a sliding door and any wall-mounted
# fixture have AABBs that OVERLAP the wall they hang on, so sampling the far endpoint would declare every
# one of them occluded by itself (bt_putting_away_toys registers 30 windows + 13 sliding doors). The cost
# is that an object within this margin BEHIND a wall reads as visible.
_VIS_WALL_MARGIN_M = 0.20
_VIS_WALL_MAX_SAMPLES = 96  # caps the per-ray march (8 m at a 0.10 m nav cell = 80)
_VIS_NAMES_MAX = 32  # visible names carried on the obs before truncation
# ISAAC_VIS_GATE: the A/B arm. 0 = measure only, the person is handed the whole registry (~290 names on
# bt_putting_away_toys) and the "only act on what you see" rule is honour-system prompt text. 1 = the
# person is handed ONLY what is in view, so an unseen id is not merely forbidden, it is unsayable.
# Read once at import: the measurement it gates must not change shape mid-episode.
_VIS_GATE = (os.getenv("ISAAC_VIS_GATE") or "0").strip().lower() in ("1", "true", "on")
PLACEMENT_SETTLE_ATTEMPTS: int = 3
PLACE_INSIDE_SETTLE_ATTEMPTS: int = PLACEMENT_SETTLE_ATTEMPTS

_PHASE_PREDICATE_ALIASES = {
    "is_raised": "raised",
    "is_unmoved": "unmoved",
    "ontop": "on",
    "remain_unmoved": "unmoved",
}
_PHASE_PREDICATE_ARITIES = {
    "held": 1,
    "inside": 2,
    "near": 2,
    "on": 2,
    "open": 1,
    "raised": 1,
    "running": 1,
    "unmoved": 1,
}


def _phase_relation_shape(value: object) -> tuple[str, tuple[str, ...]] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    parts = list(value)
    if len(parts) != 3:
        return None
    name = str(getattr(parts[0], "name", parts[0])).rsplit(".", 1)[-1].lower()
    name = _PHASE_PREDICATE_ALIASES.get(name, name)
    entities = [str(parts[1])]
    if parts[2] not in (None, "", "None"):
        entities.append(str(parts[2]))
    return name, tuple(entities)


def _has_live_relation(relations, predicate: str, subject: str, target: str) -> bool:
    return any(
        getattr(pred, "name", str(pred)).upper().endswith(predicate.upper())
        and str(left) == subject
        and str(right) == target
        for pred, left, right in relations
    )


def _has_live_inside(relations, subject: str, target: str) -> bool:
    return _has_live_relation(relations, "INSIDE", subject, target)


def _mark_placement_retry(
    meta: dict, failed_xy, *, inside: bool, reject_space: bool = False
) -> int:
    import semantic_executor as se

    return se.mark_placement_retry(
        meta,
        failed_xy,
        inside=inside,
        reject_space=reject_space,
    )


def _mark_inside_retry(meta: dict, failed_xy) -> int:
    return _mark_placement_retry(meta, failed_xy, inside=True, reject_space=True)


def _replace_goal_report(reports: list, goal, reason) -> None:
    for index in range(len(reports) - 1, -1, -1):
        if reports[index][0] is goal:
            reports[index] = (goal, reason)
            return
    reports.append((goal, reason))


def _next_command_action_followup(
    command_grammar,
    navigation_goal,
    next_command: object,
    scene,
    *,
    held_key: str | None = None,
):
    """Match the whole-program action lookahead for one online navigation goal."""
    if not isinstance(next_command, str) or not next_command.strip():
        return None
    if navigation_goal.prim is not command_grammar.Primitive.NAVIGATE_TO:
        return None
    next_goals, next_commands = command_grammar.compile_program_with_commands(
        next_command,
        scene,
        held_key,
    )
    if not next_commands:
        return None
    current_command = command_grammar.CompiledCommand(
        index=0,
        source="",
        goal_start=0,
        goal_end=1,
    )
    shifted_commands = tuple(
        command_grammar.CompiledCommand(
            index=index + 1,
            source=command.source,
            goal_start=command.goal_start + 1,
            goal_end=command.goal_end + 1,
        )
        for index, command in enumerate(next_commands)
    )
    return command_grammar.navigation_action_followup(
        [navigation_goal, *next_goals],
        (current_command, *shifted_commands),
        0,
        scene,
    )


def _same_frozen_command(expected: object, actual: str) -> bool:
    expected_text = str(expected).strip()
    actual_text = actual.strip()
    if expected_text == actual_text:
        return True
    try:
        import command_grammar

        expected_verb, _expected_params = command_grammar.parse_line(expected_text)
        actual_verb, _actual_params = command_grammar.parse_line(actual_text)
        expected_prim = command_grammar._VERB.get(expected_verb)
        actual_prim = command_grammar._VERB.get(actual_verb)
    except (AttributeError, TypeError, ValueError):
        return False
    expected_tail = expected_text.partition("]")[2].strip()
    actual_tail = actual_text.partition("]")[2].strip()
    return (
        expected_prim is not None
        and expected_prim is actual_prim
        and expected_tail == actual_tail
    )


def _import_opt(name: str):
    """Import ``name``, or return None if it cannot be imported. Used only by the setup-phase timing
    patches, where a missing module must degrade to "that phase is not timed" rather than kill a switch.
    """
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - instrumentation must never break setup
        logger.warning("setup timing: import %s failed (%s) -> not timed", name, exc)
        return None


class IsaacEnv:
    """One warm SimulationApp serving many episodes of ONE task/scene.

    Lifecycle: ``open()`` boots + builds + settles + validates + snapshots the pristine start; ``reset()``
    restores that start (symbolic + USD poses + nav grid) and re-inits the scheduler; ``score()`` reads the
    task goal off the physical relations; ``close()`` tears the app down (deliberately, once).
    """

    def __init__(
        self,
        scene: str,
        seed: int = 0,
        tag: str | None = None,
        active_gpu: int = 0,
        render_obs: bool = False,
        collect_camera: bool = True,
        goal_override: list | None = None,
        strict_acceptance: bool = False,
    ) -> None:
        self.scene = scene
        self.builder_spec_digest: str | None = None
        self.seed = seed
        self.tag = tag
        self.active_gpu = int(
            active_gpu
        )  # sim-server pool pins each server to its own GPU (RL topology)
        # render_obs: attach a per-observation egocentric FPV frame (pixels). Off by default (the symbolic
        # smokes/determinism test stay cheap -- no RTX warmup); the RL sim-server turns it ON. A multimodal
        # obs also gives Qwen3-VL a valid image_grid_thw/mrope (the text-only actor forward crashed).
        self.render_obs = bool(render_obs)
        # Camera recurrence is needed for pixels and for the optional visibility telemetry, but the
        # deterministic acceptance replay consumes neither. Keep it independently switchable so that
        # no-render gates retain every physical/compliance frame without paying for camera-only state.
        self.collect_camera = bool(render_obs or collect_camera)
        self.strict_acceptance = bool(strict_acceptance)
        self.ctx: dict | None = None
        self.app = None
        self.stage = None
        self.scene_obj: "SceneState | None" = None
        self.env = None  # semantic_env.SemanticEnv (the online per-command driver)
        self.render_start_xy: tuple[float, float] | None = None
        self.snap: _reset.EpisodeSnapshot | None = None
        # A SYNTHESIZED INSIDE goal (list of (obj_key, container_key)) overriding the scene's static BDDL
        # goal -- the dynamic TaskPack path. None = the scene's own
        # task goal. Applied at open(); re-targetable on a WARM env via set_goal()/switch_scene(goals=...).
        self.goal_override: list | None = list(goal_override) if goal_override else None
        self.goal_preds: list = []
        self._baseline_goal_indices: set[int] = set()
        # The PLACE verb the scene's ground-truth program uses for the container: "Put" (drop into an OPEN
        # bin / onto a surface -> PLACE) vs "PutInside" (reach into an interior SPACE -> PLACE_INSIDE). These
        # are DISTINCT executor producers, so telling the human the wrong one means the object never lands ->
        # zero reward. Derived from the program at open() so the human prompt suggests the verb that WORKS.
        self.put_verb: str = "PutInside"
        # Render N egocentric frames uniformly sampled across each command's motion. Evaluation can opt into
        # selected-only rendering: the renderer then visits exactly the frames delivered to the models instead
        # of replaying and reading every dense animation frame before discarding most of them.
        self.n_seg_frames: int = max(1, int(os.getenv("ISAAC_FRAMES_PER_SEG", "1")))
        self.strict_uniform_action_frames: bool = False
        self.render_selected_only: bool = os.getenv(
            "ISAAC_RENDER_SELECTED_ONLY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._seg_start: int | None = (
            None  # motion.frames index where the last step()'s segment began
        )
        self._render_seg_start_poses: dict[str, object] | None = None
        self._temporal_reset_pending = True
        self._dynamic_render_proxy = None
        self._dynamic_owned_paths: set[str] = set()
        self._dynamic_free_focus = None
        self._dynamic_focus_path: str | None = None
        self._clutter_render_replay = None
        # /World/Body mesh authored yet? setup_episode wires _body_path but never authors the visible human
        # mesh (RL training renders no body), so _render_one authors it lazily on the first render_obs frame
        # (then rewrites points per frame) -> the FPV shows the wearer's arm/hand, not a floating object.
        self._body_authored = False
        # PURE-MEASUREMENT phase timing (no behavior change): phase-name -> list of wall durations.
        # Shipped per operation in the simulator server's reply for profiling.
        # envelope (isaac_sim_server._dispatch -> meta.phases) so a MAST rollout can be cost-attributed.
        self._t: dict[str, list[float]] = {}
        self._counters: dict[str, int] = {}
        # Per-episode compliance-FK cache (see _fk_sync): the realized joints of every motion frame, the
        # exact Frame object each was realized from (staleness guard), and the running nudge_state after
        # each frame (exact resume point). Invalidated by open/reset/switch_scene/set_goal.
        self._fk_j: list = []
        self._fk_obj: list = []
        self._fk_fr: list = []
        self._fk_post: list = []
        self._fk_cam: list = []
        self._fk_cam_post: list = []
        self._fk_openable_post: list = []
        self._fk_openable_consumed: list[bool] = []
        self._fk_nudge_boxes_cache: dict[tuple, tuple] = {}
        self._fk_env: tuple | None = None
        self._fk_grid = None
        self._fk_wall_sdf = None
        self._fk_static_boxes = None
        self._camera_adapter = None
        # Per-step visibility MEASUREMENT inputs (see the ISAAC_VISCHECK section): the camera pose of every
        # frame the assistant was actually shown this step, the pose _render_one last aimed with, and the
        # command that produced them. Cleared at every step/episode boundary.
        self._eye_poses: list = []
        # The motion-segment tag of every frame the assistant was actually shown this step, parallel to
        # the b64 list in obs["frames"]. The frame buffer needs it to tell an inspect's HOLD from its
        # RELEASE; without it keep_final pins the release and the cue is dropped with nothing recording
        # that it was (ego_frames.CUE_HOLD_SEGS).
        self._frame_segs: list = []
        self._last_eye: tuple | None = None
        self._last_delivered_pose = None
        self._last_canonical_render_pose = None
        self._online_render_trace = None
        self._render_parity_trace_status = "inactive"
        self._render_command_spans: list[tuple[int, int]] = []
        self._online_trace_noted_commands = 0
        self._render_delivery_cache: dict[tuple, tuple] = {}
        self._dense_render_delivery_cache: dict[int | None, tuple] = {}
        self._dense_render_delivery_bytes = 0
        self._dense_render_delivery_complete = True
        self._rendered_dense_key: tuple | None = None
        self._last_dense_render_delivery: _LastDenseRenderDelivery | None = None
        self._last_cmd: str = ""
        self._visual_state_events = None
        self._visual_state_replay: dict = {}
        self._visual_state_initial_poses: dict[str, dict[str, object]] = {}
        self._visual_state_changed_this_command = False
        # The UNtruncated visible set of the CURRENT step, or None when the pass did not run. Under
        # _VIS_GATE this is the person's whole vocabulary, so None must mean "fall back", never "empty".
        self._vis_full: set | None = None
        self._vis_gate_fallbacks: int = 0
        # This command's per-goal outcomes, mirroring semantic_env.reports one-verdict-per-Goal but keeping
        # the plain-English ``detail`` the enum throws away. (goal, record) pairs: the goal reference pins
        # the object so identity comparison stays valid for the whole step.
        self._step_out: list[tuple[object, dict]] = []
        self._authored_trace: list[dict[str, object]] = []
        self._trace_current_at: set[str] = set()
        self._reset_counters()

    # ---- phase timing ---------------------------------------------------------------------------
    @contextmanager
    def _timed(self, name: str) -> "Iterator[None]":
        """Time the block and append the duration under ``name``. Phases NEST (``step.total`` contains
        ``step.run_goal`` which contains nothing else measured), so never sum across levels -- subtract.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._t.setdefault(name, []).append(time.perf_counter() - t0)

    def _note_time(self, name: str, dt: float) -> None:
        """Append an ALREADY-MEASURED duration under ``name`` -- for a span accumulated in pieces (the
        stand-pick cache bookkeeping brackets the scan, so a single ``with`` cannot express it) that must
        still show up as ONE sample per call, not two."""
        self._t.setdefault(name, []).append(dt)

    def timings(self, clear: bool = False) -> dict[str, dict]:
        """Per-phase stats ``{n, total_s, mean_s, p50_s, max_s}`` keyed by phase name, plus a ``"counters"``
        entry (motion-frame gauges -- NOT a phase; skip it when iterating the table). ``clear`` drops the
        recorded durations (the sim-server clears per op so each reply carries only that op's phases); the
        counters survive a clear and are re-zeroed by open()/reset()/switch_scene()."""
        out: dict[str, dict] = {
            name: {
                "n": len(xs),
                "total_s": sum(xs),
                "mean_s": statistics.mean(xs),
                "p50_s": statistics.median(xs),
                "max_s": max(xs),
            }
            for name, xs in self._t.items()
            if xs
        }
        out["counters"] = dict(self._counters)
        if clear:
            self._t.clear()
        return out

    def _reset_counters(self) -> None:
        """Zero the per-episode motion-frame gauges (the motion buffer is cleared by open/reset/switch),
        the physics-epoch counter, the stand-pick audit log and the stand-pick cache."""
        self._counters = {
            "n_steps": 0,
            "motion_frames_last": 0,
            "motion_frames_total": 0,
            # stand-pick cache gauges (see _pick_stand_cached): calls, safe-hits, "everything but cur_xy
            # matched but cur_xy is not EXACTLY the cached stand", and VERIFY-mode disagreements.
            "standpick_call": 0,
            "standpick_hit": 0,
            "standpick_near": 0,
            "standpick_mismatch": 0,
            "camera_yaw_clamps": 0,
            "camera_pitch_clamps": 0,
            "fpv_readback_retries": 0,
        }
        # Bumped every time a command's settle moves objects to a new physical rest -- the cheap "the world
        # changed" identity in the stand-pick fingerprint (a stand pick reads the LIVE object poses).
        self._world_epoch = 0
        self._stand_picks: list[tuple[tuple, tuple | None]] = []
        # fingerprint-minus-cur_xy -> the stand that scan returned. MUST be dropped wherever _world_epoch
        # is (an episode/scene boundary), which is why it lives here and not in __init__.
        self._standpick_cache: dict[tuple, tuple] = {}
        self._last_delivered_pose = None
        self._last_canonical_render_pose = None
        self._last_dense_render_delivery = None
        self._render_command_spans = []
        self._online_trace_noted_commands = 0
        self._preflight_command_cursor: int | None = 0

    def _note_motion_frames(self) -> None:
        """Record how many motion frames the command just driven produced. The cumulative total is the
        suspected reward-side cost driver: ``_compliance`` runs a per-frame FK pass over EVERY accumulated
        frame, so its cost grows with episode length."""
        motion = getattr(self.env, "motion", None)
        n_all = len(motion.frames) if motion is not None else 0
        self._counters["n_steps"] = int(self._counters.get("n_steps", 0)) + 1
        self._counters["motion_frames_last"] = int(n_all - int(self._seg_start or 0))
        self._counters["motion_frames_total"] = int(n_all)

    # ---- lifecycle ------------------------------------------------------------------------------
    @contextmanager
    def _memo_footprints(self, prefix: str = "switch") -> "Iterator[None]":
        """Memoize ``init_validation.collidable_footprints`` for the duration of ONE
        ``episode_runtime._validate_init_for_render`` call -- and not one instruction longer.

        The init gate reads the footprint table FOUR times per call (``collision_world.build``,
        ``place_start``, and ``nav_grid.build_nav_grid``'s region + furniture layers) and nothing writes to
        the stage in between, so three of those four reads recompute a byte-identical table. Each read is a
        full ``Usd.PrimRange(TraverseInstanceProxies)`` with a ``ComputeWorldBound`` per collider prim --
        seconds on a BEHAVIOR scene. The key MUST include ``exclude_prefixes`` and ``group_by_owner``:
        the nav furniture layer excludes the walls, while the collision oracle requests per-collider
        rows rather than owner unions, and both legitimately produce different tables.

        THE SCOPE IS THE CORRECTNESS ARGUMENT. The same function is called on the RUNTIME path
        (``init_validation.pick_stand_runtime`` -> ``_stand_gate_setup``) against a stage that HAS changed:
        objects settle and door leaves swing, and a swung leaf's world AABB moves (doors are not in the
        exclude set). A memo leaking into that path would hand back stale footprints and silently corrupt
        stand selection. So the cache is armed only INSIDE ``_validate_init_for_render`` and cleared on both
        entry and exit -- which is why that function is swapped too, and which also makes the seed re-sample
        loop's second call re-read the stage from scratch.

        ``ISAAC_FOOTPRINT_MEMO_VERIFY=1`` re-runs the real function on EVERY cache hit and requires the
        result to equal the cached one -- a complete proof rather than a sample.

        Module-attribute swaps restored in ``finally``: ``switch_scene``
        already uses for ``fpv_core.launch_app`` -- no simulator.physical_world edit.

        PURE MEASUREMENT (``prefix``): every REAL footprint read is timed as
        ``<prefix>.collidable_footprints`` and every cache hit as
        ``<prefix>.collidable_footprints_hit``, so the phase table's ``n`` gives the miss/hit split
        directly. The timers live HERE rather than in ``_timed_setup_internals`` because only this scope
        can see which branch a call took -- an outer wrapper sees one undifferentiated call.
        """
        import inspect

        import episode_runtime
        import init_validation as iv

        orig = iv.collidable_footprints
        orig_gate = episode_runtime._validate_init_for_render
        # Mirror the real default instead of re-declaring it, so a caller that passes the default value
        # explicitly (nav's furniture layer does, on a scene with no wall prims) hits the same entry.
        default_excl = inspect.signature(orig).parameters["exclude_prefixes"].default
        default_grouping = inspect.signature(orig).parameters["group_by_owner"].default
        verify = bool(os.getenv("ISAAC_FOOTPRINT_MEMO_VERIFY"))
        cache: dict = {}
        armed = [False]
        stat = {"hit": 0, "miss": 0, "hit_s": 0.0}

        ph_miss, ph_hit = (
            f"{prefix}.collidable_footprints",
            f"{prefix}.collidable_footprints_hit",
        )

        def memo(
            stage,
            exclude_prefixes=default_excl,
            group_by_owner=default_grouping,
        ):
            if not armed[0]:
                with self._timed(ph_miss):  # outside the gate: pure pass-through
                    return orig(
                        stage,
                        exclude_prefixes,
                        group_by_owner=group_by_owner,
                    )
            key = (id(stage), tuple(exclude_prefixes), bool(group_by_owner))
            ent = cache.get(key)
            if ent is None:
                with self._timed(ph_miss):
                    fps = orig(
                        stage,
                        exclude_prefixes,
                        group_by_owner=group_by_owner,
                    )
                # Cache a COPY and hand the caller the original list, so no consumer can mutate the cached
                # table; the stage is retained so its id() cannot be recycled under a live key.
                cache[key] = (stage, list(fps))
                stat["miss"] += 1
                return fps
            with self._timed(ph_hit):
                stat["hit"] += 1
                if verify:
                    t0 = time.perf_counter()
                    fresh = orig(
                        stage,
                        exclude_prefixes,
                        group_by_owner=group_by_owner,
                    )
                    stat["hit_s"] += time.perf_counter() - t0
                    if fresh != ent[1]:
                        raise AssertionError(
                            f"footprint memo MISMATCH exclude={tuple(exclude_prefixes)} "
                            f"group_by_owner={bool(group_by_owner)}: cached "
                            f"{len(ent[1])} owners vs fresh {len(fresh)} -- the stage changed under the memo"
                        )
                return list(ent[1])

        def gate(*a, **kw):
            cache.clear()
            armed[0] = True
            try:
                return orig_gate(*a, **kw)
            finally:
                armed[0] = False
                cache.clear()

        iv.collidable_footprints = memo
        episode_runtime._validate_init_for_render = gate
        try:
            yield
        finally:
            iv.collidable_footprints = orig
            episode_runtime._validate_init_for_render = orig_gate
            logger.info(
                "footprint memo: %d hit / %d miss%s",
                stat["hit"],
                stat["miss"],
                (
                    f" (VERIFY: every hit re-checked against a fresh read, "
                    f"{stat['hit_s']:.2f}s of recompute avoided)"
                    if verify
                    else ""
                ),
            )

    # ``setup_episode`` is ONE opaque phase (~206s on a heavy scene on MAST) and the local 2.1x cut of the
    # init gate barely moved it, so the cost is somewhere else INSIDE it. These are the module functions
    # ``setup_episode`` (and everything it calls) dispatch through, wrapped in a ``_timed`` phase for the
    # duration of that call only -- (module, attribute, phase suffix). PURE MEASUREMENT.
    _SETUP_PHASES: tuple[tuple[str, str, str], ...] = (
        ("episode_runtime", "_validate_init_for_render", "validate_init_for_render"),
        ("episode_runtime", "_apply_scene_colliders", "apply_scene_colliders"),
        ("episode_runtime", "_auto_action_layout", "auto_action_layout"),
        # the SSOT collision oracle: a full precise-geometry pass over the stage, reached from BOTH the
        # init gate and setup_episode's own settle block (cached after the first real build).
        ("episode_runtime", "_ensure_cworld", "ensure_cworld"),
        ("episode_runtime", "_place_door", "place_door"),
        ("nav_grid", "build_task_nav_grid", "build_task_nav_grid"),
        ("nav_grid", "build_nav_grid", "build_nav_grid"),
        ("init_validation", "place_start", "place_start"),
        ("init_validation", "pick_stand_runtime", "pick_stand_runtime"),
        ("smplh_anim_core", "derive_rest_place_arms", "derive_rest_place_arms"),
        ("object_physics", "settle_object_on_support", "settle_object_on_support"),
    )

    @contextmanager
    def _timed_setup(self, prefix: str) -> "Iterator[None]":
        """The whole instrumentation bracket around ONE ``episode_runtime.setup_episode`` call: the umbrella
        ``<prefix>.setup_episode`` phase, the footprint memo, and the per-function sub-phases. ExitStack
        (not a nested ``with``) so the three are entered/exited LIFO at one indent level.
        """
        with ExitStack() as stack:
            stack.enter_context(self._timed(f"{prefix}.setup_episode"))
            stack.enter_context(self._memo_footprints(prefix))
            stack.enter_context(self._timed_setup_internals(prefix))
            yield

    @contextmanager
    def _timed_setup_internals(self, prefix: str) -> "Iterator[None]":
        """Split the opaque ``<prefix>.setup_episode`` phase into ``<prefix>.<fn>`` sub-phases, for the
        duration of ONE ``setup_episode`` call and not one instruction longer.

        Same contained module-attribute-swap idiom this file already uses three times
        (``fpv_core.launch_app`` in ``switch_scene``, ``smplh_fpv.fwd`` in ``_joints_only_fk``, and the
        ``collidable_footprints`` memo) -- no simulator.physical_world edit, everything restored in ``finally``.

        PURE MEASUREMENT: each wrapper forwards ``*args``/``**kwargs`` and the return value untouched and
        adds only a ``perf_counter`` pair. A module that will not import or an attribute that is missing or
        not callable degrades to "not timed" (a warning), never to an exception.

        PHASES NEST -- never sum them. ``validate_init_for_render`` contains ``apply_scene_colliders``,
        ``ensure_cworld``, ``place_start`` and ``build_task_nav_grid``; ``build_task_nav_grid`` contains
        ``build_nav_grid``; ``place_start`` and ``build_nav_grid`` both contain ``collidable_footprints``.
        ``n`` is as informative as ``total_s`` here (init-gate re-samples, per-object settle candidates,
        per-witness stand scans)."""
        undo: list = []
        for mod_name, attr, phase in self._SETUP_PHASES:
            mod = _import_opt(mod_name)
            if mod is not None:
                self._patch_timed(mod, attr, f"{prefix}.{phase}", undo)
        self._patch_scene_builders(prefix, undo)
        try:
            yield
        finally:
            for mod, name, fn in reversed(undo):
                setattr(mod, name, fn)

    def _patch_timed(self, mod, name: str, phase: str, undo: list) -> None:
        """Replace ``mod.name`` with a same-signature wrapper that records its wall time under ``phase``,
        appending the undo entry. No-op (warn) when the attribute is absent or not callable.
        """
        fn = getattr(mod, name, None)
        if not callable(fn):
            logger.warning(
                "setup timing: %s.%s missing/not callable -> not timed",
                getattr(mod, "__name__", mod),
                name,
            )
            return

        @functools.wraps(fn)
        def timed(*args, **kwargs):
            with self._timed(phase):
                return fn(*args, **kwargs)

        setattr(mod, name, timed)
        undo.append((mod, name, fn))

    def _patch_scene_builders(self, prefix: str, undo: list) -> None:
        """Time the per-scene BUILDER call as ``<prefix>.scene_builder``.

        ``setup_episode`` dispatches with ``builders = episode_runtime._scene_builders(seed=...);
        builders[args.scene](stage, app)``, and ``_scene_builders`` REBUILDS that dict on every call (it is
        a function, not a module-level dict, so the ``build_*`` names resolve at call time), so the only
        stable hook is the factory: wrap it to return the same dict with each builder timed. The wrapper
        accepts and forwards all registry arguments so seeded raw-world dispatch and test monkeypatches
        retain the original callable contract. Same keys, same builder signature, same 5-tuple return.
        """
        s3 = _import_opt("episode_runtime")
        orig = getattr(s3, "_scene_builders", None) if s3 is not None else None
        if not callable(orig):
            logger.warning("setup timing: _scene_builders missing -> builder not timed")
            return
        phase = f"{prefix}.scene_builder"

        def wrap(build):
            @functools.wraps(build)
            def timed(*args, **kwargs):
                with self._timed(phase):
                    return build(*args, **kwargs)

            return timed

        @functools.wraps(orig)
        def builders(*args, **kwargs) -> dict:
            return {key: wrap(build) for key, build in orig(*args, **kwargs).items()}

        s3._scene_builders = builders
        undo.append((s3, "_scene_builders", orig))

    def open(self) -> dict:
        """Boot + build + settle + validate the scene ONCE, then snapshot the pristine start state."""
        import episode_runtime  # heavy (pulls torch + the whole render stack); import at call time

        with self._timed("open.total"):
            with self._timed_setup("open"):
                ctx = episode_runtime.setup_episode(self._make_args())
            if not isinstance(ctx, dict):
                raise RuntimeError(
                    f"setup_episode returned an early-exit status {ctx!r} (preview/check/abort); "
                    f"IsaacEnv needs the full scene context -- check the scene/seed."
                )
            self.ctx = ctx
            self.app = ctx["app"]
            self.stage = ctx["stage"]
            self.scene_obj = ctx["scene"]
            self.env = ctx["env"]
            self._configure_visual_state_events()
            self._configure_clutter_render_replay()
            self.render_start_xy = ctx["render_start_xy"]
            if self.render_obs:
                # FPV camera + rgb annotator + timeline (+ one-time readback prime) for render_fpv(). Additive
                # episode_runtime helper; the full-render/--step3 path is untouched (it makes its own identically).
                with self._timed("open.make_fpv_products"):
                    episode_runtime.make_fpv_products(self.ctx)
                self._body_authored = bool(self.ctx.get("body_authored"))
            if self.goal_override:
                self._apply_goal_override(self.goal_override)
            self.goal_preds = self._derive_goal()
            self.put_verb = self._scene_put_verb()
            with self._timed("open.snapshot"):
                self.snap = _reset.snapshot(self.scene_obj)
            self._fk_clear()
            self._render_seg_start_poses = None
            self._temporal_reset_pending = True
            self._dynamic_render_proxy = None
            self._dynamic_owned_paths.clear()
            self._dynamic_free_focus = None
            self._dynamic_focus_path = None
            self._begin_command("")
            self._reset_authored_trace()
            self.capture_score_baseline()
            self._reset_counters()
            logger.info(
                "IsaacEnv.open scene=%s seed=%s goal_preds=%d episode_valid=%s",
                self.scene,
                self.seed,
                len(self.goal_preds),
                ctx["render_episode_valid"],
            )
            return self.observe()

    def reset(self, start_xy: tuple[float, float] | None = None) -> dict:
        """Restore the pristine start (physics-only) and re-init the scheduler for a new episode."""
        if self.snap is None or self.env is None or self.scene_obj is None:
            raise RuntimeError("IsaacEnv.reset() called before open()")
        import episode_runtime

        self.finalize_render_parity_trace()
        with self._timed("reset.total"):
            _reset.restore(self.scene_obj, self.snap, start_xy)
            self.scene_obj.egocentric_collision_fallback_used = False
            episode_runtime._apply_openable_state(
                self.ctx["controllers"],
                episode_runtime._initial_openable_state(self.ctx["controllers"]),
            )
            self.env.reset(
                start_xy=start_xy if start_xy is not None else self.render_start_xy
            )
            self._reset_visual_state_events()
            self._clock_started = False
            self._seg_start = None  # motion cleared -> tick-0 obs renders the start frame, not a segment
            self._render_seg_start_poses = None
            self._temporal_reset_pending = True
            self._dynamic_owned_paths.clear()
            self._dynamic_free_focus = None
            self._dynamic_focus_path = None
            self._reset_clutter_render_replay()
            self._fk_clear()  # env.reset() REPLACED motion.frames -> every cached frame index is stale
            # CONSERVATIVE on purpose: reset.restore only rewrites GRASPABLE prim poses (and joint_val),
            # neither of which is ever memoized, so the memo is provably still valid here. Clearing it
            # anyway costs one cold sweep per episode (<=0.35 s against a ~167 s episode) and removes a
            # correctness argument that would have to be re-derived every time _snap_entity changes.
            self.scene_obj.invalidate_bounds()
            self._begin_command("")
            self._reset_authored_trace()
            self.capture_score_baseline()
            self._reset_counters()
            return self.observe()

    # ---- warm scene switch: rebuild a DIFFERENT scene in the SAME warm app (no re-boot) ----------
    def switch_scene(
        self,
        scene: str,
        seed: int | None = None,
        goals: list | None = None,
        builder_spec: Mapping[str, object] | None = None,
        builder_spec_digest: str | None = None,
    ) -> dict:
        """Switch this WARM env to a DIFFERENT BEHAVIOR ``scene`` WITHOUT re-launching the app: tear down the
        current scene's ``/World`` prims, rebuild the new scene via ``episode_runtime.setup_episode`` (with
        ``fpv_core.launch_app`` shimmed to return the warm app -- only one SimulationApp is ever allowed),
        re-warm RTX via ``make_fpv_products``, and re-snapshot the pristine start. Proven by
        Scene-switch profiling measured ~25-73s versus a ~67-174s cold boot across many
        switches, no stale-PhysX-handle crash). Makes the sim-server pool SCENE-AGNOSTIC so the
        online-taskgen dataset can rotate scenes per GRPO step. ``goals`` (optional) applies a synthesized
        INSIDE goal to the rebuilt scene. Returns the post-switch observation."""
        if self.app is None:
            raise RuntimeError("IsaacEnv.switch_scene() called before open()")
        registered_digest = register_runtime_builder_spec(
            scene, builder_spec, builder_spec_digest
        )

        import episode_runtime
        import fpv_core

        with self._timed("switch.total"):
            # Free the PREVIOUS scene's RTX render products + rgb annotators FIRST. make_fpv_products creates
            # them per switch (fpv_render.setup_products -> rep.create.render_product) OUTSIDE /World (under
            # /Render), so _teardown_scene's RemovePrim("/World/*") never frees their GPU framebuffers/hydra
            # textures -> they accumulate every switch -> OOM on a long multi-scene run. Must run before
            # self.ctx is replaced below (that assignment drops the only handle to them).
            with self._timed("switch.teardown_fpv"):
                self._teardown_fpv_products()
            with self._timed("switch.teardown_scene"):
                self._teardown_scene()
            self.scene = scene
            self.builder_spec_digest = None
            if seed is not None:
                self.seed = int(seed)
            self.goal_override = list(goals) if goals else None

            warm_app = self.app
            orig_launch = fpv_core.launch_app
            # setup_episode always calls fpv_core.launch_app; return the warm app so no 2nd SimulationApp is
            # constructed. Contained to this call and restored in finally.
            fpv_core.launch_app = lambda cfg: warm_app  # noqa: E731
            try:
                with self._timed_setup("switch"):
                    ctx = episode_runtime.setup_episode(self._make_args())
            finally:
                fpv_core.launch_app = orig_launch
            if not isinstance(ctx, dict):
                raise RuntimeError(
                    f"setup_episode('{scene}') returned early-exit status {ctx!r}; switch_scene needs the "
                    f"full scene context (check the scene/seed)."
                )
            self.ctx = ctx
            self.app = ctx["app"]
            self.stage = ctx["stage"]
            self.scene_obj = ctx["scene"]
            self.env = ctx["env"]
            self._configure_visual_state_events()
            self._configure_clutter_render_replay()
            self.render_start_xy = ctx["render_start_xy"]
            if self.render_obs:
                # Re-create FPV products + RTX re-warm for the new geometry. The previous scene's products
                # were already detached+destroyed by _teardown_fpv_products() at the top of this method, so
                # this no longer leaks render products across switches.
                with self._timed("switch.make_fpv_products"):
                    episode_runtime.make_fpv_products(self.ctx)
                self._body_authored = bool(self.ctx.get("body_authored"))
            if self.goal_override:
                self._apply_goal_override(self.goal_override)
            self.goal_preds = self._derive_goal()
            self.put_verb = self._scene_put_verb()
            with self._timed("switch.snapshot"):
                self.snap = _reset.snapshot(self.scene_obj)
            self._clock_started = False
            self._seg_start = None
            self._render_seg_start_poses = None
            self._temporal_reset_pending = True
            self._dynamic_render_proxy = None
            self._dynamic_owned_paths.clear()
            self._dynamic_free_focus = None
            self._dynamic_focus_path = None
            self._fk_clear()  # new scene: new geometry, new SemanticEnv, new motion buffer
            self.scene_obj.invalidate_bounds()  # belt-and-braces: ctx["scene"] is a fresh SceneState today
            self._begin_command("")
            self._reset_authored_trace()
            self.capture_score_baseline()
            self._reset_counters()
            if not self.render_obs:
                self._body_authored = False
            self.builder_spec_digest = registered_digest
            logger.info(
                "IsaacEnv.switch_scene -> %s seed=%s goal_preds=%d",
                scene,
                self.seed,
                len(self.goal_preds),
            )
            self._log_switch_mem(scene)
            return self.observe()

    def _log_switch_mem(self, scene: str) -> None:
        """Gated (``ISAAC_MEM_DEBUG``) driver-level VRAM + render-product leak probe for a warm switch."""
        if not os.getenv("ISAAC_MEM_DEBUG"):
            return
        try:
            import torch

            self._switch_n = getattr(self, "_switch_n", 0) + 1
            free, total = torch.cuda.mem_get_info(self.active_gpu)
            n_rp = sum(
                1 for p in self.stage.Traverse() if p.GetTypeName() == "RenderProduct"
            )
            logger.warning(
                "MEMDBG switch#%d scene=%s gpu_used=%.0fMB render_products=%d",
                self._switch_n,
                scene,
                (total - free) / 1e6,
                n_rp,
            )
        except Exception as exc:  # noqa: BLE001 - probe must never kill a switch
            logger.warning("MEMDBG failed: %s", exc)

    def _teardown_scene(self) -> int:
        """RemovePrim every ``/World`` child except the persistent PhysX scene ``/World/PhysicsScene`` so the
        next (non-idempotent) scene build starts clean -- the builders write prims directly under /World, so
        there is no single scene-root Xform to drop. Returns the number of prims removed.
        """
        if self.stage is None:
            return 0
        root = self.stage.GetPrimAtPath("/World")
        if not root or not root.IsValid():
            return 0
        removed = 0
        for child in list(root.GetChildren()):
            if child.GetPath().pathString == "/World/PhysicsScene":
                continue
            self.stage.RemovePrim(child.GetPath())
            removed += 1
        return removed

    def _teardown_fpv_products(self) -> None:
        """Detach + destroy the CURRENT ctx's FPV/TPV render products + rgb annotators (the previous scene's)
        so their GPU memory is released before a switch recreates them via make_fpv_products. Uses the shared
        fpv_render.teardown (rep.orchestrator.wait_until_complete + annot.detach() + rp.destroy() -- the
        proven Isaac 5.1 cleanup run_render itself calls). No-op when render_obs is off or no products exist.
        """
        if not self.render_obs or not self.ctx:
            return
        import fpv_render

        rps = [self.ctx[k] for k in ("rp_f", "rp_t") if self.ctx.get(k) is not None]
        annots = [self.ctx[k] for k in ("a_f", "a_t") if self.ctx.get(k) is not None]
        if not (rps or annots):
            return
        try:
            fpv_render.teardown(rps, annots)
        except Exception as exc:  # noqa: BLE001 - cleanup must never kill a switch
            logger.warning("switch_scene: fpv product teardown raised %s", exc)

    def set_goal(self, goals: list) -> dict:
        """Re-target this WARM env to a freshly SYNTHESIZED INSIDE goal WITHOUT a scene rebuild -- the
        cheapest online-taskgen path when only the goal changes (same scene): rewrite behavior_task["goals"]
        + re-derive goal_preds, and return the new obs. Physics is untouched (a reset still restores the
        pristine start); only the SCORED goal changes. No-op (keeps the current goal) if goals is empty.
        """
        if self.scene_obj is None:
            raise RuntimeError("IsaacEnv.set_goal() called before open()")
        if not goals:
            return self.observe()
        self._apply_goal_override(goals)
        self.goal_preds = self._derive_goal()
        self._reset_visual_state_events()
        self._fk_clear()  # mandatory: a re-target is an episode boundary for anything frame-indexed
        self._seg_start = len(self.env.motion.frames) if self.env is not None else None
        self._render_seg_start_poses = None
        self._reset_clutter_render_replay()
        self._begin_command("")  # same boundary: no command produced this observation
        self._reset_authored_trace()
        self.capture_score_baseline()
        # Belt-and-braces: a re-target changes no geometry, so no cached stand can go stale from it -- but
        # this is an episode boundary and a dropped cache only ever costs hits, never correctness.
        self._standpick_cache = {}
        self._preflight_command_cursor = 0
        return self.observe()

    def _configure_visual_state_events(self) -> None:
        """Resolve task-authored visual transitions against the newly built scene."""
        import surface_visual_effects as sve
        import visual_state_events as vse

        dialogue = dict(
            (
                (getattr(self.scene_obj, "behavior_task", None) or {}).get(
                    "episode_dialogue"
                )
                or {}
            )
        )
        entity_paths = {
            str(key): str(entity.prim_path)
            for key, entity in (getattr(self.scene_obj, "entities", None) or {}).items()
            if getattr(entity, "prim_path", None)
        }
        effect_paths = sve.materialize_surface_effects(self.scene_obj, dialogue)
        self._visual_state_events = vse.OnlineVisualStateEvents(
            dialogue,
            dict(getattr(self.scene_obj, "label_prim_paths", None) or {}),
            entity_paths,
            effect_paths=effect_paths,
        )
        self._visual_state_initial_poses = self._snapshot_visual_state_poses()
        self._visual_state_replay = {}
        self._visual_state_changed_this_command = False
        self._apply_visual_state_render_frame(-1)

    def _reset_visual_state_events(self) -> None:
        controller = getattr(self, "_visual_state_events", None)
        if controller is None:
            return
        import episode_runtime

        episode_runtime._reset_visual_state_events(
            self.stage,
            self._visual_state_replay,
        )
        self._restore_visual_state_poses()
        controller.reset()
        self._visual_state_replay = {}
        self._visual_state_changed_this_command = False
        self._apply_visual_state_render_frame(-1)
        self.scene_obj.invalidate_bounds()

    def _snapshot_visual_state_poses(self) -> dict[str, dict[str, object]]:
        controller = getattr(self, "_visual_state_events", None)
        if controller is None:
            return {}
        import semantic_executor as se
        import visual_state_events as vse

        saved: dict[str, dict[str, object]] = {}
        rest_poses = getattr(self.scene_obj, "object_rest_pose", None) or {}
        entities = getattr(self.scene_obj, "entities", None) or {}
        for event in controller.templates:
            if (
                not isinstance(event, vse.EntityVisibilityEvent)
                or not event.pose_from_entities
            ):
                continue
            for key, path in zip(event.entities, event.paths, strict=True):
                if key in saved:
                    continue
                pose = se._read_world_col(self.stage, path)
                entity = entities.get(key)
                if pose is None or entity is None:
                    raise RuntimeError(
                        f"cannot snapshot visual state pose for {key!r} at {path!r}"
                    )
                saved[key] = {
                    "path": path,
                    "pose": pose.copy(),
                    "xy": copy.deepcopy(entity.xy),
                    "z": copy.deepcopy(entity.z),
                    "rest_present": key in rest_poses,
                    "rest_pose": copy.deepcopy(rest_poses.get(key)),
                }
        return saved

    def _restore_visual_state_poses(self) -> None:
        saved = getattr(self, "_visual_state_initial_poses", None) or {}
        if not saved:
            return
        import semantic_executor as se

        entities = getattr(self.scene_obj, "entities", None) or {}
        rest_poses = getattr(self.scene_obj, "object_rest_pose", None)
        if rest_poses is None:
            raise RuntimeError("scene has no object_rest_pose registry")
        for key, row in saved.items():
            if not se._write_world_col(self.stage, row["path"], row["pose"]):
                raise RuntimeError(f"cannot restore visual state pose for {key!r}")
            entity = entities.get(key)
            if entity is None:
                raise RuntimeError(
                    f"cannot restore missing visual state entity {key!r}"
                )
            entity.xy = copy.deepcopy(row["xy"])
            entity.z = copy.deepcopy(row["z"])
            if row["rest_present"]:
                rest_poses[key] = copy.deepcopy(row["rest_pose"])
            else:
                rest_poses.pop(key, None)

    def _apply_visual_state_render_frame(self, frame_index: int | None) -> None:
        controller = getattr(self, "_visual_state_events", None)
        if controller is None:
            return
        import episode_runtime

        effective_frame = int(frame_index) if frame_index is not None else -1
        if frame_index is None and getattr(
            self, "_visual_state_changed_this_command", False
        ):
            effective_frame = 0
        episode_runtime._apply_visual_state_events(
            self.stage,
            controller,
            effective_frame,
            self._visual_state_replay,
        )

    def _note_visual_state_command(
        self, command: str, outcomes: Sequence[Mapping[str, object]]
    ) -> tuple:
        controller = getattr(self, "_visual_state_events", None)
        succeeded = bool(outcomes) and all(row.get("ok") is True for row in outcomes)
        if controller is None:
            return ()
        frames = self.env.motion.frames if self.env is not None else []
        frame = max(0, len(frames) - 1)
        triggered = controller.note_command(
            command,
            succeeded=succeeded,
            frame=frame,
        )
        if not triggered:
            return ()
        import episode_runtime
        import visual_state_events as vse

        for event in triggered:
            if (
                isinstance(event, vse.EntityVisibilityEvent)
                and event.pose_from_entities
            ):
                episode_runtime._place_variant_at_source(
                    self.stage, self.scene_obj, event
                )
        self._visual_state_changed_this_command = True
        self._world_epoch += 1
        self._fk_clear()
        self.scene_obj.invalidate_bounds()
        self._render_delivery_cache = {}
        self._dense_render_delivery_cache = {}
        self._dense_render_delivery_bytes = 0
        self._dense_render_delivery_complete = True
        self._rendered_dense_key = None
        self._last_dense_render_delivery = None
        return triggered

    def _preflight_navigation_record(self, goal_index: int) -> Mapping | None:
        reachability = getattr(self.scene_obj, "build_reachability", None)
        if not isinstance(reachability, Mapping) or reachability.get("ok") is not True:
            return None
        stands = reachability.get("navigation_stands")
        if not isinstance(stands, Mapping):
            return None
        record = stands.get(goal_index)
        if record is None:
            record = stands.get(str(goal_index))
        return record if isinstance(record, Mapping) else None

    def _preflight_goal_indices_for_command(
        self,
        command_str: str,
        goals: Sequence[object],
    ) -> tuple[int | None, ...]:
        """Match one online command to the next command in the preflighted frozen program."""
        unmatched = tuple(None for _goal in goals)
        reachability = getattr(self.scene_obj, "build_reachability", None)
        if (
            not isinstance(reachability, Mapping)
            or reachability.get("ok") is not True
            or not isinstance(reachability.get("navigation_stands"), Mapping)
        ):
            return unmatched
        cursor = getattr(self, "_preflight_command_cursor", 0)
        raw_ctx = getattr(self, "ctx", None)
        ctx = raw_ctx if isinstance(raw_ctx, Mapping) else {}
        commands = ctx.get("compiled_commands")
        if cursor is None or not commands:
            return unmatched
        if cursor >= len(commands):
            self._preflight_command_cursor = None
            return unmatched
        expected = commands[cursor]
        if not _same_frozen_command(getattr(expected, "source", ""), command_str):
            logger.info(
                "[preflight-stand] online command diverged from frozen program at "
                "command %d; disabling stand reuse for this episode",
                cursor,
            )
            self._preflight_command_cursor = None
            return unmatched
        self._preflight_command_cursor = cursor + 1
        try:
            goal_start = int(expected.goal_start)
            goal_end = int(expected.goal_end)
        except (AttributeError, TypeError, ValueError):
            self._preflight_command_cursor = None
            return unmatched

        candidates = []
        for goal_index in range(goal_start, goal_end):
            record = self._preflight_navigation_record(goal_index)
            if record is not None:
                candidates.append((goal_index, record))
        matched: set[int] = set()
        indices: list[int | None] = []
        for goal in goals:
            target_key = getattr(goal, "target_key", None)
            if getattr(goal, "target_xy", None) is not None or target_key is None:
                indices.append(None)
                continue
            match = next(
                (
                    (goal_index, record)
                    for goal_index, record in candidates
                    if goal_index not in matched
                    and str(record.get("target_key")) == str(target_key)
                ),
                None,
            )
            if match is None:
                indices.append(None)
                continue
            matched.add(match[0])
            indices.append(match[0])
        return tuple(indices)

    def _preflight_navigation_stand_xy(
        self,
        goal_index: int | None,
        target_key: object,
        stand_key: object,
    ) -> tuple[float, float] | None:
        if (os.getenv("ISAAC_PREFLIGHT_STAND_REUSE") or "1").strip().lower() in (
            "0",
            "off",
            "false",
            "no",
        ):
            return None
        if goal_index is None:
            return None
        record = self._preflight_navigation_record(goal_index)
        if (
            record is None
            or str(record.get("target_key")) != str(target_key)
            or str(record.get("stand_key")) != str(stand_key)
        ):
            return None
        xy = record.get("xy")
        if isinstance(xy, (str, bytes)) or not isinstance(xy, Sequence) or len(xy) != 2:
            return None
        try:
            stand_xy = (float(xy[0]), float(xy[1]))
        except (TypeError, ValueError):
            return None
        return stand_xy if all(math.isfinite(value) for value in stand_xy) else None

    # ---- online loop (M2): drive one high-level command through the goal loop -------------------
    def step(
        self,
        command_str: str,
        trace_event: Mapping[str, object] | None = None,
        phase_observation_requests: Sequence[Mapping[str, object]] = (),
    ) -> dict:
        """Run ONE high-level command (VirtualHome grammar) online and return the post-command observation.

        Compiles the command to goals, then drives each goal through the SAME orchestration
        episode_runtime.run_render uses (runtime nearest-reachable stand pick + closed-door open + run_goal +
        per-command physics settle) -- reused by IMPORTING those helpers, no shared-code edit. This is the
        per-tick ``env.step`` the RL AgentLoop calls."""
        import command_grammar

        if self.env is None or self.scene_obj is None:
            raise RuntimeError("IsaacEnv.step() called before open()")
        phase_requests = self._validated_phase_observation_requests(
            phase_observation_requests
        )
        with self._timed("step.total"):
            relations_before = self.scene_obj.relations(source="physics")
            held_before = set(self.scene_obj.held)
            phase_state_before = self._phase_replay_state() if phase_requests else None
            self._render_seg_start_poses = (
                _reset.snapshot_graspable_poses(self.scene_obj)
                if self.render_obs
                else None
            )
            self._seg_start = len(
                self.env.motion.frames
            )  # this command's motion = frames[_seg_start:]
            self._begin_command(command_str)
            with self._timed("step.compile"):
                held_key = next(iter(self.scene_obj.held), None)
                goals, commands = command_grammar.compile_program_with_commands(
                    command_str,
                    self.scene_obj,
                    held_key,
                )
            preflight_goal_indices = self._preflight_goal_indices_for_command(
                command_str,
                goals,
            )
            next_command = (trace_event or {}).get("_runtime_next_command")
            where0 = self._actor_xy()
            for goal_index, goal in enumerate(goals):
                followup_goal = command_grammar.navigation_action_followup(
                    goals,
                    commands,
                    goal_index,
                    self.scene_obj,
                )
                if followup_goal is None and goal_index + 1 == len(goals):
                    followup_goal = _next_command_action_followup(
                        command_grammar,
                        goal,
                        next_command,
                        self.scene_obj,
                        held_key=held_key,
                    )
                preflight_goal_index = preflight_goal_indices[goal_index]
                if preflight_goal_index is None:
                    self._drive_goal(goal, followup_goal=followup_goal)
                else:
                    self._drive_goal(
                        goal,
                        followup_goal=followup_goal,
                        preflight_goal_index=preflight_goal_index,
                    )
            self._sync_openable_command_state(int(self._seg_start))
            self._validate_action_postconditions(goals)
            self._reconcile_implicit_navigation_outcomes(
                goals,
                commands,
                command_grammar,
            )
            self._record_authored_trace(
                command_str,
                goals,
                trace_event,
                relations_before,
                held_before,
            )
            self._note_motion_frames()
            # Amortize the reward-time FK pass: realize THIS command's new frames now that its settle has
            # rewritten their rests (see _fk_sync). reward() then only realizes what is still uncached.
            self._fk_note_frames()
            self._note_render_command_span(
                int(self._seg_start), len(self.env.motion.frames)
            )
            phase_observations, phase_failures = self._capture_phase_observations(
                phase_requests,
                phase_state_before,
                int(self._seg_start),
                len(self.env.motion.frames),
            )
            walk_no_motion = self._note_walk_no_motion(goals, where0)
            self._allow_static_no_motion_capture = walk_no_motion is not None
            outcomes = self.last_step_reports()
            self._note_visual_state_command(command_str, outcomes)
            failed_zero_motion = (
                bool(outcomes)
                and any(row.get("ok") is not True for row in outcomes)
                and int(self._seg_start) >= len(self.env.motion.frames)
            )
            self._suppress_failed_zero_motion_capture = failed_zero_motion
            try:
                obs = self.observe()
            finally:
                self._suppress_failed_zero_motion_capture = False
            if phase_requests:
                obs["phase_observations"] = phase_observations
                obs["phase_observation_failures"] = phase_failures
            obs["reasons"] = [str(row.get("reason") or "") for row in outcomes]
            obs["outcomes"] = outcomes
            obs["walk_no_motion"] = walk_no_motion
            return obs

    def _reset_authored_trace(self) -> None:
        self._authored_trace = []
        self._trace_current_at = set()

    def _sync_openable_command_state(self, frame_start: int) -> None:
        ctx = getattr(self, "ctx", None)
        env = getattr(self, "env", None)
        if ctx is None or env is None or self.scene_obj is None:
            return
        controllers = ctx.get("controllers") or {}
        if not controllers:
            return
        import episode_runtime

        episode_runtime.sync_openable_motion_state(
            controllers,
            env.motion.frames[frame_start:],
        )

    def _validate_action_postconditions(self, goals: list) -> None:
        if self.scene_obj is None:
            return
        from goal_schema import Primitive, PrimitiveError

        relations = self.scene_obj.relations(source="physics")
        present = {
            (str(getattr(pred, "name", pred)).upper(), str(subject))
            for pred, subject, _target in relations
        }
        expected = {
            Primitive.OPEN: ("OPEN", True),
            Primitive.CLOSE: ("OPEN", False),
            Primitive.OPENDOOR: ("OPEN", True),
            Primitive.CLOSEDOOR: ("OPEN", False),
            Primitive.TURN_ON: ("RUNNING", True),
            Primitive.TURN_OFF: ("RUNNING", False),
            Primitive.CUT: ("CUT", True),
            Primitive.COOK: ("COOKED", True),
        }
        for goal in goals:
            requirement = expected.get(goal.prim)
            subject = goal.obj_key
            if requirement is None or subject is None:
                continue
            outcome = next(
                (row for current, row in self._step_out if current is goal),
                None,
            )
            if outcome is None or not bool(outcome.get("ok")):
                continue
            predicate, wanted = requirement
            observed = (predicate, str(subject)) in present
            if observed == wanted:
                continue
            detail = (
                f"{goal.prim.value} reported success but the runtime scorer observed "
                f"{predicate}({subject})={observed}, expected {wanted}"
            )
            if hasattr(self.env, "reports"):
                _replace_goal_report(
                    self.env.reports,
                    goal,
                    PrimitiveError.POSTCONDITION_FAILED,
                )
            report_detail = getattr(self.env, "report_detail", None)
            if isinstance(report_detail, dict):
                report_detail[id(goal)] = detail
            self._note_outcome(goal, PrimitiveError.POSTCONDITION_FAILED, detail)

    def _reconcile_implicit_navigation_outcomes(
        self,
        goals: list,
        commands: tuple,
        command_grammar,
    ) -> None:
        """Let a successfully recovered action supersede its generated walk failure."""
        from goal_schema import PrimitiveError

        outcomes = {id(goal): row for goal, row in self._step_out}
        for goal_index, nav_goal in enumerate(goals):
            followup = command_grammar.navigation_followup(
                goals,
                commands,
                goal_index,
            )
            if followup is None:
                continue
            nav_outcome = outcomes.get(id(nav_goal))
            followup_outcome = outcomes.get(id(followup))
            if (
                nav_outcome is None
                or bool(nav_outcome.get("ok"))
                or followup_outcome is None
                or not bool(followup_outcome.get("ok"))
            ):
                continue
            detail = (
                "implicit navigation was recovered by the successful "
                f"{followup.prim.value} action"
            )
            if hasattr(self.env, "reports"):
                _replace_goal_report(self.env.reports, nav_goal, PrimitiveError.OK)
            report_detail = getattr(self.env, "report_detail", None)
            if isinstance(report_detail, dict):
                report_detail[id(nav_goal)] = detail
            self._note_outcome(nav_goal, PrimitiveError.OK, detail)

    def _record_authored_trace(
        self,
        command: str,
        goals: list,
        trace_event: Mapping[str, object] | None,
        relations_before: set[tuple],
        held_before: set[str],
    ) -> None:
        assert self.scene_obj is not None
        relations_after = self.scene_obj.relations(source="physics")
        held_after = set(self.scene_obj.held)
        outcomes = self.last_step_reports()
        succeeded = bool(outcomes) and all(bool(row.get("ok")) for row in outcomes)
        actual_effects = self._relation_effects(
            relations_before,
            relations_after,
            held_before,
            held_after,
        )
        self._trace_current_at = {
            key for key in self._trace_current_at if self._actor_at_entity(key)
        }
        context = dict(trace_event or {})
        declared = [
            dict(row) for row in context.get("effects", ()) if isinstance(row, Mapping)
        ]
        parsed = __import__("command_grammar").parse_line(command)
        verb = parsed[0] if parsed is not None else ""
        if verb == "ATTACH" and succeeded:
            for effect in actual_effects:
                if effect.get("op") != "attached":
                    continue
                status = self.scene_obj.attachment_status(str(effect["subject"]))
                if status.get("parent") == effect.get("target"):
                    effect.update(status)
        waypoint = None
        authored_waypoints = [
            str(row.get("target"))
            for row in declared
            if row.get("op") == "at"
            and row.get("subject") == "human"
            and isinstance(row.get("target"), str)
        ]
        inferred_waypoints = []
        if verb in {"WALK", "RUN", "WALKTOWARDS"}:
            inferred_waypoints = [
                str(goal.target_key)
                for goal in goals
                if getattr(goal, "target_key", None)
            ][:1]
        for candidate in [*authored_waypoints, *inferred_waypoints]:
            if succeeded and self._actor_at_entity(candidate):
                actual_effects.append(
                    {"op": "at", "subject": "human", "target": candidate}
                )
                self._trace_current_at = {candidate}
                waypoint = candidate
                break
        if verb in {"CONFIRM", "WAIT"} and succeeded:
            actual_effects.append({"op": "policy_halted"})
        action = context.get("action")
        event = {
            "index": len(self._authored_trace),
            "beat_id": str(context.get("beat_id") or ""),
            "command": command,
            "action": dict(action) if isinstance(action, Mapping) else {},
            "declared_effects": declared,
            "actual_effects": actual_effects,
            "required_fact_ids": [
                str(value)
                for value in context.get("required_fact_ids", ())
                if isinstance(value, str)
            ],
            "outcomes": [dict(row) for row in outcomes],
            "ok": succeeded,
        }
        if waypoint is not None:
            event["waypoint"] = waypoint
        if any(row.get("op") == "policy_halted" for row in actual_effects):
            event["event"] = "policy_halted"
        self._authored_trace.append(event)

    @staticmethod
    def _relation_effects(
        before: set[tuple],
        after: set[tuple],
        held_before: set[str],
        held_after: set[str],
    ) -> list[dict[str, object]]:
        effects: list[dict[str, object]] = []
        effects.extend(
            {"op": "held", "subject": key} for key in sorted(held_after - held_before)
        )
        effects.extend(
            {"op": "released", "subject": key}
            for key in sorted(held_before - held_after)
        )
        for relation in sorted(after - before, key=str):
            pred, subject, target = relation
            name = str(getattr(pred, "name", pred)).lower()
            if name in {"inside", "attached", "open", "running", "cut", "cooked"}:
                effect = {"op": name, "subject": str(subject)}
                if target is not None:
                    effect["target"] = str(target)
                effects.append(effect)
        for relation in sorted(before - after, key=str):
            pred, subject, target = relation
            name = str(getattr(pred, "name", pred)).lower()
            if name in {"inside", "attached", "open", "running", "cut", "cooked"}:
                effect = {"op": f"not_{name}", "subject": str(subject)}
                if target is not None:
                    effect["target"] = str(target)
                effects.append(effect)
        return effects

    def _actor_at_entity(self, key: str) -> bool:
        import init_validation as iv

        return self.scene_obj is not None and iv.actor_near_entity(
            self.scene_obj, key, self._actor_xy()
        )

    def _actor_xy(self) -> tuple[float, float] | None:
        """The wearer's LIVE stand position. ``env.st.root_xy`` -- which ``_drive_goal`` itself reads as
        ``cur`` -- and NOT ``rig._root_xy``, which is only rewritten when the rig is next realized and is
        therefore stale for a whole command."""
        st = getattr(self.env, "st", None)
        xy = getattr(st, "root_xy", None)
        return (float(xy[0]), float(xy[1])) if xy is not None else None

    # A walk that ends where it began. 0.05 m is the codebase's own "did we move" threshold -- the
    # reach-renav in ``_drive_goal`` uses the same figure to decide a re-stand is worth running.
    _NO_MOTION_M: float = 0.05

    def _note_walk_no_motion(self, goals: list, where0: tuple | None) -> dict | None:
        """A bare ``[Walk] <x>`` that moved the body NOWHERE -- reported so the person can be told.

        ⚠️ THE DEFECT THIS EXISTS FOR. Walking to somewhere you already stand is a legitimate no-op: the
        navigator has nothing to do, so it produces no new motion frames while the engine reports
        success. This explicit marker lets strict capture reuse the current real framebuffer and lets the
        person distinguish ARRIVED from STILL WALKING instead of re-issuing the identical command until
        the budget is gone. Measured in three independent episodes
        (`the design record` section 4), twice while standing at the CORRECT answer container,
        which makes it read as a binding failure when binding had actually succeeded.

        Only a program that is NOTHING BUT navigation counts. A ``[Grab] <x>`` auto-inserts a
        NAVIGATE_TO, and when you are already at ``x`` that nav is a no-op AND THAT IS FINE -- the grab
        still happens, so there is nothing to report and saying so would be noise.
        """
        import command_grammar

        keys = [
            str(g.target_key)
            for g in goals
            if g.prim is command_grammar.Primitive.NAVIGATE_TO and g.target_key
        ]
        if not goals or len(keys) != len(goals) or where0 is None:
            return None
        outcomes = self.last_step_reports()
        if not outcomes or any(not bool(row.get("ok")) for row in outcomes):
            return None
        where1 = self._actor_xy()
        if where1 is None:
            return None
        moved = math.hypot(where1[0] - where0[0], where1[1] - where0[1])
        if moved >= self._NO_MOTION_M:
            return None
        self._counters["walk_no_motion"] = self._counters.get("walk_no_motion", 0) + 1
        logger.warning(
            "[walk] NO MOTION: %r left the body at (%.2f, %.2f) -- already standing at <%s>. "
            "The engine reports success with no new motion frames, so the person and strict capture "
            "must be told explicitly.",
            self._last_cmd,
            where1[0],
            where1[1],
            ",".join(keys),
        )
        return {
            "keys": keys,
            "moved_m": round(moved, 3),
            "accepted_already_at": True,
        }

    def _begin_command(self, command_str: str) -> None:
        """Arm the per-command records the post-step observation reads: this command's text (the visibility
        pass reports whether what it NAMED was in view), the frames the assistant will be shown, and its
        per-goal outcomes. Also cleared by every episode boundary, so a fresh episode never reports the
        previous one's rows."""
        self._last_cmd = str(command_str or "")
        self._eye_poses = []
        self._frame_segs = []
        self._step_out = []
        self._render_delivery_cache = {}
        self._dense_render_delivery_cache = {}
        self._dense_render_delivery_bytes = 0
        self._dense_render_delivery_complete = True
        self._rendered_dense_key = None
        self._static_no_motion_capture = False
        self._allow_static_no_motion_capture = False
        self._visual_state_changed_this_command = False

    def _note_render_command_span(self, start: int, end: int) -> None:
        if start < 0 or end < start:
            raise ValueError("render command spans must be ordered and non-negative")
        spans = self._render_command_spans
        if spans and spans[-1][1] != start:
            raise RuntimeError("render command spans must be contiguous")
        if not spans and start != 0:
            raise RuntimeError("the first render command span must start at frame zero")
        spans.append((start, end))
        self._sync_online_trace_command_spans()

    @staticmethod
    def _validated_phase_observation_requests(
        value: Sequence[Mapping[str, object]],
    ) -> tuple[dict[str, object], ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("phase observation requests must be an array")
        result = []
        seen = set()
        for index, raw in enumerate(value):
            if not isinstance(raw, Mapping):
                raise ValueError(f"phase observation request {index} must be an object")
            predicate = raw.get("predicate")
            if not isinstance(predicate, Mapping) or "expected" in predicate:
                raise ValueError(
                    f"phase observation request {index} must be label-free"
                )
            name = predicate.get("predicate")
            entities = predicate.get("entity_ids")
            if (
                not isinstance(name, str)
                or not name
                or isinstance(entities, (str, bytes))
                or not isinstance(entities, Sequence)
                or any(not isinstance(entity, str) or not entity for entity in entities)
            ):
                raise ValueError(f"phase observation request {index} is invalid")
            phase = raw.get("observation_phase")
            position = raw.get("capture_position")
            if (
                phase != "mid"
                or isinstance(position, bool)
                or not isinstance(position, (int, float))
                or not math.isfinite(position)
                or not 0.0 < float(position) < 1.0
            ):
                raise ValueError(
                    f"phase observation request {index} has an invalid capture"
                )
            request = {
                "predicate": {
                    "predicate": name,
                    "entity_ids": list(entities),
                },
                "observation_phase": "mid",
                "capture_position": float(position),
            }
            key = (
                name,
                tuple(entities),
                float(position),
            )
            if key not in seen:
                result.append(request)
                seen.add(key)
        return tuple(result)

    def _phase_replay_state(self) -> dict[str, object]:
        scene = self.scene_obj
        return {
            "held": set(getattr(scene, "held", ()) or ()),
            "running": set(getattr(scene, "running", ()) or ()),
            "motion_max_m": dict(getattr(scene, "motion_max_m", {}) or {}),
            "motion_max_deg": dict(getattr(scene, "motion_max_deg", {}) or {}),
            "actor_xy": copy.deepcopy(getattr(scene, "actor_xy", None)),
            "authored_trace": copy.deepcopy(self._authored_trace),
            "trace_current_at": set(self._trace_current_at),
        }

    def _restore_phase_replay_state(self, state: Mapping[str, object]) -> None:
        scene = self.scene_obj
        scene.held = set(state["held"])
        scene.running = set(state["running"])
        scene.motion_max_m = dict(state["motion_max_m"])
        scene.motion_max_deg = dict(state["motion_max_deg"])
        scene.actor_xy = copy.deepcopy(state["actor_xy"])
        self._authored_trace = copy.deepcopy(state["authored_trace"])
        self._trace_current_at = set(state["trace_current_at"])

    @contextmanager
    def _phase_replay_context(self, before: Mapping[str, object]) -> "Iterator[None]":
        import episode_runtime as s3

        scene = self.scene_obj
        controllers = self.ctx.get("controllers") or {}
        replay = getattr(self, "_clutter_render_replay", None)
        final_state = self._phase_replay_state()
        final_openable = s3._current_openable_state(controllers)
        rig = self.ctx.get("rig")
        final_object_path = getattr(rig, "_obj_path", None)
        frozen = None
        clutter_state = None
        clutter_frame = getattr(replay, "last_global_frame", None)
        try:
            frozen = _reset.freeze_graspables(scene)
            if replay is not None:
                clutter_state = replay.snapshot_state()
            self._restore_phase_replay_state(before)
            yield
        finally:
            if replay is not None:
                replay.restore_state(clutter_state or {})
                replay.last_global_frame = clutter_frame
            _reset.restore_graspables(scene, frozen)
            s3._apply_openable_state(controllers, final_openable)
            self._restore_phase_replay_state(final_state)
            if rig is not None and final_object_path is not None:
                rig._obj_path = final_object_path

    def _capture_phase_observations(
        self,
        requests: Sequence[Mapping[str, object]],
        before: Mapping[str, object] | None,
        frame_start: int,
        frame_end_exclusive: int,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if not requests:
            return [], []
        if before is None or frame_end_exclusive <= frame_start:
            return [], [
                self._phase_observation_failure(
                    request,
                    "PHASE_FRAME_UNAVAILABLE",
                    "the command produced no frame at the requested phase",
                )
                for request in requests
            ]
        if self.ctx.get("fdp_paths") or any(
            bool(getattr(frame, "free_dynamic", False))
            for frame in self.env.motion.frames[frame_start:frame_end_exclusive]
        ):
            return [], [
                self._phase_observation_failure(
                    request,
                    "PHASE_DYNAMIC_REPLAY_UNAVAILABLE",
                    "the command depends on per-frame dynamic physics replay",
                )
                for request in requests
            ]

        from fpv_segments import phase_contiguous_indices

        groups: dict[tuple[str, float], list[Mapping[str, object]]] = {}
        anchors: dict[tuple[str, float], int] = {}
        failures: list[dict[str, object]] = []
        for request in requests:
            key = (
                str(request["observation_phase"]),
                float(request["capture_position"]),
            )
            groups.setdefault(key, []).append(request)
            try:
                anchors[key] = phase_contiguous_indices(
                    frame_start,
                    frame_end_exclusive,
                    key[0],
                    key[1],
                    num_frames=1,
                )[0]
            except ValueError as error:
                failures.append(
                    self._phase_observation_failure(
                        request,
                        "PHASE_FRAME_UNAVAILABLE",
                        str(error),
                    )
                )
        if not anchors:
            return [], failures

        frames = self.env.motion.frames
        self._ensure_render_cache()
        if any(
            len(values) < frame_end_exclusive
            for values in (
                self._fk_j,
                self._fk_fr,
                self._fk_obj,
                self._fk_openable_post,
                self._fk_openable_consumed,
            )
        ):
            raise RuntimeError("phase observation replay cache is incomplete")

        observations = []
        with self._phase_replay_context(before):
            next_frame = frame_start
            for key in sorted(anchors, key=lambda value: anchors[value]):
                anchor = anchors[key]
                for frame_index in range(next_frame, anchor + 1):
                    self._apply_phase_frame(frames[frame_index], frame_index)
                next_frame = anchor + 1
                observation, unsupported = self._phase_observation_at_frame(
                    groups[key],
                    anchor,
                    frame_start,
                    frame_end_exclusive,
                )
                failures.extend(unsupported)
                if observation is not None:
                    observations.append(
                        {
                            "observation_phase": key[0],
                            "capture_position": key[1],
                            "phase_frame_index": anchor,
                            "command_frame_start": frame_start,
                            "command_frame_end_exclusive": frame_end_exclusive,
                            "observation": observation,
                        }
                    )
        return observations, failures

    def _apply_phase_frame(self, frame: object, frame_index: int) -> None:
        import episode_runtime as s3
        import smplh_anim_core as core

        scene = self.scene_obj
        s3._apply_openable_state(
            self.ctx.get("controllers") or {},
            self._fk_openable_post[frame_index],
        )
        self._apply_clutter_render_frame(frame_index)
        object_transform = self._fk_obj[frame_index]
        drove_openable = self._fk_openable_consumed[frame_index]
        entity = self._phase_frame_object_entity(frame)
        if object_transform is not None and not drove_openable:
            object_path = getattr(frame, "obj_path", None)
            if object_path:
                self.ctx["rig"]._obj_path = object_path
                core.set_object_world(self.ctx["rig"], object_transform)
            elif not getattr(scene, "grasp_objs", None):
                core.set_object_world(self.ctx["rig"], object_transform)
            for carried_path, relative_transform in getattr(frame, "carried", ()) or ():
                self.ctx["rig"]._obj_path = carried_path
                core.set_object_world(
                    self.ctx["rig"], object_transform @ relative_transform
                )
            if getattr(frame, "carried", None) and object_path:
                self.ctx["rig"]._obj_path = object_path

        kind = str((getattr(frame, "obj", ()) or ("",))[0])
        if entity is not None and kind == "carry":
            scene.held = {entity.name}
        elif entity is not None and kind == "rest":
            scene.held.discard(entity.name)
        running = getattr(frame, "running_devices", None)
        if running is not None:
            scene.running = set(running)
        root_joint = self._fk_j[frame_index][0]
        scene.actor_xy = (float(root_joint[0]), float(root_joint[1]))
        if getattr(scene, "tracked_unmoved", None):
            scene.note_motion(scene.tracked_unmoved)

    def _phase_frame_object_entity(self, frame: object):
        scene = self.scene_obj
        object_path = getattr(frame, "obj_path", None)
        if not object_path and not getattr(scene, "grasp_objs", None):
            object_path = getattr(self.ctx.get("rig"), "_obj_path", None)
        return scene._prim_to_entity(str(object_path)) if object_path else None

    def _phase_observation_at_frame(
        self,
        requests: Sequence[Mapping[str, object]],
        frame_index: int,
        frame_start: int,
        frame_end_exclusive: int,
    ) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
        frame = self.env.motion.frames[frame_index]
        relations = {
            row
            for row in self.scene_obj.relations(source="physics")
            if (
                (shape := _phase_relation_shape(row)) is not None
                and shape[0] in _PHASE_PREDICATE_ARITIES
                and (
                    shape[0] != "running"
                    or getattr(frame, "running_devices", None) is not None
                )
            )
        }
        observed = []
        failures = []
        for request in requests:
            value, reason = self._phase_predicate_value(
                request["predicate"], relations, frame
            )
            if reason is not None:
                failures.append(
                    self._phase_observation_failure(
                        request,
                        reason,
                        "the frame replay has no complete physical reader for this predicate",
                        frame_index=frame_index,
                    )
                )
                continue
            observed.append(
                {
                    "predicate": copy.deepcopy(request["predicate"]),
                    "observed": value,
                }
            )
        if not observed:
            return None, failures
        return (
            {
                "relations": relations,
                "held": set(self.scene_obj.held),
                "at": set(),
                "trace_events": copy.deepcopy(self._authored_trace),
                "score": {},
                "goal": [],
                "gate_outcomes": [],
                "actor_xy": copy.deepcopy(self.scene_obj.actor_xy),
                "observation_phase": str(requests[0]["observation_phase"]),
                "capture_position": float(requests[0]["capture_position"]),
                "phase_frame_index": frame_index,
                "command_frame_start": frame_start,
                "command_frame_end_exclusive": frame_end_exclusive,
                "predicate_observations": observed,
            },
            failures,
        )

    def _phase_predicate_value(
        self,
        predicate: Mapping[str, object],
        relations: set[tuple],
        frame: object,
    ) -> tuple[bool | None, str | None]:
        raw_name = str(predicate.get("predicate") or "").rsplit(".", 1)[-1].lower()
        negative = raw_name.startswith("not_")
        name = _PHASE_PREDICATE_ALIASES.get(
            raw_name.removeprefix("not_"), raw_name.removeprefix("not_")
        )
        entities = tuple(str(value) for value in predicate.get("entity_ids") or ())
        if _PHASE_PREDICATE_ARITIES.get(name) != len(entities):
            return None, "PHASE_PREDICATE_UNSUPPORTED"
        if any(
            entity not in self.scene_obj.entities and entity != "wearer"
            for entity in entities
        ):
            return None, "PHASE_ENTITY_UNAVAILABLE"
        if name == "running" and getattr(frame, "running_devices", None) is None:
            return None, "PHASE_PREDICATE_UNSUPPORTED"
        if name == "held":
            present = entities[0] in self.scene_obj.held
        else:
            present = any(
                _phase_relation_shape(row) == (name, entities) for row in relations
            )
        return (not present if negative else present), None

    @staticmethod
    def _phase_observation_failure(
        request: Mapping[str, object],
        reason_code: str,
        detail: str,
        *,
        frame_index: int | None = None,
    ) -> dict[str, object]:
        result = {
            "predicate": copy.deepcopy(request.get("predicate")),
            "observation_phase": request.get("observation_phase"),
            "capture_position": request.get("capture_position"),
            "reason_code": reason_code,
            "detail": detail,
        }
        if frame_index is not None:
            result["phase_frame_index"] = frame_index
        return result

    @staticmethod
    def _online_trace_is_active(session) -> bool:
        status = getattr(session, "status", None)
        if status is None:
            return True
        return str(getattr(status, "value", status)) == "active"

    def _sync_online_trace_command_spans(self) -> None:
        session = getattr(self, "_online_render_trace", None)
        if session is None:
            return
        noted = int(getattr(self, "_online_trace_noted_commands", 0))
        spans = getattr(self, "_render_command_spans", ())
        if not self._online_trace_is_active(session):
            self._online_trace_noted_commands = len(spans)
            return
        for start, _end in spans[noted:]:
            session.begin_segment(start)
        self._online_trace_noted_commands = len(spans)

    # ---- timed thin wrappers: same call, same args, same return -- only a phase timer is added. Used
    # where the call sits inside a short-circuit ``and`` chain, where a ``with`` block would change the
    # evaluation semantics.
    def _timed_pick_stand(self, *args, **kwargs):
        """``iv.pick_stand_runtime`` under the ``step.stand_pick`` timer + the audit log, served from the
        same-stand cache when that is provably free (see ``_standpick_lookup``).
        ``ISAAC_STANDPICK_CACHE=0`` restores the byte-identical uncached path."""
        if self._standpick_mode() == "off":
            return self._pick_stand_scan(args, kwargs)
        return self._pick_stand_cached(args, kwargs)

    def _pick_stand_scan(self, args: tuple, kwargs: dict):
        """The REAL scan under the ``step.stand_pick`` timer + the audit log (the cache-off path)."""
        import init_validation as iv

        with self._timed("step.stand_pick"):
            stand = iv.pick_stand_runtime(*args, **kwargs)
        self._counters["standpick_call"] += 1
        self._log_stand_pick(args, kwargs, stand)
        return stand

    # ---- stand-pick cache: skip the scan when the answer CANNOT have changed --------------------
    #
    # A "walk to X" command and the "grab X" that follows it each resolve their NAVIGATE_TO through a full
    # ~2.8s stand scan, and by the time the second one runs the human is standing exactly on the stand the
    # first one returned -- half of an episode's runtime stand picks are that structural duplicate.
    #
    # THE SAFE-HIT RULE (narrower than "equal fingerprint", because ``cur_xy`` DIFFERS across the pair, so
    # a full-fingerprint cache would never fire): remember ``fingerprint-minus-cur_xy -> returned stand S``
    # and serve S only when everything except ``cur_xy`` matches AND ``cur_xy`` IS S (bit-exact, compared
    # through ``_stand_repr``) AND ``nearest_to_contact`` is False.
    #
    # WHY THAT IS FREE (``init_validation._search_reachable_stand``). The rest of the fingerprint (key,
    # contact_override, doors_open, held, open_objs, _world_epoch) guards that nothing else the scan reads
    # has changed, so only the two things ``cur_xy`` itself feeds have to be argued:
    #   1. RANKING. With ``nearest_to_contact=False`` candidates are sorted by distance from ``cur_xy``,
    #      and they are distinct grid cell CENTRES enumerated around the CONTACT (not around ``cur_xy``),
    #      so the candidate SET is untouched and S -- itself one of those centres -- now sits at distance
    #      exactly 0, strictly below every other candidate. S is therefore evaluated FIRST.
    #   2. THE WALKABILITY PRE-CULL, the one gate that also reads ``cur_xy``: it floods the passable grid
    #      from ``cur_xy``'s cell (snapped to its nearest free cell) and requires a flooded free cell
    #      within 0.25 m -- i.e. inside the same +-2-cell box ``nearest_free`` searches at 0.10 m cells.
    #      S passed that gate on the FIRST scan, so a free cell exists within that box of S; hence the
    #      snap from S lands on a free cell inside the box, that cell is trivially in its OWN flooded
    #      component, and the gate passes again. The other gates (idle non-penetration, reach IK, arm
    #      keyframe) never read ``cur_xy`` and already passed for S.
    # S is evaluated first and passes every gate -> the scan returns S. The scan is deterministic, so this
    # is an identity, not a heuristic.
    #
    # ``ISAAC_STANDPICK_CACHE=verify`` keeps the Phase-1 shadow check armed: on every hit the real scan
    # STILL runs and the two are compared byte-exactly (``_standpick_shadow``).
    #
    # KNOWN GAP (unchanged from the audit): the door-approach picks made inside
    # ``episode_runtime._runtime_open_doors`` call ``iv.pick_stand_runtime`` directly, so they neither hit nor
    # populate this cache. Nothing to fix here -- closing it would mean editing shared simulator.physical_world code.
    _STANDPICK_CACHE_MAX = 512

    @staticmethod
    def _standpick_mode() -> str:
        """``ISAAC_STANDPICK_CACHE`` -> ``"on"`` (default) | ``"off"`` | ``"verify"``."""
        val = (os.getenv("ISAAC_STANDPICK_CACHE") or "1").strip().lower()
        if val in ("0", "off", "false", "no"):
            return "off"
        if val in ("verify", "shadow"):
            return "verify"
        return "on"

    @staticmethod
    def _standpick_key(fp: tuple) -> tuple:
        """The cache key: the full fingerprint minus the live root pose."""
        return fp[:1] + fp[3:]

    def _pick_stand_cached(self, args: tuple, kwargs: dict):
        """Cache-armed stand pick. ONE ``step.stand_pick`` sample per call covering the bookkeeping AND the
        scan-or-serve (so the cache-on/cache-off comparison is honest -- the cache pays for itself inside
        the phase it is meant to shrink); the bookkeeping alone is also recorded as
        ``step.stand_pick_ovh``."""
        import init_validation as iv

        verify = self._standpick_mode() == "verify"
        with self._timed("step.stand_pick"):
            t0 = time.perf_counter()
            fp = self._stand_pick_fingerprint(args, kwargs)
            hit = self._standpick_lookup(fp, kwargs)
            ovh = time.perf_counter() - t0
            if hit is not None:
                self._counters["standpick_hit"] += 1
            if hit is not None and not verify:
                stand = ((float(hit[0][0]), float(hit[0][1])), float(hit[1]))
            else:
                stand = iv.pick_stand_runtime(*args, **kwargs)
            t1 = time.perf_counter()
            if hit is not None and verify:
                self._standpick_shadow(fp, hit, stand)
            self._standpick_store(fp, kwargs, stand)
            ovh += time.perf_counter() - t1
        self._note_time("step.stand_pick_ovh", ovh)
        self._counters["standpick_call"] += 1
        self._log_stand_pick(args, kwargs, stand)
        return stand

    def _standpick_lookup(self, fp: tuple, kwargs: dict):
        """The cached stand for ``fp`` when the safe-hit rule fires, else None."""
        if kwargs.get("nearest_to_contact", False):
            return None  # ranked by distance to the CONTACT -> the distance-0 argument does not apply
        stand = self._standpick_cache.get(self._standpick_key(fp))
        if stand is None:
            return None
        yaw_matches = (
            fp[2] is None
            or abs((float(fp[2]) - float(stand[1]) + 180.0) % 360.0 - 180.0) <= 1e-6
        )
        if fp[1] == self._stand_repr(stand)[:2] and yaw_matches:
            return stand
        self._note_standpick_near(fp, stand)
        return None

    def _standpick_store(self, fp: tuple, kwargs: dict, stand) -> None:
        """Remember a scan's answer. A ``None`` (no clean stand) is never stored -- the hit rule compares
        ``cur_xy`` against the RETURNED stand, which a None does not have. Neither is a
        ``nearest_to_contact`` answer, which the rule may not serve. Overflow clears
        (only costs hits)."""
        if stand is None or kwargs.get("nearest_to_contact", False):
            return
        if len(self._standpick_cache) >= self._STANDPICK_CACHE_MAX:
            self._standpick_cache.clear()
        self._standpick_cache[self._standpick_key(fp)] = (
            (float(stand[0][0]), float(stand[0][1])),
            float(stand[1]),
        )

    def _note_standpick_near(self, fp: tuple, stand) -> None:
        """Everything but ``cur_xy`` matched a cached stand, but ``cur_xy`` is not EXACTLY that stand.
        This is the measurement that decides whether the exact-equality rule can fire at all, so the metric
        gap is logged: a tolerance must be argued from these numbers, never assumed."""
        # fp holds repr()s of the floats -> float() parses them back bit-exactly.
        dx = float(fp[1][0]) - float(stand[0][0])
        dy = float(fp[1][1]) - float(stand[0][1])
        self._counters["standpick_near"] += 1
        logger.info(
            "[standpick] NEAR-MISS key=%s: cur_xy is %.9f m from the cached stand -> no hit",
            fp[0],
            (dx * dx + dy * dy) ** 0.5,
        )

    def _standpick_shadow(self, fp: tuple, hit, stand) -> None:
        """VERIFY mode: the hit rule fired AND the real scan still ran -- demand byte-exact agreement."""
        got, exp = self._stand_repr(stand), self._stand_repr(hit)
        if got == exp:
            return
        self._counters["standpick_mismatch"] += 1
        logger.error(
            "[standpick] CACHE MISMATCH fp=%r cached=%r real_scan=%r", fp, exp, got
        )

    # ---- stand-pick audit (MEASUREMENT ONLY -- no caching, no logic change) ----------------------
    @staticmethod
    def _stand_repr(stand) -> tuple | None:
        """``((x, y), yaw)`` -> a full-precision comparable tuple (``repr`` round-trips a float exactly)."""
        if stand is None:
            return None
        (sx, sy), yaw = stand
        return (repr(float(sx)), repr(float(sy)), repr(float(yaw)))

    def _stand_pick_fingerprint(self, args: tuple, kwargs: dict) -> tuple:
        """The FULL input set ``iv.pick_stand_runtime`` can read, as a comparable tuple: the target key,
        the human's CURRENT root_xy at full precision, the contact override, and a cheap identity for the
        scene state that could change the answer -- what is held, which openables are open, and
        ``_world_epoch`` (bumped by every per-command settle, i.e. every time objects were physically
        moved). Two calls with an equal fingerprint had equal inputs and MUST return the same stand.
        """
        import numpy as np
        import semantic_executor as se

        cur = args[5]
        co = kwargs.get("contact_override")
        contacts = kwargs.get("contact_overrides")
        start_yaw = kwargs.get("start_yaw")
        scene = self.scene_obj
        entity = se._openable_access_entity(scene, str(args[4]))
        if entity is None:
            entity = (getattr(scene, "entities", None) or {}).get(str(args[4]))
        door_meta = getattr(entity, "meta", None) or {}
        door_policy = tuple(
            (
                field,
                (
                    None
                    if door_meta.get(field) is None
                    else repr(float(door_meta[field]))
                ),
            )
            for field in (
                "action_standoff_m",
                "content_standoff_m",
                "content_insert_standoff_m",
                "content_extract_standoff_m",
                "action_operator_lateral_offset_m",
                "content_operator_lateral_offset_m",
                "content_insert_operator_lateral_offset_m",
                "content_extract_operator_lateral_offset_m",
                "close_operator_lateral_offset_m",
                "action_lateral_limit_m",
                "sweep_standoff_m",
            )
        )
        validator_key = getattr(
            kwargs.get("candidate_validator"), "_standpick_cache_key", None
        )
        if validator_key is not None:
            door_policy += (("content_action_validator", validator_key),)
        return (
            str(args[4]),
            (repr(float(cur[0])), repr(float(cur[1]))),
            None if start_yaw is None else repr(float(start_yaw)),
            (
                None
                if co is None
                else tuple(repr(float(v)) for v in np.asarray(co).ravel())
            ),
            (
                None
                if contacts is None
                else tuple(
                    tuple(repr(float(v)) for v in point)
                    for point in np.asarray(contacts).reshape(-1, 3)
                )
            ),
            bool(kwargs.get("doors_open", True)),
            bool(kwargs.get("nearest_to_contact", False)),
            bool(kwargs.get("route_gate", False)),
            bool(kwargs.get("cue_read", False)),
            bool(kwargs.get("require_reach", True)),
            bool(kwargs.get("require_openable_sweep_clearance", True)),
            (
                None
                if kwargs.get("reach_tol") is None
                else repr(float(kwargs["reach_tol"]))
            ),
            tuple(sorted(getattr(scene, "held", None) or ())),
            tuple(sorted(getattr(scene, "open_objs", None) or ())),
            int(self._world_epoch),
            bool(kwargs.get("openable_action_phase", False)),
            (
                None
                if kwargs.get("openable_action_kind") is None
                else str(kwargs["openable_action_kind"])
            ),
            (
                None
                if kwargs.get("front_access_standoff_m") is None
                else repr(float(kwargs["front_access_standoff_m"]))
            ),
            (
                None
                if kwargs.get("validated_stand_xy") is None
                else tuple(repr(float(value)) for value in kwargs["validated_stand_xy"])
            ),
            bool(kwargs.get("centered_openable", False)),
            door_policy,
        )

    def _log_stand_pick(self, args: tuple, kwargs: dict, stand) -> None:
        """One line per runtime stand pick: the input fingerprint + the stand it returned. Answers "how
        many of these 2.8s scans are TRUE duplicates" from a real rollout before anyone risks a cache.
        Set ``ISAAC_STANDPICK_LOG=0`` to silence. NOTE: the door-approach picks made INSIDE
        ``episode_runtime._runtime_open_doors`` do not route through here and are therefore not counted.
        """
        if os.getenv("ISAAC_STANDPICK_LOG", "1") == "0":
            return
        fp = self._stand_pick_fingerprint(args, kwargs)
        out = self._stand_repr(stand)
        self._stand_picks.append((fp, out))
        logger.info("[standpick] n=%d fp=%r -> %r", len(self._stand_picks), fp, out)

    def stand_picks(self) -> list[tuple[tuple, tuple | None]]:
        """The episode's ``(input_fingerprint, returned_stand)`` audit log (see ``_log_stand_pick``)."""
        return list(self._stand_picks)

    def _timed_open_doors(self, *args, **kwargs) -> bool:
        import episode_runtime as s3

        with self._timed("step.open_doors"):
            return s3._runtime_open_doors(*args, **kwargs)

    def _timed_run_goal(self, goal):
        with self._timed("step.run_goal"):
            out = self.env.run_goal(goal)
        info = out[2]
        self._note_outcome(goal, info["reason"], info.get("detail", ""))
        return out

    # ---- per-command failure back-channel --------------------------------------------------------
    def _note_outcome(self, goal, reason, detail: str) -> None:
        """Record ONE verdict per Goal for the command currently running.

        An exact mirror of ``semantic_env._record``: a retried Goal (the OUT_OF_REACH reach-renav re-runs
        the SAME object) SUPERSEDES its earlier row in place rather than adding a second, and the match is
        on object IDENTITY because two independent commands compile to EQUAL dataclasses. The list holds
        the goal alongside its record so the reference is pinned for the whole step -- an ``id()`` key on a
        temporary Goal could be recycled under a later one.

        Why this exists at all: ``reports`` rows are ``(Goal, PrimitiveError)``, so the plain-English
        ``detail`` the primitive raised ("PICK while already holding an object") is dropped at the door and
        the model is handed a bare ``GRASP_FAILED``, which reads like "the grasp slipped".
        """
        rec = {
            "primitive": getattr(getattr(goal, "prim", None), "value", ""),
            "reason": getattr(reason, "name", str(reason)),
            "detail": str(detail or ""),
            "ok": getattr(reason, "name", "") == "OK",
        }
        for i, (prev_goal, _prev) in enumerate(self._step_out):
            if prev_goal is goal:
                self._step_out[i] = (goal, rec)
                return
        self._step_out.append((goal, rec))

    def last_step_reports(self) -> list[dict]:
        """The LAST ``step()``'s per-goal outcomes, in goal order -- see ``_note_outcome``. One row per
        compiled goal of that command and nothing from any earlier command."""
        return [dict(rec) for _goal, rec in self._step_out]

    def _report_blocked(self, goal, detail: str, reason=None) -> None:
        """File a PATH_BLOCKED verdict the executor never got to raise (no reachable stand / a door that
        will not open). Writes BOTH channels so ``reports`` and ``last_step_reports`` cannot disagree, and
        supplies the ``detail`` the bare enum cannot: those two causes need different recovery from the
        human and PATH_BLOCKED alone does not distinguish them."""
        from goal_schema import PrimitiveError

        failure = reason or PrimitiveError.PATH_BLOCKED
        self.env.reports.append((goal, failure))
        self._note_outcome(goal, failure, detail)

    def _timed_placed_objects(self, frames):
        import action_check

        with self._timed("step.placed_check"):
            return action_check._placed_objects(frames)

    def _placed_objects_for_settle(self, scene, frames):
        if not getattr(scene, "post_action_check", False):
            return {}
        if getattr(scene, "cworld", None) is None:
            return {}
        return self._timed_placed_objects(frames)

    def _settle_placement_physics(self, scene, frames, frame_start: int):
        import action_check
        import settle_guard
        import settle_trace
        import smplh_anim_core as core

        with self._timed("step.settle_clock"):
            self._ensure_settle_clock()
        settle_trace.install()
        settle_guard.install(lambda: self.scene_obj)
        with self._timed("step.settle"):
            return action_check.settle_placements(
                self.ctx["ph"],
                self.stage,
                self.ctx["rig"],
                core,
                scene.cworld,
                frames,
                frame_start,
                step_fn=self._settle_step,
            )

    def _settle_placement_if_needed(self, scene, goal, frame_start: int) -> None:
        import dynamic_render_replay
        import episode_runtime as s3
        import semantic_executor as se

        frames = self.env.motion.frames
        placed_objects = self._placed_objects_for_settle(scene, frames[frame_start:])
        if not placed_objects:
            return
        controllers = self.ctx.get("controllers") or {}
        if controllers:
            s3._apply_openable_state(
                controllers,
                s3._openable_state_after_frames(controllers, frames),
            )
        paused = dynamic_render_replay.freeze_dynamic_prims(
            self.stage, self.ctx.get("fdp_paths") or ()
        )
        try:
            _notes, slot_locked = settle_or_lock_taskpack_placement(
                scene,
                goal,
                self.stage,
                self.ctx["ph"],
                placed_objects,
                lambda: self._settle_placement_physics(scene, frames, frame_start),
            )
        finally:
            dynamic_render_replay.restore_dynamic_prims(self.stage, paused)
        if slot_locked:
            meta = goal.meta or {}
            logger.info(
                "taskpack placement slot locked: %s -> %s retained the authored release pose",
                str(meta.get("_held_object_key") or ""),
                str(meta.get("surface") or ""),
            )
        se.sync_carried_to_root(
            scene,
            str((goal.meta or {}).get("_carried_root") or ""),
            (goal.meta or {}).get("_carried") or (),
        )
        scene.note_motion()
        self._world_epoch += 1

    def _capture_inside_retry(self, goal, frame_start: int) -> dict | None:
        env, scene = self.env, self.scene_obj
        held = getattr(env.st, "held", None)
        target = str((goal.meta or {}).get("surface") or "")
        subject = str((held or {}).get("obj_key") or "")
        if not subject or not target:
            return None
        dynamic = {}
        for name in ("_space_assign", "_into_placed"):
            dynamic[name] = (
                (
                    True,
                    copy.deepcopy(getattr(scene, name)),
                )
                if hasattr(scene, name)
                else (False, None)
            )
        entity = scene.entities.get(subject)
        return {
            "subject": subject,
            "target": target,
            "frame_start": frame_start,
            "state": copy.deepcopy(env.st),
            "poses": _reset.snapshot_graspable_poses(scene),
            "held": copy.deepcopy(scene.held),
            "on": copy.deepcopy(scene.on),
            "inside": copy.deepcopy(scene.inside),
            "open_objs": copy.deepcopy(getattr(scene, "open_objs", set())),
            "dynamic": dynamic,
            "entity_grounding": (
                (
                    copy.deepcopy(entity.xy),
                    copy.deepcopy(entity.z),
                )
                if entity is not None
                else None
            ),
            "reports_len": len(env.reports),
            "report_detail": copy.deepcopy(getattr(env, "report_detail", {})),
            "report_frame_span": copy.deepcopy(getattr(env, "report_frame_span", {})),
            "report_state": copy.deepcopy(getattr(env, "report_state", {})),
            "world_epoch": self._world_epoch,
        }

    def _failed_inside_xy(
        self, retry: Mapping[str, object], goal
    ) -> tuple[float, float] | None:
        chosen = (goal.meta or {}).get("_placement_chosen_xy") or (goal.meta or {}).get(
            "_inside_chosen_xy"
        )
        if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
            return (float(chosen[0]), float(chosen[1]))
        placed = getattr(self.scene_obj, "_into_placed", {}).get(retry["target"]) or []
        if not placed:
            return None
        return (float(placed[-1][0]), float(placed[-1][1]))

    def _restore_inside_retry(self, retry: Mapping[str, object]) -> None:
        env, scene = self.env, self.scene_obj
        del env.motion.frames[int(retry["frame_start"]) :]
        env.st = copy.deepcopy(retry["state"])
        _reset.restore_graspable_poses(scene, retry["poses"])
        scene.held = copy.deepcopy(retry["held"])
        scene.on = copy.deepcopy(retry["on"])
        scene.inside = copy.deepcopy(retry["inside"])
        scene.open_objs = copy.deepcopy(retry["open_objs"])
        for name, raw in retry["dynamic"].items():
            existed, value = raw
            if existed:
                setattr(scene, name, copy.deepcopy(value))
            elif hasattr(scene, name):
                delattr(scene, name)
        grounding = retry.get("entity_grounding")
        entity = scene.entities.get(str(retry["subject"]))
        if entity is not None and grounding is not None:
            entity.xy, entity.z = copy.deepcopy(grounding)
        del env.reports[int(retry["reports_len"]) :]
        env.report_detail = copy.deepcopy(retry["report_detail"])
        env.report_frame_span = copy.deepcopy(retry["report_frame_span"])
        env.report_state = copy.deepcopy(retry["report_state"])
        self._world_epoch = int(retry["world_epoch"])
        self._fk_clear()

    def _retry_place_inside_from_alternate_space(self, goal, info, placement_retry):
        """Retry a camera-blocked insertion on another real interior shelf."""
        import command_grammar
        from goal_schema import PrimitiveError
        from scene_state import Role

        if (
            info["reason"] is not PrimitiveError.OUT_OF_REACH
            or goal.prim is not command_grammar.Primitive.PLACE_INSIDE
            or self.env.st.held is None
        ):
            return info, placement_retry
        target_key = str((goal.meta or {}).get("surface") or "")
        entities = getattr(self.scene_obj, "entities", None) or {}
        target = (
            self.scene_obj.target_view(target_key, Role.CONTAINER)
            if target_key in entities
            else None
        )
        spaces = getattr(target, "spaces", None) or ()
        tries = 0
        while (
            info["reason"] is PrimitiveError.OUT_OF_REACH
            and goal.prim is command_grammar.Primitive.PLACE_INSIDE
            and self.env.st.held is not None
            and len(spaces) > 1
            and goal.meta.get("_space_chosen") is not None
            and tries < len(spaces) - 1
        ):
            tries += 1
            rejected = set(goal.meta.get("_space_rejected") or ())
            rejected.add(str(goal.meta.pop("_space_chosen")))
            goal.meta["_space_rejected"] = rejected
            if len(rejected) >= len(spaces):
                break
            logger.warning(
                "place-inside retry: <%s> rejected an eye-blocked interior space; "
                "trying another physical shelf (%d/%d rejected)",
                target_key,
                len(rejected),
                len(spaces),
            )
            if placement_retry is not None:
                placement_retry = self._capture_inside_retry(
                    goal, len(self.env.motion.frames)
                )
            _obs, _done, info = self._timed_run_goal(goal)
        return info, placement_retry

    def _drive_goal(
        self,
        goal,
        followup_goal=None,
        *,
        preflight_goal_index: int | None = None,
    ) -> None:
        """One compiled goal, exactly as run_render's per-goal block: resolve an unresolved NAVIGATE to the
        nearest-reachable stand (opening any closed door en route), run it, retry a PLACE/ATTACH that missed
        OUT_OF_REACH from a reach-capable stand, then settle placed objects so the next command reads their
        true rest."""
        import command_grammar
        import episode_runtime as s3
        import semantic_executor as se
        from goal_schema import Goal, PrimitiveError

        env, scene, stage = self.env, self.scene_obj, self.stage
        ph, rig = self.ctx["ph"], self.ctx["rig"]
        prim = command_grammar.Primitive
        prev = len(env.motion.frames)
        nav_transit_restore = None
        placement_retry = (
            self._capture_inside_retry(goal, prev)
            if goal.prim in (prim.PLACE, prim.PLACE_INSIDE)
            else None
        )

        if (
            goal.prim is prim.NAVIGATE_TO
            and goal.target_xy is None
            and goal.target_key is not None
        ):
            if getattr(scene, "nav_grid", None) is not None:
                cur = tuple(env.st.root_xy)
                stand_request = s3.navigation_stand_request(
                    scene,
                    goal,
                    followup_goal,
                    actor_xy=cur,
                )
                stand_key = stand_request["stand_key"]
                serves_openable = stand_request["serves_openable"]
                openable_action_kind = s3._openable_action_kind(
                    scene,
                    followup_goal,
                    stand_key,
                )
                previews_openable_action = s3._navigation_previews_openable_action(
                    scene,
                    followup_goal,
                    stand_key,
                    serves_openable=serves_openable,
                )
                s3._apply_openable_content_action_reach_tolerance(
                    scene, followup_goal, stand_key
                )
                content_preview = openable_action_kind in ("insert", "extract")
                target_transit_avoid = set()
                navigation_fallback = {}
                stand = None
                cached_stand_xy = self._preflight_navigation_stand_xy(
                    preflight_goal_index,
                    goal.target_key,
                    stand_key,
                )
                front_standoffs = tuple(
                    s3._front_access_standoffs_for_goal(followup_goal, scene)
                )
                contact_attempts = tuple(stand_request["contact_attempts"])
                search_attempts = []
                if cached_stand_xy is not None and front_standoffs and contact_attempts:
                    search_attempts.append(
                        (
                            cached_stand_xy,
                            front_standoffs[0],
                            contact_attempts[0],
                        )
                    )
                search_attempts.extend(
                    (None, front_standoff, contact_attempt)
                    for front_standoff in front_standoffs
                    for contact_attempt in contact_attempts
                )
                for (
                    validated_stand_xy,
                    front_standoff,
                    contact_attempt,
                ) in search_attempts:
                    stand_contact, contact_fraction = contact_attempt
                    target_transit_avoid = set()
                    navigation_fallback = {}
                    if serves_openable:
                        followup_goal.meta["_door_contact_fraction"] = contact_fraction
                    stand = self._timed_pick_stand(
                        ph,
                        stage,
                        rig,
                        scene,
                        stand_key,
                        cur,
                        contact_override=stand_contact,
                        contact_overrides=stand_request["contact_overrides"],
                        complete_contact_trajectory=stand_request[
                            "complete_contact_trajectory"
                        ],
                        nearest_to_contact=serves_openable,
                        cue_read=stand_request["serves_read"],
                        require_reach=(
                            False if content_preview else stand_request["require_reach"]
                        ),
                        require_openable_sweep_clearance=bool(
                            stand_request["is_nav_door"]
                            or (
                                serves_openable
                                and not s3._needs_front_opening_action_stand(
                                    scene,
                                    followup_goal,
                                )
                            )
                        ),
                        openable_action_phase=bool(serves_openable),
                        openable_action_kind=openable_action_kind,
                        front_access_standoff_m=front_standoff,
                        validated_stand_xy=validated_stand_xy,
                        start_yaw=float(env.st.root_yaw),
                        gait_profile=env.g,
                        gait_cum_dist=float(env.st.gait_cum_dist),
                        route_gate=s3._ego_route_gate(scene),
                        reach_tol=s3._navigation_stand_reach_tolerance(
                            scene, stand_request, followup_goal
                        ),
                        transit_avoid_out=target_transit_avoid,
                        navigation_fallback_out=(
                            navigation_fallback
                            if getattr(
                                scene,
                                "egocentric_camera_only_collision",
                                True,
                            )
                            or not getattr(scene, "strict_acceptance", False)
                            else None
                        ),
                        candidate_validator=(
                            s3._openable_candidate_validator(
                                env,
                                rig,
                                scene,
                                followup_goal,
                                stand_key,
                            )
                            if previews_openable_action
                            else None
                        ),
                        action_state=(
                            env.st if serves_openable or content_preview else None
                        ),
                        centered_openable=stand_request["centered_openable"],
                    )
                    if stand is not None:
                        if serves_openable and contact_fraction < 1.0:
                            logger.info(
                                "[runtime-stand] %s: handle stand unavailable; using "
                                "the %.2f-span door contact",
                                stand_key,
                                contact_fraction,
                            )
                        if validated_stand_xy is not None:
                            logger.info(
                                "[preflight-stand] goal %d <%s>: cached stand "
                                "(%.2f,%.2f) passed live revalidation",
                                preflight_goal_index,
                                stand_key,
                                stand[0][0],
                                stand[0][1],
                            )
                        break
                    if validated_stand_xy is not None:
                        logger.info(
                            "[preflight-stand] goal %d <%s>: cached stand was rejected; "
                            "running the normal live search",
                            preflight_goal_index,
                            stand_key,
                        )
                if stand is None:
                    stationary_openable = se._openable_access_entity(scene, stand_key)
                    stationary_operator = bool(
                        (
                            getattr(stationary_openable, "meta", None)
                            if stationary_openable is not None
                            else None
                        )
                        and stationary_openable.meta.get("stationary_operator")
                    )
                    if (
                        not stationary_operator
                        and s3._allow_actor_near_stand_fallback(
                            stand_request,
                            content_preview=content_preview,
                        )
                        and self._actor_at_entity(stand_key)
                    ):
                        goal.target_xy = (float(cur[0]), float(cur[1]))
                        goal.target_yaw = float(env.st.root_yaw)
                        logger.info(
                            "[runtime-stand] %s has no better stand, but the actor is already "
                            "within interaction range; accepting the walk as a no-op",
                            goal.target_key,
                        )
                    else:
                        self._report_blocked(
                            goal,
                            f"no reachable place to stand near <{goal.target_key}>",
                            reason=(
                                PrimitiveError.NO_COLLISION_FREE_STANCE
                                if getattr(scene, "strict_acceptance", False)
                                else PrimitiveError.PATH_BLOCKED
                            ),
                        )
                        return
                else:
                    goal.target_xy, goal.target_yaw = stand
                    if navigation_fallback.get("selected"):
                        scene.egocentric_collision_fallback_used = True
                        goal.meta["_egocentric_navigation_fallback"] = True
                        if navigation_fallback.get("force_straight"):
                            goal.meta["_camera_safe_navigation_fallback"] = True
                        logger.warning(
                            "[runtime-stand] %s: using least-penetrating camera-clear fallback (%s)",
                            goal.target_key,
                            navigation_fallback.get("reason") or "body-only contact",
                        )
                nav_transit_restore = set(scene.nav_grid.transit_avoid)
                scene.nav_grid.transit_avoid.clear()
                scene.nav_grid.transit_avoid.update(target_transit_avoid)
                door_why: list[str] = []
                if not self._timed_open_doors(
                    env,
                    ph,
                    stage,
                    rig,
                    scene,
                    cur,
                    goal.target_xy,
                    why=door_why,
                    controllers=self.ctx.get("controllers"),
                    target_yaw=goal.target_yaw,
                    gait_profile=env.g,
                    gait_cum_dist=float(env.st.gait_cum_dist),
                    allow_camera_safe_fallback=bool(
                        navigation_fallback.get("force_straight")
                    ),
                    navigation_fallback_out=(
                        navigation_fallback
                        if getattr(scene, "egocentric_camera_only_collision", True)
                        or not getattr(scene, "strict_acceptance", False)
                        else None
                    ),
                ):
                    scene.nav_grid.transit_avoid.clear()
                    scene.nav_grid.transit_avoid.update(nav_transit_restore)
                    self._report_blocked(
                        goal,
                        (
                            door_why[-1]
                            if door_why
                            else f"a closed door blocks the way to <{goal.target_key}>"
                        ),
                    )
                    return
                if navigation_fallback.get("force_straight"):
                    scene.egocentric_collision_fallback_used = True
                    goal.meta["_egocentric_navigation_fallback"] = True
                    goal.meta["_camera_safe_navigation_fallback"] = True
                if navigation_fallback.get("path"):
                    scene.egocentric_collision_fallback_used = True
                    goal.meta["_egocentric_navigation_fallback"] = True
                    goal.meta["_camera_safe_navigation_path"] = navigation_fallback[
                        "path"
                    ]
            else:
                stand_key = s3._navigation_stand_key(scene, goal, followup_goal)
                _stand, _yaw = scene.stand_pose_for(stand_key)
                goal.target_xy = (float(_stand[0]), float(_stand[1]))
                goal.target_yaw = float(_yaw)

        if (
            goal.prim is prim.OPEN
            and goal.obj_key not in (getattr(scene, "open_objs", None) or set())
            and not s3._select_runtime_door_swing(
                scene,
                self.ctx.get("controllers"),
                goal.obj_key,
                tuple(env.st.root_xy),
            )
        ):
            logger.warning(
                "[runtime-door] '%s': neither swing direction passed the local clearance "
                "preview; preserving the authored direction",
                goal.obj_key,
            )
        prepare_front_stand = getattr(s3, "_prepare_front_opening_action_stand", None)
        front_ready, front_detail = (
            prepare_front_stand(
                env,
                ph,
                stage,
                rig,
                scene,
                goal,
                types.SimpleNamespace(pick_stand_runtime=self._timed_pick_stand),
                controllers=self.ctx.get("controllers"),
                open_doors_fn=self._timed_open_doors,
                run_goal_fn=self._timed_run_goal,
            )
            if callable(prepare_front_stand)
            else (True, None)
        )
        if not front_ready:
            self._report_blocked(
                goal,
                front_detail or "camera-safe sweep stand unavailable",
                reason=PrimitiveError.NO_COLLISION_FREE_STANCE,
            )
            return
        try:
            _obs, _done, info = self._timed_run_goal(goal)
        finally:
            if nav_transit_restore is not None:
                scene.nav_grid.transit_avoid.clear()
                scene.nav_grid.transit_avoid.update(nav_transit_restore)
        reason = info["reason"]
        if reason is PrimitiveError.OK and goal.prim in (prim.PUSH, prim.PULL):
            self._world_epoch += 1
            self._fk_clear()

        retry_request = (
            s3._retry_stand_search_request(env, scene, goal)
            if reason
            in (PrimitiveError.OUT_OF_REACH, PrimitiveError.TRAJECTORY_COLLISION)
            and getattr(scene, "nav_grid", None) is not None
            else None
        )
        if retry_request is not None:
            reach_key, search_kwargs = retry_request
            base_search_kwargs = dict(search_kwargs)
            openable_action_kind = base_search_kwargs.pop(
                "openable_action_kind",
                None,
            )
            s3._apply_openable_content_action_reach_tolerance(
                scene,
                goal,
                reach_key,
                action_kind=openable_action_kind,
            )
            content_preview = openable_action_kind in ("insert", "extract")
            if content_preview:
                base_search_kwargs["require_reach"] = False
                base_search_kwargs["action_state"] = env.st
            front_standoffs = s3._front_access_standoffs_for_goal(goal, scene)
            for standoff_index, front_standoff in enumerate(front_standoffs):
                cur = tuple(env.st.root_xy)
                reach_navigation_fallback = {}
                effective_search_kwargs = dict(base_search_kwargs)
                effective_search_kwargs["front_access_standoff_m"] = front_standoff
                pstand = self._timed_pick_stand(
                    ph,
                    stage,
                    rig,
                    scene,
                    reach_key,
                    cur,
                    **effective_search_kwargs,
                    route_gate=s3._ego_route_gate(scene),
                    start_yaw=float(env.st.root_yaw),
                    gait_profile=env.g,
                    gait_cum_dist=float(env.st.gait_cum_dist),
                    openable_action_phase=goal.prim in (prim.OPEN, prim.CLOSE),
                    openable_action_kind=openable_action_kind,
                    navigation_fallback_out=(
                        reach_navigation_fallback
                        if getattr(scene, "egocentric_camera_only_collision", True)
                        or not getattr(scene, "strict_acceptance", False)
                        else None
                    ),
                    candidate_validator=s3._openable_candidate_validator(
                        env,
                        rig,
                        scene,
                        goal,
                        reach_key,
                        action_kind=openable_action_kind,
                    ),
                )
                if pstand is None:
                    continue
                needs_move = s3._stand_pose_changed(env.st, pstand)
                doors_ready = not needs_move or self._timed_open_doors(
                    env,
                    ph,
                    stage,
                    rig,
                    scene,
                    cur,
                    pstand[0],
                    controllers=self.ctx.get("controllers"),
                    target_yaw=pstand[1],
                    gait_profile=env.g,
                    gait_cum_dist=float(env.st.gait_cum_dist),
                    allow_camera_safe_fallback=bool(
                        reach_navigation_fallback.get("force_straight")
                    ),
                    navigation_fallback_out=(
                        reach_navigation_fallback
                        if getattr(scene, "egocentric_camera_only_collision", True)
                        or not getattr(scene, "strict_acceptance", False)
                        else None
                    ),
                )
                if not doors_ready:
                    break
                if not getattr(scene, "strict_acceptance", False):
                    scene.egocentric_collision_fallback_used = True
                    goal.meta["_egocentric_reach_tol_m"] = se._OPENABLE_REACH_TOL_M
                nav_meta = {}
                if reach_navigation_fallback.get("force_straight"):
                    nav_meta["_camera_safe_navigation_fallback"] = True
                if reach_navigation_fallback.get("path"):
                    nav_meta["_camera_safe_navigation_path"] = (
                        reach_navigation_fallback["path"]
                    )
                if needs_move:
                    _obs, _done, navigation_info = self._timed_run_goal(
                        Goal(
                            prim.NAVIGATE_TO,
                            target_xy=pstand[0],
                            target_yaw=pstand[1],
                            meta=nav_meta,
                        )
                    )
                    if navigation_info["reason"] is not PrimitiveError.OK:
                        info = navigation_info
                        break
                if placement_retry is not None:
                    placement_retry = self._capture_inside_retry(
                        goal, len(env.motion.frames)
                    )
                _obs, _done, info = self._timed_run_goal(goal)
                if info["reason"] not in (
                    PrimitiveError.OUT_OF_REACH,
                    PrimitiveError.TRAJECTORY_COLLISION,
                ):
                    break
                if standoff_index + 1 < len(front_standoffs):
                    logger.info(
                        "[reach-renav] %s: nominal front-opening stand remained "
                        "unreachable; trying the next validated standoff",
                        reach_key,
                    )
        info, placement_retry = self._retry_place_inside_from_alternate_space(
            goal, info, placement_retry
        )
        reason = info["reason"]

        self._settle_placement_if_needed(scene, goal, prev)

        placement_status = (
            se.placement_relation_status(scene, goal)
            if placement_retry is not None and reason is PrimitiveError.OK
            else None
        )
        if placement_status is not None and not placement_status["ok"]:
            failed_xy = self._failed_inside_xy(placement_retry, goal)
            attempt = int(goal.meta.get("_placement_retry_attempt", 0))
            can_rechoose = goal.place_xy is None
            if can_rechoose and attempt + 1 < PLACEMENT_SETTLE_ATTEMPTS:
                self._restore_inside_retry(placement_retry)
                next_attempt = _mark_placement_retry(
                    goal.meta,
                    failed_xy,
                    inside=placement_status["predicate"] == "inside",
                    reject_space=(
                        attempt > 0
                        and se.placement_space_count(
                            scene, str(placement_retry["target"])
                        )
                        > 1
                    ),
                )
                logger.warning(
                    "placement settle retry %d/%d: live %s(%s,%s) is false; "
                    "trying a different geometry-backed site",
                    next_attempt + 1,
                    PLACEMENT_SETTLE_ATTEMPTS,
                    placement_status["predicate"].upper(),
                    placement_status["subject"],
                    placement_status["target"],
                )
                self._drive_goal(goal)
                return
            detail = (
                f"live {placement_status['predicate'].upper()} relation is false after "
                f"{attempt + 1} geometry-backed settle attempt(s): "
                f"{placement_status['subject']} -> {placement_status['target']}"
            )
            _replace_goal_report(
                env.reports, goal, PrimitiveError.UNSTABLE_AFTER_RELEASE
            )
            self._note_outcome(goal, PrimitiveError.UNSTABLE_AFTER_RELEASE, detail)
            logger.error("placement failed closed: %s", detail)

    def _ensure_settle_clock(self) -> None:
        """Lazy-init the render=FREE physics clock the per-command settle needs, via the object_physics
        side-channel ``SimulationContext`` (``step(render=False)``). RATIONALE: the RL rollout must NEVER
        render -- (1) symbolic-obs training renders zero frames (the VH "only render needed frames" win),
        and (2) ``fpv_render.advance_once`` -> ``rep.orchestrator.step`` HANGS on the MAST nodes whose RTX
        render is regressed (Hydra deviceMask:0), whereas ``get_sim().step(render=False)`` runs PhysX fine
        there (validated: isaac_phys_smoke PHYSICS_OK on MAST). Keeps >=60 Hz substeps so a placement drop
        cannot tunnel a shelf; a short warmup because the first steps do not integrate physics.
        """
        if getattr(self, "_clock_started", False):
            return
        import episode_runtime as s3
        import object_physics

        self.ctx["ph"].set_scene_timestep(self.stage, s3.FPS, substeps=5)
        self._sim = object_physics.get_sim()
        for _ in range(4):
            self._sim.step(render=False)
        self._clock_started = True

    def _settle_step(self) -> None:
        self._sim.step(
            render=False
        )  # render=FREE PhysX advance (see _ensure_settle_clock)

    # ---- terminal reward (M2): count-based goal score + dirty-episode penalty flag --------------
    def reward(self) -> dict:
        """Terminal reward ingredients for the AgentLoop: the count-based goal score (n_done/n_total) plus a
        DIRTY quality flag for any human/object contact over the whole accumulated motion. Task success is
        reported independently for ordinary contact; the acceptance runner separately rejects only the
        compliance grader's deep/sustained (egregious) geometry failures."""
        with self._timed("reward.total"):
            sc = self.score()
            comp = self._compliance()
            sc["dirty"] = bool(
                comp is not None
                and (
                    comp.get("human_clean") is False
                    or comp.get("objects_clean") is False
                    or not comp.get("ok", True)
                )
            )
            sc["compliance"] = comp
            return sc

    def _compliance(self) -> dict | None:
        """Universal post-action physics check over the accumulated motion (per-frame human-vs-scene +
        object-vs-object penetration), reused from run_render via imports. None when the scene opts out
        (no post_action_check / no oracle / no frames)."""
        import action_check
        import episode_runtime as s3
        import smplh_anim_core as core

        env, scene, stage = self.env, self.scene_obj, self.stage
        ph, rig = self.ctx["ph"], self.ctx["rig"]
        frames = env.motion.frames
        if not (
            getattr(scene, "post_action_check", False)
            and getattr(scene, "cworld", None) is not None
            and frames
        ):
            return None
        with self._timed("reward.solid_boxes"):
            nb_static = scene.solid_boxes()
        # The O(len(frames)) forward-kinematics pass over the WHOLE accumulated motion is served from the
        # per-command cache (_fk_sync realizes each command's new frames in step()); only frames the cache
        # does not already cover are realized here.
        with self._timed("reward.compliance_fk"):
            jtraj = self._fk_sync(scene, nb_static, frames)
        controllers = self.ctx.get("controllers") or {}
        last_applied: dict[str, float] | None = None

        def apply_sample_state(frame_index: int) -> None:
            nonlocal last_applied
            state = self._fk_openable_post[frame_index]
            if state == last_applied:
                return
            s3._apply_openable_state(controllers, state)
            last_applied = state

        sample_state_fn = apply_sample_state if controllers else None
        try:
            with self._timed("reward.compliance_run"):
                comp, _frames = action_check.run_action_compliance(
                    ph,
                    stage,
                    rig,
                    core,
                    scene,
                    scene.cworld,
                    frames,
                    joints_traj=jtraj,
                    settle_objects=False,
                    sample_state_fn=sample_state_fn,
                )
        finally:
            if controllers and self._fk_openable_post:
                s3._apply_openable_state(controllers, self._fk_openable_post[-1])
        nudge = self._nudge_telemetry(len(frames))
        comp["nudge"] = nudge
        if self.strict_acceptance:
            human_ok = (
                bool(comp.get("fpv_eye_clear"))
                if comp.get("egocentric_fallback_requested")
                else bool(comp.get("human_clean"))
            )
            strict_ok = human_ok and bool(comp.get("objects_clean"))
            strict_ok = strict_ok and not nudge["saturated"] and not nudge["repeated"]
            comp["strict_ok"] = strict_ok
            comp["ok"] = strict_ok
        return comp

    def _nudge_telemetry(self, frame_count: int) -> dict[str, object]:
        import humanoid_motion as wpp
        import numpy as np

        distances = [
            float(np.linalg.norm(displacement)) for displacement, _ in self._fk_post
        ]
        leans = [abs(float(lean)) for _, lean in self._fk_post]
        active = sum(
            distance > 1e-4 or lean > math.radians(0.1)
            for distance, lean in zip(distances, leans)
        )
        repeated_limit = max(6, math.ceil(max(1, frame_count) * 0.10))
        max_distance = max(distances, default=0.0)
        max_lean = max(leans, default=0.0)
        return {
            "active_frames": active,
            "frame_count": frame_count,
            "max_displacement_m": max_distance,
            "max_lean_deg": math.degrees(max_lean),
            "repeated": active >= repeated_limit,
            "saturated": (
                max_distance >= 0.95 * float(wpp._NUDGE_MAG_M)
                or max_lean >= 0.95 * float(wpp._NUDGE_LEAN_MAX)
            ),
        }

    def _timed_nudge_boxes(self, s3, scene, controllers, nb_static, fr):
        """``episode_runtime._nudge_boxes_for`` under a per-frame timer -- it runs once per motion frame in the
        compliance FK pass, so it is the natural suspect if that pass dominates."""
        with self._timed("reward.nudge_boxes"):
            return s3._nudge_boxes_for(scene, controllers, nb_static, fr)

    def _body_nudge_boxes(self, boxes):
        """Match the renderer's body-nudge policy unless strict mode requests the raw trajectory."""
        if getattr(self, "strict_acceptance", False):
            return None
        import episode_runtime as s3

        return s3._body_nudge_boxes(self.scene_obj, boxes)

    # ---- compliance FK: joints-only realize + per-command incremental cache ----------------------
    @contextmanager
    def _joints_only_fk(self) -> "Iterator[None]":
        """Swap ``smplh_fpv.fwd`` (full 6890-vertex LBS) for the joints-only ``fwd_joints`` for the block.

        ``wpp.realize`` -> ``gait.pose_place`` -> ``H.fwd(p) -> (V, J)``, but the compliance pass consumes
        ONLY ``realize(...)[1]`` (the joints) and the nudge itself reads only joints, so every vertex is
        computed and thrown away. ``fwd_joints`` returns the SAME joints by construction: with betas=0 the
        shaped template is bit-identical to the rest template, so both paths feed the identical
        ``batch_rigid_transform`` (proven empirically frame-by-frame by
        the compliance equivalence check with ``np.array_equal``). The stand-in ``V`` is an EMPTY (0,3)
        array, which survives every vertex op on this path (``V @ Rz.T``, ``V + off``, ``V + d``) and is
        discarded by the caller. ``ISAAC_COMPLIANCE_FULL_FK=1`` restores the full-vertex FK.

        Module-attribute swap restored in ``finally`` rather than a physical-world edit -- the same
        boundary technique ``switch_scene`` uses for ``fpv_core.launch_app``."""
        import numpy as np
        import smplh_fpv as fpv_fk

        if os.getenv("ISAAC_COMPLIANCE_FULL_FK"):
            yield
            return
        empty_v = np.zeros((0, 3), dtype=np.float32)
        orig_fwd = fpv_fk.fwd
        fpv_fk.fwd = lambda p: (empty_v, fpv_fk.fwd_joints(p))
        try:
            yield
        finally:
            fpv_fk.fwd = orig_fwd

    def _fk_clear(self) -> None:
        """Drop the compliance-FK cache. MANDATORY wherever motion.frames is cleared or replaced."""
        self._fk_j, self._fk_fr, self._fk_post = [], [], []
        self._fk_obj = []
        self._fk_boxes = []
        self._fk_boxes_by_frame_id = {}
        self._fk_cam, self._fk_cam_post = [], []
        self._fk_openable_post, self._fk_openable_consumed = [], []
        self._fk_nudge_boxes_cache = {}
        self._fk_env, self._fk_grid, self._fk_wall_sdf = None, None, None
        self._fk_static_boxes = None
        self._camera_adapter = None

    def _fk_note_frames(self) -> None:
        """Realize + cache the joints of every motion frame produced so far. Called once per ``step()``,
        AFTER the command's per-placement settle has rewritten its rest frames, so what is cached is final.
        """
        scene = self.scene_obj
        frames = self.env.motion.frames if self.env is not None else []
        needs_compliance = bool(
            getattr(scene, "post_action_check", False)
            and getattr(scene, "cworld", None) is not None
        )
        if not frames or not (needs_compliance or self.render_obs):
            return
        with self._timed("step.compliance_fk"):
            self._fk_sync(scene, scene.solid_boxes(), frames)

    @staticmethod
    def _fk_floats(v) -> tuple:
        import numpy as np

        return tuple(float(x) for x in np.asarray(v, dtype=np.float64).ravel())

    def _fk_env_key(self, s3, scene, controllers, nb_static) -> tuple:
        """Fingerprint of every NON-frame input the nudge reads, so a cached joint array can never outlive
        the world it was computed in. Openable position is replayed chronologically from its episode-start
        state, so the key contains immutable controller geometry rather than the stage's current final angle.
        The nav grid is compared by object identity separately."""
        probe = types.SimpleNamespace(obj=("hide", None))
        replay = s3._controllers_at_openable_state(
            controllers, s3._initial_openable_state(controllers)
        )
        boxes = s3._nudge_boxes_for(scene, replay, nb_static, probe)
        box_key = tuple(
            (str(n), self._fk_floats(mn), self._fk_floats(mx)) for n, mn, mx in boxes
        )
        ctl_key = tuple(
            (
                str(name),
                str(c.get("kind", "")),
                float(c.get("travel", 0.0)),
                self._fk_floats(c.get("world_axis", ())),
                self._fk_floats(c.get("closed_handle", ())),
                self._fk_floats(c.get("hinge_xy", ())),
                float(c.get("panel_w", 0.0)),
                float(c.get("swing_angle", 0.0)),
                float(c.get("swing_sign", 0.0)),
                tuple(self._fk_floats(corner) for corner in c.get("leaf_aabb", ())),
            )
            for name, c in sorted((controllers or {}).items())
        )
        return (box_key, ctl_key)

    def _fk_sync(self, scene, nb_static, frames: list) -> list:
        """Bring the per-episode FK cache up to ``frames`` and return the joints trajectory.

        ``wpp.realize(rig, fr, scene_boxes=..., nudge_state=ns)`` is a RECURRENCE, not a pure function:
        ``nudge_frame`` reads ``ns["d"]``/``ns["lean"]`` and writes them back (exponential smoothing +
        a per-frame rate cap), so frame i's joints depend on frames 0..i-1. Frames may therefore only be
        realized IN ORDER and none may be skipped. ``motion.frames`` is append-only and a command's settle
        rewrites only its OWN new frames, so realizing each command's tail as it lands evolves ``ns``
        exactly as one end-of-episode forward pass would.

        The cache is SELF-VALIDATING rather than trusting that argument:
          * each entry keeps a reference to the exact Frame OBJECT it was realized from, so an
            ``action_check._rewrite_rests`` replacement (``dataclasses.replace`` -> a NEW object) is caught
            by an ``is`` compare, and that index and everything after it are re-realized;
          * the whole cache is dropped if any non-frame nudge input changed (solid boxes, controller
            geometry, the nav grid or its wall SDF);
          * ``ns`` is snapshotted after every frame, so a rebuild from index k resumes from EXACTLY the
            state a single forward pass would have had there.
        Worst case it degrades to the original full recompute; it can never return different joints.
        """
        import episode_runtime as s3

        controllers = self.ctx["controllers"]
        grid = getattr(scene, "nav_grid", None) or getattr(scene, "wall_sdf_grid", None)
        sdf = getattr(grid, "wall_sdf", None) if grid is not None else None
        env_key = self._fk_env_key(s3, scene, controllers, nb_static)
        if (
            env_key != self._fk_env
            or self._fk_grid is not grid
            or self._fk_wall_sdf is not sdf
        ):
            self._fk_clear()
        k = 0
        n_cached = min(len(self._fk_fr), len(frames))
        while k < n_cached and self._fk_fr[k] is frames[k]:
            k += 1
        if k < len(
            self._fk_j
        ):  # a frame was rewritten (or removed) -> drop it and its successors
            for cached_frame in self._fk_fr[k:]:
                self._fk_boxes_by_frame_id.pop(id(cached_frame), None)
            del self._fk_j[k:]
            del self._fk_obj[k:]
            del self._fk_fr[k:]
            del self._fk_post[k:]
            del self._fk_boxes[k:]
            del self._fk_cam[k:]
            del self._fk_cam_post[k:]
            del self._fk_openable_post[k:]
            del self._fk_openable_consumed[k:]
        if k < len(frames):
            self._fk_realize(s3, scene, controllers, nb_static, grid, frames, k)
        self._fk_env, self._fk_grid, self._fk_wall_sdf = env_key, grid, sdf
        self._fk_static_boxes = nb_static
        self._counters["camera_yaw_clamps"] = sum(
            int(output.diagnostics.physiological_yaw_clamped) for output in self._fk_cam
        )
        self._counters["camera_pitch_clamps"] = sum(
            int(output.diagnostics.physiological_pitch_clamped)
            for output in self._fk_cam
        )
        return self._fk_j[: len(frames)]

    def _fk_realize(self, s3, scene, controllers, nb_static, grid, frames, lo) -> None:
        """Realize physical state, plus camera state when requested, in chronological order."""
        import humanoid_motion as wpp
        import numpy as np

        if not hasattr(self, "_fk_obj"):
            self._fk_obj = []
        rig = self.ctx["rig"]
        camera_state = None
        if self.collect_camera:
            import egocentric_camera
            import egocentric_camera_isaac

            if self._camera_adapter is None:
                self._camera_adapter = (
                    egocentric_camera_isaac.IsaacCameraAdapter.from_scene(
                        scene, nb_static
                    )
                )
            camera_state = (
                self._fk_cam_post[lo - 1]
                if lo > 0
                else egocentric_camera.INITIAL_CAMERA_STATE
            )
        # lo == 0 seeds EXACTLY as the original one-shot pass did (nav_grid only -> d=0, lean=0 via the
        # nudge's own .get defaults); lo > 0 resumes from the snapshot taken after frame lo-1.
        ns = {"nav_grid": grid}
        if lo > 0:
            d_prev, lean_prev = self._fk_post[lo - 1]
            ns["d"], ns["lean"] = d_prev.copy(), lean_prev
        openable_state = (
            dict(self._fk_openable_post[lo - 1])
            if lo > 0
            else s3._initial_openable_state(controllers)
        )
        pose_place_cache = {}
        with self._joints_only_fk():
            for frame_index, fr in enumerate(frames[lo:], start=lo):
                nudge_boxes = self._cached_nudge_boxes(
                    s3, scene, controllers, nb_static, fr, openable_state
                )
                ns["camera_boxes"] = nudge_boxes
                ns["frame_index"] = frame_index
                _vertices, joints, object_transform = wpp.realize(
                    rig,
                    fr,
                    scene_boxes=self._body_nudge_boxes(nudge_boxes),
                    nudge_state=ns,
                    pose_place_cache=pose_place_cache,
                )
                next_openable_state, consumed = s3._step_openable_state(
                    controllers, fr, openable_state
                )
                if consumed:
                    frame_boxes = self._cached_nudge_boxes(
                        s3,
                        scene,
                        controllers,
                        nb_static,
                        fr,
                        next_openable_state,
                    )
                else:
                    frame_boxes = nudge_boxes
                self._fk_j.append(joints)
                self._fk_obj.append(object_transform)
                self._fk_fr.append(fr)
                self._fk_boxes.append(frame_boxes)
                self._fk_boxes_by_frame_id[id(fr)] = frame_boxes
                self._fk_post.append(
                    (
                        np.array(ns.get("d", np.zeros(3)), dtype=np.float64),
                        float(ns.get("lean", 0.0)),
                    )
                )
                if self.collect_camera:
                    frame_adapter = self._camera_adapter
                    if float(getattr(fr, "zoom", None) or 0.0) > 0.0:
                        frame_adapter = frame_adapter.with_solid_boxes(frame_boxes)
                    camera_frame = frame_adapter.camera_frame(
                        fr,
                        joints,
                        object_transform,
                        dt_seconds=1.0 / s3.FPS,
                    )
                    camera_step = egocentric_camera.step_camera(
                        camera_frame,
                        camera_state,
                        framing=frame_adapter.framing,
                    )
                    camera_state = camera_step.state
                    self._fk_cam.append(camera_step.output)
                    self._fk_cam_post.append(camera_state)
                openable_state = next_openable_state
                self._fk_openable_post.append(dict(openable_state))
                self._fk_openable_consumed.append(bool(consumed))

    def _cached_nudge_boxes(
        self, s3, scene, controllers, nb_static, fr, openable_state
    ) -> tuple:
        """Return the exact per-frame boxes, reusing states repeated across motion frames."""
        mode = (os.getenv("ISAAC_NUDGEBOX_STATE_CACHE") or "1").strip().lower()
        key = (
            tuple(
                (str(name), float(value))
                for name, value in sorted(openable_state.items())
            ),
            tuple(
                str(path) for path in getattr(fr, "collision_exclude_paths", ()) or ()
            ),
        )
        if mode in {"0", "off", "false", "no"}:
            replay_controllers = s3._controllers_at_openable_state(
                controllers, openable_state
            )
            return tuple(
                self._timed_nudge_boxes(s3, scene, replay_controllers, nb_static, fr)
            )
        cached = self._fk_nudge_boxes_cache.get(key)
        if cached is not None:
            if mode in {"verify", "shadow"}:
                replay_controllers = s3._controllers_at_openable_state(
                    controllers, openable_state
                )
                fresh = tuple(
                    self._timed_nudge_boxes(
                        s3, scene, replay_controllers, nb_static, fr
                    )
                )
                if self._nudge_boxes_signature(fresh) != self._nudge_boxes_signature(
                    cached
                ):
                    raise AssertionError("per-state nudge-box cache mismatch")
            return cached
        replay_controllers = s3._controllers_at_openable_state(
            controllers, openable_state
        )
        boxes = tuple(
            self._timed_nudge_boxes(s3, scene, replay_controllers, nb_static, fr)
        )
        self._fk_nudge_boxes_cache[key] = boxes
        return boxes

    @classmethod
    def _nudge_boxes_signature(cls, boxes) -> tuple:
        return tuple(
            (str(name), cls._fk_floats(minimum), cls._fk_floats(maximum))
            for name, minimum, maximum in boxes
        )

    def _ensure_render_cache(self) -> None:
        frames = self.env.motion.frames if self.env is not None else []
        if not frames:
            return
        boxes = self._fk_static_boxes
        if boxes is None:
            boxes = self.scene_obj.solid_boxes()
        self._fk_sync(self.scene_obj, boxes, frames)

    def _realize_render_frame(self, frame_index: int, *, cache_ready: bool = False):
        """Reconstruct one full-mesh pose from the exact pre-frame recurrence state."""
        import episode_runtime as s3
        import humanoid_motion as wpp
        import numpy as np

        if not cache_ready:
            self._ensure_render_cache()
        frames = self.env.motion.frames
        frame = frames[frame_index]
        grid = getattr(self.scene_obj, "nav_grid", None) or getattr(
            self.scene_obj, "wall_sdf_grid", None
        )
        nudge_state = {"nav_grid": grid}
        if frame_index > 0:
            displacement, lean = self._fk_post[frame_index - 1]
            nudge_state["d"] = np.asarray(displacement, dtype=np.float64).copy()
            nudge_state["lean"] = float(lean)
        openable_state = (
            self._fk_openable_post[frame_index - 1]
            if frame_index > 0
            else s3._initial_openable_state(self.ctx["controllers"])
        )
        replay_controllers = s3._controllers_at_openable_state(
            self.ctx["controllers"], openable_state
        )
        boxes = self._timed_nudge_boxes(
            s3,
            self.scene_obj,
            replay_controllers,
            self._fk_static_boxes,
            frame,
        )
        nudge_state["camera_boxes"] = boxes
        nudge_state["frame_index"] = frame_index
        return wpp.realize(
            self.ctx["rig"],
            frame,
            scene_boxes=self._body_nudge_boxes(boxes),
            nudge_state=nudge_state,
        )

    def close(self, hard: bool = True) -> None:
        """Tear down the app. Isaac segfaults in its atexit/plugin teardown on this build, so a long-lived
        env process should ``os._exit`` after a deliberate ``app.close()`` (results already persisted).
        ``hard=False`` skips the exit for callers that manage the process lifetime themselves.
        """
        self.finalize_render_parity_trace()
        try:
            if self.app is not None:
                self.app.close()
        finally:
            self.app = None
            if hard:
                os._exit(0)

    def finalize_render_parity_trace(self) -> str:
        """Write the buffered online trace with the episode's final trajectory identity."""
        session = getattr(self, "_online_render_trace", None)
        if session is None:
            return str(getattr(self, "_render_parity_trace_status", "inactive"))
        frames = self.env.motion.frames if self.env is not None else []
        try:
            result = session.finish(frames)
        except Exception:
            self._render_parity_trace_status = "error"
            self._online_render_trace = None
            raise
        status = result.value if hasattr(result, "value") else str(result)
        self._render_parity_trace_status = status
        self._online_render_trace = None
        if status == "aborted":
            logger.warning("renderer parity trace was aborted before finalization")
        return status

    # ---- observation + reward -------------------------------------------------------------------
    def _render_one(
        self,
        fr,
        frame_index: int | None = None,
        *,
        advance_simulation: bool = True,
        consume_temporal_reset: bool = True,
    ):
        """Render one frame using the shared dense trajectory and readback policies."""
        vertices, _joints, object_transform, camera_output, drove_openable = (
            self._render_frame_state(fr, frame_index)
        )
        dynamic_replay = None
        physx_owned = False
        if self.ctx.get("fdp_paths") and frame_index is not None:
            frames = self.env.motion.frames
            dynamic_replay = self._make_dynamic_replay(frames, frame_index, _joints)
            dynamic_replay.before_frame(fr, _joints, object_frame=not drove_openable)
            physx_owned = dynamic_replay.owns_frame_object(
                fr, object_frame=not drove_openable
            )
            if self._dynamic_focus_path != dynamic_replay.focus_path:
                self._dynamic_free_focus = None
                self._dynamic_focus_path = dynamic_replay.focus_path
            state = self._dynamic_camera_state_before(frame_index)
            camera_output, state = self._step_dynamic_camera(
                fr,
                _joints,
                object_transform,
                state,
                self._dynamic_free_focus,
                physx_owned,
            )
            self._fk_cam[frame_index] = camera_output
            self._fk_cam_post[frame_index] = state
        else:
            self._ensure_dynamic_render_proxy(_joints, update=True)
        self._apply_clutter_render_frame(frame_index)
        self._author_render_frame(
            fr,
            vertices,
            None if physx_owned else object_transform,
            drove_openable,
        )
        self._apply_visual_state_render_frame(frame_index)
        self._drive_render_camera(camera_output)
        self._converge_render_history(consume=consume_temporal_reset)
        if dynamic_replay is None or not advance_simulation:
            return self._read_render_frame(
                advance=advance_simulation,
                frame_index=frame_index,
                frame=fr,
                joints=_joints,
            )
        import fpv_render

        fpv_render.advance_once(self.ctx["rcfg"])
        positions = dynamic_replay.after_step()
        focus = dynamic_replay.focus_from(positions)
        if focus is not None:
            self._dynamic_free_focus = focus
            self._dynamic_focus_path = dynamic_replay.focus_path
        return self._read_render_frame(
            advance=False,
            frame_index=frame_index,
            frame=fr,
            joints=_joints,
        )

    def _converge_render_history(self, *, consume: bool = True) -> int:
        """Clear cross-episode temporal history at fixed simulation time."""
        if not self._temporal_reset_pending:
            return 0
        import fpv_render

        refresh_frames = 16
        fpv_render.converge_at_current_time(self.ctx["rcfg"])
        if consume:
            self._temporal_reset_pending = False
        return refresh_frames

    def _render_frame_state(
        self, fr, frame_index: int | None, *, cache_ready: bool = False
    ):
        """Resolve body, object, openable and camera state for one global frame."""
        import egocentric_camera
        import episode_runtime as s3
        import humanoid_motion as wpp

        rig = self.ctx["rig"]
        if frame_index is None:
            vertices, joints, object_transform = wpp.realize(rig, fr)
            adapter = self._camera_adapter_for_frame(fr)
            camera_frame = adapter.camera_frame(
                fr, joints, object_transform, dt_seconds=1.0 / s3.FPS
            )
            camera_output = egocentric_camera.step_camera(
                camera_frame,
                framing=adapter.framing,
            ).output
        else:
            vertices, joints, object_transform = self._realize_render_frame(
                frame_index, cache_ready=cache_ready
            )
            camera_output = self._fk_cam[frame_index]
        if frame_index is None:
            drove_openable = s3._drive_drawers(self.ctx["controllers"], fr)
        else:
            s3._apply_openable_state(
                self.ctx["controllers"], self._fk_openable_post[frame_index]
            )
            drove_openable = self._fk_openable_consumed[frame_index]
        return vertices, joints, object_transform, camera_output, drove_openable

    def _camera_adapter_for_frame(self, frame):
        """Resolve focus visibility against this frame's chronological geometry."""
        import egocentric_camera_isaac
        import episode_runtime as s3

        base = self._camera_adapter
        if base is None:
            static_boxes = self._fk_static_boxes or self.scene_obj.solid_boxes()
            base = egocentric_camera_isaac.IsaacCameraAdapter.from_scene(
                self.scene_obj, static_boxes
            )
            self._camera_adapter = base
        boxes = self._fk_boxes_by_frame_id.get(id(frame))
        if boxes is None:
            static_boxes = self._fk_static_boxes or self.scene_obj.solid_boxes()
            boxes = tuple(
                s3._nudge_boxes_for(
                    self.scene_obj,
                    self.ctx["controllers"],
                    static_boxes,
                    frame,
                )
            )
        return base.with_solid_boxes(boxes)

    def _author_render_frame(
        self, fr, vertices, object_transform, drove_openable: bool
    ) -> None:
        """Author the shared per-frame body and manipulated-object state."""
        import smplh_anim_core as core

        rig = self.ctx["rig"]
        # Draw the human body mesh so the FPV shows the wearer's arm/hand holding the object (not a floating
        # object). setup_episode builds the rig + wires _body_path (enable_physics_refetch) but never AUTHORS
        # the /World/Body mesh (RL training renders no human), so author it once here from this frame's verts,
        # then rewrite its points every frame -- the exact per-frame body draw run_render does
        # (author_body_once episode_runtime.py:5037, update_body on_frame:5457).
        if not self._body_authored:
            core.author_body_once(self.stage, rig, vertices)
            self._body_authored = True
        core.update_body(rig, vertices)
        # Re-pose the manipulated object to THIS frame's pose (port of episode_runtime.on_frame:5487). Without it,
        # mid-motion frames (LookAt raise/hold, grab carry) render the object at its physics REST pose while the
        # camera frames the raised position -> the inspected bottle is invisible (we see the background). The
        # final-frame render is unaffected (obj_T there == the already-settled pose).
        if object_transform is not None and not drove_openable:
            _op = getattr(fr, "obj_path", None)
            if _op:
                self._make_render_object_kinematic(_op)
                rig._obj_path = _op
                core.set_object_world(rig, object_transform)
            elif not getattr(self.scene_obj, "grasp_objs", None):
                self._make_render_object_kinematic(rig._obj_path)
                core.set_object_world(rig, object_transform)
            for _cp, _rel in getattr(fr, "carried", ()) or ():
                self._make_render_object_kinematic(_cp)
                rig._obj_path = _cp
                core.set_object_world(rig, object_transform @ _rel)
            if getattr(fr, "carried", None) and _op:
                rig._obj_path = _op

    def _make_render_object_kinematic(self, path: str) -> None:
        """Make a scripted object pose authorable; a later free-dynamic frame releases it."""
        if not path:
            return
        import physics_human as ph
        from pxr import UsdPhysics

        prim = self.stage.GetPrimAtPath(str(path))
        if prim is not None and prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            ph.set_rigid_kinematic(prim, True)

    def _drive_render_camera(
        self,
        camera_output,
    ) -> None:
        """Author one output from the shared camera policy to the USD camera."""
        import numpy as np

        pose = self._camera_render_pose(camera_output)
        self._last_canonical_render_pose = pose
        self._apply_render_camera_pose(pose)
        self._last_render_pose = pose
        self._last_delivered_pose = pose
        self._last_eye = (
            np.asarray(pose.eye, dtype=np.float64),
            np.asarray(pose.look, dtype=np.float64),
            pose.focal_mm,
        )

    @staticmethod
    def _camera_render_pose(camera_output):
        import fpv_render

        return fpv_render.CameraRenderPose(
            tuple(float(value) for value in camera_output.eye),
            tuple(float(value) for value in camera_output.look),
            float(camera_output.focal_mm),
        )

    def _apply_render_camera_pose(self, pose) -> None:
        """Author a camera-only view without changing canonical camera trajectory state."""
        import numpy as np
        import smplh_camera

        cop = self.ctx["cop"]
        _fa = self.ctx.get("fpv_focal")
        if _fa is None:
            from pxr import UsdGeom

            _fa = self.ctx["fpv_focal"] = UsdGeom.Camera(
                cop.GetAttr().GetPrim()
            ).GetFocalLengthAttr()
        _fa.Set(pose.focal_mm)
        smplh_camera.drive_fpv_gaze(
            cop,
            pose.eye,
            pose.look,
            np.asarray(pose.eye) + np.asarray(pose.look),
            eye_push=0.0,
            aim_blend=1.0,
        )

    def _read_render_frame(
        self,
        *,
        advance: bool = True,
        frame_index: int | None = None,
        frame=None,
        joints=None,
    ):
        """Read current-scene pixels without camera search or pose recovery."""
        import fpv_render

        a_f, rcfg = self.ctx["a_f"], self.ctx["rcfg"]
        del frame_index, frame, joints

        if advance:
            fpv_render.advance_once(rcfg)
        readback = fpv_render.read_rgb(
            a_f,
            int(self.ctx["W"]),
            int(self.ctx["Hh"]),
            max_retries=fpv_render.DEFAULT_READBACK_RETRIES,
        )
        self._counters["fpv_readback_retries"] = int(
            self._counters.get("fpv_readback_retries", 0)
        ) + int(readback.diagnostics.retries)
        self._last_render_readback = readback
        self._last_readback_zero_dt = 0
        return readback.rgb

    def render_fpv(self):
        """Render ONE egocentric FPV frame at the human's CURRENT (final) pose -> HxWx3 uint8 (or None)."""
        if not self.render_obs or self.ctx is None or "a_f" not in self.ctx:
            return None
        frames = self.env.motion.frames if self.env is not None else []
        fr = frames[-1] if frames else self._start_frame()
        if fr is None:
            return None
        with self._timed("render.frame"):
            return self._render_one(
                fr,
                frame_index=len(frames) - 1 if frames else None,
                advance_simulation=bool(frames),
                consume_temporal_reset=bool(frames),
            )

    @staticmethod
    def _uniform_idxs(m: int, n: int) -> list[int]:
        """Take the end of each uniform time bin: ``1/n, 2/n, ..., 1``.

        For ten frames and four observations this returns ``[2, 4, 7, 9]``, corresponding to
        25%, 50%, 75%, and 100% of the action instead of spending a slot on its initial state.
        ``n<=1`` keeps only the final frame; ``m<=n`` keeps every available frame.
        """
        if m <= 0:
            return []
        if n <= 1:
            return [m - 1]
        if m <= n:
            return list(range(m))
        return sorted({((i * m + n - 1) // n) - 1 for i in range(1, n + 1)})

    @classmethod
    def _observation_idxs(cls, frames: list, n: int) -> list[int]:
        """Sample an action while guaranteeing that an inspect's informative hold survives."""
        if not frames:
            return []
        holds = [
            index
            for index, frame in enumerate(frames)
            if str(getattr(frame, "seg", "") or "") == "look_hold"
        ]
        if n <= 1:
            return [holds[len(holds) // 2] if holds else len(frames) - 1]
        indices = cls._uniform_idxs(len(frames), n)
        if not holds or set(indices) & set(holds):
            return indices
        hold = holds[len(holds) // 2]
        replace_at = max(0, len(indices) - 2)
        indices[replace_at] = hold
        return sorted(set(indices))

    def _seg_pick_indices(self, n: int) -> list[int | None]:
        frames = self.env.motion.frames if self.env is not None else []
        start = self._seg_start if self._seg_start is not None else len(frames)
        if start < len(frames):
            local = self._observation_idxs(frames[start:], n)
            if getattr(self, "_visual_state_changed_this_command", False) and local:
                local[-1] = len(frames[start:]) - 1
                local = sorted(set(local))
            return [start + index for index in local]
        return [len(frames) - 1] if frames else [None]

    def _strict_uniform_seg_pick_indices(self, n: int) -> list[int]:
        """Return exactly ``n`` end-of-bin indices from the current command."""
        frames = self.env.motion.frames if self.env is not None else []
        start = self._seg_start if self._seg_start is not None else len(frames)
        if start >= len(frames):
            return []
        count = len(frames) - start
        if n <= 1:
            return [len(frames) - 1]
        return [
            start + ((position * count + n - 1) // n) - 1
            for position in range(1, n + 1)
        ]

    def _seg_picks(self, n: int) -> list:
        """The ``n`` motion frames UNIFORMLY sampled across the LAST command's segment (final frame always
        included). No segment yet (open/reset/tick-0) -> one start/current frame.

        The SET of frames the assistant is shown, independent of whether pixels are produced from them: a
        scan verb emits 24 motion frames and only these ``n`` are ever rendered, so "the wearer saw X" must
        mean "X was in one of these", not "the body passed X at some point during the command".
        """
        frames = self.env.motion.frames if self.env is not None else []
        out = []
        for index in self._seg_pick_indices(n):
            frame = self._start_frame() if index is None else frames[index]
            if frame is not None:
                out.append(frame)
        return out

    def render_fpv_frames(self, n: int | None = None, freeze: bool = True) -> list:
        """Render up to ``n`` (default ``n_seg_frames``) egocentric FPV frames UNIFORMLY sampled across the
        LAST command's motion segment (VH extract_frames 4/interval) -> list[HxWx3]. The rig is re-posed to
        each sampled frame. A LookAt with one available slot uses its informative hold; otherwise the
        final result frame is included. No segment yet (open/reset/tick-0) -> one start/current frame.

        THE RENDER IS AN OBSERVER: it must not mutate any state the goal check reads. It otherwise does, at
        EVERY n including 1, because ``_render_one`` -> ``smplh_anim_core.set_object_world`` rewrites the
        object prim's world matrix from the motion frame -- and a ``("rest", xyz)`` frame carries a POSITION
        only, so the write silently replaces the physics-settled ORIENTATION with identity. Measured on
        bt_picking_up_trash: the settled can rests tilted 40deg with its world-AABB min-z at -0.02868 m, which
        is 0.56 mm past ``_relations_inside_live``'s z-band, so the put scores 0; the unbracketed render then
        stands the can upright (min-z -0.01325 m) and the SAME put scores 1 on the NEXT command -- a one-tick
        score lag, and an n-dependent one (n=4 held the bracket, n=1 did not, so the two disagreed at the
        terminal: 2/3 vs 3/3). So the bracket (``reset.freeze_graspables`` -> render -> ``restore_graspables``:
        graspables frozen KINEMATIC so the render's physics ticks cannot move them, exact pre-render poses
        rewritten afterward) is unconditional. ``freeze=False`` is the one escape hatch, for a REPLAY/video
        render where the terminal score is irrelevant and the bracket's cost across a DENSE frame loop (n>>6)
        once drove Isaac's replicator orchestrator to None."""
        if not self.render_obs or self.ctx is None or "a_f" not in self.ctx:
            return []
        with self._timed("render.total"):
            frame_count = self.n_seg_frames if n is None else int(n)
            self._static_no_motion_capture = False
            picks = (
                self._strict_uniform_seg_pick_indices(frame_count)
                if getattr(self, "strict_uniform_action_frames", False)
                else self._seg_pick_indices(frame_count)
            )
            frames = self.env.motion.frames if self.env is not None else []
            if getattr(self, "_visual_state_changed_this_command", False) and not picks:
                picks = [len(frames) - 1] if frames else [None]
            if (
                getattr(self, "strict_uniform_action_frames", False)
                and not picks
                and bool(getattr(self, "_last_cmd", ""))
            ):
                if getattr(self, "_allow_static_no_motion_capture", False):
                    return self._reuse_static_no_motion_delivery(frame_count, frames)
                raise RuntimeError(
                    "a zero-frame command was not explicitly accepted as no-motion; "
                    "refusing to synthesize capture frames"
                )
            clutter_replay = getattr(self, "_clutter_render_replay", None)
            saved_clutter = (
                clutter_replay.snapshot_state() if clutter_replay is not None else {}
            )
            if not freeze:
                self._render_delivery_cache = {}
                self._dense_render_delivery_cache = {}
                self._dense_render_delivery_bytes = 0
                self._dense_render_delivery_complete = True
                self._rendered_dense_key = None
                try:
                    return self._render_picks(picks)
                finally:
                    if clutter_replay is not None:
                        clutter_replay.restore_state(saved_clutter)
            cached = self._cached_render_delivery(picks, frames)
            if cached is not None:
                return cached
            saved = _reset.freeze_graspables(
                self.scene_obj,
                tuple(str(path) for path in (self.ctx.get("fdp_paths") or ())),
            )
            try:
                return self._render_picks(picks)
            finally:
                if clutter_replay is not None:
                    clutter_replay.restore_state(saved_clutter)
                _reset.restore_graspables(self.scene_obj, saved)

    def _reuse_static_no_motion_delivery(self, count: int, frames: list) -> list:
        """Repeat the current real framebuffer for a successful zero-motion action."""
        delivery = getattr(self, "_last_dense_render_delivery", None)
        expected_index = len(frames) - 1 if frames else None
        expected_frame = frames[-1] if frames else None
        if (
            count < 1
            or delivery is None
            or bool(getattr(self, "_temporal_reset_pending", False))
            or delivery.scene_identity != id(getattr(self, "scene_obj", None))
            or delivery.frame_index != expected_index
            or (
                expected_frame is not None
                and delivery.frame_identity != id(expected_frame)
            )
            or delivery.world_epoch != int(getattr(self, "_world_epoch", 0))
            or delivery.delivered_pose != getattr(self, "_last_delivered_pose", None)
        ):
            raise RuntimeError(
                "a zero-motion command has no matching prior real framebuffer; "
                "refusing to synthesize capture frames"
            )
        output = [self._copy_render_value(delivery.rgb) for _ in range(count)]
        self._eye_poses = (
            []
            if delivery.eye_pose is None
            else [self._copy_eye_pose(delivery.eye_pose) for _ in range(count)]
        )
        self._frame_segs = [delivery.segment for _ in range(count)]
        self._static_no_motion_capture = True
        return output

    def _render_picks(self, picks: list[int | None]) -> list:
        """Render each selected global frame -> list[HxWx3]. Always called inside the
        caller's freeze/restore bracket (except the ``freeze=False`` replay path).

        Also RETAINS each delivered frame's camera pose in ``_eye_poses`` -- ``_render_one`` computes eye +
        look for every frame and used to throw them away, and the visibility measurement needs exactly
        those. A frame whose readback failed never reached the model, so its pose is dropped with it.
        """
        out = []
        self._eye_poses = []
        self._frame_segs = []
        frames = self.env.motion.frames if self.env is not None else []
        cached = self._cached_render_delivery(picks, frames)
        if cached is not None:
            return cached
        dense_key = self._dense_execution_key(frames)
        if self._reuse_last_dense_delivery(picks, frames, out):
            self._store_render_delivery(picks, frames, out)
            self._rendered_dense_key = dense_key
            return out
        if (
            frames
            and int(self._seg_start or 0) >= len(frames)
            and not getattr(self, "_visual_state_changed_this_command", False)
        ):
            raise RuntimeError(
                "a zero-motion command has no matching prior render delivery; "
                "refusing to consume the camera or physics timeline again"
            )
        if getattr(self, "_rendered_dense_key", None) == dense_key:
            raise RuntimeError(
                "this dense segment was already rendered; the requested delivery sample "
                "is not available in the bounded lossless cache"
            )
        try:
            if picks == [None]:
                fr = self._start_frame()
                if fr is not None:
                    with self._timed("render.frame"):
                        rgb = self._render_one(
                            fr,
                            advance_simulation=False,
                            consume_temporal_reset=False,
                        )
                    self._store_dense_render_frame(None, fr, rgb)
                    self._record_render_pick(out, fr, rgb)
            elif getattr(self, "render_selected_only", False) and not (
                getattr(self, "_online_render_trace", None)
                or os.environ.get("ISAAC_RENDER_TRACE_PATH")
            ):
                self._render_selected_segment(frames, picks, out)
            else:
                self._render_dense_segment(frames, picks, out)
        finally:
            if frames:
                import episode_runtime as s3
                import smplh_anim_core as core

                self._ensure_render_cache()
                s3._apply_openable_state(
                    self.ctx["controllers"], self._fk_openable_post[-1]
                )
                if self._body_authored:
                    vertices, _joints, _object_transform = self._realize_render_frame(
                        len(frames) - 1, cache_ready=True
                    )
                    core.update_body(self.ctx["rig"], vertices)
        self._store_render_delivery(picks, frames, out)
        self._rendered_dense_key = dense_key
        return out

    def _render_selected_segment(
        self, frames: list, picks: list[int | None], out: list
    ) -> None:
        """Render only the uniformly selected observations from the latest command."""
        selected = sorted({int(index) for index in picks if index is not None})
        if not selected:
            return
        start = self._seg_start if self._seg_start is not None else selected[0]
        if start < 0 or selected[0] < start or selected[-1] >= len(frames):
            raise RuntimeError("selected render frame is outside the latest command")
        start_poses = getattr(self, "_render_seg_start_poses", None)
        if start_poses:
            _reset.restore_graspable_poses(self.scene_obj, start_poses)
        if start == 0:
            self._last_delivered_pose = None
            self._last_canonical_render_pose = None
        self._ensure_render_cache()
        for index in selected:
            frame = frames[index]
            with self._timed("render.frame"):
                rgb = self._render_one(
                    frame,
                    frame_index=index,
                    advance_simulation=True,
                )
            self._store_dense_render_frame(index, frame, rgb)
            self._record_render_pick(out, frame, rgb)
        self._refresh_camera_counters()

    @staticmethod
    def _copy_render_value(value):
        copier = getattr(value, "copy", None)
        return copier() if callable(copier) else value

    @classmethod
    def _copy_eye_pose(cls, pose):
        if pose is None:
            return None
        return tuple(cls._copy_render_value(value) for value in pose)

    def _store_dense_render_frame(self, index: int | None, frame, rgb) -> None:
        self._last_dense_render_delivery = _LastDenseRenderDelivery(
            scene_identity=id(getattr(self, "scene_obj", None)),
            frame_identity=id(frame),
            frame_index=index,
            world_epoch=int(getattr(self, "_world_epoch", 0)),
            rgb=self._copy_render_value(rgb),
            eye_pose=self._copy_eye_pose(getattr(self, "_last_eye", None)),
            segment=str(getattr(frame, "seg", "") or ""),
            delivered_pose=getattr(self, "_last_delivered_pose", None),
        )
        if not getattr(self, "_dense_render_delivery_complete", True):
            return
        cache = getattr(self, "_dense_render_delivery_cache", None)
        if cache is None:
            cache = self._dense_render_delivery_cache = {}
        encoded = self._encode_render_rgb(rgb)
        next_bytes = int(getattr(self, "_dense_render_delivery_bytes", 0)) + len(
            encoded
        )
        if next_bytes > _MAX_DENSE_RENDER_CACHE_BYTES:
            cache.clear()
            self._dense_render_delivery_bytes = 0
            self._dense_render_delivery_complete = False
            return
        cache[index] = (
            id(frame),
            encoded,
            self._copy_eye_pose(self._last_eye),
            str(getattr(frame, "seg", "") or ""),
            getattr(self, "_last_delivered_pose", None),
        )
        self._dense_render_delivery_bytes = next_bytes

    def _reuse_last_dense_delivery(
        self, picks: list[int | None], frames: list, out: list
    ) -> bool:
        if len(picks) != 1:
            return False
        index = picks[0]
        frame = self._start_frame() if index is None else frames[index]
        delivery = getattr(self, "_last_dense_render_delivery", None)
        if delivery is None or bool(getattr(self, "_temporal_reset_pending", False)):
            return False
        if (
            delivery.scene_identity != id(getattr(self, "scene_obj", None))
            or delivery.frame_identity != id(frame)
            or delivery.frame_index != index
            or delivery.world_epoch != int(getattr(self, "_world_epoch", 0))
            or delivery.delivered_pose != getattr(self, "_last_delivered_pose", None)
        ):
            return False
        out.append(self._copy_render_value(delivery.rgb))
        self._eye_poses = (
            []
            if delivery.eye_pose is None
            else [self._copy_eye_pose(delivery.eye_pose)]
        )
        self._frame_segs = [delivery.segment]
        return True

    @staticmethod
    def _encode_render_rgb(rgb) -> bytes:
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG", compress_level=1)
        return buffer.getvalue()

    @staticmethod
    def _decode_render_rgb(payload: bytes):
        import io

        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    def _render_delivery_key(self, picks: list[int | None], frames: list) -> tuple:
        return (
            *self._dense_execution_key(frames),
            tuple(picks),
        )

    def _dense_execution_key(self, frames: list) -> tuple:
        return (
            id(getattr(self, "scene_obj", None)),
            getattr(self, "_seg_start", None),
            tuple(id(frame) for frame in frames),
        )

    def _cached_render_delivery(
        self, picks: list[int | None], frames: list
    ) -> list | None:
        cached = getattr(self, "_render_delivery_cache", {}).get(
            self._render_delivery_key(picks, frames)
        )
        if cached is None:
            dense = getattr(self, "_dense_render_delivery_cache", {})
            entries = []
            for index in picks:
                entry = dense.get(index)
                if entry is None:
                    return None
                frame_id = entry[0]
                if index is not None and (
                    index < 0 or index >= len(frames) or frame_id != id(frames[index])
                ):
                    return None
                entries.append(entry)
            rgbs = tuple(self._decode_render_rgb(entry[1]) for entry in entries)
            poses = tuple(entry[2] for entry in entries if entry[2] is not None)
            segments = tuple(entry[3] for entry in entries)
            last_eye = entries[-1][2] if entries else None
        else:
            rgbs, poses, segments, last_eye, _last_delivered_pose = cached
        self._eye_poses = [self._copy_eye_pose(pose) for pose in poses]
        self._frame_segs = list(segments)
        self._last_eye = self._copy_eye_pose(last_eye)
        output = [self._copy_render_value(rgb) for rgb in rgbs]
        if cached is None:
            self._store_render_delivery(picks, frames, output)
        return output

    def _store_render_delivery(
        self, picks: list[int | None], frames: list, rgbs: list
    ) -> None:
        cache = getattr(self, "_render_delivery_cache", None)
        if cache is None:
            cache = self._render_delivery_cache = {}
        cache[self._render_delivery_key(picks, frames)] = (
            tuple(self._copy_render_value(rgb) for rgb in rgbs),
            tuple(self._copy_eye_pose(pose) for pose in self._eye_poses),
            tuple(self._frame_segs),
            self._copy_eye_pose(self._last_eye),
            getattr(self, "_last_delivered_pose", None),
        )

    def _online_trace_snapshot(
        self,
        trace_module,
        frame,
        frame_index: int,
        vertices,
        joints,
        object_transform,
        dynamic_replay,
        camera_pose,
        horizontal_aperture: float,
    ):
        dynamic_paths = tuple(self.ctx.get("fdp_paths") or ())
        owned_paths = (
            dynamic_replay.released_paths if dynamic_replay is not None else ()
        )
        clutter_replay = getattr(self, "_clutter_render_replay", None)
        persistent_paths = trace_module.persistent_scene_object_paths(
            self.scene_obj,
            clutter_replay.active_paths if clutter_replay is not None else (),
            trace_module.openable_controller_paths(self.ctx["controllers"]),
        )
        paths = trace_module.object_paths_for_frame(
            frame,
            self.ctx["rig"],
            dynamic_paths,
            persistent_paths,
        )
        solid_boxes = self._fk_boxes_by_frame_id.get(id(frame))
        if solid_boxes is None:
            solid_boxes = self._fk_boxes[frame_index]
        return trace_module.build_render_state_snapshot(
            frame=frame,
            camera_pose=camera_pose,
            width=self.ctx["W"],
            height=self.ctx["Hh"],
            horizontal_aperture=horizontal_aperture,
            body_vertices=vertices,
            joints=joints,
            object_transform=object_transform,
            object_world_transforms=trace_module.snapshot_world_transforms(
                self.stage, paths
            ),
            openable_state=self._fk_openable_post[frame_index],
            solid_boxes=solid_boxes,
            dynamic_state=trace_module.snapshot_dynamic_state(
                self.stage, dynamic_paths, owned_paths
            ),
            renderer="online",
        )

    def _render_dense_segment(
        self, frames: list, picks: list[int | None], out: list
    ) -> None:
        """Advance every new motion frame in order and read only the selected observations."""
        import fpv_render

        selected = {int(index) for index in picks if index is not None}
        if not selected:
            return
        start_poses = getattr(self, "_render_seg_start_poses", None)
        if start_poses:
            _reset.restore_graspable_poses(self.scene_obj, start_poses)
        raw_start = self._seg_start if self._seg_start is not None else min(selected)
        start = max(0, min(int(raw_start), len(frames)))
        if start >= len(frames):
            raise RuntimeError(
                "a zero-motion command has no matching prior render delivery; "
                "refusing to consume the camera or physics timeline again"
            )
        trace_module = None
        trace_session = getattr(self, "_online_render_trace", None)
        trace_aperture = None
        trace_enabled = trace_session is not None or bool(
            os.environ.get("ISAAC_RENDER_TRACE_PATH")
        )
        if trace_session is not None and not self._online_trace_is_active(
            trace_session
        ):
            trace_session = None
            trace_enabled = False
        if trace_enabled:
            from simulator.physical_world import (
                render_parity_integration as trace_module,
            )

            if trace_session is None:
                recorded_spans = tuple(getattr(self, "_render_command_spans", ()))
                first_start = recorded_spans[0][0] if recorded_spans else int(raw_start)
                trace_session = trace_module.maybe_create_online_episode_trace(
                    renderer="online",
                    start_index=first_start,
                )
                self._online_render_trace = trace_session
                if trace_session is not None:
                    self._render_parity_trace_status = "active"
                    self._online_trace_noted_commands = 1
                    self._sync_online_trace_command_spans()
            elif not getattr(self, "_render_command_spans", None):
                trace_session.begin_segment(int(raw_start))
            trace_aperture = trace_module.camera_horizontal_aperture(self.ctx["cop"])
        if start == 0:
            self._last_delivered_pose = None
            self._last_canonical_render_pose = None
        self._ensure_render_cache()
        dynamic_replay = None
        dynamic_camera_state = self._dynamic_camera_state_before(start)
        free_focus = self._dynamic_free_focus
        with ExitStack() as trace_stack:
            if trace_session is not None:
                trace_stack.enter_context(trace_session)
            for index in range(start, len(frames)):
                fr = frames[index]
                with self._timed("render.frame"):
                    (
                        vertices,
                        _joints,
                        object_transform,
                        camera_output,
                        drove_openable,
                    ) = self._render_frame_state(fr, index, cache_ready=True)
                    physx_owned = False
                    if self.ctx.get("fdp_paths"):
                        if dynamic_replay is None:
                            dynamic_replay = self._make_dynamic_replay(
                                frames, start, _joints
                            )
                        dynamic_replay.before_frame(
                            fr, _joints, object_frame=not drove_openable
                        )
                        physx_owned = dynamic_replay.owns_frame_object(
                            fr, object_frame=not drove_openable
                        )
                        if self._dynamic_focus_path != dynamic_replay.focus_path:
                            free_focus = None
                            self._dynamic_free_focus = None
                            self._dynamic_focus_path = dynamic_replay.focus_path
                        camera_output, dynamic_camera_state = self._step_dynamic_camera(
                            fr,
                            _joints,
                            object_transform,
                            dynamic_camera_state,
                            free_focus,
                            physx_owned,
                        )
                        self._fk_cam[index] = camera_output
                        self._fk_cam_post[index] = dynamic_camera_state
                    self._apply_clutter_render_frame(index)
                    self._author_render_frame(
                        fr,
                        vertices,
                        None if physx_owned else object_transform,
                        drove_openable,
                    )
                    self._drive_render_camera(camera_output)
                    if trace_session is not None:
                        trace_session.authored_pre_step(
                            index,
                            fr,
                            self._online_trace_snapshot(
                                trace_module,
                                fr,
                                index,
                                vertices,
                                _joints,
                                object_transform,
                                dynamic_replay,
                                self._last_render_pose,
                                trace_aperture,
                            ),
                        )
                    pre_step_refreshes = self._converge_render_history()
                    fpv_render.advance_once(self.ctx["rcfg"])
                    if dynamic_replay is not None:
                        positions = dynamic_replay.after_step()
                        next_focus = dynamic_replay.focus_from(positions)
                        if next_focus is not None:
                            free_focus = next_focus
                            self._dynamic_free_focus = next_focus
                            self._dynamic_focus_path = dynamic_replay.focus_path
                    if trace_session is not None:
                        trace_session.post_positive_step(
                            index,
                            fr,
                            self._online_trace_snapshot(
                                trace_module,
                                fr,
                                index,
                                vertices,
                                _joints,
                                object_transform,
                                dynamic_replay,
                                self._last_render_pose,
                                trace_aperture,
                            ),
                            zero_dt=pre_step_refreshes,
                        )
                    current_rgb = self._read_render_frame(
                        advance=False,
                        frame_index=index,
                        frame=fr,
                        joints=_joints,
                    )
                    if trace_session is not None:
                        trace_session.delivered_readback(
                            index,
                            fr,
                            self._online_trace_snapshot(
                                trace_module,
                                fr,
                                index,
                                vertices,
                                _joints,
                                object_transform,
                                dynamic_replay,
                                self._last_delivered_pose,
                                trace_aperture,
                            ),
                            readback=self._last_render_readback,
                            rgb=current_rgb,
                            zero_dt=self._last_readback_zero_dt,
                        )
                    self._store_dense_render_frame(index, fr, current_rgb)
                    rgb = current_rgb if index in selected else None
                if index in selected:
                    self._record_render_pick(out, fr, rgb)
        self._refresh_camera_counters()

    def _configure_clutter_render_replay(self) -> None:
        """Bind the shared kinematic clutter replay to the current episode stage."""
        import clutter_render_replay

        self._clutter_render_replay = clutter_render_replay.make_clutter_render_replay(
            self.ctx or {}
        )

    def _reset_clutter_render_replay(self) -> None:
        replay = getattr(self, "_clutter_render_replay", None)
        if replay is not None:
            replay.reset()

    def _apply_clutter_render_frame(self, frame_index: int | None) -> None:
        """Author clutter for a motion frame; tick-zero is the dense timeline's frame zero."""
        replay = getattr(self, "_clutter_render_replay", None)
        if replay is not None:
            replay.apply(0 if frame_index is None else frame_index)

    def _make_dynamic_replay(self, frames: list, start: int, joints):
        """Build the shared free-dynamic replay state for one chronological segment."""
        import dynamic_render_replay
        import object_physics
        import physics_human as ph

        paths = tuple(str(path).rstrip("/") for path in self.ctx["fdp_paths"])
        releasable = {
            str(getattr(frame, "obj_path", None) or "/World/Obj").rstrip("/")
            for frame in frames[start:]
            if bool(getattr(frame, "free_dynamic", False))
        }
        self._ensure_dynamic_render_proxy(joints, update=False)
        registry = {
            path: self.stage.GetPrimAtPath(path)
            for path in paths
            if path not in releasable or path in self._dynamic_owned_paths
        }
        return dynamic_render_replay.DynamicRenderReplay(
            self.stage,
            ph,
            object_physics,
            self._dynamic_render_proxy,
            paths,
            registry,
            self._dynamic_owned_paths,
        )

    def _ensure_dynamic_render_proxy(self, joints, *, update: bool) -> None:
        paths = tuple(str(path).rstrip("/") for path in self.ctx.get("fdp_paths") or ())
        if not paths:
            return
        import physics_human as ph

        if self._dynamic_render_proxy is None:
            self._dynamic_render_proxy = ph.build_proxy_rig(
                self.stage, self.ctx["rig"], joints
            )
            ph.enable_contact_reports(
                self.stage,
                [bone.path for bone in self._dynamic_render_proxy.bones] + list(paths),
            )
        if update:
            ph.update_proxy_rig(self._dynamic_render_proxy, joints)

    def _dynamic_camera_state_before(self, start: int):
        import egocentric_camera

        return (
            self._fk_cam_post[start - 1]
            if start > 0
            else egocentric_camera.INITIAL_CAMERA_STATE
        )

    def _step_dynamic_camera(
        self, fr, joints, object_transform, state, free_focus, physx_owned
    ):
        import egocentric_camera
        import episode_runtime as s3

        adapter = self._camera_adapter_for_frame(fr)
        camera_frame = adapter.camera_frame(
            fr,
            joints,
            object_transform,
            dt_seconds=1.0 / s3.FPS,
            free_focus=free_focus,
            physx_owned=physx_owned,
        )
        step = egocentric_camera.step_camera(
            camera_frame,
            state,
            framing=adapter.framing,
        )
        return step.output, step.state

    def _refresh_camera_counters(self) -> None:
        self._counters["camera_yaw_clamps"] = sum(
            int(output.diagnostics.physiological_yaw_clamped) for output in self._fk_cam
        )
        self._counters["camera_pitch_clamps"] = sum(
            int(output.diagnostics.physiological_pitch_clamped)
            for output in self._fk_cam
        )

    def _record_render_pick(self, out: list, fr, rgb) -> None:
        """Record pixels and matching camera/segment provenance for one delivered frame."""
        if rgb is None:
            return
        out.append(rgb)
        self._frame_segs.append(str(getattr(fr, "seg", "") or ""))
        if self._last_eye is not None:
            self._eye_poses.append(self._last_eye)

    def render_fpv_frames_b64(self) -> list:
        """Encode the exact validated model-facing frames as lossless base64 PNG strings."""
        try:
            import base64
            import io

            from PIL import Image

            rgbs = self.render_fpv_frames()
            out: list = []
            with self._timed("render.b64"):
                for rgb in rgbs:
                    buf = io.BytesIO()
                    # compress_level=1 (zlib fastest) instead of PIL's default 6: still LOSSLESS -- the
                    # decoded pixels are bit-identical, only the deflate search is shorter -- so the VLM
                    # sees exactly the same image with no compression artifacts. Trades a larger payload
                    # for a much cheaper encode; JPEG is deliberately NOT used (lossy).
                    Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
                    out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                    # ISAAC_FRAME_DUMP=<dir> writes the same frame to disk. This is the ONLY place the
                    # pixels the VLM receives are produced, so the dump is exactly what it sees rather
                    # than a re-render that could differ. Off unless the env var is set.
                    if _FRAME_DUMP:
                        _n = len(os.listdir(_FRAME_DUMP))
                        Image.fromarray(rgb).save(
                            os.path.join(_FRAME_DUMP, f"frame_{_n:04d}.png")
                        )
            return out
        except Exception as exc:  # noqa: BLE001 - never let a render error kill the rollout
            raise RuntimeError(f"render_fpv_frames_b64 failed: {exc}") from exc

    def _start_frame(self):
        """The shared neutral standing frame used by dense and model-facing renderers."""
        if not self.ctx:
            return None
        import episode_runtime

        return episode_runtime.make_start_frame(self.ctx)

    def observe(self) -> dict:
        """Observation: symbolic relations/held/score + (when ``render_obs``) egocentric FPV pixels --
        ``"frames"`` = the 1..n frames sampled across the last command (the assistant's temporal obs) and
        ``"frame"`` = the final one (the human's view + back-compat). The frames are the pixel obs AND make
        the trajectory multimodal (valid Qwen3-VL mrope; the text-only actor forward crashed apply_rotary).

        ``"vischeck"`` (present unless ``ISAAC_VISCHECK=0``) is the diagnostic record for the visibility
        calculation. The same uncapped set is exposed privately for the guarded Human interface; it never
        changes physics or score::

            {"n_visible": int, "n_entities": int,   # |visible set| out of the whole registry
             "visible": [str],  "truncated": bool,  # names, capped at _VIS_NAMES_MAX
             "n_containment_hidden": int,            # projected, but inside a closed receptacle
             "containment_hidden": [str],            # names, capped at _VIS_NAMES_MAX
             "n_frames": int,   "eye_src": "render"|"fk"|"none",
             "cmd": str,        "cmd_objs": [str],  # entity keys the last command named
             "cmd_visible": [bool],                 # parallel to cmd_objs
             "cmd_all_visible": bool | None}        # None = the command named no known entity

        On failure it degrades to ``{"error": "<Type>: <msg>"}`` rather than vanishing, so a silent zero can
        never be mistaken for "nothing was visible"."""
        assert self.scene_obj is not None
        with self._timed("observe.total"):
            if self.render_obs and getattr(
                self, "_suppress_failed_zero_motion_capture", False
            ):
                self._eye_poses = []
                self._frame_segs = []
                self._static_no_motion_capture = False
                frames_b64 = []
            else:
                frames_b64 = self.render_fpv_frames_b64() if self.render_obs else []
            with self._timed("observe.relations"):
                rel = self.scene_obj.relations(source="physics")
            obs = {
                "relations": rel,
                "held": set(self.scene_obj.held),
                "at": set(self._trace_current_at),
                "trace_events": [dict(row) for row in self._authored_trace],
                "score": self.score(),
                "goal": self._goal_checklist(),  # per-item {obj,target,kind,done}: the progress both agents steer by
                "put_verb": self.put_verb,  # "Put"|"PutInside": the container's working place verb (see __init__)
                "task": str(
                    self.scene
                ),  # scene key as the task id (humanized in the prompts)
            }
            if self.render_obs:
                obs["frames"] = (
                    frames_b64  # 1..n frames across the last command (the assistant's temporal obs)
                )
                obs["frame"] = (
                    frames_b64[-1] if frames_b64 else None
                )  # final frame -> the human's single-frame view + back-compat
                # Motion-segment tag per delivered frame. Emitted ONLY when it is exactly parallel to
                # the pixels -- a render that partly failed must not hand a consumer a tag list it
                # would have to guess the alignment of.
                if len(self._frame_segs) == len(frames_b64):
                    obs["frame_segs"] = list(self._frame_segs)
                if getattr(self, "_static_no_motion_capture", False):
                    obs["capture_mode"] = "static_no_motion"
            # AFTER the render: the visibility pass reads the camera poses _render_picks just retained.
            # It also has to run BEFORE the object list, because under ISAAC_VIS_GATE that list IS the
            # visible set -- the one case where the pass is not pure measurement.
            vis = self._vis_record()
            if vis is not None:
                obs["vischeck"] = vis
            # Harness-private, complete visibility set. ``vischeck.visible`` is intentionally capped
            # for logs and transport diagnostics, but the ASK Human guard must not mistake the 33rd
            # visible entity for an unseen one. Interaction views strip this private field before
            # constructing either model prompt.
            visible_keys_full = getattr(self, "_vis_full", None)
            obs["_visible_object_keys_full"] = (
                sorted(visible_keys_full) if visible_keys_full is not None else None
            )
            obs["objects"] = (
                self._actionable_objects()
            )  # [Verb] <name> keys for the 32B human
            # The human's prompt states what the list IS, and the two must never disagree: telling it
            # "this is what you can see" while handing it the whole house is what made the previous
            # round's spatial rules unenforceable.
            obs["objects_are_visible_only"] = bool(
                _VIS_GATE and self._vis_full is not None
            )
            return obs

    def _goal_checklist(self) -> list[dict]:
        """Per-predicate task checklist ``{obj, target, kind, done}`` off the CURRENT physical relations --
        the Isaac equivalent of VH's ``remaining_goal_hint``: it lets the assistant name the next concrete
        step and the frozen human see what is still TODO (so it progresses instead of repeating an action).
        """
        import isaac_goal

        if not self.goal_preds or self.scene_obj is None:
            return []
        with self._timed("observe.goal_checklist"):
            rel = self.scene_obj.relations(source="physics")
            return [
                {
                    "obj": p.a,
                    "target": p.b,
                    "kind": p.pred.value,
                    "done": bool(isaac_goal.satisfied(p, rel)),
                }
                for p in self.goal_preds
            ]

    # ---- visible-set MEASUREMENT (ISAAC_VISCHECK) ------------------------------------------------
    #
    # OBSERVATION-SIDE ONLY. Nothing below changes physics or score. The complete set is recorded for audit
    # and backs the guarded Human interface; ISAAC_VIS_GATE can additionally narrow ``_actionable_objects``.
    #
    # THE RULE. An entity is visible on a step if ANY of {AABB centre, 8 AABB corners} lands inside the FPV
    # frustum of ANY frame the assistant was actually SHOWN this step, within _VIS_MAX_DIST_M, with no WALL
    # between the eye and that sample, and with the entity subtending at least _VIS_MIN_APPARENT_PX.
    #   * centre + 8 corners, not centre alone: a box is inside a convex frustum iff all its corners are, so
    #     corners are the correct PARTIAL-visibility test for something straddling the frame edge -- and the
    #     centre is needed on top because a wall-length counter can have every corner out of frame while its
    #     middle fills the view. 9 points is the smallest set with both properties.
    #   * walls + declared closed-container containment: nav_grid.wall_sdf is the one GEOMETRIC occluder
    #     this codebase models correctly (true walls, undilated, FULL height -- so its XY footprint is valid
    #     at eye height). A movable declared INSIDE a bound, currently closed receptacle, or one covered by
    #     a live ON relation from an asset whose category is exactly ``lid``, is also hidden. All other
    #     furniture occlusion is deliberately out of scope, so the measured set remains an OVER-estimate.
    #   * the frames the assistant SAW, not every motion frame: ISAAC_FRAMES_PER_SEG of a command's frames
    #     are sampled, so 20 of a scan verb's 24 frames are never shown to anyone.

    @staticmethod
    def _vis_mode() -> str:
        """``ISAAC_VISCHECK`` -> ``"on"`` (default) | ``"off"`` | ``"verify"``. Default ON because the pass
        is observe-only and costs ~10 ms of an 11.9 s step; ``off`` removes it with no code change.
        ``verify`` re-checks every projected point against the scalar oracle (see ``_vis_project``).
        """
        val = (os.getenv("ISAAC_VISCHECK") or "1").strip().lower()
        if val in ("0", "off", "false", "no"):
            return "off"
        if val in ("verify", "shadow"):
            return "verify"
        return "on"

    def _gate_to_visible(self, names: list) -> list:
        """``ISAAC_VIS_GATE=1``: the person is told only what is IN VIEW, instead of the whole registry.

        This is the arm of the A/B. With the whole house listed, a lazy "look for the cans" is enough --
        the person picks a plausible id out of ~290 names and the assistant never has to say where
        anything is, which is the capability being trained. Narrowing the vocabulary removes the cheat at
        the root rather than policing it: an id that is not in the list cannot be typed, and the lenient
        command-repair narrows with it for free (it reconstructs commands out of this very list).

        Two names survive regardless of the frustum:
          * whatever is in the HAND -- a held object is routinely outside the sampled view (it is under
            the chin), and dropping it from the list makes ``[Put] <held> <somewhere>`` unsayable, so the
            person would be stuck carrying it for the rest of the episode;
          * nothing else. An empty view yields an empty list, and the honest move -- sweep, or say "I
            can't see anything like that" -- is the one the prompt already asks for.

        Falls back to the full registry when the pass produced nothing (``ISAAC_VISCHECK=0``, no camera
        pose, an exception): a starved vocabulary caused by a broken measurement would read as a hard
        task rather than a bug. Counted so the fallback cannot be silent."""
        if self._vis_full is None:
            self._vis_gate_fallbacks += 1
            # Into _counters, not just an attribute: a run that falls back on every step would otherwise
            # be indistinguishable from a correctly gated one, i.e. the A/B's treatment arm could
            # silently be a second copy of the control.
            self._counters["vis_gate_fallbacks"] = self._vis_gate_fallbacks
            return names
        keep = set(self._vis_full) | {
            str(h) for h in (getattr(self.scene_obj, "held", None) or ())
        }
        return [n for n in names if n in keep]

    def _vis_record(self) -> dict | None:
        """This step's visibility measurement, or None when the pass is off / there is no scene.

        Clears ``_vis_full`` first: under ISAAC_VIS_GATE the object list is built from it, and a set
        left over from the PREVIOUS step would silently let the person name things they have since
        turned away from -- a gate that leaks one step at a time is worse than no gate, because the
        numbers would look enforced."""
        self._vis_full = None
        if (
            not self.collect_camera
            or self._vis_mode() == "off"
            or self.scene_obj is None
        ):
            return None
        try:
            with self._timed("observe.vischeck"):
                rec = self._vis_compute()
        except Exception as exc:  # noqa: BLE001 - a measurement must never kill a rollout
            logger.warning("vischeck failed: %s: %s", type(exc).__name__, exc)
            return {"error": f"{type(exc).__name__}: {exc}"}
        logger.info(
            "[vischeck] cmd=%r visible=%d/%d containment_hidden=%d frames=%d(%s) "
            "cmd_objs=%s seen=%s",
            rec["cmd"],
            rec["n_visible"],
            rec["n_entities"],
            rec["n_containment_hidden"],
            rec["n_frames"],
            rec["eye_src"],
            rec["cmd_objs"],
            rec["cmd_visible"],
        )
        return rec

    def _vis_compute(self) -> dict:
        """Build the record. Shape is documented on ``observe``'s ``"vischeck"`` key."""
        poses, src = self._vis_eye_poses()
        controller = getattr(self, "_visual_state_events", None)
        hidden = controller.hidden_entity_keys() if controller is not None else set()
        names = sorted(
            str(key)
            for key in (self.scene_obj.entities or {})
            if str(key) not in hidden
        )
        projected = self._visible_names(poses, names) if (poses and names) else []
        containment_hidden = self._vis_containment_hidden(projected)
        vis = [name for name in projected if name not in containment_hidden]
        seen = set(vis)
        # The UNtruncated set, for the ISAAC_VIS_GATE object list. ``visible`` below is capped for the
        # wire and the logs; gating off a capped list would hide whatever sorts after the 32nd name.
        # None, NOT the empty set, when there was no camera pose to test against: an empty set gates the
        # person down to what is in their hand and reports the gate as working, which is the one failure
        # mode that looks like a hard task instead of a broken measurement.
        self._vis_full = seen if poses else None
        cmd_objs = self._vis_cmd_objects(self._last_cmd)
        return {
            "n_visible": len(vis),
            "n_entities": len(names),
            "visible": vis[:_VIS_NAMES_MAX],
            "truncated": len(vis) > _VIS_NAMES_MAX,
            "n_containment_hidden": len(containment_hidden),
            "containment_hidden": sorted(containment_hidden)[:_VIS_NAMES_MAX],
            "n_frames": len(poses),
            "eye_src": src,
            "cmd": self._last_cmd,
            "cmd_objs": cmd_objs,
            "cmd_visible": [n in seen for n in cmd_objs],
            # None (not True) when the command named no known entity -- "no objects to check" is a
            # different fact from "every named object was in view" and must not inflate the pass rate.
            "cmd_all_visible": all(n in seen for n in cmd_objs) if cmd_objs else None,
        }

    def _vis_containment_hidden(self, names: Sequence[str]) -> set[str]:
        """Projected entities whose declared container is currently closed.

        ``inside`` is the executor-maintained contents-to-container relation. A controller is the
        runtime proof that the destination has a physical lid/door. A detachable cover is recognized only
        from the conjunction of live ``on[lid] == container`` state and the lid entity's exact
        ``asset_category == "lid"`` metadata. Thus removing the lid immediately reveals the contents, and
        an ordinary object resting on a box cannot close it. Top-open bins with neither form of closure do
        not hide their contents. Walking the parent chain also handles a movable box inside a closed cabinet.
        This is intentionally narrower than general furniture occlusion.
        """
        inside = {
            str(subject): str(container)
            for subject, container in (
                getattr(self.scene_obj, "inside", None) or {}
            ).items()
            if subject is not None and container is not None
        }
        closed = {
            str(name)
            for name in (getattr(self.scene_obj, "_openable_controllers", None) or {})
        } - {str(name) for name in (getattr(self.scene_obj, "open_objs", None) or ())}
        closed.update(self._vis_detachable_lid_containers())
        if not inside or not closed:
            return set()

        hidden: set[str] = set()
        for name in names:
            ancestor = str(name)
            visited = {ancestor}
            while ancestor in inside:
                ancestor = inside[ancestor]
                if ancestor in visited:
                    break
                if ancestor in closed:
                    hidden.add(str(name))
                    break
                visited.add(ancestor)
        return hidden

    def _vis_detachable_lid_containers(self) -> set[str]:
        """Containers currently covered by a semantically typed loose lid."""
        entities = getattr(self.scene_obj, "entities", None) or {}
        covered: set[str] = set()
        for subject, target in (getattr(self.scene_obj, "on", None) or {}).items():
            entity = entities.get(str(subject))
            metadata = getattr(entity, "meta", None) or {}
            category = str(metadata.get("asset_category") or "").strip().lower()
            if category == "lid" and target is not None:
                covered.add(str(target))
        return covered

    def _vis_eye_poses(self) -> tuple[list, str]:
        """(eye, look, focal_mm) per shown frame, plus where the poses came from.

        ``render``: retained by ``_render_picks``, so the RL path pays nothing -- ``_render_one`` computed
        them anyway. ``fk`` reads the same dense camera cache without drawing pixels. ``none`` means no
        frame or rig was available. The measurement never advances camera state independently.
        """
        if self._eye_poses:
            return list(self._eye_poses), "render"
        if not self.collect_camera:
            return [], "none"
        import egocentric_camera
        import egocentric_camera_isaac
        import episode_runtime as s3
        import humanoid_motion as wpp
        import numpy as np

        rig = (self.ctx or {}).get("rig")
        indices = self._seg_pick_indices(self.n_seg_frames)
        if rig is None or not indices:
            return [], "none"
        frames = self.env.motion.frames if self.env is not None else []
        if frames:
            self._ensure_render_cache()
            return [
                (
                    np.asarray(self._fk_cam[index].eye, dtype=np.float64),
                    np.asarray(self._fk_cam[index].look, dtype=np.float64),
                    float(self._fk_cam[index].focal_mm),
                )
                for index in indices
                if index is not None
            ], "fk"
        frame = self._start_frame()
        if frame is None:
            return [], "none"
        with self._joints_only_fk():
            _vertices, joints, object_transform = wpp.realize(rig, frame)
        adapter = egocentric_camera_isaac.IsaacCameraAdapter.from_scene(
            self.scene_obj, self.scene_obj.solid_boxes()
        )
        camera_frame = adapter.camera_frame(
            frame, joints, object_transform, dt_seconds=1.0 / s3.FPS
        )
        output = egocentric_camera.step_camera(
            camera_frame,
            framing=adapter.framing,
        ).output
        return [
            (
                np.asarray(output.eye, dtype=np.float64),
                np.asarray(output.look, dtype=np.float64),
                float(output.focal_mm),
            )
        ], "fk"

    def _visible_names(self, poses: list, names: list) -> list[str]:
        """The subset of ``names`` visible on at least one of ``poses`` (see the section header)."""
        import numpy as np

        aabbs = self.scene_obj.entity_aabbs(names)
        names = [n for n in names if n in aabbs]
        if not names:
            return []
        pts, owner = self._vis_samples(aabbs, names)
        grid = getattr(self.scene_obj, "nav_grid", None) or getattr(
            self.scene_obj, "wall_sdf_grid", None
        )
        w = int((self.ctx or {}).get("W", 960))
        h = int((self.ctx or {}).get("Hh", 760))
        hit = np.zeros(len(names), dtype=bool)
        for pose in poses:
            eye, look = pose[:2]
            focal_mm = float(pose[2]) if len(pose) > 2 else None
            hit |= self._vis_frame_hits(
                eye, look, focal_mm, pts, owner, len(names), grid, w, h
            )
        return [n for n, v in zip(names, hit.tolist()) if v]

    @staticmethod
    def _vis_samples(aabbs: dict, names: list):
        """(9E,3) world sample points -- each entity's AABB centre then its 8 corners -- plus the (9E,) entity
        index each point belongs to. Flat rather than nested so one vectorized projection covers the whole
        scene; the fixed stride of 9 lets the per-entity reduce reshape instead of loop.
        """
        import numpy as np

        mn = np.array([aabbs[n][0] for n in names], dtype=np.float64)  # (E,3)
        mx = np.array([aabbs[n][1] for n in names], dtype=np.float64)
        # bit i of k picks mx over mn on axis i -> the 8 corners, in one broadcast
        sel = np.array(
            [[bool((k >> i) & 1) for i in range(3)] for k in range(8)], dtype=bool
        )
        corners = np.where(sel[None, :, :], mx[:, None, :], mn[:, None, :])  # (E,8,3)
        ctr = (0.5 * (mn + mx))[:, None, :]  # (E,1,3)
        pts = np.concatenate([ctr, corners], axis=1)  # (E,9,3)
        owner = np.repeat(np.arange(len(names), dtype=np.int64), 9)
        return pts.reshape(-1, 3), owner

    def _vis_frame_hits(
        self, eye, look, focal_mm, pts, owner, n_ent: int, grid, w: int, h: int
    ):
        """(n_ent,) bool: which entities are visible in the ONE frame posed at (eye, look)."""
        import numpy as np

        front, inside, ndc, dist = self._vis_project(eye, look, focal_mm, pts, w, h)
        hit = np.zeros(n_ent, dtype=bool)
        cand = np.nonzero(inside & (dist <= _VIS_MAX_DIST_M))[0]
        if cand.size:
            clear = self._vis_wall_clear(eye, pts[cand], grid)
            hit[owner[cand[clear]]] = True
        return hit & self._vis_apparent_ok(front, ndc, n_ent, w, h)

    def _vis_project(self, eye, look, focal_mm, pts, width: int, height: int):
        """Project every point at once -> (front, inside, ndc, dist), each indexed like ``pts``.

        A VECTORIZED TRANSCRIPTION of ``smplh_camera.point_in_fpv_frustum``, term for term, sharing that
        module's ``FOCAL_MM``/``H_APERTURE`` so the intrinsics stay single-sourced. It is transcribed rather
        than called because the scalar helper is one Python call per point and this pass projects ~2600
        points per frame. ``ISAAC_VISCHECK=verify`` runs the scalar helper on EVERY point and requires the
        two to agree, which is what makes the transcription a claim and not a hope.

        ``tgt`` for the oracle is ``eye + look``: it normalizes ``tgt - eye``, so any positive scale of the
        look direction gives the identical basis -- and ``drive_fpv_gaze`` aims at ``eye + look*0.6``.
        """
        import numpy as np
        from smplh_camera import FOCAL_MM, H_APERTURE

        f = np.asarray(look, dtype=np.float64)
        f = f / (np.linalg.norm(f) + 1e-9)
        up_world = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(f, up_world))) > 0.999:
            up_world = np.array([0.0, 1.0, 0.0])
        right = np.cross(f, up_world)
        right = right / (np.linalg.norm(right) + 1e-9)
        true_up = np.cross(right, f)
        p = np.asarray(pts, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
        z_cam = p @ (-f)  # camera -Z is forward -> a point in front has z_cam < 0
        front = z_cam < 0.0
        denom = np.where(front, -z_cam, 1.0)
        aspect = float(height) / float(width) if width else 760.0 / 960.0
        focal = FOCAL_MM if focal_mm is None else float(focal_mm)
        tan_h = (H_APERTURE * 0.5) / focal
        tan_v = (H_APERTURE * aspect * 0.5) / focal
        ndc = np.stack([(p @ right) / denom / tan_h, (p @ true_up) / denom / tan_v], 1)
        inside = front & (np.abs(ndc[:, 0]) <= 1.0) & (np.abs(ndc[:, 1]) <= 1.0)
        if self._vis_mode() == "verify":
            self._vis_verify(eye, look, focal, pts, width, height, inside, ndc)
        return front, inside, ndc, np.linalg.norm(p, axis=1)

    @staticmethod
    def _vis_verify(eye, look, focal_mm, pts, width, height, inside, ndc) -> None:
        """Re-run the scalar ``point_in_fpv_frustum`` on every point and require it to agree with the
        vectorized projection (``ISAAC_VISCHECK=verify``). A complete proof rather than a sample.
        """
        import numpy as np
        from smplh_camera import point_in_fpv_frustum

        tgt = np.asarray(eye, dtype=np.float64) + np.asarray(look, dtype=np.float64)
        for i, pt in enumerate(np.asarray(pts, dtype=np.float64)):
            ok, nx, ny = point_in_fpv_frustum(
                eye, tgt, pt, width, height, focal_mm=focal_mm
            )
            same_ndc = ok and abs(nx - ndc[i, 0]) < 1e-9 and abs(ny - ndc[i, 1]) < 1e-9
            if ok != bool(inside[i]) or (ok and not same_ndc):
                raise AssertionError(
                    f"vischeck projection MISMATCH at point {i} {tuple(pt)}: oracle "
                    f"(inside={ok}, ndc=({nx}, {ny})) vs vectorized "
                    f"(inside={bool(inside[i])}, ndc={tuple(ndc[i])})"
                )

    @staticmethod
    def _vis_apparent_ok(front, ndc, n_ent: int, width: int, height: int):
        """(n_ent,) bool: does the entity subtend at least ``_VIS_MIN_APPARENT_PX``?

        Measured from the NDC bounding box of its in-front samples (|ndc|<=1 spans the full frame, so one
        NDC unit is half the frame), which correctly keeps an entity that is partly off-screen but large.
        An entity STRADDLING the near plane -- some samples in front, some behind -- is treated as large
        without measuring: its projection diverges, and something the camera is inside of is never a speck.

        An entity whose AABB is DEGENERATE (the spec-pose fallback ``entity_aabbs`` returns for a prim that
        is not built) measures 0 px and is reported not-visible. In a live scene every registered entity
        has a real bound, so this only affects an offline/SPEC registry."""
        import numpy as np

        fr = np.asarray(front).reshape(n_ent, -1)
        xs, ys = ndc[:, 0].reshape(n_ent, -1), ndc[:, 1].reshape(n_ent, -1)
        # empty row (nothing in front) -> -inf - inf = -inf, i.e. "too small", never NaN
        x_span = np.where(fr, xs, -np.inf).max(1) - np.where(fr, xs, np.inf).min(1)
        y_span = np.where(fr, ys, -np.inf).max(1) - np.where(fr, ys, np.inf).min(1)
        px = np.maximum(0.5 * width * x_span, 0.5 * height * y_span)
        any_front = fr.any(axis=1)
        straddles = any_front & ~fr.all(axis=1)
        return straddles | (any_front & (px >= _VIS_MIN_APPARENT_PX))

    def _vis_wall_clear(self, eye, pts, grid):
        """(K,) bool: does the eye->point segment stay clear of every WALL?

        Marches each ray in XY through ``nav_grid.wall_sdf`` (>0 = outside the wall footprint). Walls span
        full height, so their 2D footprint is the correct occluder at eye height -- which is exactly why
        ``NavGrid.wall_sdf_at`` exists and why the combined SDF is NOT used (it would fire over a low
        counter). A scene with no wall prims has ``wall_sdf is None`` -> nothing occludes, matching
        ``wall_sdf_at``'s 10.0 branch. See ``_VIS_WALL_MARGIN_M`` for why both ray ends are skipped.
        """
        import numpy as np

        sdf = getattr(grid, "wall_sdf", None) if grid is not None else None
        if sdf is None:
            return np.ones(len(pts), dtype=bool)
        e_xy = np.asarray(eye, dtype=np.float64)[:2]
        d = np.asarray(pts, dtype=np.float64)[:, :2] - e_xy  # (K,2)
        span = np.linalg.norm(d, axis=1)
        unit = d / np.maximum(span, 1e-9)[:, None]
        hi = np.maximum(span - _VIS_WALL_MARGIN_M, _VIS_WALL_MARGIN_M)
        # One fixed sample count for every ray (a rectangular array is what makes the march vectorizable).
        # Sized so the LONGEST possible ray still steps no further than one grid cell -- callers only pass
        # points that passed the _VIS_MAX_DIST_M gate, and the XY span is at most that 3D distance -- which
        # is the correctness bound: a coarser step could stride over a one-cell-thick wall. Finer buys
        # nothing, the field is bilinear between cell centres.
        cell = max(float(getattr(grid, "cell", 0.10)), 1e-3)
        n = min(_VIS_WALL_MAX_SAMPLES, max(2, int(_VIS_MAX_DIST_M / cell) + 1))
        t = np.linspace(0.0, 1.0, n)[None, :]  # (1,n)
        r = _VIS_WALL_MARGIN_M + (hi - _VIS_WALL_MARGIN_M)[:, None] * t  # (K,n)
        val = self._vis_bilinear(
            sdf, grid, e_xy[0] + unit[:, 0:1] * r, e_xy[1] + unit[:, 1:2] * r
        )
        return (val > 0.0).all(axis=1)

    @staticmethod
    def _vis_bilinear(arr, grid, xs, ys):
        """Vectorized transcription of ``nav_grid.NavGrid._bilinear`` (cell-centre aligned, edge-clamped),
        for the same reason ``_vis_project`` transcribes the frustum test: the scalar ``wall_sdf_at`` is one
        Python call per point and an occlusion march samples tens of thousands of them per step.
        """
        import numpy as np

        fx = (xs - grid.x0) / grid.cell - 0.5
        fy = (ys - grid.y0) / grid.cell - 0.5
        i0, j0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        tx, ty = fx - i0, fy - j0

        def tap(di: int, dj: int):
            return arr[
                np.clip(j0 + dj, 0, grid.ny - 1), np.clip(i0 + di, 0, grid.nx - 1)
            ]

        return (
            tap(0, 0) * (1 - tx) * (1 - ty)
            + tap(1, 0) * tx * (1 - ty)
            + tap(0, 1) * (1 - tx) * ty
            + tap(1, 1) * tx * ty
        )

    def _vis_cmd_objects(self, cmd: str) -> list[str]:
        """The registry keys the command just executed NAMED, in order, deduped.

        Resolved through the grammar's OWN resolver, so ``<can_of_soda> (115)`` is checked under the key it
        actually binds to (``can_of_soda115``) rather than under the literal text. A reference that resolves
        to nothing is dropped: that is a parse failure, not a visibility fact."""
        import command_grammar

        out: list[str] = []
        for line in command_grammar.split_program(cmd or ""):
            parsed = command_grammar.parse_line(line)
            if not parsed:
                continue
            for name, inst in parsed[1]:
                try:
                    key = command_grammar._resolve(self.scene_obj, name, inst)
                except command_grammar.CommandParseError:
                    continue
                if key not in out:
                    out.append(key)
        return out

    def _scene_put_verb(self) -> str:
        """The PLACE verb the scene's ground-truth program uses (first place line wins) so the human is told
        the verb that actually works for THIS container: "Put" for an open bin/surface (-> PLACE), or
        "PutInside" for an interior-space container like a fridge/cabinet (-> PLACE_INSIDE). General (no
        per-scene hardcode); falls back to "PutInside" for an inside-goal scene whose program has no place
        line (e.g. an ATTACH task, where it is unused anyway)."""
        import command_grammar

        prog = (self.ctx or {}).get("program", "") or ""
        for line in command_grammar.split_program(prog):
            parsed = command_grammar.parse_line(line)
            if not parsed:
                continue
            verb = parsed[0]
            if verb in ("PUTIN", "PUTINSIDE", "INSERT"):
                return "PutInside"
            if verb in ("PUT", "PUTBACK", "PUTON", "PUTOBJBACK"):
                return "Put"
        return "PutInside"

    def _actionable_objects(self) -> list:
        """EVERY entity registered in the scene -- furniture and fixtures included, not just this
        episode's task objects.

        ``scene_state.entities`` is the authoritative registry: whatever is in it is exactly what
        ``resolve()`` accepts, so every name handed over here is commandable by construction. All five
        roles (graspable / surface / container / openable / seat) are addressable by some verb.

        Why the whole scene and not the goal set: the assistant talks like a person -- "check under the
        fridge", "the bin by the door" -- and the human can only answer with names it was given. With
        only task objects listed, run q3 had the human reason "I will walk to the fridge to get a closer
        look" and then emit NOTHING, because `fridge` was not a word it was allowed to say. It also made
        the deliberate-mistake modes inexpressible: there was nothing off-task to reach for.
        """
        ents = getattr(getattr(self, "scene_obj", None), "entities", None) or {}
        names: list = sorted(str(k) for k in ents)
        if names:
            return self._gate_to_visible(names) if _VIS_GATE else names
        # Fallback for a scene that never populated the registry: the task's own objects, as before.
        bt = getattr(self.scene_obj, "behavior_task", None) or {}
        out: list = list(bt.get("grasp", []) or [])
        if bt.get("container"):
            out.append(bt["container"])
        for _og in bt.get("goals", []) or []:
            if isinstance(_og, (list, tuple)) and len(_og) == 2:
                out.append(_og[1])
        for _cp in bt.get("carcass", []) or []:
            if isinstance(_cp, (list, tuple)) and len(_cp) == 2:
                out += [_cp[0], _cp[1]]
        for _k, _cs in (bt.get("existential_witness", {}) or {}).items():
            out.append(str(_k))
            out += [str(c) for c in (_cs or [])]
        seen: set = set()
        return [n for n in out if not (n in seen or seen.add(n))]

    def score(self) -> dict:
        """Shared achievement/prohibition score over the current PHYSICAL relations."""
        import task_score

        assert self.scene_obj is not None
        preds = self.goal_preds
        bt = getattr(self.scene_obj, "behavior_task", None) or {}
        neg_constraints = list(bt.get("neg_constraints") or [])
        trace_outcomes = list(bt.get("trace_outcomes") or [])
        aggregate_goals = list(bt.get("aggregate_goals") or [])
        if (
            not preds
            and not neg_constraints
            and not trace_outcomes
            and not aggregate_goals
        ):
            # 9.3% of run s2b's rollouts scored here, yet the _derive_goal instrumentation never fired
            # once -- so goal_preds was never DERIVED empty, it was never derived at all. That is the
            # constructor's `self.goal_preds = []` surviving to scoring time, i.e. the reward is being
            # read from an env that was rebuilt (sim-server respawn?) and never re-opened. Print what
            # distinguishes those two worlds so the next run says which.
            logger.error(
                "[goal] score(): goal_preds is EMPTY -> n_total=0, so this rollout scores 0.00 whatever "
                "the policy did. scene=%r has_task=%s opened=%s. _derive_goal never logged an empty "
                "derive, so this env most likely never ran one.",
                getattr(self, "scene", None),
                bool(getattr(self.scene_obj, "behavior_task", None)),
                getattr(self, "_opened", "unknown"),
            )
            return task_score.score_task([], [], set()).to_dict()
        with self._timed("score.total"):
            rel = self.scene_obj.relations(source="physics")
            return task_score.score_task(
                preds,
                neg_constraints,
                rel,
                trace_outcomes,
                {
                    "events": self._authored_trace,
                    "current_at": sorted(self._trace_current_at),
                },
                aggregate_goals=aggregate_goals,
                baseline_goal_indices=self._baseline_goal_indices,
            ).to_dict()

    def capture_score_baseline(self) -> dict:
        """Exclude goals already true before the evaluated policy can earn them."""
        import task_score

        if self.scene_obj is None:
            raise RuntimeError("IsaacEnv.capture_score_baseline() called before open()")
        relations = self.scene_obj.relations(source="physics")
        self._baseline_goal_indices = set(
            task_score.satisfied_goal_indices(self.goal_preds, relations)
        )
        return self.score()

    # ---- internals ------------------------------------------------------------------------------
    def _derive_goal(self) -> list:
        """Task goal predicates from scene.behavior_task (same derivation as episode_runtime's goal-check:
        per-edge ATTACHED for assembly, per-item INSIDE for multi-container, else each grasped item INSIDE
        the single container)."""
        import isaac_goal

        bt = getattr(self.scene_obj, "behavior_task", None)
        if not bt:
            # Measured in run s2a: 52/52 episode OPENS derived 2-8 predicates, yet 8% of rollouts
            # scored with n_total=0, so the task is being lost AFTER open -- and the only re-derive
            # is reset()'s. An empty derive is therefore never normal; say which key went missing so
            # the next run names the branch instead of leaving it to log archaeology.
            logger.error(
                "[goal] _derive_goal: scene_obj has NO behavior_task -> 0 predicates. Every rollout "
                "of this episode scores 0.00 no matter what the policy does."
            )
            return []
        if bt.get("goal_preds"):
            predicates = isaac_goal.from_rows(bt["goal_preds"])
            self.scene_obj.configure_goal_predicates(bt["goal_preds"])
            return predicates
        if bt.get("carcass"):
            return [isaac_goal.attached(c, p) for (c, p) in bt["carcass"]]
        if bt.get("goals"):
            return [isaac_goal.inside(o, c) for (o, c) in bt["goals"]]
        if bt.get("aggregate_goals"):
            return []
        container = bt.get("container")
        if container:
            return [isaac_goal.inside(k, container) for k in bt.get("grasp", [])]
        logger.error(
            "[goal] _derive_goal: behavior_task has none of carcass/goals/container -> 0 predicates "
            "(keys present: %s). Every rollout of this episode scores 0.00 regardless of the policy.",
            sorted(bt),
        )
        return []

    def _apply_goal_override(self, goals: list) -> None:
        """Overwrite the scene's task goal with a SYNTHESIZED INSIDE goal (list of (obj,container)) by
        rewriting scene_obj.behavior_task["goals"] BEFORE _derive_goal reads it (the "goals" branch already
        exists, so no goal-compilation change is needed). The referenced objects/containers must already
        exist + be reachable in the built scene -- the taskgen synthesizer only re-pairs the scene's OWN
        (already stand-validated) graspables + containers, so feasibility is inherited. Objects dropped from
        the goal stay physically in the scene, they simply stop being scored/commanded.
        """
        if self.scene_obj is None:
            raise RuntimeError("goal override requires a built scene")
        pairs: list[tuple[str, str]] = []
        for index, row in enumerate(goals):
            if (
                isinstance(row, (str, bytes))
                or not isinstance(row, Sequence)
                or len(row) != 2
            ):
                raise ValueError(
                    f"goal override row {index} must be a (subject, target) pair"
                )
            subject, target = row
            if not isinstance(subject, str) or not subject.strip():
                raise ValueError(
                    f"goal override row {index} subject must be a non-empty string"
                )
            if not isinstance(target, str) or not target.strip():
                raise ValueError(
                    f"goal override row {index} target must be a non-empty string"
                )
            pairs.append((subject.strip(), target.strip()))
        if not pairs:
            raise ValueError("goal override must contain at least one pair")

        entities = getattr(self.scene_obj, "entities", None)
        if not isinstance(entities, Mapping):
            raise ValueError("built scene has no entity role registry")

        def role_name(key: str) -> str:
            entity = entities.get(key)
            role = getattr(entity, "role", None)
            return str(getattr(role, "name", "")).casefold()

        bad_subjects = sorted(
            {subject for subject, _target in pairs if role_name(subject) != "graspable"}
        )
        if bad_subjects:
            raise ValueError(
                "goal override subject(s) are not registered graspables: "
                + ", ".join(bad_subjects)
            )
        bad_targets = sorted(
            {target for _subject, target in pairs if role_name(target) != "container"}
        )
        if bad_targets:
            raise ValueError(
                "goal override target(s) are not registered containers: "
                + ", ".join(bad_targets)
            )

        vocabulary = self.vocab()
        graspables = set(vocabulary["graspables"])
        containers = set(vocabulary["containers"])
        undeclared_subjects = sorted(
            {subject for subject, _target in pairs if subject not in graspables}
        )
        if undeclared_subjects:
            raise ValueError(
                "goal override subject(s) are outside the scene's validated vocabulary: "
                + ", ".join(undeclared_subjects)
            )
        undeclared_targets = sorted(
            {target for _subject, target in pairs if target not in containers}
        )
        if undeclared_targets:
            raise ValueError(
                "goal override target(s) are outside the scene's validated vocabulary: "
                + ", ".join(undeclared_targets)
            )
        witness = vocabulary["existential_witness"]
        invalid_pairs = sorted(
            (subject, target)
            for subject, target in pairs
            if subject in witness and target not in set(witness[subject])
        )
        if invalid_pairs:
            rendered = ", ".join(
                f"{subject}->{target}" for subject, target in invalid_pairs
            )
            raise ValueError(
                "goal override pair(s) are outside the scene's validated reachability "
                f"witness: {rendered}"
            )

        bt = dict(getattr(self.scene_obj, "behavior_task", None) or {})
        bt["goals"] = pairs
        bt.pop(
            "container", None
        )  # goals is authoritative; drop the stale single-container default
        self.scene_obj.behavior_task = bt

    def vocab(self) -> dict:
        """Per-scene SYNTHESIS MENU for the online-taskgen generator: the graspable KEYS + valid container
        KEYS (+ per-object candidate containers when the builder computed them). Sourced from the VALIDATED
        behavior_task (grasp / container / existential_witness) -- exactly the keys the executor accepts AND
        that the builder placed + stand-validated, so a goal re-pairing them is feasible by construction.
        Called by the offline scene-vocabulary bake; not on the episode hot path.
        """
        bt = dict(getattr(self.scene_obj, "behavior_task", None) or {})
        graspables = sorted({str(g) for g in (bt.get("grasp", []) or []) if g})
        witness = {
            str(k): [str(c) for c in (v or [])]
            for k, v in (bt.get("existential_witness", {}) or {}).items()
        }
        containers: set[str] = set()
        if bt.get("container"):
            containers.add(str(bt["container"]))
        for cs in witness.values():
            containers.update(cs)
        for _o, c in bt.get("goals", []) or []:
            if c:
                containers.add(str(c))
        return {
            "graspables": graspables,
            "containers": sorted(containers),
            "existential_witness": witness,
        }

    def _make_args(self) -> argparse.Namespace:
        """The argparse-shaped config setup_episode reads. Only ``scene`` is required (the rest are
        getattr-guarded in setup_episode); we set the full set explicitly so no default drifts.
        """
        return types.SimpleNamespace(
            scene=self.scene,
            seed=self.seed,
            tag=self.tag,
            active_gpu=self.active_gpu,
            subdir=None,
            frames=0,
            check=False,
            preview=False,
            physcheck=False,
            initcheck=False,
            navcheck=False,
            fallback_check=False,
            no_doors=False,
            close_doors_behind=False,
            natural_grasp=False,
            strict_acceptance=self.strict_acceptance,
            allow_dirty=True,
            resample=4,
            num_shards=1,
            shard_idx=0,
            shard_no_tpv=False,
            shard_fast=False,
            door_open=False,
            render_obs=self.render_obs,
        )


def _smoke() -> int:
    """M2 smoke: open a scene, drive its program ONE command at a time through step() (the online loop),
    print per-command obs + the terminal reward, then reset (proving reset-many) and close.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bt_composting_waste")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--render-obs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="render the per-command FPV frames (validates Plan B multi-frame); off = symbolic-only (cheap).",
    )
    args = ap.parse_args()

    import command_grammar

    env = IsaacEnv(args.scene, seed=args.seed, render_obs=args.render_obs)
    obs = env.open()
    print(
        f"[isaac_env] opened {args.scene}: start score={obs['score']} put_verb={obs.get('put_verb')} "
        f"goal={obs.get('goal')} n_frames={len(obs.get('frames') or [])}"
    )

    cmds = command_grammar.split_program(env.ctx["program"])
    print(f"[isaac_env] driving {len(cmds)} commands online via step()")
    for i, c in enumerate(cmds):
        o = env.step(c)
        print(
            f"[isaac_env] step {i:2d}: {c!r:46s} held={sorted(o['held'])} "
            f"n_done={o['score']['n_done']}/{o['score']['n_total']} n_frames={len(o.get('frames') or [])}"
        )

    rw = env.reward()
    print(
        f"[isaac_env] TERMINAL: success={rw['success']} "
        f"n_done={rw['n_done']}/{rw['n_total']} progress={rw['progress']:.2f} dirty={rw['dirty']}"
    )

    env.reset()
    print(f"[isaac_env] after reset (reset-many): score={env.score()}")
    print("[isaac_env] M2 smoke OK")
    env.close(
        hard=True
    )  # os._exit(0) -- never returns (Isaac teardown segfaults otherwise)
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
