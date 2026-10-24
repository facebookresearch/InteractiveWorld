"""Shared per-segment capture loop used by all three demos.

A "segment" is a contiguous run of N captured frames with a per-frame callback that sets the
scripted scene state (root pose, object pose, cameras, proxy forearm) for a normalized time
``t`` in [0,1]. ``run_segment`` advances the SINGLE render clock once per frame (capturing
BOTH the FPV and TPV products) and returns the updated global frame index.

Keeping this here avoids triplicating the loop across demo1/2/3 while leaving each demo's
choreography (the segment list + per-frame callbacks) in the demo file, as the spec lays out.
"""

from __future__ import annotations

import bisect
import logging
import os
import shutil
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from capture_sampling import (
    ACTION_QUARTILE_FRAME_COUNT,
    action_quartile_indices,
    ACTION_QUARTILE_POSITIONS,
)
from fpv_render import (
    advance_once,
    converge_at_current_time,
    DEFAULT_READBACK_RETRIES,
    read_and_save,
    validate_frame,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fpv_core import RenderCfg

LOG = logging.getLogger("fpv_segments")
CAPTURE_PLAN_VERSION = "holoassist-sparse-capture/1.0"
EGOLONGQA_CAPTURE_PLAN_VERSION = "egolongqa-sparse-capture/1.0"
EGOLONGQA_CAPTURE_PLAN_V2_VERSION = "egolongqa-sparse-capture/2.0"
PHASE_CAPTURE_PLAN_VERSION = "wearable-proactive-phase-sparse-capture/1.0"
PHASE_CAPTURE_FRAMES = 4


def uniform_endpoint_indices(
    start_frame: int, end_frame: int, num_frames: int
) -> list[int]:
    """Mirror HoloAssist's endpoint-inclusive uniform video sampling."""
    if start_frame < 0 or end_frame < start_frame or num_frames < 1:
        raise ValueError("invalid endpoint-inclusive sampling range")
    count = min(num_frames, end_frame - start_frame + 1)
    if count == 1:
        return [start_frame]
    step = (end_frame - start_frame) / (count - 1)
    return sorted(
        {
            min(int(round(start_frame + index * step)), end_frame)
            for index in range(count)
        }
    )


def official_uniform_endpoint_indices(
    start_frame: int, end_frame: int, num_frames: int
) -> list[int]:
    """Mirror EgoLongQA's official floor-based whole-video sampler."""
    if start_frame < 0 or end_frame < start_frame or num_frames < 1:
        raise ValueError("invalid endpoint-inclusive sampling range")
    count = min(num_frames, end_frame - start_frame + 1)
    if count == 1:
        return [start_frame]
    step = (end_frame - start_frame) / (count - 1)
    return [int(start_frame + index * step) for index in range(count)]


def resolve_capture_plan(
    plan: Mapping[str, object],
    command_spans: Sequence[tuple[int, int]],
    *,
    total_frames: int,
) -> dict[str, object]:
    """Resolve command windows to the exact frame indices the evaluator will read."""
    version = plan.get("contract_version")
    if version not in {
        CAPTURE_PLAN_VERSION,
        EGOLONGQA_CAPTURE_PLAN_VERSION,
        EGOLONGQA_CAPTURE_PLAN_V2_VERSION,
        PHASE_CAPTURE_PLAN_VERSION,
    }:
        raise ValueError("unsupported sparse capture-plan contract")
    if total_frames < 1 or not command_spans:
        raise ValueError("capture plan requires a non-empty rendered trajectory")
    sampler = plan.get("sampler")
    windows = plan.get("windows")
    _validate_sampler(version, sampler, plan)
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise ValueError("capture plan windows must be an array")
    _validate_phase_bindings(version, plan.get("observation_queries"), windows)
    num_frames = int(sampler.get("num_frames", 0))
    if num_frames < 1:
        raise ValueError("capture plan num_frames must be positive")
    resolved_windows = [
        _resolve_capture_window(
            version,
            row,
            command_spans,
            total_frames,
            num_frames,
            str(sampler.get("kind") or ""),
        )
        for row in windows
    ]
    selected = {
        int(index) for row in resolved_windows for index in row["frame_indices"]
    }
    if not selected:
        raise ValueError("capture plan selected no frames")
    capture_indices = sorted(selected)
    return {
        **dict(plan),
        "windows": resolved_windows,
        "capture_indices": capture_indices,
        "selected_frame_count": len(capture_indices),
        "timeline_frame_count": total_frames,
    }


def _resolve_capture_window(
    version: object,
    raw_window: object,
    command_spans: Sequence[tuple[int, int]],
    total_frames: int,
    num_frames: int,
    sampler_kind: str,
) -> dict[str, object]:
    if not isinstance(raw_window, Mapping):
        raise ValueError("capture plan window must be an object")
    command_start = int(raw_window.get("command_start", -1))
    command_end = int(raw_window.get("command_end_exclusive", -1))
    if not 0 <= command_start < command_end <= len(command_spans):
        raise ValueError("capture plan window has invalid command bounds")
    frame_start = max(0, int(command_spans[command_start][0]))
    source_end = int(command_spans[command_end - 1][1])
    phase_v2 = version == PHASE_CAPTURE_PLAN_VERSION
    exclusive_end = version in {
        EGOLONGQA_CAPTURE_PLAN_VERSION,
        EGOLONGQA_CAPTURE_PLAN_V2_VERSION,
        PHASE_CAPTURE_PLAN_VERSION,
    }
    frame_end = min(total_frames - 1, source_end - 1 if exclusive_end else source_end)
    indices = _window_indices(
        version,
        raw_window,
        frame_start,
        frame_end,
        num_frames,
        sampler_kind,
    )
    result = {
        **dict(raw_window),
        "frame_start": indices[0] if phase_v2 else frame_start,
        "frame_end_inclusive": indices[-1] if phase_v2 else frame_end,
        "frame_indices": indices,
    }
    if phase_v2:
        result.update(
            {
                "source_frame_start": frame_start,
                "source_frame_end_exclusive": frame_end + 1,
            }
        )
    return result


def _validate_sampler(
    version: object, sampler: object, plan: Mapping[str, object]
) -> None:
    if not isinstance(sampler, Mapping):
        raise ValueError("capture plan has an unsupported sampler")
    kind = sampler.get("kind")
    count = sampler.get("num_frames")
    if version == CAPTURE_PLAN_VERSION and kind == "uniform_endpoint_inclusive":
        return
    if (
        version == EGOLONGQA_CAPTURE_PLAN_VERSION
        and kind == "uniform_endpoint_inclusive"
        and plan.get("frame_end_semantics") == "exclusive"
    ):
        return
    if (
        version == EGOLONGQA_CAPTURE_PLAN_V2_VERSION
        and kind == "uniform_time_endpoint_inclusive"
        and sampler.get("index_quantization") == "floor"
        and sampler.get("sampling_domain") == "rendered_frame_time"
        and plan.get("frame_end_semantics") == "exclusive"
    ):
        return
    if (
        version == EGOLONGQA_CAPTURE_PLAN_V2_VERSION
        and kind == "action_progress_quartiles"
        and count == ACTION_QUARTILE_FRAME_COUNT
        and tuple(sampler.get("positions") or ()) == ACTION_QUARTILE_POSITIONS
        and sampler.get("index_quantization") == "preceding_frame"
        and sampler.get("sampling_domain") == "rendered_command_time"
        and plan.get("frame_end_semantics") == "exclusive"
    ):
        return
    if (
        version == PHASE_CAPTURE_PLAN_VERSION
        and kind == "phase_contiguous"
        and count == PHASE_CAPTURE_FRAMES
        and plan.get("frame_end_semantics") == "exclusive"
    ):
        _validate_observation_queries(plan.get("observation_queries"))
        return
    raise ValueError("capture plan has an unsupported sampler")


def _validate_observation_queries(value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("phase capture plan requires observation queries")
    forbidden = {"expected", "action", "instruction", "reason_code", "expected_action"}
    for query in value:
        if not isinstance(query, Mapping) or forbidden & _nested_keys(query):
            raise ValueError("observation query is invalid or contains supervision")
        if not _is_sha256(query.get("query_digest")) or not query.get("decision_id"):
            raise ValueError("observation query identity is missing")


def _is_sha256(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
    ):
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def _validate_phase_bindings(version: object, queries: object, windows: object) -> None:
    if version != PHASE_CAPTURE_PLAN_VERSION:
        return
    if not isinstance(queries, Sequence) or not isinstance(windows, Sequence):
        raise ValueError("phase capture bindings must be arrays")
    query_rows = [row for row in queries if isinstance(row, Mapping)]
    window_rows = [row for row in windows if isinstance(row, Mapping)]
    if len(query_rows) != len(queries) or len(window_rows) != len(windows):
        raise ValueError("phase capture bindings must contain objects")
    window_by_id = {str(row.get("decision_id") or ""): row for row in window_rows}
    query_digests = [str(row.get("query_digest") or "") for row in query_rows]
    if len(query_digests) != len(set(query_digests)) or len(window_by_id) != len(
        windows
    ):
        raise ValueError("phase capture decision identities must be unique")
    queries_by_id: dict[str, list[Mapping[str, object]]] = {}
    for query in query_rows:
        queries_by_id.setdefault(str(query["decision_id"]), []).append(query)
    if not queries_by_id or queries_by_id.keys() != window_by_id.keys():
        raise ValueError("phase capture queries and windows must bind one-to-one")
    for decision_id, decision_queries in queries_by_id.items():
        window = window_by_id[decision_id]
        expected_digests = sorted(str(row["query_digest"]) for row in decision_queries)
        if sorted(window.get("observation_query_digests", ())) != expected_digests:
            raise ValueError("phase capture window query digests do not match")
        for query in decision_queries:
            _validate_phase_query(query, window)


def _validate_phase_query(
    query: Mapping[str, object], window: Mapping[str, object]
) -> None:
    if any(
        query.get(key) != window.get(key)
        for key in ("observation_phase", "capture_position")
    ):
        raise ValueError("phase capture query disagrees with its window")
    minimum = query.get("minimum_visible_frames")
    predicate = query.get("predicate")
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not 1 <= minimum <= PHASE_CAPTURE_FRAMES
        or not isinstance(predicate, Mapping)
        or not predicate.get("entity_ids")
    ):
        raise ValueError("phase capture observation query is incomplete")


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return set(value) | set().union(
            *(_nested_keys(item) for item in value.values()), set()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return set().union(*(_nested_keys(item) for item in value), set())
    return set()


def _window_indices(
    version: object,
    window: Mapping[str, object],
    frame_start: int,
    frame_end: int,
    num_frames: int,
    sampler_kind: str,
) -> list[int]:
    if version == PHASE_CAPTURE_PLAN_VERSION:
        if (
            int(window.get("command_end_exclusive", -1))
            != int(window["command_start"]) + 1
        ):
            raise ValueError("phase capture requires exactly one command")
        return phase_contiguous_indices(
            frame_start,
            frame_end + 1,
            str(window.get("observation_phase") or ""),
            window.get("capture_position"),
            num_frames,
        )
    if sampler_kind == "action_progress_quartiles":
        if (
            int(window.get("command_end_exclusive", -1))
            != int(window["command_start"]) + 1
        ):
            raise ValueError("action quartile capture requires exactly one command")
        return action_quartile_indices(frame_start, frame_end + 1)
    sample_position = window.get("sample_position")
    if sample_position is None:
        if version == EGOLONGQA_CAPTURE_PLAN_V2_VERSION:
            return official_uniform_endpoint_indices(frame_start, frame_end, num_frames)
        return uniform_endpoint_indices(frame_start, frame_end, num_frames)
    if sample_position == "midpoint" and num_frames == 1:
        return [int(round(0.5 * (frame_start + frame_end)))]
    raise ValueError(
        "capture-plan sample_position supports only midpoint with num_frames=1"
    )


def phase_contiguous_indices(
    frame_start: int,
    frame_end_exclusive: int,
    phase: str,
    position: object,
    num_frames: int = PHASE_CAPTURE_FRAMES,
) -> list[int]:
    """Resolve one semantic decision phase against an end-exclusive command span."""
    if isinstance(position, bool) or not isinstance(position, (int, float)):
        raise ValueError("phase capture position must be numeric")
    numeric = float(position)
    if phase == "pre" and numeric == 0.0:
        first, lower = frame_start - num_frames, 0
    elif phase == "post" and numeric == 1.0:
        first, lower = frame_end_exclusive - num_frames, frame_start
    elif phase == "mid" and 0.0 < numeric < 1.0:
        anchor = frame_start + int((frame_end_exclusive - frame_start - 1) * numeric)
        first = anchor - int((num_frames - 1) * numeric)
        first = min(max(first, frame_start), frame_end_exclusive - num_frames)
        lower = frame_start
    else:
        raise ValueError("phase and capture position are inconsistent")
    if first < lower or first + num_frames > frame_end_exclusive:
        raise ValueError("phase capture interval has fewer than requested frames")
    return list(range(first, first + num_frames))


def materialize_compatibility_sequence(
    frame_dir: str | Path,
    *,
    total_frames: int,
    capture_indices: Sequence[int],
) -> dict[str, int]:
    """Fill unrendered indices with hard links so normal video readers keep the timeline."""
    selected = sorted({int(index) for index in capture_indices})
    if not selected or selected[0] < 0 or selected[-1] >= total_frames:
        raise ValueError("capture indices are empty or outside the timeline")
    root = Path(frame_dir)
    for index in selected:
        if not (root / f"frame_{index:05d}.png").is_file():
            raise ValueError(f"captured frame {index} is missing")
    linked = 0
    copied = 0
    for index in range(total_frames):
        target = root / f"frame_{index:05d}.png"
        if target.is_file():
            continue
        insertion = bisect.bisect_left(selected, index)
        candidates = selected[max(0, insertion - 1) : insertion + 1]
        nearest = min(candidates, key=lambda value: (abs(value - index), value))
        source = root / f"frame_{nearest:05d}.png"
        try:
            os.link(source, target)
            linked += 1
        except OSError:
            shutil.copyfile(source, target)
            copied += 1
    return {"linked": linked, "copied": copied}


@dataclass
class SegmentCtx:
    """Everything the per-frame callback needs, bundled so demos stay terse."""

    a_fpv: object
    a_tpv: object
    fpv_dir: str
    tpv_dir: str
    rcfg: "RenderCfg"
    # Clamp dt to exactly one capture period (kills first-frame camera teleport flash).
    dt: float
    # Render resolution (W,H), threaded to readback validation and PNG output.
    width: int = 1280
    height: int = 720
    fpv_retries: int = 0
    tpv_retries: int = 0
    temporal_reset_pending: bool = True
    temporal_refresh_frames: int = 16
    last_advanced_index: int | None = None
    describe_fpv: "Callable[[int], str] | None" = None
    fpv_last_readback: object | None = None
    on_authored_pre_step: "Callable[[int], None] | None" = None
    on_post_positive_step: "Callable[[int, int], None] | None" = None
    on_delivered_readback: "Callable[[int, object, object, int], None] | None" = None

    def record_fpv_readback(self, readback) -> None:
        self.fpv_last_readback = readback
        self.fpv_retries += int(readback.diagnostics.retries)

    def record_tpv_readback(self, readback) -> None:
        self.tpv_retries += int(readback.diagnostics.retries)


def _should_capture(
    index: int,
    cap_lo: int,
    cap_hi: int,
    capture_indices: Set[int] | None,
) -> bool:
    in_window = cap_lo <= index and (cap_hi < 0 or index < cap_hi)
    return in_window and (capture_indices is None or index in capture_indices)


def _refresh_temporal_state(
    ctx: SegmentCtx,
    index: int,
    *,
    fast_skip: bool,
    sparse_refresh_frames: int,
) -> int:
    if ctx.temporal_reset_pending:
        refreshes = ctx.temporal_refresh_frames
        ctx.temporal_reset_pending = False
    elif (
        fast_skip
        and sparse_refresh_frames > 0
        and ctx.last_advanced_index is not None
        and index > ctx.last_advanced_index + 1
    ):
        refreshes = sparse_refresh_frames
    else:
        return 0
    converge_at_current_time(ctx.rcfg, refreshes)
    return refreshes


def run_segment(
    ctx: SegmentCtx,
    name: str,
    n_frames: int,
    on_frame: "Callable[[float, int], None]",
    gidx: int,
    max_frames: int,
    on_post_step: "Callable[[float, int], None] | None" = None,
    cap_lo: int = 0,
    cap_hi: int = -1,
    cap_tpv: bool = True,
    fast_skip: bool = False,
    capture_indices: Set[int] | None = None,
    sparse_refresh_frames: int = 0,
) -> int:
    """Run one segment of ``n_frames`` captures; return the new global frame index.

    ``on_frame(t, gidx)`` sets all scripted state for normalized time t in [0,1] and absolute
    frame index gidx. It is called BEFORE the render step so the captured frame reflects the
    new poses. The final frame of the segment (t==1.0) is always captured (segment-final state
    is what symbolic checks read), unless the global max-frame budget is hit first.

    ``on_post_step(t, gidx)`` (optional) is called IMMEDIATELY after the single render/physics step,
    before the next frame -- the only place to read post-step DYNAMIC state (a physics-driven object's
    pose, the PhysX contact-report buffer, which is valid for exactly the step just taken). Default
    None leaves the scripted demos (demo1/2/3) untouched.

    FRAME-SHARDING capture window ``[cap_lo, cap_hi)`` (MAST multi-GPU): the render clock is STILL
    advanced for EVERY frame (``on_frame`` + ``advance_once`` + ``on_post_step`` run unchanged), so the
    single-clock physics + RTX state at each frame are identical to a full render; only the CAPTURE (the
    annotator readback + the GLOBAL-indexed PNG write + the black-frame validate) is windowed. Frames
    outside the window are read and checked but NOT written, so annotator history stays identical to a
    full render while N shards each own a disjoint global range that
    merges into one contiguous ``frame_%05d.png`` sequence. Defaults (``cap_lo=0``, ``cap_hi=-1`` sentinel
    = no upper bound, ``cap_tpv=True``) capture EVERY frame + BOTH products -> byte-identical for every
    non-sharded caller. ``cap_tpv=False`` suppresses TPV PNG writes while retaining its readback and
    retry history (optional per-shard IO saving without forking renderer state).

    SHARD-FAST ``fast_skip`` (opt-in REAL sharding speedup): when True, the EXPENSIVE ``advance_once``
    (``rep.orchestrator.step`` == the raytracing) is SKIPPED for frames OUTSIDE ``[cap_lo, cap_hi)`` so a
    shard raytraces only ~T/N frames instead of all T (the whole point of sharding -- without this every
    shard pays the full RTX cost). ``on_frame`` still runs for EVERY frame so the cheap per-frame
    kinematic / camera-smoothing / nudge state stays deterministic up to the window start; ONLY the RTX
    step (and, with it, the readback) is elided outside the window. The CALLER MUST gate ``fast_skip`` on
    the absence of per-frame physics coupling (no live free-dynamic PhysX): with physics live, each
    frame's object pose integrates from the prior ``advance_once``, so skipping would DESYNC the sim.
    Default False -> byte-identical to a full ``advance_once``-every-frame render.

    *** BYTE-IDENTICAL boundary-frame verification REQUIRED before trusting ``fast_skip`` (the RTX
        temporal denoiser carries cross-frame history, so the window's FIRST captured frames may differ
        from a full render; GPU was busy at impl time). ***
    """
    if n_frames <= 0:
        return gidx

    LOG.info("SEG %-22s frames=%d (gidx %d..)", name, n_frames, gidx)
    for i in range(n_frames):
        if gidx >= max_frames:
            LOG.warning(
                "SEG %s: hit max_frames=%d at gidx=%d; stopping early",
                name,
                max_frames,
                gidx,
            )
            break
        # t spans 0..1 inclusive so the segment-final pose is captured on the last frame.
        t = i / (n_frames - 1) if n_frames > 1 else 1.0
        # on_frame is CHEAP (pose math + camera/nudge smoothing) and carries cross-frame state, so it
        # runs for EVERY frame even under fast_skip -- that is what keeps the window-start deterministic.
        on_frame(t, gidx)
        if ctx.on_authored_pre_step is not None:
            ctx.on_authored_pre_step(gidx)

        capture_frame = _should_capture(
            gidx,
            cap_lo,
            cap_hi,
            capture_indices,
        )
        # SHARD-FAST: skip the EXPENSIVE advance_once (rep.orchestrator.step == the raytracing) for
        # frames OUTSIDE this shard's capture window, so a shard raytraces only ~T/N frames (the real
        # sharding speedup). Sound ONLY with NO per-frame physics coupling -- the caller gates fast_skip
        # on `not fdp`. The cheap kinematic/camera/nudge state was already advanced by on_frame above.
        # *** BYTE-IDENTICAL boundary-frame verification REQUIRED before trusting --shard-fast (the RTX
        #     temporal denoiser has cross-frame history -> the window's first frames may differ from a
        #     full render; GPU was busy at impl time). ***
        if fast_skip and not capture_frame:
            gidx += 1
            continue

        pre_step_refreshes = _refresh_temporal_state(
            ctx,
            gidx,
            fast_skip=fast_skip,
            sparse_refresh_frames=sparse_refresh_frames,
        )

        # SINGLE CLOCK: advance the orchestrator ONCE, then read BOTH products so the FPV and
        # TPV PNGs for this gidx share the same simulation time (no 2x double-step).
        advance_once(ctx.rcfg)
        ctx.last_advanced_index = gidx
        if on_post_step is not None:
            on_post_step(
                t, gidx
            )  # read the just-stepped dynamic pose + contact buffer here
        if ctx.on_post_positive_step is not None:
            ctx.on_post_positive_step(gidx, pre_step_refreshes)
        # Read every advanced frame even outside the shard's write window so the
        # annotator history is identical to a full render. Empty reads receive the
        # original one-extra-read fallback; no pose search or gray-frame recovery runs.
        rgb_fpv = read_and_save(
            ctx.a_fpv,
            ctx.fpv_dir,
            gidx,
            ctx.width,
            ctx.height,
            ctx.rcfg,
            max_retries=DEFAULT_READBACK_RETRIES,
            on_readback=ctx.record_fpv_readback,
            save=capture_frame,
        )
        if ctx.on_delivered_readback is not None:
            ctx.on_delivered_readback(
                gidx,
                rgb_fpv,
                ctx.fpv_last_readback,
                0,
            )
        read_and_save(
            ctx.a_tpv,
            ctx.tpv_dir,
            gidx,
            ctx.width,
            ctx.height,
            ctx.rcfg,
            max_retries=DEFAULT_READBACK_RETRIES,
            on_readback=ctx.record_tpv_readback,
            save=capture_frame and cap_tpv,
        )
        try:
            validate_frame(rgb_fpv, gidx, LOG)
        except RuntimeError as exc:
            # A single black frame should not abort an otherwise-good render; warn loudly.
            LOG.warning("SEG %s: %s", name, exc)
        gidx += 1
    return gidx
