# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ego_frames.py -- egocentric frame-window buffer for the two-agent RL loop.

``uniform_pick`` + ``EgoFrameBuffer`` define how the Isaac Sim two-agent RL loop feeds frames:
  * per command (== one "interval") the env renders up to ``per_interval`` egocentric FPV frames (4 by
    default); the loop stores them here via ``add_interval``.
  * the TRAINED assistant consumes ONLY the newest interval (``delta`` == ``intervals[-1]``) each tick --
    the append-only verl trajectory IS its growing window; re-striding the whole history would break the
    1:1 ``<|image_pad|>`` <-> ``_mm_images`` invariant, so it must NOT call fed/windowed_frames.
  * frozen-human callers choose a bounded stateless view: recent action intervals for interactive
    evaluation, or the legacy restrided cumulative window where that historical behavior is required.

The buffer is content-agnostic: it stores whatever ``add_interval`` is given (Isaac passes base64 PNG
strings; file paths work equally well). STDLIB-only -> importable in the trainer's plain-python process.

    python3 -m simulator.interaction.ego_frames   # offline selftest (no dependencies)
"""

from __future__ import annotations

import os


# Motion-segment tags naming the frames in which a [LookAt] is actually HOLDING the cue steady in
# front of the camera. ``keep_final`` alone pins the interval's LAST frame, which for an inspect is
# the RELEASE -- object already back on the bench, label out of frame -- and for a sign look-at is
# the zoom-OUT. Both are the action's chronological end and neither carries its result, so at a
# small ``per_interval`` the whole point of the command was being dropped while every counter said
# the frames had been delivered.
CUE_HOLD_SEGS = ("look_hold",)


def uniform_pick(items: list, n: int) -> list:
    """``n`` items uniformly strided across ``items``.

    ``step = len/n`` with ``int(i*step)`` -> the largest picked index is < ``len-1`` (the last item is
    structurally dropped) unless ``len <= n`` (then all items are returned). ``n <= 0`` -> ``[]``.
    """
    items = list(items)
    if not items or n <= 0:
        return []
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _uniform_end_indices(length: int, count: int) -> list[int]:
    """Choose end-of-bin indices across the complete input range."""
    if count <= 0:
        return []
    if count >= length:
        return list(range(length))
    return [((index + 1) * length + count - 1) // count - 1 for index in range(count)]


def _budgeted_interval_counts(length: int, budget: int) -> list[int]:
    """Spread a frame allowance uniformly over a chronological interval prefix."""
    if length <= 0:
        return []
    base, remainder = divmod(max(0, budget), length)
    counts = [base] * length
    for index in _uniform_end_indices(length, remainder):
        counts[index] += 1
    return counts


class EgoFrameBuffer:
    """Accumulate per-command frame intervals; expose the trained-assistant DELTA and the frozen-human
    restrided WINDOW (RL defaults: per_interval=4, max_frames=16)."""

    def __init__(
        self,
        per_interval: int = 4,
        max_frames: int = 16,
        keep_final: bool | None = None,
    ) -> None:
        self.per_interval = max(1, int(per_interval))
        self.max_frames = max(1, int(max_frames))
        # keep_final: overwrite the last sampled slot with the TRUE final frame (the action's RESULT --
        # placed object / opened door). Gated by VH_SPARSE_KEEP_FINAL; recommended ON for sparse RL
        # renders so the result frame is always in the feed. Explicit arg overrides the env gate.
        # DEFAULT ON: `uniform_pick` structurally drops the last item, so with this off the RESULT of
        # the action -- object now in hand, item now in the bin -- is never guaranteed to be in the
        # feed at all. That matters because the frame taken at the END of a grab has the can filling
        # the middle of the view, while mid-motion frames are floor, ceiling and blur. Without it the
        # assistant would have to notice "you are already holding it" from a sample that could easily
        # contain no such frame.
        self.keep_final = (
            (os.getenv("VH_SPARSE_KEEP_FINAL", "1") != "0")
            if keep_final is None
            else bool(keep_final)
        )
        # Fraction of an action's frames the sample is drawn from, counted back from the END (1.0 = the
        # whole action, as before). An action's informative part is its tail: the approach is mostly
        # walking. 0.5 spends the same 4 frames on the half that shows the outcome.
        self.tail_frac = float(os.getenv("ISAAC_FRAME_TAIL_FRAC", "1.0"))
        self.intervals: list[list] = []
        # Number of leading intervals that are PINNED: they survive every budget/restride, because they
        # are the opening look-around and the assistant's whole grasp of the room rests on them. Only
        # what comes after competes for the remaining allowance.
        self.pinned: int = 0
        # Set by add_interval whenever a cue-hold frame had to be rescued, and by how many intervals --
        # so "the cue was delivered" is a counted fact rather than an assumption. See CUE_HOLD_SEGS.
        self.n_cue_holds_pinned: int = 0
        self.n_intervals_with_cue: int = 0

    def add_interval(self, frames: list, segs: list | None = None) -> None:
        """Add one command's rendered frames as a new interval, uniformly sampled to ``per_interval``
        Empty input adds nothing (an idle/no-op tick contributes no frames).

        ``segs`` is the per-frame motion-segment tag, parallel to ``frames`` (``isaac_env`` puts it in
        ``obs["frame_segs"]``). Omit it and the selection is byte-identical to the historical one; supply
        it and an interval that contains CUE_HOLD_SEGS frames is guaranteed to deliver one of them.
        """
        fp = list(frames or [])
        sp = list(segs or [])
        if len(sp) != len(fp):
            sp = []  # a partial/misaligned tag list is worse than none: never guess an alignment
        if 0.0 < self.tail_frac < 1.0 and len(fp) > self.per_interval:
            _keep = max(self.per_interval, int(round(len(fp) * self.tail_frac)))
            fp, sp = fp[-_keep:], (sp[-_keep:] if sp else [])
        # Index-based so the tags stay aligned with the picks. uniform_pick over range(len) selects the
        # very same positions it would select over the items themselves.
        idx = uniform_pick(list(range(len(fp))), self.per_interval)
        if idx and self.keep_final and idx[-1] != len(fp) - 1:
            idx[-1] = len(fp) - 1  # guarantee the action's RESULT frame is in the feed
        holds = [i for i, s in enumerate(sp) if s in CUE_HOLD_SEGS]
        if holds:
            self.n_intervals_with_cue += 1
            if not set(idx) & set(holds):
                # The action's result IS the hold here, so it replaces -- not joins -- the final frame:
                # joining would silently spend one more of a frame budget that is already the binding
                # constraint on model memory.
                idx[-1] = holds[-1]
                idx.sort()
                self.n_cue_holds_pinned += 1
        sel = [fp[i] for i in idx]
        if sel:
            self.intervals.append(sel)

    def delta(self) -> list:
        """The NEWEST interval's frames -- the per-tick append the TRAINED assistant adds to its append-only
        trajectory (``intervals[-1]``, never the restrided window)."""
        return list(self.intervals[-1]) if self.intervals else []

    def recent_frames(self, interval_count: int = 2) -> list:
        """Flatten the newest ``interval_count`` action intervals in chronological order."""
        count = max(0, int(interval_count))
        if count == 0:
            return []
        return [frame for interval in self.intervals[-count:] for frame in interval]

    @staticmethod
    def budget_schedule(
        max_ticks: int, budget: int, per: int = 4, recent_k: int = 4
    ) -> list:
        """Frames-per-tick decided UP FRONT, so the same budget can be spent without ever un-appending.

        ``budgeted_frames`` re-strides the history on every call, which a stateless consumer can do but
        the trained assistant cannot: its frame history IS the RL trajectory, one token sequence whose
        log-probs get recomputed at training time. Deleting a frame from tick 3 at tick 20 would mean the
        stored sequence is not the one the model actually generated against -- the recomputed log-probs
        would no longer match the rollout's, which is precisely the corruption
        ``rollout_actor_probs_pearson_corr`` exists to catch.

        Fixing the split in advance sidesteps that: every tick appends exactly its allotted number and
        nothing is ever rewritten. For a full-length episode the allocation is identical to
        ``budgeted_frames`` -- last ``recent_k`` ticks whole, with the remaining allowance spread
        uniformly across the older timeline -- so both agents can be fed the same temporal coverage.
        An episode that stops early can end on a thin tick because the schedule cannot know its actual
        stopping point.
        """
        if max_ticks <= 0:
            return []
        if budget <= 0 or max_ticks * per <= budget:
            return [per] * max_ticks
        k = min(int(recent_k), max_ticks)
        n = max_ticks - k
        share = budget - k * per
        if n <= 0 or share <= 0:
            return [per] * max_ticks
        return _budgeted_interval_counts(n, share) + [per] * k

    def budgeted_frames(self, budget: int, recent_k: int = 4) -> list:
        """History squeezed into at most ``budget`` frames, spending it on the RECENT end.

        The last ``recent_k`` intervals are kept whole because they are what the next decision turns on.
        The remaining allowance is spread uniformly over the complete older timeline. If there are more
        old intervals than available frames, representative intervals are selected across the full
        prefix instead of retaining only a recent suffix. A selected one-frame interval contributes its
        final stored frame, preserving action-result and rescued cue-hold frames.

        NON-append: a past frame can drop out between calls, so this is only safe for a stateless
        consumer (the frozen human). The trained assistant's frame history IS its training trajectory and
        can only ever be appended to.
        """
        if budget <= 0:
            return self.fed_frames()
        acc = [p for itv in self.intervals for p in itv]
        if len(acc) <= budget:
            return acc
        pin = self.intervals[: self.pinned]
        pinned_frames = [p for itv in pin for p in itv]
        rest = self.intervals[self.pinned :]
        k = max(0, min(int(recent_k), len(rest)))
        recent = rest[len(rest) - k :] if k else []
        older = rest[: len(rest) - k]
        share = budget - len(pinned_frames) - sum(len(itv) for itv in recent)
        out: list = []
        if older and share > 0:
            for itv, take in zip(older, _budgeted_interval_counts(len(older), share)):
                if take <= 0:
                    continue
                sel = uniform_pick(itv, take)
                if sel and self.keep_final and itv and sel[-1] != itv[-1]:
                    sel[-1] = itv[-1]
                out += sel
        out += [p for itv in recent for p in itv]
        # Prologue first, ALWAYS -- the trim below only ever eats into what came after it.
        out = pinned_frames + (
            out[-(budget - len(pinned_frames)) :] if budget > len(pinned_frames) else []
        )
        return out

    def fed_frames(self) -> list:
        """Whole history flattened + GLOBALLY restrided to <= ``max_frames``. Re-strides every call (a
        past frame can drop out) -> NON-append; only for a non-trained consumer (the human)."""
        acc = [p for itv in self.intervals for p in itv]
        if len(acc) <= self.max_frames:
            return acc
        pinned = [p for itv in self.intervals[: self.pinned] for p in itv]
        rest = [p for itv in self.intervals[self.pinned :] for p in itv]
        room = max(0, self.max_frames - len(pinned))
        return pinned + (uniform_pick(rest, room) if room else [])

    def windowed_frames(self) -> list:
        """Causal two-tier sliding window: the last ``K`` intervals VERBATIM plus
        ``S`` uniformly-strided older-interval RESULT-keyframes (each older interval's last frame), capped at
        ``max_frames`` (oldest dropped first). Gated by ``VH_STREAM_WINDOW`` (off -> ``fed_frames``). Also
        NON-append (re-strides) -> for the frozen human only."""
        if os.environ.get("VH_STREAM_WINDOW", "0") == "0":
            return self.fed_frames()
        k = max(1, int(os.environ.get("VH_STREAM_RECENT_K", "3") or 3))
        s = max(0, int(os.environ.get("VH_STREAM_OLDER_M", "4") or 4))
        recent = self.intervals[-k:]
        older = self.intervals[: len(self.intervals) - len(recent)]
        older_keys = [itv[-1] for itv in older if itv]
        out = (uniform_pick(older_keys, s) if s else []) + [
            p for itv in recent for p in itv
        ]
        return out[-self.max_frames :] if len(out) > self.max_frames else out


def _selftest() -> int:
    """Offline selftest (no deps): uniform_pick semantics + buffer delta/fed/windowed."""
    ok = True

    def _chk(cond: bool, msg: str) -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    _chk(uniform_pick([1, 2, 3], 5) == [1, 2, 3], "uniform_pick len<=n keeps all")
    _chk(
        uniform_pick([], 4) == [] and uniform_pick([1, 2], 0) == [],
        "uniform_pick empties",
    )
    _chk(
        uniform_pick(list(range(8)), 4) == [0, 2, 4, 6],
        "uniform_pick strides, drops last",
    )

    fb = EgoFrameBuffer(per_interval=4, max_frames=16, keep_final=False)
    _chk(fb.delta() == [] and fb.fed_frames() == [], "empty buffer -> empty")
    fb.add_interval(["a1", "a2", "a3", "a4", "a5", "a6"])  # 6 -> sampled to 4
    _chk(len(fb.intervals[-1]) == 4, "add_interval samples to per_interval")
    _chk(fb.delta() == fb.intervals[-1], "delta == newest interval")
    fb.add_interval(["b1", "b2"])
    _chk(fb.delta() == ["b1", "b2"], "delta tracks the newest interval")
    fbr = EgoFrameBuffer(per_interval=4, max_frames=256, keep_final=False)
    fbr.add_interval(["a1", "a2", "a3", "a4"])
    fbr.add_interval(["b1", "b2", "b3", "b4"])
    fbr.add_interval(["c1", "c2", "c3", "c4"])
    _chk(
        fbr.recent_frames(2) == ["b1", "b2", "b3", "b4", "c1", "c2", "c3", "c4"],
        "recent_frames drops older intervals and keeps the newest two",
    )
    _chk(fbr.recent_frames(0) == [], "recent_frames zero interval count is empty")
    _chk(
        fb.fed_frames() == [f for itv in fb.intervals for f in itv],
        "fed_frames == flatten under cap",
    )

    # keep_final overwrites the last sampled slot with the true final frame.
    fbk = EgoFrameBuffer(per_interval=4, keep_final=True)
    fbk.add_interval(["x1", "x2", "x3", "x4", "x5", "x6"])
    _chk(fbk.intervals[-1][-1] == "x6", "keep_final puts the RESULT frame last")

    # THE CUE-HOLD RESCUE. A [LookAt] on a graspable renders reach / hold / hold / release; at
    # per_interval 2 the historical pick was [reach, release] -- the two frames in which the item is
    # NOT held up. Regression-pin the exact shape, because the oracle arm cannot detect its absence.
    _looked = ["f0", "f1", "f2", "f3"]
    _segs = ["look_reach", "look_hold", "look_hold", "look_release"]
    fb_old = EgoFrameBuffer(per_interval=2, keep_final=True)
    fb_old.add_interval(_looked)  # no tags -> the historical behaviour, unchanged
    _chk(fb_old.intervals[-1] == ["f0", "f3"], "no segs -> byte-identical legacy pick")
    fb_new = EgoFrameBuffer(per_interval=2, keep_final=True)
    fb_new.add_interval(_looked, _segs)
    _chk(
        fb_new.intervals[-1] == ["f0", "f2"],
        "segs -> the HOLD frame replaces the release",
    )
    _chk(
        fb_new.n_cue_holds_pinned == 1 and fb_new.n_intervals_with_cue == 1,
        "cue-hold rescue is COUNTED, not silent",
    )
    fb1 = EgoFrameBuffer(per_interval=1, keep_final=True)
    fb1.add_interval(_looked, _segs)
    _chk(
        fb1.intervals[-1] == ["f2"], "per_interval 1 delivers the hold, not the release"
    )
    fb4 = EgoFrameBuffer(per_interval=4, keep_final=True)
    fb4.add_interval(_looked, _segs)
    _chk(
        fb4.intervals[-1] == _looked and fb4.n_cue_holds_pinned == 0,
        "a hold already in the pick is left alone (no extra frame spent)",
    )
    fbg = EgoFrameBuffer(per_interval=2, keep_final=True)
    fbg.add_interval(["g0", "g1", "g2", "g3"], ["grasp", "lift", "lift", "carry"])
    _chk(
        fbg.intervals[-1] == ["g0", "g3"] and fbg.n_intervals_with_cue == 0,
        "a non-inspect interval is untouched by the rescue",
    )
    fbm = EgoFrameBuffer(per_interval=2, keep_final=True)
    fbm.add_interval(["m0", "m1"], ["look_hold"])  # misaligned tag list
    _chk(
        fbm.intervals[-1] == ["m0", "m1"],
        "a misaligned segs list is ignored, never guessed",
    )
    fbo = EgoFrameBuffer(
        per_interval=3, keep_final=True
    )  # picks 0,3 then keep_final pins 9
    fbo.add_interval(
        [f"o{i}" for i in range(10)],
        ["look_reach"] * 4 + ["look_hold"] + ["look_lower"] * 5,
    )
    _chk(
        fbo.intervals[-1] == ["o0", "o3", "o4"],
        "the rescued interval stays in chronological order and keeps its slot count",
    )

    # cap: fed_frames restrides down to max_frames.
    fbc = EgoFrameBuffer(per_interval=4, max_frames=8, keep_final=False)
    for i in range(5):
        fbc.add_interval([f"{i}_{j}" for j in range(4)])  # 20 frames total
    _chk(len(fbc.fed_frames()) == 8, "fed_frames caps at max_frames")

    # windowed_frames gated: off -> fed_frames.
    _chk(fbc.windowed_frames() == fbc.fed_frames(), "windowed off -> fed_frames")
    os.environ["VH_STREAM_WINDOW"] = "1"
    os.environ["VH_STREAM_RECENT_K"] = "2"
    os.environ["VH_STREAM_OLDER_M"] = "2"
    win = fbc.windowed_frames()
    _chk(
        0 < len(win) <= 8 and win[-1] == "4_3",
        "windowed on -> recent-dense, capped, newest last",
    )
    os.environ.pop("VH_STREAM_WINDOW", None)

    print(f"[ego_frames] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())
