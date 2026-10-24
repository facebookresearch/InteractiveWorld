# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Retain on disk EXACTLY what a served assistant was sent, and exactly what it sent back.

WHY THIS EXISTS. A model can narrate a stream of events that provably did not happen -- *"you
dropped it on the floor"*, *"you're throwing the battery towards the sinks"* -- while the episode
manifest records ``n_fail: 0`` and ``all_placed: true``. If every frame has already been discarded, it
is impossible to say afterwards whether the model had been shown usable images, or whether the answer
container was ever in view at all. **An unadjudicable result is worse than a bad one**, and the only
fix is to keep the input.

WHAT IS RETAINED, per episode, under ``<out_dir>/<tag>_io/``::

  turnNN_imgMM.png        every image in that turn's request, in request order, byte-identical to
                          the PNG bytes sent to the HTTP JSON endpoint. A repeat across turns is
                          hard-linked to its first occurrence where the filesystem allows it and
                          copied where it does not -- on filesystems that refuse ``os.link`` every
                          repeat is a real copy and the trail costs its full nominal size. ``disk``
                          reports the two counts separately so the figure is never a guess.
  turnNN_request.json     the request as ACTUALLY ASSEMBLED BY THE BACKEND -- system prompt with the
                          goal already substituted, the user message (flattened history included, if
                          that backend flattens), and the ordered image list. The image payload
                          itself is replaced by a per-image record (file, sha256, bytes, pixels,
                          position in the cumulative stream) because a base64 blob in a JSON file is
                          not readable and the bytes are already on disk beside it.
  human_turnNN_imgMM.png  every image sent over the Human HTTP boundary on that turn. These use the
                          same content-addressed store as Assistant images, so identical frames are
                          hard-linked where possible and otherwise copied without losing provenance.
  turnNN_response.txt     the raw completion, before any parsing.
  index.json / Markdown   turn -> frames delivered -> request -> raw reply -> parsed action -> what
                          the person then did -> the engine's result, plus the per-turn answer to
                          "was an answer container in the delivered frames".

The HTTP JSON backend re-sends the complete multi-turn history. Its exact provider-neutral request
and raw response are retained so a reader can audit the model input without provider-specific tools.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
from typing import Any

_PNG_SIG: bytes = b"\x89PNG\r\n\x1a\n"


def png_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) straight out of the IHDR chunk. Stdlib only -- no PIL in this process."""
    if len(data) < 24 or not data.startswith(_PNG_SIG):
        return None
    return (
        int.from_bytes(data[16:20], "big"),
        int.from_bytes(data[20:24], "big"),
    )


class IOCapture:
    """One episode's request/response trail. Every method is a no-op-safe recorder, never a gate.

    NOTHING HERE MAY EVER FAIL AN EPISODE. A recorder that can raise turns into an instrument that
    deletes the measurements it was added to preserve.
    Every public method swallows its own IO errors into ``self.errors`` and returns.
    """

    def __init__(self, out_dir: str, tag: str, meta: dict | None = None) -> None:
        self.dir: str = os.path.join(out_dir, f"{tag}_io")
        self.tag: str = tag
        self.meta: dict = dict(meta or {})
        self.turns: list[dict] = []
        self.errors: list[str] = []
        self._by_hash: dict[str, str] = {}
        self._prov: list[dict] = []
        self.n_files: int = 0
        self.n_unique: int = 0
        self.n_linked: int = 0
        self.n_copied: int = 0
        self.bytes_unique: int = 0
        self.bytes_nominal: int = 0
        self.t0: float = time.time()
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError as exc:
            self.errors.append(f"mkdir: {exc}")

    # -- turn bookkeeping ---------------------------------------------------------------------
    def begin_turn(self, turn: int) -> None:
        self.turns.append(
            {
                "turn": int(turn),
                "requests": [],
                "t_begin": round(time.time() - self.t0, 2),
            }
        )

    @property
    def active(self) -> bool:
        """False before the first tick -- a backend's startup ``probe()`` is not part of the episode."""
        return bool(self.turns)

    def _cur(self) -> dict | None:
        return self.turns[-1] if self.turns else None

    def set_frame_provenance(self, prov: list[dict]) -> None:
        """The ordered provenance for the image batch in the next backend ``generate`` call."""
        self._prov = list(prov or [])

    def current_frame_provenance(self, n_frames: int) -> list[dict]:
        """Return provenance aligned with the current input batch, failing the capture if not."""
        provenance = [dict(value) for value in self._prov]
        if len(provenance) != n_frames:
            self.errors.append(
                "incoming frame/provenance count mismatch: "
                f"{n_frames} frame(s), {len(provenance)} provenance record(s)"
            )
        if len(provenance) < n_frames:
            provenance.extend(
                {"interval": None, "source": "(unattributed)"}
                for _ in range(n_frames - len(provenance))
            )
        return provenance[:n_frames]

    # -- frames -------------------------------------------------------------------------------
    def _write_blob(self, data: bytes, name: str) -> dict:
        path = os.path.join(self.dir, name)
        digest = hashlib.sha256(data).hexdigest()
        first = self._by_hash.get(digest)
        if first is None:
            with open(path, "wb") as fh:
                fh.write(data)
            self._by_hash[digest] = name
            self.n_unique += 1
            self.n_files += 1
            self.bytes_unique += len(data)
            self.bytes_nominal += len(data)
        elif not os.path.exists(path):
            # Same bytes under a new turn/order name. Hard-link so the naming stays complete while
            # the disk cost stays at the unique frames -- a hard link is a real file to every reader.
            # Some filesystems refuse os.link (PermissionError); there this falls back to a real
            # copy and the trail costs the full nominal size. That is a fact about the filesystem,
            # not a defect: the fallback is what keeps every turnNN_imgMM.png a readable file. Both
            # counts are reported so the disk figure is never a guess.
            src = os.path.join(self.dir, first)
            try:
                os.link(src, path)
                self.n_linked += 1
            except OSError:
                shutil.copyfile(src, path)
                self.n_copied += 1
                self.bytes_unique += len(data)
            self.n_files += 1
            self.bytes_nominal += len(data)
        wh = png_size(data)
        return {
            "file": name,
            "sha256": digest[:16],
            "n_bytes": len(data),
            "px": list(wh) if wh else None,
            "same_as": first if first and first != name else None,
        }

    def save_images(
        self,
        items: list[tuple[str, Any]],
        turn: int | None = None,
        provenance: list[dict] | None = None,
    ) -> list[dict]:
        """``[("b64", str) | ("path", str), ...]`` in REQUEST ORDER -> one record per image.

        The index into ``items`` is the on-disk suffix and ``request_index``. ``global_index`` comes
        from the selected frame's provenance, so it remains the original cumulative-stream position
        even after the backend uniformly samples that stream to its image cap.
        """
        t = self.turns[-1]["turn"] if (turn is None and self.turns) else (turn or 0)
        frame_provenance = (
            self.current_frame_provenance(len(items))
            if provenance is None
            else [dict(value) for value in provenance]
        )
        if len(frame_provenance) != len(items):
            self.errors.append(
                "request frame/provenance count mismatch: "
                f"{len(items)} frame(s), {len(frame_provenance)} provenance record(s)"
            )
            frame_provenance.extend(
                {"interval": None, "source": "(unattributed)"}
                for _ in range(len(items) - len(frame_provenance))
            )
            frame_provenance = frame_provenance[: len(items)]
        out: list[dict] = []
        for i, (kind, val) in enumerate(items):
            try:
                if kind == "b64":
                    data = base64.b64decode(val)
                elif kind == "path":
                    with open(val, "rb") as fh:
                        data = fh.read()
                else:
                    data = bytes(val)
                rec = self._write_blob(data, f"turn{t:02d}_img{i:02d}.png")
            except (OSError, ValueError) as exc:
                self.errors.append(
                    f"turn{t:02d}_img{i:02d}: {type(exc).__name__}: {exc}"
                )
                rec = {"file": None, "error": f"{type(exc).__name__}: {exc}"}
            rec["request_index"] = i
            if kind == "path":
                rec["sent_as"] = val
            if i < len(frame_provenance):
                rec["provenance"] = frame_provenance[i]
                rec["global_index"] = frame_provenance[i].get("global_index")
            out.append(rec)
        return out

    def record_human_input(
        self,
        frames_b64: list[str],
        *,
        turn: int,
        provenance: list[dict] | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        raw_response: str = "",
        parsed_action: str = "",
        spoken: str = "",
    ) -> list[dict]:
        """Persist one Human model request beside the Assistant request for this tick."""
        saved: list[dict] = []
        frame_provenance = list(provenance or [])
        for index, encoded in enumerate(frames_b64):
            try:
                data = base64.b64decode(encoded)
                record = self._write_blob(
                    data, f"human_turn{int(turn):02d}_img{index:02d}.png"
                )
            except (OSError, ValueError) as exc:
                self.errors.append(
                    f"human_turn{int(turn):02d}_img{index:02d}: "
                    f"{type(exc).__name__}: {exc}"
                )
                record = {"file": None, "error": f"{type(exc).__name__}: {exc}"}
            record["request_index"] = index
            if index < len(frame_provenance):
                record["provenance"] = frame_provenance[index]
            saved.append(record)
        current = self._cur()
        if current is not None:
            current.setdefault("human_inputs", []).append(
                {
                    "turn": int(turn),
                    "n_images": len(saved),
                    "images": [record.get("file") for record in saved],
                    "frame_records": saved,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "raw_response": raw_response,
                    "parsed_action": parsed_action,
                    "spoken": spoken,
                    "transport": "human_http_images_b64",
                }
            )
        return saved

    # -- request / response -------------------------------------------------------------------
    def record_request(
        self,
        backend: str,
        payload: dict,
        images: list[dict],
        notes: dict | None = None,
    ) -> None:
        cur = self._cur()
        if cur is None:
            return  # a probe call, before any turn -- not part of the episode
        attempt = len(cur["requests"]) + 1
        name = (
            f"turn{cur['turn']:02d}_request.json"
            if attempt == 1
            else f"turn{cur['turn']:02d}_request_attempt{attempt}.json"
        )
        blob = {
            "tag": self.tag,
            "turn": cur["turn"],
            "attempt": attempt,
            "backend": backend,
            "n_images": len(images),
            "images": images,
            "payload_as_sent": payload,
            "notes": dict(notes or {}),
        }
        try:
            with open(os.path.join(self.dir, name), "w") as fh:
                json.dump(blob, fh, indent=1, default=str)
                fh.write("\n")
        except OSError as exc:
            self.errors.append(f"{name}: {exc}")
        user = payload.get("user")
        if user is None:  # The HTTP JSON shape has no single user field.
            user = "\n".join(
                str(m.get("text") or "") for m in (payload.get("messages") or [])
            )
        cur["requests"].append(
            {
                "attempt": attempt,
                "file": name,
                "backend": backend,
                "n_images": len(images),
                "images": [im.get("file") for im in images],
                "px": sorted({tuple(im["px"]) for im in images if im.get("px")}),
                "system_chars": len(str(payload.get("system") or "")),
                "user_chars": len(str(user)),
            }
        )

    def record_response(
        self, raw: str, latency_s: float | None = None, error: str = ""
    ) -> None:
        cur = self._cur()
        if cur is None:
            return
        name = f"turn{cur['turn']:02d}_response.txt"
        try:
            with open(os.path.join(self.dir, name), "w") as fh:
                fh.write(raw if raw is not None else "")
        except OSError as exc:
            self.errors.append(f"{name}: {exc}")
        cur["response"] = {
            "file": name,
            "raw": raw,
            "n_chars": len(raw or ""),
            "latency_s": round(latency_s, 2) if latency_s is not None else None,
            "error": error,
        }

    # -- the loop's half ----------------------------------------------------------------------
    def record(self, **kw: Any) -> None:
        """Merge loop-side facts (parse, frames delivered, person's action, env result) into the turn."""
        cur = self._cur()
        if cur is not None:
            cur.update(kw)

    def record_episode_frames(self, ledger: Any) -> None:
        """The whole episode's interval table, so the trail stands alone without the render log."""
        self.meta["intervals"] = list(getattr(ledger, "intervals", []))
        self.meta["answers_key"] = sorted(getattr(ledger, "answers", set()))

    # -- closeout -----------------------------------------------------------------------------
    def disk(self) -> dict:
        return {
            "n_image_files": self.n_files,
            "n_unique_images": self.n_unique,
            "n_repeats_hardlinked": self.n_linked,
            "n_repeats_copied": self.n_copied,
            "bytes_on_disk": self.bytes_unique,
            "bytes_if_every_repeat_were_a_copy": self.bytes_nominal,
            "mb_on_disk": round(self.bytes_unique / 1e6, 1),
            "mb_if_every_repeat_were_a_copy": round(self.bytes_nominal / 1e6, 1),
        }

    def close(self, summary: dict | None = None) -> dict:
        blob = {
            "tag": self.tag,
            "meta": self.meta,
            "summary": dict(summary or {}),
            "disk": self.disk(),
            "capture_errors": self.errors,
            "turns": self.turns,
        }
        try:
            with open(os.path.join(self.dir, "index.json"), "w") as fh:
                json.dump(blob, fh, indent=1, default=str)
                fh.write("\n")
            with open(os.path.join(self.dir, "index.md"), "w") as fh:
                fh.write(render_index_md(blob))
        except OSError as exc:
            self.errors.append(f"index: {exc}")
        return blob


class FrameLedger:
    """Which env step rendered each frame the assistant was handed, and what the engine saw in it.

    THE QUESTION THIS ANSWERS is the one that separates *"it never saw the bin"* from *"it saw the
    bin and misread it"*, and it has to be answered PER TURN. The engine already measures it --
    ``obs["vischeck"]`` is an AABB sample inside the frustum of the frames that step rendered, within
    the horizon, with no wall between -- but the measurement is per STEP while the assistant is
    handed a WINDOW spanning many steps. So each interval's vischeck is banked as the interval is
    added, every delivered frame is traced back to its interval, and the per-turn verdict is the
    union over the frames that turn actually carried.

    ``vischeck["visible"]`` is capped at 32 names (``isaac_env._VIS_NAMES_MAX``) and the list is
    alphabetical, so a ``unit_*`` key sorts late and can be cut. When ``truncated`` is set, ABSENCE
    from the list is not evidence of absence from the frame, and the per-turn record says so instead
    of reporting a clean zero. The frames are on disk beside it; that is the tiebreaker.
    """

    def __init__(self, answers: set[str]) -> None:
        self.answers: set[str] = {str(a) for a in answers if a}
        self.intervals: list[dict] = []
        self.owner: dict[int, int] = {}
        self.cum: list[dict] = []

    def note(self, fb: Any, source: str, obs: dict) -> None:
        """Call straight after ``fb.add_interval``. A no-op when the buffer did not actually grow."""
        vc = obs.get("vischeck") or {}
        seen = set(vc.get("visible") or [])
        while len(self.intervals) < len(getattr(fb, "intervals", [])):
            i = len(self.intervals)
            self.intervals.append(
                {
                    "interval": i,
                    "source": source,
                    "answers_visible": sorted(self.answers & seen),
                    "vischeck_n_visible": vc.get("n_visible"),
                    "vischeck_n_entities": vc.get("n_entities"),
                    "vischeck_truncated": bool(vc.get("truncated")),
                    "vischeck_error": vc.get("error"),
                    "n_frames": len(fb.intervals[i]),
                }
            )
            for fr in fb.intervals[i]:
                self.owner[id(fr)] = i

    def trace(self, frames: list) -> list[dict]:
        """Return source provenance for ``frames`` without changing delivery history."""
        out: list[dict] = []
        for fr in frames:
            i = self.owner.get(id(fr))
            out.append(
                dict(self.intervals[i])
                if i is not None and i < len(self.intervals)
                else {"interval": None, "source": "(unattributed)"}
            )
        return out

    def deliver(self, frames: list) -> list[dict]:
        """Record one turn's newly-handed frames and return their provenance, in order."""
        out = self.trace(frames)
        start = len(self.cum)
        for offset, provenance in enumerate(out):
            provenance["global_index"] = start + offset
        self.cum.extend(out)
        return out

    def answers_seen(self) -> dict:
        """The per-turn verdict over every frame delivered so far."""
        seen: set[str] = set()
        n_trunc = n_err = 0
        for p in self.cum:
            seen |= set(p.get("answers_visible") or [])
            n_trunc += bool(p.get("vischeck_truncated"))
            n_err += bool(p.get("vischeck_error"))
        missing = sorted(self.answers - seen)
        return {
            "decidable": n_err == 0 and not (missing and n_trunc),
            "n_answers": len(self.answers),
            "n_visible": len(seen),
            "visible": sorted(seen),
            "missing": missing,
            "n_frames_delivered": len(self.cum),
            "n_frames_vischeck_truncated": n_trunc,
            "n_frames_vischeck_error": n_err,
        }


def _fmt(val: Any, dash: str = "-") -> str:
    if val is None or val == "" or val == []:
        return dash
    if isinstance(val, (list, tuple)):
        return ", ".join(str(v) for v in val)
    return str(val)


def render_index_md(blob: dict) -> str:  # noqa: C901 - a linear report writer
    """The per-episode audit trail as one readable document, start to finish."""
    m = blob.get("meta") or {}
    out: list[str] = []
    out.append(f"# IO capture — `{blob.get('tag')}`\n")
    out.append(
        "Every image below is on disk in this directory at the resolution it was sent.\n"
    )
    out.append("| | |")
    out.append("|---|---|")
    for k in (
        "task_id",
        "arm",
        "assistant_backend",
        "assistant_model",
        "human_url",
        "seed",
        "goal_text",
        "seeded_say",
        "prologue_actions",
        "answer_key",
        "tree_fingerprint",
        "frame_config",
    ):
        if k in m:
            out.append(f"| `{k}` | {_fmt(m[k])} |")
    d = blob.get("disk") or {}
    out.append(
        f"| disk | {d.get('n_image_files')} image files, "
        f"{d.get('n_unique_images')} unique, **{d.get('mb_on_disk')} MB** "
        f"({d.get('n_repeats_hardlinked', 0)} repeats hard-linked, "
        f"{d.get('n_repeats_copied', 0)} copied) |"
    )
    s = blob.get("summary") or {}
    if s:
        out.append("\n## Episode verdict (from the MANIFEST)\n")
        out.append("| | |")
        out.append("|---|---|")
        for k, v in s.items():
            out.append(f"| `{k}` | {_fmt(v)} |")
    if blob.get("capture_errors"):
        out.append("\n⚠️ capture errors: " + "; ".join(blob["capture_errors"]) + "\n")

    out.append("\n## Per-turn index\n")
    out.append(
        "| turn | imgs in request | new this turn | answer container in delivered frames | "
        "assistant raw (first 90) | parsed mode | person's command | env result |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for t in blob.get("turns") or []:
        req = (t.get("requests") or [{}])[-1]
        resp = t.get("response") or {}
        av = t.get("answers_visible_cumulative") or {}
        seen = av.get("visible")
        avs = f"{av.get('n_visible')}/{av.get('n_answers')}"
        if seen:
            avs += f" ({', '.join(seen)})"
        raw = (resp.get("raw") or "").replace("\n", " ").replace("|", "\\|")[:90]
        out.append(
            f"| {t.get('turn')} | {req.get('n_images', '-')} | {t.get('n_frames_new', '-')} | "
            f"{avs} | {raw or '(empty)'} | {_fmt(t.get('parsed_mode'))} | "
            f"`{_fmt(t.get('human_command'))}` | {_fmt(t.get('env_result'))[:110]} |"
        )

    out.append("\n## Per-turn detail\n")
    for t in blob.get("turns") or []:
        n = t.get("turn")
        out.append(f"\n### turn {n}\n")
        for req in t.get("requests") or []:
            out.append(
                f"- request: `{req.get('file')}` — backend `{req.get('backend')}`, "
                f"{req.get('n_images')} images, px {_fmt(req.get('px'))}, "
                f"system {req.get('system_chars')} chars, user {req.get('user_chars')} chars"
            )
            imgs = req.get("images") or []
            if imgs:
                out.append(f"  - images in order: {', '.join(f'`{x}`' for x in imgs)}")
        if t.get("frames_new"):
            out.append(
                f"- frames NEW this turn ({t.get('n_frames_new')}): "
                f"{_fmt([f.get('source') for f in t['frames_new']])}"
            )
        av = t.get("answers_visible_cumulative") or {}
        if av:
            out.append(
                f"- `vischeck` over the delivered frames: visible={_fmt(av.get('visible'))} "
                f"missing={_fmt(av.get('missing'))} decidable={av.get('decidable')}"
            )
        avl = t.get("answers_visible_last_step") or {}
        if avl:
            out.append(
                f"- `vischeck` of the newest step alone: visible={_fmt(avl.get('visible'))} "
                f"missing={_fmt(avl.get('missing'))}"
            )
        resp = t.get("response") or {}
        out.append(
            f"- raw reply (`{resp.get('file')}`, {resp.get('n_chars')} chars, "
            f"{resp.get('latency_s')} s):"
        )
        out.append("```")
        out.append((resp.get("raw") or "(empty)").strip())
        out.append("```")
        out.append(
            f"- parsed: mode=`{_fmt(t.get('parsed_mode'))}` "
            f"say={json.dumps(t.get('parsed_say') or '')}"
        )
        out.append(
            f"- heard by the assistant this turn: {json.dumps(t.get('heard') or '')}"
        )
        out.append(
            f"- person: source=`{_fmt(t.get('human_source'))}` "
            f"said={json.dumps(t.get('human_say') or '')} "
            f"command=`{_fmt(t.get('human_command'))}`"
        )
        for human_input in t.get("human_inputs") or []:
            out.append(
                f"- Human model input: {human_input.get('n_images')} images in order: "
                f"{_fmt([f'`{name}`' for name in human_input.get('images') or []])}"
            )
            out.append(
                f"  - system prompt: {json.dumps(human_input.get('system_prompt') or '')}"
            )
            out.append(
                f"  - user prompt: {json.dumps(human_input.get('user_prompt') or '')}"
            )
            out.append("  - raw Human response:")
            out.append("```")
            out.append((human_input.get("raw_response") or "(empty)").strip())
            out.append("```")
        out.append(f"- env result: {_fmt(t.get('env_result'))}")
        if t.get("env_bad"):
            out.append(f"- ⚠️ engine refusals: {_fmt(t.get('env_bad'))}")
        if t.get("score"):
            out.append(f"- score after the step: {_fmt(t.get('score'))}")
    return "\n".join(out) + "\n"


def attach(
    backend: Any, out_dir: str, tag: str, meta: dict | None = None
) -> IOCapture | None:
    """Wire a fresh :class:`IOCapture` onto a served-model backend and its wrapper.

    ``backend`` may be the raw ``AssistantBackend`` (``episode_loop.main``) or a wrapper that
    exposes one as ``.inner``. The recorder goes on the INNER object, because
    that is where the payload is assembled, and a handle goes on the outer one so the tick loop can
    find it without knowing which driver it is running under.
    """
    inner = getattr(backend, "inner", backend)
    if not hasattr(inner, "generate"):
        return None
    cap = IOCapture(out_dir, tag, meta)
    inner.capture = cap
    try:
        backend.io_capture = cap
    except AttributeError:
        pass
    return cap


def _selftest() -> int:
    """Offline: dedup keeps the naming complete, PNG size is read without PIL, index renders."""
    import tempfile
    import zlib

    ok = True

    def chk(cond: bool, why: str) -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {why}")

    def _png(w: int, h: int, fill: int) -> bytes:
        raw = b"".join(b"\x00" + bytes([fill, fill, fill]) * w for _ in range(h))

        def chunk(kind: bytes, body: bytes) -> bytes:
            return (
                len(body).to_bytes(4, "big")
                + kind
                + body
                + zlib.crc32(kind + body).to_bytes(4, "big")
            )

        ihdr = w.to_bytes(4, "big") + h.to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
        return (
            _PNG_SIG
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )

    a, b = _png(7, 5, 10), _png(7, 5, 200)
    chk(png_size(a) == (7, 5), "png_size reads IHDR with no PIL")
    chk(png_size(b"nope") is None, "png_size refuses a non-PNG instead of guessing")

    with tempfile.TemporaryDirectory() as td:
        cap = IOCapture(td, "SELFTEST", {"task_id": "T0", "goal_text": "g"})
        cap.begin_turn(0)
        cap.set_frame_provenance(
            [{"source": "prologue:[Walk]"}, {"source": "prologue:[Grab]"}]
        )
        b64 = base64.b64encode(a).decode()
        imgs = cap.save_images([("b64", b64), ("b64", base64.b64encode(b).decode())])
        chk(
            [i["file"] for i in imgs] == ["turn00_img00.png", "turn00_img01.png"],
            "frame names carry turn AND order",
        )
        chk(
            imgs[0].get("provenance", {}).get("source") == "prologue:[Walk]",
            "a request image carries the command that rendered it",
        )
        cap.record_request("http-json", {"system": "S", "messages": []}, imgs, {"n": 1})
        cap.record_response("$interrupt$hello", 1.25)
        cap.record(
            parsed_mode="interrupt",
            parsed_say="hello",
            human_command="[Walk] <unit_c> (1)",
            env_result="ok",
            n_frames_new=2,
            answers_visible_cumulative={
                "decidable": True,
                "n_answers": 1,
                "n_visible": 1,
                "visible": ["unit_c"],
                "missing": [],
            },
        )
        cap.begin_turn(1)
        cap.set_frame_provenance(
            [
                {"source": "turn1:first"},
                {"source": "turn1:second"},
                {"source": "turn1:third"},
            ]
        )
        imgs2 = cap.save_images(
            [("b64", b64), ("b64", base64.b64encode(b).decode()), ("b64", b64)]
        )
        chk(
            len(imgs2) == 3
            and all(os.path.exists(os.path.join(cap.dir, i["file"])) for i in imgs2),
            "a repeated frame still gets its own turn-and-order filename",
        )
        chk(
            cap.n_unique == 2 and cap.n_files == 5,
            "identical bytes are stored once, linked after",
        )
        cap.record_request("http-json", {"system": "S", "messages": []}, imgs2, {})
        cap.record_response("", 0.5, error="http")
        blob = cap.close({"goal_met": False})
        chk(os.path.exists(os.path.join(cap.dir, "index.json")), "index.json written")
        md = open(os.path.join(cap.dir, "index.md")).read()
        chk(
            "turn00_img00.png" in md and "unit_c" in md,
            "index.md names the frames and the vischeck",
        )
        chk(
            blob["disk"]["n_image_files"] == 5 and "n_repeats_copied" in blob["disk"],
            "disk cost is reported, links and copies apart",
        )
        chk(not cap.errors, f"no capture errors ({cap.errors})")

    print(f"[io_capture] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())
