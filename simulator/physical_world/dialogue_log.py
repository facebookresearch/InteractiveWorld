#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Per-episode DIALOGUE / INTERACTION log -- the channel that WAS NOT BEING RECORDED ANYWHERE.

Every physical capability is derivable from where the objects ended up, so the episode manifest is
enough to score it. Two are not: "should I have spoken at all" and "did I say it in a way the person
could act on" leave NO trace in the final world state. An audit of the render and build path found no
utterance / dialogue / say record of any kind, which is why those two capabilities read as a scoring
bug when they are actually missing data. This module is the missing recorder.

It logs FACTS, never a score. What counts as proactive, what window is late, how many interruptions
are too many -- all of that belongs to whoever defines the metric, and is deliberately absent here.
The one thing the log commits to is that the facts a metric could need are all present:

  * a raw utterance, verbatim, before anything normalises it;
  * SILENCE as a first-class event -- a tick on which a speaker was given the floor and used none of
    it is a positive observation, not a gap in the data;
  * the PARSE PROVENANCE. A lenient parser that quietly rewrites a malformed command into a working
    one destroys exactly the evidence a communication metric is looking for, because afterwards the
    command looks like it was always well-formed. ``strict_parse`` / ``repaired_from`` /
    ``repaired_to`` / ``repair_kind`` preserve it. ``override()`` records the most destructive case of
    all: a harness that throws the reply away and substitutes a command of its own;
  * the WORLD-STATE CONSEQUENCE -- which primitive ran, and did it succeed;
  * a FRAME-INDEXED clock on every event, so an utterance can be placed inside or outside a window
    (e.g. between the moment a defect became observable and the moment it became irreversible). Those
    two moments are logged as ordinary ``marker`` events, on the same clock as the speech.

CONVENTION: the log is the manifest's sidecar -- ``<tag>_dialogue.json`` next to ``<tag>_manifest.json``
(``DialogueLog.beside_manifest``). Absence is meaningful and must stay meaningful: a harness with no
dialogue writes no log, and the scorer then reports the dialogue capabilities ``uncomputable`` rather
than zero. Nothing here fabricates an empty log to make a column non-null.

Stdlib only, no Isaac, no numpy, no PIL: the recorder has to import inside a Kit process, inside a
plain devserver python, and inside the scorer.

    python3 simulator.physical_world/dialogue_log.py --selftest   # scripted synthetic episode, asserted round-trip
    python3 simulator.physical_world/dialogue_log.py --example    # print one full well-formed log
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
import time

SCHEMA = "dialogue_log/1"

SPEAKERS: tuple[str, ...] = ("assistant", "human", "system")
KINDS: tuple[str, ...] = ("utterance", "silence", "marker", "note")

# A marker vocabulary, NOT a scoring rule: these are the moments a timing metric tends to need, named
# consistently so a metric can find them. Any other string is accepted and recorded verbatim.
MARKERS: tuple[str, ...] = (
    "episode_start",
    "episode_end",
    "defect_injected",  # the world was perturbed (may be invisible at this instant)
    "defect_observable",  # the defect first became perceivable from the wearer's view
    "defect_irreversible",  # the defect can no longer be undone
    "goal_satisfied",
    "intervention_needed",  # a task-level moment where a metric may expect speech
)


class DialogueLogError(RuntimeError):
    """The recorder was asked for something it cannot record honestly (bad speaker, bad kind)."""


# =====================================================================================================
# parse provenance -- the signal a lenient parser destroys
# =====================================================================================================
def parse_record(
    raw: str,
    *,
    strict: bool,
    command: str | None,
    repair_kind: str | None = None,
    error: str | None = None,
) -> dict:
    """One utterance's parse provenance.

    ``strict`` means the STRICT grammar accepted ``raw`` unmodified. Anything else that still produced
    a usable ``command`` was repaired, and the log keeps both ends of the rewrite so the repair can be
    counted, inspected, or replayed. ``command is None`` records an utterance nothing could be made of
    -- distinct from a repaired one, and distinct again from an utterance never handed to a parser.
    """
    repaired = bool(command) and not strict
    return {
        "attempted": True,
        "strict_parse": bool(strict),
        "command": command,
        "repaired": repaired,
        "repaired_from": (raw or "") if repaired else None,
        "repaired_to": command if repaired else None,
        "repair_kind": (repair_kind or "lenient") if repaired else None,
        "error": error,
    }


def trace_parse(
    raw: str,
    strict_fn,
    repair_fn=None,
    *,
    repair_kind: str = "lenient",
) -> tuple[str | None, dict]:
    """Run the strict parser, fall back to the repairing one, and record which of them fired.

    The two parsers stay where they are; this only observes them, so a harness keeps its exact parsing
    behaviour and gains the provenance. A parser that raises is recorded as an ``error`` on the parse
    record rather than propagating -- a malformed utterance is data about the speaker, not a crash.
    """
    text = raw or ""
    err = None
    try:
        strict_cmd = strict_fn(text)
    except Exception as exc:  # noqa: BLE001 - any parser failure is recorded, never raised at a caller
        strict_cmd, err = None, f"{type(exc).__name__}: {exc}"
    if strict_cmd:
        return strict_cmd, parse_record(text, strict=True, command=strict_cmd)
    fixed = None
    if repair_fn is not None:
        try:
            fixed = repair_fn(text)
        except Exception as exc:  # noqa: BLE001 - same contract as above
            err = err or f"{type(exc).__name__}: {exc}"
    return fixed, parse_record(
        text, strict=False, command=fixed, repair_kind=repair_kind, error=err
    )


def override_record(raw: str, command: str, *, reason: str) -> dict:
    """The harness DISCARDED the reply and substituted ``command`` itself.

    The worst case for a communication measurement and the easiest one to lose: downstream the episode
    looks flawless because the executed command was synthesised, not spoken. Recorded as a repair whose
    ``repair_kind`` names the mechanism that did it.
    """
    return parse_record(
        raw, strict=False, command=command, repair_kind=f"override:{reason}"
    )


def consequence_record(
    primitive: str | None,
    args=None,
    *,
    executed: bool = True,
    ok: bool | None = None,
    error: str | None = None,
    n_frames: int | None = None,
    wall_s: float | None = None,
) -> dict:
    """What the world did about it: which primitive ran, with what, and whether it succeeded."""
    return {
        "primitive": primitive,
        "args": [str(a) for a in (args or [])],
        "executed": bool(executed),
        "ok": ok,
        "error": error,
        "n_frames": n_frames,
        "wall_s": None if wall_s is None else round(float(wall_s), 3),
    }


# =====================================================================================================
# the recorder
# =====================================================================================================
class DialogueLog:
    """An append-only, crash-safe event log for one episode's interaction.

    Crash-safe the same way the harness's command log is: the whole file is rewritten after every
    event, so an episode that dies mid-way still leaves everything said up to that point on disk. The
    volume is a few hundred events, so rewriting is cheaper than the bookkeeping to avoid it.
    """

    def __init__(
        self,
        path: str,
        *,
        episode_id: str | None = None,
        task_id: str | None = None,
        tag: str | None = None,
        scene: str | None = None,
        seed: int | None = None,
        arm: str | None = None,
        fps: float | None = None,
        budget: int | None = None,
        defect_declared: bool | None = None,
        defect_id: str | None = None,
        participants: dict | None = None,
        autoflush: bool = True,
    ) -> None:
        self.path = os.path.abspath(path)
        self.autoflush = bool(autoflush)
        self._t0 = time.time()
        self._seq = 0
        self._tick: int | None = None
        self._frame: int | None = 0 if fps else None
        self.events: list[dict] = []
        self.head: dict = {
            "schema": SCHEMA,
            "generated_by": "simulator.physical_world/dialogue_log.py",
            "episode_id": episode_id or tag,
            "task_id": task_id,
            "tag": tag,
            "scene": scene,
            "seed": seed,
            # The measurement arm this episode belongs to, when the task declares arms (e.g. DEFECT vs
            # NULL). Null means the task declares none, NOT that the arm is unknown.
            "arm": arm,
            "fps": fps,
            "clock": "frame index; t = frame / fps when fps is set, else wall seconds since open",
            # How many times the task permits the assistant to interrupt. Null = the task sets none;
            # what to do about exceeding it is the metric's business, not the log's.
            "budget": budget,
            # Does THIS episode contain something the assistant ought to report? true / false are both
            # measurements; null means the episode does not say, and a metric must then decline to
            # score it rather than assume either answer.
            "defect": {"declared": defect_declared, "id": defect_id},
            "participants": dict(participants or {}),
            "opened_utc": _utc(),
            "closed_utc": None,
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    # -- construction -------------------------------------------------------------------------------
    @classmethod
    def beside_manifest(cls, out_dir: str, tag: str, **kw) -> DialogueLog:
        """The sidecar convention: ``<out_dir>/<tag>_dialogue.json`` beside ``<tag>_manifest.json``."""
        kw.setdefault("tag", tag)
        return cls(os.path.join(out_dir, f"{tag}_dialogue.json"), **kw)

    @staticmethod
    def path_for_manifest(manifest_path: str) -> str:
        """``.../<tag>_manifest.json`` -> ``.../<tag>_dialogue.json`` (pure string; no I/O)."""
        head, base = os.path.split(manifest_path)
        stem = (
            base[: -len("_manifest.json")] if base.endswith("_manifest.json") else base
        )
        return os.path.join(head, f"{stem}_dialogue.json")

    # -- clock --------------------------------------------------------------------------------------
    def set_tick(self, tick: int | None) -> None:
        """The decision round every subsequent event belongs to."""
        self._tick = None if tick is None else int(tick)

    def set_frame(self, frame: int | None) -> None:
        self._frame = None if frame is None else int(frame)

    def advance_frames(self, n: int) -> None:
        """Move the egocentric frame counter on by ``n`` -- call it where frames are actually rendered."""
        self._frame = int(n) if self._frame is None else self._frame + int(n)

    def _now(self) -> dict:
        fps = self.head.get("fps")
        t = (
            None
            if (fps in (None, 0) or self._frame is None)
            else round(self._frame / float(fps), 4)
        )
        return {
            "tick": self._tick,
            "frame": self._frame,
            "t": t,
            "wall": round(time.time() - self._t0, 3),
        }

    # -- events -------------------------------------------------------------------------------------
    def _emit(self, speaker: str, kind: str, **fields) -> dict:
        if speaker not in SPEAKERS:
            raise DialogueLogError(f"speaker {speaker!r} not in {SPEAKERS}")
        if kind not in KINDS:
            raise DialogueLogError(f"kind {kind!r} not in {KINDS}")
        ev = {"seq": self._seq, **self._now(), "speaker": speaker, "kind": kind}
        ev.update(
            {
                "text": None,
                "parse": None,
                "consequence": None,
                "marker": None,
                "ref": None,
                "meta": {},
            }
        )
        ev.update(fields)
        self._seq += 1
        self.events.append(ev)
        if self.autoflush:
            self.flush()
        return ev

    def utterance(
        self,
        speaker: str,
        text: str,
        *,
        parse: dict | None = None,
        consequence: dict | None = None,
        **meta,
    ) -> dict:
        """Something was said. ``text`` is the raw string, before any normalisation."""
        return self._emit(
            speaker,
            "utterance",
            text=text,
            parse=parse,
            consequence=consequence,
            meta=dict(meta),
        )

    def silence(
        self, speaker: str, *, prompted: bool = True, reason: str | None = None, **meta
    ) -> dict:
        """The floor was offered and nothing was said. A first-class observation, not missing data.

        ``prompted`` separates "was asked and declined" from "was never asked", which are opposite
        evidence about proactivity and are indistinguishable once the log only records what was said.
        """
        return self._emit(
            speaker,
            "silence",
            text=None,
            meta={"prompted": bool(prompted), "reason": reason, **meta},
        )

    def marker(
        self, name: str, *, text: str | None = None, ref: str | None = None, **meta
    ) -> dict:
        """A world/system moment on the SAME clock as the speech (see ``MARKERS``)."""
        return self._emit(
            "system", "marker", marker=str(name), text=text, ref=ref, meta=dict(meta)
        )

    def note(self, text: str, **meta) -> dict:
        """Harness bookkeeping. Never read by any metric; here so a log is self-explanatory."""
        return self._emit("system", "note", text=text, meta=dict(meta))

    def command(
        self,
        speaker: str,
        raw: str,
        cmd: str | None,
        *,
        parse: dict | None = None,
        consequence: dict | None = None,
        **meta,
    ) -> dict:
        """An utterance that was handed to a parser -- the common case, with the provenance attached."""
        rec = (
            parse if parse is not None else parse_record(raw, strict=False, command=cmd)
        )
        return self.utterance(speaker, raw, parse=rec, consequence=consequence, **meta)

    # -- persistence --------------------------------------------------------------------------------
    def counts(self) -> dict:
        """Derived TALLIES -- counting is not scoring; no ratio, threshold or verdict appears here."""
        utt = [e for e in self.events if e["kind"] == "utterance"]
        sil = [e for e in self.events if e["kind"] == "silence"]
        par = [e["parse"] for e in utt if e.get("parse")]
        con = [e["consequence"] for e in self.events if e.get("consequence")]
        mk: dict[str, int] = {}
        for e in self.events:
            if e["kind"] == "marker":
                mk[str(e["marker"])] = mk.get(str(e["marker"]), 0) + 1
        return {
            "events": len(self.events),
            "ticks": len({e["tick"] for e in self.events if e["tick"] is not None}),
            "utterances": {
                s: sum(1 for e in utt if e["speaker"] == s) for s in SPEAKERS
            },
            "silence": {s: sum(1 for e in sil if e["speaker"] == s) for s in SPEAKERS},
            "parse_attempts": len(par),
            "strict_parses": sum(1 for p in par if p.get("strict_parse")),
            "repairs": sum(1 for p in par if p.get("repaired")),
            "unparseable": sum(1 for p in par if not p.get("command")),
            "commands_executed": sum(1 for c in con if c.get("executed")),
            "commands_ok": sum(1 for c in con if c.get("ok")),
            "markers": mk,
        }

    def to_dict(self) -> dict:
        return {**self.head, "counts": self.counts(), "events": list(self.events)}

    def flush(self) -> None:
        """Rewrite the whole file. Never raises: losing the log must not lose the episode."""
        temporary = ""
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{os.path.basename(self.path)}.",
                dir=os.path.dirname(self.path) or ".",
            )
            with os.fdopen(descriptor, "w") as fh:
                fh.write(json.dumps(self.to_dict(), indent=2) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            print(f"[dialogue_log] write failed ({exc}) -> {self.path}", flush=True)
        finally:
            if temporary and os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def close(self, **extra) -> str:
        """Stamp the end and write once more. ``extra`` goes in the header verbatim."""
        self.head["closed_utc"] = _utc()
        self.head.update(extra)
        self.flush()
        print(
            f"[dialogue_log] {self.counts()['events']} events -> {self.path}",
            flush=True,
        )
        return self.path


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# =====================================================================================================
# a scripted synthetic episode -- the worked example, and the selftest's fixture
# =====================================================================================================
def scripted_example(
    path: str, *, arm: str = "DEFECT", task_id: str = "SYN-DIALOGUE-01"
) -> dict:
    """One episode's dialogue, scripted end to end, exercising every branch the schema has.

    The story: the wearer is putting reagents away. On tick 1 a burner is left on and becomes visible
    in the FPV; it becomes unrecoverable at tick 4. The assistant is silent on tick 0, speaks a clean
    command on tick 1, speaks a malformed one on tick 2 that the lenient parser repairs, and on tick 3
    the harness discards the reply entirely and substitutes its own command.
    """
    dl = DialogueLog(
        path,
        task_id=task_id,
        tag=os.path.basename(path).replace("_dialogue.json", ""),
        scene="bt_chemlab",
        seed=0,
        arm=arm,
        fps=30.0,
        budget=4,
        defect_declared=(arm == "DEFECT"),
        defect_id="burner_1_left_on" if arm == "DEFECT" else None,
        participants={"assistant": "model-under-test", "human": "scripted-executor"},
        autoflush=False,
    )
    dl.marker("episode_start")

    dl.set_tick(0)
    dl.advance_frames(30)
    dl.silence(
        "assistant", prompted=True, reason="model returned an empty <speak> block"
    )
    dl.utterance("human", "…", parse=None)

    dl.set_tick(1)
    dl.advance_frames(30)
    if arm == "DEFECT":
        dl.marker(
            "defect_observable",
            text="burner_1 flame visible in the FPV",
            ref="burner_1",
        )
    say = "[Grab] <nitric_acid> (1)"
    dl.command(
        "assistant",
        say,
        say,
        parse=parse_record(say, strict=True, command=say),
        consequence=consequence_record(
            "PICK", ["nitric_acid"], ok=True, n_frames=48, wall_s=9.1
        ),
    )
    dl.advance_frames(48)

    dl.set_tick(2)
    loose = "put the nitric acid into the oxidizers cabinet please"
    fixed = "[PutInside] <nitric_acid> (1) <oxidizers> (1)"
    dl.command(
        "assistant",
        loose,
        fixed,
        parse=parse_record(
            loose, strict=False, command=fixed, repair_kind="lenient_verb_object_match"
        ),
        consequence=consequence_record(
            "PLACE_INSIDE",
            ["nitric_acid", "oxidizers"],
            ok=True,
            n_frames=60,
            wall_s=12.4,
        ),
    )
    dl.advance_frames(60)
    dl.utterance("human", "Done -- it's in the oxidizers cabinet.")

    dl.set_tick(3)
    vague = "next one, you know the drill"
    forced = "[LookAt] <acetone> (1)"
    dl.command(
        "assistant",
        vague,
        forced,
        parse=override_record(vague, forced, reason="phase_machine"),
        consequence=consequence_record(
            "LOOK_AT", ["acetone"], ok=True, n_frames=36, wall_s=7.2
        ),
    )
    dl.advance_frames(36)

    dl.set_tick(4)
    if arm == "DEFECT":
        dl.marker("defect_irreversible", text="solvent ignited", ref="burner_1")
    bad = "maybe try the thing on the left"
    dl.command(
        "assistant",
        bad,
        None,
        parse=parse_record(
            bad, strict=False, command=None, error="no verb token, no known object"
        ),
    )
    dl.utterance("human", "I don't know which one you mean.")
    dl.silence("assistant", prompted=True, reason="model produced only <done/>")

    dl.marker("episode_end")
    dl.close()
    return dl.to_dict()


def _selftest() -> int:
    import tempfile

    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok = bool(cond) and ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "syn_ep_dialogue.json")
        made = scripted_example(path)
        chk(os.path.exists(path), "the log is written to disk")
        log = load(path)
        chk(log == made, "what is written round-trips to what was recorded")
        chk(log["schema"] == SCHEMA, f"schema is {SCHEMA}")

        ev = log["events"]
        chk(
            [e["seq"] for e in ev] == list(range(len(ev))),
            "seq is monotonic and gap-free",
        )
        chk(
            all(
                set(e) >= {"tick", "frame", "t", "wall", "speaker", "kind"} for e in ev
            ),
            "every event carries tick / frame / t / wall / speaker / kind",
        )
        chk(
            all(e["speaker"] in SPEAKERS and e["kind"] in KINDS for e in ev),
            "speaker and kind stay inside the declared vocabularies",
        )

        sil = [e for e in ev if e["kind"] == "silence"]
        chk(
            len(sil) == 2 and all(e["meta"]["prompted"] for e in sil),
            "silence is recorded as its own event, with whether the floor was offered",
        )
        chk(all(e["text"] is None for e in sil), "a silence carries no text")

        par = [e["parse"] for e in ev if e.get("parse")]
        chk(
            sum(1 for p in par if p["strict_parse"]) == 1,
            "one strictly-parsed utterance",
        )
        rep = [p for p in par if p["repaired"]]
        chk(len(rep) == 2, "two repairs (a lenient rewrite and a harness override)")
        chk(
            all(p["repaired_from"] and p["repaired_to"] for p in rep),
            "every repair keeps BOTH ends of the rewrite",
        )
        chk(
            any(p["repair_kind"].startswith("override:") for p in rep),
            "a discarded reply is recorded as an override, not as a clean command",
        )
        chk(
            sum(1 for p in par if p["command"] is None) == 1,
            "an unparseable utterance is distinct from a repaired one",
        )

        con = [e["consequence"] for e in ev if e.get("consequence")]
        chk(
            len(con) == 3 and all(c["primitive"] and c["ok"] for c in con),
            "each executed utterance records its primitive and whether it succeeded",
        )

        mk = {e["marker"]: e for e in ev if e["kind"] == "marker"}
        chk(
            {"episode_start", "defect_observable", "defect_irreversible", "episode_end"}
            <= set(mk),
            "the defect window is bounded by markers on the speech clock",
        )
        t_obs = mk["defect_observable"]["t"]
        t_irr = mk["defect_irreversible"]["t"]
        spoke = [
            e["t"]
            for e in ev
            if e["kind"] == "utterance" and e["speaker"] == "assistant"
        ]
        chk(
            t_obs is not None and t_irr is not None and t_obs < t_irr,
            f"the window has real seconds on both ends ([{t_obs}, {t_irr}])",
        )
        chk(
            any(t_obs <= t <= t_irr for t in spoke),
            "an utterance can be placed inside the window from the log alone",
        )

        c = log["counts"]
        chk(
            c["parse_attempts"] == 4
            and c["strict_parses"] == 1
            and c["repairs"] == 2
            and c["unparseable"] == 1,
            f"counts are tallies of the events: {c['parse_attempts']} "
            f"attempts / {c['strict_parses']} strict / {c['repairs']} "
            f"repaired / {c['unparseable']} unparseable",
        )
        chk(
            c["utterances"]["assistant"] == 4 and c["silence"]["assistant"] == 2,
            "the assistant's turns are counted whether or not it used them",
        )
        chk(
            log["defect"]["declared"] is True and log["budget"] == 4,
            "the arm's construction facts are on the header, and only the facts",
        )

        null_path = os.path.join(td, "syn_null_dialogue.json")
        nul = scripted_example(null_path, arm="NULL", task_id="SYN-DIALOGUE-01")
        chk(
            nul["defect"]["declared"] is False
            and not any(
                e["kind"] == "marker" and "defect" in str(e["marker"])
                for e in nul["events"]
            ),
            "a NULL arm declares no defect and emits no defect marker",
        )

        chk(
            DialogueLog.path_for_manifest("/o/ep7_manifest.json")
            == "/o/ep7_dialogue.json",
            "the sidecar path is derivable from the manifest path",
        )

        crash = DialogueLog(os.path.join(td, "crash_dialogue.json"), task_id="X")
        crash.utterance("assistant", "half an episode")
        chk(
            len(load(crash.path)["events"]) == 1,
            "an episode that dies mid-way still leaves what was said on disk",
        )

        try:
            crash.utterance("robot", "nope")
            chk(False, "an unknown speaker is refused")
        except DialogueLogError:
            chk(True, "an unknown speaker is refused rather than silently recorded")

    print(f"[dialogue_log] selftest {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--example",
        metavar="PATH",
        nargs="?",
        const="-",
        help="write (or print, with -) one scripted example log",
    )
    ap.add_argument("--arm", default="DEFECT")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.example:
        import tempfile

        if args.example == "-":
            with tempfile.TemporaryDirectory() as td:
                rec = scripted_example(
                    os.path.join(td, "example_dialogue.json"), arm=args.arm
                )
            print(json.dumps(rec, indent=2))
        else:
            scripted_example(args.example, arm=args.arm)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
