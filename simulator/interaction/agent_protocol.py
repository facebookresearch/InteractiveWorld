"""Assistant/Human dialogue protocol used by released TaskPack evaluation."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from .interaction_profiles import (
    ASK_HUMAN_PER_TURN_GATE,
    assistant_system as interaction_assistant_system,
    human_system as interaction_human_system,
)
from .prompts import assistant_goal, assistant_system, human_system
from .taskpack_interaction import enforce_human_turn_policy

_INTERACTION_DIR = os.path.dirname(os.path.abspath(__file__))
_SIMULATOR_DIR = os.path.dirname(_INTERACTION_DIR)
_GRAMMAR_DIR = os.path.join(_SIMULATOR_DIR, "physical_world")


_ASK_HUMAN_REPROMPT_GUARDS = frozenset(
    {
        "ask_handle_map_missing",
        "ask_flip_requires_held",
        "ask_hands_full_grab_blocked",
        "ask_inspection_grab_blocked",
        "ask_internal_interface_speech_blocked",
        "ask_multiple_commands_blocked",
        "ask_premature_completion_speech_blocked",
        "ask_redundant_walk_blocked",
        "ask_unknown_human_handle_blocked",
        "ask_unrequested_action_blocked",
        "ask_unseen_grab_blocked",
        "ask_unseen_target_blocked",
    }
)


_GUIDE_HUMAN_REPROMPT_GUARDS = frozenset(
    {
        "ask_internal_interface_speech_blocked",
        "ask_premature_completion_speech_blocked",
        "ask_unobservable_attribute_speech_blocked",
        "guide_active_instruction_mute_blocked",
        "guide_ambiguous_reference_blocked",
        "guide_failed_placement_claim_blocked",
        "guide_incomplete_instruction_completion_claim_blocked",
        "guide_incomplete_instruction_next_step_blocked",
        "guide_inference_conclusion_speech_blocked",
        "guide_instruction_mismatch_blocked",
        "guide_movable_source_conflict_blocked",
        "guide_required_inspection_missing",
        "guide_unknown_human_handle_blocked",
        "guide_unseen_movable_blocked",
        "guide_unverified_progress_speech_blocked",
        "guide_visual_evidence_speech_blocked",
        "human_motor_claim_without_action",
    }
)


_WATCH_HUMAN_REPROMPT_GUARDS = frozenset(
    {
        "human_motor_claim_without_action",
        "watch_assistant_instruction_ignored_blocked",
        "watch_assistant_state_override_blocked",
    }
)


_NARRATE_HUMAN_REPROMPT_GUARDS = frozenset()


_HUMAN_MECHANICAL_REPROMPT_GUARDS = frozenset(
    {
        "human_action_format_invalid",
        "human_action_unsupported_verb",
        "human_action_wrong_arity",
        "narrate_handle_map_missing",
        "narrate_unknown_human_handle_blocked",
    }
)


_ORCHESTRATE_HUMAN_REPROMPT_GUARDS = frozenset(
    {
        "ask_premature_completion_speech_blocked",
        "human_motor_claim_without_action",
        "orchestrate_action_target_mismatch_blocked",
        "orchestrate_completed_action_repeat_blocked",
        "orchestrate_unrequested_action_blocked",
    }
)


def human_reprompt_guard_enabled(
    interaction_family: str,
    guard: str,
    *,
    human_object_interface: str = "full_ids",
) -> bool:
    family = interaction_family.strip().upper()
    if family == "NARRATE" and guard in _HUMAN_MECHANICAL_REPROMPT_GUARDS:
        return True
    if family == "GUIDE":
        return guard in _GUIDE_HUMAN_REPROMPT_GUARDS
    if family == "WATCH":
        return guard in _WATCH_HUMAN_REPROMPT_GUARDS
    if family == "NARRATE":
        return guard in _NARRATE_HUMAN_REPROMPT_GUARDS
    if family == "ORCHESTRATE":
        return guard in _ORCHESTRATE_HUMAN_REPROMPT_GUARDS
    return bool(
        family == "ASK"
        and human_object_interface == "full_ids"
        and guard in _ASK_HUMAN_REPROMPT_GUARDS
    )


_HUMAN_AUDIT_ONLY_GUARDS = frozenset(
    {
        "orchestrate_action_order_unverified",
        "watch_assistant_command_mismatch_blocked",
        "watch_assistant_command_alignment_needs_causal_review",
        "watch_assistant_command_alignment_unverified",
    }
)


def observation_frames(obs: dict) -> list:
    """Return the observation's temporal frame segment, with the final-frame fallback for old servers."""
    return list(obs.get("frames") or ([obs["frame"]] if obs.get("frame") else []))


_ZERO_ARG_VERBS_SEED = frozenset(
    {
        "DROP",
        "RELEASE",
        "STANDUP",
        "LOOK_AROUND",
        "LOOKAROUND",
        "LOOK_SCAN",
        "LOOKSCAN",
        "SQUAT",
        "SQUATDOWN",
    }
)


_VERB_DISPLAY: dict[str, str] = {
    "PUTINSIDE": "PutInside",
    "PUTIN": "PutInside",
    "PUTBACK": "PutBack",
    "PUTOBJBACK": "PutObjBack",
    "PUTON": "PutOn",
    "STANDUP": "StandUp",
    "STANDUPRIGHT": "StandUpright",
    "LOOKAROUND": "LookAround",
    "LOOKSCAN": "LookScan",
    "LOOKAT": "LookAt",
    "OPENDOOR": "OpenDoor",
    "CLOSEDOOR": "CloseDoor",
    "TURNLEFT": "TurnLeft",
    "TURNRIGHT": "TurnRight",
    "TURNAROUND": "TurnAround",
    "WALKTOWARDS": "WalkTowards",
    "WALKFORWARD": "WalkForward",
    "SQUATDOWN": "SquatDown",
}


_VERB_ALIASES: dict[str, str] = {
    "pick up": "GRAB",
    "pick": "GRAB",
    "place": "PUTINSIDE",
    "look down": "LOOK_SCAN",
    "scan the floor": "LOOK_SCAN",
    "stand up": "STANDUP",
    "sit down": "SIT",
    "go forward": "FORWARD",
    "step forward": "FORWARD",
}


_GRAMMAR_CACHE: dict[str, Any] = {}


def _grammar_tables() -> dict[str, Any]:
    """Build the lenient tables once from the physical-world command grammar.

    ``words``  lowercase verb word (grammar token, its space spelling, and the English aliases) ->
               canonical bracket spelling.
    ``arity``  canonical spelling -> preferred object-argument count.
    ``arities`` canonical spelling -> every accepted object-argument count.
    ``loco``   the canonical spellings that take an optional SCALAR instead of objects.
    ``zero``   upper-case tokens legitimately written as a bare ``[Verb]``.

    command_grammar is stdlib-only (it pulls in goal_schema and nothing else), so importing it costs
    nothing. If it is genuinely absent the tables degrade to the walk/grab/put core rather than taking
    the episode down with them -- loudly, because a silent degrade would hide the missing dependency.
    """
    if _GRAMMAR_CACHE:
        return _GRAMMAR_CACHE
    words: dict[str, str] = {}
    arity: dict[str, int] = {}
    arities: dict[str, frozenset[int]] = {}
    loco: set[str] = set()
    zero: set[str] = set(_ZERO_ARG_VERBS_SEED)
    verb_map: dict = {}
    arity_map: dict = {}
    arities_map: dict = {}
    loco_prims: frozenset = frozenset()
    try:
        if _GRAMMAR_DIR not in sys.path:
            sys.path.insert(0, _GRAMMAR_DIR)
        import command_grammar as _cg  # noqa: PLC0415 - optional sibling, imported once and cached

        verb_map = dict(getattr(_cg, "_VERB", {}) or {})
        arity_map = dict(getattr(_cg, "_ARITY", {}) or {})
        arities_map = dict(getattr(_cg, "_ARITIES", {}) or {})
        loco_prims = frozenset(getattr(_cg, "LOCOMOTION_PRIMS", ()) or ())
    except Exception as exc:  # noqa: BLE001 - see the docstring: degrade loudly, never abort
        print(
            f"[isaac_human] command_grammar not importable from {_GRAMMAR_DIR!r} "
            f"({type(exc).__name__}: {str(exc)[:120]}); the lenient verb table is DEGRADED to "
            f"walk/grab/put and legal commands will be discarded again",
            flush=True,
        )
    for token, prim in verb_map.items():
        canon = _VERB_DISPLAY.get(token) or "".join(
            p.capitalize() for p in token.split("_")
        )
        n_args = arity_map.get(prim)
        if (
            n_args is None
        ):  # a verb the grammar knows but declares no arity for -- not repairable
            continue
        arity[canon] = int(n_args)
        arities[canon] = frozenset(
            int(value) for value in arities_map.get(prim, (n_args,))
        )
        words[token.lower()] = canon
        words[token.lower().replace("_", " ")] = canon
        words[token.lower().replace("_", "")] = canon
        if prim in loco_prims:
            loco.add(canon)
        elif int(n_args) == 0:
            # bare `[Verb]` is legal for these; NOT for locomotion, whose bare form would silently
            # discard the scalar ([Forward] 3 -> [Forward], three paces collapsed to one).
            zero.update({token, token.replace("_", "")})
    if not words:  # degraded path: the pre-fix core verbatim, so behaviour is never WORSE than before
        words = {
            "walktowards": "Walk",
            "walk": "Walk",
            "grab": "Grab",
            "putinside": "PutInside",
            "putin": "PutInside",
            "put": "Put",
        }
        arity = {"Walk": 1, "Grab": 1, "Put": 2, "PutInside": 2}
        arities = {name: frozenset({count}) for name, count in arity.items()}
    for alias, token in _VERB_ALIASES.items():
        canon = words.get(token.lower())
        if canon:
            words[alias] = canon
    _GRAMMAR_CACHE.update(
        {
            "words": words,
            "arity": arity,
            "arities": arities,
            "loco": frozenset(loco),
            "zero": frozenset(zero),
            "derived": bool(verb_map),
            # longest first, so `putinside` wins over `put` and `look at` over `look` at the same
            # offset; \b so `sit` does not fire inside "opposite" and `close` not inside "closest".
            "re": re.compile(
                r"\b("
                + "|".join(
                    re.escape(w) for w in sorted(words, key=len, reverse=True) if w
                )
                + r")\b"
            ),
        }
    )
    return _GRAMMAR_CACHE


def human_action_mechanical_guard(
    action_scope: str,
    commands: Sequence[str],
    *,
    held_object_keys: Sequence[str] = (),
) -> str:
    """Return a retry reason only for mechanically invalid non-empty action output."""
    if not action_scope.strip():
        return ""

    tables = _grammar_tables()
    bracketed_verbs = re.findall(r"\[([A-Za-z_]+)\]", action_scope)
    canonical_verbs = [tables["words"].get(verb.lower()) for verb in bracketed_verbs]
    if any(verb is None for verb in canonical_verbs):
        return "human_action_unsupported_verb"

    if not commands:
        if canonical_verbs:
            return "human_action_wrong_arity"
        return "human_action_format_invalid"

    held = {str(value) for value in held_object_keys}
    for command in commands:
        match = re.match(r"^\s*\[([A-Za-z_]+)\]", command)
        if match is None:
            return "human_action_format_invalid"
        canonical = tables["words"].get(match.group(1).lower())
        if canonical is None:
            return "human_action_unsupported_verb"
        if canonical in tables["loco"]:
            continue
        arguments = re.findall(r"<([^>]+)>\s*\((-?\d+)\)", command)
        allowed_arities = tables["arities"].get(canonical, frozenset())
        if len(arguments) in allowed_arities:
            continue
        if (
            canonical.startswith("Put")
            and len(arguments) == 1
            and len(held) == 1
            and arguments[0][0] not in held
        ):
            continue
        return "human_action_wrong_arity"
    return ""


def _arrival_clause(no_motion: dict | None) -> str:
    """ "You did NOT move -- you are ALREADY standing at <x>." for a walk that went nowhere.

    ``isaac_env.step`` sets ``obs["walk_no_motion"]`` when a program that is NOTHING BUT navigation left
    the body where it started. Kept as a free function so ``bench_eval_loop`` and the trainer compose
    the identical sentence -- two copies of this string would drift, and the person's feedback channel
    is the one place a drift is invisible until it costs a whole sweep."""
    keys = [str(k) for k in ((no_motion or {}).get("keys") or [])]
    if not keys:
        return ""
    where = ", ".join(f"<{k}>" for k in keys)
    return (
        f" You did NOT move -- you are ALREADY standing at {where}. Do not walk there again; "
        f"you are close enough to act on it now."
    )


_OBJ_ID_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*_(?:\d+|[a-z]{6})\b")


_LOOSE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")


_LOOSE_KEY_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")


def _redact_keys_loose() -> bool:
    return (os.getenv("ISAAC_REDACT_KEYS_LOOSE") or "0").strip().lower() in (
        "1",
        "true",
        "on",
    )


_GENERIC_SAY = (
    "what should i do",
    "what do i do",
    "what next",
    "whats next",
    "what now",
    "what should i do next",
    "tell me what to do",
    "waiting for instructions",
    "any instructions",
    "i need instructions",
    "what would you like me to do",
)


_SPATIAL_SAY = (
    "next to",
    "by the",
    "on the",
    "under the",
    "behind the",
    "in front of",
    "left of",
    "right of",
    "on your left",
    "on your right",
    "in the corner",
    "over there",
    "across the",
    "against the",
    "beside",
)


def _b64_to_pil(b64: str) -> Image.Image:
    """Decode one base64 PNG from the simulator into a PIL RGB image."""
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


class IsaacAgentLoop:
    def _timing_reset(self) -> None:
        """Reset per-episode timing, counters, and command provenance."""
        self._t = {"t_gen_s": 0.0, "t_human_s": 0.0, "t_step_s": 0.0}
        self._n = {
            key: 0
            for key in (
                "ticks",
                "env_steps",
                "human_calls",
                "human_none",
                "gen_tokens",
                "gen_tokens_max",
                "turn_cap_hits",
                "human_multi_action",
                "queue_played",
                "queue_interrupted",
                "queue_clipped",
                "human_said_generic",
                "say_redactions",
                "say_spatial",
                "vis_steps",
                "vis_n_sum",
                "cmd_vis_checked",
                "cmd_unseen",
                "idle_textonly_turns",
                "human_asked",
                "human_mute_idle",
                "human_spoke",
                "human_policy_beats",
                "human_policy_transitions",
                "human_policy_waits",
                "semantic_verifier_calls",
            )
        }
        self._last_vischeck = None
        self._repairs = []
        self._tk = {"gen_tokens": [], "t_gen_s": [], "t_human_s": [], "t_step_s": []}
        self._stop_reason = "unset"
        self._human_policy_state = {
            "status": "LEGACY",
            "code": "",
            "path": "",
            "message": "",
        }
        self._t0_episode = time.perf_counter()

    def _tadd(self, key: str, t0: float) -> float:
        """Fold one elapsed interval into bucket ``key``; returns the interval (for the per-tick vectors)."""
        dt = time.perf_counter() - t0
        self._t[key] = self._t.get(key, 0.0) + dt
        return dt

    def _assistant_messages(
        self,
        obs: dict,
        first: bool,
        frames: list | None = None,
        heard: str = "",
    ) -> list[dict]:
        """Build this tick's assistant chat turn from the PIXEL observation: this command's (downscaled) FPV
        frames (obs["frames"], 1..n uniformly sampled across the motion = VH extract_frames) as images + a
        terse text state (task + what's held + the goal checklist). Append-only trajectory (prior turns' frames
        stay frozen) uses only the newest interval while preserving multimodal sequence positions. The
        checklist is the Isaac analogue of VH's remaining_goal_hint -- it is WHY the assistant can name a
        concrete next step instead of rambling."""
        content: list[dict] = []
        interaction_family = str(obs.get("interaction_family") or "").strip()
        if first and interaction_family:
            opening_say = str(obs.get("interaction_opening_say") or "").strip()
            if opening_say:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Before the setup actions, the person says: "
                            f'"{opening_say}"'
                        ),
                    }
                )
        # Prefer the buffer's per_interval-sampled DELTA (so ONE per_interval knob governs both agents, as
        # VH does); fall back to the raw obs frames if not passed.
        frames = (
            frames
            if frames is not None
            else (obs.get("frames") or ([obs["frame"]] if obs.get("frame") else []))
        )
        for fb in frames:
            content.append({"type": "image", "image": self._frame_image(fb)})
        # A user turn MUST carry text. Stripping the per-turn line left `content` as images only, and a
        # chat model handed a wordless user turn returns nothing -- run p3 came back 48/48 "silence"
        # purely because every generation was empty, not because the policy chose to stay quiet. Keep it
        # to the bare segment marker: it names what just arrived without hinting at an answer, which is
        # what the old "Decide: silence, interrupt, or errordetection?" line did wrong.
        # What the PERSON said out loud since the last turn. This is the only channel from them to the
        # assistant -- it cannot see their action, only the video and this. It was never wired up, which
        # made the new "ask when stuck" behaviour useless: run p3 has the person saying "what should I
        # do?" on 37 of 48 turns into a void, and the assistant staying silent every single time because
        # the question never reached it.
        if heard:
            prefix = (
                "After the setup actions, the person says"
                if first and interaction_family
                else "The person says"
            )
            content.append({"type": "text", "text": f'{prefix}: "{heard}"'})
        content.append({"type": "text", "text": "Latest segment."})
        task = self._humanize_task(obs.get("task", ""))
        # NO checklist, NO object names, NO ids. The assistant must read the specifics off the video --
        # that is the whole redesign. Handing it the checklist is what made every evaluation sample identical.
        # Turn 0 sends the system prompt (role + goal + output contract) and the first frames; later
        # turns carry ONLY the new frames. There is deliberately no per-turn "decide now" line -- the
        # standing instruction already says to answer after each segment, and a reminder repeated
        # every tick is noise the model would start keying on. Its own prior lines are already in the
        # append-only trajectory, so the dialogue history needs no re-sending.
        if first:
            system = (
                interaction_assistant_system(interaction_family)
                if interaction_family
                else assistant_system(assistant_goal(task, obs.get("goal") or []))
            )
            if interaction_family.upper() == "ASK":
                task_context = str(obs.get("interaction_task_context") or "").strip()
                if task_context:
                    system = f"{system}\n\nTask context:\n{task_context}"
            return [
                # The ASSISTANT gets the task stated the way a user would ("put all the cans of soda
                # into the trash can"); the HUMAN keeps only the vague activity. That asymmetry is the
                # point: with both vague, the assistant had to guess what counted as trash and sent the
                # person after a microwave; with both specific, every evaluation sample reads the same answer
                # off its prompt and the advantage collapses. Counts and instance ids stay out of it --
                # those are what the video is for.
                {
                    "role": "system",
                    "content": system,
                },
                {"role": "user", "content": content},
            ]
        return [{"role": "user", "content": content}]

    def _parse_assistant(self, text: str) -> dict:
        """Map the raw output onto {mode, say}, mirroring EgoProactive's ``_parse_action``.

        Same token contract as the benchmark (``$silent$`` / ``$interrupt$<text>``) plus ``$error$`` for
        the correction branch, which the benchmark folds into a plain interrupt. Keeping the extra token
        is deliberate: without a distinct label an interrupt that lands on a mistake is indistinguishable
        from a lucky next-step tip, so the evaluation could not tell whether the model detects errors.
        The token maps straight back onto ``interrupt`` for benchmark scoring.

        Untokenised output is treated as an INTERRUPT carrying the whole text, exactly as the benchmark
        does -- not as silence. Treating it as silence would grant a leniency the test set does not.
        """
        t = (text or "").strip()
        low = t.lower()
        for tok, mode in (("$error$", "errordetection"), ("$interrupt$", "interrupt")):
            k = low.find(tok)
            if k >= 0:
                self._n[f"mode_{mode}"] = self._n.get(f"mode_{mode}", 0) + 1
                self._last_assistant_raw = t[:200]
                return {"mode": mode, "say": t[k + len(tok) :].strip(), "done": False}
        self._last_assistant_raw = t[:200]
        if "$silent$" in low and len(low.replace("$silent$", "").strip()) < 5:
            self._n["mode_silence"] = self._n.get("mode_silence", 0) + 1
            return {"mode": "silence", "say": "", "done": False}
        if len(t) < 2:
            # EMPTY, not a chosen silence. Counted apart so a run of these is visible as the generation
            # failure it is -- p3 reported 48/48 "silence" that was really 48 empty completions.
            self._n["assistant_empty"] = self._n.get("assistant_empty", 0) + 1
            return {"mode": "silence", "say": "", "done": False}
        self._n["mode_untokenised"] = self._n.get("mode_untokenised", 0) + 1
        return {"mode": "interrupt", "say": t, "done": False}

    def _frame_image(self, b64: str) -> Image.Image:
        """Decode the FPV frame + downscale to ``frame_max_side`` so it expands into fewer image-pad tokens
        (more ticks fit the response_length budget; cheaper to process). No-op if already small / cap off.
        """
        img = _b64_to_pil(b64)
        m = self.frame_max_side
        if m and max(img.size) > m:
            w, h = img.size
            s = m / float(max(w, h))
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))))
        return img

    @staticmethod
    def _humanize_task(task: str) -> str:
        """Scene key -> a readable goal phrase for the prompts (e.g. "bt_picking_up_trash" -> "picking up
        trash"). The structural goal lives in the checklist; this is just a friendly title.
        """
        t = (task or "").strip()
        for pre in ("bt_", "behavior_"):
            if t.startswith(pre):
                t = t[len(pre) :]
        return t.replace("_", " ").strip() or "complete the task"

    @staticmethod
    def _grammar_generic(put_verb: str) -> str:
        """A GOAL-FREE grammar template (placeholder names only, never drawn from the todo) so the executor
        copies the FORMAT verbatim while getting the specific names from the assistant, not from the goal.

        Written as a literal TWO-LINE sequence. The old "then, once holding it" phrasing described a
        later turn, so the 32B read the whole example as one command per reply and never once used the
        action queue."""
        return (
            "Grammar (copy the format; fill the names the assistant gave you; one command per line):\n"
            f"[Grab] <object_name> (1)\n"
            f"[{put_verb}] <object_name> (1) <container_name> (1)"
        )

    @staticmethod
    def _interaction_grammar() -> str:
        """Goal-free command reference for a person carrying out interactive guidance."""
        return """Command grammar (one command per line; use exact object keys):
    [Walk] <target> (1)
    [Run] <target> (1)
    [Forward]
    [TurnLeft]
    [TurnRight]
    [TurnAround]
    [Grab] <object> (1)
    [Flip] <object> (1)
    [Drop]
    [Put] <object> (1) <surface> (1)
    [PutInside] <object> (1) <container> (1)
    [Push] <object> (1) <target> (1)
    [Pull] <object> (1) <target> (1)
    [Open] <object> (1)
    [Close] <object> (1)
    [OpenDoor] <object> (1)
    [CloseDoor] <object> (1)
    [TurnOn] <object> (1)
    [TurnOff] <object> (1)
    [LookAt] <object> (1)
    [LookAround]
    [LookScan]
    [Sit] <object> (1)
    [StandUp]
    [Squat]
    [Cut] <target> (1) <tool> (1)
    [Stir] <target> (1) <tool> (1)
    [Pour] <target> (1) <tool> (1)
    [Wipe] <target> (1) <tool> (1)
    [Cook] <food> (1)
    [AnswerPhone] <phone> (1)
    [Attach] <child> (1) <child_point> (0) <parent> (1) <parent_point> (0)
    [Detach] <child> (1) <parent> (1)
    [Raise] <assembly_root> (1)
    [Wait]"""

    @staticmethod
    def _objlist_block(objs: list, desc: dict, vis_only: bool) -> str:
        """The OBJECTS paragraph of the person's turn prompt, in one of two shapes.

        NAMES (default, unchanged): a comma-separated spelling guide. It carries no appearance and no
        position, which is why a perfectly-worded English description could not be executed -- "the
        locker with the LINE CHARLIE placard" matched 107 names on school_chemistry and the answer,
        ``unit_b``, was not one of them.

        DESCRIPTORS (``obs["object_desc"]``, name -> phrase): one line per object, the name followed by
        what it looks like and where it is. The producer owns the phrase and owns the leak rule with it
        -- appearance, printed placard, support, bearing and distance only, never anything derived from
        the goal, the answer key, a routing rule or an object's contents.
        """
        head = (
            "OBJECTS you can SEE right now"
            if vis_only
            else "MOTOR-AVAILABLE OBJECTS in this scene (some may be out of view)"
        )
        empty = "(you cannot see anything -- look around)" if vis_only else "(none)"
        if not desc:
            return f"{head} (use these names and numbers EXACTLY): {', '.join(objs) or empty}.\n"
        lines = [f"  {n}   {desc[n]}" for n in objs if desc.get(n)]
        lines += [f"  {n}" for n in objs if not desc.get(n)]
        if not lines:
            return "OBJECTS around you: (none).\n"
        body = "\n".join(lines)
        return f"{head} -- NAME first, then appearance and known location:\n{body}\n"

    @staticmethod
    def _watch_policy_block(obs: Mapping[str, object]) -> str:
        policy = obs.get("interaction_watch_policy")
        if not isinstance(policy, Mapping) or not policy:
            return ""
        opening = [
            str(value) for value in policy.get("no_advice_opening_actions") or ()
        ]
        actions = [str(value) for value in policy.get("no_advice_actions") or ()]
        planned_speech = [
            row
            for row in policy.get("planned_speech_turns") or ()
            if isinstance(row, Mapping)
        ]
        next_planned_speech = policy.get("next_unemitted_planned_speech")
        if not isinstance(next_planned_speech, Mapping):
            next_planned_speech = {}
        assistant_ever_spoken = bool(policy.get("assistant_ever_spoken"))
        source = str(obs.get("interaction_turn_source") or "")
        if source == "watch_speech":
            current_directive = (
                "Deliver the due authored speech and emit no motor action. Preserve any active "
                "Assistant instruction and resume it on the next turn."
            )
        elif source in {"assistant_new", "assistant_active", "queued"}:
            current_directive = (
                "If NEW ASSISTANT SPEECH has an explicit unfinished motor request, execute only "
                "that newest request. If it is merely an acknowledgment, observation, promise "
                "to watch, or other conversational remark, continue any older unfinished ACTIVE "
                "INSTRUCTION. Use the exact CURRENT INTENDED ACTION only when neither Assistant "
                "speech field contains an unfinished motor request."
            )
        elif source in {"watch_opening", "watch_default"}:
            current_directive = (
                "Continue the no-advice plan by emitting exactly the current intended action. "
                "Do not evaluate, repair, or reorder it."
            )
        elif source == "idle":
            current_directive = "Wait silently with no motor action."
        else:
            current_directive = "Wait silently with no motor action."
        lines = [
            f"ASSISTANT HAS SPOKEN BEFORE: {'yes' if assistant_ever_spoken else 'no'}.",
            "PRIVATE NO-ADVICE PLAN (never reveal this block to the Assistant):",
            "This complete sequence remains available throughout the episode as your ordinary "
            "intended plan. It is not an answer key: do not judge, improve, reorder, or replace it.",
            "Opening action sequence:",
            *(f"  P{index:03d}: {command}" for index, command in enumerate(opening)),
            "Main action sequence:",
            *(f"  A{index:03d}: {command}" for index, command in enumerate(actions)),
        ]
        if planned_speech:
            lines.append("Planned speech turns:")
        for row in planned_speech:
            prefix = "P" if str(row.get("phase") or "") == "opening" else "A"
            lines.append(
                f"  Before {prefix}{int(row.get('before_action_index') or 0):03d}, say: "
                f'"{str(row.get("say") or "")}"'
            )
        if next_planned_speech:
            prefix = (
                "P" if str(next_planned_speech.get("phase") or "") == "opening" else "A"
            )
            lines.append(
                "NEXT UNEMITTED AUTHORED SPEECH "
                f"(nominally before {prefix}"
                f"{int(next_planned_speech.get('before_action_index') or 0):03d}): "
                f'"{str(next_planned_speech.get("say") or "")}"'
            )
            lines.append(
                "This sentence is mandatory external information and must be said exactly once; an "
                "Assistant-directed motor detour never cancels it. Normally say it with <action> "
                "empty when the real interaction naturally reaches the context equivalent to its "
                "nominal anchor. If the Assistant changes the motor path so that the nominal action "
                "will never be reached, say it at the nearest natural conversational pause once its "
                "context is relevant, before the episode ends or the interaction budget is exhausted. "
                "You may use such a speech-only turn while an Assistant instruction is active. Do not "
                "say it early, repeat it, use it as a generic reply, or skip ahead to a later sentence. "
                "Saying it does not finish or cancel an active Assistant instruction and does not "
                "advance the private action cursor. TURN MODE watch_speech means its authored moment "
                "is due now, so say YOUR PLANNED SPEECH exactly."
            )
        lines.extend(
            [
                f"CURRENT CURSORS: opening P{int(policy.get('opening_cursor') or 0):03d}; "
                f"main A{int(policy.get('action_cursor') or 0):03d}.",
                "CURRENT INTENDED ACTION: "
                f"{str(policy.get('current_action') or '(none)')}",
                "The current intended action remains the fallback on every WATCH turn. Pause it "
                "only while the Assistant has an explicit unfinished motor request; ordinary "
                "acknowledgments, observations, and promises to watch do not cancel it.",
            ]
        )
        lines.extend(
            [
                str(policy.get("assistant_priority") or ""),
                current_directive,
            ]
        )
        return "\n".join(line for line in lines if line) + "\n"

    @staticmethod
    def _watch_response_budget_block(
        obs: Mapping[str, object], remaining_opportunities: int
    ) -> str:
        post_speech_motor_reserve = 6
        policy = obs.get("interaction_watch_policy")
        if not isinstance(policy, Mapping):
            return ""
        next_speech = policy.get("next_unemitted_planned_speech")
        if not isinstance(next_speech, Mapping) or not next_speech:
            return ""
        planned = [
            row
            for row in policy.get("planned_speech_turns") or ()
            if isinstance(row, Mapping)
        ]
        next_key = (
            str(next_speech.get("phase") or ""),
            int(next_speech.get("before_action_index") or 0),
            str(next_speech.get("say") or ""),
        )
        next_index = next(
            (
                index
                for index, row in enumerate(planned)
                if (
                    str(row.get("phase") or ""),
                    int(row.get("before_action_index") or 0),
                    str(row.get("say") or ""),
                )
                == next_key
            ),
            len(planned) - 1,
        )
        pending = planned[max(0, next_index) :]
        pending_count = max(1, len(pending))
        inter_speech_motor_slots = sum(
            max(
                0,
                int(later.get("before_action_index") or 0)
                - int(earlier.get("before_action_index") or 0),
            )
            for earlier, later in zip(pending, pending[1:])
            if str(earlier.get("phase") or "") == str(later.get("phase") or "")
        )
        minimum_needed = (
            pending_count + inter_speech_motor_slots + post_speech_motor_reserve
        )
        lines = [
            "HUMAN RESPONSE OPPORTUNITIES REMAINING (including this turn): "
            f"{remaining_opportunities}."
        ]
        if remaining_opportunities <= minimum_needed:
            lines.append(
                f"Only {remaining_opportunities} response opportunities remain; "
                f"{pending_count} authored sentences, their natural intervening private "
                f"actions, and at least {post_speech_motor_reserve} later motor opportunities "
                "must still fit. Authored speech now has absolute priority: say the NEXT "
                "UNEMITTED AUTHORED SPEECH with <action> empty, before every active Assistant "
                "motor clause. Resume that clause next turn."
            )
        return "\n".join(lines) + "\n"

    def _interaction_human_user_text(
        self,
        obs: dict,
        assistant_say: str,
        objs: list,
        desc: dict,
        vis_only: bool,
        held: list,
    ) -> str:
        interaction_family = str(obs.get("interaction_family") or "").strip().upper()
        task_context = self._public_find_text(
            obs, str(obs.get("interaction_task_context") or "").strip()
        )
        turn_source = str(obs.get("interaction_turn_source") or "idle")
        active = self._public_find_text(
            obs, str(obs.get("interaction_active_instruction") or "").strip()
        )
        private_plan = self._public_find_text(
            obs, str(obs.get("interaction_private_plan") or "").strip()
        )
        interrupted_plan = self._public_find_text(
            obs, str(obs.get("interaction_interrupted_plan") or "").strip()
        )
        planned_speech = self._public_find_text(
            obs, str(obs.get("interaction_planned_speech") or "").strip()
        )
        watch_response_budget = ""
        if interaction_family == "WATCH":
            tick_budget = int(
                getattr(
                    self,
                    "_episode_max_ticks",
                    getattr(self, "max_ticks", 0),
                )
                or 0
            )
            ticks_used = int(getattr(self, "_n", {}).get("ticks", 0) or 0)
            if tick_budget > 0:
                remaining_opportunities = max(0, tick_budget - ticks_used + 1)
                watch_response_budget = self._watch_response_budget_block(
                    obs, remaining_opportunities
                )
        heard = self._public_find_text(obs, assistant_say.strip())
        last_result = self._public_find_text(
            obs, str(getattr(self, "_last_cmd_error", "") or "")
        )
        retry_feedback = str(obs.get("_interaction_human_retry_feedback") or "")
        episode_action_memory = self._interaction_human_episode_action_memory_text(obs)
        action_memory = self._interaction_human_action_memory_text(obs)
        active_label = "ACTIVE INSTRUCTION"
        state_timing = ""
        if interaction_family in {"ASK", "FIND"}:
            state_timing = (
                "STATE TIMING: every `currently ...` entry below is authoritative live state. "
                "A past-tense location in the assistant's answer describes where the item was before "
                "your subsequent actions. Never revisit an old source after the requested current "
                "state is already satisfied. If every requested outcome is now satisfied, do not "
                "inspect or move anything else; report completion.\n"
            )
            if turn_source == "assistant_active":
                active_label = "ORIGINAL ANSWER ABOUT THE PRE-MOVE STATE"
        find_human_control = str(
            obs.get("interaction_find_human_control") or "dynamic-guarded"
        )
        prompt_only_orchestrate = bool(
            interaction_family == "ORCHESTRATE"
            and not bool(obs.get("interaction_orchestrate_human_turn_guard", True))
        )
        public_history = (
            self._interaction_human_public_history(obs)
            if interaction_family == "NARRATE"
            or (interaction_family == "FIND" and find_human_control == "prompt-only")
            else ""
        )
        if obs.get("interaction_opaque_handles"):
            if objs:
                lines = [
                    f"  {name}   {desc.get(name, 'public reference')}" for name in objs
                ]
                if (
                    interaction_family == "FIND" and find_human_control == "prompt-only"
                ) or prompt_only_orchestrate:
                    heading = (
                        "PRIVATE ACTION REFERENCES (type the HANDLE exactly as written; "
                        "listing does not imply visibility, location, correctness, or permission)"
                    )
                else:
                    heading = (
                        "ASSISTANT-GROUNDED LANDMARKS AND CURRENTLY VISIBLE OR HELD "
                        "OBJECTS (type the HANDLE exactly as written)"
                    )
                object_block = f"{heading}:\n" + "\n".join(lines) + "\n"
            else:
                object_block = (
                    "ASSISTANT-GROUNDED LANDMARKS AND CURRENTLY VISIBLE OR HELD "
                    "OBJECTS: (none yet).\n"
                )
        elif obs.get("interaction_grounded_legend_only"):
            if objs:
                lines = [
                    f"  {name}   {desc.get(name, 'grounded reference')}"
                    for name in objs
                ]
                object_block = (
                    "PRIVATE ACTION-ONLY NAMES (copy a NAME only inside <action>; "
                    "never say it aloud):\n" + "\n".join(lines) + "\n"
                )
            else:
                object_block = "PRIVATE ACTION-ONLY NAMES: (none yet).\n"
        else:
            object_block = self._objlist_block(objs, desc, vis_only)
        grammar = self._interaction_grammar()
        if interaction_family == "NARRATE":
            grammar = grammar.replace(
                "Command grammar (one command per line; use exact object keys):",
                "PRIVATE ACTION SYNTAX (copy at most one line and use a listed name exactly):",
            )
        if obs.get("interaction_opaque_handles"):
            grammar = grammar.replace(
                "use exact object keys", "use exact handles from the current legend"
            )
        if interaction_family == "GUIDE" and obs.get("interaction_opaque_handles"):
            action_policy = (
                "For GUIDE with opaque handles, emit exactly one simulator action per response. "
                "This overrides any earlier permission to emit a short local sequence. After that "
                "action succeeds, wait for a fresh frame and current handle legend, then continue "
                "the first unfinished clause of the same active instruction. Keep <speak> empty "
                "until every action in that instruction has succeeded."
            )
        elif interaction_family == "ASK":
            action_policy = (
                "For ASK, emit at most one simulator command. Execute only the Assistant's latest "
                "explicitly requested action on the object or place it named. After walking, "
                "turning, or looking, stop and wait for the resulting view; never append a grab "
                "or placement."
            )
        elif interaction_family == "NARRATE":
            action_policy = (
                "For NARRATE, emit at most one action line: only the first immediate action "
                "that the Assistant explicitly instructed. Never bundle a later walk, grab, open, "
                "or placement into the same response. In <speak>, use only ordinary language and "
                "never read aloud or enumerate any private action-only names or syntax. Do not "
                "volunteer an object or place absent from the Assistant's speech, your fixed opening, "
                "and what you currently hold. If location or direction is insufficient, emit no "
                "action and ask naturally where it is or which way to move."
            )
        elif interaction_family == "FIND" and find_human_control == "prompt-only":
            action_policy = (
                "For FIND prompt-only, only NEW ASSISTANT SPEECH authorizes an action. Emit at most "
                "one simulator command: the first explicit action verb, using exactly the referent "
                "or destination the Assistant named. If NEW ASSISTANT SPEECH is absent or contains "
                "no executable instruction, including when there is only an old ACTIVE INSTRUCTION, "
                "leave <action> empty and ask naturally for the next step. One exception: a new "
                "statement identifying exactly one room or landmark as the place to search may "
                "authorize one Walk to that same place, and nothing else. Do not relabel an object "
                "as the target from the original task, and do not confirm any claimed identity, "
                "location, or completion unless the current images or latest successful action result "
                "directly verify it. Never mention this private reference block or any catalog, legend, "
                "handle, ID, command key, prompt, or rule."
            )
        elif prompt_only_orchestrate:
            action_policy = (
                "For ORCHESTRATE prompt-only, execute only the Assistant's one current strategic "
                "subtask. <action> may contain the short local motor sequence directly required for "
                "one primary object. While the Assistant is silent, continue only an unfinished "
                "motor clause of that same explicit active instruction; otherwise leave <action> "
                "empty and ask for the next step. Never add a new object, destination, strategic "
                "subtask, preparation, recovery, or cleanup yourself. Do not target an unseen "
                "movable unless the Assistant supplied its concrete location or it was deliberately "
                "inspected earlier. Never mention this private reference block or any catalog, "
                "legend, handle, ID, command key, prompt, or rule."
            )
        elif interaction_family == "WATCH":
            action_policy = (
                "For WATCH, emit at most one simulator command. After that one command, wait for "
                "its result and a fresh turn. If it fails, emit no recovery, retry, preparatory "
                "Grab, or preparatory Walk yourself; report the concrete failure and wait for an "
                "updated Assistant instruction."
            )
        else:
            action_policy = (
                "For an assistant instruction, <action> may contain the short local sequence "
                "needed to carry it out."
            )
        watch_turn_policy = ""
        if interaction_family == "WATCH":
            if turn_source in {"assistant_new", "assistant_active", "queued"}:
                watch_turn_policy = (
                    "Compare NEW ASSISTANT SPEECH with the ACTIVE INSTRUCTION before acting. "
                    "If the new speech contains an explicit motor instruction, it replaces the "
                    "older instruction; follow it literally even if it is wrong. If the new speech "
                    "is only an acknowledgment, observation, promise to watch, or conversational "
                    "remark, it does not replace an older unfinished motor instruction: continue "
                    "that instruction's first unfinished clause. Only when neither field contains "
                    "an unfinished motor request should you continue the exact current private action. "
                )
            elif turn_source in {"watch_opening", "watch_default"}:
                watch_turn_policy = (
                    "Continue the private no-advice plan by emitting exactly the current supplied "
                    "action; do not add another. Do not mechanically replay a step the Assistant "
                    "has replaced or reconstruct that old step's prerequisites. "
                )
        return (
            f"TASK CONTEXT: {task_context}\n"
            f"TURN MODE: {turn_source}\n"
            + watch_response_budget
            + state_timing
            + public_history
            + self._watch_policy_block(obs)
            + object_block
            + episode_action_memory
            + f"You are holding: {', '.join(held) if held else 'nothing'}.\n"
            + (
                f'NEW ASSISTANT SPEECH: "{heard}"\n'
                if heard
                else "NEW ASSISTANT SPEECH: (none)\n"
            )
            + (
                f'{active_label}: "{active}"\n'
                if active
                else f"{active_label}: (none)\n"
            )
            + action_memory
            + (
                "YOUR OWN NEXT INTENDED ACTION (private; the assistant cannot see this field): "
                f"{private_plan}\n"
                if private_plan
                else "YOUR OWN NEXT INTENDED ACTION: (none)\n"
            )
            + (
                "PRIVATE ACTION AT THE CURRENT CURSOR (pause only for an explicit unfinished "
                "Assistant motor request): "
                f"{interrupted_plan}\n"
                if interrupted_plan
                else ""
            )
            + (
                "YOUR PLANNED SPEECH FOR THIS TURN: "
                f'"{planned_speech}"\nLeave <action> empty and say that sentence now.\n'
                if planned_speech
                else ""
            )
            + (f"RESULT OF YOUR LAST ACTION: {last_result}\n" if last_result else "")
            + (
                "PRIVATE VALIDATOR FEEDBACK: Your previous response was not executed. "
                f"{retry_feedback}\n"
                if retry_feedback
                else ""
            )
            + f"{grammar}\n"
            + "Reply with <reasoning>, then <action>, then <speak>. "
            + watch_turn_policy
            + "For a WATCH planned speech turn, emit no action and say only the supplied sentence. "
            + action_policy
            + " On a WATCH idle turn, leave both <action> and <speak> empty and wait. Otherwise, if "
            "there is no safe action, leave <action> empty and say what you need."
            + (f"\n\n{ASK_HUMAN_PER_TURN_GATE}" if interaction_family == "ASK" else "")
        )

    def _interaction_human_episode_action_memory_text(
        self, obs: Mapping[str, object]
    ) -> str:
        if str(obs.get("interaction_family") or "").strip().upper() != "NARRATE":
            return ""
        raw_memory = obs.get("interaction_successful_action_history")
        if not isinstance(raw_memory, Sequence) or isinstance(raw_memory, (str, bytes)):
            return ""
        actions = [
            self._public_find_text(obs, str(row.get("action") or "").strip())
            for row in raw_memory
            if isinstance(row, Mapping) and row.get("result") == "succeeded"
        ]
        actions = [action for action in actions if action]
        if not actions:
            return ""
        history = "\n".join(
            f"  {index}. {action} -- succeeded"
            for index, action in enumerate(actions, start=1)
        )
        return (
            "YOUR OWN SUCCESSFUL ACTIONS SO FAR (facts, never a plan):\n"
            f"{history}\n"
            "These actions already happened. Never repeat one or treat a private description's "
            "original location as current after the object moved.\n"
        )

    def _interaction_human_action_memory_text(self, obs: Mapping[str, object]) -> str:
        raw_memory = obs.get("interaction_active_instruction_action_memory")
        if not isinstance(raw_memory, Sequence) or isinstance(raw_memory, (str, bytes)):
            return ""
        actions = []
        for row in raw_memory:
            if not isinstance(row, Mapping) or row.get("result") != "succeeded":
                continue
            action = self._public_find_text(obs, str(row.get("action") or "").strip())
            if action:
                actions.append(action)
        if not actions:
            return ""
        history = "\n".join(
            f"  {index}. {action} -- succeeded"
            for index, action in enumerate(actions, start=1)
        )
        return (
            "YOUR OWN SUCCESSFUL ACTIONS UNDER THIS ACTIVE INSTRUCTION "
            "(execution history only, not a plan):\n"
            f"{history}\n"
            "Do not restart or accidentally repeat a completed clause. Continue with the first "
            "unfinished clause of the active instruction, without inventing future steps.\n"
        )

    @staticmethod
    def _public_find_text(obs: Mapping[str, object], text: str) -> str:
        if not bool(obs.get("interaction_opaque_handles")) or not text:
            return text
        handle_map = obs.get("_interaction_human_handle_map")
        if not isinstance(handle_map, Mapping):
            return text
        replacements = sorted(
            (
                (str(internal), str(handle))
                for handle, internal in handle_map.items()
                if str(internal)
                and str(handle)
                and _LOOSE_KEY_RE.fullmatch(str(internal))
            ),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        redacted = text
        for internal, handle in replacements:
            redacted = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(internal)}(?![A-Za-z0-9_])",
                handle,
                redacted,
                flags=re.IGNORECASE,
            )
        assigned = {internal for internal, _handle in replacements}
        known_keys = obs.get("_interaction_human_internal_keys")
        if isinstance(known_keys, Sequence) and not isinstance(
            known_keys, (str, bytes)
        ):
            for internal in sorted(
                (
                    str(value)
                    for value in known_keys
                    if "_" in str(value) and str(value) not in assigned
                ),
                key=len,
                reverse=True,
            ):
                redacted = re.sub(
                    rf"(?<![A-Za-z0-9_]){re.escape(internal)}(?![A-Za-z0-9_])",
                    "unrecognized reference",
                    redacted,
                    flags=re.IGNORECASE,
                )
        public_handles = {str(handle).lower() for handle in handle_map}
        return _LOOSE_KEY_TOKEN_RE.sub(
            lambda match: (
                match.group(0)
                if match.group(0).lower() in public_handles
                else "unrecognized reference"
            ),
            redacted,
        )

    def _interaction_human_public_history(self, obs: Mapping[str, object]) -> str:
        opening = " ".join(
            self._public_find_text(
                obs, str(obs.get("interaction_opening_say") or "")
            ).split()
        )
        lines = ["PUBLIC CONVERSATION SO FAR (context only, not a new instruction):"]
        if opening:
            lines.append(f'  Your fixed opening: "{opening}"')
        for row in getattr(self, "_transcript", [])[-12:]:
            if not isinstance(row, Mapping):
                continue
            assistant_say = " ".join(
                self._public_find_text(obs, str(row.get("assistant_say") or "")).split()
            )
            human_say = " ".join(
                self._public_find_text(obs, str(row.get("human_say") or "")).split()
            )
            if assistant_say:
                lines.append(f'  Assistant earlier: "{assistant_say}"')
            if human_say:
                lines.append(f'  You replied earlier: "{human_say}"')
        if len(lines) == 1:
            return ""
        if str(obs.get("interaction_family") or "").strip().upper() == "NARRATE":
            lines.append(
                "Use this only to remember object roles and what already happened. Follow NEW "
                "ASSISTANT SPEECH below, or one unfinished immediate action in the ACTIVE "
                "INSTRUCTION; never treat old speech as a new instruction."
            )
        else:
            lines.append(
                "Use this only to preserve object roles and conversational context. Only NEW "
                "ASSISTANT SPEECH below may authorize the current action."
            )
        return "\n".join(lines) + "\n"

    def _human_prompt(
        self,
        obs: dict,
        assistant_say: str,
        objs: list,
        desc: dict,
        vis_only: bool,
        held: list,
        put_verb: str,
    ) -> tuple[str, str, str]:
        interaction_family = str(obs.get("interaction_family") or "").strip()
        if interaction_family:
            self._last_human_mode = str(
                obs.get("interaction_turn_source") or "interaction"
            )
            return (
                interaction_human_system(
                    interaction_family,
                    opaque_handles=bool(obs.get("interaction_opaque_handles")),
                    find_human_control=str(
                        obs.get("interaction_find_human_control") or "dynamic-guarded"
                    ),
                    orchestrate_human_turn_guard=bool(
                        obs.get("interaction_orchestrate_human_turn_guard", True)
                    ),
                ),
                self._interaction_human_user_text(
                    obs, assistant_say, objs, desc, vis_only, held
                ),
                interaction_family,
            )
        task = self._humanize_task(obs.get("task", ""))
        mode = (
            self._error_mode
            if self._error_turn is not None
            and self._n["human_calls"] == self._error_turn
            else "normal"
        )
        self._n[f"human_mode_{mode}"] = self._n.get(f"human_mode_{mode}", 0) + 1
        self._last_human_mode = mode
        system = human_system(task, mode, visible_only=vis_only, described=bool(desc))
        heard = assistant_say.strip()[:300]
        user_text = (
            self._objlist_block(objs, desc, vis_only)
            + f"You are holding: {(', '.join(held) if held else 'nothing')}.\n"
            + (
                f'The assistant just said: "{heard}"\n'
                if heard
                else "The assistant has said nothing since your last action.\n"
            )
            + f"{self._grammar_generic(put_verb)}\n"
            + (
                f"RESULT of your last action: {self._last_cmd_error}\n"
                if getattr(self, "_last_cmd_error", "")
                else ""
            )
            + "Reply with <reasoning>, then <action>, then <speak>. <action> may hold the short sequence of commands that carries out what the assistant asked -- they run one at a time. Both blocks must be present; either may be empty, but not both. If <action> is empty, <speak> must name what stopped you."
        )
        return (system, user_text, "")

    @staticmethod
    def _interaction_visible_keys(obs: Mapping[str, object]) -> object:
        full_visible = obs.get("interaction_visible_object_keys")
        if isinstance(full_visible, (list, tuple, set)):
            return full_visible
        vischeck = obs.get("vischeck")
        if bool(obs.get("objects_are_visible_only")):
            return obs.get("objects") or ()
        if isinstance(vischeck, Mapping):
            return vischeck.get("visible") or ()
        return ()

    @staticmethod
    def _interaction_policy_mapping(
        obs: Mapping[str, object], key: str, *, optional: bool = False
    ) -> Mapping[object, object] | None:
        value = obs.get(key)
        if isinstance(value, Mapping) and (value or not optional):
            return value
        return None if optional else {}

    def _interaction_human_policy_inputs(
        self,
        obs: Mapping[str, object],
        interaction_family: str,
        commands: list[str],
    ) -> dict[str, object]:
        internal_descriptions = obs.get("_interaction_internal_object_desc")
        public_descriptions = obs.get("object_desc")
        object_descriptions = (
            internal_descriptions
            if isinstance(internal_descriptions, dict)
            else public_descriptions
            if isinstance(public_descriptions, dict)
            else {}
        )
        raw_action_memory = obs.get("interaction_active_instruction_action_memory")
        action_memory = (
            tuple(
                str(row.get("action") or "")
                for row in raw_action_memory
                if isinstance(row, Mapping) and row.get("result") == "succeeded"
            )
            if isinstance(raw_action_memory, Sequence)
            and not isinstance(raw_action_memory, (str, bytes))
            else ()
        )
        enforce_ask_instruction_guard = bool(
            getattr(self, "_interaction_ask_instruction_guard", True)
        )
        human_say = self._last_human_say
        if not enforce_ask_instruction_guard:
            human_say = str(
                getattr(self, "_last_human_say_unredacted", "") or human_say
            )
        return {
            "family": interaction_family,
            "source": str(obs.get("interaction_turn_source") or "idle"),
            "commands": commands,
            "human_say": human_say,
            "human_reasoning": str(getattr(self, "_last_human_reasoning", "") or ""),
            "unredacted_human_say": str(
                getattr(self, "_last_human_say_unredacted", "") or ""
            ),
            "private_plan": str(obs.get("interaction_private_plan") or ""),
            "interrupted_plan": str(obs.get("interaction_interrupted_plan") or ""),
            "planned_speech": str(obs.get("interaction_planned_speech") or ""),
            "next_planned_speech": str(
                obs.get("interaction_next_planned_speech") or ""
            ),
            "authored_planned_speech": (
                obs.get("_interaction_watch_authored_planned_speech") or ()
            ),
            "assistant_instruction": str(
                obs.get("interaction_active_instruction") or ""
            ),
            "opening_say": str(obs.get("interaction_opening_say") or ""),
            "object_descriptions": object_descriptions,
            "motor_aliases": IsaacAgentLoop._interaction_policy_mapping(
                obs, "_interaction_motor_aliases"
            ),
            "ask_movable_object_keys": (
                obs.get("_interaction_ask_movable_object_keys") or ()
            ),
            "find_target_keys": obs.get("interaction_find_target_keys") or (),
            "visible_object_keys": IsaacAgentLoop._interaction_visible_keys(obs),
            "held_object_keys": (
                obs.get("_interaction_internal_held_object_keys")
                or obs.get("held")
                or ()
            ),
            "watch_static_locations": IsaacAgentLoop._interaction_policy_mapping(
                obs, "_interaction_watch_static_locations"
            ),
            "human_at_keys": (
                obs.get("at") if isinstance(obs.get("at"), (list, tuple, set)) else ()
            ),
            "find_grounding_aliases": IsaacAgentLoop._interaction_policy_mapping(
                obs, "_interaction_find_grounding_aliases", optional=True
            ),
            "find_object_aliases": IsaacAgentLoop._interaction_policy_mapping(
                obs, "_interaction_find_object_aliases"
            ),
            "find_target_aliases": IsaacAgentLoop._interaction_policy_mapping(
                obs, "_interaction_find_target_aliases"
            ),
            "find_destination_aliases": IsaacAgentLoop._interaction_policy_mapping(
                obs, "_interaction_find_destination_aliases"
            ),
            "find_target_destinations": IsaacAgentLoop._interaction_policy_mapping(
                obs, "_interaction_find_target_destinations"
            ),
            "find_open_object_keys": (
                obs.get("_interaction_find_open_object_keys") or ()
            ),
            "find_bound_landmark_keys": (
                obs.get("_interaction_find_bound_landmark_keys") or ()
            ),
            "last_successful_human_command": str(
                getattr(self, "_last_successful_human_command", "") or ""
            ),
            "active_instruction_successful_commands": action_memory,
            "last_human_action_result": str(getattr(self, "_last_cmd_error", "") or ""),
            "enforce_ask_instruction_guard": enforce_ask_instruction_guard,
        }

    def _apply_interaction_human_policy(
        self,
        obs: dict,
        interaction_family: str,
        commands: list[str],
    ) -> tuple[list[str], str]:
        if not interaction_family:
            return commands, ""
        if interaction_family.strip().upper() == "WATCH" and not bool(
            getattr(self, "_interaction_watch_human_turn_guard", True)
        ):
            policy = IsaacAgentLoop._interaction_human_policy_inputs(
                self, obs, interaction_family, commands
            )
            _, _, audit_guard = enforce_human_turn_policy(**policy)
            repeat_guard = (
                audit_guard if audit_guard == "watch_planned_speech_repeat" else ""
            )
            self._last_human_policy_audit = "would_block" if repeat_guard else "pass"
            self._last_human_policy_audit_reasons = (
                [repeat_guard] if repeat_guard else []
            )
            if repeat_guard:
                audit_key = f"human_policy_audit_{repeat_guard}"
                self._n[audit_key] = self._n.get(audit_key, 0) + 1
                self._n["watch_planned_speech_repeats"] = (
                    self._n.get("watch_planned_speech_repeats", 0) + 1
                )
            return commands, ""
        if interaction_family.strip().upper() == "ORCHESTRATE" and not bool(
            getattr(self, "_interaction_orchestrate_human_turn_guard", True)
        ):
            self._last_human_policy_audit = "pass"
            self._last_human_policy_audit_reasons = []
            return commands, ""
        policy = IsaacAgentLoop._interaction_human_policy_inputs(
            self, obs, interaction_family, commands
        )
        enforce_find_guard = bool(getattr(self, "_interaction_find_model_guard", True))
        enforce_guide_guard = bool(
            getattr(self, "_interaction_guide_human_turn_guard", True)
        )
        enforce_ask_guard = bool(policy["enforce_ask_instruction_guard"])
        if interaction_family.strip().upper() == "ASK" and not enforce_ask_guard:
            audit_policy = {**policy, "enforce_ask_instruction_guard": True}
            _, _, audit_guard = enforce_human_turn_policy(
                **audit_policy,
                enforce_find_model_guard=enforce_find_guard,
                enforce_guide_turn_guard=enforce_guide_guard,
            )
            guarded_commands = list(commands)
            guarded_say = str(policy["unredacted_human_say"] or policy["human_say"])
            guard = ""
        else:
            guarded_commands, guarded_say, guard = enforce_human_turn_policy(
                **policy,
                enforce_find_model_guard=enforce_find_guard,
                enforce_guide_turn_guard=enforce_guide_guard,
            )
            audit_guard = guard
        if interaction_family.strip().upper() == "FIND" and not enforce_find_guard:
            _, _, audit_guard = enforce_human_turn_policy(
                **policy,
                enforce_find_model_guard=True,
            )
        audit_reasons = [audit_guard] if audit_guard else []
        self._last_human_policy_audit_reasons = audit_reasons
        audit_only = guard in _HUMAN_AUDIT_ONLY_GUARDS
        self._last_human_policy_audit = (
            "would_block"
            if audit_only
            else "blocked"
            if guard
            else "would_block"
            if audit_reasons
            else "pass"
        )
        for reason in audit_reasons:
            key = f"human_policy_audit_{reason}"
            self._n[key] = self._n.get(key, 0) + 1
        deferred_guard = bool(
            human_reprompt_guard_enabled(
                interaction_family,
                guard,
                human_object_interface=str(
                    obs.get("interaction_human_object_interface") or "full_ids"
                ),
            )
            and getattr(self, "_defer_ask_human_guard_accounting", False)
        )
        self._last_human_say = "" if deferred_guard else guarded_say
        if deferred_guard:
            self._n["human_invalid_attempts"] = (
                self._n.get("human_invalid_attempts", 0) + 1
            )
            key = f"human_invalid_attempt_{guard}"
            self._n[key] = self._n.get(key, 0) + 1
        elif guard and not audit_only:
            key = f"human_policy_guard_{guard}"
            self._n[key] = self._n.get(key, 0) + 1
            if guard == "watch_planned_speech_repeat":
                self._n["watch_planned_speech_repeats"] = (
                    self._n.get("watch_planned_speech_repeats", 0) + 1
                )
        return guarded_commands, guard

    def _translate_and_guard_interaction_human_commands(
        self,
        obs: dict,
        interaction_family: str,
        commands: list[str],
    ) -> tuple[list[str], str]:
        self._last_human_policy_audit = "pass"
        self._last_human_policy_audit_reasons = []
        commands, handle_guard = self._translate_interaction_human_commands(
            obs, commands
        )
        self._last_human_translated_actions = list(commands)
        if not handle_guard:
            return self._apply_interaction_human_policy(
                obs, interaction_family, commands
            )
        self._last_human_say = (
            "Could you describe the object or place in ordinary words?"
        )
        self._last_human_policy_audit = "blocked"
        self._last_human_policy_audit_reasons = [handle_guard]
        audit_key = f"human_policy_audit_{handle_guard}"
        self._n[audit_key] = self._n.get(audit_key, 0) + 1
        deferred_guard = bool(
            (
                interaction_family.strip().upper() != "GUIDE"
                or bool(getattr(self, "_interaction_guide_human_turn_guard", True))
            )
            and human_reprompt_guard_enabled(
                interaction_family,
                handle_guard,
                human_object_interface=str(
                    obs.get("interaction_human_object_interface") or "full_ids"
                ),
            )
            and getattr(self, "_defer_ask_human_guard_accounting", False)
        )
        if deferred_guard:
            self._last_human_say = ""
            self._n["human_invalid_attempts"] = (
                self._n.get("human_invalid_attempts", 0) + 1
            )
            key = f"human_invalid_attempt_{handle_guard}"
            self._n[key] = self._n.get(key, 0) + 1
            return [], handle_guard
        key = f"human_policy_guard_{handle_guard}"
        self._n[key] = self._n.get(key, 0) + 1
        return [], handle_guard

    def _request_human_completion(
        self, request: Mapping[str, object], assistant_say: str, frame_count: int
    ) -> Mapping[str, object] | None:
        self._last_human_provider = {}
        data = json.dumps(request).encode("utf-8")
        attempt_limit = max(1, int(getattr(self, "human_http_attempt_limit", 3)))
        backoff_s = max(0.0, float(getattr(self, "human_http_retry_backoff_s", 0.25)))
        last_failure = ""
        for attempt in range(attempt_limit):
            nonretryable = False
            try:
                http_request = urllib.request.Request(
                    self.human_server,
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(
                    http_request,
                    timeout=float(getattr(self, "human_timeout_s", 120.0)),
                ) as response:
                    output = json.loads(response.read().decode("utf-8"))
                if isinstance(output, Mapping):
                    provider_attempts = max(
                        1, int(output.get("provider_attempts") or 1)
                    )
                    provider_errors = [
                        str(value) for value in output.get("provider_errors") or ()
                    ]
                    self._last_human_provider = {
                        "provider": str(output.get("provider") or ""),
                        "model": str(output.get("provider_model") or ""),
                        "endpoint": str(output.get("provider_endpoint") or ""),
                        "frame_count": int(output.get("provider_frame_count") or 0),
                        "attempts": provider_attempts,
                        "errors": provider_errors,
                        "finish_reason": str(
                            output.get("provider_finish_reason") or ""
                        ),
                        "finish_reasons": [
                            str(value)
                            for value in output.get("provider_finish_reasons") or ()
                        ],
                        "stop_details": dict(output.get("provider_stop_details") or {}),
                        "usage": dict(output.get("provider_usage") or {}),
                        "truncated": bool(output.get("provider_truncated")),
                        "max_output_tokens": int(
                            output.get("provider_max_output_tokens") or 0
                        ),
                    }
                    if output.get("ok") and (provider_attempts > 1 or provider_errors):
                        failed_attempts = max(
                            provider_attempts - 1, len(provider_errors)
                        )
                        self._n["human_http_attempt_failures"] = (
                            self._n.get("human_http_attempt_failures", 0)
                            + failed_attempts
                        )
                        self._n["human_http_retries"] = (
                            self._n.get("human_http_retries", 0) + provider_attempts - 1
                        )
                    if output.get("ok"):
                        return output
                    nonretryable = bool(output.get("provider_truncated"))
                    if nonretryable:
                        self._n["human_provider_truncated"] = (
                            self._n.get("human_provider_truncated", 0) + 1
                        )
                error_text = (
                    str(output.get("error"))
                    if isinstance(output, Mapping)
                    else "malformed response"
                )
                last_failure = f"NOT-OK: {error_text[:200]}"
            except Exception as error:  # noqa: BLE001 - bounded retry before hard failure
                last_failure = f"{type(error).__name__}: {str(error)[:160]}"
            self._n["human_http_attempt_failures"] = (
                self._n.get("human_http_attempt_failures", 0) + 1
            )
            if nonretryable:
                break
            if attempt + 1 < attempt_limit:
                self._n["human_http_retries"] = self._n.get("human_http_retries", 0) + 1
                if backoff_s:
                    time.sleep(min(2.0, backoff_s * (attempt + 1)))
        self._n["human_http_fail"] = self._n.get("human_http_fail", 0) + 1
        self._requeue_say(assistant_say)
        print(
            f"[isaac_human] HTTP attempts exhausted ({last_failure}) -- tick idled, "
            f"{frame_count} frames were in the payload",
            flush=True,
        )
        return None

    def _human_command(
        self, obs: dict, assistant_say: str, human_frames: list | None = None
    ) -> Optional[str]:
        """Request one goal-blind Human action from the provider-neutral endpoint. The Human receives only observable state and the Assistant instruction. TaskPack rows always use their frozen interaction-family prompt."""
        objs = obs.get("objects") or []
        held = sorted(obs.get("held", []))
        internal_held = sorted(
            obs.get("_interaction_internal_held_object_keys") or held
        )
        put_verb = obs.get("put_verb", "PutInside")
        _vis_only = bool(obs.get("objects_are_visible_only"))
        _desc = obs.get("object_desc") or {}
        system, user_text, interaction_family = self._human_prompt(
            obs, assistant_say, objs, _desc, _vis_only, held, put_verb
        )
        msg: dict = {"role": "user", "text": user_text}
        frames = (
            human_frames
            if human_frames is not None
            else [obs["frame"]]
            if obs.get("frame")
            else []
        )
        if interaction_family and (
            not bool(obs.get("interaction_human_receives_frames", True))
        ):
            frames = []
        if frames:
            msg["images_b64"] = frames
        req = {
            "system": system,
            "messages": [msg],
            "max_new_tokens": int(os.getenv("ISAAC_HUMAN_MAX_TOK", "320")),
            "temperature": 0.0,
            "do_sample": False,
        }
        out = self._request_human_completion(req, assistant_say, len(frames))
        if out is None:
            return None
        text = out.get("text", "") or ""
        _a = re.search("<action>(.*?)</action>", text, re.I | re.S)
        _scope = _a.group(1) if _a else text
        _sp = re.search("<speak>(.*?)</speak>", text, re.I | re.S)
        self._last_human_say = self._guard_human_say(
            _sp.group(1).strip() if _sp else "", objs
        )
        _rs = re.search("<reasoning>(.*?)</reasoning>", text, re.I | re.S)
        self._last_human_reasoning = _rs.group(1).strip() if _rs else ""
        if _a is None:
            self._n["human_no_action_tag"] = self._n.get("human_no_action_tag", 0) + 1
        cmds: list = self._extract_commands(_scope, objs)
        parsed_handle_commands = list(cmds)
        discarded_for_reobserve: list[str] = []
        discarded_say_for_reobserve = ""
        if (
            interaction_family.strip().upper() == "GUIDE"
            and str(obs.get("interaction_human_object_interface") or "full_ids").strip()
            == "opaque_handles"
            and (len(cmds) > 1)
        ):
            discarded_for_reobserve = list(cmds[1:])
            cmds = cmds[:1]
            self._n["guide_opaque_multi_action_responses"] = (
                self._n.get("guide_opaque_multi_action_responses", 0) + 1
            )
            self._n["guide_opaque_actions_discarded_for_reobserve"] = self._n.get(
                "guide_opaque_actions_discarded_for_reobserve", 0
            ) + len(discarded_for_reobserve)
            discarded_say_for_reobserve = (
                self._last_human_say or self._pending_human_say
            )
            self._last_human_say = ""
            self._last_human_say_unredacted = ""
            self._pending_human_say = ""
        mechanical_guard = (
            human_action_mechanical_guard(_scope, cmds, held_object_keys=held)
            if interaction_family.strip().upper() == "NARRATE" and _a is not None
            else ""
        )
        if mechanical_guard:
            cmds = []
            human_policy_guard = mechanical_guard
            self._last_human_translated_actions = []
            self._last_human_policy_audit = "blocked"
            self._last_human_policy_audit_reasons = [mechanical_guard]
            audit_key = f"human_policy_audit_{mechanical_guard}"
            self._n[audit_key] = self._n.get(audit_key, 0) + 1
            if getattr(self, "_defer_ask_human_guard_accounting", False):
                self._last_human_say = ""
                self._n["human_invalid_attempts"] = (
                    self._n.get("human_invalid_attempts", 0) + 1
                )
                invalid_key = f"human_invalid_attempt_{mechanical_guard}"
                self._n[invalid_key] = self._n.get(invalid_key, 0) + 1
            else:
                policy_key = f"human_policy_guard_{mechanical_guard}"
                self._n[policy_key] = self._n.get(policy_key, 0) + 1
        else:
            cmds, human_policy_guard = (
                self._translate_and_guard_interaction_human_commands(
                    obs, interaction_family, cmds
                )
            )
        if len(parsed_handle_commands) > 1:
            self._n["human_multi_action"] += 1
        command_before_repair = cmds[0] if cmds else None
        cmd = self._repair_one_arg_put(command_before_repair, internal_held)
        command_repair = IsaacAgentLoop._one_arg_put_repair_audit(
            command_before_repair, cmd, internal_held
        )
        episode_ticks = getattr(self, "_episode_max_ticks", self.max_ticks)
        _room = max(0, episode_ticks - self._n["ticks"])
        self._pending_cmds = cmds[1 : 1 + _room]
        deferred_say = ""
        if interaction_family and cmd and self._last_human_say:
            deferred_say = self._last_human_say
            self._pending_human_say = deferred_say
            self._last_human_say = ""
        if len(cmds) - 1 > _room:
            self._n["queue_clipped"] += 1
        if human_policy_guard and getattr(
            self, "_defer_ask_human_guard_accounting", False
        ):
            pass
        elif self._last_human_say and cmd is None:
            self._n["human_asked"] = self._n.get("human_asked", 0) + 1
        elif cmd is None:
            self._n["human_mute_idle"] = self._n.get("human_mute_idle", 0) + 1
        self._transcript.append(
            {
                "turn": self._n["human_calls"],
                "assistant_mode": getattr(self, "_last_assistant_mode", ""),
                "assistant_say": assistant_say.strip(),
                "assistant_raw": getattr(self, "_last_assistant_raw", ""),
                "human_mode": getattr(self, "_last_human_mode", "normal"),
                "human_reasoning": self._last_human_reasoning,
                "human_parsed_handle_actions": parsed_handle_commands,
                "human_translated_actions": list(
                    getattr(self, "_last_human_translated_actions", cmds)
                ),
                "human_action": cmd or "",
                "human_queued": list(self._pending_cmds),
                "human_say": self._last_human_say,
                "human_say_deferred": deferred_say,
                "human_actions_discarded_for_reobserve": discarded_for_reobserve,
                "human_say_discarded_for_reobserve": discarded_say_for_reobserve,
                "human_policy_recovery": "guide_opaque_multi_action_reobserve"
                if discarded_for_reobserve
                else "",
                "human_raw": text if interaction_family else text[:1200],
                "human_provider": dict(getattr(self, "_last_human_provider", {}) or {}),
                "human_system_prompt": system if interaction_family else "",
                "human_user_prompt": user_text if interaction_family else "",
                "human_policy_guard": human_policy_guard,
                "human_policy_audit": getattr(self, "_last_human_policy_audit", "pass"),
                "human_policy_audit_reasons": list(
                    getattr(self, "_last_human_policy_audit_reasons", ())
                ),
                "guide_human_turn_guard": bool(
                    getattr(self, "_interaction_guide_human_turn_guard", True)
                ),
                "watch_human_turn_guard": bool(
                    getattr(self, "_interaction_watch_human_turn_guard", True)
                ),
                "orchestrate_human_turn_guard": bool(
                    getattr(self, "_interaction_orchestrate_human_turn_guard", True)
                ),
                "find_human_control": str(
                    obs.get("interaction_find_human_control") or "dynamic-guarded"
                ),
                "ask_human_control": str(getattr(self, "_ask_human_control", "strict")),
                "visible_object_keys_internal": list(
                    obs.get("interaction_visible_object_keys") or ()
                ),
                "held": list(held),
                "held_internal": list(internal_held),
                "human_handle_map": dict(
                    obs.get("_interaction_human_handle_map") or {}
                ),
                "human_available_handles": list(
                    obs.get("_interaction_human_available_handles") or ()
                ),
                "human_object_interface": str(
                    obs.get("interaction_human_object_interface") or "full_ids"
                ),
                "vischeck": self._last_vischeck,
                "objects": list(objs),
                "objects_visible_only": _vis_only,
            }
        )
        if command_repair is not None:
            self._transcript[-1].update(
                {
                    "human_action_before_repair": command_before_repair,
                    "human_command_repair": command_repair,
                }
            )
        return cmd

    @staticmethod
    def _prologue_cmds(goal: list) -> list:
        """A look-around performed BEFORE the dialogue, so the assistant opens the conversation having
        seen the room instead of guessing from one frame.

        Three steps, and the count does NOT grow with the task:
          1. walk to a task object -- this is what puts the person in the RIGHT ROOM. The start pose is
             not guaranteed to be where the work is (the toy scene starts elsewhere), and an eyes-level
             pan from the wrong room shows nothing relevant.
          2. LOOK_AROUND -- eye-level pan: room layout, furniture, where the destination is.
          3. LOOK_SCAN -- a full turn with the gaze on the floor. Rendering LOOK_AROUND in the trash
             scene came back with cabinets and a range hood and none of the three cans on the ground;
             small objects on the floor sit below its bottom edge, so the level pan alone is not enough.

        Walking to ONE object, not each of them, is deliberate: eight toys would mean eighteen commands
        -- longer than the fifteen-tick conversation -- and marching the person to every single item
        hands the assistant the count and the locations, which are exactly what it is supposed to read
        off the video.
        """
        objs, tgts = [], []
        for g in goal or []:
            o, t = g.get("obj"), g.get("target")
            if o and o not in objs:
                objs.append(o)
            if t and t not in tgts:
                tgts.append(t)
        if not objs and not tgts:
            return []
        anchor = objs[len(objs) // 2] if objs else tgts[0]
        # LOOK_SCAN, not LOOK_AT: it names no object, so it discloses nothing about where things are --
        # the person simply turns with their eyes down and whatever is on the floor comes into frame.
        # LOOK_AROUND stays for the room layout; on its own it is eye-level and returned cabinets and a
        # range hood with none of the three cans in shot, which is what LOOK_SCAN exists to fix.
        # The walk only exists to get into the room the work is in (the toy scene does not start there);
        # its frames are ordinary and may be trimmed later. The two sweeps that follow are the ones that
        # get PINNED -- eye level for the layout, then gaze-down for what is on the floor.
        return [f"[Walk] <{anchor}> (1)", "[LookAround]", "[LookScan]"]

    def _pick_error_turn(self, max_ticks: int) -> None:
        """Disable synthetic Human slips in released evaluation."""
        del max_ticks
        self._error_turn = None
        self._error_mode = "normal"

    @staticmethod
    def _strip_nonaction(text: str) -> str:
        """Drop the <reasoning> and <speak> spans, including an unterminated one. Neither is a command,
        and both routinely name objects the human considered and then decided AGAINST touching.
        """
        out = re.sub(r"<(reasoning|speak)>.*?</\1>", " ", text, flags=re.I | re.S)
        return re.sub(r"<(?:reasoning|speak)>.*$", " ", out, flags=re.I | re.S)

    @staticmethod
    def _translate_interaction_human_commands(
        obs: Mapping[str, object], commands: Sequence[str]
    ) -> tuple[list[str], str]:
        if not bool(obs.get("interaction_opaque_handles")):
            return list(commands), ""
        raw_map = obs.get("_interaction_human_handle_map")
        raw_available = obs.get("_interaction_human_available_handles")
        if (
            not isinstance(raw_map, Mapping)
            or not isinstance(raw_available, Sequence)
            or isinstance(raw_available, (str, bytes))
        ):
            family = str(obs.get("interaction_family") or "find").lower()
            return [], f"{family}_handle_map_missing"
        handle_map = {str(handle): str(name) for handle, name in raw_map.items()}
        available = {str(handle) for handle in raw_available}
        translated: list[str] = []
        for command in commands:
            references = re.findall(r"<([^>]+)>\s*\((-?\d+)\)", command)
            if any(
                handle not in available or handle not in handle_map
                for handle, _ in references
            ):
                family = str(obs.get("interaction_family") or "find").lower()
                return [], f"{family}_unknown_human_handle_blocked"
            translated_command = command
            for handle, _ in references:
                translated_command = re.sub(
                    rf"<\s*{re.escape(handle)}\s*>",
                    f"<{handle_map[handle]}>",
                    translated_command,
                )
            translated.append(translated_command)
        return translated, ""

    def _guard_human_say(self, say: str, objs: list) -> str:
        """The person's speech as the ASSISTANT will hear it: object IDENTIFIERS removed.

        The human is handed the full object list; the assistant deliberately is not -- it has to read
        the scene off the video, and that is the whole point of the redesign. Anything the human echoes
        back ("put the can_of_soda_115 in the trash_can_116") hands it both the vocabulary and the
        layout for free. Two shapes are caught: a key from THIS observation, and the generic id shape,
        so a hallucinated id leaks no less than a real one. Ordinary English carries no underscore and
        is left alone.

        Only IDENTIFIER-shaped keys are redacted. A registry key that is a bare English noun
        (`microwave`) is left alone: the prompt requires the refusal to NAME what stopped the person
        ("I can't reach the microwave"), and blanking that noun returns "I can't reach the that" --
        destroying the very sentence the contract asks for while leaking nothing.

        Also counts the content-free utterances ("what should I do?"), which are a turn spent saying
        nothing and were previously indistinguishable from the person reporting a real obstacle, and
        the SPATIAL leaks -- "the can is next to the sofa", "I see three by the couch" -- which carry
        no identifier at all yet hand the assistant the layout it is supposed to read off the video.
        Counted, not redacted: rewriting free English would mangle the honest refusals too, so the
        decision on whether to police it is deferred until this measurement exists.
        """
        self._last_human_say_unredacted = say
        if not say:
            return say
        out, n = say, 0
        # longest key first: `can_of_soda_115` must go before a `can_of_soda` that is also registered.
        _shape = _LOOSE_KEY_RE if _redact_keys_loose() else _OBJ_ID_RE
        _ids = {str(o) for o in objs or [] if o and _shape.fullmatch(str(o).strip())}
        for key in sorted(_ids, key=len, reverse=True):
            out, k = re.subn(re.escape(key), "that", out, flags=re.I)
            n += k
        out, k = _OBJ_ID_RE.subn("that", out)
        n += k
        if n:
            self._n["say_redactions"] += n
        flat = re.sub(r"[^a-z0-9 ]+", " ", out.lower())
        flat = " ".join(flat.split())
        if any(p in flat for p in _GENERIC_SAY):
            self._n["human_said_generic"] += 1
        if any(p in flat for p in _SPATIAL_SAY) or re.search(r"\b\d+\b", flat):
            self._n["say_spatial"] += 1
        return out

    def _vis_absorb(self, obs: dict) -> None:
        """Fold this step's visibility MEASUREMENT into the counters, and remember it for the transcript.

        Pure measurement -- nothing branches on it. The env computes ``vischeck`` and ships it, but the
        only report on the env side is a ``logger.info`` that the default root level swallows, so without
        this the whole round would produce no data on the question it exists to answer: how big is the
        visible set, and how often does the person act on something the wearer never saw.
        """
        vc = obs.get("vischeck")
        self._last_vischeck = vc if isinstance(vc, dict) else None
        if not isinstance(vc, dict) or not vc.get("n_entities"):
            return
        self._n["vis_steps"] += 1
        self._n["vis_n_sum"] += int(vc.get("n_visible") or 0)
        seen = vc.get("cmd_all_visible")
        if seen is not None:
            self._n["cmd_vis_checked"] += 1
            if not seen:
                self._n["cmd_unseen"] += 1

    def _merge_carried_say(self, say: str) -> str:
        """Everything the assistant has said since the human was LAST actually asked, oldest first.

        A tick that drains the queue never calls the 32B, so speech spoken on that tick has no path to
        the human at all -- it would be dropped forever. Buffer it there, deliver it here.
        """
        parts = [s for s in (*self._carry_say, (say or "").strip()) if s]
        self._carry_say = []
        return " ".join(parts)

    def _requeue_say(self, say: str) -> None:
        """Undo a delivery the 32B never actually received (the HTTP call failed). Same reason as
        _merge_carried_say: speech the assistant spent a turn on must not vanish because of transport.
        """
        say = (say or "").strip()
        if say:
            self._carry_say.insert(0, say)

    @staticmethod
    def _repair_one_arg_put(cmd: Optional[str], held: list) -> Optional[str]:
        """Fill in the missing FIRST object of a one-argument put.

        The 32B renders "put it in the bucket" as ``[Put] <ice_bucket_42> (1)`` -- it names the
        destination and omits the thing in its hand, because that is how the sentence works in English.
        The grammar needs both, so the sim rejects it (``[PUT] expects 2 object arg(s), got 1``). Telling
        it twice in the prompt did not stop it: run st1 still emitted 28 of these in 32 rollouts even with
        the two-argument shape spelled out in the output format AND the rejection echoed back to it.

        The intent is unambiguous whenever exactly ONE object is held -- "put it in X" can only mean that
        object -- so repair it here instead of arguing with the model. With zero or several things held
        the sentence really is ambiguous, so leave it alone and let the rejection stand.
        """
        if not cmd or len(held) != 1:
            return cmd
        m = re.match(r"^\s*\[(Put\w*)\]\s*<([^>]+)>\s*\((\d+)\)\s*$", cmd, re.I)
        if not m:
            return cmd  # already two-argument, or not a put at all
        verb, target, idx = m.group(1), m.group(2), m.group(3)
        if target == held[0]:
            return cmd  # names the held object, not a destination -- genuinely malformed, don't guess
        fixed = f"[{verb}] <{held[0]}> (1) <{target}> ({idx})"
        print(
            f"[isaac_human] repaired one-arg put {cmd!r} -> {fixed!r} (holding {held[0]})",
            flush=True,
        )
        return fixed

    @staticmethod
    def _one_arg_put_repair_audit(
        original: Optional[str], repaired: Optional[str], held: list
    ) -> Optional[dict[str, object]]:
        if not original or repaired == original or len(held) != 1:
            return None
        match = re.match(
            r"^\s*\[(Put\w*)\]\s*<([^>]+)>\s*\((\d+)\)\s*$",
            original,
            re.I,
        )
        if match is None:
            return None
        verb, destination, destination_index = match.groups()
        expected = f"[{verb}] <{held[0]}> (1) <{destination}> ({destination_index})"
        if repaired != expected:
            return None
        return {
            "kind": "unique_held_put_source",
            "original_command": original,
            "repaired_command": repaired,
            "held_source": held[0],
            "destination": destination,
            "destination_index": int(destination_index),
            "verb": verb,
        }

    def _extract_commands(self, scope: str, objs: list) -> list:
        """Every command in one ``<action>`` block, in order. PURE: no state but the repair counters.

        The human now writes the whole short sequence that carries out one instruction ("walk over,
        then pick it up") instead of being re-prompted for every primitive. The caller plays them one
        per tick and still polls the assistant between each, so the assistant keeps its per-action
        say/stay-silent decision; an assistant interrupt no longer discards the plan, it is simply
        delivered on the next real call.

        Three tiers per line, strictest first: the STRICT grammar, then a bare zero-argument verb, then
        the lenient repair. Segments are split on newlines AND on VirtualHome's ", " --
        command_grammar.split_program accepts both, so a one-liner must not be truncated to its first
        verb.

        Separated from ``_human_command`` so the recorded transcripts can be re-parsed through the REAL
        code path (see ``bench_v2/reparse_dialogue.py``) instead of a copy of it that can drift.
        """
        cmds: list = []
        for line in [s for raw in scope.splitlines() for s in raw.split(", ")]:
            if "[" not in line:
                continue
            # finditer, not search: two commands written on ONE line are both real, and taking only the
            # first silently threw the rest of the human's plan away.
            found = [
                m.group(0).strip()
                for m in re.finditer(r"\[[A-Za-z_]+\](?:\s*<[^>]+>\s*\(\d+\))+", line)
            ]
            if found:  # verb + AT LEAST ONE object
                cmds.extend(found)
                continue
            bare = re.fullmatch(r"\s*\[([A-Za-z_]+)\]\s*", line)
            if bare:
                canonical = _grammar_tables()["words"].get(bare.group(1).lower())
                if canonical in _grammar_tables()["loco"]:
                    cmds.append(f"[{bare.group(1)}]")
                    continue
            mz = re.search(r"\[([A-Za-z_]+)\]", line)
            if mz and mz.group(1).upper() in _grammar_tables()["zero"]:
                cmds.append(f"[{mz.group(1)}]")
                continue
            # A verb with a MALFORMED object ("[Grab] microwave"). The strict pattern used to require one
            # <name> (id), so these fell through to the lenient mapper; loosening it to `*` for the
            # zero-arg verbs swallowed them as BARE verbs instead, and the grammar rejects every one.
            # Runs p5/p6/p7: 0 of 48 human actions were legal -- all bare `[Grab]` / `[Walk]` /
            # `[LookAround]` -- while the reasoning right beside them named the object it meant.
            one = self._lenient_command(line, objs)
            if one:
                cmds.append(one)
        if not cmds:
            # With no <action> tag ``scope`` is the WHOLE reply, so the non-action spans have to come
            # out before the lenient scan or it invents commands from the human's thoughts. Caught live:
            # reasoning that read "I am holding toy_figure_242 ... I need to ask for the next step"
            # became [Walk] <toy_figure_242> (1) -- walk to the thing in your own hand -- and ran.
            one = self._lenient_command(self._strip_nonaction(scope), objs)
            cmds = [one] if one else []
        return cmds

    @staticmethod
    def _lenient_verb(text: str) -> Optional[str]:
        """The verb a loose line MEANT, as a canonical bracket spelling; None if it names none.

        Two tiers, most specific first. A BRACKETED token is the strongest signal there is -- the
        person wrote `[LookAt] item_1 (1)` and only fluffed the `<>` -- and it is checked before any
        bare word so a verb quoted mid-sentence cannot outvote the one actually issued.
        """
        tab = _grammar_tables()
        mb = re.search(r"\[([A-Za-z_]+)\]", text)
        if mb:
            canon = tab["words"].get(mb.group(1).lower())
            if canon:
                return canon
            return None  # a BRACKETED verb the grammar does not have is a hallucination, not a typo
        low = text.lower()
        # The two sweeps keep their historical short-circuit: they are the only way out of "I cannot
        # see anything to name", and under ISAAC_VIS_GATE that is a state the person will be in.
        if "lookscan" in low or "look down" in low or "scan the floor" in low:
            return "LookScan"
        if "lookaround" in low or "look around" in low:
            return "LookAround"
        mw = tab["re"].search(low)
        return tab["words"].get(mw.group(1)) if mw else None

    @staticmethod
    def _named_objects(low: str, objs: list) -> list:
        """The obs object KEYS a line mentions, in order of appearance, longest match wins.

        The longest-match rule matters: `desk_tmluxa_1` is a substring of `desk_tmluxa_10`, so a plain
        scan binds `[LookAt] desk_tmluxa_10` to the wrong desk at the same offset.
        """
        hits = sorted(
            ((low.find(o.lower()), -len(o), o) for o in objs if o.lower() in low),
        )
        out: list = []
        spans: list[tuple[int, int]] = []
        for pos, neg_len, key in hits:
            end = pos - neg_len
            if any(pos >= s and end <= e for s, e in spans):
                continue  # contained in a longer key already bound at this offset
            spans.append((pos, end))
            out.append(key)
        return out

    def _lenient_command(self, text: str, objs: list) -> Optional[str]:
        """Robustness fallback: the frozen 32B often emits a LOOSE action ("Walk trash_can_116",
        "grab the can_of_soda_115", "[LookAt] item_1 (1)") instead of the strict ``[Verb] <name> (id)``
        grammar. Map the verb it named + the obs object KEYS it mentions to the canonical command so the
        action path still executes. (System-robustness, my lane; the human's DECISION quality / prompt is
        the user's model-side lane.)

        The verb table is DERIVED from command_grammar (see ``_grammar_tables``) -- the whole 21-verb
        grammar plus the four locomotion verbs, not the three this used to know. Arity comes from the
        grammar too, so a repair is emitted with the argument count the sim will actually accept.
        """
        verb = self._lenient_verb(text)
        if not verb:
            return None
        tab = _grammar_tables()
        n_args = tab["arity"].get(verb, 1)
        if verb in tab["loco"]:
            # Locomotion names no entity; its optional argument is a SCALAR, and dropping it would
            # silently turn `[Forward] 3` into one pace. Carry the tail through verbatim and let the
            # grammar validate it.
            tail = re.sub(r"^.*?\]", "", text, count=1).strip() if "]" in text else ""
            m = re.match(r"^([-+]?(?:\d+\.?\d*|\.\d+)\s*[A-Za-z]*)", tail)
            cmd = f"[{verb}] {m.group(1).strip()}" if m else f"[{verb}]"
            self._note_repair(verb, text, cmd)
            return cmd
        if n_args == 0:
            cmd = f"[{verb}]"
            self._note_repair(verb, text, cmd)
            return cmd
        named = self._named_objects(text.lower(), objs)
        if not named:
            return None
        # As many arguments as were actually named, capped at the arity. Emitting a SHORT put on
        # purpose: `_repair_one_arg_put` fills the held object in afterwards, which is the one case
        # where the missing argument is unambiguous. Anything else short is rejected by the grammar
        # with a reason the person is told -- which is the point: a rejection is feedback, a silent
        # drop is not.
        args = " ".join(f"<{k}> (1)" for k in named[:n_args])
        cmd = f"[{verb}] {args}"
        self._note_repair(verb, text, cmd)
        return cmd

    def _note_repair(self, verb: str, raw: str, cmd: str) -> None:
        """Record one lenient repair: a counter per verb, plus the provenance the dialogue log wants.

        ``{strict_parse, repaired_from, repaired_to}`` is the fact that survives the rewrite; without
        it a repaired command is indistinguishable from one the model wrote correctly, and the 12 % of
        turns this function now rescues would be invisible in the transcript.
        """
        self._n["lenient_repair"] = self._n.get("lenient_repair", 0) + 1
        key = f"lenient_repair_{verb}"
        self._n[key] = self._n.get(key, 0) + 1
        self._repairs.append(
            {"repaired_from": raw.strip()[:400], "repaired_to": cmd, "verb": verb}
        )

    def repair_provenance(self, cmd: Optional[str]) -> dict:
        """``{strict_parse, repaired_from, repaired_to, repair_kind}`` for one emitted command."""
        for rec in reversed(self._repairs):
            if rec["repaired_to"] == cmd:
                return {
                    "strict_parse": False,
                    "repaired_from": rec["repaired_from"],
                    "repaired_to": cmd,
                    "repair_kind": "lenient_verb",
                }
        return {
            "strict_parse": True,
            "repaired_from": None,
            "repaired_to": None,
            "repair_kind": None,
        }

    @staticmethod
    def _obs_frames(obs: dict) -> list:
        """The base64 FPV frames in an observation: the multi-frame segment (``frames``) if present, else the
        single final frame, else empty (an idle tick with no render)."""
        return observation_frames(obs)
