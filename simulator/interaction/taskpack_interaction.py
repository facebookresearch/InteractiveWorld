# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Data-driven TaskPack adapter for the six-family interactive evaluation."""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from .interaction_profiles import profile_for


_COMMAND_RE = re.compile(r"^\s*\[([^\]]+)\](.*)$")
_ARG_RE = re.compile(r"<([^>]+)>\s*\((-?\d+)\)")
_ANCHOR_INDEX_RE = re.compile(r"^[A-Za-z]+(\d+)$")
_RELATIONAL_COMMA_RE = re.compile(
    r",\s*(?:next to|beside|by|near|in front of|behind|inside|in|on)\b",
    re.IGNORECASE,
)
_ACTS_AS_RE = re.compile(
    r"\(([^()]+)\)\s+acts as\s+(?:an?|the)\s+(.+?)[.]?$", re.IGNORECASE
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_MOTOR_CLAIM_RE = re.compile(
    r"^(?:(?:i|we)\s+(?:am|are|was|were)\s+able\s+to\s+|"
    r"i(?:'m| am| will|'ll|'ve| have)?\s+"
    r"(?:(?:already|just|successfully)\s+)?)?(?:"
    r"walk(?:ed|ing)?|run|ran|running|turn(?:ed|ing)?|mov(?:e|ed|ing)|"
    r"head(?:ed|ing)?|reach(?:ed|ing)?|grab(?:bed|bing)?|pick(?:ed|ing)?|plac(?:e|ed|ing)|"
    r"put(?:ting)?|open(?:ed|ing)?|clos(?:e|ed|ing)|look(?:ed|ing)?|"
    r"find(?:ing)?|found|"
    r"scan(?:ned|ning)?|drop(?:ped|ping)?|push(?:ed|ing)?|pull(?:ed|ing)?|"
    r"sit(?:ting)?|sat|stand(?:ing)?|stood|squat(?:ted|ting)?|cut(?:ting)?|"
    r"stir(?:red|ring)?|pour(?:ed|ing)?|wip(?:e|ed|ing)|cook(?:ed|ing)?|"
    r"attach(?:ed|ing)?|detach(?:ed|ing)?|rais(?:e|ed|ing)|flip(?:ped|ping)?"
    r")\b",
    re.IGNORECASE,
)
_COMPLETED_MOTOR_CLAIM_RE = re.compile(
    r"^(?:(?:i(?:'ve| have)?)\s+(?:(?:already|just|successfully)\s+)?)?(?:"
    r"(?P<verb>walked|ran|turned|moved|headed|reached|grabbed|picked|placed|put|"
    r"opened|closed|looked|scanned|dropped|pushed|pulled|sat|stood|squatted|"
    r"cut|stirred|poured|wiped|cooked|attached|detached|raised|flipped)"
    r")\b",
    re.IGNORECASE,
)
_COMPLETED_MOTOR_COMMANDS: Mapping[str, frozenset[str]] = {
    "attached": frozenset({"ASSEMBLE", "ATTACH", "INSTALL"}),
    "closed": frozenset({"CLOSE", "CLOSEDOOR"}),
    "cooked": frozenset({"COOK"}),
    "cut": frozenset({"CUT"}),
    "detached": frozenset({"DETACH"}),
    "dropped": frozenset({"DROP", "PUTONFLOOR", "RELEASE"}),
    "flipped": frozenset({"FLIP"}),
    "grabbed": frozenset({"GRAB"}),
    "headed": frozenset({"WALK"}),
    "looked": frozenset({"LOOKAROUND", "LOOKAT", "LOOKSCAN"}),
    "moved": frozenset(
        {"DROP", "GRAB", "PUT", "PUTBACK", "PUTIN", "PUTINSIDE", "PUTON", "WALK"}
    ),
    "opened": frozenset({"OPEN", "OPENDOOR"}),
    "picked": frozenset({"GRAB"}),
    "placed": frozenset({"DROP", "PUT", "PUTBACK", "PUTIN", "PUTINSIDE", "PUTON"}),
    "poured": frozenset({"POUR"}),
    "pulled": frozenset({"PULL"}),
    "pushed": frozenset({"PUSH"}),
    "put": frozenset({"DROP", "PUT", "PUTBACK", "PUTIN", "PUTINSIDE", "PUTON"}),
    "raised": frozenset({"RAISE"}),
    "ran": frozenset({"WALK"}),
    "reached": frozenset({"WALK"}),
    "sat": frozenset({"SIT"}),
    "scanned": frozenset({"LOOKAROUND", "LOOKSCAN"}),
    "squatted": frozenset({"SQUAT"}),
    "stirred": frozenset({"STIR"}),
    "stood": frozenset({"STAND", "STANDUP"}),
    "turned": frozenset({"TURN", "TURNAROUND", "TURNLEFT", "TURNRIGHT"}),
    "walked": frozenset({"WALK"}),
    "wiped": frozenset({"WIPE"}),
}
_NARRATE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "currently",
        "in",
        "inside",
        "is",
        "it",
        "near",
        "of",
        "on",
        "put",
        "the",
        "this",
        "to",
        "toward",
        "towards",
        "walk",
        "you",
        "your",
    }
)
_WATCH_UNIT_END = frozenset(
    {
        "ASSEMBLE",
        "ATTACH",
        "CLOSE",
        "COOK",
        "CUT",
        "DROP",
        "INSTALL",
        "POUR",
        "PUT",
        "PUTBACK",
        "PUTIN",
        "PUTINSIDE",
        "PUTON",
        "RELEASE",
        "STIR",
        "WIPE",
    }
)
_WATCH_MOTOR_RECOVERY_VERBS = frozenset(
    {
        "LOOKAROUND",
        "LOOKAT",
        "LOOKSCAN",
        "OPEN",
        "OPENDOOR",
        "STANDUP",
        "WALK",
    }
)
_GUIDE_VISIBILITY_GATED_VERBS = frozenset({"GRAB", "LOOKAT", "WALK"})
_INSPECTED_MOVABLE_GROUNDING_FAMILIES = frozenset({"GUIDE", "NARRATE", "ORCHESTRATE"})
_FIND_VISIBILITY_GATED_VERBS = frozenset({"GRAB", "LOOKAT", "WALK"})
_FIND_PLACEMENT_VERBS = frozenset(
    {"PUT", "PUTBACK", "PUTIN", "PUTINSIDE", "PUTON", "PUTONFLOOR"}
)
_ASK_PLACEMENT_VERBS = _FIND_PLACEMENT_VERBS | frozenset(
    {"INSERT", "PUTOBJBACK", "PUT_ON_FLOOR"}
)
_FIND_SAFE_UNKEYED_VERBS = frozenset(
    {"LOOKAROUND", "LOOKSCAN", "TURN", "TURNLEFT", "TURNRIGHT", "WAIT"}
)
_NARRATE_EXPLICIT_ACTION_CUES: Mapping[str, frozenset[str]] = {
    "CLOSE": frozenset({"close", "shut"}),
    "GRAB": frozenset(
        {"collect", "fetch", "get", "grab", "lift", "pick", "retrieve", "take"}
    ),
    "LOOKAT": frozenset(
        {"check", "examine", "inspect", "look", "read", "scan", "search", "show"}
    ),
    "OPEN": frozenset({"open"}),
    "PLACE": frozenset({"bring", "leave", "place", "put", "set", "store"}),
    "WALK": frozenset(
        {"approach", "go", "head", "move", "navigate", "proceed", "step", "walk"}
    ),
    "WIPE": frozenset({"clean", "sweep", "wipe"}),
}
_NARRATE_LANDMARK_REFERENT_RE = re.compile(
    r"\b(?:there|(?:same|that)\s+(?:area|bin|cabinet|counter|countertop|desk|"
    r"door|drawer|fridge|refrigerator|shelf|surface|table))\b",
    re.IGNORECASE,
)
_FIND_ACTION_CUES: Mapping[str, frozenset[str]] = {
    "CLOSE": frozenset({"close", "put", "shut", "store"}),
    "GRAB": frozenset(
        {"bring", "collect", "fetch", "get", "grab", "lift", "pick", "retrieve", "take"}
    ),
    "DROP": frozenset({"drop", "place", "put", "release"}),
    "LOOKAT": frozenset(
        {
            "check",
            "examine",
            "get",
            "grab",
            "inspect",
            "look",
            "read",
            "retrieve",
            "see",
            "take",
        }
    ),
    "OPEN": frozenset(
        {
            "check",
            "get",
            "inspect",
            "look",
            "open",
            "place",
            "put",
            "retrieve",
            "search",
            "start",
            "store",
            "take",
        }
    ),
    "PLACE": frozenset(
        {"bring", "leave", "place", "put", "return", "set", "store", "take"}
    ),
    "TURN": frozenset({"look", "rotate", "scan", "turn"}),
    "WALK": frozenset(
        {
            "approach",
            "back",
            "bring",
            "check",
            "collect",
            "fetch",
            "get",
            "go",
            "grab",
            "head",
            "inspect",
            "look",
            "move",
            "open",
            "place",
            "pick",
            "put",
            "retrieve",
            "return",
            "search",
            "start",
            "take",
            "use",
            "walk",
        }
    ),
}
_GROUNDING_WORD_ALIASES = {
    "cabinets": "cabinet",
    "counter": "countertop",
    "counters": "countertop",
    "cupboard": "cabinet",
    "cupboards": "cabinet",
    "fridges": "fridge",
    "refrigerator": "fridge",
    "refrigerators": "fridge",
    "shelves": "shelf",
    "tables": "table",
    "worktop": "countertop",
    "worktops": "countertop",
}
_GROUNDING_LABEL_PREFIXES = frozenset(
    {"card", "item", "label", "option", "sheet", "unit"}
)
# These aliases contain only stable, Human-visible landmark attributes. Every reasonably
# nameable candidate is included, including distractors, so this table cannot act as an
# answer-key allowlist. Target-location facts and oracle-order labels are deliberately absent.
_FIND_GROUNDING_ALIASES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "FIND-13004": {
        "surf_bed": (
            "bedroom cabinet",
            "on the bedroom cabinet",
            "bedroom cabinet top",
            "bedroom cabinet surface",
        ),
        "unit_store": (
            "bedroom cabinet",
            "in the bedroom cabinet",
            "bedroom storage cabinet",
            "metal bedroom cabinet",
        ),
        "surf_bath": ("bathroom counter", "bathroom countertop"),
    },
    "FIND-14001": {
        "surf_outdoor": ("outdoor table", "garden table", "table in the garden"),
        "unit_fridge": (
            "kitchen refrigerator",
            "kitchen fridge",
            "refrigerator",
            "fridge",
        ),
        "surf_kitchen": ("kitchen counter", "kitchen countertop"),
        "surf_start": ("bedroom breakfast table", "bedroom table"),
        "tray_final": ("kitchen tray", "tray on the kitchen counter", "tray"),
    },
    "FIND-14002": {
        "unit_cabinet": (
            "living-room cabinet",
            "cabinet in the living room",
        ),
        "surf_prep": ("living-room table", "table in the living room"),
        "surf_phone": ("kitchen counter", "kitchen countertop"),
    },
    "FIND-14015": {
        "surf_original": ("living-room table",),
        "surf_remote": ("kitchen counter", "kitchen countertop"),
        "surf_phone": ("living-room coffee table", "coffee table"),
    },
    "FIND-16000": {
        "surf_prep": (
            "kitchen counter",
            "kitchen countertop",
        ),
        "unit_fridge": (
            "kitchen refrigerator",
            "kitchen fridge",
            "refrigerator",
            "fridge",
        ),
        "surf_dining": ("garden table", "outdoor table"),
        "surf_phone": ("office cabinet", "kitchen cabinet"),
    },
    "FIND-18010": {
        "unit_left": ("left bedside cabinet", "left cabinet", "cabinet on the left"),
        "unit_right": (
            "right bedside cabinet",
            "right cabinet",
            "cabinet on the right",
        ),
        "surf_start": ("hospital bed", "infirmary bed"),
        "surf_infirmary": ("infirmary desk", "desk here", "this desk"),
        "surf_break": ("corridor bench", "hallway bench"),
    },
    "FIND-22000": {
        "locker_left": (
            "left cabinet",
            "left low cabinet",
            "first low cabinet",
            "leftmost cabinet",
        ),
        "locker_middle": (
            "middle cabinet",
            "middle low cabinet",
            "center low cabinet",
            "second low cabinet",
        ),
        "locker_right": (
            "right cabinet",
            "right low cabinet",
            "last low cabinet",
            "third low cabinet",
            "rightmost cabinet",
        ),
        "surf_result": ("utility-room worktop", "worktop", "countertop"),
    },
    "FIND-28003": {
        "unit_fridge": (
            "kitchen refrigerator",
            "kitchen fridge",
            "refrigerator",
            "fridge",
        ),
        "surf_prep": ("kitchen bar", "kitchen counter"),
        "surf_dining": (
            "dining table",
            "breakfast table",
            "this dining table",
            "this table",
        ),
        "surf_living": ("living-room coffee table", "coffee table"),
    },
    "FIND-28005": {
        "fridge_remote": (
            "hallway refrigerator",
            "hallway fridge",
            "refrigerator in the hallway",
            "refrigerator",
            "fridge",
        ),
        "surf_living": (
            "table in the first living room",
            "first living-room table",
        ),
        "fridge_start": (
            "kitchen refrigerator",
            "kitchen fridge",
            "refrigerator",
            "fridge",
        ),
        "surf_work": (
            "worktable in the second living room",
            "second living-room worktable",
            "this worktable",
            "current worktable",
            "work table in the second living room",
            "second living-room work table",
            "this work table",
            "current work table",
        ),
    },
    "FIND-30001": {
        "cabinet_origin": (
            "kitchen cabinet",
            "matching cabinet in the kitchen",
        ),
        "cabinet_wrong": ("bedroom cabinet", "matching cabinet in the bedroom"),
        "surf_middle": ("living-room table", "table in the living room"),
        "surf_result": ("kitchen counter", "kitchen countertop"),
    },
    "FIND-40000": {
        "unit_files": (
            "closed low file cabinet in the office",
            "office file cabinet",
            "low filing cabinet",
        ),
        "surf_office": ("office desk", "workstation desk"),
        "surf_clipboard": ("side desk", "office side desk"),
        "surf_kitchen": ("break-room table", "table in the break room"),
    },
    "FIND-81013": {
        "fruit_table": (
            "kitchen table",
            "fruit table",
            "table edge",
        ),
        "unit_fridge": (
            "kitchen refrigerator",
            "kitchen fridge",
            "refrigerator",
            "fridge",
        ),
        "surf_work": ("living-room table", "work table"),
        "surf_result": ("kitchen counter", "kitchen countertop"),
    },
    "FIND-87000": {
        "bedroom_folder_shelf": (
            "long wooden shelf in the bedroom",
            "bedroom shelf",
        ),
        "bedroom_desk": ("bedroom desk",),
        "bedroom_result": (
            "bedroom nightstand",
            "nightstand in the bedroom",
            "nightstand",
        ),
        "bathroom_checkpoint": ("bathroom shelf", "shelf in the bathroom"),
    },
    "FIND-87004": {
        "surf_2": (
            "breakfast table in the dining room",
            "dining-room breakfast table",
        ),
        "unit_a": (
            "kitchen refrigerator",
            "kitchen fridge",
            "refrigerator",
            "fridge",
        ),
        "surf_1": ("kitchen hydration cabinet", "kitchen cabinet"),
        "seat_call": ("dining-room armchair", "armchair in the dining room"),
    },
    "FIND-91002": {
        "surf_remote": (
            "living-room coffee table",
            "coffee table in the living room",
        ),
        "surf_dining": ("breakfast table", "dining table"),
        "surf_spiced": ("kitchen counter", "kitchen countertop"),
        "surf_white": ("kitchen counter", "kitchen countertop"),
        "surf_box": ("kitchen counter", "kitchen countertop"),
        "surf_lid_start": ("kitchen counter", "kitchen countertop"),
        "surf_prep": ("kitchen counter", "kitchen countertop"),
    },
    "FIND-97003": {
        "unit_store": (
            "other living-room cabinet",
            "living-room storage cabinet",
        ),
        "surf_living": ("breakfast table", "living-room breakfast table"),
        "surf_living_aux": ("nearby console table", "console table"),
        "surf_lid_rest": ("nearby coffee table", "coffee table"),
        "surf_white": ("kitchen counter", "kitchen countertop"),
        "surf_black": ("kitchen counter", "kitchen countertop"),
        "surf_box": ("kitchen counter", "kitchen countertop"),
        "surf_blue": ("kitchen counter", "kitchen countertop"),
        "surf_lid_start": ("kitchen counter", "kitchen countertop"),
    },
    "FIND-101000": {
        "cabinet_kitchen": (
            "kitchen cabinet",
            "low cabinet in the kitchen",
        ),
        "cabinet_living": (
            "living-room cabinet",
            "cabinet in the first living room",
            "cabinet in the outer living room",
            "light wood low cabinet next to a brown armchair",
            "light wood low cabinet beside a brown armchair",
            "light wood cabinet next to a brown armchair",
            "cabinet beside the brown armchair",
            "cabinet by the brown armchair",
            "cabinet you just walked past",
            "light wood cabinet",
        ),
        "cabinet_bedroom": (
            "innermost living-room cabinet",
            "inner living-room cabinet",
            "cabinet in the second living room",
        ),
        "surf_result": ("kitchen counter", "kitchen countertop"),
    },
}
_FIND_TARGET_ALIASES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "FIND-13004": {"eraser_paper": ("eraser", "rubber eraser", "paper eraser")},
    "FIND-14001": {
        "milk_garden": ("coconut milk", "coconut-milk bottle", "milk bottle"),
        "milk_fridge": ("coconut milk", "coconut-milk bottle", "milk bottle"),
    },
    "FIND-14002": {"oil": ("sesame oil", "oil bottle")},
    "FIND-14015": {"gift_target": ("other gift box", "missing gift box")},
    "FIND-16000": {"mayonnaise": ("mayonnaise", "mayonnaise bottle")},
    "FIND-18010": {"thermometer": ("thermometer", "digital thermometer")},
    "FIND-22000": {"lunch": ("lunchbox", "lunch box", "packed lunch")},
    "FIND-28003": {
        "box": ("box", "food box", "tupperware", "closed box"),
        "mushrooms": ("shiitake", "shiitake mushroom", "mushroom"),
    },
    "FIND-28005": {
        "milk_first": ("first-shelf carton", "first milk carton", "milk carton"),
        "milk_second": ("second-shelf carton", "second milk carton", "milk carton"),
    },
    "FIND-30001": {"wallet": ("wallet",)},
    "FIND-40000": {"equipment_form": ("equipment storage form", "equipment form")},
    "FIND-81013": {"apple": ("apple",)},
    "FIND-87000": {"folder": ("blue guest folder", "blue folder")},
    "FIND-87004": {"part_1": ("cola", "cola bottle", "red cola bottle")},
    "FIND-91002": {"spiced": ("spiced bottle", "spiced rum bottle")},
    "FIND-97003": {
        "squeegee_black": ("black squeegee", "black one"),
        "squeegee_blue": ("blue squeegee", "blue one"),
    },
    "FIND-101000": {"sponge": ("cleaning sponge", "yellow sponge", "sponge")},
}
_FIND_DESTINATION_ALIASES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "FIND-13004": {"unit_store": ("bedroom storage cabinet", "bedroom cabinet")},
    "FIND-14001": {"tray_final": ("tray on the kitchen counter", "kitchen tray")},
    "FIND-14002": {"unit_cabinet": ("living-room cabinet",)},
    "FIND-14015": {"surf_remote": ("kitchen counter", "kitchen countertop")},
    "FIND-16000": {
        "unit_fridge": (
            "kitchen refrigerator",
            "kitchen fridge",
            "refrigerator",
            "fridge",
        )
    },
    "FIND-18010": {"surf_infirmary": ("infirmary desk", "this desk")},
    "FIND-22000": {"surf_result": ("worktop", "utility-room worktop")},
    "FIND-28003": {"surf_dining": ("dining table", "this dining table", "this table")},
    "FIND-28005": {
        "surf_work": (
            "this worktable",
            "current worktable",
            "this work table",
            "current work table",
        )
    },
    "FIND-30001": {"surf_result": ("kitchen counter", "kitchen countertop")},
    "FIND-40000": {"surf_kitchen": ("break-room table",)},
    "FIND-81013": {"surf_result": ("kitchen counter", "kitchen countertop")},
    "FIND-87000": {"bedroom_result": ("bedroom nightstand", "nightstand")},
    "FIND-87004": {
        "unit_a": (
            "kitchen refrigerator",
            "kitchen fridge",
            "refrigerator",
            "fridge",
        )
    },
    "FIND-91002": {"surf_dining": ("breakfast table",)},
    "FIND-97003": {
        "surf_living": ("breakfast table",),
        "surf_living_aux": ("nearby console table", "console table"),
    },
    "FIND-101000": {"surf_result": ("kitchen counter", "kitchen countertop")},
}
_FIND_OBJECT_ALIASES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "FIND-13004": {
        "eraser_folder": (
            "other eraser",
            "folder eraser",
            "rubber eraser",
        ),
        "paper": ("loose paper", "paper sheet"),
        "folder": ("folder",),
    },
    "FIND-14001": {
        "plate": ("plate",),
        "phone": ("phone", "cell phone"),
    },
    "FIND-14002": {
        "sauce": ("salsa", "salsa bottle", "other bottle"),
        "phone": ("phone", "cell phone"),
    },
    "FIND-14015": {
        "gift_decoy": ("gift box", "visible gift box"),
        "cup": ("cup", "mug"),
        "phone": ("phone", "cell phone"),
    },
    "FIND-16000": {
        "jam": ("jam", "strawberry jam", "jam jar"),
        "plate": ("plate",),
        "phone": ("phone", "cell phone"),
    },
    "FIND-18010": {"drink": ("drink", "cola", "cola bottle")},
    "FIND-28003": {
        "lid": ("lid", "food-box lid"),
        "leek": ("leek",),
        "plate": ("plate",),
    },
    "FIND-28005": {
        "milk_third": ("third milk carton", "empty milk carton", "milk carton"),
        "mug": ("mug",),
    },
    "FIND-30001": {"phone": ("phone", "cell phone")},
    "FIND-40000": {
        "food_form": ("food storage form", "food form"),
        "clipboard": ("clipboard",),
        "keyboard": ("keyboard",),
    },
    "FIND-81013": {
        "banana": ("banana",),
        "watermelon": ("watermelon", "half watermelon"),
        "pomelo": ("pomelo",),
        "work_clipboard": ("clipboard", "work clipboard"),
    },
    "FIND-87000": {
        "cover_folder": ("brown folder", "covering folder"),
        "apple": ("apple", "red apple"),
    },
    "FIND-87004": {
        "item_1": ("clipboard",),
        "item_2": ("blue water bottle", "water bottle"),
        "item_3": ("green mug", "mug"),
        "item_4": ("phone", "cell phone"),
    },
    "FIND-91002": {
        "box": ("box", "lidded box", "closed box", "storage box"),
        "lid": ("lid", "box lid", "lidded box"),
        "white": ("white bottle", "white rum bottle"),
        "prep_plate": ("plate", "prep plate"),
    },
    "FIND-97003": {
        "box": ("box", "closed box", "storage box"),
        "lid": ("lid", "box lid", "closed box"),
        "squeegee_white": ("white squeegee", "white one"),
    },
}
# Every mapping below restates a destination already requested aloud in the fixed opening.
_FIND_PUBLIC_TARGET_DESTINATIONS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "FIND-13004": {"eraser_paper": ("unit_store",)},
    "FIND-14001": {
        "milk_garden": ("tray_final",),
        "milk_fridge": ("tray_final",),
    },
    "FIND-14002": {"oil": ("unit_cabinet",)},
    "FIND-14015": {"gift_target": ("surf_remote",)},
    "FIND-16000": {"mayonnaise": ("unit_fridge",)},
    "FIND-18010": {"thermometer": ("surf_infirmary",)},
    "FIND-22000": {"lunch": ("surf_result",)},
    "FIND-28003": {
        "box": ("surf_dining",),
        "mushrooms": ("surf_dining",),
    },
    "FIND-28005": {
        "milk_first": ("surf_work",),
        "milk_second": ("surf_work",),
    },
    "FIND-30001": {"wallet": ("surf_result",)},
    "FIND-40000": {"equipment_form": ("surf_kitchen",)},
    "FIND-81013": {"apple": ("surf_result",)},
    "FIND-87000": {"folder": ("bedroom_result",)},
    "FIND-87004": {"part_1": ("unit_a",)},
    "FIND-91002": {"spiced": ("surf_dining",)},
    "FIND-97003": {
        "squeegee_black": ("surf_living",),
        "squeegee_blue": ("surf_living_aux",),
    },
    "FIND-101000": {"sponge": ("surf_result",)},
}
_FIND_REFERENT_CONTINUATIONS = frozenset(
    {"back", "continue", "it", "same", "that", "there"}
)
_FIND_REFERENT_CHANGES = frozenset({"another", "different", "next", "other"})
_FIND_ROOM_WORDS = frozenset(
    {
        "bathroom",
        "bedroom",
        "corridor",
        "dining",
        "garden",
        "hallway",
        "infirmary",
        "kitchen",
        "living",
        "office",
        "room",
        "utility",
    }
)
_ASK_VISIBILITY_GATED_VERBS = frozenset({"GRAB"})
_ASK_ACTION_CUES: Mapping[str, frozenset[str]] = {
    "ANSWERPHONE": frozenset({"answer"}),
    "ATTACH": frozenset(
        {"assemble", "attach", "connect", "fasten", "install", "mount"}
    ),
    "CLOSE": frozenset({"close", "shut"}),
    "CONFIRM": frozenset({"confirm"}),
    "COOK": frozenset({"bake", "cook", "heat", "roast"}),
    "CUT": frozenset({"chop", "cut", "slice"}),
    "DETACH": frozenset({"detach", "disassemble", "disconnect", "remove"}),
    "DROP": frozenset({"drop", "release"}),
    "FALLTOFLOOR": frozenset({"drop", "fall", "knock", "tip"}),
    "FLIP": frozenset({"flip", "turn"}),
    "GRAB": frozenset(
        {
            "collect",
            "fetch",
            "get",
            "grab",
            "lift",
            "pick",
            "remove",
            "retrieve",
            "take",
        }
    ),
    "LOOKAT": frozenset(
        {
            "angle",
            "check",
            "examine",
            "inspect",
            "look",
            "read",
            "scan",
            "search",
            "show",
        }
    ),
    "OPEN": frozenset({"open"}),
    "PLACE": frozenset({"bring", "leave", "place", "put", "return", "set", "store"}),
    "POUR": frozenset({"empty", "pour"}),
    "PULL": frozenset({"pull"}),
    "PUSH": frozenset({"push"}),
    "RAISE": frozenset({"erect", "lift", "raise"}),
    "SIT": frozenset({"sit"}),
    "SQUAT": frozenset({"crouch", "squat"}),
    "STANDUP": frozenset({"stand"}),
    "STIR": frozenset({"mix", "stir"}),
    "TURNON": frozenset({"activate", "start", "switch", "turn"}),
    "TURNOFF": frozenset({"stop", "switch", "turn"}),
    "TURN": frozenset({"look", "rotate", "turn"}),
    "USECOMPUTER": frozenset({"computer", "type", "use"}),
    "WAIT": frozenset({"hold", "stay", "wait"}),
    "WIPE": frozenset({"clean", "wipe"}),
    "WALK": frozenset(
        {
            "approach",
            "bring",
            "carry",
            "forward",
            "go",
            "head",
            "move",
            "navigate",
            "proceed",
            "return",
            "step",
            "take",
            "walk",
        }
    ),
}
_ASK_OMISSION_ACTION_CUES: Mapping[str, frozenset[str]] = {
    "ATTACH": frozenset(
        {"assemble", "attach", "connect", "fasten", "install", "mount"}
    ),
    "CLOSE": frozenset({"close", "shut"}),
    "COOK": frozenset({"bake", "cook", "heat", "roast"}),
    "CUT": frozenset({"chop", "cut", "slice"}),
    "DETACH": frozenset({"detach", "disassemble", "disconnect"}),
    "DROP": frozenset({"drop", "release"}),
    "FALLTOFLOOR": frozenset({"fall", "knock", "tip"}),
    "FLIP": frozenset({"flip"}),
    "GRAB": frozenset({"collect", "fetch", "get", "grab", "pick", "retrieve"}),
    "LOOKAT": frozenset(
        {"check", "examine", "inspect", "read", "scan", "search", "show"}
    ),
    "OPEN": frozenset({"open"}),
    "PLACE": frozenset({"leave", "place", "put", "set", "store"}),
    "POUR": frozenset({"empty", "pour"}),
    "PULL": frozenset({"pull"}),
    "PUSH": frozenset({"push"}),
    "RAISE": frozenset({"erect", "raise"}),
    "SIT": frozenset({"sit"}),
    "SQUAT": frozenset({"crouch", "squat"}),
    "STIR": frozenset({"mix", "stir"}),
    "TURNON": frozenset({"activate", "start"}),
    "TURNOFF": frozenset({"stop"}),
    "TURN": frozenset({"rotate"}),
    "USECOMPUTER": frozenset({"computer", "type"}),
    "WAIT": frozenset({"hold", "stay", "wait"}),
    "WIPE": frozenset({"clean", "wipe"}),
    "WALK": frozenset(
        {
            "approach",
            "carry",
            "forward",
            "go",
            "head",
            "move",
            "navigate",
            "proceed",
            "step",
            "walk",
        }
    ),
}
_ASK_ONE_ENTITY_ACTIONS = frozenset(
    {
        "ANSWERPHONE",
        "CLOSE",
        "COOK",
        "FALLTOFLOOR",
        "OPEN",
        "RAISE",
        "SIT",
        "TURNON",
        "TURNOFF",
        "USECOMPUTER",
    }
)
_ASK_VISUALLY_GROUNDED_SINGLE_ENTITY_ACTIONS = _ASK_ONE_ENTITY_ACTIONS | {
    "FLIP",
    "GRAB",
    "LOOKAT",
}
_ASK_SPECIFIC_DESTINATION_RE = re.compile(
    r"\b(?:atop|in|inside|into|on|onto|to|toward|towards)\s+"
    r"(?:(?:a|an|that|the|this|your)\s+)?"
    r"(?!(?:anywhere|it|one|place|somewhere|something|there|thing)\b)"
    r"[a-z][a-z0-9-]*",
    re.IGNORECASE,
)
_ASK_TWO_ENTITY_ACTIONS = frozenset(
    {"ATTACH", "CUT", "DETACH", "POUR", "PULL", "PUSH", "STIR", "WIPE"}
)
_ORCHESTRATE_STRICT_ACTIONS = frozenset({"COOK", "CUT", "POUR", "TURNOFF", "TURNON"})
_ORCHESTRATE_RECOVERY_EVIDENCE_RE = re.compile(
    r"\b(?:collision|failed|failure|out[_ -]of[_ -]reach|path[_ -]blocked|"
    r"blocked|rejected|stuck|unable|unstable)\b|"
    r"\b(?:can|could|did)(?:\s+not|n't)\b",
    re.IGNORECASE,
)
_ORCHESTRATE_REPEAT_SEQUENCE_RE = re.compile(
    r"\b(?:do|repeat)\b(?:\s+\w+){0,4}\s+same\b", re.IGNORECASE
)
_ORCHESTRATE_PRESERVATION_RE = re.compile(
    r"\b(?:keep\s+every\s+clause\s+already\s+completed|"
    r"leave\s+(?:any|the)\s+completed\b|continue\s+with\s+the\s+"
    r"(?:first|next)\s+unfinished\b)",
    re.IGNORECASE,
)
_ASK_HUMAN_TASK_CONTEXT = (
    "You have already asked the fixed question and have no private answer, route, item order, "
    "destination plan, or recovery plan. Act only on the Assistant's latest explicit physical or "
    "inspection instruction. If it has answered the question but has not explicitly told you the "
    "next physical step, leave the action empty and ask what to do next."
)
_ASK_INTERNAL_INTERFACE_SPEECH_RE = re.compile(
    r"\b(?:(?:object|landmark)_\d+|(?:motor\s+)?(?:catalog|legend)|"
    r"(?:motor|action)\s+references?|exact\s+(?:reference\s+)?name|"
    r"reference\s+(?:label|name)s?|(?:private\s+)?action(?:-only)?\s+"
    r"(?:list|name|syntax)(?:es|s)?|"
    r"(?:command|object|simulator)\s+(?:id|key)s?|"
    r"exact\s+(?:id|key)|internal\s+(?:id|key|name)|listed\s+(?:id|key|location|"
    r"object|option)s?|available\s+(?:action|command|destination|location|object|"
    r"option|target)s?|(?:destination|location|target)\s+(?:choice|option)s?|"
    r"(?:my|the)\s+(?:choices|options)\s+(?:are|include)|"
    r"(?:i\s+)?(?:can(?:not|['’]t)|am\s+unable\s+to)\s+(?:directly\s+)?target|"
    r"target(?:ing)?\s+(?:the\s+)?(?:floor|wall)|"
    r"(?:do not|don't)\s+have\s+(?:the\s+)?(?:ability|action|command))\b",
    re.IGNORECASE,
)
_GUIDE_INTERNAL_INTERFACE_SPEECH_RE = re.compile(
    r"\b(?:(?:(?:a|the|this)\s+)?handle\s+(?:named|called)\b|"
    r"(?:motor|object|action|private)\s+handles?\b)",
    re.IGNORECASE,
)
_ASK_UNOBSERVABLE_ATTRIBUTE_SPEECH_RE = re.compile(
    r"\b(?:i|we)\s+(?:can\s+)?(?:feel|felt|smell|smelled|taste|tasted|weigh|weighed)\b|"
    r"\b(?:it|this|that|the\s+(?:item|object|package|packet|box|carton|bottle|food))\s+"
    r"(?:feels?|felt|smells?|smelled|tastes?|tasted|weighs?|weighed)\b|"
    r"\b(?:frozen\s+solid|(?:hot|cold|warm|cool)\s+to\s+(?:my\s+)?touch)\b|"
    r"\b(?:hidden|covered|concealed|underside|back(?:side)?)\s+"
    r"(?:label|text|writing)\s+(?:says?|reads?|shows?)\b",
    re.IGNORECASE,
)
_GUIDE_INTERFACE_CLARIFICATION = (
    "What does it look like, and where should I look for it?"
)
_GUIDE_UNOBSERVABLE_CLARIFICATION = (
    "I can't confirm that from what I can see. Can you judge it from the view?"
)
_GUIDE_LOCAL_COMPLETION_REPORT = "That step is done. What should I do next?"
_GUIDE_INFERENCE_CONCLUSION_RE = re.compile(
    r"\b(?:means|proves?|therefore|thus)\b|(?<!label\s)\b(?:tells?\s+us|indicates?|confirms?)\b|"
    r"\b(?:so\s+)?(?:this|that|it)\s+(?:is|must\s+be)\s+(?:the\s+)?"
    r"(?:[a-z0-9-]+\s+){0,3}(?:category|fridge|refrigerator|storage|bin)\b|"
    r"\b(?:unit|station|area|section|zone|container|cabinet|shelf|bin|box|order)\s+"
    r"[a-z0-9-]+\s+(?:is|holds?|belongs?\s+to|seems?\s+to\s+be|"
    r"appears?\s+to\s+be|must\s+be)\b|"
    r"\b(?:it|this|that)(?:'s|\s+is|\s+seems?\s+to\s+be|"
    r"\s+appears?\s+to\s+be)\s+(?:the\s+)?(?:[a-z0-9-]+\s+){0,3}one\b|"
    r"\b(?:this|that)\s+one\s+(?:is|belongs?\s+to|seems?\s+to\s+be)\b|"
    r"\blooks?\s+like\s+(?:the\s+)?(?:[a-z0-9-]+\s+){0,3}one\b",
    re.IGNORECASE,
)
_GUIDE_VISUAL_EVIDENCE_REPORT_RE = re.compile(
    r"\b(?:i|we)\s+(?:can\s+)?(?:see|saw|notice|noticed|spot|spotted|found)\b|"
    r"\bthere\s+(?:is|are|was|were)\b|\bcontents?\s+(?:is|are|include|includes)\b|"
    r"\binside\s+(?:is|are|i\s+(?:can\s+)?see|we\s+(?:can\s+)?see)\b|"
    r"\b(?:is|are|was|were)\s+inside\b|"
    r"\b(?:unit|station|container|cabinet|shelf|bin|box|order)\s+[a-z0-9-]+\s+"
    r"(?:contains?|has|holds?)\b",
    re.IGNORECASE,
)
_GUIDE_NEGATIVE_VISUAL_REPORT_RE = re.compile(
    r"\b(?:can't|cannot|couldn't|could\s+not|didn't|did\s+not|don't|do\s+not|"
    r"unable\s+to)\s+(?:see|find|identify|read|tell)\b|"
    r"\b(?:see|saw|found|notice|noticed)\s+(?:nothing|no\s+[a-z0-9-]+)\b",
    re.IGNORECASE,
)
_GUIDE_PLACEMENT_SUCCESS_CLAIM_RE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:now\s+)?(?:in|inside|on|onto)\s+"
    r"(?:(?:a|an|my|the|that|this|your)\s+)?[a-z0-9]|"
    r"\b(?:placement|putting\s+it)\b[^.!?\n]{0,80}\b"
    r"(?:complete|completed|done|succeeded|successful)\b",
    re.IGNORECASE,
)
_ASK_INTERFACE_ENUMERATION_RE = re.compile(
    r"\b(?:available|listed|valid)\s+(?:choices?|destinations?|locations?|options?|targets?)\b|"
    r"\b(?:destination|location|target)\s+(?:choices?|options?)\b|"
    r"\b(?:my|the)\s+(?:choices|options)\s+(?:are|include)\b",
    re.IGNORECASE,
)
_ASK_PREMATURE_COMPLETION_SPEECH_RE = re.compile(
    r"(?:^|\b)(?:done|finished|complete)\s*[.!]*$|"
    r"\b(?:we(?:'re| are)\s+(?:all\s+)?(?:done|finished|complete)(?:\s+here)?|"
    r"(?:the|this)\s+(?:task|episode)\s+(?:is\s+|has\s+been\s+)?"
    r"(?:complete|completed|done|finished)|"
    r"everything(?:'s|\s+is|\s+has\s+been)?\s+(?:complete|completed|done|finished)|"
    r"(?:finished|completed)\s+everything|all\s+done|"
    r"that(?:'s|\s+is|\s+should\s+be)\s+(?:it|all|everything))\b",
    re.IGNORECASE,
)
_GUIDE_NEXT_STEP_SPEECH_RE = re.compile(
    r"\b(?:what(?:'s|\s+is|\s+comes)\s+next|what\s+(?:should\s+i|do\s+i)\s+do\s+"
    r"(?:next|now)|how\s+should\s+i\s+proceed|ready\s+for\s+(?:(?:the|your)\s+)?"
    r"next\s+(?:instruction|step)|give\s+me\s+(?:the\s+)?next\s+step|what\s+next|"
    r"(?:okay|ok|alright)[,;:]?\s+next)\b",
    re.IGNORECASE,
)
_GUIDE_STEP_COMPLETION_CLAIM_RE = re.compile(
    r"\b(?:i(?:'ve|\s+have)?\s+(?:completed|done|finished)\s+(?:it|what\s+you\s+asked)|"
    r"i(?:'ve|\s+have)?\s+(?:completed|done|finished)\s+(?:that|this|the)\s+"
    r"(?:step|instruction|part)|i(?:'m|\s+am)\s+(?:all\s+)?"
    r"(?:done|finished|complete)\s+(?:with\s+)?(?:that|this|the)\s+"
    r"(?:step|instruction|part)|(?:that|this|the)\s+(?:step|instruction|part)\s+"
    r"(?:is|was)\s+(?:done|finished|complete)|that\s+(?:completes|finishes)\s+"
    r"(?:that|this|the)\s+(?:step|instruction|part))\b(?=\s*(?:[,;.!?]|$))",
    re.IGNORECASE,
)
_WATCH_DYNAMIC_INTERFACE_CLAIM_RE = re.compile(
    r"\bcurrent[- ]state\s+legend\b|"
    r"\blegend\b[^.!?\n]{0,180}\b(?:already|currently|still|satisfied|complete|done)\b|"
    r"\blegend\b[^.!?\n]{0,100}\b(?:indicates|shows|confirms|says)\b"
    r"[^.!?\n]{0,100}\b(?:is|are|was|were)\s+(?:inside|in|on|at)\b|"
    r"\bsimulator\b[^.!?\n]{0,180}\b(?:actually|considers?|state)\b",
    re.IGNORECASE,
)
_WATCH_PRIVATE_PLAN_USE_RE = re.compile(
    r"\b(?:private|no[- ]advice)\s+(?:plan|script)\s+"
    r"(?:indicates|says|shows|requires|needs|calls|tells)\b|"
    r"\bas\s+I\s+(?:originally|previously)\s+intended\b|"
    r"\b(?:according\s+to|following|from|using)\s+(?:my|the)\s+"
    r"(?:(?:original|default|intended|prewritten|pre-written)\s+)?"
    r"(?:action\s+)?(?:list|plan|script|sequence|steps)\b|"
    r"\b(?:my|the)\s+(?:original|default|intended|prewritten|pre-written)\s+"
    r"(?:action\s+)?(?:list|plan|script|sequence|steps)\s+"
    r"(?:indicate|say|show|require|need|call|tell)s?\b|"
    r"\bthe\s+script\s+(?:indicates|says|shows|requires|needs|calls|tells)\b",
    re.IGNORECASE,
)
_WATCH_FALSE_ACTION_CLAIM_RE = re.compile(
    r"\b(?:i|we)\s+(?:(?:am|are|was|were)\s+able\s+to\s+|already\s+|"
    r"(?:have|'ve)\s+)(?:found|grabbed|picked|placed|put|opened|closed|"
    r"dropped|pushed|pulled|cut|stirred|poured|wiped|cooked|attached|"
    r"detached|raised|flipped|completed|finished)\b|"
    r"\b(?:i|we)\s+(?:found|grabbed|picked|placed|opened|closed|dropped|"
    r"pushed|pulled|cut|stirred|poured|wiped|cooked|attached|detached|"
    r"raised|flipped|completed|finished)\b|"
    r"(?:^|[,;.!?—-]\s*)(?:(?:okay|well|no\s+problem)[,;:—-]?\s*)?"
    r"(?:but\s+)?(?:i|we)\s+put\b|"
    r"\ball\s+finished(?:\s+here)?\b",
    re.IGNORECASE,
)
_WATCH_ALREADY_AT_RE = re.compile(
    r"\b(?:I(?:'m|\s+am)|we(?:'re|\s+are))\s+(?:already|currently)\s+"
    r"(?:at|in|inside|on)\b",
    re.IGNORECASE,
)
_WATCH_HELD_CONFLICT_RE = re.compile(
    r"\b(?:already|currently)\s+(?:holding|carrying)\b|"
    r"\b(?:is|are)\s+(?:already|currently)\s+in\s+my\s+hand\b|"
    r"\bmy\s+hands?\s+(?:is|are)\s+full\b",
    re.IGNORECASE,
)
_WATCH_GROUNDING_QUESTION_RE = re.compile(
    r"(?:^|[?]\s*)\s*(?:which|what|where|who)\b|"
    r"\b(?:can['’]t|cannot|could not|couldn['’]t|do not|don['’]t)\s+"
    r"(?:identify|find|ground|see|tell)\b|"
    r"\b(?:which|what)\s+(?:object|item|place|surface|container|cabinet|shelf)\b|"
    r"\bwhat\s+do\s+you\s+mean\b",
    re.IGNORECASE,
)
_WATCH_ACTION_OBSTACLE_RE = re.compile(
    r"\b(?:can['’]?t|cannot|couldn['’]?t|tried\s+to|unable\s+to)\b"
    r"[^.!?\n]{0,160}\b(?:close|interact|open)\b|"
    r"\bdoes(?:n['’]?t|\s+not)\s+(?:seem\s+to\s+)?have\b"
    r"[^.!?\n]{0,100}\b(?:door|handle|mechanism|way)\b",
    re.IGNORECASE,
)
_WATCH_DIRECT_OBJECT_STATE_RE = re.compile(
    r"\b(?:it|they|that|this|the\s+[a-z][a-z0-9 -]{0,60}|[a-z][a-z0-9-]{1,30})"
    r"\s+(?:is|are)\s+(?:actually\s+)?(?:already|currently|still)\s+"
    r"(?:inside|in|on|at)\b",
    re.IGNORECASE,
)
_WATCH_STATE_RELATION_RE = re.compile(
    r"\b(?:already|currently|still)\s+(?:inside|in|on|at)\b|"
    r"\b(?:is|are|was|were|remains?)\s+(?:actually\s+)?"
    r"(?:already|currently|still)\s+(?:inside|in|on|at)\b",
    re.IGNORECASE,
)
_WATCH_STATE_REFUSAL_RE = re.compile(
    r"\b(?:instruction|request|step|action)\b[^.!?\n]{0,100}\b"
    r"(?:already\s+)?(?:satisfied|complete|done)\b|"
    r"\b(?:however|but|conflicts?|mistaken|wrong)\b|"
    r"\b(?:will|should)\s+(?:ask|inform|report)\b|"
    r"\bwhere\s+(?:else|should)\b",
    re.IGNORECASE,
)
_WATCH_ACTIONABLE_DIRECTIVE_RE = re.compile(
    r"\b(?:make\s+sure|need(?:s)?\s+to|has\s+to|have\s+to|must|please|"
    r"should\s+(?:go|be|move)|belongs?\s+(?:in|inside|on|at))\b",
    re.IGNORECASE,
)
_PROMPT_ONLY_FIND_HUMAN_TASK_CONTEXT = (
    "The fixed FIND question has already been asked. You have no private answer, route, goal plan, "
    "cleanup obligation, or completion information. Execute only the first concrete physical or "
    "inspection command explicitly requested in the Assistant's newest speech. If there is no new "
    "explicit command, leave the action empty and ask naturally for the next step."
)
_PROMPT_ONLY_ORCHESTRATE_HUMAN_TASK_CONTEXT = (
    "The fixed ORCHESTRATE request has already been spoken. You have no private plan, answer, "
    "cleanup obligation, or completion information. Execute only the current local subtask "
    "explicitly requested by the Assistant. If that subtask is complete and there is no new "
    "instruction, leave the action empty and ask naturally for the next step."
)
HUMAN_OBJECT_INTERFACES: tuple[str, ...] = ("full_ids", "opaque_handles")
FIND_HUMAN_CONTROLS: tuple[str, ...] = ("prompt-only", "dynamic-guarded")
ASK_HUMAN_CONTROLS: tuple[str, ...] = ("strict", "prompt-only")
_PRIVATE_OUTCOME_KEYS = frozenset(
    {
        "_visible_object_keys_full",
        "assistant_declared_done",
        "completion",
        "completion_contract",
        "completion_state",
        "completed",
        "contract_world_success",
        "done",
        "eval_valid",
        "first_goal_tick",
        "goal_met",
        "goal_progress",
        "goal_state",
        "goal_success",
        "is_done",
        "is_success",
        "n_done",
        "n_total",
        "physical_goal_met",
        "process_requirements_met",
        "progress",
        "reward",
        "score",
        "score_requires_assistant_done",
        "success",
        "task_goal",
        "task_success",
        "verification_points",
        "verification_points_define_success",
        "verification_points_met",
    }
)
_UNARY_WORLD_PREDICATES = frozenset(
    {"closed", "cooked", "cut", "off", "open", "running"}
)


class InteractionSpecError(ValueError):
    """A TaskPack cannot be adapted without inventing interaction semantics."""


@dataclass(frozen=True)
class OpeningScript:
    prelude_say: str
    actions: tuple[str, ...]
    handoff_say: str
    utterances: tuple[OpeningUtterance, ...] = ()
    watch_utterances: tuple[OpeningUtterance, ...] = ()
    continuous_watch: bool = False
    frame_policy: str = "evidence"
    recent_intervals: int = 2

    @property
    def setup_speech(self) -> tuple[str, ...]:
        if self.continuous_watch:
            return (self.prelude_say,) if self.prelude_say else ()
        speech = ([self.prelude_say] if self.prelude_say else []) + [
            utterance.say
            for utterance in self.utterances
            if utterance.say != self.handoff_say
        ]
        return tuple(speech)


@dataclass(frozen=True)
class OpeningUtterance:
    before_action_index: int
    say: str


@dataclass(frozen=True)
class SpokenCue:
    cue_id: str
    required_phrases: tuple[str, ...]


@dataclass(frozen=True)
class WatchIntervention:
    intervention_id: str
    before_action_index: int
    earliest_opening_action_index: int | None
    dialogue_index: int
    resume_opening_action_index: int
    resume_action_index: int
    recovery_commands: tuple[str, ...]


@dataclass(frozen=True)
class HumanCommandQualifier:
    match_command: str
    qualified_command: str
    space: str | None
    place_spot: tuple[float, float]


@dataclass(frozen=True)
class TaskPackInteractionSpec:
    task_id: str
    family: str
    public_task_context: str
    human_task_context: str
    authored_tick_budget: int
    opening: OpeningScript
    default_commands: tuple[str, ...]
    oracle_commands: tuple[str, ...]
    reference_dialogue: tuple[str, ...]
    reference_dialogue_action_indices: tuple[int | None, ...]
    scripted_human_replies: tuple[str, ...]
    answer_key: tuple[Mapping[str, object], ...]
    object_descriptions: Mapping[str, str]
    motor_aliases: Mapping[str, tuple[str, ...]]
    movable_object_keys: tuple[str, ...]
    opening_locations: Mapping[str, str]
    opening_evidence_action_indices: tuple[int, ...]
    opening_evidence_fact_ids: tuple[str, ...]
    required_spoken_cues: tuple[SpokenCue, ...]
    completion_contract: Mapping[str, object]
    watch_interventions: tuple[WatchIntervention, ...]
    human_command_qualifiers: tuple[HumanCommandQualifier, ...]
    visual_state_events: tuple[Mapping[str, object], ...]

    @property
    def human_receives_frames(self) -> bool:
        return profile_for(self.family).human_receives_frames

    @property
    def assistant_receives_opening_history(self) -> bool:
        return profile_for(self.family).assistant_receives_opening_history

    @property
    def score_requires_assistant_done(self) -> bool:
        interaction = self.completion_contract
        return bool(interaction.get("requires_assistant_done", True))

    @property
    def find_target_keys(self) -> tuple[str, ...]:
        if self.family != "FIND":
            return ()
        return tuple(sorted(self.find_target_aliases))

    @property
    def find_target_destinations(self) -> Mapping[str, tuple[str, ...]]:
        if self.family != "FIND":
            return {}
        return _FIND_PUBLIC_TARGET_DESTINATIONS.get(self.task_id, {})

    @property
    def find_grounding_aliases(self) -> Mapping[str, tuple[str, ...]]:
        if self.family != "FIND":
            return {}
        return _FIND_GROUNDING_ALIASES.get(self.task_id, {})

    @property
    def find_object_aliases(self) -> Mapping[str, tuple[str, ...]]:
        if self.family != "FIND":
            return {}
        return _FIND_OBJECT_ALIASES.get(self.task_id, {})

    @property
    def find_target_aliases(self) -> Mapping[str, tuple[str, ...]]:
        if self.family != "FIND":
            return {}
        return _FIND_TARGET_ALIASES.get(self.task_id, {})

    @property
    def find_destination_aliases(self) -> Mapping[str, tuple[str, ...]]:
        if self.family != "FIND":
            return {}
        return _FIND_DESTINATION_ALIASES.get(self.task_id, {})


@dataclass(frozen=True)
class HumanTurnContext:
    source: str
    new_assistant_speech: str
    active_instruction: str
    private_plan: str
    receives_frames: bool
    preempt_queue: bool = False
    expected_watch_command: str = ""
    interrupted_plan: str = ""
    planned_speech: str = ""
    planned_speech_phase: str = ""
    planned_speech_index: int | None = None


@dataclass
class _SymbolicState:
    held: str | None
    known_entities: set[str]
    container_targets: set[str]
    surface_targets: set[str]
    locations: dict[str, str]
    attached: dict[str, str]
    open_entities: set[str]
    running: set[str]
    sitting: str | None
    raised: set[str]
    cut: set[str]
    cooked: set[str]
    ever_moved: set[str]


@dataclass(frozen=True)
class _OrchestrateActionUnit:
    action_kind: str
    primary: frozenset[str]
    destination: frozenset[str]
    named: frozenset[str]
    auxiliaries: frozenset[str]
    placement_inside: bool | None = None


def enforce_human_turn_policy(
    family: str,
    source: str,
    commands: Sequence[str],
    human_say: str,
    private_plan: str = "",
    interrupted_plan: str = "",
    planned_speech: str = "",
    next_planned_speech: str = "",
    authored_planned_speech: Sequence[str] = (),
    assistant_instruction: str = "",
    object_descriptions: Mapping[str, object] | None = None,
    motor_aliases: Mapping[str, object] | None = None,
    ask_movable_object_keys: Sequence[str] = (),
    find_target_keys: Sequence[str] = (),
    visible_object_keys: Sequence[str] = (),
    held_object_keys: Sequence[str] = (),
    find_grounding_aliases: Mapping[str, object] | None = None,
    find_object_aliases: Mapping[str, object] | None = None,
    find_target_aliases: Mapping[str, object] | None = None,
    find_destination_aliases: Mapping[str, object] | None = None,
    find_target_destinations: Mapping[str, Sequence[str]] | None = None,
    find_open_object_keys: Sequence[str] = (),
    find_bound_landmark_keys: Sequence[str] = (),
    last_successful_human_command: str = "",
    active_instruction_successful_commands: Sequence[str] = (),
    last_human_action_result: str = "",
    enforce_ask_instruction_guard: bool = True,
    enforce_find_model_guard: bool = True,
    unredacted_human_say: str = "",
    human_at_keys: Sequence[str] | None = None,
    human_reasoning: str = "",
    watch_static_locations: Mapping[str, str] | None = None,
    opening_say: str = "",
    enforce_guide_turn_guard: bool = True,
) -> tuple[list[str], str, str]:
    """Apply the interaction contract when a model reply violates it."""
    normalized_family = family.strip().upper()
    if normalized_family == "NARRATE":
        return list(commands), human_say, ""
    if normalized_family == "GUIDE" and not enforce_guide_turn_guard:
        return list(commands), human_say, ""
    normalized_source = source.strip().lower()
    original_human_say = human_say
    authored_speech = {
        str(value).strip() for value in authored_planned_speech if str(value).strip()
    }
    current_planned_speech = planned_speech.strip()
    current_planned_speech_matches = bool(
        current_planned_speech and original_human_say.strip() == current_planned_speech
    )
    next_planned_speech_matches = bool(
        normalized_family == "WATCH"
        and not commands
        and next_planned_speech.strip()
        and original_human_say.strip() == next_planned_speech.strip()
    )
    guide_completion_supported = bool(
        normalized_family == "GUIDE"
        and (
            normalized_source == "assistant_active"
            or (normalized_source == "assistant_new" and bool(commands))
        )
        and bool(assistant_instruction.strip())
        and _successful_command_completes_claimed_instruction(
            original_human_say,
            last_successful_human_command,
            assistant_instruction,
            active_instruction_successful_commands,
            commands,
            find_object_aliases or {},
            find_grounding_aliases or {},
            find_target_destinations or {},
        )
    )
    previous_success_supports_completion = bool(
        not commands
        and (
            guide_completion_supported
            or (
                normalized_family == "WATCH"
                and normalized_source in {"assistant_active", "assistant_new"}
                and _watch_nonactionable_continuation_is_allowed(
                    assistant_instruction,
                    last_successful_human_command,
                )
                and _completed_motor_claim_matches_commands(
                    original_human_say,
                    (last_successful_human_command,),
                )
            )
            or (
                normalized_family == "WATCH"
                and normalized_source in {"assistant_active", "assistant_new"}
                and _watch_completion_report_supported_by_history(
                    original_human_say,
                    assistant_instruction,
                    last_successful_human_command,
                    active_instruction_successful_commands,
                    object_descriptions or {},
                    motor_aliases or {},
                )
            )
            or (
                normalized_family == "NARRATE"
                and normalized_source in {"assistant_active", "assistant_new"}
                and _narrate_motor_claim_matches_commands(
                    original_human_say,
                    (
                        *active_instruction_successful_commands,
                        last_successful_human_command,
                    ),
                )
            )
        )
    )
    guide_deferred_completion_speech = bool(
        normalized_family == "GUIDE"
        and commands
        and not guide_completion_supported
        and _guide_completion_intent(original_human_say)
    )
    human_say, unredacted_human_say = _guide_normalized_completion_speech(
        normalized_family,
        original_human_say,
        human_say,
        unredacted_human_say,
        guide_completion_supported,
        guide_deferred_completion_speech,
    )
    if (
        normalized_family == "GUIDE"
        and normalized_source in {"assistant_new", "assistant_active"}
        and assistant_instruction.strip()
        and not commands
        and not human_say.strip()
    ):
        return [], "", "guide_active_instruction_mute_blocked"
    planned_speech_repeat = bool(
        normalized_family == "WATCH"
        and original_human_say.strip()
        and any(
            _contains_spoken_phrase(original_human_say, authored)
            for authored in authored_speech
        )
        and not (normalized_source == "watch_speech" and current_planned_speech_matches)
        and not next_planned_speech_matches
    )
    planned_speech_mismatch = bool(
        normalized_family == "WATCH"
        and normalized_source == "watch_speech"
        and not current_planned_speech_matches
    )
    if planned_speech_repeat or planned_speech_mismatch:
        human_say = ""
    response_guard = _human_response_guard(
        normalized_family,
        normalized_source,
        commands,
        human_say,
        unredacted_human_say=unredacted_human_say,
        human_reasoning=human_reasoning,
        assistant_instruction=assistant_instruction,
        opening_say=opening_say,
        object_descriptions=object_descriptions or {},
        motor_aliases=motor_aliases or {},
        narrate_entity_aliases={
            **(find_grounding_aliases or {}),
            **(find_object_aliases or {}),
            **(motor_aliases or {}),
        },
        visible_object_keys=visible_object_keys,
        held_object_keys=held_object_keys,
        last_human_action_result=last_human_action_result,
        guide_completion_supported=guide_completion_supported,
    )
    if response_guard:
        if response_guard == "watch_assistant_command_alignment_needs_causal_review":
            return list(commands), human_say, response_guard
        replacement = ""
        if normalized_family == "GUIDE":
            replacement = {
                "ask_internal_interface_speech_blocked": _GUIDE_INTERFACE_CLARIFICATION,
                "ask_unobservable_attribute_speech_blocked": (
                    _GUIDE_UNOBSERVABLE_CLARIFICATION
                ),
                "guide_failed_placement_claim_blocked": (
                    "That placement didn't succeed. What should I do next?"
                ),
            }.get(response_guard, "")
        return [], replacement, response_guard
    if planned_speech_repeat and normalized_source in {
        "assistant_new",
        "assistant_active",
    }:
        return list(commands), human_say, "watch_planned_speech_repeat"
    watch_alignment_guard = _watch_command_alignment_guard(
        normalized_family,
        normalized_source,
        commands,
        assistant_instruction,
        interrupted_plan,
        object_descriptions or {},
        motor_aliases or {},
        held_object_keys,
        watch_static_locations or {},
        last_successful_human_command,
    )
    if watch_alignment_guard:
        return (
            list(commands),
            human_say,
            "watch_assistant_command_alignment_needs_causal_review",
        )
    family_guard = _family_specific_human_guard(
        normalized_family,
        normalized_source,
        commands,
        find_target_keys,
        visible_object_keys,
        assistant_instruction,
        object_descriptions or {},
        motor_aliases or {},
        ask_movable_object_keys,
        held_object_keys,
        find_grounding_aliases,
        find_object_aliases or {},
        find_target_aliases or {},
        find_destination_aliases or {},
        find_target_destinations or {},
        find_open_object_keys,
        find_bound_landmark_keys,
        last_successful_human_command,
        active_instruction_successful_commands,
        last_human_action_result,
        enforce_ask_instruction_guard,
        enforce_find_model_guard,
        human_at_keys,
        human_reasoning,
    )
    if family_guard is not None:
        return family_guard
    watch_turn_result = _watch_plan_turn_result(
        normalized_family,
        normalized_source,
        commands,
        human_say,
        private_plan,
        planned_speech_repeat,
        planned_speech_mismatch,
        next_planned_speech_matches,
    )
    if watch_turn_result is not None:
        return watch_turn_result
    completed_narrate_action = bool(
        normalized_family == "NARRATE"
        and _completed_motor_claim_matches_commands(human_say, commands)
    )
    if (
        commands
        and normalized_family != "WATCH"
        and _MOTOR_CLAIM_RE.match(human_say.strip())
        and not (completed_narrate_action or guide_completion_supported)
    ):
        preserved_say = (
            _trailing_narrate_question(human_say)
            if normalized_family == "NARRATE"
            else ""
        )
        return (
            list(commands),
            preserved_say,
            watch_alignment_guard or "human_routine_narration_suppressed",
        )
    allowed_watch_held_question = bool(
        normalized_family == "WATCH"
        and held_object_keys
        and _WATCH_HELD_CONFLICT_RE.search(human_say)
    )
    false_action_claim = bool(
        not allowed_watch_held_question
        and (
            _MOTOR_CLAIM_RE.match(human_say.strip())
            or (
                normalized_family == "WATCH"
                and (
                    _WATCH_FALSE_ACTION_CLAIM_RE.search(human_say)
                    or _ASK_PREMATURE_COMPLETION_SPEECH_RE.search(human_say)
                )
            )
        )
    )
    if not commands and false_action_claim and not previous_success_supports_completion:
        return [], "I haven't moved yet.", "human_motor_claim_without_action"
    return (
        list(commands),
        human_say,
        watch_alignment_guard
        or (
            "human_routine_narration_suppressed"
            if guide_deferred_completion_speech
            else ""
        )
        or ("watch_planned_speech_repeat" if planned_speech_repeat else ""),
    )


def _guide_normalized_completion_speech(
    family: str,
    original_say: str,
    human_say: str,
    unredacted_say: str,
    completion_supported: bool,
    completion_deferred: bool,
) -> tuple[str, str]:
    if completion_deferred:
        return "", ""
    if (
        family == "GUIDE"
        and completion_supported
        and _ASK_PREMATURE_COMPLETION_SPEECH_RE.search(original_say)
    ):
        return _GUIDE_LOCAL_COMPLETION_REPORT, _GUIDE_LOCAL_COMPLETION_REPORT
    return human_say, unredacted_say


def _watch_plan_turn_result(
    family: str,
    source: str,
    commands: Sequence[str],
    human_say: str,
    private_plan: str,
    planned_speech_repeat: bool,
    planned_speech_mismatch: bool,
    next_planned_speech_matches: bool,
) -> tuple[list[str], str, str] | None:
    if family != "WATCH":
        return None
    if source in {"watch_opening", "watch_default"}:
        if not private_plan.strip():
            raise InteractionSpecError(
                f"WATCH {source} turn is missing its private intended action"
            )
        guard = "watch_planned_speech_repeat" if planned_speech_repeat else ""
        return list(commands), human_say, guard
    if source == "watch_speech":
        guard = "watch_planned_speech_only" if commands else ""
        if planned_speech_repeat:
            guard = "watch_planned_speech_repeat"
        elif planned_speech_mismatch:
            guard = "watch_planned_speech_mismatch"
        return [], human_say, guard
    if source == "idle":
        if next_planned_speech_matches:
            return [], human_say, ""
        guard = (
            "watch_planned_speech_repeat"
            if planned_speech_repeat
            else "watch_idle_wait"
        )
        return [], "", guard
    return None


def _human_response_guard(
    family: str,
    source: str,
    commands: Sequence[str],
    human_say: str,
    *,
    unredacted_human_say: str,
    human_reasoning: str,
    assistant_instruction: str,
    opening_say: str,
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    narrate_entity_aliases: Mapping[str, object],
    visible_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    last_human_action_result: str,
    guide_completion_supported: bool,
) -> str:
    if (
        family == "GUIDE"
        and not commands
        and _guide_failed_placement_claim(
            unredacted_human_say or human_say,
            last_human_action_result,
            assistant_instruction,
        )
    ):
        return "guide_failed_placement_claim_blocked"
    narrate_guard = _narrate_response_guard(
        family,
        commands,
        human_say,
        unredacted_human_say=unredacted_human_say,
        human_reasoning=human_reasoning,
        assistant_instruction=assistant_instruction,
        opening_say=opening_say,
        object_descriptions=object_descriptions,
        entity_aliases=narrate_entity_aliases,
        held_object_keys=held_object_keys,
    )
    if narrate_guard:
        return narrate_guard
    watch_guard = _watch_state_correction_guard(
        family,
        source,
        commands,
        unredacted_human_say or human_say,
        human_reasoning,
        assistant_instruction,
        held_object_keys,
    )
    if watch_guard:
        return watch_guard
    return _ask_speech_guard(
        family,
        commands,
        human_say,
        unredacted_human_say=unredacted_human_say,
        assistant_instruction=assistant_instruction,
        object_descriptions=object_descriptions,
        motor_aliases=motor_aliases,
        visible_object_keys=visible_object_keys,
        held_object_keys=held_object_keys,
        guide_completion_supported=guide_completion_supported,
    )


def _narrate_response_guard(
    family: str,
    commands: Sequence[str],
    human_say: str,
    *,
    unredacted_human_say: str,
    human_reasoning: str,
    assistant_instruction: str,
    opening_say: str,
    object_descriptions: Mapping[str, object],
    entity_aliases: Mapping[str, object],
    held_object_keys: Sequence[str],
) -> str:
    if family != "NARRATE":
        return ""
    raw_say = unredacted_human_say or human_say
    entity_keys = set(object_descriptions) | set(entity_aliases)
    raw_say_names_internal_key = any(
        "_" in str(key)
        and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(str(key))}(?![A-Za-z0-9_])",
            raw_say,
            flags=re.IGNORECASE,
        )
        for key in entity_keys
    )
    if (
        _ASK_INTERNAL_INTERFACE_SPEECH_RE.search(raw_say)
        or re.search(r"\[[A-Za-z_]+\]|<[^>]+>\s*\(-?\d+\)", raw_say)
        or raw_say_names_internal_key
    ):
        return "narrate_internal_interface_speech_blocked"
    public_context = " ".join((opening_say, assistant_instruction)).strip()
    if _ask_speech_reveals_ungrounded_entity(
        raw_say,
        public_context,
        object_descriptions,
        entity_aliases,
        (),
        held_object_keys,
        block_any_ungrounded=True,
    ):
        return "narrate_hidden_entity_speech_blocked"
    if len(commands) > 1:
        return "narrate_multiple_commands_blocked"
    return ""


def _watch_state_correction_guard(
    family: str,
    source: str,
    commands: Sequence[str],
    human_say: str,
    human_reasoning: str,
    assistant_instruction: str,
    held_object_keys: Sequence[str],
) -> str:
    if family != "WATCH" or source not in {"assistant_new", "assistant_active"}:
        return ""
    if not _watch_instruction_is_actionable(assistant_instruction):
        return ""
    response = " ".join((human_reasoning, human_say)).strip()
    if _WATCH_PRIVATE_PLAN_USE_RE.search(response):
        return "watch_assistant_state_override_blocked"
    if _WATCH_DYNAMIC_INTERFACE_CLAIM_RE.search(response):
        return "watch_assistant_state_override_blocked"
    if (
        _ask_action_is_requested(assistant_instruction, "WALK")
        and _WATCH_ALREADY_AT_RE.search(response)
        and not _commands_include_action_kind(commands, "WALK")
    ):
        return "watch_assistant_state_override_blocked"
    if commands:
        return ""
    if (
        _MOTOR_CLAIM_RE.match(human_say.strip())
        or _WATCH_FALSE_ACTION_CLAIM_RE.search(human_say)
        or _ASK_PREMATURE_COMPLETION_SPEECH_RE.search(human_say)
    ):
        return ""
    held_conflict = bool(held_object_keys and _WATCH_HELD_CONFLICT_RE.search(response))
    if _WATCH_DIRECT_OBJECT_STATE_RE.search(response) and not held_conflict:
        return "watch_assistant_state_override_blocked"
    if _WATCH_STATE_REFUSAL_RE.search(response) and _WATCH_STATE_RELATION_RE.search(
        response
    ):
        return "watch_assistant_state_override_blocked"
    if (
        human_say.strip()
        or held_conflict
        or _WATCH_GROUNDING_QUESTION_RE.search(response)
        or _WATCH_ACTION_OBSTACLE_RE.search(response)
    ):
        return "watch_assistant_command_alignment_needs_causal_review"
    return "watch_assistant_instruction_ignored_blocked"


def _watch_command_alignment_guard(
    family: str,
    source: str,
    commands: Sequence[str],
    assistant_instruction: str,
    interrupted_plan: str,
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    held_object_keys: Sequence[str],
    static_locations: Mapping[str, str],
    last_successful_human_command: str,
) -> str:
    if (
        family != "WATCH"
        or source not in {"assistant_new", "assistant_active"}
        or not commands
    ):
        return ""
    if not assistant_instruction.strip():
        return ""
    parsed = _watch_parse_commands(commands)
    review_guard = (
        "watch_assistant_command_alignment_needs_causal_review"
        if len(parsed) > 1
        else ""
    )
    if not _watch_instruction_is_actionable(assistant_instruction):
        if _watch_nonactionable_continuation_is_allowed(
            assistant_instruction,
            last_successful_human_command,
        ):
            return ""
        return review_guard or "watch_assistant_command_alignment_needs_causal_review"
    requested = _watch_requested_action_kinds(assistant_instruction)
    if not requested:
        if _WATCH_ACTIONABLE_DIRECTIVE_RE.search(assistant_instruction):
            return (
                review_guard or "watch_assistant_command_alignment_needs_causal_review"
            )
        return review_guard
    if _watch_repeats_nonrepeatable_action(parsed):
        return review_guard or "watch_assistant_command_alignment_needs_causal_review"
    if _watch_has_redundant_walk_run(parsed, assistant_instruction):
        return review_guard or "watch_assistant_command_alignment_needs_causal_review"
    aliases = _watch_command_aliases(parsed, object_descriptions, motor_aliases)
    if _watch_explicit_action_order_mismatch(
        parsed, assistant_instruction, requested, aliases
    ):
        return (
            "watch_assistant_command_alignment_needs_causal_review"
            if review_guard
            else "watch_assistant_command_alignment_needs_causal_review"
        )
    held = {str(key) for key in held_object_keys}
    held.update(_watch_interrupted_subjects(interrupted_plan))
    recent_subjects = _watch_command_subjects(last_successful_human_command)
    placement_grounded = _watch_grounded_names(assistant_instruction, aliases, "PLACE")
    if _watch_placement_has_referential_subject(assistant_instruction):
        placement_grounded.update(recent_subjects)
    (
        support_walk_targets,
        support_action_targets,
        support_preparation,
    ) = _watch_static_support_steps(
        parsed,
        requested,
        assistant_instruction,
        aliases,
        static_locations,
    )
    placement_preparation = _watch_placement_preparation_steps(
        parsed,
        requested,
        assistant_instruction,
        aliases,
        static_locations,
        held,
        recent_subjects,
    )
    placement_preparation.update(support_preparation)
    saw_unverified = False
    for command_index, (verb, action_kind, keyed) in enumerate(parsed):
        alignment = _watch_one_command_alignment(
            command_index,
            verb,
            action_kind,
            keyed,
            requested,
            assistant_instruction,
            aliases,
            object_descriptions,
            placement_grounded,
            held,
            support_walk_targets,
            support_action_targets,
            placement_preparation,
            recent_subjects,
            len(_find_action_clauses(assistant_instruction)) > 1,
        )
        if alignment:
            if alignment == "watch_assistant_command_alignment_needs_causal_review":
                return alignment
            saw_unverified = True
    return (
        "watch_assistant_command_alignment_needs_causal_review"
        if review_guard or saw_unverified
        else ""
    )


def _watch_parse_commands(
    commands: Sequence[str],
) -> list[tuple[str, str, list[str]]]:
    parsed: list[tuple[str, str, list[str]]] = []
    for command in commands:
        verb, arguments, _ = _command_key(command)
        keyed = [name for name, object_id in arguments if object_id != 0]
        parsed.append((verb, _find_action_kind(verb), keyed))
    return parsed


def _watch_command_aliases(
    parsed: Sequence[tuple[str, str, Sequence[str]]],
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    referenced = {key for _verb, _action_kind, keyed in parsed for key in keyed}
    known = set(object_descriptions) | set(motor_aliases) | referenced
    aliases: dict[str, tuple[str, ...]] = {}
    for raw_key in known:
        key = str(raw_key)
        description = str(object_descriptions.get(key, "")).split("; currently", 1)[0]
        aliases[key] = tuple(
            dict.fromkeys(
                (
                    *_guide_entity_aliases(key, description),
                    *_motor_alias_values(motor_aliases.get(key)),
                )
            )
        )
    return aliases


def _watch_one_command_alignment(
    command_index: int,
    verb: str,
    action_kind: str,
    keyed: Sequence[str],
    requested: set[str],
    assistant_instruction: str,
    aliases: Mapping[str, object],
    object_descriptions: Mapping[str, object],
    placement_grounded: set[str],
    held: set[str],
    support_walk_targets: set[tuple[int, str]],
    support_action_targets: set[tuple[int, str, str]],
    placement_preparation: set[tuple[int, str, str]],
    recent_subjects: set[str],
    allow_unverified_preparation: bool,
) -> str:
    preparation_matches = bool(
        len(keyed) == 1
        and (command_index, action_kind, keyed[0]) in placement_preparation
    )
    if preparation_matches:
        if verb == "GRAB":
            held.clear()
            held.add(keyed[0])
        return ""
    if action_kind not in requested:
        if action_kind == "WALK":
            walk_alignment = _watch_preparatory_walk_alignment(
                keyed,
                requested,
                assistant_instruction,
                aliases,
                placement_grounded,
                held,
                support_walk_targets,
                command_index,
            )
            if (
                allow_unverified_preparation
                and walk_alignment
                == "watch_assistant_command_alignment_needs_causal_review"
            ):
                return "watch_assistant_command_alignment_unverified"
            return walk_alignment
        return "watch_assistant_command_alignment_unverified"
    directly_grounded = _watch_directly_grounded_names(
        assistant_instruction, aliases, action_kind
    )
    if action_kind == "PLACE" and _watch_placement_has_referential_subject(
        assistant_instruction
    ):
        directly_grounded.update(recent_subjects)
    if action_kind == "WALK":
        transport_targets = _watch_transport_walk_targets(
            assistant_instruction,
            aliases,
        )
        if transport_targets:
            directly_grounded = transport_targets
    sequence_grounded = {
        key
        for index, kind, key in support_action_targets
        if index == command_index and kind == action_kind
    }
    grounded = (
        directly_grounded
        or sequence_grounded
        or _positive_alias_matches(assistant_instruction, aliases)
    )
    permitted_held = _watch_permitted_held_arguments(
        action_kind,
        keyed,
        assistant_instruction,
        directly_grounded,
        held,
    )
    if action_kind == "PLACE":
        destination_grounded = _ask_action_grounded_names(
            assistant_instruction,
            aliases,
            "PLACE",
        )
        subject_grounded = _watch_placement_subject_grounded_names(
            assistant_instruction,
            aliases,
        )
        if _watch_placement_has_referential_subject(assistant_instruction):
            subject_grounded.update(held or recent_subjects)
        if not destination_grounded:
            contextual_destinations = directly_grounded - subject_grounded
            if len(contextual_destinations) == 1:
                destination_grounded.update(contextual_destinations)
        if (
            keyed
            and subject_grounded
            and keyed[0] not in subject_grounded
            and keyed[0] not in permitted_held
        ):
            return "watch_assistant_command_alignment_needs_causal_review"
        return _watch_placement_alignment(
            verb,
            keyed,
            assistant_instruction,
            object_descriptions,
            destination_grounded,
            held,
        )
    if keyed and not grounded and not set(keyed) <= permitted_held:
        return "watch_assistant_command_alignment_unverified"
    if set(keyed) - grounded - permitted_held:
        return "watch_assistant_command_alignment_needs_causal_review"
    if verb == "GRAB" and keyed:
        held.clear()
        held.add(keyed[0])
    return ""


def _watch_placement_subject_grounded_names(
    instruction: str,
    aliases: Mapping[str, object],
) -> set[str]:
    grounded: set[str] = set()
    for clause in _find_action_clauses(instruction):
        if not _ask_action_is_requested(clause, "PLACE"):
            continue
        subject_phrase = re.split(
            r"\b(?:in|inside|into|on|onto|to|toward|towards)\b",
            clause,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        grounded.update(_positive_alias_matches(subject_phrase, aliases))
    return grounded


def _watch_placement_alignment(
    verb: str,
    keyed: Sequence[str],
    assistant_instruction: str,
    object_descriptions: Mapping[str, object],
    destination_grounded: set[str],
    held: set[str],
) -> str:
    if len(keyed) < 2:
        return "watch_assistant_command_alignment_unverified"
    if held and keyed[0] not in held:
        return "watch_assistant_command_alignment_needs_causal_review"
    destination = keyed[-1]
    requests_inside = _ask_placement_relation_matches(
        "PUTINSIDE", assistant_instruction
    )
    requests_surface = _ask_placement_relation_matches("PUTON", assistant_instruction)
    if verb in {"PUTIN", "PUTINSIDE"} and requests_surface and not requests_inside:
        return "watch_assistant_command_alignment_needs_causal_review"
    if verb not in {"PUTIN", "PUTINSIDE"} and requests_inside and not requests_surface:
        return "watch_assistant_command_alignment_needs_causal_review"
    if (
        not requests_inside
        and not requests_surface
        and not (
            _watch_destination_relation_is_supported(
                verb, destination, object_descriptions
            )
        )
    ):
        return "watch_assistant_command_alignment_unverified"
    if destination not in destination_grounded:
        return (
            "watch_assistant_command_alignment_needs_causal_review"
            if destination_grounded
            else "watch_assistant_command_alignment_unverified"
        )
    held.discard(keyed[0])
    return ""


def _watch_permitted_held_arguments(
    action_kind: str,
    keyed: Sequence[str],
    assistant_instruction: str,
    directly_grounded: set[str],
    held: set[str],
) -> set[str]:
    if not held or not keyed or action_kind not in {"DROP", "PLACE"}:
        return set()
    subject = keyed[0]
    if subject not in held:
        return set()
    explicit_subjects = directly_grounded - (
        {keyed[-1]} if action_kind == "PLACE" else set()
    )
    if explicit_subjects:
        return set()
    if re.search(
        r"\b(?:drop|leave|place|put|return|set|store)\s+"
        r"(?:it|that|this|them)\b",
        assistant_instruction,
        flags=re.IGNORECASE,
    ):
        return {subject}
    return set()


def _watch_requested_action_kinds(instruction: str) -> set[str]:
    requested = _ask_unambiguous_action_kinds(instruction)
    normalized = _normalize_find_instruction(instruction)
    for clause in _find_action_clauses(normalized):
        requested.update(_ask_unambiguous_action_kinds(clause))
    if _watch_has_transport_destination(normalized):
        requested.add("WALK")
        if _watch_transport_has_named_subject(normalized):
            requested.add("GRAB")
    if _watch_has_load_placement(normalized):
        requested.add("PLACE")
    if _watch_has_resultative_placement(normalized):
        requested.add("PLACE")
        requested.discard("WALK")
    if _watch_has_return_for_retrieval(normalized):
        requested.update({"GRAB", "WALK"})
    if _watch_reports_left_open(normalized):
        requested.add("CLOSE")
    for action_kind, cues in _ASK_OMISSION_ACTION_CUES.items():
        cue_pattern = "|".join(re.escape(cue) for cue in sorted(cues))
        if re.search(
            rf"\b(?:after\s+you|before\s+you|while\s+you|to)\s+"
            rf"(?:{cue_pattern})\b",
            normalized,
        ):
            requested.add(action_kind)
    if re.search(
        r"\b(?:swap|switch)\s+(?:over\s+)?to\b|"
        r"\buse\b[^.!?\n]{0,80}\binstead\b|"
        r"\breplace\b[^.!?\n]{0,80}\bwith\b|"
        r"\bswitch\s+out\b[^.!?\n]{0,80}\bfor\b|"
        r"\bchange\s+to\b",
        normalized,
    ):
        requested.add("GRAB")
    if re.search(
        r"\bbelongs?\s+(?:in|inside|on|at)\b",
        instruction,
        flags=re.IGNORECASE,
    ):
        requested.add("PLACE")
    if _watch_has_resultative_placement(normalized):
        requested.discard("WALK")
    return requested


def _watch_has_load_placement(instruction: str) -> bool:
    return bool(
        re.search(
            r"\bload\b[^.!?;\n]{1,100}\b(?:in|inside|into)\b",
            instruction,
        )
    )


def _watch_has_resultative_placement(instruction: str) -> bool:
    return bool(
        re.search(
            r"\b(?:need(?:s)?\s+to|should|must)\s+"
            r"(?:go|stay)\s+(?:at|in|inside|into|on|onto)\b",
            instruction,
        )
    )


def _watch_has_return_for_retrieval(instruction: str) -> bool:
    return bool(
        re.search(
            r"\b(?:come|go|head|return|walk)\s+back\s+for\b",
            instruction,
        )
    )


def _watch_reports_left_open(instruction: str) -> bool:
    return bool(
        re.search(
            r"\b(?:you\s+(?:left|leave)|do(?:n't|\s+not)\s+leave)\b"
            r"[^.!?;\n]{1,80}\bopen\b",
            instruction,
        )
    )


def _watch_has_transport_destination(instruction: str) -> bool:
    return any(
        re.search(
            r"\b(?:bring|carry|take)\b[^.!?;\n]{0,100}"
            r"\b(?:to|toward|towards)\b\s+"
            r"(?!(?:it|one|somewhere|there)\b)",
            clause,
        )
        for clause in _find_action_clauses(instruction)
    )


def _watch_transport_has_named_subject(instruction: str) -> bool:
    for clause in _find_action_clauses(instruction):
        match = _watch_transport_match(clause)
        if match is None:
            continue
        return not _watch_transport_subject_is_pronoun(match.group("subject"))
    return False


def _watch_transport_match(clause: str) -> re.Match[str] | None:
    return re.search(
        r"\b(?:bring|carry|take)\b\s+"
        r"(?P<subject>[^.!?;\n]{1,80}?)\s+"
        r"\b(?:to|toward|towards)\b\s+(?P<destination>.+)$",
        clause,
    )


def _watch_transport_subject_is_pronoun(subject: str) -> bool:
    words = [
        word
        for word in _WORD_RE.findall(subject.casefold())
        if word not in {"along", "back", "over"}
    ]
    return words in (["it"], ["that"], ["them"], ["this"])


def _watch_transport_walk_targets(
    instruction: str,
    aliases: Mapping[str, object],
) -> set[str]:
    for clause in _find_action_clauses(instruction):
        match = _watch_transport_match(clause)
        if match is None:
            continue
        destination = match.group("destination")
        grounded = _watch_key_phrase_matches(destination, aliases)
        if grounded:
            return grounded
        if re.search(r"\blaundry(?:\s+(?:area|room))?\b", destination):
            laundry_landmarks = {
                key
                for key in aliases
                if set(_WORD_RE.findall(key.casefold().replace("_", " ")))
                & {"dryer", "hamper", "washer"}
            }
            if laundry_landmarks:
                return laundry_landmarks
        grounded = _positive_alias_matches(destination, aliases)
        if grounded:
            return grounded
    return set()


def _watch_key_phrase_matches(
    text: str,
    aliases: Mapping[str, object],
) -> set[str]:
    matches = {
        key: tuple(_WORD_RE.findall(key.casefold().replace("_", " ")))
        for key in aliases
        if _contains_spoken_phrase(text, key.replace("_", " "))
    }
    return {
        key
        for key, words in matches.items()
        if not any(
            len(other_words) > len(words) and set(words) < set(other_words)
            for other_key, other_words in matches.items()
            if other_key != key
        )
    }


def _watch_interrupted_subjects(interrupted_plan: str) -> set[str]:
    if not interrupted_plan.strip():
        return set()
    verb, arguments, _ = _command_key(interrupted_plan)
    if verb not in _WATCH_UNIT_END or not arguments:
        return set()
    return {arguments[0][0]}


def _watch_command_subjects(command: str) -> set[str]:
    if not command.strip():
        return set()
    try:
        _verb, arguments, _ = _command_key(command)
    except InteractionSpecError:
        return set()
    return {arguments[0][0]} if arguments and arguments[0][1] != 0 else set()


def _watch_placement_has_referential_subject(instruction: str) -> bool:
    return bool(
        re.search(
            r"\b(?:leave|place|put|return|set|store)\s+"
            r"(?:it|that|this|them)\b",
            instruction,
            flags=re.IGNORECASE,
        )
    )


def _watch_nonactionable_continuation_is_allowed(
    instruction: str,
    last_successful_human_command: str,
) -> bool:
    normalized = _normalize_find_instruction(instruction).strip()
    monitoring_ack = bool(
        re.search(
            r"\b(?:i(?:'ll|\s+will)\s+)?(?:keep\s+an\s+eye|monitor|watch)\b"
            r"[^.!?\n]{0,100}\b(?:mistake|speak\s+up)\b",
            normalized,
        )
    )
    confirmation = bool(
        last_successful_human_command.strip()
        and re.match(r"^(?:correct|exactly|right|yes)\b", normalized)
    )
    return monitoring_ack or confirmation


def _watch_grounded_names(
    instruction: str,
    aliases: Mapping[str, object],
    action_kind: str,
) -> set[str]:
    grounded = _watch_directly_grounded_names(instruction, aliases, action_kind)
    return grounded or _positive_alias_matches(instruction, aliases)


def _watch_directly_grounded_names(
    instruction: str,
    aliases: Mapping[str, object],
    action_kind: str,
) -> set[str]:
    special = _watch_special_action_grounded_names(
        instruction,
        aliases,
        action_kind,
    )
    if special:
        return special
    if action_kind == "GRAB":
        redirected = _watch_redirect_grounded_names(instruction, aliases)
        if redirected:
            return redirected
    grounded = _ask_action_grounded_names(instruction, aliases, action_kind)
    if action_kind == "PLACE":
        grounded.update(_watch_clausewise_grounded_names(instruction, aliases))
    if grounded:
        return grounded
    key_grounded: set[str] = set()
    for clause in _find_action_clauses(instruction):
        if _ask_action_is_requested(clause, action_kind):
            key_grounded.update(_watch_key_phrase_matches(clause, aliases))
    return key_grounded


def _watch_special_action_grounded_names(
    instruction: str,
    aliases: Mapping[str, object],
    action_kind: str,
) -> set[str]:
    clauses = _find_action_clauses(instruction)
    if action_kind == "GRAB":
        identity = set()
        for clause in clauses:
            if re.match(r"^(?:it|that)(?:'s|\s+is)\b", clause):
                identity.update(_watch_key_phrase_matches(clause, aliases))
                identity.update(_positive_alias_matches(clause, aliases))
        if identity and any(
            _ask_action_is_requested(clause, "GRAB") for clause in clauses
        ):
            return identity
    for clause in clauses:
        clause_matches = _watch_key_phrase_matches(clause, aliases)
        if not clause_matches:
            clause_matches = _positive_alias_matches(clause, aliases)
        if action_kind == "PLACE" and _watch_has_load_placement(clause):
            return clause_matches
        if action_kind in {"GRAB", "WALK"} and _watch_has_return_for_retrieval(clause):
            return clause_matches
        if action_kind == "CLOSE" and _watch_reports_left_open(clause):
            return clause_matches
    return set()


def _watch_clausewise_grounded_names(
    instruction: str,
    aliases: Mapping[str, object],
) -> set[str]:
    grounded: set[str] = set()
    for clause in _find_action_clauses(instruction):
        grounded.update(_watch_key_phrase_matches(clause, aliases))
        grounded.update(_positive_alias_matches(clause, aliases))
    return grounded


def _watch_redirect_grounded_names(
    instruction: str,
    aliases: Mapping[str, object],
) -> set[str]:
    normalized = _normalize_find_instruction(instruction)
    patterns = (
        r"\b(?:swap|change)\s+(?:over\s+)?to\s+(.+?)(?:\s+instead)?[.!?]?$",
        r"\buse\s+(.+?)\s+instead[.!?]?$",
        r"\breplace\s+.+?\s+with\s+(.+?)[.!?]?$",
        r"\bswitch\s+out\s+.+?\s+for\s+(.+?)[.!?]?$",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match is not None:
            return _positive_alias_matches(match.group(1), aliases)
    return set()


def _watch_preparatory_walk_alignment(
    keyed: Sequence[str],
    requested: set[str],
    assistant_instruction: str,
    aliases: Mapping[str, object],
    placement_grounded: set[str],
    held: set[str],
    support_walk_targets: set[tuple[int, str]],
    command_index: int,
) -> str:
    if len(keyed) != 1:
        return "watch_assistant_command_alignment_unverified"
    grounded_targets = placement_grounded - held if "PLACE" in requested else set()
    for requested_kind in requested - {"PLACE", "WALK"}:
        grounded_targets.update(
            _watch_directly_grounded_names(
                assistant_instruction,
                aliases,
                requested_kind,
            )
        )
    if not grounded_targets and requested - {"WALK"}:
        grounded_targets.update(_positive_alias_matches(assistant_instruction, aliases))
    grounded_targets.update(
        target for index, target in support_walk_targets if index == command_index
    )
    grounded_targets.difference_update(held)
    if keyed[0] in grounded_targets:
        return ""
    return (
        "watch_assistant_command_alignment_needs_causal_review"
        if grounded_targets
        else "watch_assistant_command_alignment_unverified"
    )


def _watch_static_support_steps(
    parsed: Sequence[tuple[str, str, Sequence[str]]],
    requested: set[str],
    assistant_instruction: str,
    aliases: Mapping[str, object],
    static_locations: Mapping[str, str],
) -> tuple[
    set[tuple[int, str]],
    set[tuple[int, str, str]],
    set[tuple[int, str, str]],
]:
    walk_targets: set[tuple[int, str]] = set()
    action_targets: set[tuple[int, str, str]] = set()
    preparation: set[tuple[int, str, str]] = set()
    for action_index, (_verb, action_kind, action_keys) in enumerate(parsed):
        if (
            action_kind not in requested
            or action_kind in {"PLACE", "WALK"}
            or len(action_keys) != 1
        ):
            continue
        direct_targets = _watch_directly_grounded_names(
            assistant_instruction,
            aliases,
            action_kind,
        )
        action_target = action_keys[0]
        if direct_targets and action_target not in direct_targets:
            continue
        relation, separator, support = str(
            static_locations.get(action_target, "")
        ).partition(":")
        if not separator or relation not in {"in", "inside", "on"}:
            continue
        cursor = action_index - 1
        has_preparation = False
        if (
            cursor >= 0
            and relation in {"in", "inside"}
            and parsed[cursor][1] == "OPEN"
            and parsed[cursor][2] == [support]
        ):
            preparation.add((cursor, "OPEN", support))
            cursor -= 1
            has_preparation = True
        if (
            cursor >= 0
            and parsed[cursor][1] == "WALK"
            and parsed[cursor][2] == [support]
        ):
            walk_targets.add((cursor, support))
            has_preparation = True
        if has_preparation:
            action_targets.add((action_index, action_kind, action_target))
    return walk_targets, action_targets, preparation


def _watch_repeats_nonrepeatable_action(
    parsed: Sequence[tuple[str, str, Sequence[str]]],
) -> bool:
    nonrepeatable = {"CLOSE", "GRAB", "OPEN", "PLACE", "TURNON", "TURNOFF"}
    counts = Counter(
        (action_kind, tuple(keyed))
        for _verb, action_kind, keyed in parsed
        if action_kind in nonrepeatable and keyed
    )
    return any(count > 1 for count in counts.values())


def _watch_explicit_action_order_mismatch(
    parsed: Sequence[tuple[str, str, Sequence[str]]],
    assistant_instruction: str,
    requested: set[str],
    aliases: Mapping[str, object],
) -> bool:
    units = _watch_ordered_instruction_units(
        assistant_instruction,
        requested,
        aliases,
    )
    if len(units) < 2:
        return False
    cursor = 0
    used: set[int] = set()
    for _verb, action_kind, keyed in parsed:
        candidates = [
            index
            for index, (unit_kind, targets) in enumerate(units)
            if index not in used
            and unit_kind == action_kind
            and (not targets or bool(set(keyed) & targets))
        ]
        if not candidates:
            continue
        next_index = min(candidates)
        if next_index < cursor:
            return True
        cursor = next_index
        used.add(next_index)
    return False


def _watch_ordered_instruction_units(
    instruction: str,
    requested: set[str],
    aliases: Mapping[str, object],
) -> list[tuple[str, set[str]]]:
    fragments = _watch_semantically_ordered_fragments(instruction)
    units: list[tuple[str, set[str]]] = []
    for fragment in fragments:
        transport_units = _watch_transport_ordered_units(fragment, aliases)
        if transport_units:
            units.extend(transport_units)
            continue
        special_units = _watch_special_ordered_units(fragment, aliases)
        if special_units:
            units.extend(special_units)
            continue
        words = _find_action_words(fragment)
        ordered_kinds = sorted(
            (
                min(
                    index
                    for index, word in enumerate(words)
                    if word in _ASK_OMISSION_ACTION_CUES.get(action_kind, ())
                ),
                action_kind,
            )
            for action_kind in requested
            if any(
                word in _ASK_OMISSION_ACTION_CUES.get(action_kind, ()) for word in words
            )
        )
        units.extend(
            (
                action_kind,
                _watch_order_unit_targets(fragment, aliases, action_kind),
            )
            for _position, action_kind in ordered_kinds
        )
    return units


def _watch_lifecycle_instruction_units(
    instruction: str,
    aliases: Mapping[str, object],
) -> list[tuple[str, set[str]]]:
    units: list[tuple[str, set[str]]] = []
    for fragment in _watch_semantically_ordered_fragments(instruction):
        lifecycle_fragment = _watch_lifecycle_fragment(fragment)
        words = _find_action_words(lifecycle_fragment)
        if words == ["stop"]:
            continue
        positioned = _watch_lifecycle_action_positions(lifecycle_fragment, words)
        units.extend(
            (
                action_kind,
                _watch_order_unit_targets(lifecycle_fragment, aliases, action_kind),
            )
            for action_kind, _position in sorted(
                positioned.items(), key=lambda item: item[1]
            )
        )
    return units


def _watch_lifecycle_fragment(fragment: str) -> str:
    if not re.match(r"^(?:please\s+)?do not forget\b", fragment):
        return fragment
    return re.split(r"\bbefore\s+you\b", fragment, maxsplit=1)[0]


def _watch_lifecycle_action_positions(
    fragment: str, words: Sequence[str]
) -> dict[str, int]:
    requested = _watch_requested_action_kinds(fragment)
    positioned = {
        action_kind: min(
            index
            for index, word in enumerate(words)
            if word in _ASK_OMISSION_ACTION_CUES[action_kind]
        )
        for action_kind in requested
        if any(word in _ASK_OMISSION_ACTION_CUES[action_kind] for word in words)
    }
    if "PLACE" in requested and "PLACE" not in positioned:
        placement_cue = next(
            (cue for cue in ("go", "move", "belong") if cue in words), None
        )
        if placement_cue is not None:
            positioned["PLACE"] = words.index(placement_cue)
    if _watch_lifecycle_is_object_move(fragment, words):
        positioned.pop("WALK", None)
        positioned["PLACE"] = words.index("move")
    positioned.update(_watch_lifecycle_implicit_action_positions(words))
    return positioned


def _watch_lifecycle_is_object_move(fragment: str, words: Sequence[str]) -> bool:
    return bool(
        re.search(
            r"\bmove\s+(?:(?:it|them|this|that)|(?:the|a|an)\s+"
            r"[^.!?;\n]{1,60}?)\s+(?:to|into|onto)\b",
            fragment,
        )
        and _ask_has_positive_cue(words, {"move"})
    )


def _watch_lifecycle_implicit_action_positions(
    words: Sequence[str],
) -> dict[str, int]:
    positioned: dict[str, int] = {}
    for cue, companion, action_kind in (
        ("take", "from", "GRAB"),
        ("return", "to", "PLACE"),
    ):
        if cue in words and companion in words and _ask_has_positive_cue(words, {cue}):
            positioned[action_kind] = words.index(cue)
    if words and words[0] in {"continue", "finish", "resume"}:
        for action_kind, cues in _ASK_OMISSION_ACTION_CUES.items():
            positions = [index for index, word in enumerate(words) if word in cues]
            if positions and _ask_has_positive_cue(words, cues):
                positioned[action_kind] = min(positions)
    return positioned


def _watch_instruction_units_are_covered(
    parsed: Sequence[tuple[str, str, Sequence[str]]],
    units: Sequence[tuple[str, set[str]]],
) -> bool:
    observed = [(action_kind, set(keyed)) for _, action_kind, keyed in parsed]
    return all(
        any(
            observed_kind == unit_kind
            and (not targets or bool(observed_targets & targets))
            for observed_kind, observed_targets in observed
        )
        for unit_kind, targets in units
    )


def _watch_completion_report_supported_by_history(
    human_say: str,
    assistant_instruction: str,
    last_successful_human_command: str,
    successful_commands: Sequence[str],
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
) -> bool:
    history = [str(command) for command in successful_commands if str(command).strip()]
    if last_successful_human_command.strip() and (
        not history or history[-1] != last_successful_human_command
    ):
        history.append(last_successful_human_command)
    if not history or not _completed_motor_claim_matches_commands(human_say, history):
        return False
    parsed = _watch_parse_commands(history)
    aliases = _watch_command_aliases(parsed, object_descriptions, motor_aliases)
    units = _watch_lifecycle_instruction_units(assistant_instruction, aliases)
    if units:
        return _watch_instruction_units_are_covered(parsed, units)
    grounded = _positive_alias_matches(assistant_instruction, aliases)
    if re.match(
        r"^(?:please\s+)?do not forget\b.+\bbefore\s+you\b",
        _normalize_find_instruction(assistant_instruction).strip(),
    ):
        grounded.update(
            key
            for key, values in aliases.items()
            if any(
                _contains_spoken_phrase(assistant_instruction, alias)
                for alias in _motor_alias_values(values)
            )
        )
    observed = {key for _, _, keyed in parsed for key in keyed}
    return bool(grounded & observed)


def _watch_special_ordered_units(
    fragment: str,
    aliases: Mapping[str, object],
) -> list[tuple[str, set[str]]]:
    if _watch_has_load_placement(fragment):
        return [
            (
                "PLACE",
                _watch_special_action_grounded_names(fragment, aliases, "PLACE"),
            )
        ]
    if _watch_has_return_for_retrieval(fragment):
        targets = _watch_special_action_grounded_names(fragment, aliases, "GRAB")
        return [("WALK", targets), ("GRAB", targets)]
    return []


def _watch_transport_ordered_units(
    fragment: str,
    aliases: Mapping[str, object],
) -> list[tuple[str, set[str]]]:
    match = _watch_transport_match(fragment)
    if match is None:
        return []
    units: list[tuple[str, set[str]]] = []
    subject = match.group("subject")
    if not _watch_transport_subject_is_pronoun(subject):
        subject_targets = _watch_key_phrase_matches(subject, aliases)
        if not subject_targets:
            subject_targets = _positive_alias_matches(subject, aliases)
        units.append(("GRAB", subject_targets))
    units.append(("WALK", _watch_transport_walk_targets(fragment, aliases)))
    return units


def _watch_order_unit_targets(
    fragment: str,
    aliases: Mapping[str, object],
    action_kind: str,
) -> set[str]:
    if action_kind == "WALK":
        transport_targets = _watch_transport_walk_targets(fragment, aliases)
        if transport_targets:
            return transport_targets
    return _watch_directly_grounded_names(fragment, aliases, action_kind)


def _watch_semantically_ordered_fragments(instruction: str) -> tuple[str, ...]:
    normalized = _normalize_find_instruction(instruction).strip()
    prefix_before = re.fullmatch(
        r"before\s+(?:you\s+)?(.+?)[,;]\s*(.+?)[.]?",
        normalized,
    )
    if prefix_before is not None:
        later, earlier = prefix_before.groups()
        return (*_find_action_clauses(earlier), *_find_action_clauses(later))
    infix_after = re.fullmatch(r"(.+?)\s+after\s+(.+?)[.]?", normalized)
    if infix_after is not None:
        later, earlier = infix_after.groups()
        return (*_find_action_clauses(earlier), *_find_action_clauses(later))
    return _find_action_clauses(normalized)


def _watch_has_redundant_walk_run(
    parsed: Sequence[tuple[str, str, Sequence[str]]],
    assistant_instruction: str,
) -> bool:
    explicit_walks = sum(
        1
        for clause in _find_action_clauses(assistant_instruction)
        if _ask_action_is_requested(clause, "WALK")
        and _ask_has_directive_form(clause, _ASK_OMISSION_ACTION_CUES["WALK"])
    )
    allowed_run = max(1, explicit_walks)
    run_length = 0
    for _verb, action_kind, _keyed in parsed:
        run_length = run_length + 1 if action_kind == "WALK" else 0
        if run_length > allowed_run:
            return True
    return False


def _watch_placement_preparation_steps(
    parsed: Sequence[tuple[str, str, Sequence[str]]],
    requested: set[str],
    assistant_instruction: str,
    aliases: Mapping[str, object],
    static_locations: Mapping[str, str],
    held: set[str],
    recent_subjects: set[str],
) -> set[tuple[int, str, str]]:
    if "PLACE" not in requested:
        return set()
    grounded = _watch_grounded_names(assistant_instruction, aliases, "PLACE")
    if _watch_placement_has_referential_subject(assistant_instruction):
        grounded.update(recent_subjects)
    allowed: set[tuple[int, str, str]] = set()
    for final_index, (final_verb, final_kind, final_keys) in enumerate(parsed):
        if final_kind != "PLACE" or len(final_keys) < 2:
            continue
        subject = final_keys[0]
        destination = final_keys[-1]
        if destination not in grounded - held or subject not in grounded | held:
            continue
        cursor = final_index - 1
        if (
            cursor >= 0
            and final_verb in {"PUTIN", "PUTINSIDE"}
            and parsed[cursor][1] == "OPEN"
            and parsed[cursor][2] == [destination]
        ):
            allowed.add((cursor, "OPEN", destination))
            cursor -= 1
        if (
            cursor >= 0
            and parsed[cursor][1] == "WALK"
            and parsed[cursor][2] == [destination]
        ):
            allowed.add((cursor, "WALK", destination))
            cursor -= 1
        if subject not in held and cursor >= 0:
            if parsed[cursor][1] == "GRAB" and parsed[cursor][2] == [subject]:
                allowed.add((cursor, "GRAB", subject))
                cursor -= 1
                location = str(static_locations.get(subject, ""))
                relation, separator, support = location.partition(":")
                source_targets = {subject}
                if separator and relation in {"in", "inside", "on"}:
                    source_targets.add(support)
                    if (
                        cursor >= 0
                        and relation in {"in", "inside"}
                        and parsed[cursor][1] == "OPEN"
                        and parsed[cursor][2] == [support]
                    ):
                        allowed.add((cursor, "OPEN", support))
                        cursor -= 1
                if (
                    cursor >= 0
                    and parsed[cursor][1] == "WALK"
                    and len(parsed[cursor][2]) == 1
                    and parsed[cursor][2][0] in source_targets
                ):
                    allowed.add((cursor, "WALK", parsed[cursor][2][0]))
    return allowed


def _watch_destination_relation_is_supported(
    verb: str,
    destination: str,
    object_descriptions: Mapping[str, object],
) -> bool:
    words = set(
        _WORD_RE.findall(str(object_descriptions.get(destination, "")).casefold())
    )
    if verb in {"PUTIN", "PUTINSIDE"}:
        return bool(
            words
            & {
                "bin",
                "cabinet",
                "container",
                "dishwasher",
                "fridge",
                "oven",
                "refrigerator",
                "sink",
                "washer",
            }
        )
    return bool(
        words
        & {
            "area",
            "bed",
            "console",
            "counter",
            "rack",
            "shelf",
            "station",
            "table",
            "tray",
        }
    )


def _watch_instruction_is_actionable(instruction: str) -> bool:
    if not instruction.strip():
        return False
    if _watch_requested_action_kinds(instruction):
        return True
    return _WATCH_ACTIONABLE_DIRECTIVE_RE.search(instruction) is not None


def _commands_include_action_kind(commands: Sequence[str], action_kind: str) -> bool:
    for command in commands:
        try:
            verb, _, _ = _command_key(command)
        except InteractionSpecError:
            continue
        if _find_action_kind(verb) == action_kind:
            return True
    return False


def _ask_speech_guard(
    family: str,
    commands: Sequence[str],
    human_say: str,
    *,
    unredacted_human_say: str,
    assistant_instruction: str,
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    visible_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    guide_completion_supported: bool,
) -> str:
    if family not in {"ASK", "GUIDE"}:
        return ""
    raw_say = unredacted_human_say or human_say
    entity_keys = set(object_descriptions) | set(motor_aliases)
    if (
        _ASK_INTERNAL_INTERFACE_SPEECH_RE.search(raw_say)
        or (family == "GUIDE" and _GUIDE_INTERNAL_INTERFACE_SPEECH_RE.search(raw_say))
        or any(
            "_" in str(key)
            and re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(str(key))}(?![A-Za-z0-9_])",
                raw_say,
                flags=re.IGNORECASE,
            )
            for key in entity_keys
        )
    ):
        return "ask_internal_interface_speech_blocked"
    if family == "GUIDE" and _ASK_UNOBSERVABLE_ATTRIBUTE_SPEECH_RE.search(raw_say):
        return "ask_unobservable_attribute_speech_blocked"
    if family == "GUIDE" and _GUIDE_INFERENCE_CONCLUSION_RE.search(raw_say):
        return "guide_inference_conclusion_speech_blocked"
    if (
        family == "GUIDE"
        and _narrate_action_is_explicitly_requested(assistant_instruction, "LOOKAT")
        and _GUIDE_VISUAL_EVIDENCE_REPORT_RE.search(raw_say)
        and not _GUIDE_NEGATIVE_VISUAL_REPORT_RE.search(raw_say)
        and not _guide_requests_spoken_observation(assistant_instruction)
    ):
        return "guide_visual_evidence_speech_blocked"
    if (
        family == "GUIDE"
        and _GUIDE_NEXT_STEP_SPEECH_RE.search(raw_say)
        and not _ORCHESTRATE_RECOVERY_EVIDENCE_RE.search(raw_say.replace("_", " "))
        and not guide_completion_supported
    ):
        return "guide_incomplete_instruction_next_step_blocked"
    if (
        family == "GUIDE"
        and _GUIDE_STEP_COMPLETION_CLAIM_RE.search(raw_say)
        and not guide_completion_supported
    ):
        return "guide_incomplete_instruction_completion_claim_blocked"
    if (
        _ASK_PREMATURE_COMPLETION_SPEECH_RE.search(raw_say)
        and not guide_completion_supported
    ):
        return "ask_premature_completion_speech_blocked"
    if (
        family == "GUIDE"
        and assistant_instruction.strip()
        and raw_say.strip()
        and not guide_completion_supported
        and not _guide_incomplete_speech_is_allowed(raw_say, assistant_instruction)
    ):
        return "guide_unverified_progress_speech_blocked"
    if _ask_speech_reveals_ungrounded_entity(
        raw_say,
        assistant_instruction,
        object_descriptions,
        motor_aliases,
        visible_object_keys,
        held_object_keys,
    ):
        return "ask_internal_interface_speech_blocked"
    return ""


def _guide_requests_spoken_observation(instruction: str) -> bool:
    return bool(
        re.search(
            r"\b(?:describe|read|report|say|tell\s+me|what\s+(?:does|do|is|are))\b",
            instruction,
            re.IGNORECASE,
        )
    )


def _guide_incomplete_speech_is_allowed(speech: str, instruction: str) -> bool:
    if _ORCHESTRATE_RECOVERY_EVIDENCE_RE.search(speech.replace("_", " ")) or (
        _GUIDE_NEGATIVE_VISUAL_REPORT_RE.search(speech)
    ):
        return True
    if _guide_requests_spoken_observation(instruction):
        return True
    if "?" not in speech:
        return False
    clarification = re.compile(
        r"^\s*(?:"
        r"(?:which|where)\b|"
        r"what\s+(?:exactly\s+)?(?:do\s+you\s+mean|should\s+i\s+do\s+with|"
        r"(?:item|object|one|place|location|direction|instruction|step)\b)|"
        r"do\s+you\s+mean\b[^?]*\b(?:ahead|behind|left|right|near|beside|at|or)\b|"
        r"(?:can|could|would)\s+you\s+(?:clarify|repeat|specify|"
        r"tell\s+me\s+where|show\s+me\s+where|guide\s+me)\b|"
        r"should\s+i\s+(?:close|grab|inspect|look|open|pick|place|put|"
        r"take|turn|use|walk)\b[^?]*\bor\b|"
        r"is\s+it\b[^?]*\b(?:ahead|behind|left|right|near|beside|at)\b)",
        re.IGNORECASE,
    )
    return clarification.search(speech) is not None


def _successful_command_completes_claimed_instruction(
    human_say: str,
    last_successful_human_command: str,
    assistant_instruction: str,
    successful_commands: Sequence[str] = (),
    pending_commands: Sequence[str] = (),
    object_aliases: Mapping[str, object] | None = None,
    landmark_aliases: Mapping[str, object] | None = None,
    target_sources: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    if not human_say.strip():
        return False
    history = [str(command) for command in successful_commands if str(command).strip()]
    if last_successful_human_command.strip() and (
        not history or history[-1] != last_successful_human_command
    ):
        history.append(last_successful_human_command)
    history.extend(str(command) for command in pending_commands if str(command).strip())
    if not history or not _guide_completion_intent(human_say):
        return False
    aliases = {**(object_aliases or {}), **(landmark_aliases or {})}
    if _guide_turn_direction_guard(history, assistant_instruction) or (
        _guide_placement_relation_guard(
            history,
            assistant_instruction,
            object_aliases or {},
            landmark_aliases or {},
        )
    ):
        return False
    parsed = _guide_parsed_commands(history)
    units = _guide_instruction_completion_units(
        assistant_instruction,
        object_aliases or {},
        landmark_aliases or {},
        target_sources or {},
    )
    if not units:
        named = _positive_alias_matches(assistant_instruction, aliases)
        return bool(
            any(kind in {"TURN", "WALK"} for kind, _arguments in parsed)
            and not _guide_navigation_only_guard(history, assistant_instruction, named)
        )
    return _guide_completed_unit_cursor(parsed, units) == len(units)


def _guide_completion_intent(human_say: str) -> bool:
    if not human_say.strip() or _ORCHESTRATE_RECOVERY_EVIDENCE_RE.search(
        human_say.replace("_", " ")
    ):
        return False
    return bool(
        _MOTOR_CLAIM_RE.match(human_say.strip())
        or _ASK_PREMATURE_COMPLETION_SPEECH_RE.search(human_say)
        or _GUIDE_NEXT_STEP_SPEECH_RE.search(human_say)
        or _GUIDE_STEP_COMPLETION_CLAIM_RE.search(human_say)
    )


def _guide_instruction_completion_units(
    instruction: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]] | None = None,
) -> list[tuple[str, set[str], set[str]]]:
    state = _GuideInstructionState([], set(), set(), set(), [], {})
    units: list[tuple[str, set[str], set[str]]] = []
    for sentence in _guide_instruction_sentences(instruction):
        sentence_context = _guide_context_object_groups(sentence, object_aliases)
        if sentence_context:
            state.context_groups = sentence_context
            state.focus = set(sentence_context[-1])
        sentence_units = _guide_sentence_completion_units(
            sentence,
            state,
            object_aliases,
            landmark_aliases,
            target_sources or {},
        )
        if sentence_units:
            units.extend(sentence_units)
            substantive = [
                unit
                for unit in sentence_units
                if unit[0] not in {"GRAB", "LOOKAROUND", "LOOKAT", "WAIT"}
            ]
            state.last_repeat = substantive[-2:]
    return _guide_with_required_container_opens(
        units, instruction, object_aliases, landmark_aliases
    )


@dataclass
class _GuideInstructionState:
    context_groups: list[set[str]]
    focus: set[str]
    last_container: set[str]
    last_destination: set[str]
    last_repeat: list[tuple[str, set[str], set[str]]]
    placed_destinations: dict[str, set[str]]


def _guide_sentence_completion_units(
    sentence: str,
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> list[tuple[str, set[str], set[str]]]:
    units: list[tuple[str, set[str], set[str]]] = []
    clauses = _guide_action_clauses(sentence)
    if all(not _guide_clause_action(clause)[0] for clause in clauses):
        mentioned = _guide_completion_alias_candidates(sentence, object_aliases)
        if len(mentioned) == 1:
            state.focus = mentioned
    for clause in clauses:
        units.extend(
            _guide_clause_completion_units(
                clause,
                sentence,
                units,
                state,
                object_aliases,
                landmark_aliases,
                target_sources,
            )
        )
    return units


def _guide_clause_completion_units(
    clause: str,
    sentence: str,
    sentence_units: Sequence[tuple[str, set[str], set[str]]],
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> list[tuple[str, set[str], set[str]]]:
    if re.search(r"\bdo\s+(?:the\s+)?same\s+with\b", clause, re.IGNORECASE):
        return _guide_repeated_clause_units(
            clause, sentence_units, state, object_aliases
        )
    action_kind, cue_end = _guide_clause_action(clause)
    if not action_kind:
        return _guide_context_clause_units(
            clause, sentence, state, object_aliases, landmark_aliases
        )
    if action_kind == "PLACE":
        return _guide_placement_clause_units(
            clause,
            cue_end,
            sentence_units,
            state,
            object_aliases,
            landmark_aliases,
            target_sources,
        )
    return _guide_direct_clause_units(
        clause,
        sentence,
        action_kind,
        cue_end,
        state,
        object_aliases,
        landmark_aliases,
        target_sources,
    )


def _guide_repeated_clause_units(
    clause: str,
    sentence_units: Sequence[tuple[str, set[str], set[str]]],
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
) -> list[tuple[str, set[str], set[str]]]:
    repeated = _guide_context_object_groups(clause, object_aliases)
    repeated_subject = set().union(*repeated) if repeated else set()
    template = [
        unit
        for unit in sentence_units
        if unit[0] not in {"GRAB", "LOOKAROUND", "LOOKAT", "WAIT"}
    ][-2:] or state.last_repeat
    return [
        (
            action_kind,
            repeated_subject if action_kind == "PLACE" else set(),
            set(destinations),
        )
        for action_kind, _subjects, destinations in template
    ]


def _guide_context_clause_units(
    clause: str,
    sentence: str,
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> list[tuple[str, set[str], set[str]]]:
    units: list[tuple[str, set[str], set[str]]] = []
    at_context = re.match(r"\s*\bat\b(?P<target>.+)$", clause, re.IGNORECASE)
    if at_context is not None:
        contextual_destination = _guide_completion_alias_candidates(
            at_context.group("target"), landmark_aliases
        )
        if contextual_destination:
            state.last_destination = contextual_destination
    retrieval = re.search(
        r"\breturn\s+to\b(?P<source>[^.!?;,]+?)\bfor\b(?P<subject>[^,]+)",
        clause,
        re.IGNORECASE,
    )
    navigation = tuple(
        re.finditer(
            r"\b(?:approach|follow|go|head|return|walk)\b",
            clause,
            re.IGNORECASE,
        )
    )
    navigated: set[str] = set()
    if navigation:
        navigated = _guide_completion_alias_candidates(
            clause[navigation[-1].start() :], landmark_aliases
        ) or _guide_placement_destination_candidates(
            clause[navigation[-1].start() :], object_aliases, landmark_aliases
        )
        if navigated:
            state.last_destination = navigated
    scan_after_navigation = bool(
        navigation
        and re.search(
            r"\b(?:look\s+around|take\s+(?:a|one)\s+(?:quick|final)?\s*"
            r"look\s+around)\b",
            sentence[navigation[-1].end() :],
            re.IGNORECASE,
        )
    )
    if navigated and retrieval is None and scan_after_navigation:
        units.append(("WALK", set(navigated), set()))
    if retrieval is not None:
        later_navigation = next(
            (
                match
                for match in navigation
                if match.start() >= retrieval.end("subject")
            ),
            None,
        )
        retrieval_source = _guide_completion_alias_candidates(
            retrieval.group("source"), landmark_aliases
        ) or _guide_placement_destination_candidates(
            retrieval.group("source"), object_aliases, landmark_aliases
        )
        if retrieval_source and later_navigation is not None:
            units.append(("WALK", set(retrieval_source), set()))
        retrieved = _guide_subject_groups(retrieval.group("subject"), object_aliases)
        units.extend(("GRAB", subjects, set()) for subjects in retrieved)
        if retrieved:
            state.focus = set(retrieved[-1])
        if later_navigation is not None and navigated:
            units.append(("WALK", set(navigated), set()))
    carry = re.search(r"\bcarry\b", clause, re.IGNORECASE)
    if carry is not None:
        units.extend(
            _guide_carry_clause_units(
                clause, sentence, carry.end(), state, object_aliases, landmark_aliases
            )
        )
    return units


def _guide_carry_clause_units(
    clause: str,
    sentence: str,
    cue_end: int,
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> list[tuple[str, set[str], set[str]]]:
    carried = _guide_direct_action_targets(
        clause[cue_end:], "GRAB", object_aliases, landmark_aliases
    )
    if not carried and re.search(r"\b(?:it|them)\b", clause, re.IGNORECASE):
        carried = set(state.focus)
    destination = _guide_placement_destination_candidates(
        clause, object_aliases, landmark_aliases
    )
    if carried:
        state.focus = carried
    if destination:
        state.last_destination = destination
    suffix = sentence[sentence.find(clause) + len(clause) :]
    if (
        carried
        and _guide_destination_is_surface(destination, object_aliases, landmark_aliases)
        and not _guide_has_same_transport_explicit_placement(suffix)
    ):
        return [("WALK", destination, set()), ("PLACE", carried, destination)]
    if destination and not _guide_has_same_transport_explicit_placement(suffix):
        return [("WALK", destination, set())]
    return []


def _guide_placement_clause_units(
    clause: str,
    cue_end: int,
    preceding_units: Sequence[tuple[str, set[str], set[str]]],
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> list[tuple[str, set[str], set[str]]]:
    destinations = _guide_clause_placement_destinations(
        clause,
        cue_end,
        state.last_destination,
        state.last_container,
        state.placed_destinations,
        object_aliases,
        landmark_aliases,
        target_sources,
    )
    if not destinations and state.last_container:
        destinations = set(state.last_container)
    placement_text = clause[cue_end:]
    context_groups = state.context_groups or [state.focus]
    if re.match(r"\s*it\b", placement_text, re.IGNORECASE) and state.focus:
        context_groups = [state.focus]
    groups = _guide_placement_subject_groups(
        placement_text, object_aliases, context_groups
    )
    if not groups and state.focus:
        groups = [set(state.focus)]
    units: list[tuple[str, set[str], set[str]]] = []
    for subjects in groups or [set()]:
        units.extend(
            _guide_transfer_prefix_units(
                clause,
                subjects,
                destinations,
                (*preceding_units, *units),
                state.context_groups,
                target_sources,
            )
        )
        units.append(("PLACE", subjects, set(destinations)))
        if subjects:
            state.focus = set(subjects)
            if destinations:
                for subject in subjects:
                    state.placed_destinations[subject] = set(destinations)
    if destinations:
        state.last_destination = set(destinations)
    return units


def _guide_transfer_prefix_units(
    clause: str,
    subjects: set[str],
    destinations: set[str],
    preceding_units: Sequence[tuple[str, set[str], set[str]]],
    context_groups: Sequence[set[str]],
    target_sources: Mapping[str, Sequence[str]],
) -> list[tuple[str, set[str], set[str]]]:
    if not subjects or not destinations:
        return []
    recent_grab = _guide_recent_unit_matches(preceding_units, "GRAB", subjects)
    context_names = set().union(*context_groups) if context_groups else set()
    requires_transfer = bool(
        recent_grab
        or subjects & context_names
        or re.search(r"(?<!-)\bfrom\b(?!-)", clause, re.IGNORECASE)
    )
    if not requires_transfer:
        return []
    return [] if recent_grab else [("GRAB", set(subjects), set())]


def _guide_recent_unit_matches(
    units: Sequence[tuple[str, set[str], set[str]]],
    action_kind: str,
    targets: set[str],
) -> bool:
    for kind, subjects, _destinations in reversed(units):
        if kind == "PLACE":
            return False
        if kind == action_kind and (not targets or not subjects or subjects & targets):
            return True
    return False


def _guide_subject_and_destination_share_location(
    subjects: set[str],
    destinations: set[str],
    target_sources: Mapping[str, Sequence[str]],
) -> bool:
    subject_locations = {
        str(source)
        for subject in subjects
        for source in target_sources.get(subject, ())
    }
    destination_locations = {
        str(source)
        for destination in destinations
        for source in target_sources.get(destination, (destination,))
    }
    return bool(subject_locations and subject_locations & destination_locations)


def _guide_clause_placement_destinations(
    clause: str,
    cue_end: int,
    last_destination: set[str],
    last_container: set[str],
    placed_destinations: Mapping[str, set[str]],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> set[str]:
    destinations = _guide_placement_destination_candidates(
        clause, object_aliases, landmark_aliases
    )
    if (
        last_destination
        and len(destinations) != 1
        and re.search(
            r"\b(?:back\s+)?(?:on|onto|to)\s+(?:that|the\s+same)\b|\bthere\b",
            clause,
            re.IGNORECASE,
        )
    ):
        return set(last_destination)
    relative = re.search(r"\bbeside\b(?P<reference>.+)$", clause, re.IGNORECASE)
    if re.search(r"\bto\s+the\s+(?:center|left|right)\b", clause, re.IGNORECASE):
        return set(last_destination)
    if not destinations and re.search(r"\b(?:there|back)\b", clause):
        return set(last_destination)
    if not destinations and relative is not None:
        references = _guide_completion_alias_candidates(
            relative.group("reference"), object_aliases
        )
        placed = {
            destination
            for reference in references
            for destination in placed_destinations.get(reference, ())
        }
        if placed:
            return placed
        if last_container and re.search(
            r"\b(?:in|inside|into)\s+(?:right\s+)?beside\b",
            clause,
            re.IGNORECASE,
        ):
            return set(last_container)
        return {
            str(source)
            for reference in references
            for source in target_sources.get(reference, ())
        }
    if not destinations and re.match(
        r"\s*(?:each|it|them|those|these)\b", clause[cue_end:], re.IGNORECASE
    ):
        return set(last_destination)
    return destinations


def _guide_direct_clause_units(
    clause: str,
    sentence: str,
    action_kind: str,
    cue_end: int,
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> list[tuple[str, set[str], set[str]]]:
    prior_focus = set(state.focus)
    targets = _guide_direct_clause_targets(
        clause,
        action_kind,
        cue_end,
        state,
        object_aliases,
        landmark_aliases,
        target_sources,
    )
    auxiliaries = _guide_explicit_tool_targets(
        clause,
        action_kind,
        cue_end,
        prior_focus,
        targets,
        object_aliases,
        landmark_aliases,
    )
    groups = (
        _guide_subject_groups(clause[cue_end:], object_aliases)
        if action_kind in {"COOK", "CUT"}
        else []
    )
    unit_action_kind = (
        "LOOKAROUND"
        if (
            action_kind == "LOOKAT"
            and _guide_clause_requests_broad_look(clause, targets)
        )
        else action_kind
    )
    units = (
        [(unit_action_kind, subjects, set(auxiliaries)) for subjects in groups]
        if len(groups) > 1
        else [
            (
                unit_action_kind,
                set() if unit_action_kind == "LOOKAROUND" else targets,
                set(auxiliaries),
            )
        ]
    )
    if groups:
        state.focus = set(groups[-1])
    elif targets:
        state.focus = set(targets)
    if (
        action_kind == "GRAB"
        and state.last_destination
        and _guide_destination_is_surface(
            state.last_destination, object_aliases, landmark_aliases
        )
        and re.search(
            r"\btake\b[^.!?;]*\bfrom\b[^.!?;]*\bto\b",
            clause,
            re.IGNORECASE,
        )
    ):
        units.append(("WALK", set(state.last_destination), set()))
    implied = _guide_implied_transport_units(
        clause, sentence, action_kind, targets, state, object_aliases, landmark_aliases
    )
    for implied_kind, implied_subjects, implied_destinations in implied:
        if implied_kind != "PLACE" or not implied_destinations:
            continue
        for subject in implied_subjects:
            state.placed_destinations[subject] = set(implied_destinations)
    units.extend(implied)
    return units


_GUIDE_TOOL_ACTION_KINDS = frozenset({"CUT", "STIR", "WIPE"})


def _guide_explicit_tool_targets(
    clause: str,
    action_kind: str,
    cue_end: int,
    prior_focus: set[str],
    action_targets: set[str],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> set[str]:
    if action_kind not in _GUIDE_TOOL_ACTION_KINDS:
        return set()
    aliases = {**object_aliases, **landmark_aliases}
    marker = re.search(r"\b(?:using|with)\b(?P<tool>.+)$", clause[cue_end:], re.I)
    if marker is not None:
        tool_text = re.split(r"[,.!;—–]", marker.group("tool"), maxsplit=1)[0]
        explicit = _guide_completion_alias_candidates(tool_text, aliases)
        if explicit:
            return explicit
    use = re.search(r"\buse\b(?P<tool>.+?)\bto\b", clause[:cue_end], re.I)
    if use is not None:
        explicit = _guide_completion_alias_candidates(use.group("tool"), aliases)
        if explicit:
            return explicit
        if re.search(r"\bit\b", use.group("tool"), re.I):
            return set(prior_focus)
    return set(prior_focus) if prior_focus.isdisjoint(action_targets) else set()


def _guide_direct_clause_targets(
    clause: str,
    action_kind: str,
    cue_end: int,
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> set[str]:
    targets = _guide_direct_action_targets(
        clause[cue_end:], action_kind, object_aliases, landmark_aliases
    )
    if action_kind == "GRAB":
        return _guide_resolve_grab_targets(
            clause, targets, state, object_aliases, landmark_aliases
        )
    if action_kind == "LOOKAT":
        return _guide_resolve_look_targets(
            clause, cue_end, targets, state, target_sources
        )
    if action_kind in {"CLOSE", "TURNON", "TURNOFF"} and not targets:
        targets = set(state.last_container or state.last_destination)
    if action_kind == "OPEN" and not targets:
        targets = set(state.last_destination)
    if action_kind == "OPEN" and targets:
        state.last_container = set(targets)
    return targets


def _guide_resolve_grab_targets(
    clause: str,
    targets: set[str],
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> set[str]:
    focused = targets & state.focus
    if (
        len(targets) > 1
        and len(focused) == 1
        and re.search(r"\b(?:inspected|selected|that|this)\b", clause, re.IGNORECASE)
    ):
        targets = focused
    if not targets:
        targets = (
            set(state.focus)
            if state.focus
            else set().union(*state.context_groups)
            if state.context_groups
            else set()
        )
    source = _guide_grab_source_destination(clause, landmark_aliases)
    if source:
        state.last_destination = source
    destination = _guide_grab_approach_destination(
        clause, object_aliases, landmark_aliases
    )
    if destination:
        state.last_destination = destination
    return targets


def _guide_resolve_look_targets(
    clause: str,
    cue_end: int,
    targets: set[str],
    state: _GuideInstructionState,
    target_sources: Mapping[str, Sequence[str]],
) -> set[str]:
    if (
        not targets
        and state.focus
        and re.match(r"\s*(?:it|them)\b", clause[cue_end:], re.IGNORECASE)
    ):
        return set(state.focus)
    if not targets and state.last_container:
        targets = {
            str(target)
            for target, sources in target_sources.items()
            if state.last_container & {str(source) for source in sources}
        }
    return targets or set(state.last_destination)


def _guide_grab_source_destination(
    clause: str, landmark_aliases: Mapping[str, object]
) -> set[str]:
    source = re.search(
        r"(?<!-)\bfrom\b(?!-)(?P<source>.+?)(?:\b(?:into|onto|to|toward|towards)\b|$)",
        clause,
        re.IGNORECASE,
    )
    if source is None:
        source = re.match(
            r"\s*\bat\b(?P<source>.+?)(?:,|\b(?:grab|pick|take)\b)",
            clause,
            re.IGNORECASE,
        )
    if source is None:
        return set()
    return _guide_completion_alias_candidates(source.group("source"), landmark_aliases)


def _guide_clause_requests_broad_look(clause: str, targets: set[str]) -> bool:
    if re.search(
        r"\b(?:look\s+around|take\s+(?:a|one)\s+(?:quick|final)?\s*"
        r"look\s+around|inspect\s+(?:the\s+)?completed\s+order)\b",
        clause,
        re.IGNORECASE,
    ):
        return True
    return bool(not targets and re.search(r"\blook\s+over\b", clause, re.IGNORECASE))


def _guide_grab_approach_destination(
    clause: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> set[str]:
    destination = _guide_placement_destination_candidates(
        clause, object_aliases, landmark_aliases
    )
    approach = re.search(
        r"\bapproach(?:ing)?\b(?P<target>.+?)(?:\bfrom\b|$)",
        clause,
        re.IGNORECASE,
    )
    if approach is None:
        return destination
    approached = _guide_completion_alias_candidates(
        approach.group("target"), landmark_aliases
    )
    return approached or destination


def _guide_implied_transport_units(
    clause: str,
    sentence: str,
    action_kind: str,
    targets: set[str],
    state: _GuideInstructionState,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> list[tuple[str, set[str], set[str]]]:
    if action_kind != "GRAB" or not state.last_destination:
        return []
    surface = _guide_destination_is_surface(
        state.last_destination, object_aliases, landmark_aliases
    )
    if re.search(r"\bbring\b", clause, re.IGNORECASE) and surface:
        return [("PLACE", set(targets), set(state.last_destination))]
    suffix = sentence[sentence.find(clause) + len(clause) :]
    carrying = re.search(r"\b(?:bring|carry)\b", clause, re.IGNORECASE) or re.search(
        r"\b(?:take|taking)\b[^.!?;]*\bto\b",
        clause,
        re.IGNORECASE,
    )
    container_close = re.search(r"\bclose\b", sentence, re.IGNORECASE) and (
        _guide_destination_is_container(
            state.last_destination, object_aliases, landmark_aliases
        )
    )
    if (
        carrying
        and not _guide_has_same_transport_explicit_placement(suffix)
        and (surface or container_close)
    ):
        return [("PLACE", set(targets), set(state.last_destination))]
    return []


_GUIDE_ACTION_START = (
    r"(?:add|bring|carry|close|cook|cut|do|grab|inspect|leave|lift|load|look|"
    r"move|open|pause|pick|place|placing|put|refrigerate|remove|reopen|return|"
    r"set|shut|slice|start|stir|stop|take|taking|turn|use|wait|wipe)"
)


def _guide_action_clauses(sentence: str) -> tuple[str, ...]:
    if ":" in sentence:
        suffix = sentence.split(":", 1)[1].strip()
        if re.match(rf"{_GUIDE_ACTION_START}\b", suffix, re.IGNORECASE):
            sentence = suffix
    return tuple(
        clause.strip(" ,")
        for clause in re.split(
            rf"(?:,\s*(?:and\s+)?(?:then\s+)?|\b(?:and\s+)?then\s+)"
            rf"(?={_GUIDE_ACTION_START}\b)|"
            rf"\band\s+(?={_GUIDE_ACTION_START}\b)",
            sentence,
            flags=re.IGNORECASE,
        )
        if clause.strip(" ,")
    )


def _guide_clause_action(clause: str) -> tuple[str, int]:
    normalized = clause.casefold()
    if re.search(r"\b(?:do\s+not|don't|never)\b", normalized):
        return "", 0
    look = re.search(
        r"\b(?:inspect|look\s+(?:at|around|over)|take\s+(?:a|one)\s+"
        r"(?:quick|final)?\s*look\s+around)\b",
        normalized,
    )
    candidates: list[tuple[int, str, int]] = []
    if look is not None:
        candidates.append((look.start(), "LOOKAT", look.end()))
    patterns = (
        ("TURNOFF", r"\bturn\s+off\b"),
        (
            "TURNON",
            r"\bturn\s+on\b|\bstart\s+(?!with\b)(?:the\s+)?(?:cycle|fan|it|stove|flat\s+top|dryer|washer)\b",
        ),
        ("OPEN", r"\breopen\b|(?<!already[ -])\bopen\b"),
        ("CLOSE", r"\bclose\b|\bshut\b"),
        ("CUT", r"\b(?:cut|slice)\b"),
        ("STIR", r"\bstir\b"),
        ("COOK", r"\bcook\b"),
        ("WIPE", r"\bwipe\b"),
        (
            "PLACE",
            r"\b(?:add|load|move|placing|puts?|refrigerate|set)\b|"
            r"(?<!in )\bplace\b|"
            r"\breturn\s+(?!to\b|through\b|back\b|is\b)(?=[^,.;]{1,80}\bto\b)|"
            r"\bleave\b(?![^.!?;]*\b(?:alone|there|untouched)\b)",
        ),
        ("GRAB", r"\b(?:bring|grab|lift|pick(?:\s+up)?|remove|take|taking)\b"),
        ("WAIT", r"\b(?:pause|stop|wait)\b"),
    )
    for action_kind, pattern in patterns:
        match = re.search(pattern, normalized)
        if match is not None and not _guide_action_match_is_false_positive(
            action_kind, match, normalized
        ):
            candidates.append((match.start(), action_kind, match.end()))
    if not candidates:
        return "", 0
    _position, action_kind, cue_end = min(candidates, key=lambda item: item[0])
    return action_kind, cue_end


def _guide_action_match_is_false_positive(
    action_kind: str, match: re.Match[str], normalized: str
) -> bool:
    if action_kind == "GRAB" and re.search(
        r"\btake\s+(?:a|one)\s+(?:quick|final)?\s*look\b", normalized
    ):
        return True
    if action_kind == "CUT" and (
        re.search(r"\b(?:is|are|was|were)\s+cut\b", normalized)
        or normalized.startswith("the cut ")
    ):
        return True
    if action_kind == "STIR" and re.search(r"\bstir[- ]fry\b", normalized):
        return True
    if action_kind == "COOK" and re.match(r"\s+target\b", normalized[match.end() :]):
        return True
    if action_kind == "CLOSE" and re.search(
        r"\b(?:is|are|was|were)\s+(?:already\s+)?(?:closed|shut)\b", normalized
    ):
        return True
    if action_kind == "GRAB" and match.group(0) == "bring" and match.start():
        return True
    return bool(
        action_kind == "PLACE"
        and match.group(0) == "leave"
        and re.search(r"\b(?:alternative|rightmost|untouched)\b", normalized)
    )


def _guide_context_object_groups(
    sentence: str, object_aliases: Mapping[str, object]
) -> list[set[str]]:
    marker = re.search(
        r"\b(?:begin|start)\s+with\b|\buse\b|\breturn\s+to\b|"
        r"\bdo\s+(?:the\s+)?same\s+with\b",
        sentence,
        re.IGNORECASE,
    )
    if marker is None:
        return []
    context = sentence[marker.end() :]
    if marker.group(0).casefold().startswith("return") and re.search(
        r"\bfor\b", context, re.IGNORECASE
    ):
        context = re.split(r"\bfor\b", context, flags=re.IGNORECASE)[-1]
    if ":" in context:
        context = context.rsplit(":", 1)[-1]
    action_boundary = re.search(
        rf"(?:,\s*(?:and\s+)?(?:then\s+)?|\band\s+)(?={_GUIDE_ACTION_START}\b)",
        context,
        re.IGNORECASE,
    )
    if action_boundary is not None:
        context = context[: action_boundary.start()]
    groups = _guide_subject_groups(context, object_aliases)
    minimal: list[set[str]] = []
    for group in groups:
        if any(existing < group for existing in minimal):
            continue
        minimal = [existing for existing in minimal if not group < existing]
        minimal.append(group)
    return minimal


def _guide_placement_subject_groups(
    text: str,
    object_aliases: Mapping[str, object],
    context_groups: Sequence[set[str]],
) -> list[set[str]]:
    boundaries = tuple(
        re.finditer(
            r"(?<!-)\b(?:from|in|inside|into|on|onto|to|toward|towards)\b(?!-)",
            text,
            re.IGNORECASE,
        )
    )
    destination_boundaries = [
        match
        for match in boundaries
        if match.group(0).casefold() not in {"from", "toward", "towards"}
    ]
    subject_text = (
        text[: destination_boundaries[-1].start()] if destination_boundaries else text
    )
    subject_text = re.split(
        r"(?<!-)\bfrom\b(?!-)", subject_text, maxsplit=1, flags=re.IGNORECASE
    )[0]
    if re.match(r"\s*(?:each|it|them|those|these)\b", subject_text, re.IGNORECASE):
        return [set(group) for group in context_groups if group]
    groups = _guide_subject_groups(subject_text, object_aliases)
    context_names = set().union(*context_groups) if context_groups else set()
    if not context_names:
        return groups
    return [group & context_names or group for group in groups]


def _guide_subject_groups(text: str, aliases: Mapping[str, object]) -> list[set[str]]:
    groups: list[set[str]] = []
    context_keys: set[str] = set()
    for phrase in re.split(r",|\band\s+then\b|\band\b", text, flags=re.IGNORECASE):
        candidates = _guide_completion_alias_candidates(phrase, aliases)
        if not candidates:
            continue
        candidates = _guide_subject_candidates_with_context(
            candidates, context_keys, aliases
        )
        count_match = re.search(
            r"(?<!of\s)\b(?:the|all)\s+(two|three|four)\b",
            phrase,
            re.IGNORECASE,
        )
        count = {"two": 2, "three": 3, "four": 4}.get(
            count_match.group(1).casefold() if count_match is not None else "", 1
        )
        groups.extend(set(candidates) for _ in range(count))
        if len(candidates) == 1:
            context_keys = set(candidates)
    if groups:
        return groups
    candidates = _guide_completion_alias_candidates(text, aliases)
    return [candidates] if candidates else []


def _guide_subject_candidates_with_context(
    candidates: set[str],
    context_keys: set[str],
    aliases: Mapping[str, object],
) -> set[str]:
    if len(candidates) <= 1 or not context_keys:
        return candidates
    context_words = set().union(
        *(_guide_alias_identity_words(key, aliases) for key in context_keys)
    )
    scores = {
        key: len(_guide_alias_identity_words(key, aliases) & context_words)
        for key in candidates
    }
    best = max(scores.values(), default=0)
    return (
        {key for key, score in scores.items() if score == best} if best else candidates
    )


def _guide_alias_identity_words(key: str, aliases: Mapping[str, object]) -> set[str]:
    generic = _GUIDE_ALIAS_TYPE_WORDS | {
        "a",
        "an",
        "farther",
        "farthest",
        "first",
        "labelled",
        "labeled",
        "left",
        "middle",
        "nearer",
        "nearest",
        "of",
        "right",
        "second",
        "the",
    }
    return {
        word
        for alias in _motor_alias_values(aliases.get(key))
        for word in _WORD_RE.findall(alias.casefold())
        if word not in generic and not any(char.isdigit() for char in word)
    }


def _guide_direct_action_targets(
    text: str,
    action_kind: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> set[str]:
    target_text = _guide_without_negative_tail(text)
    if action_kind == "GRAB":
        target_text = re.split(
            r"\b(?:from|to|toward|towards)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return _guide_completion_alias_candidates(target_text, object_aliases)
    if action_kind in {"CUT", "STIR", "WIPE"}:
        target_text = re.split(
            r"\b(?:using|with)\b", text, maxsplit=1, flags=re.IGNORECASE
        )[0]
    if action_kind in {"OPEN", "CLOSE", "TURNON", "TURNOFF"}:
        target_text = re.split(
            r"\b(?:at|beside|from|near|with)\b",
            target_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
    candidates = _guide_completion_alias_candidates(
        target_text, {**object_aliases, **landmark_aliases}
    )
    if action_kind != "COOK":
        return candidates
    non_documents = {
        key
        for key in candidates
        if not (
            _guide_alias_identity_words(key, {**object_aliases, **landmark_aliases})
            & {"clipboard", "document", "paper", "recipe", "ticket"}
        )
    }
    return non_documents or candidates


def _guide_destination_is_surface(
    destinations: set[str],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> bool:
    aliases = {**object_aliases, **landmark_aliases}
    return any(
        set(
            _WORD_RE.findall(
                " ".join(_motor_alias_values(aliases.get(destination))).casefold()
            )
        )
        & {
            "burner",
            "cooktop",
            "counter",
            "grill",
            "pass",
            "plate",
            "shelf",
            "stove",
            "table",
        }
        for destination in destinations
    )


def _guide_destination_is_container(
    destinations: set[str],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> bool:
    aliases = {**object_aliases, **landmark_aliases}
    return any(
        set(
            _WORD_RE.findall(
                " ".join(_motor_alias_values(aliases.get(destination))).casefold()
            )
        )
        & {
            "bin",
            "box",
            "cabinet",
            "chiller",
            "dryer",
            "freezer",
            "fridge",
            "locker",
            "oven",
            "rack",
            "refrigerator",
            "washer",
        }
        for destination in destinations
    )


def _guide_has_same_transport_explicit_placement(suffix: str) -> bool:
    placement = re.search(r"\b(?:load|place|placing|put|set)\b", suffix, re.IGNORECASE)
    if placement is None:
        return False
    next_subject = re.search(
        r"\b(?:bring|grab|pick|return\s+to\b[^,.;]*\bfor\b|take|taking)\b",
        suffix,
        re.IGNORECASE,
    )
    return next_subject is None or placement.start() < next_subject.start()


def _guide_with_required_container_opens(
    units: Sequence[tuple[str, set[str], set[str]]],
    instruction: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> list[tuple[str, set[str], set[str]]]:
    expanded: list[tuple[str, set[str], set[str]]] = []
    open_containers: set[str] = set()
    explicit_open = bool(
        re.search(r"(?<!already[ -])\bopen\b", instruction, re.IGNORECASE)
    )
    for index, unit in enumerate(units):
        action_kind, subjects, destinations = unit
        if action_kind == "OPEN":
            requires_approach = explicit_open or _guide_closed_storage_needs_walk(
                subjects, instruction, object_aliases, landmark_aliases
            )
            if (
                subjects
                and requires_approach
                and _guide_recent_unit_matches(expanded, "GRAB", set())
                and not _guide_recent_unit_matches(expanded, "WALK", subjects)
            ):
                expanded.append(("WALK", set(subjects), set()))
            open_containers.update(subjects)
        elif action_kind == "CLOSE":
            open_containers.difference_update(subjects or open_containers)
        elif (
            action_kind == "PLACE"
            and destinations
            and _guide_destination_is_container(
                destinations, object_aliases, landmark_aliases
            )
            and not destinations & open_containers
            and any(
                later_kind == "CLOSE"
                and (not later_subjects or bool(later_subjects & destinations))
                for later_kind, later_subjects, _later_destinations in units[
                    index + 1 :
                ]
            )
        ):
            requires_approach = explicit_open or _guide_closed_storage_needs_walk(
                destinations, instruction, object_aliases, landmark_aliases
            )
            if not _guide_recent_unit_matches(expanded, "GRAB", subjects):
                expanded.append(("GRAB", set(subjects), set()))
            if requires_approach and not _guide_recent_unit_matches(
                expanded, "WALK", destinations
            ):
                expanded.append(("WALK", set(destinations), set()))
            expanded.append(("OPEN", set(destinations), set()))
            open_containers.update(destinations)
        expanded.append(unit)
    return expanded


def _guide_closed_storage_needs_walk(
    containers: set[str],
    instruction: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> bool:
    if not (
        re.search(r"\b(?:load|place|put)\b", instruction, re.IGNORECASE)
        and re.search(r"\b(?:close|shut)\b", instruction, re.IGNORECASE)
    ):
        return False
    aliases = {**object_aliases, **landmark_aliases}
    words = {
        word
        for container in containers
        for alias in _motor_alias_values(aliases.get(container))
        for word in _WORD_RE.findall(alias.casefold())
    }
    return bool(words & {"cabinet", "dryer", "locker", "oven", "washer"})


def _guide_completed_unit_cursor(
    commands: Sequence[tuple[str, tuple[str, ...]]],
    units: Sequence[tuple[str, set[str], set[str]]],
) -> int:
    cursor = 0
    for command in commands:
        if cursor >= len(units):
            break
        action_kind, subjects, destinations = units[cursor]
        if _guide_completion_unit_matches(command, action_kind, subjects, destinations):
            cursor += 1
            continue
        command_kind, arguments = command
        if (
            action_kind == "GRAB"
            and command_kind == "PLACE"
            and cursor + 1 < len(units)
            and _guide_completion_unit_matches(command, *units[cursor + 1])
        ):
            cursor += 2
            continue
        if command_kind in {"WALK", "WAIT"}:
            continue
        if (
            action_kind == "PLACE"
            and command_kind == "GRAB"
            and (not subjects or not arguments or arguments[0] in subjects)
        ):
            continue
    return cursor


def _guide_placement_subject_matches(
    clause: str, object_aliases: Mapping[str, object]
) -> set[str]:
    destination = re.search(
        r"\b(?:in|inside|into|on|onto|to|toward|towards)\b",
        clause,
        re.IGNORECASE,
    )
    subject_phrase = clause[: destination.start()] if destination else clause
    return _guide_completion_alias_candidates(subject_phrase, object_aliases)


def _guide_completion_placement_destination_matches(
    clause: str, aliases: Mapping[str, object]
) -> set[str]:
    strong_matches = tuple(
        re.finditer(
            r"(?<!-)\b(?:inside|into|onto)\b(?!-)",
            clause,
            re.IGNORECASE,
        )
    )
    matches = strong_matches or tuple(
        re.finditer(
            r"(?<!-)\b(?:in|on|to|toward|towards)\b(?!-)",
            clause,
            re.IGNORECASE,
        )
    )
    if not matches:
        return set()
    destination_text = re.split(
        r"(?<!-)\b(?:after|at|before|beside|in|inside|into|on|onto|to|"
        r"toward|towards|with)\b(?!-)",
        clause[matches[0].end() :],
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    destination_text = _guide_without_negative_tail(destination_text)
    return _guide_completion_alias_candidates(destination_text, aliases)


def _guide_without_negative_tail(text: str) -> str:
    return re.split(
        r"(?:,|\band\b)?\s*\b(?:but\s+)?not\b|\b(?:avoid|instead\s+of)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]


def _guide_placement_destination_candidates(
    clause: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> set[str]:
    return _guide_completion_placement_destination_matches(
        clause, {**object_aliases, **landmark_aliases}
    )


def _guide_completion_alias_candidates(
    text: str, aliases: Mapping[str, object]
) -> set[str]:
    text_words = tuple(_WORD_RE.findall(text.casefold()))
    text_word_set = _guide_expanded_alias_words(set(text_words))
    text_numbers = {word for word in text_words if any(char.isdigit() for char in word)}
    type_words = _GUIDE_ALIAS_TYPE_WORDS
    text_types = text_word_set & type_words
    matches: dict[str, tuple[int, int, int, int, int, int]] = {}
    for raw_key, raw_aliases in aliases.items():
        key = str(raw_key)
        values = _motor_alias_values(raw_aliases)
        score = _guide_alias_candidate_score(
            text, text_word_set, text_numbers, text_types, key, values
        )
        if score is not None:
            matches[key] = score
    if not matches:
        return set()
    most_specific = max(matches.values())
    return {key for key, score in matches.items() if score == most_specific}


_GUIDE_ALIAS_SYNONYM_GROUPS = (
    {"fridge", "refrigerator"},
    {"bun", "roll"},
    {"bin", "can"},
    {"burner", "cooktop", "grill", "stove"},
    {"trash", "waste"},
)
_GUIDE_ALIAS_TYPE_WORDS = {
    "bin",
    "bottle",
    "box",
    "cabinet",
    "cart",
    "carton",
    "cloth",
    "counter",
    "dryer",
    "fan",
    "folder",
    "fridge",
    "grill",
    "knife",
    "locker",
    "mug",
    "oven",
    "pan",
    "plate",
    "rack",
    "refrigerator",
    "shelf",
    "table",
    "towel",
    "washer",
}


def _guide_expanded_alias_words(words: set[str]) -> set[str]:
    expanded = set(words)
    for synonyms in _GUIDE_ALIAS_SYNONYM_GROUPS:
        if expanded & synonyms:
            expanded.update(synonyms)
    return expanded


def _guide_alias_candidate_score(
    text: str,
    text_words: set[str],
    text_numbers: set[str],
    text_types: set[str],
    key: str,
    aliases: Sequence[str],
) -> tuple[int, int, int, int, int, int] | None:
    word_sets = [set(_WORD_RE.findall(alias.casefold())) for alias in aliases]
    alias_words = _guide_expanded_alias_words(set().union(*word_sets))
    alias_numbers = {
        word for word in alias_words if any(char.isdigit() for char in word)
    }
    if text_numbers and alias_numbers and text_numbers.isdisjoint(alias_numbers):
        return None
    widths = [
        len(words)
        for alias, words in zip(aliases, word_sets)
        if _contains_spoken_phrase(text, alias)
        and (len(words) > 1 or not words <= _GUIDE_ALIAS_TYPE_WORDS)
    ]
    type_match, extra_types = (
        max(
            (
                len(words & text_types & _GUIDE_ALIAS_TYPE_WORDS),
                -len((words & _GUIDE_ALIAS_TYPE_WORDS) - text_types),
            )
            for words in word_sets
        )
        if word_sets
        else (0, 0)
    )
    overlap = text_words & alias_words
    informative = overlap - {
        "a",
        "an",
        "the",
        "of",
        "labelled",
        "labeled",
        "storage",
    }
    if not (widths or informative):
        return None
    key_words = set(_WORD_RE.findall(key.casefold().replace("_", " ")))
    return (
        len(key_words) if key_words and key_words <= text_words else 0,
        type_match,
        extra_types,
        max(widths, default=0),
        len(informative),
        len(overlap),
    )


def _guide_completion_unit_matches(
    command: tuple[str, tuple[str, ...]],
    action_kind: str,
    subjects: set[str],
    destinations: set[str],
) -> bool:
    command_kind, arguments = command
    if command_kind != action_kind:
        return False
    if subjects and (not arguments or arguments[0] not in subjects):
        return False
    if action_kind in _GUIDE_TOOL_ACTION_KINDS and destinations:
        return len(arguments) == 2 and arguments[1] in destinations
    return not destinations or (len(arguments) >= 2 and arguments[-1] in destinations)


def _ask_speech_reveals_ungrounded_entity(
    human_say: str,
    assistant_instruction: str,
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    visible_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    *,
    block_any_ungrounded: bool = False,
) -> bool:
    if not human_say.strip():
        return False
    specific_aliases: dict[str, tuple[str, ...]] = {}
    grounding_aliases: dict[str, tuple[str, ...]] = {}
    for raw_key in set(object_descriptions) | set(motor_aliases):
        key = str(raw_key)
        description = str(object_descriptions.get(raw_key, "")).split("; currently", 1)[
            0
        ]
        authored = _motor_alias_values(motor_aliases.get(raw_key))
        broad = (*_guide_entity_aliases(key, description), *authored)
        grounding_aliases[key] = tuple(dict.fromkeys(broad))
        normalized_key = tuple(_WORD_RE.findall(key.casefold().replace("_", " ")))
        candidates = (*_find_visual_description_aliases(description), *authored)
        specific_aliases[key] = tuple(
            dict.fromkeys(
                alias
                for alias in candidates
                if len(_WORD_RE.findall(alias.casefold())) >= 2
                and tuple(_WORD_RE.findall(alias.casefold())) != normalized_key
            )
        )
    matches = {
        key
        for key, aliases in specific_aliases.items()
        if any(_contains_spoken_phrase(human_say, alias) for alias in aliases)
    }
    if not matches:
        return False
    allowed = {str(key) for key in (*visible_object_keys, *held_object_keys)}
    allowed.update(
        _instruction_grounded_names(assistant_instruction, grounding_aliases)
    )
    ungrounded = matches - allowed
    if block_any_ungrounded:
        return bool(ungrounded)
    if len(ungrounded) == 1:
        return True
    return bool(len(ungrounded) > 1 and _ASK_INTERFACE_ENUMERATION_RE.search(human_say))


def _completed_motor_claim_matches_commands(
    human_say: str, commands: Sequence[str]
) -> bool:
    match = _COMPLETED_MOTOR_CLAIM_RE.match(human_say.strip())
    if match is None:
        return False
    allowed_verbs = _COMPLETED_MOTOR_COMMANDS[match.group("verb").lower()]
    for command in commands:
        try:
            verb, _, _ = _command_key(command)
        except InteractionSpecError:
            continue
        if verb in allowed_verbs:
            return True
    return False


def _narrate_motor_claim_matches_commands(
    human_say: str, commands: Sequence[str]
) -> bool:
    if _MOTOR_CLAIM_RE.match(human_say.strip()) is None:
        return False
    for command in commands:
        try:
            verb, _, _ = _command_key(command)
        except InteractionSpecError:
            continue
        if _narrate_action_is_explicitly_requested(human_say, _find_action_kind(verb)):
            return True
    return False


def _trailing_narrate_question(human_say: str) -> str:
    speech = human_say.strip()
    if not speech.endswith("?"):
        return ""
    boundary = max(
        speech.rfind(".", 0, -1),
        speech.rfind("!", 0, -1),
        speech.rfind("?", 0, -1),
        speech.rfind("\n", 0, -1),
    )
    final_clause = speech[boundary + 1 :].strip()
    match = re.search(
        r"(?:^|[.!?;,]\s+|\band\s+)(?P<question>"
        r"(?:what|where|which|how|when|who|should|could|would|can|do|does|did|"
        r"is|are|am)\b[^?]*\?)\s*$",
        final_clause,
        re.IGNORECASE,
    )
    return match.group("question").strip() if match is not None else ""


def _contains_spoken_phrase(speech: str, phrase: str) -> bool:
    """Match an authored sentence inside free-form speech despite punctuation changes."""
    speech_tokens = re.findall(r"[^\W_]+", speech.casefold(), flags=re.UNICODE)
    phrase_tokens = re.findall(r"[^\W_]+", phrase.casefold(), flags=re.UNICODE)
    if not phrase_tokens or len(phrase_tokens) > len(speech_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        speech_tokens[index : index + width] == phrase_tokens
        for index in range(len(speech_tokens) - width + 1)
    )


def _family_specific_human_guard(
    family: str,
    source: str,
    commands: Sequence[str],
    find_target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
    assistant_instruction: str,
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    ask_movable_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    find_grounding_aliases: Mapping[str, object] | None,
    find_object_aliases: Mapping[str, object],
    find_target_aliases: Mapping[str, object],
    find_destination_aliases: Mapping[str, object],
    find_target_destinations: Mapping[str, Sequence[str]],
    find_open_object_keys: Sequence[str],
    find_bound_landmark_keys: Sequence[str],
    last_successful_human_command: str,
    active_instruction_successful_commands: Sequence[str],
    last_human_action_result: str,
    enforce_ask_instruction_guard: bool,
    enforce_find_model_guard: bool,
    human_at_keys: Sequence[str] | None,
    human_reasoning: str,
) -> tuple[list[str], str, str] | None:
    cook_guard = (
        None
        if source == "idle"
        else _nonfood_cook_target_guard(family, commands, find_target_keys)
    )
    if cook_guard is not None:
        return cook_guard
    if family == "ASK" and source == "idle":
        return [], "Could you answer my question?", "ask_wait_for_answer"
    if family == "ASK":
        return _ask_human_guard(
            source,
            commands,
            visible_object_keys,
            assistant_instruction,
            object_descriptions,
            motor_aliases,
            ask_movable_object_keys,
            held_object_keys,
            last_successful_human_command,
            enforce_ask_instruction_guard,
            human_at_keys,
            human_reasoning,
        )
    if family == "NARRATE":
        return _narrate_human_guard(
            source,
            commands,
            assistant_instruction,
            find_target_keys,
            visible_object_keys,
            held_object_keys,
            find_object_aliases,
            find_grounding_aliases or {},
            find_target_destinations,
            active_instruction_successful_commands,
        )
    if family == "ORCHESTRATE":
        return _orchestrate_human_guard(
            source,
            commands,
            assistant_instruction,
            find_target_keys,
            visible_object_keys,
            held_object_keys,
            object_descriptions,
            motor_aliases,
            find_object_aliases,
            find_grounding_aliases or {},
            find_target_destinations,
            active_instruction_successful_commands,
            last_human_action_result,
        )
    if family in _INSPECTED_MOVABLE_GROUNDING_FAMILIES:
        return _guide_human_guard(
            source,
            commands,
            assistant_instruction,
            find_target_keys,
            visible_object_keys,
            object_descriptions,
            find_object_aliases,
            find_grounding_aliases or {},
            find_target_destinations,
            find_open_object_keys,
            active_instruction_successful_commands,
            held_object_keys=held_object_keys,
            enforce_inspection_sequence=True,
            enforce_source_locations=True,
            enforce_instruction_alignment=True,
        )
    if family != "FIND":
        return None
    return _find_human_guard(
        source,
        commands,
        assistant_instruction,
        find_target_keys,
        visible_object_keys,
        held_object_keys,
        object_descriptions,
        find_grounding_aliases,
        find_object_aliases,
        find_target_aliases,
        find_destination_aliases,
        find_target_destinations,
        find_open_object_keys,
        find_bound_landmark_keys,
        enforce_find_model_guard,
    )


def _nonfood_cook_target_guard(
    family: str,
    commands: Sequence[str],
    movable_object_keys: Sequence[str],
) -> tuple[list[str], str, str] | None:
    if family not in {"NARRATE", "ORCHESTRATE"}:
        return None
    movable = {str(key) for key in movable_object_keys}
    for index, command in enumerate(commands):
        verb, arguments, _ = _command_key(command)
        if verb != "COOK" or (len(arguments) == 1 and arguments[0][0] in movable):
            continue
        return (
            list(commands[:index]),
            (
                ""
                if index
                else "I need to target the food itself, not the appliance. "
                "Which food should I cook?"
            ),
            "cook_nonfood_target_blocked",
        )
    return None


def _guide_human_guard(
    source: str,
    commands: Sequence[str],
    assistant_instruction: str,
    target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
    object_descriptions: Mapping[str, object],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
    open_object_keys: Sequence[str] = (),
    active_instruction_successful_commands: Sequence[str] = (),
    held_object_keys: Sequence[str] = (),
    enforce_inspection_sequence: bool = False,
    enforce_source_locations: bool = False,
    enforce_instruction_alignment: bool = False,
) -> tuple[list[str], str, str] | None:
    if source == "idle":
        return [], "What should I do next?", "guide_wait_for_instruction"
    if enforce_source_locations:
        source_guard = _guide_source_location_guard(
            commands,
            assistant_instruction,
            target_keys,
            object_aliases,
            landmark_aliases,
            target_sources,
        )
        if source_guard is not None:
            return source_guard
    if enforce_inspection_sequence:
        inspection_guard = _guide_inspection_sequence_guard(
            commands,
            assistant_instruction,
            target_keys,
            visible_object_keys,
            object_aliases,
            target_sources,
            active_instruction_successful_commands,
        )
        if inspection_guard is not None:
            return inspection_guard
    explicitly_located = _guide_targets_with_stated_sources(
        assistant_instruction,
        target_keys,
        object_aliases,
        landmark_aliases,
        target_sources,
    )
    guarded, blocked = _without_unseen_guide_references(
        commands,
        _GUIDE_VISIBILITY_GATED_VERBS,
        target_keys,
        (*visible_object_keys, *explicitly_located),
        object_descriptions,
    )
    if not blocked:
        if enforce_instruction_alignment:
            alignment_guard = _guide_command_alignment_guard(
                commands,
                assistant_instruction,
                object_aliases,
                landmark_aliases,
                target_sources,
                visible_object_keys,
                open_object_keys,
                active_instruction_successful_commands,
                held_object_keys,
            )
            if alignment_guard:
                clarification = (
                    "Which specific one do you mean, and where is it?"
                    if alignment_guard == "guide_ambiguous_reference_blocked"
                    else ""
                )
                return [], clarification, alignment_guard
        return None
    if guarded:
        return guarded, "", "guide_unseen_movable_blocked"
    return (
        [],
        "I can't see that object yet. Where exactly should I look for it?",
        "guide_unseen_movable_blocked",
    )


def _guide_source_location_guard(
    commands: Sequence[str],
    instruction: str,
    target_keys: Sequence[str],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> tuple[list[str], str, str] | None:
    guarded, conflict = _without_guide_source_conflicts(
        commands,
        instruction,
        target_keys,
        object_aliases,
        landmark_aliases,
        target_sources,
    )
    if not conflict:
        return None
    clarification = ""
    if not guarded:
        clarification = (
            "That item is not at the place you named. Which item should I use?"
        )
    return guarded, clarification, "guide_movable_source_conflict_blocked"


def _guide_inspection_sequence_guard(
    commands: Sequence[str],
    instruction: str,
    target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
    object_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
    successful_commands: Sequence[str],
) -> tuple[list[str], str, str] | None:
    if not _guide_requires_inspection_before_close(instruction):
        return None
    inspected = _guide_directly_inspected_targets(successful_commands, target_keys)
    for index, command in enumerate(commands):
        newly_inspected = _guide_directly_inspected_targets((command,), target_keys)
        if newly_inspected:
            inspected.update(newly_inspected)
            continue
        if _find_action_kind(_command_key(command)[0]) != "CLOSE":
            continue
        required = _guide_inspection_targets_for_close(
            command,
            instruction,
            target_keys,
            visible_object_keys,
            object_aliases,
            target_sources,
        )
        if not required or not required <= inspected:
            return list(commands[:index]), "", "guide_required_inspection_missing"
    return None


def _guide_command_alignment_guard(
    commands: Sequence[str],
    instruction: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
    visible_object_keys: Sequence[str],
    open_object_keys: Sequence[str],
    successful_commands: Sequence[str],
    held_object_keys: Sequence[str],
) -> str:
    if not commands or not instruction.strip():
        return ""
    aliases = {**object_aliases, **landmark_aliases}
    detail_guard = _guide_turn_direction_guard(commands, instruction) or (
        _guide_placement_relation_guard(
            commands, instruction, object_aliases, landmark_aliases
        )
    )
    if detail_guard:
        return detail_guard
    units = _guide_resolved_instruction_units(
        instruction,
        object_aliases,
        landmark_aliases,
        target_sources,
        visible_object_keys,
        open_object_keys,
        successful_commands,
    )
    parsed_history = _guide_parsed_commands(successful_commands)
    cursor = _guide_completed_unit_cursor(parsed_history, units)
    held = {str(key) for key in held_object_keys}
    for parsed_command in parsed_history:
        _guide_advance_held(held, parsed_command)
    named = _positive_alias_matches(instruction, aliases)
    leading_navigation = _guide_leading_navigation_targets(instruction, aliases)
    if not units:
        return _guide_navigation_only_guard(
            commands, instruction, named, successful_commands
        )
    for command in commands:
        parsed = _guide_parsed_commands((command,))
        if not parsed:
            continue
        cursor, guard = _guide_align_parsed_command(
            parsed[0],
            cursor,
            units,
            instruction,
            visible_object_keys,
            target_sources,
            leading_navigation,
            parsed_history,
            held,
        )
        if guard:
            return guard
    return ""


def _guide_align_parsed_command(
    command: tuple[str, tuple[str, ...]],
    cursor: int,
    units: Sequence[tuple[str, set[str], set[str]]],
    instruction: str,
    visible_object_keys: Sequence[str],
    target_sources: Mapping[str, Sequence[str]],
    leading_navigation: set[str],
    parsed_history: list[tuple[str, tuple[str, ...]]],
    held: set[str],
) -> tuple[int, str]:
    command_kind, _arguments = command
    if not _guide_command_has_required_held_object(command, held):
        return cursor, "guide_instruction_mismatch_blocked"
    if cursor >= len(units):
        if command_kind != "WAIT":
            return cursor, "guide_instruction_mismatch_blocked"
    elif _guide_reference_is_ambiguous(
        command, units[cursor], instruction, visible_object_keys
    ):
        return cursor, "guide_ambiguous_reference_blocked"
    elif _guide_completion_unit_matches(command, *units[cursor]):
        cursor += 1
    elif not _guide_current_unit_preparation_matches(
        command,
        units[cursor],
        target_sources,
        leading_navigation,
        instruction,
        parsed_history,
        tuple(held),
        units[cursor + 1 :],
    ):
        return cursor, "guide_instruction_mismatch_blocked"
    parsed_history.append(command)
    _guide_advance_held(held, command)
    return cursor, ""


def _guide_command_has_required_held_object(
    command: tuple[str, tuple[str, ...]], held: set[str]
) -> bool:
    action_kind, arguments = command
    if action_kind == "PLACE":
        return bool(arguments and arguments[0] in held)
    if action_kind in _GUIDE_TOOL_ACTION_KINDS and len(arguments) >= 2:
        return arguments[1] in held
    return True


def _guide_advance_held(held: set[str], command: tuple[str, tuple[str, ...]]) -> None:
    action_kind, arguments = command
    if action_kind == "GRAB" and arguments:
        held.clear()
        held.add(arguments[0])
    elif action_kind == "PLACE" and arguments:
        held.discard(arguments[0])


def _guide_reference_is_ambiguous(
    command: tuple[str, tuple[str, ...]],
    unit: tuple[str, set[str], set[str]],
    instruction: str,
    visible_object_keys: Sequence[str],
) -> bool:
    command_kind, _arguments = command
    action_kind, subjects, destinations = unit
    candidates = destinations if command_kind == "PLACE" else subjects
    visible = candidates & {str(key) for key in visible_object_keys}
    if visible:
        candidates = visible
    if len(candidates) <= 1:
        return False
    if re.search(
        r"\b(?:all|both|each|two|three|four|first|second|third|last|remaining|"
        r"left(?:most)?|right(?:most)?|middle|center|front|back|near(?:er|est)?|"
        r"far(?:ther|thest)?|ahead|behind|beside|adjacent|label(?:ed|led)|UNIT\s+[A-Z])\b",
        instruction,
        re.IGNORECASE,
    ):
        return False
    return command_kind == action_kind or (
        action_kind == "PLACE" and command_kind in {"GRAB", "WALK"}
    )


def _guide_turn_direction_guard(commands: Sequence[str], instruction: str) -> str:
    requested = {
        direction
        for direction in ("around", "left", "right")
        if re.search(rf"\bturn\s+{direction}\b", instruction, re.IGNORECASE)
    }
    actual_directions = {
        "TURN": "generic",
        "TURNAROUND": "around",
        "TURN_AROUND": "around",
        "TURNLEFT": "left",
        "TURN_LEFT": "left",
        "TURNRIGHT": "right",
        "TURN_RIGHT": "right",
    }
    for command in commands:
        try:
            verb, _arguments, _scalar = _command_key(command)
        except InteractionSpecError:
            continue
        actual = actual_directions.get(verb)
        if actual is not None and actual not in requested:
            return "guide_instruction_mismatch_blocked"
    return ""


def _guide_placement_relation_guard(
    commands: Sequence[str],
    instruction: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> str:
    by_destination, all_relations = _guide_explicit_placement_relations(
        instruction, object_aliases, landmark_aliases
    )
    verb_relations = {
        "PUT": "ON",
        "PUTBACK": "ON",
        "PUTIN": "INSIDE",
        "PUTINSIDE": "INSIDE",
        "PUTON": "ON",
        "PUTONFLOOR": "ON",
    }
    for command in commands:
        try:
            verb, arguments, _scalar = _command_key(command)
        except InteractionSpecError:
            continue
        if _find_action_kind(verb) != "PLACE":
            continue
        destination = arguments[-1][0] if arguments else ""
        expected = by_destination.get(destination, set())
        if not expected and destination:
            is_container = _guide_destination_is_container(
                {destination}, object_aliases, landmark_aliases
            )
            is_surface = _guide_destination_is_surface(
                {destination}, object_aliases, landmark_aliases
            )
            if is_container and not is_surface:
                expected = {"INSIDE"}
            elif is_surface and not is_container:
                expected = {"ON"}
            elif is_container and is_surface:
                implicit = _guide_implicit_placement_relation(instruction)
                expected = {implicit} if implicit else set()
        if not expected and len(all_relations) == 1:
            expected = all_relations
        if expected and verb_relations.get(verb) not in expected:
            return "guide_instruction_mismatch_blocked"
    return ""


def _guide_implicit_placement_relation(instruction: str) -> str:
    inside = re.search(
        r"\b(?:add|inside|into|load|refrigerate)\b", instruction, re.IGNORECASE
    )
    on = re.search(
        r"\b(?:beside|on|onto|place|return|set)\b", instruction, re.IGNORECASE
    )
    if inside is not None and on is None:
        return "INSIDE"
    if on is not None and inside is None:
        return "ON"
    return ""


def _guide_explicit_placement_relations(
    instruction: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> tuple[dict[str, set[str]], set[str]]:
    aliases = {**object_aliases, **landmark_aliases}
    by_destination: dict[str, set[str]] = {}
    all_relations: set[str] = set()
    for sentence in _guide_instruction_sentences(instruction):
        for clause in _guide_action_clauses(sentence):
            action_kind, cue_end = _guide_clause_action(clause)
            if action_kind != "PLACE":
                continue
            strong_relations = tuple(
                re.finditer(
                    r"\b(?:inside|into|onto)\b",
                    clause[cue_end:],
                    re.IGNORECASE,
                )
            )
            relations = strong_relations or tuple(
                re.finditer(
                    r"\b(?:on|in)\b(?!\s+front\b)",
                    clause[cue_end:],
                    re.IGNORECASE,
                )
            )
            if not relations:
                continue
            relation = relations[0]
            relation_kind = (
                "INSIDE"
                if relation.group(0).casefold() in {"in", "inside", "into"}
                else "ON"
            )
            destinations = _guide_completion_placement_destination_matches(
                clause, aliases
            )
            if (
                relation_kind == "INSIDE"
                and destinations
                and _guide_destination_is_surface(
                    destinations, object_aliases, landmark_aliases
                )
                and relation.group(0).casefold() == "in"
            ):
                relation_kind = "ON"
            all_relations.add(relation_kind)
            for destination in destinations:
                by_destination.setdefault(destination, set()).add(relation_kind)
    return by_destination, all_relations


def _guide_current_unit_preparation_matches(
    command: tuple[str, tuple[str, ...]],
    unit: tuple[str, set[str], set[str]],
    target_sources: Mapping[str, Sequence[str]],
    leading_navigation: set[str],
    instruction: str,
    successful_commands: Sequence[tuple[str, tuple[str, ...]]],
    held_object_keys: Sequence[str],
    future_units: Sequence[tuple[str, set[str], set[str]]] = (),
) -> bool:
    command_kind, arguments = command
    action_kind, subjects, destinations = unit
    if command_kind == "TURN":
        return bool(
            re.search(r"\bturn\s+(?:around|left|right)\b", instruction, re.IGNORECASE)
            and not any(kind == "TURN" for kind, _arguments in successful_commands)
        )
    if command_kind == "WALK":
        if not arguments:
            return bool(
                re.search(r"\b(?:face|turn\s+around)\b", instruction, re.IGNORECASE)
            )
        navigation_targets = set(subjects) | set(destinations)
        for target in (*subjects, *destinations):
            if target in {str(key) for key in held_object_keys}:
                continue
            navigation_targets.update(
                str(source) for source in target_sources.get(target, ())
            )
        future_navigation_targets = {
            target
            for future_kind, future_subjects, future_destinations in future_units
            for target in (
                *(future_subjects if future_kind == "WALK" else ()),
                *(future_destinations if future_kind in {"PLACE", "WALK"} else ()),
            )
        }
        if (
            arguments[0] in future_navigation_targets
            and arguments[0] not in navigation_targets
        ):
            return False
        return bool(
            arguments[0] in navigation_targets | leading_navigation
            and not _guide_immediately_repeats_navigation(command, successful_commands)
        )
    if action_kind == "PLACE" and command_kind == "GRAB" and arguments:
        return bool(
            (not subjects or arguments[0] in subjects)
            and arguments[0] not in {str(key) for key in held_object_keys}
            and not any(
                previous_kind == "GRAB"
                and previous_arguments
                and previous_arguments[0] == arguments[0]
                for previous_kind, previous_arguments in successful_commands
            )
        )
    return False


def _guide_leading_navigation_targets(
    instruction: str, aliases: Mapping[str, object]
) -> set[str]:
    targets: set[str] = set()
    for match in re.finditer(
        r"\b(?:approach|face|go|head|return|walk)\b(?P<target>[^,.;]*)",
        instruction,
        re.IGNORECASE,
    ):
        targets.update(_positive_alias_matches(match.group("target"), aliases))
    return targets


def _guide_immediately_repeats_navigation(
    command: tuple[str, tuple[str, ...]],
    successful_commands: Sequence[tuple[str, tuple[str, ...]]],
) -> bool:
    for previous in reversed(successful_commands):
        if previous[0] not in {"TURN", "WALK"}:
            return False
        if previous == command:
            return True
    return False


def _guide_navigation_only_guard(
    commands: Sequence[str],
    instruction: str,
    named: set[str],
    successful_commands: Sequence[str] = (),
) -> str:
    parsed_history = _guide_parsed_commands(successful_commands)
    for command_kind, arguments in _guide_parsed_commands(commands):
        current = (command_kind, arguments)
        if _guide_immediately_repeats_navigation(current, parsed_history):
            return "guide_instruction_mismatch_blocked"
        if command_kind == "TURN" and re.search(
            r"\bturn\s+(?:around|left|right)\b", instruction, re.IGNORECASE
        ):
            parsed_history.append(current)
            continue
        if (
            command_kind == "WALK"
            and arguments
            and arguments[0] in named
            and re.search(
                r"\b(?:approach|face|go|head|return|walk)\b",
                instruction,
                re.IGNORECASE,
            )
        ):
            parsed_history.append(current)
            continue
        return "guide_instruction_mismatch_blocked"
    return ""


def _guide_resolved_instruction_units(
    instruction: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
    visible_object_keys: Sequence[str],
    open_object_keys: Sequence[str] = (),
    successful_commands: Sequence[str] = (),
) -> list[tuple[str, set[str], set[str]]]:
    units = [
        (action_kind, set(subjects), set(destinations))
        for action_kind, subjects, destinations in _guide_instruction_completion_units(
            instruction, object_aliases, landmark_aliases, target_sources
        )
    ]
    openable_words = {
        "cabinet",
        "chiller",
        "dryer",
        "freezer",
        "fridge",
        "locker",
        "oven",
        "refrigerator",
        "washer",
    }
    contextual_containers = {
        str(key)
        for key in visible_object_keys
        if key in landmark_aliases
        and set(
            _WORD_RE.findall(
                " ".join(_motor_alias_values(landmark_aliases[key])).casefold()
            )
        )
        & openable_words
    }
    active_container: set[str] = set()
    inspected_landmarks = [
        arguments[0]
        for action_kind, arguments in _guide_parsed_commands(successful_commands)
        if action_kind == "LOOKAT" and arguments and arguments[0] in landmark_aliases
    ]
    successful_placements = [
        arguments
        for action_kind, arguments in _guide_parsed_commands(successful_commands)
        if action_kind == "PLACE" and len(arguments) >= 2
    ]
    used_contextual_view = False
    resolved: list[tuple[str, set[str], set[str]]] = []
    for action_kind, subjects, destinations in units:
        destinations = _guide_stable_unit_destinations(
            subjects, destinations, successful_placements, visible_object_keys
        )
        if action_kind == "OPEN":
            if not subjects and len(contextual_containers) == 1:
                subjects = set(contextual_containers)
            active_container = set(subjects)
        elif action_kind == "LOOKAT" and not subjects:
            if active_container:
                subjects = {
                    str(target)
                    for target, sources in target_sources.items()
                    if active_container & {str(source) for source in sources}
                }
            elif inspected_landmarks:
                subjects = {inspected_landmarks.pop(0)}
            elif len(contextual_containers) == 1 and not used_contextual_view:
                subjects = set(contextual_containers)
                used_contextual_view = True
        elif action_kind == "CLOSE" and not subjects and active_container:
            subjects = set(active_container)
        elif action_kind == "CLOSE" and not subjects and len(open_object_keys) == 1:
            subjects = {str(open_object_keys[0])}
        resolved.append((action_kind, subjects, destinations))
    return _guide_resolve_repeated_subjects(resolved, successful_placements)


def _guide_resolve_repeated_subjects(
    units: list[tuple[str, set[str], set[str]]],
    successful_placements: Sequence[tuple[str, ...]],
) -> list[tuple[str, set[str], set[str]]]:
    used: set[str] = set()
    placement_index = 0
    for index, (action_kind, subjects, destinations) in enumerate(units):
        if action_kind != "PLACE" or not subjects:
            continue
        matched = ""
        for arguments in successful_placements[placement_index:]:
            placement_index += 1
            if arguments and arguments[0] in subjects:
                matched = arguments[0]
                break
        narrowed = {matched} if matched else subjects - used
        if not narrowed:
            narrowed = set(subjects)
        units[index] = (action_kind, narrowed, destinations)
        _guide_narrow_preceding_grab(units, index, subjects, narrowed)
        if matched:
            used.add(matched)
    return units


def _guide_narrow_preceding_grab(
    units: list[tuple[str, set[str], set[str]]],
    placement_index: int,
    original_subjects: set[str],
    narrowed_subjects: set[str],
) -> None:
    for index in range(placement_index - 1, -1, -1):
        action_kind, subjects, destinations = units[index]
        if action_kind == "PLACE":
            return
        if action_kind == "GRAB" and subjects & original_subjects:
            units[index] = (action_kind, set(narrowed_subjects), destinations)
            return


def _guide_stable_unit_destinations(
    subjects: set[str],
    destinations: set[str],
    successful_placements: Sequence[tuple[str, ...]],
    visible_object_keys: Sequence[str],
) -> set[str]:
    if len(destinations) <= 1:
        return destinations
    completed = {
        arguments[-1]
        for arguments in successful_placements
        if (not subjects or arguments[0] in subjects) and arguments[-1] in destinations
    }
    if len(completed) == 1:
        return completed
    visible = destinations & {str(key) for key in visible_object_keys}
    return visible if len(visible) == 1 else destinations


def _guide_parsed_commands(
    commands: Sequence[str],
) -> list[tuple[str, tuple[str, ...]]]:
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for command in commands:
        try:
            verb, arguments, _ = _command_key(command)
        except InteractionSpecError:
            continue
        action_kind = _find_action_kind(verb)
        if action_kind == "LOOKAT" and verb in {"LOOKAROUND", "LOOKSCAN"}:
            action_kind = "LOOKAROUND"
        parsed.append(
            (
                action_kind,
                tuple(name for name, object_id in arguments if object_id != 0),
            )
        )
    return parsed


def _guide_requires_inspection_before_close(instruction: str) -> bool:
    return _narrate_action_is_explicitly_requested(
        instruction, "LOOKAT"
    ) and _narrate_action_is_explicitly_requested(instruction, "CLOSE")


def _guide_failed_placement_claim(
    human_say: str, action_result: str, assistant_instruction: str
) -> bool:
    if not _narrate_action_is_explicitly_requested(assistant_instruction, "PLACE"):
        return False
    if not _ORCHESTRATE_RECOVERY_EVIDENCE_RE.search(action_result.replace("_", " ")):
        return False
    command_match = re.search(r"\[([^\]]+)\]", action_result)
    if (
        command_match is None
        or _find_action_kind(command_match.group(1).upper()) != "PLACE"
    ):
        return False
    if _ORCHESTRATE_RECOVERY_EVIDENCE_RE.search(human_say.replace("_", " ")):
        return False
    return _GUIDE_PLACEMENT_SUCCESS_CLAIM_RE.search(human_say) is not None


def _guide_directly_inspected_targets(
    commands: Sequence[str], target_keys: Sequence[str]
) -> set[str]:
    targets = {str(key) for key in target_keys}
    inspected: set[str] = set()
    for command in commands:
        verb, arguments, _ = _command_key(command)
        if verb not in {"LOOKAT", "LOOK_AT"} or not arguments:
            continue
        if arguments[0][0] in targets:
            inspected.add(arguments[0][0])
    return inspected


def _without_guide_source_conflicts(
    commands: Sequence[str],
    instruction: str,
    target_keys: Sequence[str],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> tuple[list[str], bool]:
    source_constraints = _guide_explicit_source_constraints(
        instruction, object_aliases, landmark_aliases
    )
    if not source_constraints:
        return list(commands), False
    targets = {str(key) for key in target_keys}
    for index, command in enumerate(commands):
        verb, arguments, _ = _command_key(command)
        if _find_action_kind(verb) not in {"GRAB", "LOOKAT", "PLACE", "WALK"}:
            continue
        referenced = {name for name, _object_id in arguments} & targets
        for target in referenced:
            required_sources = source_constraints.get(target, set())
            if not required_sources or not _guide_has_ambiguous_source_peer(
                target, targets, object_aliases
            ):
                continue
            current_sources = {str(value) for value in target_sources.get(target, ())}
            if current_sources and current_sources.isdisjoint(required_sources):
                return list(commands[:index]), True
    return list(commands), False


def _guide_has_ambiguous_source_peer(
    target: str, targets: set[str], object_aliases: Mapping[str, object]
) -> bool:
    aliases = {
        " ".join(_WORD_RE.findall(alias.casefold()))
        for alias in _motor_alias_values(object_aliases.get(target))
        if len(_WORD_RE.findall(alias.casefold())) >= 2
    }
    identity_words = _guide_alias_identity_words(target, object_aliases)
    derived_or_document = {
        "brown",
        "browned",
        "clipboard",
        "cooked",
        "cut",
        "document",
        "finished",
        "half",
        "paper",
        "recipe",
        "sliced",
        "ticket",
    }
    for other in targets:
        if other == target:
            continue
        other_aliases = {
            " ".join(_WORD_RE.findall(alias.casefold()))
            for alias in _motor_alias_values(object_aliases.get(other))
            if len(_WORD_RE.findall(alias.casefold())) >= 2
        }
        other_identity = _guide_alias_identity_words(other, object_aliases)
        broad_identity_match = bool(
            not (identity_words | other_identity) & derived_or_document
            and identity_words & other_identity
        )
        if aliases & other_aliases or broad_identity_match:
            return True
    return False


def _guide_explicit_source_constraints(
    instruction: str,
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
) -> dict[str, set[str]]:
    all_aliases = {**object_aliases, **landmark_aliases}
    specific_landmarks = {
        str(key): _guide_specific_source_aliases(str(key), aliases)
        for key, aliases in landmark_aliases.items()
    }
    constraints: dict[str, set[str]] = {}
    context_sources: set[str] = set()
    for sentence in _guide_instruction_sentences(instruction):
        destinations = _guide_completion_placement_destination_matches(
            sentence, all_aliases
        )
        if re.match(r"\s*(?:at|face|go|head|return|walk)\b", sentence, re.IGNORECASE):
            contextual = _positive_alias_matches(sentence, specific_landmarks)
            if contextual:
                context_sources = contextual
        for source_match in re.finditer(
            r"(?<!-)\bfrom\b(?!-)", sentence, re.IGNORECASE
        ):
            source_tail = sentence[source_match.end() :]
            source_tail = re.split(
                r"\b(?:in|inside|into|on|onto|to|toward|towards)\b|[,;]",
                source_tail,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            explicit_sources = _positive_alias_matches(source_tail, specific_landmarks)
            if not explicit_sources:
                continue
            context_sources = explicit_sources
            subject_text = _guide_source_subject_text(sentence[: source_match.start()])
            subject_groups = _guide_subject_groups(subject_text, object_aliases)
            for target in set().union(*subject_groups) if subject_groups else set():
                constraints.setdefault(target, set()).update(explicit_sources)
        if not context_sources:
            continue
        use_match = re.search(r"\buse\b(?P<targets>.+)$", sentence, re.IGNORECASE)
        context_targets = (
            _guide_completion_alias_candidates(
                use_match.group("targets"), object_aliases
            )
            if use_match is not None
            else _guide_placement_subject_matches(sentence, object_aliases)
        ) - destinations
        for clause in _guide_action_clauses(sentence):
            action_kind, cue_end = _guide_clause_action(clause)
            if action_kind == "GRAB":
                context_targets.update(
                    _guide_direct_action_targets(
                        clause[cue_end:],
                        "GRAB",
                        object_aliases,
                        landmark_aliases,
                    )
                )
        for target in context_targets:
            constraints.setdefault(target, set()).update(context_sources)
    return constraints


def _guide_instruction_sentences(instruction: str) -> tuple[str, ...]:
    return tuple(
        sentence.strip()
        for sentence in re.split(r"[.!?;\n]+", _normalize_find_instruction(instruction))
        if sentence.strip()
    )


def _guide_source_subject_text(prefix: str) -> str:
    starts = tuple(
        match.end()
        for match in re.finditer(
            r"\b(?:add|bring|carry|grab|move|pick(?:\s+up)?|take)\b",
            prefix,
            re.IGNORECASE,
        )
    )
    return prefix[starts[-1] :] if starts else prefix


def _guide_explicit_source_landmarks(
    instruction: str, landmark_aliases: Mapping[str, object]
) -> set[str]:
    specific_aliases = {
        str(key): _guide_specific_source_aliases(str(key), aliases)
        for key, aliases in landmark_aliases.items()
    }
    context_landmarks: set[str] = set()
    destination_landmarks: set[str] = set()
    for clause in _find_action_clauses(instruction):
        matches = _positive_alias_matches(clause, specific_aliases)
        placement = re.search(
            r"\b(?:bring|carry|move|place|put|return|set|store|transfer)\b"
            r"[^.!?;\n]{0,180}?\b(?:in|inside|into|on|onto)\b|"
            r"\b(?:bring|carry|place|put|set|store|transfer)\b"
            r"[^.!?;\n]{0,180}?\bto\b|"
            r"\breturn\s+(?!to\b)[^.!?;\n]{1,180}?\bto\b",
            clause,
            re.IGNORECASE,
        )
        if placement is None and not _narrate_action_is_explicitly_requested(
            clause, "PLACE"
        ):
            context_landmarks.update(matches)
            continue
        destination_landmarks.update(
            _guide_completion_placement_destination_matches(clause, specific_aliases)
        )
        placement_prefix = (
            clause[: placement.end()] if placement is not None else clause
        )
        source_match = re.search(
            r"\b(?:at|from|on)\b(?P<source>.+)$",
            placement_prefix,
            re.IGNORECASE,
        )
        if source_match is not None:
            context_landmarks.update(
                _positive_alias_matches(source_match.group("source"), specific_aliases)
            )
    return context_landmarks - destination_landmarks


def _guide_inspection_targets_for_close(
    close_command: str,
    instruction: str,
    target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
    object_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> set[str]:
    _verb, arguments, _ = _command_key(close_command)
    if not arguments:
        return set()
    container = arguments[0][0]
    targets = {str(key) for key in target_keys}
    in_container = {
        target
        for target in targets
        if container in {str(source) for source in target_sources.get(target, ())}
    }
    named = _positive_alias_matches(instruction, object_aliases) & targets
    if in_container:
        named_in_container = named & in_container
        if named_in_container:
            return named_in_container
        visible_in_container = set(visible_object_keys) & in_container
        if visible_in_container:
            return visible_in_container
        return in_container
    if named:
        return named
    visible_targets = set(visible_object_keys) & targets
    return visible_targets if len(visible_targets) == 1 else set()


def _orchestrate_human_guard(
    source: str,
    commands: Sequence[str],
    assistant_instruction: str,
    target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
    successful_commands: Sequence[str],
    last_action_result: str,
) -> tuple[list[str], str, str] | None:
    visibility_guard = _guide_human_guard(
        source,
        commands,
        assistant_instruction,
        target_keys,
        visible_object_keys,
        object_descriptions,
        object_aliases,
        landmark_aliases,
        target_sources,
        active_instruction_successful_commands=successful_commands,
    )
    if visibility_guard is not None or not commands:
        return visibility_guard
    guard = _orchestrate_command_alignment_guard(
        commands,
        assistant_instruction,
        object_descriptions,
        motor_aliases,
        {**landmark_aliases, **object_aliases},
        visible_object_keys,
        held_object_keys,
        successful_commands,
        last_action_result,
    )
    if not guard:
        return None
    if guard == "orchestrate_action_order_unverified":
        return list(commands), "", guard
    return [], "", guard


def _orchestrate_command_alignment_guard(
    commands: Sequence[str],
    instruction: str,
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    policy_aliases: Mapping[str, object],
    visible_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    successful_commands: Sequence[str],
    last_action_result: str,
) -> str:
    parsed = _watch_parse_commands(commands)
    aliases = _orchestrate_command_aliases(
        object_descriptions, motor_aliases, policy_aliases
    )
    if not aliases:
        return ""
    recovery_active = bool(
        _ORCHESTRATE_RECOVERY_EVIDENCE_RE.search(last_action_result.replace("_", " "))
    )
    units = _orchestrate_action_units(
        instruction,
        aliases,
        preferred_primary={str(key) for key in visible_object_keys},
        recovery_active=recovery_active,
    )
    inactive_recovery_units = (
        []
        if recovery_active or not _ORCHESTRATE_RECOVERY_EVIDENCE_RE.search(instruction)
        else _orchestrate_action_units(
            instruction,
            aliases,
            preferred_primary={str(key) for key in visible_object_keys},
            recovery_active=True,
        )
    )
    cursor = _orchestrate_consumed_unit_cursor(successful_commands, units)
    completed_cursor = cursor
    held = {str(key) for key in held_object_keys}
    order_unverified = False
    for verb, action_kind, keyed in parsed:
        match = _orchestrate_next_matching_unit(
            action_kind, keyed, verb, units, cursor, held
        )
        if match is not None:
            order_unverified |= match > cursor
            cursor = match + 1
            _orchestrate_advance_held(held, action_kind, keyed)
            continue
        if _orchestrate_is_local_preparation(action_kind, keyed, units, cursor, held):
            _orchestrate_advance_held(held, action_kind, keyed)
            continue
        if not any(
            _orchestrate_unit_matches_command(action_kind, keyed, verb, unit, held)
            for unit in units
        ) and any(
            _orchestrate_unit_matches_command(action_kind, keyed, verb, unit, held)
            for unit in inactive_recovery_units
        ):
            return "orchestrate_unrequested_action_blocked"
        guard = _orchestrate_unmatched_action_guard(
            action_kind,
            keyed,
            verb,
            units,
            cursor,
            completed_cursor,
            held,
            instruction,
            aliases,
            successful_commands,
        )
        if guard:
            return guard
        order_unverified = True
    return "orchestrate_action_order_unverified" if order_unverified else ""


def _orchestrate_command_aliases(
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    policy_aliases: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for raw_key in set(object_descriptions) | set(motor_aliases) | set(policy_aliases):
        key = str(raw_key)
        description = str(object_descriptions.get(raw_key, "")).split("; currently", 1)[
            0
        ]
        candidates = dict.fromkeys(
            (
                *_guide_entity_aliases(key, description),
                *_motor_alias_values(motor_aliases.get(raw_key)),
                *_motor_alias_values(policy_aliases.get(raw_key)),
            )
        )
        label_match = re.search(
            r"\blabel(?:led|ed)\s+['\"]([^'\"]+)['\"]",
            description,
            re.IGNORECASE,
        )
        label_words = (
            set(_find_action_words(label_match.group(1)))
            if label_match is not None
            else set()
        )
        aliases[key] = tuple(
            alias
            for alias in candidates
            if not _orchestrate_alias_is_label_text_only(alias, label_words)
        )
    return aliases


def _orchestrate_alias_is_label_text_only(alias: str, label_words: set[str]) -> bool:
    words = set(_find_action_words(alias))
    return bool(label_words and words and words <= label_words)


def _orchestrate_action_units(
    instruction: str,
    aliases: Mapping[str, object],
    *,
    preferred_primary: set[str],
    recovery_active: bool,
) -> list[_OrchestrateActionUnit]:
    active_text = _orchestrate_select_conditional_branch(instruction, recovery_active)
    cut_tools = {
        str(key)
        for key, values in aliases.items()
        if str(key) in preferred_primary
        and set(_find_action_words(" ".join(_motor_alias_values(values))))
        & {"blade", "knife", "parer", "scissors"}
    }
    units: list[_OrchestrateActionUnit] = []
    focus: set[str] = set()
    held: set[str] = set()
    origins: dict[str, set[str]] = {}
    pending_context: set[str] = set()
    for clause in _find_action_clauses(active_text):
        if _ORCHESTRATE_PRESERVATION_RE.search(clause):
            continue
        occurrences = _orchestrate_alias_occurrences(clause, aliases)
        named = {key for key, _start, _width in occurrences}
        positioned = _orchestrate_action_kind_positions(clause)
        if not positioned:
            pending_context = named or pending_context
            continue
        named.update(pending_context)
        pending_context.clear()
        for cue_index, action_kind in positioned:
            unit = _orchestrate_make_action_unit(
                clause,
                cue_index,
                action_kind,
                occurrences,
                named,
                focus,
                held,
                origins,
                preferred_primary,
                cut_tools,
            )
            units.append(unit)
            focus, held = _orchestrate_advance_instruction_state(
                unit, focus, held, origins
            )
    return units


def _orchestrate_select_conditional_branch(text: str, recovery_active: bool) -> str:
    selected: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not re.match(r"\s*if\b", sentence, re.IGNORECASE):
            selected.append(sentence)
            continue
        otherwise = re.search(r"\botherwise\b", sentence, re.IGNORECASE)
        conditional = sentence[: otherwise.start()] if otherwise else sentence
        fallback = sentence[otherwise.end() :] if otherwise else ""
        branch = conditional if recovery_active else fallback
        if recovery_active:
            _condition, comma, consequence = branch.partition(",")
            branch = consequence if comma else ""
        selected.append(branch)
    return " ".join(selected)


def _orchestrate_action_kind_positions(clause: str) -> list[tuple[int, str]]:
    words = _find_action_words(clause)
    positioned: list[tuple[int, str]] = []
    for action_kind, cues in _ASK_ACTION_CUES.items():
        if not _ask_action_is_requested(clause, action_kind):
            continue
        explicitly_directed = _ask_has_directive_form(clause, cues)
        if action_kind == "PLACE":
            explicitly_directed |= _ask_targeted_drop_requests_placement(clause)
            explicitly_directed |= _ask_transport_requests_placement(clause)
        elif action_kind == "GRAB":
            explicitly_directed |= _ask_hold_requests_grab(clause)
        if not explicitly_directed:
            continue
        indices = [index for index, word in enumerate(words) if word in cues]
        if action_kind == "PLACE" and not indices:
            indices = [
                index
                for index, word in enumerate(words)
                if word in {"bring", "carry", "take"}
            ]
        if indices:
            positioned.append((min(indices), action_kind))
    if _orchestrate_oil_action_is_requested(clause):
        oil_index = next(
            index
            for index, word in enumerate(words)
            if word in {"coat", "drizzle", "oil", "oiling"}
        )
        positioned.append((oil_index, "POUR"))
    kinds = {action_kind for _index, action_kind in positioned}
    if kinds & {"TURNOFF", "TURNON"}:
        positioned = [row for row in positioned if row[1] not in {"FLIP", "TURN"}]
    elif "FLIP" in kinds:
        positioned = [row for row in positioned if row[1] != "TURN"]
    if "PLACE" in kinds and not re.search(r"\breturn\s+to\b", clause):
        positioned = [row for row in positioned if row[1] != "WALK"]
    return sorted(set(positioned))


def _orchestrate_oil_action_is_requested(clause: str) -> bool:
    words = _find_action_words(clause)
    cues = {"coat", "drizzle", "oil", "oiling"}
    return any(
        word in cues
        and not _ask_action_cue_is_negated(words, index)
        and _ask_has_directive_form(clause, {word})
        for index, word in enumerate(words)
    )


def _orchestrate_alias_occurrences(
    clause: str, aliases: Mapping[str, object]
) -> list[tuple[str, int, int]]:
    words = _find_action_words(clause)
    occurrences: list[tuple[str, int, int]] = []
    for raw_key, raw_aliases in aliases.items():
        key = str(raw_key)
        for alias in _motor_alias_values(raw_aliases):
            alias_words = _find_action_words(alias)
            if not alias_words or _alias_is_negated(clause, set(alias_words)):
                continue
            for start in range(len(words) - len(alias_words) + 1):
                if words[start : start + len(alias_words)] == alias_words:
                    occurrences.append((key, start, len(alias_words)))
    return occurrences


def _orchestrate_direct_entity(
    occurrences: Sequence[tuple[str, int, int]], after_index: int
) -> set[str]:
    later = [row for row in occurrences if row[1] > after_index]
    if not later:
        return set()
    earliest = min(start for _key, start, _width in later)
    longest = max(width for _key, start, width in later if start == earliest)
    return {
        key for key, start, width in later if start == earliest and width == longest
    }


def _orchestrate_destination_entity(
    clause: str,
    occurrences: Sequence[tuple[str, int, int]],
    cue_index: int,
) -> set[str]:
    words = _find_action_words(clause)
    for relation_index in range(cue_index + 1, len(words)):
        if words[relation_index] not in {
            "at",
            "in",
            "inside",
            "into",
            "on",
            "onto",
            "over",
            "to",
        }:
            continue
        destination = _orchestrate_direct_entity(occurrences, relation_index)
        if destination:
            return destination
    return set()


def _orchestrate_make_action_unit(
    clause: str,
    cue_index: int,
    action_kind: str,
    occurrences: Sequence[tuple[str, int, int]],
    named: set[str],
    focus: set[str],
    held: set[str],
    origins: Mapping[str, set[str]],
    preferred_primary: set[str],
    cut_tools: set[str],
) -> _OrchestrateActionUnit:
    primary = _orchestrate_direct_entity(occurrences, cue_index)
    if (
        action_kind == "GRAB"
        and focus
        and re.search(
            r"\b(?:grab|pick|retrieve|take)\s+(?:it|them)\b|"
            r"\bpick\s+(?:it|them)\s+up\b",
            clause,
            re.IGNORECASE,
        )
    ):
        primary = set(focus)
    visible_primary = primary & preferred_primary
    if len(visible_primary) == 1:
        primary = visible_primary
    destination: set[str] = set()
    if action_kind == "POUR":
        cue = _find_action_words(clause)[cue_index]
        poured_on = (
            set()
            if cue in {"coat", "oil", "oiling"}
            else _orchestrate_destination_entity(clause, occurrences, cue_index)
        )
        primary = poured_on or primary
    elif action_kind == "PLACE":
        destination = _orchestrate_destination_entity(clause, occurrences, cue_index)
        primary = set(held) or (primary - destination)
        if not destination and len(primary) == 1:
            destination = set(origins.get(next(iter(primary)), set()))
    elif not primary and (
        _continues_find_referent(clause) or action_kind in {"CLOSE", "OPEN"}
    ):
        primary = set(focus)
    placement_inside = None
    if action_kind == "PLACE":
        placement_inside = _ask_placement_relation_matches("PUTINSIDE", clause)
    auxiliaries = set(held)
    if action_kind == "CUT" and not auxiliaries and len(cut_tools) == 1:
        auxiliaries.update(cut_tools)
    return _OrchestrateActionUnit(
        action_kind=action_kind,
        primary=frozenset(primary),
        destination=frozenset(destination),
        named=frozenset(named),
        auxiliaries=frozenset(auxiliaries),
        placement_inside=placement_inside,
    )


def _orchestrate_advance_instruction_state(
    unit: _OrchestrateActionUnit,
    focus: set[str],
    held: set[str],
    origins: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    if unit.action_kind == "GRAB" and unit.primary:
        held = set(unit.primary)
        for subject in unit.primary:
            named_origins = set(unit.named - unit.primary)
            if named_origins:
                origins[subject] = named_origins
        return set(unit.primary), held
    if unit.action_kind == "PLACE":
        held -= set(unit.primary)
        return set(unit.destination or unit.primary), held
    return (set(unit.primary) if unit.primary else focus), held


def _orchestrate_unit_matches_command(
    action_kind: str,
    keyed: Sequence[str],
    verb: str,
    unit: _OrchestrateActionUnit,
    held: set[str],
) -> bool:
    if action_kind != unit.action_kind:
        return False
    if action_kind == "WAIT":
        return not keyed
    if action_kind == "PLACE":
        if len(keyed) < 2 or not unit.destination:
            return False
        inside = verb in {"PUTIN", "PUTINSIDE"}
        if unit.placement_inside is not None and inside != unit.placement_inside:
            return False
        return keyed[0] in unit.primary | held and keyed[-1] in unit.destination
    if not keyed or keyed[0] not in unit.primary:
        return False
    if action_kind in {"CUT", "POUR"}:
        permitted = unit.named | unit.auxiliaries | held
        return len(keyed) == 2 and set(keyed[1:]) <= permitted
    return len(keyed) == 1


def _orchestrate_next_matching_unit(
    action_kind: str,
    keyed: Sequence[str],
    verb: str,
    units: Sequence[_OrchestrateActionUnit],
    cursor: int,
    held: set[str],
) -> int | None:
    return next(
        (
            index
            for index in range(cursor, len(units))
            if _orchestrate_unit_matches_command(
                action_kind, keyed, verb, units[index], held
            )
        ),
        None,
    )


def _orchestrate_consumed_unit_cursor(
    successful_commands: Sequence[str], units: Sequence[_OrchestrateActionUnit]
) -> int:
    cursor = 0
    held: set[str] = set()
    for command in successful_commands:
        verb, arguments, _ = _command_key(command)
        action_kind = _find_action_kind(verb)
        keyed = [name for name, object_id in arguments if object_id != 0]
        if cursor < len(units) and _orchestrate_unit_matches_command(
            action_kind, keyed, verb, units[cursor], held
        ):
            cursor += 1
        _orchestrate_advance_held(held, action_kind, keyed)
    return cursor


def _orchestrate_is_local_preparation(
    action_kind: str,
    keyed: Sequence[str],
    units: Sequence[_OrchestrateActionUnit],
    cursor: int,
    held: set[str],
) -> bool:
    if len(keyed) != 1 or cursor >= len(units):
        return False
    unit = units[cursor]
    target = keyed[0]
    if action_kind == "WALK":
        return target in unit.named | unit.destination | unit.primary
    if action_kind == "OPEN":
        return bool(
            (
                unit.action_kind == "PLACE"
                and unit.placement_inside
                and target in unit.destination
            )
            or (unit.action_kind == "GRAB" and target in unit.named - unit.primary)
        )
    if action_kind != "GRAB" or held:
        return False
    if unit.action_kind == "PLACE":
        return target in unit.primary
    return unit.action_kind in {"CUT", "POUR"} and target in unit.auxiliaries


def _orchestrate_unmatched_action_guard(
    action_kind: str,
    keyed: Sequence[str],
    verb: str,
    units: Sequence[_OrchestrateActionUnit],
    cursor: int,
    completed_cursor: int,
    held: set[str],
    instruction: str,
    aliases: Mapping[str, object],
    successful_commands: Sequence[str],
) -> str:
    matches_completed_unit = any(
        _orchestrate_unit_matches_command(action_kind, keyed, verb, unit, held)
        for unit in units[:completed_cursor]
    )
    if (
        matches_completed_unit
        and _orchestrate_command_already_succeeded(
            action_kind, keyed, verb, successful_commands
        )
        and not _orchestrate_repeated_support_action_is_authorized(
            action_kind,
            keyed,
            instruction,
            aliases,
            successful_commands,
        )
    ):
        return "orchestrate_completed_action_repeat_blocked"
    same_kind = any(unit.action_kind == action_kind for unit in units)
    if (
        action_kind in _ORCHESTRATE_STRICT_ACTIONS
        and _orchestrate_strict_action_is_lexically_authorized(
            action_kind, keyed, instruction, aliases, held
        )
    ):
        return ""
    if action_kind in _ORCHESTRATE_STRICT_ACTIONS and same_kind:
        return "orchestrate_action_target_mismatch_blocked"
    if action_kind in _ORCHESTRATE_STRICT_ACTIONS:
        return "orchestrate_unrequested_action_blocked"
    return ""


def _orchestrate_command_already_succeeded(
    action_kind: str,
    keyed: Sequence[str],
    verb: str,
    successful_commands: Sequence[str],
) -> bool:
    inside = verb in {"PUTIN", "PUTINSIDE"}
    for successful_command in successful_commands:
        successful_verb, arguments, _ = _command_key(successful_command)
        if _find_action_kind(successful_verb) != action_kind:
            continue
        successful_keyed = [name for name, object_id in arguments if object_id != 0]
        if list(keyed) != successful_keyed:
            continue
        if action_kind != "PLACE" or inside == (
            successful_verb in {"PUTIN", "PUTINSIDE"}
        ):
            return True
    return False


def _orchestrate_repeated_support_action_is_authorized(
    action_kind: str,
    keyed: Sequence[str],
    instruction: str,
    aliases: Mapping[str, object],
    successful_commands: Sequence[str],
) -> bool:
    if not keyed or not _ORCHESTRATE_REPEAT_SEQUENCE_RE.search(instruction):
        return False
    grounded = {
        key
        for key, _start, _width in _orchestrate_alias_occurrences(instruction, aliases)
    }
    if keyed[0] not in grounded:
        return False
    if action_kind == "WALK":
        return len(keyed) == 1
    reusable: set[str] = set()
    for successful_command in successful_commands:
        successful_verb, arguments, _ = _command_key(successful_command)
        successful_kind = _find_action_kind(successful_verb)
        successful_keyed = [name for name, object_id in arguments if object_id != 0]
        if successful_kind in {"CUT", "POUR", "STIR", "WIPE"}:
            reusable.update(successful_keyed[1:])
    return (
        action_kind in {"GRAB", "PLACE"}
        and keyed[0] in reusable
        and set(keyed) <= grounded | reusable
    )


def _orchestrate_strict_action_is_lexically_authorized(
    action_kind: str,
    keyed: Sequence[str],
    instruction: str,
    aliases: Mapping[str, object],
    held: set[str],
) -> bool:
    if not keyed:
        return False
    grounded: set[str] = set()
    action_seen = False
    carry = False
    for clause in _find_action_clauses(instruction):
        kinds = {kind for _index, kind in _orchestrate_action_kind_positions(clause)}
        requested = action_kind in kinds
        same_sequence = bool(
            action_seen and _ORCHESTRATE_REPEAT_SEQUENCE_RE.search(clause)
        )
        if requested or same_sequence or (carry and not kinds):
            grounded.update(
                key
                for key, _start, _width in _orchestrate_alias_occurrences(
                    clause, aliases
                )
            )
        action_seen |= requested
        carry = requested or same_sequence or (carry and not kinds)
        if kinds and not requested:
            carry = False
    if keyed[0] not in grounded:
        return False
    if action_kind in {"CUT", "POUR"}:
        return len(keyed) == 2 and keyed[1] in grounded | held
    return len(keyed) == 1


def _orchestrate_advance_held(
    held: set[str], action_kind: str, keyed: Sequence[str]
) -> None:
    if not keyed:
        return
    if action_kind == "GRAB":
        held.clear()
        held.add(keyed[0])
    elif action_kind in {"DROP", "PLACE"}:
        held.discard(keyed[0])


def _without_unseen_guide_references(
    commands: Sequence[str],
    gated_verbs: frozenset[str],
    target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
    object_descriptions: Mapping[str, object],
) -> tuple[list[str], bool]:
    targets = {str(key) for key in target_keys}
    visible = {str(key) for key in visible_object_keys}
    if not targets:
        return list(commands), False
    for index, command in enumerate(commands):
        verb, arguments, _ = _command_key(command)
        referenced = {name for name, _ in arguments} & targets
        if verb in gated_verbs and referenced - visible:
            return list(commands[:index]), True
        if verb == "CUT" and arguments and arguments[0][0] in visible:
            visible.update(
                _guide_cut_result_keys(arguments[0][0], targets, object_descriptions)
            )
    return list(commands), False


def _guide_cut_result_keys(
    source: str,
    target_keys: Collection[str],
    object_descriptions: Mapping[str, object],
) -> set[str]:
    source_description = _guide_base_description(object_descriptions.get(source, ""))
    if not source_description:
        return set()
    produced_description = f"half {source_description}".casefold()
    return {
        key
        for key in target_keys
        if _guide_base_description(object_descriptions.get(key, "")).casefold()
        == produced_description
    }


def _guide_base_description(description: object) -> str:
    return (
        _guide_description_without_location(str(description))
        .split("; public motor references:", 1)[0]
        .strip()
    )


def _narrate_human_guard(
    source: str,
    commands: Sequence[str],
    assistant_instruction: str,
    target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
    active_instruction_successful_commands: Sequence[str],
) -> tuple[list[str], str, str] | None:
    if source == "idle":
        return [], "What should I do next?", "narrate_wait_for_instruction"
    targets = {str(key) for key in target_keys}
    held = {str(key) for key in held_object_keys}
    located = _narrate_targets_with_stated_sources(
        assistant_instruction,
        target_keys,
        object_aliases,
        landmark_aliases,
        target_sources,
    )
    if assistant_instruction.strip():
        for index, command in enumerate(commands):
            if _narrate_repeats_successful_action(
                command,
                assistant_instruction,
                active_instruction_successful_commands,
            ):
                return (
                    list(commands[:index]),
                    "",
                    "narrate_completed_action_repeat_blocked",
                )
            if _narrate_command_action_is_requested(command, assistant_instruction):
                continue
            return list(commands[:index]), "", "narrate_unrequested_action_blocked"
    named = _positive_alias_matches(assistant_instruction, object_aliases)
    inspected_and_described = {str(key) for key in visible_object_keys} & named
    operable = held | inspected_and_described | located
    for index, command in enumerate(commands):
        verb, arguments, _ = _command_key(command)
        referenced = {name for name, _ in arguments} & targets
        blocked = verb in {"GRAB", "LOOKAT", "WALK"} and bool(referenced - operable)
        if blocked:
            return (
                list(commands[:index]),
                (
                    ""
                    if index
                    else "I can't safely target that object yet. Where should I go or look?"
                ),
                "narrate_ungrounded_movable_blocked",
            )
    return None


def _narrate_repeats_successful_action(
    command: str,
    instruction: str,
    successful_commands: Sequence[str],
) -> bool:
    if re.search(
        r"\b(?:again|another|one\s+more|repeat|twice)\b", instruction, re.IGNORECASE
    ):
        return False
    command_key = _command_key(command)
    for successful in successful_commands:
        try:
            if _command_key(successful) == command_key:
                return True
        except InteractionSpecError:
            continue
    return False


def _narrate_command_action_is_requested(command: str, instruction: str) -> bool:
    verb, _, _ = _command_key(command)
    action_kind = _find_action_kind(verb)
    if _narrate_action_is_explicitly_requested(instruction, action_kind):
        return True
    if verb in {
        "LOOKAROUND",
        "LOOK_AROUND",
    } and _narrate_action_is_explicitly_requested(instruction, "TURN"):
        return True
    if action_kind in {"LOOKAT", "WAIT"} and not any(
        _narrate_action_is_explicitly_requested(instruction, candidate)
        for candidate in set(_ASK_ACTION_CUES) | set(_NARRATE_EXPLICIT_ACTION_CUES)
    ):
        return True
    return bool(
        action_kind in {"LOOKAT", "WALK"}
        and any(
            _narrate_action_is_explicitly_requested(instruction, primary)
            for primary in ("GRAB", "PLACE")
        )
    )


def _find_human_guard(
    source: str,
    commands: Sequence[str],
    assistant_instruction: str,
    find_target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    object_descriptions: Mapping[str, object],
    find_grounding_aliases: Mapping[str, object] | None,
    find_object_aliases: Mapping[str, object],
    find_target_aliases: Mapping[str, object],
    find_destination_aliases: Mapping[str, object],
    find_target_destinations: Mapping[str, Sequence[str]],
    find_open_object_keys: Sequence[str],
    find_bound_landmark_keys: Sequence[str],
    enforce_find_model_guard: bool,
) -> tuple[list[str], str, str] | None:
    if source == "idle":
        return [], "Could you answer my question?", "find_wait_for_answer"
    if not enforce_find_model_guard:
        return None
    visible_guarded, unseen_blocked = _without_unseen_find_targets(
        commands, find_target_keys, visible_object_keys
    )
    grounded, grounding_blocked = _without_ungrounded_find_references(
        visible_guarded,
        assistant_instruction,
        find_target_keys,
        visible_object_keys,
        held_object_keys,
        object_descriptions,
        find_grounding_aliases,
        find_object_aliases,
        find_target_aliases,
        find_destination_aliases,
        find_target_destinations,
        find_open_object_keys,
        find_bound_landmark_keys,
    )
    if grounding_blocked:
        return (
            grounded,
            "Which exact object or place should I use?",
            "find_instruction_grounding_blocked",
        )
    if unseen_blocked:
        if grounded:
            return grounded, "", "find_unseen_target_blocked"
        return (
            [],
            "I can't see the item yet. What should I inspect here?",
            "find_unseen_target_blocked",
        )
    return None


def _without_unseen_find_targets(
    commands: Sequence[str],
    find_target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
) -> tuple[list[str], bool]:
    return _without_unseen_references(
        commands,
        _FIND_VISIBILITY_GATED_VERBS,
        find_target_keys,
        visible_object_keys,
    )


def _guide_targets_with_stated_sources(
    assistant_instruction: str,
    target_keys: Sequence[str],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> set[str]:
    if not assistant_instruction.strip():
        return set()
    target_set = {str(target) for target in target_keys}
    targets = {
        str(key)
        for key, aliases in object_aliases.items()
        if str(key) in target_set
        and _guide_alias_is_positively_mentioned(assistant_instruction, aliases)
    }
    sources = _positive_alias_matches(
        assistant_instruction,
        {
            str(key): _guide_specific_source_aliases(str(key), aliases)
            for key, aliases in landmark_aliases.items()
        },
    )
    return {
        target
        for target in targets
        if sources.intersection(
            str(source) for source in target_sources.get(target, ())
        )
    }


def _narrate_targets_with_stated_sources(
    assistant_instruction: str,
    target_keys: Sequence[str],
    object_aliases: Mapping[str, object],
    landmark_aliases: Mapping[str, object],
    target_sources: Mapping[str, Sequence[str]],
) -> set[str]:
    if not assistant_instruction.strip():
        return set()
    targets = {str(target) for target in target_keys}
    source_aliases = {
        str(key): _guide_specific_source_aliases(str(key), aliases)
        for key, aliases in landmark_aliases.items()
    }
    grounded: set[str] = set()
    for clause in _find_action_clauses(assistant_instruction):
        named_targets = {
            str(key)
            for key, aliases in object_aliases.items()
            if str(key) in targets
            and _guide_alias_is_positively_mentioned(clause, aliases)
        }
        named_sources = _positive_alias_matches(clause, source_aliases)
        located = {
            target
            for target in named_targets
            if named_sources.intersection(
                str(source) for source in target_sources.get(target, ())
            )
        }
        if len(located) == 1:
            grounded.update(located)
    return grounded


def _guide_alias_is_positively_mentioned(text: str, aliases: object) -> bool:
    spoken_words = _grounding_alias_words(text)
    return any(
        alias_words
        and alias_words <= spoken_words
        and not _alias_is_negated(text, alias_words)
        for alias_words in _grounding_alias_word_sets(aliases)
    )


def _alias_phrase_is_explicitly_mentioned(text: str, aliases: object) -> bool:
    spoken = _WORD_RE.findall(text.lower().replace("_", " "))
    for alias in _motor_alias_values(aliases):
        phrase = _WORD_RE.findall(alias.lower().replace("_", " "))
        if phrase and any(
            spoken[index : index + len(phrase)] == phrase
            for index in range(len(spoken) - len(phrase) + 1)
        ):
            return True
    return False


def _without_ungrounded_find_references(
    commands: Sequence[str],
    assistant_instruction: str,
    find_target_keys: Sequence[str],
    visible_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    object_descriptions: Mapping[str, object],
    find_grounding_aliases: Mapping[str, object] | None,
    find_object_aliases: Mapping[str, object],
    find_target_aliases: Mapping[str, object],
    find_destination_aliases: Mapping[str, object],
    find_target_destinations: Mapping[str, Sequence[str]],
    find_open_object_keys: Sequence[str],
    find_bound_landmark_keys: Sequence[str],
) -> tuple[list[str], bool]:
    if find_grounding_aliases is None:
        return list(commands), False
    state = _FindGuardState(
        targets={str(key) for key in find_target_keys},
        visible={str(key) for key in visible_object_keys},
        held={str(key) for key in held_object_keys},
        opened={str(key) for key in find_open_object_keys},
        bound={str(key) for key in find_bound_landmark_keys},
        destinations={
            str(subject): {str(target) for target in target_keys}
            for subject, target_keys in find_target_destinations.items()
        },
    )
    movable_aliases = dict(find_object_aliases)
    movable_aliases.update(find_target_aliases)
    _augment_visible_find_movable_aliases(
        movable_aliases,
        object_descriptions,
        state.visible | state.held,
    )
    for index, command in enumerate(commands):
        verb, arguments, _ = _command_key(command)
        keyed = [name for name, object_id in arguments if object_id != 0]
        if not keyed:
            if _find_unkeyed_command_is_grounded(
                verb, assistant_instruction, state.held
            ):
                continue
            return list(commands[:index]), True
        action_kind = _find_action_kind(verb)
        if _find_action_is_explicitly_forbidden(assistant_instruction, action_kind):
            return list(commands[:index]), True
        grounding = _find_command_grounding(
            assistant_instruction,
            action_kind,
            state,
            find_grounding_aliases,
            find_object_aliases,
            find_destination_aliases,
            movable_aliases,
        )
        if not _find_command_is_grounded(
            verb, keyed, assistant_instruction, state, grounding
        ):
            return list(commands[:index]), True
        _advance_find_guard_state(state, verb, keyed)
    return list(commands), False


def _augment_visible_find_movable_aliases(
    aliases: dict[str, object],
    descriptions: Mapping[str, object],
    visible_or_held: set[str],
) -> None:
    for name in visible_or_held & set(aliases):
        visual_aliases = _find_visual_description_aliases(descriptions.get(name, ""))
        if not visual_aliases:
            continue
        existing = aliases.get(name, ())
        values = (
            tuple(str(value) for value in existing)
            if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes))
            else (str(existing),)
        )
        aliases[name] = tuple(dict.fromkeys((*values, *visual_aliases)))


def _find_visual_description_aliases(description: object) -> tuple[str, ...]:
    text = str(description).split("; currently", 1)[0].strip()
    if "; visually rendered as " in text:
        text = text.split("; visually rendered as ", 1)[1].strip()
    text = re.sub(r",\s*in\s+[^,;]+$", "", text, flags=re.IGNORECASE).strip()
    if not text:
        return ()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    label_match = re.search(r"\blabelled\s+['\"]?([^'\"]+)", text, re.IGNORECASE)
    category_parts = [
        part for part in parts if not part.lower().startswith("labelled ")
    ]
    category = category_parts[-1] if category_parts else ""
    label = label_match.group(1).strip() if label_match is not None else ""
    candidates = [text, category, label, f"{label} {category}".strip()]
    candidates.extend(sorted(_grounding_alias_words(text) - {"labelled"}))
    return tuple(
        dict.fromkeys(candidate for candidate in candidates if candidate.strip())
    )


@dataclass
class _FindGuardState:
    targets: set[str]
    visible: set[str]
    held: set[str]
    opened: set[str]
    bound: set[str]
    destinations: dict[str, set[str]]


@dataclass(frozen=True)
class _FindCommandGrounding:
    landmarks: set[str]
    helpers: set[str]
    targets: set[str]
    destinations: set[str]
    continuation: bool


def _find_command_grounding(
    instruction: str,
    action_kind: str,
    state: _FindGuardState,
    landmark_aliases: Mapping[str, object],
    object_aliases: Mapping[str, object],
    destination_aliases: Mapping[str, object],
    movable_aliases: Mapping[str, object],
) -> _FindCommandGrounding:
    groundable = state.visible | state.held
    visible_or_held_aliases = {
        name: aliases for name, aliases in movable_aliases.items() if name in groundable
    }
    landmark_candidates = _instruction_alias_candidates(instruction, landmark_aliases)
    movable = _action_grounded_names(instruction, visible_or_held_aliases, action_kind)
    landmarks = _action_grounded_names(instruction, landmark_aliases, action_kind)
    whole_instruction_fallback = action_kind in {"LOOKAT", "WALK"} or (
        action_kind in {"CLOSE", "OPEN"}
        and _RELATIONAL_COMMA_RE.search(instruction) is not None
    )
    if (
        not landmarks
        and whole_instruction_fallback
        and _find_action_is_requested(instruction, action_kind)
    ):
        mentioned_landmarks = _instruction_grounded_names(instruction, landmark_aliases)
        if len(mentioned_landmarks) == 1:
            landmarks = mentioned_landmarks
    destinations = _action_grounded_names(instruction, destination_aliases, action_kind)
    helpers = movable - state.targets
    targets = movable & state.targets
    continuation = _action_continues_find_referent(instruction, action_kind)
    if continuation:
        if not landmarks and len(landmark_candidates) <= 1:
            landmarks = state.bound & set(landmark_aliases)
        if not helpers:
            helpers = state.bound & set(object_aliases) & groundable
        if not targets:
            targets = state.bound & state.targets & groundable
        if not destinations:
            destinations = state.bound & set(destination_aliases)
    if continuation and not (helpers | targets):
        visible_named = state.visible & (state.targets | set(object_aliases))
        if len(visible_named) == 1:
            targets |= visible_named & state.targets
            helpers |= visible_named - state.targets
    return _FindCommandGrounding(
        landmarks=landmarks,
        helpers=helpers,
        targets=targets,
        destinations=destinations,
        continuation=continuation,
    )


def _find_command_is_grounded(
    verb: str,
    keyed: Sequence[str],
    instruction: str,
    state: _FindGuardState,
    grounding: _FindCommandGrounding,
) -> bool:
    permitted = grounding.landmarks | grounding.helpers | grounding.targets
    permitted |= _grounded_open_find_references(verb, instruction, state, grounding)
    held_targets = state.targets & state.held
    permitted |= {
        destination
        for target in held_targets
        for destination in state.destinations.get(target, set())
        if destination in grounding.destinations
    }
    if verb in _FIND_PLACEMENT_VERBS:
        if not _find_placement_subject_is_grounded(keyed, state, grounding):
            return False
        permitted.add(keyed[0])
    if verb == "GRAB" and (
        keyed[0] not in state.visible
        or keyed[0] not in grounding.targets | grounding.helpers
    ):
        return False
    return all(name in permitted for name in keyed)


def _grounded_open_find_references(
    verb: str,
    instruction: str,
    state: _FindGuardState,
    grounding: _FindCommandGrounding,
) -> set[str]:
    if verb not in {"CLOSE", "CLOSEDOOR"} or not _find_action_is_requested(
        instruction, "CLOSE"
    ):
        return set()
    if len(state.opened) == 1 and grounding.continuation:
        return set(state.opened)
    return state.opened & (grounding.landmarks | grounding.helpers)


def _find_placement_subject_is_grounded(
    keyed: Sequence[str],
    state: _FindGuardState,
    grounding: _FindCommandGrounding,
) -> bool:
    if not keyed or keyed[0] not in state.held:
        return False
    grounded_subjects = grounding.targets | grounding.helpers
    if grounding.continuation and len(state.held) == 1:
        grounded_subjects |= state.held
    return keyed[0] in grounded_subjects


def _advance_find_guard_state(
    state: _FindGuardState, verb: str, keyed: Sequence[str]
) -> None:
    if verb == "GRAB":
        state.held.add(keyed[0])
        state.bound = {keyed[0]}
    elif verb in _FIND_PLACEMENT_VERBS:
        state.held.discard(keyed[0])
        if len(keyed) > 1:
            state.bound = {keyed[-1]}
    if verb in {"OPEN", "OPENDOOR"}:
        state.opened.add(keyed[0])
    elif verb in {"CLOSE", "CLOSEDOOR"}:
        state.opened.discard(keyed[0])
    if verb in {"LOOKAT", "OPEN", "OPENDOOR", "WALK"}:
        state.bound = {keyed[0]}


def _find_unkeyed_command_is_grounded(
    verb: str, instruction: str, held: set[str]
) -> bool:
    if verb == "WAIT":
        return True
    if verb == "DROP":
        return bool(held) and _find_action_is_requested(instruction, "DROP")
    if verb not in _FIND_SAFE_UNKEYED_VERBS:
        return False
    return _find_action_is_requested(instruction, _find_action_kind(verb))


def _without_unseen_ask_targets(
    commands: Sequence[str],
    visible_object_keys: Sequence[str],
    movable_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
) -> tuple[list[str], str]:
    visible = {str(key) for key in visible_object_keys}
    movable = {str(key) for key in movable_object_keys}
    held = {str(key) for key in held_object_keys}
    for index, command in enumerate(commands):
        verb, arguments, _ = _command_key(command)
        referenced = {name for name, _ in arguments}
        if verb in _ASK_VISIBILITY_GATED_VERBS and (index > 0 or referenced - visible):
            return list(commands[:index]), "ask_unseen_grab_blocked"
        observable = visible | held if verb == "LOOKAT" else visible
        unseen_movables = {
            name
            for name in referenced - observable
            if name in movable or name.lower().startswith("item_")
        }
        if verb in {"LOOKAT", "WALK"} and unseen_movables:
            return list(commands[:index]), "ask_unseen_target_blocked"
    return list(commands), ""


def _ask_human_guard(
    source: str,
    commands: Sequence[str],
    visible_object_keys: Sequence[str],
    assistant_instruction: str,
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    movable_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    last_successful_human_command: str,
    enforce_instruction_guard: bool,
    human_at_keys: Sequence[str] | None,
    human_reasoning: str,
) -> tuple[list[str], str, str] | None:
    if len(commands) > 1:
        return [], "", "ask_multiple_commands_blocked"
    proposed = list(commands[:1])
    guarded, unseen_reason = _without_unseen_ask_targets(
        proposed, visible_object_keys, movable_object_keys, held_object_keys
    )
    if unseen_reason:
        return (
            guarded,
            "" if guarded else "I can't see that object yet. Where should I look?",
            unseen_reason,
        )
    if enforce_instruction_guard:
        grounded, instruction_guard = _without_unrequested_ask_actions(
            guarded,
            assistant_instruction,
            object_descriptions,
            motor_aliases,
            visible_object_keys,
            movable_object_keys,
            held_object_keys,
            last_successful_human_command,
            human_at_keys,
            human_reasoning,
        )
        if instruction_guard:
            return (
                grounded,
                "" if grounded else "What exact object or place should I use next?",
                instruction_guard,
            )
        if not grounded and _ask_new_instruction_requires_action(
            source,
            assistant_instruction,
            object_descriptions,
            motor_aliases,
            visible_object_keys,
            movable_object_keys,
            held_object_keys,
            last_successful_human_command,
            human_at_keys,
        ):
            return [], "", "ask_unrequested_action_blocked"
    return None


def _ask_new_instruction_requires_action(
    source: str,
    assistant_instruction: str,
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    visible_object_keys: Sequence[str],
    movable_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    last_successful_human_command: str,
    human_at_keys: Sequence[str] | None,
) -> bool:
    if source != "assistant_new" or not assistant_instruction.strip():
        return False
    action_kinds = _ask_unambiguous_action_kinds(assistant_instruction)
    if len(action_kinds) != 1:
        return False
    action_kind = next(iter(action_kinds))
    aliases = {
        str(key): _ask_entity_aliases(
            str(key), str(description).split("; currently", 1)[0]
        )
        for key, description in object_descriptions.items()
    }
    authored_aliases = {
        str(key): _motor_alias_values(value)
        for key, value in motor_aliases.items()
        if str(key) in aliases
    }
    grounded = _ask_action_grounded_names(
        assistant_instruction, authored_aliases, action_kind
    )
    if not grounded:
        grounded = _ask_action_grounded_names(
            assistant_instruction, aliases, action_kind
        )
    return _ask_action_is_currently_executable(
        action_kind,
        assistant_instruction,
        grounded,
        {str(key) for key in visible_object_keys},
        {str(key) for key in movable_object_keys},
        {str(key) for key in held_object_keys},
        last_successful_human_command,
        None if human_at_keys is None else {str(key) for key in human_at_keys},
    )


def _ask_unambiguous_action_kinds(instruction: str) -> set[str]:
    words = _find_action_words(instruction)
    requested = {
        action_kind
        for action_kind, cues in _ASK_OMISSION_ACTION_CUES.items()
        if _ask_has_positive_cue(words, cues)
        and _ask_has_directive_form(instruction, cues)
    }
    normalized = " ".join(words)
    if _ask_has_directive_form(instruction, {"hold"}) and _ask_hold_requests_grab(
        instruction
    ):
        requested.discard("WAIT")
        requested.add("GRAB")
    if _ask_has_directive_form(instruction, {"put", "set"}) and re.search(
        r"\b(?:put|set)\b(?:\s+\w+){0,5}\s+down\b", normalized
    ):
        requested.discard("PLACE")
        requested.add("DROP")
    if _ask_has_directive_form(instruction, {"answer"}) and re.search(
        r"\banswer\s+(?:the\s+)?phone\b", normalized
    ):
        requested.add("ANSWERPHONE")
    if _ask_has_directive_form(
        instruction, {"activate", "start", "switch", "turn"}
    ) and _ask_action_is_requested(instruction, "TURNON"):
        requested.add("TURNON")
    if _ask_has_directive_form(
        instruction, {"stop", "switch", "turn"}
    ) and _ask_action_is_requested(instruction, "TURNOFF"):
        requested.add("TURNOFF")
    if requested & {"TURNON", "TURNOFF"}:
        requested.discard("TURN")
    if _ask_has_directive_form(instruction, {"turn"}) and re.search(
        r"\bturn\s+(?:left|right|around)\b", normalized
    ):
        requested.add("TURN")
    if _ask_has_directive_form(instruction, {"look"}) and re.search(
        r"\blook\s+(?:around|at)\b", normalized
    ):
        requested.add("LOOKAT")
    if _ask_has_directive_form(instruction, {"stand"}) and re.search(
        r"\bstand\s+up\b", normalized
    ):
        requested.add("STANDUP")
    return requested


def _ask_has_positive_cue(words: Sequence[str], cues: Collection[str]) -> bool:
    negative = {"avoid", "except", "never", "not", "rather", "skip", "without"}
    for index, word in enumerate(words):
        if word not in cues:
            continue
        context = words[max(0, index - 4) : min(len(words), index + 4)]
        if not negative.intersection(context):
            return True
    return False


def _ask_has_directive_form(text: str, cues: Collection[str]) -> bool:
    directive_prefixes = (
        ("can", "you"),
        ("could", "you"),
        ("go", "ahead", "and"),
        ("i", "need", "you", "to"),
        ("i", "want", "you", "to"),
        ("next", "step", "is", "to"),
        ("please",),
        ("you", "must"),
        ("you", "need", "to"),
        ("you", "should"),
    )
    for clause in _find_action_clauses(text):
        words = _find_action_words(clause)
        for index, word in enumerate(words):
            if word not in cues:
                continue
            prefix = tuple(words[:index])
            if not prefix or any(
                prefix[-len(candidate) :] == candidate
                for candidate in directive_prefixes
            ):
                return True
    return False


def _ask_action_is_currently_executable(
    action_kind: str,
    instruction: str,
    grounded: set[str],
    visible: set[str],
    movable: set[str],
    held: set[str],
    last_successful_human_command: str,
    current_at: set[str] | None,
) -> bool:
    primary = _ask_primary_action_is_executable(
        action_kind,
        instruction,
        grounded,
        visible,
        movable,
        held,
        last_successful_human_command,
        current_at,
    )
    if primary is not None:
        return primary
    if action_kind in {"DROP", "FLIP"}:
        return len(held) == 1 and (
            action_kind == "DROP" or not grounded or grounded == held
        )
    if action_kind == "LOOKAT" and not grounded:
        return bool(
            re.search(
                r"\b(?:look\s+around|scan\s+(?:around|the\s+(?:area|room)))\b",
                instruction,
                re.IGNORECASE,
            )
        )
    if action_kind == "TURN" and not grounded:
        return True
    if action_kind in {"SQUAT", "STANDUP", "WAIT"}:
        return True
    if action_kind in _ASK_ONE_ENTITY_ACTIONS:
        return len(grounded) == 1
    if action_kind in _ASK_TWO_ENTITY_ACTIONS:
        return len(grounded) == 2
    return False


def _ask_primary_action_is_executable(
    action_kind: str,
    instruction: str,
    grounded: set[str],
    visible: set[str],
    movable: set[str],
    held: set[str],
    last_successful_human_command: str,
    current_at: set[str] | None,
) -> bool | None:
    if action_kind == "GRAB":
        return len(grounded) == 1 and grounded <= visible and not held
    if action_kind == "PLACE":
        destinations = grounded - held
        relation_is_explicit = _ask_placement_relation_matches(
            "PUT", instruction
        ) or _ask_placement_relation_matches("PUTINSIDE", instruction)
        return len(held) == 1 and len(destinations) == 1 and relation_is_explicit
    if action_kind != "WALK":
        return None
    if not grounded:
        return bool(re.search(r"\bforward\b", instruction, re.IGNORECASE))
    if len(grounded) != 1 or grounded & movable - visible:
        return False
    if current_at is not None and grounded <= current_at:
        return False
    return not _ask_last_walk_matches(last_successful_human_command, grounded)


def _ask_last_walk_matches(command: str, grounded: set[str]) -> bool:
    try:
        verb, arguments, _ = _command_key(command)
    except InteractionSpecError:
        return False
    targets = {name for name, object_id in arguments if object_id}
    return verb == "WALK" and targets == grounded


def _without_unrequested_ask_actions(
    commands: Sequence[str],
    assistant_instruction: str,
    object_descriptions: Mapping[str, object],
    motor_aliases: Mapping[str, object],
    visible_object_keys: Sequence[str],
    movable_object_keys: Sequence[str],
    held_object_keys: Sequence[str],
    last_successful_human_command: str,
    human_at_keys: Sequence[str] | None,
    human_reasoning: str,
) -> tuple[list[str], str]:
    visible = {str(key) for key in visible_object_keys}
    held = {str(key) for key in held_object_keys}
    aliases = {
        str(key): _ask_entity_aliases(
            str(key), str(description).split("; currently", 1)[0]
        )
        for key, description in object_descriptions.items()
    }
    authored_aliases = {
        str(key): _motor_alias_values(value)
        for key, value in motor_aliases.items()
        if str(key) in aliases
    }
    visible_aliases = {
        str(key): entity_aliases
        for key, entity_aliases in aliases.items()
        if str(key) in visible
    }
    current_locations = _ask_current_locations(object_descriptions, set(aliases))
    for index, command in enumerate(commands):
        guard = _ask_instruction_action_guard(
            command,
            assistant_instruction,
            last_successful_human_command,
            held,
            authored_aliases,
            aliases,
            visible_aliases,
            {str(key) for key in movable_object_keys},
            None if human_at_keys is None else {str(key) for key in human_at_keys},
            current_locations,
            human_reasoning,
        )
        if guard:
            return list(commands[:index]), guard
    return list(commands), ""


def _ask_instruction_action_guard(
    command: str,
    assistant_instruction: str,
    last_successful_human_command: str,
    held: set[str],
    authored_aliases: Mapping[str, object],
    aliases: Mapping[str, object],
    visible_aliases: Mapping[str, object],
    movable: set[str],
    current_at: set[str] | None,
    current_locations: Mapping[str, str],
    human_reasoning: str,
) -> str:
    verb, arguments, _ = _command_key(command)
    action_kind = _find_action_kind(verb)
    state_guard = _ask_action_state_guard(
        verb,
        command,
        assistant_instruction,
        last_successful_human_command,
        held,
        current_at,
    )
    if state_guard:
        return state_guard
    if (
        verb == "GRAB"
        and _ask_action_is_requested(assistant_instruction, "LOOKAT")
        and not _ask_action_is_requested(assistant_instruction, "GRAB")
    ):
        return "ask_inspection_grab_blocked"
    if verb == "DROP" and _ask_targeted_drop_requests_placement(assistant_instruction):
        return "ask_unrequested_action_blocked"
    if action_kind not in _ASK_ACTION_CUES:
        return ""
    keyed = [name for name, object_id in arguments if object_id != 0]
    if _ask_preparatory_action_is_permitted(
        verb,
        keyed,
        assistant_instruction,
        held,
        authored_aliases,
        aliases,
        current_at,
        current_locations,
        movable,
    ):
        return ""
    if (
        verb == "WALK"
        and _ask_action_is_requested(assistant_instruction, "PLACE")
        and not re.search(
            r"\b(?:approach|go|head|move|navigate|proceed|return|walk)\b",
            assistant_instruction,
            re.IGNORECASE,
        )
    ):
        return "ask_unrequested_action_blocked"
    if not _ask_action_is_requested(assistant_instruction, action_kind):
        return "ask_unrequested_action_blocked"
    if not keyed:
        return ""
    grounded = _ask_grounded_command_names(
        assistant_instruction,
        action_kind,
        authored_aliases,
        aliases,
        visible_aliases,
    )
    grounded.update(
        _ask_contextual_placement_destinations(
            assistant_instruction,
            held,
            authored_aliases,
            aliases,
            current_locations,
        )
    )
    visually_available = set(visible_aliases) | held
    visual_fallback_candidates = (
        (set(visible_aliases) & movable) | held
        if action_kind in {"FLIP", "GRAB", "LOOKAT"}
        else visually_available
    )
    if (
        verb in _ASK_VISUALLY_GROUNDED_SINGLE_ENTITY_ACTIONS
        and len(keyed) == 1
        and not grounded
        and visual_fallback_candidates == {keyed[0]}
    ):
        return ""
    if _ask_visually_reasoned_grab_is_permitted(
        verb,
        keyed,
        assistant_instruction,
        human_reasoning,
        grounded,
        visible_aliases,
        movable,
    ):
        return ""
    if _ask_visually_grounded_placement_is_permitted(
        verb,
        keyed,
        assistant_instruction,
        held,
        grounded,
        current_at,
    ):
        return ""
    return (
        ""
        if _ask_keyed_action_is_permitted(
            verb, keyed, assistant_instruction, held, grounded
        )
        else "ask_unrequested_action_blocked"
    )


def _ask_preparatory_action_is_permitted(
    verb: str,
    keyed: Sequence[str],
    assistant_instruction: str,
    held: set[str],
    authored_aliases: Mapping[str, object],
    aliases: Mapping[str, object],
    current_at: set[str] | None,
    current_locations: Mapping[str, str],
    movable: set[str],
) -> bool:
    return bool(
        _ask_preparatory_orientation_matches(verb, assistant_instruction)
        or _ask_held_visual_inspection_is_permitted(
            verb,
            keyed,
            assistant_instruction,
            held,
            authored_aliases,
            aliases,
            movable,
        )
        or _ask_walk_prepares_requested_placement(
            verb,
            keyed,
            assistant_instruction,
            held,
            authored_aliases,
            aliases,
            current_at,
            current_locations,
        )
        or _ask_walk_prepares_requested_grab(
            verb,
            keyed,
            assistant_instruction,
            held,
            authored_aliases,
            aliases,
            movable,
            current_at,
        )
    )


def _ask_uniquely_grounded_names(
    text: str, descriptions: Mapping[str, object]
) -> set[str]:
    spatial_words = {
        "back",
        "center",
        "corner",
        "edge",
        "front",
        "left",
        "middle",
        "near",
        "right",
        "side",
    }
    spoken_words = _grounding_words(text)
    matches = {
        str(name): (spoken_words & _description_grounding_words(description))
        - spatial_words
        for name, description in descriptions.items()
    }
    return {
        name
        for name, words in matches.items()
        if words
        and not any(
            words <= other_words
            for other_name, other_words in matches.items()
            if other_name != name and other_words
        )
    }


def _ask_current_locations(
    descriptions: Mapping[str, object], known_names: set[str]
) -> dict[str, str]:
    locations: dict[str, str] = {}
    for raw_name, raw_description in descriptions.items():
        match = re.search(
            r"\bcurrently\s+(?:in|inside|on)\s+([A-Za-z][A-Za-z0-9_]*)\b",
            str(raw_description),
            re.IGNORECASE,
        )
        if match is None:
            continue
        destination = match.group(1)
        if destination in known_names:
            locations[str(raw_name)] = destination
    return locations


def _ask_contextual_placement_destinations(
    instruction: str,
    held: set[str],
    authored_aliases: Mapping[str, object],
    aliases: Mapping[str, object],
    current_locations: Mapping[str, str],
) -> set[str]:
    if not _ask_action_is_requested(instruction, "PLACE"):
        return set()
    clauses = _find_action_clauses(instruction)
    negated_entities = set().union(
        *(
            _ask_negated_alias_names(clause, authored_aliases)
            | _ask_negated_alias_names(clause, aliases)
            for clause in clauses
        )
    )
    negated_destinations = negated_entities | {
        current_locations[name]
        for name in negated_entities
        if name in current_locations
    }
    destinations: set[str] = set()
    relation = re.compile(r"\b(?:alongside|beside|near|next\s+to|with)\b", re.I)
    for clause in clauses:
        if relation.search(clause) is None:
            continue
        anchors = _positive_alias_matches(clause, authored_aliases)
        anchors.update(_positive_alias_matches(clause, aliases))
        for anchor in anchors - held - negated_entities:
            destination = current_locations.get(anchor)
            if destination:
                destinations.add(destination)
    if _ask_transport_continues_named_destination(instruction):
        for clause in clauses:
            destinations.update(
                _ask_placement_destination_matches(clause, authored_aliases)
            )
            destinations.update(_ask_placement_destination_matches(clause, aliases))
    return destinations - held - negated_destinations


def _ask_visually_reasoned_grab_is_permitted(
    verb: str,
    keyed: Sequence[str],
    instruction: str,
    human_reasoning: str,
    grounded: set[str],
    visible_aliases: Mapping[str, object],
    movable: set[str],
) -> bool:
    if verb != "GRAB" or len(keyed) != 1 or grounded:
        return False
    target = keyed[0]
    if target not in visible_aliases or target not in movable:
        return False
    visual_words = set(_find_action_words(instruction))
    has_visual_attribute = bool(
        visual_words
        & {
            "black",
            "blue",
            "brown",
            "clear",
            "green",
            "large",
            "long",
            "orange",
            "red",
            "round",
            "short",
            "small",
            "square",
            "tall",
            "white",
            "yellow",
        }
    )
    has_spatial_selector = bool(
        visual_words
        & {
            "back",
            "center",
            "corner",
            "edge",
            "front",
            "left",
            "middle",
            "right",
            "side",
        }
    )
    if not has_visual_attribute or not has_spatial_selector:
        return False
    perceived = re.search(
        r"\b(?:can|could)\s+see\s+(.+?)(?=[.;]|\b(?:that|which)\s+"
        r"(?:correspond|fit|match))",
        human_reasoning,
        re.IGNORECASE,
    )
    if perceived is None:
        return False
    if not re.search(
        r"\b(?:correspond(?:s|ed)?|fit(?:s|ted)?|match(?:es|ed|ing)?)\b",
        human_reasoning,
        re.I,
    ):
        return False
    visible_movable_aliases = {
        name: value for name, value in visible_aliases.items() if name in movable
    }
    return _positive_alias_matches(perceived.group(1), visible_movable_aliases) == {
        target
    }


def _ask_held_visual_inspection_is_permitted(
    verb: str,
    keyed: Sequence[str],
    instruction: str,
    held: set[str],
    authored_aliases: Mapping[str, object],
    aliases: Mapping[str, object],
    movable: set[str],
) -> bool:
    if verb != "LOOKAT" or len(held) != 1 or set(keyed) != held:
        return False
    normalized = _normalize_find_instruction(instruction)
    visual_intent = bool(
        re.search(r"\bclose\s+to\s+(?:my|the|your\s+)?eyes?\b", normalized)
        or (
            re.search(r"\b(?:edge|face|label|side|text|weight)\b", normalized)
            and re.search(r"\bvisible\b", normalized)
        )
    )
    if not visual_intent:
        return False
    positive_presentation = False
    for clause in _find_action_clauses(normalized):
        words = _find_action_words(clause)
        for index, word in enumerate(words):
            if word in {"bring", "hold", "lift", "raise", "show"} and not (
                _ask_action_cue_is_negated(words, index)
            ):
                positive_presentation = True
    if not positive_presentation:
        return False
    all_aliases = {name: value for name, value in aliases.items() if name in movable}
    all_aliases.update(
        {name: value for name, value in authored_aliases.items() if name in movable}
    )
    named = _uniquely_grounded_names(instruction, all_aliases)
    return not named or named <= held


def _ask_transport_continues_named_destination(instruction: str) -> bool:
    clauses = _find_action_clauses(instruction)
    prior_has_destination = False
    for clause in clauses:
        words = _find_action_words(clause)
        for index, word in enumerate(words):
            if word not in {"bring", "carry", "take"}:
                continue
            if _ask_action_cue_is_negated(words, index) or (
                word == "take" and _ask_take_is_look_idiom(words, index)
            ):
                continue
            if "there" in words[index + 1 :] and prior_has_destination:
                return True
        prior_has_destination |= _ASK_SPECIFIC_DESTINATION_RE.search(clause) is not None
    return False


def _ask_entity_aliases(key: str, description: str) -> tuple[str, ...]:
    spatial_only = {
        "back",
        "center",
        "corner",
        "edge",
        "front",
        "left",
        "middle",
        "near",
        "right",
        "side",
    }
    aliases = {
        alias
        for alias in _guide_entity_aliases(key, description)
        if alias.casefold() not in spatial_only
    }
    aliases.update(
        re.sub(r"\bcabinets?\b", "chest", alias, flags=re.IGNORECASE)
        for alias in tuple(aliases)
        if re.search(r"\bcabinets?\b", alias, re.IGNORECASE)
    )
    return tuple(sorted(aliases, key=lambda alias: (len(alias.split()), alias)))


def _ask_grounded_command_names(
    assistant_instruction: str,
    action_kind: str,
    authored_aliases: Mapping[str, object],
    aliases: Mapping[str, object],
    visible_aliases: Mapping[str, object],
) -> set[str]:
    grounded = _ask_action_grounded_names(
        assistant_instruction, authored_aliases, action_kind
    )
    if not grounded:
        grounded = _ask_action_grounded_names(
            assistant_instruction, aliases, action_kind
        ) | _ask_action_grounded_names(
            assistant_instruction, visible_aliases, action_kind
        )
    if (
        not grounded
        and action_kind == "PLACE"
        and _ask_has_specific_destination_directive(assistant_instruction)
    ):
        return set()
    if not grounded:
        grounded = _ask_uniquely_grounded_names(assistant_instruction, aliases)
    return grounded


def _ask_keyed_action_is_permitted(
    verb: str,
    keyed: Sequence[str],
    assistant_instruction: str,
    held: set[str],
    grounded: set[str],
) -> bool:
    permitted = set(grounded)
    if verb in _ASK_PLACEMENT_VERBS:
        if len(held) != 1 or keyed[0] not in held:
            return False
        if not _ask_placement_relation_matches(verb, assistant_instruction):
            return False
        permitted.add(keyed[0])
    if verb == "DROP" and len(held) == 1 and keyed[0] in held:
        permitted.add(keyed[0])
    return all(name in permitted for name in keyed)


def _ask_walk_prepares_requested_placement(
    verb: str,
    keyed: Sequence[str],
    assistant_instruction: str,
    held: set[str],
    authored_aliases: Mapping[str, object],
    aliases: Mapping[str, object],
    current_at: set[str] | None,
    current_locations: Mapping[str, str],
) -> bool:
    destination_directive = _ask_has_specific_destination_directive(
        assistant_instruction
    )
    if verb != "WALK" or len(held) != 1 or not destination_directive:
        return False
    placement_grounded = _ask_action_grounded_names(
        assistant_instruction, authored_aliases, "PLACE"
    )
    if not placement_grounded:
        placement_grounded = _ask_action_grounded_names(
            assistant_instruction, aliases, "PLACE"
        )
    placement_grounded.update(
        _ask_contextual_placement_destinations(
            assistant_instruction,
            held,
            authored_aliases,
            aliases,
            current_locations,
        )
    )
    destinations = placement_grounded - held
    grounded_walk = bool(
        len(keyed) == 1
        and keyed[0] in destinations
        and (current_at is None or keyed[0] not in current_at)
    )
    return grounded_walk


def _ask_has_specific_destination_directive(instruction: str) -> bool:
    return bool(
        _ask_action_is_requested(instruction, "PLACE")
        and _ASK_SPECIFIC_DESTINATION_RE.search(instruction)
    )


def _ask_walk_prepares_requested_grab(
    verb: str,
    keyed: Sequence[str],
    assistant_instruction: str,
    held: set[str],
    authored_aliases: Mapping[str, object],
    aliases: Mapping[str, object],
    movable: set[str],
    current_at: set[str] | None,
) -> bool:
    if (
        verb != "WALK"
        or len(keyed) != 1
        or held
        or not _ask_action_is_requested(assistant_instruction, "GRAB")
    ):
        return False
    destination = keyed[0]
    if current_at is not None and destination in current_at:
        return False
    movable_aliases = {
        name: value for name, value in aliases.items() if name in movable
    }
    authored_movable_aliases = {
        name: value for name, value in authored_aliases.items() if name in movable
    }
    destination_aliases = {
        name: value for name, value in aliases.items() if name not in movable
    }
    authored_destination_aliases = {
        name: value for name, value in authored_aliases.items() if name not in movable
    }
    grab_targets = _ask_action_grounded_names(
        assistant_instruction, authored_movable_aliases, "GRAB"
    )
    if not grab_targets:
        grab_targets = _ask_action_grounded_names(
            assistant_instruction, movable_aliases, "GRAB"
        )
    pronoun_grab = any(
        _ask_action_is_requested(clause, "GRAB") and _continues_find_referent(clause)
        for clause in _find_action_clauses(assistant_instruction)
    )
    for clause in _find_instruction_clauses(assistant_instruction):
        mentioned_targets = _positive_alias_matches(clause, authored_movable_aliases)
        if not mentioned_targets:
            mentioned_targets = _positive_alias_matches(clause, movable_aliases)
        if len(mentioned_targets) != 1 or (
            grab_targets and mentioned_targets.isdisjoint(grab_targets)
        ):
            continue
        if not grab_targets and not pronoun_grab:
            continue
        positive_destinations = _positive_alias_matches(
            clause, authored_destination_aliases
        )
        if not positive_destinations:
            positive_destinations = _positive_alias_matches(clause, destination_aliases)
        negated_destinations = _ask_negated_alias_names(
            clause, authored_destination_aliases
        ) | _ask_negated_alias_names(clause, destination_aliases)
        positive_destinations -= negated_destinations
        for match in re.finditer(
            r"\b(?:at|beside|by|from|in|inside|near|on)\b", clause, re.IGNORECASE
        ):
            local_subject = re.split(
                r"[,;]|\b(?:and|then)\b",
                clause[: match.start()],
                flags=re.IGNORECASE,
            )[-1]
            local_targets = _positive_alias_matches(
                local_subject, authored_movable_aliases
            )
            if not local_targets:
                local_targets = _positive_alias_matches(local_subject, movable_aliases)
            if len(local_targets) != 1 or (
                grab_targets and local_targets.isdisjoint(grab_targets)
            ):
                continue
            location_phrase = re.split(
                r"[,;]|\b(?:and|but|then|to|toward|towards)\b",
                clause[match.end() :],
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            grounded = _positive_alias_matches(
                location_phrase, authored_destination_aliases
            )
            if not grounded:
                grounded = _positive_alias_matches(location_phrase, destination_aliases)
            grounded -= negated_destinations
            if destination in grounded & positive_destinations:
                return True
    return False


def _ask_visually_grounded_placement_is_permitted(
    verb: str,
    keyed: Sequence[str],
    assistant_instruction: str,
    held: set[str],
    grounded: set[str],
    current_at: set[str] | None,
) -> bool:
    return bool(
        verb in _ASK_PLACEMENT_VERBS
        and len(keyed) == 2
        and len(held) == 1
        and keyed[0] in held
        and grounded - held == {keyed[1]}
        and current_at is not None
        and keyed[1] in current_at
        and _ask_has_specific_destination_directive(assistant_instruction)
        and _ask_placement_relation_matches(verb, assistant_instruction)
    )


def _ask_preparatory_orientation_matches(verb: str, instruction: str) -> bool:
    direction_patterns = {
        "TURNAROUND": r"\b(?:behind\s+you|turn\s+around)\b",
        "TURN_AROUND": r"\b(?:behind\s+you|turn\s+around)\b",
        "TURNLEFT": (
            r"\b(?:(?:on|to)\s+(?:the\s+|your\s+)?left|"
            r"(?:diagonally\s+)?forward[-\s]left|leftward)\b"
        ),
        "TURN_LEFT": (
            r"\b(?:(?:on|to)\s+(?:the\s+|your\s+)?left|"
            r"(?:diagonally\s+)?forward[-\s]left|leftward)\b"
        ),
        "TURNRIGHT": (
            r"\b(?:(?:on|to)\s+(?:the\s+|your\s+)?right|"
            r"(?:diagonally\s+)?forward[-\s]right|rightward)\b"
        ),
        "TURN_RIGHT": (
            r"\b(?:(?:on|to)\s+(?:the\s+|your\s+)?right|"
            r"(?:diagonally\s+)?forward[-\s]right|rightward)\b"
        ),
    }
    compound_direction_patterns = {
        "TURNLEFT": r"\b(?:(?:diagonally\s+)?forward[-\s]left|leftward)\b",
        "TURN_LEFT": r"\b(?:(?:diagonally\s+)?forward[-\s]left|leftward)\b",
        "TURNRIGHT": r"\b(?:(?:diagonally\s+)?forward[-\s]right|rightward)\b",
        "TURN_RIGHT": r"\b(?:(?:diagonally\s+)?forward[-\s]right|rightward)\b",
    }
    compound_motion_patterns = {
        "TURNLEFT": (
            r"\b(?:(?:go|head|move|navigate|proceed|step|walk)\s+|"
            r"(?:bring|carry|take)\s+(?:(?:a|an|my|that|the|this|your)\s+)?"
            r"(?:[a-z0-9-]+\s+){0,4}?)(?:diagonally\s+)?"
            r"(?:forward[-\s]left|leftward)\b"
        ),
        "TURN_LEFT": (
            r"\b(?:(?:go|head|move|navigate|proceed|step|walk)\s+|"
            r"(?:bring|carry|take)\s+(?:(?:a|an|my|that|the|this|your)\s+)?"
            r"(?:[a-z0-9-]+\s+){0,4}?)(?:diagonally\s+)?"
            r"(?:forward[-\s]left|leftward)\b"
        ),
        "TURNRIGHT": (
            r"\b(?:(?:go|head|move|navigate|proceed|step|walk)\s+|"
            r"(?:bring|carry|take)\s+(?:(?:a|an|my|that|the|this|your)\s+)?"
            r"(?:[a-z0-9-]+\s+){0,4}?)(?:diagonally\s+)?"
            r"(?:forward[-\s]right|rightward)\b"
        ),
        "TURN_RIGHT": (
            r"\b(?:(?:go|head|move|navigate|proceed|step|walk)\s+|"
            r"(?:bring|carry|take)\s+(?:(?:a|an|my|that|the|this|your)\s+)?"
            r"(?:[a-z0-9-]+\s+){0,4}?)(?:diagonally\s+)?"
            r"(?:forward[-\s]right|rightward)\b"
        ),
    }
    direction = direction_patterns.get(verb)
    if direction is None:
        return False
    compound_direction = compound_direction_patterns.get(verb)
    compound_motion = compound_motion_patterns.get(verb)
    for clause in _find_action_clauses(instruction):
        if not re.search(direction, clause, re.IGNORECASE):
            continue
        if re.search(r"\b(?:avoid|ignore|never|not|skip|without)\b", clause):
            continue
        has_compound_direction = bool(
            compound_direction and re.search(compound_direction, clause, re.IGNORECASE)
        )
        if has_compound_direction and not (
            compound_motion and re.search(compound_motion, clause, re.IGNORECASE)
        ):
            continue
        if (
            re.search(r"\b(?:ahead|forward|straight)\b", clause)
            and not has_compound_direction
        ):
            continue
        if _ask_action_is_requested(clause, "WALK"):
            return True
    return False


def _ask_action_state_guard(
    verb: str,
    command: str,
    assistant_instruction: str,
    last_successful_human_command: str,
    held: set[str],
    current_at: set[str] | None,
) -> str:
    if verb == "GRAB" and held:
        return "ask_hands_full_grab_blocked"
    if verb == "DROP" and not held:
        return "ask_unrequested_action_blocked"
    if verb == "FLIP":
        _, arguments, _ = _command_key(command)
        targets = {name for name, object_id in arguments if object_id != 0}
        if len(held) != 1 or targets != held:
            return "ask_flip_requires_held"
    if verb == "WALK" and current_at is not None:
        _, arguments, _ = _command_key(command)
        targets = [name for name, object_id in arguments if object_id != 0]
        if len(targets) == 1 and targets[0] in current_at:
            return "ask_redundant_walk_blocked"
    if verb in _ASK_PLACEMENT_VERBS and current_at is not None:
        _, arguments, _ = _command_key(command)
        destinations = [name for name, object_id in arguments[1:] if object_id != 0]
        if len(destinations) != 1 or destinations[0] not in current_at:
            return "ask_unrequested_action_blocked"
    repeated_walk = bool(
        verb == "WALK"
        and last_successful_human_command
        and command == last_successful_human_command
        and _ask_action_is_requested(assistant_instruction, "PLACE")
    )
    return "ask_redundant_walk_blocked" if repeated_walk else ""


def _ask_action_grounded_names(
    instruction: str,
    aliases: Mapping[str, object],
    action_kind: str,
) -> set[str]:
    clauses = _find_action_clauses(instruction)
    grounded: set[str] = set()
    for clause in clauses:
        if _ask_action_is_requested(clause, action_kind):
            grounded.update(
                _ask_placement_destination_matches(clause, aliases)
                if action_kind == "PLACE"
                else _positive_alias_matches(clause, aliases)
            )
    explicitly_negated = (
        set().union(*(_ask_negated_alias_names(clause, aliases) for clause in clauses))
        if action_kind == "PLACE"
        else set()
    )
    return grounded - explicitly_negated


def _ask_placement_destination_matches(
    clause: str, aliases: Mapping[str, object]
) -> set[str]:
    grounded: set[str] = set()
    for match in re.finditer(
        r"\b(?:in|inside|into|on|onto|to|toward|towards)\b",
        clause,
        re.IGNORECASE,
    ):
        destination_phrase = re.split(
            r"\b(?:and\s+then|then)\b",
            clause[match.end() :],
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        grounded.update(_positive_alias_matches(destination_phrase, aliases))
    return grounded


def _ask_negated_alias_names(clause: str, aliases: Mapping[str, object]) -> set[str]:
    spoken_words = _grounding_alias_words(clause)
    return {
        str(name)
        for name, description_aliases in aliases.items()
        for alias_words in _grounding_alias_word_sets(description_aliases)
        if alias_words
        and alias_words <= spoken_words
        and _ask_alias_is_negated(clause, alias_words)
    }


def _ask_action_is_requested(text: str, action_kind: str) -> bool:
    if action_kind == "PLACE" and _ask_targeted_drop_requests_placement(text):
        return True
    if action_kind == "PLACE" and _ask_transport_requests_placement(text):
        return True
    if action_kind == "GRAB" and _ask_hold_requests_grab(text):
        return True
    if action_kind == "DROP" and re.search(
        r"\b(?:put|set)\b(?:\s+\w+){0,5}\s+down\b",
        " ".join(_find_action_words(_normalize_find_instruction(text))),
    ):
        return True
    cues = _ASK_ACTION_CUES.get(action_kind, frozenset())
    if not cues:
        return False
    words = _find_action_words(_normalize_find_instruction(text))
    for index, word in enumerate(words):
        if word not in cues:
            continue
        context = words[max(0, index - 4) : min(len(words), index + 4)]
        if _ask_action_cue_is_negated(words, index):
            continue
        if (
            action_kind == "GRAB"
            and word == "take"
            and _ask_take_is_look_idiom(words, index)
        ):
            continue
        if (
            action_kind in {"TURNON", "TURNOFF"}
            and word in {"switch", "turn"}
            and action_kind.removeprefix("TURN").lower() not in context
        ):
            continue
        return True
    return False


def _ask_hold_requests_grab(instruction: str) -> bool:
    for clause in _find_action_clauses(instruction):
        words = _find_action_words(clause)
        for index, word in enumerate(words):
            if word != "hold":
                continue
            if _ask_action_cue_is_negated(words, index):
                continue
            following = words[index + 1 : index + 8]
            if not following or following[0] in {"on", "still"}:
                continue
            if tuple(following[:2]) in {("your", "ground"), ("your", "position")}:
                continue
            if {"closer", "nearer"}.intersection(following) or following[0] in {
                "a",
                "an",
                "it",
                "my",
                "that",
                "the",
                "this",
                "your",
            }:
                return True
    return False


def _ask_take_is_look_idiom(words: Sequence[str], index: int) -> bool:
    modifiers = {
        "a",
        "an",
        "another",
        "better",
        "careful",
        "close",
        "closer",
        "good",
        "one",
        "quick",
        "second",
    }
    following = words[index + 1 : index + 6]
    if "look" not in following:
        return False
    look_index = following.index("look")
    return all(word in modifiers for word in following[:look_index])


def _ask_transport_requests_placement(instruction: str) -> bool:
    for clause in _find_action_clauses(instruction):
        words = _find_action_words(clause)
        for index, word in enumerate(words):
            if word not in {"bring", "carry", "take"}:
                continue
            if _ask_action_cue_is_negated(words, index) or (
                word == "take" and _ask_take_is_look_idiom(words, index)
            ):
                continue
            if {"to", "toward", "towards"}.intersection(words[index + 1 :]):
                return True
    return _ask_transport_continues_named_destination(instruction)


def _ask_action_cue_is_negated(words: Sequence[str], index: int) -> bool:
    context = set(words[max(0, index - 4) : min(len(words), index + 4)])
    return bool(
        context
        & {
            "avoid",
            "except",
            "never",
            "no",
            "not",
            "nowhere",
            "rather",
            "skip",
            "without",
        }
    )


def _ask_alias_is_negated(clause: str, alias_words: set[str]) -> bool:
    words = [
        _GROUNDING_WORD_ALIASES.get(word, word)
        for word in _WORD_RE.findall(clause.lower().replace("_", " "))
    ]
    positions = [index for index, word in enumerate(words) if word in alias_words]
    if not positions:
        return False
    start = max(0, min(positions) - 5)
    end = min(len(words), max(positions) + 5)
    return bool(
        set(words[start:end])
        & {
            "avoid",
            "except",
            "never",
            "no",
            "not",
            "nowhere",
            "rather",
            "skip",
            "without",
        }
    )


def _ask_targeted_drop_requests_placement(instruction: str) -> bool:
    destination = re.compile(
        r"\b(?:in|inside|into|on|onto|to)\s+"
        r"(?:(?:a|an|that|the|this|your)\s+)?([a-z][a-z0-9-]*)",
        re.IGNORECASE,
    )
    for clause in _find_action_clauses(instruction):
        if not _ask_action_is_requested(clause, "DROP"):
            continue
        match = destination.search(clause)
        if match and match.group(1).lower() not in {"floor", "ground"}:
            return True
    return False


def _ask_placement_relation_matches(verb: str, instruction: str) -> bool:
    relation_words: set[str] = set()
    for clause in _find_action_clauses(instruction):
        if _ask_action_is_requested(clause, "PLACE"):
            relation_words.update(_WORD_RE.findall(clause.lower()))
    if _ask_transport_continues_named_destination(instruction):
        relation_words.update(_WORD_RE.findall(instruction.lower()))
    if verb in {"PUTIN", "PUTINSIDE"}:
        return bool(relation_words & {"in", "inside", "into"})
    return bool(
        relation_words & {"atop", "on", "onto"}
        or re.search(
            r"\bin\s+(?:the\s+)?(?:center|middle)\s+of(?:\s+\w+){0,6}\s+"
            r"(?:surface|top)\b",
            instruction,
            re.IGNORECASE,
        )
    )


def _without_unseen_references(
    commands: Sequence[str],
    gated_verbs: frozenset[str],
    target_keys: Sequence[str] | None,
    visible_object_keys: Sequence[str],
) -> tuple[list[str], bool]:
    targets = None if target_keys is None else {str(key) for key in target_keys}
    visible = {str(key) for key in visible_object_keys}
    if targets == set():
        return list(commands), False
    for index, command in enumerate(commands):
        verb, arguments, _ = _command_key(command)
        referenced = {name for name, _ in arguments}
        guarded_references = referenced if targets is None else referenced & targets
        if verb in gated_verbs and guarded_references - visible:
            return list(commands[:index]), True
    return list(commands), False


def _without_private_outcome_state(
    observation: Mapping[str, object],
) -> dict[str, object]:
    return {
        str(key): value
        for key, value in observation.items()
        if str(key) not in _PRIVATE_OUTCOME_KEYS
    }


def _watch_interrupted_destination_guard(
    commands: Sequence[str],
    interrupted_plan: str,
    assistant_instruction: str,
    descriptions: Mapping[str, object],
) -> list[str] | None:
    if not commands or not interrupted_plan:
        return None
    expected_verb, expected_args, _ = _command_key(interrupted_plan)
    if expected_verb not in _WATCH_UNIT_END or len(expected_args) < 2:
        return None
    terminal = next(
        (
            _command_key(command)
            for command in reversed(commands)
            if _command_key(command)[0] in _WATCH_UNIT_END
        ),
        None,
    )
    if terminal is None or len(terminal[1]) < 2:
        return None
    expected_subject, expected_target = expected_args[0][0], expected_args[-1][0]
    proposed_subject, proposed_target = terminal[1][0][0], terminal[1][-1][0]
    if proposed_subject != expected_subject or proposed_target == expected_target:
        return None
    grounded = _uniquely_grounded_names(assistant_instruction, descriptions)
    if proposed_target in grounded:
        return None
    return [interrupted_plan]


def _uniquely_grounded_names(text: str, descriptions: Mapping[str, object]) -> set[str]:
    spoken_words = _grounding_words(text)
    matches = {
        str(name): spoken_words & _description_grounding_words(description)
        for name, description in descriptions.items()
    }
    return {
        name
        for name, words in matches.items()
        if words
        and not any(
            words <= other_words
            for other_name, other_words in matches.items()
            if other_name != name and other_words
        )
    }


def _instruction_grounded_names(
    text: str, descriptions: Mapping[str, object]
) -> set[str]:
    grounded: set[str] = set()
    for clause in _find_instruction_clauses(text):
        grounded.update(_positive_alias_matches(clause, descriptions))
    return grounded


def _instruction_alias_candidates(
    text: str, descriptions: Mapping[str, object]
) -> set[str]:
    candidates: set[str] = set()
    for clause in _find_instruction_clauses(text):
        spoken_words = _grounding_alias_words(clause)
        candidates.update(
            str(name)
            for name, aliases in descriptions.items()
            if any(
                alias_words
                and alias_words <= spoken_words
                and not _alias_is_negated(clause, alias_words)
                for alias_words in _grounding_alias_word_sets(aliases)
            )
        )
    return candidates


def _action_grounded_names(
    text: str,
    descriptions: Mapping[str, object],
    action_kind: str,
) -> set[str]:
    clauses = _find_action_clauses(text)
    grounded: set[str] = set()
    prior_matches: set[str] = set()
    for clause in clauses:
        matches = _positive_alias_matches(clause, descriptions)
        if _find_action_is_requested(clause, action_kind):
            if matches:
                grounded.update(matches)
            elif _continues_find_referent(clause) and len(prior_matches) == 1:
                grounded.update(prior_matches)
        if matches:
            prior_matches.update(matches)
    return grounded


def _positive_alias_matches(
    clause: str, descriptions: Mapping[str, object]
) -> set[str]:
    spoken_words = _grounding_alias_words(clause)
    matches = {
        str(name): tuple(
            alias_words
            for alias_words in _grounding_alias_word_sets(aliases)
            if alias_words
            and alias_words <= spoken_words
            and not _alias_is_negated(clause, alias_words)
        )
        for name, aliases in descriptions.items()
    }
    grounded = {
        name
        for name, alias_matches in matches.items()
        if alias_matches
        and any(
            not any(
                alias_words <= other_alias_words
                for other_name, other_matches in matches.items()
                if other_name != name
                for other_alias_words in other_matches
            )
            for alias_words in alias_matches
        )
    }
    if len(grounded) > 1 and "or" in _WORD_RE.findall(clause.lower()):
        return set()
    return grounded


def _best_positive_alias_match(
    clause: str,
    descriptions: Mapping[str, object],
    excluded_descriptions: Mapping[str, object],
) -> set[str]:
    spoken_words = _grounding_alias_words(clause)
    excluded_words = {
        word
        for aliases in excluded_descriptions.values()
        for alias_words in _grounding_alias_word_sets(aliases)
        for word in alias_words
    }

    def score_alias(alias_words: set[str]) -> int:
        score = len(spoken_words & alias_words)
        if score == 1 and not (alias_words & spoken_words - excluded_words):
            return 0
        return score

    scores = {
        str(name): max(
            (
                score_alias(alias_words)
                for alias_words in _grounding_alias_word_sets(aliases)
                if alias_words and not _alias_is_negated(clause, alias_words)
            ),
            default=0,
        )
        for name, aliases in descriptions.items()
    }
    best = max(scores.values(), default=0)
    winners = {name for name, score in scores.items() if score == best and score > 0}
    return winners if len(winners) == 1 else set()


def _find_instruction_clauses(text: str) -> tuple[str, ...]:
    normalized = _normalize_find_instruction(text)
    normalized = re.sub(
        r",\s*(?=(?:(?:then|and)\s+)?(?:do not|never|avoid|skip|open|close|"
        r"shut|walk|go|head|move|grab|pick|take|get|retrieve|put|place|bring|"
        r"look|inspect|check|search|return)\b)",
        ";",
        normalized,
    )
    return tuple(
        clause.strip()
        for clause in re.split(
            r"[.!?;\n\u2013\u2014]+|\b(?:but|however|instead)\b", normalized
        )
        if clause.strip()
    )


def _find_action_clauses(text: str) -> tuple[str, ...]:
    normalized = _normalize_find_instruction(text)
    action_words = {
        word
        for cues in (*_FIND_ACTION_CUES.values(), *_ASK_ACTION_CUES.values())
        for word in cues
    }
    action_start = (
        "(?:"
        + "|".join(
            re.escape(word)
            for word in sorted(
                action_words
                | {"avoid", "closing", "do not", "never", "opening", "skip"},
                key=lambda value: (-len(value), value),
            )
        )
        + ")"
    )
    normalized = re.sub(rf"\band\s+(?={action_start}\b)", ";", normalized)
    normalized = re.sub(rf"\b(?:and\s+)?then\s+(?={action_start}\b)", ";", normalized)
    normalized = re.sub(rf"\b(?:after|before)\s+(?={action_start}\b)", ";", normalized)
    return tuple(
        clause.strip()
        for clause in re.split(
            r"[,.!?;\n\u2013\u2014]+|\b(?:but|however|instead)\b",
            normalized,
        )
        if clause.strip()
    )


def _normalize_find_instruction(text: str) -> str:
    normalized = text.lower().replace("’", "'")
    normalized = re.sub(r"\b(?:don't|do not)\b", "do not", normalized)
    normalized = re.sub(r"\b(should|could|would|must|can)n't\b", r"\1 not", normalized)
    normalized = re.sub(r"\b(?:isn't|is not)\b", "is not", normalized)
    normalized = re.sub(r"\b(?:wasn't|was not)\b", "was not", normalized)
    return re.sub(r"\b(?:there is )?no need to\b", "do not", normalized)


def _alias_is_negated(clause: str, alias_words: set[str]) -> bool:
    words = [
        _GROUNDING_WORD_ALIASES.get(word, word)
        for word in _WORD_RE.findall(clause.lower().replace("_", " "))
    ]
    positions = [index for index, word in enumerate(words) if word in alias_words]
    if not positions:
        return False
    start = max(0, min(positions) - 5)
    end = min(len(words), max(positions) + 5)
    context = set(words[start:end])
    negators = {
        "avoid",
        "except",
        "never",
        "no",
        "not",
        "nowhere",
        "rather",
        "skip",
        "without",
    }
    # A label such as "NO GLASS" can be part of the object's identifying alias.
    return bool((context & negators) - alias_words)


def _find_action_kind(verb: str) -> str:
    if verb in _ASK_PLACEMENT_VERBS:
        return "PLACE"
    aliases = {
        "ANSWER_PHONE": "ANSWERPHONE",
        "ASSEMBLE": "ATTACH",
        "CLOSE_DOOR": "CLOSE",
        "CLOSEDOOR": "CLOSE",
        "DISASSEMBLE": "DETACH",
        "ERECT": "RAISE",
        "FALL_TO_FLOOR": "FALLTOFLOOR",
        "INSTALL": "ATTACH",
        "LOOK_AROUND": "LOOKAT",
        "LOOK_AT": "LOOKAT",
        "LOOK_SCAN": "LOOKAT",
        "OPEN_DOOR": "OPEN",
        "OPENDOOR": "OPEN",
        "PULL_RACK": "PULL",
        "PULLRACK": "PULL",
        "PUSH_RACK": "PUSH",
        "PUSHRACK": "PUSH",
        "RELEASE": "DROP",
        "REMOVE": "DETACH",
        "RUN": "WALK",
        "SQUATDOWN": "SQUAT",
        "STANDUPRIGHT": "RAISE",
        "SWITCH_OFF": "TURNOFF",
        "SWITCH_ON": "TURNON",
        "SWITCHOFF": "TURNOFF",
        "SWITCHON": "TURNON",
        "TURN_OFF": "TURNOFF",
        "TURN_ON": "TURNON",
        "TURN_OVER": "FLIP",
        "TURN_AROUND": "TURN",
        "TURN_LEFT": "TURN",
        "TURN_RIGHT": "TURN",
        "TURNOVER": "FLIP",
        "TYPE": "USECOMPUTER",
        "USE_COMPUTER": "USECOMPUTER",
        "WALKTOWARDS": "WALK",
    }
    if verb in aliases:
        return aliases[verb]
    if verb in {"LOOKAROUND", "LOOKSCAN"}:
        return "LOOKAT"
    if verb in {"TURN", "TURNAROUND", "TURNLEFT", "TURNRIGHT"}:
        return "TURN"
    if verb in {"FORWARD", "WALKFORWARD", "WALK_FORWARD"}:
        return "WALK"
    if verb in {"CLOSE", "CLOSEDOOR"}:
        return "CLOSE"
    if verb in {"OPEN", "OPENDOOR"}:
        return "OPEN"
    return verb


def _find_action_is_requested(text: str, action_kind: str) -> bool:
    cues = _FIND_ACTION_CUES.get(action_kind, frozenset())
    if not cues:
        return False
    words = _find_action_words(text)
    negative = {"avoid", "except", "never", "not", "rather", "skip", "without"}
    for index, word in enumerate(words):
        if word not in cues:
            continue
        context = words[max(0, index - 4) : min(len(words), index + 4)]
        if not negative.intersection(context):
            return True
    return False


def _narrate_action_is_explicitly_requested(text: str, action_kind: str) -> bool:
    if action_kind == "WALK" and re.search(r"\breturn\s+to\b", text, re.I):
        return True
    if action_kind == "GRAB" and re.search(r"\bstart\s+with\b", text, re.I):
        return True
    if action_kind == "WAIT" and re.search(
        r"\b(?:do not touch|keep holding|leave\b[^.!?;]{0,80}\b(?:alone|in place|where)|"
        r"stay where)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if action_kind == "LOOKAT" and re.search(
        r"\b(?:confirm|check|verify|visible|remain|stayed|still\s+be)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if action_kind == "PLACE" and _ask_transport_requests_placement(text):
        return True
    cues = _NARRATE_EXPLICIT_ACTION_CUES.get(
        action_kind, _ASK_ACTION_CUES.get(action_kind, frozenset())
    )
    negative = {"avoid", "except", "never", "not", "rather", "skip", "without"}
    words = _find_action_words(text)
    return any(
        word in cues
        and not negative.intersection(
            words[max(0, index - 4) : min(len(words), index + 4)]
        )
        for index, word in enumerate(words)
    )


def _action_continues_find_referent(text: str, action_kind: str) -> bool:
    words = set(_WORD_RE.findall(text.lower().replace("_", " ")))
    if (
        "there" in words
        and words & _FIND_ROOM_WORDS
        and not words & {"it", "same", "that"}
    ):
        return False
    return any(
        _find_action_is_requested(clause, action_kind)
        and _continues_find_referent(clause)
        for clause in _find_action_clauses(text)
    )


def _find_action_is_explicitly_forbidden(text: str, action_kind: str) -> bool:
    cues = _FIND_ACTION_CUES.get(action_kind, frozenset())
    negative = {"avoid", "except", "never", "not", "rather", "skip", "without"}
    saw_forbidden_cue = False
    for clause in _find_action_clauses(text):
        words = _find_action_words(clause)
        for index, word in enumerate(words):
            if word not in cues:
                continue
            context = words[max(0, index - 4) : min(len(words), index + 4)]
            if not negative.intersection(context):
                return False
            saw_forbidden_cue = True
    return saw_forbidden_cue


def _find_action_words(text: str) -> list[str]:
    stems = {
        "approaching": "approach",
        "checking": "check",
        "closing": "close",
        "collecting": "collect",
        "examining": "examine",
        "fetching": "fetch",
        "grabbing": "grab",
        "heading": "head",
        "inspecting": "inspect",
        "lifting": "lift",
        "looking": "look",
        "moving": "move",
        "opening": "open",
        "picking": "pick",
        "placing": "place",
        "retrieving": "retrieve",
        "reopen": "open",
        "reopening": "open",
        "returning": "return",
        "searching": "search",
        "set": "set",
        "shut": "shut",
        "storing": "store",
        "taking": "take",
        "walking": "walk",
        "wiping": "wipe",
    }
    return [
        stems.get(word, word)
        for word in _WORD_RE.findall(text.lower().replace("_", " "))
    ]


def _description_grounding_words(description: object) -> set[str]:
    if isinstance(description, Sequence) and not isinstance(description, (str, bytes)):
        return {word for alias in description for word in _grounding_words(str(alias))}
    return _grounding_words(str(description))


def _guide_entity_phrases(key: str, description: str) -> tuple[str, ...]:
    description = description.split("; public motor references:", 1)[0]
    normalized_key = " ".join(_WORD_RE.findall(key.lower().replace("_", " ")))
    rendered = description.split("; visually rendered as ", 1)
    phrases = [normalized_key]
    if len(rendered) == 2:
        phrases.append(rendered[0].strip())
        phrases.append(rendered[1].split(", in ", 1)[0].strip())
    else:
        phrases.append(description.split(", in ", 1)[0].strip())
    return tuple(
        sorted(
            {phrase for phrase in phrases if phrase},
            key=lambda alias: (len(alias.split()), alias),
        )
    )


def _guide_entity_aliases(key: str, description: str) -> tuple[str, ...]:
    phrases = _guide_entity_phrases(key, description)
    aliases = set(phrases)
    generic = {
        "area",
        "item",
        "object",
        "part",
        "place",
        "room",
        "surf",
        "surface",
        "unit",
    }
    aliases.update(
        word
        for phrase in phrases
        for word in _WORD_RE.findall(phrase.lower())
        if len(word) > 2 and not word.isdigit() and word not in generic
    )
    return tuple(sorted(aliases, key=lambda alias: (len(alias.split()), alias)))


def _narrate_entity_aliases(description: str) -> tuple[str, ...]:
    return _guide_entity_phrases("", description)


def _motor_alias_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(alias) for alias in value)
    return (str(value),)


def _guide_landmark_aliases(key: str, description: str) -> tuple[str, ...]:
    return _guide_specific_source_aliases(key, _guide_entity_aliases(key, description))


def _orchestrate_landmark_aliases(
    key: str, description: str, authored_aliases: Sequence[str]
) -> tuple[str, ...]:
    stable_description = description.split("; public motor references:", 1)[0].strip()
    without_room_index = re.sub(r"(?<=\D)\s+\d+$", "", stable_description)
    aliases = tuple(
        dict.fromkeys(
            (
                *_guide_entity_aliases(key, description),
                stable_description,
                without_room_index,
                *authored_aliases,
            )
        )
    )
    return _guide_specific_source_aliases(key, aliases)


def _guide_specific_source_aliases(key: str, aliases: object) -> tuple[str, ...]:
    key_phrase = " ".join(_WORD_RE.findall(key.lower().replace("_", " ")))
    values = (
        aliases
        if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes))
        else (aliases,)
    )
    return tuple(
        str(alias)
        for alias in values
        if " ".join(_WORD_RE.findall(str(alias).lower().replace("_", " ")))
        == key_phrase
        or len(_grounding_alias_words(str(alias))) >= 2
    )


def _guide_description_without_location(description: str) -> str:
    return description.rsplit(", in ", 1)[0]


def _guide_description_without_uninspected_label(description: str) -> str:
    return re.sub(
        r",?\s*label(?:l)?ed\s+(['\"]).*?\1",
        "",
        description,
        flags=re.IGNORECASE,
    ).strip(" ,")


def _grounding_alias_word_sets(aliases: object) -> tuple[set[str], ...]:
    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
        return tuple(_grounding_alias_words(str(alias)) for alias in aliases)
    return (_grounding_alias_words(str(aliases)),)


def _grounding_alias_words(value: str) -> set[str]:
    relational_words = {"in", "inside", "into", "on"}
    raw_words = _WORD_RE.findall(value.lower().replace("_", " "))
    words = {
        word
        for word in raw_words
        if (len(word) > 1 or word.isdigit())
        and (word not in _NARRATE_STOPWORDS or word in relational_words)
    }
    words.update(
        f"{raw_words[index - 1]}_{word}"
        for index, word in enumerate(raw_words)
        if index > 0
        and len(word) == 1
        and word.isalpha()
        and raw_words[index - 1] in _GROUNDING_LABEL_PREFIXES
    )
    return {_GROUNDING_WORD_ALIASES.get(word, word) for word in words}


def _continues_find_referent(text: str) -> bool:
    words = set(_WORD_RE.findall(text.lower().replace("_", " ")))
    continuations = words & _FIND_REFERENT_CONTINUATIONS
    if continuations == {"there"} and words & _FIND_ROOM_WORDS:
        return False
    return bool(continuations) and not bool(words & _FIND_REFERENT_CHANGES)


def _grounding_words(value: str) -> set[str]:
    words = {
        word
        for word in _WORD_RE.findall(value.lower().replace("_", " "))
        if (len(word) > 1 or word.isdigit()) and word not in _NARRATE_STOPWORDS
    }
    return {_GROUNDING_WORD_ALIASES.get(word, word) for word in words}


def load_opening_catalog(path: str) -> dict[str, Mapping[str, object]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("tasks") if isinstance(payload, Mapping) else None
    if not isinstance(rows, Mapping):
        raise InteractionSpecError(f"{path}: expected a tasks object")
    return {
        str(task_id): copy.deepcopy(dict(value))
        for task_id, value in rows.items()
        if isinstance(value, Mapping)
    }


def _human_task_context(
    family: str, authored: Mapping[str, object], public_context: str
) -> str:
    if family == "WATCH":
        return public_context
    if family == "ASK":
        return _ASK_HUMAN_TASK_CONTEXT
    return str(authored.get("human_task_context") or public_context).strip()


def _task_scoped_motor_aliases(
    authored: Mapping[str, object],
    family: str,
    descriptions: Mapping[str, str],
    task_id: str,
) -> dict[str, tuple[str, ...]]:
    raw = authored.get("motor_aliases")
    if raw is None:
        return {}
    if family not in {"ASK", "GUIDE", "NARRATE", "ORCHESTRATE"}:
        raise InteractionSpecError(
            f"{task_id}.motor_aliases is supported only for ASK, GUIDE, NARRATE, "
            "and ORCHESTRATE"
        )
    mapping = _required_mapping(raw, f"{task_id}.motor_aliases")
    aliases = _parse_task_scoped_motor_aliases(mapping, descriptions, task_id)
    _validate_motor_alias_description_coverage(aliases, descriptions, task_id)
    _validate_unique_motor_aliases(aliases, task_id)
    return aliases


def _parse_task_scoped_motor_aliases(
    mapping: Mapping[str, object],
    descriptions: Mapping[str, str],
    task_id: str,
) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_values in mapping.items():
        key = _required_text(raw_key, f"{task_id}.motor_aliases key")
        if key not in descriptions:
            raise InteractionSpecError(
                f"{task_id}.motor_aliases references unknown object {key!r}"
            )
        values = _string_sequence(raw_values, f"{task_id}.motor_aliases.{key}")
        if not values:
            raise InteractionSpecError(
                f"{task_id}.motor_aliases.{key} must not be empty"
            )
        if any(";" in value or "\n" in value for value in values):
            raise InteractionSpecError(
                f"{task_id}.motor_aliases.{key} must contain plain phrases"
            )
        aliases[key] = tuple(dict.fromkeys(values))
    return aliases


def _validate_motor_alias_description_coverage(
    aliases: Mapping[str, tuple[str, ...]],
    descriptions: Mapping[str, str],
    task_id: str,
) -> None:
    description_groups: dict[str, set[str]] = {}
    for key, description in descriptions.items():
        normalized = " ".join(_WORD_RE.findall(description.lower()))
        description_groups.setdefault(normalized, set()).add(key)
    for candidates in description_groups.values():
        configured = candidates & aliases.keys()
        if configured and configured != candidates:
            raise InteractionSpecError(
                f"{task_id}.motor_aliases must cover every indistinguishable candidate: "
                f"{sorted(candidates)!r}"
            )


def _validate_unique_motor_aliases(
    aliases: Mapping[str, tuple[str, ...]], task_id: str
) -> None:
    for key, values in aliases.items():
        for value in values:
            if _positive_alias_matches(value, aliases) != {key}:
                raise InteractionSpecError(
                    f"{task_id}.motor_aliases.{key} phrase {value!r} is not unique"
                )


def _oracle_script_override(
    authored: Mapping[str, object], family: str, task_id: str
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]] | None:
    raw_override = authored.get("oracle_script_override")
    if raw_override is None:
        return None
    if family != "NARRATE":
        raise InteractionSpecError(
            f"{task_id}.oracle_script_override is supported only for NARRATE"
        )
    path = f"{task_id}.oracle_script_override"
    override = _required_mapping(raw_override, path)
    expected_fields = {"override_id", "commands", "dialogue"}
    if set(override) != expected_fields:
        raise InteractionSpecError(
            f"{path} must contain exactly {sorted(expected_fields)!r}"
        )
    _required_text(override.get("override_id"), f"{path}.override_id")
    commands = tuple(
        _render_command(row, f"{path}.commands[{index}]")
        for index, row in enumerate(
            _mapping_sequence(override.get("commands"), f"{path}.commands")
        )
    )
    dialogue_rows = _mapping_sequence(override.get("dialogue"), f"{path}.dialogue")
    if not commands or not dialogue_rows:
        raise InteractionSpecError(f"{path} commands and dialogue must not be empty")
    dialogue: list[str] = []
    action_indices: list[int] = []
    for index, row in enumerate(dialogue_rows):
        row_path = f"{path}.dialogue[{index}]"
        if set(row) != {"before_action_index", "say"}:
            raise InteractionSpecError(
                f"{row_path} must contain exactly before_action_index and say"
            )
        dialogue.append(_required_text(row.get("say"), f"{row_path}.say"))
        action_indices.append(
            _required_int(
                row.get("before_action_index"),
                f"{row_path}.before_action_index",
            )
        )
    if action_indices[0] != 0 or any(
        left >= right for left, right in zip(action_indices, action_indices[1:])
    ):
        raise InteractionSpecError(
            f"{path}.dialogue indices must start at zero and strictly increase"
        )
    if action_indices[-1] >= len(commands):
        raise InteractionSpecError(
            f"{path}.dialogue index exceeds the command sequence"
        )
    return commands, tuple(dialogue), tuple(action_indices)


def _interaction_oracle_script(
    authored: Mapping[str, object],
    runtime: Mapping[str, object],
    family: str,
    task_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int | None, ...]]:
    override = _oracle_script_override(authored, family, task_id)
    if override is not None:
        tick_budget = _positive_int(
            runtime.get("tick_budget"), f"{task_id}.runtime.tick_budget"
        )
        if len(override[0]) > tick_budget:
            raise InteractionSpecError(
                f"{task_id}.oracle_script_override has {len(override[0])} commands "
                f"but tick_budget is {tick_budget}"
            )
        return override
    commands = tuple(
        _render_command(row, f"$.runtime.oracle_commands[{index}]")
        for index, row in enumerate(
            _mapping_sequence(
                runtime.get("oracle_commands"), "$.runtime.oracle_commands"
            )
        )
    )
    dialogue_rows = [
        row
        for row in _mapping_sequence(runtime.get("dialogue"), "$.runtime.dialogue")
        if str(row.get("say") or "").strip()
    ]
    dialogue = tuple(str(row.get("say") or "").strip() for row in dialogue_rows)
    action_indices = tuple(_dialogue_action_index(row) for row in dialogue_rows)
    return commands, dialogue, action_indices


def adapt_taskpack(
    pack: Mapping[str, object],
    opening_catalog: Mapping[str, Mapping[str, object]],
) -> TaskPackInteractionSpec:
    task_id = _required_text(pack.get("task_id"), "$.task_id")
    story = _required_mapping(pack.get("story"), "$.story")
    runtime = _required_mapping(pack.get("runtime"), "$.runtime")
    source = _required_mapping(story.get("source_contract"), "$.story.source_contract")
    family = _required_text(
        source.get("family"), "$.story.source_contract.family"
    ).upper()
    profile = profile_for(family)

    authored = _interaction_opening(story, opening_catalog.get(task_id), task_id)
    prologue = _required_mapping(story.get("prologue"), "$.story.prologue")
    raw_actions = authored.get("actions", prologue.get("actions"))
    actions = _string_sequence(raw_actions, f"{task_id}.opening.actions")
    prelude_say = str(authored.get("prelude_say") or "").strip()
    continuous_watch = bool(authored.get("continuous_watch", False))
    if continuous_watch and family != "WATCH":
        raise InteractionSpecError(
            f"{task_id}.continuous_watch is supported only for WATCH tasks"
        )
    utterances = _opening_utterances(
        authored.get("utterances"),
        len(actions),
        task_id,
        allow_after_last=continuous_watch,
    )
    handoff_say = str(authored.get("handoff_say") or "").strip()
    if not prelude_say and not utterances and not handoff_say:
        raise InteractionSpecError(
            f"{task_id}.opening needs prelude_say, handoff_say, or at least one "
            "timed utterance"
        )
    frame_policy, recent_intervals = _opening_frame_selection(authored, task_id)
    completion_contract = _completion_contract(authored, task_id)
    human_command_qualifiers = _human_command_qualifiers(authored, task_id)
    public_context = str(
        authored.get("public_task_context") or handoff_say or prelude_say
    ).strip()
    human_context = _human_task_context(family, authored, public_context)
    authored_tick_budget = _positive_int(
        runtime.get("tick_budget"), f"{task_id}.runtime.tick_budget"
    )
    if continuous_watch:
        authored_tick_budget += len(actions) + len(utterances)

    default_commands = tuple(
        _render_command(row, f"$.runtime.default_commands[{index}]")
        for index, row in enumerate(
            _mapping_sequence(
                runtime.get("default_commands"), "$.runtime.default_commands"
            )
        )
    )
    if profile.has_default_plan and not default_commands:
        raise InteractionSpecError(
            f"{task_id}: WATCH requires runtime.default_commands"
        )
    if not profile.has_default_plan:
        default_commands = ()

    oracle_commands, dialogue, dialogue_action_indices = _interaction_oracle_script(
        authored,
        runtime,
        family,
        task_id,
    )
    if profile.has_default_plan and not oracle_commands:
        raise InteractionSpecError(f"{task_id}: WATCH requires runtime.oracle_commands")
    if profile.has_default_plan:
        default_commands, oracle_commands = _drop_shared_redundant_opening_walk(
            actions, default_commands, oracle_commands
        )
    if (
        profile.has_default_plan
        and _first_difference(default_commands, oracle_commands) is None
    ):
        raise InteractionSpecError(
            f"{task_id}: WATCH default plan must diverge from the oracle plan"
        )
    if profile.has_default_plan and authored_tick_budget < len(default_commands):
        raise InteractionSpecError(
            f"{task_id}: WATCH tick_budget {authored_tick_budget} cannot execute "
            f"its {len(default_commands)}-action silent branch"
        )
    watch_interventions = _watch_interventions(
        authored,
        actions,
        default_commands,
        oracle_commands,
        len(dialogue),
        task_id,
    )
    dialogue = _watch_reference_dialogue(
        authored, dialogue, watch_interventions, task_id
    )
    watch_utterances = _opening_utterances(
        authored.get("watch_utterances"),
        len(default_commands),
        f"{task_id}.watch",
        allow_after_last=True,
    )
    authored_tick_budget += int(continuous_watch) * len(watch_utterances)
    spoken_cues = _required_spoken_cues(
        authored,
        (
            prelude_say,
            *(utterance.say for utterance in utterances),
            *(utterance.say for utterance in watch_utterances),
            handoff_say,
        ),
        task_id,
    )
    raw_scripted_replies = authored.get("scripted_human_replies")
    scripted_human_replies = (
        ()
        if raw_scripted_replies is None
        else _string_sequence(raw_scripted_replies, f"{task_id}.scripted_human_replies")
    )
    if scripted_human_replies and len(scripted_human_replies) != len(dialogue):
        raise InteractionSpecError(
            f"{task_id}.scripted_human_replies must align with runtime.dialogue"
        )
    executable = _required_mapping(pack.get("executable_spec"), "$.executable_spec")
    goal = _required_mapping(executable.get("goal"), "$.executable_spec.goal")
    answer_key = tuple(
        copy.deepcopy(dict(row))
        for row in _mapping_sequence(
            goal.get("answer_key"), "$.executable_spec.goal.answer_key"
        )
    )
    if profile.has_default_plan:
        _validate_watch_branches(
            task_id,
            executable,
            actions,
            default_commands,
            oracle_commands,
            answer_key,
        )
    descriptions, locations = _world_legend(
        executable,
        story,
        () if continuous_watch else actions,
        hide_movable_rooms=family == "FIND",
    )
    movable_keys = _world_movable_keys(executable)
    motor_aliases = _task_scoped_motor_aliases(authored, family, descriptions, task_id)
    visual_state_events = tuple(
        copy.deepcopy(dict(row))
        for row in _mapping_sequence(
            runtime.get("visual_state_events") or (),
            f"{task_id}.runtime.visual_state_events",
        )
    )
    evidence_indices, evidence_fact_ids = _opening_evidence(
        story, len(actions), task_id
    )
    return TaskPackInteractionSpec(
        task_id=task_id,
        family=family,
        public_task_context=public_context,
        human_task_context=human_context,
        authored_tick_budget=authored_tick_budget,
        opening=OpeningScript(
            prelude_say=prelude_say,
            actions=actions,
            handoff_say=handoff_say,
            utterances=utterances,
            watch_utterances=watch_utterances,
            continuous_watch=continuous_watch,
            frame_policy=frame_policy,
            recent_intervals=recent_intervals,
        ),
        default_commands=default_commands,
        oracle_commands=oracle_commands,
        reference_dialogue=dialogue,
        reference_dialogue_action_indices=dialogue_action_indices,
        scripted_human_replies=scripted_human_replies,
        answer_key=answer_key,
        object_descriptions=descriptions,
        motor_aliases=motor_aliases,
        movable_object_keys=movable_keys,
        opening_locations=locations,
        opening_evidence_action_indices=evidence_indices,
        opening_evidence_fact_ids=evidence_fact_ids,
        required_spoken_cues=spoken_cues,
        completion_contract=completion_contract,
        watch_interventions=watch_interventions,
        human_command_qualifiers=human_command_qualifiers,
        visual_state_events=visual_state_events,
    )


class TaskPackInteractionSession:
    """Mutable per-episode state layered over an immutable interaction spec."""

    def __init__(
        self,
        spec: TaskPackInteractionSpec,
        *,
        human_object_interface: str = "full_ids",
    ) -> None:
        if human_object_interface not in HUMAN_OBJECT_INTERFACES:
            raise ValueError(
                "human_object_interface must be one of "
                f"{HUMAN_OBJECT_INTERFACES!r}, got {human_object_interface!r}"
            )
        self.spec = spec
        self.human_object_interface = human_object_interface
        self.active_instruction = ""
        self._active_instruction_successful_commands: list[str] = []
        self.waiting_for_assistant = spec.family != "WATCH"
        self.successful_actions = 0
        self.opening_index = (
            0 if spec.opening.continuous_watch else len(spec.opening.actions)
        )
        self.watch_index = 0
        self._watch_resume_opening_index: int | None = None
        self._watch_resume_index: int | None = None
        self._watch_override_started = False
        self._watch_assistant_ever_spoken = False
        self._watch_latest_assistant_speech = ""
        self._watch_interrupted_plan = ""
        self._watch_pending_intervention: WatchIntervention | None = None
        self._watch_recovery_progress = 0
        self._watch_attempted_ids: set[str] = set()
        self._watch_timely_ids: set[str] = set()
        self._watch_missed_ids: set[str] = set()
        self._emitted_opening_utterances: set[int] = set()
        self._emitted_watch_utterances: set[int] = set()
        self.locations = dict(spec.opening_locations)
        # A FIND opening location is the answer to the memory question. Keep the
        # verifier state above, but disclose only locations caused after handoff.
        self._human_disclosed_locations: dict[str, str] | None = (
            {} if spec.family == "FIND" else None
        )
        self._guide_inspected_movable_keys: set[str] = set()
        self._narrate_confirmed_landmark_keys: set[str] = set()
        if (
            spec.family in _INSPECTED_MOVABLE_GROUNDING_FAMILIES
            and spec.family != "NARRATE"
        ):
            for command in spec.opening.actions:
                self._update_guide_inspected_movable(command)
        self._find_bound_landmark_keys: set[str] = set()
        self._find_human_handles: dict[str, str] = {}
        self._find_human_handle_counts = {"landmark": 0, "object": 0}
        self.successful_commands: list[str] = []
        self._successful_process_commands: list[str] = []
        self._successful_watch_steps: list[tuple[str, bool]] = []
        self.counters: dict[str, int] = {
            "human_command_qualifications": 0,
            "queue_preemptions": 0,
            "watch_opening_matches": 0,
            "watch_opening_mismatches": 0,
            "watch_opening_execution_failures": 0,
            "watch_default_matches": 0,
            "watch_default_mismatches": 0,
            "watch_default_execution_failures": 0,
            "watch_motor_recovery_actions": 0,
            "watch_motor_recovery_failures": 0,
            "watch_planned_speech_emitted": 0,
            "watch_planned_speech_mismatches": 0,
            "watch_overrides": 0,
            "watch_interventions": 0,
            "watch_intervention_attempts": 0,
            "watch_premature_interventions": 0,
            "watch_timely_interventions": 0,
            "watch_late_interventions": 0,
            "watch_missed_interventions": 0,
            "watch_premature_done": 0,
        }

    @property
    def opening(self) -> OpeningScript:
        return self.spec.opening

    @property
    def answer_key(self) -> tuple[Mapping[str, object], ...]:
        return self.spec.answer_key

    @property
    def assistant_receives_opening_history(self) -> bool:
        return self.spec.assistant_receives_opening_history

    @property
    def score_requires_assistant_done(self) -> bool:
        return self.spec.score_requires_assistant_done

    def qualify_human_command(
        self, command: str
    ) -> tuple[str, Mapping[str, object] | None]:
        """Apply an authored placement qualifier to one generic Human placement."""
        try:
            verb, args, scalar = _semantic_command_key(command)
        except InteractionSpecError:
            return command, None
        if verb not in {"PUTINSIDE", "PUTON"} or len(args) != 2 or scalar:
            return command, None
        command_key = verb, args, scalar
        for qualifier in self.spec.human_command_qualifiers:
            if _semantic_command_key(qualifier.match_command) != command_key:
                continue
            self.counters["human_command_qualifications"] += 1
            return qualifier.qualified_command, {
                "kind": (
                    "authored_putinside_qualifier"
                    if verb == "PUTINSIDE"
                    else "authored_put_qualifier"
                ),
                "matched_command": qualifier.match_command,
                "original_command": command,
                "qualified_command": qualifier.qualified_command,
                "space": qualifier.space,
                "place_spot": list(qualifier.place_spot),
            }
        return command, None

    def opening_utterances_before(self, action_index: int) -> tuple[str, ...]:
        return tuple(
            utterance.say
            for utterance in self.spec.opening.utterances
            if utterance.before_action_index == action_index
        )

    def _due_watch_speech(self) -> tuple[str, int, OpeningUtterance] | None:
        """Return the next authored Human speech turn whose action boundary was reached."""
        if self.spec.family != "WATCH" or not self.spec.opening.continuous_watch:
            return None
        for index, utterance in enumerate(self.spec.opening.utterances):
            if (
                index not in self._emitted_opening_utterances
                and utterance.before_action_index <= self.opening_index
            ):
                return "opening", index, utterance
        if self.opening_index < len(self.spec.opening.actions):
            return None
        for index, utterance in enumerate(self.spec.opening.watch_utterances):
            if (
                index not in self._emitted_watch_utterances
                and utterance.before_action_index <= self.watch_index
            ):
                return "watch", index, utterance
        return None

    def _next_unemitted_watch_speech(
        self,
    ) -> tuple[str, int, OpeningUtterance] | None:
        if self.spec.family != "WATCH" or not self.spec.opening.continuous_watch:
            return None
        for phase, utterances, emitted in (
            (
                "opening",
                self.spec.opening.utterances,
                self._emitted_opening_utterances,
            ),
            (
                "watch",
                self.spec.opening.watch_utterances,
                self._emitted_watch_utterances,
            ),
        ):
            for index, utterance in enumerate(utterances):
                if index not in emitted:
                    return phase, index, utterance
        return None

    def _note_flexible_watch_speech(self, human_say: str) -> bool:
        next_speech = self._next_unemitted_watch_speech()
        if next_speech is None:
            return False
        phase, speech_index, utterance = next_speech
        if human_say.strip() != utterance.say.strip():
            return False
        if phase == "opening":
            self._emitted_opening_utterances.add(speech_index)
        else:
            self._emitted_watch_utterances.add(speech_index)
        self.counters["watch_planned_speech_emitted"] += 1
        return True

    def _note_watch_speech(self, turn: HumanTurnContext, human_say: str) -> None:
        if turn.planned_speech_index is None:
            return
        if human_say.strip() != turn.planned_speech.strip():
            self.counters["watch_planned_speech_mismatches"] += 1
            return
        if turn.planned_speech_phase == "opening":
            self._emitted_opening_utterances.add(turn.planned_speech_index)
        elif turn.planned_speech_phase == "watch":
            self._emitted_watch_utterances.add(turn.planned_speech_index)
        self.counters["watch_planned_speech_emitted"] += 1

    def _due_watch_speech_turn(
        self,
        *,
        new_assistant_speech: str = "",
        preempt_queue: bool = False,
    ) -> HumanTurnContext | None:
        due_speech = self._due_watch_speech()
        if due_speech is None:
            return None
        phase, speech_index, utterance = due_speech
        self.waiting_for_assistant = False
        return HumanTurnContext(
            source="watch_speech",
            new_assistant_speech=new_assistant_speech,
            active_instruction=self.active_instruction,
            private_plan="",
            receives_frames=self.spec.human_receives_frames,
            preempt_queue=preempt_queue,
            interrupted_plan=self._watch_interrupted_plan,
            planned_speech=utterance.say,
            planned_speech_phase=phase,
            planned_speech_index=speech_index,
        )

    def pin_opening_intervals(self) -> bool:
        return self.spec.opening.frame_policy != "all_actions_budgeted"

    def assistant_opening_frames(
        self, frame_buffer: object, frame_cap: int
    ) -> tuple[list, int]:
        opening_frames = frame_buffer.fed_frames()
        if not self.assistant_receives_opening_history:
            current_frames = opening_frames[-1:]
            return current_frames, max(0, len(opening_frames) - 1)
        if self.spec.opening.frame_policy != "all_actions_budgeted":
            return opening_frames, 0
        selected = frame_buffer.budgeted_frames(
            frame_cap, recent_k=self.spec.opening.recent_intervals
        )
        return selected, max(0, len(opening_frames) - len(selected))

    def assistant_observation(
        self, observation: Mapping[str, object], *, first: bool
    ) -> dict[str, object]:
        visible_observation = _without_private_outcome_state(observation)
        return {
            **visible_observation,
            "task": self.spec.public_task_context,
            "goal": [],
            "interaction_family": self.spec.family,
            "interaction_task_context": self.spec.public_task_context,
            "interaction_opening_say": (
                " ".join(self.spec.opening.setup_speech)
                if first and not self.spec.opening.continuous_watch
                else ""
            ),
        }

    def first_handoff_say(self) -> str:
        if self.spec.opening.continuous_watch:
            return self.spec.opening.prelude_say
        return self.spec.opening.handoff_say

    def assert_tick_budget(self, tick_budget: int) -> None:
        minimum = len(self.spec.default_commands)
        if self.spec.opening.continuous_watch:
            minimum += (
                len(self.spec.opening.actions)
                + len(self.spec.opening.utterances)
                + len(self.spec.opening.watch_utterances)
            )
        if self.spec.family == "WATCH" and tick_budget < minimum:
            raise InteractionSpecError(
                f"{self.spec.task_id}: runtime tick budget {tick_budget} cannot execute "
                f"the {minimum}-action silent branch"
            )

    def keep_opening_start_frame(self) -> bool:
        if self.spec.opening.continuous_watch:
            return True
        if self.spec.opening.frame_policy == "all_actions_budgeted":
            return False
        return not self.spec.opening_evidence_action_indices

    def score_baseline_ready(self) -> bool:
        """Whether all unscored setup actions have completed.

        A continuous WATCH row may open its intervention window before the final
        authored opening action.  That action is the mistake the Assistant is
        expected to prevent, so the baseline must be captured immediately before
        that window rather than after a successful recovery.
        """
        boundaries = [
            window.earliest_opening_action_index
            for window in self.spec.watch_interventions
            if window.earliest_opening_action_index is not None
        ]
        boundary = min(boundaries, default=len(self.spec.opening.actions))
        return self.opening_index >= boundary

    def keep_opening_frame(self, command: str, index: int, total: int) -> bool:
        if self.spec.opening.frame_policy == "all_actions_budgeted":
            return True
        verb, _, _ = _command_key(command)
        if self.spec.opening_evidence_action_indices:
            return (
                index == total - 1
                or index in self.spec.opening_evidence_action_indices
                or verb in {"LOOKAT", "LOOKSCAN"}
            )
        return index == total - 1 or verb in {
            "CLOSE",
            "GRAB",
            "LOOKAT",
            "LOOKSCAN",
            "OPEN",
            "PUT",
            "PUTINSIDE",
            "PUTON",
        }

    def _watch_private_cursor(self) -> tuple[str, str]:
        if self.spec.family != "WATCH":
            return "", ""
        if self.spec.opening.continuous_watch and self.opening_index < len(
            self.spec.opening.actions
        ):
            return "watch_opening", self.spec.opening.actions[self.opening_index]
        if self.watch_index < len(self.spec.default_commands):
            return "watch_default", self.spec.default_commands[self.watch_index]
        return "", ""

    def prepare_human_turn(
        self, assistant_say: str, pending_commands: Sequence[str]
    ) -> HumanTurnContext:
        speech = assistant_say.strip()
        new_instruction = False
        watch_private_source, watch_private_command = self._watch_private_cursor()
        if self.spec.family == "WATCH" and speech:
            self._watch_assistant_ever_spoken = True
        speech_changed = bool(
            speech
            and (
                speech != self._watch_latest_assistant_speech
                if self.spec.family == "WATCH"
                else speech != self.active_instruction
            )
        )
        if speech_changed:
            new_instruction = True
            if pending_commands:
                self.counters["queue_preemptions"] += len(pending_commands)
            preserving_watch_instruction = bool(
                self.spec.family == "WATCH" and self.active_instruction
            )
            if preserving_watch_instruction:
                self.active_instruction = (
                    f"{self.active_instruction}\n\nNEWER ASSISTANT SPEECH:\n{speech}"
                )
            else:
                self.active_instruction = speech
                self._active_instruction_successful_commands.clear()
            if self.spec.family == "WATCH":
                self._watch_latest_assistant_speech = speech
            self.waiting_for_assistant = False
            if (
                self.spec.family == "WATCH"
                and not preserving_watch_instruction
                and self.opening_index < len(self.spec.opening.actions)
            ):
                self._watch_interrupted_plan = self.spec.opening.actions[
                    self.opening_index
                ]
                window = self._watch_pending_intervention
                self._watch_resume_opening_index = (
                    window.resume_opening_action_index
                    if window is not None
                    else self.opening_index
                )
                self._watch_resume_index = (
                    window.resume_action_index if window is not None else None
                )
                if (
                    window is not None
                    and self._watch_resume_opening_index is None
                    and self._watch_resume_index is None
                ):
                    self._watch_resume_opening_index = self.opening_index
                self._watch_override_started = False
            elif (
                self.spec.family == "WATCH"
                and not preserving_watch_instruction
                and self.watch_index < len(self.spec.default_commands)
            ):
                self._watch_interrupted_plan = self.spec.default_commands[
                    self.watch_index
                ]
                window = self._watch_pending_intervention
                self._watch_resume_opening_index = None
                self._watch_resume_index = (
                    window.resume_action_index
                    if window is not None and window.resume_action_index is not None
                    else self.watch_index
                )
                self._watch_override_started = False
        speech_turn = self._due_watch_speech_turn(
            new_assistant_speech=speech if new_instruction else "",
            preempt_queue=bool(new_instruction and pending_commands),
        )
        if speech_turn is not None:
            return speech_turn
        if new_instruction:
            return HumanTurnContext(
                source="assistant_new",
                new_assistant_speech=speech,
                active_instruction=self.active_instruction,
                private_plan=watch_private_command,
                receives_frames=self.spec.human_receives_frames,
                preempt_queue=bool(pending_commands),
                expected_watch_command=watch_private_command,
                interrupted_plan=self._watch_interrupted_plan,
            )
        if pending_commands:
            return HumanTurnContext(
                source="queued",
                new_assistant_speech="",
                active_instruction=self.active_instruction,
                private_plan="",
                receives_frames=self.spec.human_receives_frames,
                interrupted_plan=self._watch_interrupted_plan,
            )
        if self.active_instruction:
            return HumanTurnContext(
                source="assistant_active",
                new_assistant_speech="",
                active_instruction=self.active_instruction,
                private_plan=watch_private_command,
                receives_frames=self.spec.human_receives_frames,
                expected_watch_command=watch_private_command,
                interrupted_plan=self._watch_interrupted_plan,
            )
        if watch_private_source:
            self.waiting_for_assistant = False
            return HumanTurnContext(
                source=watch_private_source,
                new_assistant_speech="",
                active_instruction="",
                private_plan=watch_private_command,
                receives_frames=self.spec.human_receives_frames,
                expected_watch_command=watch_private_command,
            )
        self.waiting_for_assistant = True
        return HumanTurnContext(
            source="idle",
            new_assistant_speech="",
            active_instruction="",
            private_plan="",
            receives_frames=self.spec.human_receives_frames,
        )

    def _find_open_object_keys(self) -> tuple[str, ...]:
        open_objects: set[str] = set()
        for command in self.successful_commands:
            verb, arguments, _ = _command_key(command)
            keys = [name for name, object_id in arguments if object_id != 0]
            if not keys:
                continue
            if verb in {"OPEN", "OPENDOOR"}:
                open_objects.add(keys[0])
            elif verb in {"CLOSE", "CLOSEDOOR"}:
                open_objects.discard(keys[0])
        return tuple(sorted(open_objects))

    def _update_find_bound_landmark(self, command: str) -> None:
        if self.spec.family != "FIND":
            return
        verb, arguments, _ = _command_key(command)
        if verb not in {
            "CLOSE",
            "CLOSEDOOR",
            "GRAB",
            "LOOKAT",
            "LOOKSCAN",
            "OPEN",
            "OPENDOOR",
            *_FIND_PLACEMENT_VERBS,
            "WALK",
        }:
            return
        keys = [name for name, object_id in arguments if object_id != 0]
        if not keys:
            return
        self._find_bound_landmark_keys = {
            keys[-1] if verb in _FIND_PLACEMENT_VERBS else keys[0]
        }

    def _update_narrate_confirmed_landmark(
        self,
        turn: HumanTurnContext,
        command: str,
        observation: Mapping[str, object] | None,
    ) -> None:
        if self.spec.family != "NARRATE" or not turn.active_instruction:
            return
        verb, arguments, _ = _command_key(command)
        if verb not in {"LOOKAT", "WALK"} or len(arguments) != 1:
            return
        target, object_id = arguments[0]
        if object_id == 0 or target in self.spec.movable_object_keys:
            return
        action_kind = _find_action_kind(verb)
        if not _narrate_action_is_explicitly_requested(
            turn.active_instruction, action_kind
        ):
            return
        landmark_aliases, object_aliases, _ = self._guide_policy_grounding()
        confirmation_aliases = {
            name: self.spec.motor_aliases.get(name) or aliases
            for name, aliases in landmark_aliases.items()
        }
        requested_landmarks = _action_grounded_names(
            turn.active_instruction,
            confirmation_aliases,
            action_kind,
        )
        requested_movables = _action_grounded_names(
            turn.active_instruction,
            object_aliases,
            action_kind,
        )
        if verb == "WALK":
            requested_movables.difference_update(
                str(name) for name in (observation or {}).get("held", ())
            )
        if requested_landmarks == {target} and not requested_movables:
            self._narrate_confirmed_landmark_keys.add(target)

    def _update_guide_inspected_movable(self, command: str) -> None:
        if self.spec.family not in _INSPECTED_MOVABLE_GROUNDING_FAMILIES:
            return
        verb, arguments, _ = _command_key(command)
        movable = set(self.spec.movable_object_keys)
        narrate_handled = self.spec.family == "NARRATE" and verb in {
            "DROP",
            "GRAB",
            "PUT",
            "PUTBACK",
            "PUTIN",
            "PUTINSIDE",
            "PUTON",
            "PUTONFLOOR",
            "RELEASE",
        }
        if verb == "LOOKAT" or narrate_handled:
            self._guide_inspected_movable_keys.update(
                name
                for name, object_id in arguments
                if object_id != 0 and name in movable
            )
            return
        if verb != "CUT" or not arguments:
            return
        self._guide_inspected_movable_keys.update(
            _guide_cut_result_keys(
                arguments[0][0], movable, self.spec.object_descriptions
            )
        )
        self._guide_inspected_movable_keys.update(
            self._guide_visual_event_objects(command, movable)
        )

    def _guide_visual_event_objects(self, command: str, movable: set[str]) -> set[str]:
        try:
            command_key = _command_key(command)[:2]
        except InteractionSpecError:
            return set()
        occurrence = 1 + sum(
            1
            for previous in self.successful_commands
            if _command_key(previous)[:2] == command_key
        )
        revealed: set[str] = set()
        for event in self.spec.visual_state_events:
            after_command = str(event.get("after_command") or "")
            if not after_command or int(event.get("occurrence") or 1) != occurrence:
                continue
            try:
                event_key = _command_key(after_command)[:2]
            except InteractionSpecError:
                continue
            if event_key != command_key or not bool(event.get("visible", True)):
                continue
            objects = event.get("objects") or ()
            if isinstance(objects, Sequence) and not isinstance(objects, (str, bytes)):
                revealed.update(str(name) for name in objects if str(name) in movable)
        return revealed

    def _update_narrate_inspection_visibility(
        self,
        command: str,
        observation: Mapping[str, object] | None,
    ) -> None:
        if self.spec.family != "NARRATE" or observation is None:
            return
        verb, _, _ = _command_key(command)
        if not (verb.startswith("LOOK") or verb.startswith("TURN")):
            return
        movable = set(self.spec.movable_object_keys)
        visible = set(self._human_visible_object_keys(observation, required=False))
        self._guide_inspected_movable_keys.update(movable & visible)

    def _find_human_motor_objects(
        self,
        turn: HumanTurnContext,
        visible_object_keys: Sequence[str],
        held_object_keys: Sequence[str],
    ) -> list[str]:
        movable = set(self.spec.movable_object_keys)
        visible_or_held = {
            str(name) for name in (*visible_object_keys, *held_object_keys)
        }
        visible_movables = movable - set(self.spec.find_grounding_aliases)
        exposed = self._find_human_landmarks(
            turn, visible_object_keys, held_object_keys
        ) | (visible_movables & visible_or_held)
        return [name for name in self.spec.object_descriptions if name in exposed]

    def _find_human_landmarks(
        self,
        turn: HumanTurnContext,
        visible_object_keys: Sequence[str],
        held_object_keys: Sequence[str],
    ) -> set[str]:
        instruction = turn.new_assistant_speech or turn.active_instruction
        if not instruction:
            return set()
        aliases = self.spec.find_grounding_aliases
        state = _FindGuardState(
            targets=set(self.spec.find_target_keys),
            visible={str(name) for name in visible_object_keys},
            held={str(name) for name in held_object_keys},
            opened=set(self._find_open_object_keys()),
            bound=set(self._find_bound_landmark_keys),
            destinations={
                str(subject): {str(target) for target in target_keys}
                for subject, target_keys in self.spec.find_target_destinations.items()
            },
        )
        held_destinations = {
            destination
            for target in state.targets & state.held
            for destination in state.destinations.get(target, set())
        }
        grounded: set[str] = set()
        for action_kind in ("CLOSE", "LOOKAT", "OPEN", "PLACE", "WALK"):
            action_grounding = _find_command_grounding(
                instruction,
                action_kind,
                state,
                aliases,
                {},
                self.spec.find_destination_aliases,
                {},
            )
            grounded.update(action_grounding.landmarks)
            grounded.update(action_grounding.destinations & held_destinations)
            if action_kind == "CLOSE":
                grounded.update(
                    _grounded_open_find_references(
                        "CLOSE", instruction, state, action_grounding
                    )
                )
        return grounded

    def _find_human_descriptions(
        self, descriptions: Mapping[str, str], objects: Sequence[str]
    ) -> dict[str, str]:
        exposed: dict[str, str] = {}
        for name in objects:
            description = descriptions.get(name, "")
            aliases = self.spec.find_grounding_aliases.get(name, ())
            if aliases:
                references = "; ".join(aliases)
                description = f"public landmark references: {references}"
            else:
                visual_descriptions = _find_visual_description_aliases(description)
                if visual_descriptions:
                    description = visual_descriptions[0]
            exposed[name] = description
        return exposed

    def _ask_human_motor_objects(
        self,
        turn: HumanTurnContext,
        visible_object_keys: Sequence[str],
        held_object_keys: Sequence[str],
    ) -> list[str]:
        instruction = turn.new_assistant_speech or turn.active_instruction
        exposed = {str(name) for name in (*visible_object_keys, *held_object_keys)}
        movable = set(self.spec.movable_object_keys)
        if instruction:
            descriptions = self._current_descriptions()
            description_aliases = {
                key: _guide_entity_aliases(key, description.split("; currently", 1)[0])
                for key, description in descriptions.items()
            }
            authored_aliases = dict(self.spec.motor_aliases)
            for action_kind in _ASK_ACTION_CUES:
                grounded = _ask_action_grounded_names(
                    instruction, authored_aliases, action_kind
                )
                if not grounded:
                    grounded = _ask_action_grounded_names(
                        instruction, description_aliases, action_kind
                    )
                exposed.update(name for name in grounded if name not in movable)
        return [name for name in self.spec.object_descriptions if name in exposed]

    def _ask_human_descriptions(
        self, descriptions: Mapping[str, str], objects: Sequence[str]
    ) -> dict[str, str]:
        exposed: dict[str, str] = {}
        for name in objects:
            description = descriptions.get(name, "").split(
                "; public motor references:", 1
            )[0]
            visual_aliases = _find_visual_description_aliases(description)
            parts = list(visual_aliases[:1])
            motor_aliases = self.spec.motor_aliases.get(name, ())
            if motor_aliases:
                parts.append(f"public motor references: {'; '.join(motor_aliases)}")
            exposed[name] = "; ".join(parts) or "visible object"
        return exposed

    def _opaque_human_legend_enabled(self) -> bool:
        return bool(
            (self.spec.family == "FIND" and self.spec.find_grounding_aliases)
            or (
                self.spec.family in {"ASK", "GUIDE", "ORCHESTRATE"}
                and self.human_object_interface == "opaque_handles"
            )
        )

    def _opaque_human_handle_kind(self, name: str) -> str:
        if self.spec.family in {"ASK", "GUIDE", "ORCHESTRATE"}:
            return "object" if name in self.spec.movable_object_keys else "landmark"
        return "landmark" if name in self.spec.find_grounding_aliases else "object"

    def _find_human_handle(self, name: str) -> str:
        existing = self._find_human_handles.get(name)
        if existing is not None:
            return existing
        kind = self._opaque_human_handle_kind(name)
        self._find_human_handle_counts[kind] += 1
        handle = f"{kind}_{self._find_human_handle_counts[kind]}"
        self._find_human_handles[name] = handle
        return handle

    def _project_human_handles(
        self,
        objects: Sequence[str],
        descriptions: Mapping[str, str],
        internal_held: Sequence[str],
        *,
        enabled: bool,
    ) -> tuple[list[str], dict[str, str], list[str], dict[str, str], list[str]]:
        if not enabled:
            return (
                list(objects),
                dict(descriptions),
                list(internal_held),
                {},
                [],
            )
        internal_to_handle = {name: self._find_human_handle(name) for name in objects}
        human_objects = [internal_to_handle[name] for name in objects]
        human_descriptions = {
            internal_to_handle[name]: description
            for name, description in descriptions.items()
        }
        human_held = [
            internal_to_handle[name]
            for name in internal_held
            if name in internal_to_handle
        ]
        handle_map = {handle: name for name, handle in self._find_human_handles.items()}
        return (
            human_objects,
            human_descriptions,
            human_held,
            handle_map,
            list(human_objects),
        )

    def _human_visible_object_keys(
        self, observation: Mapping[str, object], *, required: bool
    ) -> list[str]:
        raw_visible = observation.get("_visible_object_keys_full")
        if isinstance(raw_visible, Sequence) and not isinstance(
            raw_visible, (str, bytes)
        ):
            return [str(name) for name in raw_visible]
        if not required:
            return []
        if bool(observation.get("objects_are_visible_only")):
            return [str(name) for name in observation.get("objects") or ()]
        vischeck = observation.get("vischeck")
        if isinstance(vischeck, Mapping) and not bool(vischeck.get("truncated")):
            visible = vischeck.get("visible")
            if isinstance(visible, Sequence) and not isinstance(visible, (str, bytes)):
                return [str(name) for name in visible]
        raise InteractionSpecError(
            f"{self.spec.task_id}: guarded interaction observation lacks complete visibility"
        )

    def _human_object_names(
        self,
        observation: Mapping[str, object],
        turn: HumanTurnContext,
        visible_object_keys: Sequence[str],
        *,
        prompt_only_find: bool,
        prompt_only_orchestrate: bool,
        guarded_find_legend: bool,
        opaque_ask_legend: bool,
    ) -> list[str]:
        raw_objects = [str(name) for name in observation.get("objects") or ()]
        held = [str(name) for name in observation.get("held") or ()]
        if prompt_only_find and self.human_object_interface == "full_ids":
            return list(dict.fromkeys((*raw_objects, *held)))
        allowed = set(self.spec.object_descriptions)
        objects = [str(name) for name in raw_objects if str(name) in allowed]
        objects.extend(name for name in held if name in allowed and name not in objects)
        if prompt_only_find or prompt_only_orchestrate:
            return list(self.spec.object_descriptions)
        if guarded_find_legend:
            return self._find_human_motor_objects(turn, visible_object_keys, held)
        if opaque_ask_legend:
            return self._ask_human_motor_objects(turn, visible_object_keys, held)
        return objects

    def _human_object_descriptions(
        self,
        objects: Sequence[str],
        visible_object_keys: Sequence[str],
        held_object_keys: Sequence[str],
        instruction: str,
        *,
        prompt_only_find: bool,
        prompt_only_orchestrate: bool,
        opaque_find_legend: bool,
        opaque_ask_legend: bool,
    ) -> dict[str, str]:
        movable = set(self.spec.movable_object_keys)
        human_locations = self._human_disclosed_locations
        if (
            prompt_only_find
            or prompt_only_orchestrate
            or self.spec.family in _INSPECTED_MOVABLE_GROUNDING_FAMILIES
        ):
            human_locations = {}
        descriptions = (
            {
                key: description.split("; currently", 1)[0]
                for key, description in self.spec.object_descriptions.items()
            }
            if self.spec.family == "WATCH"
            else self._current_descriptions(locations=human_locations)
        )
        if self.spec.family in _INSPECTED_MOVABLE_GROUNDING_FAMILIES:
            descriptions = {
                key: (
                    _guide_description_without_uninspected_label(
                        _guide_description_without_location(description)
                    )
                    if key in movable and key not in self._guide_inspected_movable_keys
                    else _guide_description_without_location(description)
                    if key in movable
                    else description
                )
                for key, description in descriptions.items()
            }
        alias_visible = (
            set(visible_object_keys)
            | {str(name) for name in held_object_keys}
            | self._guide_inspected_movable_keys
        )
        alias_visible.update(
            name
            for name, aliases in self.spec.motor_aliases.items()
            if name in movable
            and _alias_phrase_is_explicitly_mentioned(instruction, aliases)
        )
        if self.spec.motor_aliases:
            descriptions = {
                name: (
                    f"{description}; public motor references: "
                    f"{'; '.join(self.spec.motor_aliases[name])}"
                    if name in self.spec.motor_aliases
                    and (name not in movable or name in alias_visible)
                    else description
                )
                for name, description in descriptions.items()
            }
        if opaque_find_legend:
            return self._find_human_descriptions(descriptions, objects)
        if opaque_ask_legend:
            return self._ask_human_descriptions(descriptions, objects)
        return descriptions

    def human_observation(
        self, observation: Mapping[str, object], turn: HumanTurnContext
    ) -> dict[str, object]:
        visible_observation = _without_private_outcome_state(observation)
        find_human_control = str(
            observation.get("_interaction_find_human_control") or "dynamic-guarded"
        )
        if find_human_control not in FIND_HUMAN_CONTROLS:
            raise InteractionSpecError(
                f"unknown FIND Human control mode {find_human_control!r}"
            )
        prompt_only_find = bool(
            self.spec.family == "FIND" and find_human_control == "prompt-only"
        )
        prompt_only_orchestrate = bool(
            self.spec.family == "ORCHESTRATE"
            and not bool(
                observation.get("_interaction_orchestrate_human_turn_guard", True)
            )
        )
        human_task_context = (
            _PROMPT_ONLY_FIND_HUMAN_TASK_CONTEXT
            if prompt_only_find
            else _PROMPT_ONLY_ORCHESTRATE_HUMAN_TASK_CONTEXT
            if prompt_only_orchestrate
            else self.spec.human_task_context
        )
        guarded_find_legend = bool(
            self.spec.family == "FIND"
            and self.spec.find_grounding_aliases
            and not prompt_only_find
        )
        opaque_find_legend = (
            guarded_find_legend
            or (prompt_only_find and self.human_object_interface == "opaque_handles")
            or prompt_only_orchestrate
        )
        opaque_ask_legend = bool(
            self.spec.family in {"ASK", "GUIDE", "ORCHESTRATE"}
            and self.human_object_interface == "opaque_handles"
        )
        opaque_human_legend = opaque_find_legend or opaque_ask_legend
        policy_visible_object_keys = self._human_visible_object_keys(
            observation, required=opaque_human_legend
        )
        if opaque_ask_legend and self.spec.family == "GUIDE":
            policy_visible_object_keys.extend(
                name
                for name in sorted(self._guide_inspected_movable_keys)
                if name not in policy_visible_object_keys
            )
        instruction = turn.new_assistant_speech or turn.active_instruction
        held_object_keys = [str(name) for name in observation.get("held") or ()]
        objects = self._human_object_names(
            observation,
            turn,
            policy_visible_object_keys,
            prompt_only_find=prompt_only_find,
            prompt_only_orchestrate=prompt_only_orchestrate,
            guarded_find_legend=guarded_find_legend,
            opaque_ask_legend=opaque_ask_legend,
        )
        descriptions = self._human_object_descriptions(
            objects,
            policy_visible_object_keys,
            held_object_keys,
            instruction,
            prompt_only_find=prompt_only_find,
            prompt_only_orchestrate=prompt_only_orchestrate,
            opaque_find_legend=opaque_find_legend,
            opaque_ask_legend=opaque_ask_legend,
        )
        grounded_only = self.spec.family == "NARRATE"
        if grounded_only:
            objects = list(self.spec.object_descriptions)
            descriptions = {
                name: description
                for name, description in descriptions.items()
                if name in self.spec.object_descriptions
            }
        policy_target_keys = self.spec.find_target_keys
        if self.spec.family in _INSPECTED_MOVABLE_GROUNDING_FAMILIES:
            policy_target_keys = self.spec.movable_object_keys
            if self.spec.family == "NARRATE":
                policy_visible_object_keys = sorted(self._guide_inspected_movable_keys)
            else:
                policy_visible_object_keys.extend(
                    name
                    for name in sorted(self._guide_inspected_movable_keys)
                    if name not in policy_visible_object_keys
                )
        guide_grounding_aliases: Mapping[str, tuple[str, ...]] = {}
        guide_object_aliases: Mapping[str, tuple[str, ...]] = {}
        guide_target_sources: Mapping[str, tuple[str, ...]] = {}
        if self.spec.family in _INSPECTED_MOVABLE_GROUNDING_FAMILIES:
            (
                guide_grounding_aliases,
                guide_object_aliases,
                guide_target_sources,
            ) = self._guide_policy_grounding()
        internal_held = held_object_keys
        (
            human_objects,
            human_descriptions,
            human_held,
            handle_map,
            available_handles,
        ) = self._project_human_handles(
            objects,
            descriptions,
            internal_held,
            enabled=opaque_human_legend,
        )
        watch_policy = self._human_watch_policy()
        next_planned_speech = watch_policy.get("next_unemitted_planned_speech", {})
        return {
            **visible_observation,
            "task": human_task_context,
            "goal": [],
            "objects": human_objects,
            "object_desc": human_descriptions,
            "held": human_held,
            "interaction_family": self.spec.family,
            "interaction_task_context": human_task_context,
            "interaction_opening_say": " ".join(
                (*self.spec.opening.setup_speech, self.spec.opening.handoff_say)
            ).strip(),
            "interaction_turn_source": turn.source,
            "interaction_active_instruction": turn.active_instruction,
            "interaction_active_instruction_action_memory": [
                {"action": command, "result": "succeeded"}
                for command in self._active_instruction_successful_commands
            ],
            "interaction_successful_action_history": [
                {"action": command, "result": "succeeded"}
                for command in (
                    self._successful_process_commands
                    if self.spec.family == "NARRATE"
                    else ()
                )
            ],
            "interaction_private_plan": turn.private_plan,
            "interaction_interrupted_plan": turn.interrupted_plan,
            "interaction_planned_speech": turn.planned_speech,
            "interaction_next_planned_speech": (
                str(next_planned_speech.get("say") or "")
                if isinstance(next_planned_speech, Mapping)
                else ""
            ),
            "interaction_watch_policy": watch_policy,
            "interaction_human_receives_frames": turn.receives_frames,
            "interaction_grounded_legend_only": grounded_only,
            "interaction_find_target_keys": list(policy_target_keys),
            "interaction_visible_object_keys": policy_visible_object_keys,
            "interaction_opaque_handles": opaque_human_legend,
            "interaction_human_object_interface": (
                "opaque_handles" if opaque_human_legend else self.human_object_interface
            ),
            "interaction_find_human_control": find_human_control,
            "interaction_orchestrate_human_turn_guard": (not prompt_only_orchestrate),
            "objects_are_visible_only": (
                False
                if opaque_human_legend
                else bool(observation.get("objects_are_visible_only"))
            ),
            "_interaction_human_handle_map": handle_map,
            "_interaction_human_available_handles": available_handles,
            "_interaction_human_internal_keys": list(self.spec.object_descriptions),
            "_interaction_ask_movable_object_keys": list(self.spec.movable_object_keys),
            "_interaction_internal_object_desc": descriptions,
            "_interaction_internal_held_object_keys": internal_held,
            "_interaction_watch_static_locations": (
                dict(self.spec.opening_locations) if self.spec.family == "WATCH" else {}
            ),
            "_interaction_watch_authored_planned_speech": [
                utterance.say
                for utterances in (
                    self.spec.opening.utterances,
                    self.spec.opening.watch_utterances,
                )
                for utterance in utterances
            ],
            "_interaction_motor_aliases": {
                name: list(aliases) for name, aliases in self.spec.motor_aliases.items()
            },
            "_interaction_find_grounding_aliases": {
                name: list(aliases)
                for name, aliases in (
                    guide_grounding_aliases or self.spec.find_grounding_aliases
                ).items()
            },
            "_interaction_find_object_aliases": {
                name: list(aliases)
                for name, aliases in (
                    guide_object_aliases or self.spec.find_object_aliases
                ).items()
            },
            "_interaction_find_target_aliases": {
                name: list(aliases)
                for name, aliases in self.spec.find_target_aliases.items()
            },
            "_interaction_find_destination_aliases": {
                name: list(aliases)
                for name, aliases in self.spec.find_destination_aliases.items()
            },
            "_interaction_find_target_destinations": {
                name: list(destinations)
                for name, destinations in (
                    guide_target_sources or self.spec.find_target_destinations
                ).items()
            },
            "_interaction_find_open_object_keys": list(self._find_open_object_keys()),
            "_interaction_find_bound_landmark_keys": list(
                self._find_bound_landmark_keys
            ),
        }

    def _guide_policy_grounding(
        self,
    ) -> tuple[
        dict[str, tuple[str, ...]],
        dict[str, tuple[str, ...]],
        dict[str, tuple[str, ...]],
    ]:
        movable = set(self.spec.movable_object_keys)
        object_aliases = {
            key: (
                self.spec.motor_aliases.get(key)
                or _narrate_entity_aliases(self.spec.object_descriptions.get(key, ""))
                if self.spec.family == "NARRATE"
                else tuple(
                    dict.fromkeys(
                        (
                            *_guide_entity_aliases(
                                key, self.spec.object_descriptions.get(key, "")
                            ),
                            *self.spec.motor_aliases.get(key, ()),
                        )
                    )
                )
            )
            for key in movable
        }
        target_sources: dict[str, tuple[str, ...]] = {}
        source_keys = set(self.spec.object_descriptions) - movable
        for key in movable:
            _relation, separator, source = self.locations.get(key, "").partition(":")
            if not separator or not source:
                continue
            target_sources[key] = (source,)
            source_keys.add(source)
        if self.spec.family in {"NARRATE", "ORCHESTRATE"}:
            landmark_aliases = {
                key: _orchestrate_landmark_aliases(
                    key,
                    self.spec.object_descriptions.get(key, ""),
                    self.spec.motor_aliases.get(key, ()),
                )
                for key in source_keys
            }
        else:
            landmark_aliases = {
                key: tuple(
                    dict.fromkeys(
                        (
                            *_guide_landmark_aliases(
                                key, self.spec.object_descriptions.get(key, "")
                            ),
                            *self.spec.motor_aliases.get(key, ()),
                        )
                    )
                )
                for key in source_keys
            }
        return landmark_aliases, object_aliases, target_sources

    def _human_watch_policy(self) -> Mapping[str, object]:
        if self.spec.family != "WATCH":
            return {}
        next_speech = self._next_unemitted_watch_speech()
        next_speech_policy = (
            {
                "phase": "opening" if next_speech[0] == "opening" else "action",
                "before_action_index": next_speech[2].before_action_index,
                "say": next_speech[2].say,
            }
            if next_speech is not None
            else {}
        )
        policy: dict[str, object] = {
            "assistant_ever_spoken": self._watch_assistant_ever_spoken,
            "next_unemitted_planned_speech": next_speech_policy,
            "planned_speech_turns": [
                {
                    "phase": phase,
                    "before_action_index": utterance.before_action_index,
                    "say": utterance.say,
                }
                for phase, utterances, emitted in (
                    (
                        "opening",
                        self.spec.opening.utterances,
                        self._emitted_opening_utterances,
                    ),
                    (
                        "action",
                        self.spec.opening.watch_utterances,
                        self._emitted_watch_utterances,
                    ),
                )
                for index, utterance in enumerate(utterances)
                if index not in emitted
            ],
            "assistant_priority": (
                "A new Assistant instruction takes priority while it remains unfinished. Follow "
                "it literally, even if it is wrong, without changing it from private-plan or task "
                "reasoning. Once it is complete, continue the no-advice plan from the current "
                "supplied cursor; never replay the interrupted step or rebuild its prerequisites. "
                "If an action fails, report it and wait instead of recovering or retrying. Authored "
                "planned speech is mandatory at its natural real-world moment, speech-only, and "
                "does not cancel the active instruction or advance the private action cursor."
            ),
        }
        policy.update(
            {
                "no_advice_opening_actions": list(self.spec.opening.actions),
                "no_advice_actions": list(self.spec.default_commands),
                "opening_cursor": self.opening_index,
                "action_cursor": self.watch_index,
                "current_action": (
                    self.spec.opening.actions[self.opening_index]
                    if self.opening_index < len(self.spec.opening.actions)
                    else (
                        self.spec.default_commands[self.watch_index]
                        if self.watch_index < len(self.spec.default_commands)
                        else ""
                    )
                ),
            }
        )
        return policy

    def note_model_idle(self, turn: HumanTurnContext, human_say: str) -> None:
        if turn.source == "watch_speech":
            self._note_watch_speech(turn, human_say)
            return
        if self.spec.family == "WATCH" and self._note_flexible_watch_speech(human_say):
            return
        if turn.source not in {"assistant_new", "assistant_active"}:
            self.waiting_for_assistant = True
            return
        if not human_say.strip():
            return
        if self.spec.family == "GUIDE":
            landmark_aliases, object_aliases, target_sources = (
                self._guide_policy_grounding()
            )
            if _successful_command_completes_claimed_instruction(
                human_say,
                (
                    self._active_instruction_successful_commands[-1]
                    if self._active_instruction_successful_commands
                    else ""
                ),
                turn.active_instruction,
                self._active_instruction_successful_commands,
                (),
                object_aliases,
                landmark_aliases,
                target_sources,
            ):
                self._finish_active_instruction()
            return
        if self.spec.family == "WATCH" and human_say.rstrip().endswith("?"):
            self.waiting_for_assistant = True
            return
        self._finish_active_instruction(
            watch_recovery_complete=self._watch_recovery_is_complete()
        )

    def note_assistant_turn(
        self,
        *,
        mode: str,
        say: str,
        done: bool,
        goal_success: bool,
    ) -> None:
        if self.spec.family != "WATCH":
            return
        if done:
            if (
                self.opening_index < len(self.spec.opening.actions)
                or self.watch_index < len(self.spec.default_commands)
                or not goal_success
            ):
                self.counters["watch_premature_done"] += 1
            return
        if mode == "silence" or not say.strip():
            return
        self.counters["watch_interventions"] += 1
        due = self._due_watch_intervention()
        if due is not None:
            self._watch_attempted_ids.add(due.intervention_id)
            if self._watch_pending_intervention is None:
                self._watch_pending_intervention = due
                self._watch_recovery_progress = 0
                self.counters["watch_intervention_attempts"] += 1
        elif self._watch_missed_ids:
            self.counters["watch_late_interventions"] += 1
        else:
            self.counters["watch_premature_interventions"] += 1

    def process_requirements_met(self) -> bool:
        if self.spec.family == "WATCH":
            required = tuple(
                str(command)
                for command in self.spec.completion_contract.get(
                    "required_successful_commands", ()
                )
            )
            commands_met = self._watch_required_commands_met(required)
            forbidden_clear = not any(
                self._watch_successful_step_matches(step, str(command))
                for command in self.spec.completion_contract.get(
                    "forbidden_successful_commands", ()
                )
                for step in self._successful_watch_steps
            )
            sequences_met = all(
                self._watch_command_subsequence_holds(sequence)
                for sequence in self.spec.completion_contract.get(
                    "required_command_sequences", ()
                )
            )
            order_guards_met = all(
                not self._watch_forbidden_before_violated(rule)
                for rule in self.spec.completion_contract.get("forbidden_before", ())
            )
        else:
            required = tuple(
                str(command)
                for command in self.spec.completion_contract.get(
                    "required_successful_commands", ()
                )
            )
            commands_met = self._process_required_commands_met(required)
            forbidden_clear = not any(
                self._process_commands_equivalent(observed, str(forbidden))
                for forbidden in self.spec.completion_contract.get(
                    "forbidden_successful_commands", ()
                )
                for observed in self._successful_process_commands
            )
            sequences_met = all(
                self._process_command_subsequence_holds(sequence)
                for sequence in self.spec.completion_contract.get(
                    "required_command_sequences", ()
                )
            )
            order_guards_met = all(
                not self._process_forbidden_before_violated(rule)
                for rule in self.spec.completion_contract.get("forbidden_before", ())
            )
        watch_met = self.spec.family != "WATCH" or (
            self.counters["watch_premature_interventions"] == 0
            and self.counters["watch_late_interventions"] == 0
            and self.counters["watch_missed_interventions"] == 0
            and self.counters["watch_premature_done"] == 0
            and len(self._watch_timely_ids) == len(self.spec.watch_interventions)
        )
        return (
            commands_met
            and forbidden_clear
            and sequences_met
            and order_guards_met
            and watch_met
        )

    def _watch_successful_step_matches(
        self, step: tuple[str, bool], expected: str
    ) -> bool:
        command, inside_verified = step
        return _watch_semantic_commands_equivalent(
            command,
            expected,
            inside_verified=inside_verified,
        )

    def _watch_required_commands_met(self, required: Sequence[str]) -> bool:
        owners: dict[int, int] = {}

        def assign(required_index: int, seen: set[int]) -> bool:
            for observed_index, step in enumerate(self._successful_watch_steps):
                if observed_index in seen or not self._watch_successful_step_matches(
                    step, required[required_index]
                ):
                    continue
                seen.add(observed_index)
                previous = owners.get(observed_index)
                if previous is None or assign(previous, seen):
                    owners[observed_index] = required_index
                    return True
            return False

        return all(assign(index, set()) for index in range(len(required)))

    def _watch_command_subsequence_holds(self, required: Sequence[str]) -> bool:
        cursor = 0
        for step in self._successful_watch_steps:
            if cursor < len(required) and self._watch_successful_step_matches(
                step, str(required[cursor])
            ):
                cursor += 1
        return cursor == len(required)

    def _watch_forbidden_before_violated(self, rule: Mapping[str, object]) -> bool:
        command = str(rule["command"])
        prerequisite = str(rule["prerequisite"])
        prerequisite_seen = False
        for step in self._successful_watch_steps:
            if self._watch_successful_step_matches(step, prerequisite):
                prerequisite_seen = True
            if (
                self._watch_successful_step_matches(step, command)
                and not prerequisite_seen
            ):
                return True
        return False

    def scripted_oracle_chunk(self) -> tuple[tuple[str, ...], str]:
        """Return the oracle actions belonging to the current authored GUIDE step."""
        if self.spec.family == "WATCH" and self._watch_pending_intervention is not None:
            window = self._watch_pending_intervention
            return window.recovery_commands, (
                "That's done."
                if window.dialogue_index + 1 == len(self.spec.reference_dialogue)
                else "That's done. I'll keep going."
            )
        start = self._oracle_action_progress()
        later_anchors = [
            anchor
            for anchor in self.spec.reference_dialogue_action_indices
            if anchor is not None and anchor > start
        ]
        end = min(later_anchors, default=len(self.spec.oracle_commands))
        commands = self.spec.oracle_commands[start:end]
        if not commands:
            return (), ""
        dialogue_index = max(
            (
                index
                for index, anchor in enumerate(
                    self.spec.reference_dialogue_action_indices
                )
                if anchor is not None and anchor <= start
            ),
            default=0,
        )
        if self.spec.scripted_human_replies:
            reply = self.spec.scripted_human_replies[dialogue_index]
        elif end < len(self.spec.oracle_commands):
            reply = "That's done. What should I do next?"
        else:
            reply = "That's done."
        return commands, reply

    def note_step(
        self,
        turn: HumanTurnContext,
        command: str,
        *,
        succeeded: bool,
        human_say: str = "",
        action_sequence_complete: bool = False,
        motor_recovery: bool = False,
        observation: Mapping[str, object] | None = None,
        human_command: str | None = None,
    ) -> None:
        inside_verified = bool(
            succeeded
            and observation is not None
            and _watch_inside_world_holds(command, observation)
        )
        if succeeded:
            process_command = self._normalized_process_command(command)
            self._update_locations(command)
            self._update_find_bound_landmark(command)
            self._update_narrate_confirmed_landmark(turn, command, observation)
            self._update_guide_inspected_movable(command)
            self._update_narrate_inspection_visibility(command, observation)
            self.successful_commands.append(command)
            self._successful_process_commands.append(process_command)
            self.successful_actions += 1
            if turn.active_instruction and turn.source in {
                "assistant_new",
                "assistant_active",
                "queued",
            }:
                self._active_instruction_successful_commands.append(
                    human_command or command
                )
            if self.spec.family == "WATCH":
                self._successful_watch_steps.append((command, inside_verified))
        watch_script_step = self._note_watch_script_step(
            turn,
            command,
            succeeded=succeeded,
            motor_recovery=motor_recovery,
            inside_verified=inside_verified,
        )
        if watch_script_step:
            if self._should_finish_active_instruction(
                turn,
                command,
                succeeded=succeeded,
                human_say=human_say,
                action_sequence_complete=action_sequence_complete,
            ):
                self._finish_active_instruction()
            return
        recovery_complete = self._note_watch_override_step(
            turn,
            command,
            succeeded=succeeded,
            inside_verified=inside_verified,
            observation=observation,
        )
        if self._should_finish_active_instruction(
            turn,
            command,
            succeeded=succeeded,
            human_say=human_say,
            action_sequence_complete=action_sequence_complete,
        ):
            self._finish_active_instruction(watch_recovery_complete=recovery_complete)

    def _note_watch_script_step(
        self,
        turn: HumanTurnContext,
        command: str,
        *,
        succeeded: bool,
        motor_recovery: bool,
        inside_verified: bool,
    ) -> bool:
        private_source, private_command = self._watch_private_cursor()
        if (
            self.spec.family == "WATCH"
            and turn.source in {"assistant_new", "assistant_active"}
            and self._watch_pending_intervention is None
            and turn.expected_watch_command
            and turn.expected_watch_command == private_command
            and self._watch_command_matches(
                command,
                private_command,
                succeeded=succeeded,
                inside_verified=inside_verified,
            )
        ):
            if private_source == "watch_opening":
                self._note_watch_opening_step(
                    turn,
                    command,
                    succeeded=succeeded,
                    motor_recovery=motor_recovery,
                    inside_verified=inside_verified,
                )
            elif private_source == "watch_default":
                self._note_watch_default_step(
                    turn,
                    command,
                    succeeded=succeeded,
                    motor_recovery=motor_recovery,
                    inside_verified=inside_verified,
                )
            else:
                return False
            _, next_private_command = self._watch_private_cursor()
            self._watch_interrupted_plan = next_private_command
            if private_source == "watch_opening":
                self._watch_resume_opening_index = self.opening_index
            else:
                self._watch_resume_index = self.watch_index
            return True
        if turn.source == "watch_opening":
            self._note_watch_opening_step(
                turn,
                command,
                succeeded=succeeded,
                motor_recovery=motor_recovery,
                inside_verified=inside_verified,
            )
            return True
        if turn.source == "watch_default":
            self._note_watch_default_step(
                turn,
                command,
                succeeded=succeeded,
                motor_recovery=motor_recovery,
                inside_verified=inside_verified,
            )
            return True
        return False

    def _note_watch_opening_step(
        self,
        turn: HumanTurnContext,
        command: str,
        *,
        succeeded: bool,
        motor_recovery: bool,
        inside_verified: bool,
    ) -> None:
        matches_plan = self._watch_command_matches(
            command,
            turn.expected_watch_command,
            succeeded=succeeded,
            inside_verified=inside_verified,
        )
        if succeeded and matches_plan:
            due = self._due_watch_intervention()
            if (
                due is not None
                and due.earliest_opening_action_index == self.opening_index
            ):
                self._watch_missed_ids.add(due.intervention_id)
                self.counters["watch_missed_interventions"] += 1
            self.opening_index += 1
            self.counters["watch_opening_matches"] += 1
        elif matches_plan:
            self.counters["watch_opening_execution_failures"] += 1
        elif motor_recovery and self._is_watch_motor_recovery(command):
            self._note_watch_motor_recovery(succeeded)
        else:
            self.counters["watch_opening_mismatches"] += 1

    def _note_watch_default_step(
        self,
        turn: HumanTurnContext,
        command: str,
        *,
        succeeded: bool,
        motor_recovery: bool,
        inside_verified: bool,
    ) -> None:
        matches_plan = self._watch_command_matches(
            command,
            turn.expected_watch_command,
            succeeded=succeeded,
            inside_verified=inside_verified,
        )
        if succeeded and matches_plan:
            due = self._due_watch_intervention()
            if due is not None:
                self._watch_missed_ids.add(due.intervention_id)
                self.counters["watch_missed_interventions"] += 1
            self.watch_index += 1
            self.counters["watch_default_matches"] += 1
        elif matches_plan:
            self.counters["watch_default_execution_failures"] += 1
        elif motor_recovery and self._is_watch_motor_recovery(command):
            self._note_watch_motor_recovery(succeeded)
        else:
            self.counters["watch_default_mismatches"] += 1

    def _note_watch_override_step(
        self,
        turn: HumanTurnContext,
        command: str,
        *,
        succeeded: bool,
        inside_verified: bool,
        observation: Mapping[str, object] | None,
    ) -> bool:
        if self.spec.family == "WATCH" and turn.source in {
            "assistant_new",
            "assistant_active",
            "queued",
        }:
            self._watch_override_started = True
            self._advance_watch_resume(
                command,
                succeeded=succeeded,
                inside_verified=inside_verified,
            )
            recovery_complete = self._advance_watch_recovery(
                command,
                succeeded=succeeded,
                inside_verified=inside_verified,
                observation=observation,
            )
            return recovery_complete
        return False

    def _should_finish_active_instruction(
        self,
        turn: HumanTurnContext,
        command: str,
        *,
        succeeded: bool,
        human_say: str,
        action_sequence_complete: bool,
    ) -> bool:
        if not succeeded or turn.source not in {
            "assistant_new",
            "assistant_active",
            "queued",
        }:
            return False
        if (
            self.spec.family == "WATCH"
            and not self._watch_active_instruction_goals_complete(
                turn.active_instruction
            )
        ):
            return False
        if self.spec.family == "GUIDE":
            landmark_aliases, object_aliases, target_sources = (
                self._guide_policy_grounding()
            )
            return _successful_command_completes_claimed_instruction(
                human_say,
                command,
                turn.active_instruction,
                self._active_instruction_successful_commands,
                (),
                object_aliases,
                landmark_aliases,
                target_sources,
            )
        sequence_finished = action_sequence_complete and (
            (
                self.spec.family == "WATCH"
                and _command_key(command)[0] in _WATCH_UNIT_END
            )
            or (self.spec.family == "NARRATE" and turn.source == "queued")
        )
        return bool(human_say.strip() or sequence_finished)

    def _watch_active_instruction_goals_complete(self, instruction: str) -> bool:
        parsed = _watch_parse_commands(self._active_instruction_successful_commands)
        aliases = _watch_command_aliases(
            parsed,
            self.spec.object_descriptions,
            self.spec.motor_aliases,
        )
        units = _watch_lifecycle_instruction_units(instruction, aliases)
        if len(units) <= 1:
            return True
        return _watch_instruction_units_are_covered(parsed, units)

    def _is_watch_motor_recovery(self, command: str) -> bool:
        verb, _, _ = _command_key(command)
        return verb in _WATCH_MOTOR_RECOVERY_VERBS

    def _watch_command_matches(
        self,
        command: str,
        expected: str,
        *,
        succeeded: bool,
        inside_verified: bool,
    ) -> bool:
        command_key = _command_key(command)
        expected_key = _command_key(expected)
        if command_key == expected_key:
            return True
        if not succeeded:
            return False
        if _watch_putinside_parent_space_equivalent(command_key, expected_key):
            return inside_verified
        if _semantic_command_key(command) != _semantic_command_key(expected):
            return False
        verb, args, _ = command_key
        expected_verb, _, _ = expected_key
        if {verb, expected_verb} != {"PUT", "PUTON"}:
            return False
        subject, target = (name for name, _ in args)
        return self.locations.get(subject) == f"on:{target}"

    def _note_watch_motor_recovery(self, succeeded: bool) -> None:
        self.counters["watch_motor_recovery_actions"] += 1
        if not succeeded:
            self.counters["watch_motor_recovery_failures"] += 1

    def assistant_state(self, observation: Mapping[str, object]) -> dict[str, object]:
        score = observation.get("score")
        score = score if isinstance(score, Mapping) else {}
        physical_goal_success = bool(score.get("success"))
        verification_points_met = all(
            bool(point.get("ok"))
            for point in self.verification_results(score, observation)
        )
        verification_points_define_success = bool(
            self.spec.completion_contract.get(
                "verification_points_define_success", False
            )
        )
        contract_world_success = (
            verification_points_met
            if verification_points_define_success
            else physical_goal_success
        )
        due = self._due_watch_intervention()
        next_window = next(
            (
                window
                for window in self.spec.watch_interventions
                if window.intervention_id not in self._watch_timely_ids
                and window.intervention_id not in self._watch_missed_ids
            ),
            None,
        )
        return {
            "family": self.spec.family,
            "waiting_for_assistant": self.waiting_for_assistant,
            "has_active_instruction": bool(self.active_instruction),
            "successful_actions": self.successful_actions,
            "oracle_action_progress": self._oracle_action_progress(),
            "watch_opening_index": self.opening_index,
            "watch_opening_total": len(self.spec.opening.actions),
            "watch_index": self.watch_index,
            "watch_trigger_index": (
                next_window.before_action_index if next_window is not None else None
            ),
            "watch_due_intervention": (
                self._watch_intervention_state(due) if due is not None else None
            ),
            "watch_default_complete": (
                self.opening_index >= len(self.spec.opening.actions)
                and self.watch_index >= len(self.spec.default_commands)
            ),
            "physical_goal_success": physical_goal_success,
            "contract_world_success": contract_world_success,
            "process_requirements_met": self.process_requirements_met(),
            "verification_points_met": verification_points_met,
            "goal_success": (
                contract_world_success
                and self.process_requirements_met()
                and verification_points_met
            ),
        }

    def _oracle_action_progress(self) -> int:
        """Match the authored oracle prefix while ignoring optional waits and extras."""
        observed = (
            self._successful_watch_steps
            if self.spec.family == "WATCH"
            else [(command, False) for command in self._successful_process_commands]
        )
        strict_progress = self._matched_oracle_progress(observed)
        inferred_progress = self._matched_oracle_progress(
            observed, allow_omitted_walks=True
        )
        if self.spec.family in {"GUIDE", "NARRATE", "ORCHESTRATE"}:
            dialogue_progress = self._matched_guide_dialogue_progress(observed)
            return max(strict_progress, inferred_progress, dialogue_progress)
        return max(strict_progress, inferred_progress)

    def _matched_guide_dialogue_progress(
        self, observed: Sequence[tuple[str, bool]]
    ) -> int:
        observed_index = 0
        anchors = self.spec.reference_dialogue_action_indices
        if not anchors or anchors[0] != 0 or any(index is None for index in anchors):
            return 0
        resolved = tuple(index for index in anchors if index is not None)
        if any(
            not 0 <= index < len(self.spec.oracle_commands) for index in resolved
        ) or any(left >= right for left, right in zip(resolved, resolved[1:])):
            return 0
        boundaries = resolved[1:] + (len(self.spec.oracle_commands),)
        progress = 0
        for boundary in boundaries:
            pending = self._guide_block_requirements(progress, boundary)
            quarter_turns = {"TURNLEFT": 0, "TURNRIGHT": 0}
            while pending and observed_index < len(observed):
                candidate = observed[observed_index]
                match = next(
                    (
                        index
                        for index, expected in enumerate(pending)
                        if self._guide_successful_step_matches(candidate, expected)
                    ),
                    None,
                )
                if match is not None:
                    pending.pop(match)
                    quarter_turns = {"TURNLEFT": 0, "TURNRIGHT": 0}
                elif self.spec.family == "NARRATE":
                    self._consume_full_scan_equivalent(
                        pending, candidate, quarter_turns
                    )
                observed_index += 1
            if pending:
                return progress
            progress = boundary
        return progress

    def _guide_block_requirements(self, start: int, end: int) -> list[str]:
        block = list(self.spec.oracle_commands[start:end])
        verbs = [_command_key(command)[0] for command in block]
        has_non_wait = any(verb != "WAIT" for verb in verbs)
        required = []
        for index, command in enumerate(block):
            verb = verbs[index]
            if verb == "WAIT" and has_non_wait:
                continue
            if verb == "WALK" and any(
                later not in {"WAIT", "WALK"} for later in verbs[index + 1 :]
            ):
                continue
            required.append(command)
        return required

    def _consume_full_scan_equivalent(
        self,
        pending: list[str],
        step: tuple[str, bool],
        quarter_turns: dict[str, int],
    ) -> None:
        scan_indices = [
            index
            for index, expected in enumerate(pending)
            if _command_key(expected) == ("TURNAROUND", (), "")
        ][:2]
        verb, args, scalar = _command_key(step[0])
        if len(scan_indices) < 2 or args or scalar:
            quarter_turns.update(TURNLEFT=0, TURNRIGHT=0)
            return
        if verb == "LOOKAROUND":
            for index in reversed(scan_indices):
                pending.pop(index)
            quarter_turns.update(TURNLEFT=0, TURNRIGHT=0)
            return
        if verb not in quarter_turns:
            quarter_turns.update(TURNLEFT=0, TURNRIGHT=0)
            return
        other = "TURNRIGHT" if verb == "TURNLEFT" else "TURNLEFT"
        quarter_turns[other] = 0
        quarter_turns[verb] += 1
        if quarter_turns[verb] < 4:
            return
        for index in reversed(scan_indices):
            pending.pop(index)
        quarter_turns.update(TURNLEFT=0, TURNRIGHT=0)

    def _guide_successful_step_matches(
        self, step: tuple[str, bool], expected: str
    ) -> bool:
        command, inside_verified = step
        if self.spec.family == "NARRATE" and self._process_command_key(
            command
        ) == self._process_command_key(expected):
            return True
        if _surface_placement_args_match(_command_key(command), _command_key(expected)):
            return True
        if (
            self.spec.family == "NARRATE"
            and _putinside_args_match_with_optional_details(
                _command_key(command), _command_key(expected)
            )
        ):
            return True
        if (
            self.spec.family == "ORCHESTRATE"
            and _putinside_args_match_with_optional_spot(
                _command_key(command), _command_key(expected)
            )
        ):
            return True
        return _watch_semantic_commands_equivalent(
            command,
            expected,
            inside_verified=inside_verified,
        )

    def _process_command_key(
        self, command: str
    ) -> tuple[str, tuple[tuple[str, int], ...], str]:
        return _semantic_command_key(command)

    def _process_commands_equivalent(self, observed: str, expected: str) -> bool:
        if self._process_command_key(observed) == self._process_command_key(expected):
            return True
        return _surface_placement_args_match(
            _command_key(observed), _command_key(expected)
        )

    def _process_required_commands_met(self, required: Sequence[str]) -> bool:
        owners: dict[int, int] = {}

        def assign(required_index: int, seen: set[int]) -> bool:
            for observed_index, observed in enumerate(
                self._successful_process_commands
            ):
                if observed_index in seen or not self._process_commands_equivalent(
                    observed, required[required_index]
                ):
                    continue
                seen.add(observed_index)
                previous = owners.get(observed_index)
                if previous is None or assign(previous, seen):
                    owners[observed_index] = required_index
                    return True
            return False

        return all(assign(index, set()) for index in range(len(required)))

    def _process_command_subsequence_holds(self, required: Sequence[str]) -> bool:
        cursor = 0
        for observed in self._successful_process_commands:
            if cursor < len(required) and self._process_commands_equivalent(
                observed, str(required[cursor])
            ):
                cursor += 1
        return cursor == len(required)

    def _process_forbidden_before_violated(self, rule: Mapping[str, object]) -> bool:
        command = str(rule["command"])
        prerequisite = str(rule["prerequisite"])
        prerequisite_seen = False
        for observed in self._successful_process_commands:
            if self._process_commands_equivalent(observed, prerequisite):
                prerequisite_seen = True
            if (
                self._process_commands_equivalent(observed, command)
                and not prerequisite_seen
            ):
                return True
        return False

    def _normalized_process_command(self, command: str) -> str:
        command_key = _semantic_command_key(command)
        if self.spec.family != "NARRATE" or command_key[0] != "WIPE":
            return command
        matches = [
            expected
            for expected in self.spec.oracle_commands
            if (expected_key := _semantic_command_key(expected))[0] == "WIPE"
            and self._narrate_wipe_execution_matches(command_key, expected_key)
        ]
        matched_keys = {_semantic_command_key(expected) for expected in matches}
        return matches[0] if len(matched_keys) == 1 else command

    def _narrate_wipe_execution_matches(
        self,
        observed: tuple[str, tuple[tuple[str, int], ...], str],
        expected: tuple[str, tuple[tuple[str, int], ...], str],
    ) -> bool:
        _, observed_args, observed_scalar = observed
        _, expected_args, expected_scalar = expected
        if (
            observed_scalar
            or expected_scalar
            or len(observed_args) != 2
            or len(expected_args) != 2
        ):
            return False
        expected_tool, effect = expected_args
        if (
            expected_tool[0] not in self.spec.movable_object_keys
            or expected_tool[0] not in self.spec.object_descriptions
            or effect[0] not in self.spec.movable_object_keys
            or effect[0] not in self.spec.object_descriptions
            or not self._narrate_is_broken_glass(effect[0])
        ):
            return False
        if observed_args == expected_args or observed_args == (effect, expected_tool):
            return True
        if observed_args[0] == expected_tool:
            surface = observed_args[1]
        elif observed_args[1] == expected_tool:
            surface = observed_args[0]
        else:
            return False
        relation, separator, support = self.locations.get(effect[0], "").partition(":")
        return bool(
            surface[1] != 0
            and surface[0] in self.spec.object_descriptions
            and surface[0] not in self.spec.movable_object_keys
            and separator
            and relation == "on"
            and support == surface[0]
        )

    def _narrate_is_broken_glass(self, name: str) -> bool:
        aliases = (
            name,
            self.spec.object_descriptions.get(name, ""),
            *self.spec.motor_aliases.get(name, ()),
        )
        words = {word for alias in aliases for word in _grounding_words(alias)}
        return {"broken", "glass"} <= words

    def _matched_oracle_progress(
        self,
        observed: Sequence[tuple[str, bool]],
        *,
        allow_omitted_walks: bool = False,
    ) -> int:
        observed_index = 0
        progress = 0
        for index, command in enumerate(self.spec.oracle_commands):
            verb = _command_key(command)[0]
            if verb == "WAIT":
                if not allow_omitted_walks or progress == index:
                    progress = index + 1
                continue
            if allow_omitted_walks and verb == "WALK":
                continue
            while observed_index < len(observed) and not (
                self._guide_successful_step_matches(observed[observed_index], command)
                if self.spec.family == "NARRATE"
                else self._watch_successful_step_matches(
                    observed[observed_index], command
                )
            ):
                observed_index += 1
            if observed_index == len(observed):
                break
            observed_index += 1
            progress = index + 1
        return progress

    def metrics(self) -> dict[str, object]:
        return {
            "family": self.spec.family,
            "human_object_interface": self.human_object_interface,
            "opening_prelude": self.spec.opening.prelude_say,
            "opening_handoff": self.spec.opening.handoff_say,
            "opening_utterances": [
                {
                    "before_action_index": utterance.before_action_index,
                    "say": utterance.say,
                }
                for utterance in self.spec.opening.utterances
            ],
            "opening_frame_policy": self.spec.opening.frame_policy,
            "opening_recent_intervals": self.spec.opening.recent_intervals,
            "human_receives_frames": self.spec.human_receives_frames,
            "assistant_receives_opening_history": (
                self.spec.assistant_receives_opening_history
            ),
            "authored_tick_budget": self.spec.authored_tick_budget,
            "watch_opening_index": self.opening_index,
            "watch_opening_total": len(self.spec.opening.actions),
            "watch_default_index": self.watch_index,
            "watch_default_total": len(self.spec.default_commands),
            "watch_silent_branch_proven_failure": self.spec.family == "WATCH",
            "waiting_for_assistant": self.waiting_for_assistant,
            "active_instruction": self.active_instruction,
            "successful_actions": self.successful_actions,
            "oracle_action_progress": self._oracle_action_progress(),
            "successful_commands": list(self.successful_commands),
            "required_successful_commands": list(
                self.spec.completion_contract.get("required_successful_commands", ())
            ),
            "forbidden_successful_commands": list(
                self.spec.completion_contract.get("forbidden_successful_commands", ())
            ),
            "required_command_sequences": [
                list(sequence)
                for sequence in self.spec.completion_contract.get(
                    "required_command_sequences", ()
                )
            ],
            "forbidden_before": [
                dict(rule)
                for rule in self.spec.completion_contract.get("forbidden_before", ())
            ],
            "verification_points": [
                dict(point)
                for point in self.spec.completion_contract.get(
                    "verification_points", ()
                )
            ],
            "watch_required_interventions": len(self.spec.watch_interventions),
            "watch_attempted_intervention_ids": sorted(self._watch_attempted_ids),
            "watch_timely_intervention_ids": sorted(self._watch_timely_ids),
            "watch_missed_intervention_ids": sorted(self._watch_missed_ids),
            "process_requirements_met": self.process_requirements_met(),
            **self.counters,
        }

    def verification_results(
        self,
        score: Mapping[str, object],
        observation: Mapping[str, object],
    ) -> list[dict[str, object]]:
        observed = Counter(
            self._process_command_key(command)
            for command in self._successful_process_commands
        )
        results: list[dict[str, object]] = []
        for raw_point in self.spec.completion_contract.get("verification_points", ()):
            point = dict(raw_point)
            kind = str(point.get("kind") or "")
            if kind == "world_state":
                subject = str(point.get("subject") or "")
                if _predicate_name(point.get("pred")) == "unmoved":
                    ok = (
                        subject in self.spec.opening_locations
                        and self.locations.get(subject)
                        == self.spec.opening_locations[subject]
                    )
                else:
                    ok = _world_verification_with_locations_holds(
                        point,
                        score,
                        observation,
                        self.locations,
                    )
            elif kind == "world_state_any":
                ok = any(
                    _world_verification_with_locations_holds(
                        alternative,
                        score,
                        observation,
                        self.locations,
                    )
                    for alternative in point.get("alternatives") or ()
                    if isinstance(alternative, Mapping)
                )
            elif kind == "world_state_not":
                ok = not _world_verification_with_locations_holds(
                    point,
                    score,
                    observation,
                    self.locations,
                )
            elif kind == "exact_relation_count":
                ok = sum(
                    _world_verification_with_locations_holds(
                        {
                            "pred": point.get("pred"),
                            "subject": candidate,
                            "target": point.get("target"),
                        },
                        score,
                        observation,
                        self.locations,
                    )
                    for candidate in point.get("candidates") or ()
                    if isinstance(candidate, str)
                ) == point.get("count")
            elif kind == "exact_count_inside":
                ok = _aggregate_verification_holds(point, score)
            elif kind == "successful_command":
                command = str(point.get("command") or "")
                ok = (
                    any(
                        self._watch_successful_step_matches(step, command)
                        for step in self._successful_watch_steps
                    )
                    if self.spec.family == "WATCH"
                    else observed[self._process_command_key(command)] > 0
                )
            elif kind == "intervention":
                ok = str(point.get("intervention_id") or "") in self._watch_timely_ids
            else:
                ok = False
            results.append({**point, "ok": ok})
        return results

    def _due_watch_intervention(self) -> WatchIntervention | None:
        return next(
            (
                window
                for window in self.spec.watch_interventions
                if window.intervention_id not in self._watch_timely_ids
                and window.intervention_id not in self._watch_missed_ids
                and (
                    (
                        window.earliest_opening_action_index is not None
                        and self.opening_index >= window.earliest_opening_action_index
                    )
                    or (
                        self.opening_index >= len(self.spec.opening.actions)
                        and window.before_action_index == self.watch_index
                    )
                )
            ),
            None,
        )

    def _watch_intervention_state(
        self, window: WatchIntervention
    ) -> Mapping[str, object]:
        return {
            "id": window.intervention_id,
            "before_action_index": window.before_action_index,
            "earliest_opening_action_index": (window.earliest_opening_action_index),
            "dialogue_index": window.dialogue_index,
        }

    def _advance_watch_resume(
        self,
        command: str,
        *,
        succeeded: bool,
        inside_verified: bool,
    ) -> None:
        if not succeeded or self._watch_resume_index is None:
            return
        if self._watch_resume_index >= len(self.spec.default_commands):
            return
        expected = self.spec.default_commands[self._watch_resume_index]
        if self._watch_command_matches(
            command,
            expected,
            succeeded=succeeded,
            inside_verified=inside_verified,
        ):
            self._watch_resume_index += 1

    def _advance_watch_recovery(
        self,
        command: str,
        *,
        succeeded: bool,
        inside_verified: bool,
        observation: Mapping[str, object] | None,
    ) -> bool:
        window = self._watch_pending_intervention
        if not succeeded or window is None:
            return False
        if self._watch_recovery_progress >= len(window.recovery_commands):
            return True
        expected = window.recovery_commands[self._watch_recovery_progress]
        if self._watch_command_matches(
            command,
            expected,
            succeeded=succeeded,
            inside_verified=inside_verified,
        ):
            self._watch_recovery_progress += 1
        else:
            candidate = self._watch_recovery_progress
            while candidate < len(window.recovery_commands) and (
                _watch_position_step_world_holds(
                    window.recovery_commands[candidate], observation
                )
            ):
                candidate += 1
            if candidate < len(
                window.recovery_commands
            ) and self._watch_command_matches(
                command,
                window.recovery_commands[candidate],
                succeeded=succeeded,
                inside_verified=inside_verified,
            ):
                self._watch_recovery_progress = candidate + 1
        complete = self._watch_recovery_progress >= len(window.recovery_commands)
        if complete and window.intervention_id not in self._watch_timely_ids:
            self._watch_timely_ids.add(window.intervention_id)
            self.counters["watch_timely_interventions"] += 1
        return complete

    def _watch_recovery_is_complete(self) -> bool:
        window = self._watch_pending_intervention
        return bool(
            window is not None
            and self._watch_recovery_progress >= len(window.recovery_commands)
        )

    def _finish_active_instruction(
        self, *, watch_recovery_complete: bool = False
    ) -> None:
        self.active_instruction = ""
        self._active_instruction_successful_commands.clear()
        self.waiting_for_assistant = True
        if self.spec.family == "WATCH":
            self._watch_latest_assistant_speech = ""
            window = self._watch_pending_intervention
            if (
                window is not None
                and window.intervention_id not in self._watch_timely_ids
                and window.intervention_id not in self._watch_missed_ids
            ):
                self._watch_missed_ids.add(window.intervention_id)
                self.counters["watch_missed_interventions"] += 1
            if self._watch_override_started and self._watch_resume_index is not None:
                self.watch_index = self._watch_resume_index
            if (
                watch_recovery_complete
                and self._watch_override_started
                and self._watch_resume_index is not None
            ):
                self.counters["watch_overrides"] += 1
            if (
                self._watch_override_started
                and self._watch_resume_opening_index is not None
            ):
                self.opening_index = self._watch_resume_opening_index
            self._watch_resume_opening_index = None
            self._watch_resume_index = None
            self._watch_override_started = False
            self._watch_interrupted_plan = ""
            self._watch_pending_intervention = None
            self._watch_recovery_progress = 0

    def _current_descriptions(
        self, *, locations: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        visible_locations = self.locations if locations is None else locations
        descriptions: dict[str, str] = {}
        for key, base in self.spec.object_descriptions.items():
            location = visible_locations.get(key)
            if location:
                relation, separator, target_key = location.partition(":")
                if not separator:
                    target_key = relation
                    relation_text = "at"
                else:
                    relation_text = {
                        "attached": "attached to",
                        "floor": "on the floor in",
                        "in": "inside",
                        "inside": "inside",
                        "near": "near",
                        "on": "on",
                    }.get(relation.lower(), "at")
                target = self.spec.object_descriptions.get(target_key, target_key)
                descriptions[key] = (
                    f"{base}; currently {relation_text} {target_key} ({target})"
                )
            else:
                descriptions[key] = base
        return descriptions

    def _update_locations(self, command: str) -> None:
        verb, args, _ = _command_key(command)
        names = [name for name, _ in args]
        if verb == "GRAB" and names:
            self.locations.pop(names[0], None)
            if self._human_disclosed_locations is not None:
                self._human_disclosed_locations.pop(names[0], None)
        elif (
            verb in {"PUT", "PUTBACK", "PUTIN", "PUTINSIDE", "PUTON"}
            and len(names) >= 2
        ):
            relation = "inside" if verb in {"PUTIN", "PUTINSIDE"} else "on"
            self.locations[names[0]] = f"{relation}:{names[1]}"
            if self._human_disclosed_locations is not None:
                self._human_disclosed_locations[names[0]] = f"{relation}:{names[1]}"
        elif verb in {"ATTACH", "ASSEMBLE", "INSTALL"} and len(names) >= 3:
            self.locations[names[0]] = f"attached:{names[2]}"
            if self._human_disclosed_locations is not None:
                self._human_disclosed_locations[names[0]] = f"attached:{names[2]}"


def _interaction_opening(
    story: Mapping[str, object],
    catalog_row: Mapping[str, object] | None,
    task_id: str,
) -> Mapping[str, object]:
    interaction = story.get("interaction")
    if isinstance(interaction, Mapping) and isinstance(
        interaction.get("opening"), Mapping
    ):
        return {**dict(interaction), **dict(interaction["opening"])}
    if catalog_row is not None:
        return catalog_row
    raise InteractionSpecError(
        f"{task_id}: no task-specific interaction opening in TaskPack or opening catalog"
    )


def _opening_utterances(
    value: object,
    action_count: int,
    task_id: str,
    *,
    allow_after_last: bool = False,
) -> tuple[OpeningUtterance, ...]:
    rows = _mapping_sequence(value, f"{task_id}.opening.utterances")
    utterances: list[OpeningUtterance] = []
    previous_index = -1
    for row_index, row in enumerate(rows):
        path = f"{task_id}.opening.utterances[{row_index}]"
        raw_index = row.get("before_action_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise InteractionSpecError(f"{path}.before_action_index must be an integer")
        upper_bound = action_count if allow_after_last else action_count - 1
        if not 0 <= raw_index <= upper_bound:
            raise InteractionSpecError(
                f"{path}.before_action_index {raw_index} is outside the "
                f"{action_count}-action sequence"
            )
        if raw_index < previous_index:
            raise InteractionSpecError(
                f"{task_id}.opening.utterances must be ordered by action index"
            )
        utterances.append(
            OpeningUtterance(
                before_action_index=raw_index,
                say=_required_text(row.get("say"), f"{path}.say"),
            )
        )
        previous_index = raw_index
    return tuple(utterances)


def _opening_frame_selection(
    authored: Mapping[str, object], task_id: str
) -> tuple[str, int]:
    value = authored.get("frame_selection")
    if value is None:
        return "evidence", 2
    selection = _required_mapping(value, f"{task_id}.opening.frame_selection")
    policy = str(selection.get("policy") or "evidence").strip()
    if policy not in {"evidence", "all_actions_budgeted"}:
        raise InteractionSpecError(
            f"{task_id}.opening.frame_selection.policy has unsupported value {policy!r}"
        )
    recent = _positive_int(
        selection.get("recent_intervals", 2),
        f"{task_id}.opening.frame_selection.recent_intervals",
    )
    return policy, recent


def _completion_contract(
    authored: Mapping[str, object], task_id: str
) -> dict[str, object]:
    value = authored.get("completion_contract")
    if value is None:
        return {"requires_assistant_done": True}
    contract = _required_mapping(value, f"{task_id}.completion_contract")
    requires_done = contract.get("requires_assistant_done", True)
    if not isinstance(requires_done, bool):
        raise InteractionSpecError(
            f"{task_id}.completion_contract.requires_assistant_done must be boolean"
        )
    verification_points_define_success = contract.get(
        "verification_points_define_success", False
    )
    if not isinstance(verification_points_define_success, bool):
        raise InteractionSpecError(
            f"{task_id}.completion_contract.verification_points_define_success "
            "must be boolean"
        )
    raw_required_commands = contract.get("required_successful_commands")
    required_commands = (
        ()
        if raw_required_commands is None
        else _string_sequence(
            raw_required_commands,
            f"{task_id}.completion_contract.required_successful_commands",
        )
    )
    for command in required_commands:
        _command_key(command)
    forbidden_commands = _optional_command_sequence(
        contract.get("forbidden_successful_commands"),
        f"{task_id}.completion_contract.forbidden_successful_commands",
    )
    required_sequences = _command_sequences(
        contract.get("required_command_sequences"),
        f"{task_id}.completion_contract.required_command_sequences",
    )
    forbidden_before = _forbidden_before_rules(
        contract.get("forbidden_before"),
        f"{task_id}.completion_contract.forbidden_before",
    )
    verification_points = _verification_points(
        contract.get("verification_points"), task_id
    )
    return {
        **contract,
        "requires_assistant_done": requires_done,
        "verification_points_define_success": verification_points_define_success,
        "required_successful_commands": required_commands,
        "forbidden_successful_commands": forbidden_commands,
        "required_command_sequences": required_sequences,
        "forbidden_before": forbidden_before,
        "verification_points": verification_points,
    }


def _optional_command_sequence(value: object, path: str) -> tuple[str, ...]:
    commands = () if value is None else _string_sequence(value, path)
    for command in commands:
        _command_key(command)
    return commands


def _command_sequences(value: object, path: str) -> tuple[tuple[str, ...], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InteractionSpecError(f"{path} must be an array")
    sequences: list[tuple[str, ...]] = []
    for index, raw in enumerate(value):
        commands = _string_sequence(raw, f"{path}[{index}]")
        if not commands:
            raise InteractionSpecError(f"{path}[{index}] must not be empty")
        for command in commands:
            _command_key(command)
        sequences.append(commands)
    return tuple(sequences)


def _forbidden_before_rules(value: object, path: str) -> tuple[Mapping[str, str], ...]:
    rows = _mapping_sequence(value, path)
    rules: list[Mapping[str, str]] = []
    for index, row in enumerate(rows):
        command = _required_text(row.get("command"), f"{path}[{index}].command")
        prerequisite = _required_text(
            row.get("prerequisite"), f"{path}[{index}].prerequisite"
        )
        _command_key(command)
        _command_key(prerequisite)
        rules.append({"command": command, "prerequisite": prerequisite})
    return tuple(rules)


def _verification_points(
    value: object, task_id: str
) -> tuple[Mapping[str, object], ...]:
    rows = _mapping_sequence(
        value, f"{task_id}.completion_contract.verification_points"
    )
    points: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        path = f"{task_id}.completion_contract.verification_points[{index}]"
        point_id = _required_text(row.get("id"), f"{path}.id")
        kind = _required_text(row.get("kind"), f"{path}.kind")
        if point_id in seen_ids:
            raise InteractionSpecError(
                f"{task_id}: duplicate verification point id {point_id!r}"
            )
        if kind == "world_state":
            _required_text(row.get("pred"), f"{path}.pred")
            _required_text(row.get("subject"), f"{path}.subject")
        elif kind == "world_state_any":
            alternatives = _mapping_sequence(
                row.get("alternatives"), f"{path}.alternatives"
            )
            if not alternatives:
                raise InteractionSpecError(f"{path}.alternatives must not be empty")
            for alternative_index, alternative in enumerate(alternatives):
                alternative_path = f"{path}.alternatives[{alternative_index}]"
                _required_text(alternative.get("pred"), f"{alternative_path}.pred")
                _required_text(
                    alternative.get("subject"), f"{alternative_path}.subject"
                )
        elif kind == "world_state_not":
            _required_text(row.get("pred"), f"{path}.pred")
            _required_text(row.get("subject"), f"{path}.subject")
        elif kind == "exact_relation_count":
            _required_text(row.get("pred"), f"{path}.pred")
            candidates = _string_sequence(row.get("candidates"), f"{path}.candidates")
            _required_text(row.get("target"), f"{path}.target")
            count = row.get("count")
            if len(candidates) != len(set(candidates)):
                raise InteractionSpecError(f"{path}.candidates must be unique")
            if (
                not candidates
                or not isinstance(count, int)
                or isinstance(count, bool)
                or not 0 <= count <= len(candidates)
            ):
                raise InteractionSpecError(
                    f"{path} must have a non-empty candidate set and valid count"
                )
        elif kind == "exact_count_inside":
            label = _required_text(row.get("label"), f"{path}.label")
            candidates = _string_sequence(row.get("candidates"), f"{path}.candidates")
            _required_text(row.get("target"), f"{path}.target")
            count = row.get("count")
            if len(candidates) != len(set(candidates)):
                raise InteractionSpecError(f"{path}.candidates must be unique")
            if (
                not candidates
                or not isinstance(count, int)
                or isinstance(count, bool)
                or not 1 <= count <= len(candidates)
            ):
                raise InteractionSpecError(
                    f"{path} must have a non-empty candidate set and executable count"
                )
            if label != point_id:
                raise InteractionSpecError(
                    f"{path}.label must equal its verification point id"
                )
        elif kind == "successful_command":
            _command_key(_required_text(row.get("command"), f"{path}.command"))
        elif kind == "intervention":
            _required_text(row.get("intervention_id"), f"{path}.intervention_id")
        else:
            raise InteractionSpecError(f"{path}.kind has unsupported value {kind!r}")
        points.append(copy.deepcopy(dict(row)))
        seen_ids.add(point_id)
    return tuple(points)


def _required_spoken_cues(
    authored: Mapping[str, object],
    speech: Sequence[str],
    task_id: str,
) -> tuple[SpokenCue, ...]:
    contract = _required_mapping(
        authored.get("information_contract"),
        f"{task_id}.opening.information_contract",
    )
    rows = _mapping_sequence(
        contract.get("spoken_cues"),
        f"{task_id}.opening.information_contract.spoken_cues",
    )
    if not rows:
        raise InteractionSpecError(
            f"{task_id}: information_contract.spoken_cues must not be empty"
        )
    actual_speech = _normalize_spoken_text(" ".join(speech))
    seen_ids: set[str] = set()
    cues: list[SpokenCue] = []
    for index, row in enumerate(rows):
        path = f"{task_id}.opening.information_contract.spoken_cues[{index}]"
        cue_id = _required_text(row.get("id"), f"{path}.id")
        if cue_id in seen_ids:
            raise InteractionSpecError(f"{task_id}: duplicate spoken cue id {cue_id!r}")
        phrases = _string_sequence(
            row.get("required_phrases"), f"{path}.required_phrases"
        )
        if not phrases:
            raise InteractionSpecError(f"{path}.required_phrases must not be empty")
        missing = [
            phrase
            for phrase in phrases
            if _normalize_spoken_text(phrase) not in actual_speech
        ]
        if missing:
            raise InteractionSpecError(
                f"{task_id}: opening speech does not state cue {cue_id!r}; "
                f"missing phrases {missing!r}"
            )
        seen_ids.add(cue_id)
        cues.append(SpokenCue(cue_id=cue_id, required_phrases=phrases))
    return tuple(cues)


def _world_legend(
    executable: Mapping[str, object],
    story: Mapping[str, object],
    opening_actions: Sequence[str],
    *,
    hide_movable_rooms: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    world = _required_mapping(executable.get("world"), "$.executable_spec.world")
    descriptions, locations = _base_world_legend(
        world, hide_movable_rooms=hide_movable_rooms
    )
    _apply_live_semantic_descriptions(descriptions, story)
    _apply_opening_locations(locations, opening_actions)
    return descriptions, locations


def _world_movable_keys(executable: Mapping[str, object]) -> tuple[str, ...]:
    world = _required_mapping(executable.get("world"), "$.executable_spec.world")
    return tuple(
        sorted(
            str(key)
            for field in ("objects", "parts")
            for key, value in (
                world.get(field).items()
                if isinstance(world.get(field), Mapping)
                else ()
            )
            if isinstance(value, Mapping)
        )
    )


def _world_verification_holds(
    point: Mapping[str, object],
    score: Mapping[str, object],
    observation: Mapping[str, object],
) -> bool:
    pred = _predicate_name(point.get("pred"))
    subject = str(point.get("subject") or "")
    target_value = point.get("target")
    target = None if target_value is None else str(target_value)
    scored = _matching_world_score_result(point, score)
    if scored is not None:
        return scored
    if pred == "held":
        return subject in {str(value) for value in observation.get("held") or ()}
    if pred == "at" and subject == "human":
        return target in {str(value) for value in observation.get("at") or ()}
    relations = {
        (
            _predicate_name(parts[0]),
            str(parts[1]),
            None if parts[2] is None else str(parts[2]),
        )
        for row in observation.get("relations") or ()
        if isinstance(row, Sequence)
        and not isinstance(row, (str, bytes))
        and len(parts := list(row)) == 3
    }
    if pred == "closed":
        return ("open", subject, None) not in relations
    if pred == "off":
        return ("running", subject, None) not in relations
    if pred in _UNARY_WORLD_PREDICATES:
        return (pred, subject, None) in relations
    return target is not None and (pred, subject, target) in relations


def _matching_world_score_result(
    point: Mapping[str, object], score: Mapping[str, object]
) -> bool | None:
    pred = _predicate_name(point.get("pred"))
    subject = str(point.get("subject") or "")
    target_value = point.get("target")
    target = None if target_value is None else str(target_value)
    for row in (*list(score.get("sub_goals") or ()), *list(score.get("goals") or ())):
        if not isinstance(row, Mapping):
            continue
        row_target = row.get("target")
        if (
            _predicate_name(row.get("pred")) == pred
            and str(row.get("subject") or "") == subject
            and (
                pred in _UNARY_WORLD_PREDICATES
                or (None if row_target is None else str(row_target)) == target
            )
        ):
            return bool(row.get("ok"))
    return None


def _world_verification_with_locations_holds(
    point: Mapping[str, object],
    score: Mapping[str, object],
    observation: Mapping[str, object],
    locations: Mapping[str, str],
) -> bool:
    scored = _matching_world_score_result(point, score)
    if scored is not None:
        return scored
    pred = _predicate_name(point.get("pred"))
    subject = str(point.get("subject") or "")
    target = str(point.get("target") or "")
    if pred in {"inside", "on"} and locations.get(subject) == f"{pred}:{target}":
        return True
    return _world_verification_holds(point, score, observation)


def _aggregate_verification_holds(
    point: Mapping[str, object], score: Mapping[str, object]
) -> bool:
    expected_candidates = {
        str(value) for value in point.get("candidates") or () if isinstance(value, str)
    }
    for row in score.get("aggregates") or ():
        if not isinstance(row, Mapping):
            continue
        actual_candidates = {
            str(value)
            for value in row.get("candidates") or ()
            if isinstance(value, str)
        }
        if (
            str(row.get("kind") or "") == "exact_count_inside"
            and str(row.get("label") or "") == str(point.get("label") or "")
            and actual_candidates == expected_candidates
            and str(row.get("target") or "") == str(point.get("target") or "")
            and row.get("count") == point.get("count")
        ):
            return bool(row.get("ok"))
    return False


def _predicate_name(value: object) -> str:
    return str(getattr(value, "name", value) or "").lower().removeprefix("pred.")


def _base_world_legend(
    world: Mapping[str, object],
    *,
    hide_movable_rooms: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    descriptions: dict[str, str] = {}
    locations: dict[str, str] = {}
    for field in ("objects", "containers", "fixtures", "parts"):
        values = world.get(field)
        if not isinstance(values, Mapping):
            continue
        for raw_key, raw_value in values.items():
            if not isinstance(raw_value, Mapping):
                continue
            key = str(raw_key)
            # A movable's authored room can itself answer a FIND "which room" query.
            descriptions[key] = _describe_entity(
                raw_value,
                include_room=not (hide_movable_rooms and field in {"objects", "parts"}),
            )
            place = str(raw_value.get("place") or "")
            if ":" in place:
                locations[key] = place
    return descriptions, locations


def _apply_live_semantic_descriptions(
    descriptions: dict[str, str], story: Mapping[str, object]
) -> None:
    for row in _mapping_sequence(story.get("evidence"), "$.story.evidence"):
        if str(row.get("channel") or "").lower() != "live":
            continue
        match = _ACTS_AS_RE.search(str(row.get("statement") or "").strip())
        if match is None:
            continue
        key, semantic_description = match.groups()
        if key in descriptions:
            descriptions[key] = (
                f"{semantic_description}; visually rendered as {descriptions[key]}"
            )


def _apply_opening_locations(
    locations: dict[str, str], opening_actions: Sequence[str]
) -> None:
    for command in opening_actions:
        verb, args, _ = _command_key(command)
        names = [name for name, _ in args]
        if verb == "GRAB" and names:
            locations.pop(names[0], None)
        elif (
            verb in {"PUT", "PUTBACK", "PUTIN", "PUTINSIDE", "PUTON"}
            and len(names) >= 2
        ):
            relation = "inside" if verb in {"PUTIN", "PUTINSIDE"} else "on"
            locations[names[0]] = f"{relation}:{names[1]}"
        elif verb in {"ATTACH", "ASSEMBLE", "INSTALL"} and len(names) >= 3:
            locations[names[0]] = f"attached:{names[2]}"


def _describe_entity(value: Mapping[str, object], *, include_room: bool = True) -> str:
    bind = value.get("bind")
    bind = bind if isinstance(bind, Mapping) else {}
    category = str(value.get("cat") or bind.get("cat") or "object").replace("_", " ")
    details = [category]
    tint = str(value.get("tint") or "").strip()
    if tint:
        details.insert(0, tint)
    decal = str(value.get("decal") or "").strip()
    if decal:
        details.append(f"labelled {decal!r}")
    room = (
        str(value.get("room") or "").replace("_", " ").strip() if include_room else ""
    )
    if room:
        details.append(f"in {room}")
    return ", ".join(details)


def _human_command_qualifiers(
    authored: Mapping[str, object], task_id: str
) -> tuple[HumanCommandQualifier, ...]:
    raw = authored.get("human_command_qualifiers")
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise InteractionSpecError(
            f"{task_id}.human_command_qualifiers must be an object"
        )
    qualifiers: list[HumanCommandQualifier] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...], str]] = set()
    for raw_command, raw_qualifier in raw.items():
        path = f"{task_id}.human_command_qualifiers[{raw_command!r}]"
        match_command = _required_text(raw_command, path)
        command_key = _semantic_command_key(match_command)
        verb, args, scalar = command_key
        if verb not in {"PUTINSIDE", "PUTON"} or len(args) != 2 or scalar:
            raise InteractionSpecError(
                f"{path} must key a generic two-argument Put or PutInside command"
            )
        if command_key in seen:
            raise InteractionSpecError(f"{path} duplicates another command matcher")
        seen.add(command_key)
        qualifier = _required_mapping(raw_qualifier, path)
        unexpected = set(qualifier) - {"space", "place_spot"}
        if unexpected:
            raise InteractionSpecError(
                f"{path} has unsupported fields: {sorted(unexpected)}"
            )
        raw_space = qualifier.get("space")
        if verb == "PUTINSIDE":
            space = _required_text(raw_space, f"{path}.space")
        else:
            if raw_space is not None:
                raise InteractionSpecError(
                    f"{path}.space is supported only for PutInside commands"
                )
            space = None
        place_spot = _normalized_place_spot(
            {"verb": verb, "place_spot": qualifier.get("place_spot")},
            path,
        )
        if place_spot is None:
            raise InteractionSpecError(f"{path}.place_spot is required")
        rendered_args = [args[0][0], args[1][0]]
        rendered_indices = [args[0][1], args[1][1]]
        if space is not None:
            rendered_args.append(space)
            rendered_indices.append(0)
        qualified_command = _render_command(
            {
                "verb": "PutInside" if verb == "PUTINSIDE" else "Put",
                "args": rendered_args,
                "indices": rendered_indices,
                "place_spot": list(place_spot),
            },
            path,
        )
        qualifiers.append(
            HumanCommandQualifier(
                match_command=match_command,
                qualified_command=qualified_command,
                space=space,
                place_spot=place_spot,
            )
        )
    return tuple(qualifiers)


def _render_command(value: Mapping[str, object], path: str) -> str:
    verb = _required_text(value.get("verb"), f"{path}.verb")
    raw_args = value.get("args")
    if isinstance(raw_args, (str, bytes)) or not isinstance(raw_args, Sequence):
        raise InteractionSpecError(f"{path}.args must be an array")
    args = [str(item) for item in raw_args]
    raw_indices = value.get("indices")
    if raw_indices is None:
        attach = verb.upper() in {"ASSEMBLE", "ATTACH", "INSTALL"}
        container_space = verb.upper() in {"PUTIN", "PUTINSIDE"}
        indices = [
            0 if (attach and index in {1, 3}) or (container_space and index == 2) else 1
            for index in range(len(args))
        ]
    else:
        if isinstance(raw_indices, (str, bytes)) or not isinstance(
            raw_indices, Sequence
        ):
            raise InteractionSpecError(f"{path}.indices must be an array")
        indices = [int(item) for item in raw_indices]
        if len(indices) != len(args):
            raise InteractionSpecError(f"{path}.indices must align with args")
    rendered = f"[{verb}]" + "".join(
        f" <{argument}> ({indices[index]})" for index, argument in enumerate(args)
    )
    scalar = value.get("scalar_arg")
    if scalar is not None:
        if args:
            raise InteractionSpecError(f"{path}.scalar_arg cannot accompany args")
        rendered += f" {_required_text(scalar, f'{path}.scalar_arg')}"
    place_spot = _normalized_place_spot(value, path)
    if place_spot is not None:
        rendered += f" @place_spot={place_spot[0]!r}:{place_spot[1]!r}"
    _command_key(rendered)
    return rendered


def _normalized_place_spot(
    value: Mapping[str, object], path: str
) -> tuple[float, float] | None:
    raw = value.get("place_spot")
    if raw is None:
        return None
    verb = str(value.get("verb") or "").upper()
    if verb not in {"PUT", "PUTBACK", "PUTON", "PUTIN", "PUTINSIDE", "INSERT"}:
        raise InteractionSpecError(
            f"{path}.place_spot is allowed only on Put/PutOn/PutInside commands"
        )
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise InteractionSpecError(f"{path}.place_spot must be a numeric [u, v] pair")
    values = list(raw)
    if len(values) != 2 or any(
        not isinstance(item, (int, float)) or isinstance(item, bool) for item in values
    ):
        raise InteractionSpecError(f"{path}.place_spot must be a numeric [u, v] pair")
    u, v = float(values[0]), float(values[1])
    if (
        not math.isfinite(u)
        or not math.isfinite(v)
        or not -1.0 <= u <= 1.0
        or not 0.0 <= v <= 1.0
    ):
        raise InteractionSpecError(
            f"{path}.place_spot requires finite -1 <= u <= 1 and 0 <= v <= 1"
        )
    return u, v


def _command_key(
    command: str,
) -> tuple[str, tuple[tuple[str, int], ...], str]:
    match = _COMMAND_RE.fullmatch(command)
    if match is None:
        raise InteractionSpecError(f"invalid simulator command {command!r}")
    verb = match.group(1).strip().upper()
    tail = match.group(2).strip()
    args = tuple((name, int(index)) for name, index in _ARG_RE.findall(tail))
    scalar = _ARG_RE.sub("", tail).strip()
    if "<" in scalar or ">" in scalar or "(" in scalar or ")" in scalar:
        raise InteractionSpecError(f"invalid simulator command {command!r}")
    return verb, args, scalar


def _semantic_command_key(
    command: str,
) -> tuple[str, tuple[tuple[str, int], ...], str]:
    verb, args, scalar = _command_key(command)
    if verb in {"PUT", "PUTON"} and len(args) == 2 and not scalar:
        verb = "PUTON"
    return verb, args, scalar


def _surface_placement_args_match(
    observed: tuple[str, tuple[tuple[str, int], ...], str],
    expected: tuple[str, tuple[tuple[str, int], ...], str],
) -> bool:
    observed_verb, observed_args, observed_scalar = observed
    expected_verb, expected_args, expected_scalar = expected
    return (
        observed_verb in {"PUT", "PUTON"}
        and expected_verb in {"PUT", "PUTON"}
        and observed_args == expected_args
        and (
            not observed_scalar
            or not expected_scalar
            or observed_scalar == expected_scalar
        )
    )


def _putinside_args_match_with_optional_spot(
    observed: tuple[str, tuple[tuple[str, int], ...], str],
    expected: tuple[str, tuple[tuple[str, int], ...], str],
) -> bool:
    observed_verb, observed_args, observed_scalar = observed
    expected_verb, expected_args, expected_scalar = expected
    if (
        observed_verb != "PUTINSIDE"
        or expected_verb != "PUTINSIDE"
        or observed_args != expected_args
    ):
        return False
    if observed_scalar and expected_scalar:
        return observed_scalar == expected_scalar
    optional_scalar = observed_scalar or expected_scalar
    return optional_scalar.startswith("@place_spot=")


def _putinside_args_match_with_optional_details(
    observed: tuple[str, tuple[tuple[str, int], ...], str],
    expected: tuple[str, tuple[tuple[str, int], ...], str],
) -> bool:
    observed_verb, observed_args, observed_scalar = observed
    expected_verb, expected_args, expected_scalar = expected
    if (
        observed_verb != "PUTINSIDE"
        or expected_verb != "PUTINSIDE"
        or not _putinside_parent_space_args_match(observed_args, expected_args)
    ):
        return False
    if observed_scalar == expected_scalar:
        return True
    if observed_scalar and expected_scalar:
        return False
    return (observed_scalar or expected_scalar).startswith("@place_spot=")


def _putinside_parent_space_args_match(
    observed_args: tuple[tuple[str, int], ...],
    expected_args: tuple[tuple[str, int], ...],
) -> bool:
    if (
        len(observed_args) not in {2, 3}
        or len(expected_args) not in {2, 3}
        or observed_args[:2] != expected_args[:2]
    ):
        return False
    if observed_args == expected_args:
        return True
    if len(observed_args) == len(expected_args):
        return False
    parent = observed_args[1][0]
    child = (observed_args if len(observed_args) == 3 else expected_args)[2]
    child_name, child_index = child
    return bool(
        child_index >= 0
        and re.fullmatch(rf"{re.escape(parent)}_space[1-9][0-9]*", child_name)
    )


def _watch_putinside_parent_space_equivalent(
    observed: tuple[str, tuple[tuple[str, int], ...], str],
    expected: tuple[str, tuple[tuple[str, int], ...], str],
) -> bool:
    observed_verb, observed_args, observed_scalar = observed
    expected_verb, expected_args, expected_scalar = expected
    if (
        observed_verb != "PUTINSIDE"
        or expected_verb != "PUTINSIDE"
        or observed_scalar
        or expected_scalar
        or len(observed_args) == len(expected_args)
    ):
        return False
    return _putinside_parent_space_args_match(observed_args, expected_args)


def _watch_semantic_commands_equivalent(
    observed: str,
    expected: str,
    *,
    inside_verified: bool,
) -> bool:
    if _semantic_command_key(observed) == _semantic_command_key(expected):
        return True
    return inside_verified and _watch_putinside_parent_space_equivalent(
        _command_key(observed), _command_key(expected)
    )


def _watch_inside_world_holds(command: str, observation: Mapping[str, object]) -> bool:
    verb, args, scalar = _command_key(command)
    if verb != "PUTINSIDE" or len(args) not in {2, 3} or scalar:
        return False
    subject = args[0][0]
    parent = args[1][0]
    score = observation.get("score")
    score = score if isinstance(score, Mapping) else {}
    return _world_verification_holds(
        {"pred": "inside", "subject": subject, "target": parent},
        score,
        observation,
    )


def _watch_position_step_world_holds(
    command: str, observation: Mapping[str, object] | None
) -> bool:
    if observation is None:
        return False
    verb, args, scalar = _command_key(command)
    if verb != "WALK" or len(args) != 1 or scalar:
        return False
    target, _ = args[0]
    return target in {str(value) for value in observation.get("at") or ()}


def _first_difference(left: Sequence[str], right: Sequence[str]) -> int | None:
    for index, (left_command, right_command) in enumerate(zip(left, right)):
        if _command_key(left_command) != _command_key(right_command):
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _command_subsequence_holds(
    observed: Sequence[tuple[str, tuple[tuple[str, int], ...], str]],
    required: Sequence[str],
    *,
    key_fn=_command_key,
) -> bool:
    cursor = 0
    required_keys = [key_fn(command) for command in required]
    for command in observed:
        if cursor < len(required_keys) and command == required_keys[cursor]:
            cursor += 1
    return cursor == len(required_keys)


def _forbidden_before_violated(
    observed: Sequence[tuple[str, tuple[tuple[str, int], ...], str]],
    rule: Mapping[str, object],
    *,
    key_fn=_command_key,
) -> bool:
    command = key_fn(str(rule["command"]))
    prerequisite = key_fn(str(rule["prerequisite"]))
    prerequisite_seen = False
    for current in observed:
        if current == prerequisite:
            prerequisite_seen = True
        if current == command and not prerequisite_seen:
            return True
    return False


def _drop_shared_redundant_opening_walk(
    opening_actions: Sequence[str],
    default_commands: tuple[str, ...],
    oracle_commands: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Remove a shared first walk when setup already leaves the person at that target."""
    if not opening_actions or not default_commands or not oracle_commands:
        return default_commands, oracle_commands
    _, opening_args, _ = _command_key(opening_actions[-1])
    if not opening_args:
        return default_commands, oracle_commands
    opening_anchor = opening_args[-1]
    default_verb, default_args, default_scalar = _command_key(default_commands[0])
    oracle_verb, oracle_args, oracle_scalar = _command_key(oracle_commands[0])
    if (
        default_verb != "WALK"
        or oracle_verb != "WALK"
        or default_scalar
        or oracle_scalar
        or len(default_args) != 1
        or default_args != oracle_args
        or default_args[0] != opening_anchor
    ):
        return default_commands, oracle_commands
    return default_commands[1:], oracle_commands[1:]


def _watch_observable_trigger(
    default_commands: Sequence[str], oracle_commands: Sequence[str]
) -> int | None:
    """Return the last safe assistant turn before the first divergent action."""
    return _first_difference(default_commands, oracle_commands)


def _watch_interventions(
    authored: Mapping[str, object],
    opening_commands: Sequence[str],
    default_commands: Sequence[str],
    oracle_commands: Sequence[str],
    dialogue_count: int,
    task_id: str,
) -> tuple[WatchIntervention, ...]:
    raw_rows = authored.get("watch_interventions")
    if raw_rows is None:
        return _legacy_watch_interventions(
            opening_commands, default_commands, oracle_commands, dialogue_count
        )
    rows = _mapping_sequence(raw_rows, f"{task_id}.watch_interventions")
    if not default_commands and rows:
        raise InteractionSpecError(
            f"{task_id}.watch_interventions requires a WATCH default plan"
        )
    interventions: list[WatchIntervention] = []
    seen_ids: set[str] = set()
    previous_index = -1
    for row_index, row in enumerate(rows):
        path = f"{task_id}.watch_interventions[{row_index}]"
        intervention = _parse_watch_intervention(
            row,
            row_index=row_index,
            path=path,
            opening_count=len(opening_commands),
            default_count=len(default_commands),
            oracle_commands=oracle_commands,
            dialogue_count=dialogue_count,
        )
        if intervention.intervention_id in seen_ids:
            raise InteractionSpecError(
                f"{task_id}: duplicate WATCH intervention id "
                f"{intervention.intervention_id!r}"
            )
        if intervention.before_action_index <= previous_index:
            raise InteractionSpecError(
                f"{task_id}.watch_interventions must use strictly increasing action indices"
            )
        interventions.append(intervention)
        previous_index = intervention.before_action_index
        seen_ids.add(intervention.intervention_id)
    return tuple(interventions)


def _watch_reference_dialogue(
    authored: Mapping[str, object],
    dialogue: tuple[str, ...],
    interventions: Sequence[WatchIntervention],
    task_id: str,
) -> tuple[str, ...]:
    raw_rows = authored.get("watch_interventions")
    if raw_rows is None:
        return dialogue
    rows = _mapping_sequence(raw_rows, f"{task_id}.watch_interventions")
    overridden = list(dialogue)
    for row_index, (row, intervention) in enumerate(zip(rows, interventions)):
        if "assistant_say" not in row:
            continue
        overridden[intervention.dialogue_index] = _required_text(
            row.get("assistant_say"),
            f"{task_id}.watch_interventions[{row_index}].assistant_say",
        )
    return tuple(overridden)


def _legacy_watch_interventions(
    opening_commands: Sequence[str],
    default_commands: Sequence[str],
    oracle_commands: Sequence[str],
    dialogue_count: int,
) -> tuple[WatchIntervention, ...]:
    if not default_commands:
        return ()
    trigger = _watch_observable_trigger(default_commands, oracle_commands)
    if trigger is None or dialogue_count < 1:
        return ()
    return (
        WatchIntervention(
            intervention_id="watch_intervention_0",
            before_action_index=trigger,
            earliest_opening_action_index=None,
            dialogue_index=0,
            resume_opening_action_index=len(opening_commands),
            resume_action_index=len(default_commands),
            recovery_commands=tuple(oracle_commands),
        ),
    )


def _parse_watch_intervention(
    row: Mapping[str, object],
    *,
    row_index: int,
    path: str,
    opening_count: int,
    default_count: int,
    oracle_commands: Sequence[str],
    dialogue_count: int,
) -> WatchIntervention:
    intervention_id = _required_text(row.get("id"), f"{path}.id")
    before = _required_int(
        row.get("before_action_index"), f"{path}.before_action_index"
    )
    dialogue_index = _required_int(
        row.get("dialogue_index", row_index), f"{path}.dialogue_index"
    )
    resume_opening = _required_int(
        row.get("resume_opening_action_index", opening_count),
        f"{path}.resume_opening_action_index",
    )
    resume = _required_int(
        row.get("resume_action_index", default_count), f"{path}.resume_action_index"
    )
    earliest = _optional_bounded_int(
        row.get("earliest_opening_action_index"),
        f"{path}.earliest_opening_action_index",
        opening_count,
    )
    _validate_watch_intervention_indices(
        path,
        before,
        resume,
        resume_opening,
        dialogue_index,
        default_count,
        opening_count,
        dialogue_count,
    )
    raw_recovery = row.get("recovery_commands")
    recovery = (
        tuple(oracle_commands)
        if raw_recovery is None
        else _string_sequence(raw_recovery, f"{path}.recovery_commands")
    )
    if not recovery:
        raise InteractionSpecError(f"{path}.recovery_commands must not be empty")
    for command in recovery:
        _command_key(command)
    return WatchIntervention(
        intervention_id,
        before,
        earliest,
        dialogue_index,
        resume_opening,
        resume,
        recovery,
    )


def _required_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InteractionSpecError(f"{path} must be an integer")
    return value


def _optional_bounded_int(value: object, path: str, maximum: int) -> int | None:
    if value is None:
        return None
    parsed = _required_int(value, path)
    if not 0 <= parsed <= maximum:
        raise InteractionSpecError(f"{path} must be between 0 and {maximum}")
    return parsed


def _validate_watch_intervention_indices(
    path: str,
    before: int,
    resume: int,
    resume_opening: int,
    dialogue_index: int,
    default_count: int,
    opening_count: int,
    dialogue_count: int,
) -> None:
    if not 0 <= before < default_count:
        raise InteractionSpecError(
            f"{path}.before_action_index {before} is outside the "
            f"{default_count}-action default plan"
        )
    if not before < resume <= default_count:
        raise InteractionSpecError(
            f"{path}.resume_action_index must be after the intervention and no "
            f"greater than {default_count}"
        )
    if not 0 <= resume_opening <= opening_count:
        raise InteractionSpecError(
            f"{path}.resume_opening_action_index must be between 0 and {opening_count}"
        )
    if not 0 <= dialogue_index < dialogue_count:
        raise InteractionSpecError(
            f"{path}.dialogue_index {dialogue_index} is outside the "
            f"{dialogue_count}-utterance reference dialogue"
        )


def _validate_watch_branches(
    task_id: str,
    executable: Mapping[str, object],
    opening_commands: Sequence[str],
    default_commands: Sequence[str],
    oracle_commands: Sequence[str],
    answer_key: Sequence[Mapping[str, object]],
) -> None:
    if not answer_key:
        raise InteractionSpecError(
            f"{task_id}: WATCH requires a non-empty answer_key to prove the silent branch fails"
        )
    after_opening = _symbolic_state(executable)
    _apply_symbolic_commands(after_opening, opening_commands)
    after_default = copy.deepcopy(after_opening)
    after_oracle = copy.deepcopy(after_opening)
    _apply_symbolic_commands(after_default, default_commands)
    _apply_symbolic_commands(after_oracle, oracle_commands)
    default_results = [_symbolic_goal_holds(row, after_default) for row in answer_key]
    oracle_results = [_symbolic_goal_holds(row, after_oracle) for row in answer_key]
    if any(result is None for result in oracle_results):
        unsupported = sorted(
            {
                str(row.get("pred") or "")
                for row, result in zip(answer_key, oracle_results)
                if result is None
            }
        )
        raise InteractionSpecError(
            f"{task_id}: WATCH cannot prove the silent branch for unsupported goals {unsupported}"
        )
    if not all(oracle_results):
        raise InteractionSpecError(
            f"{task_id}: WATCH oracle_commands do not satisfy every answer_key goal"
        )
    if all(default_results):
        raise InteractionSpecError(
            f"{task_id}: WATCH default_commands still satisfy every answer_key goal"
        )


def _symbolic_state(executable: Mapping[str, object]) -> _SymbolicState:
    world = _required_mapping(executable.get("world"), "$.executable_spec.world")
    locations: dict[str, str] = {}
    known_entities = {str(room) for room in world.get("rooms") or ()}
    container_targets: set[str] = set()
    surface_targets: set[str] = set()
    for field in ("objects", "parts", "containers", "fixtures"):
        values = world.get(field)
        if not isinstance(values, Mapping):
            continue
        for raw_key, raw_value in values.items():
            if not isinstance(raw_value, Mapping):
                continue
            place = str(raw_value.get("place") or "").strip()
            known_entities.add(str(raw_key))
            affordances = {
                str(value).strip().lower()
                for value in raw_value.get("affordances") or ()
            }
            if field == "containers" or "container" in affordances:
                container_targets.add(str(raw_key))
            if field == "fixtures" or "surface" in affordances:
                surface_targets.add(str(raw_key))
            if place:
                locations[str(raw_key)] = place
    containers = world.get("containers")
    if isinstance(containers, Mapping):
        for target, raw_value in containers.items():
            if not isinstance(raw_value, Mapping):
                continue
            for subject in raw_value.get("preload") or ():
                locations[str(subject)] = f"in:{target}"
    attached = {
        str(row.get("child")): str(row.get("parent"))
        for row in world.get("initial_attachments") or ()
        if isinstance(row, Mapping) and row.get("child") and row.get("parent")
    }
    return _SymbolicState(
        held=None,
        known_entities=known_entities,
        container_targets=container_targets,
        surface_targets=surface_targets,
        locations=locations,
        attached=attached,
        open_entities=set(),
        running=set(),
        sitting=None,
        raised=set(),
        cut=set(),
        cooked=set(),
        ever_moved=set(),
    )


def _apply_symbolic_commands(state: _SymbolicState, commands: Sequence[str]) -> None:
    for command in commands:
        _apply_symbolic_command(state, command)


def _apply_symbolic_command(state: _SymbolicState, command: str) -> None:
    verb, indexed_args, _ = _command_key(command)
    args = [name for name, _ in indexed_args]
    unknown = [
        name
        for name, instance_index in indexed_args
        if instance_index != 0 and name not in state.known_entities
    ]
    if unknown:
        raise InteractionSpecError(
            f"symbolic WATCH branch command {command!r} names unknown entities {unknown}"
        )
    if _apply_symbolic_transfer(state, verb, args, command):
        return
    if _apply_symbolic_attachment(state, verb, args, command):
        return
    _apply_symbolic_state_change(state, verb, args, command)


def _apply_symbolic_transfer(
    state: _SymbolicState, verb: str, args: list[str], command: str
) -> bool:
    if verb == "GRAB" and args:
        if state.held is not None:
            raise InteractionSpecError(
                f"symbolic WATCH branch cannot execute {command!r}; already holding {state.held}"
            )
        state.held = args[0]
        state.locations[args[0]] = "held"
        state.ever_moved.add(args[0])
        return True
    if verb == "DROP":
        if state.held is None:
            raise InteractionSpecError(
                f"symbolic WATCH branch cannot execute {command!r}; the hand is empty"
            )
        state.locations[state.held] = "floor"
        state.ever_moved.add(state.held)
        state.held = None
        return True
    if verb in {"PUTIN", "PUTINSIDE"}:
        if len(args) >= 2:
            _require_symbolic_held(state, args[0], command)
            _require_symbolic_target(
                state.container_targets, args[1], command, "container"
            )
            _symbolic_place(state, args[0], f"in:{args[1]}")
        return True
    if verb in {"PUT", "PUTBACK", "PUTON"}:
        if len(args) >= 2:
            _require_symbolic_held(state, args[0], command)
            _require_symbolic_target(state.surface_targets, args[1], command, "surface")
            _symbolic_place(state, args[0], f"on:{args[1]}")
        return True
    return verb == "GRAB"


def _apply_symbolic_attachment(
    state: _SymbolicState, verb: str, args: list[str], command: str
) -> bool:
    if verb in {"ATTACH", "ASSEMBLE", "INSTALL"}:
        if len(args) >= 3:
            _require_symbolic_held(state, args[0], command)
            state.attached[args[0]] = args[2]
            state.locations[args[0]] = f"attached:{args[2]}"
            state.ever_moved.add(args[0])
            state.held = None
        return True
    if verb == "DETACH":
        if len(args) >= 2:
            if state.held is not None:
                raise InteractionSpecError(
                    f"symbolic WATCH branch cannot execute {command!r}; "
                    f"already holding {state.held}"
                )
            state.attached.pop(args[0], None)
            state.locations[args[0]] = "held"
            state.ever_moved.add(args[0])
            state.held = args[0]
        return True
    return False


def _apply_symbolic_state_change(
    state: _SymbolicState, verb: str, args: list[str], command: str
) -> None:
    if _apply_symbolic_open_or_power(state, verb, args):
        return
    if verb == "RAISE" and args:
        state.raised.add(args[0])
        state.ever_moved.add(args[0])
    elif verb == "SIT" and args:
        state.sitting = args[0]
    elif verb == "STANDUP":
        state.sitting = None
    elif verb == "CUT" and args:
        if len(args) >= 2:
            _require_symbolic_held(state, args[1], command)
        state.cut.add(args[0])
    elif verb in {"STIR", "POUR", "WIPE"} and len(args) >= 2:
        _require_symbolic_held(state, args[1], command)
    elif verb == "COOK" and args:
        state.cooked.add(args[0])
    elif verb in {"PUSH", "PULL"} and len(args) >= 2:
        state.locations[args[0]] = f"near:{args[1]}"
        state.ever_moved.add(args[0])


def _apply_symbolic_open_or_power(
    state: _SymbolicState, verb: str, args: list[str]
) -> bool:
    if verb in {"OPEN", "OPENDOOR"} and args:
        state.open_entities.add(args[0])
    elif verb in {"CLOSE", "CLOSEDOOR"} and args:
        state.open_entities.discard(args[0])
    elif verb == "TURNON" and args:
        state.running.add(args[0])
    elif verb == "TURNOFF" and args:
        state.running.discard(args[0])
    else:
        return False
    return True


def _symbolic_place(state: _SymbolicState, subject: str, location: str) -> None:
    state.locations[subject] = location
    state.ever_moved.add(subject)
    if state.held == subject:
        state.held = None


def _require_symbolic_held(state: _SymbolicState, subject: str, command: str) -> None:
    if state.held != subject:
        raise InteractionSpecError(
            f"symbolic WATCH branch cannot execute {command!r}; "
            f"the hand holds {state.held!r}, not {subject!r}"
        )


def _require_symbolic_target(
    allowed: set[str], target: str, command: str, affordance: str
) -> None:
    if target not in allowed:
        raise InteractionSpecError(
            f"symbolic WATCH branch cannot execute {command!r}; "
            f"{target!r} is not a declared {affordance}"
        )


def _symbolic_goal_holds(
    row: Mapping[str, object], state: _SymbolicState
) -> bool | None:
    pred = str(row.get("pred") or "").strip().lower().replace("-", "_")
    negative = pred.startswith("not_")
    if negative:
        pred = pred[4:]
    subject = str(row.get("subject") or "")
    target = str(row.get("target") or "")
    result = _positive_symbolic_goal_holds(pred, subject, target, state)
    if result is None:
        return None
    if negative:
        result = not result
    expected = row.get("truth", True)
    return result if bool(expected) else not result


def _positive_symbolic_goal_holds(
    pred: str, subject: str, target: str, state: _SymbolicState
) -> bool | None:
    location_prefix = {"inside": "in", "in": "in", "on": "on", "near": "near"}
    if pred in location_prefix:
        return state.locations.get(subject) == f"{location_prefix[pred]}:{target}"
    if pred == "attached":
        return state.attached.get(subject) == target
    if pred == "held":
        return state.held == subject
    membership = {
        "open": state.open_entities,
        "running": state.running,
        "turned_on": state.running,
        "raised": state.raised,
        "cut": state.cut,
        "cooked": state.cooked,
    }
    if pred in membership:
        return subject in membership[pred]
    if pred in {"closed", "off", "turned_off"}:
        active = state.open_entities if pred == "closed" else state.running
        return subject not in active
    if pred == "sitting":
        return state.sitting == subject
    if pred == "unmoved":
        return subject not in state.ever_moved
    return None


def _dialogue_action_index(row: Mapping[str, object]) -> int | None:
    anchor = str(row.get("anchor_beat_id") or "").strip()
    match = _ANCHOR_INDEX_RE.fullmatch(anchor)
    return int(match.group(1)) if match is not None else None


def _opening_evidence(
    story: Mapping[str, object], action_count: int, task_id: str
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    indices: set[int] = set()
    fact_ids: list[str] = []
    for row in _mapping_sequence(story.get("evidence"), "$.story.evidence"):
        if str(row.get("channel") or "").lower() != "prologue":
            continue
        raw_index = row.get("source_index")
        if raw_index is None:
            raise InteractionSpecError(
                f"{task_id}: prologue evidence has no source_index"
            )
        if isinstance(raw_index, bool):
            raise InteractionSpecError(
                f"{task_id}: prologue evidence source_index must be an integer"
            )
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise InteractionSpecError(
                f"{task_id}: prologue evidence source_index must be an integer"
            ) from exc
        if not 0 <= index < action_count:
            raise InteractionSpecError(
                f"{task_id}: prologue evidence source_index {index} is outside "
                f"the {action_count}-action opening"
            )
        fact_id = str(row.get("id") or f"opening_action_{index}").strip()
        indices.add(index)
        fact_ids.append(fact_id)
    return tuple(sorted(indices)), tuple(fact_ids)


def _normalize_spoken_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _meaningful_words(value: str) -> set[str]:
    return {
        word
        for word in _WORD_RE.findall(value.lower().replace("_", " "))
        if len(word) > 1 and word not in _NARRATE_STOPWORDS
    }


def _required_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InteractionSpecError(f"{path} must be an object")
    return value


def _mapping_sequence(value: object, path: str) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InteractionSpecError(f"{path} must be an array")
    rows: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise InteractionSpecError(f"{path}[{index}] must be an object")
        rows.append(item)
    return tuple(rows)


def _string_sequence(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InteractionSpecError(f"{path} must be an array")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise InteractionSpecError(f"{path} contains an empty command")
    return result


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool):
        raise InteractionSpecError(f"{path}: expected a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise InteractionSpecError(f"{path}: expected a positive integer") from exc
    if number <= 0:
        raise InteractionSpecError(f"{path}: expected a positive integer")
    return number


def _required_text(value: object, path: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise InteractionSpecError(f"{path} must be non-empty text")
    return result
