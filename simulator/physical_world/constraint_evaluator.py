"""Restaurant "closing: break down the line safely" benchmark task -- MULTI-ROOM DATA + goal/negative layer.

Task 1 of the assistant benchmark (see the design record). It restores what a restaurant looks like
JUST AFTER CLOSING and asks the AI assistant to direct the (flawless) human to put EACH item back where
it belongs, across THREE connected zones of a multi-room restaurant (verified connected by doorways):

  * DINING room (front)  -- bus the tables: dirty dishes/glasses, leftover scraps, soiled napkins.
  * KITCHEN (back)       -- put the line away by the food-safety cold-chain + sort waste/used-oil.
  * CORRIDOR (back)      -- return cleaning chemicals to their cabinet, AWAY from food (a different room).

Difficulty = professional knowledge (HACCP/ServSafe cold-chain, raw/ready-to-eat separation, chemical
separation, waste + grease sorting), not dexterity. Spreading the work across rooms keeps every area
uncrowded and realistic. The honest success signal is per-item INSIDE(item, correct_container) + a few
NEGATIVE constraints -- all grounded on the one relation the live scorer computes honestly (INSIDE-a-
container, unit level). The action string that achieves this is authored by the builder (reference
solution); at eval time the AI assistant authors it.

STRICTLY ADDITIVE: a NEW task-scoped module importing ONLY isaac_goal. behavior_import.build_behavior_closing
consumes ITEMS/CONTAINERS to build the scene; episode_runtime's gated scorer branch consumes build_task_fields()
via scene.behavior_task (goals + neg_constraints). Nothing in the shared engine changes.

    bash hand/run_isaac.sh simulator.physical_world/constraint_evaluator.py   # offline goal/negative selftest (no Isaac)
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from isaac_goal import check_goal, goal_progress, inside, Pred  # noqa: E402

SCENE = "restaurant_hotel"  # multi-room (dining + kitchen + corridors + lobby); rooms verified connected

# Every negative-constraint kind ``eval_neg_constraints`` implements. SINGLE SOURCE OF TRUTH: spec_validator
# imports THIS rather than keeping its own copy, so a newly implemented kind cannot be validated against a
# stale table.
NEG_KINDS: frozenset[str] = frozenset(
    {"not_inside", "all_placed", "no_share_group", "capacity"}
)

# Of those four, exactly ONE can be EARNED. ``all_placed`` demands that every item ended up in some
# correct container, so an agent that does nothing fails it -- it is an ACHIEVEMENT wearing a negative
# constraint's shape. The other three are PROHIBITIONS: true at t = 0 by construction (nothing is
# inside anything yet, so nothing is inside anything WRONG) and only ever broken, never earned.
#
# Measured on the two-agent run: 24/24 ``not_inside`` and 4/4 ``capacity`` "satisfied" against 0/42
# achievement predicates, giving ``n_sub_goals_ok / n_sub_goals`` a 33-50 % (mean 37.6 %) do-nothing
# floor. A prohibition is therefore a GATE -- it can only WITHHOLD credit -- and never a numerator.
# Single source of truth for that split across runtime scoring and released evaluation.
NEG_ACHIEVEMENT_KINDS: frozenset[str] = frozenset({"all_placed"})
NEG_PROHIBITION_KINDS: frozenset[str] = NEG_KINDS - NEG_ACHIEVEMENT_KINDS
AGGREGATE_ACHIEVEMENT_KINDS: frozenset[str] = frozenset({"exact_count_inside"})
POSITIVE_PROHIBITION_PREDS: frozenset[str] = frozenset(
    {"unmoved", "is_unmoved", "remain_unmoved"}
)

ACHIEVEMENT = "achievement"
PROHIBITION = "prohibition"


def sub_goal_class(entry: dict) -> str:
    """``achievement`` (credit can be EARNED) or ``prohibition`` (credit can only be WITHHELD).

    Accepts either shape: a manifest sub-goal (positive ``{pred, subject, target, ok}`` / negative
    ``{kind, label, ok}``) or a spec sub-goal (negatives are ``kind="neg"`` + ``neg_kind``).
    An ``unmoved`` row or any positive row marked ``baseline_excluded`` is a prohibition:
    it can invalidate the final state but can never award progress.

    An unrecognised negative kind is treated as a PROHIBITION -- the conservative direction is to
    withhold credit, never to invent it. ``validate_neg_kinds`` is the build-time gate that stops one
    reaching here at all.
    """
    if entry.get("baseline_excluded") is True:
        return PROHIBITION
    pred = str(entry.get("pred") or "").casefold()
    if pred in POSITIVE_PROHIBITION_PREDS:
        return PROHIBITION
    if pred or entry.get("subject"):
        return ACHIEVEMENT  # a positive goal
    kind = str(entry.get("kind") or "")
    if kind == "neg":
        kind = str(entry.get("neg_kind") or "")
    return (
        ACHIEVEMENT
        if kind in NEG_ACHIEVEMENT_KINDS | AGGREGATE_ACHIEVEMENT_KINDS
        else PROHIBITION
    )


def score_sub_goals(sub: list[dict]) -> dict:
    """Split one episode's sub-goal booleans into the two classes and score ONLY the achievements.

    SINGLE SOURCE OF TRUTH for the rule, shared by ``episode_runtime._task_goal_record`` (which writes
    the manifest) and the release evaluator (which reads it), so the two cannot disagree.

    ``progress_sub_goals`` is the achievement ratio, gated: every prohibition must hold or the whole
    ratio is withheld. A task with no achievement at all scores 1.0 when the gate holds, matching the
    existing "no positive goals" convention in ``progress``.
    """
    ach = [r for r in sub if sub_goal_class(r) == ACHIEVEMENT]
    pro = [r for r in sub if sub_goal_class(r) == PROHIBITION]
    n_ach_ok = sum(1 for r in ach if r.get("ok"))
    n_pro_ok = sum(1 for r in pro if r.get("ok"))
    ratio = (n_ach_ok / len(ach)) if ach else 1.0
    gate = n_pro_ok == len(pro)
    return {
        "sub_goal_scoring": "achievements_only_gated_by_prohibitions",
        "n_achievements": len(ach),
        "n_achievements_ok": n_ach_ok,
        "progress_achievements": ratio,
        "n_prohibitions": len(pro),
        "n_prohibitions_ok": n_pro_ok,
        "prohibitions_ok": gate,
        "progress_sub_goals": ratio if gate else 0.0,
    }


class NegKindError(ValueError):
    """A task spec declared a negative-constraint kind that no evaluator branch implements."""


def validate_neg_kinds(neg_list: list[dict]) -> None:
    """Raise on any unimplemented kind. Pure + Isaac-free, so it belongs at BUILD / spec-authoring time --
    NOT at score time, where episode_runtime's blanket ``except Exception`` around the goal block would
    convert the raise into ``task_goal: null`` with ``episode_valid`` still true (a false pass by another
    route). Deliberately NOT called from ``build_task_fields`` -- that accessor is shared with the offline
    spec_validator, which must REPORT an unknown kind (NEG001), not die on it."""
    unknown = sorted({str(c.get("kind")) for c in neg_list} - NEG_KINDS)
    if unknown:
        raise NegKindError(
            f"unimplemented negative-constraint kind(s) {unknown}; "
            f"implemented: {sorted(NEG_KINDS)}"
        )


# --- ITEMS (the graspables): where each REALISTICALLY sits at close -> where it BELONGS -------------
# key -> {cat, model, zone, place, target}
#   zone  : which room it starts in (dining / kitchen).
#   place : how the builder seats it initially -- "table" (a dining table top), "prepline" (the kitchen
#           prep counter, i.e. food left out on the line), "counter" (a kitchen worktop).
#   target: the container entity it must end up INSIDE (a CONTAINERS key = the rule-correct destination).
ITEMS: dict[str, dict] = {
    # DINING (front) -- bus 3 dirty dishes into the bus tub (a deep tote reliably holds ~3; stacking a 4th,
    # or a 2nd tub in the same corner, destabilises the physics). The tables look FULL because many more
    # dirty dishes sit on them as set-dressing (DECOR) -- only these 3 are actually bussed. Models chosen to
    # READ on camera: a real tumbler (water_glass/evaida ~0.16 m, not the tiny bbpraa) and a leftover MEAL
    # (club_sandwich, not a single french fry) for the compost.
    "plate_1": {
        "cat": "plate",
        "model": "aewthq",
        "zone": "dining",
        "place": "table",
        "target": "bus_tub",
    },
    "plate_2": {
        "cat": "plate",
        "model": "aewthq",
        "zone": "dining",
        "place": "table",
        "target": "bus_tub",
    },
    "water_glass_1": {
        "cat": "water_glass",
        "model": "evaida",
        "zone": "dining",
        "place": "table",
        "target": "bus_tub",
    },
    "food_scraps": {
        "cat": "hamburger",
        "model": "fyswch",
        "zone": "dining",
        "place": "table",
        "target": "compost_bin",
    },
    # KITCHEN (back) -- cold chain: raw proteins vs ready-to-eat/dairy in SEPARATE fridges (HACCP). cheese
    # (swiss_cheese) reads far better than the paper-thin cold_cuts sliver and is still ready-to-eat/dairy.
    "raw_chicken": {
        "cat": "chicken_breast",
        "model": "iuuyre",
        "zone": "kitchen",
        "place": "prepline",
        "target": "raw_fridge",
    },
    "raw_steak": {
        "cat": "steak",
        "model": "akoxpw",
        "zone": "kitchen",
        "place": "prepline",
        "target": "raw_fridge",
    },
    "raw_salmon": {
        "cat": "salmon",
        "model": "glrmfn",
        "zone": "kitchen",
        "place": "prepline",
        "target": "raw_fridge",
    },
    "cheese_block": {
        "cat": "swiss_cheese",
        "model": "hwxeto",
        "zone": "kitchen",
        "place": "prepline",
        "target": "rte_fridge",
    },
    "milk": {
        "cat": "carton_of_milk",
        "model": "atyqub",
        "zone": "kitchen",
        "place": "prepline",
        "target": "rte_fridge",
    },
    # KITCHEN -- waste sorting: used oil to the grease caddy, two recyclable cans to recycling. (A tall beer
    # bottle balanced on the rim after landing on the first can -- tall items dropped onto a prior item don't
    # seat; two SHORT cans nest low + both score INSIDE reliably.)
    "used_oil": {
        "cat": "oil_bottle",
        "model": "gcixra",
        "zone": "kitchen",
        "place": "counter",
        "target": "grease_caddy",
    },
    "empty_can": {
        "cat": "can_of_soda",
        "model": "bfrzvk",
        "zone": "kitchen",
        "place": "counter",
        "target": "recycling_bin",
    },
    "empty_can_2": {
        "cat": "can_of_soda",
        "model": "bfrzvk",
        "zone": "kitchen",
        "place": "counter",
        "target": "recycling_bin",
    },
    # KITCHEN counter -- a cleaning chemical left out, must go to the chemical bin away from food. ONLY ONE
    # tall bottle: two tall bottles dropped into one open bin topple (the 2nd lands on the 1st + rolls out --
    # verified across a shelved cabinet, a narrow tote, a ledged bin, and a plain deep bin). An open bin
    # reliably holds MANY SHORT items or ONE TALL item; a 2nd chemical would need a genuinely partitioned
    # unit. The negative "no chemical in a food fridge" + the separate-room rule still exercise the concept.
    "bleach": {
        "cat": "bottle_of_bleach_agent",
        "model": "lfjumk",
        "zone": "kitchen",
        "place": "counter",
        "target": "chem_cabinet",
    },
}

# --- CONTAINERS (the destinations) -----------------------------------------------------------------
# ``pick`` DISAMBIGUATOR (scene-sourced units only). The kitchen holds THREE fridge instances, so
# ``cat: "fridge"`` alone does not say which appliance becomes the RAW unit -- and the answer key
# (raw proteins -> raw_fridge) is fixed, so an unstated rule lets the ground truth drift onto a
# different appliance whenever the layout, the anchor or the candidate iteration order moves. ``by``
# names the selection rule the builder applies and ``rank`` the position in it, giving a TOTAL order
# reproducible from the spec alone. ``nearest_room_anchor`` = ascending distance from the container's
# ``room`` anchor, ties broken on the scene instance name (never on build order).
PICK_NEAREST: str = "nearest_room_anchor"
# name -> spec.  source "scene" = reuse an in-scene furniture prim (resolve by cat); "import" = add a baked
# (cat/model). kind "swing" = a revolute door unit ([Open]/[Close]); "open" = top-open bin/basin.
# room = which zone it lives in (drives realistic placement). A "scene" fridge is assigned from the kitchen's
# real openable fridges (the diner/hotel jtqazu doors open via a task-local lax leaf reader; see the builder).
# Distinguishability WITHOUT floating markers, done the way a real kitchen does it:
#   * ``recolor_body`` tints the whole BIN to the standard waste colour (real bins are solid-coloured plastic)
#     using ``color``; black-glass fridges can't be recoloured so they keep their look.
#   * ``sign`` = a real 2D LABEL SIGN (a PNG under labels/) the builder mounts FLAT on the unit's front/door
#     (RAW/R.T.E. on the fridge doors, COMPOST/RECYCLE/USED OIL/CHEMICAL on the bins). Swap the PNG to
#     restyle. ``None`` = no sign. The 3 kitchen bins are also grouped into a ROW against a wall (a real
#     waste station) instead of scattered -- see build_behavior_closing.
CONTAINERS: dict[str, dict] = {
    # bus_tub = a DEEP tote (0.34x0.24x0.47), not a flat tray. ONE tub holds the 3 bussed dishes; extra
    # dishes are DECOR. Recoloured a neutral bus-tub gray; no sign (obviously a dish tub in the dining room).
    "bus_tub": {
        "source": "import",
        "cat": "tote",
        "model": "bnkjle",
        "kind": "open",
        "room": "dining",
        "recolor_body": True,
        "color": (0.30, 0.33, 0.38),
        "sign": "bus",
        "label": "bus tub (deep tote)",
    },
    "compost_bin": {
        "source": "import",
        "cat": "compost_bin",
        "model": "fvkdos",
        "kind": "open",
        "room": "kitchen",
        "recolor_body": True,
        "color": (0.20, 0.52, 0.26),
        "sign": "compost",
        "label": "compost bin (food waste)",
    },
    "recycling_bin": {
        "source": "import",
        "cat": "recycling_bin",
        "model": "mdojox",
        "kind": "open",
        "room": "kitchen",
        "recolor_body": True,
        "color": (0.14, 0.42, 0.72),
        "sign": "recycle",
        "label": "recycling (cans/bottles)",
    },
    "grease_caddy": {
        "source": "import",
        "cat": "bucket",
        "model": "bdhvnt",
        "kind": "open",
        "room": "kitchen",
        "recolor_body": True,
        "color": (0.74, 0.56, 0.12),
        "sign": "oil",
        "label": "used-oil / grease caddy",
    },
    "raw_fridge": {
        "source": "scene",
        "cat": "fridge",
        "kind": "swing",
        "room": "kitchen",
        "recolor_body": False,
        "color": None,
        "sign": "raw",
        "label": "RAW cold unit (raw proteins)",
        "pick": {"by": PICK_NEAREST, "rank": 0},
    },
    "rte_fridge": {
        "source": "scene",
        "cat": "fridge",
        "kind": "swing",
        "room": "kitchen",
        "recolor_body": False,
        "color": None,
        "sign": "dairy",
        "label": "ready-to-eat / dairy cold unit",
        "pick": {"by": PICK_NEAREST, "rank": 1},
    },
    # chem_cabinet = a SIMPLE deep-wide open bin (trash_can/cjmezk, 0.38x0.38x0.42, 0 joints) -- a plain box
    # (no shelf/ledge) that holds the 1 bleach jug LOW inside (honest INSIDE). Recoloured RED + a CHEMICAL
    # sign, in a room away from the food. (Earlier chem tries failed on geometry: a shelved cabinet bounced
    # bottles out; a narrow tote/ledged bin toppled them -- see restaurant-task LEARNINGS.)
    "chem_cabinet": {
        "source": "import",
        "cat": "trash_can",
        "model": "cjmezk",
        "kind": "open",
        "room": "corridor",
        "recolor_body": True,
        "color": (0.78, 0.16, 0.10),
        "sign": "chem",
        "label": "chemical bin (away from food)",
    },
    # in-scene basin: registered CONTAINER ONLY so the "no oil in the sink" negative is scorable.
    "sink": {
        "source": "scene",
        "cat": "commercial_kitchen_sink",
        "kind": "open",
        "room": "kitchen",
        "recolor_body": False,
        "color": None,
        "sign": None,
        "label": "prep sink (must NOT receive grease)",
    },
}

ANSWER_KEY: dict[str, str] = {k: v["target"] for k, v in ITEMS.items()}

# --- rule sets (for the negative constraints) ------------------------------------------------------
COLD_UNITS: set[str] = {"raw_fridge", "rte_fridge"}
CHEMICALS: set[str] = {"bleach"}
OILS: set[str] = {"used_oil"}
PERISHABLES: set[str] = {
    "raw_chicken",
    "raw_steak",
    "raw_salmon",
    "cheese_block",
    "milk",
}
SINK: str = "sink"

# --- DECOR (set-dressing) --------------------------------------------------------------------------
# Props that make it look like a REAL just-closed restaurant, NOT a tidy puzzle board. These are NOT
# graspable, NOT scored, and NOT obstacles: the builder places each as a plain USD reference (no
# scene.register), so they are invisible to the program builder, the scorer, and the nav grid. Two roles:
# (1) "leave it on the table" items a shift would NOT bus (condiments, a stray wine glass) -- the user's
# point that not everything gets collected; (2) end-of-shift clutter (dirty pans/boards, a bussing cart,
# a broom). ``on`` routes placement: dining/prep/counter table tops, or floor_<room> along a wall.
# Placement is RESILIENT -- a prop whose model is not baked is silently skipped, so it never breaks the build.
DECOR: list[dict] = [
    # dining tables -- what STAYS on the table at close (condiments/tableware), plus used napkins
    {"cat": "salt_shaker", "model": "fztfsi", "on": "dining"},
    {"cat": "salt_shaker", "model": "fztfsi", "on": "dining"},
    {"cat": "bottle_of_catsup", "model": "bcqfxb", "on": "dining"},
    {"cat": "bottle_of_catsup", "model": "bcqfxb", "on": "dining"},
    {"cat": "bottle_of_wine", "model": "dawhco", "on": "dining"},
    {"cat": "wineglass", "model": "aakcyj", "on": "dining"},
    {"cat": "wineglass", "model": "aakcyj", "on": "dining"},
    {"cat": "dinner_napkin", "model": "etxjzg", "on": "dining"},
    {"cat": "dinner_napkin", "model": "etxjzg", "on": "dining"},
    # dining tables -- MORE dirty dishes left out (static; makes the tables look full even though only 3 bus)
    {"cat": "plate", "model": "aewthq", "on": "dining"},
    {"cat": "plate", "model": "aewthq", "on": "dining"},
    {"cat": "plate", "model": "aewthq", "on": "dining"},
    {"cat": "water_glass", "model": "evaida", "on": "dining"},
    {"cat": "water_glass", "model": "evaida", "on": "dining"},
    {"cat": "coffee_cup", "model": "ckkwmj", "on": "dining"},
    # prep line -- dirty prep clutter left out at close
    {"cat": "cutting_board", "model": "aibvew", "on": "prep"},
    {"cat": "chopping_board", "model": "afwefw", "on": "prep"},
    {"cat": "mixing_bowl", "model": "bsgybx", "on": "prep"},
    {"cat": "frying_pan", "model": "aewpzn", "on": "prep"},
    {"cat": "tray", "model": "avotsj", "on": "prep"},
    # counter -- more line clutter
    {"cat": "baking_sheet", "model": "yhurut", "on": "counter"},
    {"cat": "stockpot", "model": "azoiaq", "on": "counter"},
    {"cat": "coffee_cup", "model": "ckkwmj", "on": "counter"},
    # floors -- bussing / janitorial gear along the walls
    {"cat": "dish_rack_cart", "model": "rniujn", "on": "floor_kitchen"},
    {"cat": "scrub_brush", "model": "hsejyi", "on": "floor_kitchen"},
    {"cat": "broom", "model": "tpyvbt", "on": "floor_corridor"},
    {"cat": "bucket", "model": "bdhvnt", "on": "floor_corridor"},
]


def items_in_zone(zone: str) -> list[str]:
    return [k for k, v in ITEMS.items() if v["zone"] == zone]


def pick_order(cat: str) -> list[str]:
    """Container keys claiming scene category ``cat``, in declared ``pick.rank`` order.

    The builder assigns the ranked scene instances to THIS list, so which appliance carries which answer
    key is fixed by the spec rather than by whichever candidate the scene happened to yield first.
    """
    keys = [
        k
        for k, c in CONTAINERS.items()
        if c.get("source") == "scene" and c.get("cat") == cat and c.get("pick")
    ]
    return sorted(keys, key=lambda k: int(CONTAINERS[k]["pick"]["rank"]))


def build_task_fields() -> tuple[list[tuple[str, str]], list[dict]]:
    """Return (goals, neg_constraints) for scene.behavior_task, consumed by episode_runtime's gated scorer.

    goals: [(item_key, container_name), ...] -> check_goal([inside(o, c) ...]).
    neg_constraints: dicts for episode_runtime._eval_neg_constraints / eval_neg_constraints:
      {"kind":"not_inside","objs":[...],"containers":[...],"label":...}  fail if any obj INSIDE any container.
      {"kind":"all_placed","objs":[...],"label":...}                     fail if any obj is INSIDE no container.
    """
    goals = [(k, ANSWER_KEY[k]) for k in ITEMS]
    negs: list[dict] = [
        {
            "kind": "not_inside",
            "objs": sorted(CHEMICALS),
            "containers": sorted(COLD_UNITS),
            "label": "no_chemical_in_a_food_fridge",
        },
        {
            "kind": "not_inside",
            "objs": sorted(OILS),
            "containers": [SINK],
            "label": "no_oil_in_the_sink",
        },
        {
            "kind": "all_placed",
            "objs": sorted(PERISHABLES),
            "label": "no_perishable_left_on_the_line",
        },
    ]
    return goals, negs


def eval_neg_constraints(neg_list: list[dict], relations: set) -> list[dict]:
    """Pure set logic over the INSIDE pairs in a relation set (isaac_goal has no NOT/absent predicate).

    Returns one verdict dict per constraint, in input order:
    ``{"idx","label","kind","ok","offenders"}``. Shared with episode_runtime. ``idx`` is the constraint's
    position in ``neg_list``, so a verdict can be joined back to the constraint that produced it even
    when two constraints share a label or a kind (``label`` alone is not a key).
    """
    inside_pairs = {(a, b) for (p, a, b) in relations if p is Pred.INSIDE}
    placed = {a for (a, _b) in inside_pairs}
    out: list[dict] = []
    for c in neg_list:
        # A MISSING "kind" is the limit case of an unknown one: reported below, never raised. A KeyError
        # here would be swallowed by the scorer's blanket except and become task_goal:null -- a false pass.
        kind = str(c.get("kind"))
        if kind == "not_inside":
            objs, conts = set(c["objs"]), set(c["containers"])
            bad = sorted(a for (a, b) in inside_pairs if a in objs and b in conts)
            out.append(
                {
                    "label": c.get("label", kind),
                    "kind": kind,
                    "ok": not bad,
                    "offenders": bad,
                }
            )
        elif kind == "all_placed":
            missing = sorted(o for o in c["objs"] if o not in placed)
            out.append(
                {
                    "label": c.get("label", kind),
                    "kind": kind,
                    "ok": not missing,
                    "offenders": missing,
                }
            )
        elif kind == "no_share_group":
            # fail if any storage unit holds two mutually-incompatible hazard groups. group_of + the incompatibility
            # matrix + the unit set are self-contained on the constraint dict (no signature change).
            gof = c.get("group_of", {})
            incompat = [tuple(p) for p in c.get("incompatible", [])]
            units = set(c.get("units") or {b for (_a, b) in inside_pairs})
            per_unit: dict[str, set] = {}
            for a, b in inside_pairs:
                if b in units and a in gof:
                    per_unit.setdefault(b, set()).add(gof[a])
            bad = sorted(
                f"{u}:{g1}+{g2}"
                for u, gs in per_unit.items()
                for (g1, g2) in incompat
                if g1 in gs and g2 in gs
            )
            out.append(
                {
                    "label": c.get("label", kind),
                    "kind": kind,
                    "ok": not bad,
                    "offenders": bad,
                }
            )
        elif kind == "capacity":
            # fail if any unit's INSIDE count exceeds its capacity (units absent from `cap` are unlimited).
            caps = c.get("cap", {})
            counts: dict[str, int] = {}
            for _a, b in inside_pairs:
                counts[b] = counts.get(b, 0) + 1
            bad = sorted(
                f"{u}:{counts[u]}>{caps[u]}" for u in caps if counts.get(u, 0) > caps[u]
            )
            out.append(
                {
                    "label": c.get("label", kind),
                    "kind": kind,
                    "ok": not bad,
                    "offenders": bad,
                }
            )
        else:
            # UNKNOWN kind -> never a silent pass. A constraint nobody evaluates cannot have been
            # satisfied, so it is reported VIOLATED and names itself as the offender. Structured, not
            # raised: a raise here is swallowed by the scorer's blanket except and becomes task_goal:null.
            out.append(
                {
                    "label": c.get("label", kind),
                    "kind": kind,
                    "ok": False,
                    "offenders": [f"unimplemented_kind:{kind}"],
                }
            )
    # Exactly one verdict is appended per constraint above, so enumerate(out) IS the input position.
    for i, verdict in enumerate(out):
        verdict["idx"] = i
    return out


# =====================================================================================================
# Offline selftest (no Isaac): goal + negative logic against synthetic relation sets.
# =====================================================================================================
# One minimally-populated constraint per implemented kind: enough fields that its branch runs without a
# KeyError, empty enough that a compliant world yields no offenders. Keyed by kind so a member added to
# NEG_KINDS without a branch (or a branch deleted) is caught by _selftest_neg_kinds below.
_KIND_PROBES: dict[str, dict] = {
    "not_inside": {"kind": "not_inside", "objs": [], "containers": []},
    "all_placed": {"kind": "all_placed", "objs": []},
    "no_share_group": {"kind": "no_share_group"},
    "capacity": {"kind": "capacity"},
}


def _selftest_neg_kinds(chk, correct: set) -> None:
    """The unknown-kind contract: no kind outside NEG_KINDS may report CLEAN, at score time or build time."""
    chk(
        set(_KIND_PROBES) == set(NEG_KINDS),
        "the drift probe covers every NEG_KINDS member",
    )
    chk(
        all(
            eval_neg_constraints([_KIND_PROBES[k]], correct)[0]["offenders"] == []
            for k in sorted(NEG_KINDS)
        ),
        "every NEG_KINDS member reaches a real evaluator branch",
    )
    for bogus, why in (
        ({"kind": "tall_low", "label": "x"}, "an UNKNOWN kind"),
        ({"label": "x"}, "a MISSING kind"),
    ):
        v = eval_neg_constraints([bogus], correct)[0]
        chk(
            not v["ok"] and v["offenders"] == [f"unimplemented_kind:{v['kind']}"],
            f"{why} reports VIOLATED and names itself, even in a fully-correct world",
        )
    try:
        validate_neg_kinds([{"kind": "tall_low"}])
        chk(False, "validate_neg_kinds raises on an unimplemented kind")
    except NegKindError as exc:
        chk(
            "tall_low" in str(exc), "validate_neg_kinds raises on an unimplemented kind"
        )
    try:
        validate_neg_kinds([{"kind": k} for k in sorted(NEG_KINDS)])
        silent = True
    except NegKindError:
        silent = False
    chk(silent, "validate_neg_kinds is silent on every implemented kind")


def _selftest_sub_goal_classes(chk) -> None:
    """The prohibition/achievement split, and the do-nothing floor it removes."""
    chk(
        NEG_ACHIEVEMENT_KINDS | NEG_PROHIBITION_KINDS == NEG_KINDS
        and not (NEG_ACHIEVEMENT_KINDS & NEG_PROHIBITION_KINDS),
        "every registered negative kind is classified exactly once",
    )
    chk(
        sub_goal_class({"kind": "all_placed", "label": "all_placed"}) == ACHIEVEMENT,
        "all_placed is an ACHIEVEMENT -- an agent that does nothing does not satisfy it",
    )
    chk(
        all(
            sub_goal_class({"kind": k, "label": k}) == PROHIBITION
            for k in NEG_PROHIBITION_KINDS
        ),
        f"{sorted(NEG_PROHIBITION_KINDS)} are prohibitions",
    )
    chk(
        sub_goal_class({"pred": "inside", "subject": "a", "target": "b"}) == ACHIEVEMENT
        and sub_goal_class({"kind": "neg", "neg_kind": "capacity", "label": "x"})
        == PROHIBITION,
        "both the manifest shape and the spec shape classify correctly",
    )
    # The T1-CHEM-01 shape: one positive, one all_placed, one not_inside. Doing nothing used to pay
    # 1/3; it must now pay nothing, while a completed episode still pays 1.00.
    idle = [
        {"pred": "inside", "subject": "item_1", "target": "unit_b", "ok": False},
        {"kind": "all_placed", "label": "all_placed", "ok": False},
        {"kind": "not_inside", "label": "not_inside", "ok": True},
    ]
    got = score_sub_goals(idle)
    chk(
        sum(1 for r in idle if r["ok"]) / len(idle) == 1 / 3
        and got["progress_sub_goals"] == 0.0,
        f"do-nothing floor 33.3% -> {got['progress_sub_goals']:.1%}",
    )
    chk(
        got["n_achievements"] == 2
        and got["n_prohibitions"] == 1
        and got["prohibitions_ok"],
        "both classes are reported separately, prohibitions held",
    )
    done = [dict(r, ok=True) for r in idle]
    chk(
        score_sub_goals(done)["progress_sub_goals"] == 1.0,
        "an episode that actually does the task still scores 1.00",
    )
    broke = [dict(r, ok=True) for r in idle[:2]] + [dict(idle[2], ok=False)]
    chk(
        score_sub_goals(broke)["progress_sub_goals"] == 0.0
        and score_sub_goals(broke)["progress_achievements"] == 1.0,
        "a broken prohibition WITHHOLDS the credit the achievements earned, and says it did",
    )


def _selftest() -> int:
    ok = True

    def _chk(cond, msg):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    goals, negs = build_task_fields()
    _chk(len(goals) == len(ITEMS), f"goals cover every item ({len(goals)})")
    _chk(
        all(c in CONTAINERS for _, c in goals),
        "every goal container is defined in CONTAINERS",
    )
    _chk(
        set(items_in_zone("dining")) and set(items_in_zone("kitchen")),
        "items span dining + kitchen zones",
    )
    _chk(
        all(CONTAINERS[ANSWER_KEY[k]] for k in ITEMS),
        "every item's target resolves to a container",
    )
    _chk(
        pick_order("fridge") == ["raw_fridge", "rte_fridge"],
        "the two cold units carry a declared, ranked pick disambiguator",
    )
    _shared: dict[str, list[str]] = {}
    for _k, _c in CONTAINERS.items():
        if _c.get("source") == "scene":
            _shared.setdefault(str(_c.get("cat")), []).append(_k)
    _ambig = sorted(
        cat
        for cat, keys in _shared.items()
        if len(keys) > 1
        and len(
            {
                int(CONTAINERS[k]["pick"]["rank"])
                for k in keys
                if CONTAINERS[k].get("pick")
            }
        )
        != len(keys)
    )
    _chk(
        not _ambig,
        f"no scene category is claimed by two containers without distinct ranks (offenders={_ambig})",
    )

    correct = {(Pred.INSIDE, k, ANSWER_KEY[k]) for k in ITEMS}
    _chk(
        check_goal([inside(o, c) for o, c in goals], correct),
        "correct placement -> goal MET",
    )
    _chk(
        abs(goal_progress([inside(o, c) for o, c in goals], correct) - 1.0) < 1e-9,
        "progress == 1.0 when correct",
    )
    _chk(
        all(r["ok"] for r in eval_neg_constraints(negs, correct)),
        "correct placement -> all negatives OK",
    )

    bad1 = set(correct)
    bad1.discard((Pred.INSIDE, "bleach", "chem_cabinet"))
    bad1.add((Pred.INSIDE, "bleach", "rte_fridge"))
    n1 = eval_neg_constraints(negs, bad1)
    _chk(
        any(r["label"] == "no_chemical_in_a_food_fridge" and not r["ok"] for r in n1),
        "bleach in a fridge -> flagged",
    )
    _chk(
        not check_goal([inside(o, c) for o, c in goals], bad1),
        "bleach mis-placed -> goal UNMET",
    )

    bad2 = set(correct)
    bad2.add((Pred.INSIDE, "used_oil", "sink"))
    _chk(
        any(
            r["label"] == "no_oil_in_the_sink" and not r["ok"]
            for r in eval_neg_constraints(negs, bad2)
        ),
        "oil in the sink -> flagged",
    )

    bad3 = {r for r in correct if r[1] != "raw_chicken"}
    _chk(
        any(
            r["label"] == "no_perishable_left_on_the_line" and not r["ok"]
            for r in eval_neg_constraints(negs, bad3)
        ),
        "perishable left out -> flagged",
    )

    _selftest_neg_kinds(_chk, correct)

    # idx is the join key back to the input constraint: positional, present on every branch, and
    # unambiguous where two constraints share a label.
    _idx = eval_neg_constraints(
        negs + [{"kind": "nope", "label": negs[0]["label"]}], correct
    )
    _chk(
        [r["idx"] for r in _idx] == list(range(len(negs) + 1)),
        "eval_neg_constraints stamps idx = position in the input list (unknown kind included)",
    )
    _chk(
        all(_idx[i]["label"] == negs[i]["label"] for i in range(len(negs))),
        "idx-th verdict describes the idx-th constraint",
    )

    _selftest_sub_goal_classes(_chk)

    print(
        f"[constraint_evaluator] selftest {'PASS' if ok else 'FAIL'} ({len(ITEMS)} items, {len(CONTAINERS)} containers, scene={SCENE})"
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest())
