#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""TaskPack world builder for frozen InteractiveWorld benchmark episodes."""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import os
import re
import sys
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_SIMULATOR_DIR = os.path.dirname(_THIS_DIR)
_RELEASE_DIR = os.path.dirname(_SIMULATOR_DIR)
_TASKPACK_DIR = os.path.join(_RELEASE_DIR, "benchmark", "tasks")

import command_grammar  # noqa: E402
import runtime_capabilities  # noqa: E402
from simulator.taskpack_support.bundled_assets import generated_label_cache  # noqa: E402


logger: logging.Logger = logging.getLogger(__name__)

SPEC_DIR: str = _TASKPACK_DIR
GENERATED_LABEL_DIR: str = str(generated_label_cache())
SCENE_JSON_ROOT: str = os.path.join(
    os.environ.get(
        "INTERACTIVEWORLD_BEHAVIOR_DATA", "/opt/interactiveworld/behavior-data"
    ),
    "behavior-1k-assets",
    "scenes",
)
FLOOR_SUPPORT_INDEX: str = os.path.join(
    _SIMULATOR_DIR, "taskpack_support", "floor_support_index.json"
)
SCENE_CAPABILITY_INDEX: str = os.path.join(
    _SIMULATOR_DIR, "taskpack_support", "scene_capabilities.json"
)

_RUNTIME_SPEC: dict | None = None
_RUNTIME_SPEC_DIGEST: str | None = None
_FLOOR_SUPPORT_DATA: dict | None = None
_FLOOR_SUPPORT_GRIDS: dict[tuple[str, str], object] = {}

# Reading distance floor for the START pose: the cue decal is ~0.1 m of artwork, so a start
# closer than this would let the model answer from frame 0 without the [LookAt] inspect.
MIN_CUE_DIST_M: float = 2.2
MAX_CUE_DIST_M: float = 6.0
# Floor a person needs IN FRONT of a swung-open door to stand square-on and reach the shelf.
DOOR_FRONT_CLEAR_M: float = 0.95
# Imported furniture must not occupy the swept/approach lane of an architectural door.  The
# scene door's closed leaf is already a siting obstacle, but it is only a thin slab; without this
# lane a cabinet can legally fit a few centimetres behind it and make the only inter-room route
# physically impossible once the leaf opens.
ARCHITECTURAL_DOOR_SWING_KEEP_OUT_M: float = 0.90
ARCHITECTURAL_DOOR_JAMB_KEEP_OUT_M: float = 0.12
# Keep genuinely embedded support approaches out while still allowing small body brushes to be
# graded as dirty at render time rather than turning them into build failures.
SUPPORT_STAND_MIN_CLEAR_M: float = 0.20
# Gap demanded between two sited units, on top of each unit's own footprint half-extent. The
# HARD guarantee that no two scored interiors overlap is ``assert_disjoint``; this only has to
# leave a person room to walk between them, so it stays small enough to fit a real kitchen.
UNIT_KEEPOUT_M: float = 0.55
# Scene-authored transforms and their live USD bounds can differ by a few millimetres.  Apply this
# only when both units are pinned to exact scene instances; inferred native picks still need the
# full separation threshold so the selector cannot silently choose an ambiguous pair.
EXACT_SCENE_UNIT_SEPARATION_TOLERANCE_M: float = 0.005
# A grid cell counts as walkable (nav-component flood fill) at this furniture clearance.
WALKABLE_CLEAR_M: float = 0.34
# A declared stand must have a walkable lattice cell locally.  Without a bound, ``_component_at``
# maps a stand inside furniture to whichever component happens to be nearest, even when it is on
# the other side of a cabinet bank.
STAND_COMPONENT_SNAP_M: float = 0.30
MIN_ROOM_HULL_SPAN_M: float = 1.0


class SpecBuildError(RuntimeError):
    """A spec could not be turned into a scorable world. ALWAYS raised at BUILD time: a raise
    inside episode_runtime's score-time goal block is swallowed by its blanket ``except`` and
    becomes ``task_goal: null`` with ``episode_valid`` still true -- a false pass."""


class SimulatorBuildError(SpecBuildError):
    """A valid runtime view exposed an invariant failure in the simulator builder."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


class UnsupportedSpec(SpecBuildError):
    """No world recipe exists for this task id yet (slice A ships exactly one)."""


class UnitBodyUnusable(SpecBuildError):
    """This baked body cannot play a destination container (no drivable swing leaf, or its own
    interior swallows a [PutInside]). Recoverable: the siting stage takes the next body down the
    category's ladder. Escaping this module still fails the build, never a silent skip."""

    def __init__(self, key: str, cat: str, model: str, why: str) -> None:
        super().__init__(f"unit {key} ({cat}/{model}) {why}")
        self.key, self.cat, self.model = key, cat, model


# =====================================================================================
# spec io
# =====================================================================================
def spec_path(task_id: str) -> str:
    return os.path.join(SPEC_DIR, f"{task_id}.json")


def _runtime_adapter():
    from simulator.taskpack_support.runtime_adapter import (
        canonical_digest,
        verify_runtime_view,
    )

    return canonical_digest, verify_runtime_view


def register_runtime_spec(runtime_view: dict, expected_digest: str) -> str:
    """Install one verified in-memory builder view, replacing the previous dynamic view."""
    if not isinstance(runtime_view, dict):
        raise SpecBuildError("runtime builder spec must be a JSON object")
    if not isinstance(expected_digest, str) or not expected_digest:
        raise SpecBuildError("runtime builder spec requires a non-empty digest")
    candidate = copy.deepcopy(runtime_view)
    canonical_digest, verify_runtime_view = _runtime_adapter()
    try:
        verify_runtime_view(candidate)
    except ValueError as exc:
        raise SpecBuildError(f"invalid runtime builder spec: {exc}") from exc
    actual_digest = canonical_digest(candidate)
    if actual_digest != expected_digest:
        raise SpecBuildError(
            "runtime builder spec digest mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    task_id = str(candidate.get("task_id") or "")
    if not task_id:
        raise SpecBuildError("runtime builder spec has no task_id")
    global _RUNTIME_SPEC, _RUNTIME_SPEC_DIGEST
    _RUNTIME_SPEC = candidate
    _RUNTIME_SPEC_DIGEST = actual_digest
    return actual_digest


def clear_runtime_spec() -> None:
    """Remove the current dynamic view; disk-backed specs remain available."""
    global _RUNTIME_SPEC, _RUNTIME_SPEC_DIGEST
    _RUNTIME_SPEC = None
    _RUNTIME_SPEC_DIGEST = None


def _registered_runtime_spec(task_id: str) -> dict | None:
    if _RUNTIME_SPEC is None or str(_RUNTIME_SPEC.get("task_id") or "") != task_id:
        return None
    candidate = copy.deepcopy(_RUNTIME_SPEC)
    canonical_digest, verify_runtime_view = _runtime_adapter()
    try:
        verify_runtime_view(candidate)
    except ValueError as exc:
        raise SpecBuildError(f"runtime builder registry drift: {exc}") from exc
    actual_digest = canonical_digest(candidate)
    if actual_digest != _RUNTIME_SPEC_DIGEST:
        raise SpecBuildError(
            "runtime builder registry drift: "
            f"registered {_RUNTIME_SPEC_DIGEST}, current {actual_digest}"
        )
    return candidate


def runtime_spec_digest(task_id: str) -> str | None:
    """Return the verified digest only when ``task_id`` is the current dynamic view."""
    return (
        _RUNTIME_SPEC_DIGEST if _registered_runtime_spec(task_id) is not None else None
    )


def load_spec(task_id: str) -> dict:
    import json

    runtime_spec = _registered_runtime_spec(task_id)
    if runtime_spec is not None:
        return _apply_unit_rename(runtime_spec)
    p = spec_path(task_id)
    if not os.path.exists(p):
        raise SpecBuildError(f"no spec at {p}")
    with open(p) as f:
        return _apply_unit_rename(json.load(f))


def _apply_unit_rename(spec: dict) -> dict:
    """Rewrite ``human_program`` names that spec_privacy re-keyed but left behind in the prose.

    ``unit_rename`` is that pass's own ledger (original -> current). A program that still says
    ``[LookAt] unit_01(1)`` names an entity no longer in the roster, and ``command_grammar`` raises
    ``no scene entity for <unit_01>`` at BUILD time. Only names that are NOT current roster keys are
    rewritten, so the transform can turn an unresolvable name into a resolvable one and nothing
    else -- a rename whose source is still a live key is left strictly alone."""
    ren = {str(k): str(v) for k, v in (spec.get("unit_rename") or {}).items()}
    live = set(spec.get("roster", {}).get("containers") or [])
    ren = {k: v for k, v in ren.items() if k not in live and v in live}
    hp = str(spec.get("human_program") or "")
    if not ren or not hp:
        return spec
    for src, dst in sorted(ren.items(), key=lambda kv: -len(kv[0])):
        hp = re.sub(rf"(?<![\w<]){re.escape(src)}(?![\w>])", dst, hp)
        hp = hp.replace(f"<{src}>", f"<{dst}>")
    spec["human_program"] = hp
    return spec


def spec_ids() -> list[str]:
    if not os.path.isdir(SPEC_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(SPEC_DIR) if f.endswith(".json"))


# =====================================================================================
# recipe -- the authored, coordinate-free layer
# =====================================================================================
@dataclass(frozen=True)
class UnitRecipe:
    """One destination container: which baked category plays it + how its placard reads."""

    key: str  # abstract spec key ("unit_a")
    cat: str
    model: str
    png: str  # label PNG basename under GENERATED_LABEL_DIR
    title: str  # placard header, e.g. "COLD ZONE"
    # An OPEN receptacle (bay / cart / tote) has no door leaf, so its occupancy is countable from
    # the aisle. Every occupancy row depends on that: a count behind a closed door is not a cue.
    open_top: bool = False
    # Categories this role may fall back to when EVERY baked model of ``cat`` fails the container
    # ACCEPTANCE probe. Same semantic role, different family -- a cold unit stays a cold unit. This
    # is authored rather than inferred because "what else could play this part" is a task-design
    # judgement, and a wrong guess would change what the picture means.
    alt_cats: tuple[str, ...] = ()
    binding_source: str = "auto"
    pick_rank: int | None = None
    pick_name: str | None = None
    pick_prim_path: str | None = None
    pick_asset_sha256: str | None = None
    pick_instance_digest: str | None = None
    tint: str = ""  # plain colour word painted over the body -- see ``behavior_import.tint_rgb``
    room: str = ""  # hard binding for canonical BTS rows; empty keeps legacy rehoming
    move_mode: str = ""
    initially_open: bool = False
    siting_xy: tuple[float, float] | None = None
    siting_front_yaw_deg: float | None = None


@dataclass(frozen=True)
class ClassRecipe:
    """One INTRINSIC class (an acid is an acid on every seed): how it prints + which icon says it.

    Used by ``cue_mode="intrinsic"`` rows, where the spec authors the item->class map as a literal
    ``class_map`` and the UNIT placard is what has to be derived (from the answer key)."""

    key: str  # the spec's class token ("ox")
    title: str  # what the placard / decal prints ("OXIDISER")
    icon: str  # label_generator icon id for the item decal


@dataclass(frozen=True)
class NumberRecipe:
    """One NUMERIC cue channel: a per-key integer DRAWN from the seed and PRINTED on a label.

    This is the whole point of ``cue_number``: the value the scorer's capacity uses and the value a
    reader sees are the SAME draw, generated into the PNG at build time, so the label text and the
    answer key cannot drift apart."""

    channel: str  # the name the spec's cue_number expressions look up
    domain: str  # "units" | "items" -- which roster the keys come from
    keys: tuple[str, ...]
    kind: str  # "count" (a bare integer) | "date" (a printed date code)
    lo: int
    hi: int
    title: str
    sub: str = ""
    month: int = 5  # date codes render as 2026-<month>-<value>; fixed so the draw is the only variable


@dataclass(frozen=True)
class PreloadRecipe:
    """Stock that is ALREADY inside a unit at t=0 (PRESTAGE-INIT).

    The count is seeded, never authored, and it is re-measured off the built stage before it reaches
    ``BuildEnv`` -- a preload the scorer cannot see would silently shift every capacity."""

    unit: str
    cat: str
    model: str
    lo: int = 1
    hi: int = 2
    from_number: str = (
        ""  # channel whose per-unit drawn value IS the count (a FULL bay)
    )
    keys: tuple[str, ...] = ()
    initial_space: str = ""  # exact measured <unit>_spaceN; empty keeps automatic packing
    tint: str = ""
    front_facing: bool = False
    front_show_back: bool = False
    spot: tuple[float, float] | None = None


@dataclass(frozen=True)
class ItemRecipe:
    """One graspable: which baked category plays it + which decal PNGs it may carry."""

    key: str
    cat: str
    model: str
    pngs: tuple[
        str, ...
    ]  # one per possible cue class, index-aligned with Recipe.classes
    icons: tuple[str, ...]  # label_generator icon ids, index-aligned with Recipe.classes
    initial_place: str = ""
    # WHERE ON THE SURFACE the author put it: u across the front edge (-1 far left .. +1 far right),
    # v into the depth (0 front edge .. 1 back). None = unauthored, and ``seat_items`` then spaces the
    # row evenly exactly as it always has. This is the ONE piece of t=0 geometry an author may state,
    # and it exists because "these over here, those over there" is not decoration: a story whose
    # difficulty is telling two piles apart is unbuildable when the builder rows everything out at
    # equal pitch from the list index. A whole compiled batch wrote 113 of these and none was read.
    spot: tuple[float, float] | None = None
    tint: str = ""  # plain colour word painted over the body -- see ``behavior_import.tint_rgb``
    affordances: tuple[str, ...] = ()
    scale: float | tuple[float, float, float] = 1.0
    workstation_posture: str = ""
    workstation_screen_style: str = ""


@dataclass(frozen=True)
class PartRecipe:
    """One flat-pack part: which baked category plays it + how the MANUAL names its ROLE.

    ``label`` is the only authored string, and it is rule-level on purpose -- the manual pairs it
    with a per-seed LETTER, so the assistant still has to look at a board to learn which physical
    plank that role is. The connector names are NOT authored: they are read off the part's own
    BEHAVIOR metadata (``derive_mates``)."""

    key: str  # abstract spec key ("bd_0")
    cat: str
    model: str
    label: str
    affordances: tuple[str, ...] = ()
    initial_place: str = ""
    spot: tuple[float, float] | None = None
    tint: str = ""


@dataclass(frozen=True)
class Mate:
    """One scored ATTACHED edge, fully resolved against the two parts' own BEHAVIOR metadata."""

    child: str
    parent: str
    anon: str  # the spec's KEY-ANON connector base ("shelfsideL")
    child_point: str  # real metadata key ("shelfsideyujrmwM")
    parent_point: str  # real metadata key ("shelfsideyujrmwF")
    cp_idx: int
    pp_idx: int


@dataclass(frozen=True)
class Recipe:
    """The full non-geometric world description for one task id.

    Two shapes are supported and they are mutually exclusive: a CONTAINER row fills
    ``units``/``items``/``support_cats`` and is scored by ``INSIDE``; an ASSEMBLY row fills
    ``parts``/``manual_key`` and is scored by ``ATTACHED``."""

    task_id: str
    scene: str
    room: str = ""  # BEHAVIOR room instance the whole episode happens in
    support_cats: tuple[
        str, ...
    ] = ()  # categories that may hold the graspables ("the pass")
    units: tuple[UnitRecipe, ...] = ()
    items: tuple[ItemRecipe, ...] = ()
    classes: tuple[
        str, ...
    ] = ()  # the cue vocabulary, permuted over units/parts per seed
    # Which units the seeded class permutation moves over. Empty = every unit (slice A/B). A row whose
    # ``no_share_group`` names a PAIR of units must restrict the shuffle to that pair, or on some seeds
    # neither incompatible class lands on the pair and the constraint is unfireable that seed.
    class_units: tuple[str, ...] = ()
    cue_channel: str = "item_class"
    placard_title: str = "COLD ZONE"
    # How the cue is carried. "target": the class vocabulary permutes over the units and the item
    # decal is DERIVED from its answer-key target (slice A). "intrinsic": the spec authors the
    # item->class map, so the UNIT placard is derived instead and the seed permutes which physical
    # box plays which key. "number": the cue is a drawn integer printed on a card.
    cue_mode: str = "target"
    class_defs: tuple[ClassRecipe, ...] = ()  # cue_mode "intrinsic"
    numbers: tuple[NumberRecipe, ...] = ()  # cue_mode "number"
    preloads: tuple[PreloadRecipe, ...] = ()
    # "distinct": every unit is a different category, so the role is carried by geometry as well as
    # by the placard. "twin": the units are deliberately the SAME body and only the label tells them
    # apart -- the honest shape for a row whose difficulty IS reading the label. "pair_twin": the
    # ``class_units`` pair shares ONE body (only its placard tells the two apart, which is what a
    # ``no_share_group`` over that pair needs) and every other unit is a distinct body.
    unit_shape: str = "distinct"
    # Unscored graspables: staged, decal'd and inspected exactly like the scored ones, never a target.
    # They are what makes ``set(pool:items) minus key_subjects`` a non-empty set of REAL bodies, so a
    # ``not_inside{the spares}`` is a constraint about the picture rather than about an empty set.
    decoys: tuple[ItemRecipe, ...] = ()
    # The band a DECOY prints. Kept OUT of ``classes`` on purpose: ``classes`` is one entry per unit
    # (``zone_permutation`` enforces that), and a spare's band names no unit at all -- which is what
    # makes it a spare. It occupies artwork slot len(classes) in every staged item's ``pngs``.
    decoy_class: str = ""
    # How many staging surfaces the graspables are split across. A row whose authored program looks at
    # ``tray_1`` and ``tray_2`` has TWO, and dropping one would drop an authored [LookAt].
    n_supports: int = 1
    support_keys: tuple[str, ...] = ()  # the LookAt keys those surfaces answer to
    support_bindings: tuple[str, ...] = ()
    support_names: tuple[str, ...] = ()
    support_models: tuple[str, ...] = ()
    support_siting_xy: tuple[tuple[float, float] | None, ...] = ()
    support_siting_front_yaws: tuple[float | None, ...] = ()
    support_approach_yaws: tuple[float | None, ...] = ()
    support_operator_stands: tuple[tuple[float, float] | None, ...] = ()
    support_items: tuple[tuple[str, ...], ...] = ()
    support_capabilities: tuple[frozenset[str], ...] = ()
    support_title: str = "STAGING TRAY"
    card_key: str = (
        ""  # a free-standing rule board (the ledger card), reusing the manual body
    )
    card_title: str = ""
    # Room-level cards the human must read when the AUTHORED program contains no [LookAt] at all.
    # Only consulted in that case, so it can never displace an authored observation.
    observe: tuple[str, ...] = ()
    # --- assembly rows only ---------------------------------------------------------------
    parts: tuple[PartRecipe, ...] = ()
    manual_key: str = (
        ""  # the rule-level MANUAL entity, i.e. THE CUE for an assembly row
    )
    part_png: str = ""  # printf-style stem for the per-letter board decals ("part_%s")
    hull: str = "furniture"  # "furniture" = the room's own furniture hull; "floor" = the floor prim
    look_zoom: bool = True
    rooms: tuple[str, ...] = ()  # canonical BTS room set; empty for legacy recipes
    support_rooms: tuple[str, ...] = ()
    frozen_bindings: bool = False
    initially_open_room_doors: bool = False
    initially_open_room_door_names: tuple[str, ...] = ()
    initially_open_room_door_reverse_swing_names: tuple[str, ...] = ()

    def is_assembly(self) -> bool:
        return bool(self.parts)

    def keys(self) -> set[str]:
        return (
            {u.key for u in self.units}
            | {i.key for i in self.items}
            | {p.key for p in self.parts}
        )

    def staged(self) -> tuple[ItemRecipe, ...]:
        """Every graspable on the bench, scored or not, in staging order."""
        return tuple(self.items) + tuple(self.decoys)

    def permuted_units(self) -> tuple[str, ...]:
        return self.class_units or tuple(u.key for u in self.units)


# Same-role fallback families for the container ACCEPTANCE probe. Every one of these was measured
# INSIDE-clean with zero interpenetration on a live stage; the probe still re-measures each of them
# in the row's own room, so this list can only ever OFFER a body, never bless one.
_COLD_ALTS: tuple[str, ...] = (
    "freezer",
    "grocery_fridge",
    "display_fridge",
    "fridge",
    "metal_bottom_cabinet",
    "bottom_cabinet",
)
_LAB_ALTS: tuple[str, ...] = (
    "metal_bottom_cabinet",
    "bottom_cabinet",
    "cabinet_base",
    "locker",
)


def _t1_dinr_01() -> Recipe:
    """Prep-fridge loading: four allergen-zoned cold units of FOUR PHYSICALLY DISTINCT
    categories (so the zone role is carried by geometry as well as by the placard), one
    lidded tub on the pass whose lid decal names the allergen. Categories/models are the
    baked cold-storage families that carry a revolute base_link door, so the uniform
    ``[Open] ... [Close]`` bracket in the authored program is legal on all four."""
    zones = ("NUTS", "GLUTEN", "DAIRY", "SOY")
    icons = ("allergen_nuts", "allergen_gluten", "allergen_milk", "allergen_soy")
    stem = "t1_dinr_01__"
    units = (
        UnitRecipe(
            "unit_a",
            "display_fridge",
            "wlxyjn",
            f"{stem}zone_placards_1.png",
            "COLD ZONE",
            alt_cats=_COLD_ALTS,
        ),
        UnitRecipe(
            "unit_b",
            "wine_fridge",
            "tgcuwh",
            f"{stem}zone_placards_2.png",
            "COLD ZONE",
            alt_cats=_COLD_ALTS,
        ),
        UnitRecipe(
            "unit_c",
            "metal_bottom_cabinet",
            "mvsgdm",
            f"{stem}zone_placards_3.png",
            "COLD ZONE",
            alt_cats=_COLD_ALTS,
        ),
        UnitRecipe(
            "unit_d",
            "fridge",
            "juwaoh",
            f"{stem}zone_placards_4.png",
            "COLD ZONE",
            alt_cats=_COLD_ALTS,
        ),
    )
    item = ItemRecipe(
        "item_1",
        "tupperware",
        "mkstwr",
        tuple(f"{stem}allergen_icon_lid_{i + 1}.png" for i in range(4)),
        icons,
    )
    return Recipe(
        task_id="T1-DINR-01",
        scene="restaurant_diner",
        room="kitchen_0",
        support_cats=("commercial_kitchen_table",),
        units=units,
        items=(item,),
        classes=zones,
    )


def _t3_hall_01() -> Recipe:
    """Flat-pack shelf build: six boards laid out flat in an empty hall, five scored ATTACHED mates
    plus the closing ``[Raise]``. ``bd_2`` and ``bd_5`` are the SAME category and the same silhouette
    (``bookcase_side``, 0.26 x 2.02 m); only ``bd_5``'s metadata exposes the ``shelfbase`` female, and
    only the per-seed letter decal + the rule-level manual say which plank that is. The manual states
    the ROLE of each letter and the ORDER of the mates, and never names a board."""
    parts = (
        PartRecipe("bd_0", "bookcase_back", "gjsnrt", "BACK PANEL - LAY IT FLAT"),
        PartRecipe("bd_1", "bookcase_top", "pfiole", "TOP PANEL"),
        PartRecipe("bd_2", "bookcase_side", "yujrmw", "PLAIN SIDE PANEL"),
        PartRecipe("bd_3", "bookcase_shelf", "ymtnqa", "SHELF"),
        PartRecipe("bd_4", "bookcase_baseboard", "hlhneo", "BASEBOARD"),
        PartRecipe("bd_5", "bookcase_side", "bxfkjj", "SIDE WITH THE BASEBOARD SOCKET"),
    )
    return Recipe(
        task_id="T3-HALL-01",
        scene="hall_arch_wood",
        room="empty_room_0",
        parts=parts,
        classes=("A", "B", "C", "D", "E", "F"),
        cue_channel="part_letter",
        manual_key="manual",
        part_png="part_%s.png",
        hull="floor",
        placard_title="FLAT-PACK SHELF UNIT",
        # A [LookAt] on a Role.GRASPABLE under look_zoom GRABS the object and raises it to the face
        # (``_look_at_inspect_grabbable``). The authored program looks at two 2.02 m planks, so the
        # zoom inspect is off here and the boards are read from the standing FPV, as the shipped
        # bookcase build does.
        look_zoom=False,
    )


def _t1_chem_02() -> Recipe:
    """Pair exclusion: three reagent bottles, two under-filled lockers, one question per bottle.

    The classes are INTRINSIC (the spec authors ``group_of_expr = class_map``: an oxidiser is an
    oxidiser on every seed), so the derivation runs the other way from T1-DINR-01 -- the UNIT placard
    is derived from the answer key, and BG-PERMUTE moves which physical cabinet plays which key.
    The three bottles are the SAME model: only the GHS pictogram tells them apart."""
    stem = "t1_chem_02__"
    units = (
        UnitRecipe(
            "unit_a",
            "locker",
            "gemgfz",
            f"{stem}locker_placard_a.png",
            "HAZARD LOCKER",
            alt_cats=_LAB_ALTS,
        ),
        UnitRecipe(
            "unit_c",
            "metal_bottom_cabinet",
            "mvsgdm",
            f"{stem}locker_placard_c.png",
            "HAZARD LOCKER",
            alt_cats=_LAB_ALTS,
        ),
    )
    items = tuple(
        ItemRecipe(k, "reagent_bottle", "tnjpsf", (), ())
        for k in ("item_1", "item_2", "item_3")
    )
    return Recipe(
        task_id="T1-CHEM-02",
        scene="school_chemistry",
        room="chemistry_lab_0",
        support_cats=("lab_table",),
        units=units,
        items=items,
        cue_mode="intrinsic",
        class_defs=(
            ClassRecipe("ox", "OXIDISER", "ghs_flame_circle"),
            ClassRecipe("flam", "FLAMMABLE", "ghs_flame"),
        ),
        placard_title="HAZARD LOCKER",
        observe=("unit_a", "unit_c"),
    )


def _t1_gcaf_01() -> Recipe:
    """Planogram facings: ONE tagged shelf bay whose printed limit is a DRAWN number, pre-loaded to
    exactly that limit, plus a back-stock trolley. Both destinations are OPEN receptacles, so the
    occupancy is countable from the aisle with no door to open -- which is what makes the count a
    cue at all. ``from_number`` binds the preload to the printed limit, so the bay the tag says is
    full really is full and ``capacity`` is the constraint the wrong answer trips."""
    stem = "t1_gcaf_01__"
    # TWIN bays on purpose. A ``shopping_cart`` back-stock was tried first and never came to rest: the
    # released case slid on the cart's wheeled hull and the post-place settle did not converge, hanging
    # the run before it could score. Identical bays are also the honest shape here -- the cue is a
    # printed NUMBER, so the body must not leak which destination is which.
    units = (
        UnitRecipe(
            "unit_a",
            "grocery_shelf",
            "fgfrwb",
            f"{stem}shelf_tag",
            "FACINGS MAX",
            open_top=True,
        ),
        UnitRecipe(
            "unit_e",
            "grocery_shelf",
            "fgfrwb",
            f"{stem}backstock.png",
            "BACK-STOCK",
            open_top=True,
        ),
    )
    return Recipe(
        task_id="T1-GCAF-01",
        scene="grocery_store_cafe",
        room="grocery_store_0",
        support_cats=("checkout_counter",),
        units=units,
        items=(ItemRecipe("item_1", "carton_of_milk", "xmugpm", (), ()),),
        cue_mode="number",
        numbers=(
            NumberRecipe(
                "facing_limit",
                "units",
                ("unit_a",),
                "count",
                2,
                4,
                "FACINGS MAX",
                "this bay",
            ),
        ),
        preloads=(
            PreloadRecipe(
                "unit_a", "carton_of_milk", "atyqub", from_number="facing_limit"
            ),
        ),
        unit_shape="twin",
        placard_title="FACINGS MAX",
    )


def _t3_groc_02() -> Recipe:
    """FIFO back-stock: four date-coded cartons, three IDENTICAL open bays that already hold older
    stock. Each bay's range card lists the use-by codes it takes, and both the card and the carton
    decals are printed from the SAME per-seed draw. ``cap = preload + key_load`` means the oracle
    fills every bay exactly to its cap, so one carton in the wrong bay overflows it."""
    stem = "t3_groc_02__"
    bays = (("u_front", "FRONT BAY"), ("u_mid", "MID BAY"), ("u_back", "BACK BAY"))
    units = tuple(
        UnitRecipe(k, "grocery_shelf", "fgfrwb", f"{stem}bay_{k}", t, open_top=True)
        for k, t in bays
    )
    items = tuple(
        ItemRecipe(k, "carton_of_milk", "xmugpm", (), ())
        for k in ("obj_1", "obj_2", "obj_3", "obj_4")
    )
    return Recipe(
        task_id="T3-GROC-02",
        scene="grocery_store_half_stocked",
        room="grocery_store_0",
        support_cats=("countertop",),
        units=units,
        items=items,
        cue_mode="number",
        numbers=(
            NumberRecipe(
                "date_code",
                "items",
                tuple(i.key for i in items),
                "date",
                4,
                27,
                "USE BY",
                "date code",
            ),
        ),
        preloads=tuple(
            PreloadRecipe(k, "carton_of_milk", "atyqub", lo=1, hi=2) for k, _t in bays
        ),
        unit_shape="twin",
        placard_title="BAY RANGE CARD",
    )


def _t6_rdin_01() -> Recipe:
    """Allergen put-away (PROSE shape A -- placard sweep, per-item survey, per-item placement).

    The seeded class permutation is restricted to ``unit_a``/``unit_b``: those are the two units the
    spec's ``no_share_group`` names, and a shuffle over all four would land both incompatible classes
    off the pair on most seeds, leaving the constraint unfireable that seed. ``unit_a``'s cap is
    ``preload + 1``, so the one contended line is exactly full once its keyed pack is in."""
    stem = "t6_rdin_01__"
    zones = ("ALLERGEN-FREE", "ALLERGEN-BEARING", "FROZEN", "HOT LINE - NO STORAGE")
    icons = ("allergen_soy", "allergen_nuts", "allergen_gluten", "allergen_milk")
    # BODIES PINNED BY MEASUREMENT, not by taste. ``wine_fridge/tgcuwh`` (136 mm), ``fridge/juwaoh``
    # (6 mm) and ``display_fridge/wlxyjn`` (19 mm) all accept a [PutInside] with ZERO failed
    # primitives and then eject the object -- juwaoh launched a tub 210 m under the floor at 320 m/s
    # and cost this row's control the constraint it was aimed at. The four below are the bodies the
    # container-ACCEPTANCE probe measured at 0 mm on a live stage; the probe re-measures them here
    # anyway, in this room, so this comment can never become the thing that is trusted.
    units = (
        UnitRecipe(
            "unit_a",
            "fridge",
            "jtqazu",
            f"{stem}line_placard_1.png",
            "PREP LINE",
            alt_cats=_COLD_ALTS,
        ),
        UnitRecipe(
            "unit_b",
            "metal_bottom_cabinet",
            "mvsgdm",
            f"{stem}line_placard_2.png",
            "PREP LINE",
            alt_cats=_COLD_ALTS,
        ),
        UnitRecipe(
            "unit_c",
            "grocery_shelf",
            "fgfrwb",
            f"{stem}line_placard_3.png",
            "PREP LINE",
            open_top=True,
            alt_cats=_COLD_ALTS,
        ),
        UnitRecipe(
            "unit_forbidden",
            "oven",
            "amblrk",
            f"{stem}line_placard_4.png",
            "PREP LINE",
            alt_cats=("microwave", "dishwasher", "metal_bottom_cabinet"),
        ),
    )
    items = tuple(
        ItemRecipe(
            k,
            "tupperware",
            "mkstwr",
            tuple(f"{stem}pack_icon_{i + 1}.png" for i in range(4)),
            icons,
        )
        for k in ("item_1", "item_2", "item_3", "item_4")
    )
    return Recipe(
        task_id="T6-RDIN-01",
        scene="restaurant_diner",
        # The DINING room, not the kitchen: with four units sited the compact kitchen splits into
        # three nav components and the walk to the bench is path_blocked at render time (measured:
        # "[bench][WARN] AABB-proxy nav components ... [2, 4, 6]"). Nothing in the row is about the
        # kitchen -- the delivery is put away off the pass.
        room="dining_room_0",
        support_cats=("pedestal_table", "booth", "bench"),
        units=units,
        items=items,
        classes=zones,
        class_units=("unit_a", "unit_b"),
        preloads=(PreloadRecipe("unit_a", "bowl", "fiarri", lo=1, hi=2),),
        placard_title="PREP LINE",
    )


def _t6_rhot_01() -> Recipe:
    """HACCP leftovers put-away (PROSE shape A at N=6). ``unit_c`` is an OPEN shelf pre-loaded with
    exactly three cases, so the spec's ``capacity{unit_c: 5}`` is occupants(3) + key load(2) and the
    oracle fills it to the brim -- one misrouted container overflows it and nothing else."""
    stem = "t6_rhot_01__"
    zones = ("RAW", "RTE", "SHELF-STABLE", "HOT PASS - NO STORAGE")
    icons = ("allergen_nuts", "allergen_milk", "allergen_gluten", "allergen_soy")
    units = (
        # OPEN cold display, not a doored fridge. Both baked fridges swallowed item_1 here with
        # ZERO failed primitives (juwaoh at 320 m/s, jtqazu at 312 m/s, both 200 m under the floor)
        # and the executor's own drop probe reported EVERY interior board of both as
        # "blocked(... collision__N)" while resting the object ON that same mesh. An open
        # receptacle takes the interior_drop path instead of the board path -- the path unit_c has
        # always used here without a single ejection. The row's program has no [Open]/[Close], so
        # nothing about it depends on there being a door.
        UnitRecipe(
            "unit_a",
            "grocery_fridge",
            "vzwcql",
            f"{stem}band_1.png",
            "HANDLING CLASS",
            open_top=True,
            alt_cats=_COLD_ALTS,
        ),
        UnitRecipe(
            "unit_b",
            "grocery_fridge",
            "vzwcql",
            f"{stem}band_2.png",
            "HANDLING CLASS",
            open_top=True,
            alt_cats=_COLD_ALTS,
        ),
        UnitRecipe(
            "unit_c",
            "grocery_shelf",
            "fgfrwb",
            f"{stem}band_3.png",
            "HANDLING CLASS",
            open_top=True,
        ),
        UnitRecipe(
            "unit_forbidden",
            "oven",
            "amblrk",
            f"{stem}band_4.png",
            "HANDLING CLASS",
            alt_cats=("microwave", "dishwasher", "metal_bottom_cabinet"),
        ),
    )
    items = tuple(
        ItemRecipe(
            k,
            "tupperware",
            "mkstwr",
            tuple(f"{stem}lid_{i + 1}.png" for i in range(4)),
            icons,
        )
        for k in ("item_1", "item_2", "item_3", "item_4", "item_5", "item_6")
    )
    return Recipe(
        task_id="T6-RHOT-01",
        scene="restaurant_hotel",
        room="kitchen_0",
        support_cats=("commercial_kitchen_table",),
        units=units,
        items=items,
        classes=zones,
        class_units=("unit_a", "unit_b"),
        preloads=(PreloadRecipe("unit_c", "bowl", "fiarri", lo=3, hi=3),),
        unit_shape="pair_twin",
        placard_title="HANDLING CLASS",
    )


def _t6_rhot_02() -> Recipe:
    """Wash-list set difference (PROSE shape C -- named non-roster cue props, then a loop over the
    REMAINING items). The requisition card is a free-standing board listing FIVE codes; one of them
    is the piece already inside the machine, so the card minus the two trays is the answer and a card
    printed only from the unplaced pieces would hand it over."""
    stem = "t6_rhot_02__"
    keys = ("item_1", "item_2", "item_3", "item_4", "item_5", "item_6", "item_7")
    # Re-keyed with spec_privacy.PINNED_RENAME: the WASH RACK (the answer) is unit_b and the
    # OVERFLOW RACK is unit_a, so the alphabetically-first key is the DECOY. The placards and the
    # bodies do not move -- only the abstract key does.
    units = (
        UnitRecipe(
            "unit_b",
            "grocery_shelf",
            "fgfrwb",
            f"{stem}unit_a.png",
            "WASH RACK",
            open_top=True,
        ),
        UnitRecipe(
            "unit_a",
            "shelf",
            "yursyw",
            f"{stem}unit_o.png",
            "OVERFLOW RACK",
            open_top=True,
        ),
    )
    return Recipe(
        task_id="T6-RHOT-02",
        scene="restaurant_hotel",
        room="kitchen_0",
        support_cats=("commercial_kitchen_table",),
        units=units,
        items=tuple(ItemRecipe(k, "plate", "aewthq", (), ()) for k in keys),
        cue_mode="number",
        numbers=(
            NumberRecipe(
                "item_code",
                "items",
                keys + ("pre_unit_b_0",),
                "count",
                1000,
                9999,
                "ITEM CODE",
                "stamped",
            ),
        ),
        preloads=(PreloadRecipe("unit_b", "plate", "aewthq", lo=1, hi=1),),
        n_supports=2,
        support_keys=("tray_1", "tray_2"),
        support_title="STAGING TRAY",
        card_key="card",
        card_title="WASH LIST",
        placard_title="WASH RACK",
    )


def _t6_hsf_02() -> Recipe:
    """Priority parcel under a hard action budget (PROSE shape B -- survey ALL twelve, place only the
    MANDATORY four). Eight of the twelve are DECOYS: they carry a real body and a real decal, are in
    ``pool:items``, and are what makes ``not_inside{items minus key_subjects}`` a constraint about the
    picture. The row is the BENCH_ORDER carrier: with a final-state scorer an ordering defect is only
    scoreable through a budget, and this is the wave-1 row that publishes one."""
    stem = "t6_hsf_02__"
    zones = ("PRIORITY - PARCEL", "RESTRICTED - QUARANTINE", "SPARE - PLAIN BAND")
    icons = ("allergen_nuts", "allergen_gluten", "allergen_soy")
    units = (
        # The spec asks for an IMPORTED carton as the parcel. The only baked cartons are 0.23 m
        # (too small for four 0.19 m pouches) or 1.0-1.2 m (no room instance in this scene has a
        # metre of clear floor beside a work surface), so the parcel is played by a compact cupboard.
        # Declared, not hidden: the scored term is INSIDE(item, unit_a) and the body does not enter it.
        UnitRecipe(
            "unit_a",
            "bottom_cabinet",
            "dajebq",
            f"{stem}band_1.png",
            "PARCEL",
            alt_cats=("bottom_cabinet_no_top", "metal_bottom_cabinet", "cabinet_base"),
        ),
        UnitRecipe(
            "unit_b",
            "bottom_cabinet",
            "ccjinv",
            f"{stem}band_2.png",
            "QUARANTINE",
            alt_cats=("bottom_cabinet_no_top", "metal_bottom_cabinet", "cabinet_base"),
        ),
    )

    def _bag(k: str) -> ItemRecipe:
        return ItemRecipe(
            k,
            "bag_of_jerky",
            "wblype",
            tuple(f"{stem}band_item_{i + 1}.png" for i in range(3)),
            icons,
        )

    keyed = ("item_2", "item_5", "item_8", "item_11")
    spares = tuple(f"item_{i}" for i in range(1, 13) if f"item_{i}" not in keyed)
    return Recipe(
        task_id="T6-HSF-02",
        scene="house_single_floor",
        room="utility_room_0",
        support_cats=("countertop", "breakfast_table", "bar"),
        units=units,
        items=tuple(_bag(k) for k in keyed),
        decoys=tuple(_bag(k) for k in spares),
        classes=zones[:2],
        decoy_class=zones[2],
        class_units=("unit_a", "unit_b"),
        n_supports=2,
        placard_title="PARCEL LINE",
    )


def _bay_units(
    stem: str, keys, cat: str, model: str, title: str, alts
) -> tuple[UnitRecipe, ...]:
    """N OPEN destination bays of ONE body. Deliberately identical: these rows publish five to nine
    destinations, and the honest shape for a bank that big is one where only the PLACARD tells the
    bays apart -- distinct silhouettes at N=9 would hand a reader a second, unintended channel.
    Open-top because none of these four rows' authored programs contains an [Open] / [Close]."""
    return tuple(
        UnitRecipe(
            k, cat, model, f"{stem}bay_{i + 1}.png", title, open_top=True, alt_cats=alts
        )
        for i, k in enumerate(keys)
    )


def _bottles(keys, stem: str, n_class: int, icon: str) -> tuple[ItemRecipe, ...]:
    return tuple(
        ItemRecipe(
            k,
            "reagent_bottle",
            "tnjpsf",
            tuple(f"{stem}decal_{i + 1}.png" for i in range(n_class)),
            tuple(icon for _ in range(n_class)),
        )
        for k in keys
    )


_SHELF_ALTS: tuple[str, ...] = (
    "grocery_shelf",
    "bookcase",
    "shelf",
    "commercial_kitchen_shelf",
)

# GHS bands, one per bay. Nine of them: a nine-bay bank needs nine distinct placards or the
# seeded permutation cannot be one class per unit.
_GHS_BANDS: tuple[str, ...] = (
    "OXIDISER",
    "CORROSIVE ACID",
    "FLAMMABLE",
    "TOXIC",
    "CAUSTIC BASE",
    "WATER-REACTIVE",
    "PEROXIDE FORMER",
    "COMPRESSED GAS",
    "INERT / GENERAL",
)
_CALL_RANGES: tuple[str, ...] = (
    "QA 1-99",
    "QC 100-199",
    "QD 200-299",
    "QH 300-399",
    "QK 400-499",
    "QL 500-599",
    "QP 600-699",
)


def _t3_chem_01() -> Recipe:
    """NINE reagent bays, six bottles: the widest destination bank in the portfolio, and the row
    the greedy one-at-a-time siter could not place at all (it declares 9 containers against a
    measured wall of 4). Everything above the siting is slice A's proven ``target`` shape."""
    stem = "t3_chem_01__"
    keys = tuple(f"unit_{c}" for c in "abcdefghi")
    return Recipe(
        task_id="T3-CHEM-01",
        scene="school_chemistry",
        room="chemistry_lab_0",
        support_cats=("lab_table", "desk"),
        units=_bay_units(stem, keys, "shelf", "yursyw", "HAZARD BAY", _SHELF_ALTS),
        items=_bottles(
            ("obj_1", "obj_2", "obj_3", "obj_4", "obj_5", "obj_6"), stem, 9, "ghs_flame"
        ),
        classes=_GHS_BANDS,
        unit_shape="twin",
        placard_title="HAZARD BAY",
    )


def _t3_chem_02() -> Recipe:
    """Six bays, five bottles, two of one class. ``no_share_group`` reads the spec's own class map,
    so the pair is a property of the row rather than of the seed; ``not_inside`` names the two bays
    no answer reaches."""
    stem = "t3_chem_02__"
    keys = tuple(f"unit_{c}" for c in "abcdef")
    return Recipe(
        task_id="T3-CHEM-02",
        scene="school_chemistry",
        room="chemistry_lab_0",
        support_cats=("lab_table", "desk"),
        units=_bay_units(stem, keys, "shelf", "yursyw", "HAZARD BAY", _SHELF_ALTS),
        items=_bottles(
            ("obj_1", "obj_2", "obj_3", "obj_4", "obj_5"), stem, 6, "ghs_flame"
        ),
        classes=_GHS_BANDS[:6],
        unit_shape="twin",
        placard_title="HAZARD BAY",
    )


def _t1_offc_01() -> Recipe:
    """Battery segregation: five kerbside bins, one loose cell whose printed chemistry code is the
    only discriminator. The spec's own ``assets_needed`` records that this scene contains NO bin at
    all -- all five bodies are author-added into a room the scene never furnished, which is why the
    row needs a solver that can pick the ROOM as well as the slots. The authored ``copy_room_0`` has
    no work surface at all; the solver is expected to say so and move."""
    stem = "t1_offc_01__"
    streams = ("LI-ION", "NIMH", "CR2032", "ALKALINE", "GENERAL WASTE")
    icons = (
        "battery_liion",
        "battery_nimh",
        "battery_coin",
        "battery_alkaline",
        "no_batteries",
    )
    keys = ("unit_a", "unit_b", "unit_c", "unit_d", "unit_e")
    return Recipe(
        task_id="T1-OFFC-01",
        scene="office_cubicles_right",
        room="copy_room_0",
        support_cats=("desk", "coffee_table", "conference_table", "countertop"),
        units=_bay_units(
            stem,
            keys,
            "recycling_bin",
            "mdojox",
            "BATTERY STREAM",
            ("compost_bin", "trash_can", "shopping_cart", "carton"),
        ),
        items=(
            ItemRecipe(
                "item_1",
                "battery",
                "dcjyzg",
                tuple(f"{stem}cell_{i + 1}.png" for i in range(5)),
                icons,
            ),
        ),
        classes=streams,
        unit_shape="twin",
        placard_title="BATTERY STREAM",
    )


def _t3_hdfu_01() -> Recipe:
    """Six books, SEVEN call-number bays in a domestic upstairs study: the densest floor plan the
    siter meets, and the one where the authored bedroom is plainly too small for the bank."""
    stem = "t3_hdfu_01__"
    keys = tuple(f"unit_{c}" for c in "abcdefg")
    return Recipe(
        task_id="T3-HDFU-01",
        scene="house_double_floor_upper",
        room="bedroom_1",
        support_cats=("bed", "desk", "coffee_table", "breakfast_table"),
        units=_bay_units(
            stem, keys, "shelf", "yursyw", "CALL-NUMBER RANGE", _SHELF_ALTS
        ),
        items=tuple(
            ItemRecipe(
                k,
                "hardback",
                "acbrnv",
                tuple(f"{stem}spine_{i + 1}.png" for i in range(7)),
                tuple("call_number" for _ in range(7)),
            )
            for k in ("obj_1", "obj_2", "obj_3", "obj_4", "obj_5", "obj_6")
        ),
        classes=_CALL_RANGES,
        unit_shape="twin",
        placard_title="CALL-NUMBER RANGE",
    )


def recipes() -> dict[str, Recipe]:
    """Explicit mapping (no import-time registration side effects)."""
    return {
        r.task_id: r
        for r in (
            _t1_dinr_01(),
            _t3_hall_01(),
            _t1_chem_02(),
            _t1_gcaf_01(),
            _t3_groc_02(),
            _t6_rdin_01(),
            _t6_rhot_01(),
            _t6_rhot_02(),
            _t6_hsf_02(),
            # 5+ destination banks: only siteable since the set-level solver replaced the greedy one
            _t3_chem_01(),
            _t3_chem_02(),
            _t1_offc_01(),
            _t3_hdfu_01(),
        )
    }


# =====================================================================================
# DERIVED recipes -- a scene-CAPABILITY query keyed on the roster
#
# The thirteen recipes above are PINS: rows whose bodies were hand-measured, and where a pin
# exists it WINS. Every other spec is resolved here, from four measured sources and no per-row
# authoring:
#
#   * the spec's own ``roster`` + ``unit_roles`` (the pre-anonymisation semantic key spec_privacy
#     kept, e.g. ``unit_cold`` / ``sharps_bin``) + its free-text ``assets_needed``;
#   * the BAKED cache's per-model metadata -- ``role_hint``, ``bbox_size``, ``mass``, ``joints``
#     -- which is where "has a drivable door" and "is graspable" are MEASURED rather than assumed;
#   * the frozen scene-capability index -- measured rooms, door containers,
#     open_receptacles, work_surfaces), measured off the scene files;
#   * ``behavior_import.scene_objects`` -- the scene's own inventory with ``in_rooms``, so
#     "prefer what this scene already contains, in a room that can stage the graspables" is a
#     query rather than a guess.
#
# Determinism: every step is a sorted list or a ``random.Random(f"{task_id}:...")`` draw, so the
# same spec resolves to the same assets forever. Failure is ``RoleUnsatisfied``, which names the
# role, the scene and everything tried -- a role that cannot be satisfied never degrades into a
# substitute body, because a body that does not carry the role changes what the task tests.
# =====================================================================================
class RoleUnsatisfied(SpecBuildError):
    """No baked asset satisfies one of a spec's abstract roles in its own scene."""

    def __init__(
        self, task_id: str, key: str, role: str, scene: str, tried: list[str]
    ) -> None:
        super().__init__(
            f"{task_id}: roster key {key!r} needs the role {role!r} and nothing in scene "
            f"{scene!r} or in the baked cache satisfies it. Tried, in order: {tried}. Refusing "
            f"to substitute a body that does not carry the role -- that changes what is tested."
        )
        self.key, self.role, self.scene, self.tried = key, role, scene, list(tried)


# Affordance bands, all read off MEASURED metadata rather than tuned. The receptacle floor is the
# narrowest body the acceptance probe has ever cleared (``bottom_cabinet/dajebq``, 0.278 m) and the
# graspable ceiling the largest staged one (``tupperware/mkstwr``, 0.22 m); the mass terms are what
# separate a piece of furniture from clutter of the same size. ``_selftest_derived`` re-asserts that
# every probe-clean body still survives these numbers, so tightening one cannot pass silently.
RECEPTACLE_MIN_SPAN_M: float = 0.27
RECEPTACLE_MIN_HEIGHT_M: float = 0.40
RECEPTACLE_MAX_SPAN_M: float = 3.0
RECEPTACLE_MIN_MASS_KG: float = 3.0
# A corrugated packing box is intentionally low but still a real floor-level
# receptacle. Keep this exception category-scoped so trays and portable food
# containers do not silently enter the fixed-unit bank.
LOW_OPEN_RECEPTACLE_MIN_HEIGHT_M: float = 0.20
LOW_OPEN_RECEPTACLE_CATEGORIES: frozenset[str] = frozenset({"packing_box"})
# Raised from 0.40 on 2026-08-16. The old span excluded 64 categories that weigh 3 kg or less
# -- pizza_box, baking_sheet, rolling_pin, ice_bucket, wicker_basket, coffee_maker -- purely
# for being a hand's width too long, while the MASS gate below is what actually decides
# whether one hand can carry a thing. Deliberately NOT raised further: at 0.6 m the list
# starts admitting furniture, and a bicycle or a desk top is something a person CARRIES with
# two hands, which is a motion the executor does not have.
GRASPABLE_MAX_SPAN_M: float = 0.55
GRASPABLE_MAX_MASS_KG: float = 4.0
# The LIGHT-AND-LONG lane, added 2026-08-17 alongside dropping the role-hint gate (see
# ``graspable_models``). It exists because the span above is a proxy for "not furniture" and the
# comment over it already says so: mass is what decides whether one hand can carry a thing. A
# hi-vis VEST measures 0.58 m and 0.5 kg -- three centimetres past a proxy for a property it
# manifestly has. Measured over all 1829 categories this lane admits 19 of them and every one is
# soft goods or a thin stick (life_jacket, jeans, cardigan, belt, rolled_yoga_mat, violin,
# architectural_plan); at 1.5 kg the ceiling is far under anything the two-handed-carry comment
# warns about, and no bicycle or desk top is anywhere near it.
GRASPABLE_LONG_SPAN_M: float = 0.70
GRASPABLE_LONG_MASS_KG: float = 1.5
# A DECAL floor, not a grasp floor -- ``mount_item_decal`` puts artwork on the body's own bbox face
# and below this the print is unreadable at reading range. Kept OUT of the carry test on purpose:
# the two were one condition until 2026-08-17, which made 92 categories the grasp engine is happy
# with (battery, charger, computer_mouse, dip_candle) unpickable for the unrelated reason that they
# are too small to sticker.
DECAL_MIN_SPAN_M: float = 0.04
WORK_SURFACE_Z_M: tuple[float, float] = (0.35, 1.40)

# Object-local measurements of Nutrition Facts / nutritional-information panels already printed
# in the shipped BEHAVIOR textures. These are gaze anchors only: publishing one must never author
# geometry, replace a material, or cover the package with generated artwork.
_NATIVE_NUTRITION_PANELS: dict[
    tuple[str, str],
    dict[str, object],
] = {
    ("box_of_almond_milk", "oiiqwq"): {
        "local": (0.0, 0.0375, -0.006),
        "face": (0.0, 1.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "size": (0.066, 0.140),
        "glyph_height_ratio": 0.035,
        "content_box_uv": (0.03, 0.05, 0.97, 0.95),
        "curved": False,
    },
    ("box_of_cereal", "wrlalk"): {
        "local": (0.018, -0.0525, 0.012),
        "face": (0.0, -1.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "size": (0.030, 0.090),
        "glyph_height_ratio": 0.035,
        "content_box_uv": (0.03, 0.05, 0.97, 0.95),
        "curved": False,
    },
    ("can_of_beans", "ojqgjz"): {
        "local": (0.002, -0.0394, 0.0165),
        "face": (0.0, -1.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "size": (0.040, 0.064),
        "glyph_height_ratio": 0.055,
        "content_box_uv": (0.03, 0.05, 0.97, 0.95),
        "inspect_grip_local": (0.0, 0.0, 0.046),
        "curved": True,
    },
    ("box_of_crackers", "cmdigf"): {
        "local": (0.0315, -0.0465, -0.0865),
        "face": (1.0, 0.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "size": (0.050, 0.010),
        "glyph_height_ratio": 0.350,
        "content_box_uv": (0.03, 0.12, 0.97, 0.88),
        "inspect_grip_local": (0.0, 0.0, 0.085),
        "curved": False,
    },
    ("bag_of_cookies", "ksjtde"): {
        "local": (0.078, 0.058, 0.0255),
        "face": (0.0, 0.0, 1.0),
        "up": (-1.0, 0.0, 0.0),
        "size": (0.048, 0.025),
        "glyph_height_ratio": 0.220,
        "content_box_uv": (0.03, 0.06, 0.97, 0.94),
        "mount": "top",
        "look_target_glyph_px": 28,
        "look_held_distance_m": 0.34,
        "curved": False,
    },
    ("box_of_cane_sugar", "pozpqi"): {
        "local": (0.0, 0.03835, -0.007),
        "face": (0.0, 1.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "size": (0.040, 0.084),
        "glyph_height_ratio": 0.040,
        "content_box_uv": (0.03, 0.03, 0.97, 0.97),
        "inspect_grip_local": (0.0, 0.0, 0.060),
        "curved": False,
    },
    ("bottle_of_chocolate_sauce", "tsyldw"): {
        "local": (-0.0173, 0.0065, -0.005),
        "face": (-1.0, 0.0, 0.0),
        "up": (0.0, 0.0, 1.0),
        "size": (0.070, 0.055),
        "glyph_height_ratio": 0.050,
        "content_box_uv": (0.03, 0.03, 0.97, 0.97),
        "curved": True,
    },
    ("bottle_of_pop", "rhclfg"): {
        "local": (-0.0226, -0.0049, 0.0064),
        "face": (-0.988, -0.157, 0.0),
        "up": (0.0, 0.0, 1.0),
        "size": (0.020, 0.043),
        "glyph_height_ratio": 0.060,
        "content_box_uv": (0.03, 0.03, 0.97, 0.97),
        "inspect_grip_local": (0.0, 0.0, -0.060),
        "curved": True,
    },
}

# Same-role families, keyed by the tokens a spec's own ``unit_roles`` ledger carries. This is the
# ONE authored table in the derived layer and it is rule-level, not per-row: eight families serve
# all 127 rows, and every category in them is affordance-CHECKED before it is offered.
_ROLE_FAMILIES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("cold", "chill", "fridge", "frozen", "freez", "dairy", "bev", "cool"),
        ("fridge", "grocery_fridge", "display_fridge", "wine_fridge", "freezer"),
    ),
    (
        (
            "bin",
            "trash",
            "waste",
            "garbage",
            "recycl",
            "compost",
            "weee",
            "sharps",
            "reject",
        ),
        ("recycling_bin", "trash_can", "compost_bin", "ashcan"),
    ),
    (("cart", "trolley"), ("shopping_cart", "serving_cart", "carton")),
    (("hot", "oven", "hood", "vent", "cook"), ("oven", "microwave", "dishwasher")),
    (
        ("lock", "cab", "drawer", "undersink", "store", "hold", "chest", "dry"),
        (
            "metal_bottom_cabinet",
            "bottom_cabinet",
            "locker",
            "cabinet_base",
            "bottom_cabinet_no_top",
        ),
    ),
    (
        ("shelf", "bay", "rack", "back", "stock", "ambient", "general", "overflow"),
        ("shelf", "grocery_shelf", "bookcase", "commercial_kitchen_shelf"),
    ),
    (("sink",), ("commercial_kitchen_sink", "sink")),
    (("tray", "tote", "box", "parcel"), ("carton", "packing_box", "crate")),
)
# Consulted only when neither the row's own ``assets_needed`` nor its role ledger names anything
# usable. Ordered by how many distinct rows their bodies have already been sited and scored in.
_DEFAULT_OPEN: tuple[str, ...] = (
    "shelf",
    "grocery_shelf",
    "bookcase",
    "recycling_bin",
    "trash_can",
    "carton",
)
_DEFAULT_DOOR: tuple[str, ...] = (
    "metal_bottom_cabinet",
    "bottom_cabinet",
    "locker",
    "fridge",
    "display_fridge",
    "oven",
)
_DEFAULT_GRASP: tuple[str, ...] = (
    "tupperware",
    "reagent_bottle",
    "carton_of_milk",
    "hardback",
    "plate",
    "bag_of_jerky",
    "bowl",
    "battery",
)
# Bodies the container-ACCEPTANCE probe has MEASURED on a live stage. Clean ones are offered
# first and silent ones last; neither list can bless a body, because ``assert_container_accepts``
# re-measures whatever is chosen in the row's own room before the build is allowed to finish.
_CLEAN_BODIES: frozenset[tuple[str, str]] = frozenset(
    {
        ("fridge", "jtqazu"),
        ("metal_bottom_cabinet", "mvsgdm"),
        ("grocery_shelf", "fgfrwb"),
        ("shelf", "yursyw"),
        ("grocery_fridge", "vzwcql"),
        ("trash_can", "wklill"),
        ("bottom_cabinet", "dajebq"),
        ("bottom_cabinet", "ccjinv"),
        ("locker", "gemgfz"),
        ("oven", "amblrk"),
    }
)
_SILENT_BODIES: frozenset[tuple[str, str]] = frozenset(
    {
        ("wine_fridge", "tgcuwh"),
        ("oven", "cgtaer"),
        ("fridge", "dszchb"),
        ("fridge", "fexqbj"),
        ("display_fridge", "wlxyjn"),
        ("fridge", "juwaoh"),
        ("recycling_bin", "mdojox"),
    }
)
# The GRASPABLE twins of the two tables above, measured through the real
# ``semantic_executor.produce_pick`` (pose planner + CCD warp + live per-shape SDF finger wrap) run
# on each body at a pose the reach envelope is measured to cover, so a failure is a property of the
# BODY alone. Like the container tables, neither
# can BLESS a body -- ``assert_grasp_accepts`` re-picks whatever is chosen in the row's own room.
_GRASP_CLEAN: frozenset[tuple[str, str]] = frozenset(
    {
        ("tupperware", "mkstwr"),
        ("reagent_bottle", "tnjpsf"),
        ("bag_of_jerky", "wblype"),
        ("plate", "aewthq"),
        ("plate", "fhdyrj"),
        ("hardback", "acbrnv"),
        ("hardback", "znlewl"),
        ("carton_of_milk", "xmugpm"),
        ("battery", "dcjyzg"),
        ("bowl", "byzaxy"),
        ("beer_glass", "lymciz"),
    }
)
# EMPTY, and that is the measurement, not an omission: all 11 bodies the 127 specs can stage --
# 413 staged instances over 121 rows -- seat a grip, ``beer_glass/lymciz`` included (seat_err
# 0.9 mm, 4 of 5 fingers in contact). The slice that blamed that body for T6-RURB-01's 0.00 was
# wrong about WHERE the defect lived: it was the staging SURFACE, not the object.
_GRASP_SILENT: frozenset[tuple[str, str]] = frozenset()
_OBSERVATION_ONLY_EXACT_OBJECTS: frozenset[tuple[str, str, str, str]] = frozenset(
    {("ASK-30000", "item_3", "petfood_bag", "fhfqys")}
)
# The printed cue vocabulary for a derived row. Nine entries because the widest declared bank in
# the portfolio is nine. Call signs rather than colour or hazard words on purpose: the band is an
# ARBITRARY tag whose binding to a unit is re-permuted every seed, and a word that also described
# something visible in the room would hand a reader a second, unintended channel.
_DERIVED_BANDS: tuple[str, ...] = (
    "LINE ALFA",
    "LINE BRAVO",
    "LINE CHARLIE",
    "LINE DELTA",
    "LINE ECHO",
    "LINE FOXTROT",
    "LINE GOLF",
    "LINE HOTEL",
    "LINE INDIA",
)
_DERIVED_DECOY_BAND: str = "LINE ZULU"
# One distinct glyph per band. Decoration, not the cue: both halves of the pair print the band
# TEXT (``_labels_target`` passes ``text=zone`` to the placard and to the decal), so the icon only
# has to be per-slot distinguishable at 640 px.
_DERIVED_ICONS: tuple[str, ...] = (
    "allergen_nuts",
    "allergen_gluten",
    "allergen_milk",
    "allergen_soy",
    "allergen_egg",
    "recycle",
    "weee",
    "biohazard",
    "fallback",
    "no_symbol",
)

_ASSET_INDEX: dict[str, dict] | None = None
_CATEGORY_AFFORDANCES: dict[str, frozenset[str]] | None = None
_RECON: dict[str, dict] | None = None
_SCENE_INV: dict[str, dict[str, set[str]]] = {}


def category_affordances(cat: str) -> frozenset[str]:
    """Semantic affordances published by the shared asset index for one category."""
    global _CATEGORY_AFFORDANCES
    if _CATEGORY_AFFORDANCES is None:
        import json

        path = os.path.join(
            _SIMULATOR_DIR,
            "taskpack_support",
            "asset_index.json",
        )
        with open(path) as handle:
            categories = json.load(handle).get("categories") or {}
        _CATEGORY_AFFORDANCES = {
            str(name): frozenset(str(value) for value in row.get("affordances") or [])
            for name, row in categories.items()
        }
    return _CATEGORY_AFFORDANCES.get(str(cat), frozenset())


def asset_index() -> dict[str, dict]:
    """category -> measured capability over EVERY baked model. Read once, cached.

    ``role`` is the bake's own ``role_hint``; ``door`` is the set of models carrying a drivable
    base_link revolute-Z leaf (the affordance ``[Open]`` / ``[Close]`` needs, and the same test
    ``_selftest_assets`` gates on); ``models`` maps model -> (bx, by, bz, mass) in metres/kg."""
    global _ASSET_INDEX
    if _ASSET_INDEX is not None:
        return _ASSET_INDEX
    import behavior_import as bi

    out: dict[str, dict] = {}
    root = bi.CACHE_USD
    for cat in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, cat)
        if not os.path.isdir(d):
            continue
        for model in sorted(f[:-4] for f in os.listdir(d) if f.endswith(".usd")):
            _index_model(out, cat, model)
    _ASSET_INDEX = out
    return out


def _index_model(out: dict[str, dict], cat: str, model: str) -> None:
    import json

    import behavior_import as bi

    p = os.path.join(bi.CACHE_SPEC, cat, f"{model}.json")
    if not os.path.exists(p):
        return
    try:
        with open(p) as f:
            sp = json.load(f)
    except (OSError, ValueError):
        return
    bb = list(sp.get("bbox_size") or [0.0, 0.0, 0.0])
    e = out.setdefault(cat, {"role": "", "models": {}, "door": set()})
    e["role"] = e["role"] or str(sp.get("role_hint") or "")
    e["models"][model] = (
        float(bb[0]),
        float(bb[1]),
        float(bb[2]),
        float(sp.get("mass") or 0.0),
    )
    if _has_base_door(sp):
        e["door"].add(model)


def _has_base_door(sp: dict) -> bool:
    """A drivable base-link leaf suitable for a closed destination container."""
    import behavior_import as bi

    return any(
        j.get("type") == "prismatic"
        and str(j.get("body0", "")).rstrip("/").endswith("base_link")
        for j in (sp.get("joints") or [])
    ) or bool(bi._container_revolute_joints(sp, accept_blank_vertical=True))


def _body_rank(cat: str, model: str, foot: float) -> tuple:
    """Sort key for one candidate body: measured-clean first, measured-silent last, then the
    smallest footprint (the binding resource in every room the siter has met)."""
    ev = (
        0
        if (cat, model) in _CLEAN_BODIES
        else (2 if (cat, model) in _SILENT_BODIES else 1)
    )
    return (ev, round(foot, 4), model)


_RECEPTACLE_VOCAB: frozenset[str] | None = None


def receptacle_vocab() -> frozenset[str]:
    """Categories that may play a DESTINATION container, measured off recon's own capability sheet.

    ``door_containers`` and ``open_receptacles`` are what recon counted, per scene, over all 51
    scenes; ``work_surfaces`` and ``architectural_doors`` are subtracted because a body recon calls
    a bench plays the PASS and a body it calls a doorway is not furniture at all. Without the
    subtraction the affordance filter alone happily offered ``door`` (a base_link revolute leaf) and
    ``coffee_table`` (a bench-height SURFACE) as places to put things inside.

    The author-added families are unioned in because four of the portfolio's rows deliberately bring
    a body the scene does not own (T1-OFFC-01's spec records that its scene contains no bin at all).
    """
    global _RECEPTACLE_VOCAB
    if _RECEPTACLE_VOCAB is not None:
        return _RECEPTACLE_VOCAB
    yes: set[str] = set(_DEFAULT_OPEN) | set(_DEFAULT_DOOR)
    for _keys, cats in _ROLE_FAMILIES:
        yes |= set(cats)
    yes.update(("drying_rack", "shoe_rack"))
    no: set[str] = set()
    for s in recon_scenes().values():
        yes |= set(_split_hist(s.get("door_containers")))
        yes |= set(_split_hist(s.get("open_receptacles")))
        no |= set(_split_hist(s.get("work_surfaces")))
        no |= set(_split_hist(s.get("architectural_doors")))
    _RECEPTACLE_VOCAB = frozenset(yes - no)
    return _RECEPTACLE_VOCAB


def _split_hist(entries) -> list[str]:
    """``["fridgex1", "ovenx2"]`` -> ``["fridge", "oven"]``."""
    out = []
    for e in entries or []:
        m = re.match(r"^(.*?)x(\d+)$", str(e))
        if m:
            out.append(m.group(1))
    return out


def unit_models(cat: str, want_door: bool) -> list[str]:
    """Baked models of ``cat`` that can hold a released graspable, best first.

    ``want_door`` is not a preference: a row whose authored program brackets its put with
    ``[Open]`` / ``[Close]`` cannot run its own oracle on a body with no leaf to drive."""
    e = asset_index().get(cat)
    if not e or cat not in receptacle_vocab():
        return []
    if str(e.get("role")) not in ("SURFACE", "OPENABLE"):
        return []
    got = []
    for m, (bx, by, bz, mass) in sorted(e["models"].items()):
        if want_door and m not in e["door"]:
            continue
        if min(bx, by) < RECEPTACLE_MIN_SPAN_M or max(bx, by) > RECEPTACLE_MAX_SPAN_M:
            continue
        minimum_height = (
            LOW_OPEN_RECEPTACLE_MIN_HEIGHT_M
            if not want_door and cat in LOW_OPEN_RECEPTACLE_CATEGORIES
            else RECEPTACLE_MIN_HEIGHT_M
        )
        if bz < minimum_height or mass < RECEPTACLE_MIN_MASS_KG:
            continue
        got.append((_body_rank(cat, m, 0.5 * max(bx, by)), m))
    return [m for _r, m in sorted(got)]


def _carryable(bx: float, by: float, bz: float, mass: float) -> bool:
    """Can one hand carry this measured body? Two lanes, both mass-bounded, no role hint involved."""
    span = max(bx, by, bz)
    if mass > GRASPABLE_MAX_MASS_KG:
        return False
    return span <= GRASPABLE_MAX_SPAN_M or (
        span <= GRASPABLE_LONG_SPAN_M and mass <= GRASPABLE_LONG_MASS_KG
    )


def graspable_models(cat: str, decal: bool = True) -> list[str]:
    """Baked models of ``cat`` a person can pick up, best first; with ``decal``, only ones that can
    also carry printed artwork.

    CARRYING IS DECIDED BY MEASUREMENT, NOT BY THE BAKE'S ROLE HINT (changed 2026-08-17). BEHAVIOR
    labels each model SURFACE / GRASPABLE / SEAT / OPENABLE, and requiring the label to read
    GRASPABLE was refusing to let a person pick up a ``vest``, a ``textbook``, a ``laptop`` or a
    ``table_lamp`` -- all of them labelled SURFACE, all of them obviously carryable, and 21 of the
    objects one compiled batch asked for. The bbox and the mass are OUR OWN measurements of the same
    asset and they answer the question the label was standing in for, so ``_carryable`` decides and
    the hint is ignored.

    This can only ever ADD, never remove or reorder what is already there: measured over all 1829
    categories, 152 gain a first model, 0 lose one, and the 3 whose existing list changes at all
    (``picture`` / ``signpost`` / ``toilet_paper_dispenser``, all already GRASPABLE) keep every model
    and their relative order -- the light-and-long lane simply admits a longer one, which the
    largest-first sort puts at the head. No spec on disk picks any of those three, so all 314
    derived picks are byte-identical. The heaviest newly admitted body over the whole library is
    ``food_processor`` at 4.0 kg / 0.46 m -- clutter, not furniture. Nothing here BLESSES a body:
    ``assert_grasp_accepts`` still re-measures whichever one is chosen in the row's own room, and the
    body ladder bumps past one whose grip the engine cannot seat.

    ``decal`` stays ON by default so every existing caller is unchanged; a row that prints nothing on
    its graspables passes False and gets the bodies that are merely too small to sticker."""
    e = asset_index().get(cat)
    if not e:
        return []
    got = []
    for m, (bx, by, bz, mass) in sorted(e["models"].items()):
        if not _carryable(bx, by, bz, mass):
            continue
        if decal and max(bx, by) < DECAL_MIN_SPAN_M:
            continue
        # Measured-SILENT bodies go LAST, never out: the same rule the container probe's
        # ``_body_rank`` follows. Only ``assert_grasp_accepts``, in the row's own room, may reject.
        got.append(((cat, m) in _GRASP_SILENT, -round(max(bx, by), 4), m))
    return [m for _s, _r, m in sorted(got)]


def recon_scenes() -> dict[str, dict]:
    """Return the frozen per-scene capability sheet keyed by scene id."""
    global _RECON
    if _RECON is None:
        import json

        with open(SCENE_CAPABILITY_INDEX) as f:
            _RECON = {str(s["scene_id"]): s for s in json.load(f).get("scenes") or []}
    return _RECON


def _recon_cats(scene: str, field: str) -> list[str]:
    """One ``catxN`` histogram recon measured for a scene -> categories, most instances first."""
    got = []
    for entry in (recon_scenes().get(scene) or {}).get(field) or []:
        m = re.match(r"^(.*?)x(\d+)$", str(entry))
        if m:
            got.append((-int(m.group(2)), m.group(1)))
    return [c for _n, c in sorted(got)]


def _selftest_derived(chk) -> None:
    """The derived layer's own invariants, independent of any one row."""
    vocab = receptacle_vocab()
    chk(
        not (
            vocab
            & {"door", "gate", "openable_window", "coffee_table", "desk", "countertop"}
        ),
        "the receptacle vocabulary excludes doorways and work surfaces",
    )
    for cat, model in sorted(_CLEAN_BODIES):
        chk(
            model in unit_models(cat, False),
            f"{cat}/{model} (probe-clean) survives the filter",
        )
    ids = spec_ids()
    chk(
        all(
            derive_recipe(load_spec(t)) == derive_recipe(load_spec(t)) for t in ids[:12]
        ),
        "derive_recipe is deterministic over 12 specs",
    )
    _selftest_arm_pins(chk, ids)
    _selftest_cue_legible(chk)


def _selftest_cue_legible(chk) -> None:
    """The CUE-LEGIBLE optics, and a FAULT INJECTION proving the gate can actually reject.

    Two things are checked separately because they fail for different reasons: the arithmetic (does
    ``cue_px`` reproduce the render's optics) and the verdict (does ``assert_cue_legible`` refuse a
    world it should). A check that has never rejected anything is not evidence.
    """
    # Inside the focal cap the zoom law holds the target at a FIXED angular share of the frame, so
    # the delivered size is scale- AND distance-invariant. That invariance is the law itself, and it
    # is why "the placard is in the frame" was never the same question as "the placard is legible".
    chk(
        abs(cue_px(0.46, 0.9) - cue_px(0.23, 0.45)) < 1e-6,
        "cue_px: the delivered size depends on span/dist ONLY -- halving both changes nothing",
    )
    # ⭐ CALIBRATION, against pixels I measured myself in frames the model was actually sent
    # (EVIDENCE/R_WFA_offc_item_decal_DELIVERED_x10.png and Q_WFA_offc_delivered/). Without these two
    # the whole gate is an untested formula; with them it is a model with a known 5% error.
    chk(
        abs(cue_px(0.037, INSPECT_FWD_M) - 22.0) <= 3.0,
        f"cue_px vs MEASURED: T1-OFFC-01 cell decal delivered at ~22 px, model says "
        f"{cue_px(0.037, INSPECT_FWD_M):.0f}",
    )
    chk(
        abs(cue_px(0.46, 1.41) - 70.0) <= 8.0,
        f"cue_px vs MEASURED: its bin placards delivered at ~70 px, model says "
        f"{cue_px(0.46, 1.41):.0f}",
    )
    chk(
        not CUE_LOOK_ZOOM_DELIVERED,
        "the model assumes NO look-zoom, because isaac_env._render_one never sets the focal",
    )
    chk(
        cue_px(0.46, 0.9) > cue_px(0.46, 3.0) > cue_px(0.46, 8.0),
        "cue_px: further away is smaller, monotonically",
    )
    chk(
        abs(cue_px(0.92, 2.0) - cue_px(0.46, 1.0)) < 1e-6,
        "cue_px: twice the artwork at twice the distance is the same number of pixels",
    )
    chk(
        cue_px(0.46, 0.9, delivered_w_px=1024) == 2.0 * cue_px(0.46, 0.9),
        "cue_px is measured at the DELIVERED resolution, not the render's",
    )

    # FAULT INJECTION. A stub world holding one ANSWER container whose placard is a knee-height
    # sliver read from across the room -- the T5-GROCA-01 shape -- must be refused when armed.
    class _E:
        def __init__(self, meta):
            self.meta = meta

    class _Sc:
        entities = {
            "unit_a": _E(
                {
                    "label_span": 0.05,
                    "label_pos": (4.0, 0.0, 0.55),
                    "front_stand": (0.0, 0.0),
                }
            )
        }

    class _R:
        @staticmethod
        def staged():
            return []

    class _W:
        spec = {"answer_key": [{"subject": "o", "pred": "inside", "target": "unit_a"}]}
        scene, recipe = _Sc(), _R()

    rows = cue_legibility(_W())
    chk(
        rows["unit_a"]["px"] < CUE_MIN_PX_TEXT and not rows["unit_a"]["ok"],
        f"fault injection: a 0.05 m placard 4 m away measures {rows['unit_a']['px']:.0f} px, under the floor",
    )
    chk(
        rows["unit_a"]["depression_deg"] > 0.0,
        "the reader's depression angle onto a knee-height placard is reported, not assumed 0",
    )
    _prev = os.environ.get("BENCH_CUE_LEGIBLE")
    try:
        os.environ["BENCH_CUE_LEGIBLE"] = "raise"
        _raised = False
        try:
            assert_cue_legible(_W())
        except SpecBuildError:
            _raised = True
        chk(
            _raised,
            "fault injection: ARMED, assert_cue_legible REFUSES the unreadable answer",
        )
        os.environ["BENCH_CUE_LEGIBLE"] = "warn"
        assert_cue_legible(_W())  # must not raise
        chk(
            True,
            "unarmed, the same world only warns -- the gate can never be what breaks a build",
        )
    finally:
        if _prev is None:
            os.environ.pop("BENCH_CUE_LEGIBLE", None)
        else:
            os.environ["BENCH_CUE_LEGIBLE"] = _prev


def _selftest_arm_pins(chk, ids: list[str]) -> None:
    """A split row's ARMS must still reach their row's pin (``_pin_for_arm``).

    Two halves, because the live half cannot see the defect yet and the synthetic half cannot see
    the portfolio. LIVE: every shipped ``<row>.<arm>`` spec resolves to its row's pin if the row
    has one and to the derived layer if it does not -- the assertion that would have FAILED before
    this rule existed, the day any pinned row was split. SYNTHETIC: the rule itself, over ledgers
    handed in by hand, because no arm of a pinned row ships today and an untested fallback is a
    fallback that will be wrong the first time it fires."""
    pins = recipes()
    for tid in [t for t in ids if "." in t]:
        want = pins.get(tid) or pins.get(
            str(load_spec(tid).get("source_task_id") or "")
        )
        got = recipe_for(tid)
        # BY VALUE, not by identity: ``recipes()`` builds a fresh table on every call (no
        # import-time registration), so the pin ``recipe_for`` returns is never the same OBJECT as
        # the one this loop holds. ``Recipe`` is a frozen dataclass of tuples, so ``==`` is the
        # structural question actually being asked. Measured: an identity test here reports a false
        # FAIL the first time a pinned row is split, which is the only time this check matters.
        chk(
            got == want if want is not None else got.task_id == tid,
            f"{tid} resolves to {'its row pin' if want is not None else 'the derived layer'}",
        )
    pinned = sorted(pins)[0]
    unpinned = next(t for t in ids if "." not in t and t not in pins)
    cases = [
        (
            f"{pinned}.feasible",
            {"source_task_id": pinned},
            pins[pinned],
            f"an arm of the pinned {pinned} inherits its row's pin",
        ),
        (
            pinned,
            {"source_task_id": pinned},
            None,
            "the fallback stands aside for an unsuffixed id (the exact lookup owns those)",
        ),
        (
            f"{unpinned}.arm",
            {"source_task_id": unpinned},
            None,
            f"an arm of the unpinned {unpinned} still reaches the derived layer",
        ),
        (
            "T9-NOPE-01.arm",
            {},
            None,
            "a spec with no source_task_id ledger still derives",
        ),
    ]
    for tid, ledger, want, msg in cases:
        chk(_pin_for_arm(pins, tid, ledger) is want, msg)


def scene_inventory(scene: str) -> dict[str, set[str]]:
    """category -> the room instances THIS scene puts it in, from the scene file's own
    ``in_rooms`` annotation. This is the "what does the scene actually contain" half of the
    capability query, and the reason a room-scoped preference is possible at all."""
    if scene in _SCENE_INV:
        return _SCENE_INV[scene]
    import behavior_import as bi

    out: dict[str, set[str]] = {}
    for o in bi.scene_objects(scene):
        out.setdefault(str(o["cat"]), set()).update(o["in_rooms"])
    _SCENE_INV[scene] = out
    return out


def _is_work_surface(cat: str) -> bool:
    e = asset_index().get(cat) or {}
    if str(e.get("role")) != "SURFACE":
        return False
    return any(
        WORK_SURFACE_Z_M[0] <= bz <= WORK_SURFACE_Z_M[1]
        and min(bx, by) >= 0.35
        and mass >= 5.0
        for bx, by, bz, mass in (e.get("models") or {}).values()
    )


def derived_support_cats(scene: str) -> tuple[str, ...]:
    """Categories in THIS scene that can stage the graspables at working height.

    ``_pick_support`` only ever picks a prim the SCENE ITSELF placed -- it never imports one -- so
    this is a pure query of the scene's own inventory: recon's measured ``work_surfaces`` first,
    then any other native category the bake calls a bench-height SURFACE."""
    inv = scene_inventory(scene)
    out = [c for c in _recon_cats(scene, "work_surfaces") if c in inv]
    out += [c for c in sorted(inv) if c not in out and _is_work_surface(c)]
    return tuple(out)


def _object_kind_cats(spec: dict) -> tuple[str, ...]:
    """Baked graspable categories the row's ``object_kinds`` names, in the order given.

    This is the channel v2 did not have. Until 2026-08-11 only 11 bodies had ever been measured, so
    ``pick_graspable`` put ``_DEFAULT_GRASP`` first and every one of 45 designs staged the same
    ``tupperware/mkstwr``. The bake (482 -> 1726 categories) and the batch grasp probe removed the
    reason for that, and this reads the designer's own choice of WHAT THE PERSON CARRIES.

    Returns () when the key is absent, which is what keeps the change additive: an empty bank
    contributes nothing to ``_ranked_cats`` and the ranking is then byte-identical to before.
    Unknown or unbaked names are dropped SILENTLY here on purpose -- the spec-time validator is
    where a bad name must be named, and a build that dies on a typo wastes 70 s of design."""
    want = spec.get("object_kinds")
    if not isinstance(want, (list, tuple)):
        return ()
    known = asset_index()
    out: list[str] = []
    for raw in want:
        c = str(raw).strip().lower().replace(" ", "_")
        if c in known and c not in out:
            out.append(c)
    return tuple(out)


def _named_cats(spec: dict) -> tuple[str, ...]:
    """Baked categories the row's own ``assets_needed`` free text names, in first-mention order.

    107 of the 127 rows name at least one, and it is the strongest available signal: it is the
    task author saying what the picture is supposed to show."""
    txt = " ".join(str(a) for a in (spec.get("assets_needed") or [])).lower()
    hits = []
    for cat in asset_index():
        if len(cat) <= 3:
            continue
        m = re.search(r"\b" + re.escape(cat) + r"\b", txt)
        if m:
            hits.append((m.start(), cat))
    return tuple(c for _i, c in sorted(hits))


def _role_tokens(spec: dict) -> tuple[str, ...]:
    """Every lowercase word the row's ``unit_roles`` ledger carries, deduped and sorted."""
    got: set[str] = set()
    for v in (spec.get("unit_roles") or {}).values():
        got.update(w for w in re.split(r"[^a-z0-9]+", str(v).lower()) if w)
    return tuple(sorted(got))


def _family_cats(spec: dict) -> tuple[str, ...]:
    """Categories the row's role ledger points at, via the rule-level same-role family table."""
    toks = _role_tokens(spec)
    out: list[str] = []
    for keys, cats in _ROLE_FAMILIES:
        if not any(t.startswith(k) or k in t for t in toks for k in keys):
            continue
        out += [c for c in cats if c not in out]
    return tuple(out)


def _pick_body(spec: dict, key: str, role: str, ranked, models) -> tuple[str, str]:
    """Walk a ranked category list and return the first (cat, model) that satisfies ``role``."""
    tried: list[str] = []
    for cat in ranked:
        got = models(cat)
        if got:
            return cat, got[0]
        tried.append(cat)
    raise RoleUnsatisfied(
        str(spec["task_id"]), key, role, str(spec["scene"]), tried[:24]
    )


def _ranked_cats(
    spec: dict, banks: tuple[tuple[str, ...], ...], prefer_native: bool = True
) -> list[str]:
    """The candidate categories, best first: what the row ASKS for, then what its role ledger
    implies, then the proven bank, then everything the scene itself already contains.

    ``prefer_native`` floats a category the scene already owns to the front of the whole list. That
    is right for a CONTAINER (an imported body that matches the room's own furniture reads as part
    of the scene) and wrong for a GRASPABLE: on T6-RURB-01 it lifted the scene-native
    ``beer_glass`` over every body with measured grasp evidence, and every ``[Grab]`` no-opped."""
    inv = scene_inventory(str(spec["scene"]))
    out: list[str] = []
    for group in (*banks, tuple(sorted(inv))):
        for c in group:
            if c not in out:
                out.append(c)
    if not prefer_native:
        return out
    native = {c for c in out if c in inv}
    return sorted(
        out, key=lambda c: (out.index(c) - (len(out) if c in native else 0), c)
    )


def pick_unit_body(spec: dict, want_door: bool) -> tuple[str, str, tuple[str, ...]]:
    """(category, model, the SAME-ROLE ladder behind it) for every derived destination container.

    The ladder is not decoration. ``site_accepting_units`` repairs a body whose interior swallows a
    ``[PutInside]`` by walking it, and a unit with an empty ``alt_cats`` fails the whole build the
    moment its own category runs out -- measured on T2-WS0-01, where all twelve baked
    ``bottom_cabinet_no_top`` models were rejected by the acceptance probe."""
    banks = (
        _named_cats(spec),
        _family_cats(spec),
        _DEFAULT_DOOR if want_door else _DEFAULT_OPEN,
    )
    role = "door container" if want_door else "open receptacle"
    ranked = _ranked_cats(spec, banks)
    cat, model = _pick_body(
        spec, "containers", role, ranked, lambda c: unit_models(c, want_door)
    )
    alts = tuple(c for c in ranked if c != cat and unit_models(c, want_door))[:8]
    return cat, model, alts


def pick_item_body(spec: dict) -> tuple[str, str]:
    """The ONE body every derived graspable is played by, chosen by affordance.

    The bank order is DELIBERATELY the opposite of the unit's: bodies with measured grasp evidence
    come first and the row's ``assets_needed`` only breaks ties. Two reasons, one of them measured:

    * the graspable is not the cue. What the reader has to read is the DECAL; which produce the
      body depicts is not what the row tests, while the CONTAINER body is what the picture is about.
    * ``role_hint == "GRASPABLE"`` does not mean the grasp engine can seat it. T6-RURB-01 picked
      ``beer_glass/lymciz`` (a thin-walled hollow) off its own ``assets_needed`` and every
      ``[PutInside]`` came back ``place_unstable (PLACE with an empty hand)`` -- four silent
      no-ops and a 0.00 with the build reporting no error at all.
    """
    # object_kinds FIRST, then the proven default bank. With no object_kinds the leading group is
    # empty and the order is exactly what it was; ``_selftest_object_kinds`` asserts that.
    banks = (_object_kind_cats(spec), _DEFAULT_GRASP, _named_cats(spec))
    return _pick_body(
        spec,
        "objects",
        "graspable that can carry a decal",
        _ranked_cats(spec, banks, prefer_native=False),
        graspable_models,
    )


def derived_room(spec: dict, supports: tuple[str, ...]) -> str:
    """The room the episode happens in: the biggest room instance that owns a staging surface.

    Left to ``candidate_rooms`` / ``plan_layout`` to overrule -- they choose the room, the bodies
    and every slot together -- but a concrete room here gives ``_compose`` a real anchor."""
    inv = scene_inventory(str(spec["scene"]))
    rooms: set[str] = set()
    for cat in supports:
        rooms |= inv.get(cat, set())
    if not rooms:
        return ""
    scene = str(spec["scene"])
    sized = []
    for r in sorted(rooms):
        x0, y0, x1, y1 = room_box(scene, r)
        sized.append((-(x1 - x0) * (y1 - y0), r))
    return sorted(sized)[0][1]


def derived_props(spec: dict, roster: set[str]) -> tuple[str, ...]:
    """The named PROPS this row's observation phase looks at but its roster does not name.

    Both program shapes reach here. ``unit_placard(s)`` is the unit sweep and needs no prop;
    anything else -- a prose ``card`` / ``tray_1``, a tokenised ``[LookAt] <par_sheet>`` -- is a
    real sign the row demands, and it becomes a registered, placarded staging surface. The
    alternative is not "drop the look": ``prose_look_keys`` refuses the row outright and
    ``command_grammar`` raises ``no scene entity for <par_sheet>`` at build time, which is exactly
    the failure this turns into a buildable world.

    Declared deviation: the sign is realised as a card on a work surface, not as the specific prop
    the row's prose imagines. It is readable, walk-up and in the room; it is not a bespoke body."""
    if is_prose(spec):
        named = parse_prose(str(spec["human_program"])).named_looks
    else:
        named = tuple(dict.fromkeys(authored_look_keys(str(spec["human_program"]))))
    # A look at the row's PRE-ANONYMISATION unit name is a look at that unit, not at a prop:
    # spec_privacy re-keyed the roster and left ``unit_rename`` as the ledger. T2-BN1-01 looks at
    # ``unit_01..unit_04``; building four cards for them would place four signs beside the four
    # units they already are.
    rename = {str(k): str(v) for k, v in (spec.get("unit_rename") or {}).items()}
    return tuple(
        n
        for n in named
        if n not in ("unit_placard", "unit_placards")
        and n not in roster
        and n not in rename
    )


def _derived_items(spec: dict, cat: str, model: str, n_slot: int):
    """The graspables, split into SCORED items and DECOYS by whether the answer key names them.

    A staged body with no answer-key row is not a defect: it is what makes
    ``not_inside{items minus key_subjects}`` a constraint about the picture. It has to be declared
    a decoy, though, or ``cue_class_for`` has no class to print on it."""
    stem = _stem(str(spec["task_id"]))
    keyed = {
        str(g["subject"])
        for g in (spec.get("answer_key") or [])
        if g.get("pred", "inside") == "inside"
    }
    pngs = tuple(f"{stem}__band_{i + 1}.png" for i in range(n_slot))
    icons = tuple(_DERIVED_ICONS[i % len(_DERIVED_ICONS)] for i in range(n_slot))
    items, decoys = [], []
    for k in sorted(spec["roster"]["objects"]):
        (items if k in keyed else decoys).append(ItemRecipe(k, cat, model, pngs, icons))
    for k in _spare_keys(spec, len(items) + len(decoys)):
        decoys.append(ItemRecipe(k, cat, model, pngs, icons))
    return tuple(items), tuple(decoys)


def _spare_keys(spec: dict, have: int) -> list[str]:
    """Extra SPARE keys a prose row's own published survey count demands beyond its roster.

    ``[LookAt] item_i(1) xN`` is the author stating how many bodies are on the bench, and
    ``assert_prose_arithmetic`` refuses a world that stages a different number. The roster carries
    only the SCORED keys, so the difference is real staged stock the row needs and the spec does not
    enumerate -- the same eight spares T6-HSF-02 was hand-authored with."""
    if not is_prose(spec):
        return []
    want = parse_prose(str(spec["human_program"])).n_look
    if want <= have:
        return []
    named = set(spec["roster"]["objects"])
    out, i = [], 1
    while len(out) < want - have:
        k = f"spare_{i}"
        if k not in named:
            out.append(k)
        i += 1
    return out


def derive_recipe(spec: dict) -> Recipe:
    """Resolve one spec's abstract roster to concrete assets, with nothing authored per row."""
    if isinstance(spec.get("bts_source"), dict):
        _canonical_digest, verify_runtime_view = _runtime_adapter()
        try:
            verify_runtime_view(spec)
        except ValueError as exc:
            raise SpecBuildError(f"invalid bts runtime view: {exc}") from exc
        return derive_bts_recipe(spec)
    tid = str(spec["task_id"])
    if any(g.get("pred") == "attached" for g in (spec.get("goals") or [])):
        raise UnsupportedSpec(
            f"{tid} is an ASSEMBLY row: its parts must expose real BEHAVIOR attach-point "
            f"metadata and a mate graph, which no affordance query can invent"
        )
    units_keys = sorted(spec["roster"]["containers"])
    if not units_keys or not spec["roster"]["objects"]:
        raise UnsupportedSpec(
            f"{tid} declares {len(units_keys)} container(s) and "
            f"{len(spec['roster']['objects'])} object(s); a container row needs both"
        )
    want_door = _needs_door(spec)
    ucat, umodel, ualts = pick_unit_body(spec, want_door)
    icat, imodel = pick_item_body(spec)
    bands, class_units = _derived_bands(spec, units_keys)
    items, decoys = _derived_items(spec, icat, imodel, len(bands) + 1)
    return _assemble_derived(
        spec, (ucat, umodel, ualts), bands, class_units, items, decoys, want_door
    )


def derive_bts_recipe(spec: dict) -> Recipe:
    """Consume the concrete world declared by a ``bts/1.0`` runtime view."""
    source = dict(spec["bts_source"])
    world = dict(source.get("world") or {})
    initially_open_room_door_names = tuple(
        str(value) for value in world.get("initially_open_room_door_names") or ()
    )
    initially_open_room_door_reverse_swing_names = tuple(
        str(value)
        for value in world.get("initially_open_room_door_reverse_swing_names") or ()
    )
    unknown_reverse_swing_names = tuple(
        name
        for name in initially_open_room_door_reverse_swing_names
        if name not in initially_open_room_door_names
    )
    if unknown_reverse_swing_names:
        raise UnsupportedSpec(
            f"{spec['task_id']}: reverse-swing room doors must also appear in "
            "initially_open_room_door_names: "
            f"{list(unknown_reverse_swing_names)}"
        )
    if world.get("object_groups"):
        raise UnsupportedSpec(
            f"{spec['task_id']}: object_groups must be compiled into explicit world.objects"
        )
    rooms = tuple(str(value) for value in spec.get("rooms") or ())
    room = str(spec.get("room") or (rooms[0] if rooms else ""))
    if any(g.get("pred") == "attached" for g in (spec.get("goals") or [])):
        return _derive_bts_assembly_recipe(spec, source, world, room)
    units = _bts_units(spec, world, room)
    items, decoys, preloads = _bts_items(spec, world)
    (
        support_keys,
        support_cats,
        support_items,
        support_rooms,
        support_names,
        support_models,
        support_siting_xy,
        support_siting_yaws,
        support_approach_yaws,
        support_operator_stands,
    ) = _bts_supports(spec, world, items + decoys, room)
    fixture_capabilities = program_fixture_capabilities(
        str(spec.get("human_program") or "")
    )
    bodies = {(unit.cat, unit.model) for unit in units}
    printed = _printed_block(spec)
    printed_items = {
        str(key)
        for key, value in (printed.get("items") or {}).items()
        if _printed_value_present(value)
    }
    object_rows = dict(world.get("objects") or {})
    printed_units = {
        str(key)
        for key, value in (printed.get("units") or {}).items()
        if _printed_value_present(value)
    }
    printed_walls = _printed_wall_entries(printed)
    wall_anchors = tuple(str(row["on"]) for row in printed_walls.values())
    roster = dict(spec.get("roster") or {})
    observe = tuple(
        [
            str(key)
            for key in roster.get("objects") or []
            if str(key) in printed_items
            and isinstance(object_rows.get(str(key)), Mapping)
            and (
                object_rows[str(key)].get("needs_lookat") is True
                or object_rows[str(key)].get("required_label_evidence") is True
            )
        ]
        + [
            str(key)
            for key in roster.get("containers") or []
            if str(key) in printed_units
        ]
        + list(wall_anchors)
    )
    shape = (
        "twin"
        if len(bodies) == 1
        else ("distinct" if len(bodies) == len(units) else "declared")
    )
    return Recipe(
        task_id=str(spec["task_id"]),
        scene=str(spec["scene"]),
        room=room,
        rooms=rooms or ((room,) if room else ()),
        support_cats=tuple(dict.fromkeys(support_cats)),
        units=units,
        items=items,
        cue_mode=(
            "printed" if printed_items or printed_units or printed_walls else "none"
        ),
        preloads=preloads,
        unit_shape=shape,
        decoys=decoys,
        n_supports=len(support_keys),
        support_keys=support_keys,
        support_bindings=support_cats,
        support_names=support_names,
        support_models=support_models,
        support_siting_xy=support_siting_xy,
        support_siting_front_yaws=support_siting_yaws,
        support_approach_yaws=support_approach_yaws,
        support_operator_stands=support_operator_stands,
        support_items=support_items,
        support_capabilities=tuple(
            fixture_capabilities.get(key, frozenset()) for key in support_keys
        ),
        support_rooms=support_rooms,
        observe=observe,
        frozen_bindings=world.get("layout") == "frozen",
        initially_open_room_doors=world.get("initially_open_room_doors") is True,
        initially_open_room_door_names=initially_open_room_door_names,
        initially_open_room_door_reverse_swing_names=(
            initially_open_room_door_reverse_swing_names
        ),
    )


def _derive_bts_assembly_recipe(
    spec: dict, source: dict, world: dict, room: str
) -> Recipe:
    _validate_bts_mate_contract(source, world)
    initially_open_room_door_names = tuple(
        str(value) for value in world.get("initially_open_room_door_names") or ()
    )
    initially_open_room_door_reverse_swing_names = tuple(
        str(value)
        for value in world.get("initially_open_room_door_reverse_swing_names") or ()
    )
    unknown_reverse_swing_names = tuple(
        name
        for name in initially_open_room_door_reverse_swing_names
        if name not in initially_open_room_door_names
    )
    if unknown_reverse_swing_names:
        raise UnsupportedSpec(
            f"{spec['task_id']}: reverse-swing room doors must also appear in "
            "initially_open_room_door_names: "
            f"{list(unknown_reverse_swing_names)}"
        )
    parts = _bts_parts(spec, world, room)
    items, decoys, _preloads = _bts_items(spec, world)
    (
        support_keys,
        support_cats,
        support_items,
        support_rooms,
        support_names,
        support_models,
        support_siting_xy,
        support_siting_yaws,
        support_approach_yaws,
        support_operator_stands,
    ) = _bts_supports(spec, world, items + decoys, room)
    fixture_capabilities = program_fixture_capabilities(
        str(spec.get("human_program") or "")
    )
    return Recipe(
        task_id=str(spec["task_id"]),
        scene=str(spec["scene"]),
        room=room,
        rooms=tuple(str(value) for value in spec.get("rooms") or (room,)),
        support_cats=tuple(dict.fromkeys(support_cats)),
        items=items,
        decoys=decoys,
        n_supports=len(support_keys),
        support_keys=support_keys,
        support_bindings=support_cats,
        support_names=support_names,
        support_models=support_models,
        support_siting_xy=support_siting_xy,
        support_siting_front_yaws=support_siting_yaws,
        support_approach_yaws=support_approach_yaws,
        support_operator_stands=support_operator_stands,
        support_items=support_items,
        support_capabilities=tuple(
            fixture_capabilities.get(key, frozenset()) for key in support_keys
        ),
        support_rooms=support_rooms,
        parts=parts,
        cue_mode="none",
        hull="floor",
        look_zoom=True,
        frozen_bindings=world.get("layout") == "frozen",
        initially_open_room_doors=world.get("initially_open_room_doors") is True,
        initially_open_room_door_names=initially_open_room_door_names,
        initially_open_room_door_reverse_swing_names=(
            initially_open_room_door_reverse_swing_names
        ),
    )


def _bts_parts(spec: dict, world: dict, room: str) -> tuple[PartRecipe, ...]:
    rows = dict(world.get("parts") or {})
    edges = [
        (str(goal.get("subject")), str(goal.get("target")))
        for goal in spec.get("goals") or []
        if goal.get("pred") == "attached"
    ]
    edges.extend(
        (str(row.get("child")), str(row.get("parent")))
        for row in world.get("initial_attachments") or []
        if isinstance(row, dict)
    )
    nodes = {key for edge in edges for key in edge}
    if not nodes.issubset(rows):
        raise UnsupportedSpec(
            f"{spec['task_id']}: assembly world.parts is missing scored or initial mate nodes "
            f"{sorted(nodes - set(rows))}"
        )
    referenced = {
        str(value)
        for command in ((spec.get("bts_source") or {}).get("program") or {}).get(
            "commands", ()
        )
        for value in command.get("args") or ()
    }
    unused = sorted(set(rows) - nodes - referenced)
    if unused:
        raise UnsupportedSpec(
            f"{spec['task_id']}: disconnected assembly parts are never used by the program: {unused}"
        )
    invalid_places = []
    for key in sorted(rows):
        row = dict(rows[key])
        cat, model = str(row.get("cat") or ""), str(row.get("model") or "")
        if not cat or not model:
            raise RoleUnsatisfied(
                str(spec["task_id"]), key, "assembly part", str(spec["scene"]), [cat]
            )
        place = str(row.get("place") or "")
        on_support = place.startswith("on:") and place[3:] in (
            world.get("fixtures") or {}
        )
        if place != f"floor:{room}" and not on_support:
            invalid_places.append(f"{key}={place!r}")
    if invalid_places:
        raise UnsupportedSpec(
            f"{spec['task_id']}: assembly parts must use floor:{room} or a declared "
            f"support; invalid placements: {invalid_places}"
        )

    parts: list[PartRecipe] = []
    for key in sorted(rows):
        row = dict(rows[key])
        cat, model = str(row.get("cat") or ""), str(row.get("model") or "")
        place = str(row.get("place") or "")
        parts.append(
            PartRecipe(
                key,
                cat,
                model,
                str(row.get("decal") or ""),
                affordances=tuple(
                    str(value) for value in row.get("affordances") or ()
                ),
                initial_place=place,
                spot=_bts_spot(spec, key, row),
                tint=str(row.get("tint") or ""),
            )
        )
    return tuple(parts)


def _validate_bts_mate_contract(source: dict, world: dict) -> None:  # noqa: C901
    task_id = str(source.get("task_id") or "")
    parts = dict(world.get("parts") or {})
    answers = [
        dict(row)
        for row in (source.get("goal") or {}).get("answer_key") or []
        if row.get("pred") == "attached"
    ]
    goals = [
        dict(row)
        for row in (source.get("goal") or {}).get("goals") or []
        if row.get("pred") == "attached"
    ]
    if not answers or [(row.get("subject"), row.get("target")) for row in answers] != [
        (row.get("subject"), row.get("target")) for row in goals
    ]:
        raise UnsupportedSpec(
            f"{task_id}: attached answer_key/goals are empty or drifted"
        )
    attaches = [
        dict(row)
        for row in (source.get("program") or {}).get("commands") or []
        if str(row.get("verb") or "").upper() in {"ASSEMBLE", "ATTACH", "INSTALL"}
    ]
    detaches = [
        dict(row)
        for row in (source.get("program") or {}).get("commands") or []
        if str(row.get("verb") or "").upper() == "DETACH"
    ]
    initial_rows = [dict(row) for row in world.get("initial_attachments") or []]
    initial = {str(row.get("child") or ""): row for row in initial_rows}
    if len(initial) != len(initial_rows) or "" in initial:
        raise UnsupportedSpec(
            f"{task_id}: world.initial_attachments must name each child exactly once"
        )
    attach_by_child: dict[str, list[dict]] = {}
    for command in attaches:
        args = [str(value) for value in command.get("args") or []]
        child = args[0] if args else ""
        if not child:
            raise UnsupportedSpec(f"{task_id}: Attach command is missing its child")
        attach_by_child.setdefault(child, []).append(command)
    detached_children: set[str] = set()
    for command in detaches:
        args = [str(value) for value in command.get("args") or []]
        if len(args) != 2:
            raise UnsupportedSpec(f"{task_id}: Detach must name child and parent")
        detached_children.add(args[0])
    children: set[str] = set()
    parent_slots: set[tuple[str, str]] = set()
    for index, answer in enumerate(answers):
        child, parent = (
            str(answer.get("subject") or ""),
            str(answer.get("target") or ""),
        )
        connector = str(answer.get("connector") or "")
        initial_row = initial.get(child)
        if initial_row and (
            str(initial_row.get("parent") or "") != parent
            or str(initial_row.get("connector") or "") != connector
        ):
            raise UnsupportedSpec(
                f"{task_id}: initial mate {child} disagrees with its scored parent/connector"
            )
        commands = attach_by_child.get(child, [])
        needs_attach = initial_row is None or child in detached_children
        if needs_attach and not commands:
            raise UnsupportedSpec(
                f"{task_id}: scored mate {child}->{parent} is neither preserved initially nor attached"
            )
        expected_args = [child, f"{connector}M", parent, f"{connector}F"]
        if any(
            [str(value) for value in command.get("args") or []] != expected_args
            for command in commands
        ):
            raise UnsupportedSpec(
                f"{task_id}: mate {index} must execute exact child/connector/parent pairing"
            )
        if child in children:
            raise UnsupportedSpec(f"{task_id}: part {child} is attached more than once")
        children.add(child)
        parent_slot = (parent, connector)
        if parent_slot in parent_slots:
            raise UnsupportedSpec(
                f"{task_id}: parent {parent} connector {connector} is reused; "
                "bts/1.0 commands cannot select a second connector instance"
            )
        parent_slots.add(parent_slot)
        child_row, parent_row = (
            dict(parts.get(child) or {}),
            dict(parts.get(parent) or {}),
        )
        if child_row.get("male") != f"{connector}M" or f"{connector}F" not in (
            parent_row.get("females") or []
        ):
            raise UnsupportedSpec(
                f"{task_id}: world.parts connector declarations disagree with mate {child}->{parent}"
            )
    scored_parents = {str(row["subject"]): str(row["target"]) for row in answers}
    parents = dict(scored_parents)
    for initial_row in initial_rows:
        child = str(initial_row.get("child") or "")
        if child in children:
            continue
        parent = str(initial_row.get("parent") or "")
        connector = str(initial_row.get("connector") or "")
        parent_slot = (parent, connector)
        if not child or not parent or not connector or parent_slot in parent_slots:
            raise UnsupportedSpec(
                f"{task_id}: initial mate {child}->{parent} has an invalid or reused connector {connector!r}"
            )
        child_row, parent_row = (
            dict(parts.get(child) or {}),
            dict(parts.get(parent) or {}),
        )
        if child_row.get("male") != f"{connector}M" or f"{connector}F" not in (
            parent_row.get("females") or []
        ):
            raise UnsupportedSpec(
                f"{task_id}: world.parts connector declarations disagree with initial mate {child}->{parent}"
            )
        children.add(child)
        parent_slots.add(parent_slot)
        parents[child] = parent
    extra_attaches = sorted(set(attach_by_child) - children)
    if extra_attaches:
        raise UnsupportedSpec(
            f"{task_id}: Attach commands name unscored children {extra_attaches}"
        )
    extra_detaches = sorted(detached_children - set(initial) - set(attach_by_child))
    if extra_detaches:
        raise UnsupportedSpec(
            f"{task_id}: Detach requires an earlier or initial attachment for {extra_detaches}"
        )
    _validate_mate_action_sequence(
        task_id,
        (source.get("program") or {}).get("commands") or [],
        initial,
        parents,
        scored_parents,
    )
    graph_nodes = children | set(parents.values())
    roots = graph_nodes - children
    if not roots:
        raise UnsupportedSpec(f"{task_id}: assembly mate graph has no root")
    for node in sorted(children):
        seen: set[str] = set()
        cursor = node
        while cursor not in roots:
            if cursor in seen or cursor not in parents:
                raise UnsupportedSpec(
                    f"{task_id}: mate graph is cyclic or disconnected at {node}"
                )
            seen.add(cursor)
            cursor = parents[cursor]


def _validate_mate_action_sequence(
    task_id: str,
    commands,
    initial: dict[str, dict],
    declared_parents: dict[str, str],
    scored_parents: dict[str, str],
) -> None:
    attached = {
        child: str(row.get("parent") or "") for child, row in initial.items()
    }
    attached_by_program: set[str] = set()
    for command in commands:
        verb = str(command.get("verb") or "").upper()
        args = [str(value) for value in command.get("args") or []]
        if verb in {"ASSEMBLE", "ATTACH", "INSTALL"} and len(args) >= 3:
            child, parent = args[0], args[2]
            if child in attached and child in attached_by_program:
                raise UnsupportedSpec(
                    f"{task_id}: Attach repeats {child}->{parent} without an intervening Detach"
                )
            if declared_parents.get(child) != parent:
                raise UnsupportedSpec(
                    f"{task_id}: Attach {child}->{parent} is outside the declared mate graph"
                )
            attached[child] = parent
            attached_by_program.add(child)
        elif verb == "DETACH" and len(args) == 2:
            child, parent = args
            if attached.get(child) != parent:
                raise UnsupportedSpec(
                    f"{task_id}: Detach {child}->{parent} has no matching current attachment"
                )
            attached.pop(child)
            attached_by_program.discard(child)
    missing = sorted(
        f"{child}->{parent}"
        for child, parent in scored_parents.items()
        if attached.get(child) != parent
    )
    if missing:
        raise UnsupportedSpec(
            f"{task_id}: final assembly program does not preserve scored mates {missing}"
        )


def _bts_units(spec: dict, world: dict, room: str) -> tuple[UnitRecipe, ...]:
    rows = dict(world.get("containers") or {})
    keys = [str(key) for key in (spec.get("roster") or {}).get("containers") or []]
    missing = [key for key in keys if key not in rows]
    if missing:
        raise UnsupportedSpec(
            f"{spec['task_id']}: roster container(s) {missing} absent from world.containers"
        )
    resolved: dict[str, UnitRecipe] = {}
    for key in keys:
        resolved[key] = _bts_unit(spec, key, dict(rows[key]), rows, room, resolved)
    return tuple(resolved[key] for key in keys)


def _exact_scene_candidate(
    spec: dict,
    cat: str,
    room: str,
    pick: dict,
) -> dict:
    if pick.get("by") != "scene_instance":
        raise UnsupportedSpec(
            f"{spec['task_id']}: scene binding requires exact scene_instance identity, "
            f"not {pick.get('by')!r}"
        )
    contract_version = "scene-container-binding/unavailable"
    digest_for = None
    try:
        import scene_furniture_units as sfu

        contract_version = sfu.SCENE_BINDING_CONTRACT_VERSION
        digest_for = sfu.scene_instance_digest
        runtime_candidates = [
            candidate
            for candidate in sfu.candidates(str(spec["scene"]))
            if candidate.get("cat") == cat and candidate.get("room") == room
        ]
    except (ImportError, OSError, ValueError):
        runtime_candidates = []
    rank = int(pick["rank"]) if pick.get("rank") is not None else None
    name = str(pick.get("name") or "")
    prim_path = str(pick.get("stage_prim_path") or "")
    asset_sha256 = str(pick.get("asset_sha256") or "")
    instance_digest = str(pick.get("instance_digest") or "")
    identity_matches = [
        candidate
        for candidate in runtime_candidates
        if candidate.get("name") == name
        and candidate.get("stage_prim_path") == prim_path
        and candidate.get("asset_sha256") == asset_sha256
        and candidate.get("room_rank") == rank
    ]
    exact = [
        candidate
        for candidate in identity_matches
        if candidate.get("instance_digest") == instance_digest
        and digest_for is not None
        and candidate.get("instance_digest") == digest_for(dict(candidate))
    ]
    if len(exact) == 1:
        return dict(exact[0])
    compatible = _legacy_open_only_scene_candidate(
        identity_matches,
        instance_digest,
        digest_for,
    )
    if compatible is not None:
        logger.warning(
            "%s: accepting exact scene instance %s after the additive open-to-swing "
            "registration upgrade",
            spec["task_id"],
            name,
        )
        return compatible
    raise UnsupportedSpec(
        f"{spec['task_id']}: exact scene instance {name!r} at {prim_path!r} "
        f"does not resolve uniquely in {room} under {contract_version}"
    )


def _legacy_open_only_scene_candidate(
    identity_matches: Sequence[dict],
    instance_digest: str,
    digest_for,
) -> dict | None:
    if len(identity_matches) != 1 or digest_for is None:
        return None
    candidate = identity_matches[0]
    if candidate.get("registration_modes") != ["open", "swing"]:
        return None
    if candidate.get("instance_digest") != digest_for(dict(candidate)):
        return None
    legacy_candidate = dict(candidate)
    legacy_candidate["registration_modes"] = ["open"]
    if digest_for(legacy_candidate) != instance_digest:
        return None
    return dict(candidate)


def _bts_unit_siting(
    spec: Mapping[str, object], key: str, row: Mapping[str, object], source: str
) -> tuple[tuple[float, float] | None, float | None]:
    raw_xy = row.get("siting_xy")
    raw_yaw = row.get("siting_front_yaw_deg")
    if (raw_xy is None) != (raw_yaw is None):
        raise UnsupportedSpec(
            f"{spec['task_id']}: container {key} must declare siting_xy and "
            "siting_front_yaw_deg together"
        )
    if raw_xy is None:
        return None, None
    if (
        not isinstance(raw_xy, (list, tuple))
        or len(raw_xy) != 2
        or any(isinstance(value, bool) for value in (*raw_xy, raw_yaw))
    ):
        raise UnsupportedSpec(
            f"{spec['task_id']}: container {key} has invalid explicit siting values"
        )
    try:
        x, y, yaw = (float(value) for value in (*raw_xy, raw_yaw))
    except (TypeError, ValueError) as error:
        raise UnsupportedSpec(
            f"{spec['task_id']}: container {key} has non-numeric explicit siting values"
        ) from error
    if (
        not all(math.isfinite(value) for value in (x, y, yaw))
        or not -180.0 <= yaw <= 180.0
    ):
        raise UnsupportedSpec(
            f"{spec['task_id']}: container {key} explicit siting must be finite and "
            "its yaw must be within -180..180 degrees"
        )
    if source == "scene":
        raise UnsupportedSpec(
            f"{spec['task_id']}: scene-bound container {key} cannot declare an imported "
            "siting transform"
        )
    return (x, y), yaw


def _bts_unit(
    spec: dict,
    key: str,
    row: dict,
    all_rows: dict,
    room: str,
    resolved: dict[str, UnitRecipe],
) -> UnitRecipe:
    declared_room = str(row.get("room") or room)
    legal_rooms = {str(value) for value in spec.get("rooms") or ()}
    if legal_rooms and declared_room not in legal_rooms:
        raise UnsupportedSpec(
            f"{spec['task_id']}: container {key} declares room {declared_room!r}; "
            f"legal rooms are {sorted(legal_rooms)}"
        )
    bind = dict(row.get("bind") or {})
    source = str(bind.get("source") or "")
    if source == "clone":
        parent = str(bind.get("of") or "")
        if parent not in resolved:
            if parent not in all_rows:
                raise UnsupportedSpec(
                    f"{spec['task_id']}: {key} clones unknown {parent}"
                )
            resolved[parent] = _bts_unit(
                spec, parent, dict(all_rows[parent]), all_rows, room, resolved
            )
        base = resolved[parent]
        cat, model, binding_source = base.cat, base.model, "clone"
        rank = None
        pick_name = pick_prim_path = pick_asset_sha256 = pick_instance_digest = None
    elif source in {"scene", "import"}:
        cat = str(bind.get("cat") or "")
        want_door = bool((row.get("door") or {}).get("required"))
        models = unit_models(cat, want_door)
        authored_model = str(bind.get("model") or "")
        model = authored_model or (str(models[0]) if models else "")
        if authored_model and authored_model not in models:
            raise RoleUnsatisfied(
                str(spec["task_id"]),
                key,
                "exact container model",
                str(spec["scene"]),
                [f"{cat}/{authored_model}"],
            )
        if not cat or not model:
            raise RoleUnsatisfied(
                str(spec["task_id"]), key, "container", str(spec["scene"]), [cat]
            )
        binding_source = source
        pick = dict(bind.get("pick") or {})
        rank = int(pick["rank"]) if pick.get("rank") is not None else None
        pick_name = pick_prim_path = pick_asset_sha256 = pick_instance_digest = None
        if source == "scene":
            available = scene_inventory(str(spec["scene"])).get(cat, set())
            if declared_room and declared_room not in available:
                raise RoleUnsatisfied(
                    str(spec["task_id"]),
                    key,
                    f"scene container in {declared_room}",
                    str(spec["scene"]),
                    [cat],
                )
            pick_name = str(pick.get("name") or "")
            pick_prim_path = str(pick.get("stage_prim_path") or "")
            pick_asset_sha256 = str(pick.get("asset_sha256") or "")
            pick_instance_digest = str(pick.get("instance_digest") or "")
            resolved_pick = _exact_scene_candidate(spec, cat, declared_room, pick)
            model = str(resolved_pick["model"])
            pick_instance_digest = str(resolved_pick["instance_digest"])
    else:
        raise UnsupportedSpec(
            f"{spec['task_id']}: container {key} bind.source={source!r} is not realizable"
        )
    siting_xy, siting_yaw = _bts_unit_siting(spec, key, row, source)
    open_top = not bool((row.get("door") or {}).get("required"))
    # ALTERNATIVE CATEGORIES, or the ladder has nowhere to go. A derived row used to carry none, so
    # when every body of its one category failed the acceptance probe the build died with
    # "exhausted <cat> + alt_cats ()" -- the message asking for exactly this. The emitter measures
    # them (`catalog.containers_that_fit`, the same interior data the compile checks use) and puts
    # them on the row; anything with no list keeps the old behaviour.
    alts = tuple(
        str(c) for c in (row.get("alt_cats") or ()) if str(c) and str(c) != cat
    )
    return UnitRecipe(
        key,
        cat,
        model,
        f"{_stem(str(spec['task_id']))}__{key}.png",
        str(row.get("label") or "DESTINATION"),
        open_top=open_top,
        alt_cats=alts,
        binding_source=binding_source,
        pick_rank=rank,
        pick_name=pick_name,
        pick_prim_path=pick_prim_path,
        pick_asset_sha256=pick_asset_sha256,
        pick_instance_digest=pick_instance_digest,
        tint=str(row.get("tint") or ""),
        room=declared_room,
        move_mode=str(row.get("move_mode") or ""),
        initially_open=bool(row.get("initially_open")),
        siting_xy=siting_xy,
        siting_front_yaw_deg=siting_yaw,
    )


def _bts_items(
    spec: dict, world: dict
) -> tuple[tuple[ItemRecipe, ...], tuple[ItemRecipe, ...], tuple[PreloadRecipe, ...]]:
    rows = dict(world.get("objects") or {})
    printed_items = dict((_printed_block(spec).get("items") or {}))
    items: list[ItemRecipe] = []
    decoys: list[ItemRecipe] = []
    preloads: list[PreloadRecipe] = []
    for key in [
        str(value) for value in (spec.get("roster") or {}).get("objects") or []
    ]:
        row = dict(rows.get(key) or {})
        if not row:
            raise UnsupportedSpec(
                f"{spec['task_id']}: roster object {key} is undeclared"
            )
        cat = str(row.get("cat") or "")
        # A bts row only ever decals its graspables under ``cue_mode="printed"`` (see
        # ``derive_bts_recipe``), so an unprinted row must not lose a body for being too small to
        # sticker -- that floor is about artwork, not about the hand.
        models = graspable_models(
            cat, decal=_printed_value_present(printed_items.get(key))
        )
        authored_model = str(row.get("model") or "")
        model = authored_model or (str(models[0]) if models else "")
        exact_observation_only = _validate_observation_only_exact_object(
            spec, key, cat, authored_model
        )
        if authored_model and authored_model not in models and not exact_observation_only:
            raise RoleUnsatisfied(
                str(spec["task_id"]),
                key,
                "exact graspable model",
                str(spec["scene"]),
                [f"{cat}/{authored_model}"],
            )
        if not cat or not model:
            raise RoleUnsatisfied(
                str(spec["task_id"]), key, "graspable", str(spec["scene"]), [cat]
            )
        place = str(row.get("place") or "")
        if place.startswith("in:"):
            unit = place[3:]
            preloads.append(
                PreloadRecipe(
                    unit,
                    cat,
                    model,
                    lo=1,
                    hi=1,
                    keys=(key,),
                    initial_space=_bts_initial_space(spec, key, row, unit),
                    tint=str(row.get("tint") or ""),
                    front_facing=bool(row.get("front_facing")),
                    front_show_back=bool(row.get("front_show_back")),
                    spot=_bts_spot(spec, key, row),
                )
            )
            continue
        if row.get("initial_space") not in (None, ""):
            raise UnsupportedSpec(
                f"{spec['task_id']}: object {key} initial_space requires an in: placement"
            )
        if not (place.startswith("on:") or place.startswith("floor:")):
            raise UnsupportedSpec(
                f"{spec['task_id']}: object {key} place={place!r} needs a generic placement producer"
            )
        item = ItemRecipe(
            key,
            cat,
            model,
            (),
            (),
            initial_place=place,
            spot=_bts_spot(spec, key, row),
            tint=str(row.get("tint") or ""),
            affordances=tuple(
                str(value) for value in row.get("affordances") or ()
            ),
            scale=_bts_scale(spec, key, row),
            workstation_posture=str(row.get("workstation_posture") or ""),
            workstation_screen_style=str(row.get("workstation_screen_style") or ""),
        )
        (items if bool(row.get("scored")) else decoys).append(item)
    return tuple(items), tuple(decoys), tuple(preloads)


def _is_observation_only_exact_object(
    task_id: str, key: str, cat: str, model: str
) -> bool:
    return (task_id, key, cat, model) in _OBSERVATION_ONLY_EXACT_OBJECTS


def _validate_observation_only_exact_object(
    spec: dict, key: str, cat: str, model: str
) -> bool:
    task_id = str(spec.get("task_id") or "")
    if not _is_observation_only_exact_object(task_id, key, cat, model):
        return False
    for goal in (*list(spec.get("goals") or []), *list(spec.get("answer_key") or [])):
        if str(goal.get("subject") or "") == key and str(
            goal.get("pred") or ""
        ).casefold() not in {"unmoved", "is_unmoved", "remain_unmoved"}:
            raise UnsupportedSpec(
                f"{task_id}: observation-only exact object {key} has a mutable goal"
            )
    for token in program_tokens(str(spec.get("human_program") or "")):
        verb_match = re.match(r"\[([A-Za-z_]+)\]", token)
        arguments = re.findall(r"<([^>]+)>", token)
        verb = verb_match.group(1).replace("_", "").upper() if verb_match else ""
        if key in arguments and verb not in {"WALK", "RUN", "WALKTOWARDS", "LOOKAT"}:
            raise UnsupportedSpec(
                f"{task_id}: observation-only exact object {key} cannot run [{verb}]"
            )
    return True


def _bts_initial_space(spec: dict, key: str, row: dict, unit: str) -> str:
    raw = row.get("initial_space")
    if raw is None or raw == "":
        return ""
    if not isinstance(raw, str) or not re.fullmatch(
        rf"{re.escape(unit)}_space[1-9][0-9]*", raw
    ):
        raise UnsupportedSpec(
            f"{spec['task_id']}: object {key} initial_space={raw!r} must name "
            f"a numbered space of {unit}"
        )
    return raw


def _bts_spot(spec: dict, key: str, row: dict) -> tuple[float, float] | None:
    """``world.objects[key].spot`` as a clamped (u, v), or None when the author left it out.

    Refused rather than repaired when it is present but malformed: a spot that silently becomes
    ``None`` puts the object back on the evenly-spaced fallback, which is precisely the arrangement
    the author was overriding, and the row would then build and score while showing a different
    picture from the one it describes."""
    raw = row.get("spot")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise UnsupportedSpec(
            f"{spec['task_id']}: object {key} spot={raw!r} must be a [u, v] pair"
        )
    try:
        u, v = float(raw[0]), float(raw[1])
    except (TypeError, ValueError) as exc:
        raise UnsupportedSpec(
            f"{spec['task_id']}: object {key} spot={raw!r} is not numeric"
        ) from exc
    if u != u or v != v:  # NaN survives float() and would place the body nowhere
        raise UnsupportedSpec(
            f"{spec['task_id']}: object {key} spot={raw!r} is not finite"
        )
    return (min(1.0, max(-1.0, u)), min(1.0, max(0.0, v)))


def _bts_scale(
    spec: dict, key: str, row: dict
) -> float | tuple[float, float, float]:
    raw = row.get("scale", 1.0)
    if isinstance(raw, (list, tuple)):
        if len(raw) != 3:
            raise UnsupportedSpec(
                f"{spec['task_id']}: object {key} scale={raw!r} must be a scalar or [x, y, z]"
            )
        try:
            scale_xyz = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise UnsupportedSpec(
                f"{spec['task_id']}: object {key} scale={raw!r} is not numeric"
            ) from exc
        if any(
            not math.isfinite(value) or not 0.1 <= value <= 2.0
            for value in scale_xyz
        ):
            raise UnsupportedSpec(
                f"{spec['task_id']}: object {key} scale={raw!r} components must be "
                "finite and in [0.1, 2.0]"
            )
        return scale_xyz
    try:
        scale = float(raw)
    except (TypeError, ValueError) as exc:
        raise UnsupportedSpec(
            f"{spec['task_id']}: object {key} scale={raw!r} is not numeric"
        ) from exc
    if not math.isfinite(scale) or not 0.5 <= scale <= 2.0:
        raise UnsupportedSpec(
            f"{spec['task_id']}: object {key} scale={raw!r} must be finite and in [0.5, 2.0]"
        )
    return scale


def _scale_xyz(
    scale: float | tuple[float, float, float],
) -> tuple[float, float, float]:
    if isinstance(scale, tuple):
        return tuple(float(value) for value in scale)
    scalar = float(scale)
    return (scalar, scalar, scalar)


def _serialized_scale(scale: float | tuple[float, float, float]) -> float | list[float]:
    if isinstance(scale, tuple):
        return [float(value) for value in scale]
    return float(scale)


def _movable_support_groups(
    items: tuple[ItemRecipe, ...], task_id: str = ""
) -> tuple[tuple[str, tuple[ItemRecipe, ...]], ...]:
    """Return movable-on-movable placements in dependency order."""
    ordered = tuple(items)
    by_key = {item.key: item for item in ordered}
    pending = [
        item
        for item in ordered
        if item.initial_place.startswith("on:")
        and item.initial_place[3:] in by_key
    ]
    resolved = set(by_key) - {item.key for item in pending}
    groups: list[tuple[str, tuple[ItemRecipe, ...]]] = []
    while pending:
        ready = [item for item in pending if item.initial_place[3:] in resolved]
        if not ready:
            edges = sorted(
                (item.key, item.initial_place[3:]) for item in pending
            )
            prefix = f"{task_id}: " if task_id else ""
            raise UnsupportedSpec(
                f"{prefix}movable initial supports contain a cycle: {edges}"
            )
        ready_by_parent: dict[str, list[ItemRecipe]] = {}
        for item in ready:
            ready_by_parent.setdefault(item.initial_place[3:], []).append(item)
        groups.extend(
            (parent, tuple(children))
            for parent, children in ready_by_parent.items()
        )
        ready_keys = {item.key for item in ready}
        resolved.update(ready_keys)
        pending = [item for item in pending if item.key not in ready_keys]
    return tuple(groups)


def _fixture_item_supports(
    items: tuple[ItemRecipe, ...], task_id: str
) -> tuple[str, ...]:
    movable_keys = {item.key for item in items}
    _movable_support_groups(items, task_id)
    return tuple(
        dict.fromkeys(
            item.initial_place[3:]
            for item in items
            if item.initial_place.startswith("on:")
            and item.initial_place[3:] not in movable_keys
        )
    )


def _non_movable_support_keys(
    items: tuple[ItemRecipe, ...], candidates: Collection[str]
) -> tuple[str, ...]:
    movable_keys = {item.key for item in items}
    return tuple(
        key for key in dict.fromkeys(candidates) if key not in movable_keys
    )


def _bts_supports(
    spec: dict,
    world: dict,
    items: tuple[ItemRecipe, ...],
    room: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, ...], ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[float, float] | None, ...],
    tuple[float | None, ...],
    tuple[float | None, ...],
    tuple[tuple[float, float] | None, ...],
]:
    fixtures = dict(world.get("fixtures") or {})
    item_supports = _fixture_item_supports(
        items, str(spec.get("task_id") or "")
    )
    part_supports = tuple(
        dict.fromkeys(
            str(row.get("place") or "")[3:]
            for row in (world.get("parts") or {}).values()
            if str(row.get("place") or "").startswith("on:")
        )
    )
    route_waypoints = tuple(
        str(waypoint)
        for outcome in spec.get("trace_outcomes") or []
        if isinstance(outcome, dict) and outcome.get("kind") == "ordered_route"
        for waypoint in outcome.get("ordered_waypoints") or []
        if isinstance(waypoint, str)
    )
    # DESTINATION tops, not just the ones things start on. `on` goals name a surface the row never
    # stages anything onto at t=0, and a top nobody starts on was never registered -- so the first
    # row scored purely by `on` died at `compile_program` with "no scene entity for <surf_2>",
    # having built its world perfectly. Every declared fixture the program walks to, puts onto, or
    # the answer key scores against is a real destination and has to exist.
    goal_tops = tuple(
        str(row.get("target"))
        for row in (spec.get("answer_key") or []) + (spec.get("goals") or [])
        if str(row.get("pred") or "").lower() == "on"
        and str(row.get("target")) in fixtures
    )
    prog = str(spec.get("human_program") or "")
    prog_tops = tuple(
        m.group(1) for m in re.finditer(r"<([^>]+)>", prog) if m.group(1) in fixtures
    )
    wall_anchors = tuple(
        str(row["on"])
        for row in _printed_wall_entries(_printed_block(spec)).values()
        if str(row["on"]) in fixtures
    )
    keys = _non_movable_support_keys(
        items,
        (
            *item_supports,
            *part_supports,
            *route_waypoints,
            *goal_tops,
            *prog_tops,
            *wall_anchors,
        ),
    )
    fixture_capabilities = program_fixture_capabilities(
        str(spec.get("human_program") or "")
    )
    cats: list[str] = []
    groups: list[tuple[str, ...]] = []
    rooms: list[str] = []
    names: list[str] = []
    models: list[str] = []
    siting_xy: list[tuple[float, float] | None] = []
    siting_yaws: list[float | None] = []
    approach_yaws: list[float | None] = []
    operator_stands: list[tuple[float, float] | None] = []
    inventory = scene_inventory(str(spec["scene"]))
    for key in keys:
        fixture = dict(fixtures.get(key) or {})
        if "container" in fixture_capabilities.get(key, frozenset()):
            raise UnsupportedSpec(
                f"{spec['task_id']}: fixture {key} is not a receptacle; declare a "
                "world.containers entry for PutInside"
            )
        bind = dict(fixture.get("bind") or {})
        source = str(bind.get("source") or "")
        if fixture and source not in {"scene", "import"}:
            raise UnsupportedSpec(
                f"{spec['task_id']}: support fixture {key} bind.source={source!r} "
                "must be scene or import"
            )
        cat = str(bind.get("cat") or key)
        if (
            cat == "openable_window"
            and "surface" in fixture_capabilities.get(key, frozenset())
        ):
            raise UnsupportedSpec(
                f"{spec['task_id']}: openable window fixture {key} cannot be used as a "
                "PutOn surface"
            )
        # The room the FIXTURE declares, not the episode's one room. Navigation has always crossed
        # rooms, and an explicitly sited imported top is still checked against that declared room.
        rm = str(fixture.get("room") or room or "")
        legal_rooms = {str(value) for value in spec.get("rooms") or ()}
        if legal_rooms and rm not in legal_rooms:
            raise UnsupportedSpec(
                f"{spec['task_id']}: support {key} declares room {rm!r}; "
                f"legal rooms are {sorted(legal_rooms)}"
            )
        raw_xy = fixture.get("siting_xy")
        raw_siting_yaw = fixture.get("siting_front_yaw_deg")
        if (raw_xy is None) != (raw_siting_yaw is None):
            raise UnsupportedSpec(
                f"{spec['task_id']}: fixture {key} must declare siting_xy and "
                "siting_front_yaw_deg together"
            )
        explicit_xy = None
        explicit_siting_yaw = None
        if raw_xy is not None:
            if (
                not isinstance(raw_xy, (list, tuple))
                or len(raw_xy) != 2
                or any(isinstance(value, bool) for value in (*raw_xy, raw_siting_yaw))
            ):
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} has invalid explicit siting values"
                )
            try:
                x, y, yaw = (float(value) for value in (*raw_xy, raw_siting_yaw))
            except (TypeError, ValueError) as error:
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} has non-numeric explicit siting values"
                ) from error
            if (
                not all(math.isfinite(value) for value in (x, y, yaw))
                or not -180.0 <= yaw <= 180.0
            ):
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} siting must be finite and yaw "
                    "within -180..180 degrees"
                )
            explicit_xy, explicit_siting_yaw = (x, y), yaw

        raw_approach = fixture.get("approach_yaw_deg")
        explicit_approach = None
        if raw_approach is not None:
            if isinstance(raw_approach, bool):
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} has boolean approach_yaw_deg"
                )
            try:
                explicit_approach = float(raw_approach)
            except (TypeError, ValueError) as error:
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} has non-numeric approach_yaw_deg"
                ) from error
            if not math.isfinite(explicit_approach) or not -180.0 <= explicit_approach <= 180.0:
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} approach_yaw_deg must be finite "
                    "and within -180..180 degrees"
                )

        raw_stand = fixture.get("operator_stand_xy")
        explicit_stand = None
        if raw_stand is not None:
            if (
                not isinstance(raw_stand, (list, tuple))
                or len(raw_stand) != 2
                or any(isinstance(value, bool) for value in raw_stand)
            ):
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} operator_stand_xy must be a "
                    "two-number pair"
                )
            try:
                stand_x, stand_y = (float(value) for value in raw_stand)
            except (TypeError, ValueError) as error:
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} operator_stand_xy is not numeric"
                ) from error
            if not all(math.isfinite(value) for value in (stand_x, stand_y)):
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} operator_stand_xy must be finite"
                )
            if explicit_approach is None:
                raise UnsupportedSpec(
                    f"{spec['task_id']}: fixture {key} operator_stand_xy requires "
                    "approach_yaw_deg"
                )
            explicit_stand = (stand_x, stand_y)

        model = str(bind.get("model") or "")
        if source == "scene":
            if explicit_xy is not None:
                raise UnsupportedSpec(
                    f"{spec['task_id']}: scene-bound fixture {key} cannot declare an "
                    "imported siting transform"
                )
            if cat not in inventory or (rm and rm not in inventory.get(cat, set())):
                raise RoleUnsatisfied(
                    str(spec["task_id"]),
                    key,
                    f"support in {rm}",
                    str(spec["scene"]),
                    [cat],
                )
        else:
            if not model or explicit_xy is None or explicit_siting_yaw is None:
                raise UnsupportedSpec(
                    f"{spec['task_id']}: imported fixture {key} requires an exact model, "
                    "siting_xy, and siting_front_yaw_deg"
                )
            if model not in baked_models(cat):
                raise RoleUnsatisfied(
                    str(spec["task_id"]),
                    key,
                    "exact imported support model",
                    str(spec["scene"]),
                    [f"{cat}/{model}"],
                )
        cats.append(cat)
        rooms.append(rm)
        names.append(str((bind.get("pick") or {}).get("name") or ""))
        models.append(model)
        siting_xy.append(explicit_xy)
        siting_yaws.append(explicit_siting_yaw)
        approach_yaws.append(explicit_approach)
        operator_stands.append(explicit_stand)
        groups.append(
            tuple(item.key for item in items if item.initial_place == f"on:{key}")
        )
    return (
        keys,
        tuple(cats),
        tuple(groups),
        tuple(rooms),
        tuple(names),
        tuple(models),
        tuple(siting_xy),
        tuple(siting_yaws),
        tuple(approach_yaws),
        tuple(operator_stands),
    )


def share_group_classes(spec: dict) -> tuple[list[str], list[str]]:
    """(the class strings a ``cue_class`` no_share_group names, the units it names).

    These are NOT decoration. ``constraint_resolver.assert_fireable`` builds a synthetic violating world out
    of the constraint's OWN vocabulary, so if the printed bands are not the literal strings the
    spec's ``incompatible`` pair names, ``group_of`` can never contain both and the constraint is
    refused as unfireable. The spec is therefore the source of its own cue words."""
    classes: list[str] = []
    units: list[str] = []
    for c in spec.get("negs") or []:
        if c.get("kind") != "no_share_group":
            continue
        if str((c.get("group_of_expr") or {}).get("op")) != "cue_class":
            continue
        for pair in c.get("incompatible") or []:
            classes += [str(g) for g in pair if str(g) not in classes]
        units += [str(u) for u in (c.get("units") or []) if str(u) not in units]
    return classes, units


def _derived_bands(
    spec: dict, unit_keys: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(one printed band per unit, the units the seeded permutation may move them over).

    A row with an incompatible pair pins those literal classes onto the units the constraint names
    and restricts the shuffle to exactly those units -- the shape T6-RDIN-01 was hand-authored into
    after a seed on which neither incompatible class landed on the pair left it unfireable."""
    named, nsg_units = share_group_classes(spec)
    nsg_units = [u for u in nsg_units if u in unit_keys]
    if len(named) > max(len(nsg_units), 0):
        raise UnsupportedSpec(
            f"{spec['task_id']}: no_share_group names {len(named)} class(es) {named} but only "
            f"{len(nsg_units)} of its units {nsg_units} are in the roster; one band per unit is "
            f"impossible, so the constraint could not be printed honestly"
        )
    pinned = dict(zip(nsg_units, named))
    spare = [b for b in _DERIVED_BANDS if b not in named]
    out: list[str] = []
    for k in unit_keys:
        if k in pinned:
            out.append(pinned[k])
        elif spare:
            out.append(spare.pop(0))
        else:
            raise UnsupportedSpec(
                f"{spec['task_id']} declares {len(unit_keys)} destinations; the derived cue "
                f"vocabulary prints {len(_DERIVED_BANDS)} distinguishable bands"
            )
    return tuple(out), tuple(nsg_units)


def _assemble_derived(
    spec, body, bands, class_units, items, decoys, want_door: bool
) -> Recipe:
    """Bolt the resolved assets onto the abstract roster. No coordinate, no per-row string."""
    tid, stem = str(spec["task_id"]), _stem(str(spec["task_id"]))
    supports = derived_support_cats(str(spec["scene"]))
    if not supports:
        raise RoleUnsatisfied(tid, "the pass", "work surface", str(spec["scene"]), [])
    props = derived_props(
        spec, set(spec["roster"]["objects"]) | set(spec["roster"]["containers"])
    )
    # ``open_top`` is a property of the BODY, not of the program. A body with a drivable leaf
    # registered open-top drops the object past a CLOSED door: measured on T4-DINER-02, where four
    # doored ``metal_bottom_cabinet`` bays taken open-top scored 0.25 with the executor never
    # opening anything. ``_register_swing_unit`` opens the leaf itself, so a doored body is always a
    # swing unit -- and T6-RDIN-01 (a row with no authored [Open]) has scored 1.00 that way.
    doored = body[1] in (asset_index().get(body[0]) or {}).get("door", set())
    del want_door
    units = tuple(
        UnitRecipe(
            k,
            body[0],
            body[1],
            f"{stem}__placard_{i + 1}.png",
            "ROUTING BAND",
            open_top=not doored,
            alt_cats=body[2],
        )
        for i, k in enumerate(sorted(spec["roster"]["containers"]))
    )
    no_look = not any(
        t.startswith("[LookAt]") for t in program_tokens(str(spec["human_program"]))
    )
    # A row that ships its OWN printed strings prints those instead of the derived LINE vocabulary,
    # and prints them PER BODY rather than per class (see ``_printed_block``). ``classes`` still
    # carries the unit strings so anything reading it generically reads the truth; nothing in this
    # mode indexes artwork by class.
    printed = _printed_block(spec)
    p_units = {str(k): str(v) for k, v in (printed.get("units") or {}).items()}
    if printed and sorted(p_units) != sorted(spec["roster"]["containers"]):
        raise SpecBuildError(
            f"{tid}: printed.units covers {sorted(p_units)} but the roster declares "
            f"{sorted(spec['roster']['containers'])}; a half-printed bank is unsolvable"
        )
    return Recipe(
        task_id=tid,
        scene=str(spec["scene"]),
        room=derived_room(spec, supports),
        support_cats=supports,
        units=units,
        items=items,
        decoys=decoys,
        classes=(
            tuple(p_units[k] for k in sorted(spec["roster"]["containers"]))
            if printed
            else bands
        ),
        cue_mode="printed" if printed else "target",
        class_units=tuple(class_units),
        decoy_class=_DERIVED_DECOY_BAND if decoys else "",
        unit_shape="twin",
        placard_title="ROUTING BAND",
        observe=tuple(u.key for u in units) if (no_look and not is_prose(spec)) else (),
        n_supports=max(1, len(props)),
        support_keys=props,
    )


def recipe_for(task_id: str) -> Recipe:
    """The AUTHORED pin if the row has one, else the derived scene-capability resolution.

    A pin always wins: the thirteen hand-measured rows are the ones whose worlds have been sited,
    scored and controlled end to end, and a derivation that quietly re-bodied one of them would
    regress a proven row to prove a general one.

    A pin is keyed on the ROW, not on the string. ``g0_normalise.variants_for`` ships a split row
    as ``<row>.<arm>`` specs and RETIRES the unsuffixed id, so an exact-id lookup alone would miss
    the pin on EVERY arm -- including the arm that was already shipping -- and silently re-body a
    measured world into a derived guess. ``source_task_id`` is each arm's own ledger of the row it
    came from, and ``_pin_for_arm`` follows it."""
    runtime_spec = _registered_runtime_spec(task_id)
    if runtime_spec is not None:
        return derive_recipe(_apply_unit_rename(runtime_spec))
    pins = recipes()
    r = pins.get(task_id)
    if r is not None:
        return r
    spec = load_spec(task_id)
    r = _pin_for_arm(pins, task_id, spec)
    return r if r is not None else derive_recipe(spec)


def _pin_for_arm(pins: dict[str, Recipe], task_id: str, spec: dict) -> Recipe | None:
    """The pin an ARM inherits from its source row, or None when the row has no pin.

    The pin is returned VERBATIM, its own ``task_id`` included, and that is deliberate rather than
    an oversight. Every PNG ``_labels_target`` / ``_labels_number`` authors is a function of the
    RECIPE alone (``class_order``, the drawn numbers, the authored basenames) and ``label_generator``
    SKIPS a file that already exists -- so two arms re-stamped with their own ids would author two
    different texts into one basename and whichever arm built first would silently supply the
    artwork for the second. Sharing the recipe keeps the artwork identical and leaves the arms to
    differ where they should: ``cue_class_for`` mounts each item's decal from THAT ARM's answer key.

    An arm that needs a different BODY -- an extra hold cabinet, a deeper preload, a limit the
    printed number must NOT bind -- is not this function's business. It gets its own entry in
    ``recipes()`` under the full arm id, which the exact-id lookup already prefers; no second
    mechanism is needed and none is offered here.

    If the arm's roster keys no longer match the pin (``spec_privacy.BUILDER_PINNED`` is keyed on the
    exact id too, so an arm of a pinned row is free to be re-lettered), ``assert_recipe_roles``
    raises and names both repairs. That failure is loud, which is the whole difference from the
    silent derivation this function replaces."""
    src = str(spec.get("source_task_id") or "")
    if not src or src == task_id:
        return None
    return pins.get(src)


# =====================================================================================
# seeded appearance permutation (BG-PERMUTE) -- the ANSWER stays fixed, the look moves
# =====================================================================================
def zone_permutation(recipe: Recipe, seed: int) -> dict[str, str]:
    """unit key -> cue class for this seed. The answer key is a property of the SPEC and
    never moves; what moves is which class is printed on which unit, so a model that
    memorised "the dairy tub goes in the third fridge" gets nothing."""
    import random

    classes = list(recipe.classes)
    if recipe.cue_mode == "printed":
        # ⛔ IDENTITY, never shuffled. In ``target`` mode the permutation is safe because the item's
        # decal is DERIVED from the permuted zone, so picture and key move together. A ``printed``
        # row pins both halves to the spec, so a shuffle here desynchronises three things at once and
        # MEASURED it did: the spec said unit_b="Discount"/unit_c="Send Back", the PNG hung on unit_b
        # said "Discount" (keyed by unit, unshuffled), while this map told the scene -- and through it
        # the person's object list -- that unit_b was "Send Back". The assistant then read "Discount"
        # off the shelf in the picture, said "Discount", and the person looked "Discount" up and
        # walked to a DIFFERENT shelf. Every arm in that wave scored 0.000 while doing the right
        # thing, and nothing in the logs said why.
        return {u.key: c for u, c in zip(recipe.units, classes)}
    if recipe.cue_mode == "target" and len(classes) != len(recipe.units):
        raise SpecBuildError(
            f"{recipe.task_id}: the seeded zone permutation needs one class per unit, but the "
            f"recipe declares {len(classes)} class(es) {list(classes)} against {len(recipe.units)} "
            f"unit(s) {[u.key for u in recipe.units]}. A silent index slip here would print the "
            f"wrong placard on the answer unit."
        )
    moving = [
        i
        for i, u in enumerate(recipe.units)
        if u.key in recipe.permuted_units() and i < len(classes)
    ]
    drawn = [classes[i] for i in moving]
    random.Random(f"{recipe.task_id}:{int(seed)}").shuffle(drawn)
    for i, c in zip(moving, drawn):
        classes[i] = c
    return {u.key: c for u, c in zip(recipe.units, classes)}


def cue_class_for(spec: dict, recipe: Recipe, seed: int) -> dict[str, str]:
    """item key -> the class its decal must show.

    Two derivations, never an authored coincidence. ``cue_mode="target"``: the class printed on the
    unit the ANSWER KEY sends the item to (slice A) -- the appearance permutes, the answer does not.
    ``cue_mode="intrinsic"``: the spec's own ``class_map``, because a hazard class belongs to the
    bottle, not to the seed; there the UNIT placard is the derived half (``unit_class_for``)."""
    if recipe.cue_mode == "intrinsic":
        return intrinsic_classes(spec, recipe)
    if recipe.cue_mode in ("number", "printed"):
        # the cue is a printed NUMBER / a per-item STRING, not a class shared with a unit; no
        # cue_class channel is published and the artwork is keyed by the body that wears it
        return {}
    zones = zone_permutation(recipe, seed)
    out: dict[str, str] = {}
    for g in spec.get("answer_key") or []:
        if g.get("pred", "inside") != "inside":
            continue
        target = str(g.get("target"))
        if target not in zones:
            raise SpecBuildError(
                f"answer key targets {target!r}, which is not a recipe unit"
            )
        out[str(g.get("subject"))] = zones[target]
    out.update(decoy_classes(recipe, seed))
    missing = [i.key for i in recipe.staged() if i.key not in out]
    if missing:
        raise SpecBuildError(f"no answer-key row for graspable(s) {missing}")
    return out


def decoy_classes(recipe: Recipe, seed: int) -> dict[str, str]:
    """decoy key -> the class its decal shows: a class NO scored unit's placard claims this seed.

    Derived, not authored. A spare that showed a live class would be a spare the rule table says
    belongs somewhere, which is exactly the partition the row asks the reader to compute -- and the
    ``not_inside{spares}`` would then be scoring a contradiction."""
    import random

    if not recipe.decoys:
        return {}
    if recipe.decoy_class:
        return {d.key: recipe.decoy_class for d in recipe.decoys}
    live = set(zone_permutation(recipe, seed).values())
    spare = [c for c in recipe.classes if c not in live] or list(recipe.classes)
    rng = random.Random(f"{recipe.task_id}:decoy:{int(seed)}")
    return {d.key: spare[rng.randrange(len(spare))] for d in recipe.decoys}


def staged_order(recipe: Recipe, seed: int) -> list[ItemRecipe]:
    """The graspables in BENCH order -- scored ones and decoys interleaved by the seed.

    A row whose spares are declared after its keyed items would stage them in two blocks, and the
    partition the reader is supposed to compute from the decals would be legible from position alone
    ("STAGING-ORDER EXPLOIT" is a published chance number for exactly this)."""
    import random

    got = list(recipe.staged())
    if recipe.decoys:
        random.Random(f"{recipe.task_id}:stage:{int(seed)}").shuffle(got)
    return got


def intrinsic_classes(spec: dict, recipe: Recipe) -> dict[str, str]:
    """item key -> its authored class, read from the spec's own ``group_of_expr.class_map``.

    Read from the SPEC rather than the recipe on purpose: the same literal that the resolver hands
    ``no_share_group`` is the one the decal prints, so the picture cannot disagree with the scorer."""
    known = {c.key for c in recipe.class_defs}
    for c in spec.get("negs") or []:
        expr = c.get("group_of_expr") or {}
        if str(expr.get("op")) != "class_map":
            continue
        got = {str(k): str(v) for k, v in dict(expr.get("v") or {}).items()}
        unknown = sorted(set(got.values()) - known)
        if unknown:
            raise SpecBuildError(
                f"class_map uses class(es) {unknown} the recipe cannot print"
            )
        missing = [i.key for i in recipe.items if i.key not in got]
        if missing:
            raise SpecBuildError(f"class_map does not classify graspable(s) {missing}")
        return got
    raise SpecBuildError(
        f"{spec['task_id']}: cue_mode='intrinsic' but no class_map is authored"
    )


def unit_class_for(spec: dict, recipe: Recipe) -> dict[str, str]:
    """unit key -> the class its placard must print: the class of every item the ANSWER KEY routes
    there. Derived, and ASSERTED unique -- two incompatible classes keyed into one unit would mean
    the spec contradicts its own ``no_share_group`` and no placard could be honest."""
    classes = intrinsic_classes(spec, recipe)
    per: dict[str, set[str]] = {}
    for g in spec.get("answer_key") or []:
        if g.get("pred", "inside") == "inside" and str(g.get("subject")) in classes:
            per.setdefault(str(g.get("target")), set()).add(classes[str(g["subject"])])
    split = {u: sorted(v) for u, v in per.items() if len(v) != 1}
    if split:
        raise SpecBuildError(
            f"the answer key routes >1 class into {split} -- no placard is honest"
        )
    missing = [u.key for u in recipe.units if u.key not in per]
    if missing:
        raise SpecBuildError(
            f"no answer-key row reaches unit(s) {missing}; their placard is undefined"
        )
    return {u: next(iter(v)) for u, v in per.items()}


def unit_assets(recipe: Recipe, seed: int) -> dict[str, tuple[str, str]]:
    """unit key -> the (category, model) that plays it THIS seed.

    For a ``target`` row the appearance already moves (the zone placards permute), so the assets are
    fixed. For an ``intrinsic`` / ``number`` row the class or the count is pinned to the answer key,
    so the permutable thing is which physical body plays which key -- otherwise 'the tall locker is
    the oxidiser cabinet' is memorisable across seeds."""
    import random

    fixed = {u.key: (u.cat, u.model) for u in recipe.units}
    if (
        recipe.frozen_bindings
        or recipe.cue_mode in {"target", "none"}
        or recipe.unit_shape == "twin"
        or len(fixed) < 2
    ):
        return fixed
    keys = [u.key for u in recipe.units]
    bodies = [fixed[k] for k in keys]
    random.Random(f"{recipe.task_id}:assets:{int(seed)}").shuffle(bodies)
    return dict(zip(keys, bodies))


def _draw_one(rng, nr: NumberRecipe) -> dict[str, int]:
    """One channel's per-key draw. ``date`` codes are drawn WITHOUT replacement: two cartons sharing
    a use-by code would make the bay range cards ambiguous and the row unanswerable."""
    keys = list(nr.keys)
    if nr.kind == "date":
        span = list(range(int(nr.lo), int(nr.hi) + 1))
        if len(span) < len(keys):
            raise SpecBuildError(
                f"channel {nr.channel!r} cannot draw {len(keys)} distinct codes"
            )
        return dict(zip(keys, sorted(rng.sample(span, len(keys)))))
    return {k: rng.randint(int(nr.lo), int(nr.hi)) for k in keys}


def draw_numbers(recipe: Recipe, seed: int) -> dict[str, dict[str, int]]:
    """channel -> {key: drawn integer}. This IS ``BuildEnv.cue_number``, and the same values are
    what ``ensure_number_labels`` prints, so no reader and no scorer can see different numbers."""
    import random

    out: dict[str, dict[str, int]] = {}
    for nr in recipe.numbers:
        out[nr.channel] = _draw_one(
            random.Random(f"{recipe.task_id}:{nr.channel}:{int(seed)}"), nr
        )
    return out


def number_text(nr: NumberRecipe, value: int) -> str:
    """How a drawn value PRINTS. A date code renders as a real date so the FIFO rule reads."""
    if nr.kind == "date":
        return f"2026-{int(nr.month):02d}-{int(value):02d}"
    return str(int(value))


# =====================================================================================
# ATTACHED -- mates derived from the parts' OWN metadata, never from the spec's anon names
# =====================================================================================
def attach_points_of(recipe: Recipe) -> dict[str, dict]:
    """abstract key -> ``{point_name: [{pos, quat}, ...]}`` from the part's ORIGINAL BEHAVIOR
    metadata (the bake strips the ``meta__`` attach Xforms, so this is the only source)."""
    import behavior_import as bi

    out = {p.key: bi._read_attach_points(p.cat, p.model) for p in recipe.parts}
    empty = sorted(k for k, v in out.items() if not v)
    if empty:
        raise SpecBuildError(
            f"part(s) {empty} carry NO attach-point metadata -- unmateable"
        )
    return out


def _male_point(points: dict, key: str, declared: str = "") -> str:
    """The part's SOLE Male connector. More than one would make the mate ambiguous, and the spec's
    connector name is KEY-ANON'd, so there is nothing to disambiguate it with."""
    males = sorted(k for k in points if k.endswith("M"))
    if declared:
        matches = [key for key in males if key[:-1].lower() == declared.lower()]
        if len(matches) != 1:
            raise SpecBuildError(
                f"part {key} has no unique declared Male connector {declared!r}; "
                f"metadata exposes {males}"
            )
        return matches[0]
    if len(males) != 1:
        raise SpecBuildError(
            f"part {key} exposes {len(males)} Male connectors {males}, need exactly 1"
        )
    return males[0]


def authored_attach_idx(human_program: str) -> list[tuple[str, int, int]]:
    """Per authored ``[Attach]``, in order: (child key, child connector idx, parent connector idx)."""
    import re

    out: list[tuple[str, int, int]] = []
    for tok in program_tokens(human_program):
        if not tok.startswith("[Attach]"):
            continue
        args = re.findall(r"<([^>]*)>\s*\(\s*(\d+)\s*\)", tok)
        if len(args) != 4:
            raise SpecBuildError(f"[Attach] is arity-4, got {len(args)} args: {tok!r}")
        out.append((args[0][0], int(args[1][1]), int(args[3][1])))
    return out


def derive_mates(spec: dict, recipe: Recipe, points: dict[str, dict]) -> list[Mate]:
    """The scored and frozen-initial tree edges -> resolved ``Mate``s. The connector NAMES come from the parts'
    metadata (the child's sole Male, then the same-base Female on the parent), so the KEY-ANON'd spec
    string is never trusted as a lookup key -- it is only cross-checked (``assert_anon_connectors``).
    A connector INDEX is authored (the back exposes six ``shelfbackF`` slots) and clamped to what the
    part really has, rather than left for ``produce_attach`` to silently clamp at render time."""
    authored = {
        c: (ci, pi) for c, ci, pi in authored_attach_idx(str(spec["human_program"]))
    }
    out: list[Mate] = []
    is_bts = isinstance(spec.get("bts_source"), dict)
    rows = list(spec.get("answer_key_raw") or [])
    if is_bts:
        rows = [
            [row.get("subject"), row.get("target"), row.get("connector")]
            for row in ((spec.get("bts_source") or {}).get("goal") or {}).get(
                "answer_key", ()
            )
            if row.get("pred") == "attached"
        ]
        seen_children = {str(row[0]) for row in rows}
        for initial in (spec.get("world") or {}).get("initial_attachments") or []:
            child = str(initial.get("child") or "")
            if child in seen_children:
                continue
            rows.append(
                [
                    child,
                    str(initial.get("parent") or ""),
                    str(initial.get("connector") or ""),
                ]
            )
            seen_children.add(child)
    for row in rows:
        child, parent, anon = str(row[0]), str(row[1]), str(row[2])
        if child not in points or parent not in points:
            raise SpecBuildError(
                f"mate {child}->{parent} names a key outside the recipe parts"
            )
        cp = _male_point(points[child], child, anon if is_bts else "")
        pp = f"{cp[:-1]}F"
        plist = points[parent].get(pp)
        if not plist:
            raise SpecBuildError(
                f"parent {parent} has no {pp} female for child {child} -- the answer key states a "
                f"mate the geometry cannot make"
            )
        ci, pi = authored.get(child, (0, 0))
        out.append(
            Mate(
                child,
                parent,
                anon,
                cp,
                pp,
                min(ci, len(points[child][cp]) - 1),
                min(pi, len(plist) - 1),
            )
        )
    if not out:
        raise SpecBuildError(
            f"{spec['task_id']}: no ATTACHED answer-key row to build a mate from"
        )
    return out


def assert_anon_connectors(mates: list[Mate], recipe: Recipe) -> None:
    """KEY-ANON both ways. The spec must NOT carry the real connector base (``shelfsidebxfkjj`` names
    the very board the part-ID trap is about -> one probe turn is a no-pixels solve), yet the two must
    still describe the SAME joint, or the goal string and the built world have quietly diverged."""
    models = {p.model for p in recipe.parts}
    for m in mates:
        stem = m.anon.rstrip("LR") or m.anon
        real = m.child_point[:-1]
        if not real.startswith(stem):
            raise SpecBuildError(
                f"anon connector {m.anon!r} does not name the real joint {real!r}"
            )
        leak = sorted(mid for mid in models if mid in m.anon.lower())
        if leak:
            raise SpecBuildError(
                f"connector {m.anon!r} LEAKS model id(s) {leak} into the spec (KEY-ANON)"
            )
    anon = [m.anon for m in mates]
    if len(set(anon)) != len(anon):
        raise SpecBuildError(f"anon connector names are not unique: {anon}")
    print(
        f"[bench] {len(mates)} mate(s) resolved: "
        + " ".join(
            f"{m.child}-[{m.anon}]->{m.parent}({m.parent_point}#{m.pp_idx})"
            for m in mates
        )
    )


def assembly_roots(mates: list[Mate], recipe: Recipe) -> list[str]:
    """Every part that is never a child, one root per independent assembly."""
    roots = sorted({m.parent for m in mates} - {m.child for m in mates})
    if not roots:
        raise SpecBuildError("assembly mate graph has no root")
    part_keys = {part.key for part in recipe.parts}
    if not set(roots).issubset(part_keys):
        raise SpecBuildError(f"assembly roots are absent from recipe parts: {roots}")
    return roots


def assembly_root(mates: list[Mate], recipe: Recipe) -> str:
    """Primary root retained for camera/cue placement in legacy single-root code."""
    return assembly_roots(mates, recipe)[0]


# =====================================================================================
# labels -- generate deterministically, then PROVE every declared PNG is on disk
# =====================================================================================
def ensure_labels(
    recipe: Recipe,
    spec: dict | None = None,
    draw: dict[str, dict[str, int]] | None = None,
    label_dir: str | None = None,
) -> dict[str, str]:
    """Author every PNG this row needs and return {png_name: abs_path}.

    ``label_generator`` is byte-stable and skips an existing file, so re-running a build never
    perturbs the artwork. Raises if anything is still absent afterwards -- the mounting code
    SILENTLY SKIPS a missing PNG, so a gap here renders an unlabelled, unsolvable episode."""
    label_dir = str(label_dir or GENERATED_LABEL_DIR)
    os.makedirs(label_dir, exist_ok=True)
    out: dict[str, str] = {}
    source_spec = dict(spec or {})
    if recipe.cue_mode == "none":
        return out
    if recipe.cue_mode == "intrinsic":
        _labels_intrinsic(recipe, source_spec, out, label_dir)
    elif recipe.cue_mode == "number":
        _labels_number(recipe, source_spec, dict(draw or {}), out, label_dir)
    elif recipe.cue_mode == "printed":
        _labels_printed(recipe, source_spec, out, label_dir)
    else:
        _labels_target(recipe, out, label_dir, source_spec)
    absent = sorted(n for n, p in out.items() if not os.path.exists(p))
    if absent:
        raise SpecBuildError(f"label PNG(s) still absent after generation: {absent}")
    return out


def _printed_block(spec: dict) -> dict:
    """The row's OWN printed strings, or ``{}`` for a row that uses the derived cue vocabulary.

    ⛔ WHY THIS EXISTS, measured 2026-08-12. The derived builder is a DECAL MACHINE: one graspable
    body plays every item and one container body plays every unit (``pick_item_body``: *"the
    graspable is not the cue. What the reader has to read is the DECAL"*). That is a good fit for a
    GENERATED task whose discriminator is a printed string -- and the v4 generator writes exactly
    those -- but the strings were being dropped at the adapter and replaced with ``LINE ALFA`` /
    ``LINE BRAVO``. A grocery row whose six bottles are told apart by ``EXP 2022`` vs ``EXP 2026``
    therefore built as **six identical bottles wearing LINE-name bands**: the thing the row is
    scored on did not exist in the world.

    ⭐ And it would not have looked broken. The oracle walks by KEY and never reads a label, so it
    still scores 1.00; only a model can fail. Build green + oracle 1.00 + models near chance reads
    as "the models are weak" -- the project's banner failure, one more time.

    Shape::

        "printed": {
          "units":  {"unit_a": "Main", "unit_b": "Discount", "unit_c": "Send Back"},
          "items":  {"item_1": "EXP 2025", ...},          # one per staged graspable
          "unit_title": "SECTION", "item_title": "BEST BEFORE", "item_sub": "expires",
          "board_title": "ROUTING RULES",                  # optional printed rule board
          "board_rows": [["Past years", "Send Back"], ["This month", "Discount"]]
        }

    Unlike ``target`` mode the two halves are INDEPENDENT: nothing derives the unit placard from the
    item decal or back. That is the whole point -- a routing rule ("past years -> send back") is a
    task precisely because the two strings differ and something in the world has to join them.
    """
    p = spec.get("printed")
    if isinstance(p, dict):
        return dict(p)
    world = dict(spec.get("world") or {})
    units = _inline_printed_rows(world.get("containers"))
    items = _inline_printed_rows(world.get("objects"))
    return {"units": units, "items": items, "sparse": True} if units or items else {}


def item_decal_mount_face(spec: Mapping[str, object], item_key: str) -> str | None:
    """Return an explicitly authored item face without changing legacy auto placement."""
    faces = _printed_block(dict(spec)).get("item_mount_faces")
    if faces is None:
        return None
    if not isinstance(faces, Mapping):
        raise SpecBuildError("printed.item_mount_faces must be an object")
    face = faces.get(item_key)
    if face is None:
        return None
    if not isinstance(face, str):
        raise SpecBuildError(f"item decal mount face for {item_key} must be a string")
    return face


def item_decal_look_held_distance(
    spec: Mapping[str, object], item_key: str
) -> float | None:
    """Return an explicitly authored close-reading distance for one held item label."""
    items = _printed_block(dict(spec)).get("items")
    value = items.get(item_key) if isinstance(items, Mapping) else None
    content = value.get("content") if isinstance(value, Mapping) else None
    args = content.get("args") if isinstance(content, Mapping) else None
    raw = args.get("look_held_distance_m") if isinstance(args, Mapping) else None
    if raw is None:
        return None
    try:
        distance = float(raw)
    except (TypeError, ValueError) as exc:
        raise SpecBuildError(
            f"item decal held-look distance for {item_key} must be numeric"
        ) from exc
    if not 0.30 <= distance <= 0.80:
        raise SpecBuildError(
            f"item decal held-look distance for {item_key} must be within 0.30..0.80 m"
        )
    return distance


def item_decal_look_target_glyph_px(
    spec: Mapping[str, object], item_key: str
) -> int | None:
    """Return an explicitly authored held-label planning target for one item."""
    items = _printed_block(dict(spec)).get("items")
    value = items.get(item_key) if isinstance(items, Mapping) else None
    content = value.get("content") if isinstance(value, Mapping) else None
    args = content.get("args") if isinstance(content, Mapping) else None
    raw = args.get("look_target_glyph_px") if isinstance(args, Mapping) else None
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise SpecBuildError(
            f"item decal held-look glyph target for {item_key} must be an integer"
        )
    try:
        target = int(raw)
    except (TypeError, ValueError) as exc:
        raise SpecBuildError(
            f"item decal held-look glyph target for {item_key} must be an integer"
        ) from exc
    if target != raw or not 10 <= target <= 32:
        raise SpecBuildError(
            f"item decal held-look glyph target for {item_key} must be an integer "
            "within 10..32 px"
        )
    return target


def _inline_printed_rows(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(raw, dict):
            continue
        label = raw.get("label")
        if isinstance(label, str) and label.strip():
            out[str(key)] = label.strip()
    return out


def _printed_value_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value.get("kind") and value.get("content"))
    return bool(str(value).strip())


def _structured_label_fields(
    value: Mapping[str, object], *, title: str, sub: str
) -> tuple[str, str, dict[str, object]]:
    content = value.get("content")
    if not isinstance(content, Mapping):
        raise SpecBuildError("structured printed label requires an object content field")
    args = content.get("args")
    fields = dict(args) if isinstance(args, Mapping) else {}
    source_kind = str(value.get("kind") or "short_string")
    renderer = str(fields.pop("renderer", source_kind))
    if renderer == "nutrition_report":
        fields.setdefault("title", str(content.get("text") or title))
        fields.setdefault("rows", list(content.get("lines") or ()))
    else:
        for key in ("text", "number"):
            if content.get(key) is not None:
                fields.setdefault(key, content[key])
        if content.get("lines") is not None:
            fields.setdefault(
                "rows" if renderer == "range_card" else "text",
                content["lines"],
            )
        icons = content.get("icons")
        if icons:
            fields.setdefault("icon", list(icons)[0])  # pyre-ignore[16]
        fields.setdefault("title", title)
    fields.setdefault("sub", sub)
    stem = str(value.get("png") or fields.get("title") or renderer)
    return renderer, stem, fields


def _printed_label_fields(
    value: object, *, title: str, sub: str
) -> tuple[str, str, dict[str, object]]:
    if isinstance(value, Mapping):
        return _structured_label_fields(value, title=title, sub=sub)
    text = str(value).strip()
    return "short_string", text, {"title": title, "text": text, "sub": sub}


def _slug(text: str, n: int = 24) -> str:
    """A filename-safe stem for a printed string.

    ⚠️ The text MUST reach the file name. ``label_generator`` never overwrites, so a name that did not
    carry the string would freeze the first row's artwork onto every later row that reused the name
    and silently decouple the sticker from the world -- the exact trap ``_labels_number`` documents
    for drawn numbers.
    """
    raw = str(text).lower()
    s = re.sub(r"[^a-z0-9]+", "_", raw).strip("_") or "blank"
    if len(s) <= n:
        return s
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{s[: n - len(digest) - 1]}_{digest}"


def _labels_printed(
    recipe: Recipe, spec: dict, out: dict[str, str], label_dir: str
) -> None:
    """Slice D: every string comes from the SPEC, one card per unit and one decal per ITEM.

    Keyed ``unit:<k>`` / ``item:<k>`` rather than by class, exactly as the ``number`` mode is: in
    this mode two items routed to the same unit still print DIFFERENT strings (``Coffee Pods`` and
    ``Biscuits`` both go to the pantry), so there is no class to index artwork by.
    """
    from label_generator import label_generator

    printed = _printed_block(spec)
    units = {str(k): v for k, v in (printed.get("units") or {}).items()}
    items = {str(k): v for k, v in (printed.get("items") or {}).items()}
    sparse = bool(printed.get("sparse"))
    stem = _stem_of(recipe)
    unit_keys = {unit.key for unit in recipe.units}
    item_order = [item.key for item in recipe.staged()] + [
        key for preload in recipe.preloads for key in preload.keys
    ]
    item_order = list(dict.fromkeys(item_order))
    item_keys = set(item_order)
    unknown_u = sorted(set(units) - unit_keys)
    unknown_i = sorted(set(items) - item_keys)
    if unknown_u or unknown_i:
        raise SpecBuildError(
            f"{recipe.task_id}: printed strings name unknown unit(s) {unknown_u} / "
            f"unstaged item(s) {unknown_i}"
        )
    missing_u = (
        []
        if sparse
        else [u.key for u in recipe.units if not _printed_value_present(units.get(u.key))]
    )
    missing_i = (
        []
        if sparse
        else [key for key in item_order if not _printed_value_present(items.get(key))]
    )
    if missing_u or missing_i:
        raise SpecBuildError(
            f"{recipe.task_id}: cue_mode='printed' but no string for unit(s) {missing_u} / "
            f"item(s) {missing_i}; an unprinted body is an unsolvable row, not a default"
        )
    for u in recipe.units:
        value = units.get(u.key)
        if not _printed_value_present(value):
            continue
        kind, label_stem, fields = _printed_label_fields(
            value,
            title=str(printed.get("unit_title") or recipe.placard_title),
            sub=str(printed.get("unit_sub") or ""),
        )
        p = os.path.join(label_dir, f"{stem}__pu_{_slug(label_stem)}.png")
        label_generator(kind, p, mount="container", **fields)
        out[f"unit:{u.key}"] = p
    for key in item_order:
        value = items.get(key)
        if not _printed_value_present(value):
            continue
        kind, label_stem, fields = _printed_label_fields(
            value,
            title=str(printed.get("item_title") or ""),
            sub=str(printed.get("item_sub") or ""),
        )
        p = os.path.join(label_dir, f"{stem}__pi_{_slug(label_stem)}.png")
        label_generator(kind, p, mount="graspable", **fields)
        out[f"item:{key}"] = p
    _labels_printed_walls(recipe, printed, out, label_dir)
    _labels_props(recipe, out, label_dir, spec)
    _labels_printed_board(recipe, printed, out, label_dir)


def _printed_wall_entries(printed: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw_walls = printed.get("walls") or {}
    if not isinstance(raw_walls, Mapping):
        raise SpecBuildError("printed.walls must be an object")
    entries: dict[str, Mapping[str, object]] = {}
    anchors: set[str] = set()
    for raw_key, raw in raw_walls.items():
        key = str(raw_key)
        if not isinstance(raw, Mapping):
            raise SpecBuildError(f"printed.walls.{key} must be an object")
        anchor = str(raw.get("on") or "")
        label = raw.get("label")
        if not anchor or not isinstance(label, Mapping):
            raise SpecBuildError(
                f"printed.walls.{key} requires an anchor in 'on' and an object 'label'"
            )
        if anchor in anchors:
            raise SpecBuildError(f"wall anchor {anchor!r} carries more than one label")
        anchors.add(anchor)
        entries[key] = raw
    return entries


def _labels_printed_walls(
    recipe: Recipe,
    printed: Mapping[str, object],
    out: dict[str, str],
    label_dir: str,
) -> None:
    from label_generator import label_generator

    for key, mount in _printed_wall_entries(printed).items():
        label = mount["label"]
        if not isinstance(label, Mapping):
            raise SpecBuildError(f"printed.walls.{key}.label must be an object")
        content = label.get("content")
        title = str(content.get("text") or "") if isinstance(content, Mapping) else ""
        kind, label_stem, fields = _printed_label_fields(
            label,
            title=title,
            sub="",
        )
        name = f"{_stem_of(recipe)}__pw_{_slug(f'{key}_{label_stem}', 40)}.png"
        path = os.path.join(label_dir, name)
        label_generator(kind, path, mount="wall", **fields)
        out[f"wall:{key}"] = path


def _labels_printed_board(
    recipe: Recipe, printed: dict, out: dict[str, str], label_dir: str
) -> None:
    """The free-standing RULE BOARD, mounted on the row's first staging prop.

    Only a row whose rule cannot be inferred needs one: "past years -> send back" is arbitrary, so
    without the board the task is unanswerable; "toner cartridge -> IT equipment" is common sense
    and the board would be scaffolding. Runs AFTER ``_labels_props`` on purpose -- it replaces that
    prop's generic "STAGED HERE" card rather than being replaced by it.
    """
    from label_generator import label_generator

    rows = [
        (str(a), str(b)) for a, b in (printed.get("board_rows") or []) if str(a).strip()
    ]
    if not rows or not recipe.support_keys:
        return
    key = recipe.support_keys[0]
    name = (
        f"{_stem_of(recipe)}__board_" + _slug("_".join(a for a, _ in rows), 40) + ".png"
    )
    p = os.path.join(label_dir, name)
    label_generator(
        "range_card",
        p,
        mount="container",
        title=str(printed.get("board_title") or "RULES"),
        rows=rows,
        sub="",
    )
    out[f"prop:{key}"] = p


def _labels_target(
    recipe: Recipe,
    out: dict[str, str],
    label_dir: str,
    spec: Mapping[str, object] | None = None,
) -> None:
    """Slice A: one zone placard per class + one item decal per class, index-aligned with
    ``classes`` so the seeded permutation only swaps WHICH png hangs on WHICH unit."""
    from label_generator import label_generator

    zones = class_order(recipe)
    for u, zone in zip(recipe.units, zones):
        p = os.path.join(label_dir, u.png)
        label_generator(
            "short_string",
            p,
            mount="container",
            title=u.title,
            text=zone,
            sub="allergen zone",
        )
        out[u.png] = p
    for item in recipe.staged():
        for png, icon, zone in zip(item.pngs, item.icons, zones):
            p = os.path.join(label_dir, png)
            label_generator(
                "pictogram", p, mount="graspable", icon=icon, text=zone, sub="contains"
            )
            out[png] = p
    _labels_props(recipe, out, label_dir, spec)


def _auto_support_labels_enabled(spec: Mapping[str, object]) -> bool:
    for block_name in ("printed", "cues"):
        block = spec.get(block_name)
        if isinstance(block, Mapping) and block.get("auto_support_labels") is False:
            return False
    return True


def _labels_props(
    recipe: Recipe,
    out: dict[str, str],
    label_dir: str,
    spec: Mapping[str, object] | None = None,
) -> None:
    """The non-unit props a prose row looks at: the staging trays and the free-standing rule card."""
    from label_generator import label_generator

    if not _auto_support_labels_enabled(spec or {}):
        return
    printed = _printed_block(dict(spec or {}))
    wall_anchors = {
        str(entry.get("on"))
        for entry in _printed_wall_entries(printed).values()
    }
    for k in recipe.support_keys:
        if k in wall_anchors:
            continue
        if re.match(r"^(?:surf(?:ace)?|fixture)(?:[_-]?[a-z0-9]+)$", k, re.I):
            continue
        p = os.path.join(label_dir, f"{_stem_of(recipe)}__{k}.png")
        label_generator(
            "short_string",
            p,
            mount="container",
            title=recipe.support_title,
            text=k.replace("_", " ").upper(),
            sub="staged here",
        )
        out[f"prop:{k}"] = p


def assembly_label_pngs(
    recipe: Recipe, label_dir: str = GENERATED_LABEL_DIR
) -> dict[str, str]:
    """Generate only explicitly authored decals for free flat-pack parts."""
    from label_generator import label_generator

    out: dict[str, str] = {}
    for part in recipe.parts:
        text = part.label.strip()
        if not text:
            continue
        path = os.path.join(
            label_dir, f"{_stem_of(recipe)}__part_{_slug(text)}.png"
        )
        label_generator(
            "short_string",
            path,
            mount="graspable",
            title="FLAT-PACK BOOKCASE",
            text=text,
            sub="read before assembly",
        )
        out[part.key] = path
    absent = sorted(key for key, path in out.items() if not os.path.exists(path))
    if absent:
        raise SpecBuildError(f"assembly label PNG(s) still absent for {absent}")
    return out


def _labels_card(
    recipe: Recipe, spec: dict, draw, out: dict[str, str], label_dir: str
) -> None:
    """The LEDGER card: every code the requisition lists, INCLUDING the one already inside the unit.

    The card is generated from the same draw the pieces are stamped from, and it deliberately lists
    one code more than the bench carries -- the difference between the card and the trays IS the
    task, so a card built only from the unplaced items would hand the answer over."""
    from label_generator import label_generator

    if not recipe.card_key:
        return
    unit = _sole_scored_unit(reference_answers(spec))[0]
    subs = _keyed_subjects(spec, unit) + _filler_keys(
        unit, _pinned_preload(recipe, unit)
    )
    rows = [
        (number_text(nr, draw[nr.channel][s]), "WASH")
        for nr in recipe.numbers
        if nr.domain == "items"
        for s in sorted(subs)
        if s in draw.get(nr.channel, {})
    ]
    if not rows:
        raise SpecBuildError(f"card {recipe.card_key!r} has no coded rows to print")
    name = f"{_stem_of(recipe)}__card_" + "_".join(r[0] for r in rows) + ".png"
    p = os.path.join(label_dir, name)
    label_generator(
        "range_card", p, mount="container", title=recipe.card_title, rows=rows, sub=""
    )
    out[f"prop:{recipe.card_key}"] = p


def _pinned_preload(recipe: Recipe, unit: str) -> int:
    """How many fillers ``unit`` starts with, and it must be PINNED (``lo == hi``). A card that lists
    the already-done pieces cannot be printed off a count that the seed still moves."""
    got = [p for p in recipe.preloads if p.unit == unit]
    loose = [p.unit for p in got if p.lo != p.hi]
    if loose:
        raise SpecBuildError(
            f"a ledger card needs a pinned preload; {loose} still draws lo!=hi"
        )
    return sum(int(p.lo) for p in got)


def _labels_intrinsic(
    recipe: Recipe, spec: dict, out: dict[str, str], label_dir: str
) -> None:
    """One placard per unit printing the class its answer-key items carry, one GHS pictogram per
    class for the bottles. The PNG name carries the CLASS, never the unit key, so the per-seed asset
    permutation cannot leave a stale placard on a body."""
    from label_generator import label_generator

    titles = {c.key: c for c in recipe.class_defs}
    stem = recipe.task_id.lower().replace("-", "_")
    for cls in sorted(
        {
            *unit_class_for(spec, recipe).values(),
            *intrinsic_classes(spec, recipe).values(),
        }
    ):
        cd = titles[cls]
        p = os.path.join(label_dir, f"{stem}__placard_{cls}.png")
        label_generator(
            "short_string",
            p,
            mount="container",
            title=recipe.placard_title,
            text=cd.title,
            sub="store here",
        )
        out[f"placard:{cls}"] = p
        q = os.path.join(label_dir, f"{stem}__ghs_{cls}.png")
        label_generator(
            "pictogram",
            q,
            mount="graspable",
            icon=cd.icon,
            text=cd.title,
            sub="hazard class",
        )
        out[f"decal:{cls}"] = q
    _labels_props(recipe, out, label_dir, spec)


def _labels_number(
    recipe: Recipe,
    spec: dict,
    draw: dict[str, dict[str, int]],
    out: dict[str, str],
    label_dir: str,
) -> None:
    """The numeric channel. Every card is generated FROM THE DRAWN VALUE and its file name carries
    that value -- ``label_generator`` never overwrites, so a name that did not carry it would freeze one
    seed's number onto every later seed and silently decouple the sign from the world."""
    from label_generator import label_generator

    for nr in recipe.numbers:
        got = draw.get(nr.channel) or {}
        if sorted(got) != sorted(nr.keys):
            raise SpecBuildError(
                f"channel {nr.channel!r} drew {sorted(got)}, expected {sorted(nr.keys)}"
            )
        for k, v in sorted(got.items()):
            txt = number_text(nr, v)
            p = os.path.join(label_dir, f"{_stem_of(recipe)}__{nr.channel}_{v}.png")
            kind = "date" if nr.kind == "date" else "number"
            label_generator(
                kind,
                p,
                mount=("graspable" if nr.domain == "items" else "container"),
                title=nr.title,
                text=txt,
                number=txt,
                sub=nr.sub,
            )
            out[f"{nr.channel}:{k}"] = p
    coded = {k for nr in recipe.numbers if nr.domain == "items" for k in nr.keys}
    for it in (
        recipe.staged()
    ):  # an item whose cue lives on the UNIT still needs an identity decal
        if it.key in coded:
            continue
        p = os.path.join(label_dir, f"{_stem_of(recipe)}__item_card.png")
        label_generator(
            "short_string",
            p,
            mount="graspable",
            title="CASE",
            text="ONE FACING",
            sub="stock",
        )
        out[f"item:{it.key}"] = p
    _labels_bay_cards(recipe, spec, draw, out, label_dir)
    _labels_props(recipe, out, label_dir, spec)
    _labels_card(recipe, spec, draw, out, label_dir)


def _labels_bay_cards(
    recipe: Recipe,
    spec: dict,
    draw: dict[str, dict[str, int]],
    out: dict[str, str],
    label_dir: str,
) -> None:
    """The UNIT half of a numeric row. A unit that carries its own drawn number gets that card; a
    unit that receives item-coded stock gets a RANGE CARD listing exactly the codes the answer key
    sends it, so reading the carton and reading the card is a complete, pixel-only solve."""
    from label_generator import label_generator

    unit_ch = {nr.channel: nr for nr in recipe.numbers if nr.domain == "units"}
    item_ch = [nr for nr in recipe.numbers if nr.domain == "items"]
    for u in recipe.units:
        hit = [c for c, nr in unit_ch.items() if u.key in nr.keys]
        if hit:
            out[f"unit:{u.key}"] = out[f"{hit[0]}:{u.key}"]
            continue
        rows = [
            (number_text(nr, draw[nr.channel][s]), "ACCEPT")
            for nr in item_ch
            for s in sorted(_keyed_subjects(spec, u.key))
            if s in draw.get(nr.channel, {})
        ]
        name = (
            f"{_stem_of(recipe)}__card_{u.key}_" + "_".join(r[0] for r in rows) + ".png"
        )
        p = os.path.join(
            label_dir, name if rows else f"{_stem_of(recipe)}__card_{u.key}.png"
        )
        if rows:
            label_generator(
                "range_card", p, mount="container", title=u.title, rows=rows, sub=""
            )
        else:
            label_generator(
                "short_string",
                p,
                mount="container",
                title=u.title,
                text=u.title,
                sub="",
            )
        out[f"unit:{u.key}"] = p


def _stem_of(recipe: Recipe) -> str:
    return _stem(recipe.task_id)


def _stem(task_id: str) -> str:
    return str(task_id).lower().replace("-", "_")


def _keyed_subjects(spec: dict, unit: str) -> list[str]:
    return [
        str(g["subject"])
        for g in spec.get("answer_key") or []
        if g.get("pred", "inside") == "inside" and str(g.get("target")) == unit
    ]


def class_order(recipe: Recipe) -> list[str]:
    """Artwork slot order: one slot per unit class, then the decoy band if the row declares one."""
    return list(recipe.classes) + ([recipe.decoy_class] if recipe.decoy_class else [])


def _class_index(recipe: Recipe, zone: str, n_slots: int, what: str) -> int:
    """``classes.index(zone)`` with the two failure modes named instead of an IndexError.

    A cue class the recipe never declares, or a class list longer than the artwork it indexes,
    both used to surface as a bare ``IndexError`` deep in the label pass -- the kind of crash
    that gets "fixed" by disabling the permutation, which silently reopens the naming leak.
    """
    classes = class_order(recipe)
    if zone not in classes:
        raise SpecBuildError(
            f"{recipe.task_id}: {what} asks for cue class {zone!r}, which the recipe does not "
            f"declare; it ships {classes}"
        )
    idx = classes.index(zone)
    if idx >= n_slots:
        raise SpecBuildError(
            f"{recipe.task_id}: {what} indexes artwork slot {idx} of class {zone!r}, but only "
            f"{n_slots} slot(s) exist ({len(classes)} class(es) declared)"
        )
    return idx


def placard_png_for(recipe: Recipe, unit_key: str, zone: str) -> str:
    """The ``ensure_labels`` key of the card this unit carries. For a ``target`` row the placards are
    index-aligned with ``classes``, so the seeded permutation swaps WHICH png hangs on WHICH unit;
    for the other two modes the card is keyed by the derived class / by the unit itself."""
    if recipe.cue_mode == "intrinsic":
        return f"placard:{zone}"
    if recipe.cue_mode in ("number", "printed"):
        return f"unit:{unit_key}"
    idx = _class_index(recipe, zone, len(recipe.units), f"placard for {unit_key}")
    return recipe.units[idx].png


def decal_png_for(item: ItemRecipe, recipe: Recipe, zone: str) -> str:
    """The ``ensure_labels`` key of the decal this graspable carries."""
    if recipe.cue_mode == "intrinsic":
        return f"decal:{zone}"
    if recipe.cue_mode == "printed":
        return f"item:{item.key}"
    if recipe.cue_mode == "number":
        chans = [
            nr.channel
            for nr in recipe.numbers
            if nr.domain == "items" and item.key in nr.keys
        ]
        if len(chans) > 1:
            raise SpecBuildError(
                f"{item.key} is carried by {len(chans)} item number channel(s) {chans}"
            )
        return f"{chans[0]}:{item.key}" if chans else f"item:{item.key}"
    return item.pngs[
        _class_index(recipe, zone, len(item.pngs), f"decal for {item.key}")
    ]


def letter_permutation(recipe: Recipe, seed: int) -> dict[str, str]:
    """part key -> the letter stamped on that board, for this seed. The ANSWER (which board carries
    the ``shelfbase`` female) is a property of the geometry and never moves; what moves is which
    letter is printed on it, so a model that memorised 'C is the socket side' gets nothing and has to
    read the manual AND the boards."""
    import random

    letters = list(recipe.classes)
    random.Random(f"{recipe.task_id}:letters:{int(seed)}").shuffle(letters)
    return {p.key: c for p, c in zip(recipe.parts, letters)}


def board_pngs(
    recipe: Recipe, letters: dict[str, str], label_dir: str = GENERATED_LABEL_DIR
) -> dict[str, str]:
    """part key -> its letter decal PNG. These six ship on disk; a MISSING one is a silent skip in
    ``apply_board_labels``, which renders two indistinguishable side panels and an unsolvable row."""
    out = {
        k: os.path.join(label_dir, recipe.part_png % v.lower())
        for k, v in letters.items()
    }
    absent = sorted(k for k, p in out.items() if not os.path.exists(p))
    if absent:
        raise SpecBuildError(
            f"board letter PNG(s) absent for {absent}: {[out[k] for k in absent]}"
        )
    return out


def manual_rows(
    recipe: Recipe, letters: dict[str, str], mates: list[Mate]
) -> list[tuple[str, str]]:
    """The MANUAL's part table: one row per board, ROLE on the left and this seed's LETTER on the
    right. Rule-level -- it names roles and letters, never a board key, so it cannot be followed
    without looking at the boards."""
    order = list(
        dict.fromkeys([*assembly_roots(mates, recipe), *(m.child for m in mates)])
    )
    by_key = {p.key: p for p in recipe.parts}
    return [(by_key[k].label, letters[k]) for k in order if k in by_key]


def ensure_manual(
    recipe: Recipe, seed: int, rows, label_dir: str = GENERATED_LABEL_DIR
) -> list[str]:
    """Author the two manual panels for this seed and PROVE both are on disk. The file name carries
    the seed because ``label_generator`` never overwrites: one fixed name would freeze seed 0's letter
    table onto every later seed and silently decouple the sign from the boards."""
    from label_generator import label_generator

    stem = recipe.task_id.lower().replace("-", "_")
    order = [
        ("1. BOTH SIDES ONTO THE BACK", ""),
        ("2. TOP ONTO THE BACK", ""),
        ("3. BASEBOARD ONTO THE SOCKET SIDE", ""),
        ("4. SHELF INTO A BACK SLOT", ""),
        ("5. RAISE THE UNIT UPRIGHT", "LAST"),
        ("ONCE RAISED THE TOP IS OUT OF REACH", ""),
    ]
    out = []
    for tag, title, table in (
        ("parts", recipe.placard_title, rows),
        ("order", "ASSEMBLY ORDER", order),
    ):
        p = os.path.join(label_dir, f"{stem}__manual_{tag}_s{int(seed)}.png")
        label_generator("range_card", p, mount="wall", title=title, rows=table, sub="")
        out.append(p)
    absent = [p for p in out if not os.path.exists(p)]
    if absent:
        raise SpecBuildError(f"manual panel(s) still absent after generation: {absent}")
    return out


# =====================================================================================
# program -- authored opening kept VERBATIM, answer slots filled
# =====================================================================================
def program_tokens(human_program: str) -> list[str]:
    """Split an authored program into complete commands without losing scalar arguments.

    Two purely SYNTACTIC normalisations run first, because two whole families write the same
    commands in a different notation and the difference is not a difference in meaning:

    * ``<u_prog_SUBJECT>`` -> ``<?>``. Nineteen T4 rows write the answer slot in that named form.
      Measured: no ``u_prog_*`` name appears in any roster and every one names a real answer
      subject. Without the rewrite ``fill_answer_slots`` leaves it verbatim and the wrong-answer
      control silently drives the ORACLE program.
    * ``name(1)`` -> ``<name>(1)``. About forty rows (the whole T2 recall family) omit the angle
      brackets. Without this the command compiler sees bare prose instead of entity arguments, so
      ``[Open] unit_01(1), ... [PutInside] jar_3(1) <?>(1)`` cannot be grounded."""
    import re

    hp = re.sub(r"<\s*u_prog_[^>]+>", "<?>", str(human_program))
    hp = re.sub(r"(?<![<\w])([A-Za-z_][A-Za-z0-9_]*)\s*\((\d+)\)", r"<\1>(\2)", hp)
    toks = command_grammar.split_program(hp)
    if not toks or any(re.match(r"^\[[A-Za-z_]+\]", token) is None for token in toks):
        raise SpecBuildError(f"unparseable human_program: {human_program!r}")
    return [" ".join(t.split()) for t in toks]


def program_fixture_capabilities(human_program: str) -> dict[str, frozenset[str]]:
    """Return physical capabilities required of each named command argument."""
    if not str(human_program).strip():
        return {}
    requirements: dict[str, set[str]] = {}
    verb_requirements = {
        "OPEN": (0, "openable"),
        "CLOSE": (0, "openable"),
        "OPENDOOR": (0, "openable"),
        "CLOSEDOOR": (0, "openable"),
        "TURNON": (0, "toggleable"),
        "TURNOFF": (0, "toggleable"),
        "SIT": (0, "seat"),
        "PUT": (1, "surface"),
        "PUTON": (1, "surface"),
        "PUTBACK": (1, "surface"),
        "PUTOBJBACK": (1, "surface"),
        "PUTIN": (1, "container"),
        "PUTINSIDE": (1, "container"),
    }
    for token in program_tokens(human_program):
        verb_match = re.match(r"\[([A-Za-z_]+)\]", token)
        if verb_match is None:
            continue
        requirement = verb_requirements.get(verb_match.group(1).replace("_", "").upper())
        if requirement is None:
            continue
        argument_index, capability = requirement
        arguments = re.findall(r"<([^>?]+)>", token)
        if argument_index >= len(arguments):
            continue
        requirements.setdefault(arguments[argument_index], set()).add(capability)
    return {key: frozenset(values) for key, values in requirements.items()}


def authored_prefix(human_program: str) -> list[str]:
    """The commands BEFORE the first answer slot. These are the observation the task is
    about (walk up, look at the cue) and are copied verbatim: an audit found generated
    programs silently dropping the authored ``[LookAt]``, which still scores 1.00 while
    the human never looks at the cue."""
    toks = program_tokens(human_program)
    for i, t in enumerate(toks):
        if "<?>" in t:
            return toks[:i]
    return toks


def opening_prefix(human_program: str) -> list[str]:
    """The authored OBSERVATION opening: everything before the first command that touches an answer.
    Copied byte-for-byte into every generated program (oracle and control alike) and re-asserted, so
    the audited failure -- a generated program that drops the authored ``[LookAt]`` and still scores
    1.00 while the human never looked at the cue -- cannot ship under either goal shape."""
    return _prefix_len_toks(program_tokens(human_program))


def _subject_of(tok: str, answers: dict[str, str]) -> str | None:
    """The answer slot a command belongs to: its first object argument that HAS an answer."""
    import re

    for name in re.findall(r"<([^>?]+)>", tok):
        if name in answers:
            return name
    return None


def split_answer(value: str) -> tuple[str, str | None]:
    """``'unit_a'`` -> (unit_a, None); ``'bd_0:shelfback'`` -> (bd_0, shelfback). The optional second
    field re-routes an ATTACH's PARENT CONNECTOR as well as its parent, which is what a *scoreable*
    wrong answer needs: the emitted relation carries no connector term, so only the parent is
    scoreable, but a wrong parent that does not expose the child's female cannot mate at all."""
    tgt, _, conn = str(value).partition(":")
    return tgt.strip(), (conn.strip() or None)


def fill_answer_slots(
    toks: list[str], answers: dict[str, str], key: dict[str, str] | None = None
) -> list[str]:
    """PER-ITEM answer filling (this is what retires the single shared ``_sole_answer`` slot).

    A ``<?>`` resolves from its own command's subject when the command names one, else from the most
    recent command that did. So ``[Grab] <g1>, [Open] <?>, [PutInside] <g1> <?>, [Close] <?>,
    [Grab] <g2>, [Open] <?>, ...`` routes each garment to ITS OWN hamper. A ``[Walk]`` is
    auto-inserted whenever a FILLED slot changes target.

    ``key`` (the reference answers) additionally makes an AUTHORED target controllable: about a
    third of the portfolio writes ``[PutInside] <obj_1>(1) <u_back>(1)`` instead of ``<?>``, and
    without this a wrong-answer control would silently drive the oracle program. Re-routing an
    authored target inserts NO walk -- command_grammar's auto-nav does that -- so the control stays
    a one-command diff from the oracle."""
    out: list[str] = []
    carry: str | None = None
    walked: str | None = None
    for t in toks:
        subj = _subject_of(t, answers)
        if subj is not None:
            carry = subj
        if "<?>" in t:
            if carry is None:
                raise SpecBuildError(
                    f"answer slot in {t!r} has no resolvable subject (answers={answers})"
                )
            tgt, _conn = split_answer(answers[carry])
            # ...unless the command being filled IS the walk. Nineteen rows author
            # ``[Walk] <u_prog_item_1>(1), [PutInside] <item_1>(1) <u_prog_item_1>(1)``; emitting an
            # auto-walk in front of the authored one doubles every approach and pushes the program
            # over its own published command count.
            if walked != tgt and not t.startswith("[Walk]"):
                out.append(f"[Walk] <{tgt}> (1)")
            walked = tgt
            out.append(t.replace("<?>", f"<{tgt}>"))
            continue
        out.append(_retarget(t, carry, answers, key))
    return out


def _retarget(
    tok: str, carry: str | None, answers: dict[str, str], key: dict[str, str] | None
) -> str:
    """Swap an AUTHORED target that names the current subject's reference answer for the answered
    one. Only the reference target is ever rewritten, so a command naming some third entity (a
    crate, a placard bank) is left alone."""
    if key is None or carry is None or carry not in key:
        return tok
    want, _c = split_answer(answers.get(carry, key[carry]))
    ref = str(key[carry])
    if want == ref or f"<{ref}>" not in tok:
        return tok
    return tok.replace(f"<{ref}>", f"<{want}>")


def insert_observation(toks: list[str], recipe: Recipe) -> list[str]:
    """Ensure every required printed cue is observed before the first manipulation.

    T1-CHEM-02's ``design.cue`` is "the GHS pictogram on each bottle at grasp range, plus the two
    cabinet placards", and its authored program opens on ``[Grab]``: as written the human never
    looks at either. That is the audited failure verbatim, so the builder ADDS the observation
    rather than scoring a blind run. Missing item looks are inserted as one sweep after the scripted
    opening; missing unit looks first walk to the labelled unit because ``[LookAt]`` deliberately
    does not auto-navigate. Adding observation can never make a wrong answer right, so this is a
    one-way safe transform."""
    if not recipe.observe:
        return toks
    authored = {_first_arg(token) for token in toks if token.startswith("[LookAt]")}
    missing = [key for key in recipe.observe if key not in authored]
    if not missing:
        return toks
    units = {unit.key for unit in recipe.units}
    # AFTER the authored opening, never before it. ``build_program`` re-asserts that opening as the
    # generated program's byte prefix, so a sweep spliced at index 0 fails the assertion on every row
    # whose opening is a [Walk] rather than a [Grab]. T1-CHEM-02's opening is empty, so it is
    # byte-identical either way -- this only widens the transform to rows that authored one.
    n_pre = len(_prefix_len_toks(toks))
    observation: list[str] = []
    for key in missing:
        if key in units:
            observation.append(f"[Walk] <{key}> (1)")
        observation.append(f"[LookAt] <{key}> (1)")
    out = list(toks[:n_pre]) + observation + list(toks[n_pre:])
    return out


def insert_dialogue_pauses(toks: list[str], spec: dict | None) -> list[str]:
    """Insert a visible speech beat before commands whose guidance must arrive first."""
    dialogue = dict((spec or {}).get("episode_dialogue") or {})
    pending = [
        dict(turn)
        for turn in dialogue.get("assistant_turns") or []
        if isinstance(turn, dict) and turn.get("pause_before")
    ]
    if not pending:
        return toks
    counts: dict[str, int] = {}
    out: list[str] = []
    for token in toks:
        counts[token] = counts.get(token, 0) + 1
        occurrence = counts[token]
        pause = any(
            str(turn.get("command") or "") == token
            and int(turn.get("occurrence") or occurrence) == occurrence
            for turn in pending
        )
        if pause:
            out.append("[Wait]")
        out.append(token)
    return out


def _prefix_len_toks(toks: list[str]) -> list[str]:
    """``opening_prefix``, on tokens that are already split."""
    for i, t in enumerate(toks):
        if "<?>" in t or t.split(" ", 1)[0] in ("[Grab]", "[Attach]"):
            return toks[:i]
    return list(toks)


def _first_arg(tok: str) -> str:
    import re

    m = re.search(r"<([^>?]+)>", tok)
    return m.group(1) if m else ""


def authored_look_keys(human_program: str) -> list[str]:
    """Every key the AUTHORED program looks at. This is the general form of the audited failure --
    a generated program that drops an authored ``[LookAt]`` and still scores 1.00 -- because it
    demands the compiled program keep ALL of them, not just a hand-listed cue."""
    import re

    out: list[str] = []
    for t in program_tokens(human_program):
        if not t.startswith("[LookAt]"):
            continue
        for name in re.findall(r"<([^>?]+)>", t):
            if name not in out:
                out.append(name)
    return out


def look_keys(
    human_program: str,
    recipe: Recipe,
    spec: dict | None = None,
    key: dict[str, str] | None = None,
    seed: int = 0,
) -> list[str]:
    """Every key the EXECUTED program must look at: the authored ones, else the inserted ones."""
    if spec is not None and is_prose(spec):
        pp = parse_prose(str(spec["human_program"]))
        return list(
            dict.fromkeys(
                prose_look_keys(pp, recipe, dict(key or reference_answers(spec)), seed)
            )
        )
    out: list[str] = []
    for token in insert_observation(program_tokens(human_program), recipe):
        key = _first_arg(token) if token.startswith("[LookAt]") else ""
        if key and key not in out:
            out.append(key)
    return out


def emit_attach(mate: Mate, answers: dict[str, str]) -> str:
    """One arity-4 ``[Attach] <child>(i) <childConnector>(i) <parent>(i) <parentConnector>(i)``.

    The child and its Male connector are geometry; the PARENT and its Female are the ANSWER, so a
    per-mate override re-routes exactly the term ``ATTACHED(child, parent)`` scores on."""
    parent, conn = split_answer(answers.get(mate.child, mate.parent))
    point = mate.parent_point if conn is None else f"{conn}F"
    idx = mate.pp_idx if conn is None else 0
    return (
        f"[Attach] <{mate.child}> (1) <{mate.child_point}> ({mate.cp_idx}) "
        f"<{parent}> (1) <{point}> ({idx})"
    )


def build_attach_program(
    toks: list[str], mates: list[Mate], answers: dict[str, str]
) -> list[str]:
    """Regenerate authored ``[Attach]`` tokens from metadata mates; copy everything else verbatim.

    Selected mates must retain answer-key order. A provenance-restore program may
    attach only the one initially-mated child it detached; the builder's canonical
    contract validator separately proves every other scored mate is preserved in
    ``world.initial_attachments``.
    """
    out: list[str] = []
    by_child = {mate.child: (index, mate) for index, mate in enumerate(mates)}
    seen: set[str] = set()
    last_index = -1
    for t in toks:
        if not t.startswith("[Attach]"):
            out.append(t)
            continue
        child = _first_arg(t)
        if child not in by_child:
            raise SpecBuildError(
                f"program attaches {child!r}, which is not one of the scored mates {sorted(by_child)}"
            )
        index, mate = by_child[child]
        if child in seen or index <= last_index:
            raise SpecBuildError(
                f"authored Attach order repeats or reverses scored mate {child!r}"
            )
        out.append(emit_attach(mate, answers))
        seen.add(child)
        last_index = index
    return out


def prose_observation(
    pp: ProseProgram, recipe: Recipe, key: dict[str, str], seed: int = 0
) -> list[str]:
    """The expanded OBSERVATION phase: the prose's own looks, in the prose's own order.

    One ``[Walk]`` is emitted before a per-item survey and nowhere else. ``command_grammar`` auto-navs
    every MANIPULATION but deliberately not ``[LookAt]`` ("gaze from where we stand"), while a
    ``[LookAt]`` on a graspable under ``look_zoom`` is a GRASP-RANGE inspect -- so a survey issued
    from the start pose would reach for a bench eight metres away and honest-fail. The walk restores
    exactly the auto-nav the row's WALK-FREE convention assumes; every other walk stays deleted."""
    out = [f"[LookAt] <{k}> (1)" for k in prose_look_keys(pp, recipe, key, seed)]
    if pp.n_look:
        first = staged_order(recipe, seed)[0].key
        at = len(out) - pp.n_look
        out.insert(
            at,
            f"[Walk] <{recipe.support_keys[0] if recipe.support_keys else first}> (1)",
        )
    return out


def expand_prose(
    spec: dict,
    recipe: Recipe,
    answers: dict[str, str],
    key: dict[str, str],
    seed: int = 0,
) -> list[str]:
    """A prose program -> real tokens. Observation phase first, then the placement queue."""
    pp = parse_prose(str(spec["human_program"]))
    assert_prose_arithmetic(pp, recipe, key, seed)
    out = [
        t
        for item, unit in pp.prologue
        for t in (f"[Grab] <{item}> (1)", f"[PutInside] <{item}> (1) <{unit}> (1)")
    ]
    out += prose_observation(pp, recipe, key, seed)
    queue = order_override(pp.loop_subjects(recipe, key, seed), key)
    ordered = bool(os.environ.get("BENCH_ORDER", "").strip())
    for subj, tgt in queue:
        # BENCH_ORDER owns the routing of its own queue (the two controls are mutually exclusive),
        # so a queued decoy really does go where the wrong ORDER sends it.
        dest = split_answer(tgt if ordered else answers.get(subj, tgt))[0]
        out += [f"[Grab] <{subj}> (1)", f"[PutInside] <{subj}> (1) <{dest}> (1)"]
    return out


def assert_prose_opening(
    toks, pp: ProseProgram, recipe: Recipe, key, seed: int = 0
) -> None:
    """The prose analogue of the verbatim authored opening: the observation phase the row authored
    must be the program's own prefix, entire and in order. Dropping a look stays impossible."""
    want = prose_observation(pp, recipe, key, seed)
    head = toks[2 * len(pp.prologue) :][: len(want)]
    if head != want:
        raise SpecBuildError(
            f"generated program dropped the authored observation {want!r}: {head!r}"
        )


def build_program(
    human_program: str,
    answers: dict[str, str],
    mates: list[Mate] | None = None,
    key: dict[str, str] | None = None,
    recipe: Recipe | None = None,
    spec: dict | None = None,
    seed: int = 0,
) -> str:
    """Authored opening verbatim + the answered tail. ``answers`` maps a SUBJECT (a graspable, or an
    assembly child) to the target the ANSWERER chose; a wrong answer changes only this dict, so every
    control drives the identical world."""
    if spec is not None and recipe is not None and is_prose(spec):
        return build_prose_program(spec, recipe, answers, dict(key or {}), seed)
    toks = program_tokens(human_program)
    if recipe is not None and not mates:
        toks = insert_observation(toks, recipe)
    toks = insert_dialogue_pauses(toks, spec)
    filled = (
        build_attach_program(toks, mates, answers)
        if mates
        else fill_answer_slots(toks, answers, key)
    )
    pre = insert_dialogue_pauses(opening_prefix(human_program), spec)
    # The opening is compared AFTER the same answer re-routing, not against the authored literal.
    # A row whose opening already contains a put to its reference target (``[PutInside] <obj_2>(1)
    # <unit_c>(1), [Grab] ...``) MUST see that target re-routed by the control -- asserting the raw
    # authored bytes there would forbid the only wrong answer the row can express. The prefix has no
    # ``<?>`` by construction, so no auto-walk is inserted and the two runs are index-aligned.
    want = fill_answer_slots(pre, answers, key) if (pre and not mates) else pre
    if want and filled[: len(want)] != want:
        raise SpecBuildError(
            f"generated program dropped the authored opening {want!r}: {filled[: len(want)]!r}"
        )
    return ", ".join(filled)


def build_prose_program(spec: dict, recipe: Recipe, answers, key, seed: int = 0) -> str:
    """Expand, then apply the row's own budget. The ORACLE must fit: a truncated oracle would score
    its own row 0, so the row's K (or its world) is wrong and the build has to say so."""
    pp = parse_prose(str(spec["human_program"]))
    toks = expand_prose(spec, recipe, answers, key, seed)
    assert_prose_opening(toks, pp, recipe, key, seed)
    k = prose_budget(spec, pp, len(expand_prose(spec, recipe, key, key, seed)))
    if k and answers == key and len(toks) > k:
        raise SpecBuildError(
            f"the ORACLE program is {len(toks)} commands over its own budget K={k}"
        )
    if not k and os.environ.get("BENCH_ORDER", "").strip():
        raise SpecBuildError(
            "BENCH_ORDER was set on a row with no budget: with a final-state scorer a pure "
            "re-ordering leaves the scored world identical, so it would be an unrendered control"
        )
    return ", ".join(apply_budget(toks, k))


def answer_override(default: dict[str, str]) -> dict[str, str]:
    """``BENCH_ANSWER='item_1=unit_a'`` (or ``'bd_4=bd_0:shelfback'``) re-routes the PROGRAM without
    touching the SCORED goals -- exactly the wrong-answer control the slice has to demonstrate."""
    raw = os.environ.get("BENCH_ANSWER", "").strip()
    if not raw:
        return dict(default)
    out = dict(default)
    for part in raw.split(","):
        if "=" not in part:
            raise SpecBuildError(
                f"BENCH_ANSWER entry {part!r} is not '<subject>=<target>[:<connector>]'"
            )
        k, v = part.split("=", 1)
        if k.strip() not in out:
            raise SpecBuildError(
                f"BENCH_ANSWER subject {k.strip()!r} is not an answer slot {sorted(out)}"
            )
        out[k.strip()] = v.strip()
    print(f"[bench] BENCH_ANSWER override active: {out} (scored goals UNCHANGED)")
    return out


def reference_answers(spec: dict) -> dict[str, str]:
    """subject -> the target its ANSWER KEY row sends it to, for every scoreable predicate."""
    return {
        str(g["subject"]): str(g["target"])
        for g in spec.get("answer_key") or []
        if g.get("pred", "inside") in ("inside", "attached")
    }


# =====================================================================================
# PROSE programs -- 22 rows (every T6) author their program as English, not as tokens
# =====================================================================================
# The prose is NOT free text: it is a loop macro over the fixed grammar, and every row is one of
# four shapes. It is parsed, never paraphrased -- and the parse is checked against the author's OWN
# published arithmetic (``= 1 + 4 + 8 = 13 grammar commands``), so a mis-read shape is a build error
# rather than a silently different task.
_PR_PROLOGUE = re.compile(r"PROLOGUE\s*\(assistant muted\):(.*?)\.\s+MAIN:", re.S)
_PR_PRO_PAIR = re.compile(
    r"\[Grab\]\s*(\w+)\s*\(1\)\s*,\s*\[PutInside\]\s*\1\s*\(1\)\s*(\w+)\s*\(1\)"
)
_PR_LOOK_N = re.compile(r"\[LookAt\]\s*item_i\s*\(1\)\s*x\s*(\d+)")
_PR_LOOK_NAMED = re.compile(r"\[LookAt\]\s*(?!item_i\b)(\w+)\s*\(1\)")
_PR_LOOP = re.compile(
    r"then per (item|remaining item|mandatory item) i:\s*"
    r"\[Grab\]\s*item_i\s*\(1\)\s*,\s*\[PutInside\]\s*item_i\s*\(1\)\s*(\w+)\s*\(1\)"
)
_PR_TOTAL = re.compile(r"=\s*(\d+)\s+grammar commands")
_PR_BUDGET = re.compile(r"K\s*=\s*(\d+)")


@dataclass(frozen=True)
class ProseProgram:
    """One parsed prose program. Every field is READ from the prose; none is inferred."""

    prologue: tuple[
        tuple[str, str], ...
    ]  # (item, unit) placements made with the assistant MUTED
    named_looks: tuple[
        str, ...
    ]  # authored [LookAt] on a named key ("unit_placards", "card", ...)
    n_look: int  # the authored ``[LookAt] item_i(1) xN``; 0 = no per-item survey
    loop: str  # "" | "all" (every staged item) | "key" (only the answer key's subjects)
    target: str  # "" = the answer slot ``u``; else the literal unit the prose names
    total: int  # the author's own published grammar-command total
    budget: int  # the author's own published K, 0 = the row declares no budget

    def loop_subjects(
        self, recipe: Recipe, key: dict[str, str], seed: int = 0
    ) -> list[str]:
        got = [i.key for i in staged_order(recipe, seed)]
        return [k for k in got if k in key] if self.loop == "key" else got


def is_prose(spec: dict) -> bool:
    return not bool((spec.get("cmds") or {}).get("program_tokenisable", True))


def parse_prose(human_program: str) -> ProseProgram:
    """English loop macro -> a ProseProgram. Refuses anything it does not recognise."""
    hp = " ".join(str(human_program).split())
    pro, body = _split_prologue(hp)
    body = _PR_LOOP.sub(" ", _PR_LOOK_N.sub(" ", body))
    named = tuple(dict.fromkeys(_PR_LOOK_NAMED.findall(body)))
    loop_m = _PR_LOOP.search(hp)
    look_m = _PR_LOOK_N.search(hp)
    total_m = _PR_TOTAL.search(hp)
    if loop_m is None or total_m is None:
        raise UnsupportedSpec(
            f"prose program has no recognisable loop/total: {human_program!r}"
        )
    kind = {"item": "all", "remaining item": "key", "mandatory item": "key"}[
        loop_m.group(1)
    ]
    tgt = "" if loop_m.group(2) == "u" else loop_m.group(2)
    bud = _PR_BUDGET.search(hp)
    return ProseProgram(
        prologue=pro,
        named_looks=named,
        n_look=int(look_m.group(1)) if look_m else 0,
        loop=kind,
        target=tgt,
        total=int(total_m.group(1)),
        budget=int(bud.group(1)) if bud else 0,
    )


def _split_prologue(hp: str) -> tuple[tuple[tuple[str, str], ...], str]:
    """Peel a ``PROLOGUE (assistant muted): ... MAIN:`` phase off the front, if the row has one."""
    m = _PR_PROLOGUE.search(hp)
    if m is None:
        return (), hp
    pairs = tuple((a, b) for a, b in _PR_PRO_PAIR.findall(m.group(1)))
    if not pairs:
        raise UnsupportedSpec(
            f"prologue phase parses to no [Grab]/[PutInside] pair: {m.group(1)!r}"
        )
    return pairs, hp[m.end() :]


def prose_arithmetic(
    pp: ProseProgram, recipe: Recipe, key: dict[str, str], seed: int = 0
) -> int:
    """The author's OWN command count, recomputed from the parse. Equality with ``pp.total`` is what
    proves the parse read the shape the author wrote -- a survey loop mistaken for a placement loop,
    or a missed named look, changes this number and fails the build."""
    return (
        2 * len(pp.prologue)
        + len(pp.named_looks)
        + pp.n_look
        + 2 * len(pp.loop_subjects(recipe, key, seed))
    )


def assert_prose_arithmetic(
    pp: ProseProgram, recipe: Recipe, key: dict[str, str], seed: int = 0
) -> None:
    got = prose_arithmetic(pp, recipe, key, seed)
    if got != pp.total:
        raise SpecBuildError(
            f"prose parse gives {got} grammar commands but the row publishes {pp.total} "
            f"(prologue={len(pp.prologue)} named={list(pp.named_looks)} survey={pp.n_look} "
            f"loop={len(pp.loop_subjects(recipe, key, seed))}) -- the parse read a different shape"
        )
    if pp.n_look and pp.n_look != len(recipe.staged()):
        raise SpecBuildError(
            f"the prose surveys {pp.n_look} items but the recipe stages {len(recipe.staged())} "
            f"({[i.key for i in recipe.staged()]}) -- the authored N pins the world's item count"
        )


def prose_look_keys(
    pp: ProseProgram, recipe: Recipe, key: dict[str, str], seed: int = 0
) -> list[str]:
    """The concrete entities the prose's observation phase looks at, in order.

    ``unit_placards`` (plural) is a SWEEP over every unit; ``unit_placard`` (singular) is the row's
    one scored destination. Anything else is a named prop the recipe has to have registered -- a
    prose row that looks at a ``card`` the recipe never builds is refused, never quietly dropped."""
    out: list[str] = []
    props = set(recipe.support_keys) | ({recipe.card_key} if recipe.card_key else set())
    for name in pp.named_looks:
        if name == "unit_placards":
            out += [u.key for u in recipe.units]
        elif name == "unit_placard":
            out += _sole_scored_unit(key)
        elif name in props:
            out.append(name)
        else:
            raise UnsupportedSpec(
                f"the prose looks at {name!r}, which is neither 'unit_placard(s)' nor one of the "
                f"recipe's registered props {sorted(props)}; the recipe must build it"
            )
    if pp.n_look:
        out += [i.key for i in staged_order(recipe, seed)]
    return out


def _sole_scored_unit(key: dict[str, str]) -> list[str]:
    got = sorted(set(key.values()))
    if len(got) != 1:
        raise SpecBuildError(f"'unit_placard' (singular) but the key names {got}")
    return got


def order_override(default: list[str], key: dict[str, str]) -> list[tuple[str, str]]:
    """``BENCH_ORDER='item_1=unit_a,item_2=unit_a,...'`` -- the ORDERING analogue of ``BENCH_ANSWER``.

    ``BENCH_ANSWER`` can only re-route a subject the oracle already handles, so it cannot express the
    defect a BUDGET row is about: working the queue in the wrong ORDER (or over the wrong SELECTION)
    so the budget runs out before the mandatory items are reached. This replaces the queue itself.
    The scored goals are untouched; ``=<target>`` is optional and defaults to the answer key's."""
    raw = os.environ.get("BENCH_ORDER", "").strip()
    if not raw:
        return [(s, key[s]) for s in default]
    if os.environ.get("BENCH_ANSWER", "").strip():
        raise SpecBuildError(
            "BENCH_ANSWER and BENCH_ORDER are two different defect classes (destination vs order); "
            "setting both would render one control and report the other"
        )
    out: list[tuple[str, str]] = []
    for part in raw.split(","):
        subj, _, tgt = part.strip().partition("=")
        subj, tgt = subj.strip(), tgt.strip()
        if not tgt and subj not in key:
            raise SpecBuildError(
                f"BENCH_ORDER entry {subj!r} has no answer-key target; write '{subj}=<unit>'"
            )
        out.append((subj, tgt or key[subj]))
    print(f"[bench] BENCH_ORDER override active: {out} (scored goals UNCHANGED)")
    return out


def prose_budget(spec: dict, pp: ProseProgram, oracle_n: int) -> int:
    """The budget in EMITTED commands, derived from the row's own two published numbers.

    The author counts a placard SWEEP as one command and lets auto-nav supply the walks; the engine
    needs one ``[LookAt]`` per placard and one ``[Walk]`` to the bench before a grasp-range inspect.
    So the absolute K does not transfer -- the SLACK does, and the slack is what the row is about
    ("there is not enough budget left to correct a mistake"). ``K_emitted = oracle + (K - oracle)``."""
    if not pp.budget:
        return 0
    declared = int((spec.get("cmds") or {}).get("budget_k") or 0)
    slack = int(pp.budget) - int(pp.total)
    if slack < 0:
        raise SpecBuildError(
            f"published K={pp.budget} is under the row's own oracle {pp.total}"
        )
    if declared and declared != pp.budget:
        print(
            f"[bench] BUDGET NOTE: cmds.budget_k={declared} but the prose states K={pp.budget}; "
            f"the prose is used and the SLACK ({slack}) is what transfers"
        )
    k = int(oracle_n) + slack
    print(f"[bench] budget: oracle={oracle_n} emitted commands, K={k} (slack {slack})")
    return k


def apply_budget(toks: list[str], k: int) -> list[str]:
    """Truncate a queued command stream at the budget, then drop a stranded trailing ``[Grab]``.

    The ORACLE is never truncated (``assert_budget_fits``); only a control whose ORDER or SELECTION
    spends the budget elsewhere is, and then the drop is printed command by command so the failure
    is legible in the log rather than inferred from the score."""
    if k <= 0 or len(toks) <= k:
        return toks
    cut = list(toks[:k])
    while cut and cut[-1].startswith("[Grab]"):
        cut.pop()
    print(
        f"[bench] BUDGET TRUNCATION at K={k}: dropped {len(toks) - len(cut)} command(s) {toks[len(cut) :]}"
    )
    return cut


# =====================================================================================
# stage geometry helpers (all metres MEASURED, none authored)
# =====================================================================================
def _wbbox(stage, pp: str):
    import fpv_scene

    rng = fpv_scene.compute_world_bbox(stage.GetPrimAtPath(pp))
    mn, mx = rng.GetMin(), rng.GetMax()
    return (
        (float(mn[0]), float(mn[1]), float(mn[2])),
        (float(mx[0]), float(mx[1]), float(mx[2])),
    )


def _ctr(stage, pp: str) -> tuple[float, float]:
    (ax, ay, _az), (bx, by, _bz) = _wbbox(stage, pp)
    return (0.5 * (ax + bx), 0.5 * (ay + by))


def _clearance(boxes, x: float, y: float, skip: str | None = None) -> float:
    best = 1e9
    for nm, mn, mx in boxes:
        if skip is not None and nm == skip:
            continue
        dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
        dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
        best = min(best, math.hypot(dx, dy))
    return best


def _approach_clearance(boxes, x: float, y: float, yaw: float, foot_half: float) -> float:
    """Minimum obstacle clearance along the whole door-to-human approach corridor."""
    angle = math.radians(yaw)
    front = (-math.sin(angle), math.cos(angle))
    return min(
        _clearance(boxes, x + front[0] * distance, y + front[1] * distance)
        for distance in (
            foot_half + 0.12,
            foot_half + 0.30,
            foot_half + 0.50,
            foot_half + 0.72,
            foot_half + DOOR_FRONT_CLEAR_M,
        )
    )


def _best_yaw(boxes, x: float, y: float, foot_half: float = 0.0) -> float:
    """The cardinal direction with the clearest complete approach corridor."""
    best_a, best_c = 0.0, -1.0
    for a_deg in (0.0, 90.0, 180.0, 270.0):
        if foot_half > 0.0:
            c = _approach_clearance(boxes, x, y, a_deg, foot_half)
        else:
            a = math.radians(a_deg)
            c = _clearance(boxes, x - math.sin(a) * 1.0, y + math.cos(a) * 1.0)
        if c > best_c:
            best_a, best_c = a_deg, c
    return best_a


# Structural categories are assigned to a room but are its SHELL, not its contents: including
# them in the room hull pushes the hull out to the wall centre-lines, and an unbounded slot
# search then happily sites a unit -- or the human's start -- inside a wall.
_STRUCTURAL: frozenset[str] = frozenset(
    {"walls", "floors", "ceilings", "door", "window", "openable_window", "roof", "lawn"}
)


def _room_objects(scene_name: str, room: str, *, structural: bool) -> list[dict]:
    import behavior_import as bi

    return [
        obj
        for obj in bi.scene_objects(scene_name)
        if room in (obj.get("in_rooms") or [])
        and (structural or obj.get("cat") not in _STRUCTURAL)
    ]


def _room_content_hull(
    scene_name: str, room: str
) -> tuple[float, float, float, float] | None:
    points = [obj["pos"] for obj in _room_objects(scene_name, room, structural=False)]
    if not points:
        return None
    xs, ys = (
        [float(point[0]) for point in points],
        [float(point[1]) for point in points],
    )
    box = (min(xs), min(ys), max(xs), max(ys))
    if (
        box[2] - box[0] < MIN_ROOM_HULL_SPAN_M
        or box[3] - box[1] < MIN_ROOM_HULL_SPAN_M
    ):
        return None
    return box


def room_floor_regions(
    scene_name: str, room: str
) -> list[tuple[float, float, float, float, float]]:
    """Measured oriented XY floor rectangles as ``(cx, cy, hx, hy, yaw)``."""
    import behavior_import as bi

    regions = []
    for obj in _room_objects(scene_name, room, structural=True):
        if str(obj.get("cat", "")).rstrip("s") != "floor":
            continue
        try:
            bbox = bi.read_spec(f"{obj['cat']}/{obj['model']}").get("bbox_size")
        except (KeyError, OSError, ValueError):
            continue
        if not bbox:
            continue
        scale = obj.get("scale") or [1.0, 1.0, 1.0]
        if isinstance(scale, (int, float)):
            scale = [scale, scale, scale]
        width, depth = (
            float(bbox[0]) * float(scale[0]),
            float(bbox[1]) * float(scale[1]),
        )
        qx, qy, qz, qw = (float(value) for value in obj["ori"])
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        x, y = float(obj["pos"][0]), float(obj["pos"][1])
        regions.append((x, y, 0.5 * width, 0.5 * depth, yaw))
    return sorted(regions, key=lambda row: (-4.0 * row[2] * row[3], row))


def room_floor_boxes(
    scene_name: str, room: str
) -> list[tuple[float, float, float, float]]:
    """Axis-aligned envelopes of the separate measured room floor slabs."""
    boxes = []
    for x, y, hx, hy, yaw in room_floor_regions(scene_name, room):
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        half_x = hx * c + hy * s
        half_y = hx * s + hy * c
        boxes.append((x - half_x, y - half_y, x + half_x, y + half_y))
    return sorted(
        boxes,
        key=lambda box: (-(box[2] - box[0]) * (box[3] - box[1]), box),
    )


def _floor_support_mask(xs, ys, regions):
    import numpy as np

    supported = np.zeros(xs.shape, dtype=bool)
    for cx, cy, hx, hy, yaw in regions:
        dx, dy = xs - cx, ys - cy
        c, s = math.cos(yaw), math.sin(yaw)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        supported |= (np.abs(local_x) <= hx) & (np.abs(local_y) <= hy)
    return supported


def floor_support_index() -> dict:
    """Load the pre-rasterized counterpart of Isaac's real floor-mesh mask."""
    import json

    global _FLOOR_SUPPORT_DATA
    if _FLOOR_SUPPORT_DATA is not None:
        return _FLOOR_SUPPORT_DATA
    try:
        with open(FLOOR_SUPPORT_INDEX, encoding="utf-8") as source:
            data = json.load(source)
    except FileNotFoundError:
        data = {}
    except (OSError, json.JSONDecodeError) as error:
        raise SpecBuildError(
            f"cannot read measured floor support index {FLOOR_SUPPORT_INDEX}: {error}"
        ) from error
    if data and data.get("schema") != "floor-support/1.0":
        raise SpecBuildError(
            f"unsupported measured floor support schema {data.get('schema')!r}"
        )
    _FLOOR_SUPPORT_DATA = data
    return data


def _measured_floor_grid(scene_name: str, room: str):
    """Decode one room's RLE floor mask, or None when the bake predates this scene."""
    key = (scene_name, room)
    if key in _FLOOR_SUPPORT_GRIDS:
        return _FLOOR_SUPPORT_GRIDS[key]
    record = (
        (floor_support_index().get("scenes") or {})
        .get(scene_name, {})
        .get("rooms", {})
        .get(room)
    )
    if not record:
        _FLOOR_SUPPORT_GRIDS[key] = None
        return None
    import numpy as np
    from scipy import ndimage

    nx, ny = int(record["nx"]), int(record["ny"])
    support = np.zeros((ny, nx), dtype=bool)
    for j, i0, i1 in record.get("runs") or []:
        support[int(j), int(i0) : int(i1) + 1] = True
    clearance = np.maximum(ndimage.distance_transform_edt(support) - 0.5, 0.0)
    clearance *= float(record["cell"])
    grid = (
        float(record["x0"]),
        float(record["y0"]),
        float(record["cell"]),
        support,
        clearance,
    )
    _FLOOR_SUPPORT_GRIDS[key] = grid
    return grid


def _floor_support_samples(scene_name: str, room: str, xs, ys):
    """Return (supported, edge clearance) for points in a room's measured floor."""
    import numpy as np

    grid = _measured_floor_grid(scene_name, room)
    if grid is None:
        regions = room_floor_regions(scene_name, room)
        supported = _floor_support_mask(xs, ys, regions)
        clearance = np.zeros(xs.shape, dtype=np.float64)
        for cx, cy, hx, hy, yaw in regions:
            dx, dy = xs - cx, ys - cy
            c, s = math.cos(yaw), math.sin(yaw)
            local_x = c * dx + s * dy
            local_y = -s * dx + c * dy
            inset = np.minimum(hx - np.abs(local_x), hy - np.abs(local_y))
            clearance = np.maximum(clearance, np.where(inset >= 0.0, inset, 0.0))
        return supported, clearance

    x0, y0, cell, support, clearance = grid
    ii = np.floor((xs - x0) / cell).astype(np.int64)
    jj = np.floor((ys - y0) / cell).astype(np.int64)
    valid = (ii >= 0) & (jj >= 0) & (ii < support.shape[1]) & (jj < support.shape[0])
    sampled_support = np.zeros(xs.shape, dtype=bool)
    sampled_clearance = np.zeros(xs.shape, dtype=np.float64)
    sampled_support[valid] = support[jj[valid], ii[valid]]
    sampled_clearance[valid] = clearance[jj[valid], ii[valid]]
    return sampled_support, sampled_clearance


def floor_supported_at(scene_name: str, room: str, x: float, y: float) -> bool:
    """Whether Isaac's baked floor mesh supports this room-local world point."""
    import numpy as np

    supported, _clearance = _floor_support_samples(
        scene_name,
        room,
        np.asarray([float(x)]),
        np.asarray([float(y)]),
    )
    return bool(supported[0])


def _room_floor_box(scene_name: str, room: str) -> tuple[float, float, float, float]:
    boxes = room_floor_boxes(scene_name, room)
    if not boxes:
        raise SpecBuildError(
            f"scene {scene_name!r} has neither a usable furniture hull nor measured floor "
            f"geometry in room {room!r}"
        )
    return boxes[0]


def room_anchor(scene_name: str, room: str) -> tuple[float, float]:
    """Stable layout anchor from room contents, or the largest real floor slab when empty."""
    hull = _room_content_hull(scene_name, room)
    if hull is None:
        x0, y0, x1, y1 = _room_floor_box(scene_name, room)
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    points = [obj["pos"] for obj in _room_objects(scene_name, room, structural=False)]
    return (
        float(sum(float(point[0]) for point in points) / len(points)),
        float(sum(float(point[1]) for point in points) / len(points)),
    )


def room_prims(scene_name: str, room: str, structural: bool = True) -> set[str]:
    """Instance names the scene file assigns to ``room`` (used to keep siting in-room)."""
    import json

    p = os.path.join(SCENE_JSON_ROOT, scene_name, "json", f"{scene_name}_best.json")
    with open(p) as f:
        d = json.load(f)
    init = d["objects_info"]["init_info"]
    return {
        n
        for n, info in init.items()
        if isinstance(info, dict)
        and room in (info.get("args", {}).get("in_rooms") or [])
        and (structural or info.get("args", {}).get("category") not in _STRUCTURAL)
    }


def room_box(scene_name: str, room: str) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) hull of the room's own furniture. Everything the episode places has to
    land inside it -- the raw scene-wide clearance grid happily offers the most 'open' cells in
    the void outside the building, which is where an unbounded slot search walks off to. An empty
    or one-object room falls back to its largest measured floor slab instead of inventing a hull
    across disconnected slabs."""
    return _room_content_hull(scene_name, room) or _room_floor_box(scene_name, room)


# =====================================================================================
# the build world (a small bag so each stage stays a short, testable function)
# =====================================================================================
@dataclass
class World:
    stage: object
    app: object
    spec: dict
    recipe: Recipe
    seed: int
    scene: object = None
    placed: dict = field(default_factory=dict)
    anchor: tuple[float, float] = (0.0, 0.0)
    zones: dict[str, str] = field(default_factory=dict)
    cue_class: dict[str, str] = field(default_factory=dict)
    used_slots: list = field(default_factory=list)
    room_slots: dict[str, list] = field(default_factory=dict)
    front_dir: dict = field(default_factory=dict)
    reg_pp: dict = field(default_factory=dict)
    decals: dict = field(default_factory=dict)
    boxes_of: dict = field(default_factory=dict)  # scored container -> measured AABB
    outer_of: dict = field(
        default_factory=dict
    )  # scored container -> measured OUTER world AABB
    numbers: dict = field(
        default_factory=dict
    )  # cue_number channel -> {key: drawn int}
    preload: dict = field(
        default_factory=dict
    )  # unit key -> how many fillers were seated
    preload_keys: dict = field(
        default_factory=dict
    )  # unit key -> the filler entity keys
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # room hull
    walls: list = field(
        default_factory=list
    )  # measured wall AABBs (NOT in solid_boxes())
    zspan: dict = field(
        default_factory=dict
    )  # body -> measured (bottom_z, top_z); see siting_boxes
    light_rig: dict = field(
        default_factory=dict
    )  # bench_lighting's manifest row for this scene
    # --- set-level siting -------------------------------------------------------------------
    room: str = ""  # the room the SOLVER chose (seeded from recipe.room, which is only a preference)
    layout: list = field(default_factory=list)  # the committed [UnitPlan], one per unit
    supports: list = field(
        default_factory=list
    )  # the committed [(prim_path, approach_yaw)]
    support_options: list = field(
        default_factory=list
    )  # ranked same-room alternatives, requested category first, parallel to support_keys
    last_support_options: list = field(default_factory=list, repr=False)
    support_contact_xy: dict[int, tuple[float, float]] = field(default_factory=dict)
    support_stand_xy: dict[int, tuple[float, float]] = field(default_factory=dict)
    last_support_sites: dict[
        tuple[str, float], tuple[tuple[float, float], tuple[float, float]]
    ] = field(default_factory=dict, repr=False)
    support_stands: list = field(
        default_factory=list
    )  # where the human stands at each support
    room_support_stands: dict[str, list] = field(default_factory=dict)
    nav_strict: bool = (
        True  # search phase: demand ONE nav component before relaxing to a WARN
    )
    stage_band: bool = (
        True  # search phase: demand a WORK-SURFACE staging pass; see stageable_top
    )
    body_bump: dict = field(
        default_factory=dict
    )  # (cat, model) -> ladder rungs the probe demanded
    frozen_body_ladders: dict = field(
        default_factory=dict
    )  # (cat, model) -> failure-aware same-category repair order
    scene_unit_fallbacks: dict = field(
        default_factory=dict
    )  # imported unit key -> same-category scene-native candidate
    scene_unit_rejections: set = field(
        default_factory=set
    )  # scene-native paths already rejected by registration/acceptance
    binding_repairs: list[dict[str, object]] = field(default_factory=list)
    binding_repair_seconds: float = 0.0
    item_bump: dict = field(
        default_factory=dict
    )  # the SAME ledger for the GRASP probe's graspables
    support_skip: set = field(
        default_factory=set
    )  # staging surfaces the grasp probe has rejected
    support_import_attempts: dict = field(
        default_factory=dict
    )  # (room, category, fixture index) -> models already tried
    imported_support_paths: dict = field(
        default_factory=dict
    )  # imported support path -> synthetic key in placed
    # --- assembly rows only ---------------------------------------------------------------
    mates: list = field(
        default_factory=list
    )  # ordered, resolved Mate list (the scored edges)
    part_of: dict = field(default_factory=dict)  # abstract key -> PartRecipe
    root_key: str = ""  # the pre-placed assembly hub (derived, never authored)
    root_keys: tuple[str, ...] = ()  # one root per independent assembly component
    letters: dict = field(
        default_factory=dict
    )  # part key -> this seed's stamped letter
    manual_xy: tuple[float, float] = (0.0, 0.0)


def wall_dist(world: World, x: float, y: float) -> float:
    """Distance to the nearest WALL. ``scene.solid_boxes()`` carries furniture only, so without
    this term the 'most open' cell in the grid is happily the inside of a wall or a doorway."""
    if not world.walls:
        return 1e9
    best = 1e9
    for mn, mx in world.walls:
        dx = max(float(mn[0]) - x, 0.0, x - float(mx[0]))
        dy = max(float(mn[1]) - y, 0.0, y - float(mx[1]))
        best = min(best, math.hypot(dx, dy))
    return best


# A floor-standing unit collides with what is IN ITS OWN Z BAND. ``solid_boxes()`` reports every
# body as one floor->top xy pillar, which turns a ground slab and an overhead body into solid
# floor. Measured on ``house_double_floor_lower``: ``paver_sekqqq_0`` is a 27 x 15 m patio whose
# top is 0.17 m up and ``roof_pgwmer_0`` is a 14 x 11 m roof 8 m up, and between them every one of
# the garden's 17112 cells AND all 416 kitchen cells read clearance 0.00 -- so no unit could be
# sited anywhere in that scene, at any body rung, in any room. That is not a room being small.
SITE_STEP_M: float = (
    0.20  # a top at or below this is ground you STAND ON, not an obstacle
)
SITE_SPAN_M: float = (
    2.00  # a bottom at or above this is overhead of a floor-standing unit
)


def body_zspans(world: World) -> dict[str, tuple[float, float]]:
    """Measured (bottom_z, top_z) per placed body -- the z range ``solid_boxes()`` flattens away.

    One BBoxCache, and only the bodies not measured yet -- ``site_units`` re-asks after every unit
    it places, and re-measuring the whole scene each time would cost thousands of bounds. A body
    that cannot be measured is simply absent, and ``siting_boxes`` then keeps it: a proxy that
    fails to measure must not be able to delete an obstacle."""
    todo = [nm for nm in world.placed if nm not in world.zspan]
    if not todo or world.stage is None:
        return world.zspan
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    for nm in todo:
        prim = world.stage.GetPrimAtPath(world.placed[nm][0])
        if not (prim and prim.IsValid()):
            continue
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = rng.GetMin(), rng.GetMax()
        if float(mx[2]) >= float(mn[2]):
            world.zspan[nm] = (float(mn[2]), float(mx[2]))
    return world.zspan


def in_site_band(fz: float, lo: float, hi: float) -> bool:
    """Does a body spanning ``lo..hi`` obstruct a floor-standing unit on a floor at ``fz``?"""
    return hi > fz + SITE_STEP_M and lo < fz + SITE_SPAN_M


def siting_boxes(world: World) -> list:
    """``solid_boxes()`` restricted to the bodies that actually occupy a unit's own z band.

    Strictly a RELAXATION -- it only ever removes obstacles -- so it cannot unsite a world that
    already sites. What it removes is the ground (floor slabs, lawn, pavers, driveways, a pool)
    and the sky (roofs, pergola beams), neither of which a body standing on the floor can hit."""
    fz = float(getattr(world.scene, "floor_z", 0.0))
    zs = body_zspans(world)
    out = []
    for nm, mn, mx in world.scene.solid_boxes():
        lo, hi = zs.get(nm, (float(mn[2]), float(mx[2])))
        if not in_site_band(fz, lo, hi):
            continue
        out.append((nm, mn, mx))
    return out


def _wall_boxes(world: World) -> list:
    """Measured AABBs of the scene's wall prims (``scene.wall_prims``), which the entity-derived
    ``solid_boxes()`` never contains."""
    out = []
    for wp in getattr(world.scene, "wall_prims", None) or []:
        try:
            out.append(_wbbox(world.stage, wp))
        except (RuntimeError, ValueError, AttributeError):
            continue
    print(f"[bench] wall prims measured: {len(out)}")
    return out


def _grid(world: World):
    """Clearance grid over the ROOM's own hull, rebuilt after every placement so a newly sited
    unit is an obstacle to the next one. Clamped to the room because the scene-wide grid's most
    'open' cells are the empty void outside the building."""
    import numpy as np

    boxes = siting_boxes(world)
    x0, y0, x1, y1 = world.box
    step = 0.25
    xs = np.arange(x0, x1 + 1e-6, step)
    ys = np.arange(y0, y1 + 1e-6, step)
    floor_support = None
    floor_prims = list(getattr(world.scene, "floor_prims", None) or [])
    if floor_prims:
        import nav_grid

        floor_support = nav_grid._raster_floor_support(
            world.stage,
            floor_prims,
            float(xs[0] - 0.5 * step),
            float(ys[0] - 0.5 * step),
            step,
            len(xs),
            len(ys),
            float(world.scene.floor_z),
        )
        if not floor_support.any():
            raise SpecBuildError(
                f"{world.recipe.task_id}: room hull has no modelled floor support"
            )
    elif world.recipe.frozen_bindings:
        if not room_floor_regions(world.recipe.scene, world.room):
            raise SpecBuildError(
                f"{world.recipe.task_id}: room has no measured floor support"
            )
        mx_, my_ = np.meshgrid(xs, ys, indexing="ij")
        floor_support, _floor_clearance = _floor_support_samples(
            world.recipe.scene,
            world.room,
            mx_.ravel(),
            my_.ravel(),
        )
        floor_support = floor_support.reshape(len(xs), len(ys)).T
        if not floor_support.any():
            raise SpecBuildError(
                f"{world.recipe.task_id}: room hull has no measured floor support"
            )
    cells = [
        (float(x), float(y), _clearance(boxes, float(x), float(y)))
        for i, x in enumerate(xs)
        for j, y in enumerate(ys)
        if floor_support is None or bool(floor_support[j, i])
    ]
    return cells, boxes


# Free floor demanded AROUND a sited unit, best first. The ladder relaxes the MARGIN, never the
# unit's own half-extent: at every rung the cell still fits the body, it just has less elbow
# room. A real diner kitchen is wall-to-wall furniture in AABB terms, so a ladder that scaled
# the whole requirement ran out of floor before the fourth cold unit was placed.
_SLOT_MARGINS: tuple[tuple[float, float], ...] = (
    (0.55, 2.5),
    (0.42, 3.5),
    (0.30, 5.0),
    (0.20, 7.0),
    (0.12, 12.0),
    (0.06, 12.0),
)
# A pushed/pulled unit needs room for the human on whichever side the route selects.
# The hand contact is 0.38 m beyond the body's footprint and the standing body itself
# needs roughly another 0.30 m, so ordinary wall-hugging container slots are invalid.
_MOVABLE_UNIT_CLEAR_M: float = 0.75


def _find_slot(world: World, cells, fh: float, near=None, dcap_max: float = 8.0):
    """Nearest cell to ``near`` (default: the room anchor) that fits a unit of half-extent
    ``fh``, is outside every reserved keep-out and clear of the walls. Chaining ``near`` to the
    previously sited unit is what makes four separately sited units come out as ONE cold line
    the human can work along, rather than scattered across four rooms."""
    ax, ay = near if near is not None else world.anchor
    rej: dict = {}
    for margin, dcap0 in _SLOT_MARGINS:
        cf = fh + margin
        dcap = min(dcap0, dcap_max)
        best, bestscore = None, 1e18
        rej = {"clear": 0, "far": 0, "reserved": 0, "wall": 0, "ok": 0}
        for x, y, c in cells:
            if c < cf:
                rej["clear"] += 1
                continue
            if math.hypot(x - ax, y - ay) > dcap:
                rej["far"] += 1
                continue
            if any(
                math.hypot(x - ux, y - uy) < ur for (ux, uy, ur) in world.used_slots
            ):
                rej["reserved"] += 1
                continue
            wd = wall_dist(world, x, y)
            if wd < fh + 0.06:  # the unit's own body would clip into the wall
                rej["wall"] += 1
                continue
            rej["ok"] += 1
            # tie-break toward hugging a wall: a cold line stands against one, not mid-aisle
            score = math.hypot(x - ax, y - ay) + 1.3 * max(0.0, wd - (fh + 0.18))
            if score < bestscore:
                bestscore, best = score, (x, y)
        if best is not None:
            return best, cf
    print(f"[bench][slot] exhausted (fh={fh:.2f} cells={len(cells)}): last rung {rej}")
    return None, 0.0


def _drop_to_floor(world: World, prim, pp: str, x: float, y: float, quat) -> None:
    import fpv_scene

    fz = float(world.scene.floor_z)
    fpv_scene.set_world_pose(prim, (float(x), float(y), fz), quat, (1.0, 1.0, 1.0))
    (_bx, _by, base_z), _mx = _wbbox(world.stage, pp)
    if abs(float(base_z) - fz) > 0.02:
        fpv_scene.set_world_pose(
            prim, (float(x), float(y), fz - (float(base_z) - fz)), quat, (1.0, 1.0, 1.0)
        )


def _door_quat(
    world: World,
    pp: str,
    cat: str,
    model: str,
    cell,
    boxes,
    desired_yaw: float | None = None,
):
    """Rotate an imported swing unit so its baked door faces the openest side -- the
    leaf-derived stand then lands on open floor instead of against a wall."""
    import behavior_import as bi

    of = bi._closing_leaf_fields(world.scene, pp, {"cat": cat, "model": model})
    if not of or not of.get("leaf_bbox"):
        return (1.0, 0.0, 0.0, 0.0)
    (mn, mx) = _wbbox(world.stage, pp)
    ucx, ucy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
    d0 = _door_normal(of, ucx, ucy)
    a = math.radians(
        float(desired_yaw)
        if desired_yaw is not None
        else _best_yaw(boxes, cell[0], cell[1])
    )
    d1 = (-math.sin(a), math.cos(a))
    off = math.atan2(d0[0] * d1[1] - d0[1] * d1[0], d0[0] * d1[0] + d0[1] * d1[1])
    return (math.cos(off / 2.0), 0.0, 0.0, math.sin(off / 2.0))


# =====================================================================================
# stage 2 -- adopt/import the destination containers
# =====================================================================================
def _door_normal(leaf_fields: dict, tcx: float, tcy: float) -> tuple[float, float]:
    """Outward normal of the door side, from the ACTUAL swinging leaf: unit centre -> leaf
    centre. This is where the placard hangs and where the human stands to reach the interior."""
    boxes = tuple(leaf_fields.get("leaf_bboxes") or (leaf_fields["leaf_bbox"],))
    lmn = tuple(min(box[0][axis] for box in boxes) for axis in range(3))
    lmx = tuple(max(box[1][axis] for box in boxes) for axis in range(3))
    fdx, fdy = 0.5 * (lmn[0] + lmx[0]) - tcx, 0.5 * (lmn[1] + lmx[1]) - tcy
    fn = math.hypot(fdx, fdy) or 1.0
    return (fdx / fn, fdy / fn)


def _primary_leaf_handle_xy(
    leaf_fields: dict, tcx: float, tcy: float
) -> tuple[float, float]:
    """Return the primary leaf's free edge, the long-axis edge nearest the unit centre."""
    lmn, lmx = leaf_fields["leaf_bbox"]
    cx, cy = 0.5 * (lmn[0] + lmx[0]), 0.5 * (lmn[1] + lmx[1])
    if float(lmx[0]) - float(lmn[0]) >= float(lmx[1]) - float(lmn[1]):
        edges = ((float(lmn[0]), cy), (float(lmx[0]), cy))
    else:
        edges = ((cx, float(lmn[1])), (cx, float(lmx[1])))
    return min(edges, key=lambda point: math.hypot(point[0] - tcx, point[1] - tcy))


def _flip_down_geometry(
    leaf_fields: Mapping[str, object], front: tuple[float, float]
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float],
    float,
]:
    """World hinge, axis, and closed handle for one vertical front flap."""
    hinge = tuple(float(value) for value in leaf_fields["hinge_xyz"])
    leaf_minimum, leaf_maximum = leaf_fields["leaf_bbox"]
    handle_xy = (
        0.5 * (float(leaf_minimum[0]) + float(leaf_maximum[0])),
        0.5 * (float(leaf_minimum[1]) + float(leaf_maximum[1])),
    )
    axis = (-float(front[1]), float(front[0]), 0.0)
    return hinge, axis, handle_xy, float(leaf_maximum[2])


def _interior_clamp(mnz: float, mxz: float):
    """Clamp a desired height into the unit's REAL interior span, so a chest-height drop on a
    tall fridge and on a 0.92 m undercounter unit both land on something."""
    lo, hi = mnz + 0.08, mxz - 0.12

    def _clamp(v: float) -> float:
        return min(max(v, min(lo, hi)), max(lo, hi))

    return _clamp


def swing_unit_access(
    cat: str,
    model: str,
    leaf_fields: dict,
    unit_center_xy: tuple[float, float],
    floor_z: float,
    clamp,
    spaces,
) -> tuple[float, float, object, dict[str, object]]:
    import behavior_import as bi

    handle_z = clamp(float(floor_z) + 1.00)
    drop_z = clamp(float(floor_z) + 0.90)
    meta = bi.front_opening_access_metadata(
        cat,
        model,
        leaf_fields,
        unit_center_xy,
    )
    uses_front_aperture = bi.is_front_drum_category(cat) or bool(
        leaf_fields.get("front_flip")
    )
    if not uses_front_aperture or "aperture_center" not in meta:
        return handle_z, drop_z, spaces, meta

    aperture_z = float(meta["aperture_center"][2])
    return clamp(aperture_z + 0.15), clamp(aperture_z), (), meta


def _register_lidded_unit(
    world: World,
    key: str,
    pp: str,
    cat: str,
    model: str,
    leaf_fields: dict | None = None,
) -> bool:
    """Register a top-opening articulated container from its measured lid geometry."""
    import behavior_import as bi
    import numpy as np
    from scene_state import Role

    of = leaf_fields or bi._closing_leaf_fields(
        world.scene, pp, {"cat": cat, "model": model}
    )
    if not of or not of.get("horizontal_top_lid") or not of.get("leaf_bbox"):
        return False
    (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(world.stage, pp)
    (lmn, lmx) = of["leaf_bbox"]
    tcx, tcy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    hx, hy = 0.5 * (mxx - mnx), 0.5 * (mxy - mny)
    lcx, lcy = 0.5 * (lmn[0] + lmx[0]), 0.5 * (lmn[1] + lmx[1])
    lhx, lhy = 0.5 * (lmx[0] - lmn[0]), 0.5 * (lmx[1] - lmn[1])
    ayaw = _best_yaw(siting_boxes(world), tcx, tcy)
    a = math.radians(ayaw)
    front = np.array([-math.sin(a), math.cos(a)], dtype=np.float64)
    lid_depth = abs(float(front[0])) * lhx + abs(float(front[1])) * lhy
    hinge_xy = np.array([lcx, lcy], dtype=np.float64) - front * lid_depth
    handle_xy = np.array([lcx, lcy], dtype=np.float64) + front * lid_depth
    hinge_z = 0.5 * (float(lmn[2]) + float(lmx[2]))
    hinge_axis = (float(front[1]), float(-front[0]), 0.0)
    band = 0.05 * max(0.1, float(mxz) - float(mnz))
    inner_z = float(mnz) + 0.04
    world.front_dir[key] = (float(front[0]), float(front[1]))
    world.scene.register(
        key,
        Role.CONTAINER,
        xy=(tcx, tcy),
        z=inner_z,
        height=max(0.10, (float(mxz) - 0.03) - inner_z),
        half=(max(0.05, hx * 0.85), max(0.05, hy * 0.85)),
        prim_path=pp,
        approach_yaw=ayaw,
        foot_xy=(tcx, tcy),
        foot_half=(float(hx), float(hy)),
        top_z=float(mxz),
        clearance=0.30,
        hinge_xy=(float(hinge_xy[0]), float(hinge_xy[1])),
        hinge_axis=hinge_axis,
        swing_angle=of["swing_angle"],
        swing_sign=1.0,
        leaf_paths=of["leaf_paths"],
        leaf_hinges=of.get("leaf_hinges", ()),
        joint_name=of["joint_name"],
        open_thresh=0.10,
        handle_xy=(float(handle_xy[0]), float(handle_xy[1])),
        handle_z=hinge_z,
        drop_z=max(float(world.scene.floor_z) + 0.02, float(mxz) - band - 0.03),
        spaces=bi._extract_spaces(world.scene.stage, pp, key),
        meta={
            "open_top": True,
            "openable_kind": "lid",
            "auto_open_before_place": True,
            "hinge_xyz": (
                float(hinge_xy[0]),
                float(hinge_xy[1]),
                hinge_z,
            ),
        },
    )
    _record_scored_box(world, key)
    return True


def _register_swing_unit(world: World, key: str, pp: str, cat: str, model: str) -> bool:
    """Register a front-opening unit as a CONTAINER with a reachable interior drop.

    Mirrors the shipped restaurant-closing registration: the door side comes from the ACTUAL
    swinging leaf, the drop lands at a chest-height shelf clamped inside the real interior,
    and ``_extract_spaces`` splits real shelves so ``PutInside`` targets a board, not air."""
    import behavior_import as bi
    from scene_state import Role

    of = bi._closing_leaf_fields(world.scene, pp, {"cat": cat, "model": model})
    if not of or of.get("kind") != "swing":
        return False
    if of.get("horizontal_top_lid"):
        return _register_lidded_unit(world, key, pp, cat, model, of)
    (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(world.stage, pp)
    tcx, tcy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    thx, thy = max(0.05, 0.5 * (mxx - mnx)), max(0.05, 0.5 * (mxy - mny))
    fdir = _door_normal(of, tcx, tcy)
    world.front_dir[key] = fdir
    hd = abs(fdir[0]) * thx + abs(fdir[1]) * thy
    inset = min(0.22, max(0.0, hd - 0.05))
    clamp = _interior_clamp(float(mnz), float(mxz))
    inner_z = float(mnz) + 0.06
    spaces = bi._extract_spaces(world.scene.stage, pp, key)
    handle_z, drop_z, spaces, access_meta = swing_unit_access(
        cat,
        model,
        of,
        (tcx, tcy),
        float(world.scene.floor_z),
        clamp,
        spaces,
    )
    hinge_axis = (0.0, 0.0, 1.0)
    handle_xy = _primary_leaf_handle_xy(of, tcx, tcy)
    if of.get("front_flip"):
        hinge_xyz, hinge_axis, handle_xy, handle_z = _flip_down_geometry(of, fdir)
        access_meta.update(
            openable_kind="flip_down",
            hinge_xyz=hinge_xyz,
        )
    container_spec = (
        ((world.spec.get("world") or {}).get("containers") or {}).get(key) or {}
    )
    door_policy = container_spec.get("door") or {}
    if isinstance(door_policy, Mapping):
        if "place_upright" in door_policy and not isinstance(
            door_policy["place_upright"], bool
        ):
            raise SpecBuildError("door.place_upright must be a boolean")
        action_policy = _validated_action_door_lateral_policy(door_policy)
        for policy_field in (
            "hinge_source",
            "stationary_operator",
            "close_reposition",
            "action_standoff_m",
            "open_standoff_m",
            "close_standoff_m",
            "content_standoff_m",
            "content_insert_standoff_m",
            "content_insert_exact_rest_crouch",
            "content_extract_standoff_m",
            "content_gaze_lift_m",
            "content_camera_pullback_m",
            "content_insert_camera_pullback_m",
            "content_insert_camera_forward_m",
            "content_insert_camera_lateral_m",
            "content_extract_camera_pullback_m",
            "place_upright",
            "operator_lateral_offset_m",
            "sweep_standoff_m",
            "operator_stand_xy",
            "opening_pullback_m",
            "closing_pullback_m",
            "closing_swing_retreat_m",
            "closing_camera_door_lateral_offset_m",
            "closing_camera_door_local_offset_m",
            "action_camera_clearance_m",
            "action_camera_focus",
        ):
            if policy_field in door_policy:
                access_meta[policy_field] = door_policy[policy_field]
        access_meta.update(action_policy)
    world.scene.register(
        key,
        Role.CONTAINER,
        xy=(tcx + fdir[0] * (hd - inset), tcy + fdir[1] * (hd - inset)),
        z=inner_z,
        height=max(0.15, (float(mxz) - 0.1) - inner_z),
        half=(max(0.05, thx * 0.8), max(0.05, thy * 0.8)),
        prim_path=pp,
        approach_yaw=math.degrees(math.atan2(-fdir[0], fdir[1])),
        foot_xy=(tcx, tcy),
        foot_half=(thx, thy),
        top_z=float(mxz),
        clearance=0.35,
        hinge_xy=of["hinge_xy"],
        hinge_axis=hinge_axis,
        swing_angle=of["swing_angle"],
        swing_sign=1.0,
        leaf_paths=of["leaf_paths"],
        leaf_hinges=of.get("leaf_hinges", ()),
        joint_name=of["joint_name"],
        open_thresh=0.10,
        handle_xy=handle_xy,
        handle_z=handle_z,
        drop_z=drop_z,
        spaces=spaces,
        meta=access_meta,
    )
    _record_scored_box(world, key)
    return True


def _validated_action_door_lateral_policy(
    door_policy: Mapping[str, object],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for policy_key, lower, upper in (
        ("open_standoff_m", 0.0, 1.50),
        ("close_standoff_m", 0.0, 1.50),
        ("content_insert_standoff_m", 0.0, 1.50),
        ("content_extract_standoff_m", 0.0, 1.50),
        ("content_insert_camera_forward_m", 0.0, 0.40),
        ("content_insert_camera_lateral_m", -0.30, 0.30),
        ("action_operator_lateral_offset_m", -1.50, 1.50),
        ("content_operator_lateral_offset_m", -1.50, 1.50),
        ("content_insert_operator_lateral_offset_m", -1.50, 1.50),
        ("content_extract_operator_lateral_offset_m", -1.50, 1.50),
        ("close_operator_lateral_offset_m", -1.50, 1.50),
        ("action_lateral_limit_m", 0.0, 1.50),
    ):
        if policy_key not in door_policy:
            continue
        value = door_policy[policy_key]
        if isinstance(value, bool):
            raise SpecBuildError(
                f"door.{policy_key} must be a finite distance in metres"
            )
        try:
            distance = float(value)
        except (TypeError, ValueError) as error:
            raise SpecBuildError(
                f"door.{policy_key} must be a finite distance in metres"
            ) from error
        in_range = (
            lower <= distance <= upper
            if policy_key
            in {
                "action_operator_lateral_offset_m",
                "content_operator_lateral_offset_m",
                "content_insert_operator_lateral_offset_m",
                "content_extract_operator_lateral_offset_m",
                "close_operator_lateral_offset_m",
                "content_insert_camera_forward_m",
                "content_insert_camera_lateral_m",
            }
            else lower < distance <= upper
        )
        if not math.isfinite(distance) or not in_range:
            raise SpecBuildError(
                f"door.{policy_key} must be between {lower:.2f} and {upper:.2f} metres"
            )
        normalized[policy_key] = distance
    limit = normalized.get("action_lateral_limit_m")
    for offset_key in (
        "action_operator_lateral_offset_m",
        "content_operator_lateral_offset_m",
        "content_insert_operator_lateral_offset_m",
        "content_extract_operator_lateral_offset_m",
        "close_operator_lateral_offset_m",
    ):
        offset = normalized.get(offset_key)
        if offset is not None and limit is not None and abs(offset) > limit:
            raise SpecBuildError(
                f"door.{offset_key} must not exceed door.action_lateral_limit_m"
            )
    return normalized


def _register_slide_unit(
    world: World,
    key: str,
    pp: str,
    cat: str,
    model: str,
    leaf_fields: dict | None = None,
) -> bool:
    """Register one scene drawer as an articulated destination container."""
    import behavior_import as bi
    from scene_state import Role

    fields = leaf_fields or bi._closing_leaf_fields(
        world.scene, pp, {"cat": cat, "model": model}
    )
    if not fields or fields.get("kind") != "slide" or not fields.get("leaf_bbox"):
        return False
    (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(world.stage, pp)
    leaf_min, leaf_max = fields["leaf_bbox"]
    body_center = (0.5 * (mnx + mxx), 0.5 * (mny + mxy))
    front = _leaf_front_direction(fields, *body_center)
    if front is None:
        return False
    leaf_center = (
        0.5 * (float(leaf_min[0]) + float(leaf_max[0])),
        0.5 * (float(leaf_min[1]) + float(leaf_max[1])),
    )
    leaf_half = (
        0.5 * (float(leaf_max[0]) - float(leaf_min[0])),
        0.5 * (float(leaf_max[1]) - float(leaf_min[1])),
    )
    travel = float(fields["travel"])
    open_offset = (front[0] * travel, front[1] * travel)
    body_half = (0.5 * (mxx - mnx), 0.5 * (mxy - mny))
    leaf_depth = abs(front[0]) * leaf_half[0] + abs(front[1]) * leaf_half[1]
    handle_xy = (
        leaf_center[0] + front[0] * leaf_depth,
        leaf_center[1] + front[1] * leaf_depth,
    )
    inner_z = float(leaf_min[2]) + 0.02
    inner_top = float(leaf_max[2]) - 0.02
    world.front_dir[key] = front
    world.scene.register(
        key,
        Role.CONTAINER,
        xy=leaf_center,
        z=inner_z,
        height=max(0.05, inner_top - inner_z),
        half=(max(0.03, leaf_half[0] - 0.035), max(0.03, leaf_half[1] - 0.035)),
        prim_path=pp,
        approach_yaw=math.degrees(math.atan2(-front[0], front[1])),
        foot_xy=body_center,
        foot_half=body_half,
        top_z=float(mxz),
        clearance=0.35,
        joint_name=str(fields["joint_name"]),
        axis=(float(front[0]), float(front[1]), 0.0),
        travel=travel,
        leaf_paths=tuple(str(path) for path in fields["leaf_paths"]),
        open_thresh=0.10,
        handle_xy=handle_xy,
        handle_z=0.5 * (float(leaf_min[2]) + float(leaf_max[2])),
        drop_z=inner_z,
        meta={
            "asset_category": cat,
            "asset_model": model,
            "openable_kind": "slide",
            "drawer_open_offset_xy": open_offset,
        },
    )
    _record_scored_box(world, key)
    return True


def _hide_static_slide_leaf(
    stage,
    root_path: str,
    leaf_paths: Sequence[str],
) -> bool:
    """Hide and de-collide a leaf that must remain open for the episode."""
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return False

    def _deinstance(prim) -> None:
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
        for child in prim.GetChildren():
            _deinstance(child)

    _deinstance(root)
    hidden = False
    for path in leaf_paths:
        leaf = stage.GetPrimAtPath(str(path))
        if not leaf or not leaf.IsValid():
            continue
        UsdGeom.Imageable(leaf).MakeInvisible()
        for prim in Usd.PrimRange(leaf):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        hidden = True
    return hidden


def _register_static_front_unit(
    world: World,
    key: str,
    pp: str,
    cat: str,
    model: str,
) -> bool:
    """Register a dishwasher as an already-open front container."""
    import behavior_import as bi
    from scene_state import Role

    if cat != "dishwasher":
        return False
    leaf_fields = bi._static_open_leaf_fields(
        world.scene,
        pp,
        {"cat": cat, "model": model},
    )
    if (
        not leaf_fields
        or not leaf_fields.get("leaf_bbox")
        or not leaf_fields.get("leaf_paths")
    ):
        return False

    (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(world.stage, pp)
    tcx, tcy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    thx, thy = max(0.05, 0.5 * (mxx - mnx)), max(0.05, 0.5 * (mxy - mny))
    front = _door_normal(leaf_fields, tcx, tcy)
    world.front_dir[key] = front
    front_depth = abs(front[0]) * thx + abs(front[1]) * thy
    inset = min(0.22, max(0.0, front_depth - 0.05))
    clamp = _interior_clamp(float(mnz), float(mxz))
    spaces = bi._extract_spaces(world.scene.stage, pp, key)
    geometry_leaf = {**leaf_fields, "kind": "swing"}
    _handle_z, drop_z, spaces, access_meta = swing_unit_access(
        cat,
        model,
        geometry_leaf,
        (tcx, tcy),
        float(world.scene.floor_z),
        clamp,
        spaces,
    )
    if not _hide_static_slide_leaf(
        world.stage,
        pp,
        tuple(str(path) for path in leaf_fields["leaf_paths"]),
    ):
        return False

    access_meta.update(open_top=True, static_open_front=True)
    world.scene.register(
        key,
        Role.CONTAINER,
        xy=(
            tcx + front[0] * (front_depth - inset),
            tcy + front[1] * (front_depth - inset),
        ),
        z=float(mnz) + 0.06,
        height=max(0.15, (float(mxz) - 0.10) - (float(mnz) + 0.06)),
        half=(max(0.05, thx * 0.80), max(0.05, thy * 0.80)),
        prim_path=pp,
        approach_yaw=math.degrees(math.atan2(-front[0], front[1])),
        foot_xy=(tcx, tcy),
        foot_half=(thx, thy),
        top_z=float(mxz),
        clearance=0.35,
        drop_z=drop_z,
        spaces=spaces,
        meta=access_meta,
    )
    _record_scored_box(world, key)
    return True


def _register_open_top_unit(
    world: World,
    key: str,
    pp: str,
    cat: str,
    model: str,
) -> bool:
    if cat == "dishwasher":
        return _register_static_front_unit(world, key, pp, cat, model)
    return _register_lidded_unit(world, key, pp, cat, model) or _register_open_unit(
        world,
        key,
        pp,
        cat,
        model,
    )


def _register_open_unit(world: World, key: str, pp: str, cat: str, model: str) -> bool:
    """Register an OPEN receptacle (shelf bay / trolley / tote) as a CONTAINER.

    Ported from the shipped restaurant-closing ``_register_open``: no leaf, no hinge, and the drop
    lands just under the rim so the released object's base ends up inside the outer bbox z-band that
    ``_relations_inside_live`` scores on. Occupancy rows NEED this: a count behind a closed door is
    not a cue, and a door leaf swung open also inflates the unit's xy footprint."""
    import behavior_import as bi
    from scene_state import Role

    del cat, model
    (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(world.stage, pp)
    hx, hy = 0.5 * (mxx - mnx), 0.5 * (mxy - mny)
    tcx, tcy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    band = 0.05 * max(
        0.1, float(mxz) - float(mnz)
    )  # match _relations_inside_live's z-band
    inner_z = float(mnz) + 0.04
    ayaw = _best_yaw(siting_boxes(world), tcx, tcy)
    a = math.radians(ayaw)
    container_spec = (
        ((world.spec.get("world") or {}).get("containers") or {}).get(key) or {}
    )
    door_policy = container_spec.get("door") or {}
    clearance = 0.30
    if isinstance(door_policy, Mapping) and door_policy.get("content_standoff_m") is not None:
        clearance = float(door_policy["content_standoff_m"])
        if not math.isfinite(clearance) or not 0.0 < clearance <= 1.5:
            raise ValueError(
                f"open container {key!r} has invalid content_standoff_m={clearance!r}"
            )
    world.front_dir[key] = (
        -math.sin(a),
        math.cos(a),
    )  # the person's side -> where the card hangs
    world.scene.register(
        key,
        Role.CONTAINER,
        xy=(tcx, tcy),
        z=inner_z,
        height=max(0.10, (float(mxz) - 0.03) - inner_z),
        half=(max(0.05, hx * 0.85), max(0.05, hy * 0.85)),
        prim_path=pp,
        approach_yaw=ayaw,
        foot_xy=(tcx, tcy),
        foot_half=(float(hx), float(hy)),
        top_z=float(mxz),
        clearance=clearance,
        drop_z=max(float(world.scene.floor_z) + 0.02, float(mxz) - band - 0.03),
        spaces=bi._extract_spaces(world.scene.stage, pp, key),
        meta={"open_top": True, "content_standoff_m": clearance},
    )
    _record_scored_box(world, key)
    return True


_WATER_FIXTURE_PROFILES = {
    ("furniture_sink", "zexzrc"): {
        "control_local": (
            -0.03573794462273053,
            -0.11208835670815454,
            0.5596566768099684,
        ),
        "press_dir_local": (0.0, 0.0, -1.0),
        "stream_lower_local": (
            -0.005927487790659214,
            -0.06749552159433048,
            0.3594817819072472,
        ),
        "stream_upper_local": (
            -0.005927487790659214,
            -0.06749552159433048,
            0.48291053778379955,
        ),
    },
}


def _water_fixture_profile_world(stage, pp: str, cat: str, model: str):
    profile = _WATER_FIXTURE_PROFILES.get((str(cat), str(model)))
    if profile is None:
        return None
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(pp)
    if not prim or not prim.IsValid():
        return None
    transform = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(prim)
    control = transform.Transform(Gf.Vec3d(*profile["control_local"]))
    press_dir = transform.TransformDir(Gf.Vec3d(*profile["press_dir_local"]))
    direction_norm = math.sqrt(sum(float(value) ** 2 for value in press_dir))
    if direction_norm < 1e-8:
        return None
    press_dir = tuple(float(value) / direction_norm for value in press_dir)
    lower = transform.Transform(Gf.Vec3d(*profile["stream_lower_local"]))
    upper = transform.Transform(Gf.Vec3d(*profile["stream_upper_local"]))
    stream = (
        float(upper[0]),
        float(upper[1]),
        float(lower[2]),
        float(upper[2]),
    )
    values = (*control, *press_dir, *stream)
    if not all(math.isfinite(float(value)) for value in values) or stream[3] <= stream[2]:
        return None
    return tuple(float(value) for value in control), press_dir, stream


def register_asset_affordances(
    world: World, key: str, pp: str, cat: str, model: str
) -> None:
    """Attach typed controls declared by asset metadata to an already registered entity."""
    if "toggleable" not in category_affordances(cat):
        return
    from scene_state import Role

    owner = world.scene.resolve(key)
    (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(world.stage, pp)
    cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    front = world.front_dir.get(key)
    if front is None:
        angle = math.radians(float(owner.approach_yaw))
        front = (-math.sin(angle), math.cos(angle))
    norm = math.hypot(float(front[0]), float(front[1])) or 1.0
    front = (float(front[0]) / norm, float(front[1]) / norm)
    half_x, half_y = 0.5 * (mxx - mnx), 0.5 * (mxy - mny)
    front_depth = abs(front[0]) * half_x + abs(front[1]) * half_y
    control_xy = (
        cx + front[0] * (front_depth + 0.01),
        cy + front[1] * (front_depth + 0.01),
    )
    control_z = float(mnz) + 0.85 * max(0.02, float(mxz) - float(mnz))
    if owner.handle_z is not None and control_z - float(owner.handle_z) > 0.25:
        control_z = float(owner.handle_z) + 0.12
    press_dir = (-front[0], -front[1], 0.0)
    profile = _water_fixture_profile_world(world.stage, pp, cat, model)
    profile_meta = {}
    if profile is not None:
        control, press_dir, water_stream = profile
        control_xy = control[:2]
        control_z = control[2]
        profile_meta["water_stream_segment"] = water_stream
    world.scene.register_affordance(
        key,
        Role.TOGGLEABLE,
        prim_path=pp,
        xy=(float(control_xy[0]), float(control_xy[1])),
        z=control_z,
        approach_yaw=float(owner.approach_yaw),
        meta={
            "press_dir": press_dir,
            "asset_category": str(cat),
            "asset_model": str(model),
            "affordance_bbox": (
                (float(mnx), float(mny), float(mnz)),
                (float(mxx), float(mxy), float(mxz)),
            ),
            **profile_meta,
        },
    )


def register_staged_asset_affordances(world: World, key: str, cat: str) -> None:
    """Register controls for a staged movable before its USD prim is realized."""
    if "toggleable" not in category_affordances(cat):
        return
    from scene_state import Role

    owner = world.scene.resolve(key)
    if owner.xy is None or owner.z is None:
        raise SpecBuildError(
            f"staged toggleable {key!r} needs a registered position"
        )
    front = world.front_dir.get(key)
    if front is None:
        angle = math.radians(float(owner.approach_yaw))
        front = (-math.sin(angle), math.cos(angle))
    norm = math.hypot(float(front[0]), float(front[1])) or 1.0
    front = (float(front[0]) / norm, float(front[1]) / norm)
    world.scene.register_affordance(
        key,
        Role.TOGGLEABLE,
        xy=(float(owner.xy[0]), float(owner.xy[1])),
        z=float(owner.z),
        meta={
            "press_dir": (-front[0], -front[1], 0.0),
            "asset_category": str(cat),
        },
    )


def _record_scored_box(world: World, key: str) -> None:
    """Snapshot the exact box the scorer will use for INSIDE, for the disjointness gate and for
    ``behavior_task.container_boxes`` (which is what G1-DISJOINT reads offline).

    The OUTER world AABB is recorded too, because that -- not the registered box -- is what
    ``_relations_inside_live`` actually tests against. With preloaded stock the difference stops
    being cosmetic: one carton counted in two bays inflates both occupancies and both caps."""
    e = world.scene.entities[key]
    world.boxes_of[key] = (
        e.xy[0] - e.half[0],
        e.xy[1] - e.half[1],
        e.z,
        e.xy[0] + e.half[0],
        e.xy[1] + e.half[1],
        e.z + e.height,
    )
    (omn, omx) = _wbbox(world.stage, e.prim_path)
    world.outer_of[key] = (omn[0], omn[1], omn[2], omx[0], omx[1], omx[2])


def _reserve_unit(world: World, key: str, pp: str, foot_half: float) -> None:
    """Keep-out for the unit itself PLUS the clear floor a person needs in FRONT of its door,
    so the next unit cannot be sited into this one's stand and block the approach."""
    cx, cy = _ctr(world.stage, pp)
    fdir = world.front_dir.get(key) or (0.0, -1.0)
    room = next((plan.room for plan in world.layout if plan.key == key), world.room)
    reservations = [(cx, cy, foot_half + UNIT_KEEPOUT_M)]
    (mnx, mny, _z0), (mxx, mxy, _z1) = _wbbox(world.stage, pp)
    hd = abs(fdir[0]) * 0.5 * (mxx - mnx) + abs(fdir[1]) * 0.5 * (mxy - mny)
    sx = cx + fdir[0] * (hd + DOOR_FRONT_CLEAR_M)
    sy = cy + fdir[1] * (hd + DOOR_FRONT_CLEAR_M)
    reservations.append((sx, sy, 0.45))
    world.used_slots.extend(reservations)
    world.room_slots.setdefault(room, []).extend(reservations)
    world.scene.entities[key].meta["front_stand"] = (float(sx), float(sy))


# =====================================================================================
# stage 1b -- SET-LEVEL layout: the room, the bodies and every unit's slot, chosen TOGETHER
# =====================================================================================
@dataclass(frozen=True)
class UnitPlan:
    """One unit's committed slot. Produced by the set-level solver, consumed by ``site_units``."""

    key: str
    cat: str
    model: str
    xy: tuple[float, float]
    fh: float
    stand: tuple[float, float]
    room: str = ""
    front_yaw_deg: float | None = None


@dataclass
class _Field:
    """A room's cell lattice plus the two distance fields the slot test reads, measured ONCE.

    Placing a unit is then a pointwise ``min`` against its footprint, so the search can UNDO a
    placement for free. The greedy siter rebuilt the whole grid after every placement, which is
    exactly why it could not backtrack: undoing meant recomputing."""

    xs: object
    ys: object
    clear: object
    wall: object
    floor: object | None = None


def _dist_field(xs, ys, mn, mx):
    import numpy as np

    dx = np.maximum(np.maximum(float(mn[0]) - xs, 0.0), xs - float(mx[0]))
    dy = np.maximum(np.maximum(float(mn[1]) - ys, 0.0), ys - float(mx[1]))
    return np.hypot(dx, dy)


def _architectural_door_keepouts(world: World) -> list[tuple[str, tuple, tuple]]:
    """Return room-local door boxes padded through their real swing/approach lane."""
    import behavior_import as bi

    _solid, _walls, by_name, _zspan = offline_geom(
        world.recipe.scene, world.recipe.support_bindings
    )
    keepouts = []
    for obj in bi.scene_objects(world.recipe.scene):
        if obj.get("cat") != "door" or world.room not in (obj.get("in_rooms") or ()):
            continue
        name = str(obj.get("name") or "")
        bounds = by_name.get(name)
        if bounds is None:
            continue
        minimum, maximum = bounds
        x_span = float(maximum[0]) - float(minimum[0])
        y_span = float(maximum[1]) - float(minimum[1])
        x_pad = (
            ARCHITECTURAL_DOOR_JAMB_KEEP_OUT_M
            if x_span >= y_span
            else ARCHITECTURAL_DOOR_SWING_KEEP_OUT_M
        )
        y_pad = (
            ARCHITECTURAL_DOOR_SWING_KEEP_OUT_M
            if x_span >= y_span
            else ARCHITECTURAL_DOOR_JAMB_KEEP_OUT_M
        )
        keepouts.append(
            (
                f"portal:{name}",
                (
                    float(minimum[0]) - x_pad,
                    float(minimum[1]) - y_pad,
                    float(minimum[2]),
                ),
                (
                    float(maximum[0]) + x_pad,
                    float(maximum[1]) + y_pad,
                    float(maximum[2]),
                ),
            )
        )
    return keepouts


def _room_field(world: World, box) -> _Field:
    """Vectorised twin of ``_grid`` + ``wall_dist`` over one room's hull."""
    import numpy as np

    x0, y0, x1, y1 = box
    gx, gy = np.arange(x0, x1 + 1e-6, 0.25), np.arange(y0, y1 + 1e-6, 0.25)
    mx_, my_ = np.meshgrid(gx, gy, indexing="ij")
    xs, ys = mx_.ravel(), my_.ravel()
    floor_regions = room_floor_regions(world.recipe.scene, world.room)
    floor_clear = None
    if floor_regions:
        supported, floor_clear = _floor_support_samples(
            world.recipe.scene, world.room, xs, ys
        )
        xs, ys = xs[supported], ys[supported]
        floor_clear = floor_clear[supported]
    elif world.recipe.frozen_bindings:
        raise SpecBuildError(
            f"{world.recipe.task_id}: room has no measured floor support"
        )
    clear = np.full(xs.shape, 1e9)
    for _nm, mn, mx in [
        *siting_boxes(world),
        *_architectural_door_keepouts(world),
    ]:
        clear = np.minimum(clear, _dist_field(xs, ys, mn, mx))
    wall = np.full(xs.shape, 1e9)
    for mn, mx in world.walls:
        wall = np.minimum(wall, _dist_field(xs, ys, mn, mx))
    return _Field(
        xs,
        ys,
        clear,
        wall if world.walls else np.full(xs.shape, 1e9),
        floor_clear,
    )


def _rung_candidates(
    fld: _Field,
    clear,
    fh: float,
    rung,
    near,
    reserved,
    min_margin: float = 0.0,
    branch: int | None = None,
):
    """One rung's admissible cells, ranked by the GREEDY siter's own score."""
    import numpy as np

    margin, dcap0 = rung
    margin = max(float(margin), float(min_margin))
    d = np.hypot(fld.xs - near[0], fld.ys - near[1])
    ok = (
        (clear >= fh + margin)
        & (fld.wall >= fh + max(0.06, float(min_margin)))
        & (d <= min(dcap0, 8.0))
    )
    if min_margin and fld.floor is not None:
        ok &= fld.floor >= fh + float(min_margin)
    for ux, uy, ur in reserved:
        ok &= np.hypot(fld.xs - ux, fld.ys - uy) >= ur
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    score = d[idx] + 1.3 * np.maximum(0.0, fld.wall[idx] - (fh + 0.18))
    order = np.argsort(score, kind="stable")[: branch or _SLOT_BRANCH]
    return [(float(fld.xs[idx[i]]), float(fld.ys[idx[i]])) for i in order]


def _slot_candidates(
    fld: _Field,
    clear,
    fh: float,
    near,
    reserved,
    min_margin: float = 0.0,
    branch: int | None = None,
    fanout: int | None = None,
):
    """Every cell this unit may take, TIGHTEST rung first.

    The margin ladder is walked PER UNIT, exactly as the greedy siter walked it, so the first leaf
    the search reaches is the layout greedy would have built when greedy could build one at all.
    What is new is that a dead end further down no longer ends the story: the search comes back
    and takes this unit's next cell (and, when that runs out, its next RUNG)."""
    seen, out = set(), []
    fanout = fanout or _SLOT_FANOUT
    for rung in _SLOT_MARGINS:
        for xy in _rung_candidates(
            fld,
            clear,
            fh,
            rung,
            near,
            reserved,
            min_margin=min_margin,
            branch=branch,
        ):
            k = (round(xy[0], 3), round(xy[1], 3))
            if k in seen:
                continue
            seen.add(k)
            out.append(xy)
        if len(out) >= fanout:
            break
    return out[:fanout]


# How many cells one unit may be tried in per rung / in total, and how many placements the whole
# search may attempt. Budgets, not answers: exhausting them fails LOUDLY.
_SLOT_BRANCH: int = 14
_SLOT_FANOUT: int = 42
_SLOT_BUDGET: int = 9000


def _push_plan(world: World, st: dict, u: UnitPlan, fld: _Field) -> dict:
    """The state after committing ``u``: it becomes an obstacle, its own keep-out and the clear
    floor in FRONT of its door become reserved, and the next unit chains to it."""
    import numpy as np

    x, y, fh = u.xy[0], u.xy[1], u.fh
    box = ((x - fh, y - fh, 0.0), (x + fh, y + fh, 2.0))
    return {
        "clear": np.minimum(st["clear"], _dist_field(fld.xs, fld.ys, box[0], box[1])),
        "boxes": st["boxes"] + [(f"plan:{u.key}", box[0], box[1])],
        "reserved": st["reserved"]
        + [(x, y, fh + UNIT_KEEPOUT_M), (u.stand[0], u.stand[1], 0.45)],
        "placed": st["placed"] + [u],
        "near": (x, y),
    }


def _explicit_unit_plan(
    world: World,
    unit: UnitRecipe,
    cat: str,
    model: str,
    fh: float,
    st: dict,
) -> UnitPlan:
    import numpy as np

    if unit.siting_xy is None or unit.siting_front_yaw_deg is None:
        raise SpecBuildError(
            f"{world.recipe.task_id}: container {unit.key} has an incomplete explicit siting"
        )
    x, y = unit.siting_xy
    yaw = unit.siting_front_yaw_deg
    x0, y0, x1, y1 = world.box
    if not x0 <= x <= x1 or not y0 <= y <= y1:
        raise SpecBuildError(
            f"{world.recipe.task_id}: explicit siting for {unit.key} at ({x:.3f},{y:.3f}) "
            f"is outside room {world.room!r} hull {(x0, y0, x1, y1)}"
        )
    min_margin = _MOVABLE_UNIT_CLEAR_M if unit.move_mode else 0.0
    supported, floor_clearance = _floor_support_samples(
        world.recipe.scene,
        world.room,
        np.asarray([x], dtype=np.float64),
        np.asarray([y], dtype=np.float64),
    )
    required_floor = fh + min_margin
    if not bool(supported[0]) or float(floor_clearance[0]) < required_floor:
        raise SpecBuildError(
            f"{world.recipe.task_id}: explicit siting for {unit.key} at ({x:.3f},{y:.3f}) "
            f"has floor clearance {float(floor_clearance[0]):.3f}m, below "
            f"the required {required_floor:.3f}m"
        )
    obstacle_boxes = [*st["boxes"], *_architectural_door_keepouts(world)]
    body_clearance = _clearance(obstacle_boxes, x, y)
    required_body = fh + min_margin
    if body_clearance < required_body:
        raise SpecBuildError(
            f"{world.recipe.task_id}: explicit siting for {unit.key} at ({x:.3f},{y:.3f}) "
            f"has obstacle clearance {body_clearance:.3f}m, below the required "
            f"{required_body:.3f}m"
        )
    wall_boxes = [
        (f"wall:{index}", minimum, maximum)
        for index, (minimum, maximum) in enumerate(world.walls)
    ]
    required_wall = fh + max(0.06, min_margin)
    wall_clearance = _clearance(wall_boxes, x, y)
    if wall_clearance < required_wall:
        raise SpecBuildError(
            f"{world.recipe.task_id}: explicit siting for {unit.key} at ({x:.3f},{y:.3f}) "
            f"has wall clearance {wall_clearance:.3f}m, below the required "
            f"{required_wall:.3f}m"
        )
    for reserved_x, reserved_y, reserved_radius in st["reserved"]:
        distance = math.hypot(x - reserved_x, y - reserved_y)
        if distance < reserved_radius:
            raise SpecBuildError(
                f"{world.recipe.task_id}: explicit siting for {unit.key} at "
                f"({x:.3f},{y:.3f}) overlaps a reserved layout lane by "
                f"{reserved_radius - distance:.3f}m"
            )
    angle = math.radians(yaw)
    front = (-math.sin(angle), math.cos(angle))
    stand = (
        x + front[0] * (fh + DOOR_FRONT_CLEAR_M),
        y + front[1] * (fh + DOOR_FRONT_CLEAR_M),
    )
    stand_supported, _stand_clearance = _floor_support_samples(
        world.recipe.scene,
        world.room,
        np.asarray([stand[0]], dtype=np.float64),
        np.asarray([stand[1]], dtype=np.float64),
    )
    if not bool(stand_supported[0]):
        raise SpecBuildError(
            f"{world.recipe.task_id}: explicit siting for {unit.key} puts its front stand "
            f"({stand[0]:.3f},{stand[1]:.3f}) off the measured floor"
        )
    approach_clearance = _approach_clearance(
        [*obstacle_boxes, *wall_boxes], x, y, yaw, fh
    )
    if approach_clearance < WALKABLE_CLEAR_M:
        raise SpecBuildError(
            f"{world.recipe.task_id}: explicit siting for {unit.key} has only "
            f"{approach_clearance:.3f}m of front-approach clearance; "
            f"{WALKABLE_CLEAR_M:.3f}m is required"
        )
    return UnitPlan(
        unit.key,
        cat,
        model,
        (x, y),
        fh,
        stand,
        world.room,
        yaw,
    )


def _plan_unit(
    world: World, fld: _Field, units, i: int, st: dict, budget
) -> list | None:
    """DFS with real backtracking: try every admissible cell for unit ``i``, and on a dead end
    UNDO it and try the next one. The greedy siter committed the first cell and could only fail."""
    if i == len(units):
        return st["placed"] if _layout_ok(world, fld, st) else None
    key, cat, model, fh = units[i]
    unit = next(recipe for recipe in world.recipe.units if recipe.key == key)
    if unit.siting_xy is not None:
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        plan = _explicit_unit_plan(world, unit, cat, model, fh, st)
        return _plan_unit(
            world,
            fld,
            units,
            i + 1,
            _push_plan(world, st, plan, fld),
            budget,
        )
    min_margin = _MOVABLE_UNIT_CLEAR_M if unit.move_mode else 0.0
    one_unit = len(units) == 1
    candidates = _slot_candidates(
        fld,
        st["clear"],
        fh,
        st["near"],
        st["reserved"],
        min_margin=min_margin,
        branch=96 if one_unit else None,
        fanout=256 if one_unit else None,
    )
    approach_boxes = list(st["boxes"]) + [
        (f"wall:{index}", minimum, maximum)
        for index, (minimum, maximum) in enumerate(world.walls)
    ]
    ranked = []
    for x, y in candidates:
        yaw = _best_yaw(approach_boxes, x, y, foot_half=fh)
        clearance = _approach_clearance(approach_boxes, x, y, yaw, fh)
        if clearance >= WALKABLE_CLEAR_M:
            ranked.append(
                (-min(clearance, 2.0), math.hypot(x - st["near"][0], y - st["near"][1]), x, y, yaw)
            )
    for _clearance, _distance, x, y, yaw in sorted(ranked):
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        a = math.radians(yaw)
        fd = (-math.sin(a), math.cos(a))
        stand = (
            x + fd[0] * (fh + DOOR_FRONT_CLEAR_M),
            y + fd[1] * (fh + DOOR_FRONT_CLEAR_M),
        )
        plan = UnitPlan(key, cat, model, (x, y), fh, stand, world.room, yaw)
        got = _plan_unit(
            world, fld, units, i + 1, _push_plan(world, st, plan, fld), budget
        )
        if got is not None:
            return got
    return None


def _layout_ok(world: World, fld: _Field, st: dict) -> bool:
    """Leaf test: the human must be able to WALK from the pass to every unit's door.

    A stand is valid only when a walkable cell is local to the declared point. Connectivity remains
    a preference because this AABB proxy can close aisles that the runtime triangle raster reopens;
    only that component comparison may relax after every connected layout has been exhausted."""
    cells = list(zip(fld.xs.tolist(), fld.ys.tolist(), st["clear"].tolist()))
    walk, comp = _walk_components(cells)
    points = [p.stand for p in st["placed"]]
    support_stands = world.room_support_stands.get(world.room, world.support_stands)
    points.extend(support_stands)
    ids = [
        _component_at(walk, comp, point, max_distance=STAND_COMPONENT_SNAP_M)
        for point in points
    ]
    if any(component is None for component in ids):
        return False
    return not world.nav_strict or len(set(ids)) <= 1


def baked_models(cat: str) -> list[str]:
    """Every model of ``cat`` that has actually been baked to USD."""
    import behavior_import as bi

    d = os.path.join(bi.CACHE_USD, cat)
    if not os.path.isdir(d):
        return []
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".usd"))


def _foot_of(cat: str, model: str) -> float:
    import behavior_import as bi

    bb = bi.read_spec(f"{cat}/{model}").get("bbox_size") or [9.0, 9.0, 9.0]
    return 0.5 * max(float(bb[0]), float(bb[1]))


def _cat_rungs(
    cat: str, f0: float, skip: str = "", want_door: bool = False
) -> list[tuple[str, str]]:
    """One category's models: the SMALLER ones biggest-first (give up as little size as possible),
    then the LARGER ones smallest-first -- a bigger body that WORKS beats an unbuildable row.

    ``want_door`` filters to bodies with a drivable leaf, using the builder's OWN ``unit_models``.
    Walking every baked model instead spent the whole 60-swap budget on bodies the very next check
    rejects for "no drivable swing leaf" -- measured on FIND-3, which burned an hour cycling
    door-less `grocery_shelf` bodies for a unit its program opens."""
    pool = unit_models(cat, want_door) if want_door else baked_models(cat)
    small, large = [], []
    for m in pool or baked_models(cat):
        if m == skip:
            continue
        try:
            f = _foot_of(cat, m)
        except OSError:
            continue
        (small if f < f0 - 0.01 else large).append((f, m))
    return [(cat, m) for _f, m in sorted(small, reverse=True) + sorted(large)]


def _unit_wants_door(recipe: Recipe, key: str) -> bool:
    """Does this unit's own program open it? ``open_top`` is the recipe's record of that."""
    u = next((x for x in recipe.units if x.key == key), None)
    return bool(u is not None and not u.open_top)


def body_ladder(
    cat: str, model: str, alts: tuple[str, ...] = (), want_door: bool = False
) -> list[tuple[str, str]]:
    """The authored body first, then the rest of its own category, then the authored ``alt_cats``.

    A pure function of (cat, model, alts), so a ``twin`` / ``pair_twin`` pair that shares one body
    at level 0 still shares one body at every level: the ladder cannot break a row's geometry."""
    try:
        f0 = _foot_of(cat, model)
    except OSError:
        return [(cat, model)]
    out = [(cat, model)] + _cat_rungs(cat, f0, skip=model, want_door=want_door)
    for a in alts:
        out += _cat_rungs(a, f0, want_door=want_door)
    return out


def same_category_body_ladder(
    cat: str, model: str, *, want_door: bool = False
) -> list[tuple[str, str]]:
    """The authored model, then probe-clean models of the same category before unknown ones."""
    ladder = body_ladder(cat, model, want_door=want_door)
    order = {body: index for index, body in enumerate(ladder)}
    return [
        ladder[0],
        *sorted(
            ladder[1:],
            key=lambda body: (body not in _CLEAN_BODIES, order[body]),
        ),
    ]


def same_category_repair_ladder(
    cat: str,
    model: str,
    *,
    want_door: bool = False,
    reason: str = "putinside_acceptance",
) -> list[tuple[str, str]]:
    """Keep the authored body first, then order same-category repairs for the failure."""
    ladder = same_category_body_ladder(cat, model, want_door=want_door)
    if reason != "container_siting" or len(ladder) < 3:
        return ladder
    order = {body: index for index, body in enumerate(ladder)}

    def rank(body: tuple[str, str]) -> tuple[float, int]:
        try:
            footprint = _foot_of(*body)
        except OSError:
            footprint = float("inf")
        return footprint, order[body]

    return [ladder[0], *sorted(ladder[1:], key=rank)]


def _alts_of(recipe: Recipe) -> dict[tuple[str, str], tuple[str, ...]]:
    return {(u.cat, u.model): u.alt_cats for u in recipe.units}


def unit_bodies_at(
    recipe: Recipe, seed: int, level: int, bump=None, repair_ladders=None
) -> dict[str, tuple[str, str]]:
    """This seed's asset assignment, ``level`` rungs down each unit's own body ladder, plus any
    per-BODY ``bump`` the acceptance repair asked for.

    ``bump`` is keyed by the (cat, model) the recipe authored, never by the unit key: a ``twin`` /
    ``pair_twin`` pair shares one authored body, so bumping the body moves BOTH halves of the pair
    and the row's own "only the placard tells them apart" geometry survives the repair."""
    if recipe.frozen_bindings:
        out: dict[str, tuple[str, str]] = {}
        unit_by_key = {unit.key: unit for unit in recipe.units}
        for key, (cat, model) in unit_assets(recipe, seed).items():
            unit = unit_by_key[key]
            if unit.binding_source == "scene":
                out[key] = (cat, model)
                continue
            index = int(level) + int((bump or {}).get((cat, model), 0))
            if index <= 0:
                out[key] = (cat, model)
                continue
            ladder = (repair_ladders or {}).get((cat, model)) or same_category_body_ladder(
                cat, model, want_door=_unit_wants_door(recipe, key)
            )
            out[key] = ladder[min(index, len(ladder) - 1)]
        return out
    alts = _alts_of(recipe)
    out: dict[str, tuple[str, str]] = {}
    unit_by_key = {unit.key: unit for unit in recipe.units}
    for key, (cat, model) in unit_assets(recipe, seed).items():
        unit = unit_by_key[key]
        if unit.binding_source == "scene":
            out[key] = (unit.cat, unit.model)
            continue
        lad = body_ladder(
            cat,
            model,
            alts.get((cat, model), ()),
            want_door=_unit_wants_door(recipe, key),
        )
        i = int(level) + int((bump or {}).get((cat, model), 0))
        got = lad[min(i, len(lad) - 1)]
        # "distinct" means the ROLE is carried by geometry as well as by the placard, so two units
        # must never collide on one body after a repair swap; step the later one on until it does not.
        while recipe.unit_shape == "distinct" and got in out.values():
            i += 1
            if i >= len(lad):
                break
            got = lad[i]
        out[key] = got
    return out


def ladder_depth(recipe: Recipe, seed: int) -> int:
    if recipe.frozen_bindings:
        return 1
    alts = _alts_of(recipe)
    return max(
        [1]
        + [
            len(
                body_ladder(
                    c, m, alts.get((c, m), ()), want_door=_unit_wants_door(recipe, k)
                )
            )
            for k, (c, m) in unit_assets(recipe, seed).items()
            if next(unit for unit in recipe.units if unit.key == k).binding_source
            != "scene"
        ]
    )


def scene_rooms(scene_name: str) -> dict[str, set[str]]:
    """room -> the non-structural CATEGORIES the scene file puts in it."""
    import json

    p = os.path.join(SCENE_JSON_ROOT, scene_name, "json", f"{scene_name}_best.json")
    with open(p) as f:
        d = json.load(f)
    out: dict[str, set[str]] = {}
    for info in d["objects_info"]["init_info"].values():
        args = (info or {}).get("args") or {}
        cat = str(args.get("category"))
        if cat in _STRUCTURAL:
            continue
        for r in args.get("in_rooms") or []:
            out.setdefault(str(r), set()).add(cat)
    return out


# ⭐ THE STAGING BAND. A support is not "a body of an authored category" -- it is a body the human
# can pick a tub OFF. Measured over the 49 rows batch ``g3o2`` built on the cluster, against the
# top of the elected pass:
#
#   scored 1.00 (23 rows): tops 0.58 .. 1.50 m, every one of them
#   missed  (26 rows)    : 9 of the 10 lowest tops -- 7 of them a garden PAVER (top 0.00, footprint
#                          27.3 x 14.5 m) or a LAWN (top 0.12, 100.8 x 15.8 m) -- scored 0.00/0.50
#
# ``derived_support_cats`` admits ``paver``, ``lawn``, ``fixed_window``, ``hot_tub``, ``bathtub``
# and ``public_trash_can`` alongside ``countertop``, and ``derived_room`` ranks rooms by AREA, so
# the outdoor garden wins the "owns a staging surface" test in every BEHAVIOR house. The row then
# builds -- and stages its stock on the terrace, where it scores 0.00 with no build error. The same
# admission is what leaves ``bathroom_0`` holding one 4 cm deep WINDOW at 2.25 m as T1-HOTL-01's
# only candidate pass. Both are the same defect: PARTS were checked, a WORK SURFACE was not.
#
# The band is a SEARCH PREFERENCE, never a veto: ``plan_layout`` walks the whole room x body ladder
# with it on and, only if nothing sites at all, walks it again with it off. So no world that builds
# today can lose its layout -- the same property that made ``siting_boxes`` safe.
STAGE_TOP_MIN_M: float = (
    0.55  # under this nothing has ever scored 1.00 (lowest that did: 0.58)
)
STAGE_TOP_MAX_M: float = (
    1.60  # over this nothing has ever scored 1.00 (highest that did: 1.50)
)
STAGE_SIDE_MAX_M: float = (
    8.00  # a terrace/lawn is not furniture (widest that scored 1.00: 5.76)
)
STAGE_DEPTH_MIN_M: float = (
    0.12  # a 4 cm window PANE is not a surface you can set a tub down on
)


def stageable_top(world: World, cmn, cmx) -> bool:
    """Is this body's top a work surface a human can stage graspables on and pick them back up?

    Pure geometry off the SAME collision box ``_pick_support`` ranks with, measured against the
    room floor, so the live and the offline pick agree by construction."""
    fz = float(getattr(world.scene, "floor_z", 0.0) or 0.0)
    top = float(cmx[2]) - fz
    sx, sy = float(cmx[0] - cmn[0]), float(cmx[1] - cmn[1])
    return (
        STAGE_TOP_MIN_M <= top <= STAGE_TOP_MAX_M
        and max(sx, sy) <= STAGE_SIDE_MAX_M
        and min(sx, sy) >= STAGE_DEPTH_MIN_M
    )


def _room_has_stageable(world: World, room: str) -> bool:
    """Does ``room`` own at least one in-band support of an authored category? Offline geometry, so
    ``candidate_rooms`` can drop a room BEFORE the siter spends a ladder on it."""
    import behavior_import as bi

    try:
        _s, _w, by_name, _z = offline_geom(
            world.recipe.scene, world.recipe.support_bindings
        )
        cat_of = {o["name"]: o["cat"] for o in bi.scene_objects(world.recipe.scene)}
    except (OSError, ValueError, KeyError):
        return True  # unmeasurable -> keep the room; the band may never DELETE a candidate blindly
    in_room = room_prims(world.recipe.scene, room)
    cats = set(world.recipe.support_cats)
    return any(
        cat_of.get(nm) in cats and stageable_top(world, *by_name[nm])
        for nm in by_name
        if nm in in_room
    )


def candidate_rooms(world: World) -> list[str]:
    """The AUTHORED room first, then every other room in the scene that owns a staging surface of
    the recipe's own ``support_cats``, biggest furniture hull first (floor is the binding
    resource). A room with nothing to stage the graspables on is not a candidate at all.

    While ``world.stage_band`` holds, "a staging surface" means an in-band WORK SURFACE
    (``stageable_top``), so a garden whose only pass is the terrace slab is not a candidate."""
    recipe = world.recipe
    rooms = scene_rooms(recipe.scene)
    out = [recipe.room] if recipe.room else []
    rest = []
    for r, cats in rooms.items():
        if r in out or not (cats & set(recipe.support_cats)):
            continue
        x0, y0, x1, y1 = room_box(recipe.scene, r)
        rest.append(((x1 - x0) * (y1 - y0), r))
    got = out + [r for _a, r in sorted(rest, reverse=True)]
    if not world.stage_band:
        return got
    keep = [r for r in got if _room_has_stageable(world, r)]
    return keep or got


def _plan_in_room(world: World, room: str, bodies: dict) -> tuple[list | None, str]:
    """Try to lay the WHOLE unit set out in one room. Restores ``world`` on failure."""
    keep = _layout_context(world)
    world.room, world.box = room, room_box(world.recipe.scene, room)
    world.anchor = room_anchor(world.recipe.scene, room)
    supports: list[tuple[str, float]] = []
    if world.recipe.n_supports:
        try:
            supports = _pick_supports(world, world.recipe.n_supports)
        except SpecBuildError as e:
            _restore_layout_context(world, keep)
            return None, f"no staging surface ({e})"
        world.anchor = _ctr(world.stage, supports[0][0])
    fld = _room_field(world, world.box)
    units = _unit_search_rows(world.recipe.units, bodies)
    got = _search_orders(world, fld, units)
    if got is None:
        why = _why_stuck(world, fld, units)
        _restore_layout_context(world, keep)
        return None, why
    world.supports = supports
    return got, ""


def _unit_search_rows(units, bodies: dict) -> list[tuple[str, str, str, float]]:
    import behavior_import as bi

    rows = []
    for unit in units:
        cat, model = bodies[unit.key]
        bbox = bi.read_spec(f"{cat}/{model}").get("bbox_size") or [0.5, 0.5]
        rows.append(
            (unit.key, cat, model, 0.5 * max(float(value) for value in bbox[:2]))
        )
    return rows


def _uses_declared_rooms(recipe: Recipe) -> bool:
    return bool(
        recipe.rooms or recipe.support_rooms or any(unit.room for unit in recipe.units)
    )


def _layout_context(world: World) -> tuple:
    return (
        world.room,
        world.box,
        world.anchor,
        list(world.used_slots),
        list(world.support_stands),
        {room: list(rows) for room, rows in world.room_slots.items()},
        {room: list(rows) for room, rows in world.room_support_stands.items()},
        list(world.supports),
        dict(world.support_contact_xy),
        dict(world.support_stand_xy),
    )


def _restore_layout_context(world: World, context: tuple) -> None:
    (
        world.room,
        world.box,
        world.anchor,
        world.used_slots,
        world.support_stands,
        world.room_slots,
        world.room_support_stands,
        world.supports,
        world.support_contact_xy,
        world.support_stand_xy,
    ) = context


def _room_anchor_with_support(world: World, room: str) -> tuple[float, float]:
    for (path, _yaw), support_room in zip(world.supports, world.recipe.support_rooms):
        if support_room == room:
            return _ctr(world.stage, path)
    return room_anchor(world.recipe.scene, room)


def _fixed_scene_plans(world: World, bodies: dict) -> list[UnitPlan]:
    plans = []
    scene_units = [
        unit
        for unit in world.recipe.units
        if unit.binding_source == "scene"
        or unit.key in getattr(world, "scene_unit_fallbacks", {})
    ]
    if not scene_units:
        return plans
    boxes = siting_boxes(world)
    for unit in scene_units:
        fallback = getattr(world, "scene_unit_fallbacks", {}).get(unit.key)
        path = str((fallback or {}).get("path") or unit.pick_prim_path or "")
        if not path or not world.stage.GetPrimAtPath(path):
            raise SpecBuildError(
                f"{world.recipe.task_id}: exact scene unit {unit.key} is absent at {path!r}"
            )
        (mnx, mny, _), (mxx, mxy, _) = _wbbox(world.stage, path)
        x, y = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        fh = 0.5 * max(mxx - mnx, mxy - mny)
        angle = math.radians(_best_yaw(boxes, x, y))
        front = (-math.sin(angle), math.cos(angle))
        stand = (
            x + front[0] * (fh + DOOR_FRONT_CLEAR_M),
            y + front[1] * (fh + DOOR_FRONT_CLEAR_M),
        )
        cat, model = (
            (str(fallback["cat"]), str(fallback["model"]))
            if fallback
            else bodies[unit.key]
        )
        room = str((fallback or {}).get("room") or unit.room)
        plans.append(UnitPlan(unit.key, cat, model, (x, y), fh, stand, room))
    return plans


def _import_groups(
    recipe: Recipe, excluded: Collection[str] = ()
) -> dict[str, list[UnitRecipe]]:
    groups: dict[str, list[UnitRecipe]] = {}
    legal = set(recipe.rooms)
    excluded_keys = set(excluded)
    for unit in recipe.units:
        if unit.binding_source == "scene" or unit.key in excluded_keys:
            continue
        room = unit.room or recipe.room
        if not room or (legal and room not in legal):
            raise SpecBuildError(
                f"{recipe.task_id}: imported unit {unit.key} declares invalid room {room!r}; "
                f"legal rooms are {sorted(legal)}"
            )
        groups.setdefault(room, []).append(unit)
    return groups


def _plan_declared_group(
    world: World, room: str, units: list[UnitRecipe], bodies: dict
) -> tuple[list[UnitPlan] | None, str]:
    world.room = room
    world.box = room_box(world.recipe.scene, room)
    world.anchor = _room_anchor_with_support(world, room)
    world.used_slots = list(world.room_slots.get(room, ()))
    world.support_stands = list(world.room_support_stands.get(room, ()))
    field = _room_field(world, world.box)
    rows = _unit_search_rows(units, bodies)
    got = _search_orders(world, field, rows)
    if got is None:
        return None, f"room={room} phase=layout detail={_why_stuck(world, field, rows)}"
    return got, ""


def _flatten_room_values(values: dict[str, list]) -> list:
    return [row for room in sorted(values) for row in values[room]]


def _plan_declared_rooms(world: World, bodies: dict) -> tuple[list | None, str]:
    context = _layout_context(world)
    world.used_slots, world.support_stands = [], []
    world.room_slots, world.room_support_stands = {}, {}
    try:
        world.supports = (
            _pick_supports(world, world.recipe.n_supports)
            if world.recipe.n_supports
            else []
        )
        plans = _fixed_scene_plans(world, bodies)
        for plan in plans:
            world.room_slots.setdefault(plan.room, []).append(
                (plan.stand[0], plan.stand[1], 0.45)
            )
            world.room_support_stands.setdefault(plan.room, []).append(plan.stand)
        for room, units in _import_groups(
            world.recipe, getattr(world, "scene_unit_fallbacks", {})
        ).items():
            got, why = _plan_declared_group(world, room, units, bodies)
            if got is None:
                _restore_layout_context(world, context)
                return None, why
            plans.extend(got)
    except SpecBuildError as error:
        _restore_layout_context(world, context)
        return None, f"phase=binding detail={error}"
    world.used_slots = _flatten_room_values(world.room_slots)
    world.support_stands = _flatten_room_values(world.room_support_stands)
    world.room = world.recipe.room or next(iter(world.recipe.rooms), "")
    world.box = room_box(world.recipe.scene, world.room)
    world.anchor = _room_anchor_with_support(world, world.room)
    return plans, ""


def _search_orders(world: World, fld: _Field, units) -> list | None:
    """ROSTER order first (that is the order the greedy siter used, so a world greedy could build
    comes out byte-identical), then MOST-CONSTRAINED-FIRST: the biggest body has the fewest
    admissible cells, and placing it last is how a room that could hold the set reports full."""
    import numpy as np

    by_key = {unit.key: unit for unit in world.recipe.units}
    explicit = [row for row in units if by_key[row[0]].siting_xy is not None]
    automatic = [row for row in units if by_key[row[0]].siting_xy is None]
    orders = [explicit + automatic]
    size_order = explicit + sorted(automatic, key=lambda row: -row[3])
    if size_order != orders[0]:
        orders.append(size_order)
    for order in orders:
        st = {
            "clear": np.array(fld.clear),
            "boxes": list(siting_boxes(world)),
            "reserved": list(world.used_slots),
            "placed": [],
            "near": world.anchor,
        }
        got = _plan_unit(world, fld, order, 0, st, [_SLOT_BUDGET])
        if got is not None:
            return got
        if len(automatic) < 2:
            break
    return None


def _why_stuck(world: World, fld: _Field, units) -> str:
    """Name the BINDING constraint, per unit, at the loosest rung -- never a bare 'no room'."""
    import numpy as np

    if len(fld.xs) == 0:
        return (
            "0 measured floor cells at the simulator elevation; choose a room whose "
            "room table reports nonzero floor cells, or change the scene"
        )
    margin, dcap = _SLOT_MARGINS[-1]
    out = []
    for key, _c, _m, fh in units:
        fits = int(np.count_nonzero(fld.clear >= fh + margin))
        offwall = int(
            np.count_nonzero((fld.clear >= fh + margin) & (fld.wall >= fh + 0.06))
        )
        near = int(
            np.count_nonzero(
                np.hypot(fld.xs - world.anchor[0], fld.ys - world.anchor[1]) <= dcap
            )
        )
        out.append(
            f"{key}(fh={fh:.2f}): {fits} cells fit, {offwall} off-wall, {near} within {dcap:.0f}m"
        )
    return f"{len(fld.xs)} cells; " + "; ".join(out)


def plan_layout(world: World) -> list[UnitPlan]:
    """⭐ Choose the ROOM, the BODIES and every unit's slot TOGETHER, with backtracking.

    Four nested preferences, strongest first: stage on a real WORK SURFACE, then keep the AUTHORED
    BODY, then keep the AUTHORED ROOM. Every interaction stand must be local to walkable floor and
    connected layouts are preferred, then the conservative AABB proxy may relax connectivity only
    after every such layout is exhausted. The local-stand and full approach-corridor checks remain
    hard constraints. The staging band is still a relaxation. Running out of rooms or bodies is a
    loud, attributed failure."""
    if _uses_declared_rooms(world.recipe):
        return _search_declared_room_layout(world)
    tried: list[str] = []
    for band in (True, False):
        world.stage_band = band
        for level in range(ladder_depth(world.recipe, world.seed)):
            bodies = unit_bodies_at(
                world.recipe,
                world.seed,
                level,
                world.body_bump,
                getattr(world, "frozen_body_ladders", {}),
            )
            for strict in (True, False):
                world.nav_strict = strict
                for room in candidate_rooms(world):
                    got, why = _plan_in_room(world, room, bodies)
                    if got is not None:
                        if not band:
                            print(
                                f"[bench] STAGING BAND relaxed: nothing in "
                                f"{STAGE_TOP_MIN_M}-{STAGE_TOP_MAX_M} m could hold this row; "
                                f"staging in {room} out of band"
                            )
                        return _commit_layout(world, got, room, level, strict)
                    if strict and band:
                        tried.append(
                            f"{room} @ L{level} {sorted(set(bodies.values()))}: {why}"
                        )
    raise SpecBuildError(
        f"{world.recipe.task_id}: cannot site {len(world.recipe.units)} unit(s) anywhere in "
        f"{world.recipe.scene}. Rooms x bodies tried: " + " | ".join(tried)
    )


def _search_declared_room_layout(world: World) -> list[UnitPlan]:
    tried: list[str] = []
    for band in (True, False):
        world.stage_band = band
        for level in range(ladder_depth(world.recipe, world.seed)):
            bodies = unit_bodies_at(
                world.recipe,
                world.seed,
                level,
                world.body_bump,
                getattr(world, "frozen_body_ladders", {}),
            )
            for strict in (True, False):
                world.nav_strict = strict
                got, why = _plan_declared_rooms(world, bodies)
                if got is not None:
                    if not band:
                        print(
                            f"[bench] STAGING BAND relaxed for declared rooms "
                            f"{list(world.recipe.rooms)}"
                        )
                    return _commit_declared_layout(world, got, level, strict)
                tried.append(
                    f"body_level={level} nav_strict={strict} stage_band={band} {why}"
                )
    raise SpecBuildError(
        f"{world.recipe.task_id}: cannot site declared multi-room layout in "
        f"{world.recipe.scene}. Rejections: " + " | ".join(tried)
    )


# =====================================================================================
# the OFFLINE ROOM CHECK -- ask whether the scene has ROOM, not just whether it has PARTS
# =====================================================================================
# ``derive_recipe`` matches roles to assets by affordance and stops there, so a spec that no room
# can physically hold still passes offline and dies twenty minutes later in Isaac. Everything the
# live siter reads is measurable without Isaac: the scene JSON gives every body's pose, scale and
# room, the bake cache gives its bbox, and the room hull, anchor and support pick are already pure
# functions of those. So the SAME predicates -- ``_room_field``, ``_search_orders``, the margin
# ladder, the body ladder, the door-front reservation -- can be run offline in milliseconds.
_OFFLINE_GEOM: dict[object, tuple] = {}


def _offline_box(o, dz: float):
    """(name, mn, mx) of one scene body from its baked bbox, its scale and its yaw. None when the
    model was never baked -- an unbaked body is not in the live scene either."""
    import behavior_import as bi
    import numpy as np

    try:
        spec = bi.read_spec(f"{o['cat']}/{o['model']}")
        bb = spec.get("bbox_size")
        offset = spec.get("base_link_offset") or [0.0, 0.0, 0.0]
    except (OSError, ValueError):
        return None
    if not bb:
        return None
    sc = o.get("scale") or [1.0, 1.0, 1.0]
    if isinstance(sc, (int, float)):
        sc = [sc, sc, sc]
    w, h, d = (float(bb[i]) * float(sc[i]) for i in range(3))
    ox, oy, oz = (float(offset[i]) * float(sc[i]) for i in range(3))
    qx, qy, qz, qw = (float(v) for v in o["ori"])
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    c, s = abs(cosine), abs(sine)
    hx, hy = 0.5 * (w * c + h * s), 0.5 * (w * s + h * c)
    x, y, z = (float(v) for v in o["pos"])
    rotated_ox = cosine * ox - sine * oy
    rotated_oy = sine * ox + cosine * oy
    x, y, z = x + rotated_ox, y + rotated_oy, z + oz
    z += dz
    return (
        str(o["name"]),
        np.array([x - hx, y - hy, z - 0.5 * d]),
        np.array([x + hx, y + hy, z + 0.5 * d]),
    )


def offline_geom(scene: str, include_categories: Sequence[str] = ()) -> tuple:
    """(solid boxes, wall boxes, name -> box, name -> (bottom, top)) for one scene, floor top at 0.

    Mirrors ``_compose_behavior_scene``: skipped visual categories remain absent unless the task
    explicitly binds one as a fixture; ``floors`` and ``walls`` are not scene_state obstacles.
    The whole scene is shifted so the highest floor slab's top is z = 0."""
    included = tuple(sorted({str(category) for category in include_categories}))
    cache_key = (scene, included)
    if cache_key in _OFFLINE_GEOM:
        return _OFFLINE_GEOM[cache_key]
    import behavior_import as bi

    objs = bi.scene_objects(scene)
    tops = [
        b[2][2]
        for b in (_offline_box(o, 0.0) for o in objs if o["cat"] == "floors")
        if b
    ]
    dz = -float(max(tops)) if tops else 0.0
    solid, walls, by_name, zspan = [], [], {}, {}
    for o in objs:
        if (
            o["cat"] in bi._SKIP_CATS and o["cat"] not in included
        ) or o["cat"] == "floors":
            continue
        box = _offline_box(o, dz)
        if box is None:
            continue
        nm, mn, mx = box
        if o["cat"] == "walls":
            walls.append((mn, mx))
            continue
        # ``solid_boxes()`` reports a floor->top pillar; ``siting_boxes`` restores the real z span
        # from ``zspan``, so hand it the same two numbers it would measure off the stage.
        solid.append((nm, mn.copy(), mx))
        solid[-1][1][2] = 0.0
        by_name[nm] = (mn, mx)
        zspan[nm] = (float(mn[2]), float(mx[2]))
    _OFFLINE_GEOM[cache_key] = (solid, walls, by_name, zspan)
    return _OFFLINE_GEOM[cache_key]


class _OfflineScene:
    """The two things the siting predicates ask a scene for."""

    floor_z: float = 0.0

    def __init__(self, boxes: list) -> None:
        self._boxes = boxes

    def solid_boxes(self) -> list:
        return self._boxes


def offline_world(recipe: Recipe, seed: int = 0) -> World:
    """A ``World`` whose geometry came from the scene JSON and the bake cache, not from Isaac."""
    solid, walls, by_name, zspan = offline_geom(
        recipe.scene, recipe.support_bindings
    )
    w = World(stage=None, app=None, spec={}, recipe=recipe, seed=seed)
    w.scene = _OfflineScene(list(solid))
    w.walls = walls
    w.room = recipe.room
    w.nav_strict = False
    w.placed = {nm: (f"offline:{nm}", {}) for nm in by_name}
    w.zspan = dict(
        zspan
    )  # same length as ``placed`` -> ``body_zspans`` never touches a stage
    return w


def _offline_support_pool(
    world: World,
    by_name: Mapping[str, tuple],
    in_room: set[str],
    allowed: set[str],
    names: tuple[str, ...] | None,
    fixture_index: int,
) -> list[str]:
    import behavior_import as bi

    objects_by_name = {
        str(obj["name"]): obj for obj in bi.scene_objects(world.recipe.scene)
    }
    required = (
        frozenset(world.recipe.support_capabilities[fixture_index])
        if fixture_index < len(world.recipe.support_capabilities)
        else frozenset()
    )
    requires_worktop = support_requires_worktop(world.recipe, fixture_index)
    payload_profile = _support_payload_profile(world, fixture_index)
    scene_boxes = [
        (name, minimum, maximum)
        for name, (minimum, maximum) in by_name.items()
    ]
    pool = []
    for name in by_name:
        obj = objects_by_name.get(name) or {}
        if (
            obj.get("cat") not in allowed
            or name not in in_room
            or (names and name not in names)
            or not _support_passes_stage_band(
                world, by_name[name], requires_worktop
            )
        ):
            continue
        if "openable" in required:
            try:
                asset_spec = bi.read_spec(f"{obj['cat']}/{obj['model']}")
            except (KeyError, OSError, ValueError):
                continue
            if not _has_support_openable(asset_spec):
                continue
        minimum, maximum = by_name[name]
        if (
            names is None
            and payload_profile is not None
            and not _support_has_payload_clearance(
                name,
                (minimum, maximum),
                float(maximum[2]),
                payload_profile,
                scene_boxes,
            )
        ):
            continue
        pool.append(name)
    return pool


def _best_offline_support(
    world: World,
    room: str,
    pool: Sequence[str],
    by_name: Mapping[str, tuple],
    boxes: Sequence[tuple],
    used: set[str],
    skip: set[str],
    has_floor: bool,
    fixture_index: int,
) -> tuple[tuple | None, int]:
    import behavior_import as bi

    metadata_by_name = {
        str(obj["name"]): obj for obj in bi.scene_objects(world.recipe.scene)
    }
    seatable = _fixture_is_seatable(world, fixture_index)
    best = None
    rejected_off_floor = 0
    for name in pool:
        if name in used or name in skip:
            continue
        minimum, maximum = by_name[name]
        center_x = 0.5 * (minimum[0] + maximum[0])
        center_y = 0.5 * (minimum[1] + maximum[1])
        half_x = 0.5 * (maximum[0] - minimum[0])
        half_y = 0.5 * (maximum[1] - minimum[1])
        reachable = int(float(maximum[2]) + 0.05 <= reach_max_z())
        declared_yaw = _declared_support_value(
            world, "support_approach_yaws", fixture_index
        )
        angles = (
            (float(declared_yaw),)
            if declared_yaw is not None
            else support_approach_yaws(
                metadata_by_name.get(name) or {},
                seatable=seatable,
                openable_fields=None,
                center_x=center_x,
                center_y=center_y,
            )
        )
        for angle in angles:
            site, rejected = _best_fixture_interaction_site(
                world,
                fixture_index,
                center=(center_x, center_y),
                half=(half_x, half_y),
                yaw=float(angle),
                boxes=boxes,
                obstacle_name=name,
                room=room,
                has_floor=has_floor,
            )
            rejected_off_floor += rejected
            if site is None:
                continue
            clearance, contact, stand = site
            option = (
                reachable,
                clearance,
                name,
                stand[0],
                stand[1],
                float(angle),
                contact[0],
                contact[1],
            )
            if best is None or option[:2] > best[:2]:
                best = option
    return best, rejected_off_floor


def _offline_supports(
    w: World,
    room: str,
    n: int,
    categories: tuple[str, ...] | None = None,
    skip: set[str] | None = None,
    names: tuple[str, ...] | None = None,
    fixture_index: int = 0,
) -> list[str]:
    """Offline twin of ``_pick_supports``: the n in-room bodies of the recipe's own support
    categories with the most open approach side, each reserving the cell the human stands on."""
    _solid, _walls, by_name, _z = offline_geom(
        w.recipe.scene, w.recipe.support_bindings
    )
    in_room = room_prims(w.recipe.scene, room)
    boxes = siting_boxes(w)
    allowed = set(categories or w.recipe.support_cats)
    got: list[str] = []
    has_floor = bool(room_floor_regions(w.recipe.scene, room))
    rejected_off_floor = 0
    for _ in range(max(1, int(n))):
        support_index = fixture_index + len(got)
        pool = _offline_support_pool(
            w,
            by_name,
            in_room,
            allowed,
            names,
            support_index,
        )
        best, rejected = _best_offline_support(
            w,
            room,
            pool,
            by_name,
            boxes,
            set(got),
            skip or set(),
            has_floor,
            support_index,
        )
        rejected_off_floor += rejected
        if best is None:
            raise SpecBuildError(
                f"no {sorted(allowed)} in {room} "
                f"{_fixture_requirement_phrase(w, support_index)}"
                + (
                    f"; {rejected_off_floor} approach pose(s) were outside modeled floor"
                    if rejected_off_floor
                    else ""
                )
                + "; object spot coordinates do not move a scene fixture's approach pose; "
                "change the fixture category, room, or scene"
            )
        got.append(best[2])
        w.supports.append((best[2], best[5]))
        w.support_contact_xy[support_index] = (float(best[6]), float(best[7]))
        w.support_stand_xy[support_index] = (float(best[3]), float(best[4]))
        slot = (best[3], best[4], 0.50)
        stand = (float(best[3]), float(best[4]))
        w.used_slots.append(slot)
        w.support_stands.append(stand)
        w.room_slots.setdefault(room, []).append(slot)
        w.room_support_stands.setdefault(room, []).append(stand)
    return got


def _offline_explicit_support(
    world: World,
    room: str,
    fixture_index: int,
    by_name: dict[str, tuple],
    used: set[str],
) -> str:
    import numpy as np

    key = world.recipe.support_keys[fixture_index]
    category = world.recipe.support_bindings[fixture_index]
    model = world.recipe.support_models[fixture_index]
    xy = world.recipe.support_siting_xy[fixture_index]
    siting_yaw = world.recipe.support_siting_front_yaws[fixture_index]
    if not model or xy is None or siting_yaw is None:
        raise SpecBuildError(
            f"{world.recipe.task_id}: imported fixture {key} has incomplete explicit siting"
        )
    name = f"support_explicit_{key}"
    bounds = _support_siting_bounds(category, model, xy, siting_yaw)
    _validate_explicit_support_site(
        world,
        key,
        room,
        bounds,
        siting_boxes(world),
    )
    minimum, maximum = bounds
    world.scene._boxes.append(
        (name, np.asarray(minimum, dtype=np.float64), np.asarray(maximum, dtype=np.float64))
    )
    world.placed[name] = (
        f"offline:{name}",
        {"cat": category, "model": model, "room": room},
    )
    world.zspan[name] = (float(minimum[2]), float(maximum[2]))
    explicit_bounds = getattr(world, "explicit_support_bounds", None)
    if explicit_bounds is None:
        explicit_bounds = {}
        world.explicit_support_bounds = explicit_bounds
    explicit_bounds[name] = bounds
    by_name[name] = bounds
    best, rejected = _best_offline_support(
        world,
        room,
        (name,),
        by_name,
        siting_boxes(world),
        used,
        set(),
        bool(room_floor_regions(world.recipe.scene, room)),
        fixture_index,
    )
    if best is None:
        raise SpecBuildError(
            f"{world.recipe.task_id}: explicit support {key} has no valid approach"
            + (
                f"; {rejected} approach pose(s) were outside modeled floor"
                if rejected
                else ""
            )
        )
    world.supports.append((name, float(best[5])))
    world.support_contact_xy[fixture_index] = (float(best[6]), float(best[7]))
    world.support_stand_xy[fixture_index] = (float(best[3]), float(best[4]))
    slot = (float(best[3]), float(best[4]), 0.50)
    stand = (float(best[3]), float(best[4]))
    world.used_slots.append(slot)
    world.support_stands.append(stand)
    world.room_slots.setdefault(room, []).append(slot)
    world.room_support_stands.setdefault(room, []).append(stand)
    return name


def _offline_plan_in_room(w: World, room: str, bodies: dict) -> tuple[list | None, str]:
    """``_plan_in_room`` with the stage-bound support pick swapped for its offline twin."""
    keep = _layout_context(w)
    w.room, w.box = room, room_box(w.recipe.scene, room)
    w.anchor = room_anchor(w.recipe.scene, room)
    try:
        sup = _offline_supports(w, room, w.recipe.n_supports)
    except SpecBuildError as e:
        _restore_layout_context(w, keep)
        return None, f"no staging surface ({e})"
    (cmn, cmx) = offline_geom(
        w.recipe.scene, w.recipe.support_bindings
    )[2][sup[0]]
    w.anchor = (0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1]))
    fld = _room_field(w, w.box)
    units = [
        (u.key, bodies[u.key][0], bodies[u.key][1], _foot_of(*bodies[u.key]))
        for u in w.recipe.units
    ]
    got = _search_orders(w, fld, units)
    if got is None:
        why = _why_stuck(w, fld, units)
        _restore_layout_context(w, keep)
        return None, why
    return got, ""


def _offline_declared_supports(w: World) -> dict[str, tuple[float, float]]:
    anchors: dict[str, tuple[float, float]] = {}
    if not w.recipe.n_supports:
        return anchors
    used: set[str] = set()
    by_name = dict(offline_geom(w.recipe.scene, w.recipe.support_bindings)[2])
    for index in range(w.recipe.n_supports):
        room = (
            w.recipe.support_rooms[index]
            if index < len(w.recipe.support_rooms)
            else w.recipe.room
        )
        categories = (
            (w.recipe.support_bindings[index],)
            if index < len(w.recipe.support_bindings)
            else None
        )
        exact_name = (
            w.recipe.support_names[index]
            if index < len(w.recipe.support_names)
            else ""
        )
        explicit_xy = _declared_support_value(w, "support_siting_xy", index)
        name = (
            _offline_explicit_support(w, room, index, by_name, used)
            if explicit_xy is not None
            else _offline_supports(
                w,
                room,
                1,
                categories,
                used,
                (exact_name,) if exact_name else None,
                fixture_index=index,
            )[0]
        )
        used.add(name)
        if room not in anchors:
            cmn, cmx = by_name[name]
            anchors[room] = (0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1]))
    return anchors


def _offline_plan_declared_rooms(
    w: World, bodies: dict
) -> tuple[list[UnitPlan] | None, str]:
    keep = _layout_context(w)
    w.used_slots, w.support_stands = [], []
    w.room_slots, w.room_support_stands = {}, {}
    w.supports = []
    try:
        anchors = _offline_declared_supports(w)
        plans: list[UnitPlan] = []
        for room, units in _import_groups(w.recipe).items():
            w.room, w.box = room, room_box(w.recipe.scene, room)
            w.anchor = anchors.get(room) or room_anchor(w.recipe.scene, room)
            w.used_slots = list(w.room_slots.get(room, ()))
            w.support_stands = list(w.room_support_stands.get(room, ()))
            field = _room_field(w, w.box)
            rows = [
                (unit.key, *bodies[unit.key], _foot_of(*bodies[unit.key]))
                for unit in units
            ]
            got = _search_orders(w, field, rows)
            if got is None:
                why = _why_stuck(w, field, rows)
                _restore_layout_context(w, keep)
                return None, f"room={room} phase=layout detail={why}"
            plans.extend(got)
    except (OSError, KeyError, SpecBuildError) as error:
        _restore_layout_context(w, keep)
        return None, f"phase=binding detail={error}"
    return plans, ""


# The offline search walks the SAME room x body ladder as ``plan_layout``, but a 150-rung ladder
# over a 50000-cell garden is a minute of CPU, and a check nobody can afford to run is not a gate.
# The cap is on the ROOM x BODY pairs tried, and exhausting it is reported as UNKNOWN, never FAIL.
OFFLINE_PAIRS: int = 24


def _predict_siting_details(
    recipe: Recipe, seed: int
) -> tuple[bool | None, str, World | None, list[UnitPlan], str]:
    if recipe.is_assembly() or not recipe.units:
        return None, "not a container row", None, [], ""
    if _uses_declared_rooms(recipe):
        return _predict_declared_siting_details(recipe, seed)
    w = offline_world(recipe, seed)
    tried, pairs = [], 0
    # Same two-pass relaxation as ``plan_layout``: the WORK-SURFACE band first, then everything.
    # The budget is per pass, so a row that needs the relaxation still gets a full ladder.
    for band in (True, False):
        w.stage_band, pairs = band, 0
        for level in range(ladder_depth(recipe, seed)):
            bodies = unit_bodies_at(recipe, seed, level, w.body_bump)
            for room in candidate_rooms(w):
                pairs += 1
                if pairs > OFFLINE_PAIRS:
                    if band:
                        break
                    return (
                        None,
                        f"budget: {pairs - 1} room x body pairs tried, none sited; "
                        f"first rejections: {' | '.join(tried[:2])}",
                        None,
                        [],
                        "",
                    )
                got, why = _offline_plan_in_room(w, room, bodies)
                if got is not None:
                    return (
                        True,
                        f"{room} @ body rung {level}"
                        + ("" if band else " (band relaxed)"),
                        w,
                        got,
                        room,
                    )
                if band:
                    tried.append(f"{room}@L{level}: {why}")
            else:
                continue
            break
    return False, "; ".join(tried[:3]), None, [], ""


def _predict_declared_siting_details(
    recipe: Recipe, seed: int
) -> tuple[bool | None, str, World | None, list[UnitPlan], str]:
    world = offline_world(recipe, seed)
    tried = []
    pairs = 0
    for band in (True, False):
        world.stage_band = band
        for level in range(ladder_depth(recipe, seed)):
            pairs += max(1, len(_import_groups(recipe)))
            if pairs > OFFLINE_PAIRS:
                return (
                    None,
                    f"budget: {pairs - 1} room x body pairs tried, none sited; "
                    f"first rejections: {' | '.join(tried[:2])}",
                    None,
                    [],
                    "",
                )
            bodies = unit_bodies_at(recipe, seed, level, world.body_bump)
            got, why = _offline_plan_declared_rooms(world, bodies)
            if got is not None:
                rooms = sorted({plan.room for plan in got} | set(recipe.support_rooms))
                return (
                    True,
                    f"{','.join(rooms)} @ body rung {level}"
                    + ("" if band else " (band relaxed)"),
                    world,
                    got,
                    "",
                )
            if band:
                tried.append(f"L{level}: {why}")
    return False, "; ".join(tried[:3]), None, [], ""


def predict_siting(recipe: Recipe, seed: int = 0) -> tuple[bool | None, str]:
    """Can the scene hold this recipe using the live builder's layout search?"""
    sited, reason, _world, _plans, _room = _predict_siting_details(recipe, seed)
    return sited, reason


def predict_staging(recipe: Recipe, seed: int = 0) -> tuple[bool, str]:
    """Run the builder's support and floor packing arithmetic without starting Isaac."""
    items = list(recipe.staged())
    if recipe.is_assembly() and not items:
        return True, ""
    if not items and not recipe.preloads:
        return False, (
            f"{max(1, int(recipe.n_supports))} declared support(s) and no staged graspable at all; "
            "every [Grab] in the program would fail to ground"
        )

    try:
        movable_groups = _movable_support_groups(tuple(items), recipe.task_id)
    except SpecBuildError as error:
        return False, str(error)
    movable_on_items = {
        item.key for _parent, children in movable_groups for item in children
    }
    on_items = {
        item.key
        for item in items
        if item.initial_place.startswith("on:")
        and item.key not in movable_on_items
    }
    floor_items = [item for item in items if item.initial_place.startswith("floor:")]
    assigned = {key for group in recipe.support_items for key in group}
    if recipe.support_items:
        unknown = sorted(assigned - {item.key for item in items})
        missing = sorted(on_items - assigned)
        nested = sorted(assigned & movable_on_items)
        misplaced = sorted(
            item.key for item in floor_items if item.key in assigned
        )
        if unknown or missing or nested or misplaced:
            return False, (
                "support_items does not match initial placement: "
                f"unknown={unknown}, missing_on_support={missing}, "
                f"movable_nested_listed={nested}, floor_listed={misplaced}"
            )
    elif floor_items and recipe.n_supports:
        return False, (
            "floor-staged objects share an implicit support group; the live builder would seat "
            "them twice instead of preserving their declared floor placement"
        )
    elif on_items and not recipe.n_supports:
        return False, "objects declare on:<surface> placement but no staging surface exists"

    try:
        _offline_stage_geometry(recipe, seed)
    except (KeyError, OSError, SpecBuildError, ValueError) as error:
        return False, str(error)
    return True, ""


def _offline_staging_world(recipe: Recipe, seed: int) -> tuple[World, list[UnitPlan]]:
    if recipe.units:
        sited, reason, world, plans, _room = _predict_siting_details(recipe, seed)
        if sited is not True or world is None:
            raise SpecBuildError(f"layout unavailable for staging check: {reason}")
        return world, plans
    last_error = ""
    for band in (True, False):
        world = offline_world(recipe, seed)
        world.stage_band = band
        try:
            if recipe.n_supports:
                if _uses_declared_rooms(recipe):
                    _offline_declared_supports(world)
                else:
                    _offline_supports(world, recipe.room, recipe.n_supports)
            return world, []
        except (KeyError, OSError, SpecBuildError, ValueError) as error:
            last_error = str(error)
    raise SpecBuildError(last_error or "no staging surface can be selected")


def _reserve_offline_plans(world: World, plans: list[UnitPlan]) -> None:
    for plan in plans:
        # The live floor-item pass rebuilds its clearance grid after the imported units exist, so
        # their full footprints are real obstacles there.  Reserving only the centre keep-out made
        # the offline twin optimistic: it could approve a floor object in space occupied by the
        # unit's outer body, then the live builder found no slot.  Add the same planned square body
        # used by the layout solver to the offline scene before checking support/floor staging.
        world.scene._boxes.append(
            (
                f"plan:{plan.key}",
                (plan.xy[0] - plan.fh, plan.xy[1] - plan.fh, 0.0),
                (plan.xy[0] + plan.fh, plan.xy[1] + plan.fh, 2.0),
            )
        )
        world.room_slots.setdefault(plan.room, []).extend(
            [
                (plan.xy[0], plan.xy[1], plan.fh + UNIT_KEEPOUT_M),
                (plan.stand[0], plan.stand[1], 0.45),
            ]
        )


def _check_offline_support_capacity(
    world: World,
) -> dict[str, tuple[float, float]]:
    if not world.recipe.n_supports:
        return {}
    supports = list(world.supports)
    if len(supports) != max(1, int(world.recipe.n_supports)):
        raise SpecBuildError(
            f"selected {len(supports)} staging surfaces for "
            f"{world.recipe.n_supports} declared surfaces"
        )
    by_name = dict(
        offline_geom(world.recipe.scene, world.recipe.support_bindings)[2]
    )
    by_name.update(getattr(world, "explicit_support_bounds", {}))
    positions: dict[str, tuple[float, float]] = {}
    for index, ((name, yaw), block) in enumerate(
        zip(supports, _item_blocks(world, len(supports)))
    ):
        if not block:
            continue
        cmn, cmx = by_name[name]
        cx, cy = 0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1])
        hx, hy = 0.5 * (cmx[0] - cmn[0]), 0.5 * (cmx[1] - cmn[1])
        angle = math.radians(yaw)
        fwd = (math.sin(angle), -math.cos(angle))
        perp = (-fwd[1], fwd[0])
        depth = min(
            hx / max(abs(fwd[0]), 1e-6), hy / max(abs(fwd[1]), 1e-6)
        )
        span = min(
            hx / max(abs(perp[0]), 1e-6), hy / max(abs(perp[1]), 1e-6)
        )
        inset = max(0.08, depth * 0.45)
        try:
            _bodies, offsets, _gap, _wrapped, _orientations = _resolve_seat_layout(
                world, block, fwd, perp, span, depth, inset
            )
        except SpecBuildError as error:
            support = (
                world.recipe.support_keys[index]
                if index < len(world.recipe.support_keys)
                else f"support_{index + 1}"
            )
            category = (
                world.recipe.support_cats[index]
                if index < len(world.recipe.support_cats)
                else "surface"
            )
            room = (
                world.recipe.support_rooms[index]
                if index < len(world.recipe.support_rooms)
                else world.recipe.room
            )
            keys = [item.key for item in block]
            raise SpecBuildError(
                f"support {support} ({category} in {room}) cannot stage its "
                f"{len(block)} assigned objects {keys}: {error}. Split these objects across "
                f"two or more scene-backed surfaces in {room}; add or reuse a fixture and "
                "change every moved world.items[*].place to on:<that fixture>. Adding an "
                "unused fixture does not change the layout"
            ) from error
        ex, ey = cx - fwd[0] * inset, cy - fwd[1] * inset
        for item, (across, into) in zip(block, offsets):
            positions[item.key] = (
                float(ex + perp[0] * across + fwd[0] * into),
                float(ey + perp[1] * across + fwd[1] * into),
            )
    return positions


def _check_offline_floor_capacity(
    world: World, floor_items: list[ItemRecipe]
) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    by_room: dict[str, list[ItemRecipe]] = {}
    for item in floor_items:
        room = item.initial_place[6:] or world.recipe.room
        by_room.setdefault(room, []).append(item)
    for room, items in by_room.items():
        world.room = room
        world.box = room_box(world.recipe.scene, room)
        world.anchor = room_anchor(world.recipe.scene, room)
        world.used_slots = list(world.room_slots.get(room, ()))
        cells, _boxes = _grid(world)
        for item in items:
            cat, model = staged_body(world, item)
            bbox = staged_bbox(world, item)
            half = 0.5 * max(float(bbox[0]), float(bbox[1]))
            slot, reserve = _find_slot(
                world, cells, half, near=world.anchor, dcap_max=5.0
            )
            if slot is None:
                raise SpecBuildError(
                    f"room={room} phase=floor item={item.key} detail=no clear slot "
                    f"for {cat}/{model}. The authored spot field does not control physical "
                    "packing; move floor items onto reachable surfaces, move the interaction "
                    "to a room/scene with enough measured floor, or reduce only unscored "
                    "redundant clutter"
                )
            reservation = (float(slot[0]), float(slot[1]), float(reserve))
            world.used_slots.append(reservation)
            world.room_slots.setdefault(room, []).append(reservation)
            positions[item.key] = (float(slot[0]), float(slot[1]))
    return positions


def _movable_support_layout(
    world: World,
    parent: ItemRecipe,
    children: tuple[ItemRecipe, ...],
    yaw: float,
):
    parent_bbox = staged_bbox(world, parent)
    angle = math.radians(float(yaw))
    fwd = (math.sin(angle), -math.cos(angle))
    perp = (math.cos(angle), math.sin(angle))
    return _resolve_seat_layout(
        world,
        children,
        fwd,
        perp,
        0.5 * float(parent_bbox[0]),
        0.5 * float(parent_bbox[1]),
        0.0,
    )


def _check_offline_movable_support_capacity(
    world: World, positions: dict[str, tuple[float, float]]
) -> None:
    by_key = {item.key: item for item in world.recipe.staged()}
    for parent_key, children in _movable_support_groups(
        world.recipe.staged(), world.recipe.task_id
    ):
        parent_xy = positions.get(parent_key)
        if parent_xy is None:
            raise SpecBuildError(
                f"{world.recipe.task_id}: movable support {parent_key!r} "
                "has no initial staged position"
            )
        _bodies, offsets, _gap, _wrapped, _orientations = _movable_support_layout(
            world, by_key[parent_key], children, 0.0
        )
        for child, (across, into) in zip(children, offsets):
            positions[child.key] = (
                float(parent_xy[0] + across),
                float(parent_xy[1] - into),
            )


def _offline_stage_geometry(
    recipe: Recipe, seed: int
) -> tuple[World, dict[str, tuple[float, float]]]:
    world, plans = _offline_staging_world(recipe, seed)
    _reserve_offline_plans(world, plans)
    positions = _check_offline_support_capacity(world)
    floor_items = [
        item for item in recipe.staged() if item.initial_place.startswith("floor:")
    ]
    positions.update(_check_offline_floor_capacity(world, floor_items))
    _check_offline_movable_support_capacity(world, positions)
    return world, positions


def _predict_start_in_world(
    recipe: Recipe,
    world: World,
    room: str,
    cue_xy: tuple[float, float] | None = None,
) -> tuple[bool, str]:
    import numpy as np

    cue_item = recipe.items[0] if recipe.items else None
    try:
        world.room, world.box = room, room_box(recipe.scene, room)
        world.anchor = room_anchor(recipe.scene, room)
        if cue_xy is not None:
            cue = cue_xy
        elif _uses_declared_rooms(recipe):
            if cue_item is None or not cue_item.initial_place.startswith("on:"):
                return True, ""
            support = cue_item.initial_place[3:]
            index = recipe.support_keys.index(support)
            categories = (
                (recipe.support_bindings[index],)
                if index < len(recipe.support_bindings)
                else None
            )
            exact_name = (
                recipe.support_names[index]
                if index < len(recipe.support_names)
                else ""
            )
            sup = _offline_supports(
                world,
                room,
                1,
                categories,
                names=(exact_name,) if exact_name else None,
                fixture_index=index,
            )
        else:
            sup = _offline_supports(world, room, world.recipe.n_supports)
    except (SpecBuildError, OSError, KeyError):
        return True, ""  # cannot measure it -> cannot veto on it
    if cue_xy is None:
        cmn, cmx = offline_geom(recipe.scene, recipe.support_bindings)[2][sup[0]]
        cue = (0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1]))
    fld = _room_field(world, world.box)
    d = np.hypot(fld.xs - cue[0], fld.ys - cue[1])
    ok = (d >= MIN_CUE_DIST_M) & (d <= MAX_CUE_DIST_M) & (fld.wall >= 0.40)
    for ux, uy, ur in world.used_slots:
        ok &= np.hypot(fld.xs - ux, fld.ys - uy) >= ur
    if int(np.count_nonzero(ok)) > 0:
        return True, ""
    x0, y0, x1, y1 = world.box
    return False, (
        f"{room} ({x1 - x0:.1f} x {y1 - y0:.1f} m, {len(fld.xs)} cells) has no cell "
        f"{MIN_CUE_DIST_M}-{MAX_CUE_DIST_M} m from the cue AND 0.40 m off the walls; the room "
        "cannot host a start where the cue is illegible at t=0"
    )


def predict_start_pose(
    recipe: Recipe, room: str = "", seed: int = 0
) -> tuple[bool, str]:
    """Check the start pose against the offline-staged item geometry."""
    cue_item = recipe.items[0] if recipe.items else None
    if _uses_declared_rooms(recipe) and cue_item is not None:
        room = _initial_item_room(recipe, cue_item)
    room = room or recipe.room
    if not room:
        return True, ""
    try:
        world, item_xy = _offline_stage_geometry(recipe, seed)
    except (KeyError, OSError, SpecBuildError, ValueError) as error:
        return False, str(error)
    world.used_slots = list(world.room_slots.get(room, ()))
    cue = item_xy.get(cue_item.key) if cue_item is not None else None
    return _predict_start_in_world(recipe, world, room, cue)


def predict_start_pose_after_siting(
    recipe: Recipe, seed: int = 0
) -> tuple[bool | None, str]:
    """Check the start against the same unit and item layout the offline builder chose."""
    try:
        world, item_xy = _offline_stage_geometry(recipe, seed)
    except (KeyError, OSError, SpecBuildError, ValueError) as error:
        return False, str(error)
    cue_item = recipe.items[0] if recipe.items else None
    room = world.room or recipe.room
    if _uses_declared_rooms(recipe) and cue_item is not None:
        room = _initial_item_room(recipe, cue_item)
    world.used_slots = list(world.room_slots.get(room, ()))
    cue = item_xy.get(cue_item.key) if cue_item is not None else None
    return _predict_start_in_world(recipe, world, room, cue)


def _offline_assembly_order(recipe: Recipe, spec: dict) -> list[PartRecipe]:
    points = attach_points_of(recipe)
    mates = derive_mates(spec, recipe, points)
    roots = assembly_roots(mates, recipe)
    initial_children = {
        str(row.get("child") or "")
        for row in (spec.get("world") or {}).get("initial_attachments") or []
        if isinstance(row, dict)
    }
    structural = [mate.child for mate in mates if mate.child in initial_children]
    by_key = {part.key: part for part in recipe.parts}
    free = sorted(
        set(by_key) - set(roots) - set(structural),
        key=lambda key: (len(key), key),
    )
    return [by_key[key] for key in [*roots, *structural, *free]]


def predict_assembly_siting(
    recipe: Recipe, spec: dict, seed: int = 0
) -> tuple[bool | None, str]:
    """Can the exact flat-pack and its manual fit on the declared room's floor?"""
    if not recipe.is_assembly():
        return None, "not an assembly row"
    try:
        import behavior_import as bi

        world = offline_world(recipe, seed)
        world.room = recipe.room
        floor = _room_floor_box(recipe.scene, recipe.room)
        world.anchor = (0.5 * (floor[0] + floor[2]), 0.5 * (floor[1] + floor[3]))
        world.box = (
            max(world.anchor[0] - ASSEMBLY_SPAN_M, floor[0] + 0.7),
            max(world.anchor[1] - ASSEMBLY_SPAN_M, floor[1] + 0.7),
            min(world.anchor[0] + ASSEMBLY_SPAN_M, floor[2] - 0.7),
            min(world.anchor[1] + ASSEMBLY_SPAN_M, floor[3] - 0.7),
        )
        if world.box[2] - world.box[0] < 2.0 or world.box[3] - world.box[1] < 2.0:
            return False, f"assembly floor hull {world.box} is smaller than 2 m"
        cells, _boxes = _grid(world)
        near = None
        for part in _offline_assembly_order(recipe, spec):
            bbox = bi.read_spec(f"{part.cat}/{part.model}").get("bbox_size") or [
                0.3,
                0.3,
                0.05,
            ]
            half = 0.5 * max(float(bbox[0]), float(bbox[1]))
            slot, _margin = _find_slot(
                world, cells, half, near=near, dcap_max=ASSEMBLY_SPAN_M
            )
            if slot is None:
                return False, f"no open floor in {recipe.room} for board {part.key}"
            world.used_slots.append((slot[0], slot[1], half + 0.30))
            if near is None:
                near = slot
        if recipe.manual_key:
            slot, _margin = _find_slot(
                world,
                cells,
                MANUAL_HALF_W,
                near=world.anchor,
                dcap_max=ASSEMBLY_SPAN_M,
            )
            if slot is None:
                return False, f"no open floor in {recipe.room} for the manual board"
    except (KeyError, OSError, SpecBuildError, ValueError) as error:
        return None, f"offline assembly measurement unavailable: {error}"
    return True, f"{len(recipe.parts)} board(s) fit in {recipe.room}"


def room_report(task_id: str, seed: int = 0) -> dict:
    """Every offline room verdict for one row, as data. The single entry point for the gate."""
    spec = load_spec(task_id)
    recipe = recipe_for(task_id)
    if recipe.is_assembly():
        sited, why_site = predict_assembly_siting(recipe, spec, seed)
    else:
        sited, why_site = predict_siting(recipe, seed)
    staged, why_stage = predict_staging(recipe, seed)
    # ``choose_start`` runs in the room the SOLVER committed, which is only seeded from the
    # recipe's own preference -- judging the preference would condemn rows the solver rehomes.
    start_room = (
        _initial_item_room(recipe, recipe.items[0])
        if _uses_declared_rooms(recipe) and recipe.items
        else (why_site.split(" @ ")[0] if sited else "")
    )
    start, why_start = predict_start_pose(recipe, start_room, seed)
    fails = []
    if sited is False:
        fails.append(
            f"N1 no room in {recipe.scene} sites {len(recipe.units)} unit(s): {why_site}"
        )
    if not staged:
        fails.append(f"N3 {why_stage}")
    if not start:
        fails.append(f"N5 {why_start}")
    return {
        "task_id": task_id,
        "scene": recipe.scene,
        "room": recipe.room,
        "n_units": len(recipe.units),
        "sited": sited,
        "why_sited": why_site,
        "staged": staged,
        "start": start,
        "fails": fails,
    }


def _commit_layout(
    world: World, got: list, room: str, level: int, strict: bool
) -> list[UnitPlan]:
    world.layout = list(got)
    if not strict:
        print(
            "[bench][WARN] no layout puts every unit door on ONE AABB-proxy nav component; the "
            "render's precise_nav triangle raster decides at render time"
        )
    print(
        f"[bench] LAYOUT: room={room} body_level={level} nav_connected={strict} "
        + ", ".join(
            f"{p.key}={p.cat}/{p.model}@({p.xy[0]:.2f},{p.xy[1]:.2f})" for p in got
        )
    )
    # ⭐ ROOM-MOVED. The solver relocating a bank is CORRECT behaviour -- it is what stops an authored
    # room that cannot physically hold the units from failing the build. What was missing is anyone
    # noticing WHERE it landed: T1-OFFC-01 names `copy_room_0`, a 0.08 x 1.03 m sliver, and the whole
    # battery bank was silently rehomed into `bathroom_0`, so an office row rendered as a washroom for
    # two days. The move is now one loud line instead of two lines nobody diffed.
    want = str(getattr(world.recipe, "room", "") or "")
    if want and room and room != want:
        print(
            f"[bench] ⚠️ ROOM-MOVED: the recipe asked for {want!r}; the set-level solver could not "
            f"fit {len(got)} unit(s) there and used {room!r} instead. CHECK WHAT {room!r} LOOKS "
            f"LIKE -- the frames are the task."
        )
    return world.layout


def _commit_declared_layout(
    world: World, got: list[UnitPlan], level: int, strict: bool
) -> list[UnitPlan]:
    world.layout = list(got)
    if not strict:
        print(
            "[bench][WARN] at least one declared room needs render-time precise_nav "
            "validation; its local AABB proxy was disconnected"
        )
    by_room: dict[str, list[UnitPlan]] = {}
    for plan in got:
        by_room.setdefault(plan.room, []).append(plan)
    detail = " | ".join(
        f"{room}: "
        + ", ".join(
            f"{plan.key}={plan.cat}/{plan.model}@({plan.xy[0]:.2f},{plan.xy[1]:.2f})"
            for plan in plans
        )
        for room, plans in by_room.items()
    )
    print(
        f"[bench] DECLARED LAYOUT: body_level={level} nav_connected={strict} {detail}"
    )
    return world.layout


def _stage_prim_identity(prim) -> dict[str, str]:
    """Exact stage path plus the digest of the USD reference that created it."""
    import hashlib

    path, name = str(prim.GetPath()), str(prim.GetName())
    refs = prim.GetMetadata("references")
    items = []
    if refs is not None:
        try:
            items = list(refs.GetAddedOrExplicitItems())
        except AttributeError:
            items = []
    assets = sorted(
        str(getattr(item, "assetPath", "") or "")
        for item in items
        if str(getattr(item, "assetPath", "") or "")
    )
    asset_path = assets[0] if len(assets) == 1 else ""
    asset_ref = ""
    asset_sha256 = ""
    if asset_path:
        parts = asset_path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            asset_ref = "/".join(parts[-2:])
        try:
            digest = hashlib.sha256()
            with open(asset_path, "rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            asset_sha256 = f"sha256:{digest.hexdigest()}"
        except OSError:
            pass
    return {
        "path": path,
        "name": name,
        "asset_ref": asset_ref,
        "asset_sha256": asset_sha256,
    }


def _scene_native_pool(world: World) -> dict[str, list[dict[str, str]]]:
    """``{category: [stage paths]}`` of the scene's OWN containers, unused so far this build.

    A room with five bookcases still had four lockers trucked into it, because the builder only ever
    imported. Those imports are also the single biggest source of build failures (run d4: 12 of 14
    were "cannot site N units") and the reason a task can look nothing like the room it is set in.
    Empty dict -> every unit imports exactly as before."""
    if os.getenv("BENCH_SCENE_UNITS", "1").strip() == "0":
        return {}
    try:
        import scene_furniture_units as sfu
    except Exception:  # noqa: BLE001 - module absent -> import path, unchanged
        return {}
    from pxr import Usd

    # world.placed holds EVERY body the scene owns, not just this row's -- reading it as "taken"
    # marked all 94 prims used and emptied the pool on the first real run. What this row has claimed
    # is world.reg_pp, which only ``site_units`` writes, one entry per unit it has placed.
    taken = {str(v) for v in (world.reg_pp or {}).values()}
    try:
        cands = sfu.candidates(str(world.recipe.scene))
    except (OSError, ValueError):
        return {}
    prims = [
        _stage_prim_identity(prim)
        for prim in Usd.PrimRange(world.stage.GetPseudoRoot())
    ]
    pool = scene_native_bindings(cands, prims, taken)
    if not pool:
        # Silence here would read as "this scene owns nothing", which is usually false: say which of
        # the three joins failed (no candidates / no prim on the stage yet / all already taken).
        n_b = sum(1 for row in prims if row["name"].startswith("b_"))
        print(
            f"[bench] scene-native pool EMPTY: {len(cands)} candidate(s) from the scene JSON, "
            f"{n_b} b_* prim(s) on the stage, {len(taken)} already taken"
        )
    else:
        rooms = sorted(
            {row["room"] for rows in pool.values() for row in rows if row.get("room")}
        )
        print(
            f"[bench] scene-native pool in {rooms}: "
            + ", ".join(f"{k}x{len(v)}" for k, v in sorted(pool.items()))
        )
    return pool


def scene_native_bindings(
    candidates: Sequence[Mapping[str, object]],
    prims: Sequence[Mapping[str, object] | tuple[str, str]],
    taken: set[str],
) -> dict[str, list[dict[str, str]]]:
    """Join scene inventory instances to their exact stage prims once each."""
    normalized = []
    for raw in prims:
        if isinstance(raw, Mapping):
            row = {
                key: str(raw.get(key) or "")
                for key in ("path", "name", "asset_ref", "asset_sha256")
            }
        else:
            path, name = raw
            row = {
                "path": str(path),
                "name": str(name),
                "asset_ref": "",
                "asset_sha256": "",
            }
        if row["path"] not in taken:
            normalized.append(row)
    by_path = {row["path"]: row for row in normalized}
    pool: dict[str, list[dict[str, str]]] = {}
    seen: set[str] = set()
    for candidate in candidates:
        name = str(candidate.get("name") or "")
        cat = str(candidate.get("cat") or "")
        model = str(candidate.get("model") or "")
        path = str(candidate.get("stage_prim_path") or "")
        digest = str(candidate.get("instance_digest") or "")
        asset_ref = str(candidate.get("asset_ref") or "")
        asset_sha256 = str(candidate.get("asset_sha256") or "")
        room = str(
            candidate.get("room") or next(iter(candidate.get("rooms") or ()), "")
        )
        if (
            not name
            or not cat
            or not model
            or path != f"/World/b_{name}"
            or candidate.get("stage_prim_name") != f"b_{name}"
            or not digest
            or not asset_ref
            or not asset_sha256
        ):
            continue
        actual = by_path.get(path)
        if (
            actual is None
            or actual["name"] != f"b_{name}"
            or actual["asset_ref"] != asset_ref
            or actual["asset_sha256"] != asset_sha256
            or path in seen
        ):
            continue
        pool.setdefault(cat, []).append(
            {
                "path": path,
                "cat": cat,
                "model": model,
                "name": name,
                "asset_sha256": asset_sha256,
                "instance_digest": digest,
                "room": room,
            }
        )
        seen.add(path)
    for rows in pool.values():
        rows.sort(key=lambda row: row["path"])
    return pool


def same_category_scene_fallbacks(
    world: World, keys: Sequence[str]
) -> dict[str, dict[str, str]]:
    """Choose unused scene-native bodies without changing a frozen unit's category."""
    if getattr(world, "stage", None) is None:
        return {}
    pool = _scene_native_pool(world)
    current = unit_bodies_at(
        world.recipe,
        world.seed,
        0,
        world.body_bump,
        getattr(world, "frozen_body_ladders", {}),
    )
    units = {unit.key: unit for unit in world.recipe.units}
    rejected = set(getattr(world, "scene_unit_rejections", set()))
    used = {
        str(row.get("path") or "")
        for row in getattr(world, "scene_unit_fallbacks", {}).values()
    } | {str(path) for path in getattr(world, "reg_pp", {}).values()}
    selected: dict[str, dict[str, str]] = {}
    for key in keys:
        unit = units.get(key)
        if unit is None or unit.binding_source == "scene" or key not in current:
            continue
        category = current[key][0]
        rows = [
            row
            for row in pool.get(category, ())
            if row.get("path") not in rejected
            and row.get("path") not in used
            and (
                not unit.room
                or not row.get("room")
                or row.get("room") == unit.room
            )
        ]
        if not rows:
            continue
        row = dict(rows[0])
        row["cat"] = category
        selected[key] = row
        used.add(row["path"])
    return selected


def _enable_same_category_scene_fallbacks(
    world: World, keys: Sequence[str]
) -> list[str]:
    selected = same_category_scene_fallbacks(world, keys)
    if not selected:
        return []
    fallbacks = getattr(world, "scene_unit_fallbacks", {})
    fallbacks.update(selected)
    world.scene_unit_fallbacks = fallbacks
    logger.info(
        f"[bench] frozen same-category scene fallback: "
        f"{ {key: row['path'] for key, row in selected.items()} }"
    )
    return list(selected)


def _reject_scene_unit_fallback(world: World, key: str) -> None:
    fallback = getattr(world, "scene_unit_fallbacks", {}).pop(key, None)
    if not fallback:
        return
    rejected = getattr(world, "scene_unit_rejections", set())
    rejected.add(str(fallback.get("path") or ""))
    world.scene_unit_rejections = rejected


def _take_scene_native(
    world: World, u, p, pool: dict[str, list[dict[str, str]]]
) -> tuple[str, str]:
    """``(prim path, model)`` for this unit, or ``("", "")``.

    Separation is the whole of the logic. The first run took the three nearest cabinets and the
    build died on ``scored containers overlap (INSIDE is not exclusive)``: three destinations whose
    interiors intersect cannot be told apart by a containment test, so the task is unscorable no
    matter what the person does. Imported units never hit this because the siter reserves a slot per
    unit; a scene's own furniture is wherever the scene put it, often in a row, so the separation has
    to be asked for here."""
    rows = [
        row
        for row in pool.get(p.cat) or []
        if not p.room or not row.get("room") or row["room"] == p.room
    ]
    if not rows:
        return "", ""
    if u.pick_instance_digest:
        indexed = [
            (index, row)
            for index, row in enumerate(rows)
            if row["path"] == u.pick_prim_path
            and row["name"] == u.pick_name
            and row["asset_sha256"] == u.pick_asset_sha256
            and row["instance_digest"] == u.pick_instance_digest
        ]
        if len(indexed) != 1:
            raise SpecBuildError(
                f"{world.recipe.task_id}: exact scene instance {u.pick_name!r} "
                f"at {u.pick_prim_path!r} is absent or its stage asset digest drifted"
            )
    else:
        indexed = list(enumerate(rows))
        if u.pick_rank is not None:
            indexed = [indexed[u.pick_rank]] if 0 <= u.pick_rank < len(indexed) else []
    used = {
        key: (_wbbox(world.stage, path), path)
        for key, path in (world.reg_pp or {}).items()
        if world.stage.GetPrimAtPath(path)
    }
    exact_scene_keys = {
        unit.key
        for unit in world.recipe.units
        if unit.binding_source == "scene" and unit.pick_instance_digest
    }
    import scene_furniture_units as sfu

    for _i, row in indexed:
        path, model = row["path"], row["model"]
        try:
            (mnx, mny, _z0), (mxx, mxy, _z1) = _wbbox(world.stage, path)
        except Exception:  # noqa: BLE001 - an unreadable prim is simply not a candidate
            continue
        cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
        ok = True
        blocked_by = None
        blocked_distance = None
        for used_key, (((umnx, umny, _), (umxx, umxy, _)), used_path) in used.items():
            ux, uy = 0.5 * (umnx + umxx), 0.5 * (umny + umxy)
            distance = ((cx - ux) ** 2 + (cy - uy) ** 2) ** 0.5
            if not _scene_unit_pair_has_minimum_separation(
                distance,
                minimum_m=sfu.SCENE_UNIT_MIN_SEP_M,
                exact_scene_pair=(
                    u.key in exact_scene_keys and used_key in exact_scene_keys
                ),
            ):
                ok = False
                blocked_by = f"{used_key}@{used_path}"
                blocked_distance = distance
                break
        if ok:
            return path, model
        if u.pick_instance_digest:
            raise SpecBuildError(
                f"{world.recipe.task_id}: exact scene instance {u.pick_name!r} at {path} "
                f"is only {float(blocked_distance or 0.0):.3f}m from {blocked_by}; "
                f"minimum is {sfu.SCENE_UNIT_MIN_SEP_M:.3f}m"
            )
    return "", ""


def _scene_unit_pair_has_minimum_separation(
    distance_m: float,
    *,
    minimum_m: float,
    exact_scene_pair: bool,
) -> bool:
    tolerance_m = (
        EXACT_SCENE_UNIT_SEPARATION_TOLERANCE_M if exact_scene_pair else 0.0
    )
    return distance_m + tolerance_m >= minimum_m


def site_units(world: World) -> None:
    """Place every unit: reuse one of the SCENE'S OWN containers when the room already has one of
    the planned category, else import at the PLANNED slot. Orient, register, reserve.

    Raises rather than skip: a task whose unit_c never registered would score on three of four
    containers and pass wrong answers. A reused scene body is marked ``scene_owned`` so
    ``assert_placard_rides`` measures -- on this stage, for this row -- that its placard actually
    swings with its door instead of hanging in the opening."""
    import behavior_import as bi
    import fpv_scene

    pool = _scene_native_pool(world)
    n_native = 0
    for u, p in zip(world.recipe.units, _ordered_plan(world)):
        fallback = getattr(world, "scene_unit_fallbacks", {}).get(u.key)
        use_native = u.binding_source == "scene" or (
            u.binding_source == "auto" and not u.open_top
        ) or fallback is not None
        native, nmodel = (
            (str(fallback["path"]), str(fallback["model"]))
            if fallback is not None
            else (_take_scene_native(world, u, p, pool) if use_native else ("", ""))
        )
        if native:
            world.placed[u.key] = (native, {"cat": p.cat, "model": nmodel})
            registered = (
                _register_open_top_unit(world, u.key, native, p.cat, nmodel)
                if u.open_top
                else (
                    _register_slide_unit(world, u.key, native, p.cat, nmodel)
                    or _register_swing_unit(world, u.key, native, p.cat, nmodel)
                )
            )
            if registered:
                register_asset_affordances(world, u.key, native, p.cat, nmodel)
                _register_unit_mobility(world, u)
                ent = world.scene.entities[u.key]
                if not hasattr(ent, "meta") or ent.meta is None:
                    ent.meta = {}
                ent.meta[SCENE_UNIT_MARK] = True
                _reserve_unit(world, u.key, native, p.fh)
                world.reg_pp[u.key] = native
                n_native += 1
                print(f"[bench] unit {u.key} = SCENE-OWNED {p.cat}/{nmodel} @ {native}")
                continue
            world.placed.pop(u.key, None)
            if fallback is not None:
                _reject_scene_unit_fallback(world, u.key)
                raise UnitBodyUnusable(
                    u.key,
                    p.cat,
                    nmodel,
                    "scene-native same-category fallback cannot register as the declared container kind",
                )
            if u.binding_source == "scene":
                raise SpecBuildError(
                    f"{world.recipe.task_id}: scene-bound unit {u.key} ({p.cat}) exists but "
                    "cannot register as the declared container kind"
                )
        elif u.binding_source == "scene":
            raise SpecBuildError(
                f"{world.recipe.task_id}: scene-bound unit {u.key} could not bind "
                f"category {p.cat!r} at room rank {u.pick_rank!r}"
            )
        pp = f"/World/bu_{bi._safe(u.key)}"
        prim = fpv_scene.add_reference(
            world.stage, pp, bi.baked_usd(f"{p.cat}/{p.model}")
        )
        _drop_to_floor(world, prim, pp, p.xy[0], p.xy[1], (1.0, 0.0, 0.0, 0.0))
        quat = (
            (1.0, 0.0, 0.0, 0.0)
            if u.open_top
            else _door_quat(
                world,
                pp,
                p.cat,
                p.model,
                p.xy,
                siting_boxes(world),
                desired_yaw=(
                    p.front_yaw_deg
                    if p.front_yaw_deg is not None
                    else math.degrees(
                        math.atan2(
                            -(p.stand[0] - p.xy[0]),
                            p.stand[1] - p.xy[1],
                        )
                    )
                ),
            )
        )
        _drop_to_floor(world, prim, pp, p.xy[0], p.xy[1], quat)
        world.placed[u.key] = (pp, {"cat": p.cat, "model": p.model})
        registered = (
            _register_open_top_unit(world, u.key, pp, p.cat, p.model)
            if u.open_top
            else (
                _register_slide_unit(world, u.key, pp, p.cat, p.model)
                or _register_swing_unit(world, u.key, pp, p.cat, p.model)
            )
        )
        if not registered:
            raise UnitBodyUnusable(
                u.key, p.cat, p.model, "has no drivable swing or slide leaf"
            )
        register_asset_affordances(world, u.key, pp, p.cat, p.model)
        _register_unit_mobility(world, u)
        _reserve_unit(world, u.key, pp, p.fh)
        world.reg_pp[u.key] = pp
        print(
            f"[bench] unit {u.key} = {p.cat}/{p.model} @ ({p.xy[0]:.2f},{p.xy[1]:.2f}) "
            f"open_top={u.open_top} zone={world.zones.get(u.key, '-')} drop_z="
            f"{world.scene.entities[u.key].drop_z:.2f}"
        )


def _register_unit_mobility(world: World, recipe: UnitRecipe) -> None:
    if not recipe.move_mode:
        return
    if recipe.move_mode != "push_pull":
        raise UnsupportedSpec(
            f"{world.recipe.task_id}: unit {recipe.key} has unsupported move_mode "
            f"{recipe.move_mode!r}"
        )
    world.scene.add_mobility(recipe.key, push=True, pull=True)
    world.scene.entities[recipe.key].meta["move_mode"] = recipe.move_mode


_MAX_BODY_SWAPS: int = 20
# ``UnitBodyUnusable`` is not an acceptance MEASUREMENT -- it is a metadata fact about the body
# (a doorless ``shelf`` or ``trash_can`` rung offered to a unit that needs a drivable swing leaf),
# known before a single probe runs, and it advances the ladder by exactly ONE rung. Measured on the
# six N4 rows of batch ``g3o2``: the last 3 of every row's 20 swaps were spent that way --
# ``shelf/vanltx``, ``shelf/kxvbui``, ``shelf/wiiyfk`` for T1-GCAF-02 -- so the probe budget ran out
# on bodies that could never have been probed at all. They get their own, larger allowance.
_MAX_UNUSABLE_SWAPS: int = 60
_MAX_FROZEN_SAME_CATEGORY_SWAPS: int = 4


def site_accepting_units(world: World) -> None:
    """Plan + site the unit set, and REPAIR any body whose interior swallows the probe.

    The acceptance gate in ``_finish`` is a hard build failure by design, so this loop is what
    makes it satisfiable without an author hand-picking a model: a unit that fails the probe has
    its BODY advanced one rung down its own ladder and the whole layout is re-planned (a different
    body has a different footprint, so the slots must move with it). Exhausting the ladder is a
    LOUD failure -- it never degrades to "3 of 4 containers score"."""
    if world.recipe.frozen_bindings:
        _site_frozen_units(world)
        return
    probed = unusable = 0
    while probed < _MAX_BODY_SWAPS and unusable < _MAX_UNUSABLE_SWAPS:
        attempt = probed + unusable
        plan_layout(world)
        try:
            site_units(world)
            bad = container_defects(world)
        except UnitBodyUnusable as e:
            bodies, bad, why = {e.key: (e.cat, e.model)}, [e.key], str(e)
            unusable += 1
        else:
            if not bad:
                return
            bodies = {
                k: (world.placed[k][1]["cat"], world.placed[k][1]["model"]) for k in bad
            }
            why = "swallow the probe"
            probed += 1
        authored = unit_assets(world.recipe, world.seed)
        for key in bodies:
            b = authored[key]
            world.body_bump[b] = world.body_bump.get(b, 0) + 1
            lad = body_ladder(
                b[0],
                b[1],
                _alts_of(world.recipe).get(b, ()),
                want_door=_unit_wants_door(world.recipe, key),
            )
            if world.body_bump[b] >= len(lad):
                raise SpecBuildError(
                    f"{world.recipe.task_id}: unit {key} has exhausted {b[0]!r} + alt_cats "
                    f"{_alts_of(world.recipe).get(b, ())} -- all {len(lad)} baked model(s) either "
                    f"have no drivable leaf or swallow a [PutInside]. The row needs another "
                    f"category on its UnitRecipe.alt_cats, not another model."
                )
        print(
            f"[bench] ACCEPTANCE REPAIR {attempt + 1} (probed {probed}/{_MAX_BODY_SWAPS}, "
            f"unusable {unusable}/{_MAX_UNUSABLE_SWAPS}): {sorted(bodies.values())} "
            f"-> next body down the ladder for {sorted(bad)} ({why})"
        )
        _undo_units(world)
    raise SpecBuildError(
        f"{world.recipe.task_id}: no baked body in the authored categories accepts a [PutInside] "
        f"after {probed} probed + {unusable} unusable swap(s) "
        f"(bumps={sorted(world.body_bump.items())})"
    )


def _site_frozen_units(world: World) -> None:
    """Keep frozen task semantics while repairing an incompatible physical model.

    A frozen binding fixes the logical category as well as the initial model. The live acceptance
    probe is more precise than the offline compiler, so it may discover that the selected model
    swallows the task's real payload. In that case only another model of the SAME category is legal:
    a fridge may become another fridge, never a bin or cabinet. The bounded, deterministic ladder
    keeps existing TaskPacks usable without silently changing what their story means.
    """
    authored = unit_assets(world.recipe, world.seed)
    repair_started: float | None = None
    last_bodies: dict[str, tuple[str, str]] = {}
    retry_keys = [
        unit.key for unit in world.recipe.units if unit.binding_source != "scene"
    ]
    repair_reasons: dict[str, str] = {}

    for repair_round in range(_MAX_FROZEN_SAME_CATEGORY_SWAPS + 1):
        bad, why = _try_frozen_candidate(world, retry_keys)
        if not bad:
            _publish_frozen_binding_repairs(
                world, authored, repair_started, repair_reasons
            )
            return

        last_bodies = _current_frozen_bodies(world, bad)
        failure_reason = (
            "container_siting" if "cannot be sited" in why else "putinside_acceptance"
        )
        for key in bad:
            repair_reasons.setdefault(key, failure_reason)
        if repair_round >= _MAX_FROZEN_SAME_CATEGORY_SWAPS:
            break
        if repair_started is None:
            repair_started = time.monotonic()
        _advance_frozen_same_category_bodies(
            world, authored, bad, last_bodies, failure_reason
        )
        retry_keys = list(bad)
        logger.info(
            f"[bench] FROZEN SAME-CATEGORY REPAIR {repair_round + 1}/"
            f"{_MAX_FROZEN_SAME_CATEGORY_SWAPS}: {last_bodies} -> "
            f"{unit_bodies_at(world.recipe, world.seed, 0, world.body_bump, getattr(world, 'frozen_body_ladders', {}))} "
            f"({why})"
        )
        _undo_units(world)

    raise SpecBuildError(
        f"{world.recipe.task_id}: frozen container body {last_bodies} remains physically "
        f"incompatible; no same-category model could be sited and accept [PutInside] after "
        f"{_MAX_FROZEN_SAME_CATEGORY_SWAPS} replacement(s)"
    )


def _plan_frozen_candidate(
    world: World, retry_keys: Sequence[str]
) -> tuple[bool, str]:
    try:
        plan_layout(world)
        return True, ""
    except SpecBuildError as error:
        if not retry_keys:
            raise
        message = str(error)
        layout_failure = "phase=layout" in message or (
            "cannot site " in message
            and "unit(s)" in message
            and "phase=binding" not in message
        )
        fallback_keys = (
            _enable_same_category_scene_fallbacks(world, retry_keys)
            if layout_failure
            else []
        )
        if not fallback_keys:
            return False, f"container body cannot be sited: {error}"
        try:
            plan_layout(world)
        except SpecBuildError as fallback_error:
            for key in fallback_keys:
                _reject_scene_unit_fallback(world, key)
            return False, (
                f"container body cannot be sited: {error}; same-category scene fallback "
                f"also failed: {fallback_error}"
            )
        return True, str(error)


def _try_frozen_candidate(
    world: World, retry_keys: Sequence[str]
) -> tuple[list[str], str]:
    planned, original_siting_error = _plan_frozen_candidate(world, retry_keys)
    if not planned:
        return list(retry_keys), original_siting_error
    try:
        site_units(world)
    except UnitBodyUnusable as error:
        detail = str(error)
        if original_siting_error:
            detail = (
                f"container body cannot be sited: {original_siting_error}; "
                f"same-category scene fallback unusable: {detail}"
            )
        return [error.key], detail
    bad = container_defects(world)
    for key in bad:
        _reject_scene_unit_fallback(world, key)
    why = "swallow the PutInside acceptance probe"
    if bad and original_siting_error:
        why = (
            f"container body cannot be sited: {original_siting_error}; "
            "same-category scene fallback swallowed the PutInside acceptance probe"
        )
    return bad, why


def _current_frozen_bodies(
    world: World, keys: Sequence[str]
) -> dict[str, tuple[str, str]]:
    current = unit_bodies_at(
        world.recipe,
        world.seed,
        level=0,
        bump=world.body_bump,
        repair_ladders=getattr(world, "frozen_body_ladders", {}),
    )
    return {
        key: (
            str((world.placed.get(key) or (None, {}))[1].get("cat", current[key][0])),
            str((world.placed.get(key) or (None, {}))[1].get("model", current[key][1])),
        )
        for key in keys
    }


def _advance_frozen_same_category_bodies(
    world: World,
    authored: dict[str, tuple[str, str]],
    bad: Sequence[str],
    current: dict[str, tuple[str, str]],
    failure_reason: str,
) -> None:
    units = {unit.key: unit for unit in world.recipe.units}
    pending: dict[tuple[str, str], int] = {}
    exhausted: list[str] = []
    for key in bad:
        unit = units[key]
        base = authored[key]
        if unit.binding_source == "scene":
            exhausted.append(f"{key} is bound to an exact scene instance")
            continue
        if current[key][0] != base[0]:
            raise SpecBuildError(
                f"{world.recipe.task_id}: frozen binding repair changed {key} category "
                f"from {base[0]!r} to {current[key][0]!r}"
            )
        ladders = getattr(world, "frozen_body_ladders", {})
        if base not in ladders:
            ladders[base] = same_category_repair_ladder(
                base[0],
                base[1],
                want_door=_unit_wants_door(world.recipe, key),
                reason=failure_reason,
            )
            world.frozen_body_ladders = ladders
        ladder = ladders[base]
        next_index = int(world.body_bump.get(base, 0)) + 1
        if next_index >= len(ladder):
            exhausted.append(f"{key} has no remaining {base[0]!r} model")
            continue
        pending[base] = next_index
    if exhausted:
        raise SpecBuildError(
            f"{world.recipe.task_id}: frozen container body {current} cannot be sited or accept "
            f"[PutInside]; same-category repair unavailable ({'; '.join(exhausted)})"
        )
    world.body_bump.update(pending)


def _publish_frozen_binding_repairs(
    world: World,
    authored: dict[str, tuple[str, str]],
    repair_started: float | None,
    repair_reasons: dict[str, str],
) -> None:
    resolved = {
        key: (str(world.placed[key][1]["cat"]), str(world.placed[key][1]["model"]))
        for key in authored
    }
    repairs = []
    for key, original in authored.items():
        selected = resolved[key]
        if selected == original:
            continue
        if selected[0] != original[0]:
            raise SpecBuildError(
                f"{world.recipe.task_id}: frozen binding repair changed {key} category "
                f"from {original[0]!r} to {selected[0]!r}"
            )
        repairs.append(
            {
                "key": key,
                "reason": repair_reasons.get(
                    key,
                    "container_siting"
                    if key in getattr(world, "scene_unit_fallbacks", {})
                    else "shared_body_binding",
                ),
                "original_category": original[0],
                "original_model": original[1],
                "resolved_category": selected[0],
                "resolved_model": selected[1],
                "same_category": True,
            }
        )
    world.binding_repairs = repairs
    world.binding_repair_seconds = (
        max(0.0, time.monotonic() - repair_started)
        if repairs and repair_started is not None
        else 0.0
    )
    world.scene.resolved_unit_bindings = {
        key: {"category": category, "model": model}
        for key, (category, model) in resolved.items()
    }
    world.scene.container_binding_repairs = list(repairs)
    world.scene.container_binding_repair_seconds = world.binding_repair_seconds
    if repairs:
        logger.info(
            f"[bench] frozen binding repairs={repairs} "
            f"elapsed={world.binding_repair_seconds:.2f}s"
        )


def _undo_units(world: World) -> None:
    """Remove every sited unit -- prims, registrations, keep-outs -- so the next attempt plans in
    the ORIGINAL room, not in one this attempt already filled."""
    for key, pp in list(world.reg_pp.items()):
        # A scene-owned body belongs to the STAGE, not to this attempt: `site_units` records the
        # scene's own prim path here when it reuses a native container, and an unconditional
        # RemovePrim across up to 20+60 acceptance swaps deletes the room's real furniture for the
        # rest of the build. Drop only what this attempt imported.
        ent = world.scene.entities.get(key)
        if not (getattr(ent, "meta", None) or {}).get(SCENE_UNIT_MARK):
            world.stage.RemovePrim(pp)
        world.scene.entities.pop(key, None)
        getattr(world.scene, "_static_aabb", {}).pop(key, None)
        world.placed.pop(key, None)
        world.zspan.pop(
            key, None
        )  # the next attempt sites it elsewhere; its measured span moves
    world.reg_pp, world.layout, world.supports = {}, [], []
    world.used_slots, world.support_stands = [], []
    world.room_slots, world.room_support_stands = {}, {}
    world.front_dir, world.boxes_of, world.outer_of = {}, {}, {}


def _ordered_plan(world: World) -> list[UnitPlan]:
    """The plan back in ROSTER order (the solver searches biggest-body-first)."""
    by_key = {p.key: p for p in world.layout}
    missing = [u.key for u in world.recipe.units if u.key not in by_key]
    if missing:
        raise SpecBuildError(f"the layout solver left unit(s) {missing} unplanned")
    return [by_key[u.key] for u in world.recipe.units]


# =====================================================================================
# stage 3 -- decals. A missing PNG is a SILENT SKIP upstream, so every mount is asserted.
# =====================================================================================
def _slab_corners(corners, normal, depth: float):
    """The same rectangle given THICKNESS: its four corners pushed back along ``-normal``.

    ⛔ WHY A LABEL NEEDS DEPTH. Measured across four renders on ``BOTTLE0817``: a zero-thickness
    quad standing 6 mm and 16 mm proud of a refracting glass bottle does not appear at all, while
    the identical quad at 120 mm renders with its text plainly legible. The prim is carried, active,
    textured, upright and dead centre in frame in every one of those runs -- the only variable is
    how close its plane sits to the glass. A real label is not a plane, and giving it the thickness
    it physically has is both the honest model and the thing the renderer can resolve.
    """
    n = [float(v) for v in normal]
    m = math.sqrt(sum(v * v for v in n)) or 1.0
    n = [v / m * float(depth) for v in n]
    return [(c[0] - n[0], c[1] - n[1], c[2] - n[2]) for c in corners]


def _textured_quad(
    stage, path: str, corners, normal, png: str, *, self_lit: bool = True
) -> None:
    """A double-sided, NON-COLLIDING textured quad. Collision is disabled because
    the init collider pass would otherwise rebuild a decal into a boundingCube that blocks
    the doorway it is mounted on (the bug that made reaching arms clip in from the side)."""
    import behavior_import as bi
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    back = _slab_corners(corners, normal, _DECAL_THICK_M)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in list(corners) + list(back)])
    # Front face first (it carries the artwork), then the four sides and the back, so the label is
    # a solid card rather than a plane: see _slab_corners for what a plane does against glass.
    mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
    mesh.CreateFaceVertexIndicesAttr(
        [0, 1, 2, 3]  # front
        + [7, 6, 5, 4]  # back
        + [4, 5, 1, 0]  # bottom
        + [3, 2, 6, 7]  # top
        + [0, 3, 7, 4]  # left
        + [1, 5, 6, 2]  # right
    )
    mesh.CreateNormalsAttr([Gf.Vec3f(*normal)] * 24)
    mesh.SetNormalsInterpolation("faceVarying")
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    face_uv = [Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]
    st.Set(face_uv * 6)
    bi._bind_label_png(stage, mesh.GetPrim(), path, png, self_lit=self_lit)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(False)


def _publish_label_readability(ent, png: str) -> None:
    import label_visibility as lv

    readability = lv.texture_readability(str(png))
    ent.meta["label_texture"] = str(png)
    ratio = float(readability.get("glyph_height_ratio") or 0.0)
    if ratio > 0.0:
        ent.meta["label_primary_glyph_height_ratio"] = ratio
    line_count = int(readability.get("line_count") or 0)
    if line_count:
        ent.meta["label_primary_line_count"] = line_count
    content_box = readability.get("content_box_uv")
    if content_box is not None:
        ent.meta["label_primary_text_bbox_norm_xyxy"] = content_box


def _wrapped_quad(
    stage,
    path: str,
    center_xy,
    radius: float,
    nrm,
    zc: float,
    h: float,
    half_arc: float,
    png: str,
) -> None:
    """A cylinder-section strip WRAPPING a round body, like a real bottle label.

    Same construction as ``behavior_import._reagent_decal``'s reagent band -- u runs across the arc,
    v up -- and the same winding as ``_textured_quad``, so the artwork reads the same way round on
    either mount. A flat card on a round body is only square-on from ONE direction and goes edge-on
    a few steps to the side; a wrap always presents the part of itself that faces the reader.
    """
    import behavior_import as bi
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    cx, cy = float(center_xy[0]), float(center_xy[1])
    c_ang = math.atan2(float(nrm[1]), float(nrm[0]))
    n_seg = 16
    z_bot, z_top = zc - 0.5 * h, zc + 0.5 * h
    ring, ring_n = [], []
    for i in range(n_seg + 1):
        th = c_ang - half_arc + (2.0 * half_arc) * (i / float(n_seg))
        ox, oy = math.cos(th), math.sin(th)
        ring.append((cx + radius * ox, cy + radius * oy))
        ring_n.append((ox, oy))
    pts = [Gf.Vec3f(float(x), float(y), float(z_bot)) for (x, y) in ring] + [
        Gf.Vec3f(float(x), float(y), float(z_top)) for (x, y) in ring
    ]
    counts, indices, sts, norms = [], [], [], []
    for i in range(n_seg):
        t0, t1 = (n_seg + 1) + i, (n_seg + 1) + i + 1
        counts.append(4)
        indices += [i, i + 1, t1, t0]
        u0, u1 = i / float(n_seg), (i + 1) / float(n_seg)
        sts += [
            Gf.Vec2f(u0, 0.0),
            Gf.Vec2f(u1, 0.0),
            Gf.Vec2f(u1, 1.0),
            Gf.Vec2f(u0, 1.0),
        ]
        n0 = Gf.Vec3f(float(ring_n[i][0]), float(ring_n[i][1]), 0.0)
        n1 = Gf.Vec3f(float(ring_n[i + 1][0]), float(ring_n[i + 1][1]), 0.0)
        norms += [n0, n1, n1, n0]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr(norms)
    mesh.SetNormalsInterpolation("faceVarying")
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(sts)
    bi._bind_label_png(stage, mesh.GetPrim(), path, png)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(False)


def _mount_lid_chamfer(
    stage, root: str, center_xy, nrm, radius: float, z_hi: float, size: float, png: str
) -> None:
    """The 'seen from above on the pass' card, leaning back 45 deg on the body's shoulder.

    It used to lie FLAT (+Z normal). That is fine looking DOWN at the object on a shelf and useless
    the moment it is raised to the face, where the eye is level with it and a horizontal card is
    seen edge-on -- the compressed line of pixels above the bottle in ``BOTTLE0816`` frame 376.
    Leaning it back splits the two viewpoints instead of serving one and breaking the other.
    """
    nx, ny = float(nrm[0]), float(nrm[1])
    ux, uy = -ny, nx  # across, same tangent sense as the band's u
    k = math.sqrt(0.5)  # cos45 == sin45
    s = min(float(size), 1.2 * float(radius))
    # Hinge the card's LOWER edge on the top rim of the band and lean it back over the shoulder, so
    # it rests on the body instead of hovering above the lid.
    cxp = center_xy[0] + nx * (radius - 0.5 * s * k)
    cyp = center_xy[1] + ny * (radius - 0.5 * s * k)
    czp = float(z_hi) + 0.5 * s * k
    vx, vy, vz = -nx * k, -ny * k, k  # up and away from the reader
    corners = [
        (
            cxp - ux * s * 0.5 - vx * s * 0.5,
            cyp - uy * s * 0.5 - vy * s * 0.5,
            czp - vz * s * 0.5,
        ),
        (
            cxp + ux * s * 0.5 - vx * s * 0.5,
            cyp + uy * s * 0.5 - vy * s * 0.5,
            czp - vz * s * 0.5,
        ),
        (
            cxp + ux * s * 0.5 + vx * s * 0.5,
            cyp + uy * s * 0.5 + vy * s * 0.5,
            czp + vz * s * 0.5,
        ),
        (
            cxp - ux * s * 0.5 + vx * s * 0.5,
            cyp - uy * s * 0.5 + vy * s * 0.5,
            czp + vz * s * 0.5,
        ),
    ]
    _textured_quad(
        stage,
        f"{root}/Cue_lid",
        corners,
        (nx * k, ny * k, k),
        _retypeset_for(png, 1.0, "lid"),
    )


def _leaf_frame(world: World, key: str, pp: str):
    """(leaf_path, world->leaf inverse transform) so a door placard RIDES its swinging leaf
    instead of hanging in the door opening once the unit is opened."""
    from pxr import Usd, UsdGeom

    ent = world.scene.entities.get(key)
    lp = ent.leaf_paths[0] if (ent and getattr(ent, "leaf_paths", None)) else None
    root = world.stage.GetPrimAtPath(pp) if lp else None
    if not (root and root.IsValid()):
        return None, None

    def _deinst(pm):
        if pm.IsInstanceable():
            pm.SetInstanceable(False)
        for c in pm.GetChildren():
            _deinst(c)

    _deinst(root)
    lname = lp.rstrip("/").split("/")[-1]
    leaf = next((p for p in Usd.PrimRange(root) if p.GetName() == lname), None)
    if leaf is None:
        return None, None
    inv = (
        UsdGeom.XformCache(Usd.TimeCode.Default())
        .GetLocalToWorldTransform(leaf)
        .GetInverse()
    )
    return str(leaf.GetPath()), inv


def _placard_layout(
    face_width: float,
    *,
    adhesive: bool,
    large_adhesive: bool = False,
) -> tuple[float, float, float, bool]:
    if adhesive:
        width = (
            max(0.20, min(0.85 * face_width, 0.46))
            if large_adhesive
            else max(0.16, min(0.55 * face_width, 0.24))
        )
        return width, width * (320.0 / 900.0), _DECAL_GAP_M, False
    width = max(0.20, min(0.85 * face_width, 0.46))
    return width, width * 0.62, 0.02, True


def _unit_label_mount_style(world: World, key: str) -> str:
    printed = _printed_block(world.spec)
    units = printed.get("units")
    if not isinstance(units, Mapping):
        return ""
    value = units.get(key)
    if not isinstance(value, Mapping):
        return ""
    content = value.get("content")
    if not isinstance(content, Mapping):
        return ""
    args = content.get("args")
    if not isinstance(args, Mapping):
        return ""
    return str(args.get("mount_style") or "")


def mount_placard(world: World, key: str, pp: str, png: str) -> None:
    """Mount a unit's zone placard FLAT on its door, at a standing reader's eye band, and
    record its world pose so the FPV look-at can frame it square-on."""
    import behavior_import as bi
    from pxr import Gf

    if not os.path.exists(png):
        raise SpecBuildError(f"placard PNG missing for {key}: {png}")
    (mnx, mny, mnz), (mxx, mxy, mxz) = _wbbox(world.stage, pp)
    cx, cy = 0.5 * (mnx + mxx), 0.5 * (mny + mxy)
    fd = world.front_dir[key]
    hx, hy = 0.5 * (mxx - mnx), 0.5 * (mxy - mny)
    face_w = 2.0 * (abs(fd[1]) * hx + abs(fd[0]) * hy)  # the door face's own width
    mount_style = _unit_label_mount_style(world, key)
    adhesive = mount_style in {"adhesive", "large_adhesive"}
    sw, sh, gap, self_lit = _placard_layout(
        face_w,
        adhesive=adhesive,
        large_adhesive=mount_style == "large_adhesive",
    )
    zc = float(mnz) + min(0.78, max(0.35, 0.62)) * (float(mxz) - float(mnz))
    reach = abs(fd[0]) * hx + abs(fd[1]) * hy
    px, py = cx + fd[0] * (reach + gap), cy + fd[1] * (reach + gap)
    ux, uy = -fd[1], fd[0]
    hw = sw * 0.5
    corners = [
        (px - ux * hw, py - uy * hw, zc - sh * 0.5),
        (px + ux * hw, py + uy * hw, zc - sh * 0.5),
        (px + ux * hw, py + uy * hw, zc + sh * 0.5),
        (px - ux * hw, py - uy * hw, zc + sh * 0.5),
    ]
    ent = world.scene.entities[key]
    ent.meta["label_face"] = (float(fd[0]), float(fd[1]))
    ent.meta["label_face3"] = (float(fd[0]), float(fd[1]), 0.0)
    ent.meta["label_up3"] = (0.0, 0.0, 1.0)
    ent.meta["label_pos"] = (float(px), float(py), float(zc))
    ent.meta["label_span"] = float(max(sw, sh))
    ent.meta["label_size"] = (float(sw), float(sh))
    _publish_label_readability(ent, png)
    leaf_path, inv = _leaf_frame(world, key, pp)
    nrm = (float(fd[0]), float(fd[1]), 0.0)
    if leaf_path:
        corners = [tuple(inv.Transform(Gf.Vec3d(*c))) for c in corners]
        nrm = tuple(inv.TransformDir(Gf.Vec3d(*nrm)))
    path = f"{leaf_path or '/World'}/Placard_{bi._safe(key)}"
    _textured_quad(world.stage, path, corners, nrm, png, self_lit=self_lit)
    world.decals[key] = path


WALL_LABEL_WIDTH_M: float = 0.64
WALL_LABEL_HEIGHT_M: float = WALL_LABEL_WIDTH_M * (600.0 / 900.0)
WALL_LABEL_EDGE_MARGIN_M: float = 0.06
WALL_LABEL_ANCHOR_GAP_M: float = 0.10
WALL_LABEL_SURFACE_GAP_M: float = 0.006
WALL_LABEL_READER_STANDOFF_M: float = 1.10
FIXTURE_FACE_LABEL_SURFACE_GAP_M: float = 0.001
FIXTURE_FACE_LABEL_INSET: float = 0.94
FIXTURE_FACE_MIN_WIDTH_M: float = 0.06
FIXTURE_FACE_MIN_HEIGHT_M: float = 0.04
WallBox = tuple[Sequence[float], Sequence[float]]


@dataclass(frozen=True)
class WallLabelFrame:
    corners: tuple[tuple[float, float, float], ...]
    normal: tuple[float, float, float]
    center: tuple[float, float, float]
    stand: tuple[float, float]
    size: tuple[float, float]


@dataclass(frozen=True)
class FixtureFaceSurface:
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    tangent: tuple[float, float]
    size: tuple[float, float]
    depth: float


def _wall_label_size(mount: Mapping[str, object]) -> tuple[float, float]:
    label = mount.get("label")
    content = label.get("content") if isinstance(label, Mapping) else None
    args = content.get("args") if isinstance(content, Mapping) else None
    fields = args if isinstance(args, Mapping) else {}
    try:
        width = float(fields.get("wall_width_m") or WALL_LABEL_WIDTH_M)
        height = float(fields.get("wall_height_m") or WALL_LABEL_HEIGHT_M)
    except (TypeError, ValueError) as exc:
        raise SpecBuildError("wall label dimensions must be numeric") from exc
    if not 0.20 <= width <= 1.50 or not 0.20 <= height <= 1.50:
        raise SpecBuildError(
            f"wall label dimensions {(width, height)} must each be within 0.20..1.50 m"
        )
    return width, height


def _wall_face_candidate(
    wall: WallBox,
    anchor_xy: tuple[float, float],
    width: float,
    reader_xy: tuple[float, float] | None = None,
) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    mn, mx = wall
    x0, y0 = float(mn[0]), float(mn[1])
    x1, y1 = float(mx[0]), float(mx[1])
    ax, ay = anchor_xy
    margin = WALL_LABEL_EDGE_MARGIN_M + 0.5 * width
    along_x = (x1 - x0) >= (y1 - y0)
    lo, hi = (x0, x1) if along_x else (y0, y1)
    if hi - lo < 2.0 * margin:
        return None
    along = min(max(ax if along_x else ay, lo + margin), hi - margin)
    mid = 0.5 * ((y0 + y1) if along_x else (x0 + x1))
    anchor_across = ay if along_x else ax
    low, high = (y0, y1) if along_x else (x0, x1)
    reader_across = (
        (reader_xy[1] if along_x else reader_xy[0])
        if reader_xy is not None
        else anchor_across
    )
    positive_face = reader_across >= mid
    face = high if positive_face else low
    sign = 1.0 if positive_face else -1.0
    contact = (along, face) if along_x else (face, along)
    normal = (0.0, sign) if along_x else (sign, 0.0)
    if reader_xy is not None:
        to_reader = (reader_xy[0] - contact[0], reader_xy[1] - contact[1])
        if to_reader[0] * normal[0] + to_reader[1] * normal[1] < 0.0:
            normal = (-normal[0], -normal[1])
    tangent = (-normal[1], normal[0])
    distance = math.hypot(contact[0] - ax, contact[1] - ay)
    return distance, contact, normal, tangent


def _shift_wall_contact(
    contact: tuple[float, float],
    tangent: tuple[float, float],
    wall: WallBox,
    width: float,
    shift: float,
) -> tuple[float, float]:
    margin = WALL_LABEL_EDGE_MARGIN_M + 0.5 * width
    cx = contact[0] + tangent[0] * shift
    cy = contact[1] + tangent[1] * shift
    if abs(tangent[0]) > abs(tangent[1]):
        cx = min(max(cx, float(wall[0][0]) + margin), float(wall[1][0]) - margin)
    else:
        cy = min(max(cy, float(wall[0][1]) + margin), float(wall[1][1]) - margin)
    return cx, cy


def _wall_label_corners(
    center: tuple[float, float, float],
    tangent: tuple[float, float],
    size: tuple[float, float],
) -> tuple[tuple[float, float, float], ...]:
    cx, cy, cz = center
    hw, hh = 0.5 * size[0], 0.5 * size[1]
    return (
        (cx - tangent[0] * hw, cy - tangent[1] * hw, cz - hh),
        (cx + tangent[0] * hw, cy + tangent[1] * hw, cz - hh),
        (cx + tangent[0] * hw, cy + tangent[1] * hw, cz + hh),
        (cx - tangent[0] * hw, cy - tangent[1] * hw, cz + hh),
    )


def _select_wall_face(
    walls: Sequence[WallBox],
    anchor_xy: tuple[float, float],
    size: tuple[float, float],
    reader_xy: tuple[float, float] | None = None,
) -> tuple[
    tuple[float, tuple[float, float], tuple[float, float], tuple[float, float]],
    WallBox,
]:
    width, height = size
    candidates = [
        (candidate, wall)
        for wall in walls
        if float(wall[1][2]) - float(wall[0][2])
        >= height + 2.0 * WALL_LABEL_EDGE_MARGIN_M
        if (
            candidate := _wall_face_candidate(
                wall,
                anchor_xy,
                width,
                reader_xy,
            )
        )
        is not None
    ]
    if not candidates:
        raise SpecBuildError("no measured wall segment can fit the requested wall label")
    return min(candidates, key=lambda row: row[0][0])


def _wall_label_frame(
    walls: Sequence[WallBox],
    *,
    anchor_xy: tuple[float, float],
    anchor_top: float,
    anchor_half: tuple[float, float],
    floor_z: float,
    side: str,
    size: tuple[float, float],
    reader_xy: tuple[float, float] | None = None,
) -> WallLabelFrame:
    width, height = size
    if side not in {"above", "behind", "front", "left", "right"}:
        raise SpecBuildError(f"unsupported wall label side {side!r}")
    (_distance, contact, normal, tangent), wall = _select_wall_face(
        walls,
        anchor_xy,
        size,
        reader_xy,
    )
    shift = 0.0
    if side in {"left", "right"}:
        tangent_half = abs(tangent[0]) * anchor_half[0] + abs(tangent[1]) * anchor_half[1]
        shift = (tangent_half + 0.5 * width + WALL_LABEL_ANCHOR_GAP_M) * (
            -1.0 if side == "left" else 1.0
        )
    cx, cy = _shift_wall_contact(contact, tangent, wall, width, shift)
    cx += normal[0] * WALL_LABEL_SURFACE_GAP_M
    cy += normal[1] * WALL_LABEL_SURFACE_GAP_M
    desired_z = floor_z + 1.40
    if side == "above":
        desired_z = max(
            desired_z,
            anchor_top + WALL_LABEL_ANCHOR_GAP_M + 0.5 * height,
        )
    z0 = float(wall[0][2]) + WALL_LABEL_EDGE_MARGIN_M + 0.5 * height
    z1 = float(wall[1][2]) - WALL_LABEL_EDGE_MARGIN_M - 0.5 * height
    cz = min(max(desired_z, z0), z1)
    center = (cx, cy, cz)
    corners = _wall_label_corners(center, tangent, size)
    reader_offset = WALL_LABEL_READER_STANDOFF_M - WALL_LABEL_SURFACE_GAP_M
    stand = reader_xy or (
        cx + normal[0] * reader_offset,
        cy + normal[1] * reader_offset,
    )
    return WallLabelFrame(corners, (*normal, 0.0), center, stand, size)


def _wall_label_anchor(world: World, key: str) -> tuple[object, float, tuple[float, float]]:
    entity = world.scene.entities.get(key)
    if entity is None:
        raise SpecBuildError(f"wall label anchor {key!r} is not a registered scene entity")
    raw_top = getattr(entity, "top_z", None)
    top = float(raw_top if raw_top is not None else getattr(entity, "z", 0.0))
    raw_half = getattr(entity, "foot_half", None)
    if raw_half is None:
        raw_half = getattr(entity, "half", (0.0, 0.0))
    half = (float(raw_half[0]), float(raw_half[1]))
    return entity, top, half


def _fixture_face_wide_bottom(
    projected_points: Sequence[tuple[float, float]],
    width: float,
    z_min: float,
    z_max: float,
) -> float:
    """Return the bottom of the broad rectangular face above a narrow pointer."""
    height = z_max - z_min
    tolerance = max(0.0005, 0.01 * height)
    levels: list[tuple[float, list[float]]] = []
    for z, tangent in sorted(projected_points):
        if not levels or z - levels[-1][0] > tolerance:
            levels.append((z, [tangent]))
        else:
            levels[-1][1].append(tangent)
    for z, tangents in levels:
        if max(tangents) - min(tangents) >= 0.65 * width:
            return z
    return z_min


def _fixture_face_surface(
    points: Sequence[Sequence[float]],
    reader_xy: tuple[float, float],
) -> FixtureFaceSurface | None:
    xyz = [tuple(float(value) for value in point[:3]) for point in points]
    if len(xyz) < 3:
        return None
    mean_x = sum(point[0] for point in xyz) / len(xyz)
    mean_y = sum(point[1] for point in xyz) / len(xyz)
    variance_x = sum((point[0] - mean_x) ** 2 for point in xyz) / len(xyz)
    variance_y = sum((point[1] - mean_y) ** 2 for point in xyz) / len(xyz)
    covariance = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in xyz
    ) / len(xyz)
    spread = math.hypot(variance_x - variance_y, 2.0 * covariance)
    if 0.5 * (variance_x + variance_y + spread) <= 1e-10:
        return None
    angle = 0.5 * math.atan2(2.0 * covariance, variance_x - variance_y)
    wide_axis = (math.cos(angle), math.sin(angle))
    normal = (-wide_axis[1], wide_axis[0])
    midpoint = (mean_x, mean_y)
    if (
        (reader_xy[0] - midpoint[0]) * normal[0]
        + (reader_xy[1] - midpoint[1]) * normal[1]
        < 0.0
    ):
        normal = (-normal[0], -normal[1])
    tangent = (-normal[1], normal[0])
    tangent_values = [
        point[0] * tangent[0] + point[1] * tangent[1] for point in xyz
    ]
    normal_values = [
        point[0] * normal[0] + point[1] * normal[1] for point in xyz
    ]
    width = max(tangent_values) - min(tangent_values)
    depth = max(normal_values) - min(normal_values)
    z_min = min(point[2] for point in xyz)
    z_max = max(point[2] for point in xyz)
    face_bottom = _fixture_face_wide_bottom(
        [(point[2], tangent) for point, tangent in zip(xyz, tangent_values)],
        width,
        z_min,
        z_max,
    )
    height = z_max - face_bottom
    if width < FIXTURE_FACE_MIN_WIDTH_M or height < FIXTURE_FACE_MIN_HEIGHT_M:
        return None
    if depth > 0.40 * min(width, height):
        return None
    tangent_center = 0.5 * (min(tangent_values) + max(tangent_values))
    normal_front = max(normal_values) + FIXTURE_FACE_LABEL_SURFACE_GAP_M
    center = (
        tangent[0] * tangent_center + normal[0] * normal_front,
        tangent[1] * tangent_center + normal[1] * normal_front,
        0.5 * (face_bottom + z_max),
    )
    return FixtureFaceSurface(
        center=center,
        normal=(normal[0], normal[1], 0.0),
        tangent=tangent,
        size=(width, height),
        depth=depth,
    )


def select_fixture_face_surface(
    point_groups: Sequence[Sequence[Sequence[float]]],
    reader_xy: tuple[float, float],
) -> FixtureFaceSurface | None:
    """Select the broad, thin, upright collision child that forms a sign face."""
    candidates = [
        surface
        for points in point_groups
        if (surface := _fixture_face_surface(points, reader_xy)) is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda surface: (
            surface.size[0] * surface.size[1] / max(surface.depth, 0.002),
            surface.center[2],
        ),
    )


def fixture_face_label_frame(
    point_groups: Sequence[Sequence[Sequence[float]]],
    reader_xy: tuple[float, float],
    aspect: float,
) -> WallLabelFrame | None:
    """Fit a decal directly over the native upper sign board, if one is present."""
    surface = select_fixture_face_surface(point_groups, reader_xy)
    if surface is None:
        return None
    max_width = FIXTURE_FACE_LABEL_INSET * surface.size[0]
    max_height = FIXTURE_FACE_LABEL_INSET * surface.size[1]
    width = min(max_width, max_height * max(float(aspect), 1e-6))
    height = min(max_height, width / max(float(aspect), 1e-6))
    center = surface.center
    normal = surface.normal
    tangent = surface.tangent
    corners = (
        (
            center[0] - tangent[0] * width * 0.5,
            center[1] - tangent[1] * width * 0.5,
            center[2] - height * 0.5,
        ),
        (
            center[0] + tangent[0] * width * 0.5,
            center[1] + tangent[1] * width * 0.5,
            center[2] - height * 0.5,
        ),
        (
            center[0] + tangent[0] * width * 0.5,
            center[1] + tangent[1] * width * 0.5,
            center[2] + height * 0.5,
        ),
        (
            center[0] - tangent[0] * width * 0.5,
            center[1] - tangent[1] * width * 0.5,
            center[2] + height * 0.5,
        ),
    )
    return WallLabelFrame(
        corners,
        normal,
        center,
        reader_xy,
        (width, height),
    )


def _fixture_collision_point_groups(stage, prim_path: str) -> list[list[tuple[float, float, float]]]:
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not (prim and prim.IsValid()):
        return []
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    groups = []
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if child.GetTypeName() != "Mesh" or "/collisions/" not in str(child.GetPath()):
            continue
        points = UsdGeom.Mesh(child).GetPointsAttr().Get() or []
        transform = xform_cache.GetLocalToWorldTransform(child)
        groups.append(
            [
                tuple(
                    float(value)
                    for value in transform.Transform(
                        Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                    )
                )
                for point in points
            ]
        )
    return [group for group in groups if group]


def _fixture_face_label_frame_from_bounds(
    minimum,
    maximum,
    reader_xy: tuple[float, float],
    aspect: float,
) -> WallLabelFrame:
    cx = 0.5 * (float(minimum[0]) + float(maximum[0]))
    cy = 0.5 * (float(minimum[1]) + float(maximum[1]))
    cz = 0.5 * (float(minimum[2]) + float(maximum[2]))
    hx = 0.5 * (float(maximum[0]) - float(minimum[0]))
    hy = 0.5 * (float(maximum[1]) - float(minimum[1]))
    dz = float(maximum[2]) - float(minimum[2])
    dx, dy = float(reader_xy[0]) - cx, float(reader_xy[1]) - cy
    norm = math.hypot(dx, dy)
    normal = (dx / norm, dy / norm) if norm > 1e-6 else (0.0, -1.0)
    tangent = (-normal[1], normal[0])
    face_width = 2.0 * (abs(tangent[0]) * hx + abs(tangent[1]) * hy)
    max_width = max(0.01, 0.88 * face_width)
    max_height = max(0.01, 0.74 * dz)
    width = min(max_width, max_height * max(float(aspect), 1e-6))
    height = min(max_height, width / max(float(aspect), 1e-6))
    reach = abs(normal[0]) * hx + abs(normal[1]) * hy
    center = (
        cx + normal[0] * (reach + WALL_LABEL_SURFACE_GAP_M),
        cy + normal[1] * (reach + WALL_LABEL_SURFACE_GAP_M),
        cz,
    )
    return WallLabelFrame(
        _wall_label_corners(center, tangent, (width, height)),
        (normal[0], normal[1], 0.0),
        center,
        reader_xy,
        (width, height),
    )


def _mount_fixture_face_label(
    world: World, label_id: str, anchor: str, png: str, entity
) -> None:
    import behavior_import as bi
    import label_generator as ml

    stand = entity.meta.get("front_stand")
    if not isinstance(stand, (list, tuple)) or len(stand) < 2:
        raise SpecBuildError(f"fixture-face label anchor {anchor!r} has no reader stand")
    reader_xy = (float(stand[0]), float(stand[1]))
    aspect = float(ml.document_texture_aspect(png) or 1.4)
    frame = fixture_face_label_frame(
        _fixture_collision_point_groups(world.stage, entity.prim_path),
        reader_xy,
        aspect,
    )
    if frame is None:
        bounds = bi._collision_bbox(world.stage, entity.prim_path) or _wbbox(
            world.stage, entity.prim_path
        )
        frame = _fixture_face_label_frame_from_bounds(
            bounds[0], bounds[1], reader_xy, aspect
        )
    texture = _retypeset_for(
        png,
        frame.size[0] / max(frame.size[1], 1e-6),
        "fixture_face",
    )
    path = f"/World/FixtureLabel_{_slug(label_id, 48)}"
    _textured_quad(
        world.stage,
        path,
        frame.corners,
        frame.normal,
        texture,
        self_lit=False,
    )
    entity.meta.update(
        label_face=frame.normal[:2],
        label_face3=frame.normal,
        label_pos=frame.center,
        label_read_in_place=True,
        label_size=frame.size,
        label_span=max(frame.size),
        label_up3=(0.0, 0.0, 1.0),
    )
    _publish_label_readability(entity, texture)
    world.decals[f"wall:{label_id}"] = path


def mount_wall_labels(world: World, pngs: Mapping[str, str]) -> None:
    """Mount explicit documents on measured scene walls beside their declared anchors."""
    printed = _printed_block(world.spec)
    for label_id, mount in _printed_wall_entries(printed).items():
        anchor = str(mount["on"])
        entity, anchor_top, anchor_half = _wall_label_anchor(world, anchor)
        if entity.meta.get("label_texture"):
            raise SpecBuildError(
                f"wall label anchor {anchor!r} already carries another printed cue"
            )
        png_key = f"wall:{label_id}"
        png = pngs.get(png_key)
        if not png or not os.path.exists(png):
            raise SpecBuildError(f"wall label PNG missing for {label_id}: {png}")
        declared_stand = entity.meta.get("front_stand")
        reader_xy = (
            (float(declared_stand[0]), float(declared_stand[1]))
            if isinstance(declared_stand, (list, tuple))
            and len(declared_stand) >= 2
            else None
        )
        side = str(mount.get("side") or "above")
        anchor_path = str(getattr(entity, "prim_path", ""))
        anchor_asset = _support_asset(world, anchor_path) if anchor_path else None
        if side == "front" and anchor_asset is not None and anchor_asset[0] == "grocery_sign":
            if reader_xy is None:
                raise SpecBuildError(
                    f"fixture-face label anchor {anchor!r} has no reader stand"
                )
            _mount_fixture_face_label(world, label_id, anchor, png, entity)
            continue
        frame = _wall_label_frame(
            world.walls,
            anchor_xy=(float(entity.xy[0]), float(entity.xy[1])),
            anchor_top=anchor_top,
            anchor_half=anchor_half,
            floor_z=float(world.scene.floor_z),
            side=side,
            size=_wall_label_size(mount),
            reader_xy=reader_xy,
        )
        path = f"/World/WallLabel_{_slug(label_id, 48)}"
        _textured_quad(
            world.stage,
            path,
            frame.corners,
            frame.normal,
            png,
            self_lit=False,
        )
        entity.meta.update(
            label_face=frame.normal[:2],
            label_face3=frame.normal,
            label_pos=frame.center,
            label_read_in_place=True,
            label_size=frame.size,
            label_span=max(frame.size),
            label_up3=(0.0, 0.0, 1.0),
            front_stand=frame.stand,
        )
        _publish_label_readability(entity, png)
        world.decals[png_key] = path


_PANEL_CACHE_V = 3

# How far the decal stands off the wall it is stuck to. A real label is flush, but this body is
# REFRACTING GLASS: measured on BOTTLE0817g, a card 2-4 mm proud reads cleanly when the bottle is
# looked down on at rest and is swallowed by the glass when the same card is viewed at 35 deg with
# the bottle raised. Overridable so the gap can be measured rather than argued about.
_DECAL_GAP_M = float(os.getenv("ISAAC_DECAL_GAP_MM", "4")) / 1000.0
# A printed label is about this thick once it is stuck on. See ``_slab_corners``.
_DECAL_THICK_M = float(os.getenv("ISAAC_DECAL_THICK_MM", "1.5")) / 1000.0


def _panel_cache_path(cat: str, model: str) -> str:
    import behavior_import as bi

    return os.path.join(bi.CACHE_SPEC, cat, f"{model}.panel.json")


def _surface_samples(stage, spacing: float = 0.004):
    """Points ON a baked body's visible surface, in its own frame.

    Vertices alone are not enough: these meshes are low-poly (this bottle is 526 points over 281 mm)
    and leave whole height bands with no vertex at all, so a profile taken off them reads as holes
    in the body. Sampling the face EDGES fills them, and for a convex planar face the furthest point
    from the object's axis always lies on its boundary, so edges are also sufficient.

    ⚠️ The samples are spaced by LENGTH, not a fixed count per edge. A fixed count is fine on a
    bottle (many small facets) and fails completely on a BOX: four vertical edges at six samples
    each populate six height slices out of twenty-four, the other eighteen read as zero girth, and
    the band search then stops at the first "empty" slice. Measured on ``box_of_cereal/hcwbzw``,
    that produced a 33 x 20 mm label stuck to the bottom 22 mm of a 273 mm box.

    Collision hulls are skipped and duplicate copies are dropped: a BEHAVIOR bake carries the same
    geometry twice (``/<model>/...`` and ``/visuals/...``) plus dozens of convex hulls, and the
    hulls -- 32 of them here, each denser than the shell -- would otherwise outvote the real wall.
    """
    import numpy as np
    from pxr import Usd, UsdGeom

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    seen, out = set(), []
    step = max(1e-4, float(spacing))
    for prim in Usd.PrimRange(
        stage.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    ):
        path = str(prim.GetPath()).lower()
        if "collision" in path or "collider" in path:
            continue
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        index = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not index:
            continue
        m = xc.GetLocalToWorldTransform(prim)
        V = np.array([tuple(m.Transform(q)) for q in pts], dtype=np.float64)
        sig = (len(V), tuple(np.round(V.min(0), 5)), tuple(np.round(V.max(0), 5)))
        if sig in seen:
            continue
        seen.add(sig)
        chunks, k = [V], 0
        for c in counts:
            face = list(index[k : k + c])
            k += c
            for i in range(c):
                a, b = V[face[i]], V[face[(i + 1) % c]]
                n = int(
                    min(128, max(2, math.ceil(float(np.linalg.norm(b - a)) / step)))
                )
                t = np.linspace(0.0, 1.0, n)[:, None]
                chunks.append(a[None, :] * (1.0 - t) + b[None, :] * t)
        out.append(np.vstack(chunks))
    if not out:
        raise ValueError("no visible mesh")
    return np.vstack(out)


def _slice_profile(points, blo, bb) -> dict:
    """Radius-vs-height profile of a body's surface, in its LOCAL frame.

    Returns the band of height a label can sit on -- the run from the BASE upwards over which the
    girth holds up -- plus how ROUND the body is there. Both are needed to stick a label on it: the
    band says where there is surface, the roundness says whether the label has to curve to stay on
    it (a flat card on a round bottle goes edge-on; a curved one on a box lifts off its faces).

    Scanned from the base rather than "the longest band anywhere", because a tapered body has no
    constant-girth section at all and the longest run then lands on the SHOULDER -- measured on this
    bottle, which is a continuous taper: the naive rule chose 129-211 mm, its narrowing neck.
    """
    import numpy as np

    P = np.asarray(points, dtype=np.float64)
    cx, cy = float(blo[0]), float(blo[1])
    z0 = float(blo[2]) - 0.5 * float(bb[2])
    z1 = float(blo[2]) + 0.5 * float(bb[2])
    rad = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    n_slices = 24
    edges = np.linspace(z0, z1, n_slices + 1)
    step = (z1 - z0) / n_slices
    mid = 0.5 * (edges[:-1] + edges[1:])
    girth = np.array(
        [
            float(rad[np.abs(P[:, 2] - m) <= 0.75 * step].max())
            if np.any(np.abs(P[:, 2] - m) <= 0.75 * step)
            else 0.0
            for m in mid
        ]
    )
    # 3-slice median: a moulded body has ridges and seams, and one narrow slice between two full
    # ones is a feature of the moulding, not the end of the wall.
    smooth = np.array(
        [
            float(np.median(girth[max(0, s - 1) : min(n_slices, s + 2)]))
            for s in range(n_slices)
        ]
    )
    fat = float(smooth.max())
    if fat <= 1e-6:
        raise ValueError("no measurable girth")
    # A slice with no samples in it is MISSING DATA, not a waist. Treating it as zero girth is what
    # let one unsampled slice end the band two slices above the floor (see ``_surface_samples``).
    top = n_slices
    for s in range(n_slices):
        if girth[s] > 1e-6 and smooth[s] < 0.80 * fat:
            top = s
            break
    top = max(top, 2)
    lo, hi = float(edges[0]), float(edges[top])
    # HOW FAR THE WALL REACHES IN A GIVEN DIRECTION, per height slice: the shape's SUPPORT FUNCTION,
    # h(n) = max over surface points of dot(p - axis, n). That is exactly the quantity a flat card
    # mounted facing ``n`` needs, and it is the one thing neither the bounding box nor a radius can
    # give. Three measured failures on this bottle, one per wrong answer:
    #
    #   * half the bounding box (37.6 mm) -- the box is 75.2 mm deep while the mesh spans 78.0, so
    #     the card lands inside the glass;
    #   * the max RADIUS in the card's sector (48.8 mm) -- radius is dominated by the CORNERS of a
    #     rounded-rectangle section, so the card is pushed 7 mm past the face and hangs in the air
    #     beside the bottle instead of sticking to it;
    #   * one figure for the whole body -- its base flares to 57 mm while the wall at label height
    #     is ~39 mm.
    #
    # The support function has none of those failure modes: it is directional, so a corner 45 deg
    # away cannot inflate it, and it is per-slice, so a flared base cannot either.
    dirs = np.array(
        [
            [
                math.cos(-math.pi + (k + 0.5) * math.pi / 8.0),
                math.sin(-math.pi + (k + 0.5) * math.pi / 8.0),
            ]
            for k in range(16)
        ]
    )
    slot = np.clip(np.digitize(P[:, 2], edges) - 1, 0, n_slices - 1)
    rel = np.stack([P[:, 0] - cx, P[:, 1] - cy], axis=1)
    proj = (
        rel @ dirs.T
    )  # (n_points, 16) how far each point reaches along each direction
    grid = [
        [
            float(proj[slot == s, k].max()) if np.any(slot == s) else 0.0
            for k in range(16)
        ]
        for s in range(n_slices)
    ]
    seen = [v for v in (max(col) for col in zip(*grid[:top])) if v > 1e-6]
    roundness = (min(seen) / max(seen)) if len(seen) >= 12 and max(seen) > 1e-6 else 0.0
    return {
        "radius": float(np.median(seen)) if seen else fat,
        "grid": grid,
        "grid_z0": float(edges[0]),
        "grid_dz": float(step),
        "z_lo": lo,
        "z_hi": hi,
        "roundness": float(roundness),
    }


def _body_panel(cat: str, model: str, bb, blo) -> dict:
    """Where on this body a label can be stuck, measured from the baked mesh (cached per model).

    ⛔ WHY THIS IS MEASURED AND NOT A FRACTION OF THE BBOX. The old rule took the band's height as
    ``0.60 * bbox_z`` centred a tenth above the middle. On ``lemon_juice_bottle/qsdqik`` (87 x 75 x
    281 mm) that is a 168 mm strip running from 84 mm to 253 mm above the base, while the body's
    girth holds only to 211 mm -- so the label's top third hung in the air alongside the neck. In
    ``BOTTLE0816`` frame 376 that is the grey card that reads as floating behind the bottle rather
    than stuck to it. A bbox says how big a thing is, never where its surface is.

    Falls back to the bbox's lower-middle when the mesh cannot be read: still clear of the neck of
    anything bottle-shaped, which is the case the fraction got wrong.
    """
    import json

    path = _panel_cache_path(cat, model)
    try:
        with open(path) as f:
            got = json.load(f)
        # ⚠️ The MEANING of the grid is versioned, not just its presence. It held max RADIUS per
        # sector before and holds the SUPPORT FUNCTION now; a cache written by the old code reads
        # cleanly and mounts every decal 7 mm off the object, silently. Bump on any change to what
        # ``_slice_profile`` returns.
        if int(got.get("v") or 0) == _PANEL_CACHE_V:
            return got
    except (OSError, ValueError):
        pass
    out = None
    try:
        import behavior_import as bi
        from pxr import Usd

        stage = Usd.Stage.Open(bi.baked_usd(f"{cat}/{model}"))
        out = _slice_profile(_surface_samples(stage), blo, bb)
    except Exception as exc:  # noqa: BLE001 -- a build must never die on a measurement
        print(f"[bench] body panel unmeasurable for {cat}/{model}: {exc}")
    if out is None:
        out = {
            "radius": 0.5 * min(float(bb[0]), float(bb[1])),
            "reach16": [],
            "z_lo": float(blo[2]) - 0.40 * float(bb[2]),
            "z_hi": float(blo[2]) + 0.05 * float(bb[2]),
            "roundness": 0.0,
        }
    out["v"] = _PANEL_CACHE_V
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, sort_keys=True)
    except OSError:
        pass
    return out


# The band wraps this far each way from the face it presents to the reader. It is
# ``label_visibility.MAX_OFF_NORMAL_DEG`` -- the angle this codebase already calls the edge of
# legibility -- so the WHOLE strip is inside the readable fan when viewed head-on, and going
# off-axis rotates fresh label into the fan instead of shearing what is in it.
def _decal_half_arc_rad() -> float:
    import label_visibility as lv

    return math.radians(lv.MAX_OFF_NORMAL_DEG)


def _wall_reach(panel: dict, nrm, bbox_reach: float, z_lo: float, z_hi: float) -> float:
    """How far the body's wall stands out TOWARDS ``nrm``, over the height band ``z_lo..z_hi``.

    Reads ``_slice_profile``'s support grid; falls back to (and never goes below) the bounding-box
    half-extent, so a body that could not be measured still gets a card outside its box.

    Only the two directions either side of ``nrm`` are consulted -- NOT a fan of neighbours. A flat
    card facing ``nrm`` is a PLANE: every point on it is the same distance along ``nrm``, so the
    support in that one direction is the whole answer. Widening the fan re-introduces exactly the
    bug this replaced, because a corner 22 deg off pushes the plane out past the face it is meant
    to lie on and the card ends up hanging in the air beside the object.
    """
    grid = panel.get("grid") or []
    dz = float(panel.get("grid_dz") or 0.0)
    z0 = float(panel.get("grid_z0") or 0.0)
    if not grid or dz <= 0.0:
        return float(bbox_reach)
    s_lo = max(0, min(len(grid) - 1, int(math.floor((z_lo - z0) / dz))))
    s_hi = max(0, min(len(grid) - 1, int(math.ceil((z_hi - z0) / dz))))
    # Sector k covers the direction at -pi + (k + 0.5) * pi/8; take the two whose centres bracket
    # ``nrm``, since a 22.5 deg quantisation would otherwise under-read the wall by up to 2%.
    t = (math.atan2(float(nrm[1]), float(nrm[0])) + math.pi) / (
        2.0 * math.pi
    ) * 16.0 - 0.5
    k0, k1 = int(math.floor(t)) % 16, int(math.ceil(t)) % 16
    near = [
        grid[s][k]
        for s in range(s_lo, s_hi + 1)
        for k in (k0, k1)
        if len(grid[s]) == 16
    ]
    return max([float(bbox_reach)] + near)


def _retypeset_for(png: str, aspect: float, tag: str) -> str:
    """The same label, re-set for a mount of this width/height ratio. Falls back to ``png``.

    ⚠️ The RATIO is in the file name. ``label_generator`` never overwrites, and the label cache is shared
    across every row and every rebuild, so a name that did not carry the shape would freeze the
    first body's proportions onto every later one -- the same trap ``_slug`` documents for the
    printed string, and it would be invisible: the label is there, it is just set for another
    object's geometry.
    """
    import label_generator as ml

    stem = png[:-4] if png.lower().endswith(".png") else png
    out = f"{stem}__{tag}{round(max(0.0, aspect) * 100):04d}.png"
    got = ml.retypeset(png, out, aspect)
    return str(got) if got is not None else png


def _top_item_decal_layout(
    png: str, face_w: float, face_h: float
) -> tuple[float, float, str, bool]:
    import label_generator as ml

    document_aspect = ml.document_texture_aspect(png)
    if document_aspect is None:
        width, height = 0.82 * face_w, 0.62 * face_h
        reshaped = _retypeset_for(png, width / max(height, 1e-6), "top")
        return width, height, reshaped, True
    max_width, max_height = 0.90 * face_w, 0.90 * face_h
    width = min(max_width, max_height * document_aspect)
    height = width / max(document_aspect, 1e-6)
    return width, height, png, False


def choose_item_decal_mount(
    bbox_size,
    roundness: float,
    panel_height_m: float | None = None,
    *,
    explicit_face: str | None = None,
) -> str:
    """Choose the one label face a resting object exposes most clearly to a wearer.

    An explicit authored face wins. Otherwise flat, shallow, non-round bodies are read from above
    while resting on a support, and tall or round bodies keep the side/wrap mount that remains
    readable when held upright.
    """
    if explicit_face == "top":
        return "top"
    if explicit_face == "front":
        return "side"
    if explicit_face is not None:
        raise SpecBuildError(f"unsupported item decal mount face {explicit_face!r}")
    if not isinstance(bbox_size, (list, tuple)) or len(bbox_size) < 3:
        return "side"
    bx, by, bz = (abs(float(value)) for value in bbox_size[:3])
    short = min(bx, by)
    if (
        panel_height_m is not None
        and float(panel_height_m) < DECAL_MIN_SPAN_M
        and short >= DECAL_MIN_SPAN_M
    ):
        return "top"
    if short >= DECAL_MIN_SPAN_M and bz <= 0.75 * short and roundness < 0.85:
        return "top"
    return "side"


def _mount_top_item_decal(stage, root: str, png: str, blo, bb, nrm) -> dict:
    cx, cy = float(blo[0]), float(blo[1])
    hx, hy = 0.5 * float(bb[0]), 0.5 * float(bb[1])
    ux, uy = -float(nrm[1]), float(nrm[0])
    vx, vy = -float(nrm[0]), -float(nrm[1])
    face_w = 2.0 * (abs(ux) * hx + abs(uy) * hy)
    face_h = 2.0 * (abs(vx) * hx + abs(vy) * hy)
    width, height, top_png, self_lit = _top_item_decal_layout(
        png, face_w, face_h
    )
    z = float(blo[2]) + 0.5 * float(bb[2]) + _DECAL_GAP_M
    corners = [
        (
            cx - ux * width * 0.5 - vx * height * 0.5,
            cy - uy * width * 0.5 - vy * height * 0.5,
            z,
        ),
        (
            cx + ux * width * 0.5 - vx * height * 0.5,
            cy + uy * width * 0.5 - vy * height * 0.5,
            z,
        ),
        (
            cx + ux * width * 0.5 + vx * height * 0.5,
            cy + uy * width * 0.5 + vy * height * 0.5,
            z,
        ),
        (
            cx - ux * width * 0.5 + vx * height * 0.5,
            cy - uy * width * 0.5 + vy * height * 0.5,
            z,
        ),
    ]
    path = f"{root}/Cue_top"
    _textured_quad(
        stage,
        path,
        corners,
        (0.0, 0.0, 1.0),
        top_png,
        self_lit=self_lit,
    )
    return {
        "path": path,
        "local": (cx, cy, z),
        "size": (width, height),
        "span": max(width, height),
        "texture": top_png,
    }


def _rest_orientation(ent) -> tuple[float, float, float, float]:
    value = (getattr(ent, "meta", None) or {}).get("rest_orientation")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(component) for component in value)
    return (1.0, 0.0, 0.0, 0.0)


def _label_world_xy(ent, local_x: float, local_y: float) -> tuple[float, float]:
    yaw = math.radians(float((getattr(ent, "meta", None) or {}).get("rest_yaw_deg") or 0.0))
    x = math.cos(yaw) * local_x - math.sin(yaw) * local_y
    y = math.sin(yaw) * local_x + math.cos(yaw) * local_y
    return float(ent.xy[0]) + x, float(ent.xy[1]) + y


def _park_item_decal_anchor(world: World, root: str, ent) -> None:
    import fpv_scene

    fpv_scene.set_world_pose(
        world.stage.GetPrimAtPath(root),
        (float(ent.xy[0]), float(ent.xy[1]), float(ent.z)),
        _rest_orientation(ent),
    )


def mount_item_decal(
    world: World,
    item: ItemRecipe,
    png: str,
    yaw_deg: float,
    *,
    park_anchor: bool = True,
    mount_face: str | None = None,
) -> None:
    """Mount one cue on the face the wearer can actually inspect.

    Shallow boxes use their top face and are read in place; tall or round bodies use the measured
    side/wrap wall and can be raised for inspection. The cue is authored in the graspable anchor's
    local frame under ``/World/Obj_<key>``, so it rides the object when carried.

    ⛔ WHY THE FLAT BAND WAS REPLACED, measured on ``BOTTLE0816`` frame 376 (every number below is
    the code's own arithmetic, and the render agrees with it to 3 px):

    * the band was ``0.62*girth`` wide by ``0.60*height`` tall -- two independent fractions of two
      unrelated dimensions, so on a bottle it came out 46.6 x 168.4 mm, a 1:3.6 strip. The artwork
      is a 620x760 canvas stretched across the whole quad, so every glyph was compressed to 34% of
      its width: 5.9 px wide and 23 px tall on screen. Legible artwork, illegible mount.
    * half that strip was above the bottle's straight wall, hanging in the air by the neck
      (see ``_body_panel``).
    * the lid card lay FLAT (+Z). Raised to the face it is seen edge-on and reads as a line.

    So the shape now comes from the body (``_body_panel``) and the ARTWORK is re-set for that shape
    (``_retypeset_for``) instead of being stretched onto it -- the layout engine re-breaks the lines
    to fill whatever box it is handed, which is what turns one squeezed line into two big ones.
    """
    import behavior_import as bi

    if not os.path.exists(png):
        raise SpecBuildError(f"cue decal PNG missing for {item.key}: {png}")
    root = f"/World/Obj_{bi._safe(item.key)}"
    world.stage.DefinePrim(root, "Xform")
    cat, model = staged_body(world, item)
    sp = bi.read_spec(f"{cat}/{model}")
    bb = staged_bbox(world, item)
    scale_xyz = _scale_xyz(item.scale)
    blo = tuple(
        float(value) * scale_xyz[axis]
        for axis, value in enumerate(
            sp.get("base_link_offset") or [0.0, 0.0, 0.0]
        )
    )
    panel = _body_panel(cat, model, bb, blo)
    ent = world.scene.entities[item.key]
    look_held_distance = item_decal_look_held_distance(world.spec, item.key)
    if look_held_distance is not None:
        ent.meta["look_held_distance_m"] = look_held_distance
    look_target_glyph_px = item_decal_look_target_glyph_px(world.spec, item.key)
    if look_target_glyph_px is not None:
        ent.meta["look_target_glyph_px"] = look_target_glyph_px
    rest_yaw = float(ent.meta.get("rest_yaw_deg") or 0.0)
    a = math.radians(float(yaw_deg) - rest_yaw)
    nrm = (
        -math.sin(a),
        math.cos(a),
    )  # object centre -> the reader, in the anchor frame
    world_a = math.radians(float(yaw_deg))
    world_nrm = (-math.sin(world_a), math.cos(world_a))
    cx, cy = float(blo[0]), float(blo[1])
    hx, hy = 0.5 * float(bb[0]), 0.5 * float(bb[1])
    if (
        choose_item_decal_mount(
            bb,
            float(panel["roundness"]),
            float(panel["z_hi"]) - float(panel["z_lo"]),
            explicit_face=mount_face,
        )
        == "top"
    ):
        top = _mount_top_item_decal(world.stage, root, png, blo, bb, nrm)
        ent.meta["label_face"] = (0.0, 0.0)
        ent.meta["label_face3"] = (0.0, 0.0, 1.0)
        ent.meta["label_face_local"] = (0.0, 0.0, 1.0)
        ent.meta["label_local"] = top["local"]
        ent.meta["label_span"] = float(top["span"])
        ent.meta["label_size"] = top["size"]
        ent.meta["label_up3"] = (
            -float(world_nrm[0]),
            -float(world_nrm[1]),
            0.0,
        )
        ent.meta["label_up_local"] = (-float(nrm[0]), -float(nrm[1]), 0.0)
        ent.meta["label_curved"] = False
        ent.meta["label_mount"] = "top"
        ent.meta["label_read_in_place"] = True
        _publish_label_readability(ent, str(top["texture"]))
        if ent.xy is not None:
            label_x, label_y = _label_world_xy(
                ent, float(top["local"][0]), float(top["local"][1])
            )
            ent.meta["label_pos"] = (
                label_x,
                label_y,
                float(ent.z) + float(top["local"][2]),
            )
        print(
            f"[bench] item decal {item.key}: top {top['size'][0] * 1000:.0f}x"
            f"{top['size'][1] * 1000:.0f}mm"
        )
        world.decals[item.key] = str(top["path"])
        if park_anchor:
            _park_item_decal_anchor(world, root, ent)
        return
    ux, uy = -nrm[1], nrm[0]  # across the face the reader sees
    half_arc = _decal_half_arc_rad()
    curved = float(panel["roundness"]) >= 0.85
    # How far out the patch sits, and how wide it may be. Curved: the arc a reader takes in at once
    # (its own developed length). Flat: the old 0.62 margin, but across the face they are LOOKING at
    # -- it used to take ``min(bbox_x, bbox_y)`` whichever way the card pointed, i.e. the depth of
    # the object as often as its width. On this bottle that is 46.6 mm instead of 54.3.
    bbox_reach = abs(nrm[0]) * hx + abs(nrm[1]) * hy
    z_lo, z_hi = float(panel["z_lo"]), float(panel["z_hi"])
    zc = 0.5 * (z_lo + z_hi)
    face_w = 2.0 * (abs(ux) * hx + abs(uy) * hy)
    # Two passes, because on a curved body the width follows the standoff and the standoff follows
    # the height the band ends up occupying. The first pass measures over the whole wall (an upper
    # bound), the second over the band that pass produced.
    stand = _wall_reach(panel, nrm, bbox_reach, z_lo, z_hi) + _DECAL_GAP_M
    dev_w = 2.0 * stand * half_arc if curved else 0.62 * face_w
    # Height comes from the wall that exists, capped at what the typesetter can fill: it breaks a
    # string over at most 3 lines, so a box much taller than that is empty paper that only shrinks
    # the glyphs it surrounds.
    h = min(0.90 * (z_hi - z_lo), 2.0 * dev_w)
    stand = (
        _wall_reach(panel, nrm, bbox_reach, zc - 0.5 * h, zc + 0.5 * h) + _DECAL_GAP_M
    )
    if curved:
        dev_w = 2.0 * stand * half_arc
        h = min(0.90 * (z_hi - z_lo), 2.0 * dev_w)
    side_png = _retypeset_for(png, dev_w / max(h, 1e-6), "band")
    if curved:
        _wrapped_quad(
            world.stage,
            f"{root}/Cue_side",
            (cx, cy),
            stand,
            nrm,
            zc,
            h,
            half_arc,
            side_png,
        )
    else:
        px, py = cx + nrm[0] * stand, cy + nrm[1] * stand
        side = [
            (px - ux * dev_w * 0.5, py - uy * dev_w * 0.5, zc - h * 0.5),
            (px + ux * dev_w * 0.5, py + uy * dev_w * 0.5, zc - h * 0.5),
            (px + ux * dev_w * 0.5, py + uy * dev_w * 0.5, zc + h * 0.5),
            (px - ux * dev_w * 0.5, py - uy * dev_w * 0.5, zc + h * 0.5),
        ]
        _textured_quad(
            world.stage, f"{root}/Cue_side", side, (nrm[0], nrm[1], 0.0), side_png
        )
    # ONE label per object. There used to be a second card over the shoulder as well, for "what a
    # person sees looking down at it on the pass" -- but two labels on one bottle is two things to
    # read and no way to tell which one the task means, and the shoulder card is the one that goes
    # edge-on the moment the object is raised to the face. The body band serves both viewpoints.
    # The band's own pose, in the ANCHOR's frame. ``label_pos`` is the world point it occupies while
    # the object sits at its registered rest; ``label_local`` is what a consumer needs once the
    # object has MOVED -- the raised inspect carries the label to the face, and a reader that only
    # has the rest pose is measuring an empty shelf (which is exactly what the manifest's
    # label_visibility rows did for every graspable).
    lx, ly = cx + nrm[0] * stand, cy + nrm[1] * stand
    ent.meta["label_face"] = (float(world_nrm[0]), float(world_nrm[1]))
    ent.meta["label_face3"] = (
        float(world_nrm[0]),
        float(world_nrm[1]),
        0.0,
    )
    ent.meta["label_face_local"] = (float(nrm[0]), float(nrm[1]), 0.0)
    ent.meta["label_local"] = (float(lx), float(ly), float(zc))
    # The patch's own extent. ``label_span`` is what a reader can take in square-on (the chord of a
    # wrap, the whole card of a flat one); ``label_size`` is the patch, which is what tells a
    # consumer where the writing STOPS -- the raised inspect uses it to grip clear of it.
    ent.meta["label_span"] = float(
        2.0 * stand * math.sin(half_arc) if curved else dev_w
    )
    ent.meta["label_size"] = (float(dev_w), float(h))
    ent.meta["label_up3"] = (0.0, 0.0, 1.0)
    ent.meta["label_up_local"] = (0.0, 0.0, 1.0)
    ent.meta["label_curved"] = bool(curved)
    ent.meta["label_mount"] = "side"
    ent.meta["label_read_in_place"] = False
    _publish_label_readability(ent, side_png)
    if ent.xy is not None:
        ez = float(ent.z) if ent.z is not None else 0.0
        label_x, label_y = _label_world_xy(ent, float(lx), float(ly))
        ent.meta["label_pos"] = (
            label_x,
            label_y,
            ez + zc,
        )
    print(
        f"[bench] item decal {item.key}: {'wrap' if curved else 'flat'} "
        f"{dev_w * 1000:.0f}x{h * 1000:.0f}mm on z[{panel['z_lo']:.3f},{panel['z_hi']:.3f}] "
        f"round={panel['roundness']:.2f}"
    )
    world.decals[item.key] = f"{root}/Cue_side"
    # Park the anchor at its registered seat. The render's ``build_object`` re-poses it anyway, but
    # until then the decal quads sit at the WORLD ORIGIN -- which makes every t=0 measurement of this
    # graspable meaningless, and T0-EXACT is a measurement of exactly that.
    if park_anchor:
        _park_item_decal_anchor(world, root, ent)


def _rotate_z(vector: tuple[float, float, float], yaw_deg: float) -> tuple[float, float, float]:
    angle = math.radians(float(yaw_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
        vector[2],
    )


def publish_native_nutrition_labels(world: World) -> None:
    """Point LookAt at a package's own printed panel without adding any rendered decal."""
    objects = dict(((world.spec.get("world") or {}).get("objects") or {}))
    for item in world.recipe.staged():
        row = objects.get(item.key)
        if not isinstance(row, Mapping) or row.get("cue") != "native_nutrition":
            continue
        category, model = staged_body(world, item)
        panel = _NATIVE_NUTRITION_PANELS.get((category, model))
        if panel is None:
            raise SpecBuildError(
                f"{world.recipe.task_id}: {item.key} requests its native nutrition panel, "
                f"but {category}/{model} has no measured panel anchor"
            )
        ent = world.scene.entities.get(item.key)
        if ent is None or ent.xy is None or ent.z is None:
            raise SpecBuildError(
                f"{world.recipe.task_id}: native-label item {item.key} was not seated"
            )
        local = tuple(float(value) for value in panel["local"])  # pyre-ignore[6]
        face = tuple(float(value) for value in panel["face"])  # pyre-ignore[6]
        up = tuple(float(value) for value in panel["up"])  # pyre-ignore[6]
        size = tuple(float(value) for value in panel["size"])  # pyre-ignore[6]
        rest_yaw = float(ent.meta.get("rest_yaw_deg") or 0.0)
        world_local = _rotate_z(local, rest_yaw)
        world_face = _rotate_z(face, rest_yaw)
        world_up = _rotate_z(up, rest_yaw)
        ent.meta.update(
            {
                "label_pos": (
                    float(ent.xy[0]) + world_local[0],
                    float(ent.xy[1]) + world_local[1],
                    float(ent.z) + world_local[2],
                ),
                "label_local": local,
                "label_face": world_face[:2],
                "label_face3": world_face,
                "label_face_local": face,
                "label_up3": world_up,
                "label_up_local": up,
                "label_span": max(size),
                "label_size": size,
                "label_primary_glyph_height_ratio": float(
                    panel["glyph_height_ratio"]
                ),
                "label_primary_text_bbox_norm_xyxy": tuple(
                    float(value) for value in panel["content_box_uv"]
                ),
                "label_curved": bool(panel["curved"]),
                "label_mount": str(panel.get("mount") or "native_package"),
                "label_source": "native_asset_texture",
                "inspect_in_place": False,
            }
        )
        for meta_field in ("look_target_glyph_px", "look_held_distance_m"):
            if panel.get(meta_field) is not None:
                ent.meta[meta_field] = panel[meta_field]
        if panel.get("inspect_grip_local") is not None:
            ent.meta["inspect_grip_local"] = tuple(
                float(value) for value in panel["inspect_grip_local"]
            )
        print(
            f"[bench] native nutrition panel {item.key}: {category}/{model} "
            f"{size[0] * 1000:.0f}x{size[1] * 1000:.0f}mm"
        )


# =====================================================================================
# stage 4 -- seat the graspables on "the pass"
# =====================================================================================
def _finish_support_pick(
    world: World,
    best,
    target_room: str,
    reserve: bool,
    fixture_index: int,
):
    if reserve:
        interaction = world.last_support_sites.get((best[2], best[3]))
        if interaction is not None:
            world.support_contact_xy[fixture_index] = interaction[0]
            world.support_stand_xy[fixture_index] = interaction[1]
        _reserve_support_stand(
            world,
            best[2],
            best[3],
            target_room,
            stand_xy=world.support_stand_xy.get(fixture_index),
        )
        print(
            f"[bench] pass = {best[2]} approach_yaw={best[3]:.0f} clear={best[1]:.2f}m "
            f"in_reach={bool(best[0])}"
        )
    return best[2], best[3]


def _required_support_capabilities(world: World, fixture_index: int) -> frozenset[str]:
    if fixture_index >= len(world.recipe.support_keys):
        return frozenset()
    authored = getattr(world.recipe, "support_capabilities", ())
    if fixture_index < len(authored):
        return frozenset(authored[fixture_index])
    requirements = program_fixture_capabilities(
        str(world.spec.get("human_program") or "")
    )
    return requirements.get(world.recipe.support_keys[fixture_index], frozenset())


def _declared_support_value(world: World, field: str, fixture_index: int):
    values = getattr(world.recipe, field, ())
    return values[fixture_index] if fixture_index < len(values) else None


def support_requires_worktop(recipe: Recipe, fixture_index: int) -> bool:
    """Whether a declared fixture must provide usable horizontal staging geometry."""
    support_items = getattr(recipe, "support_items", ())
    assigned = (
        support_items[fixture_index]
        if fixture_index < len(support_items)
        else ()
    )
    support_capabilities = getattr(recipe, "support_capabilities", ())
    capabilities = (
        frozenset(support_capabilities[fixture_index])
        if fixture_index < len(support_capabilities)
        else frozenset()
    )
    return _capabilities_require_worktop(capabilities, bool(assigned))


def _capabilities_require_worktop(
    capabilities: frozenset[str], has_payload: bool
) -> bool:
    return has_payload or "surface" in capabilities


def _support_passes_stage_band(
    world: World, bounds: tuple, requires_worktop: bool
) -> bool:
    return (
        not world.stage_band
        or not requires_worktop
        or stageable_top(world, *bounds)
    )


def _support_asset(world: World, path: str) -> tuple[str, str] | None:
    for placed_path, metadata in world.placed.values():
        if str(placed_path) != str(path):
            continue
        category = str(metadata.get("cat") or "")
        model = str(metadata.get("model") or "")
        if category and model:
            return category, model
    return None


def _has_support_openable(spec: Mapping[str, object]) -> bool:
    joints = spec.get("joints") or []
    return _has_base_door(dict(spec)) or any(
        isinstance(joint, Mapping) and joint.get("type") == "prismatic"
        for joint in joints
    )


def _support_openable_fields(world: World, path: str) -> dict | None:
    import behavior_import as bi

    asset = _support_asset(world, path)
    if asset is None:
        return None
    category, model = asset
    return bi._closing_leaf_fields(
        world.scene,
        path,
        {"cat": category, "model": model},
    )


def _leaf_front_direction(
    leaf_fields: Mapping[str, object], center_x: float, center_y: float
) -> tuple[float, float] | None:
    leaf_bbox = leaf_fields.get("leaf_bbox")
    if not leaf_bbox:
        return None
    minimum, maximum = leaf_bbox
    dx = 0.5 * (float(minimum[0]) + float(maximum[0])) - center_x
    dy = 0.5 * (float(minimum[1]) + float(maximum[1])) - center_y
    norm = math.hypot(dx, dy)
    if norm <= 1e-5:
        return None
    return dx / norm, dy / norm


def _openable_support_yaws(
    leaf_fields: Mapping[str, object], center_x: float, center_y: float
) -> tuple[float, ...]:
    if leaf_fields.get("horizontal_top_lid"):
        return (0.0, 90.0, 180.0, 270.0)
    front = _leaf_front_direction(leaf_fields, center_x, center_y)
    if front is None:
        return (0.0, 90.0, 180.0, 270.0)
    return (math.degrees(math.atan2(-front[0], front[1])) % 360.0,)


def _thin_panel_approach_yaws(
    leaf_fields: Mapping[str, object],
) -> tuple[float, float]:
    """The two face-normal approaches to a thin world-space panel."""
    minimum, maximum = leaf_fields["leaf_bbox"]
    x_span = float(maximum[0]) - float(minimum[0])
    y_span = float(maximum[1]) - float(minimum[1])
    return (90.0, 270.0) if x_span <= y_span else (0.0, 180.0)


def support_body_ladder(
    category: str, *, want_openable: bool = False
) -> list[tuple[str, str]]:
    """Same-category support models, with human-height work surfaces tried first."""
    import behavior_import as bi

    rows = []
    for model in baked_models(category):
        try:
            spec = bi.read_spec(f"{category}/{model}")
            if want_openable and not _has_support_openable(spec):
                continue
            bbox = spec.get("bbox_size") or ()
            if len(bbox) < 3 or min(float(value) for value in bbox[:3]) <= 0.0:
                continue
            width, depth, height = (float(value) for value in bbox[:3])
        except (OSError, TypeError, ValueError):
            continue
        ergonomic = int(not (STAGE_TOP_MIN_M <= height <= STAGE_TOP_MAX_M))
        rows.append(
            (
                ergonomic,
                abs(height - 0.85),
                abs(min(width, depth) - 0.55),
                width * depth,
                model,
            )
        )
    return [(category, row[-1]) for row in sorted(rows)]


def _remove_imported_support(world: World, path: str) -> None:
    key = getattr(world, "imported_support_paths", {}).pop(path, None)
    if key is None:
        return
    if world.stage.GetPrimAtPath(path):
        world.stage.RemovePrim(path)
    world.placed.pop(key, None)
    world.zspan.pop(key, None)


def _support_siting_bounds(
    category: str,
    model: str,
    xy: tuple[float, float],
    yaw_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    import behavior_import as bi

    spec = bi.read_spec(f"{category}/{model}")
    bbox = spec.get("bbox_size") or ()
    if len(bbox) < 3 or min(float(value) for value in bbox[:3]) <= 0.0:
        raise SpecBuildError(
            f"explicit support {category}/{model} has no valid measured bounding box"
        )
    width, depth, height = (float(value) for value in bbox[:3])
    angle = math.radians(float(yaw_deg))
    half_x = 0.5 * (abs(math.cos(angle)) * width + abs(math.sin(angle)) * depth)
    half_y = 0.5 * (abs(math.sin(angle)) * width + abs(math.cos(angle)) * depth)
    return (
        (float(xy[0]) - half_x, float(xy[1]) - half_y, 0.0),
        (float(xy[0]) + half_x, float(xy[1]) + half_y, height),
    )


def _xy_aabb_clearance(left: tuple, right: tuple) -> float:
    left_minimum, left_maximum = left
    right_minimum, right_maximum = right
    dx = max(
        float(right_minimum[0]) - float(left_maximum[0]),
        float(left_minimum[0]) - float(right_maximum[0]),
        0.0,
    )
    dy = max(
        float(right_minimum[1]) - float(left_maximum[1]),
        float(left_minimum[1]) - float(right_maximum[1]),
        0.0,
    )
    return math.hypot(dx, dy)


def _validate_explicit_support_site(
    world: World,
    key: str,
    room: str,
    bounds: tuple,
    obstacles: Sequence[tuple],
) -> None:
    minimum, maximum = bounds
    x0, y0, x1, y1 = room_box(world.recipe.scene, room)
    if (
        float(minimum[0]) < x0
        or float(maximum[0]) > x1
        or float(minimum[1]) < y0
        or float(maximum[1]) > y1
    ):
        raise SpecBuildError(
            f"{world.recipe.task_id}: explicit support {key} lies outside room {room!r}"
        )
    inset = 0.04
    floor_samples = (
        (float(minimum[0]) + inset, float(minimum[1]) + inset),
        (float(minimum[0]) + inset, float(maximum[1]) - inset),
        (float(maximum[0]) - inset, float(minimum[1]) + inset),
        (float(maximum[0]) - inset, float(maximum[1]) - inset),
    )
    if not all(
        floor_supported_at(world.recipe.scene, room, x, y) for x, y in floor_samples
    ):
        raise SpecBuildError(
            f"{world.recipe.task_id}: explicit support {key} is not fully on modeled floor"
        )
    for obstacle_name, obstacle_minimum, obstacle_maximum in obstacles:
        if (
            float(obstacle_maximum[2]) <= float(minimum[2]) + 0.01
            or float(obstacle_minimum[2]) >= float(maximum[2]) - 0.01
        ):
            continue
        clearance = _xy_aabb_clearance(
            (minimum, maximum), (obstacle_minimum, obstacle_maximum)
        )
        if clearance < 0.05:
            raise SpecBuildError(
                f"{world.recipe.task_id}: explicit support {key} is only "
                f"{clearance:.3f}m from scene obstacle {obstacle_name!r}; 0.050m is required"
            )


def _materialize_explicit_support(
    world: World, fixture_index: int, room: str
) -> str:
    import behavior_import as bi
    import fpv_scene

    key = world.recipe.support_keys[fixture_index]
    category = world.recipe.support_bindings[fixture_index]
    model = world.recipe.support_models[fixture_index]
    xy = world.recipe.support_siting_xy[fixture_index]
    siting_yaw = world.recipe.support_siting_front_yaws[fixture_index]
    if not model or xy is None or siting_yaw is None:
        raise SpecBuildError(
            f"{world.recipe.task_id}: imported fixture {key} has incomplete explicit siting"
        )
    placed_name = f"support_explicit_{bi._safe(key)}"
    path = f"/World/bs_{bi._safe(room)}_{bi._safe(key)}"
    existing = world.placed.get(placed_name)
    if existing is not None and str(existing[0]) == path:
        return placed_name

    original = (world.room, world.box, world.anchor)
    world.room = room
    world.box = room_box(world.recipe.scene, room)
    world.anchor = room_anchor(world.recipe.scene, room)
    obstacles = list(siting_boxes(world))
    try:
        if world.stage.GetPrimAtPath(path):
            world.stage.RemovePrim(path)
        prim = fpv_scene.add_reference(
            world.stage, path, bi.baked_usd(f"{category}/{model}")
        )
        angle = math.radians(float(siting_yaw))
        quaternion = (math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle))
        _drop_to_floor(
            world,
            prim,
            path,
            float(xy[0]),
            float(xy[1]),
            quaternion,
        )
        bounds = bi._collision_bbox(world.stage, path) or _wbbox(world.stage, path)
        _validate_explicit_support_site(world, key, room, bounds, obstacles)
        world.placed[placed_name] = (
            path,
            {
                "cat": category,
                "model": model,
                "room": room,
            },
        )
        world.zspan.pop(placed_name, None)
        imported_paths = getattr(world, "imported_support_paths", None)
        if imported_paths is None:
            imported_paths = {}
            world.imported_support_paths = imported_paths
        imported_paths[path] = placed_name
        logger.info(
            f"[bench] imported explicit support {key}={category}/{model} "
            f"at ({float(xy[0]):.3f},{float(xy[1]):.3f}) yaw={float(siting_yaw):.1f}"
        )
        return placed_name
    except Exception:
        if world.stage.GetPrimAtPath(path):
            world.stage.RemovePrim(path)
        raise
    finally:
        world.room, world.box, world.anchor = original


def _import_same_category_support(
    world: World,
    categories: Sequence[str],
    room: str,
    fixture_index: int,
) -> str:
    """Materialize one reachable same-category surface after scene fixtures are exhausted."""
    import behavior_import as bi
    import fpv_scene

    original = (world.room, world.box, world.anchor)
    world.room = room
    world.box = room_box(world.recipe.scene, room)
    world.anchor = room_anchor(world.recipe.scene, room)
    try:
        cells, boxes = _grid(world)
        required = _required_support_capabilities(world, fixture_index)
        requires_worktop = support_requires_worktop(world.recipe, fixture_index)
        attempts_by_fixture = getattr(world, "support_import_attempts", None)
        if attempts_by_fixture is None:
            attempts_by_fixture = {}
            world.support_import_attempts = attempts_by_fixture
        for category in categories:
            attempt_key = (room, category, int(fixture_index))
            attempted = attempts_by_fixture.setdefault(attempt_key, set())
            for _candidate_category, model in support_body_ladder(
                category,
                want_openable="openable" in required,
            ):
                if model in attempted:
                    continue
                attempted.add(model)
                spec = bi.read_spec(f"{category}/{model}")
                bbox = spec.get("bbox_size") or [9.0, 9.0, 9.0]
                half = 0.5 * max(float(bbox[0]), float(bbox[1]))
                slot, _reserve = _find_slot(
                    world, cells, half, near=world.anchor, dcap_max=12.0
                )
                if slot is None:
                    continue
                key = f"support_fallback_{fixture_index}_{len(attempted)}"
                path = f"/World/bs_{bi._safe(room)}_{bi._safe(category)}_{fixture_index}"
                if world.stage.GetPrimAtPath(path):
                    world.stage.RemovePrim(path)
                prim = fpv_scene.add_reference(
                    world.stage, path, bi.baked_usd(f"{category}/{model}")
                )
                orientation = (
                    _door_quat(world, path, category, model, slot, boxes)
                    if "openable" in required
                    else (1.0, 0.0, 0.0, 0.0)
                )
                _drop_to_floor(
                    world,
                    prim,
                    path,
                    float(slot[0]),
                    float(slot[1]),
                    orientation,
                )
                bounds = bi._collision_bbox(world.stage, path) or _wbbox(
                    world.stage, path
                )
                if not _support_passes_stage_band(
                    world, bounds, requires_worktop
                ):
                    world.stage.RemovePrim(path)
                    continue
                world.placed[key] = (
                    path,
                    {"cat": category, "model": model, "room": room},
                )
                world.zspan.pop(key, None)
                imported_paths = getattr(world, "imported_support_paths", None)
                if imported_paths is None:
                    imported_paths = {}
                    world.imported_support_paths = imported_paths
                imported_paths[path] = key
                logger.info(
                    f"[bench] imported same-category staging fallback "
                    f"{category}/{model} in {room} at {path}"
                )
                return path
    finally:
        world.room, world.box, world.anchor = original
    return ""


def _support_payload_profile(
    world: World, fixture_index: int
) -> tuple[tuple[float, float], float] | None:
    blocks = _item_blocks(world, max(1, int(world.recipe.n_supports)))
    if fixture_index >= len(blocks) or not blocks[fixture_index]:
        return None
    half_x = half_y = height = 0.0
    for item in blocks[fixture_index]:
        size = staged_bbox(world, item)
        if len(size) < 3 or min(float(value) for value in size[:3]) <= 0.0:
            return None
        radius = 0.5 * max(float(size[0]), float(size[1]))
        half_x = max(half_x, radius)
        half_y = max(half_y, radius)
        height = max(height, float(size[2]))
    return (half_x, half_y), height


def _support_has_payload_clearance(
    support_name: str,
    support_bounds,
    support_top_z: float,
    payload_profile: tuple[tuple[float, float], float],
    scene_boxes,
) -> bool:
    """Whether some patch of a support has room for its tallest staged payload."""
    minimum, maximum = support_bounds
    half_xy, height = payload_profile
    hx, hy = float(half_xy[0]), float(half_xy[1])
    x_lo, x_hi = float(minimum[0]) + hx, float(maximum[0]) - hx
    y_lo, y_hi = float(minimum[1]) + hy, float(maximum[1]) - hy
    if x_lo > x_hi or y_lo > y_hi:
        return False
    xs = list(dict.fromkeys((x_lo, 0.5 * (x_lo + x_hi), x_hi)))
    ys = list(dict.fromkeys((y_lo, 0.5 * (y_lo + y_hi), y_hi)))
    base_z = float(support_top_z)
    top_z = base_z + float(height)
    for x in xs:
        for y in ys:
            blocked = False
            for name, obstacle_min, obstacle_max in scene_boxes:
                if name == support_name:
                    continue
                if float(obstacle_max[2]) <= base_z + 0.006:
                    continue
                if float(obstacle_min[2]) >= top_z - 0.006:
                    continue
                if x + hx <= float(obstacle_min[0]) or x - hx >= float(
                    obstacle_max[0]
                ):
                    continue
                if y + hy <= float(obstacle_min[1]) or y - hy >= float(
                    obstacle_max[1]
                ):
                    continue
                blocked = True
                break
            if not blocked:
                return True
    return False


def seat_front_direction(
    metadata: Mapping[str, object],
) -> tuple[float, float] | None:
    """Return a scene seat's world-space front direction from its XYZW rotation."""
    orientation = metadata.get("ori")
    if not isinstance(orientation, Sequence) or isinstance(
        orientation, (str, bytes)
    ):
        return None
    try:
        x, y, z, w = (float(value) for value in orientation)
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-8:
        return None
    x, y, z, w = (value / norm for value in (x, y, z, w))
    front_x = 1.0 - 2.0 * (y * y + z * z)
    front_y = 2.0 * (x * y + z * w)
    planar_norm = math.hypot(front_x, front_y)
    if not math.isfinite(planar_norm) or planar_norm <= 1e-8:
        return None
    return front_x / planar_norm, front_y / planar_norm


def support_approach_yaws(
    metadata: Mapping[str, object],
    *,
    seatable: bool,
    openable_fields: Mapping[str, object] | None,
    center_x: float,
    center_y: float,
) -> tuple[float, ...]:
    if openable_fields is not None:
        # A sliding window's joint travels within the pane.  Its leaf-centre offset therefore
        # points along the glass rather than toward the room, unlike a drawer face.  Approach the
        # pane from either normal and let floor/clearance checks select the indoor side.
        if metadata.get("cat") == "openable_window" and openable_fields.get(
            "kind"
        ) == "slide":
            return _thin_panel_approach_yaws(openable_fields)
        return _openable_support_yaws(openable_fields, center_x, center_y)
    seat_front = seat_front_direction(metadata) if seatable else None
    if seat_front is None:
        return (0.0, 90.0, 180.0, 270.0)
    return (math.degrees(math.atan2(-seat_front[0], seat_front[1])) % 360.0,)


def _fixture_is_seatable(world: World, fixture_index: int) -> bool:
    if fixture_index >= len(world.recipe.support_keys):
        return False
    capabilities = getattr(world.recipe, "support_capabilities", ())
    if fixture_index < len(capabilities) and "seat" in capabilities[fixture_index]:
        return True
    key = world.recipe.support_keys[fixture_index]
    fixtures = (world.spec.get("world") or {}).get("fixtures") or {}
    return bool((fixtures.get(key) or {}).get("seatable"))


def _fixture_is_seat_only(world: World, fixture_index: int) -> bool:
    if not _fixture_is_seatable(world, fixture_index):
        return False
    capabilities = _required_support_capabilities(world, fixture_index)
    support_items = getattr(world.recipe, "support_items", ())
    assigned = support_items[fixture_index] if fixture_index < len(support_items) else ()
    return capabilities == frozenset({"seat"}) and not assigned


def _fixture_requirement_phrase(world: World, fixture_index: int) -> str:
    if _fixture_is_seat_only(world, fixture_index):
        return "with a clear, floor-supported front seating approach"
    return "to stage the graspables on"


def _fixture_interaction_sites(
    world: World,
    fixture_index: int,
    *,
    center: tuple[float, float],
    half: tuple[float, float],
    yaw: float,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Return safe candidate contact/stand pairs for one fixture approach.

    A wide seat can have a coffee table in front of its midpoint while one end remains open. Seat-
    only fixtures may therefore use an off-centre cushion position, but stay on the authored front
    side and retain the same clearance and floor gates as every other fixture. Work surfaces stay
    centred so authored object spots and support-capacity checks remain unchanged.
    """
    declared_stand = _declared_support_value(
        world, "support_operator_stands", fixture_index
    )
    if declared_stand is not None:
        return (
            (
                (float(center[0]), float(center[1])),
                (float(declared_stand[0]), float(declared_stand[1])),
            ),
        )

    angle = math.radians(float(yaw))
    forward = (math.sin(angle), -math.cos(angle))
    across = (-forward[1], forward[0])
    depth = abs(forward[0]) * half[0] + abs(forward[1]) * half[1]
    offsets = (0.0,)
    if _fixture_is_seat_only(world, fixture_index):
        span = min(
            half[0] / max(abs(across[0]), 1e-6),
            half[1] / max(abs(across[1]), 1e-6),
        )
        offsets = (0.0, 0.25 * span, -0.25 * span, 0.50 * span, -0.50 * span)
    return tuple(
        (
            (
                float(center[0] + across[0] * offset),
                float(center[1] + across[1] * offset),
            ),
            (
                float(center[0] + across[0] * offset - forward[0] * (depth + 0.55)),
                float(center[1] + across[1] * offset - forward[1] * (depth + 0.55)),
            ),
        )
        for offset in offsets
    )


def _best_fixture_interaction_site(
    world: World,
    fixture_index: int,
    *,
    center: tuple[float, float],
    half: tuple[float, float],
    yaw: float,
    boxes: Sequence[tuple],
    obstacle_name: str,
    room: str,
    has_floor: bool,
) -> tuple[
    tuple[float, tuple[float, float], tuple[float, float]] | None,
    int,
]:
    best = None
    rejected_off_floor = 0
    for contact, stand in _fixture_interaction_sites(
        world,
        fixture_index,
        center=center,
        half=half,
        yaw=yaw,
    ):
        stand_x, stand_y = stand
        if has_floor and not floor_supported_at(
            world.recipe.scene,
            room,
            float(stand_x),
            float(stand_y),
        ):
            rejected_off_floor += 1
            continue
        clearance = min(
            _clearance(boxes, stand_x, stand_y, obstacle_name),
            wall_dist(world, stand_x, stand_y),
        )
        if clearance < SUPPORT_STAND_MIN_CLEAR_M:
            continue
        option = (float(clearance), contact, stand)
        if best is None or option[0] > best[0]:
            best = option
    return best, rejected_off_floor


def _live_support_candidate_geometry(
    world: World,
    name: str,
    path: str,
    metadata: Mapping[str, object],
    *,
    allowed: Sequence[str],
    target_room: str,
    in_room: set[str],
    names: tuple[str, ...] | None,
    required: frozenset[str],
    requires_worktop: bool,
    payload_profile,
    boxes: Sequence[tuple],
    fixture_index: int,
):
    import behavior_import as bi

    if metadata.get("cat") not in allowed or (
        name not in in_room and metadata.get("room") != target_room
    ):
        return None
    if names and name not in names:
        return None
    if path in world.support_skip:
        return None
    category = str(metadata.get("cat") or "")
    if "toggleable" in required and "toggleable" not in category_affordances(category):
        return None
    openable_fields = (
        _support_openable_fields(world, path) if "openable" in required else None
    )
    if "openable" in required and openable_fields is None:
        logger.info(
            f"[bench] support candidate {path} rejected: fixture "
            f"{fixture_index + 1} requires a drivable openable"
        )
        return None
    minimum, maximum = bi._collision_bbox(world.stage, path) or _wbbox(
        world.stage, path
    )
    if not _support_passes_stage_band(world, (minimum, maximum), requires_worktop):
        return None
    worktop_z = _worktop_z(world.stage, path, float(maximum[2]))
    # Exact scene-instance bindings are checked at their authored item spots by the
    # later support preflight. The coarse probe can otherwise mistake built-in
    # cubicle partitions for a completely blocked worktop.
    if names is None and payload_profile is not None and not _support_has_payload_clearance(
        name,
        (minimum, maximum),
        worktop_z,
        payload_profile,
        boxes,
    ):
        logger.info(
            f"[bench] support candidate {path} rejected: no vertical clearance "
            f"for fixture {fixture_index + 1} payload"
        )
        return None
    return minimum, maximum, openable_fields


def _rank_support_options(
    world: World,
    allowed: Sequence[str],
    target_room: str,
    skip: set[str],
    names: tuple[str, ...] | None,
    fixture_index: int = 0,
):
    boxes = siting_boxes(world)
    payload_profile = _support_payload_profile(world, fixture_index)
    required = _required_support_capabilities(world, fixture_index)
    requires_worktop = _capabilities_require_worktop(
        required, payload_profile is not None
    )
    seatable = _fixture_is_seatable(world, fixture_index)
    in_room = room_prims(world.recipe.scene, target_room)
    has_floor = bool(room_floor_regions(world.recipe.scene, target_room))
    best = None
    best_by_path: dict[str, tuple[int, float, str, float]] = {}
    site_by_path: dict[
        str, tuple[tuple[float, float], tuple[float, float]]
    ] = {}
    for name, (path, metadata) in world.placed.items():
        geometry = _live_support_candidate_geometry(
            world,
            name,
            path,
            metadata,
            allowed=allowed,
            target_room=target_room,
            in_room=in_room,
            names=names,
            required=required,
            requires_worktop=requires_worktop,
            payload_profile=payload_profile,
            boxes=boxes,
            fixture_index=fixture_index,
        )
        if geometry is None:
            continue
        minimum, maximum, openable_fields = geometry
        center_x = 0.5 * (minimum[0] + maximum[0])
        center_y = 0.5 * (minimum[1] + maximum[1])
        half_x = 0.5 * (maximum[0] - minimum[0])
        half_y = 0.5 * (maximum[1] - minimum[1])
        reachable = int(float(maximum[2]) + 0.05 <= reach_max_z())
        declared_yaw = _declared_support_value(
            world, "support_approach_yaws", fixture_index
        )
        angles = (
            (float(declared_yaw),)
            if declared_yaw is not None
            else support_approach_yaws(
                metadata,
                seatable=seatable,
                openable_fields=openable_fields,
                center_x=center_x,
                center_y=center_y,
            )
        )
        for angle in angles:
            site, _rejected = _best_fixture_interaction_site(
                world,
                fixture_index,
                center=(center_x, center_y),
                half=(half_x, half_y),
                yaw=float(angle),
                boxes=boxes,
                obstacle_name=name,
                room=target_room,
                has_floor=has_floor,
            )
            if site is None:
                continue
            clearance, contact, stand = site
            option = (reachable, clearance, path, float(angle))
            prior = best_by_path.get(path)
            if prior is None or option[:2] > prior[:2]:
                best_by_path[path] = option
                site_by_path[path] = (contact, stand)
            if path not in skip and (best is None or option[:2] > best[:2]):
                best = option
    world.last_support_sites = {
        (row[2], row[3]): site_by_path[path]
        for path, row in best_by_path.items()
    }
    return best, best_by_path


def _pick_imported_support(
    world: World,
    skip: set[str],
    categories: tuple[str, ...] | None,
    names: tuple[str, ...] | None,
    room: str | None,
    target_room: str,
    allowed: Sequence[str],
    reserve: bool,
    fixture_index: int,
):
    if names is not None:
        return None
    for _attempt in range(8):
        imported = _import_same_category_support(
            world, allowed, target_room, fixture_index
        )
        if not imported:
            return None
        try:
            return _pick_support(
                world,
                skip=skip,
                categories=categories,
                names=names,
                room=room,
                reserve=reserve,
                fixture_index=fixture_index,
                allow_import=False,
            )
        except SpecBuildError:
            _remove_imported_support(world, imported)
    return None


def _pick_support(
    world: World,
    skip: set[str] | None = None,
    categories: tuple[str, ...] | None = None,
    names: tuple[str, ...] | None = None,
    room: str | None = None,
    reserve: bool = True,
    fixture_index: int = 0,
    allow_import: bool = True,
):
    """The in-room support with the most open floor on its best approach side. Returns
    (prim_path, approach_yaw): the yaw a person stands at, i.e. the side the cue faces."""
    target_room = room or world.room
    allowed = categories or world.recipe.support_cats
    for path in set(getattr(world, "support_skip", set())):
        _remove_imported_support(world, path)
    skipped = skip or set()
    best, best_by_path = _rank_support_options(
        world,
        allowed,
        target_room,
        skipped,
        names,
        fixture_index=fixture_index,
    )
    world.last_support_options = [
        (row[2], row[3])
        for row in sorted(
            best_by_path.values(), key=lambda row: (row[0], row[1]), reverse=True
        )
    ]
    if best is None:
        imported_pick = (
            _pick_imported_support(
                world,
                skipped,
                categories,
                names,
                room,
                target_room,
                allowed,
                reserve,
                fixture_index,
            )
            if allow_import
            else None
        )
        if imported_pick is not None:
            return imported_pick
        raise SpecBuildError(
            f"no {allowed} in {target_room} "
            f"{_fixture_requirement_phrase(world, fixture_index)}"
            + (
                f" (the grasp probe already rejected {sorted(world.support_skip)})"
                if world.support_skip
                else ""
            )
            + (
                f" (staging band {STAGE_TOP_MIN_M}-{STAGE_TOP_MAX_M} m ON)"
                if world.stage_band
                else ""
            )
            + "; object spot coordinates do not move a scene fixture's approach pose; "
            "change the fixture category, room, or scene"
        )
    return _finish_support_pick(
        world,
        best,
        target_room,
        reserve,
        fixture_index,
    )


def _pick_supports(world: World, n: int) -> list[tuple[str, float]]:
    """The n clearest in-room supports, each reserved. A row that looks at ``tray_1`` and ``tray_2``
    needs two REAL surfaces; splitting one bench in half would render one tray and drop a look."""
    got: list[tuple[str, float]] = []
    options: list[list[tuple[str, float]]] = []
    bindings = world.recipe.support_bindings
    rooms = world.recipe.support_rooms
    names = getattr(world.recipe, "support_names", ())
    reserved_paths = {
        world.placed[name][0] for name in names if name and name in world.placed
    }
    for index in range(max(1, int(n))):
        categories = (bindings[index],) if index < len(bindings) else None
        exact_names = (names[index],) if index < len(names) and names[index] else None
        room = rooms[index] if index < len(rooms) else world.room
        explicit_xy = _declared_support_value(world, "support_siting_xy", index)
        if explicit_xy is not None:
            exact_names = (_materialize_explicit_support(world, index, room),)
        current_paths = {
            world.placed[name][0]
            for name in (exact_names or ())
            if name in world.placed
        }
        pp, yaw = _pick_support(
            world,
            skip=({p for p, _y in got} | reserved_paths) - current_paths,
            categories=categories,
            names=exact_names,
            room=room,
            fixture_index=index,
            allow_import=explicit_xy is None,
        )
        got.append((pp, yaw))
        ranked = list(world.last_support_options)
        if (
            exact_names is None
            and categories is not None
            and set(categories) != set(world.recipe.support_cats)
        ):
            _pick_support(
                world,
                categories=world.recipe.support_cats,
                room=room,
                reserve=False,
                fixture_index=index,
                allow_import=False,
            )
            ranked.extend(world.last_support_options)
        options.append(list(dict.fromkeys(ranked)))
    world.support_options = options
    return got


def _reserve_support_stand(
    world: World,
    pp: str,
    yaw: float,
    room: str,
    *,
    stand_xy: tuple[float, float] | None = None,
) -> None:
    """Keep the pass's own approach cell out of the unit slot search, so no cold unit is sited
    into the floor the human needs to stand on to pick the tub up."""
    import behavior_import as bi

    (cmn, cmx) = bi._collision_bbox(world.stage, pp) or _wbbox(world.stage, pp)
    cx, cy = 0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1])
    hx, hy = 0.5 * (cmx[0] - cmn[0]), 0.5 * (cmx[1] - cmn[1])
    a = math.radians(yaw)
    fwd = (math.sin(a), -math.cos(a))
    hd = abs(fwd[0]) * hx + abs(fwd[1]) * hy
    sx, sy = (
        stand_xy
        if stand_xy is not None
        else (cx - fwd[0] * (hd + 0.55), cy - fwd[1] * (hd + 0.55))
    )
    slot = (sx, sy, 0.50)
    stand = (float(sx), float(sy))
    world.used_slots.append(slot)
    world.support_stands.append(stand)
    world.room_slots.setdefault(room, []).append(slot)
    world.room_support_stands.setdefault(room, []).append(stand)


def _item_blocks(world: World, n: int) -> list[list[ItemRecipe]]:
    """The staged graspables split across ``n`` surfaces in contiguous blocks, scored ones first.

    One function, two readers: ``seat_all_items`` stages by it and the grasp repair reads it to
    learn WHICH surface carries the graspable the human could not pick up.

    A row may declare more placarded surfaces than it stages graspables -- a sign is a thing to
    READ, not a thing to hold, and ``register_supports`` mounts every card whether or not stock
    sits under it. Those surfaces get an EMPTY block, at the end, so the block index still names
    the support the grasp repair has to drop."""
    if world.recipe.support_items:
        by_key = {item.key: item for item in world.recipe.staged()}
        return [
            [by_key[key] for key in group if key in by_key]
            for group in world.recipe.support_items
        ]
    movable_children = {
        item.key
        for _parent, children in _movable_support_groups(
            world.recipe.staged(), world.recipe.task_id
        )
        for item in children
    }
    items = [
        item
        for item in staged_order(world.recipe, world.seed)
        if not item.initial_place.startswith("floor:")
        and item.key not in movable_children
    ]
    k = min(max(1, int(n)), len(items))
    if not k:
        return [[] for _ in range(n)]
    blocks = [items[i * len(items) // k : (i + 1) * len(items) // k] for i in range(k)]
    return blocks + [[] for _ in range(max(0, n - k))]


def seat_all_items(world: World, supports: list[tuple[str, float]]) -> None:
    """Split the staged graspables (scored ones first, then the decoys) across the supports in
    contiguous blocks, so ``tray_1`` and ``tray_2`` each carry a readable group rather than an
    interleaved mix, and publish every support prim to the scene."""
    from scene_state import Role

    for (pp, yaw), block in zip(supports, _item_blocks(world, len(supports))):
        if block:
            seat_items(world, pp, yaw, block)
    # EVERY declared top needs an exact static collider, not just the ones things start on.
    # BEHAVIOR tops are instanced prims: without the de-instancing pass this list drives, an object
    # released over one falls straight THROUGH it. Measured on NARRATE-3, the first row scored
    # purely by `on`: both objects were placed with their xy dead on the destination (1 mm and
    # 8 cm inside a 1.08 m half-extent) and came to rest 168 mm BELOW the bench top, so all four
    # goals read false while the video showed the person putting them down correctly.
    _tops = {
        pp
        for index, (pp, _yaw) in enumerate(supports)
        if not _fixture_is_seat_only(world, index)
    }
    _declared = set((((world.spec or {}).get("world") or {}).get("fixtures") or {}))
    for key, ent in world.scene.entities.items():
        if key in _declared and ent.role is Role.SURFACE and ent.prim_path:
            _tops.add(str(ent.prim_path))
    world.scene.behavior_support_prims = sorted(_tops)


def _undo_items(world: World) -> None:
    """Un-stage every graspable -- registrations, the render's build list, the support publication
    -- so the next attempt seats them on the NEW surface instead of on top of the old row."""
    for it in world.recipe.staged():
        world.scene.entities.pop(it.key, None)
        getattr(world.scene, "_static_aabb", {}).pop(it.key, None)
    world.scene.grasp_objs = []
    world.scene.behavior_support_prims = []


def _grasp_repair(world: World, bad: list[tuple[str, str]], supports) -> str:
    """Advance ONE rung: the staging SURFACE first, then the graspable BODY.

    Surface first because that is where the measured failure lives -- T6-RURB-01's ``[Grab]`` did
    not fail on ``beer_glass/lymciz`` (probed here: seat_err 3.3 mm, 4 of 5 fingers in contact), it
    failed because approach clearance alone elected a WINDOW 2.62 m up as the pass. Swapping the
    body first would have re-bodied a blameless object and left the row at 0.00."""
    bad_keys = {k for k, _w in bad}
    drop = [
        supports[i][0]
        for i, blk in enumerate(_item_blocks(world, len(supports)))
        if bad_keys & {it.key for it in blk}
        and supports[i][0] not in world.support_skip
    ]
    if drop:
        world.support_skip.update(drop)
        return f"drop staging surface(s) {sorted(drop)}"
    if world.recipe.frozen_bindings:
        return ""
    for key, _why in bad:
        b = (_staged_of(world, key).cat, _staged_of(world, key).model)
        lad = body_ladder(b[0], b[1])
        if world.item_bump.get(b, 0) + 1 >= len(lad):
            return ""
        world.item_bump[b] = world.item_bump.get(b, 0) + 1
    return f"bump graspable bodies {sorted(world.item_bump.items())}"


def seat_reachable_items(world: World, supports: list[tuple[str, float]]):
    """Stage the graspables, and REPAIR any staging the human cannot actually pick up from.

    The gate in ``_finish`` is a hard build failure by design; this loop is what makes it
    satisfiable without an author hand-picking a surface. Exhausting both ladders is a LOUD
    failure -- it never degrades to "the row renders and scores 0.00"."""
    if not world.recipe.staged():
        world.scene.behavior_support_prims = sorted(
            {
                path
                for index, (path, _yaw) in enumerate(supports)
                if not _fixture_is_seat_only(world, index)
            }
        )
        world.supports = supports
        return supports
    enable_runtime_geometry(world.scene)
    support_keys = _support_staged_keys(world.recipe)
    for attempt in range(_MAX_GRASP_REPAIRS):
        seat_all_items(world, supports)
        bad = grasp_defects(world, support_keys)
        if not bad:
            world.supports = supports
            return supports
        step = _grasp_repair(world, bad, supports)
        if not step:
            break
        print(
            f"[bench] GRASP REPAIR {attempt + 1}/{_MAX_GRASP_REPAIRS}: "
            f"{[k for k, _w in bad]} unreachable ({bad[0][1]}) -> {step}"
        )
        _undo_items(world)
        world.support_stands = []
        supports = _pick_supports(world, world.recipe.n_supports)
        # ``_plan_in_room`` anchored the episode (exposure focus, TPV framing) on the surface it
        # picked; a repair that moves the pass has to move the anchor with it.
        world.anchor = _ctr(world.stage, supports[0][0])
    raise SpecBuildError(
        f"{world.recipe.task_id}: no staging surface in {world.room} lets the human PICK the "
        f"graspables up after {_MAX_GRASP_REPAIRS} repair(s) (surfaces rejected: "
        f"{sorted(world.support_skip)}; body bumps: {sorted(world.item_bump.items())}). The row "
        f"needs another support category on its Recipe.support_cats, not another model."
    )


def _support_staged_keys(recipe: Recipe) -> set[str]:
    """Return only graspables that this pass actually seats on a support."""
    if recipe.support_items:
        return {key for group in recipe.support_items for key in group}
    movable_children = {
        item.key
        for _parent, children in _movable_support_groups(
            recipe.staged(), recipe.task_id
        )
        for item in children
    }
    return {
        item.key
        for item in recipe.staged()
        if not item.initial_place.startswith("floor:")
        and item.key not in movable_children
    }


def _worktop_z(stage, pp: str, fallback: float) -> float:
    """The z things actually come to rest on, not the top of the tallest thing attached to it.

    A lab bench carries a reagent rack: its collision bbox tops out at 1.05 m while the worktop is
    at 0.88 m. Registering the bbox max made `surface_top` 168 mm too high, so an object the person
    put down correctly -- xy dead on target -- read 168 mm BELOW its own surface and every `on`
    goal scored false. Measured on NARRATE-3, the first row scored purely by `on`.

    The worktop is the highest collision face that is WIDE, DEEP and THIN: it spans most of the
    footprint and is a slab. A rack shelf fails the depth test, a leg fails the thinness test. No
    candidate (a plain slab desk, one mesh) -> the bbox top, exactly as before.
    """
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(pp)
    if not (prim and prim.IsValid()):
        return fallback
    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    boxes = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() != "Mesh" or "/collisions/" not in str(p.GetPath()):
            continue
        rng = bc.ComputeWorldBound(p).ComputeAlignedRange()
        a, b = rng.GetMin(), rng.GetMax()
        if b[0] >= a[0]:
            boxes.append(
                (
                    (float(a[0]), float(a[1]), float(a[2])),
                    (float(b[0]), float(b[1]), float(b[2])),
                )
            )
    if len(boxes) < 2:
        return fallback
    ww = max(b[0] for _a, b in boxes) - min(a[0] for a, _b in boxes)
    dd = max(b[1] for _a, b in boxes) - min(a[1] for a, _b in boxes)
    hh = max(b[2] for _a, b in boxes) - min(a[2] for a, _b in boxes)
    if min(ww, dd, hh) <= 1e-3:
        return fallback
    # A worktop does not arrive as one mesh: convex decomposition hands back dozens of chunks (64
    # on lab_table_vujgjq), none of them spanning half the footprint. So band the chunk TOPS by
    # height and ask which band actually covers the bench -- the worktop covers most of it, the
    # reagent rack above it covers a strip.
    foot = ww * dd
    tops = sorted({round(b[2], 3) for _a, b in boxes}, reverse=True)
    for level in tops:
        area = 0.0
        for a, b in boxes:
            if abs(b[2] - level) <= 0.02:
                area += max(0.0, b[0] - a[0]) * max(0.0, b[1] - a[1])
        if area >= 0.45 * foot:
            return float(level)
    return fallback


def _support_surface_fields(
    pp: str,
    yaw: float,
    center: tuple[float, float],
    half: tuple[float, float],
    top_z: float,
) -> dict[str, object]:
    return {
        "xy": center,
        "z": float(top_z),
        "half": half,
        "prim_path": pp,
        "approach_yaw": float(yaw),
        "foot_xy": center,
        "foot_half": half,
        "top_z": float(top_z),
        "clearance": 0.45,
    }


def _register_slide_support(
    world: World,
    key: str,
    pp: str,
    yaw: float,
    category: str,
    model: str,
    fields: Mapping[str, object],
    surface_fields: Mapping[str, object],
) -> None:
    from scene_state import Role

    center = surface_fields["xy"]
    front = (
        None
        if category == "openable_window"
        else _leaf_front_direction(fields, float(center[0]), float(center[1]))
    )
    if front is None:
        angle = math.radians(yaw)
        front = (-math.sin(angle), math.cos(angle))
    leaf_minimum, leaf_maximum = fields["leaf_bbox"]
    handle_xy = (
        0.5 * (float(leaf_minimum[0]) + float(leaf_maximum[0])),
        0.5 * (float(leaf_minimum[1]) + float(leaf_maximum[1])),
    )
    handle_z = 0.5 * (float(leaf_minimum[2]) + float(leaf_maximum[2]))
    world.front_dir[key] = front
    primary_role = Role.OPENABLE if category == "openable_window" else Role.SURFACE
    world.scene.register(
        key,
        primary_role,
        **surface_fields,
        joint_name=str(fields["joint_name"]),
        axis=(float(front[0]), float(front[1]), 0.0),
        travel=float(fields["travel"]),
        leaf_paths=tuple(str(path) for path in fields["leaf_paths"]),
        handle_xy=handle_xy,
        handle_z=handle_z,
        meta={
            "asset_category": category,
            "asset_model": model,
            "openable_kind": "slide",
        },
    )
    if primary_role is not Role.OPENABLE:
        world.scene.register_affordance(
            key,
            Role.OPENABLE,
            xy=handle_xy,
            z=handle_z,
            prim_path=pp,
            approach_yaw=float(yaw),
            joint_name=str(fields["joint_name"]),
            axis=(float(front[0]), float(front[1]), 0.0),
            travel=float(fields["travel"]),
            leaf_paths=tuple(str(path) for path in fields["leaf_paths"]),
            handle_xy=handle_xy,
            handle_z=handle_z,
            meta={"openable_kind": "slide"},
        )


def _register_openable_support(
    world: World,
    key: str,
    pp: str,
    yaw: float,
    category: str,
    model: str,
    fields: Mapping[str, object],
    surface_fields: Mapping[str, object],
) -> None:
    from scene_state import Role

    if fields.get("kind") == "slide":
        _register_slide_support(
            world, key, pp, yaw, category, model, fields, surface_fields
        )
        return
    if not _register_swing_unit(world, key, pp, category, model):
        raise SpecBuildError(f"support {key} at {pp} lost its openable leaf")
    world.boxes_of.pop(key, None)
    world.outer_of.pop(key, None)
    world.scene.register_affordance(key, Role.SURFACE, **surface_fields)


def register_supports(
    world: World, supports: list[tuple[str, float]], pngs: dict[str, str]
) -> None:
    """Make each staging surface a LookAt-able entity with its own card, so an authored
    ``[LookAt] tray_1`` resolves to a real sign read square-on rather than failing to ground."""
    import behavior_import as bi
    from scene_state import Role

    required_by_key = program_fixture_capabilities(
        str(world.spec.get("human_program") or "")
    )
    for index, (key, (pp, yaw)) in enumerate(
        zip(world.recipe.support_keys, supports)
    ):
        (cmn, cmx) = bi._collision_bbox(world.stage, pp) or _wbbox(world.stage, pp)
        cx, cy = 0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1])
        hx, hy = 0.5 * (cmx[0] - cmn[0]), 0.5 * (cmx[1] - cmn[1])
        topz = _worktop_z(world.stage, pp, float(cmx[2]))
        a = math.radians(yaw)
        fd = (-math.sin(a), math.cos(a))
        surface_fields = _support_surface_fields(
            pp,
            yaw,
            (float(cx), float(cy)),
            (hx, hy),
            float(topz),
        )
        required = required_by_key.get(key, frozenset())
        seatable = _fixture_is_seatable(world, index)
        seat_only = _fixture_is_seat_only(world, index)
        contact_xy = getattr(world, "support_contact_xy", {}).get(
            index, (float(cx), float(cy))
        )
        stand_xy = getattr(world, "support_stand_xy", {}).get(index)
        asset = _support_asset(world, pp)
        openable_fields = _support_openable_fields(world, pp)
        if "openable" in required and asset is not None and openable_fields is not None:
            _register_openable_support(
                world,
                key,
                pp,
                yaw,
                asset[0],
                asset[1],
                openable_fields,
                surface_fields,
            )
            fd = world.front_dir[key]
        else:
            if "openable" in required:
                raise SpecBuildError(
                    f"support {key} at {pp} cannot satisfy required Open/Close actions"
                )
            world.front_dir[key] = fd
            if seat_only:
                seat_fields = dict(surface_fields)
                seat_fields["xy"] = contact_xy
                seat_fields["meta"] = {
                    "sit_yaw": (float(yaw) + 180.0) % 360.0,
                    "seat_contact_xy": contact_xy,
                }
                world.scene.register(key, Role.SEAT, **seat_fields)
            else:
                world.scene.register(key, Role.SURFACE, **surface_fields)
        if seatable and not seat_only:
            world.scene.register_affordance(
                key,
                Role.SEAT,
                xy=contact_xy,
                z=float(topz),
                half=(hx, hy),
                prim_path=pp,
                foot_xy=(float(cx), float(cy)),
                foot_half=(hx, hy),
                top_z=float(topz),
                meta={"sit_yaw": (float(yaw) + 180.0) % 360.0},
            )
        if asset is not None:
            register_asset_affordances(world, key, pp, asset[0], asset[1])
        declared_stand = _declared_support_value(
            world, "support_operator_stands", index
        )
        if declared_stand is not None:
            world.scene.entities[key].meta["operator_stand_xy"] = (
                float(declared_stand[0]),
                float(declared_stand[1]),
            )
            world.scene.entities[key].meta["stationary_operator"] = True
        world.scene.entities[key].meta["front_stand"] = stand_xy or (
            float(contact_xy[0] + fd[0] * (abs(fd[0]) * hx + abs(fd[1]) * hy + 0.55)),
            float(contact_xy[1] + fd[1] * (abs(fd[0]) * hx + abs(fd[1]) * hy + 0.55)),
        )
        prop_png = pngs.get(f"prop:{key}")
        if prop_png:
            mount_placard(world, key, pp, prop_png)
        print(f"[bench] support {key} = {pp} yaw={yaw:.0f}")
    world.scene.initial_support_options = {}
    model_by_path = {
        str(path): str(obj.get("model") or obj.get("cat") or "")
        for path, obj in world.placed.values()
    }
    for index, (key, selected) in enumerate(zip(world.recipe.support_keys, supports)):
        ranked = (
            world.support_options[index]
            if index < len(world.support_options)
            else [selected]
        )
        buckets: dict[str, list[tuple[str, float]]] = {}
        for value in ranked:
            buckets.setdefault(model_by_path.get(value[0], value[0]), []).append(value)
        diversified = [
            bucket[offset]
            for offset in range(max((len(bucket) for bucket in buckets.values()), default=0))
            for bucket in buckets.values()
            if offset < len(bucket)
        ]
        ordered = [selected]
        required = required_by_key.get(key, frozenset())
        if f"prop:{key}" not in pngs and "openable" not in required:
            ordered.extend(
                value for value in diversified if value[0] != selected[0]
            )
        world.scene.initial_support_options[key] = [
            {
                "path": path,
                "requires_worktop": support_requires_worktop(world.recipe, index),
                "yaw": float(option_yaw),
                "top_z": float(
                    _worktop_z(
                        world.stage,
                        path,
                        float(
                            (
                                bi._collision_bbox(world.stage, path)
                                or _wbbox(world.stage, path)
                            )[1][2]
                        ),
                    )
                ),
            }
            for path, option_yaw in ordered
        ]


def _workstation_yaw(direction) -> float:
    return math.degrees(math.atan2(float(direction[0]), -float(direction[1])))


def _uses_computer_target(spec: dict, key: str) -> bool:
    for token in program_tokens(str(spec.get("human_program") or "")):
        verb_match = re.match(r"\[([A-Za-z_]+)\]", token)
        verb = verb_match.group(1).replace("_", "").upper() if verb_match else ""
        if verb not in {"USECOMPUTER", "TYPE"}:
            continue
        if key in re.findall(r"<([^>?]+)>", token):
            return True
    return False


def _standing_workstation_pose(
    support_xy,
    support_half,
    front_dir,
    keyboard_bbox,
    spot=None,
) -> tuple[tuple[float, float], float, tuple[float, float, float, float]]:
    """Seat a keyboard across the real front edge for a standing two-hand reach."""
    fx, fy = float(front_dir[0]), float(front_dir[1])
    norm = math.hypot(fx, fy)
    if norm <= 1e-6:
        fx, fy = 0.0, -1.0
    else:
        fx, fy = fx / norm, fy / norm
    hx, hy = float(support_half[0]), float(support_half[1])
    depth = abs(fx) * hx + abs(fy) * hy
    forward_half = 0.5 * min(float(keyboard_bbox[0]), float(keyboard_bbox[1]))
    front_offset = max(0.0, depth - forward_half - 0.04)
    across_x, across_y = fy, -fx
    span = abs(across_x) * hx + abs(across_y) * hy
    across_half = 0.5 * max(float(keyboard_bbox[0]), float(keyboard_bbox[1]))
    authored_across = (
        max(-1.0, min(1.0, float(spot[0])))
        if spot is not None
        else 0.0
    )
    across_offset = authored_across * max(0.0, span - across_half - 0.04)
    xy = (
        float(support_xy[0]) + fx * front_offset + across_x * across_offset,
        float(support_xy[1]) + fy * front_offset + across_y * across_offset,
    )
    standing_yaw = _workstation_yaw((-fx, -fy)) % 360.0
    object_yaw = standing_yaw - 90.0
    orientation = (
        math.cos(math.radians(object_yaw) * 0.5),
        0.0,
        0.0,
        math.sin(math.radians(object_yaw) * 0.5),
    )
    return xy, standing_yaw, orientation


def _configure_standing_workstation(
    world: World,
    keyboard: ItemRecipe,
    support_key: str,
    support,
    monitor: ItemRecipe | None = None,
) -> dict[str, object] | None:
    if not _uses_computer_target(world.spec, keyboard.key):
        return None
    import behavior_import as bi

    category, model = staged_body(world, keyboard)
    bbox = bi.read_spec(f"{category}/{model}").get("bbox_size") or [
        0.18,
        0.48,
        0.02,
    ]
    keyboard_xy, standing_yaw, orientation = _standing_workstation_pose(
        support.xy,
        support.half,
        world.front_dir.get(support_key) or (0.0, -1.0),
        bbox,
        keyboard.spot,
    )
    keyboard_entity = world.scene.resolve(keyboard.key)
    keyboard_entity.xy = keyboard_xy
    keyboard_entity.foot_xy = keyboard_xy
    keyboard_entity.approach_yaw = standing_yaw
    keyboard_entity.meta.update(
        {
            "interaction_anchor": support_key,
            "rest_orientation": orientation,
            "rest_yaw_deg": (standing_yaw - 90.0) % 360.0,
            "standing_workstation": True,
        }
    )
    getattr(world.scene, "_static_aabb", {}).pop(keyboard.key, None)
    for object_spec in world.scene.grasp_objs:
        if object_spec.get("key") == keyboard.key:
            object_spec["orientation"] = orientation
    workstation = None
    screen_style = (
        monitor.workstation_screen_style
        if monitor is not None
        else keyboard.workstation_screen_style
    )
    if (monitor is not None and monitor.key in world.scene.entities) or screen_style:
        if monitor is not None:
            monitor_category, monitor_model = staged_body(world, monitor)
            monitor_bbox = bi.read_spec(f"{monitor_category}/{monitor_model}").get(
                "bbox_size"
            ) or [0.19, 0.37, 0.34]
        else:
            monitor_bbox = [0.19, 0.37, 0.34]
        front_x, front_y = world.front_dir.get(support_key) or (0.0, -1.0)
        front_norm = math.hypot(float(front_x), float(front_y))
        if front_norm <= 1e-6:
            front_x, front_y = 0.0, -1.0
        else:
            front_x = float(front_x) / front_norm
            front_y = float(front_y) / front_norm
        across_x, across_y = front_y, -front_x
        depth = abs(front_x) * float(support.half[0]) + abs(front_y) * float(
            support.half[1]
        )
        monitor_forward_half = 0.5 * min(
            float(monitor_bbox[0]), float(monitor_bbox[1])
        )
        monitor_back_offset = max(0.0, depth - monitor_forward_half - 0.04)
        keyboard_across = (
            (float(keyboard_xy[0]) - float(support.xy[0])) * across_x
            + (float(keyboard_xy[1]) - float(support.xy[1])) * across_y
        )
        monitor_xy = (
            float(support.xy[0])
            - front_x * monitor_back_offset
            + across_x * keyboard_across,
            float(support.xy[1])
            - front_y * monitor_back_offset
            + across_y * keyboard_across,
        )
        desk_top = float(
            support.top_z if support.top_z is not None else support.z
        )
        screen_center = (
            float(monitor_xy[0]),
            float(monitor_xy[1]),
            desk_top + 0.27,
        )
        screen_key = monitor.key if monitor is not None else keyboard.key
        screen_entity = world.scene.resolve(screen_key)
        screen_meta = {
            "screen_center": screen_center,
            "screen_normal": (front_x, front_y, 0.0),
            "screen_right": (across_x, across_y, 0.0),
            "workstation_screen_style": screen_style,
        }
        if monitor is not None:
            screen_entity.xy = monitor_xy
            screen_entity.foot_xy = monitor_xy
            screen_entity.approach_yaw = standing_yaw
            screen_meta.update(
                {
                "rest_orientation": orientation,
                "rest_yaw_deg": (standing_yaw - 90.0) % 360.0,
                }
            )
            getattr(world.scene, "_static_aabb", {}).pop(monitor.key, None)
            for object_spec in world.scene.grasp_objs:
                if object_spec.get("key") == monitor.key:
                    object_spec["orientation"] = orientation
        screen_entity.meta.update(screen_meta)
        keyboard_entity.meta["workstation_monitor"] = screen_key
        mouse_lateral = 0.5 * max(float(bbox[0]), float(bbox[1])) + 0.065
        workstation = {
            "desk": support_key,
            "seat": "",
            "monitor": screen_key,
            "mouse_center": (
                float(keyboard_xy[0]) - across_x * mouse_lateral,
                float(keyboard_xy[1]) - across_y * mouse_lateral,
                desk_top + 0.022,
            ),
            "procedural_monitor": monitor is None,
            "screen_center": screen_center,
            "screen_style": screen_style,
            "yaw": standing_yaw,
        }
    print(
        f"[bench] standing workstation {keyboard.key}: {support_key} "
        f"front=({keyboard_xy[0]:.2f},{keyboard_xy[1]:.2f}) "
        f"yaw={standing_yaw:.0f}"
    )
    return workstation


def _configure_workstations(world: World) -> None:
    """Configure seated monitor workstations and reachable standing keyboards."""
    import numpy as np
    from pxr import Usd, UsdGeom

    in_room_cache: dict[str, set[str]] = {}
    workstations: dict[str, dict[str, object]] = {}
    for keyboard in world.recipe.staged():
        if keyboard.cat != "keyboard" or not keyboard.initial_place.startswith("on:"):
            continue
        support_key = keyboard.initial_place[3:]
        support = world.scene.entities.get(support_key)
        if support is None or support.xy is None:
            continue
        monitor = next(
            (
                item
                for item in world.recipe.staged()
                if item.cat == "monitor"
                and item.initial_place == keyboard.initial_place
            ),
            None,
        )
        if monitor is None or monitor.key not in world.scene.entities:
            workstation = _configure_standing_workstation(
                world,
                keyboard,
                support_key,
                support,
            )
            if workstation is not None:
                workstations[keyboard.key] = workstation
            continue
        if keyboard.workstation_posture == "standing":
            workstation = _configure_standing_workstation(
                world,
                keyboard,
                support_key,
                support,
                monitor,
            )
            if workstation is not None:
                workstations[keyboard.key] = workstation
            continue
        fixture = dict(
            ((world.spec.get("world") or {}).get("fixtures") or {}).get(
                support_key
            )
            or {}
        )
        room = str(fixture.get("room") or world.room)
        if room not in in_room_cache:
            in_room_cache[room] = set(room_prims(world.recipe.scene, room))
        candidates = [
            (name, path, row)
            for name, (path, row) in world.placed.items()
            if name in in_room_cache[room]
            and any(
                token in str(row.get("cat") or "")
                for token in ("chair", "stool", "armchair")
            )
        ]
        if not candidates:
            raise SpecBuildError(
                f"{world.recipe.task_id}: keyboard {keyboard.key} has no chair near {support_key}"
            )
        desk_xy = np.asarray(support.xy, dtype=np.float64)
        chair_name, chair_path, _chair_row = min(
            candidates,
            key=lambda value: float(
                np.linalg.norm(
                    np.asarray(_ctr(world.stage, value[1])[:2], dtype=np.float64)
                    - desk_xy
                )
            ),
        )
        chair_center = np.asarray(_ctr(world.stage, chair_path), dtype=np.float64)
        distance = float(np.linalg.norm(desk_xy - chair_center[:2]))
        if distance > 1.25:
            raise SpecBuildError(
                f"{world.recipe.task_id}: nearest chair {chair_name} is {distance:.2f}m "
                f"from workstation desk {support_key}"
            )
        toward_desk = (desk_xy - chair_center[:2]) / max(distance, 1e-6)
        side = np.array([-toward_desk[1], toward_desk[0]], dtype=np.float64)
        seated_yaw = _workstation_yaw(toward_desk)
        approach_yaw = _workstation_yaw(-toward_desk)
        chair_entity = world.scene.resolve(chair_name)
        chair_prim = world.stage.GetPrimAtPath(chair_path)
        transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            chair_prim
        )
        root_z = float(transform.ExtractTranslation()[2])
        if not 0.35 <= root_z <= 0.65:
            root_z = 0.50
        chair_entity.xy = (float(chair_center[0]), float(chair_center[1]))
        chair_entity.z = root_z
        chair_entity.top_z = root_z
        chair_entity.approach_yaw = approach_yaw
        chair_entity.clearance = 0.08
        chair_entity.meta["sit_yaw"] = seated_yaw
        chair_entity.meta["sit_back_offset_m"] = 0.22

        hx, hy = support.half
        depth = abs(toward_desk[0]) * float(hx) + abs(toward_desk[1]) * float(hy)
        keyboard_xy = desk_xy - toward_desk * max(0.05, depth - 0.20)
        monitor_xy = desk_xy + toward_desk * min(0.18, 0.55 * depth)
        mouse_xy = keyboard_xy - side * 0.18
        desk_top = float(support.top_z if support.top_z is not None else support.z)
        object_yaw = seated_yaw - 90.0
        orientation = (
            math.cos(math.radians(object_yaw) * 0.5),
            0.0,
            0.0,
            math.sin(math.radians(object_yaw) * 0.5),
        )
        for item, xy in ((keyboard, keyboard_xy), (monitor, monitor_xy)):
            entity = world.scene.resolve(item.key)
            entity.xy = (float(xy[0]), float(xy[1]))
            entity.foot_xy = entity.xy
            entity.approach_yaw = seated_yaw
            entity.meta["rest_orientation"] = orientation
            entity.meta["rest_yaw_deg"] = seated_yaw
            for object_spec in world.scene.grasp_objs:
                if object_spec.get("key") == item.key:
                    object_spec["orientation"] = orientation
        keyboard_entity = world.scene.resolve(keyboard.key)
        keyboard_entity.meta.update(
            {
                "interaction_anchor": chair_name,
                "workstation_seat": chair_name,
                "workstation_monitor": monitor.key,
            }
        )
        monitor_entity = world.scene.resolve(monitor.key)
        monitor_entity.meta.update(
            {
                "screen_center": (
                    float(monitor_xy[0]),
                    float(monitor_xy[1]),
                    desk_top + 0.27,
                ),
                "screen_normal": (
                    float(-toward_desk[0]),
                    float(-toward_desk[1]),
                    0.0,
                ),
                "screen_right": (float(side[0]), float(side[1]), 0.0),
            }
        )
        workstations[keyboard.key] = {
            "desk": support_key,
            "seat": chair_name,
            "monitor": monitor.key,
            "mouse_center": (
                float(mouse_xy[0]),
                float(mouse_xy[1]),
                desk_top + 0.022,
            ),
            "yaw": seated_yaw,
        }
        print(
            f"[bench] workstation {keyboard.key}: {support_key} + {chair_name} + "
            f"{monitor.key}, seated_yaw={seated_yaw:.0f}"
        )
    world.scene.workstations = workstations


def staged_body(world: World, it: ItemRecipe) -> tuple[str, str]:
    """The body a staged graspable is ACTUALLY played by: the recipe's, advanced down its own
    ladder by however many rungs the GRASP probe has demanded.

    ``world.item_bump`` is the graspable's own ledger, deliberately not the unit's ``body_bump``:
    the two repair loops must not be able to bump each other's bodies when a category happens to
    serve both roles. No bump -> the authored body, byte-identical."""
    b = (it.cat, it.model)
    if world.recipe.frozen_bindings:
        return b
    lad = body_ladder(b[0], b[1])
    return lad[min(int(world.item_bump.get(b, 0)), len(lad) - 1)]


def staged_bbox(world: World, item: ItemRecipe) -> tuple[float, float, float]:
    import behavior_import as bi

    category, model = staged_body(world, item)
    raw = bi.read_spec(f"{category}/{model}").get("bbox_size") or [0.2, 0.2, 0.2]
    scale_xyz = _scale_xyz(item.scale)
    return tuple(
        float(value) * scale_xyz[axis] for axis, value in enumerate(raw[:3])
    )


def staged_build_params(world: World, item: ItemRecipe) -> dict:
    import behavior_import as bi

    return bi.behavior_build_params("/".join(staged_body(world, item)), item.scale)


_SEAT_GAP_M: float = (
    0.06  # the same clearance the even-row ``pitch`` below already leaves
)
_SEAT_GAP_LADDER_M: tuple[float, ...] = (_SEAT_GAP_M, 0.04, 0.02)


def _seat_offsets(
    spots: list, halves: list, span: float, depth: float, inset: float, pitch: float
) -> list[tuple[float, float]]:
    """Per item, ``(across, into)`` metres from the seating anchor: ``across`` along the front edge,
    ``into`` along the approach axis away from the reader.

    Pure geometry, no stage -- this is the whole of what an authored ``spot`` changes, so it can be
    checked offline. Items with no spot keep today's evenly-spaced row EXACTLY: with none authored,
    ``free`` is every index, ``across`` is the same ``(i - (n-1)/2) * pitch``, ``into`` is 0, the
    de-overlap sweep is a no-op (``pitch`` already exceeds any two halves plus the gap) and the clamp
    is skipped. A byte-identical fallback is the property that lets this land without re-measuring
    every shipped row.

    An authored ``u`` spans the usable half-width -- the surface's own half-extent along the front
    edge, less the body's half -- so ``u = +-1`` puts the body's EDGE at the rim rather than its
    centre, and it cannot be authored into a position it would topple out of. ``v`` does the same
    across the depth. Authored positions are de-overlapped in AUTHOR ORDER (the first one written
    keeps its place) and clamped inside the surface. The caller verifies the resulting rectangular
    footprints and packs them into multiple rows when that clamp leaves too little room."""
    n = len(spots)
    free = [i for i in range(n) if spots[i] is None]
    out: list[list[float]] = [[0.0, 0.0] for _ in range(n)]
    for j, i in enumerate(free):
        out[i][0] = (j - (len(free) - 1) / 2.0) * pitch
    for i, sp in enumerate(spots):
        if sp is None:
            continue
        u, v = float(sp[0]), float(sp[1])
        out[i][0] = u * max(0.0, span - halves[i])
        out[i][1] = (2.0 * v - 1.0) * max(0.0, depth - halves[i]) + inset
    for i in range(n):
        for k in range(i):
            gap = halves[i] + halves[k] + _SEAT_GAP_M
            if abs(out[i][1] - out[k][1]) >= gap:
                continue
            delta = out[i][0] - out[k][0]
            if abs(delta) >= gap:
                continue
            out[i][0] = out[k][0] + (gap if delta >= 0.0 else -gap)
        if spots[i] is not None:
            lim = max(0.0, span - halves[i])
            out[i][0] = min(lim, max(-lim, out[i][0]))
    return [(row[0], row[1]) for row in out]


def _stable_seat_overhang(half_extent: float) -> float:
    return min(0.05, 0.20 * float(half_extent))


def seat_rectangles_fit(
    offsets: list[tuple[float, float]],
    across_halves: list[float],
    depth_halves: list[float],
    span: float,
    depth: float,
    inset: float,
    gap: float,
) -> bool:
    for i, (across, into) in enumerate(offsets):
        if (
            abs(across) + across_halves[i]
            > span + _stable_seat_overhang(across_halves[i]) + 1e-6
        ):
            return False
        if (
            abs(into - inset) + depth_halves[i]
            > depth + _stable_seat_overhang(depth_halves[i]) + 1e-6
        ):
            return False
        for k in range(i):
            across_clear = abs(across - offsets[k][0]) + 1e-6 >= (
                across_halves[i] + across_halves[k] + gap
            )
            depth_clear = abs(into - offsets[k][1]) + 1e-6 >= (
                depth_halves[i] + depth_halves[k] + gap
            )
            if not across_clear and not depth_clear:
                return False
    return True


_seat_rectangles_fit = seat_rectangles_fit


def _balanced_rows(n: int, n_rows: int) -> list[list[int]]:
    width, extra = divmod(n, n_rows)
    rows: list[list[int]] = []
    start = 0
    for row_index in range(n_rows):
        stop = start + width + int(row_index < extra)
        rows.append(list(range(start, stop)))
        start = stop
    return rows


def _seat_axis_candidates(
    desired, low, high, center, occupied, half, other_halves, axis, gap
):
    values = [min(high, max(low, desired)), low, high, center]
    for index, x, y in occupied:
        other_center = x if axis == 0 else y
        other_half = other_halves[index]
        values.extend(
            (
                other_center - other_half - half - gap,
                other_center + other_half + half + gap,
            )
        )
    return sorted(
        {
            round(float(value), 9)
            for value in values
            if low - 1e-9 <= value <= high + 1e-9
        },
        key=lambda value: (abs(value - desired), abs(value - center), value),
    )


def _freeform_seat_offsets(
    items,
    across_halves,
    depth_halves,
    span,
    depth,
    inset,
    gap,
):
    desired = [
        (
            float(item.spot[0]) * max(0.0, span - across_halves[index]),
            (2.0 * float(item.spot[1]) - 1.0)
            * max(0.0, depth - depth_halves[index])
            + inset,
        )
        if item.spot is not None
        else (0.0, inset)
        for index, item in enumerate(items)
    ]
    order = sorted(
        range(len(items)),
        key=lambda index: (
            -across_halves[index] * depth_halves[index],
            -max(across_halves[index], depth_halves[index]),
            index,
        ),
    )
    occupied = []
    offsets: list[tuple[float, float] | None] = [None] * len(items)
    nodes = 0

    def place(position: int) -> bool:
        nonlocal nodes
        if position == len(order):
            return True
        if nodes >= 50000:
            return False
        index = order[position]
        ah, dh = across_halves[index], depth_halves[index]
        ax = _stable_seat_overhang(ah)
        ay = _stable_seat_overhang(dh)
        x_low, x_high = -span - ax + ah, span + ax - ah
        y_low, y_high = inset - depth - ay + dh, inset + depth + ay - dh
        if x_low > x_high or y_low > y_high:
            return False
        xs = _seat_axis_candidates(
            desired[index][0],
            x_low,
            x_high,
            0.0,
            occupied,
            ah,
            across_halves,
            0,
            gap,
        )
        ys = _seat_axis_candidates(
            desired[index][1],
            y_low,
            y_high,
            inset,
            occupied,
            dh,
            depth_halves,
            1,
            gap,
        )
        candidates = sorted(
            ((x, y) for x in xs for y in ys),
            key=lambda xy: (
                (xy[0] - desired[index][0]) ** 2
                + (xy[1] - desired[index][1]) ** 2,
                xy,
            ),
        )
        for x, y in candidates:
            nodes += 1
            if any(
                abs(x - ox) < ah + across_halves[other] + gap - 1e-9
                and abs(y - oy) < dh + depth_halves[other] + gap - 1e-9
                for other, ox, oy in occupied
            ):
                continue
            offsets[index] = (x, y)
            occupied.append((index, x, y))
            if place(position + 1):
                return True
            occupied.pop()
            offsets[index] = None
        return False

    return [offset for offset in offsets if offset is not None] if place(0) else None


def _packed_seat_offsets(
    across_halves: list[float],
    depth_halves: list[float],
    span: float,
    depth: float,
    inset: float,
    items=None,
) -> tuple[list[tuple[float, float]], float] | None:
    """Pack items into the fewest deterministic rows that fit with positive clearance."""
    n = len(across_halves)
    for gap in _SEAT_GAP_LADDER_M:
        for n_rows in range(1, n + 1):
            rows = _balanced_rows(n, n_rows)
            widths = [
                sum(2.0 * across_halves[i] for i in row) + gap * max(0, len(row) - 1)
                for row in rows
            ]
            row_depths = [max(depth_halves[i] for i in row) for row in rows]
            total_depth = sum(2.0 * value for value in row_depths) + gap * (n_rows - 1)
            width_fits = all(
                width
                <= 2.0 * span
                + _stable_seat_overhang(across_halves[row[0]])
                + _stable_seat_overhang(across_halves[row[-1]])
                + 1e-6
                for row, width in zip(rows, widths)
            )
            depth_allowance = _stable_seat_overhang(row_depths[0]) + _stable_seat_overhang(
                row_depths[-1]
            )
            if not width_fits or total_depth > 2.0 * depth + depth_allowance + 1e-6:
                continue
            offsets: list[tuple[float, float] | None] = [None] * n
            row_y = -0.5 * total_depth
            for row, width, row_half in zip(rows, widths, row_depths):
                row_y += row_half
                across = -0.5 * width
                for index in row:
                    across += across_halves[index]
                    offsets[index] = (across, inset + row_y)
                    across += across_halves[index] + gap
                row_y += row_half + gap
            packed = [offset for offset in offsets if offset is not None]
            if seat_rectangles_fit(
                packed,
                across_halves,
                depth_halves,
                span,
                depth,
                inset,
                gap,
            ):
                return packed, gap
        if items is not None:
            freeform = _freeform_seat_offsets(
                items,
                across_halves,
                depth_halves,
                span,
                depth,
                inset,
                gap,
            )
            if freeform is not None:
                return freeform, gap
    return None


def _seat_body_geometry(world: World, items, fwd, perp, quarter_turn: bool):
    bodies = {item.key: "/".join(staged_body(world, item)) for item in items}
    bbox_halves = [
        tuple(0.5 * float(value) for value in staged_bbox(world, item)[:2])
        for item in items
    ]
    if quarter_turn:
        bbox_halves = [(values[1], values[0]) for values in bbox_halves]
    across_halves = [
        abs(perp[0]) * values[0] + abs(perp[1]) * values[1] for values in bbox_halves
    ]
    depth_halves = [
        abs(fwd[0]) * values[0] + abs(fwd[1]) * values[1] for values in bbox_halves
    ]
    return bodies, bbox_halves, across_halves, depth_halves


def fit_seat_offsets(items, bbox_halves, across_halves, depth_halves, span, depth, inset):
    halves = [max(values) for values in bbox_halves]
    pitch = max(0.20, (max(halves) * 2.0 + _SEAT_GAP_M) if len(items) > 1 else 0.0)
    offsets = _seat_offsets(
        [item.spot for item in items], halves, span, depth, inset, pitch
    )
    if seat_rectangles_fit(
        offsets,
        across_halves,
        depth_halves,
        span,
        depth,
        inset,
        _SEAT_GAP_M,
    ):
        return offsets, _SEAT_GAP_M, False
    packed = _packed_seat_offsets(
        across_halves,
        depth_halves,
        span,
        depth,
        inset,
        items=items,
    )
    return (*packed, True) if packed is not None else None


def _bump_largest_staged_body(world: World, items) -> tuple[str, str] | None:
    import behavior_import as bi

    if world.recipe.frozen_bindings:
        return None
    candidates = []
    seen = set()
    for item in items:
        base = (item.cat, item.model)
        if base in seen:
            continue
        seen.add(base)
        ladder = body_ladder(*base)
        index = int(world.item_bump.get(base, 0))
        if index + 1 >= len(ladder):
            continue
        size = bi.read_spec("/".join(ladder[index])).get("bbox_size") or [0.2, 0.2, 0.2]
        preserve_mount = getattr(world.recipe, "cue_mode", "none") != "none"
        mount = choose_item_decal_mount(size, roundness=0.0)
        for next_index in range(index + 1, len(ladder)):
            next_size = bi.read_spec("/".join(ladder[next_index])).get(
                "bbox_size"
            ) or [0.2, 0.2, 0.2]
            if preserve_mount and choose_item_decal_mount(
                next_size, roundness=0.0
            ) != mount:
                continue
            candidates.append(
                (
                    float(size[0]) * float(size[1]),
                    base,
                    next_index,
                    ladder[next_index],
                )
            )
            break
    if not candidates:
        return None
    _area, base, next_index, next_body = max(
        candidates, key=lambda row: (row[0], row[1])
    )
    world.item_bump[base] = int(next_index)
    return "/".join(base), "/".join(next_body)


def _resolve_seat_layout(world: World, items, fwd, perp, span, depth, inset):
    while True:
        for quarter_turn in (False, True):
            bodies, bbox_halves, across_halves, depth_halves = _seat_body_geometry(
                world, items, fwd, perp, quarter_turn
            )
            fit = fit_seat_offsets(
                items,
                bbox_halves,
                across_halves,
                depth_halves,
                span,
                depth,
                inset,
            )
            if fit is not None:
                offsets, gap, wrapped = fit
                quat = (
                    (0.70710678, 0.0, 0.0, 0.70710678)
                    if quarter_turn
                    else (1.0, 0.0, 0.0, 0.0)
                )
                return (
                    bodies,
                    offsets,
                    gap,
                    wrapped,
                    {item.key: quat for item in items},
                )
        bumped = _bump_largest_staged_body(world, items)
        if bumped is None:
            raise SpecBuildError(
                f"{world.recipe.task_id}: {len(items)} staged objects do not fit on the "
                "selected support without overlap"
            )
        print(f"[bench] STAGING FIT: {bumped[0]} -> {bumped[1]}")


def seat_items(world: World, support_pp: str, yaw: float, only=None) -> None:
    """Row the graspables on the support's COLLISION solid, on the approach side, spaced by
    their own bboxes -- never on the visual overhang, which has no collider to settle on.

    An ``ItemRecipe.spot`` overrides the row for that body (``_seat_offsets``). The anti-cheat
    shuffle in ``staged_order`` needs no exception for it: ``_item_blocks`` consults that shuffle
    only when ``recipe.support_items`` is empty, and every spec that can carry a spot is a bts row,
    which fills ``support_items`` from its fixture-backed ``on:<surface>`` places. Children of a
    movable plate or tray are seated later by ``seat_movable_support_items``."""
    import behavior_import as bi
    from scene_state import Role

    (cmn, cmx) = bi._collision_bbox(world.stage, support_pp) or _wbbox(
        world.stage, support_pp
    )
    cx, cy = 0.5 * (cmn[0] + cmx[0]), 0.5 * (cmn[1] + cmx[1])
    hx, hy = 0.5 * (cmx[0] - cmn[0]), 0.5 * (cmx[1] - cmn[1])
    top = float(cmx[2])
    a = math.radians(yaw)
    fwd = (math.sin(a), -math.cos(a))
    perp = (-fwd[1], fwd[0])
    depth = min(hx / max(abs(fwd[0]), 1e-6), hy / max(abs(fwd[1]), 1e-6))
    span = min(hx / max(abs(perp[0]), 1e-6), hy / max(abs(perp[1]), 1e-6))
    inset = max(0.08, depth * 0.45)
    ex, ey = cx - fwd[0] * inset, cy - fwd[1] * inset
    items = list(world.recipe.staged() if only is None else only)
    bodies, offsets, packed_gap, wrapped, orientations = _resolve_seat_layout(
        world, items, fwd, perp, span, depth, inset
    )
    world.scene.grasp_objs = list(getattr(world.scene, "grasp_objs", None) or []) + [
        {
            "key": it.key,
            "path": f"/World/Obj_{bi._safe(it.key)}",
            "build": staged_build_params(world, it),
            "orientation": orientations[it.key],
            "initial_place": it.initial_place,
            "support_path": support_pp,
        }
        for it in items
    ]
    if wrapped:
        print(
            f"[bench] staging wrap: {len(items)} object(s) on {support_pp} use "
            f"{len({round(value[1], 4) for value in offsets})} rows at "
            f"{packed_gap * 1000:.0f}mm clearance"
        )
    for it, (off, into) in zip(items, offsets):
        sx = ex + perp[0] * off + fwd[0] * into
        sy = ey + perp[1] * off + fwd[1] * into
        asset_category, asset_model = bodies[it.key].split("/", 1)
        bbox = staged_bbox(world, it)
        foot_half = (
            (max(0.12, 0.5 * bbox[0]), max(0.12, 0.5 * bbox[1]))
            if _scale_xyz(it.scale) != (1.0, 1.0, 1.0)
            else (0.12, 0.12)
        )
        world.scene.register(
            it.key,
            Role.GRASPABLE,
            xy=(float(sx), float(sy)),
            z=float(top) + 0.05,
            prim_path=f"/World/Obj_{bi._safe(it.key)}",
            approach_yaw=float(yaw),
            foot_xy=(float(sx), float(sy)),
            foot_half=foot_half,
            top_z=float(top),
            clearance=0.34,
            meta={
                "asset_category": asset_category,
                "asset_model": asset_model,
                "asset_scale": _serialized_scale(it.scale),
            },
        )
        register_staged_asset_affordances(world, it.key, asset_category)
        ent = world.scene.entities[it.key]
        ent.meta["rest_orientation"] = orientations[it.key]
        ent.meta["rest_yaw_deg"] = (
            90.0 if abs(float(orientations[it.key][3])) > 0.5 else 0.0
        )
        print(
            f"[bench] {it.key} = {bodies[it.key]} @ ({sx:.2f},{sy:.2f},{top + 0.05:.2f})"
        )
    if only is None:
        world.scene.behavior_support_prims = [support_pp]


def seat_floor_items(world: World) -> None:
    """Place every ``floor:<room>`` graspable on measured clear floor."""
    items = [
        item
        for item in world.recipe.staged()
        if item.initial_place.startswith("floor:")
    ]
    if not items:
        return
    context = (world.room, world.box, world.anchor)
    by_room: dict[str, list[ItemRecipe]] = {}
    for item in items:
        by_room.setdefault(item.initial_place[6:] or world.recipe.room, []).append(item)
    for room, room_items in by_room.items():
        world.room = room
        world.box = room_box(world.recipe.scene, room)
        world.anchor = room_anchor(world.recipe.scene, room)
        cells, boxes = _grid(world)
        for item in room_items:
            _seat_floor_item(world, item, room, cells, boxes)
    world.room, world.box, world.anchor = context


def _seat_floor_item(world: World, item: ItemRecipe, room: str, cells, boxes) -> None:
    import behavior_import as bi
    from scene_state import Role

    cat, model = staged_body(world, item)
    bbox = staged_bbox(world, item)
    half = 0.5 * max(float(bbox[0]), float(bbox[1]))
    slot, reserve = _find_slot(world, cells, half, near=world.anchor, dcap_max=5.0)
    if slot is None:
        raise SpecBuildError(
            f"{world.recipe.task_id}: room={room} phase=floor item={item.key} "
            f"detail=no clear slot for {cat}/{model}. The authored spot field does not "
            "control physical packing; move floor items onto reachable surfaces, move the "
            "interaction to a room/scene with enough measured floor, or reduce only unscored "
            "redundant clutter"
        )
    x, y = slot
    yaw = _best_yaw(boxes, x, y)
    path = f"/World/Obj_{bi._safe(item.key)}"
    world.scene.grasp_objs = list(getattr(world.scene, "grasp_objs", None) or []) + [
        {
            "key": item.key,
            "path": path,
            "build": staged_build_params(world, item),
            "initial_place": item.initial_place,
            "support_path": None,
        }
    ]
    world.scene.register(
        item.key,
        Role.GRASPABLE,
        xy=(float(x), float(y)),
        z=float(world.scene.floor_z) + 0.05,
        prim_path=path,
        approach_yaw=float(yaw),
        foot_xy=(float(x), float(y)),
        foot_half=(half, half),
        top_z=float(world.scene.floor_z) + float(bbox[2]),
        clearance=max(0.34, half + 0.12),
        meta={
            "asset_category": cat,
            "asset_model": model,
            "asset_scale": _serialized_scale(item.scale),
        },
    )
    register_staged_asset_affordances(world, item.key, cat)
    reservation = (float(x), float(y), float(reserve))
    world.used_slots.append(reservation)
    world.room_slots.setdefault(room, []).append(reservation)
    print(f"[bench] {item.key} = {cat}/{model} on floor in {room} @ ({x:.2f},{y:.2f})")


def seat_movable_support_items(world: World) -> None:
    """Register items on staged movable surfaces after those surfaces have been placed."""
    import behavior_import as bi
    from scene_state import Role

    by_key = {item.key: item for item in world.recipe.staged()}
    for parent_key, children in _movable_support_groups(
        world.recipe.staged(), world.recipe.task_id
    ):
        parent = by_key[parent_key]
        support = world.scene.entities.get(parent_key)
        if support is None or support.xy is None or not support.prim_path:
            raise SpecBuildError(
                f"{world.recipe.task_id}: movable support {parent_key!r} must be "
                "staged before its contents"
            )
        yaw = float((support.meta or {}).get("rest_yaw_deg", 0.0))
        angle = math.radians(yaw)
        fwd = (math.sin(angle), -math.cos(angle))
        perp = (math.cos(angle), math.sin(angle))
        bodies, offsets, packed_gap, wrapped, orientations = _movable_support_layout(
            world, parent, children, yaw
        )
        parent_bbox = staged_bbox(world, parent)
        support_base = (
            float(world.scene.floor_z)
            if parent.initial_place.startswith("floor:")
            else float(support.top_z if support.top_z is not None else support.z)
        )
        support_top = support_base + float(parent_bbox[2])
        specs = []
        for child, (across, into) in zip(children, offsets):
            sx = float(support.xy[0] + perp[0] * across + fwd[0] * into)
            sy = float(support.xy[1] + perp[1] * across + fwd[1] * into)
            path = f"/World/Obj_{bi._safe(child.key)}"
            specs.append(
                {
                    "key": child.key,
                    "path": path,
                    "build": staged_build_params(world, child),
                    "orientation": orientations[child.key],
                    "initial_place": child.initial_place,
                    "support_path": str(support.prim_path),
                    "movable_support": parent_key,
                    "movable_support_offset_xy": (
                        float(sx - support.xy[0]),
                        float(sy - support.xy[1]),
                    ),
                }
            )
            asset_category, asset_model = bodies[child.key].split("/", 1)
            bbox = staged_bbox(world, child)
            foot_half = (
                (max(0.12, 0.5 * bbox[0]), max(0.12, 0.5 * bbox[1]))
                if _scale_xyz(child.scale) != (1.0, 1.0, 1.0)
                else (0.12, 0.12)
            )
            world.scene.register(
                child.key,
                Role.GRASPABLE,
                xy=(sx, sy),
                z=support_top + 0.05,
                prim_path=path,
                approach_yaw=float(support.approach_yaw or 0.0),
                foot_xy=(sx, sy),
                foot_half=foot_half,
                top_z=support_top,
                clearance=0.34,
                meta={
                    "asset_category": asset_category,
                    "asset_model": asset_model,
                    "asset_scale": _serialized_scale(child.scale),
                    "initial_movable_support": parent_key,
                },
            )
            register_staged_asset_affordances(world, child.key, asset_category)
            entity = world.scene.entities[child.key]
            entity.meta["rest_orientation"] = orientations[child.key]
            entity.meta["rest_yaw_deg"] = (
                90.0 if abs(float(orientations[child.key][3])) > 0.5 else 0.0
            )
            print(
                f"[bench] {child.key} = {bodies[child.key]} on movable "
                f"{parent_key} @ ({sx:.2f},{sy:.2f},{support_top + 0.05:.2f})"
            )
        world.scene.grasp_objs = list(
            getattr(world.scene, "grasp_objs", None) or []
        ) + specs
        if wrapped:
            print(
                f"[bench] movable staging wrap: {len(children)} object(s) on "
                f"{parent_key} use {len({round(value[1], 4) for value in offsets})} "
                f"rows at {packed_gap * 1000:.0f}mm clearance"
            )


# =====================================================================================
# stage 4a -- PRESTAGE: stock that is already inside a unit at t=0
# =====================================================================================
def _preload_recipe_counts(
    recipe: Recipe, seed: int, numbers: dict[str, dict[str, int]]
) -> tuple[int, ...]:
    import random

    rng = random.Random(f"{recipe.task_id}:preload:{int(seed)}")
    counts: list[int] = []
    for pr in recipe.preloads:
        if pr.from_number:
            chan = numbers.get(pr.from_number) or {}
            if pr.unit not in chan:
                raise SpecBuildError(
                    f"preload for {pr.unit} needs channel {pr.from_number!r}, drew {sorted(chan)}"
                )
            count = int(chan[pr.unit])
            if pr.keys and len(pr.keys) != count:
                raise SpecBuildError(
                    f"preload for {pr.unit} names {len(pr.keys)} objects but "
                    f"channel {pr.from_number!r} drew {count}"
                )
        elif pr.keys:
            count = len(pr.keys)
        else:
            count = rng.randint(int(pr.lo), int(pr.hi))
        counts.append(count)
    return tuple(counts)


def preload_counts(
    recipe: Recipe, seed: int, numbers: dict[str, dict[str, int]]
) -> dict[str, int]:
    """unit key -> how many fillers it starts with. Seeded, never authored. ``from_number`` pins the
    count to a DRAWN, PRINTED number (a bay that its own tag says is full really is full)."""
    out: dict[str, int] = {}
    for pr, count in zip(
        recipe.preloads, _preload_recipe_counts(recipe, seed, numbers)
    ):
        out[pr.unit] = out.get(pr.unit, 0) + count
    return out


def _named_preload_shelves(
    world: World, unit: str
) -> dict[str, tuple[float, float, float, float, float]]:
    ent = world.scene.entities[unit]
    top = float(getattr(ent, "top_z", 0.0) or (float(ent.z) + float(ent.height)))
    return {
        str(name): (
            float(top_z) + 0.004,
            float(cx),
            float(cy),
            float(hx),
            float(hy),
        )
        for name, top_z, _bottom_z, cx, cy, hx, hy in (
            getattr(ent, "spaces", None) or []
        )
        if float(top_z) <= top - 0.02
    }


def _preload_shelves(
    world: World, unit: str
) -> list[tuple[float, float, float, float, float]]:
    """Every board the filler stock may rest on, roomiest first, as (base_z, cx, cy, hx, hy).

    ALL of them, not just the largest. Filling one shelf in a single row and keeping its middle
    lane clear needs about three object-widths of board, so a cabinet that plainly holds two
    microscopes -- one per shelf, which is where a person would put them -- refused to stage them
    at all: "cannot seat 2 filler(s) of half 0.134 m in a 0.265 m interior". Spreading down the
    shelves is both what the furniture is for and what the room would look like.

    The unit's own top face is excluded (an object there is ON the lid, not inside). A front-loading
    drum intentionally has no extracted shelf, so its fallback uses the measured aperture drop;
    other units without a measured board keep using their interior floor."""
    ent = world.scene.entities[unit]
    named = _named_preload_shelves(world, unit)
    if named:
        shelves = list(named.values())
        return order_preload_shelves(shelves, float(world.scene.floor_z))
    meta = getattr(ent, "meta", None) or {}
    if meta.get("access_mode") == "front_drum":
        drop_z = getattr(ent, "drop_z", None)
        if drop_z is None or not math.isfinite(float(drop_z)):
            aperture = meta.get("aperture_center")
            drop_z = aperture[2] if aperture is not None and len(aperture) >= 3 else None
        if drop_z is None or not math.isfinite(float(drop_z)):
            raise SpecBuildError(f"front-drum preload {unit!r} has no aperture drop")
        return [
            (
                float(drop_z),
                float(ent.xy[0]),
                float(ent.xy[1]),
                float(ent.half[0]),
                float(ent.half[1]),
            )
        ]
    return [(float(ent.z) + 0.01, ent.xy[0], ent.xy[1], ent.half[0], ent.half[1])]


def order_preload_shelves(
    shelves: Sequence[tuple[float, float, float, float, float]], floor_z: float
) -> list[tuple[float, float, float, float, float]]:
    """Put retrievable waist/chest-height boards before floor or overhead storage.

    Capacity is still handled by ``preload_placements`` across every board.  This ordering only
    decides where the first named objects go, so a cabinet with spare shelves no longer hides a
    later ``Grab`` target on its largest but nearly floor-level interior merely because that board
    has a few more square centimetres of area.
    """

    def rank(shelf):
        base_z, _cx, _cy, hx, hy = shelf
        height = float(base_z) - float(floor_z)
        comfortable = 0 if 0.45 <= height <= 1.30 else 1
        return comfortable, abs(height - 0.85), -float(hx) * float(hy)

    return sorted(shelves, key=rank)


def _preload_shelf(world: World, unit: str):
    """The roomiest board -- kept for callers that want one shelf. See ``_preload_shelves``."""
    return _preload_shelves(world, unit)[0]


def _preload_offsets(
    n: int, span: float, half: float, keep_lane: bool = True
) -> list[float]:
    """Offsets along the unit's long interior axis.

    With ``keep_lane`` the fillers alternate OUTWARD from the centre so the middle -- where
    ``PutInside`` releases -- stays free, because a filler under the drop point is interpenetrated
    by the very object the row scores.

    Without it they are packed centred, which is what a shelf nobody puts anything into should
    look like. Reserving the lane unconditionally cost a whole board: the first filler sits a full
    pitch off centre, so a shelf with room for one or two objects seated NONE of them, and
    NARRATE-3 -- whose cabinet is only ever opened to take microscopes OUT -- could not be staged.
    """
    pitch = 2.0 * half + 0.03
    limit = span - half - 0.02
    out: list[float] = []
    for i in range(n):
        if keep_lane:
            o = pitch * ((i // 2) + 1) * (1.0 if i % 2 == 0 else -1.0)
        else:
            o = pitch * (i - (n - 1) / 2.0)
        if abs(o) > limit:
            raise SpecBuildError(
                f"cannot seat {n} filler(s) of half {half:.3f} m in a {span:.3f} m interior"
                + (" without covering the drop lane" if keep_lane else "")
            )
        out.append(o)
    return out


def _preload_capacity(span: float, half: float, keep_lane: bool = True) -> int:
    """How many fillers one board takes, asked forwards instead of found by an exception."""
    pitch = 2.0 * half + 0.03
    limit = span - half - 0.02
    if pitch <= 0 or limit <= 0:
        return 0
    if keep_lane:
        return max(0, 2 * int(limit / pitch))
    n = 0
    while n < 64 and abs(pitch * (n - n / 2.0)) <= limit:
        n += 1
    return n


def _receives_puts(world: World, unit: str) -> bool:
    """Does anything get released INTO this unit during the episode?

    Read off the row's own program and answer key. A unit that is only ever opened and emptied has
    no drop lane to protect, and pretending it does throws away the middle of every shelf."""
    spec = world.spec or {}
    if f"<{unit}>" in str(spec.get("human_program") or "") and "[PutInside]" in str(
        spec.get("human_program") or ""
    ):
        for verb, args in _spec_steps(str(spec.get("human_program") or "")):
            if verb.upper() == "PUTINSIDE" and len(args) > 1 and args[1] == unit:
                return True
    return any(
        str(row.get("pred") or "").lower() == "inside"
        and str(row.get("target")) == unit
        for row in (spec.get("answer_key") or []) + (spec.get("goals") or [])
    )


def _preloads_clear_before_first_put(world: World, unit: str, entries) -> bool:
    """Whether every named preload leaves ``unit`` before its first incoming placement."""
    preload_keys = {str(entry[1]) for entry in entries}
    if not preload_keys:
        return False
    removed = set()
    for verb, args in _spec_steps(str((world.spec or {}).get("human_program") or "")):
        action = verb.upper()
        if action == "GRAB" and args and args[0] in preload_keys:
            removed.add(args[0])
        elif action == "PUTINSIDE" and len(args) > 1 and args[1] == unit:
            return removed == preload_keys
    return False


def _spec_steps(text: str) -> list[tuple[str, list[str]]]:
    return [
        (m.group(1), re.findall(r"<([^>]+)>", m.group(2)))
        for m in re.finditer(r"\[(\w+)\]((?:\s*<[^>]+>\s*\(\d+\))*)", text)
    ]


def _seat_prim_base(
    world: World,
    prim,
    pp: str,
    x: float,
    y: float,
    base_z: float,
    orientation=(1.0, 0.0, 0.0, 0.0),
    *,
    center_xy: bool = False,
    scale: float | tuple[float, float, float] = 1.0,
) -> None:
    """Put a prim down with its measured BASE flush on ``base_z`` (the same two-step measure-and-
    correct ``_drop_to_floor`` uses for the units)."""
    import fpv_scene

    fpv_scene.set_world_pose(
        prim, (float(x), float(y), float(base_z)), orientation, scale
    )
    (mnx, mny, got), (mxx, mxy, _top) = _wbbox(world.stage, pp)
    correction_x = float(x) - 0.5 * (float(mnx) + float(mxx)) if center_xy else 0.0
    correction_y = float(y) - 0.5 * (float(mny) + float(mxy)) if center_xy else 0.0
    correction_z = float(base_z) - float(got)
    if (
        abs(correction_x) > 0.002
        or abs(correction_y) > 0.002
        or abs(correction_z) > 0.002
    ):
        fpv_scene.set_world_pose(
            prim,
            (
                float(x) + correction_x,
                float(y) + correction_y,
                float(base_z) + correction_z,
            ),
            orientation,
            scale,
        )


def _group_preload_entries(
    world: World,
) -> dict[str, list[tuple[PreloadRecipe, str, list[float], float]]]:
    import behavior_import as bi

    grouped: dict[str, list[tuple[PreloadRecipe, str, list[float], float]]] = {}
    counts = _preload_recipe_counts(world.recipe, world.seed, world.numbers)
    generated_index: dict[str, int] = {}
    reserved_keys = {
        key for preload in world.recipe.preloads for key in preload.keys
    } | {item.key for item in world.recipe.staged()}
    used_keys: set[str] = set()
    for pr, count in zip(world.recipe.preloads, counts):
        bb = list(
            bi.read_spec(f"{pr.cat}/{pr.model}").get("bbox_size") or [0.1, 0.1, 0.15]
        )
        half = 0.5 * max(float(bb[0]), float(bb[1]))
        for index in range(count):
            if index < len(pr.keys):
                key = pr.keys[index]
                if key in used_keys:
                    raise SpecBuildError(f"duplicate preload object key {key!r}")
            else:
                next_index = generated_index.get(pr.unit, 0)
                key = f"pre_{pr.unit}_{next_index}"
                while key in reserved_keys or key in used_keys:
                    next_index += 1
                    key = f"pre_{pr.unit}_{next_index}"
                generated_index[pr.unit] = next_index + 1
            used_keys.add(key)
            grouped.setdefault(pr.unit, []).append((pr, key, bb, half))
    return grouped


def seat_preloads(world: World) -> None:
    """Seat the filler stock INSIDE its unit and register each piece as a scoreable GRASPABLE.

    Registered as GRASPABLE with a real prim on purpose: ``_relations_inside_live`` only counts
    graspables, and ``capacity`` counts INSIDE pairs -- filler the scorer cannot see would make
    ``cap = preload + key_load`` a number with no world behind it."""
    grouped = _group_preload_entries(world)

    movable_targets = set(((world.spec.get("world") or {}).get("objects") or {}))
    for unit, entries in grouped.items():
        if unit in movable_targets:
            pinned = [
                entry[1]
                for entry in entries
                if getattr(entry[0], "initial_space", "")
            ]
            if pinned:
                raise SpecBuildError(
                    f"movable container {unit} cannot honor initial_space for {pinned}"
                )
            _defer_movable_preloads(world, unit, entries)
            continue
        _seat_fixed_preloads(world, unit, entries)


def mount_preload_decals(world: World, pngs: dict[str, str]) -> None:
    """Mount authored printed cues on named objects that start inside a fixed container."""
    if world.recipe.cue_mode != "printed":
        return
    for preload in world.recipe.preloads:
        for key in preload.keys:
            png = pngs.get(f"item:{key}")
            if not png:
                continue
            entity = world.scene.entities.get(key)
            if entity is None or not entity.prim_path:
                raise SpecBuildError(
                    f"labeled preload {key!r} was not registered after seating"
                )
            prim = world.stage.GetPrimAtPath(entity.prim_path)
            if not prim or not prim.IsValid():
                raise SpecBuildError(
                    f"labeled preload {key!r} has no materialized prim at {entity.prim_path}"
                )
            mount_item_decal(
                world,
                ItemRecipe(key, preload.cat, preload.model, (), ()),
                png,
                float(entity.approach_yaw or 0.0),
                park_anchor=False,
                mount_face=item_decal_mount_face(world.spec, key),
            )


def mount_staged_item_decals(
    world: World,
    pngs: Mapping[str, str],
    fallback_yaw: float,
) -> None:
    """Mount each staged label upright for that item's own support approach."""
    for item in world.recipe.staged():
        png_key = decal_png_for(
            item,
            world.recipe,
            world.cue_class.get(item.key, ""),
        )
        if png_key not in pngs:
            continue
        entity = world.scene.entities.get(item.key)
        if entity is None:
            raise SpecBuildError(
                f"labeled staged item {item.key!r} was not registered after seating"
            )
        approach_yaw = getattr(entity, "approach_yaw", None)
        if (
            os.environ.get("ISAAC_LEGACY_ASK12802_REPORT_MOUNT_YAW") == "1"
            and world.recipe.task_id == "ASK-12802"
            and item.key == "item_1"
        ):
            approach_yaw = fallback_yaw
        mount_item_decal(
            world,
            item,
            pngs[png_key],
            float(fallback_yaw if approach_yaw is None else approach_yaw),
            mount_face=item_decal_mount_face(world.spec, item.key),
        )


_PRELOAD_STACK_GAP_M = 0.01
_PRELOAD_XY_GAP_M = 0.02
_PRELOAD_EDGE_M = 0.02
_MAX_PRELOAD_ASSIGNMENT_STATES = 5000


def _flat_preload_placements(entries, shelves, keep_lane: bool):
    seated = []
    left = list(entries)
    for base_z, cx, cy, hx, hy in shelves:
        if not left:
            break
        along = 0 if hx >= hy else 1
        span = hx if along == 0 else hy
        take = left[: _preload_capacity(span, max(e[3] for e in left), keep_lane)]
        if not take:
            continue
        offsets = _preload_offsets(
            len(take), span, max(e[3] for e in take), keep_lane
        )
        for entry, offset in zip(take, offsets):
            x, y = (cx + offset, cy) if along == 0 else (cx, cy + offset)
            bbox = entry[2]
            if (
                0.5 * float(bbox[0]) + _PRELOAD_EDGE_M > float(hx)
                or 0.5 * float(bbox[1]) + _PRELOAD_EDGE_M > float(hy)
            ):
                return None
            seated.append((entry, (base_z, x, y, 0.0)))
        left = left[len(take) :]
    return None if left else seated


def _preload_shelf_ceilings(shelves, top_z: float) -> list[float]:
    ceilings = []
    for base_z, _cx, _cy, _hx, _hy in shelves:
        higher = [
            other[0] - 0.03 for other in shelves if other[0] > base_z + 0.03
        ]
        ceilings.append(min([float(top_z) - 0.03, *higher]))
    return ceilings


def _stack_preload_entries(entries, shelves, ceilings):
    groups: dict[tuple, list] = {}
    for entry in entries:
        preload, _key, bbox, _half = entry
        if len(bbox) < 3 or min(float(value) for value in bbox[:3]) <= 0.0:
            raise SpecBuildError(f"invalid preload bbox {bbox!r}")
        dims = tuple(float(value) for value in bbox[:3])
        groups.setdefault((preload.cat, preload.model, dims), []).append(entry)
    clearance = max(
        ceiling - shelf[0] for shelf, ceiling in zip(shelves, ceilings)
    )
    stacks = []
    for (_cat, _model, dims), grouped in groups.items():
        width, depth, height = dims
        capacity = int(
            (clearance + _PRELOAD_STACK_GAP_M)
            / (height + _PRELOAD_STACK_GAP_M)
        )
        if capacity < 1:
            raise SpecBuildError(
                f"preload height {height:.3f} m exceeds available clearance {clearance:.3f} m"
            )
        for start in range(0, len(grouped), capacity):
            chunk = grouped[start : start + capacity]
            stack_height = len(chunk) * height + (len(chunk) - 1) * _PRELOAD_STACK_GAP_M
            stacks.append(
                {
                    "entries": chunk,
                    "hx": 0.5 * width,
                    "hy": 0.5 * depth,
                    "height": stack_height,
                }
            )
    return stacks


def _preload_xy_candidates(
    shelf, half_x: float, half_y: float, keep_lane: bool
):
    _base_z, cx, cy, hx, hy = shelf
    sx, sy = float(half_x), float(half_y)
    if sx + _PRELOAD_EDGE_M > hx or sy + _PRELOAD_EDGE_M > hy:
        return []
    xs = [cx - hx + _PRELOAD_EDGE_M + sx, cx + hx - _PRELOAD_EDGE_M - sx, cx]
    ys = [cy - hy + _PRELOAD_EDGE_M + sy, cy + hy - _PRELOAD_EDGE_M - sy, cy]
    candidates = list(dict.fromkeys((float(x), float(y)) for x in xs for y in ys))
    if keep_lane:
        lane_x, lane_y = max(0.04, 0.20 * hx), max(0.04, 0.20 * hy)
        candidates = [
            (x, y)
            for x, y in candidates
            if abs(x - cx) >= sx + lane_x + _PRELOAD_XY_GAP_M
            or abs(y - cy) >= sy + lane_y + _PRELOAD_XY_GAP_M
        ]
    return sorted(candidates, key=lambda xy: -(abs(xy[0] - cx) + abs(xy[1] - cy)))


def _preload_rect_clear(x, y, half_x: float, half_y: float, occupied) -> bool:
    return all(
        abs(x - ox) >= float(half_x) + ohx + _PRELOAD_XY_GAP_M
        or abs(y - oy) >= float(half_y) + ohy + _PRELOAD_XY_GAP_M
        for ox, oy, ohx, ohy in occupied
    )


def _preload_orientations(stack) -> list[tuple[float, float, float]]:
    half_x, half_y = float(stack["hx"]), float(stack["hy"])
    orientations = [(half_x, half_y, 0.0)]
    if abs(half_x - half_y) > 1e-6:
        orientations.append((half_y, half_x, 90.0))
    return orientations


def _preload_candidate_slots(
    stack,
    shelves,
    ceilings,
    keep_lane: bool,
    occupied,
    shelf_state,
):
    seen_shelves = set()
    for shelf_index, shelf in enumerate(shelves):
        current_shelf = shelf_state(shelf_index)
        if current_shelf in seen_shelves:
            continue
        seen_shelves.add(current_shelf)
        if shelf[0] + float(stack["height"]) > ceilings[shelf_index] + 1e-9:
            continue
        for half_x, half_y, yaw in _preload_orientations(stack):
            for x, y in _preload_xy_candidates(
                shelf, half_x, half_y, keep_lane
            ):
                if _preload_rect_clear(
                    x, y, half_x, half_y, occupied[shelf_index]
                ):
                    yield shelf_index, x, y, half_x, half_y, yaw


def _assign_preload_stacks(stacks, shelves, ceilings, keep_lane: bool):
    order = sorted(
        range(len(stacks)),
        key=lambda index: (
            -float(stacks[index]["hx"]) * float(stacks[index]["hy"]),
            -float(stacks[index]["height"]),
        ),
    )
    occupied = {index: [] for index in range(len(shelves))}
    assigned = {}
    visited = set()

    def shelf_state(index):
        _base_z, cx, cy, hx, hy = shelves[index]
        return (
            round(float(hx), 5),
            round(float(hy), 5),
            round(float(ceilings[index]) - float(_base_z), 5),
            tuple(
                sorted(
                    (
                        round(float(x) - float(cx), 5),
                        round(float(y) - float(cy), 5),
                        round(float(ohx), 5),
                        round(float(ohy), 5),
                    )
                    for x, y, ohx, ohy in occupied[index]
                )
            ),
        )

    def place(position: int) -> bool:
        if position == len(order):
            return True
        state = (
            position,
            tuple(sorted(shelf_state(index) for index in range(len(shelves)))),
        )
        if state in visited or len(visited) >= _MAX_PRELOAD_ASSIGNMENT_STATES:
            return False
        visited.add(state)
        stack_index = order[position]
        stack = stacks[stack_index]
        for shelf_index, x, y, half_x, half_y, yaw in _preload_candidate_slots(
            stack,
            shelves,
            ceilings,
            keep_lane,
            occupied,
            shelf_state,
        ):
            assigned[stack_index] = (shelf_index, x, y, yaw)
            occupied[shelf_index].append((x, y, half_x, half_y))
            if place(position + 1):
                return True
            occupied[shelf_index].pop()
            assigned.pop(stack_index, None)
        return False

    if not place(0):
        raise SpecBuildError(
            f"cannot place {len(stacks)} preload stack(s) on {len(shelves)} shelf space(s)"
        )
    return assigned


def _preload_axis_travel(
    dx: float,
    dy: float,
    hx: float,
    hy: float,
    half_x: float,
    half_y: float,
) -> float:
    limits = []
    if abs(dx) > 1e-9:
        limits.append((hx - _PRELOAD_EDGE_M - half_x) / abs(dx))
    if abs(dy) > 1e-9:
        limits.append((hy - _PRELOAD_EDGE_M - half_y) / abs(dy))
    return min(limits, default=0.0)


def _spotted_preload_placements(entries, shelf, top_z: float, front_dir):
    base_z, cx, cy, hx, hy = (float(value) for value in shelf)
    fx, fy = (float(value) for value in (front_dir or (0.0, -1.0)))
    norm = math.hypot(fx, fy)
    if norm <= 1e-9:
        raise SpecBuildError("preload spots require a non-zero container front direction")
    fx, fy = fx / norm, fy / norm
    lx, ly = -fy, fx
    authored = []
    for entry in entries:
        preload, key, bbox, _half = entry
        spot = getattr(preload, "spot", None)
        if spot is None:
            raise SpecBuildError("authored preload spots cannot be mixed with automatic packing")
        half_x, half_y = 0.5 * float(bbox[0]), 0.5 * float(bbox[1])
        height = float(bbox[2])
        if base_z + height > float(top_z) - 0.03 + 1e-9:
            raise SpecBuildError(f"preload {key} exceeds its authored shelf headroom")

        authored.append(
            (
                entry,
                key,
                half_x,
                half_y,
                max(-1.0, min(1.0, float(spot[0]))),
                1.0 - 2.0 * max(0.0, min(1.0, float(spot[1]))),
            )
        )

    across_limits = [
        _preload_axis_travel(lx, ly, hx, hy, half_x, half_y) / abs(u)
        for _entry, _key, half_x, half_y, u, _depth in authored
        if abs(u) > 1e-9
    ]
    depth_limits = [
        _preload_axis_travel(fx, fy, hx, hy, half_x, half_y) / abs(depth)
        for _entry, _key, half_x, half_y, _u, depth in authored
        if abs(depth) > 1e-9
    ]
    max_across = max(0.0, min(across_limits, default=0.0))
    max_depth = max(0.0, min(depth_limits, default=0.0))
    steps = 100
    across_candidates = [max_across * index / steps for index in range(steps, -1, -1)]
    depth_candidates = [max_depth * index / steps for index in range(steps, -1, -1)]
    best = None
    best_score = None
    for across_span in across_candidates:
        for depth_span in depth_candidates:
            occupied = []
            placements = []
            for entry, _key, half_x, half_y, u, depth in authored:
                x = cx + lx * across_span * u + fx * depth_span * depth
                y = cy + ly * across_span * u + fy * depth_span * depth
                if (
                    abs(x - cx) + half_x + _PRELOAD_EDGE_M > hx + 1e-9
                    or abs(y - cy) + half_y + _PRELOAD_EDGE_M > hy + 1e-9
                    or not _preload_rect_clear(x, y, half_x, half_y, occupied)
                ):
                    break
                occupied.append((x, y, half_x, half_y))
                placements.append((entry, (base_z, x, y, 0.0)))
            else:
                across_share = across_span / max_across if max_across > 1e-9 else 0.0
                depth_share = depth_span / max_depth if max_depth > 1e-9 else 0.0
                score = (
                    min(across_share, depth_share),
                    across_share + depth_share,
                    across_share,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best = placements
    if best is None:
        raise SpecBuildError("authored preload spots cannot fit without overlap")
    return best


def _unconstrained_preload_placements(
    entries,
    shelves,
    top_z: float,
    keep_lane: bool,
    *,
    front_dir=None,
    allow_legacy_multi_shelf_preload_spots: bool = False,
):
    if any(getattr(entry[0], "spot", None) is not None for entry in entries):
        if len(shelves) == 1:
            return _spotted_preload_placements(entries, shelves[0], top_z, front_dir)
        if allow_legacy_multi_shelf_preload_spots is not True:
            raise SpecBuildError("authored preload spots require one named shelf")
    flat = _flat_preload_placements(entries, shelves, keep_lane)
    if flat is not None:
        return flat
    ceilings = _preload_shelf_ceilings(shelves, top_z)
    stacks = _stack_preload_entries(entries, shelves, ceilings)
    assigned = _assign_preload_stacks(stacks, shelves, ceilings, keep_lane)
    placements = []
    for stack_index, stack in enumerate(stacks):
        shelf_index, x, y, yaw = assigned[stack_index]
        z = float(shelves[shelf_index][0])
        for entry in stack["entries"]:
            placements.append((entry, (z, x, y, yaw)))
            z += float(entry[2][2]) + _PRELOAD_STACK_GAP_M
    return placements


def preload_placements(
    entries,
    shelves,
    top_z: float,
    keep_lane: bool,
    *,
    named_shelves=None,
    front_dir=None,
    allow_legacy_multi_shelf_preload_spots: bool = False,
):
    """Place preloads, honoring an authored initial container space when present."""
    requested = {
        str(getattr(entry[0], "initial_space", "") or "") for entry in entries
    } - {""}
    if not requested:
        return _unconstrained_preload_placements(
            entries,
            shelves,
            top_z,
            keep_lane,
            front_dir=front_dir,
            allow_legacy_multi_shelf_preload_spots=(
                allow_legacy_multi_shelf_preload_spots
            ),
        )

    by_name = dict(named_shelves or {})
    missing = sorted(requested - set(by_name))
    if missing:
        raise SpecBuildError(
            f"initial preload space(s) {missing} are unavailable; have {sorted(by_name)}"
        )

    pinned: dict[str, list] = {}
    flexible = []
    for entry in entries:
        space = str(getattr(entry[0], "initial_space", "") or "")
        if space:
            pinned.setdefault(space, []).append(entry)
        else:
            flexible.append(entry)

    placements_by_key = {}
    occupied_shelves = {by_name[name] for name in pinned}
    ceilings = _preload_shelf_ceilings(shelves, top_z)
    for name, group in pinned.items():
        shelf = by_name[name]
        shelf_index = next(
            (index for index, candidate in enumerate(shelves) if candidate == shelf),
            None,
        )
        if shelf_index is None:
            raise SpecBuildError(
                f"initial preload space {name!r} is not a usable shelf"
            )
        shelf_top = float(ceilings[shelf_index]) + 0.03
        for entry, placement in _unconstrained_preload_placements(
            group,
            [shelf],
            shelf_top,
            keep_lane,
            front_dir=front_dir,
            allow_legacy_multi_shelf_preload_spots=(
                allow_legacy_multi_shelf_preload_spots
            ),
        ):
            placements_by_key[entry[1]] = (entry, placement)

    if flexible:
        available = [shelf for shelf in shelves if shelf not in occupied_shelves]
        if not available:
            raise SpecBuildError(
                "unassigned preloads have no shelf left after authored initial spaces"
            )
        for entry, placement in _unconstrained_preload_placements(
            flexible,
            available,
            top_z,
            keep_lane,
            front_dir=front_dir,
            allow_legacy_multi_shelf_preload_spots=(
                allow_legacy_multi_shelf_preload_spots
            ),
        ):
            placements_by_key[entry[1]] = (entry, placement)

    return [placements_by_key[entry[1]] for entry in entries]


def _oriented_preload_half_extents(bbox, placement) -> tuple[float, float]:
    half_x, half_y = 0.5 * float(bbox[0]), 0.5 * float(bbox[1])
    yaw = float(placement[3]) if len(placement) > 3 else 0.0
    return (half_y, half_x) if int(round(abs(yaw) / 90.0)) % 2 else (half_x, half_y)


_FRONT_DRUM_PRELOAD_DEPTH_MARGIN_M = 0.015
_FRONT_DRUM_FACE_OUT_TILT_DEG = 75.0
_FRONT_DRUM_VISIBLE_CENTER_DROP_M = 0.075


@dataclass(frozen=True, slots=True)
class _FrontDrumPreloadPose:
    orientation: tuple[float, float, float, float]
    base_z: float
    center_xy: tuple[float, float]
    foot_half: tuple[float, float]
    center_drop: float
    visible_side: str
    aperture_center: tuple[float, float, float]
    entry_axis: tuple[float, float]


def _front_drum_preload_pose(entity, preload, bbox) -> _FrontDrumPreloadPose | None:
    if not bool(getattr(preload, "front_facing", False)):
        return None
    meta = getattr(entity, "meta", None) or {}
    if meta.get("access_mode") != "front_drum":
        raise SpecBuildError("front_facing preload requires a front-drum container")
    aperture = meta.get("aperture_center")
    entry_axis = meta.get("entry_axis")
    if aperture is None or len(aperture) != 3 or entry_axis is None or len(entry_axis) != 2:
        raise SpecBuildError("front_facing preload requires a measured drum aperture")
    ex, ey = float(entry_axis[0]), float(entry_axis[1])
    norm = math.hypot(ex, ey)
    if norm <= 1e-9:
        raise SpecBuildError("front_facing preload has a zero drum entry axis")
    ex, ey = ex / norm, ey / norm
    size_x, size_y, size_z = (float(value) for value in bbox[:3])
    if min(size_x, size_y, size_z) <= 0.0:
        raise SpecBuildError("front_facing preload has a non-positive bounding box")
    if size_z > 0.35 * min(size_x, size_y):
        raise SpecBuildError("front_facing preload must have a thin local-Z face")

    half_long = 0.5 * max(size_x, size_y)
    half_short = 0.5 * min(size_x, size_y)
    half_depth = 0.5 * size_z
    tilt = math.radians(_FRONT_DRUM_FACE_OUT_TILT_DEG)
    sine, cosine = math.sin(tilt), math.cos(tilt)
    vertical_half = sine * half_long + cosine * half_depth
    lateral_half = half_short
    aperture_half_width = float(meta.get("aperture_half_width") or 0.0)
    aperture_half_height = float(meta.get("aperture_half_height") or 0.0)
    if lateral_half + _PRELOAD_EDGE_M > aperture_half_width:
        raise SpecBuildError("front_facing preload is too wide for the drum aperture")
    if vertical_half + _PRELOAD_EDGE_M > aperture_half_height:
        raise SpecBuildError("front_facing preload is too tall for the drum aperture")
    center_drop = min(
        _FRONT_DRUM_VISIBLE_CENTER_DROP_M,
        aperture_half_height - vertical_half - _PRELOAD_EDGE_M,
    )

    inward_half = cosine * half_long + sine * half_depth
    center_xy = (
        float(aperture[0])
        + ex * (inward_half + _FRONT_DRUM_PRELOAD_DEPTH_MARGIN_M),
        float(aperture[1])
        + ey * (inward_half + _FRONT_DRUM_PRELOAD_DEPTH_MARGIN_M),
    )
    phi = math.atan2(ey, ex)
    half_tilt = 0.5 * tilt
    if size_x >= size_y:
        half_yaw = 0.5 * phi
        orientation = (
            math.cos(half_yaw) * math.cos(half_tilt),
            math.sin(half_yaw) * math.sin(half_tilt),
            -math.cos(half_yaw) * math.sin(half_tilt),
            math.sin(half_yaw) * math.cos(half_tilt),
        )
    else:
        half_yaw = 0.5 * (phi - 0.5 * math.pi)
        orientation = (
            math.cos(half_yaw) * math.cos(half_tilt),
            math.cos(half_yaw) * math.sin(half_tilt),
            math.sin(half_yaw) * math.sin(half_tilt),
            math.sin(half_yaw) * math.cos(half_tilt),
        )
    visible_side = "front"
    if bool(getattr(preload, "front_show_back", False)):
        w, x, y, z = orientation
        orientation = (
            (-x, w, z, -y)
            if size_x >= size_y
            else (-y, -z, w, x)
        )
        visible_side = "back"
    foot_half = (
        abs(ex * cosine) * half_long
        + abs(ey) * half_short
        + abs(ex * sine) * half_depth,
        abs(ey * cosine) * half_long
        + abs(ex) * half_short
        + abs(ey * sine) * half_depth,
    )
    return _FrontDrumPreloadPose(
        orientation=orientation,
        base_z=float(aperture[2]) - center_drop - vertical_half,
        center_xy=center_xy,
        foot_half=foot_half,
        center_drop=center_drop,
        visible_side=visible_side,
        aperture_center=tuple(float(value) for value in aperture),
        entry_axis=(ex, ey),
    )


def _center_front_drum_preload(world: World, unit: str, seated):
    if len(seated) != 1:
        return None
    entity = world.scene.entities[unit]
    meta = getattr(entity, "meta", None) or {}
    if meta.get("access_mode") != "front_drum":
        return None
    aperture = meta.get("aperture_center")
    entry_axis = meta.get("entry_axis")
    if aperture is None or len(aperture) < 2 or entry_axis is None or len(entry_axis) != 2:
        raise SpecBuildError(f"front-drum preload {unit!r} has no measured aperture axis")
    ex, ey = float(entry_axis[0]), float(entry_axis[1])
    norm = math.hypot(ex, ey)
    if norm <= 1e-9:
        raise SpecBuildError(f"front-drum preload {unit!r} has a zero entry axis")
    ex, ey = ex / norm, ey / norm
    lateral_x, lateral_y = -ey, ex
    entry, placement = seated[0]
    bbox = entry[2]
    face_out_pose = _front_drum_preload_pose(entity, entry[0], bbox)
    if face_out_pose is not None:
        return [
            (
                entry,
                (
                    face_out_pose.base_z,
                    *face_out_pose.center_xy,
                    *placement[3:],
                ),
            )
        ]
    half_x, half_y = _oriented_preload_half_extents(bbox, placement)
    lateral_half = abs(lateral_x) * half_x + abs(lateral_y) * half_y
    aperture_half_width = float(meta.get("aperture_half_width") or 0.0)
    if (
        aperture_half_width > 0.0
        and lateral_half + _PRELOAD_EDGE_M > aperture_half_width
    ):
        raise SpecBuildError(
            f"preload {entry[1]!r} is too wide for {unit!r}'s measured aperture"
        )
    depth_half = abs(ex) * half_x + abs(ey) * half_y
    depth = depth_half + _FRONT_DRUM_PRELOAD_DEPTH_MARGIN_M
    centered = (
        float(aperture[0]) + ex * depth,
        float(aperture[1]) + ey * depth,
    )
    return [
        (
            entry,
            (placement[0], centered[0], centered[1], *placement[3:]),
        )
    ]


def _frontload_preload_placements(
    world: World,
    unit: str,
    seated,
    shelves,
    *,
    center_front_drum: bool = False,
):
    """Shift each shelf row toward its open face without changing its internal packing."""
    if center_front_drum:
        centered = _center_front_drum_preload(world, unit, seated)
        if centered is not None:
            return centered
    return _shift_preload_rows_toward_front(world, unit, seated, shelves)


def _shift_preload_rows_toward_front(world: World, unit: str, seated, shelves):
    front = world.front_dir.get(unit)
    if front is None:
        return seated
    fx, fy = float(front[0]), float(front[1])
    norm = math.hypot(fx, fy)
    if norm <= 1e-9:
        return seated
    fx, fy = fx / norm, fy / norm
    grouped: dict[int, list[int]] = {}
    for index, ((_preload, _key, _bbox, _half), placement) in enumerate(seated):
        base_z, px, py = placement[:3]
        candidates = [
            shelf_index
            for shelf_index, shelf in enumerate(shelves)
            if float(shelf[0]) <= float(base_z) + 1e-6
            and abs(float(px) - float(shelf[1])) <= float(shelf[3]) + 1e-6
            and abs(float(py) - float(shelf[2])) <= float(shelf[4]) + 1e-6
        ]
        if candidates:
            grouped.setdefault(max(candidates, key=lambda value: shelves[value][0]), []).append(
                index
            )
    shifted = list(seated)
    for shelf_index, indices in grouped.items():
        _z, cx, cy, hx, hy = shelves[shelf_index]
        limits = []
        for index in indices:
            (_preload, _key, bbox, _half), placement = seated[index]
            _base_z, px, py = placement[:3]
            ox, oy = _oriented_preload_half_extents(bbox, placement)
            if fx > 1e-6:
                limits.append((float(cx) + float(hx) - _PRELOAD_EDGE_M - ox - px) / fx)
            elif fx < -1e-6:
                limits.append((float(cx) - float(hx) + _PRELOAD_EDGE_M + ox - px) / fx)
            if fy > 1e-6:
                limits.append((float(cy) + float(hy) - _PRELOAD_EDGE_M - oy - py) / fy)
            elif fy < -1e-6:
                limits.append((float(cy) - float(hy) + _PRELOAD_EDGE_M + oy - py) / fy)
        travel = max(0.0, min(limits, default=0.0))
        for index in indices:
            entry, placement = seated[index]
            base_z, px, py = placement[:3]
            shifted[index] = (
                entry,
                (base_z, px + fx * travel, py + fy * travel, *placement[3:]),
            )
    return shifted


def _seat_fixed_preloads(world: World, unit: str, entries) -> None:
    import behavior_import as bi
    import fpv_scene
    from scene_state import Role

    shelves = _preload_shelves(world, unit)
    named_shelves = _named_preload_shelves(world, unit)
    receives_puts = _receives_puts(world, unit)
    entity = world.scene.entities[unit]
    is_front_drum = (getattr(entity, "meta", None) or {}).get(
        "access_mode"
    ) == "front_drum"
    cleared_before_put = bool(
        receives_puts
        and is_front_drum
        and _preloads_clear_before_first_put(world, unit, entries)
    )
    front_facing = [
        entry for entry in entries if bool(getattr(entry[0], "front_facing", False))
    ]
    if front_facing and (not is_front_drum or len(entries) != 1):
        raise SpecBuildError(
            "front_facing preload requires exactly one object in a front-drum container"
        )
    keep_lane = receives_puts and not cleared_before_put
    if front_facing and keep_lane:
        raise SpecBuildError(
            "front_facing preload must leave the drum before its first incoming placement"
        )
    lane_status = (
        "reserved"
        if keep_lane
        else (
            "free (preloads leave before the first put)"
            if cleared_before_put
            else "free (nothing is put in)"
        )
    )
    print(
        f"[bench] preload {unit}: {len(shelves)} board(s) usable, "
        f"drop lane {lane_status}"
    )
    try:
        seated = preload_placements(
            entries,
            shelves,
            float(world.scene.entities[unit].top_z),
            keep_lane,
            named_shelves=named_shelves,
            front_dir=world.front_dir.get(unit),
            allow_legacy_multi_shelf_preload_spots=(
                (world.spec.get("world") or {}).get(
                    "allow_legacy_multi_shelf_preload_spots"
                )
                is True
            ),
        )
        seated = _frontload_preload_placements(
            world,
            unit,
            seated,
            shelves,
            center_front_drum=cleared_before_put or bool(front_facing),
        )
    except SpecBuildError as exc:
        raise SpecBuildError(f"{unit} cannot hold its preloads: {exc}") from exc
    world.scene.grasp_objs = list(getattr(world.scene, "grasp_objs", None) or [])
    support_path = str(world.scene.resolve(unit).prim_path or "") or None
    keys: list[str] = []
    levels = []
    for (pr, key, bb, _half), placement in seated:
        base_z, px, py = placement[:3]
        yaw = float(placement[3]) if len(placement) > 3 else 0.0
        face_out_pose = _front_drum_preload_pose(entity, pr, bb)
        if face_out_pose is None:
            angle = math.radians(yaw)
            orientation = (
                math.cos(0.5 * angle),
                0.0,
                0.0,
                math.sin(0.5 * angle),
            )
            foot_half = _oriented_preload_half_extents(bb, placement)
        else:
            orientation = face_out_pose.orientation
            foot_half = face_out_pose.foot_half
        pp = f"/World/Obj_{bi._safe(key)}"
        prim = fpv_scene.add_reference(
            world.stage,
            pp,
            bi.baked_usd(f"{pr.cat}/{pr.model}"),
            isolate_xform=True,
        )
        _seat_prim_base(
            world,
            prim,
            pp,
            px,
            py,
            base_z,
            orientation,
            center_xy=face_out_pose is not None,
        )
        if face_out_pose is not None:
            print(
                f"[bench] preload {key}: face-out at the front drum "
                f"({_FRONT_DRUM_FACE_OUT_TILT_DEG:.0f}deg tilt, "
                f"visible={face_out_pose.visible_side}, "
                f"center-drop={face_out_pose.center_drop * 1000.0:.0f}mm)"
            )
        mn, mx = _wbbox(world.stage, pp)
        if face_out_pose is not None:
            actual_center = tuple(
                0.5 * (float(mn[axis]) + float(mx[axis])) for axis in range(3)
            )
            print(
                f"[bench] preload {key}: front-drum actual-center="
                f"({actual_center[0]:.3f},{actual_center[1]:.3f},{actual_center[2]:.3f}) "
                f"bbox-min=({float(mn[0]):.3f},{float(mn[1]):.3f},{float(mn[2]):.3f}) "
                f"bbox-max=({float(mx[0]):.3f},{float(mx[1]):.3f},{float(mx[2]):.3f}) "
                f"orientation-wxyz={tuple(round(value, 6) for value in face_out_pose.orientation)} "
                f"aperture={tuple(round(value, 3) for value in face_out_pose.aperture_center)} "
                f"entry-axis={tuple(round(value, 3) for value in face_out_pose.entry_axis)}"
            )
        world.scene.grasp_objs.append(
            {
                "key": key,
                "path": pp,
                "build": bi.behavior_build_params(f"{pr.cat}/{pr.model}", 1.0),
                "orientation": orientation,
                "initial_place": f"in:{unit}",
                "support_path": support_path,
                "support_z": float(mn[2]),
                "prebuilt": True,
            }
        )
        actual_xy = (0.5 * (float(mn[0]) + float(mx[0])), 0.5 * (float(mn[1]) + float(mx[1])))
        world.scene.register(
            key,
            Role.GRASPABLE,
            xy=actual_xy,
            z=float(mn[2]),
            prim_path=pp,
            approach_yaw=yaw,
            foot_xy=actual_xy,
            foot_half=foot_half,
            top_z=float(mx[2]),
            clearance=0.20,
            meta={"asset_category": pr.cat, "asset_model": pr.model},
        )
        register_staged_asset_affordances(world, key, pr.cat)
        world.scene.declare_inside(key, unit)
        keys.append(key)
        levels.append(float(mn[2]))
    world.preload[unit] = world.preload.get(unit, 0) + len(entries)
    world.preload_keys.setdefault(unit, []).extend(keys)
    print(
        f"[bench] preload {unit}: {len(entries)} object(s) on "
        f"z={min(levels):.2f}..{max(levels):.2f} -> {keys}"
    )


def _defer_movable_preloads(world: World, unit: str, entries) -> None:
    import behavior_import as bi
    from scene_state import Role

    deferred = list(getattr(world.scene, "deferred_preloads", None) or [])
    keys = []
    for pr, key, bbox, half in entries:
        deferred.append(
            {
                "target": unit,
                "key": key,
                "cat": pr.cat,
                "model": pr.model,
                "bbox_size": tuple(float(value) for value in bbox),
                "half": float(half),
            }
        )
        if key not in world.scene.entities:
            target = world.scene.entities.get(unit)
            xy = (
                target.xy
                if target is not None and target.xy is not None
                else world.anchor
            )
            z = (
                float(target.z)
                if target is not None and target.z is not None
                else float(world.scene.floor_z) + 0.5
            )
            world.scene.register(
                key,
                Role.GRASPABLE,
                xy=(float(xy[0]), float(xy[1])),
                z=z,
                prim_path=f"/World/Obj_{bi._safe(key)}",
                foot_half=(float(half), float(half)),
                clearance=0.20,
                meta={"asset_category": pr.cat, "asset_model": pr.model},
            )
        keys.append(key)
    world.scene.deferred_preloads = deferred
    world.preload[unit] = world.preload.get(unit, 0) + len(entries)
    world.preload_keys.setdefault(unit, []).extend(keys)
    print(
        f"[bench] preload {unit}: defer {len(entries)} movable-target object(s) -> {keys}"
    )


# =====================================================================================
# stage 4b -- ASSEMBLY rows: the flat kit, the manual, the letters
# =====================================================================================
# Half-width of the square the whole kit + the manual + the start must fit in. The hull is clamped
# to this around the anchor so ``_grid`` stays a few thousand cells: hall_arch_wood's floor prim is
# ~100 m across and a full-hull lattice is ~80 k cells rebuilt per part.
ASSEMBLY_SPAN_M: float = 7.0


def floor_box(world: World) -> tuple[float, float, float, float] | None:
    """MEASURED world AABB of the room's own ``floors`` prim(s).

    ``room_box``'s furniture hull is the right answer for a furnished room and the WRONG one for an
    empty hall: all seven of ``empty_room_0``'s non-structural prims are pillars on a single line, so
    the hull degenerates to a zero-height strip and the slot search has nowhere to look."""
    names = room_prims(world.recipe.scene, world.room)
    got = []
    for nm in names:
        ent = world.placed.get(nm)
        if not ent or str((ent[1] or {}).get("cat")) != "floors":
            continue
        (mn, mx) = _wbbox(world.stage, ent[0])
        got.append((mn[0], mn[1], mx[0], mx[1]))
    if not got:
        return None
    return (
        min(b[0] for b in got),
        min(b[1] for b in got),
        max(b[2] for b in got),
        max(b[3] for b in got),
    )


def assembly_anchor(world: World, fallback: tuple[float, float]) -> tuple[float, float]:
    """Centre of the room's own floor slab, else the composed scene centroid. The centroid alone is
    the wrong anchor for a multi-room shell: ``hall_arch_wood``'s object mass is its bathroom, so the
    scene centroid can sit in a room the flat-pack does not fit in."""
    fb = floor_box(world)
    if fb is None:
        print(
            f"[bench][WARN] no floors prim in {world.room}; anchoring on the scene centroid"
        )
        return fallback
    return (0.5 * (fb[0] + fb[2]), 0.5 * (fb[1] + fb[3]))


def assembly_hull(
    world: World, span: float = ASSEMBLY_SPAN_M
) -> tuple[float, float, float, float]:
    """Workable hull = the anchor's ``span`` window, clipped to the room floor when one is measurable."""
    ax, ay = world.anchor
    lo = [ax - span, ay - span]
    hi = [ax + span, ay + span]
    fb = floor_box(world)
    if fb is not None:
        lo = [max(lo[0], fb[0] + 0.7), max(lo[1], fb[1] + 0.7)]
        hi = [min(hi[0], fb[2] - 0.7), min(hi[1], fb[3] - 0.7)]
        print(f"[bench] room floor AABB {[round(v, 2) for v in fb]} (inset 0.70 m)")
    if hi[0] - lo[0] < 2.0 or hi[1] - lo[1] < 2.0:
        raise SpecBuildError(
            f"assembly hull {lo}..{hi} is too small to lay a flat-pack out in"
        )
    return (lo[0], lo[1], hi[0], hi[1])


def site_parts(
    world: World,
    points: dict[str, dict],
    supports: list[tuple[str, float]] | None = None,
) -> None:
    """Place assembly parts on their declared floor or support staging locations."""
    roots = world.root_keys or (world.root_key,)
    initial_children = {
        str(row.get("child") or "")
        for row in (world.spec.get("world") or {}).get("initial_attachments") or []
    }
    structural = [mate.child for mate in world.mates if mate.child in initial_children]
    ordered = [
        *roots,
        *structural,
        *sorted(set(world.part_of) - set(roots) - set(structural)),
    ]
    floor_parts = [world.part_of[key] for key in ordered if not world.part_of[key].initial_place.startswith("on:")]
    _site_floor_parts(world, floor_parts, points)
    _site_surface_parts(world, ordered, supports or [])
    fixed = {*roots, *structural}
    for key in ordered:
        entity = world.scene.entities[key]
        entity.attach = points[key]
        entity.meta["post_assembly_affordances"] = list(world.part_of[key].affordances)
        if key in fixed:
            entity.meta["inspect_in_place"] = True
        world.reg_pp[key] = str(entity.prim_path)
    world.scene.assembly_root = world.root_key
    world.scene.assembly_roots = tuple(roots)
    world.scene.assembly_required = {
        mate.child: mate.parent for mate in world.mates
    }
    _configure_assembly_affordances(world)


def _site_floor_parts(
    world: World, parts: list[PartRecipe], points: dict[str, dict]
) -> None:
    if not parts:
        return
    import behavior_import as bi
    from scene_state import Role

    cells, _boxes = _grid(world)
    near = None
    fz = float(world.scene.floor_z)
    world.scene.grasp_objs = list(getattr(world.scene, "grasp_objs", None) or [])
    for part in parts:
        key = part.key
        bb = bi.read_spec(f"{part.cat}/{part.model}").get("bbox_size") or [
            0.3,
            0.3,
            0.05,
        ]
        fh = 0.5 * max(float(bb[0]), float(bb[1]))
        slot, got = _find_slot(world, cells, fh, near=near, dcap_max=ASSEMBLY_SPAN_M)
        if slot is None:
            raise SpecBuildError(f"no open floor in {world.room} for board {key}")
        pp = f"/World/Obj_{bi._safe(key)}"
        world.scene.grasp_objs.append(
            {
                "key": key,
                "path": pp,
                "build": bi.behavior_build_params(f"{part.cat}/{part.model}", 1.0),
                "initial_place": part.initial_place or f"floor:{world.room}",
                "support_path": None,
            }
        )
        world.scene.register(
            key,
            Role.GRASPABLE,
            xy=(float(slot[0]), float(slot[1])),
            z=fz + 0.05,
            prim_path=pp,
            approach_yaw=0.0,
            clearance=0.34,
        )
        world.used_slots.append((slot[0], slot[1], fh + 0.30))
        if near is None:
            near = slot  # every later board sits near the ROOT, so the kit reads as one workspace
        print(
            f"[bench] board {key} = {part.cat}/{part.model} @ ({slot[0]:.2f},{slot[1]:.2f}) "
            f"clear>={got:.2f}m half={fh:.2f}m points={sorted(points[key])}"
        )


def _site_surface_parts(
    world: World, ordered: list[str], supports: list[tuple[str, float]]
) -> None:
    support_map = dict(zip(world.recipe.support_keys, supports))
    for support_key in world.recipe.support_keys:
        parts = [
            world.part_of[key]
            for key in ordered
            if world.part_of[key].initial_place == f"on:{support_key}"
        ]
        if not parts:
            continue
        if support_key not in support_map:
            raise SpecBuildError(f"no live support binding for assembly parts on {support_key}")
        support_path, yaw = support_map[support_key]
        seat_items(
            world,
            support_path,
            yaw,
            only=[
                ItemRecipe(
                    part.key,
                    part.cat,
                    part.model,
                    (),
                    (),
                    initial_place=part.initial_place,
                    spot=part.spot,
                    tint=part.tint,
                    affordances=part.affordances,
                )
                for part in parts
            ],
        )


def _configure_assembly_affordances(world: World) -> None:
    raise_targets = {
        args[0]
        for verb, args in _spec_steps(str(world.spec.get("human_program") or ""))
        if verb.upper() in {"RAISE", "ERECT", "STANDUPRIGHT"} and args
    }
    for name, entity in world.scene.entities.items():
        if not (entity.meta or {}).get("post_assembly_affordances"):
            continue
        component = world.scene.assembly_component(name)
        entity.meta["assembly_affordance_requires_raised"] = bool(
            component & raise_targets
        )


def _manual_face(world: World) -> tuple[float, float]:
    """Outward normal of the manual's reading face: it looks back at the kit, so the reader stands
    BETWEEN the sign and the boards and only has to turn round to start work."""
    mx, my = world.manual_xy
    rx, ry = world.scene.entities[world.root_key].xy
    dx, dy = float(rx) - mx, float(ry) - my
    n = math.hypot(dx, dy) or 1.0
    return (dx / n, dy / n)


# The manual is a free-standing two-panel board. Sized so a standing reader at the derived stand
# fills the FPV with it: this row runs with ``look_zoom`` off, so the sign has to be legible from the
# plain standing camera rather than from a focal inspect.
MANUAL_HALF_W: float = 0.66
MANUAL_H: float = 0.92
MANUAL_BASE_Z: float = 0.34
MANUAL_THICK: float = 0.06
# A container row's rule card is a single panel in a WORKING room, not a two-panel board in an empty
# hall: it has to fit between benches, so it is narrower and it may stand further from the anchor.
CARD_HALF_W: float = 0.34
CARD_DCAP_M: float = 7.0


def site_manual(world: World, half: float = MANUAL_HALF_W, dcap: float = 0.0) -> None:
    """Site the MANUAL board on clear floor a short walk off the kit and reserve its reading stand."""
    cells, _boxes = _grid(world)
    slot, got = _find_slot(
        world, cells, half, near=world.anchor, dcap_max=dcap or ASSEMBLY_SPAN_M
    )
    if slot is None:
        raise SpecBuildError(
            f"no open floor in {world.recipe.room} for the manual board"
        )
    world.manual_xy = (float(slot[0]), float(slot[1]))
    world.used_slots.append((slot[0], slot[1], MANUAL_HALF_W + 0.30))
    print(f"[bench] manual board @ ({slot[0]:.2f},{slot[1]:.2f}) clear>={got:.2f}m")


def mount_manual(world: World, panels: list[str]) -> None:
    """Build the manual board and mount its panels, then register it as the LookAt-able CUE."""
    import fpv_scene
    from pxr import UsdGeom

    nx, ny = _manual_face(world)
    ux, uy = -ny, nx
    mx, my = world.manual_xy
    fz = float(world.scene.floor_z)
    zc = fz + MANUAL_BASE_Z + 0.5 * MANUAL_H
    body = UsdGeom.Cube.Define(world.stage, "/World/BenchManual")
    body.CreateSizeAttr(1.0)
    yaw = math.atan2(uy, ux)
    fpv_scene.set_world_pose(
        body.GetPrim(),
        (mx, my, zc),
        (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
        (2.0 * MANUAL_HALF_W, MANUAL_THICK, MANUAL_H),
    )
    pw = (2.0 * MANUAL_HALF_W - 0.10) / len(panels)
    for i, png in enumerate(panels):
        off = (i - (len(panels) - 1) / 2.0) * pw
        cx, cy = (
            mx + ux * off + nx * (0.5 * MANUAL_THICK + 0.01),
            my + uy * off + ny * (0.5 * MANUAL_THICK + 0.01),
        )
        hw, hh = 0.5 * (pw - 0.03), 0.5 * (MANUAL_H - 0.06)
        corners = [
            (cx - ux * hw, cy - uy * hw, zc - hh),
            (cx + ux * hw, cy + uy * hw, zc - hh),
            (cx + ux * hw, cy + uy * hw, zc + hh),
            (cx - ux * hw, cy - uy * hw, zc + hh),
        ]
        _textured_quad(
            world.stage, f"/World/BenchManual_p{i}", corners, (nx, ny, 0.0), png
        )
    _register_manual(world, (nx, ny), (ux, uy), zc)


def _register_manual(world: World, nrm, up, zc: float) -> None:
    """Register the manual as a SURFACE the human can walk to and read. ``approach_yaw`` puts the
    reader on the face's own side, so ``[LookAt]`` aims at artwork rather than at the back board."""
    from scene_state import Role

    mx, my = world.manual_xy
    nx, ny = nrm
    ux, uy = up
    hx = abs(ux) * MANUAL_HALF_W + abs(nx) * 0.5 * MANUAL_THICK
    hy = abs(uy) * MANUAL_HALF_W + abs(ny) * 0.5 * MANUAL_THICK
    top = zc + 0.5 * MANUAL_H
    board = world.recipe.manual_key or world.recipe.card_key
    ent = world.scene.register(
        board,
        Role.SURFACE,
        xy=(mx, my),
        z=top,
        half=(hx, hy),
        prim_path="/World/BenchManual",
        approach_yaw=math.degrees(math.atan2(-nx, ny)),
        foot_xy=(mx, my),
        foot_half=(hx, hy),
        top_z=top,
        clearance=0.55,
    )
    ent.meta["label_face"] = (float(nx), float(ny))
    ent.meta["label_pos"] = (float(mx + nx * 0.05), float(my + ny * 0.05), float(zc))
    ent.meta["label_span"] = float(2.0 * MANUAL_HALF_W)
    world.decals[board] = "/World/BenchManual_p0"
    sx = mx + nx * (abs(nx) * hx + abs(ny) * hy + 0.55)
    sy = my + ny * (abs(nx) * hx + abs(ny) * hy + 0.55)
    world.used_slots.append((sx, sy, 0.42))
    ent.meta["front_stand"] = (float(sx), float(sy))


def assert_attached_clean_at_t0(
    world: World, mates: list[Mate], initial_attachments: list[dict]
) -> None:
    """The builder-time registry must contain no unmaterialized symbolic mates.

    USD part prims are built later by ``setup_episode``. Frozen initial mates are
    therefore carried as an explicit build contract and physically applied there;
    declaring them here would create the exact symbolic-only false positive this
    gate is intended to reject.
    """
    import spec_validator as sv
    from spec_validator import Goal

    declared = dict(getattr(world.scene, "attached", {}) or {})
    if declared:
        raise SpecBuildError(f"a mate is already declared at t=0: {declared}")
    rel0 = (
        world.scene.relations(source="geometric")
        | world.scene._relations_attached_live()
    )
    initial_edges = {
        (str(row["child"]), str(row["parent"])) for row in initial_attachments
    }
    goals = [
        Goal("attached", m.child, m.parent)
        for m in mates
        if (m.child, m.parent) not in initial_edges
    ]
    pre = sv.pre_satisfied_predicates(goals, rel0)
    if pre:
        raise SpecBuildError(
            f"mate(s) already true at t=0: {[(g.subject, g.target) for g in pre]}"
        )
    print(
        f"[bench] t=0 builder registry: ATTACHED=[] (declared={len(declared)}); "
        f"{len(initial_attachments)} initial mate(s) deferred to live verified realization, "
        f"0 of {len(goals)} action-created mates pre-satisfied"
    )


# =====================================================================================
# stage 5 -- MANDATORY negative-constraint resolution
# =====================================================================================
def env_tables(recipe: Recipe, preload, numbers, cue_class, fillers=(), mates=()):
    """The five tables ``constraint_resolver`` reads, built in ONE place so the offline selftest proves the
    same env the live build hands the resolver -- a stand-in that merely resembles it proves nothing.

    ``preload`` carries EVERY unit, not just the loaded ones: ``{"op": "preload"}`` on an empty unit
    must resolve to 0, and a missing key would abort a build that is perfectly well defined."""
    import constraint_resolver as gr

    items = [i.key for i in recipe.staged()] + [m.child for m in mates]
    units = [u.key for u in recipe.units] + sorted({m.parent for m in mates})
    return gr.BuildEnv(
        preload={u.key: int(dict(preload).get(u.key, 0)) for u in recipe.units},
        cue_number={c: dict(v) for c, v in dict(numbers).items()},
        cue_class={recipe.cue_channel: dict(cue_class)},
        pool={
            "items": items,
            "staged": items,
            "preload": list(fillers),
            "containers": units,
        },
        unit_set={
            "all_units": [u.key for u in recipe.units] or [p.key for p in recipe.parts]
        },
    )


def build_env(world: World):
    """The ONE channel between the builder and the resolver, published AFTER the preloads are seated
    and the seeded cue draw is known."""
    return env_tables(
        world.recipe,
        world.preload,
        world.numbers,
        world.cue_class,
        [k for ks in world.preload_keys.values() for k in ks],
        world.mates,
    )


def offline_env(spec: dict, recipe: Recipe, seed: int):
    """The same env WITHOUT a stage: the seeded draws are pure functions of (task_id, seed), so the
    selftest can resolve every constraint for real. Only the preload COUNT is trusted here -- that
    the world really shows that many is what ``assert_preload_matches`` proves on the live stage."""
    numbers = draw_numbers(recipe, seed)
    preload = preload_counts(recipe, seed, numbers)
    fillers = [
        f"pre_{u}_{i}" for u, n in sorted(preload.items()) for i in range(int(n))
    ]
    return env_tables(
        recipe, preload, numbers, cue_class_for(spec, recipe, seed), fillers
    )


def measured_inside(world: World) -> set:
    """The INSIDE pairs a LIVE physics read of the stage gives at t=0 -- the same
    ``_relations_inside_live`` the scorer runs, on the same prims, with nothing declared."""
    from isaac_goal import Pred

    rel = world.scene.relations(source="physics")
    return {(a, b) for (p, a, b) in rel if p is Pred.INSIDE}


def assert_preload_matches(world: World, env) -> None:
    """``constraint_resolver``'s own warning, made unfalsifiable: *the preload count must match the scorer's
    t=0 INSIDE count or the cap drifts silently*. Re-count the LIVE stage and demand equality --
    a filler that missed its shelf, or a stray body that reads into a bay, is a BUILD failure."""
    got: dict[str, int] = {u.key: 0 for u in world.recipe.units}
    for _a, b in measured_inside(world):
        got[b] = got.get(b, 0) + 1
    drift = {
        u: (int(env.preload.get(u, 0)), n)
        for u, n in got.items()
        if int(env.preload.get(u, 0)) != n
    }
    if drift:
        raise SpecBuildError(
            f"preload DRIFT (declared, measured) = {drift}. cap = preload + key_load would be "
            f"computed from a number the scorer never sees; every capacity verdict would be wrong."
        )
    print(f"[bench] preload == measured t=0 INSIDE count: {got}")


def assert_t0_exact(world: World) -> set:
    """T0-EXACT. The measured t=0 relation set must equal the DECLARED one exactly -- not merely
    contain it. ``pre_satisfied_predicates`` only asks whether a goal is already true; it cannot see
    an EXTRA relation (a spare carton that landed in the wrong bay) that silently moves a cap."""
    declared = {(k, u) for u, keys in world.preload_keys.items() for k in keys}
    got = measured_inside(world)
    if got != declared:
        raise SpecBuildError(
            f"t=0 INSIDE set is not the declared one: missing={sorted(declared - got)} "
            f"unexpected={sorted(got - declared)}"
        )
    if getattr(world.scene, "attached", None):
        raise SpecBuildError(
            f"t=0 declares ATTACHED mates on a container row: {world.scene.attached}"
        )
    print(
        f"[bench] T0-EXACT: measured INSIDE == declared ({len(declared)} pair(s)) {sorted(declared)}"
    )
    return got


def resolve_negatives(world: World) -> list[dict]:
    """THE chokepoint. 466 negative fields ship EMPTY portfolio-wide and only this resolver
    fills them; an empty required field makes ``eval_neg_constraints`` report ok=True for
    every world forever. Calling it is mandatory and the result is re-asserted here, so a
    future refactor that routes around ``resolve_negs`` still cannot ship a dead constraint."""
    import constraint_resolver as gr
    from spec_validator import REQUIRED_NEG_FIELDS

    env = build_env(world)
    negs = gr.resolve_negs(list(world.spec.get("negs") or []), world.spec, env)
    for c in negs:
        empty = [
            f for f in REQUIRED_NEG_FIELDS.get(str(c.get("kind")), ()) if not c.get(f)
        ]
        if empty or not c.get("resolved"):
            raise SpecBuildError(
                f"negative {c.get('label')!r} left field(s) {empty} EMPTY after constraint_resolver "
                f"-- eval_neg_constraints would read that as 'nothing to check' (silent pass)"
            )
        print(
            f"[bench] neg {c['label']}: kind={c['kind']} "
            + " ".join(
                f"{f}={c.get(f)}" for f in REQUIRED_NEG_FIELDS.get(str(c["kind"]), ())
            )
        )
    declared = len(world.spec.get("negs") or [])
    if len(negs) != declared:
        raise SpecBuildError(
            f"resolver returned {len(negs)} constraints for {declared} declared"
        )
    print(
        f"[bench] constraint_resolver.resolve_negs: {len(negs)}/{declared} constraint(s) resolved"
    )
    return negs


# =====================================================================================
# stage 6 -- a FIXED, validated, connected start pose
# =====================================================================================
def _walk_components(cells):
    """Flood-fill the walkable cells of the clearance grid into connected components, so the
    start can be proved to share a component with every place the program sends the human."""
    walk = {
        (round(x / 0.25), round(y / 0.25)): (x, y)
        for x, y, c in cells
        if c >= WALKABLE_CLEAR_M
    }
    comp: dict = {}
    n = 0
    for k in walk:
        if k in comp:
            continue
        n += 1
        stack = [k]
        comp[k] = n
        while stack:
            i, j = stack.pop()
            for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if nb in walk and nb not in comp:
                    comp[nb] = n
                    stack.append(nb)
    return walk, comp


def _component_at(walk, comp, xy, max_distance: float | None = None) -> int | None:
    best, bk = 1e9, None
    for k, (x, y) in walk.items():
        d = math.hypot(x - xy[0], y - xy[1])
        if d < best:
            best, bk = d, k
    if bk is None or (max_distance is not None and best > float(max_distance)):
        return None
    return comp.get(bk)


def choose_start(world: World, cue_xy, targets: list[tuple[float, float]]):
    """Deterministic start: the most open walkable cell that is FAR ENOUGH from the cue that
    it is not already legible at t=0, near enough to be the same shift, and in the same nav
    component as the cue. ``scene.near_start`` then makes the seeded init snap to this exact
    hint instead of sampling a diverse random cell.

    The component test runs on this builder's per-entity AABB pillars, which OVER-block a
    cluttered diner's aisles; the render's ``precise_nav`` rasterizes collision triangles and
    reopens them, and ``init_validation.place_start`` re-confirms the snapped cell with the exact
    overlap gate. So the component split only ORDERS the candidate tiers -- it never vetoes. A
    proxy strictly more pessimistic than the engine must not be able to reject a buildable
    world; the only hard floor kept here is the honest one, ``MIN_CUE_DIST_M``."""
    cells, _boxes = _grid(world)
    walk, comp = _walk_components(cells)
    cid = _component_at(walk, comp, cue_xy)
    spans = sorted({_component_at(walk, comp, t) for t in targets} - {None})
    if len(spans) != 1:
        print(
            f"[bench][WARN] AABB-proxy nav components for the {len(targets)} program targets: "
            f"{spans} (cue in {cid}); precise_nav triangle raster decides at render time"
        )
    for tier, pool in (
        ("cue-component", [c for c in cells if comp.get(_cell_key(c)) == cid]),
        ("any-walkable", [c for c in cells if _cell_key(c) in comp]),
        ("room", list(cells)),
    ):
        best = _best_start(world, pool, cue_xy)
        if best is not None:
            print(
                f"[bench] start ({best[2]:.2f},{best[3]:.2f}) clear={best[0]:.2f}m "
                f"cue_dist={best[1]:.2f}m tier={tier} cue_component={cid} targets={spans}"
            )
            return (float(best[2]), float(best[3])), float(best[1])
    raise SpecBuildError(
        f"no cell in {world.room} is >= {MIN_CUE_DIST_M} m from the cue and clear of "
        "the walls -- the room cannot host a start where the cue is illegible at t=0"
    )


def _cell_key(cell):
    return (round(cell[0] / 0.25), round(cell[1] / 0.25))


def _best_start(world: World, pool, cue_xy):
    """Most open cell in ``pool`` at a legal reading distance from the cue, off the walls and
    out of every reserved stand. Deterministic: ranked on (clearance, -distance)."""
    best = None
    for x, y, c in sorted(pool):
        d = math.hypot(x - cue_xy[0], y - cue_xy[1])
        if not (MIN_CUE_DIST_M <= d <= MAX_CUE_DIST_M):
            continue
        if any(math.hypot(x - ux, y - uy) < ur for (ux, uy, ur) in world.used_slots):
            continue
        if wall_dist(world, x, y) < 0.40:  # a body-radius of daylight off every wall
            continue
        if best is None or (c, -d) > (best[0], -best[1]):
            best = (c, d, x, y)
    return best


def start_override(world: World) -> tuple[float, float] | None:
    """Decode an explicit ``start_xy:<x>,<y>`` demo variant."""
    prefix = "start_xy:"
    variant = str(world.spec.get("variant") or "")
    if not variant.startswith(prefix):
        return None
    values = variant[len(prefix) :].split(",")
    if len(values) != 2:
        raise SpecBuildError(f"invalid start override variant {variant!r}")
    try:
        point = float(values[0]), float(values[1])
    except ValueError as error:
        raise SpecBuildError(f"invalid start override variant {variant!r}") from error
    if not all(math.isfinite(value) for value in point):
        raise SpecBuildError(f"invalid start override variant {variant!r}")
    return point


# =====================================================================================
# stage 7 -- the honesty gates
# =====================================================================================
def _overlap_forensics(
    world: World,
    pairs: list,
    *,
    boxes: Mapping[str, tuple] | None = None,
    box_label: str = "registered",
) -> str:
    """Name each overlapping box, its source prim, and whether boxes are distinct at all."""
    keys = sorted({k for p in pairs for k in p[:2]})
    areas = {round(float(p[2]), 6) for p in pairs}
    lines = [
        f"\n  {len(pairs)} pair(s) over {len(keys)} container(s); "
        f"{len(areas)} DISTINCT overlap area(s): {sorted(areas)}"
    ]
    if len(areas) == 1 and len(pairs) > 2:
        lines.append(
            "  ^ ONE area across every pair -> the boxes are almost certainly the SAME volume "
            "(a shared proxy), not real collisions. Compare the boxes below before touching the guard."
        )
    selected_boxes = world.boxes_of if boxes is None else boxes
    seen: dict[tuple, list[str]] = {}
    for k in keys:
        b = selected_boxes.get(k)
        pp = (world.reg_pp or {}).get(k)
        if b is None:
            lines.append(f"  {k:10s} {box_label}_box=MISSING prim={pp}")
            continue
        rb = tuple(round(float(v), 4) for v in b)
        seen.setdefault(rb, []).append(k)
        lines.append(
            f"  {k:10s} {box_label}_box=({rb[0]:+.3f},{rb[1]:+.3f},{rb[2]:.3f}).."
            f"({rb[3]:+.3f},{rb[4]:+.3f},{rb[5]:.3f}) "
            f"size=({rb[3] - rb[0]:.3f}x{rb[4] - rb[1]:.3f}x{rb[5] - rb[2]:.3f}) prim={pp}"
        )
    dupes = {b: ks for b, ks in seen.items() if len(ks) > 1}
    if dupes:
        for b, ks in dupes.items():
            lines.append(
                f"  ⚠️ IDENTICAL {box_label} box shared by {len(ks)} containers {ks}: {b} "
                "-- these are not distinct geometries; the geometry is the bug, not the guard."
            )
    return "\n".join(lines)


def _release_footprints(world: World, key: str) -> list[tuple[str, tuple]]:
    """Conservative XY regions in which ``PutInside`` may leave an object's centre."""
    box = world.boxes_of.get(key)
    if box is None or len(box) != 6:
        raise SpecBuildError(
            f"cannot prove container {key!r} disjoint: registered interior bbox is missing"
        )
    footprints = [("registered interior", (box[0], box[1], box[3], box[4]))]
    entity = getattr(world.scene, "entities", {}).get(key)
    for space in getattr(entity, "spaces", None) or ():
        if len(space) < 7:
            raise SpecBuildError(
                f"cannot prove container {key!r} disjoint: malformed release space {space!r}"
            )
        name, cx, cy, hx, hy = (
            str(space[0]),
            float(space[3]),
            float(space[4]),
            float(space[5]),
            float(space[6]),
        )
        values = (cx, cy, hx, hy)
        if not all(math.isfinite(value) for value in values) or hx < 0.0 or hy < 0.0:
            raise SpecBuildError(
                f"cannot prove container {key!r} disjoint: invalid release space {space!r}"
            )
        footprints.append((name, (cx - hx, cy - hy, cx + hx, cy + hy)))
    return footprints


def _closed_xy_overlap(left: tuple, right: tuple) -> bool:
    """Whether two closed XY rectangles share even one point."""
    return max(float(left[0]), float(right[0])) <= min(
        float(left[2]), float(right[3])
    ) and max(float(left[1]), float(right[1])) <= min(
        float(left[3]), float(right[4])
    )


def _ambiguous_outer_overlaps(world: World, outer_pairs: list) -> list[tuple]:
    """Outer overlaps reachable by a legal release into either member of the pair."""
    conflicts = []
    for left, right, _area in outer_pairs:
        for source, target in ((left, right), (right, left)):
            target_outer = world.outer_of.get(target)
            if target_outer is None or len(target_outer) != 6:
                raise SpecBuildError(
                    f"cannot prove containers {left!r}/{right!r} disjoint: "
                    f"live OUTER bbox for {target!r} is missing"
                )
            for release_name, footprint in _release_footprints(world, source):
                if _closed_xy_overlap(footprint, target_outer):
                    conflicts.append((source, release_name, target))
    return conflicts


def assert_disjoint(world: World) -> list:
    """No legal container release may score INSIDE two containers.

    Live INSIDE tests an object's centre against the container's OUTER AABB. Adjacent native
    cabinets can have slightly interpenetrating decorative shells even though their registered
    interiors are separated. Permit that shell-only overlap only after proving, in both directions,
    that neither registered interior nor any geometry-derived release space reaches the other's
    live footprint.
    """
    import spec_validator as sv

    pairs = sv.container_xy_overlap_pairs(world.boxes_of)
    if pairs:
        raise SpecBuildError(
            f"scored containers overlap (INSIDE is not exclusive): {pairs}"
            + _overlap_forensics(world, pairs)
        )
    # OUTER overlap alone is not ambiguous: adjacent native cabinets often share a few centimetres
    # of trim. It becomes ambiguous only when a legal release into either container can put an
    # object centre in the other's OUTER footprint. The check is deliberately bidirectional and
    # treats boundary contact as overlap because live INSIDE uses inclusive comparisons.
    outer = sv.container_xy_overlap_pairs(world.outer_of)
    conflicts = _ambiguous_outer_overlaps(world, outer)
    if conflicts:
        details = "; ".join(
            f"{source}/{release_name} release footprint enters {target} live OUTER"
            for source, release_name, target in conflicts
        )
        raise SpecBuildError(
            "scored containers' OUTER bboxes overlap and a legal release can score "
            f"INSIDE both: {details}; outer_pairs={outer}"
            + _overlap_forensics(
                world,
                outer,
                boxes=world.outer_of,
                box_label="outer",
            )
        )
    print(
        f"[bench] container_xy_overlap_pairs = [] over {len(world.boxes_of)} scored boxes "
        f"({len(outer)} safe shell-only overlap(s) over "
        f"{len(world.outer_of)} measured OUTER bboxes)"
    )
    return pairs


PROBE_KEY: str = "__accept_probe__"


def probe_sites(world: World, unit: str) -> list[tuple[str, float, float, float]]:
    """Every ``(name, x, y, base_z)`` a ``[PutInside] <unit>`` can release into.

    ``semantic_executor.produce_place`` picks ONE interior SPACE per put (upper levels first,
    auto-distributed across free spaces), so a single unit has as many release sites as it has
    boards; a unit with no boards falls back to its registered interior drop. Probing only the
    first would leave the others unproven, and it is the ones the oracle never happens to use
    that silently swallow a control's object."""
    from scene_state import Role

    owner = world.scene.entities[unit]
    target_view = getattr(world.scene, "target_view", None)
    ent = target_view(unit, Role.CONTAINER) if callable(target_view) else None
    ent = ent or owner
    sites = [
        (str(sp[0]), float(sp[3]), float(sp[4]), float(sp[1]))
        for sp in (getattr(ent, "spaces", None) or ())
    ]
    if sites:
        return sites
    geometry = _open_top_probe_geometry(world, unit, ent)
    drop = geometry[2] if geometry is not None else getattr(ent, "drop_z", None)
    z = float(drop if drop is not None else float(ent.z) + float(ent.height))
    if geometry is None:
        return [("interior_drop", float(ent.xy[0]), float(ent.xy[1]), z)]
    required = _minimum_joint_aggregate_count(world.spec, unit)
    if required <= 1:
        return [("interior_drop", geometry[0][0], geometry[0][1], z)]
    return _packed_open_top_probe_sites(world, unit, *geometry, required)


def _open_top_probe_geometry(world: World, unit: str, entity):
    meta = getattr(entity, "meta", None) or {}
    if meta.get("open_top") and not (getattr(entity, "spaces", None) or ()):
        if entity.xy is None or entity.drop_z is None:
            return None
        return (
            (float(entity.xy[0]), float(entity.xy[1])),
            (float(entity.half[0]), float(entity.half[1])),
            float(entity.drop_z),
        )
    if (unit, "inside") not in set(
        getattr(world.scene, "_graspable_targets", None) or ()
    ):
        return None
    item = next(
        (candidate for candidate in world.recipe.staged() if candidate.key == unit),
        None,
    )
    if item is None or entity.xy is None:
        return None
    size = staged_bbox(world, item)
    yaw = math.radians(float((entity.meta or {}).get("rest_yaw_deg", 0.0)))
    world_x = abs(math.cos(yaw)) * float(size[0]) + abs(math.sin(yaw)) * float(
        size[1]
    )
    world_y = abs(math.sin(yaw)) * float(size[0]) + abs(math.cos(yaw)) * float(
        size[1]
    )
    half = (
        max(0.05, 0.5 * world_x * 0.92),
        max(0.05, 0.5 * world_y * 0.92),
    )
    top = float(entity.z or 0.0) + float(size[2])
    drop_z = top - 0.05 * max(0.1, float(size[2])) - 0.03
    return (float(entity.xy[0]), float(entity.xy[1])), half, drop_z


def _packed_open_top_probe_sites(
    world: World,
    unit: str,
    center: tuple[float, float],
    target_half: tuple[float, float],
    drop_z: float,
    count: int,
) -> list[tuple[str, float, float, float]]:
    """Mirror the runtime's strict open-top footprint packing for a capacity witness."""
    import behavior_import as bi
    import placement_packing

    bodies = container_payload_bodies(world, unit) or [_probe_body(world, unit)]
    sizes = [
        bi.read_spec("/".join(body)).get("bbox_size") or [0.15, 0.15, 0.15]
        for body in bodies
    ]
    object_half = (
        max(0.02, 0.5 * max(float(size[0]) for size in sizes)),
        max(0.02, 0.5 * max(float(size[1]) for size in sizes)),
    )
    try:
        root_xy, _yaw = world.scene.stand_pose_for(unit)
    except (KeyError, TypeError, ValueError):
        root_xy = center
    selected = placement_packing.strict_pack_sites(
        center,
        target_half,
        object_half,
        count,
        root_xy,
    )
    return [
        (f"interior_drop_{index + 1}", float(x), float(y), float(drop_z))
        for index, (x, y) in enumerate(selected)
    ]


def _probe_body(world: World, unit: str | None = None) -> tuple[str, str]:
    """The body the probe is made of: the largest thing that actually GOES INTO ``unit``.

    A probe smaller than the thing the task really puts away would clear a lip the real object
    cannot, so it has to be the largest -- but the largest of the RIGHT set. Taking the row's
    largest graspable regardless of destination probed NARRATE-3's cabinet with the packing box
    left on the floor as an obstacle, a 0.38 x 0.47 m body that never goes near it, and refused 29
    bodies across three categories for failing to swallow something nobody was ever going to put
    in them. Falls back to the row's largest graspable when nothing is routed into the unit, which
    is the old behaviour and the right default for a destination the oracle never fills.
    """
    import behavior_import as bi

    def _span(body) -> float:
        bb = bi.read_spec("/".join(body)).get("bbox_size") or [0.15, 0.15, 0.15]
        return max(float(v) for v in bb)

    wanted = container_payload_bodies(world, unit) if unit else []
    if not wanted and unit:
        # Nothing is ever put into this unit -- it is a SOURCE, opened only to take things out.
        # The gate still probes it, because a wrong-answer control may route an object here, so it
        # is probed with the largest body this row puts inside ANYTHING. The row's largest
        # graspable is the wrong yardstick for that: FIND-3 measured its lockers against a tote
        # that only ever rides a bench, exhausted every one of 21 native lockers plus three alt
        # categories, and died on a container the task never fills.
        for other in world.recipe.units:
            wanted += container_payload_bodies(world, other.key)
        wanted = list(dict.fromkeys(wanted))
    pool = wanted or [staged_body(world, it) for it in world.recipe.staged()]
    best, bestv = None, -1.0
    for body in pool:
        v = _span(body)
        if v > bestv:
            best, bestv = body, v
    return best or ("tupperware", "mkstwr")


def _recipe_item_body(recipe: Recipe, key: str) -> tuple[str, str] | None:
    item = next((candidate for candidate in recipe.staged() if candidate.key == key), None)
    if item is not None:
        return item.cat, item.model
    for preload in recipe.preloads:
        if key in preload.keys:
            return preload.cat, preload.model
    return None


def _program_putinside_routes(spec: Mapping[str, object]) -> list[tuple[str, str, str]]:
    """Return ``(object, container, space)`` routes; an empty space is dynamic."""
    source = spec.get("bts_source")
    source_program = source.get("program") if isinstance(source, Mapping) else None
    direct_program = spec.get("program")
    commands = None
    for program in (source_program, direct_program):
        if isinstance(program, Mapping) and isinstance(program.get("commands"), list):
            commands = program["commands"]
            break
    if commands is not None:
        out = []
        for command in commands:
            if not isinstance(command, Mapping):
                continue
            verb = str(command.get("verb") or "").casefold()
            args = command.get("args")
            if verb not in {"putin", "putinside"} or not isinstance(args, list):
                continue
            if len(args) < 2:
                continue
            out.append(
                (
                    str(args[0]),
                    str(args[1]),
                    str(args[2]) if len(args) >= 3 and args[2] else "",
                )
            )
        return out

    program = str(spec.get("human_program") or "")
    return [
        (match.group(1), match.group(2), str(match.group(3) or ""))
        for match in re.finditer(
            r"\[(?:PutIn|PutInside)\]\s*<([^>]+)>\s*\(\d+\)\s*"
            r"<([^>]+)>\s*\(\d+\)(?:\s*<([^>]+)>\s*\(\d+\))?",
            program,
        )
    ]


def _dynamic_container_payloads_by_space(
    world: World, unit: str
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str]]]:
    """Runtime payload bodies partitioned by explicit or dynamic release space."""
    pinned: dict[str, list[tuple[str, str]]] = {}
    universal: list[tuple[str, str]] = []
    routed: set[str] = set()
    for key, target, space in _program_putinside_routes(world.spec or {}):
        if target != unit:
            continue
        body = _recipe_item_body(world.recipe, key)
        if body is None:
            continue
        routed.add(key)
        if space:
            pinned.setdefault(space, []).append(body)
        else:
            universal.append(body)

    answers = reference_answers(world.spec)
    for key, val in answers.items():
        if split_answer(str(val))[0] != unit or key in routed:
            continue
        body = _recipe_item_body(world.recipe, key)
        if body is not None:
            universal.append(body)
    for aggregate in world.spec.get("aggregate_goals") or []:
        if (
            aggregate.get("kind") != "exact_count_inside"
            or str(aggregate.get("target") or "") != unit
        ):
            continue
        for key in aggregate.get("candidates") or []:
            body = _recipe_item_body(world.recipe, str(key))
            if body is not None:
                universal.append(body)
    return (
        {key: list(dict.fromkeys(values)) for key, values in pinned.items()},
        list(dict.fromkeys(universal)),
    )


def _container_payloads_by_space(
    world: World, unit: str
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str]]]:
    """Return exact-space preloads and bodies that may occupy any selected space."""
    pinned: dict[str, list[tuple[str, str]]] = {}
    universal: list[tuple[str, str]] = []
    for preload in world.recipe.preloads:
        if preload.unit != unit:
            continue
        body = (preload.cat, preload.model)
        initial_space = str(getattr(preload, "initial_space", "") or "")
        if initial_space:
            pinned.setdefault(initial_space, []).append(body)
        else:
            universal.append(body)
    dynamic_pinned, dynamic_universal = _dynamic_container_payloads_by_space(
        world, unit
    )
    for space, bodies in dynamic_pinned.items():
        pinned.setdefault(space, []).extend(bodies)
    universal.extend(dynamic_universal)
    return (
        {key: list(dict.fromkeys(values)) for key, values in pinned.items()},
        list(dict.fromkeys(universal)),
    )


def container_payload_bodies(world: World, unit: str) -> list[tuple[str, str]]:
    """(cat, model) of everything the row puts inside ``unit`` -- preloaded, placed or scored."""
    pinned, universal = _container_payloads_by_space(world, unit)
    out = [body for bodies in pinned.values() for body in bodies]
    out.extend(universal)
    return list(dict.fromkeys(out))


def _container_probe_plan(
    world: World, unit: str
) -> list[tuple[tuple[str, float, float, float], list[tuple[str, str]]]] | None:
    """Pair each exercised release site with every body that may actually occupy it."""
    available = probe_sites(world, unit)
    by_name = {site[0]: site for site in available}
    pinned, universal = _container_payloads_by_space(world, unit)
    if set(pinned) - set(by_name):
        return None

    selected = (
        list(available[: sites_needed(world, unit)])
        if universal or not pinned
        else []
    )
    selected_names = {site[0] for site in selected}
    for name in pinned:
        if name not in selected_names:
            selected.append(by_name[name])
            selected_names.add(name)

    fallback = None
    plan = []
    for site in selected:
        bodies = list(dict.fromkeys(pinned.get(site[0], []) + universal))
        if not bodies:
            if fallback is None:
                fallback = _probe_body(world, unit)
            bodies = [fallback]
        plan.append((site, bodies))
    return plan


def _collision_mesh_enabled(prim, root) -> bool:
    """Whether ``prim`` and its ancestors keep their authored collider enabled."""
    from pxr import UsdPhysics

    root_path = str(root.GetPath()).rstrip("/")
    current = prim
    while current and current.IsValid():
        if current.HasAPI(UsdPhysics.CollisionAPI):
            value = UsdPhysics.CollisionAPI(current).GetCollisionEnabledAttr().Get()
            if value is not None and not bool(value):
                return False
        if str(current.GetPath()).rstrip("/") == root_path:
            break
        current = current.GetParent()
    return True


def _collision_meshes(
    stage, pp: str, skip: tuple[str, ...] = ()
) -> list[tuple[str, tuple, tuple]]:
    """World AABB of every ``/collisions/`` guide mesh under ``pp`` -- the geometry PhysX turns
    into the convex hulls the released object has to come to rest against.

    ``skip`` drops the SWING LEAF's own meshes: a ``[PutInside]`` opens the door before it releases,
    so a closed leaf standing in the drop lane at t=0 is not a defect and must not read as one. (No
    measured case in this slice was a leaf -- every offender was ``base_link`` -- so this only
    closes a false-positive class, it does not move a number.)"""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(pp)
    if not (prim and prim.IsValid()):
        return []
    bc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    out = []
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        path = str(p.GetPath())
        if p.GetTypeName() != "Mesh" or "/collisions/" not in path:
            continue
        if not _collision_mesh_enabled(p, prim):
            continue
        if any(
            path.startswith(str(s).rstrip("/") + "/") or path == str(s) for s in skip
        ):
            continue
        rng = bc.ComputeWorldBound(p).ComputeAlignedRange()
        a, b = rng.GetMin(), rng.GetMax()
        if b[0] >= a[0]:
            out.append((path, (a[0], a[1], a[2]), (b[0], b[1], b[2])))
    return out


def _deepest_overlap(box, meshes) -> tuple[float, str]:
    """Deepest AABB interpenetration (the standard min-over-axes overlap) of ``box`` with any
    collision mesh. A body RESTING on a board overlaps it in xy and by ~0 in z, so the min-axis
    depth is ~0; a body EMBEDDED in a hull overlaps on all three axes at once."""
    (mnx, mny, mnz), (mxx, mxy, mxz) = box
    worst, who = 0.0, ""
    for nm, a, b in meshes:
        ox = min(mxx, b[0]) - max(mnx, a[0])
        oy = min(mxy, b[1]) - max(mny, a[1])
        oz = min(mxz, b[2]) - max(mnz, a[2])
        d = min(ox, oy, oz)
        if d > worst:
            worst, who = d, nm
    return worst, who


def _dot3(a, b) -> float:
    return sum(float(a[index]) * float(b[index]) for index in range(3))


def _centered_interval_overlap(radius_a: float, center_b: float, radius_b: float) -> float:
    return min(radius_a, center_b + radius_b) - max(-radius_a, center_b - radius_b)


def _aabb_obb_overlap_depth(box, obb) -> float:
    """Return zero for separated/touching boxes, else their shallowest SAT overlap in metres."""
    (minimum, maximum) = box
    name, center, axes, half = obb
    del name
    a_center = tuple(
        0.5 * (float(minimum[index]) + float(maximum[index])) for index in range(3)
    )
    a_half = tuple(
        max(0.0, 0.5 * (float(maximum[index]) - float(minimum[index])))
        for index in range(3)
    )
    translation = tuple(float(center[index]) - a_center[index] for index in range(3))
    rotation = tuple(
        tuple(float(axes[column][row]) for column in range(3))
        for row in range(3)
    )
    absolute = tuple(tuple(abs(value) for value in row) for row in rotation)
    depth = math.inf

    for row in range(3):
        radius_a = a_half[row]
        radius_b = sum(float(half[column]) * absolute[row][column] for column in range(3))
        overlap = _centered_interval_overlap(radius_a, translation[row], radius_b)
        if overlap <= 0.0:
            return 0.0
        depth = min(depth, overlap)

    for column in range(3):
        radius_a = sum(a_half[row] * absolute[row][column] for row in range(3))
        radius_b = float(half[column])
        overlap = _centered_interval_overlap(
            radius_a, _dot3(translation, axes[column]), radius_b
        )
        if overlap <= 0.0:
            return 0.0
        depth = min(depth, overlap)

    for row in range(3):
        row_next = (row + 1) % 3
        row_last = (row + 2) % 3
        for column in range(3):
            column_next = (column + 1) % 3
            column_last = (column + 2) % 3
            axis_norm = math.sqrt(max(0.0, 1.0 - rotation[row][column] ** 2))
            if axis_norm <= 1e-8:
                continue
            radius_a = (
                a_half[row_next] * absolute[row_last][column]
                + a_half[row_last] * absolute[row_next][column]
            )
            radius_b = (
                float(half[column_next]) * absolute[row][column_last]
                + float(half[column_last]) * absolute[row][column_next]
            )
            distance = (
                translation[row_last] * rotation[row_next][column]
                - translation[row_next] * rotation[row_last][column]
            )
            overlap = _centered_interval_overlap(radius_a, distance, radius_b)
            if overlap <= 0.0:
                return 0.0
            depth = min(depth, overlap / axis_norm)
    return 0.0 if not math.isfinite(depth) else float(depth)


def _collision_obbs(stage, pp: str, skip: tuple[str, ...] = ()) -> list[tuple]:
    """Local collision bounds transformed as oriented boxes, without world-AABB rotation inflation."""
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(pp)
    if not (prim and prim.IsValid()):
        return []
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide],
        useExtentsHint=True,
    )
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = []
    for collision in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        path = str(collision.GetPath())
        if collision.GetTypeName() != "Mesh" or "/collisions/" not in path:
            continue
        if not _collision_mesh_enabled(collision, prim):
            continue
        if any(
            path.startswith(str(value).rstrip("/") + "/") or path == str(value)
            for value in skip
        ):
            continue
        extent = UsdGeom.Mesh(collision).GetExtentAttr().Get(Usd.TimeCode.Default())
        if not extent or len(extent) != 2:
            continue
        local_minimum, local_maximum = extent
        if local_maximum[0] < local_minimum[0]:
            continue
        local_center = Gf.Vec3d(
            *(0.5 * float(local_minimum[index] + local_maximum[index]) for index in range(3))
        )
        local_half = tuple(
            max(0.0, 0.5 * float(local_maximum[index] - local_minimum[index]))
            for index in range(3)
        )
        transform = xforms.GetLocalToWorldTransform(collision)
        world_center = transform.Transform(local_center)
        directions = [
            transform.TransformDir(Gf.Vec3d(*(1.0 if index == axis else 0.0 for index in range(3))))
            for axis in range(3)
        ]
        lengths = [math.sqrt(sum(float(value) ** 2 for value in direction)) for direction in directions]
        if any(length <= 1e-9 for length in lengths):
            continue
        axes = tuple(
            tuple(float(direction[index]) / length for index in range(3))
            for direction, length in zip(directions, lengths)
        )
        if any(
            abs(_dot3(axes[left], axes[right])) > 1e-4
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            world_range = cache.ComputeWorldBound(collision).ComputeAlignedRange()
            world_minimum, world_maximum = world_range.GetMin(), world_range.GetMax()
            center = tuple(
                0.5 * float(world_minimum[index] + world_maximum[index])
                for index in range(3)
            )
            axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            half = tuple(
                max(0.0, 0.5 * float(world_maximum[index] - world_minimum[index]))
                for index in range(3)
            )
        else:
            center = tuple(float(world_center[index]) for index in range(3))
            half = tuple(local_half[index] * lengths[index] for index in range(3))
        out.append((path, center, axes, half))
    return out


def _deepest_oriented_overlap(box, meshes) -> tuple[float, str]:
    worst, who = 0.0, ""
    for mesh in meshes:
        depth = _aabb_obb_overlap_depth(box, mesh)
        if depth > worst:
            worst, who = depth, str(mesh[0])
    return worst, who


# How deep the probe may sit in a container's own collision geometry. A body released with its
# base ON a board's own top overlaps that board by EXACTLY zero, so any measurable depth means a
# DIFFERENT hull of the same unit occupies the release lane. Measured on the two rows this slice
# repaired: the good shelves read 0 mm, the ones that swallow read 6 / 21 / 92 mm -- and 6 mm was
# enough, because the executor's raycast then seats the object another 23 mm down and PhysX
# resolves the penetration by LAUNCHING it (measured: settled 210 m under the floor at 320 m/s,
# with zero failed primitives). That is the whole defect class, and it is invisible to a purely
# positional read: the object was INSIDE at the release pose and gone one settle later.
PROBE_PEN_MAX_M: float = 0.002


def _acceptance_probe_candidates(entity, site, probe_box) -> list[tuple[float, float]]:
    """Use the executor's placement lattice for a static front-opening shelf."""
    name, x, y, _base_z = site
    if not bool((getattr(entity, "meta", None) or {}).get("static_open_front")):
        return [(float(x), float(y))]
    space = next(
        (value for value in (getattr(entity, "spaces", None) or ()) if value[0] == name),
        None,
    )
    if space is None or len(space) < 7:
        return [(float(x), float(y))]
    from semantic_executor import _candidate_set

    minimum, maximum = probe_box
    object_half = (
        0.5 * (float(maximum[0]) - float(minimum[0])),
        0.5 * (float(maximum[1]) - float(minimum[1])),
    )
    candidates, _radius_x, _radius_y = _candidate_set(
        (float(space[3]), float(space[4])),
        (float(space[5]), float(space[6])),
        object_half,
        (),
        (),
    )
    return candidates or [(float(x), float(y))]


def _support_collision_paths(
    meshes: Sequence[tuple[str, tuple, tuple]],
    xy: tuple[float, float],
    base_z: float,
) -> tuple[str, ...]:
    """Collider paths a runtime support ray could hit at this declared rest plane."""
    import physics_human

    x, y = float(xy[0]), float(xy[1])
    eps = 1e-6
    return tuple(
        str(path)
        for path, minimum, maximum in meshes
        if float(minimum[0]) - eps <= x <= float(maximum[0]) + eps
        and float(minimum[1]) - eps <= y <= float(maximum[1]) + eps
        and physics_human._within_support_tolerance(
            float(maximum[2]), float(base_z), 0.03
        )
    )


def _without_support_owner(meshes, support_paths):
    """Mirror the runtime rule that a collider cannot block the surface it belongs to."""
    if not support_paths:
        return list(meshes)
    import physics_human

    return [
        mesh
        for mesh in meshes
        if not any(
            physics_human._same_physics_owner(str(mesh[0]), str(support))
            for support in support_paths
        )
    ]


def _acceptance_probe_base_z(entity, site) -> float:
    """Map an authored site to the probe's bbox-base resting height.

    Shelf sites author a contact/base z. A shelf-less ``interior_drop`` instead
    authors the runtime object's origin release z, while the live relation is
    read after settling, so its static probe belongs on the registered floor.
    """
    name, _x, _y, site_z = site
    if (
        name.startswith("interior_drop")
        and (getattr(entity, "meta", None) or {}).get("open_top")
        and not (getattr(entity, "spaces", None) or ())
    ):
        interior_floor_z = getattr(entity, "z", None)
        try:
            interior_floor_z = float(interior_floor_z)
        except (TypeError, ValueError) as error:
            raise SpecBuildError(
                "open-top interior acceptance probe has no finite interior floor z"
            ) from error
        if not math.isfinite(interior_floor_z):
            raise SpecBuildError(
                "open-top interior acceptance probe has no finite interior floor z"
            )
        return interior_floor_z
    return float(site_z)


def _probe_once(world: World, unit: str, site, cat: str, model: str) -> bool:
    """Score one site's resting probe with live relations and collision depth."""
    import behavior_import as bi
    import fpv_scene
    from scene_state import Role

    name, x, y, release_z = site
    target_entity = world.scene.entities[unit]
    probe_base_z = _acceptance_probe_base_z(target_entity, site)
    pp = f"/World/Obj_{bi._safe(PROBE_KEY)}"
    prim = fpv_scene.add_reference(world.stage, pp, bi.baked_usd(f"{cat}/{model}"))
    _seat_prim_base(world, prim, pp, x, y, probe_base_z)
    world.scene.register(
        PROBE_KEY,
        Role.GRASPABLE,
        xy=(float(x), float(y)),
        z=probe_base_z,
        prim_path=pp,
        approach_yaw=0.0,
        clearance=0.20,
    )
    leaves = tuple(getattr(target_entity, "leaf_paths", None) or ())
    collision_meshes = _collision_meshes(world.stage, world.reg_pp[unit], leaves)
    collision_obbs = _collision_obbs(world.stage, world.reg_pp[unit], leaves)
    probe_box = _wbbox(world.stage, pp)
    candidates = _acceptance_probe_candidates(target_entity, site, probe_box)
    static_front_space = bool(
        name != "interior_drop"
        and (getattr(target_entity, "meta", None) or {}).get("static_open_front")
    )
    ok = False
    pen, who, overlap_kind = math.inf, "no-support", "aabb"
    chosen = (float(x), float(y))
    chosen_inside = False
    chosen_memberships: set[str] = set()
    chosen_support = not static_front_space
    best_rank = (-1, -1, -1, -math.inf)
    tested = 0
    for candidate_x, candidate_y in candidates:
        tested += 1
        _seat_prim_base(world, prim, pp, candidate_x, candidate_y, probe_base_z)
        probe_entity = world.scene.entities[PROBE_KEY]
        probe_entity.xy = (float(candidate_x), float(candidate_y))
        probe_entity.z = probe_base_z
        getattr(world.scene, "_static_aabb", {}).pop(PROBE_KEY, None)
        got = {b for (a, b) in measured_inside(world) if a == PROBE_KEY}
        probe_box = _wbbox(world.stage, pp)
        support_paths = (
            _support_collision_paths(
                collision_meshes,
                (float(candidate_x), float(candidate_y)),
                probe_base_z,
            )
            if static_front_space
            else ()
        )
        candidate_meshes = (
            _without_support_owner(collision_meshes, support_paths)
            if support_paths
            else collision_meshes
        )
        candidate_obbs = (
            _without_support_owner(collision_obbs, support_paths)
            if support_paths
            else collision_obbs
        )
        candidate_pen, candidate_who = _deepest_overlap(
            probe_box, candidate_meshes
        )
        candidate_kind = "aabb"
        if candidate_pen > PROBE_PEN_MAX_M:
            if candidate_obbs and len(candidate_obbs) == len(candidate_meshes):
                candidate_pen, candidate_who = _deepest_oriented_overlap(
                    probe_box, candidate_obbs
                )
                candidate_kind = "obb"
        inside = got == {unit}
        contains_target = unit in got
        supported = not static_front_space or bool(support_paths)
        candidate_ok = (
            inside
            and supported
            and (name == "interior_drop" or candidate_pen <= PROBE_PEN_MAX_M)
        )
        rank = (
            int(inside),
            int(contains_target),
            int(supported),
            -float(candidate_pen),
        )
        if candidate_ok or rank > best_rank:
            best_rank = rank
            pen, who, overlap_kind = (
                candidate_pen,
                candidate_who,
                candidate_kind,
            )
            chosen = (float(candidate_x), float(candidate_y))
            chosen_inside = inside
            chosen_memberships = got
            chosen_support = supported
        if candidate_ok:
            ok = True
            break
    world.scene.entities.pop(PROBE_KEY, None)
    getattr(world.scene, "_static_aabb", {}).pop(PROBE_KEY, None)
    world.stage.RemovePrim(pp)
    # The penetration test applies to a BOARD site only. A shelved put rests the object's base ON
    # the board, so the honest expectation there is exactly 0 mm and anything more is another hull
    # in the lane. An ``interior_drop`` site authors an origin RELEASE height just under the rim;
    # its probe is instead seated on the registered interior floor to match the post-settle scorer.
    # Measuring that pose against the bin's own floor/wall hull would condemn every open receptacle
    # in the portfolio, so the live INSIDE read remains the whole test for those sites.
    print(
        f"[bench][ACCEPT]{'' if ok else ' ✗'} {unit}/{name} @ "
        f"({chosen[0]:.2f},{chosen[1]:.2f}) "
        f"release_z={float(release_z):.2f} settled_base_z={probe_base_z:.2f} "
        f"candidates={tested}/{len(candidates)} "
        f"INSIDE={'yes' if chosen_inside else 'no'} "
        f"memberships={sorted(chosen_memberships)} "
        f"support={'yes' if chosen_support else 'no'} "
        f"pen={pen * 1000:.0f}mm({overlap_kind})"
        + (
            f" into {who}"
            if (pen > PROBE_PEN_MAX_M and name != "interior_drop")
            else ""
        )
    )
    return ok


def _preload_planned(world: World, unit: str) -> int:
    """How many pieces of filler stock the RECIPE will put in ``unit``, before any are seated."""
    try:
        counts = _preload_recipe_counts(world.recipe, world.seed, world.numbers)
    except Exception:  # noqa: BLE001 - a count that cannot be drawn yet is simply unknown here
        return 0
    return sum(
        int(count)
        for pr, count in zip(world.recipe.preloads, counts)
        if pr.unit == unit
    )


def _minimum_joint_aggregate_count(spec: Mapping[str, object], unit: str) -> int:
    """Return the smallest object set satisfying every exact count for one target."""
    goals = [
        row
        for row in spec.get("aggregate_goals") or []
        if row.get("kind") == "exact_count_inside"
        and str(row.get("target") or "") == unit
    ]
    if not goals:
        return 0
    candidate_sets = [
        frozenset(str(value) for value in row.get("candidates") or [])
        for row in goals
    ]
    required = tuple(int(row.get("count") or 0) for row in goals)
    for index, (candidates, count) in enumerate(zip(candidate_sets, required)):
        if count < 0 or count > len(candidates):
            raise SpecBuildError(
                f"aggregate goal {index} for {unit} requires {count} of "
                f"{len(candidates)} distinct candidates"
            )
    states: dict[tuple[int, ...], int] = {(0,) * len(goals): 0}
    for candidate in sorted(set().union(*candidate_sets)):
        updated = dict(states)
        membership = tuple(candidate in candidates for candidates in candidate_sets)
        for counts, selected in states.items():
            next_counts = tuple(
                count + int(member)
                for count, member in zip(counts, membership)
            )
            if any(
                count > limit for count, limit in zip(next_counts, required)
            ):
                continue
            updated[next_counts] = min(
                updated.get(next_counts, selected + 1), selected + 1
            )
        states = updated
    if required not in states:
        raise SpecBuildError(
            f"aggregate goals for {unit} have no simultaneously satisfying candidate set"
        )
    return states[required]


def _aggregate_target_loads(spec: Mapping[str, object]) -> dict[str, int]:
    targets = tuple(
        dict.fromkeys(
            str(row.get("target") or "")
            for row in spec.get("aggregate_goals") or []
            if row.get("kind") == "exact_count_inside"
        )
    )
    return {
        target: _minimum_joint_aggregate_count(spec, target)
        for target in targets
        if target
    }


def sites_needed(world: World, unit: str) -> int:
    """How many release sites this unit has to accept, derived from the row's OWN answer key.

    ``semantic_executor`` fills a container's interior spaces TOP-DOWN, one per put, so a unit that
    the oracle sends two objects to uses its top TWO sites. Exact answer-key routes also keep one
    wrong-answer control site: a control whose object vanishes fires the wrong constraint. An
    ``exact_count_inside`` goal needs only its satisfying physical witness; its synthetic
    over-count witness belongs to the scorer tests and must not turn a one-item order into a
    two-site container requirement. The preload already occupies sites of its own."""
    answers = reference_answers(world.spec)
    keyed = sum(1 for v in answers.values() if split_answer(str(v))[0] == unit)
    aggregate = _minimum_joint_aggregate_count(world.spec, unit)
    # The RECIPE's preload count, not the running tally. `seat_preloads` fills `world.preload`
    # AFTER `site_accepting_units` has finished swapping bodies, so reading the tally made the
    # repair loop probe fewer sites than the final `assert_container_accepts` would -- it passed a
    # body on its top site, the preload arrived, and the last gate then failed a site nothing was
    # left to swap for. Measured on NARRATE-3: repair probed 1 site, the final gate probed 2, and
    # the build died at `_finish` with no ladder left.
    staged = int(world.preload.get(unit, 0)) or _preload_planned(world, unit)
    return staged + max(keyed + 1, aggregate)


def container_defects(world: World) -> list[str]:
    """The units whose top-down release sites do NOT all accept. Keys, so the caller can swap."""
    bad = []
    for u in world.recipe.units:
        plan = _container_probe_plan(world, u.key)
        if plan is None or any(
            not _probe_once(world, u.key, site, cat, model)
            for site, bodies in plan
            for cat, model in bodies
        ):
            bad.append(u.key)
    return bad


SCENE_UNIT_MARK: str = "scene_owned"
# A placard that does not move with its door is invisible the moment the person opens the unit, and
# the episode then scores a reader who could not read. 234.3 mm is what a measured 88-degree swing
# produced on bottom_cabinet/dajebq in Merom_1_int; 20 mm is a floor far below any real swing and
# far above float noise.
PLACARD_RIDE_MIN_M: float = 0.02


def _placard_centroid(stage, path: str) -> tuple[float, float, float]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get() or ()
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    acc = [0.0, 0.0, 0.0]
    for pt in pts:
        w = xf.Transform(pt)
        acc = [acc[0] + w[0], acc[1] + w[1], acc[2] + w[2]]
    n = max(1, len(pts))
    return (acc[0] / n, acc[1] / n, acc[2] / n)


def assert_placard_rides(world: World) -> None:
    """⭐ PLACARD RIDES ITS DOOR. For every SCENE-OWNED destination with a swing leaf, open the door
    and demand the placard moved with it.

    An IMPORTED unit gets this for free: ``behavior_import`` records ``leaf_paths`` as it places the
    body, and ``_leaf_frame`` parents the placard under the leaf. A SCENE-owned prim is registered
    by a different route, and if ``leaf_paths`` is empty the placard is parented to /World instead --
    it then hangs in the door opening the moment the unit is opened, and the cue the whole task turns
    on is gone. Measured, not assumed: this drives the leaf through the render's own controller and
    reads the placard's world centroid before and after.

    Checked HERE, on demand, rather than by pre-qualifying every container in all 51 scenes: only the
    furniture a task actually chose is worth a stage."""
    import math

    stage = world.stage
    bad = []
    for key, path in sorted((world.decals or {}).items()):
        ent = world.scene.entities.get(key)
        if ent is None or not getattr(ent, "leaf_paths", None):
            continue
        if not getattr(ent, "meta", {}).get(SCENE_UNIT_MARK):
            continue  # an imported unit is already proven by the path that placed it
        ang = float(getattr(ent, "swing_angle", 0.0) or 0.0)
        if ang <= 1e-6:
            continue  # a drawer/open bay has no swing to ride
        import episode_runtime as s3

        before = _placard_centroid(stage, path)
        ctrl = s3._place_door(stage, world.scene, key)
        if not ctrl:
            bad.append(f"{key}: no swing controller, so the placard cannot ride")
            continue
        ctrl["rot"].Set(math.degrees(ang))
        ctrl["cur_angle"] = ang
        for ex_rot, ex_sign in ctrl.get("extra_leaves", ()):
            ex_rot.Set(math.degrees(ex_sign * abs(ang)))
        after = _placard_centroid(stage, path)
        moved = sum((a - b) ** 2 for a, b in zip(after, before)) ** 0.5
        # Put it back: the assertion must not leave the world open.
        ctrl["rot"].Set(0.0)
        ctrl["cur_angle"] = 0.0
        for ex_rot, _sign in ctrl.get("extra_leaves", ()):
            ex_rot.Set(0.0)
        if moved < PLACARD_RIDE_MIN_M:
            bad.append(
                f"{key} ({path}) moved {moved * 1000:.1f} mm when its door swung "
                f"{math.degrees(ang):.0f} deg -- it is hanging in the opening, not on the leaf"
            )
    if bad:
        raise SpecBuildError("placard does not ride its door: " + "; ".join(bad))


def assert_container_accepts(world: World, before: set) -> None:
    """⭐ CONTAINER ACCEPTANCE. Drop a probe into every release site of every registered container
    and demand it reads back ``INSIDE`` through the SAME ``_relations_inside_live`` the scorer runs.

    A container that takes a ``[PutInside]`` with zero failed primitives and then does not read back
    INSIDE is the worst defect this benchmark can ship: the command succeeds, the object leaves the
    score, and the row looks like a task failure rather than a world failure. Measured on two ADDED
    fridge bodies, it cost one row its oracle 1.00 and made another's control fire the wrong
    constraint. So it FAILS THE BUILD, by name -- never a warning."""
    keys = container_defects(world)
    bad = [
        f"{k} ({world.placed[k][1]['cat']}/{world.placed[k][1]['model']}) over its top "
        f"{min(sites_needed(world, k), len(probe_sites(world, k)))} of "
        f"{len(probe_sites(world, k))} release site(s)"
        for k in keys
    ]
    after = measured_inside(world)
    if after != before:
        raise SpecBuildError(
            f"the acceptance probe did not restore the world: {after ^ before}"
        )
    if bad:
        raise SpecBuildError(
            "SILENT CONTAINER -- a [PutInside] would succeed with 0 failed primitives and the "
            f"object would not read back INSIDE: {bad}. Swap the body or fix its drop."
        )
    n = sum(
        min(sites_needed(world, u.key), len(probe_sites(world, u.key)))
        for u in world.recipe.units
    )
    print(
        f"[bench] container ACCEPTANCE: {len(world.recipe.units)} unit(s), {n} release site(s) "
        f"probed, all INSIDE and clear of the unit's own collision hulls"
    )


# =====================================================================================
# stage 5b -- GRASP ACCEPTANCE. The container probe, one layer up: a [Grab] that comes back
# with an EMPTY HAND is scored as a task failure, not as a world failure.
# =====================================================================================
# The pick reach envelope, MEASURED on the shipped rig through the real ``produce_pick`` with the
# pose planner on (``scene.unified_geom``, which ``light_and_flag`` sets and step3 sets for every
# scene): z = 0.30 / 0.60 / ... / 2.00 m all seat from a 0.40 or 0.55 m stand; z = 2.67 m is 677 mm
# short from EVERY stand distance. Used only to DEMOTE a staging surface, never to veto one -- a
# room whose only surface is high still builds, and then fails at the probe by name. (2.67 m is not
# hypothetical: ``_pick_support`` ranks by approach clearance alone, and on T6-RURB-01 that elected
# ``fixed_window/vapbnz`` -- a WINDOW, 2.62 m up -- as the pass, which is the whole 0.00.)
GRASP_REACH_MAX_Z_M: float = 2.00


def reach_max_z() -> float:
    """``GRASP_REACH_MAX_Z_M``, overridable by ``BENCH_REACH_MAX_Z`` so the demotion can be A/B'd
    against a build that ranks staging surfaces by approach clearance alone (what shipped before).
    A ranking term that changes which world gets built has to be falsifiable without an edit."""
    return float(os.environ.get("BENCH_REACH_MAX_Z", GRASP_REACH_MAX_Z_M))


# How far the object's origin may sit from the wrist origin in the solved grip. The seat the rig
# carries is 0.11 m (``set_object: seat_t=[-0.028, 0.0782, 0.073]``) and ``produce_pick`` already
# refuses a seat error over its own 30/90 mm tol, so this only catches a grip that reports success
# while holding the body at arm's length.
GRASP_SEAT_MAX_M: float = 0.35
GRASP_KEY: str = "__grasp_probe__"
_MAX_GRASP_REPAIRS: int = 8
_PROBE_RIG = None


def probe_rig():
    """The rig ``episode_runtime.setup_episode`` hands the executor -- built once per process.

    The same three calls in the same order, because a probe that solved the reach on a DIFFERENT
    arm would be testing a human the render never poses. ``derive_rest_place_arms`` is two batched
    IK solves (~13 s here), so it is paid once and cached for every item of every attempt."""
    global _PROBE_RIG
    if _PROBE_RIG is None:
        import smplh_anim_core as core

        rig = core.load_family("side_wrap")
        build, otw = core.load_object_def("cup")
        core.set_object(rig, build, otw)
        core.derive_rest_place_arms(rig)
        _PROBE_RIG = rig
    return _PROBE_RIG


def _stand_yaw(stand, obj) -> float:
    """The heading that puts ``obj`` straight in front of a human at ``stand`` -- the same
    ``fwd = (sin yaw, -cos yaw)`` convention ``_pick_support`` reserves its approach cell with."""
    dx, dy = float(obj[0]) - float(stand[0]), float(obj[1]) - float(stand[1])
    if math.hypot(dx, dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dx, -dy))


def grasp_stands(world: World, key: str) -> list[tuple[float, float, float]]:
    """Where the probe may stand to pick ``key`` up. Every one of these is a stand the BUILDER
    itself already reserved or derived.

    ``init_validation.pick_stand_runtime`` picks the NEAREST REACHABLE cell at render time, so a
    probe pinned to one hand-chosen stand would be STRICTER than the runtime it is modelling and
    would condemn a long bench whose far end is simply approached from another cell. Use a bounded
    5 cm ladder across the runtime's near-contact band: thin, low objects can need one cell closer
    than a bulky object even on the same support, and every rung still runs the real reach and
    collision gates."""
    ent = world.scene.entities[key]
    ox, oy = float(ent.xy[0]), float(ent.xy[1])
    yaw = float(getattr(ent, "approach_yaw", 0.0) or 0.0)
    a = math.radians(yaw)
    fwd = (math.sin(a), -math.cos(a))
    out = [
        (ox - fwd[0] * d, oy - fwd[1] * d, yaw)
        for d in (0.55, 0.50, 0.45, 0.40, 0.35, 0.30)
    ]
    right = (math.cos(a), math.sin(a))
    out += [
        (
            ox - fwd[0] * distance + right[0] * lateral,
            oy - fwd[1] * distance + right[1] * lateral,
            _stand_yaw(
                (
                    ox - fwd[0] * distance + right[0] * lateral,
                    oy - fwd[1] * distance + right[1] * lateral,
                ),
                (ox, oy),
            ),
        )
        for distance in (0.45, 0.40, 0.35)
        for lateral in (-0.10, -0.05, 0.05, 0.10)
    ]
    out += [
        (float(sx), float(sy), _stand_yaw((sx, sy), (ox, oy)))
        for sx, sy in world.support_stands
    ]
    return out


def _grasp_once(world: World, key: str, stand) -> tuple[bool, str]:
    """Run the REAL ``produce_pick`` from ONE stand and read back whether the hand is holding.

    Dispatched through ``semantic_executor.step_one`` -- the executor's own entry point -- so the
    reach plan, the CCD warp, the live per-shape SDF finger wrap and the failure taxonomy are the
    render's, not a re-implementation."""
    import numpy as np
    import semantic_executor as se
    from goal_schema import Goal, Primitive

    rig = probe_rig()
    rig._stage = world.stage
    rig._obj_path = world.scene.entities[key].prim_path
    st = se.make_state(rig, world.scene, start_xy=(float(stand[0]), float(stand[1])))
    st.root_yaw = float(stand[2])
    frames, reason, detail = se.step_one(
        rig, None, world.scene, st, Goal(prim=Primitive.PICK, obj_key=key)
    )
    if st.held is None:
        return False, f"{getattr(reason, 'value', reason)} {detail}".strip()
    carry = [f.obj[1] for f in frames if f.obj and f.obj[0] == "carry"]
    if not carry:
        return False, "the hand reports HELD but no frame carries the object"
    if len({id(t) for t in carry}) != 1:
        return (
            False,
            f"the object rides {len({id(t) for t in carry})} wrist transforms, not one",
        )
    # ``otw`` is the standard COLUMN-vector 4x4 (``wrist_world_T`` writes T[:3, 3]), not the USD row
    # convention the rest of this module reads bboxes with.
    off = float(np.linalg.norm(np.asarray(st.held["otw"], dtype=float)[:3, 3]))
    if off > GRASP_SEAT_MAX_M:
        return (
            False,
            f"degenerate grip: the object sits {off * 1000:.0f}mm from the wrist",
        )
    return True, f"{len(carry)} carry frame(s), grip seat {off * 1000:.0f}mm"


def _build_probe_prim(world: World, it: ItemRecipe) -> str:
    """Reference the graspable's body in at a SCRATCH path, at the pose it was staged at.

    Two reasons it is not built at the item's own ``/World/Obj_<key>``. The graspables are
    published as ``scene.grasp_objs`` and only built at RENDER time, so without a real body
    ``object_origin`` falls back to the registered pose and the live finger wrap has no mesh to
    close on -- and by the time the gate runs, that path already carries the item's cue DECAL, so
    a probe that reused it would either skip the body (and silently grasp the decal Xform's
    identity transform) or build-and-remove under the artwork the render is about to mount."""
    import behavior_import as bi
    import fpv_scene

    ent = world.scene.entities[it.key]
    pp = f"/World/Obj_{bi._safe(GRASP_KEY)}"
    prim = fpv_scene.add_reference(
        world.stage, pp, bi.baked_usd("/".join(staged_body(world, it)))
    )
    _seat_prim_base(
        world,
        prim,
        pp,
        float(ent.xy[0]),
        float(ent.xy[1]),
        float(ent.z),
        _rest_orientation(ent),
        scale=it.scale,
    )
    return pp


def grasp_defects(world: World, only: set[str] | None = None) -> list[tuple[str, str]]:
    """The staged graspables NO reserved stand can pick up -> [(key, why)]. Keys, so the
    caller can repair. A key is a defect only when EVERY candidate stand fails, so a stand the
    runtime would never choose can never condemn a row on its own."""
    # The probe re-points an entity at a scratch body, so it must hand BOTH the prim path and the
    # world-AABB cache back exactly as it found them -- including the ABSENCE of a cache entry.
    # Popping instead of restoring is not neutral: the render's own geometry (solid boxes, nav,
    # containment) reads that cache, and a build whose graspables were merely UNCACHED renders a
    # measurably different episode (T6-MER-01: 2321 frames vs 2305).
    #
    # ⭐ AND IT MUST HAND BACK THE SYMBOLIC STATE. ``_grasp_once`` runs the executor's REAL
    # ``produce_pick``, which ends in ``scene.declare_pick(key)`` -- ``scene.held.add(key)`` plus
    # ``scene.on.pop(key)``. Restoring the prim pose is not restoring that. Left unrestored, every
    # graspable the probe successfully picked up stays in ``SceneState.held`` for the life of the
    # built world, so the episode SHIPS with the hand already full: ``T1-DINR-02`` starts "holding"
    # all three tubs, the loop's observation reads ``held: [item_1, ...]`` on turn 1, and the human
    # refuses to [Grab] what it is told it already has. A replayed ``--check`` program never notices
    # (it calls PICK regardless and scores containment from physics), which is why the sweep reads
    # 1.00 on rows the eval loop reads 0.00. ``assert_hand_empty`` below makes it loud.
    aabb = getattr(world.scene, "_static_aabb", {})
    held0 = set(getattr(world.scene, "held", None) or ())
    on0 = dict(getattr(world.scene, "on", None) or {})
    bad: list[tuple[str, str]] = []
    try:
        for it in world.recipe.staged():
            if only is not None and it.key not in only:
                continue
            if _is_observation_only_exact_object(
                world.recipe.task_id, it.key, it.cat, it.model
            ):
                continue
            ent = _registered_entity(world, it.key, phase="grasp acceptance")
            keep, why, cached = (
                str(ent.prim_path),
                "",
                (it.key in aabb, aabb.get(it.key)),
            )
            ent.prim_path = _build_probe_prim(world, it)
            try:
                for stand in grasp_stands(world, it.key):
                    ok, why = _grasp_once(world, it.key, stand)
                    if ok:
                        break
                else:
                    bad.append((it.key, why))
                    continue
                print(
                    f"[bench][GRASP] {it.key} = {'/'.join(staged_body(world, it))} HELD -- {why}"
                )
            finally:
                world.stage.RemovePrim(ent.prim_path)
                ent.prim_path = keep
                aabb.pop(it.key, None)
                if cached[0]:
                    aabb[it.key] = cached[1]
    finally:
        dirty = sorted((getattr(world.scene, "held", None) or set()) - held0)
        if hasattr(world.scene, "held"):
            world.scene.held.clear()
            world.scene.held.update(held0)
        if hasattr(world.scene, "on"):
            world.scene.on.clear()
            world.scene.on.update(on0)
        # Printed every build, not only on failure: this is the line that tells whoever reads the
        # log that the hand the episode ships with is the hand the spec asked for.
        print(
            f"[bench][HAND] grasp probe left held={dirty} -> restored to "
            f"{sorted(getattr(world.scene, 'held', None) or ())}"
        )
    return bad


def _registered_entity(world: World, key: str, *, phase: str):
    entity = world.scene.entities.get(key)
    if entity is not None:
        return entity
    raise SimulatorBuildError(
        "SCENE_ENTITY_NOT_REGISTERED",
        f"{world.recipe.task_id}: {phase} requires registered entity {key!r}",
        context={"task_id": world.recipe.task_id, "phase": phase, "entity": key},
    )


def _assert_container_entities_registered(world: World) -> None:
    required = {
        *(unit.key for unit in world.recipe.units),
        *(item.key for item in world.recipe.staged()),
        *(key for preload in world.recipe.preloads for key in preload.keys),
    }
    missing = sorted(required - set(world.scene.entities))
    if missing:
        raise SimulatorBuildError(
            "SCENE_ENTITY_NOT_REGISTERED",
            f"{world.recipe.task_id}: container build left entities unregistered: {missing}",
            context={
                "task_id": world.recipe.task_id,
                "phase": "container build",
                "entities": missing,
            },
        )


def _cue_item(world: World) -> ItemRecipe:
    cue = next(iter(world.recipe.staged()), None)
    if cue is None:
        cue = next(
            (
                ItemRecipe(
                    preload.keys[0],
                    preload.cat,
                    preload.model,
                    (),
                    (),
                    initial_place=f"in:{preload.unit}",
                )
                for preload in world.recipe.preloads
                if preload.keys
            ),
            None,
        )
    if cue is None:
        raise SimulatorBuildError(
            "SCENE_ENTITY_NOT_REGISTERED",
            f"{world.recipe.task_id}: container build has no named cue item",
            context={"task_id": world.recipe.task_id, "phase": "start cue"},
        )
    _registered_entity(world, cue.key, phase="start cue")
    return cue


def assert_hand_empty(world: World) -> None:
    """⭐ THE HAND IS EMPTY AT t=0 unless the spec says otherwise -- a HARD build failure.

    Same defect shape as ``assert_container_accepts`` and ``assert_grasp_accepts``: the engine
    reports success, the world looks fine, and the task silently becomes unwinnable. A world that
    ships with ``SceneState.held`` non-empty tells the human it is already carrying the very objects
    the task is about, so every ``[Grab]`` is refused as redundant and every ``[PutInside]`` acts on
    a phantom grasp -- with no failed primitive anywhere. Measured on 8/8 recorded episodes before
    ``grasp_defects`` learned to restore the symbolic state.

    No spec in the portfolio stages an item IN THE HAND; if one ever does it must say so, and this
    is where it would be declared."""
    held = sorted(getattr(world.scene, "held", None) or ())
    if not held:
        return
    raise SpecBuildError(
        f"{world.recipe.task_id}: the episode would start with the hand ALREADY HOLDING {held}. "
        f"No spec declares a staged-in-hand item, so every [Grab] on {held} would be refused as "
        f"redundant and every following put would act on a phantom grasp, scoring 0.00 with no "
        f"failed primitive. Something in the build ran a PICK and did not hand the symbolic state "
        f"back (see grasp_defects)."
    )


def assert_grasp_accepts(world: World, before: set) -> None:
    """⭐ GRASP ACCEPTANCE. Pick every staged graspable up with the executor's OWN producer and
    demand the hand comes back holding it.

    This is ``assert_container_accepts`` one layer up, and the same defect shape: the command
    reports, the world looks fine, and the row silently becomes unscoreable. Measured on
    T6-RURB-01, where the staging surface elected by approach clearance alone was a WINDOW 2.62 m
    up: every ``[Grab]`` came back ``out_of_reach``, every following ``[PutInside]`` failed
    ``place_unstable (PLACE with an empty hand)``, and the row scored 0.00 with no build error.
    So it FAILS THE BUILD, by name -- never a warning."""
    bad = grasp_defects(world)
    after = measured_inside(world)
    if after != before:
        raise SpecBuildError(
            f"the grasp probe did not restore the world: {after ^ before}"
        )
    if bad:
        named = [
            f"{k} ({'/'.join(staged_body(world, _staged_of(world, k)))}): {why}"
            for k, why in bad
        ]
        raise SpecBuildError(
            f"{world.recipe.task_id}: UNGRASPABLE STAGING -- a [Grab] would come back with an "
            f"EMPTY HAND and every following [PutInside] would fail place_unstable, scoring 0.00 "
            f"with no failed primitive of its own: {named}. Re-stage on a reachable surface or "
            f"swap the body."
        )
    print(
        f"[bench] grasp ACCEPTANCE: "
        f"{sum(not _is_observation_only_exact_object(world.recipe.task_id, it.key, it.cat, it.model) for it in world.recipe.staged())} "
        f"graspable(s) picked up with the "
        f"executor's own produce_pick, every hand non-empty and rigidly carrying"
    )


def _staged_of(world: World, key: str) -> ItemRecipe:
    return next(it for it in world.recipe.staged() if it.key == key)


def cue_stands(
    world: World,
) -> dict[str, tuple[float, float, float, float, float, float, float]]:
    """Every cue-bearing entity -> stand, yaw, face, and artwork-center coordinates.

    ``face`` is the OUTWARD NORMAL of the mounted artwork, measured off the same ``front_dir`` the
    decal was hung on, and ``yaw`` is the heading that turns a reader TOWARDS it. Published into
    ``behavior_task`` so the render side can PIN the pose instead of re-picking a nearer one that
    happens to stand behind the board."""
    out: dict[str, tuple[float, float, float, float, float, float, float]] = {}
    for key, fd in world.front_dir.items():
        ent = world.scene.entities.get(key)
        stand = ent.meta.get("front_stand") if ent is not None else None
        if stand is None:
            continue
        face = ent.meta.get("label_face", fd)
        if not isinstance(face, (list, tuple)) or len(face) < 2:
            face = fd
        nx, ny = float(face[0]), float(face[1])
        label_pos = ent.meta.get("label_pos")
        cue_xy = (
            (float(label_pos[0]), float(label_pos[1]))
            if isinstance(label_pos, (list, tuple)) and len(label_pos) >= 2
            else (float(ent.xy[0]), float(ent.xy[1]))
        )
        out[key] = (
            float(stand[0]),
            float(stand[1]),
            math.degrees(math.atan2(nx, -ny)) % 360.0,
            nx,
            ny,
            *cue_xy,
        )
    for mount in _printed_wall_entries(_printed_block(world.spec)).values():
        key = str(mount["on"])
        if key in out:
            continue
        ent = world.scene.entities.get(key)
        meta = ent.meta if ent is not None else {}
        face, stand = meta.get("label_face"), meta.get("front_stand")
        if (
            not isinstance(face, (list, tuple))
            or len(face) < 2
            or not isinstance(stand, (list, tuple))
            or len(stand) < 2
        ):
            continue
        nx, ny = float(face[0]), float(face[1])
        label_pos = meta.get("label_pos")
        cue_xy = (
            (float(label_pos[0]), float(label_pos[1]))
            if isinstance(label_pos, (list, tuple)) and len(label_pos) >= 2
            else (float(ent.xy[0]), float(ent.xy[1]))
        )
        out[key] = (
            float(stand[0]),
            float(stand[1]),
            math.degrees(math.atan2(nx, -ny)) % 360.0,
            nx,
            ny,
            *cue_xy,
        )
    for key in (world.recipe.manual_key, world.recipe.card_key):
        ent = world.scene.entities.get(key) if key else None
        if ent is None or "label_face" not in ent.meta or "front_stand" not in ent.meta:
            continue
        nx, ny = ent.meta["label_face"]
        sx, sy = ent.meta["front_stand"]
        out[key] = (
            float(sx),
            float(sy),
            math.degrees(math.atan2(nx, -ny)) % 360.0,
            float(nx),
            float(ny),
            float(ent.meta.get("label_pos", ent.xy)[0]),
            float(ent.meta.get("label_pos", ent.xy)[1]),
        )
    return out


# How far off square-on a cue stand may be. cos(50 deg): past this the 0.1 m of artwork stops
# resolving in the FPV, and at 90 deg the reader is looking at the blank back board.
CUE_FACE_MIN_DOT: float = 0.64


def assert_cue_facing(world: World) -> None:
    """CUE-OBSERVED. Every declared cue stand must be on the ARTWORK's side of its own surface.

    A stand that is merely NEAR the cue scores the assistant on a sign it was never shown: the
    oracle routes from the answer key, so ``check_goal`` still reaches 1.00 while the FPV holds a
    blank back board for the whole episode. Measured off the same ``front_dir`` the decal was hung
    on, so the geometry that MOUNTS the artwork and the geometry that READS it cannot drift apart.

    NOTE the limit of what this file can guarantee: the runtime stand re-pick lives in
    ``episode_runtime``/``init_validation`` and may still discard the declared pose. This asserts the
    DECLARATION is right and publishes it as ``behavior_task["cue_stands"]``; honouring it is the
    render side's half."""
    bad = []
    for key, cue in cue_stands(world).items():
        sx, sy, _yaw, fx, fy = cue[:5]
        entity = world.scene.entities[key]
        label_pos = entity.meta.get("label_pos")
        cx, cy = (
            (float(label_pos[0]), float(label_pos[1]))
            if isinstance(label_pos, (list, tuple)) and len(label_pos) >= 2
            else entity.xy
        )
        dx, dy = sx - float(cx), sy - float(cy)
        n = math.hypot(dx, dy) or 1.0
        dot = (dx / n) * fx + (dy / n) * fy
        if dot < CUE_FACE_MIN_DOT:
            off = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            bad.append(
                f"{key}: stand is {off:.0f} deg off the artwork normal ({fx:+.2f},{fy:+.2f})"
            )
    if bad:
        raise SpecBuildError(
            "CUE-OBSERVED: a declared cue stand does not FACE its artwork -- the episode would "
            f"score a sign it never showed: {bad}"
        )
    print(
        f"[bench] CUE-OBSERVED: {len(cue_stands(world))} cue stand(s) square-on to their artwork"
    )


# --------------------------------------------------------------------------------------------------
#  CUE-LEGIBLE: the artwork must survive the DELIVERED downscale, not just exist in the world
# --------------------------------------------------------------------------------------------------
# ``assert_cue_facing`` asks whether the reader is on the right SIDE of the board. It passed 3/3 on a
# row whose frames delivered 1/3 readable placards, because "in front of it" and "big enough to read"
# are different questions and only the first was ever asked. Everything the cue work of 2026-08-11
# turned up -- unit_a's black frame, GROCA's knee-height crates, the T1-OFFC-01 cell at 36x54 px -- is
# the second question going unasked.
#
# The measurement is ANALYTIC, not a render: the FPV camera is a pinhole with a known aperture and a
# known focal cap, and the look-zoom law is 20 lines of arithmetic. Validated against the one
# independently measured number in the record -- the T1-OFFC-01 cell decal, measured by cropping a real
# delivered frame at 36 x 54 px; this model predicts 39 px for the same decal (see ``_selftest_cue_px``).
READER_EYE_Z: float = 1.60  # standing FPV eye height (smplh_camera: head + EYE_OFFSET)
FPV_DELIVERED_W_PX: int = (
    512  # fidelity_probe --frame-max-side default: the width a model is SHOWN
)
INSPECT_FWD_M: float = (
    0.40  # semantic_executor._INSPECT_FWD_M: a held item's distance from the eye
)
# Floors on the artwork's LONGER on-screen dimension, in delivered pixels.
#   TEXT  -- the discriminator is printed words (every unit placard). The caption band is ~0.2 of the
#            label and the fitted glyphs ~0.5 of that, so 90 px of label is ~9 px of cap height, which
#            is the bottom of the documented Latin legibility range.
#   ICON  -- the discriminator is a pictogram (every item decal). 36 px was read correctly at true size
#            by a human on 2026-08-11 (EVIDENCE/I_offc_icons_BEFORE_AFTER_true36x54.png); 32 is that
#            with a little margin, and it is deliberately NOT a text floor.
CUE_MIN_PX_TEXT: float = 60.0
CUE_MIN_PX_ICON: float = 18.0
# ⛔⛔ THE LOOK-ZOOM IS INERT ON THE PATH THE MODELS SEE. `episode_runtime.drive_cams` raises the focal on
# a look-zoom frame so the cue fills ~45 % of the width -- and `isaac_env._render_one`, which renders
# EVERY frame an eval episode or an RL rollout delivers, never touches the focal at all: it calls
# `smplh_camera.drive_fpv_gaze` directly and leaves FOCAL_MM at 16. Only the offline video render
# (`run_render`) gets the zoom. Measured, not inferred: with the zoom the model predicts 125 px for the
# T1-OFFC-01 cell decal and 222 px for its bin placards; the delivered frames measure ~22 px and ~70 px,
# and the no-zoom model predicts 21 px and 74 px. So the floors below are calibrated to the optics that
# actually ship. Porting the ~10-line focal block into `_render_one` would multiply every cue by ~3x --
# the single largest legibility win available -- but `isaac_env` is shared with live RL training, so it
# is REPORTED rather than shipped here.
CUE_LOOK_ZOOM_DELIVERED: bool = False
# Above this depression angle from the eye a vertical placard is being read edge-on and its own
# container's rim starts to cut it. Warned, not raised: it is a strong predictor, not a proof.
CUE_MAX_DEPRESSION_DEG: float = 55.0


def cue_px(
    span_m: float,
    dist_m: float,
    delivered_w_px: int = FPV_DELIVERED_W_PX,
    zoom_span_m: float | None = None,
) -> float:
    """On-screen size, in DELIVERED pixels, of ``span_m`` of artwork read from ``dist_m``.

    Reproduces the render's own optics exactly (``episode_runtime`` look-zoom + ``smplh_camera``): the
    zoom raises the focal until the target subtends 45 % of frame width, but never past ``FOCAL_MM x
    6`` -- and it is that cap, not the camera, that decides whether a small cue survives. Below the
    cap every cue lands at 0.45 x width; above it the cue shrinks linearly with its own size, which is
    why a 0.037 m battery and a 0.46 m placard are worlds apart in readability while both are "in the
    frame".

    ``zoom_span_m`` is what the zoom is AIMED at, when that differs from what must be READ: a [LookAt]
    on a graspable sizes the zoom off the whole object while the decal is a fraction of one face.
    Conflating the two flatters every item row by ~3x.
    """
    import smplh_camera

    span_m, dist_m = float(max(span_m, 0.0)), float(max(dist_m, 1e-3))
    if span_m <= 0.0:
        return 0.0
    focal = float(smplh_camera.FOCAL_MM)
    if CUE_LOOK_ZOOM_DELIVERED:
        aim = float(zoom_span_m) if zoom_span_m else span_m
        hfov0 = 2.0 * math.atan(0.5 * smplh_camera.H_APERTURE / focal)
        hfov_t = min(hfov0, (2.0 * math.atan(0.5 * aim / dist_m)) / 0.45)
        focal = min(0.5 * smplh_camera.H_APERTURE / math.tan(hfov_t / 2.0), focal * 6.0)
    subtend = 2.0 * math.atan(0.5 * span_m / dist_m)
    return float(delivered_w_px) * (
        focal * math.tan(subtend / 2.0) / (0.5 * smplh_camera.H_APERTURE)
    )


def _cue_detail_contract(meta: dict, span: float, held: bool) -> tuple[str, float, float]:
    ratio = float(meta.get("label_primary_glyph_height_ratio") or 0.0)
    size = meta.get("label_size") or ()
    if ratio > 0.0:
        import label_visibility as lv

        height = float(size[1]) if len(size) >= 2 else span
        return "text", height * ratio, float(lv.MIN_GLYPH_PX)
    return (
        ("icon", span, CUE_MIN_PX_ICON)
        if held
        else ("text", span, CUE_MIN_PX_TEXT)
    )


def _required_observation_cues(spec: Mapping[str, object]) -> set[str]:
    explicit = spec.get("required_visual_cue_ids") or ()
    required = (
        {str(key) for key in explicit}
        if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes))
        else set()
    )
    if spec.get("template") != "long_horizon_episodic_recall":
        return required
    printed = spec.get("printed")
    if not isinstance(printed, Mapping):
        return required
    printed_keys: set[str] = set()
    for group in ("items", "units"):
        entries = printed.get(group)
        if isinstance(entries, Mapping):
            printed_keys.update(str(key) for key in entries)
    walls = printed.get("walls")
    if isinstance(walls, Mapping):
        printed_keys.update(
            str(row.get("on")) for row in walls.values() if isinstance(row, Mapping)
        )
    looks = set(authored_look_keys(str(spec.get("human_program") or "")))
    return required | (printed_keys & looks)


def _missing_cue_row(
    *, key: str, answers: set[str], required: set[str], reason: str, span: float = 0.0
) -> dict[str, object]:
    return {
        "kind": "text",
        "span_m": round(span, 4),
        "dist_m": 0.0,
        "depression_deg": 0.0,
        "px": 0.0,
        "floor_px": CUE_MIN_PX_TEXT,
        "is_answer": key in answers,
        "required": key in required,
        "ok": False,
        "reason": reason,
    }


def _cue_optics(
    world: World, key: str, meta: Mapping[str, object], held: bool
) -> tuple[float, float, float] | str:
    span = float(meta.get("label_span") or 0.0)
    if held:
        (mnx, mny, mnz), (mxx, mxy, mxz) = world.scene.object_aabb(key)
        zoom_span = 0.5 * max(float(mxz - mnz), float(min(mxx - mnx, mxy - mny)))
        return INSPECT_FWD_M, 0.0, zoom_span
    stand, lpos = meta.get("front_stand"), meta.get("label_pos")
    if lpos is None:
        return "missing_label_position"
    if stand is None:
        return "missing_front_stand"
    dx, dy = float(lpos[0]) - float(stand[0]), float(lpos[1]) - float(stand[1])
    dz = READER_EYE_Z - float(lpos[2])
    flat = math.hypot(dx, dy)
    return math.hypot(flat, dz), math.degrees(
        math.atan2(abs(dz), max(flat, 1e-3))
    ), span


def _cue_legibility_row(
    world: World,
    key: str,
    ent: object,
    *,
    held_keys: set[str],
    answers: set[str],
    required: set[str],
    delivered_w_px: int,
) -> dict[str, object] | None:
    meta = getattr(ent, "meta", None) or {}
    span = float(meta.get("label_span") or 0.0)
    if span <= 0.0:
        return (
            _missing_cue_row(
                key=key, answers=answers, required=required, reason="missing_label_span"
            )
            if key in required
            else None
        )
    held = key in held_keys
    optics = _cue_optics(world, key, meta, held)
    if isinstance(optics, str):
        return (
            _missing_cue_row(
                key=key,
                answers=answers,
                required=required,
                reason=optics,
                span=span,
            )
            if key in required
            else None
        )
    dist, depression, zoom_span = optics
    kind, detail_span, floor_px = _cue_detail_contract(meta, span, held)
    px = cue_px(detail_span, dist, delivered_w_px, zoom_span_m=zoom_span)
    return {
        "kind": kind,
        "span_m": round(span, 4),
        "dist_m": round(dist, 3),
        "depression_deg": round(depression, 1),
        "px": round(px, 1),
        "floor_px": floor_px,
        "is_answer": key in answers,
        "required": key in required,
        "ok": px >= floor_px,
        "reason": None,
    }


def cue_legibility(world: World, delivered_w_px: int = FPV_DELIVERED_W_PX) -> dict:
    """Per cue-bearing entity: how big its artwork is on screen, from the stand the reader will use.

    Typeset labels use their measured glyph height, whether mounted on a unit or an item. Legacy
    textures without readability metadata retain the older text/icon fallback floors.
    """
    rows: dict[str, dict[str, object]] = {}
    answers = {
        str(g.get("target"))
        for g in (world.spec.get("answer_key") or [])
        if isinstance(g, dict)
    }
    required = _required_observation_cues(world.spec)
    held_keys = {item.key for item in world.recipe.staged()}
    for key, ent in world.scene.entities.items():
        row = _cue_legibility_row(
            world,
            key,
            ent,
            held_keys=held_keys,
            answers=answers,
            required=required,
            delivered_w_px=delivered_w_px,
        )
        if row is not None:
            rows[key] = row
    for key in sorted(required - set(world.scene.entities)):
        rows[key] = _missing_cue_row(
            key=key,
            answers=answers,
            required=required,
            reason="missing_scene_entity",
        )
    return rows


def _report_cue_legibility(rows: Mapping[str, Mapping[str, object]]) -> None:
    for key in sorted(rows):
        row = rows[key]
        flag = "OK " if row["ok"] else "LOW"
        suffix = f" reason={row['reason']}" if row.get("reason") else ""
        print(
            f"[bench] CUE-LEGIBLE {flag} {key}: {row['px']:.0f} px "
            f"(floor {row['floor_px']:.0f}, {row['kind']}) span={row['span_m']:.3f}m "
            f"dist={row['dist_m']:.2f}m depression={row['depression_deg']:.0f}deg"
            f"{' ANSWER' if row['is_answer'] else ''}"
            f"{' REQUIRED' if row['required'] else ''}{suffix}"
        )
    graze = [
        f"{key} at {row['depression_deg']:.0f} deg"
        for key, row in sorted(rows.items())
        if row["depression_deg"] > CUE_MAX_DEPRESSION_DEG
    ]
    if graze:
        print(
            f"[bench] CUE-LEGIBLE WARN: read edge-on, the container's own rim will cut them: {graze}"
        )


def _cue_failure_description(key: str, row: Mapping[str, object]) -> str:
    if row.get("reason"):
        return f"{key}: {row['reason']}"
    return (
        f"{key}: {row['px']:.0f} px < {row['floor_px']:.0f} "
        f"({row['kind']}, span {row['span_m']:.3f} m at {row['dist_m']:.2f} m)"
    )


def _cue_failures(rows: Mapping[str, Mapping[str, object]]) -> tuple[list[str], list[str]]:
    bad: list[str] = []
    nonblocking: list[str] = []
    for key, row in sorted(rows.items()):
        if row["ok"]:
            continue
        blocking = bool(row["required"] or row["is_answer"] or row["kind"] == "icon")
        if blocking:
            bad.append(_cue_failure_description(key, row))
        else:
            nonblocking.append(key)
    return bad, nonblocking


def assert_cue_legible(world: World) -> None:
    """Require answer and observation-evidence cues to survive delivered downscaling.

    Observation episodes additionally require every printed cue named by an authored ``LookAt``.
    Low non-evidence text remains advisory for compatibility with benchmark decoys.

    ``BENCH_CUE_LEGIBLE`` still defaults to ``warn`` for legacy callers. Strict TaskPack validation
    arms it explicitly.
    """
    rows = cue_legibility(world)
    if not rows:
        print(
            "[bench] CUE-LEGIBLE: no cue-bearing entity in this world -- nothing to measure"
        )
        return
    _report_cue_legibility(rows)
    bad, nonblocking = _cue_failures(rows)
    if not bad:
        if nonblocking:
            print(
                f"[bench] CUE-LEGIBLE WARN: {len(nonblocking)} non-required cue(s) below "
                f"their floor: {nonblocking}; every required discriminator is above its floor"
            )
            return
        print(
            f"[bench] CUE-LEGIBLE: {len(rows)} cue(s) measured, every discriminator above its floor"
        )
        return
    msg = (
        "CUE-LEGIBLE: a discriminator does not survive the delivered downscale -- the assistant "
        f"would be scored on a cue it was never shown: {bad}"
    )
    if os.getenv("BENCH_CUE_LEGIBLE", "warn") != "raise":
        print(f"[bench] ⚠️ CUE-LEGIBLE WARN (set BENCH_CUE_LEGIBLE=raise to arm): {msg}")
        return
    raise SpecBuildError(msg)


def assert_not_presatisfied(world: World, goals: list[tuple]) -> set:
    """Measure goals already true at t=0 for baseline-excluded scoring.

    Accepts ``(subject, target)`` or ``(pred, subject, target)``. The pair form is read as INSIDE,
    which is what every caller meant when INSIDE was the only scored predicate; a triple says so
    itself. ``relations(source="geometric")`` has always emitted ON, so an ON goal that is already
    true at t=0 is measured by the same line. The live scorer captures the complete physical
    baseline after setup and excludes such rows from achievement credit while retaining them as
    final-state constraints."""
    import spec_validator as sv
    from spec_validator import Goal

    rel0 = world.scene.relations(source="geometric")
    want = [Goal(*g) if len(g) == 3 else Goal("inside", *g) for g in goals]
    pre = sv.pre_satisfied_predicates(want, rel0)
    if pre:
        logger.info(
            "[bench] t=0 baseline-excluded goals: %s",
            [(g.pred, g.subject, g.target) for g in pre],
        )
    inside0 = sorted((a, b) for (p, a, b) in rel0 if p.name == "INSIDE")
    logger.info(
        "[bench] t=0 relations: %d total, INSIDE=%s; baseline_excluded=%d",
        len(rel0),
        inside0 or "[]",
        len(pre),
    )
    return rel0


def assert_look_goal(world: World, program: str, cue_keys: list[str]) -> None:
    """The audited failure: a generated program that drops the authored ``[LookAt]`` still
    scores 1.00 while the human never looks at the cue. Compile and prove the goal is there."""
    import command_grammar as cg
    from goal_schema import Primitive

    goals = cg.compile_program(program, world.scene)
    looked = {g.obj_key for g in goals if g.prim is Primitive.LOOK_AT}
    missing = [k for k in cue_keys if k not in looked]
    if missing:
        raise SpecBuildError(
            f"compiled program has no LOOK_AT goal for cue(s) {missing} (looked at {sorted(looked)}); "
            f"the human would never see the cue yet could still score 1.00"
        )
    print(
        f"[bench] LOOK_AT goals present for {sorted(looked)} ({len(goals)} goals compiled)"
    )


def assert_decals(world: World) -> None:
    r = world.recipe
    printed = _printed_block(world.spec)
    if r.cue_mode == "printed" and printed.get("sparse"):
        want = list((printed.get("units") or {}).keys()) + list(
            (printed.get("items") or {}).keys()
        )
        want += [f"wall:{key}" for key in _printed_wall_entries(printed)]
    else:
        want = (
            [u.key for u in r.units]
            + [i.key for i in r.items]
            + [p.key for p in r.parts]
        )
    if r.manual_key:
        want.append(r.manual_key)
    missing = [k for k in want if k not in world.decals]
    if missing:
        raise SpecBuildError(
            f"decal never mounted for {missing} -- the cue would be unreadable"
        )
    print(f"[bench] decals mounted: {len(world.decals)}/{len(want)}")


# =====================================================================================
# stage 8 -- lighting + scene flags (the proven restaurant-closing set)
# =====================================================================================
def enable_runtime_geometry(scene) -> None:
    """Enable the geometry behavior that step3 applies to every rendered scene.

    Acceptance probes execute real primitives before the builder returns, so they
    must use the same reach planner and tolerances as the eventual episode.
    """
    scene.soft_init = True
    scene.unified_geom = True
    scene.post_action_check = True


def light_and_flag(
    world: World, focus: tuple[float, float], spots_xy: list | None = None
) -> None:
    """Set the render flags, then hand the LIGHT FIELD to ``bench_lighting``.

    The four hand-placed spheres this used to author were tuned on the handful of scenes slice A
    rendered; a derived row can be sited in any of 51. ``rig_scene_lights`` puts lamps on one global
    3.5 m grid over the scene's OWN ceiling slabs with intensity scaled as spacing^2, so the areal
    density is the same in a bedroom and a train station, and it derives a sun for scenes that ship
    ``lawn`` / ``driveway``. It sets the field's SHAPE; ``auto_expose``'s 12-viewpoint median-p92
    meter still sets its LEVEL, and its manifest row is published beside ``autoexp``."""
    del (
        spots_xy
    )  # the rig is derived from the scene's own ceilings, not from the unit slots
    import behavior_import as bi
    import bench_lighting

    s = world.scene
    enable_runtime_geometry(s)
    s.plan_nav = True
    s.precise_nav = True
    s.tpv_chase = (
        not world.recipe.is_assembly()
    )  # an assembly needs a FIXED wide cam, not a chase
    s.fpv_headlamp = True
    s.look_zoom = bool(
        world.recipe.look_zoom
    )  # adaptive FPV zoom on look-at -> a legible decal
    s.near_start = True  # pin the seeded init to our validated start hint
    s.auto_expose = True
    s.expose_z = 1.8 if world.recipe.is_assembly() else 1.0
    s.expose_xy = (float(focus[0]), float(focus[1]))
    if world.recipe.is_assembly():
        # The hall's own door is far off every path; clearing it makes the shared TPV gate use our
        # fixed camera rather than a door/chase cam that frames the human instead of the assembly.
        s.doors = {}
    world.light_rig = bench_lighting.rig_scene_lights(
        world.stage, world.recipe.scene, world.placed
    )
    bi._register_mdl()


# =====================================================================================
# the orchestrator
# =====================================================================================
def assert_recipe_roles(spec: dict, recipe: Recipe) -> None:
    """Refuse to build when the recipe and the spec disagree about the container role keys.

    The frozen privacy pass permutes those keys so the answer is not systematically
    ``unit_a``.  A permutation that changes the key SET lands here; one WITHIN the set is why
    the pinned rows carry ``unit_roles`` -- the ledger of which authored key became which.
    """
    missing = sorted(set(spec["roster"]["containers"]) - set(recipe.keys()))
    if missing:
        raise SpecBuildError(
            f"{spec['task_id']}: the spec roster names container key(s) {missing} the recipe does "
            f"not bind (recipe keys {sorted(recipe.keys())}). spec_privacy renamed this row "
            f"(unit_roles={spec.get('unit_roles')}); re-key the recipe to match, or pin the row in "
            f"spec_privacy.BUILDER_PINNED -- never disable the permutation."
        )


def authoring_seed(spec: dict) -> int:
    """⭐ ONE TASK = ONE FIXED WORLD. The seed that draws this row's layout, zone permutation, decoy
    classes, staged order, unit assets, numbers and letters is a property of the SPEC, not of the
    caller: ``spec["world"]["authoring_seed"]``. For a benchmark, a fixed environment per task is
    the correct thing -- the seed belongs on the MODEL side, and `spec["world"]["seed_scope"]`
    records exactly that. Sourcing it here means every downstream draw (they all take
    ``world.seed``) inherits the frozen value by construction rather than by convention."""
    return int(((spec.get("world") or {}).get("authoring_seed", 0)))


def build_spec_scene(stage, app, task_id: str, seed: int | None = None):
    """Build ``spec:<task_id>`` and return step3's 5-tuple (scene, {}, program, look, eye)."""
    spec = load_spec(task_id)
    recipe = recipe_for(task_id)
    assert_recipe_roles(spec, recipe)
    frozen = authoring_seed(spec)
    # ``BENCH_SEED`` stays an explicit A/B override (the same idiom as ``BENCH_REACH_MAX_Z``): a
    # frozen layout that cannot be varied on purpose is not falsifiable. A caller-passed ``--seed``
    # no longer chooses the world -- it is reported when it disagrees so a harness that still
    # believes it is sweeping seeds finds out in the log rather than in the numbers.
    if seed is not None and int(seed) != frozen:
        print(
            f"[bench] caller seed {int(seed)} ignored; {task_id}'s world is frozen at {frozen}"
        )
    seed = int(os.environ.get("BENCH_SEED", frozen))
    if seed != frozen:
        print(
            f"[bench] WORLD SEED overridden {frozen} -> {seed} via BENCH_SEED; "
            f"this is NOT {task_id}'s frozen layout"
        )
    world = World(stage=stage, app=app, spec=spec, recipe=recipe, seed=seed)
    if recipe.is_assembly():
        return build_assembly_scene(world)
    return build_container_scene(world)


def build_assembly_scene(world: World):
    """ASSEMBLY row (scored by ``ATTACHED``): resolve the mates off the parts' own metadata, lay the
    kit out flat, stand the rule-level manual beside it and stamp this seed's letters on the boards."""
    recipe, spec = world.recipe, world.spec
    points = attach_points_of(recipe)
    world.part_of = {p.key: p for p in recipe.parts}
    world.mates = derive_mates(spec, recipe, points)
    if not isinstance(spec.get("bts_source"), dict):
        assert_anon_connectors(world.mates, recipe)
    world.root_keys = tuple(assembly_roots(world.mates, recipe))
    world.root_key = world.root_keys[0]
    if recipe.cue_mode == "none":
        world.letters = {}
        world.cue_class = {}
        labels = assembly_label_pngs(recipe)
        _compose(world, hull_mode="floor")
        supports = _pick_supports(world, recipe.n_supports) if recipe.n_supports else []
        has_staged_support_items = any(
            item.initial_place.startswith("on:") for item in recipe.staged()
        )
        if supports and has_staged_support_items:
            supports = seat_reachable_items(world, supports)
        elif supports:
            world.supports = supports
        if supports:
            register_supports(world, supports, {})
        site_parts(world, points, supports)
        seat_floor_items(world)
        seat_movable_support_items(world)
        world.scene.board_labels = dict(labels)
        world.decals.update(
            {key: f"/World/Obj_{key}/letter_label" for key in labels}
        )
        missing = sorted(
            part.key for part in recipe.parts if part.label and part.key not in labels
        )
        if missing:
            raise SpecBuildError(f"assembly decals were not generated for {missing}")
        return _finish_assembly(world)
    world.letters = letter_permutation(recipe, world.seed)
    world.cue_class = {
        recipe.manual_key: "".join(world.letters[p.key] for p in recipe.parts)
    }
    boards = board_pngs(recipe, world.letters)
    panels = ensure_manual(
        recipe, world.seed, manual_rows(recipe, world.letters, world.mates)
    )
    _compose(world, hull_mode="floor")
    site_parts(world, points)
    site_manual(world)
    mount_manual(world, panels)
    world.scene.board_labels = dict(boards)
    world.decals.update(
        {key: f"/World/Obj_{key}/letter_label" for key in boards}
    )  # applied by the shipped render-time ``apply_board_labels``
    assert_decals(world)
    return _finish_assembly(world)


def _validated_scene_door_omissions(
    scene_name: str, requested, scene_objects
) -> tuple[str, ...]:
    names = tuple(str(name) for name in (requested or ()))
    if not names:
        return ()
    inventory = {
        str(row.get("name") or ""): str(row.get("cat") or "")
        for row in scene_objects
        if isinstance(row, Mapping) and row.get("name")
    }
    unknown = sorted(set(names) - set(inventory))
    if unknown:
        raise SpecBuildError(
            f"{scene_name} has no native scene instance(s) named {unknown}"
        )
    not_doors = sorted(name for name in names if inventory[name] != "door")
    if not_doors:
        raise SpecBuildError(
            f"scene_omit_door_instances names non-door scene instance(s) {not_doors}"
        )
    return tuple(dict.fromkeys(names))


def _compose(world: World, hull_mode: str = "furniture"):
    """Load the scene, measure the walls, and fix the anchor + the hull the slot search works in."""
    import behavior_import as bi

    recipe = world.recipe
    world.room = world.room or recipe.room
    world_spec = world.spec.get("world") or {}
    scene_omit_instances = tuple(
        str(name) for name in (world_spec.get("scene_omit_instances") or ())
    )
    requested_door_omissions = world_spec.get("scene_omit_door_instances") or ()
    scene_omit_door_instances = (
        _validated_scene_door_omissions(
            recipe.scene,
            requested_door_omissions,
            bi.scene_objects(recipe.scene),
        )
        if requested_door_omissions
        else ()
    )
    world.scene, world.placed, _surf, ctd, _ft, _dz = bi._compose_behavior_scene(
        world.stage,
        world.app,
        recipe.scene,
        None,
        None,
        None,
        skip_names=tuple(
            dict.fromkeys((*scene_omit_instances, *scene_omit_door_instances))
        ),
        include_categories=recipe.support_bindings,
    )
    world.walls = _wall_boxes(world)
    if hull_mode == "floor":
        world.anchor = assembly_anchor(world, (float(ctd[0]), float(ctd[1])))
        world.box = assembly_hull(world)
    else:
        world.anchor = room_anchor(recipe.scene, world.room)
        world.box = room_box(recipe.scene, world.room)
    print(
        f"[bench] {recipe.task_id} scene={recipe.scene} room={world.room} seed={world.seed} "
        f"anchor=({world.anchor[0]:.2f},{world.anchor[1]:.2f}) hull={[round(v, 2) for v in world.box]}"
    )


def build_container_scene(world: World):
    """CONTAINER row (scored by ``INSIDE``): the slice-A path, unchanged."""
    recipe, spec = world.recipe, world.spec
    world.numbers = draw_numbers(recipe, world.seed)
    world.zones = (
        {}
        if recipe.cue_mode == "none"
        else (
            unit_class_for(spec, recipe)
            if recipe.cue_mode == "intrinsic"
            else zone_permutation(recipe, world.seed)
        )
    )
    world.cue_class = (
        {} if recipe.cue_mode == "none" else cue_class_for(spec, recipe, world.seed)
    )
    pngs = ensure_labels(recipe, spec, world.numbers)
    _compose(world, hull_mode="furniture")
    # The pass is chosen BEFORE the units so the cold line forms next to the work the human is
    # already doing (and so no unit is sited into the pass's own approach) -- but the ROOM, the
    # pass and every unit slot are now chosen TOGETHER, with backtracking, by ``plan_layout``.
    site_accepting_units(world)
    if recipe.cue_mode != "none":
        for u in recipe.units:
            png_key = placard_png_for(recipe, u.key, world.zones.get(u.key, ""))
            if png_key not in pngs:
                continue
            mount_placard(
                world,
                u.key,
                world.reg_pp[u.key],
                pngs[png_key],
            )
    supports = seat_reachable_items(world, world.supports) if recipe.n_supports else []
    if supports:
        register_supports(world, supports, pngs)
        _configure_workstations(world)
    seat_floor_items(world)
    seat_movable_support_items(world)
    seat_preloads(world)
    _assert_container_entities_registered(world)
    _publish_graspable_target_affordances(world)
    publish_native_nutrition_labels(world)
    mount_preload_decals(world, pngs)
    support_pp = supports[0][0] if supports else ""
    support_yaw = supports[0][1] if supports else 0.0
    mount_card(world, pngs)
    if recipe.cue_mode != "none":
        mount_staged_item_decals(world, pngs, support_yaw)
        mount_wall_labels(world, pngs)
        assert_decals(world)
    return _finish(world, support_pp)


def _initial_movable_target_affordances(
    recipe: Recipe, object_keys: set[str]
) -> list[tuple[str, str]]:
    return [
        (item.initial_place[3:], "on")
        for item in recipe.staged()
        if item.initial_place.startswith("on:")
        and item.initial_place[3:] in object_keys
    ]


def _publish_graspable_target_affordances(world: World) -> None:
    object_keys = set(((world.spec.get("world") or {}).get("objects") or {}))
    targets: list[tuple[str, str]] = [
        (preload.unit, "inside")
        for preload in world.recipe.preloads
        if preload.unit in object_keys
    ]
    for predicate, _subject, target in scored_relations(world.spec):
        if target in object_keys and predicate in {"inside", "on"}:
            targets.append((target, predicate))
    for aggregate in world.spec.get("aggregate_goals") or []:
        target = str(aggregate.get("target") or "")
        if aggregate.get("kind") == "exact_count_inside" and target in object_keys:
            targets.append((target, "inside"))
    for verb, args in _spec_steps(str(world.spec.get("human_program") or "")):
        if len(args) < 2 or args[1] not in object_keys:
            continue
        normalized = verb.upper()
        if normalized in {"PUTIN", "PUTINSIDE"}:
            targets.append((args[1], "inside"))
        elif normalized in {"PUT", "PLACE", "PUTON"}:
            targets.append((args[1], "on"))
    targets.extend(
        _initial_movable_target_affordances(world.recipe, object_keys)
    )
    for item in world.recipe.staged():
        for affordance in item.affordances:
            if affordance == "container":
                targets.append((item.key, "inside"))
            elif affordance == "surface":
                targets.append((item.key, "on"))
    world.scene._graspable_targets = list(dict.fromkeys(targets))


def mount_card(world: World, pngs: dict[str, str]) -> None:
    """Stand the row's free-standing rule board (the ledger card) on clear floor and register it.

    It reuses the ASSEMBLY row's proven manual body verbatim: a container row's card has exactly the
    same job -- a rule-level sign, off the units, that the human has to walk up to and read."""
    key = world.recipe.card_key
    if not key:
        return
    world.root_key = _cue_item(world).key
    site_manual(world, half=CARD_HALF_W, dcap=CARD_DCAP_M)
    mount_manual(world, [pngs[f"prop:{key}"]])


# The predicates a built container row may be scored on. ``on`` joined ``inside`` on 2026-08-17
# with ``scene_state._relations_on_live``; before that the physics branch could not emit it and a
# goal that used it scored 0.00 for ever. It is listed HERE as well as in
# ``spec_validate.SCORED_PREDS`` because the two answer different questions -- that one is "may a
# spec ask for it", this one is "does the built task carry it to the scorer" -- and for six weeks
# the first said yes while the second silently dropped every row.
SCORED_RELATION_PREDS: tuple[str, ...] = tuple(
    capability.predicate
    for capability in runtime_capabilities.PREDICATES
    if capability.runtime_scorable
)


def scored_inside(spec: dict) -> list[tuple[str, str]]:
    """The SCORED ``INSIDE`` edges, straight off the spec's goals -- the ``scored_edges`` twin."""
    return [
        (str(g["subject"]), str(g["target"]))
        for g in spec.get("goals") or []
        if g.get("pred", "inside") == "inside"
    ]


def scored_relations(spec: dict) -> list[tuple[str, str, str]]:
    """``(pred, subject, target)`` for every scored placement, INSIDE and ON alike.

    ``scored_inside``'s superset, kept beside it rather than replacing it because the pair-shaped
    list is what ``_gate_world``'s callers and the shipped ``behavior_task["goals"]`` field have
    always been; this adds the predicate without moving anything that already works."""
    rows: list[tuple[str, str, str]] = []
    for goal in spec.get("goals") or []:
        predicate = runtime_capabilities.canonical_predicate(
            goal.get("pred", "inside")
        )
        if predicate in SCORED_RELATION_PREDS:
            rows.append(
                (predicate, str(goal["subject"]), str(goal.get("target") or ""))
            )
    return rows


def _gate_world(world: World, goals: list[tuple[str, str]]) -> None:
    """Every geometry and ACCEPTANCE gate, in the order that makes each one meaningful.

    The ORDER is load-bearing: geometry first (disjointness, T0-EXACT), then the two acceptance
    probes against that exact t0 relation set, then the preload equality -- so no cap is ever
    computed from a preload the built world does not actually show. The two probes are siblings:
    ``assert_container_accepts`` proves the object can be PUT somewhere it reads back INSIDE, and
    ``assert_grasp_accepts`` proves the human can PICK IT UP in the first place. A row that fails
    either one renders perfectly and scores 0.00."""
    enable_runtime_geometry(world.scene)
    assert_disjoint(world)
    assert_not_presatisfied(world, goals)
    t0 = assert_t0_exact(world)
    _assert_aggregate_not_presatisfied(getattr(world, "spec", {}), t0)
    assert_aggregate_release_sites(world)
    # TWICE, and the two calls catch different things. Here, beside ``assert_t0_exact``, it asserts
    # the invariant the OWNER stated -- the person starts with nothing in their hands -- against the
    # world as built, before any probe has touched it.
    assert_hand_empty(world)
    assert_container_accepts(world, t0)
    assert_placard_rides(world)
    assert_grasp_accepts(world, t0)
    # And again LAST, because the probes above each run a REAL primitive: this is the call that
    # catches a probe which did not hand the symbolic state back.
    assert_hand_empty(world)
    assert_cue_facing(world)
    assert_cue_legible(world)
    assert_preload_matches(world, build_env(world))


def _assert_aggregate_not_presatisfied(spec: dict, relations: set) -> None:
    for index, aggregate in enumerate(spec.get("aggregate_goals") or []):
        if aggregate.get("kind") != "exact_count_inside":
            raise SpecBuildError(
                f"unsupported aggregate goal {aggregate.get('kind')!r} at index {index}"
            )
        candidates = {str(value) for value in aggregate.get("candidates") or []}
        target = str(aggregate.get("target") or "")
        expected = int(aggregate.get("count") or 0)
        actual = len(
            {
                str(subject)
                for pred, subject, relation_target in relations
                if str(getattr(pred, "value", pred)).casefold() == "inside"
                and str(subject) in candidates
                and str(relation_target) == target
            }
        )
        if actual == expected:
            raise SpecBuildError(
                f"aggregate goal {index} is already true at t0: {actual} == {expected}"
            )


def assert_aggregate_release_sites(world: World) -> None:
    """Require enough physical sites to realize each aggregate's successful state."""
    spec = getattr(world, "spec", {})
    targets = tuple(
        dict.fromkeys(
            str(aggregate.get("target") or "")
            for aggregate in spec.get("aggregate_goals") or []
            if aggregate.get("kind") == "exact_count_inside"
        )
    )
    for target in targets:
        aggregate_count = _minimum_joint_aggregate_count(spec, target)
        required = int(world.preload.get(target, 0)) + aggregate_count
        available = len(probe_sites(world, target))
        if available < required:
            raise SpecBuildError(
                f"target {target} needs {required} distinct release sites for preload + "
                f"exact count across its joint aggregate goals, but the built container "
                f"has {available}"
            )


def _publish_tints(world: World) -> None:
    """Hand the render-time painter (``behavior_import.apply_object_tints``) every authored colour.

    Published from the recipe rather than re-read from the spec so the assembly and container paths
    say it once each, and empty for every row that authors none -> ``apply_object_tints`` no-ops and
    those scenes are byte-identical."""
    tinted = {
        r.key: r.tint
        for r in (*world.recipe.units, *world.recipe.staged(), *world.recipe.parts)
        if getattr(r, "tint", "")
    }
    tinted.update(
        {
            key: preload.tint
            for preload in world.recipe.preloads
            if preload.tint
            for key in preload.keys
        }
    )
    if tinted:
        world.scene.obj_tints = tinted


def _publish_interaction_hints(world: World) -> None:
    """Validate runtime-authored grasp/camera hints and attach them to scene entities."""
    import interaction_hints
    from scene_state import Role

    runtime_world = dict((world.spec or {}).get("world") or {})
    for section in ("objects", "fixtures", "containers"):
        rows = runtime_world.get(section) or {}
        if not isinstance(rows, Mapping):
            raise SpecBuildError(f"world.{section} must be an object")
        for raw_key, raw_row in rows.items():
            key = str(raw_key)
            if not isinstance(raw_row, Mapping):
                continue
            authored = {
                name: raw_row[name]
                for name in ("grasp_hint", "look_region")
                if name in raw_row
            }
            if not authored:
                continue
            entity = world.scene.entities.get(key)
            if entity is None:
                raise SpecBuildError(
                    f"world.{section}.{key} carries interaction hints but was not registered"
                )
            try:
                if "grasp_hint" in authored:
                    if entity.role is not Role.GRASPABLE:
                        raise ValueError("grasp_hint requires a graspable entity")
                    entity.meta["grasp_hint"] = interaction_hints.normalize_grasp_hint(
                        authored["grasp_hint"]
                    )
                if "look_region" in authored:
                    entity.meta["look_region"] = interaction_hints.normalize_look_region(
                        authored["look_region"], entity_key=key
                    )
            except ValueError as exc:
                raise SpecBuildError(f"world.{section}.{key}: {exc}") from exc


def _finish(world: World, support_pp: str):
    """Resolve the negatives, pin the start, emit the program and publish the scored task."""
    _publish_tints(world)
    _publish_interaction_hints(world)
    world.scene.label_prim_paths = dict(world.decals)
    goals = scored_inside(world.spec)
    relations = scored_relations(world.spec)
    _gate_world(world, relations)
    negs = resolve_negatives(world)
    key = reference_answers(world.spec)
    answers = answer_override(key)
    program = build_program(
        str(world.spec["human_program"]),
        answers,
        key=key,
        recipe=world.recipe,
        spec=world.spec,
        seed=world.seed,
    )
    cue_item = _cue_item(world)
    cue_room = _initial_item_room(world.recipe, cue_item)
    if cue_room:
        world.room = cue_room
        world.box = room_box(world.recipe.scene, cue_room)
        world.anchor = room_anchor(world.recipe.scene, cue_room)
    cue_xy = _registered_entity(world, cue_item.key, phase="start cue").xy
    targets = [cue_xy] + [
        world.scene.entities[u.key].meta["front_stand"] for u in world.recipe.units
    ]
    explicit_start = start_override(world)
    if explicit_start is None:
        world.scene.start_xy, cue_dist = choose_start(world, cue_xy, targets)
    else:
        world.scene.start_xy = explicit_start
        cue_dist = math.hypot(
            explicit_start[0] - cue_xy[0], explicit_start[1] - cue_xy[1]
        )
        print(
            f"[bench] explicit demo start ({explicit_start[0]:.2f},{explicit_start[1]:.2f}) "
            f"cue_dist={cue_dist:.2f}m"
        )
    assert_look_goal(
        world,
        program,
        look_keys(
            str(world.spec["human_program"]), world.recipe, world.spec, key, world.seed
        ),
    )
    world.scene.behavior_task = {
        "scene": world.recipe.scene,
        "task": world.spec["task_id"],
        "allow_initial_support_object_stands": bool(
            (world.spec.get("world") or {}).get(
                "allow_initial_support_object_stands", False
            )
        ),
        "allow_legacy_close_navigation_preview": bool(
            (world.spec.get("world") or {}).get(
                "allow_legacy_close_navigation_preview", False
            )
        ),
        "allow_narrow_support_preflight_routes": bool(
            (world.spec.get("world") or {}).get(
                "allow_narrow_support_preflight_routes", False
            )
        ),
        "goals": goals,
        # The SAME edges with their predicate attached. ``goals`` stays pair-shaped and INSIDE-only
        # so every existing reader is byte-identical; ``episode_runtime`` prefers this when it is here.
        # Without it an ON goal is either dropped (it is not in ``goals``) or, worse, scored as an
        # INSIDE -- a wrong number rather than a missing one.
        "goal_preds": [
            {"pred": p, "subject": s, "target": t} for (p, s, t) in relations
        ],
        "aggregate_goals": list(world.spec.get("aggregate_goals") or []),
        "aggregate_target_loads": _aggregate_target_loads(world.spec),
        "neg_constraints": negs,
        "trace_outcomes": list(world.spec.get("trace_outcomes") or []),
        "grasp": list(
            dict.fromkeys(
                [i.key for i in world.recipe.items]
                + [key for preload in world.recipe.preloads for key in preload.keys]
            )
        ),
        "container": None,
        "caps": list(world.spec.get("caps_by_sub_goal") or []),
        "sub_goals": list(world.spec.get("sub_goals") or []),
        "answer_key": reference_answers(world.spec),
        "answered": dict(answers),
        "zones": dict(world.zones),
        "cue_class": dict(world.cue_class),
        "cue_number": {c: dict(v) for c, v in world.numbers.items()},
        "required_label_looks": list(world.recipe.observe),
        "episode_dialogue": dict(world.spec.get("episode_dialogue") or {}),
        "initially_open": [
            unit.key for unit in world.recipe.units if unit.initially_open
        ],
        "initially_open_room_doors": world.recipe.initially_open_room_doors,
        "initially_open_room_door_names": list(
            world.recipe.initially_open_room_door_names
        ),
        "initially_open_room_door_reverse_swing_names": list(
            world.recipe.initially_open_room_door_reverse_swing_names
        ),
        "preload": dict(world.preload),
        "preload_keys": {u: list(k) for u, k in world.preload_keys.items()},
        "seed": world.seed,
        "start_cue_dist_m": round(float(cue_dist), 3),
        "container_boxes": {
            k: [round(v, 4) for v in b] for k, b in world.boxes_of.items()
        },
        "cue_stands": {
            k: [round(v, 4) for v in t] for k, t in cue_stands(world).items()
        },
    }
    world.scene.configure_goal_predicates(world.scene.behavior_task["goal_preds"])
    light_and_flag(world, world.anchor)
    world.scene.behavior_task["light_rig"] = dict(world.light_rig)
    print(f"[bench] program: {program}")
    ax, ay = world.anchor
    return world.scene, {}, program, (ax, ay, 1.0), (ax - 3.0, ay - 3.0, 3.0)


def _initial_item_room(recipe: Recipe, item: ItemRecipe) -> str:
    by_key = {row.key: row for row in recipe.staged()}
    current = item
    seen = set()
    while current.key not in seen:
        seen.add(current.key)
        if current.initial_place.startswith("floor:"):
            return current.initial_place[6:] or recipe.room
        if current.initial_place.startswith("on:"):
            support = current.initial_place[3:]
            parent = by_key.get(support)
            if parent is not None:
                current = parent
                continue
            try:
                index = recipe.support_keys.index(support)
            except ValueError:
                return recipe.room
            if index < len(recipe.support_rooms):
                return recipe.support_rooms[index]
            return recipe.room
        break
    if current.initial_place.startswith("in:"):
        unit_key = current.initial_place[3:]
        unit = next((row for row in recipe.units if row.key == unit_key), None)
        if unit is not None and unit.room:
            return unit.room
    return recipe.room


def scored_edges(spec: dict) -> list[tuple[str, str]]:
    """The SCORED ``ATTACHED`` edges, straight off the spec's goals. Deliberately NOT read from the
    answers: the wrong-answer control has to change what the human does and nothing else."""
    return [
        (str(g["subject"]), str(g["target"]))
        for g in spec.get("goals") or []
        if g.get("pred") == "attached"
    ]


def resolved_initial_attachments(spec: dict, mates: list[Mate]) -> list[dict]:
    """Resolve canonical initial mate rows through the same metadata-derived mates.

    The TaskPack carries anonymized connector stems. The live reset contract must
    instead name the exact Male/Female metadata points and authored instance
    indices that ``produce_attach`` uses. A row outside the frozen assembly tree
    is rejected rather than guessed.
    """
    raw_rows = list((spec.get("world") or {}).get("initial_attachments") or [])
    resolved: list[dict] = []
    for index, raw in enumerate(raw_rows):
        child = str(raw.get("child") or "")
        parent = str(raw.get("parent") or "")
        connector = str(raw.get("connector") or "")
        matches = [
            mate
            for mate in mates
            if (mate.child, mate.parent, mate.anon) == (child, parent, connector)
        ]
        if len(matches) != 1:
            raise SpecBuildError(
                f"initial attachment {index} {child}-[{connector}]->{parent} "
                f"resolves to {len(matches)} metadata mate(s), need exactly 1"
            )
        mate = matches[0]
        resolved.append(
            {
                "child": mate.child,
                "parent": mate.parent,
                "connector": mate.anon,
                "child_point": mate.child_point,
                "parent_point": mate.parent_point,
                "child_point_idx": mate.cp_idx,
                "parent_point_idx": mate.pp_idx,
            }
        )
    return resolved


def _finish_assembly(world: World):
    """Resolve the negatives, pin the start, emit the ordered attach program, publish the task."""
    _publish_tints(world)
    _publish_interaction_hints(world)
    _publish_graspable_target_affordances(world)
    world.scene.label_prim_paths = dict(world.decals)
    negs = resolve_negatives(world)
    edges = scored_edges(world.spec)
    scored_mates = [
        (mate.child, mate.parent)
        for mate in world.mates
        if (mate.child, mate.parent) in set(edges)
    ]
    if scored_mates != edges:
        raise SpecBuildError(f"scored metadata mates {scored_mates} != goals {edges}")
    initial_attachments = resolved_initial_attachments(world.spec, world.mates)
    fixed_parts = {
        str(attachment[key])
        for attachment in initial_attachments
        for key in ("child", "parent")
    } | set(world.root_keys)
    assert_disjoint(world)
    assert_attached_clean_at_t0(world, world.mates, initial_attachments)
    answers = answer_override(reference_answers(world.spec))
    program = build_program(
        str(world.spec["human_program"]), answers, mates=world.mates
    )
    part_targets = [world.scene.entities[p.key].xy for p in world.recipe.parts]
    if world.recipe.manual_key:
        read_stand = world.scene.entities[world.recipe.manual_key].meta["front_stand"]
        targets = [read_stand, *part_targets]
        cue_xy = world.manual_xy
    else:
        targets = part_targets
        cue_xy = world.scene.entities[world.root_key].xy
    world.scene.start_xy, cue_dist = choose_start(world, cue_xy, targets)
    looks = authored_look_keys(str(world.spec["human_program"]))
    required_looks = looks + [
        key for key in [world.recipe.manual_key] if key and key not in looks
    ]
    assert_look_goal(world, program, required_looks)
    assert_flat_reach(world)
    assert_hand_empty(world)
    if world.recipe.cue_mode != "none":
        assert_cue_facing(world)
        assert_cue_legible(world)
    world.scene.behavior_task = {
        "scene": world.recipe.scene,
        "task": world.spec["task_id"],
        "allow_initial_support_object_stands": bool(
            (world.spec.get("world") or {}).get(
                "allow_initial_support_object_stands", False
            )
        ),
        "allow_legacy_close_navigation_preview": bool(
            (world.spec.get("world") or {}).get(
                "allow_legacy_close_navigation_preview", False
            )
        ),
        "allow_narrow_support_preflight_routes": bool(
            (world.spec.get("world") or {}).get(
                "allow_narrow_support_preflight_routes", False
            )
        ),
        "carcass": edges,  # step3's ATTACHED scorer branch
        "goal_preds": [
            {"pred": pred, "subject": subject, "target": target}
            for pred, subject, target in scored_relations(world.spec)
        ],
        "neg_constraints": negs,
        "trace_outcomes": list(world.spec.get("trace_outcomes") or []),
        "grasp": [
            part.key for part in world.recipe.parts if part.key not in fixed_parts
        ]
        + [item.key for item in world.recipe.staged()],
        "container": None,
        "caps": list(world.spec.get("caps_by_sub_goal") or []),
        "sub_goals": list(world.spec.get("sub_goals") or []),
        "answer_key": reference_answers(world.spec),
        "answered": dict(answers),
        "mates": [
            {
                "child": m.child,
                "parent": m.parent,
                "anon": m.anon,
                "child_point": m.child_point,
                "parent_point": m.parent_point,
                "pp_idx": m.pp_idx,
            }
            for m in world.mates
        ],
        "initial_attachments": initial_attachments,
        "letters": dict(world.letters),
        "root": world.root_key,
        "roots": list(world.root_keys),
        "required_label_looks": [
            part.key for part in world.recipe.parts if part.label
        ],
        "episode_dialogue": dict(world.spec.get("episode_dialogue") or {}),
        "initially_open_room_doors": world.recipe.initially_open_room_doors,
        "initially_open_room_door_names": list(
            world.recipe.initially_open_room_door_names
        ),
        "initially_open_room_door_reverse_swing_names": list(
            world.recipe.initially_open_room_door_reverse_swing_names
        ),
        "seed": world.seed,
        "start_cue_dist_m": round(float(cue_dist), 3),
        "cue_stands": {
            k: [round(v, 4) for v in t] for k, t in cue_stands(world).items()
        },
    }
    world.scene.configure_goal_predicates(world.scene.behavior_task["goal_preds"])
    world.scene.refresh_assembly_affordances()
    return _publish_assembly(world, program)


def _publish_assembly(world: World, program: str):
    """Light the kit, frame it with a fixed wide 3/4 camera and hand step3 its 5-tuple."""
    rx, ry = world.scene.entities[world.root_key].xy
    light_and_flag(world, (float(rx), float(ry)))
    world.scene.behavior_task["light_rig"] = dict(world.light_rig)
    print(f"[bench] program: {program}")
    return (
        world.scene,
        {},
        program,
        (float(rx), float(ry), 1.0),
        (float(rx) - 6.8, float(ry) - 6.8, 4.8),
    )


def assert_flat_reach(world: World) -> None:
    """RAISE-REACH, measured rather than assumed: with the kit FLAT every connector sits under a
    metre, and the root's own long axis says how high the top mate travels once ``[Raise]`` stands
    the unit up. The ordering the row scores is only real if the post-raise height is out of reach
    (~0.78 m), because there is no ``[Lower]`` token to undo a premature ``[Raise]``."""
    import behavior_import as bi

    fz = float(world.scene.floor_z)
    spans = {}
    for root_key in world.root_keys or (world.root_key,):
        root = world.part_of[root_key]
        bbox = bi.read_spec(f"{root.cat}/{root.model}").get("bbox_size") or [
            0.8,
            2.02,
            0.02,
        ]
        spans[root_key] = max(float(value) for value in bbox)
    flat_z = max(float(world.scene.entities[m.child].z) for m in world.mates) - fz
    if flat_z > 1.0:
        raise SpecBuildError(
            f"the flat kit already sits {flat_z:.2f} m up -- nothing is reachable-first"
        )
    print(
        f"[bench] RAISE-REACH: flat connectors <= {flat_z:.2f} m over the floor; root long "
        f"axes are {spans}, so a [Raise]-first order puts each top mate out of reach"
    )


def spec_scene_builders() -> dict:
    """Builders for every disk spec plus the single verified in-memory runtime spec."""
    ids = spec_ids()
    if _RUNTIME_SPEC is not None:
        task_id = str(_RUNTIME_SPEC.get("task_id") or "")
        if task_id and _registered_runtime_spec(task_id) is not None:
            ids.append(task_id)
    return {
        f"spec:{tid}": (lambda t: lambda st, ap: build_spec_scene(st, ap, t))(tid)
        for tid in sorted(set(ids))
    }


# =====================================================================================
# offline selftest (no Isaac): everything that does not need a stage
# =====================================================================================
def _selftest_prose(chk, spec: dict, recipe: Recipe) -> None:
    """The PROSE layer: the parse matches the author's own arithmetic, the observation phase is the
    program's prefix entire, the survey N pins the item count, and a wrong answer still moves."""
    hp = str(spec["human_program"])
    key = reference_answers(spec)
    pp = parse_prose(hp)
    chk(
        prose_arithmetic(pp, recipe, key) == pp.total,
        f"the parse reproduces the row's OWN count ({pp.total} grammar commands)",
    )
    toks = expand_prose(spec, recipe, key, key)
    obs = prose_observation(pp, recipe, key)
    chk(
        toks[2 * len(pp.prologue) :][: len(obs)] == obs,
        f"the {len(obs)}-command observation phase is the prefix",
    )
    looks = [t for t in toks if t.startswith("[LookAt]")]
    chk(
        len(looks) == len(prose_look_keys(pp, recipe, key)),
        f"every authored look is emitted ({len(looks)} of them)",
    )
    puts = [t for t in toks if t.startswith("[PutInside]")]
    chk(
        len(puts) == len(pp.loop_subjects(recipe, key)) + len(pp.prologue),
        f"one [PutInside] per queued subject ({len(puts)})",
    )
    chk(
        not any("<?>" in t for t in toks),
        "no unfilled answer slot survives the expansion",
    )
    other = _selftest_wrong_target(spec, recipe)
    wrong = expand_prose(spec, recipe, {k: other for k in key}, key)
    # The MAIN phase only. A prologue put is made with the assistant muted, to a unit the prose
    # names literally; re-routing it would put the control's diff inside the phase the row has
    # already declared unscored.
    main = wrong[2 * len(pp.prologue) :]
    chk(
        _routed_puts(main, key, other),
        f"a wrong answer re-routes every keyed [PutInside] onto {other}",
    )
    chk(
        wrong[: len(obs)] == toks[: len(obs)],
        "the wrong answer keeps the identical observation phase",
    )


def _verbs(program: str) -> list[str]:
    """The non-``[Walk]`` verb sequence of a compiled program."""
    return [
        t.split(" ", 1)[0]
        for t in program_tokens(program)
        if not t.startswith("[Walk]")
    ]


def _routed_puts(toks: list[str], key: dict[str, str], other: str) -> bool:
    """The ``[PutInside]`` that CARRIES each answer -- the LAST one per answer subject -- now
    targets ``other``.

    Scoped to the answer subjects on purpose: a program that also puts a non-roster prop away
    (``[PutInside] hamper_1(1) cupboard_1(1)``) is not re-routed by an ANSWER, and demanding it be
    would make the control a two-command diff instead of the one the row is about. Scoped to the
    LAST put per subject for the same reason: a recall row stages its item into a tray and only
    afterwards commits it to the answer unit, and that staging put is narrative, not answer -- an
    answer override leaves it alone, correctly."""
    last: dict[str, str] = {}
    for t in toks:
        if t.startswith("[PutInside]") and _first_arg(t) in key:
            last[_first_arg(t)] = t
    return bool(last) and all(f"<{other}>" in t for t in last.values())


def _sweep_extra(pp: ProseProgram, recipe: Recipe) -> int:
    """How many EXTRA commands the placard sweep costs over the author's count of 1."""
    return (len(recipe.units) - 1) if "unit_placards" in pp.named_looks else 0


def _selftest_order(chk, spec: dict, recipe: Recipe) -> None:
    """BENCH_ORDER, offline: the oracle fits its own budget and a wrong ORDER truncates."""
    import constraint_resolver as gr
    from isaac_goal import check_goal, inside, Pred
    from constraint_evaluator import eval_neg_constraints

    key = reference_answers(spec)
    pp = parse_prose(str(spec["human_program"]))
    oracle = expand_prose(spec, recipe, key, key)
    k = prose_budget(spec, pp, len(oracle))
    chk(k >= len(oracle), f"the ORACLE ({len(oracle)}) fits its own budget K={k}")
    staged = [i.key for i in staged_order(recipe, 0)]
    unit = sorted(set(key.values()))[0]
    os.environ["BENCH_ORDER"] = ",".join(f"{s}={unit}" for s in staged)
    try:
        bad_toks = apply_budget(expand_prose(spec, recipe, key, key), k)
    finally:
        del os.environ["BENCH_ORDER"]
    chk(
        len(bad_toks) < len(staged) * 2 + len(oracle) - 2 * len(key),
        f"the wrong ORDER is TRUNCATED to {len(bad_toks)} commands by K={k}",
    )
    done = [
        t.split("<")[1].split(">")[0] for t in bad_toks if t.startswith("[PutInside]")
    ]
    rel = {(Pred.INSIDE, o, unit) for o in done}
    chk(
        not check_goal([inside(o, c) for o, c in key.items()], rel),
        "wrong ORDER -> goal UNMET",
    )
    negs = gr.resolve_negs(
        list(spec.get("negs") or []), spec, offline_env(spec, recipe, 0)
    )
    fired = [r for r in eval_neg_constraints(negs, rel) if not r["ok"]]
    chk(
        bool(fired) and all(r["offenders"] for r in fired),
        f"wrong ORDER -> {[(r['kind'], r['offenders']) for r in fired]}",
    )


def _selftest_program(chk, spec: dict, recipe: Recipe) -> None:
    if is_prose(spec):
        _selftest_prose(chk, spec, recipe)
        if parse_prose(str(spec["human_program"])).budget:
            _selftest_order(chk, spec, recipe)
        return
    hp = str(spec["human_program"])
    key = reference_answers(spec)
    prog = build_program(hp, key, key=key, recipe=recipe)
    chk(bool(prog), "authored opening is preserved (build_program re-asserts it)")
    chk("<?>" not in prog, "no unfilled answer slot survives")
    n = len(program_tokens(prog))
    cmds = spec.get("cmds") or {}
    lo = int(cmds.get("grammar_cmds_from_program") or len(program_tokens(hp)))
    obs = len(insert_observation(program_tokens(hp), recipe)) - len(program_tokens(hp))
    hi = int(cmds.get("expanded_cmds") or n) + obs
    # The spec's expanded count includes the [Walk]s ``command_grammar`` auto-inserts at COMPILE
    # time; the generated string legitimately carries only the ones the answer slots forced, plus
    # any observation the row failed to author (``obs``, which the spec's count never budgeted for).
    chk(lo <= n <= hi, f"program is {n} commands (spec grammar={lo} expanded+obs={hi})")
    other = _selftest_wrong_target(spec, recipe)
    wrong = build_program(hp, {k: other for k in key}, key=key, recipe=recipe)
    chk(f"<{other}>" in wrong, f"a wrong answer re-routes the program onto {other}")
    chk(
        _routed_puts(program_tokens(wrong), key, other),
        f"every keyed [PutInside] in the wrong-answer program targets {other}",
    )
    chk(prog != wrong, "the wrong-answer program differs from the oracle program")
    # The control must change DESTINATIONS and nothing else. Walk counts legitimately differ (the
    # oracle visits several units, the control one), so they are excluded; ``build_program`` has
    # already re-asserted the authored opening, modulo the answer re-routing it exists to express.
    chk(
        _verbs(prog) == _verbs(wrong),
        "the control changes only destinations, not the actions",
    )
    looks = look_keys(hp, recipe)
    chk(bool(looks), f"the executed program looks at {looks}")
    chk(
        all(f"<{k}>" in prog for k in looks),
        "every authored [LookAt] subject survives into the program",
    )


def _selftest_wrong_target(spec: dict, recipe: Recipe) -> str:
    """The destination the wrong-answer control sends everything to.

    A unit the answer key does not already use (so the control really moves), and -- among those --
    the FIRST in declared order that makes a negative constraint fire. Choosing blind used to pick a
    unit whose cap happened to swallow the whole load, which reported "no negative fires" about the
    chooser rather than about the row. Deterministic: declared order, first hit."""
    key = reference_answers(spec)
    cands = [u.key for u in recipe.units if u.key not in set(key.values())]
    if not cands:
        return sorted(set(key.values()))[0]
    for u in cands:
        if _fires_on(spec, recipe, u):
            return u
    return cands[0]


def _fires_on(spec: dict, recipe: Recipe, unit: str) -> bool:
    """Does routing every scored subject into ``unit`` trip at least one negative?"""
    return _fires_with(spec, recipe, {str(g["subject"]): unit for g in spec["goals"]})


def _neg_world(spec: dict, recipe: Recipe) -> tuple[list, set]:
    """``(resolved negatives, the t=0 filler relations)`` -- resolved ONCE.

    ``_fires_with`` re-resolves the whole constraint set per call, which is affordable for two
    hard-coded candidate answers and not for a search over them."""
    import constraint_resolver as gr
    from isaac_goal import Pred

    env = offline_env(spec, recipe, 0)
    negs = gr.resolve_negs(list(spec.get("negs") or []), spec, env)
    base = {
        (Pred.INSIDE, k, u) for u, n in env.preload.items() for k in _filler_keys(u, n)
    }
    return negs, base


def _fires_in(negs: list, base: set, answers: dict[str, str]) -> list[dict]:
    """The constraints one destination assignment trips AND names an offender for."""
    from isaac_goal import Pred
    from constraint_evaluator import eval_neg_constraints

    world = {(Pred.INSIDE, s, u) for s, u in answers.items()} | base
    return [
        r for r in eval_neg_constraints(negs, world) if not r["ok"] and r["offenders"]
    ]


def _fires_with(spec: dict, recipe: Recipe, answers: dict[str, str]) -> bool:
    import constraint_resolver as gr
    from isaac_goal import Pred
    from constraint_evaluator import eval_neg_constraints

    try:
        negs, base = _neg_world(spec, recipe)
    except (SpecBuildError, gr.ResolveError, KeyError):
        return False
    bad = {(Pred.INSIDE, s, u) for s, u in answers.items()} | base
    return any(not r["ok"] for r in eval_neg_constraints(negs, bad))


def _reroute_candidates(key: dict[str, str], order: list[str]):
    """Every world within ONE and then TWO subject re-routes of the answer key, in declared order.

    A RELATIONAL negative is invisible to a uniform dump and to a uniform rotation:
    ``no_share_group`` fires only when two subjects of INCOMPATIBLE groups land in the SAME unit,
    and a per-subject ``not_inside`` only when ONE named subject lands in the ONE unit it is
    forbidden from. A uniform dump puts everything in a unit that names no group and forbids
    nothing, so it trips neither -- which used to be reported as 'this row has no negative that
    fires' when it is really 'the chooser looked in two places'. A minimal diff is the better
    control anyway: it isolates the row's own defect instead of confounding it with five others."""
    import itertools

    subs = sorted(key)
    for n in (1, 2):
        for picks in itertools.combinations(subs, n):
            for dests in itertools.product(order, repeat=n):
                cand = {**key, **dict(zip(picks, dests))}
                if cand != key:
                    yield cand


def wrong_answers(spec: dict, recipe: Recipe) -> dict[str, str]:
    """The wrong-answer control, as a per-subject ``BENCH_ANSWER`` map.

    Three shapes, in order, first one that trips a negative: send EVERYTHING to one unused unit,
    ROTATE each subject one unit along the declared order, then SEARCH the worlds one and two
    re-routes from the key. The rotation exists because a row whose only negatives are
    ``all_placed`` plus a ``capacity`` at least as large as its own load cannot be violated by a
    single-destination dump; the search exists because a RELATIONAL negative
    (``no_share_group``, per-subject ``not_inside``) cannot be violated by either uniform shape.
    Reporting 'no negative fires' from a chooser that only ever looked in two places is a fact
    about the chooser, not about the row. All three are expressible verbatim as ``BENCH_ANSWER``.

    Falls back to the flat dump when the search is exhausted, i.e. when the row has NO
    destination-only world that trips a negative and its discrimination is carried by the goal."""
    key = reference_answers(spec)
    flat = {s: _selftest_wrong_target(spec, recipe) for s in key}
    if _fires_with(spec, recipe, flat):
        return flat
    order = [u.key for u in recipe.units]
    rot = {
        s: order[(order.index(t) + 1) % len(order)] if t in order else t
        for s, t in key.items()
    }
    if rot != key and _fires_with(spec, recipe, rot):
        return rot
    return _searched_wrong_answers(spec, recipe, key, order) or flat


def _searched_wrong_answers(
    spec: dict, recipe: Recipe, key: dict[str, str], order: list[str]
) -> dict[str, str] | None:
    """The first one- or two-subject re-route of ``key`` that trips a negative AND names an
    offender, or ``None`` when no such world exists."""
    import constraint_resolver as gr

    try:
        negs, base = _neg_world(spec, recipe)
    except (SpecBuildError, gr.ResolveError, KeyError):
        return None
    for cand in _reroute_candidates(key, order):
        if _fires_in(negs, base, cand):
            return cand
    return None


def _selftest_resolver(chk, spec: dict, recipe: Recipe) -> None:
    import constraint_resolver as gr
    from spec_validator import REQUIRED_NEG_FIELDS

    env = offline_env(spec, recipe, 0)
    negs = gr.resolve_negs(list(spec.get("negs") or []), spec, env)
    chk(
        len(negs) == len(spec.get("negs") or []),
        f"resolver returned {len(negs)} constraint(s)",
    )
    empty = [
        (c.get("label"), f)
        for c in negs
        for f in REQUIRED_NEG_FIELDS.get(str(c.get("kind")), ())
        if not c.get(f)
    ]
    chk(
        not empty,
        f"no required negative field is EMPTY after resolution (offenders={empty})",
    )
    raw_empty = [
        c.get("label")
        for c in (spec.get("negs") or [])
        for f in REQUIRED_NEG_FIELDS.get(str(c.get("kind")), ())
        if not c.get(f)
    ]
    chk(
        bool(raw_empty),
        f"the SHIPPED spec really does ship {sorted(set(raw_empty))} empty",
    )


def _selftest_score(chk, spec: dict) -> None:
    """The scored contract, on synthetic relation sets: the reference answer must pass and a
    wrong routing must fail -- proved here so a red run cannot be blamed on the renderer."""
    import constraint_resolver as gr
    from isaac_goal import check_goal, inside, Pred
    from constraint_evaluator import eval_neg_constraints

    recipe = recipe_for(str(spec["task_id"]))
    env = offline_env(spec, recipe, 0)
    negs = gr.resolve_negs(list(spec.get("negs") or []), spec, env)
    goals = [(str(g["subject"]), str(g["target"])) for g in spec["goals"]]
    # The t=0 world the scorer sees: the answer key PLUS the filler stock already inside the units.
    # Leaving the fillers out is exactly the drift ``assert_preload_matches`` exists to stop, and it
    # would make every capacity verdict below pass for the wrong reason.
    base = {
        (Pred.INSIDE, k, u) for u, n in env.preload.items() for k in _filler_keys(u, n)
    }
    ok = {(Pred.INSIDE, o, c) for o, c in goals} | base
    chk(
        check_goal([inside(o, c) for o, c in goals], ok), "reference answer -> goal MET"
    )
    chk(
        all(r["ok"] for r in eval_neg_constraints(negs, ok)),
        "reference answer -> negatives OK",
    )
    wrong = wrong_answers(spec, recipe)
    bad = {
        (Pred.INSIDE, o, wrong.get(o, _selftest_wrong_target(spec, recipe)))
        for o, _c in goals
    } | base
    chk(
        not check_goal([inside(o, c) for o, c in goals], bad),
        "wrong answer -> goal UNMET",
    )
    fired = [r for r in eval_neg_constraints(negs, bad) if not r["ok"]]
    # A BUDGET row's own defect class is the ORDER, not the destination: with every subject sent to
    # one unit its caps still hold, and that is exactly why BENCH_ORDER exists. ``_selftest_order``
    # carries the "a negative FIRES and names its offender" burden for those rows instead.
    budgeted = is_prose(spec) and bool(parse_prose(str(spec["human_program"])).budget)
    named = _goal_offenders(goals, bad)
    chk(
        bool(fired) or budgeted or bool(named),
        "wrong answer -> a negative constraint FIRES, or -- no destination-only world trips one, "
        f"``wrong_answers`` having searched them -- the missed goal NAMES {named}",
    )
    chk(
        all(r["offenders"] for r in fired),
        f"every fired constraint NAMES its offender: {[(r['kind'], r['offenders']) for r in fired]}",
    )
    unplaced = _goal_offenders(goals, base)
    chk(
        any(not r["ok"] for r in eval_neg_constraints(negs, base)) or bool(unplaced),
        f"an unplaced item -> all_placed FIRES (or, on a row declaring none, NAMES {unplaced})",
    )
    _selftest_neg_kinds(chk, spec, negs, env, goals, base)


def _goal_offenders(goals: list[tuple[str, str]], world: set) -> list[str]:
    """The subjects of the sub-goals a world MISSES, read off the same per-sub-goal records the
    live scorer reports in its manifest.

    This is the WEAK form of "the control names its offender". It is admissible only where the
    STRONG form is unavailable -- where ``wrong_answers`` has searched the destination-only worlds
    and found none that trips a negative, which is exactly when ``fired`` comes back empty. Rows
    whose negatives CAN be tripped by a wrong destination still have to trip one. Nothing is lost
    on the per-kind side either: ``_selftest_neg_kinds`` proves every declared kind fires and names
    an offender in some world, independently of whether that world is a re-routed answer key."""
    from isaac_goal import goal_report, inside

    preds = [inside(o, c) for o, c in goals]
    return [str(r["subject"]) for r in goal_report(preds, world) if not r["ok"]]


def _filler_keys(unit: str, n: int) -> list[str]:
    return [f"pre_{unit}_{i}" for i in range(int(n))]


def _selftest_neg_kinds(chk, spec: dict, negs, env, goals, base) -> None:
    """Per-KIND discrimination, on synthetic relation sets: each declared kind must have a world
    that fires it and name the offender. A kind that only ever fires as collateral of a missed goal
    is not carrying any score."""
    from isaac_goal import Pred
    from constraint_evaluator import eval_neg_constraints

    kinds = {str(c.get("kind")) for c in negs}
    ok = {(Pred.INSIDE, o, c) for o, c in goals} | base
    if "capacity" in kinds:
        cap = next(c for c in negs if c.get("kind") == "capacity")
        unit = sorted(cap["cap"])[0]
        over = ok | {
            (Pred.INSIDE, f"extra_{i}", unit) for i in range(int(cap["cap"][unit]) + 1)
        }
        got = [r for r in eval_neg_constraints([cap], over)][0]
        chk(
            not got["ok"] and got["offenders"],
            f"capacity FIRES on an over-filled {unit}: {got['offenders']}",
        )
        chk(
            all(r["ok"] for r in eval_neg_constraints([cap], ok)),
            "capacity is satisfied by the answer key",
        )
    if "no_share_group" in kinds:
        nsg = next(c for c in negs if c.get("kind") == "no_share_group")
        g1, g2 = tuple(nsg["incompatible"][0])
        pair = [o for o, cls in nsg["group_of"].items() if cls == g1][:1]
        pair += [o for o, cls in nsg["group_of"].items() if cls == g2][:1]
        unit = str(nsg["units"][0])
        mixed = base | {(Pred.INSIDE, o, unit) for o in pair}
        got = eval_neg_constraints([nsg], mixed)[0]
        chk(
            not got["ok"] and got["offenders"] == [f"{unit}:{g1}+{g2}"],
            f"no_share_group CATCHES the incompatible pair {pair} in {unit}: {got['offenders']}",
        )
        chk(
            all(r["ok"] for r in eval_neg_constraints([nsg], ok)),
            "no_share_group is satisfied by the key",
        )
    if "all_placed" in kinds:
        ap = next(c for c in negs if c.get("kind") == "all_placed")
        got = eval_neg_constraints([ap], base)[0]
        chk(
            not got["ok"] and got["offenders"],
            f"all_placed FIRES on an empty world: {got['offenders']}",
        )
        chk(
            all(r["ok"] for r in eval_neg_constraints([ap], ok)),
            "all_placed is satisfied by the key",
        )
    for ni in [c for c in negs if c.get("kind") == "not_inside"]:
        # The forbidden pair, taken from the constraint's OWN resolved lists rather than from a
        # re-routed answer key: a ``not_inside`` usually names a DECOY the answer key never moves,
        # so no wrong answer can ever exercise it and only a direct world can.
        pair = (
            (str(ni["objs"][0]), str(ni["containers"][0]))
            if ni.get("objs") and ni.get("containers")
            else None
        )
        if pair is None:
            chk(
                False,
                f"not_inside {ni.get('label')!r} resolved to an EMPTY pair, so nothing can fire it",
            )
            continue
        got = eval_neg_constraints([ni], base | {(Pred.INSIDE, pair[0], pair[1])})[0]
        chk(
            not got["ok"] and got["offenders"] == [pair[0]],
            f"not_inside {ni.get('label')!r} FIRES on {pair[0]} in {pair[1]}: {got['offenders']}",
        )
        chk(
            all(r["ok"] for r in eval_neg_constraints([ni], ok)),
            f"not_inside {ni.get('label')!r} holds for the key",
        )
    del spec, env


def _selftest_assets(chk, spec: dict, recipe: Recipe) -> None:
    import behavior_import as bi

    for u in recipe.units:
        sp = bi.read_spec(f"{u.cat}/{u.model}")
        revs = [
            j
            for j in (sp.get("joints") or [])
            if j.get("type") == "revolute"
            and j.get("axis") in ("Z", "")
            and str(j.get("body0", "")).rstrip("/").endswith("base_link")
        ]
        want_door = _needs_door(spec)
        chk(
            bool(revs) or u.open_top,
            f"{u.key} ({u.cat}/{u.model}) has a drivable base_link door or is an open receptacle",
        )
        chk(
            not (want_door and u.open_top),
            f"{u.key}: the program brackets [Open]/[Close], so it may not be an open receptacle",
        )
        chk(os.path.exists(bi.baked_usd(f"{u.cat}/{u.model}")), f"{u.key} USD is baked")
    bodies = {(u.cat, u.model) for u in recipe.units}
    pair = {(u.cat, u.model) for u in recipe.units if u.key in recipe.permuted_units()}
    want = {
        "twin": len(bodies) == 1,
        # the class pair must be ONE body (only the placard tells them apart) and the rest distinct
        "pair_twin": len(pair) == 1
        and len(bodies) == len(recipe.units) - len(recipe.permuted_units()) + 1,
        "declared": True,
    }.get(recipe.unit_shape, len(bodies) == len(recipe.units))
    chk(
        want,
        f"unit_shape={recipe.unit_shape}: {len(bodies)} body/bodies for {len(recipe.units)} units",
    )
    for it in recipe.staged():
        chk(
            os.path.exists(bi.baked_usd(f"{it.cat}/{it.model}")),
            f"{it.key} USD is baked",
        )
    for pr in recipe.preloads:
        chk(
            os.path.exists(bi.baked_usd(f"{pr.cat}/{pr.model}")),
            f"preload for {pr.unit} is baked",
        )
    pngs = ensure_labels(recipe, spec, draw_numbers(recipe, 0))
    if recipe.cue_mode == "none":
        chk(not pngs, "cue_mode=none authors no synthetic label")
        return
    printed = _printed_block(spec)
    sparse = recipe.cue_mode == "printed" and bool(printed.get("sparse"))
    expected_units = (
        set((printed.get("units") or {}).keys())
        if sparse
        else {unit.key for unit in recipe.units}
    )
    expected_items = (
        set((printed.get("items") or {}).keys())
        if sparse
        else {item.key for item in recipe.staged()}
    )
    chk(
        len(pngs) >= len(expected_units) + len(expected_items),
        f"{len(pngs)} label PNG(s) exist on disk",
    )
    for u in recipe.units:
        if u.key not in expected_units:
            continue
        chk(
            placard_png_for(recipe, u.key, world_zone(spec, recipe, u.key, 0)) in pngs,
            f"{u.key} card exists",
        )
    for it in recipe.staged():
        if it.key not in expected_items:
            continue
        chk(
            decal_png_for(it, recipe, cue_class_for(spec, recipe, 0).get(it.key, ""))
            in pngs,
            f"{it.key} decal exists",
        )
    explicit_board = bool((_printed_block(spec).get("board_rows") or []))
    support_cards = [
        key
        for key in recipe.support_keys
        if not re.match(
            r"^(?:surf(?:ace)?|fixture)(?:[_-]?[a-z0-9]+)$", key, re.I
        )
    ]
    if explicit_board and recipe.support_keys:
        support_cards.append(recipe.support_keys[0])
    for k in list(dict.fromkeys(support_cards)) + (
        [recipe.card_key] if recipe.card_key else []
    ):
        chk(f"prop:{k}" in pngs, f"prop {k} card exists")


def _needs_door(spec: dict) -> bool:
    """The authored program brackets its put with [Open]/[Close] -> every destination must have a
    drivable leaf, or the oracle's own program cannot run."""
    return any(
        t.startswith(("[Open]", "[Close]"))
        for t in program_tokens(str(spec["human_program"]))
    )


def world_zone(spec: dict, recipe: Recipe, unit: str, seed: int) -> str:
    """What this unit's card prints at ``seed`` -- the same derivation ``build_container_scene`` uses."""
    if recipe.cue_mode == "intrinsic":
        return unit_class_for(spec, recipe).get(unit, "")
    return zone_permutation(recipe, seed).get(unit, "")


def _selftest_mates(chk, spec: dict, recipe: Recipe) -> None:
    """The ATTACHED layer, offline: the mates resolve off real metadata and the part-ID trap is real."""
    import behavior_import as bi

    points = attach_points_of(recipe)
    mates = derive_mates(spec, recipe, points)
    chk(
        [(m.child, m.parent) for m in mates] == scored_edges(spec),
        f"the {len(mates)} mates are the spec's SCORED edges, in the answer key's order",
    )
    assert_anon_connectors(mates, recipe)
    chk(
        True,
        "every anon connector names its real joint and leaks no model id (KEY-ANON)",
    )
    chk(
        assembly_root(mates, recipe) == "bd_0",
        "the assembly root is DERIVED as bd_0 (never a child)",
    )
    depth2 = [m for m in mates if m.parent != assembly_root(mates, recipe)]
    chk(
        len(depth2) == 1,
        f"exactly one depth-2 mate ({[(m.child, m.parent) for m in depth2]})",
    )
    trap = depth2[0]
    others = [
        p
        for p in recipe.parts
        if p.key != trap.parent and p.cat == recipe_part(recipe, trap.parent).cat
    ]
    chk(
        bool(others),
        "the depth-2 parent has a same-category twin (the part-ID trap is real)",
    )
    chk(
        all(trap.parent_point not in points[o.key] for o in others),
        f"only {trap.parent} exposes {trap.parent_point} -- its identical twin cannot take the baseboard",
    )
    for p in recipe.parts:
        chk(
            os.path.exists(bi.baked_usd(f"{p.cat}/{p.model}")),
            f"{p.key} ({p.cat}/{p.model}) USD is baked",
        )
    ls = [letter_permutation(recipe, s) for s in range(12)]
    chk(
        ls[0] == letter_permutation(recipe, 0),
        "the seeded letter permutation is deterministic",
    )
    chk(
        ls[0] != ls[1],
        "the letter-to-board binding MOVES between seeds (anti-memorisation)",
    )
    chk(
        {v for m in ls for v in m.values()} == set(recipe.classes),
        "over 12 seeds every letter is used while the ANSWER (which board has the socket) is fixed",
    )
    chk(bool(board_pngs(recipe, ls[0])), "every board letter decal PNG is on disk")


def recipe_part(recipe: Recipe, key: str) -> PartRecipe:
    return next(p for p in recipe.parts if p.key == key)


def _selftest_attach_program(chk, spec: dict, recipe: Recipe) -> None:
    """Program synthesis for the attach shape: verbatim opening, ordered mates, [Raise] last."""
    hp = str(spec["human_program"])
    mates = derive_mates(spec, recipe, attach_points_of(recipe))
    pre = opening_prefix(hp)
    chk(
        pre[:2]
        == [f"[Walk] <{recipe.manual_key}>(1)", f"[LookAt] <{recipe.manual_key}>(1)"]
        and not any(t.startswith(("[Grab]", "[Attach]", "[Raise]")) for t in pre),
        f"the authored opening is the pure OBSERVATION phase, verbatim: {pre}",
    )
    prog = build_program(hp, reference_answers(spec), mates=mates)
    toks = program_tokens(prog)
    n_want = int((spec.get("cmds") or {}).get("grammar_cmds_from_program") or len(toks))
    chk(
        len(toks) == n_want,
        f"program is {len(toks)} commands (spec grammar_cmds={n_want})",
    )
    chk(toks[-1].startswith("[Raise]"), "[Raise] is the LAST command (RAISE-REACH)")
    att = [t for t in toks if t.startswith("[Attach]")]
    chk(len(att) == len(mates), f"one arity-4 [Attach] per mate ({len(att)})")
    chk(
        all(
            f"<{m.child_point}>" in t and f"<{m.parent_point}>" in t
            for m, t in zip(mates, att)
        ),
        "each [Attach] carries the REAL metadata connector names, not the spec's anon ones",
    )
    chk(
        all(
            m.anon not in t for m, t in zip(mates, att) if m.anon != m.parent_point[:-1]
        ),
        "the anon connector strings never reach the executed program",
    )
    trap = next(m for m in mates if m.parent != assembly_root(mates, recipe))
    wrong = build_program(
        hp, {**reference_answers(spec), trap.child: "bd_0:shelfback"}, mates=mates
    )
    chk(wrong != prog, "a wrong-PARENT answer re-routes the program")
    chk(
        len(program_tokens(wrong)) == len(toks)
        and sum(a != b for a, b in zip(program_tokens(wrong), toks)) == 1,
        "the wrong answer changes EXACTLY one command -- everything else is identical",
    )
    chk(
        wrong.startswith(", ".join(pre)),
        "the wrong-answer program keeps the same authored opening",
    )


def _selftest_attach_score(chk, spec: dict) -> None:
    """The scored contract on synthetic ATTACHED sets: the oracle passes, a wrong PARENT does not."""
    from isaac_goal import attached, check_goal, goal_progress, Pred

    edges = scored_edges(spec)
    goal = [attached(c, p) for c, p in edges]
    ok = {(Pred.ATTACHED, c, p) for c, p in edges}
    chk(
        check_goal(goal, ok) and goal_progress(goal, ok) == 1.0,
        "reference program -> goal MET 1.00",
    )
    trap = next((c, p) for c, p in edges if p != "bd_0")
    bad = {r for r in ok if r[1] != trap[0]} | {(Pred.ATTACHED, trap[0], "bd_0")}
    chk(
        not check_goal(goal, bad),
        f"{trap[0]} mated to the WRONG PARENT bd_0 -> goal UNMET",
    )
    chk(
        abs(goal_progress(goal, bad) - (len(edges) - 1) / len(edges)) < 1e-9,
        f"the wrong parent costs exactly one of {len(edges)} mates ({goal_progress(goal, bad):.2f})",
    )
    chk(
        not check_goal(goal, ok - {(Pred.ATTACHED, trap[0], trap[1])}),
        "a skipped mate -> goal UNMET",
    )


def _selftest_container(chk, spec: dict, recipe: Recipe) -> None:
    _selftest_roster(chk, spec, recipe)
    _selftest_permute(chk, spec, recipe)
    _selftest_program(chk, spec, recipe)
    _selftest_resolver(chk, spec, recipe)
    _selftest_score(chk, spec)
    _selftest_assets(chk, spec, recipe)
    _selftest_grasp(chk, spec, recipe)
    _selftest_room(chk, spec, recipe)
    if recipe.preloads or recipe.numbers:
        _selftest_prestage(chk, spec, recipe)


def _selftest_room(chk, spec: dict, recipe: Recipe) -> None:
    """Does the scene have ROOM, not just PARTS -- the question the offline gate never asked.

    ``predict_siting`` runs the live solver's own predicates over geometry rebuilt from the scene
    JSON and the bake cache. It is allowed to answer UNKNOWN (it caps the room x body pairs it
    walks), and UNKNOWN passes: this gate exists to catch a scene that provably cannot hold the
    row, not to fail one the search merely ran out of time on."""
    sited, why = predict_siting(recipe, 0)
    chk(
        sited is not False,
        f"{spec['task_id']}: some room in {recipe.scene} sites all {len(recipe.units)} unit(s) "
        f"at their real footprints -- {why}",
    )
    staged, why_stage = predict_staging(recipe, 0)
    chk(staged, f"{spec['task_id']}: the row stages a graspable -- {why_stage}")
    room = (
        _initial_item_room(recipe, recipe.items[0])
        if _uses_declared_rooms(recipe) and recipe.items
        else (why.split(" @ ")[0] if sited else "")
    )
    start, why_start = predict_start_pose(recipe, room)
    chk(
        start,
        f"{spec['task_id']}: the sited room can host a legal start -- {why_start}",
    )


def _selftest_grasp(chk, spec: dict, recipe: Recipe) -> None:
    """The BODY half of grasp acceptance, offline: no staged graspable may be played by a body the
    real ``produce_pick`` has been MEASURED to fail on.

    The offline acceptance probe does the measuring through the same executor entry point and live
    per-shape SDF wrap, at a pose inside the measured reach envelope -- and this consults the
    result exactly as ``_selftest_assets`` consults the baked metadata. It is deliberately the WEAK
    half: a body that grips fine on a bench is still ungraspable on a window sill 2.62 m up, which
    is the failure that actually shipped, and only ``assert_grasp_accepts`` can see it."""
    bad = sorted(
        f"{it.key} ({it.cat}/{it.model})"
        for it in recipe.staged()
        if not _is_observation_only_exact_object(
            recipe.task_id, it.key, it.cat, it.model
        )
        if (it.cat, it.model) in _GRASP_SILENT
    )
    chk(
        not bad,
        f"{spec['task_id']}: every staged graspable has a measured-CLEAN grasp body"
        + (f" -- SILENT: {bad}" if bad else ""),
    )


def _selftest_roster(chk, spec: dict, recipe: Recipe) -> None:
    """Every roster key is BOUND to a body, and nothing outside the roster is SCORED.

    Two directions, not one equality, because a DECOY is legitimately outside the roster (T6-HSF-02
    stages eight spares that make ``not_inside{items minus key_subjects}`` a constraint about the
    picture) while a roster object the answer key never names is legitimately INSIDE it (a third of
    the T4 rows ship those). Both must still be bound, or the row renders a key with no body."""
    roster = set(spec["roster"]["objects"]) | set(spec["roster"]["containers"])
    bound = recipe.keys() | {d.key for d in recipe.decoys}
    chk(
        not (roster - bound),
        f"every roster key is bound ({sorted(roster - bound)} unbound)",
    )
    chk(
        not (recipe.keys() - roster),
        f"the recipe scores nothing outside the roster ({sorted(recipe.keys() - roster)})",
    )


def _selftest_permute(chk, spec: dict, recipe: Recipe) -> None:
    """BG-PERMUTE: SOMETHING about the appearance has to move with the seed while the answer key
    stays put, or a model that memorised one seed's world answers every seed of the row."""
    if recipe.cue_mode == "target":
        z0 = zone_permutation(recipe, 0)
        chk(
            z0 == zone_permutation(recipe, 0),
            "the seeded zone permutation is deterministic",
        )
        chk(
            len({tuple(sorted(zone_permutation(recipe, s).items())) for s in range(12)})
            > 1,
            "the class-to-unit binding MOVES over 12 seeds (anti-memorisation)",
        )
        # 12 seeds is the row's declared power budget, and over it the cue must exercise the whole
        # class vocabulary its own permutation group can reach, while the answer key never moves.
        # ``movable`` is read at the permuted units' OWN indices: ``classes`` is index-aligned with
        # ``units``, so zipping it against ``permuted_units()`` only agrees when the permuted pair
        # happens to be the first units declared, which is true of the pins and not in general.
        idx = [
            i for i, u in enumerate(recipe.units) if u.key in recipe.permuted_units()
        ]
        movable = {recipe.classes[i] for i in idx if i < len(recipe.classes)}
        reach = {
            zone_permutation(recipe, s)[u]
            for s in range(12)
            for u in recipe.permuted_units()
        }
        first = _moving_cue_item(spec, recipe)
        chk(
            first is not None and movable <= reach,
            f"over 12 seeds the cue on {first} moves and the pair sees every class it can "
            f"({sorted(reach)}, movable {sorted(movable)})",
        )
        return
    moved = [
        s
        for s in range(12)
        if unit_assets(recipe, s) != unit_assets(recipe, 0)
        or draw_numbers(recipe, s) != draw_numbers(recipe, 0)
    ]
    chk(
        bool(moved),
        f"the appearance MOVES over 12 seeds (differs from seed 0 on {len(moved)})",
    )
    chk(
        unit_assets(recipe, 3) == unit_assets(recipe, 3),
        "the seeded asset permutation is deterministic",
    )
    chk(
        draw_numbers(recipe, 3) == draw_numbers(recipe, 3),
        "the seeded number draw is deterministic",
    )


def _moving_cue_item(spec: dict, recipe: Recipe) -> str | None:
    """A SCORED graspable whose printed cue really moves over the row's 12-seed power budget.

    BG-PERMUTE's claim is about the row, not about ``items[0]``: on a row whose shuffle is
    restricted to a ``no_share_group`` pair, an item keyed to a unit outside that pair legitimately
    keeps one class forever, and it is the items keyed INTO the pair that carry the anti-memorisation
    burden. Returning None means NO item's cue moves, which is the real defect."""
    for it in recipe.items:
        if len({cue_class_for(spec, recipe, s)[it.key] for s in range(12)}) > 1:
            return it.key
    return None


def _selftest_prestage(chk, spec: dict, recipe: Recipe) -> None:
    """The PRESTAGE + cue_number contract, offline: the draw is seeded, the printed value IS the
    resolved value, and a preload that the cap depends on is actually declared."""
    import constraint_resolver as gr

    env = offline_env(spec, recipe, 0)
    counts = preload_counts(recipe, 0, draw_numbers(recipe, 0))
    chk(
        all(int(env.preload[u]) == int(n) for u, n in counts.items()),
        f"BuildEnv.preload carries the drawn counts {counts}",
    )
    chk(
        set(env.preload) == {u.key for u in recipe.units},
        "every unit appears in BuildEnv.preload (an empty unit must resolve to 0, not raise)",
    )
    negs = gr.resolve_negs(list(spec.get("negs") or []), spec, env)
    caps = [c for c in negs if c.get("kind") == "capacity"]
    for c in caps:
        for u, v in c["cap"].items():
            floor = int(env.preload.get(u, 0))
            chk(
                int(v) >= floor,
                f"cap[{u}]={v} is at least its own preload {floor} (t=0 is legal)",
            )
    pngs = ensure_labels(recipe, spec, draw_numbers(recipe, 0))
    for nr in recipe.numbers:
        for k, v in draw_numbers(recipe, 0)[nr.channel].items():
            chk(
                str(v) in os.path.basename(pngs[f"{nr.channel}:{k}"]),
                f"the {nr.channel} card for {k} carries the DRAWN value {v} in its file name",
            )
        if nr.domain == "units":
            chk(
                all(
                    int(env.cue_number[nr.channel][k])
                    == int(draw_numbers(recipe, 0)[nr.channel][k])
                    for k in nr.keys
                ),
                f"BuildEnv.cue_number[{nr.channel!r}] is the same draw the card prints",
            )


def _selftest_assembly(chk, spec: dict, recipe: Recipe) -> None:
    _selftest_roster(chk, spec, recipe)
    chk(
        not (spec.get("negs") or []),
        "this row declares no negative constraint (resolver -> [])",
    )
    _selftest_mates(chk, spec, recipe)
    _selftest_attach_program(chk, spec, recipe)
    _selftest_attach_score(chk, spec)


def _selftest(ids: list[str] | None = None) -> int:
    """Gate the AUTHORED pins (the shipped, comparable count) or, with ``--all``, every spec.

    The default stays the pin set so the count is comparable slice to slice; ``--all`` runs the
    derived layer too for a complete scale check."""
    ok = True
    n_pass = 0
    n_fail = 0

    def chk(cond, msg):
        nonlocal ok, n_pass, n_fail
        ok = ok and bool(cond)
        n_pass, n_fail = n_pass + bool(cond), n_fail + (not cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")

    for tid in ids if ids is not None else sorted(recipes()):
        print(f"[task_world_builder] {tid}")
        spec = load_spec(tid)
        recipe = recipe_for(tid)
        chk(
            spec["scene"] == recipe.scene,
            f"recipe scene matches the spec ({recipe.scene})",
        )
        if recipe.is_assembly():
            _selftest_assembly(chk, spec, recipe)
        else:
            _selftest_container(chk, spec, recipe)
    builders = spec_scene_builders()
    chk(
        all(f"spec:{t}" in builders for t in recipes()),
        f"the scene registry exposes {sorted('spec:' + t for t in recipes())}",
    )
    _selftest_derived(chk)
    print(
        f"[task_world_builder] selftest {'PASS' if ok else 'FAIL'} -- {n_pass} PASS / {n_fail} FAIL "
        f"over {len(ids if ids is not None else recipes())} row(s) ({len(spec_ids())} specs on disk)"
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(_selftest(spec_ids() if "--all" in sys.argv else None))
