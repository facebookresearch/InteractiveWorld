#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""scene_furniture_units.py -- use the furniture a BEHAVIOR scene ALREADY OWNS as task destinations.

THE PROBLEM THIS SOLVES
-----------------------
Every task built so far IMPORTS its destinations: a room with five bookcases still gets four fresh
lockers trucked in, all the same model, sited by a 20-round acceptance search that is also the single
biggest source of build failures. Run d4: 12 of 14 failures were "cannot site N units".

WHY IT LOOKED HARD, AND WHY IT IS NOT
-------------------------------------
``mount_placard`` (task_world_builder.py:4014) needs two things an imported unit gets for free and a
scene-owned prim appeared not to have:

  * ``world.front_dir[key]``  -- which way the readable face points
  * ``ent.leaf_paths[0]``     -- WHICH CHILD LINK IS THE SWINGING DOOR, so ``_leaf_frame`` (:4057)
    parents the placard UNDER the leaf and the placard RIDES the door instead of being left hanging
    in the opening once it swings. That is a hard requirement, not a nicety.

Both already exist. ``behavior_import._openable_leaf_fields(scene, pp, o)`` derives them from the
BAKED JOINT SPEC (revolute-Z -> swing door + hinge; prismatic -> drawer + travel) and cares only that
``o`` carries ``cat``/``model`` -- not who put the prim on the stage. Measured on Merom_1_int: 14
scene-owned openable instances, and ``bottom_cabinet/dajebq`` reports 4 revolute-Z joints with
``body1=/dajebq/link_0``. So this module is WIRING, not derivation: it finds the scene's own
containers, hands each to the SAME registration path an imported unit takes, and refuses anything it
cannot prove.

WHAT IT REFUSES, AND WHY EACH REFUSAL IS LOAD-BEARING
------------------------------------------------------
* a door-required binding with no baked swing joint -> a placard would not ride a door. Open
  receptacles are valid through the builder's separate leaf-free registration path.
* the acceptance probe says a released object does not land inside -> the goal can never score.
* the prim is not reachable from any stand pose -> the person can never put anything in it.
None of these is guessed: each is measured on the live stage, because a proxy veto that destroys a
buildable world is the most expensive mistake this pipeline can make.

    python3 scene_furniture_units.py --scan Merom_1_int          # offline: what the scene offers
    bash hand/run_isaac.sh simulator.physical_world/scene_furniture_units.py --check Merom_1_int
                                                                 # on a stage: placard rides the door
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

_HERE: str = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_BEHAVIOR_DATA_ROOT = os.environ.get(
    "INTERACTIVEWORLD_BEHAVIOR_DATA", "/opt/interactiveworld/behavior-data"
)
_BEHAVIOR_CACHE_ROOT = os.environ.get(
    "INTERACTIVEWORLD_BEHAVIOR_CACHE", "/opt/interactiveworld/behavior-cache"
)
SCENES_ROOT: str = os.path.join(_BEHAVIOR_DATA_ROOT, "behavior-1k-assets", "scenes")
SPEC_ROOT: str = os.path.join(_BEHAVIOR_CACHE_ROOT, "spec")
USD_ROOT: str = os.path.join(_BEHAVIOR_CACHE_ROOT, "usd")
SCENE_BINDING_CONTRACT_VERSION: str = "scene-container-binding/1.1"
SCENE_UNIT_MIN_SEP_M: float = 0.60


# --------------------------------------------------------------------------- offline: what exists
def scene_instances(scene: str) -> list[dict]:
    """Every placed object in a scene as ``{name, cat, model}``, from the scene's own JSON."""
    init = _scene_init(scene)
    out = []
    for name, rec in init.items():
        args = (rec or {}).get("args") or {}
        cat, model = args.get("category"), args.get("model")
        if cat and model:
            out.append({"name": str(name), "cat": str(cat), "model": str(model)})
    return out


def _scene_init(scene: str) -> dict:
    doc = _scene_document(scene)
    init = (doc.get("objects_info") or {}).get("init_info") or {}
    return init if isinstance(init, dict) else {}


def _scene_document(scene: str) -> dict:
    path = os.path.join(SCENES_ROOT, scene, "json", f"{scene}_best.json")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc if isinstance(doc, dict) else {}


def _sha256_file(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return f"sha256:{digest.hexdigest()}"


def _asset_identity(cat: str, model: str) -> tuple[str, str] | None:
    asset_ref = f"{cat}/{model}.usd"
    digest = _sha256_file(os.path.join(USD_ROOT, asset_ref))
    return (asset_ref, digest) if digest is not None else None


def _scene_pose(state: dict, name: str) -> tuple[list[float], list[float]] | None:
    registry = (state.get("registry") or {}).get("object_registry") or {}
    root = (registry.get(name) or {}).get("root_link") or {}
    pos, ori = root.get("pos"), root.get("ori")
    if (
        not isinstance(pos, list)
        or len(pos) != 3
        or not isinstance(ori, list)
        or len(ori) != 4
    ):
        return None
    values = [*pos, *ori]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        return None
    return ([float(value) for value in pos], [float(value) for value in ori])


_INSTANCE_DIGEST_FIELDS: tuple[str, ...] = (
    "name",
    "cat",
    "model",
    "rooms",
    "room",
    "stage_prim_name",
    "stage_prim_path",
    "asset_ref",
    "asset_sha256",
    "source_expected_file_hash",
    "position",
    "orientation",
    "scale",
    "registration_modes",
)


def scene_instance_digest(row: dict) -> str:
    payload = {
        "version": SCENE_BINDING_CONTRACT_VERSION,
        **{field: row.get(field) for field in _INSTANCE_DIGEST_FIELDS},
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _xy_distance(left: dict, right: dict) -> float:
    return math.hypot(
        float(left["position"][0]) - float(right["position"][0]),
        float(left["position"][1]) - float(right["position"][1]),
    )


def _compatible_category_rows(rows: list[dict]) -> list[dict]:
    """Deterministic fail-closed subset accepted by the runtime separation gate."""
    selected: list[dict] = []
    for row in sorted(rows, key=lambda item: str(item["stage_prim_path"])):
        if all(_xy_distance(row, prior) >= SCENE_UNIT_MIN_SEP_M for prior in selected):
            selected.append(row)
    return selected


def bindable_instances_from_init(
    init: dict,
    container_models: dict[
        str, tuple[str, ...] | list[str] | set[str] | frozenset[str]
    ],
    door_models: dict[str, tuple[str, ...] | list[str] | set[str] | frozenset[str]]
    | None = None,
    *,
    state: dict | None = None,
    asset_identity=None,
) -> list[dict]:
    """Exact scene instances with a provable, runtime-consumable prim identity."""
    allowed = {
        str(cat): frozenset(str(model) for model in models)
        for cat, models in container_models.items()
    }
    doors = {
        str(cat): frozenset(str(model) for model in models)
        for cat, models in (door_models or {}).items()
    }
    rows: list[dict] = []
    state = state if isinstance(state, dict) else {}
    identity_for = asset_identity or _asset_identity
    for name, rec in sorted(init.items(), key=lambda row: str(row[0])):
        args = (rec or {}).get("args") or {}
        cat = str(args.get("category") or "")
        model = str(args.get("model") or "")
        if not cat or model not in allowed.get(cat, frozenset()):
            continue
        raw_rooms = args.get("in_rooms") or args.get("rooms") or []
        if isinstance(raw_rooms, str):
            raw_rooms = [raw_rooms]
        rooms = [str(room) for room in raw_rooms if str(room)]
        if not rooms:
            continue
        pose = _scene_pose(state, str(name))
        expected_file_hash = str(args.get("expected_file_hash") or "")
        asset = identity_for(cat, model)
        scale = args.get("scale")
        if (
            pose is None
            or not expected_file_hash
            or asset is None
            or not isinstance(scale, list)
            or len(scale) != 3
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in scale
            )
        ):
            continue
        asset_ref, asset_sha256 = asset
        modes = ["open"]
        if model in doors.get(cat, frozenset()):
            modes.append("swing")
        row = {
            "name": str(name),
            "cat": cat,
            "model": model,
            "rooms": rooms,
            "room": rooms[0],
            "stage_prim_name": f"b_{name}",
            "stage_prim_path": f"/World/b_{name}",
            "asset_ref": str(asset_ref),
            "asset_sha256": str(asset_sha256),
            "source_expected_file_hash": expected_file_hash,
            "position": pose[0],
            "orientation": pose[1],
            "scale": [float(value) for value in scale],
            "registration_modes": modes,
        }
        row["instance_digest"] = scene_instance_digest(row)
        rows.append(row)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["room"]), str(row["cat"])), []).append(row)
    rows = [
        row
        for key in sorted(grouped)
        for row in _compatible_category_rows(grouped[key])
    ]
    by_room: dict[str, list[dict]] = {}
    for row in rows:
        by_room.setdefault(str(row["room"]), []).append(row)
    for room_rows in by_room.values():
        for row in room_rows:
            row["conflicts"] = sorted(
                str(other["instance_digest"])
                for other in room_rows
                if other is not row and _xy_distance(row, other) < SCENE_UNIT_MIN_SEP_M
            )
    rows.sort(key=lambda row: (row["room"], row["cat"], row["stage_prim_name"]))
    ranks: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["room"]), str(row["cat"]))
        row["room_rank"] = ranks.get(key, 0)
        ranks[key] = int(row["room_rank"]) + 1
    return rows


def _builder_model_contract() -> tuple[
    dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]
]:
    try:
        import task_world_builder as bb  # noqa: PLC0415

        categories = tuple(bb.asset_index())
        return (
            {cat: tuple(bb.unit_models(cat, False)) for cat in categories},
            {cat: tuple(bb.unit_models(cat, True)) for cat in categories},
        )
    except Exception:  # noqa: BLE001 - offline readers use the committed contract mirror
        pass
    idx = os.path.join(os.path.dirname(_HERE), "taskpack_support", "asset_index.json")
    try:
        with open(idx, encoding="utf-8") as fh:
            categories = (json.load(fh).get("categories") or {}).items()
    except (OSError, ValueError):
        return {}, {}
    open_models: dict[str, tuple[str, ...]] = {}
    door_models: dict[str, tuple[str, ...]] = {}
    for cat, row in categories:
        builder = (row or {}).get("builder") or {}
        open_models[str(cat)] = tuple(
            str(model) for model in builder.get("container_models") or []
        )
        door_models[str(cat)] = tuple(
            str(model) for model in builder.get("door_container_models") or []
        )
    return open_models, door_models


def joint_facts(cat: str, model: str) -> dict:
    """Summarize the baked joints and the builder's supported openable kind.

    ``kind`` mirrors what ``_openable_leaf_fields`` will decide, so a scan can predict the outcome
    without a stage: a supported revolute door wins over prismatic exactly as it does there."""
    path = os.path.join(SPEC_ROOT, cat, f"{model}.json")
    try:
        with open(path, encoding="utf-8") as fh:
            spec = json.load(fh)
            joints = spec.get("joints") or []
    except (OSError, ValueError):
        return {
            "n_joints": 0,
            "n_revolute_z": 0,
            "n_prismatic": 0,
            "kind": "no_spec",
        }
    rev = [
        j
        for j in joints
        if j.get("type") == "revolute"
        and j.get("axis") == "Z"
        and str(j.get("body0", "")).rstrip("/").endswith("base_link")
    ]
    declared = {
        str(row[-1])
        for row in (spec.get("openable_joint_ids") or ())
        if isinstance(row, (list, tuple)) and row
    }
    front_flip = [
        joint
        for joint in joints
        if cat == "oven"
        and joint.get("type") == "revolute"
        and joint.get("axis") in ("X", "Y")
        and str(joint.get("name") or "") in declared
        and str(joint.get("body0", "")).rstrip("/").endswith("base_link")
    ]
    pri = [j for j in joints if j.get("type") == "prismatic"]
    kind = "swing" if rev or front_flip else ("slide" if pri else "none")
    return {
        "n_joints": len(joints),
        "n_revolute_z": len(rev),
        "n_prismatic": len(pri),
        "kind": kind,
    }


def _receptacle_vocab() -> frozenset[str]:
    """The categories the builder will bind as a CONTAINER, read live from task_world_builder.

    A joint is not a container. The first scan of Merom_1_int returned 39 "candidates" of which 11
    were doors, 10 were openable windows and 4 were table lamps -- all articulated, none able to hold
    a released object. Gating on the builder's own receptacle vocabulary rather than on a hand-list
    means this can never disagree with what the builder will actually accept."""
    try:
        import task_world_builder as bb  # noqa: PLC0415

        return frozenset(bb.receptacle_vocab())
    except Exception:  # noqa: BLE001 - offline scan on a box without Isaac
        pass
    # The offline mirror: the index records the same capability the builder computed.
    idx = os.path.join(os.path.dirname(_HERE), "taskpack_support", "asset_index.json")
    try:
        with open(idx, encoding="utf-8") as fh:
            cats = (json.load(fh).get("categories") or {}).items()
        return frozenset(
            c for c, row in cats if (row.get("builder") or {}).get("container_models")
        )
    except (OSError, ValueError):
        return frozenset()


def instance_rooms(scene: str) -> dict[str, list[str]]:
    """``{instance name: [room ids]}`` from the scene JSON.

    The room is the field that decides whether a set of the scene's own cabinets can be ONE task.
    Merom_1_int's 12 candidates sit in three different rooms (childs_room_0, living_room_0,
    kitchen_0); taking the first three handed a task destinations the person cannot visit in one
    episode. Ignoring this is why the first scene-native build failed."""
    try:
        init = _scene_init(scene)
    except (OSError, ValueError):
        return {}
    out: dict[str, list[str]] = {}
    for name, rec in init.items():
        args = (rec or {}).get("args") or {}
        rooms = args.get("in_rooms") or args.get("rooms") or []
        if isinstance(rooms, str):
            rooms = [rooms]
        out[str(name)] = [str(r) for r in rooms]
    return out


def candidates(scene: str) -> list[dict]:
    """Scene-owned instances accepted by the same model contract as ``site_units``."""
    open_models, door_models = _builder_model_contract()
    doc = _scene_document(scene)
    rows = bindable_instances_from_init(
        (doc.get("objects_info") or {}).get("init_info") or {},
        open_models,
        door_models,
        state=doc.get("state") or {},
    )
    for row in rows:
        facts = joint_facts(str(row["cat"]), str(row["model"]))
        row.update(facts)
        if facts["kind"] not in ("swing", "slide"):
            row["kind"] = "open"
    return rows


def scan(scene: str) -> dict:
    inst = scene_instances(scene)
    cand = candidates(scene)
    by_cat: dict[str, int] = {}
    for c in cand:
        by_cat[c["cat"]] = by_cat.get(c["cat"], 0) + 1
    return {
        "scene": scene,
        "n_instances": len(inst),
        "n_candidates": len(cand),
        "n_swing": sum(1 for c in cand if c["kind"] == "swing"),
        "n_slide": sum(1 for c in cand if c["kind"] == "slide"),
        "n_open": sum(1 for c in cand if c["kind"] == "open"),
        "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
        "candidates": cand,
    }


# ------------------------------------------------------------------- on a stage: the real proof
def check_on_stage(scene: str, gpu: int = 0) -> int:
    """Boot the scene, register ONE scene-owned swing container, mount a placard, OPEN THE DOOR, and
    assert the placard moved with the leaf.

    This is the whole point of the module. A placard that stays put when the door swings is the
    failure the owner named, and no amount of offline joint-spec reading proves its absence -- only a
    stage does. The scene is brought up through ``IsaacEnv.switch_scene``, the same path
    ``build_and_validate`` uses, so a pass here is a pass on the real pipeline rather than on a
    bespoke harness.
    """
    cand = [c for c in candidates(scene) if c["kind"] == "swing"]
    if not cand:
        print(f"[check] {scene} owns no swing container -- nothing to prove here")
        return 2
    print(f"[check] {scene}: {len(cand)} scene-owned swing container(s)")

    from simulator.environment.isaac_env import IsaacEnv

    # open() boots the app and builds the FIRST scene; switch_scene only works on a warm env
    # (it raises "called before open()" otherwise). Passing the scene straight to open() is the
    # shortest path build_and_validate itself uses.
    env = IsaacEnv(f"gen_{scene}", seed=0, active_gpu=gpu, render_obs=False)
    env.open()
    scene_obj = env.scene_obj
    stage = env.stage
    print(f"[check] scene up: {len(getattr(scene_obj, 'entities', {}) or {})} entities")

    import task_world_builder as bb  # noqa: PLC0415

    world = _as_world(bb, env, scene_obj, stage)

    tried = []
    for target in cand[:6]:
        pp = _find_prim(stage, target)
        if not pp:
            tried.append(f"{target['name']}: no prim on stage")
            continue
        try:
            ok = bb._register_swing_unit(
                world, "unit_scene", pp, target["cat"], target["model"]
            )
        except Exception as exc:  # noqa: BLE001 - a refusal is data, not a crash
            import traceback

            tb = traceback.format_exc().strip().splitlines()
            # The LAST frame inside our tree is the one that matters; a bare message like
            # "'Stage' object has no attribute 'stage'" cost four guesses before this was added.
            frames = [ln.strip() for ln in tb if "isaacsim_try" in ln]
            tried.append(
                f"{target['name']}: {type(exc).__name__}: {exc}\n"
                + "\n".join(f"        {f}" for f in frames[-3:])
            )
            continue
        if not ok:
            tried.append(f"{target['name']}: _register_swing_unit returned False")
            continue
        return _prove_ride(bb, world, stage, "unit_scene", pp, target)

    print("[check] no scene-owned container could be registered:")
    for t in tried:
        print(f"    {t}")
    return 5


def _as_world(bb, env, scene_obj, stage):
    """The ``World`` task_world_builder's registration helpers expect, around an already-built scene.

    task_world_builder builds its own World when it imports a unit; here the stage is the scene's, so the
    fields those helpers touch (stage / scene / front_dir / decals / boxes_of / outer_of) are filled
    and nothing else is invented. If a helper reaches for a field that is missing, that is a REAL
    coupling worth seeing as an AttributeError rather than papering over with a mock."""
    w = bb.World.__new__(bb.World)
    w.stage, w.scene, w.env = stage, scene_obj, env
    w.front_dir, w.boxes_of, w.outer_of, w.decals = {}, {}, {}, {}
    return w


def _prove_ride(bb, world, stage, key: str, pp: str, target: dict) -> int:
    """Mount, open, measure. Returns 0 only when the placard actually moved."""
    ent = world.scene.entities[key]
    print(f"[check] registered {target['cat']}/{target['model']}")
    print(f"[check]   leaf_paths = {getattr(ent, 'leaf_paths', None)}")
    print(f"[check]   front_dir  = {world.front_dir.get(key)}")

    png = _make_probe_placard()
    bb.mount_placard(world, key, pp, png)
    decal = world.decals[key]
    before = _world_centroid(stage, decal)
    print(f"[check] placard prim {decal}")
    print(f"[check]   BEFORE open  {tuple(round(v, 4) for v in before)}")

    parented_under_leaf = bool(getattr(ent, "leaf_paths", None)) and decal.startswith(
        str(ent.leaf_paths[0]).rstrip("/")
    )
    print(f"[check]   parented under the leaf: {parented_under_leaf}")

    _drive_open(world, key)
    after = _world_centroid(stage, decal)
    delta = sum((a - b) ** 2 for a, b in zip(after, before)) ** 0.5
    print(f"[check]   AFTER  open  {tuple(round(v, 4) for v in after)}")
    print(f"[check]   delta = {delta * 1000:.1f} mm")

    if delta < 0.02:
        print(
            "[check] FAIL -- the placard did NOT ride the door; it is hanging in the opening"
        )
        return 1
    print("[check] PASS -- the placard rides the door")
    return 0


def _find_prim(stage, target: dict) -> str:
    """The stage path of a scene-owned instance. behavior_import names them /World/b_<cat>_<model>_<i>."""
    from pxr import Usd  # noqa: PLC0415

    want = f"b_{target['cat']}_{target['model']}_"
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        name = prim.GetName()
        if name.startswith(want):
            return str(prim.GetPath())
    return ""


def _make_probe_placard() -> str:
    import label_generator  # noqa: PLC0415

    out = "/tmp/scene_unit_probe_placard.png"
    label_generator.label_generator(
        "short_string",
        out,
        mount="container",
        title="PROBE",
        text="LINE ALFA",
        sub="ride test",
    )
    return out


def _world_centroid(stage, path: str) -> tuple[float, float, float]:
    from pxr import UsdGeom  # noqa: PLC0415

    prim = stage.GetPrimAtPath(path)
    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    acc = [0.0, 0.0, 0.0]
    for p in pts:
        w = xf.Transform(p)
        acc = [acc[0] + w[0], acc[1] + w[1], acc[2] + w[2]]
    n = max(1, len(pts))
    return (acc[0] / n, acc[1] / n, acc[2] / n)


def _drive_open(world, key: str) -> None:
    """Swing the leaf through the SAME code the render uses -- ``episode_runtime._place_door``.

    Not a hand-rolled rotate: baking FLATTENS each leaf's link transform, so a leaf's local origin
    sits at the asset root and rotating about it sweeps a multi-metre arc (episode_runtime:1920, the
    "broken door"). ``_place_container_leaves`` recomputes the TRUE hinge as the vertical panel edge
    farther from the footprint centre and opens toward the room. Driving anything else would prove a
    placard rides a door that the real pipeline never swings that way.

    ``_place_door`` returns None immediately when ``e.leaf_paths`` is empty (:1816) -- which is
    exactly the field this whole module exists to populate for scene-owned furniture."""
    import math  # noqa: PLC0415

    import episode_runtime as s3  # noqa: PLC0415

    # _place_door BUILDS the swing controller and leaves the leaf CLOSED ("cur_angle": 0.0 at
    # episode_runtime:1913). The render opens it by setting the controller's rotate op --
    # `c["rot"].Set(degrees)` at :1714, inside _drive_drawers. Do exactly that, including the
    # extra_leaves loop, so a 4-door cabinet opens the way the render opens it.
    ctrl = s3._place_door(world.stage, world.scene, key)
    if not ctrl:
        print("[check]   _place_door returned nothing -- no controller, no swing")
        return
    ent = world.scene.entities[key]
    # swing_angle is ALREADY RADIANS -- behavior_import:683 stores math.radians(min(upper, 88)).
    # Converting again turned 88 deg into 1.5 deg and the placard moved 4.6 mm, which read as a FAIL.
    ang = float(getattr(ent, "swing_angle", 0.0) or 0.0) or math.radians(88.0)
    ctrl["rot"].Set(math.degrees(ang))
    ctrl["cur_angle"] = ang
    for ex_rot, ex_sign in ctrl.get("extra_leaves", ()):
        ex_rot.Set(math.degrees(ex_sign * abs(ang)))
    print(
        f"[check]   drove {len(ctrl.get('extra_leaves', ())) + 1} leaf/leaves to "
        f"{math.degrees(ang):.0f} deg"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", metavar="SCENE")
    ap.add_argument("--scan-all", action="store_true")
    ap.add_argument("--check", metavar="SCENE")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args(argv)

    if args.scan:
        rep = scan(args.scan)
        print(json.dumps({k: v for k, v in rep.items() if k != "candidates"}, indent=1))
        for c in rep["candidates"][:20]:
            print(
                f"   {c['kind']:<6} {c['cat']:<24} {c['model']}  joints={c['n_joints']}"
            )
        return 0
    if args.scan_all:
        tot = 0
        for scene in sorted(os.listdir(SCENES_ROOT)):
            try:
                rep = scan(scene)
            except (OSError, ValueError):
                continue
            tot += rep["n_candidates"]
            print(
                f"  {scene:<34} {rep['n_candidates']:>3} candidate(s) "
                f"(swing {rep['n_swing']}, slide {rep['n_slide']}) of {rep['n_instances']}"
            )
        print(f"\n{tot} scene-owned openable container instance(s) across all scenes")
        return 0
    if args.check:
        return check_on_stage(args.check, args.gpu)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
