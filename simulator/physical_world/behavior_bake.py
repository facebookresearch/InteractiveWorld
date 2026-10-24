# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bake BEHAVIOR-1K objects into the plain-USD cache used by the simulator.

Each model's ``.encrypted.usd`` is decrypted with the BEHAVIOR key, its relative texture
paths are re-rooted to absolute paths inside the BEHAVIOR data directory, and the physics
that OmniGibson ships for its own runtime is stripped, so the simulator can own all bodies
and joints. A ``spec/<category>/<model>.json`` with bounding box, mass, role hint and joint
facts is written beside every baked USD.

``pxr`` is importable only inside a running Isaac Sim application, so ``main`` boots one
headless app and bakes many models in it. Run through ``launch/prepare_assets.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RELEASE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _path in (_THIS_DIR, _RELEASE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DATA_ROOT = os.environ.get(
    "INTERACTIVEWORLD_BEHAVIOR_DATA", "/opt/interactiveworld/behavior-data"
)
ASSETS = os.path.join(DATA_ROOT, "behavior-1k-assets")
OBJ_ROOT = os.path.join(ASSETS, "objects")
SCENE_ROOT = os.path.join(ASSETS, "scenes")
META_ROOT = os.path.join(ASSETS, "metadata")
KEY_PATH = os.path.join(DATA_ROOT, "omnigibson.key")
CACHE_ROOT = os.environ.get(
    "INTERACTIVEWORLD_BEHAVIOR_CACHE", "/opt/interactiveworld/behavior-cache"
)
CACHE_USD = os.path.join(CACHE_ROOT, "usd")
CACHE_SPEC = os.path.join(CACHE_ROOT, "spec")

# Tentative role keywords; behavior_import refines roles with scene context.
_SEAT_KW = (
    "chair",
    "sofa",
    "couch",
    "bench",
    "stool",
    "armchair",
    "ottoman",
    "loveseat",
    "recliner",
)
_STRUCT_KW = (
    "walls",
    "floors",
    "ceilings",
    "driveway",
    "lawn",
    "roof",
    "fence",
    "wall",
    "floor",
    "ceiling",
)

_AVG_SPECS: dict | None = None


def _avg_specs() -> dict:
    global _AVG_SPECS
    if _AVG_SPECS is None:
        with open(os.path.join(META_ROOT, "avg_category_specs.json")) as f:
            _AVG_SPECS = json.load(f)
    return _AVG_SPECS


def _decrypt_bytes(enc_path: str) -> bytes:
    """Decrypt one ``.encrypted.usd`` with the BEHAVIOR Fernet key (no OmniGibson import)."""
    from cryptography.fernet import Fernet

    with open(KEY_PATH, "rb") as f:
        key = f.read()
    with open(enc_path, "rb") as f:
        return Fernet(key).decrypt(f.read())


def encrypted_usd_path(cat: str, model: str) -> str:
    return os.path.join(OBJ_ROOT, cat, model, "usd", f"{model}.encrypted.usd")


def _load_metadata(cat: str, model: str) -> dict:
    mp = os.path.join(OBJ_ROOT, cat, model, "misc", "metadata.json")
    if os.path.exists(mp):
        with open(mp) as f:
            return json.load(f)
    return {}


def role_hint(cat: str, meta: dict) -> str:
    """Tentative role for a model from its category and metadata."""
    spec = _avg_specs().get(cat, {})
    mass = float(spec.get("mass", 1.0))
    bbox = meta.get("bbox_size") or [0.3, 0.3, 0.3]
    maxdim = max(bbox) if bbox else 0.3
    name = cat.lower()
    if any(k in name for k in _STRUCT_KW):
        return "STRUCTURE"
    if meta.get("openable_joint_ids"):
        return "OPENABLE"
    if any(k in name for k in _SEAT_KW):
        return "SEAT"
    if mass >= 15.0 or maxdim >= 0.6:
        return "SURFACE"
    if mass <= 3.0 and maxdim <= 0.35:
        return "GRASPABLE"
    return "SURFACE"


def _extract_joints(stage, UsdPhysics) -> list:
    """Every prismatic and revolute joint, recorded before physics is stripped."""
    out = []
    for pr in stage.Traverse():
        is_p = pr.IsA(UsdPhysics.PrismaticJoint)
        is_r = pr.IsA(UsdPhysics.RevoluteJoint)
        if not (is_p or is_r):
            continue
        _nm = pr.GetName().lower()
        if "meta__" in _nm or "fillable" in _nm:
            continue
        j = UsdPhysics.PrismaticJoint(pr) if is_p else UsdPhysics.RevoluteJoint(pr)

        def _get(attr):
            a = attr()
            return a.Get() if a else None

        def _rel(nm):
            r = pr.GetRelationship(nm)
            t = r.GetTargets() if r else []
            return str(t[0]) if t else None

        out.append(
            {
                "name": pr.GetName(),
                "path": str(pr.GetPath()),
                "type": "prismatic" if is_p else "revolute",
                "axis": str(_get(j.GetAxisAttr)),
                "lower": _get(j.GetLowerLimitAttr),
                "upper": _get(j.GetUpperLimitAttr),
                "body0": _rel("physics:body0"),
                "body1": _rel("physics:body1"),
            }
        )
    return out


def _strip_physics(stage, UsdPhysics, PhysxSchema) -> int:
    """Remove the per-object physics scene, rigid bodies, joints and OmniGibson helper links.

    Nested rigid bodies and dangling joints make PhysX fail once the asset is referenced
    under a simulator-owned body.
    """
    remove_paths = []
    for pr in stage.Traverse():
        nm = pr.GetName().lower()
        is_meta = ("meta__" in nm) or ("fillable" in nm)
        is_joint = (
            pr.IsA(UsdPhysics.Joint)
            or pr.IsA(UsdPhysics.PrismaticJoint)
            or pr.IsA(UsdPhysics.RevoluteJoint)
            or pr.IsA(UsdPhysics.FixedJoint)
        )
        if pr.IsA(UsdPhysics.Scene) or is_meta or is_joint:
            remove_paths.append(pr.GetPath())
            continue
        pr.RemoveAPI(UsdPhysics.RigidBodyAPI)
        pr.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if PhysxSchema is not None:
            for api in ("PhysxRigidBodyAPI", "PhysxArticulationAPI"):
                try:
                    pr.RemoveAPI(getattr(PhysxSchema, api))
                except Exception:  # noqa: BLE001
                    pass
    for p in remove_paths:
        stage.RemovePrim(p)
    return len(remove_paths)


def bake_model(cat: str, model: str, pxr_mods: dict, overwrite: bool = False) -> dict:
    """Bake one model into the cache and write its spec; returns a report row."""
    Usd = pxr_mods["Usd"]
    UsdUtils = pxr_mods["UsdUtils"]
    UsdPhysics = pxr_mods["UsdPhysics"]
    PhysxSchema = pxr_mods.get("PhysxSchema")

    enc = encrypted_usd_path(cat, model)
    if not os.path.exists(enc):
        return {"cat": cat, "model": model, "ok": False, "err": "encrypted usd missing"}
    out_usd = os.path.join(CACHE_USD, cat, f"{model}.usd")
    spec_path = os.path.join(CACHE_SPEC, cat, f"{model}.json")
    if os.path.exists(out_usd) and os.path.exists(spec_path) and not overwrite:
        return {"cat": cat, "model": model, "ok": True, "cached": True}

    os.makedirs(os.path.dirname(out_usd), exist_ok=True)
    enc_dir = os.path.dirname(enc)
    with open(out_usd, "wb") as f:
        f.write(_decrypt_bytes(enc))

    stage = Usd.Stage.Open(out_usd)

    def _reroot(ap: str) -> str:
        # MDL modules resolve through the MDL search path instead.
        if not ap or ".mdl" in ap or os.path.isabs(ap):
            return ap
        return os.path.normpath(os.path.join(enc_dir, ap))

    UsdUtils.ModifyAssetPaths(stage.GetRootLayer(), _reroot)
    joints = _extract_joints(stage, UsdPhysics)
    _strip_physics(stage, UsdPhysics, PhysxSchema)
    stage.GetRootLayer().Save()
    del stage

    meta = _load_metadata(cat, model)
    aspec = _avg_specs().get(cat, {})
    spec = {
        "category": cat,
        "model": model,
        "usd": out_usd,
        "role_hint": role_hint(cat, meta),
        "bbox_size": meta.get("bbox_size"),
        "base_link_offset": meta.get("base_link_offset"),
        "orientations": meta.get("orientations"),
        "mass": aspec.get("mass"),
        "density": aspec.get("density"),
        "openable_joint_ids": meta.get("openable_joint_ids"),
        "link_tags": meta.get("link_tags"),
        "joints": joints,
    }
    os.makedirs(os.path.dirname(spec_path), exist_ok=True)
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    return {"cat": cat, "model": model, "ok": True, "cached": False}


def scene_models(scene_name: str) -> list:
    """Unique ``(category, model)`` pairs of the DatasetObjects in a scene's ``*_best.json``."""
    sj = os.path.join(SCENE_ROOT, scene_name, "json", f"{scene_name}_best.json")
    with open(sj) as f:
        d = json.load(f)
    init = d.get("objects_info", {}).get("init_info", {})
    pairs: list[tuple[str, str]] = []
    seen: set = set()
    for _name, info in init.items():
        args = info.get("args", info) if isinstance(info, dict) else {}
        cat = args.get("category")
        model = args.get("model")
        cls = info.get("class_name") if isinstance(info, dict) else None
        if cls is not None and cls != "DatasetObject":
            continue
        if cat and model and (cat, model) not in seen:
            seen.add((cat, model))
            pairs.append((cat, model))
    return pairs


def _read_model_list(path: str) -> list[tuple[str, str]]:
    with open(path) as f:
        refs = [line.strip() for line in f if line.strip()]
    return [tuple(ref.split("/", 1)) for ref in refs]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-list", required=True, help="file with one cat/model per line")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args(argv)
    if not os.path.isfile(KEY_PATH):
        print(f"[bake] BEHAVIOR key not found: {KEY_PATH}", file=sys.stderr)
        return 2

    pairs = _read_model_list(args.model_list)[args.shard_index :: args.num_shards]
    if not pairs:
        return 0

    import fpv_core

    app = fpv_core.launch_app(
        fpv_core.AppCfg(
            headless=True,
            active_gpu=args.gpu,
            physics_gpu=args.gpu,
            disable_viewport_updates=True,
        )
    )
    from pxr import Usd, UsdPhysics, UsdUtils

    pxr_mods = {"Usd": Usd, "UsdUtils": UsdUtils, "UsdPhysics": UsdPhysics}
    try:
        from pxr import PhysxSchema

        pxr_mods["PhysxSchema"] = PhysxSchema
    except Exception:  # noqa: BLE001
        pass

    failures = []
    baked = cached = 0
    for index, (cat, model) in enumerate(pairs, start=1):
        try:
            row = bake_model(cat, model, pxr_mods, overwrite=args.overwrite)
        except Exception as exc:  # noqa: BLE001 - one bad model must not stop the shard
            row = {"cat": cat, "model": model, "ok": False, "err": repr(exc)}
        if not row["ok"]:
            failures.append(f"{cat}/{model}: {row.get('err')}")
        elif row.get("cached"):
            cached += 1
        else:
            baked += 1
        if index % 200 == 0 or index == len(pairs):
            print(
                f"[bake] shard {args.shard_index}/{args.num_shards}: "
                f"{index}/{len(pairs)} (baked {baked}, cached {cached}, "
                f"failed {len(failures)})",
                flush=True,
            )
    for failure in failures:
        print(f"[bake] FAIL {failure}", file=sys.stderr)
    app.close()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
