#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare every external asset InteractiveWorld needs from the original downloads.

Run inside the Isaac Sim Python through ``launch/prepare_assets.sh``:

* bakes the BEHAVIOR-1K 3.9.0 objects into ``<output>/behavior-cache``;
* copies the OmniGibson MDL materials next to the cache;
* converts the MANO-site ``SMPLH_male.pkl`` into a chumpy-free pickle;
* extracts the ACCAD forward-walk clip from the AMASS archive;
* writes ``<output>/env.sh`` with every variable the simulator reads.

Every step is idempotent and verifies the frozen checksums of the release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST = RELEASE_ROOT / "simulator" / "world_catalog" / "behavior_assets.json"
DEPENDENCIES = RELEASE_ROOT / "simulator" / "external_dependencies.json"
SMPLH_MEMBER = "SMPLH_male.pkl"
WALK_MEMBER = "ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _required_sha256(name: str) -> str:
    manifest = json.loads(DEPENDENCIES.read_text(encoding="utf-8"))
    for entry in manifest["required_external_assets"]:
        if entry["name"] == name:
            return f"sha256:{entry['required_sha256']}"
    raise KeyError(name)


def _log(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


# ---------------------------------------------------------------- BEHAVIOR-1K


def _check_behavior_data(data: Path, manifest: dict) -> None:
    assets = data / "behavior-1k-assets"
    version_file = assets / "VERSION"
    if not version_file.is_file():
        raise SystemExit(f"{assets} is not a BEHAVIOR-1K dataset (VERSION missing)")
    version = version_file.read_text(encoding="utf-8").strip()
    expected = manifest["behavior_asset_version"]
    if version != expected:
        raise SystemExit(f"BEHAVIOR-1K {expected} is required, found {version}")
    if not (data / "omnigibson.key").is_file():
        raise SystemExit(
            f"{data / 'omnigibson.key'} is missing; accept the BEHAVIOR data license "
            "with the BEHAVIOR-1K installer to obtain it"
        )
    mismatched = []
    for ref, identity in manifest["frozen_identities"].items():
        cat, model = ref.removesuffix(".usd").split("/")
        source = assets / "objects" / cat / model / "usd" / f"{model}.encrypted.usd"
        if not source.is_file() or _sha256(source) != identity["source_sha256"]:
            mismatched.append(ref)
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} BEHAVIOR objects differ from the frozen release, "
            f"e.g. {mismatched[:3]}"
        )
    _log(f"BEHAVIOR-1K {version} verified at {data}")


def _bake(
    data: Path, cache: Path, manifest: dict, workers: int, gpus: list[int]
) -> None:
    models = manifest["bake_models"]
    pending = [
        ref
        for ref in models
        if not (cache / "usd" / f"{ref}.usd").is_file()
        or not (cache / "spec" / f"{ref}.json").is_file()
    ]
    _log(f"{len(models) - len(pending)}/{len(models)} objects already baked")
    if not pending:
        return
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["INTERACTIVEWORLD_BEHAVIOR_DATA"] = str(data)
    environment["INTERACTIVEWORLD_BEHAVIOR_CACHE"] = str(cache)
    environment.setdefault("ISAAC_PYTHON", sys.executable)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    workers = max(1, min(workers, len(pending)))
    with tempfile.TemporaryDirectory() as staging:
        model_list = Path(staging) / "models.txt"
        model_list.write_text("\n".join(pending) + "\n", encoding="utf-8")
        logs = []
        processes = []
        for index in range(workers):
            log = (cache / f"bake_shard_{index}.log").open("w")
            logs.append(log)
            command = [
                str(RELEASE_ROOT / "launch" / "run_isaac.sh"),
                str(RELEASE_ROOT / "simulator" / "physical_world" / "behavior_bake.py"),
                "--model-list",
                str(model_list),
                "--shard-index",
                str(index),
                "--num-shards",
                str(workers),
                "--gpu",
                str(gpus[index % len(gpus)]),
            ]
            processes.append(
                subprocess.Popen(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            )
        _log(
            f"baking {len(pending)} objects with {workers} Isaac worker(s); "
            f"logs: {cache}/bake_shard_*.log"
        )
        codes = [process.wait() for process in processes]
        for log in logs:
            log.close()
    missing = [ref for ref in models if not (cache / "usd" / f"{ref}.usd").is_file()]
    if any(codes) or missing:
        raise SystemExit(
            f"baking failed for {len(missing)} objects (exit codes {codes}); "
            f"see {cache}/bake_shard_*.log and rerun to resume"
        )
    _log(f"baked {len(pending)} objects into {cache}")


def _find_mdl_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    spec = importlib.util.find_spec("omnigibson")
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidate = Path(location) / "materials"
            if candidate.is_dir():
                return candidate
    raise SystemExit(
        "cannot locate the OmniGibson MDL materials; pass "
        "--mdl-dir /path/to/BEHAVIOR-1K/OmniGibson/omnigibson/materials"
    )


def _copy_materials(source: Path, cache: Path) -> None:
    files = sorted(source.glob("*.mdl"))
    if not files:
        raise SystemExit(f"no .mdl files in {source}")
    target = cache / "materials"
    target.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copyfile(path, target / path.name)
    _log(f"copied {len(files)} MDL materials into {target}")


# ---------------------------------------------------------------- human model


class _ChumpyNode:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _ChumpyFreeUnpickler(pickle.Unpickler):
    """Load an SMPL-family pickle without chumpy installed."""

    def find_class(self, module, name):
        if "chumpy" in module:
            return type(name, (_ChumpyNode,), {"__module__": module})
        try:
            return super().find_class(module, name)
        except Exception:  # noqa: BLE001 - unknown legacy classes become inert nodes
            return type(name, (_ChumpyNode,), {"__module__": module})


def _resolve_chumpy(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, _ChumpyNode):
        state = value.__dict__
        if "idxs" in state and "a" in state:
            flat = np.asarray(_resolve_chumpy(state["a"])).ravel()
            out = flat[np.asarray(state["idxs"]).astype(np.int64)]
            shape = state.get("preferred_shape")
            return out.reshape(tuple(shape)) if shape is not None else out
        for key in ("x", "_result", "r", "a"):
            if key in state and isinstance(state[key], np.ndarray):
                return state[key]
    raise ValueError("unresolved chumpy node")


def _plain_array(value):
    import numpy as np

    if "scipy.sparse" in str(type(value)):
        return np.asarray(value.todense())
    if isinstance(value, _ChumpyNode) or "chumpy" in str(type(value)):
        return np.asarray(_resolve_chumpy(value))
    return np.asarray(value)


def _read_smplh_source(source: Path) -> bytes:
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = [
                name
                for name in archive.namelist()
                if Path(name).name == SMPLH_MEMBER and "__MACOSX" not in name
            ]
            if len(members) != 1:
                raise SystemExit(f"{source} does not contain exactly one {SMPLH_MEMBER}")
            return archive.read(members[0])
    return source.read_bytes()


def _convert_smplh(source: Path, destination: Path) -> None:
    expected = _required_sha256("SMPL+H male body model")
    if destination.is_file() and _sha256(destination) == expected:
        _log(f"SMPL+H model already prepared at {destination}")
        return
    model = _ChumpyFreeUnpickler(
        io.BytesIO(_read_smplh_source(source)), encoding="latin1"
    ).load()
    clean = {
        key: (value if key in ("bs_style", "bs_type") else _plain_array(value))
        for key, value in model.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(clean, handle, protocol=4)
    actual = _sha256(destination)
    if actual != expected:
        raise SystemExit(
            f"converted SMPL+H model {destination} has {actual}, expected {expected}; "
            "use SMPLH_male.pkl from the MANO 'Models & Code' download (mano_v1_2.zip) "
            "and run this tool with the Isaac Sim Python"
        )
    _log(f"SMPL+H model written to {destination}")


def _extract_walk_clip(source: Path, destination: Path) -> None:
    expected = _required_sha256("ACCAD/AMASS forward walk clip")
    if destination.is_file() and _sha256(destination) == expected:
        _log(f"walk clip already prepared at {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        candidate = source / Path(WALK_MEMBER).relative_to("ACCAD")
        if not candidate.is_file():
            candidate = source / WALK_MEMBER
        shutil.copyfile(candidate, destination)
    else:
        with tarfile.open(source) as archive:
            member = archive.getmember(WALK_MEMBER)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SystemExit(f"{WALK_MEMBER} is not a file in {source}")
            destination.write_bytes(extracted.read())
    actual = _sha256(destination)
    if actual != expected:
        raise SystemExit(
            f"walk clip {destination} has {actual}, expected {expected}; use the AMASS "
            "ACCAD 'SMPL-X N' archive"
        )
    _log(f"walk clip written to {destination}")


# ---------------------------------------------------------------- environment file


def _write_environment(output: Path, values: dict[str, Path]) -> Path:
    path = output / "env.sh"
    lines = ["# Generated by InteractiveWorld launch/prepare_assets.sh"]
    lines.append(f"export ISAAC_PYTHON={sys.executable}")
    lines.append(f"export ISAAC_ENV_DIR={sys.prefix}")
    for name, value in values.items():
        lines.append(f"export {name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory that receives the cache, human assets and env.sh",
    )
    parser.add_argument(
        "--behavior-data",
        type=Path,
        help="OmniGibson data directory containing behavior-1k-assets/ and omnigibson.key",
    )
    parser.add_argument(
        "--mdl-dir",
        type=Path,
        help="OmniGibson materials directory (default: found via the installed omnigibson)",
    )
    parser.add_argument(
        "--smplh",
        type=Path,
        help="mano_v1_2.zip from the MANO website, or the SMPLH_male.pkl inside it",
    )
    parser.add_argument(
        "--accad",
        type=Path,
        help="AMASS ACCAD 'SMPL-X N' archive (ACCAD.tar.bz2) or its extracted folder",
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel bake processes")
    parser.add_argument(
        "--gpus", default="0", help="comma-separated physical GPUs for baking"
    )
    args = parser.parse_args(argv)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    values: dict[str, Path] = {}
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    cache = output / "behavior-cache"

    if args.behavior_data is not None:
        data = args.behavior_data.expanduser().resolve()
        _check_behavior_data(data, manifest)
        gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
        _bake(data, cache, manifest, args.workers, gpus or [0])
        _copy_materials(_find_mdl_dir(args.mdl_dir), cache)
        values["INTERACTIVEWORLD_BEHAVIOR_DATA"] = data
        values["INTERACTIVEWORLD_BEHAVIOR_CACHE"] = cache
    if args.smplh is not None:
        smplh = output / "human" / SMPLH_MEMBER
        _convert_smplh(args.smplh.expanduser().resolve(), smplh)
        values["INTERACTIVEWORLD_SMPLH_MODEL"] = smplh
    if args.accad is not None:
        walk = output / "human" / WALK_MEMBER
        _extract_walk_clip(args.accad.expanduser().resolve(), walk)
        values["INTERACTIVEWORLD_WALK_CLIP"] = walk
    if not values:
        parser.error("nothing to prepare; pass --behavior-data, --smplh and/or --accad")

    previous = output / "env.sh"
    if previous.is_file():
        for line in previous.read_text(encoding="utf-8").splitlines():
            name, _, value = line.removeprefix("export ").partition("=")
            if name.startswith("INTERACTIVEWORLD_") and name not in values:
                values[name] = Path(value)
    environment_file = _write_environment(output, dict(sorted(values.items())))
    _log(f"wrote {environment_file}")
    _log(f"next: source {environment_file} && ./launch/simulator.sh doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
