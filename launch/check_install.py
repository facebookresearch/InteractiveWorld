#!/usr/bin/env python3
"""Validate packaged files and externally installed simulator dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_external(entry: dict[str, object]) -> Path | None:
    variable = str(entry["environment_variable"])
    value = os.environ.get(variable) or str(entry.get("default_path") or "")
    return Path(value).expanduser() if value else None


def _runtime_checks() -> list[dict[str, object]]:
    executable_value = os.environ.get("ISAAC_PYTHON") or ""
    executable = Path(executable_value).expanduser() if executable_value else None
    executable_absolute = executable is not None and executable.is_absolute()
    checks: list[dict[str, object]] = [
        {
            "name": "Isaac Sim Python executable",
            "environment_variable": "ISAAC_PYTHON",
            "path": str(executable) if executable is not None else None,
            "absolute": executable_absolute,
            "exists": bool(executable_absolute and executable.is_file()),
            "executable": bool(
                executable_absolute
                and executable.is_file()
                and os.access(executable, os.X_OK)
            ),
            "ok": bool(
                executable_absolute
                and executable.is_file()
                and os.access(executable, os.X_OK)
            ),
        }
    ]

    environment_value = os.environ.get("ISAAC_ENV_DIR") or ""
    environment = Path(environment_value).expanduser() if environment_value else None
    environment_absolute = environment is not None and environment.is_absolute()
    checks.append(
        {
            "name": "Optional Isaac environment root",
            "environment_variable": "ISAAC_ENV_DIR",
            "path": str(environment) if environment is not None else None,
            "required": False,
            "absolute": environment_absolute if environment is not None else None,
            "exists": environment.is_dir() if environment_absolute else False,
            "ok": environment is None
            or bool(environment_absolute and environment.is_dir()),
        }
    )

    label_value = os.environ.get("INTERACTIVEWORLD_LABEL_ASSET_ROOT") or ""
    label_root = Path(label_value).expanduser() if label_value else None
    label_absolute = label_root is not None and label_root.is_absolute()
    checks.append(
        {
            "name": "Optional TaskPack label asset override",
            "environment_variable": "INTERACTIVEWORLD_LABEL_ASSET_ROOT",
            "path": str(label_root) if label_root is not None else None,
            "required": False,
            "absolute": label_absolute if label_root is not None else None,
            "exists": label_root.is_dir() if label_absolute else False,
            "ok": label_root is None or bool(label_absolute and label_root.is_dir()),
        }
    )

    generated_value = os.environ.get("INTERACTIVEWORLD_GENERATED_LABEL_CACHE") or ""
    generated = Path(generated_value).expanduser() if generated_value else None
    generated_absolute = generated is not None and generated.is_absolute()
    checks.append(
        {
            "name": "Optional generated-label cache",
            "environment_variable": "INTERACTIVEWORLD_GENERATED_LABEL_CACHE",
            "path": str(generated) if generated is not None else None,
            "required": False,
            "absolute": generated_absolute if generated is not None else None,
            "ok": generated is None or bool(generated_absolute),
        }
    )
    return checks


def _task_asset_references(value: object) -> set[str]:
    references: set[str] = set()
    if isinstance(value, dict):
        if value.get("renderer") == "document_scan" and isinstance(
            value.get("source"), str
        ):
            references.add(value["source"])
        if isinstance(value.get("texture"), str):
            references.add(value["texture"])
        for child in value.values():
            references.update(_task_asset_references(child))
    elif isinstance(value, list):
        for child in value:
            references.update(_task_asset_references(child))
    return references


def _bundled_label_check(release_root: Path) -> dict[str, object]:
    manifest_path = release_root / "benchmark" / "tasks" / "assets" / "manifest.json"
    if not manifest_path.is_file():
        return {
            "name": "Bundled TaskPack label assets",
            "path": str(manifest_path),
            "ok": False,
            "error": "asset manifest is missing",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        asset_root = (manifest_path.parent / str(manifest["asset_root"])).resolve()
        if not asset_root.is_relative_to(manifest_path.parent):
            raise ValueError("asset_root escapes the task asset directory")
        expected_files = {
            str(entry["path"]): entry for entry in manifest.get("files", [])
        }
        task_manifest_path = (
            manifest_path.parent / str(manifest["selection"]["task_manifest"])
        ).resolve()
        task_root = task_manifest_path.parent
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        referenced_files: set[str] = set()
        for task in task_manifest["tasks"]:
            taskpack_path = (task_root / str(task["taskpack_path"])).resolve()
            if not taskpack_path.is_relative_to(task_root):
                raise ValueError("taskpack_path escapes the released task directory")
            taskpack = json.loads(taskpack_path.read_text(encoding="utf-8"))
            referenced_files.update(_task_asset_references(taskpack))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "name": "Bundled TaskPack label assets",
            "path": str(manifest_path),
            "ok": False,
            "error": f"invalid asset manifest: {error}",
        }

    missing: list[str] = []
    invalid: list[str] = []
    for relative, entry in expected_files.items():
        path = (asset_root / relative).resolve()
        if not path.is_relative_to(asset_root) or not path.is_file():
            missing.append(relative)
            continue
        expected = str(entry.get("sha256") or "").removeprefix("sha256:")
        if path.stat().st_size != entry.get("size_bytes") or _sha256(path) != expected:
            invalid.append(relative)
    actual_files = (
        {
            path.relative_to(asset_root).as_posix()
            for path in asset_root.rglob("*")
            if path.is_file()
        }
        if asset_root.is_dir()
        else set()
    )
    unexpected = sorted(actual_files - set(expected_files))
    unmanifested_references = sorted(referenced_files - set(expected_files))
    unreferenced_manifest_files = sorted(set(expected_files) - referenced_files)
    count_ok = manifest.get("file_count") == len(expected_files)
    return {
        "name": "Bundled TaskPack label assets",
        "path": str(asset_root),
        "ok": bool(
            count_ok
            and not missing
            and not invalid
            and not unexpected
            and not unmanifested_references
            and not unreferenced_manifest_files
        ),
        "file_count": len(expected_files),
        "missing": sorted(missing),
        "invalid": sorted(invalid),
        "unexpected": unexpected,
        "unmanifested_references": unmanifested_references,
        "unreferenced_manifest_files": unreferenced_manifest_files,
    }


def check_install() -> dict[str, object]:
    release_root = Path(__file__).resolve().parents[1]
    simulator_root = release_root / "simulator"
    manifest = json.loads(
        (simulator_root / "external_dependencies.json").read_text(encoding="utf-8")
    )
    checks = _runtime_checks()

    for entry in manifest["required_external_assets"]:
        root = _resolve_external(entry)
        required = (
            [root / value for value in entry.get("required_subdirectories", [])]
            + [root / value for value in entry.get("required_files", [])]
            if root is not None
            else []
        )
        expected_sha256 = entry.get("required_sha256")
        actual_sha256 = (
            _sha256(root)
            if root is not None and root.is_file() and expected_sha256
            else None
        )
        exists = root is not None and root.exists()
        absolute = root is not None and root.is_absolute()
        checksum_ok = not expected_sha256 or actual_sha256 == expected_sha256
        checks.append(
            {
                "name": entry["name"],
                "environment_variable": entry["environment_variable"],
                "path": str(root) if root is not None else None,
                "absolute": absolute,
                "exists": exists,
                "ok": absolute
                and exists
                and checksum_ok
                and all(path.exists() for path in required),
                "missing": [str(path) for path in required if not path.exists()],
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
            }
        )

    checks.append(_bundled_label_check(release_root))

    for entry in manifest["bundled_human_assets"]:
        path = simulator_root / str(entry["path"])
        expected = str(entry["sha256"])
        actual = _sha256(path) if path.is_file() else None
        checks.append(
            {
                "name": str(entry["path"]),
                "path": str(path),
                "ok": actual == expected,
                "expected_sha256": expected,
                "actual_sha256": actual,
            }
        )

    return {
        "schema_version": "interactiveworld-install-check/1.0",
        "ok": all(bool(check["ok"]) for check in checks),
        "checks": checks,
        "deferred_to_isaac_runtime": {
            "python_packages": manifest["python_packages"],
            "isaac_provided_modules": manifest["isaac_provided_modules"],
            "reason": "These imports are checked only inside the selected Isaac Python runtime.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = check_install()
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(args.strict and not result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
