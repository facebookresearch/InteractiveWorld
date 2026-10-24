#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Derive a new TaskPack from a bundled one and print its pinned SHA256.

ASK-12802 asks which snack fits the remaining calorie budget and then has the
person carry it to the kitchen breakfast table. The derived task keeps the scene,
objects, labels and reasoning, but moves the destination to the living-room table
where the nutrition report lies. Every place that names the destination changes
together: the goal, the oracle program, the Human policy and the story text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

BASE_TASK = "ASK-12802"
NEW_TASK = "MY-ASK-12802-LIVINGROOM"
OLD_DESTINATION, NEW_DESTINATION = "surf_3", "surf_call"
PHRASES = (
    ("kitchen breakfast table", "living-room table"),
    ("breakfast table", "living-room table"),
)


def _rewrite(value: object, *, path: tuple[str, ...] = ()) -> object:
    if isinstance(value, dict):
        return {
            key: (
                item
                if path + (key,) == ("executable_spec", "world", "fixtures")
                else _rewrite(item, path=path + (key,))
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite(item, path=path) for item in value]
    if isinstance(value, str):
        text = re.sub(rf"\b{OLD_DESTINATION}\b", NEW_DESTINATION, value)
        for old, new in PHRASES:
            text = text.replace(old, new)
        return text.replace(BASE_TASK, NEW_TASK)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="where to write the new TaskPack")
    args = parser.parse_args()

    release_root = Path(__file__).resolve().parents[1]
    if str(release_root) not in sys.path:
        sys.path.insert(0, str(release_root))
    from simulator import TaskCatalog, TaskSpec

    base = TaskCatalog.bundled().get(BASE_TASK).taskpack
    taskpack = _rewrite(base)
    taskpack["lineage"] = {
        "source": "examples/05_derive_taskpack.py",
        "base_task_id": BASE_TASK,
        "base_taskpack_sha256": TaskCatalog.bundled().get(BASE_TASK).source_sha256,
    }

    args.output.write_text(json.dumps(taskpack, indent=2) + "\n", encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(args.output.read_bytes()).hexdigest()
    task = TaskSpec.load(args.output, expected_sha256=digest)
    oracle = " ".join(
        f"--action '[{row['verb']}]"
        + "".join(f" <{name}> (1)" for name in row["args"])
        + "'"
        for row in taskpack["runtime"]["oracle_commands"]
    )
    print(f"task_id  {task.task_id}   {digest}")
    print(f"scene    {task.scene}   preparation {task.preparation_mode}")
    print(f"oracle   {oracle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
