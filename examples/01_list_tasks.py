#!/usr/bin/env python3
"""List the bundled structured tasks without starting Isaac Sim."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    release_root = str(Path(__file__).resolve().parents[1])
    if release_root not in sys.path:
        sys.path.insert(0, release_root)

    from simulator import TaskCatalog

    tasks = TaskCatalog.bundled().list()
    rows = [
        {
            "task_id": task.task_id,
            "category": task.category,
            "source_family": task.source_family,
            "taskpack_sha256": task.taskpack_sha256,
        }
        for task in tasks
    ]
    json.dump({"count": len(rows), "tasks": rows}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
