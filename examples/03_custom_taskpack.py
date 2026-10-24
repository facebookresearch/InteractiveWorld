#!/usr/bin/env python3
"""Load a hash-pinned custom TaskPack and execute semantic actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("taskpack", type=Path)
    parser.add_argument("taskpack_sha256", help="sha256:... digest of the file")
    parser.add_argument("--action", action="append", default=[])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    release_root = str(Path(__file__).resolve().parents[1])
    if release_root not in sys.path:
        sys.path.insert(0, release_root)

    from simulator import Simulator, SimulatorConfig, TaskSpec

    task = TaskSpec.load(
        args.taskpack,
        expected_sha256=args.taskpack_sha256,
    )
    config = SimulatorConfig(
        gpu=args.gpu,
        render_observations=not args.no_render,
    )
    with Simulator.launch(task=task, config=config) as simulator:
        preparation = simulator.prepare_task()
        results = simulator.run(args.action)
        score = simulator.score()

    json.dump(
        {
            "task_id": task.task_id,
            "taskpack_digest": task.taskpack_digest,
            "prepared_actions": len(preparation.steps),
            "executed_actions": len(results),
            "success": score.success,
            "progress": score.progress,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
