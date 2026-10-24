#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Explicitly apply a structured TaskPack handoff before agent actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="NARRATE-104000")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    release_root = str(Path(__file__).resolve().parents[1])
    if release_root not in sys.path:
        sys.path.insert(0, release_root)

    from simulator import Simulator, SimulatorConfig

    config = SimulatorConfig(
        gpu=args.gpu,
        seed=0,
        render_observations=not args.no_render,
    )
    with Simulator.launch(task_id=args.task, config=config) as simulator:
        raw_observation = simulator.observe()
        preparation = simulator.prepare_task()

    json.dump(
        {
            "task_id": args.task,
            "raw_scene": raw_observation.scene,
            "preparation_actions": len(preparation.steps),
            "handoff_visible_objects": preparation.observation.objects,
            "score_baseline": {
                "progress": preparation.score_baseline.progress,
                "n_done": preparation.score_baseline.n_done,
                "n_total": preparation.score_baseline.n_total,
            },
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
