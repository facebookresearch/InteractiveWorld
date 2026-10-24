#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Open a world and move its human with semantic Python actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="Rs_int")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    release_root = str(Path(__file__).resolve().parents[1])
    if release_root not in sys.path:
        sys.path.insert(0, release_root)

    from simulator import Action, Simulator, SimulatorConfig

    actions = (
        Action.look_around(),
        Action.walk_forward(),
        Action.turn_left(),
    )
    events: list[dict[str, object]] = []
    config = SimulatorConfig(
        gpu=args.gpu,
        seed=0,
        render_observations=not args.no_render,
    )
    with Simulator.launch(scene=args.scene, config=config) as simulator:
        initial = simulator.observe()
        for action in actions:
            result = simulator.step(action)
            events.append(
                {
                    "action": action.to_command(),
                    "accepted": result.success,
                    "terminated": result.terminated,
                    "error_code": result.error_code,
                }
            )
            if result.terminated or result.truncated:
                break

    json.dump(
        {
            "scene": args.scene,
            "initial_location": initial.at,
            "visible_objects": initial.objects,
            "events": events,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
