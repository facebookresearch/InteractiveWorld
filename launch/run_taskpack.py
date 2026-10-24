#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run the frozen benchmark TaskPack evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main(argv: list[str] | None = None) -> int:
    release_root = str(Path(__file__).resolve().parents[1])
    if release_root not in sys.path:
        sys.path.insert(0, release_root)

    from benchmark.evaluation.run_taskpack import main as evaluate_taskpack

    return evaluate_taskpack(argv)


if __name__ == "__main__":
    raise SystemExit(main())
