#!/usr/bin/env python3
"""Source-tree entrypoint for the InteractiveWorld benchmark runner."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True


def _entrypoint() -> int:
    release_root = Path(__file__).resolve().parents[1]
    if str(release_root) not in sys.path:
        sys.path.insert(0, str(release_root))
    from benchmark.runner.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
