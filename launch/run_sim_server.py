#!/usr/bin/env python3
"""Start one persistent simulator process."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True


def main(argv: list[str] | None = None) -> int:
    release_root = str(Path(__file__).resolve().parents[1])
    if release_root not in sys.path:
        sys.path.insert(0, release_root)

    from simulator.transport.isaac_sim_server import main as serve

    original_argv = sys.argv
    try:
        if argv is not None:
            sys.argv = [original_argv[0], *argv]
        result = serve()
    finally:
        sys.argv = original_argv
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
