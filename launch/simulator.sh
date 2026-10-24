#!/usr/bin/env bash
set -euo pipefail

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "${RELEASE_ROOT}/launch/simulator.py" "$@"
