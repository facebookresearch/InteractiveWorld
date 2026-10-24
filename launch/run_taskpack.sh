#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
exec "${SCRIPT_DIR}/run_isaac.sh" "${SCRIPT_DIR}/run_taskpack.py" "$@"
