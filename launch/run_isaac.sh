#!/bin/bash
set -euo pipefail

if [[ -z "${ISAAC_PYTHON:-}" ]]; then
  echo "Set ISAAC_PYTHON to the Isaac Sim 5.1 Python executable." >&2
  exit 2
fi
if [[ "${ISAAC_PYTHON}" != /* ]]; then
  echo "ISAAC_PYTHON must be an absolute path: ${ISAAC_PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "ISAAC_PYTHON is not an executable file: ${ISAAC_PYTHON}" >&2
  exit 2
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export OMNI_KIT_REGISTRY_DISABLED="${OMNI_KIT_REGISTRY_DISABLED:-1}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${ISAAC_ENV_DIR:-}" ]]; then
  if [[ "${ISAAC_ENV_DIR}" != /* ]]; then
    echo "ISAAC_ENV_DIR must be an absolute path: ${ISAAC_ENV_DIR}" >&2
    exit 2
  fi
  if [[ ! -d "${ISAAC_ENV_DIR}" ]]; then
    echo "ISAAC_ENV_DIR is not a directory: ${ISAAC_ENV_DIR}" >&2
    exit 2
  fi
  export LD_LIBRARY_PATH="${ISAAC_ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec "${ISAAC_PYTHON}" "$@"
