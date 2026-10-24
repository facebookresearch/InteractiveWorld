"""Versioned constants shared by the public simulator client and server."""

from __future__ import annotations


PROTOCOL_VERSION = "interactiveworld-sim/1.0"
OBSERVATION_PROFILES = frozenset({"agent", "debug"})
PUBLIC_OPERATIONS = (
    "health",
    "describe",
    "open_world",
    "switch_world",
    "open_task",
    "switch_task",
    "reset",
    "step",
    "observe",
    "render",
    "score",
    "reward",
    "capture_score_baseline",
    "close",
    "shutdown",
)
REQUIRED_CLIENT_OPERATIONS = frozenset(PUBLIC_OPERATIONS) - {"close"}


__all__ = [
    "OBSERVATION_PROFILES",
    "PROTOCOL_VERSION",
    "PUBLIC_OPERATIONS",
    "REQUIRED_CLIENT_OPERATIONS",
]
