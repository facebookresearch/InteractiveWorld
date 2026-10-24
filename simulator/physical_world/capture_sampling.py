# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Pure frame-index samplers shared by rendering and dataset validation."""

from __future__ import annotations


ACTION_QUARTILE_POSITIONS = (0.25, 0.5, 0.75, 1.0)
ACTION_QUARTILE_FRAME_COUNT = len(ACTION_QUARTILE_POSITIONS)


def action_quartile_indices(frame_start: int, frame_end_exclusive: int) -> list[int]:
    """Select the last frame at each quarter boundary of one action."""
    span = frame_end_exclusive - frame_start
    if frame_start < 0 or span < ACTION_QUARTILE_FRAME_COUNT:
        raise ValueError("action window has fewer than four frames")
    return [
        frame_start + (span * numerator - 1) // ACTION_QUARTILE_FRAME_COUNT
        for numerator in range(1, ACTION_QUARTILE_FRAME_COUNT + 1)
    ]
