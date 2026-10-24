# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Exact opt-in contracts for placements constrained by a physical fixture slot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar


_Notes = TypeVar("_Notes")


def declared_slot(
    dialogue: Mapping[str, object], child: str, parent: str
) -> tuple[str, str] | None:
    rows = dialogue.get("kinematic_place_slots") or ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        pair = (str(raw.get("child") or ""), str(raw.get("parent") or ""))
        if pair == (child, parent):
            return pair
    return None


def settle_or_lock(
    dialogue: Mapping[str, object],
    child: str,
    parent: str,
    lock_slot: Callable[[], bool],
    settle_physics: Callable[[], _Notes],
) -> tuple[_Notes | dict[object, object], bool]:
    if declared_slot(dialogue, child, parent) is not None and lock_slot():
        return {}, True
    return settle_physics(), False
