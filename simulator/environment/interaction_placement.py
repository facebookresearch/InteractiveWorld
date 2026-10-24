# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Exact opt-in TaskPack placement slots for the interactive Isaac runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import placement_slots


def _rest_shape(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(size) for size in shape)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    if len(value) == 3:
        return (3,)
    if len(value) == 4 and all(
        isinstance(row, Sequence)
        and not isinstance(row, (str, bytes))
        and len(row) == 4
        for row in value
    ):
        return (4, 4)
    return ()


def settle_or_lock_taskpack_placement(
    scene,
    goal,
    stage,
    physics,
    placed_objects: Mapping[str, tuple[int, object]],
    settle_physics: Callable[[], object],
    *,
    read_world: Callable[[object, str], object] | None = None,
    write_world: Callable[[object, str, object], bool] | None = None,
) -> tuple[object, bool]:
    """Lock one declared child-parent release pose, otherwise run normal settling."""
    behavior_task = getattr(scene, "behavior_task", None) or {}
    dialogue = behavior_task.get("episode_dialogue") or {}
    meta = getattr(goal, "meta", None) or {}
    child = str(meta.get("_held_object_key") or "")
    parent = str(meta.get("surface") or "")
    if read_world is None or write_world is None:
        import semantic_executor as se

        read_world = read_world or se._read_world_col
        write_world = write_world or se._write_world_col
    assert read_world is not None and write_world is not None

    def lock_slot() -> bool:
        entity = (getattr(scene, "entities", None) or {}).get(child)
        path = str(getattr(entity, "prim_path", None) or "")
        row = placed_objects.get(path)
        if row is None:
            return False
        rest = row[1]
        shape = _rest_shape(rest)
        if shape != (4, 4):
            current = read_world(stage, path)
            if current is None or shape != (3,):
                return False
            current[:3, 3] = rest
            rest = current
        if not write_world(stage, path, rest):
            return False
        prim = stage.GetPrimAtPath(path)
        if prim is not None and prim.IsValid():
            physics.set_rigid_kinematic(prim, True)
        return True

    return placement_slots.settle_or_lock(
        dialogue,
        child,
        parent,
        lock_slot,
        settle_physics,
    )
