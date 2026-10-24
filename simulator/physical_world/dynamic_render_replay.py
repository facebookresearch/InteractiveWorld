# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared per-frame replay for objects that become PhysX-owned during rendering."""

from __future__ import annotations

from collections.abc import Iterable


def snapshot_dynamic_prims(stage, paths: Iterable[str]) -> dict[str, object]:
    """Capture the full command-boundary state of dynamic prims without stepping time."""
    import object_physics
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    states = {}
    for raw_path in paths:
        path = str(raw_path).rstrip("/")
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            continue
        cache.Clear()
        pose = object_physics._read_pose(cache, prim)
        rigid = UsdPhysics.RigidBodyAPI(prim)
        kinematic = rigid.GetKinematicEnabledAttr()
        was_kinematic = bool(kinematic.Get()) if kinematic.HasAuthoredValue() else False
        states[path] = (
            was_kinematic,
            pose,
            rigid.GetVelocityAttr().Get(),
            rigid.GetAngularVelocityAttr().Get(),
        )
    return states


def freeze_dynamic_prims(stage, paths: Iterable[str]) -> dict[str, object]:
    """Pause dynamic prims across non-visual physics work and return a restore token."""
    import physics_human

    states = snapshot_dynamic_prims(stage, paths)
    for path in states:
        prim = stage.GetPrimAtPath(path)
        if prim is not None and prim.IsValid():
            physics_human.set_rigid_kinematic(prim, True)
    return states


def restore_dynamic_prims(stage, states: dict[str, object]) -> None:
    """Restore pose, ownership and velocity captured by ``snapshot_dynamic_prims``."""
    import fpv_scene
    import physics_human
    from pxr import UsdPhysics

    for path, (was_kinematic, pose, velocity, angular_velocity) in states.items():
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            continue
        physics_human.set_rigid_kinematic(prim, True)
        fpv_scene.set_world_pose(prim, pose[0], pose[1])
        physics_human.set_rigid_kinematic(prim, bool(was_kinematic))
        rigid = UsdPhysics.RigidBodyAPI(prim)
        if velocity is not None:
            rigid.GetVelocityAttr().Set(velocity)
        if angular_velocity is not None:
            rigid.GetAngularVelocityAttr().Set(angular_velocity)


class DynamicRenderReplay:
    """Release declared dynamic objects and read their post-step poses chronologically."""

    def __init__(
        self,
        stage,
        physics_human,
        object_physics,
        proxy,
        dynamic_paths: Iterable[str],
        registry: dict[str, object] | None = None,
        owned_paths: set[str] | None = None,
    ) -> None:
        self._stage = stage
        self._physics_human = physics_human
        self._object_physics = object_physics
        self._proxy = proxy
        self._dynamic_paths = {str(path).rstrip("/") for path in dynamic_paths}
        self.registry = registry if registry is not None else {}
        self.released_paths = owned_paths if owned_paths is not None else set()
        self.focus_path: str | None = None

    def before_frame(self, frame, joints, *, object_frame: bool = True) -> str | None:
        """Drive the collision proxy and release the frame's object at most once."""
        if self._proxy is not None:
            self._physics_human.update_proxy_rig(self._proxy, joints)
        if not object_frame:
            return None
        kind = str(frame.obj[0])
        if kind not in ("rest", "carry"):
            return None
        path = str(getattr(frame, "obj_path", None) or "/World/Obj").rstrip("/")
        if kind == "carry" and path in self.released_paths:
            prim = self._stage.GetPrimAtPath(path)
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"free_dynamic prim is missing: {path}")
            self._physics_human.set_rigid_kinematic(prim, True)
            self.released_paths.remove(path)
            self.registry.pop(path, None)
            if self.focus_path == path:
                self.focus_path = None
            return None
        if path in self.released_paths:
            self.focus_path = path
        if not bool(getattr(frame, "free_dynamic", False)):
            return None
        if path not in self._dynamic_paths:
            raise RuntimeError(
                f"free_dynamic frame references undeclared dynamic prim: {path}"
            )
        self.focus_path = path
        if path in self.released_paths:
            return None
        prim = self._stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"free_dynamic prim is missing: {path}")
        self._physics_human.release_to_dynamic([prim], step_fn=None)
        self.registry[path] = prim
        self.released_paths.add(path)
        return path

    def owns_frame_object(self, frame, *, object_frame: bool = True) -> bool:
        """Whether PhysX, rather than scripted rest authoring, owns this frame's object."""
        if not object_frame:
            return False
        if str(frame.obj[0]) not in ("rest", "carry"):
            return False
        path = str(getattr(frame, "obj_path", None) or "/World/Obj").rstrip("/")
        return path in self.released_paths

    def after_step(self) -> dict[str, object]:
        """Read every PhysX-owned prim after the render clock's positive time step."""
        positions = {}
        for path, prim in self.registry.items():
            if prim is None or not prim.IsValid():
                continue
            positions[path] = self._object_physics.read_prim_xyz(self._stage, path)
        return positions

    def focus_from(self, positions: dict[str, object]):
        """Return the active released object's latest position, if available."""
        return positions.get(self.focus_path) if self.focus_path is not None else None
