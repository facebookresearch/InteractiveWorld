# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Isaac adapters for the renderer-independent egocentric camera policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import egocentric_camera as camera
import numpy as np


def frame_focus(
    frame, object_transform, free_focus=None, *, physx_owned: bool = False
) -> np.ndarray | None:
    """Return the world point the FPV should frame for one motion frame."""
    import humanoid_motion as wpp

    gaze = getattr(frame, "gaze", None)
    if gaze is not None:
        return np.asarray(gaze, dtype=np.float64)
    kind, value = frame.obj
    if getattr(frame, "free_dynamic", False) or physx_owned:
        return wpp.object_rest_position(free_focus if free_focus is not None else value)
    if kind == "rest":
        return wpp.object_rest_position(value)
    if kind == "carry" and object_transform is not None:
        return np.asarray(object_transform[:3, 3], dtype=np.float64)
    return None


class FocusVisibility:
    """Generic line-of-sight test used to decide whether a narrow zoom is safe."""

    def __init__(
        self,
        grid,
        solid_boxes: Sequence[tuple[str, Sequence[float], Sequence[float]]],
        *,
        samples: int = 24,
        endpoint_fraction: float = 0.92,
    ) -> None:
        if samples < 3:
            raise ValueError("samples must be at least 3")
        if not 0.5 < endpoint_fraction < 1.0:
            raise ValueError("endpoint_fraction must be between 0.5 and 1.0")
        self._grid = grid
        self._boxes = tuple(
            (
                str(name),
                np.asarray(minimum, dtype=np.float64),
                np.asarray(maximum, dtype=np.float64),
            )
            for name, minimum, maximum in solid_boxes
        )
        self._samples = int(samples)
        self._endpoint_fraction = float(endpoint_fraction)

    @property
    def grid(self):
        return self._grid

    def __call__(self, origin: Sequence[float], focus: Sequence[float]) -> bool:
        start = np.asarray(origin, dtype=np.float64)
        end = np.asarray(focus, dtype=np.float64)
        direction = end - start
        if float(np.linalg.norm(direction)) <= 1e-8:
            return True
        if self._wall_blocks(start, direction):
            return False
        return not any(
            self._box_blocks(start, direction, minimum, maximum)
            for _name, minimum, maximum in self._boxes
        )

    def _wall_blocks(self, start: np.ndarray, direction: np.ndarray) -> bool:
        wall_sdf_at = getattr(self._grid, "wall_sdf_at", None)
        if not callable(wall_sdf_at):
            return False
        for index in range(1, self._samples):
            fraction = index / self._samples
            if fraction >= self._endpoint_fraction:
                break
            point = start + fraction * direction
            if float(wall_sdf_at(float(point[0]), float(point[1]))) < 0.0:
                return True
        return False

    def _box_blocks(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
    ) -> bool:
        end = start + direction
        target_slop = 0.02
        if np.all(end >= minimum - target_slop) and np.all(
            end <= maximum + target_slop
        ):
            return False
        enter = 0.0
        leave = 1.0
        for axis in range(3):
            delta = float(direction[axis])
            if abs(delta) <= 1e-9:
                if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                    return False
                continue
            near = float((minimum[axis] - start[axis]) / delta)
            far = float((maximum[axis] - start[axis]) / delta)
            if near > far:
                near, far = far, near
            enter = max(enter, near)
            leave = min(leave, far)
            if enter > leave:
                return False
        return leave > 0.02 and enter < self._endpoint_fraction


@dataclass(frozen=True)
class IsaacCameraAdapter:
    """Scene-level immutable inputs for the shared camera policy."""

    framing: camera.SceneFraming
    doorways: tuple[camera.Doorway, ...]
    focus_visibility: FocusVisibility

    @classmethod
    def from_scene(
        cls,
        scene,
        solid_boxes: Sequence[tuple[str, Sequence[float], Sequence[float]]] = (),
    ) -> IsaacCameraAdapter:
        grid = getattr(scene, "nav_grid", None) or getattr(scene, "wall_sdf_grid", None)
        doors = tuple(
            camera.Doorway((float(door.center_xy[0]), float(door.center_xy[1])))
            for door in (getattr(scene, "doors", None) or {}).values()
        )
        return cls(
            framing=camera.SceneFraming(floor_z=float(scene.floor_z)),
            doorways=doors,
            focus_visibility=FocusVisibility(grid, solid_boxes),
        )

    def with_solid_boxes(
        self,
        solid_boxes: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    ) -> IsaacCameraAdapter:
        """Return the same camera policy with current-frame visibility solids."""
        return replace(
            self,
            focus_visibility=FocusVisibility(
                self.focus_visibility.grid,
                tuple(solid_boxes),
            ),
        )

    def camera_frame(
        self,
        frame,
        joints,
        object_transform,
        *,
        dt_seconds: float,
        free_focus=None,
        physx_owned: bool = False,
    ) -> camera.CameraFrame:
        import humanoid_motion as wpp

        focus = frame_focus(
            frame,
            object_transform,
            free_focus=free_focus,
            physx_owned=physx_owned,
        )
        mode: camera.CameraMode
        if getattr(frame, "fpv_mode", "") == "walk" or focus is None:
            mode = "walk"
        elif (
            getattr(frame, "gaze_locked", False)
            or getattr(frame, "gaze", None) is not None
        ):
            mode = "gaze"
        else:
            mode = "manip"
        requested_zoom = float(getattr(frame, "zoom", None) or 0.0)
        camera_eye_offset = getattr(frame, "camera_eye_offset", None)
        if camera_eye_offset is None:
            camera_eye_offset = (0.0, 0.0, 0.0)
        return camera.CameraFrame(
            head_world=joints[wpp.H.HEAD_J],
            body_forward=wpp._body_forward(frame.yaw),
            mode=mode,
            dt_seconds=float(dt_seconds),
            focus_world=focus,
            focus_visible=(
                focus is not None
                and requested_zoom > 0.0
                and (
                    bool(getattr(frame, "focus_visibility_verified", False))
                    or self.focus_visibility(joints[wpp.H.HEAD_J], focus)
                )
            ),
            requested_zoom=requested_zoom,
            zoom_span=getattr(frame, "zoom_span", None),
            doorways=self.doorways,
            eye_offset_world=tuple(float(value) for value in camera_eye_offset),
        )
