# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Shared dense-frame replay for kinematic clutter settle trajectories."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence


DEFAULT_REPLAY_FRAMES = 18


def _normalize_path(path: object) -> str:
    normalized = str(path).rstrip("/")
    return normalized or "/"


def _clutter_path(item: object) -> str:
    if isinstance(item, (tuple, list)):
        if not item:
            return ""
        item = item[0]
    return _normalize_path(item)


class ClutterRenderReplay:
    """Author a recorded settle trajectory from absolute dense-frame indices."""

    def __init__(
        self,
        stage,
        trajectory: Sequence[Mapping[object, object]] | None,
        clutter: Iterable[object] | None,
        excluded_paths: Iterable[object] = (),
        *,
        replay_frames: int = DEFAULT_REPLAY_FRAMES,
        pose_writer: Callable[[object, object, object], None] | None = None,
    ) -> None:
        if replay_frames < 1:
            raise ValueError("replay_frames must be positive")
        self._stage = stage
        self._replay_frames = int(replay_frames)
        self._trajectory = tuple(
            {_normalize_path(path): pose for path, pose in trajectory_frame.items()}
            for trajectory_frame in (trajectory or ())
        )
        excluded = {_normalize_path(path) for path in excluded_paths}
        declared = {path for item in (clutter or ()) if (path := _clutter_path(item))}
        self._prims = {
            path: stage.GetPrimAtPath(path)
            for path in declared - excluded
            if stage is not None
        }
        self._pose_writer = pose_writer
        self.last_global_frame: int | None = None

    @property
    def active_paths(self) -> tuple[str, ...]:
        """Kinematic paths owned by this replay, excluding every PhysX-owned path."""
        return tuple(sorted(self._prims))

    def reset(self) -> None:
        """Start a new episode whose dense trajectory begins again at global frame zero."""
        self.last_global_frame = None

    def snapshot_state(self) -> dict[str, object]:
        """Capture replay-owned clutter so rendering remains a pure observer."""
        if self._stage is None or not self._prims:
            return {}
        import dynamic_render_replay

        return dynamic_render_replay.snapshot_dynamic_prims(
            self._stage, self.active_paths
        )

    def restore_state(self, state: dict[str, object]) -> None:
        """Restore clutter after a render-only trajectory replay."""
        if self._stage is None or not state:
            return
        import dynamic_render_replay

        dynamic_render_replay.restore_dynamic_prims(self._stage, state)

    def trajectory_index(self, global_frame: int) -> int | None:
        """Map an absolute dense-frame index onto the recorded settle trajectory."""
        frame = int(global_frame)
        if frame < 0:
            raise ValueError("global_frame must be non-negative")
        if not self._trajectory:
            return None
        if self._replay_frames == 1:
            return len(self._trajectory) - 1
        visible_frame = min(frame, self._replay_frames - 1)
        index = int(
            round(
                visible_frame / (self._replay_frames - 1) * (len(self._trajectory) - 1)
            )
        )
        return max(0, min(len(self._trajectory) - 1, index))

    def apply(self, global_frame: int) -> int | None:
        """Author the clutter pose for one absolute dense frame and return its source index."""
        index = self.trajectory_index(global_frame)
        self.last_global_frame = int(global_frame)
        if index is None:
            return None
        writer = self._pose_writer
        for path, pose in self._trajectory[index].items():
            prim = self._prims.get(path)
            if prim is None or not prim.IsValid():
                continue
            if writer is None:
                import fpv_scene

                writer = fpv_scene.set_world_pose
                self._pose_writer = writer
            position, orientation = pose
            writer(prim, position, orientation)
        return index


def make_clutter_render_replay(
    context: Mapping[str, object],
    *,
    replay_frames: int = DEFAULT_REPLAY_FRAMES,
    pose_writer: Callable[[object, object, object], None] | None = None,
) -> ClutterRenderReplay:
    """Build the renderer-independent clutter replay from an episode context."""
    return ClutterRenderReplay(
        context.get("stage"),
        context.get("clutter_traj"),
        context.get("scene_clutter"),
        context.get("fdp_paths") or (),
        replay_frames=replay_frames,
        pose_writer=pose_writer,
    )
