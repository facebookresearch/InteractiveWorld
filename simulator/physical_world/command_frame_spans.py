# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Authoritative source-command boundaries over an append-only Frame timeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class CompiledCommandRange(Protocol):
    index: int
    goal_start: int
    goal_end: int


class CommandFrameSpanTracker:
    """Record one Frame span per source command while Goals execute in order."""

    def __init__(self, commands: Sequence[CompiledCommandRange]) -> None:
        self._commands = tuple(commands)
        self._goal_to_command = self._validate_commands()
        self._spans: list[tuple[int, int]] = []
        self._active_command: int | None = None
        self._active_start = 0
        self._next_goal = 0
        self._last_frame_count = 0
        self._stopped = False
        self._finished = False

    def _validate_commands(self) -> tuple[int, ...]:
        goal_to_command = []
        expected_goal = 0
        for expected_index, command in enumerate(self._commands):
            if command.index != expected_index:
                raise ValueError("compiled command indexes must be contiguous")
            if command.goal_start != expected_goal or command.goal_end < expected_goal:
                raise ValueError("compiled command Goal ranges must be contiguous")
            goal_to_command.extend(
                [command.index] * (command.goal_end - command.goal_start)
            )
            expected_goal = command.goal_end
        return tuple(goal_to_command)

    def _check_frame_count(self, frame_count: int) -> None:
        if frame_count < self._last_frame_count:
            raise ValueError("Frame count must be append-only")
        self._last_frame_count = frame_count

    def _close_active(self, frame_count: int) -> None:
        if self._active_command is None:
            return
        self._spans.append((self._active_start, frame_count))
        self._active_command = None

    def _record_empty_before(self, command_index: int, frame_count: int) -> None:
        while len(self._spans) < command_index:
            command = self._commands[len(self._spans)]
            if command.goal_start != command.goal_end:
                raise RuntimeError("a non-empty source command was skipped")
            self._spans.append((frame_count, frame_count))

    def begin_goal(
        self, goal_index: int, frame_count: int, *, frame_limit: int = 0
    ) -> int | None:
        """Begin a Goal, or return None when a source-command boundary hits the cap."""

        if self._finished or self._stopped:
            raise RuntimeError("command span tracking is no longer active")
        if goal_index != self._next_goal or goal_index >= len(self._goal_to_command):
            raise ValueError("Goals must begin once in chronological order")
        if frame_limit < 0:
            raise ValueError("frame_limit must be non-negative")
        self._check_frame_count(frame_count)
        command_index = self._goal_to_command[goal_index]
        if command_index != self._active_command:
            self._close_active(frame_count)
            if frame_limit and frame_count >= frame_limit:
                self._stopped = True
                return None
            self._record_empty_before(command_index, frame_count)
            self._active_command = command_index
            self._active_start = frame_count
        self._next_goal += 1
        return command_index

    def finish(
        self, frame_count: int, *, frame_limit: int = 0
    ) -> tuple[tuple[int, int], ...]:
        """Close the last executed command and preserve trailing zero-Goal commands."""

        if self._finished:
            return tuple(self._spans)
        if frame_limit < 0:
            raise ValueError("frame_limit must be non-negative")
        self._check_frame_count(frame_count)
        self._close_active(frame_count)
        cap_reached = bool(frame_limit and frame_count >= frame_limit)
        if not self._stopped and not cap_reached:
            if self._next_goal != len(self._goal_to_command):
                raise RuntimeError("not every compiled Goal was executed")
            self._record_empty_before(len(self._commands), frame_count)
        elif not self._stopped and self._next_goal != len(self._goal_to_command):
            raise RuntimeError("not every compiled Goal was executed")
        self._finished = True
        return tuple(self._spans)
