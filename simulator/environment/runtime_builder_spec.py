# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Strict in-process installation of a dynamic TaskPack builder spec."""

from __future__ import annotations

from collections.abc import Mapping


def register_runtime_builder_spec(
    scene: str,
    builder_spec: Mapping[str, object] | None,
    builder_spec_digest: str | None,
) -> str | None:
    """Verify and install the dynamic spec for ``scene``, or clear it for a disk scene."""
    from simulator.physical_world import task_world_builder

    if builder_spec is None:
        if builder_spec_digest is not None:
            raise ValueError("builder_spec_digest was provided without builder_spec")
        task_world_builder.clear_runtime_spec()
        return None
    if not isinstance(builder_spec_digest, str) or not builder_spec_digest:
        raise ValueError("builder_spec requires a non-empty builder_spec_digest")

    from simulator.taskpack_support.runtime_adapter import (
        canonical_digest,
        verify_runtime_view,
    )

    verify_runtime_view(builder_spec)
    task_id = str(builder_spec.get("task_id") or "")
    expected_scene = f"spec:{task_id}"
    if not task_id or scene != expected_scene:
        raise ValueError(
            f"builder_spec task_id/scene mismatch: task_id={task_id!r}, scene={scene!r}"
        )
    actual_digest = canonical_digest(builder_spec)
    if actual_digest != builder_spec_digest:
        raise ValueError(
            "builder_spec digest mismatch: "
            f"expected {builder_spec_digest}, got {actual_digest}"
        )
    return task_world_builder.register_runtime_spec(dict(builder_spec), actual_digest)
