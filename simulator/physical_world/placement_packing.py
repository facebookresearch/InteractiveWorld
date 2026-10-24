# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pure geometry shared by runtime placement and build-time capacity probes."""

from __future__ import annotations

import math
from collections.abc import Sequence


MIN_GAP_M = 0.02
EDGE_MARGIN_M = 0.02


def stable_overhang(half_extent: float) -> float:
    return min(0.05, 0.20 * float(half_extent))


def fit_yaw(
    object_half: tuple[float, float],
    target_half: tuple[float, float],
    *,
    allow_quarter_turn: bool,
) -> float | None:
    def fits(candidate: tuple[float, float]) -> bool:
        return bool(
            candidate[0] <= target_half[0] + stable_overhang(candidate[0]) + 1e-6
            and candidate[1] <= target_half[1] + stable_overhang(candidate[1]) + 1e-6
        )

    if fits(object_half):
        return 0.0
    if allow_quarter_turn and fits((object_half[1], object_half[0])):
        return math.pi / 2.0
    return None


def placement_candidates(
    center: tuple[float, float],
    radius: tuple[float, float],
    object_half: tuple[float, float],
    occupied: Sequence[tuple[float, float, float, float]],
    *,
    min_gap: float = 0.0,
) -> list[tuple[float, float]]:
    """Finite set complete for axis-aligned rectangle contacts and target edges."""
    cx, cy = center
    rx, ry = radius
    ohx, ohy = object_half
    fractions = (-1.0, -0.85, -0.567, -0.283, 0.0, 0.283, 0.567, 0.85, 1.0)
    xs = [cx + fraction * rx for fraction in fractions]
    ys = [cy + fraction * ry for fraction in fractions]
    for qx, qy, qhx, qhy in occupied:
        xs.extend((qx - qhx - ohx - min_gap, qx + qhx + ohx + min_gap))
        ys.extend((qy - qhy - ohy - min_gap, qy + qhy + ohy + min_gap))

    def axis(values: Sequence[float], axis_center: float, axis_radius: float):
        lower, upper = axis_center - axis_radius, axis_center + axis_radius
        unique: dict[float, float] = {}
        for value in values:
            if lower - 1e-9 <= value <= upper + 1e-9:
                unique.setdefault(round(float(value), 6), float(value))
        return list(unique.values())

    return [(x, y) for x in axis(xs, cx, rx) for y in axis(ys, cy, ry)]


def candidate_set(
    center: tuple[float, float],
    target_half: tuple[float, float],
    object_half: tuple[float, float],
    occupied: Sequence[tuple[float, float, float, float]],
    rejected: Sequence[tuple[float, float]],
    *,
    edge_margin: float = EDGE_MARGIN_M,
    min_gap: float = MIN_GAP_M,
    preferred: Sequence[tuple[float, float]] = (),
    pile_anchors: Sequence[tuple[float, float, float, float]] = (),
) -> tuple[list[tuple[float, float]], float, float]:
    cx, cy = center
    hx, hy = target_half
    ohx, ohy = object_half
    free_x, free_y = max(0.0, hx - ohx), max(0.0, hy - ohy)
    edge_x = min(float(edge_margin), free_x)
    edge_y = min(float(edge_margin), free_y)
    rx, ry = free_x - edge_x, free_y - edge_y
    candidates = placement_candidates(
        center,
        (rx, ry),
        object_half,
        occupied,
        min_gap=min_gap,
    )
    candidates = list(
        dict.fromkeys(
            [
                (float(point[0]), float(point[1]))
                for point in preferred
                if len(point) >= 2
            ]
            + [
                (float(row[0]), float(row[1]))
                for row in pile_anchors
                if abs(float(row[0]) - cx) <= rx + 1e-9
                and abs(float(row[1]) - cy) <= ry + 1e-9
            ]
            + candidates
        )
    )

    def is_rejected(point: tuple[float, float]) -> bool:
        return any(
            abs(point[0] - float(row[0])) < 2.0 * ohx + min_gap
            and abs(point[1] - float(row[1])) < 2.0 * ohy + min_gap
            for row in rejected
        )

    return [point for point in candidates if not is_rejected(point)], rx, ry


def order_candidates(
    candidates: list[tuple[float, float]],
    center: tuple[float, float],
    radius: tuple[float, float],
    root_xy: tuple[float, float],
    deepest: bool,
    nearest: bool = False,
    *,
    reserve_space: bool = False,
) -> None:
    cx, cy = center
    if reserve_space:
        px, py = float(root_xy[0]), float(root_xy[1])
        candidates.sort(
            key=lambda point: (
                -((point[0] - cx) ** 2 + (point[1] - cy) ** 2),
                -((point[0] - px) ** 2 + (point[1] - py) ** 2),
            )
        )
        return
    if nearest:
        px, py = float(root_xy[0]), float(root_xy[1])
        candidates.sort(
            key=lambda point: (
                (point[0] - px) ** 2 + (point[1] - py) ** 2,
                (point[0] - cx) ** 2 + (point[1] - cy) ** 2,
            )
        )
        return
    if not deepest:
        candidates.sort(key=lambda point: (point[0] - cx) ** 2 + (point[1] - cy) ** 2)
        return
    rx, ry = radius
    px, py = float(root_xy[0]), float(root_xy[1])

    def deep_key(point: tuple[float, float]):
        nx = abs(point[0] - cx) / rx if rx > 1e-9 else 0.0
        ny = abs(point[1] - cy) / ry if ry > 1e-9 else 0.0
        edge_only = max(nx, ny) > 0.90
        distance = (point[0] - px) ** 2 + (point[1] - py) ** 2
        return edge_only, -distance

    candidates.sort(key=deep_key)


def strict_pack_sites(
    center: tuple[float, float],
    target_half: tuple[float, float],
    object_half: tuple[float, float],
    count: int,
    root_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    """Run the runtime's strict open-top packing ladder without reach/physics filters."""
    if fit_yaw(object_half, target_half, allow_quarter_turn=False) is None:
        return []
    occupied: list[tuple[float, float, float, float]] = []
    selected: list[tuple[float, float]] = []
    ohx, ohy = object_half
    for _index in range(count):
        reserve_space = count > len(occupied) + 1
        if reserve_space:
            margins = (0.0,)
        elif occupied:
            margins = (EDGE_MARGIN_M, 0.01, 0.004, 0.0)
        else:
            margins = (EDGE_MARGIN_M,)
        gaps = (MIN_GAP_M, 0.01, 0.004, 0.0) if occupied else (MIN_GAP_M,)
        chosen = None
        for min_gap in gaps:
            for edge_margin in margins:
                candidates, radius_x, radius_y = candidate_set(
                    center,
                    target_half,
                    object_half,
                    occupied,
                    (),
                    edge_margin=edge_margin,
                    min_gap=min_gap,
                )
                order_candidates(
                    candidates,
                    center,
                    (radius_x, radius_y),
                    root_xy,
                    True,
                    reserve_space=reserve_space,
                )
                chosen = next(
                    (
                        point
                        for point in candidates
                        if not any(
                            abs(point[0] - qx) < ohx + qhx + min_gap
                            and abs(point[1] - qy) < ohy + qhy + min_gap
                            for qx, qy, qhx, qhy in occupied
                        )
                    ),
                    None,
                )
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is None:
            break
        occupied.append((chosen[0], chosen[1], ohx, ohy))
        selected.append(chosen)
    return selected
