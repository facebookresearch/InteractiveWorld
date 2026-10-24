# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""collision_world.py -- the SINGLE precise-geometry source of truth (SSOT oracle).

WHY THIS EXISTS. Without a shared source, the FPV benchmark represents ONE piece of
furniture with 6+ incompatible geometries: PhysX/clearance uses the real per-link
convex-hull collider, nav a Minkowski-inflated world AABB, the stand generator the
TARGET's own visual foot_half AABB (no z, neighbour-blind), reach/nudge a floor->top
SOLID pillar, the ``--check`` gate a hardcoded prim-path AABB, and the goal is purely
symbolic. Those pictures disagree, so one subsystem picks a spot another judges
penetrating -- they fight, and the referee is blind.

WHAT THIS DOES. ``CollisionWorld`` is built ONCE per episode from the SAME real PhysX
collider set the init non-penetration gate and the object settle already use
(``init_validation.collidable_footprints`` over every ``UsdPhysics.CollisionAPI`` prim
= OG per-link convexHull + Isaac-native colliders), and every consumer AND the
post-action physics validator reads geometry from THIS ONE object. No consumer derives
its own AABB from a per-target visual bbox.

STRICTLY ADDITIVE + GATED. Nothing reads this unless ``scene.unified_geom`` is set; the
object is only built on the seeded BEHAVIOR path. Non-BEHAVIOR / hand-tuned scenes never
construct it, so they are byte-identical.

Design (matching ``physics_human`` / ``object_physics``): ``from __future__ import
annotations`` + LAZY pxr/isaacsim imports inside every method, so importing this module
before ``launch_app()`` never pulls in Kit.
"""

from __future__ import annotations

import logging

import numpy as np

LOG: logging.Logger = logging.getLogger("collision_world")

# An owner whose collider top sits within this of the floor is floor-level trim
# (thresholds/rugs); it is a valid SUPPORT but never a human OBSTACLE.
_FLOOR_TOL_M: float = 0.05


class CollisionWorld:
    """One precise-geometry view of the scene, shared by every subsystem + the validator.

    Holds the real-collider footprint set (``(owner, cx, cy, hx, hy, bottom_z, top_z)``
    per obstacle, derived from every ``UsdPhysics.CollisionAPI`` prim's world bbox) plus
    a live ``ph`` handle for the exact PhysX queries (human overlap, contact separation).
    The footprint z-range is REAL (``bottom_z..top_z``), so the space UNDER a wall cabinet
    is free -- unlike a floor->top solid pillar.
    """

    def __init__(
        self,
        footprints: list[tuple[str, float, float, float, float, float, float]],
        floor_z: float,
        scene_paths: list[str],
        ph,
        stage,
    ) -> None:
        self.footprints = footprints
        self.by_owner: dict[
            str, tuple[str, float, float, float, float, float, float]
        ] = {f[0]: f for f in footprints}
        self.floor_z = float(floor_z)
        # Non-floor obstacle owner paths = the human-vs-scene overlap obstacle set
        # (identical to the init gate's set, so gate and generator agree).
        self.scene_paths = list(scene_paths)
        self._ph = ph
        self._stage = stage
        # cworld OWNS the UNDILATED precise nav triangle raster (built lazily by build_nav_grid on
        # first use, keyed to the nav-grid shape) so nav AND the stand search read ONE precise map.
        self._precise_occ = None

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, stage, scene, ph) -> "CollisionWorld":
        """Construct from the SAME real collider set the init gate uses. Cache on the caller."""
        import init_validation as iv

        floor_z = float(getattr(scene, "floor_z", 0.0))
        fps = iv.collidable_footprints(stage, group_by_owner=False)
        scene_paths = iv._non_floor_paths(fps, floor_z=floor_z)
        cw = cls(fps, floor_z, scene_paths, ph, stage)
        LOG.info(
            "CollisionWorld: %d colliders, %d non-floor obstacles, floor_z=%.3f",
            len(fps),
            len(scene_paths),
            floor_z,
        )
        return cw

    def refresh(self, scene) -> None:
        """Re-read live collider footprints after a movable fixture changes pose."""
        fresh = type(self).build(self._stage, scene, self._ph)
        self.footprints = fresh.footprints
        self.by_owner = fresh.by_owner
        self.scene_paths = fresh.scene_paths
        self._precise_occ = None

    @staticmethod
    def _without_owners(paths: list[str], exclude_owners: tuple[str, ...]) -> list[str]:
        excluded = tuple(str(owner).rstrip("/") for owner in exclude_owners)
        if not excluded:
            return paths
        return [
            path
            for path in paths
            if not any(path == root or path.startswith(root + "/") for root in excluded)
        ]

    # -------------------------------------------------- human vs scene (gate)

    def human_overlap(
        self,
        J: np.ndarray,
        joint_subset: tuple[int, ...] | None = None,
        exclude_sphere: tuple[np.ndarray, float] | None = None,
        exclude_owners: tuple[str, ...] = (),
    ) -> list[tuple[str, str]]:
        """The AUTHORITATIVE lag-free human-vs-scene penetration query (PhysX overlap sphere).

        Same source as the init non-pen gate, so a pose the gate accepts and a pose the
        swept validator accepts are decided by ONE query over the SAME obstacle set.
        """
        import init_validation as iv

        js = joint_subset if joint_subset is not None else iv._CORE_J
        paths = self._without_owners(self.scene_paths, exclude_owners)
        return self._ph.human_scene_overlap(
            self._stage,
            J,
            paths,
            joint_subset=js,
            exclude_sphere=exclude_sphere,
        )

    def human_depth(
        self,
        J: np.ndarray,
        joint_subset: tuple[int, ...] | None = None,
        exclude_owners: tuple[str, ...] = (),
    ) -> tuple[float, str, bool]:
        """Worst capsule penetration depth (m) + description + SATURATION flag for this pose, on the
        SAME obstacle set as ``human_overlap`` -- so the depth always explains the boolean.

        The depth saturates at the capsule radius (see ``physics_human.human_scene_depth``); when the
        third element is True the value is a LOWER BOUND, never a maximum.
        """
        import init_validation as iv

        js = joint_subset if joint_subset is not None else iv._CORE_J
        paths = self._without_owners(self.scene_paths, exclude_owners)
        return self._ph.human_scene_depth(self._stage, J, paths, joint_subset=js)

    def human_exceeds_depth(
        self,
        J: np.ndarray,
        allowed_m: float,
        joint_subset: tuple[int, ...] | None = None,
        exclude_owners: tuple[str, ...] = (),
    ) -> bool:
        """Cheap early-exit form of ``human_depth`` for planner threshold gates."""
        import init_validation as iv

        js = joint_subset if joint_subset is not None else iv._CORE_J
        paths = self._without_owners(self.scene_paths, exclude_owners)
        return self._ph.human_scene_exceeds_depth(
            self._stage,
            J,
            paths,
            allowed_m,
            joint_subset=js,
        )

    def sphere_overlap(
        self,
        center: np.ndarray,
        radius: float,
        exclude_owners: tuple[str, ...] = (),
    ) -> str | None:
        """Return the first scene owner intersecting a world-space probe sphere."""
        paths = self._without_owners(self.scene_paths, exclude_owners)
        return self._ph.scene_sphere_overlap(
            self._stage,
            center,
            radius,
            paths,
        )

    # ------------------------------------------------ object vs object (gate)

    def object_penetration(
        self, prim_paths: list[str], sep_tol: float = 0.002
    ) -> list[tuple[str, str, float]]:
        """Every object-object pair whose contact separation is < -sep_tol (real PhysX contacts)."""
        import object_physics as ophys

        return ophys.all_pairs_separation(
            self._ph, self._stage, list(prim_paths), sep_tol=sep_tol
        )

    # ---------------------------------------------------- support geometry

    def _covers_xy(
        self,
        fp: tuple[str, float, float, float, float, float, float],
        x: float,
        y: float,
        pad: float,
    ) -> bool:
        _, cx, cy, hx, hy, _, _ = fp
        return abs(x - cx) <= hx + pad and abs(y - cy) <= hy + pad

    def support_top_z(
        self,
        xy: tuple[float, float],
        exclude_owners: tuple[str, ...] = (),
        include_owners: tuple[str, ...] = (),
        max_z: float | None = None,
        pad: float = 0.0,
    ) -> float:
        """REAL collider top under ``xy`` -- the SUPPORT's measured top, never the moved object's own.

        Returns the highest ``top_z`` over every selected collider whose footprint covers
        ``xy``. ``exclude_owners`` removes moving objects; non-empty ``include_owners``
        restricts the query to a declared support subtree. Supports above ``max_z`` are
        ignored so a wall cabinet overhead is not mistaken for the shelf below. Falls back
        to ``floor_z`` -- the floor is always a valid support.
        """
        x, y = float(xy[0]), float(xy[1])
        excl = tuple(str(owner).rstrip("/") for owner in (exclude_owners or ()))
        incl = tuple(str(owner).rstrip("/") for owner in (include_owners or ()))
        best = self.floor_z
        for fp in self.footprints:
            owner, _, _, _, _, _, tz = fp
            if any(owner == root or owner.startswith(root + "/") for root in excl):
                continue
            if incl and not any(
                owner == root or owner.startswith(root + "/") for root in incl
            ):
                continue
            if max_z is not None and tz > float(max_z) + 1e-6:
                continue
            if self._covers_xy(fp, x, y, pad) and tz > best:
                best = float(tz)
        return best

    def footprint(
        self, owner: str
    ) -> tuple[str, float, float, float, float, float, float] | None:
        """The real-collider footprint tuple for ``owner`` (or None if it has no collider)."""
        return self.by_owner.get(owner)

    def boxes(
        self,
        z_lo: float | None = None,
        z_hi: float | None = None,
        exclude_owners: tuple[str, ...] = (),
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Real per-owner boxes ``(name, mn, mx)`` with the REAL ``bottom_z..top_z`` z-range.

        Drop-in replacement for ``scene_state.solid_boxes`` (reach/nudge), but the box uses
        the collider's real bottom (so the space under a wall cabinet is FREE), not a
        floor->top solid pillar. Optional ``[z_lo, z_hi]`` keeps only owners overlapping
        that band (e.g. the human pass band for the nudge).
        """
        excl = tuple(str(owner).rstrip("/") for owner in (exclude_owners or ()))
        out: list[tuple[str, np.ndarray, np.ndarray]] = []
        for owner, cx, cy, hx, hy, bz, tz in self.footprints:
            if any(owner == root or owner.startswith(root + "/") for root in excl):
                continue
            if tz - self.floor_z <= _FLOOR_TOL_M:
                continue  # floor / floor-level trim is a SUPPORT, never a reach/nudge obstacle
            if z_hi is not None and bz > float(z_hi):
                continue
            if z_lo is not None and tz < float(z_lo):
                continue
            mn = np.array([cx - hx, cy - hy, bz], dtype=np.float64)
            mx = np.array([cx + hx, cy + hy, tz], dtype=np.float64)
            out.append((owner, mn, mx))
        return out

    def neighbors_between(
        self,
        a_xy: tuple[float, float],
        b_xy: tuple[float, float],
        pad: float = 0.30,
        exclude_owners: tuple[str, ...] = (),
        z_lo: float | None = None,
        z_hi: float | None = None,
    ) -> list[str]:
        """Owners whose footprint (expanded by ``pad``) straddles the a->b corridor.

        Lets the stand search back off from the RIGHT obstacle: the piece of furniture
        actually BETWEEN the stand and the contact, not just the target's own footprint.
        Optional ``[z_lo, z_hi]`` restricts to obstacles in the human pass band.
        """
        ax, ay = float(a_xy[0]), float(a_xy[1])
        bx, by = float(b_xy[0]), float(b_xy[1])
        excl = set(exclude_owners or ())
        n = 12
        pts = [(ax + (bx - ax) * t / n, ay + (by - ay) * t / n) for t in range(n + 1)]
        hit: list[str] = []
        for fp in self.footprints:
            owner, _, _, _, _, bz, tz = fp
            if owner in excl:
                continue
            if z_hi is not None and bz > float(z_hi):
                continue
            if z_lo is not None and tz < float(z_lo):
                continue
            if any(self._covers_xy(fp, px, py, pad) for px, py in pts):
                hit.append(owner)
        return hit


# ------------------------------------------------------------------ selftest


def _selftest() -> None:
    """Pure-geometry checks (NO Isaac boot): support_top_z, boxes z-range, neighbors_between."""
    logging.basicConfig(level=logging.INFO)
    fps = [
        ("/World/Floor", 0.0, 0.0, 10.0, 10.0, -0.02, 0.0),
        ("/World/Table", 1.0, 0.0, 0.5, 0.4, 0.0, 0.75),
        ("/World/WallCab", 1.0, 2.0, 0.4, 0.2, 1.40, 2.00),
        ("/World/Chair", -1.5, 0.0, 0.25, 0.25, 0.0, 0.45),
    ]
    cw = CollisionWorld(
        fps, floor_z=0.0, scene_paths=[f[0] for f in fps[1:]], ph=None, stage=None
    )

    # support under the table = table top; excluding the table => floor.
    assert abs(cw.support_top_z((1.0, 0.0)) - 0.75) < 1e-9, cw.support_top_z((1.0, 0.0))
    assert (
        abs(cw.support_top_z((1.0, 0.0), exclude_owners=("/World/Table",)) - 0.0) < 1e-9
    )
    assert (
        abs(cw.support_top_z((1.0, 0.0), include_owners=("/World/Table",)) - 0.75)
        < 1e-9
    )
    assert (
        abs(cw.support_top_z((1.0, 0.0), include_owners=("/World/WallCab",)) - 0.0)
        < 1e-9
    )
    # a drop capped below the wall cabinet must NOT pick the cabinet top.
    assert abs(cw.support_top_z((1.0, 2.0), max_z=1.0) - 0.0) < 1e-9
    assert abs(cw.support_top_z((1.0, 2.0)) - 2.0) < 1e-9
    # empty air over the floor => floor_z.
    assert abs(cw.support_top_z((5.0, 5.0)) - 0.0) < 1e-9

    # boxes carry the REAL bottom_z (wall cabinet bottom = 1.40, not the floor).
    wc = next(b for b in cw.boxes() if b[0] == "/World/WallCab")
    assert abs(float(wc[1][2]) - 1.40) < 1e-9, wc
    # the human pass band [0.2, 1.9] excludes the floor trim and includes table+cabinet.
    band = {b[0] for b in cw.boxes(z_lo=0.2, z_hi=1.9)}
    assert (
        "/World/Table" in band
        and "/World/WallCab" in band
        and "/World/Floor" not in band
    ), band

    # a stand at (-0.2,0) reaching the table at (1.0,0): the chair is not between; nothing else is.
    nb = cw.neighbors_between(
        (-0.2, 0.0), (1.0, 0.0), pad=0.1, exclude_owners=("/World/Table",)
    )
    assert "/World/Chair" not in nb, nb
    # walking THROUGH the chair corridor flags it.
    nb2 = cw.neighbors_between((-3.0, 0.0), (0.0, 0.0), pad=0.1)
    assert "/World/Chair" in nb2, nb2

    print("[collision_world] _selftest PASS")


if __name__ == "__main__":
    _selftest()
