"""Scene builders for the dynamic cup demo: an OFFICE environment + a SIMPLE-ROOM fallback.

Reuses the body-agnostic fpv_scene helpers (add_reference / set_world_pose / compute_world_bbox /
local_bbox_min_z). Two scenes:

  * OFFICE (the final target): references Environments/Office/office.usd (meter-scale, but a large
    multi-section stage whose main floor sits near z=0). We place two meter-scale office desks
    (SM_TableB.usd, top z=0.75) at the A and B positions on that floor.
  * SIMPLE-ROOM (the de-risk fallback for the first deliverable): references
    Environments/Simple_Room/simple_room.usd (floor z=0) + two cm-scale PackingTables
    (metersPerUnit=0.01 -> referenced at scale 0.01 so they come out meter-sized).

Both return a SceneLayout with the two table XY positions, the two table top-z values, and the
floor z, so the driver seats the body (feet at floor z) + the cup (base on a table top).

pxr is imported lazily (inside fpv_scene); importing this module before launch_app() is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd  # noqa: F401

import fpv_scene
from fpv_core import ASSET_ROOT

OFFICE_USD = ASSET_ROOT + "/Environments/Office/office.usd"
OFFICE_DESK_USD = ASSET_ROOT + "/Environments/Office/Props/SM_TableB.usd"
SIMPLE_ROOM_USD = ASSET_ROOT + "/Environments/Simple_Room/simple_room.usd"
PACKING_TABLE_USD = ASSET_ROOT + "/Props/PackingTable/packing_table.usd"
# NOTE on the PackingTable "cm-scale" flag: its layer metadata is metersPerUnit=0.01, but its raw
# geometry coordinates already describe a ~1 m table (world bbox top z~=1.083 at scale 1.0). USD does
# NOT auto-convert units across a reference, and the proven fpv_scene.add_table references it at
# scale 1.0 and gets a correct ~1 m table -- so we reference it UNSCALED too (scaling by 0.01 would
# shrink it to ~11 mm). The mpu flag is informational here, not a scale we must apply.
PACKING_TABLE_SCALE = 1.0


@dataclass
class SceneLayout:
    """Where the body + cup go: table A/B world XY, their top-z, and the floor z (feet rest)."""

    floor_z: float
    table_a_xy: tuple[float, float]
    table_b_xy: tuple[float, float]
    table_a_top_z: float
    table_b_top_z: float


def build_simple_room(
    stage: "Usd.Stage",
    table_a_xy: tuple[float, float] = (0.0, 0.0),
    table_b_xy: tuple[float, float] | None = (0.0, -2.0),
) -> SceneLayout:
    """Simple-room + PackingTable(s) (cm-scale -> referenced at 0.01). Floor z=0.

    Table A at the body's start XY; table B straight ahead along -Y (native forward) ONLY when
    ``table_b_xy`` is given. Pass ``table_b_xy=None`` (the single-table pick demo) to build ONLY table A
    -- a second co-located table is what cut through the body in the GRAB path. Returns the computed
    table-top z values (NOT the cm 1.08 default -- with scale 0.01 the top lands ~0.43 m).
    """
    room = fpv_scene.add_reference(stage, "/World/Room", SIMPLE_ROOM_USD)
    fpv_scene.set_world_pose(room, (0.0, 0.0, 0.0))

    top_a = _add_packing_table(stage, "TableA", table_a_xy)
    b_xy = table_b_xy if table_b_xy is not None else table_a_xy
    top_b = (
        _add_packing_table(stage, "TableB", table_b_xy)
        if table_b_xy is not None
        else top_a
    )
    print(
        f"[scene] simple-room: floor_z=0.0 tableA{table_a_xy} top={top_a:.3f} tableB{b_xy} top={top_b:.3f}"
        + ("" if table_b_xy is not None else "  (B skipped: single-table)")
    )
    return SceneLayout(0.0, table_a_xy, b_xy, top_a, top_b)


def _add_packing_table(stage: "Usd.Stage", name: str, xy: tuple[float, float]) -> float:
    """Reference packing_table.usd (unscaled ~1 m table), seat its base on the floor, return top-z."""
    from pxr import UsdGeom

    prim = fpv_scene.add_reference(stage, f"/World/{name}", PACKING_TABLE_USD)
    fpv_scene.set_world_pose(
        prim, (float(xy[0]), float(xy[1]), 0.0), scale=PACKING_TABLE_SCALE
    )
    # Seat the bbox bottom on the floor (z=0), then read the world-bbox top-z.
    min_z = fpv_scene.local_bbox_min_z(prim)
    fpv_scene.set_world_pose(
        prim, (float(xy[0]), float(xy[1]), -min_z), scale=PACKING_TABLE_SCALE
    )
    rng = _world_bbox_safe(stage, prim, UsdGeom)
    return float(rng.GetMax()[2])


def build_office(
    stage: "Usd.Stage",
    table_a_xy: tuple[float, float] = (0.0, 0.0),
    table_b_xy: tuple[float, float] | None = (0.0, -2.0),
) -> SceneLayout:
    """Office environment + meter-scale office desk(s) (SM_TableB.usd).

    DeskA is always placed at ``table_a_xy``. DeskB is placed at ``table_b_xy`` ONLY when it is given;
    pass ``table_b_xy=None`` (the single-table pick demo) to build ONLY DeskA. A SECOND desk co-located
    at the same XY -- which is what the GRAB path used to do -- left an un-repositioned 2.8 m SM_TableB
    slab spanning the body's footprint (the slab that cut through the legs). The office stage is large
    (z ~ -86..+53) but its main floor furniture sits near z=0; we assume the main floor is z=0.
    """
    office = fpv_scene.add_reference(stage, "/World/Office", OFFICE_USD)
    fpv_scene.set_world_pose(office, (0.0, 0.0, 0.0))

    floor_z = 0.0  # office main floor is at z~=0 (the multi-section stage's furniture sits there)
    top_a = _add_office_desk(stage, "DeskA", table_a_xy, floor_z)
    b_xy = table_b_xy if table_b_xy is not None else table_a_xy
    top_b = (
        _add_office_desk(stage, "DeskB", table_b_xy, floor_z)
        if table_b_xy is not None
        else top_a
    )
    _add_office_lights(stage, table_a_xy, b_xy)
    print(
        f"[scene] office: floor_z={floor_z:.2f} deskA{table_a_xy} top={top_a:.3f} deskB{b_xy} top={top_b:.3f}"
        + ("" if table_b_xy is not None else "  (B skipped: single-table)")
    )
    return SceneLayout(floor_z, table_a_xy, b_xy, top_a, top_b)


def _add_office_lights(
    stage: "Usd.Stage", ta_xy: tuple[float, float], tb_xy: tuple[float, float]
) -> None:
    """The office is an ENCLOSED interior, so the scene DomeLight is occluded by its ceiling and the
    desk/cup fall into deep shadow. Add bright downward RectLights just under the ceiling, spanning the
    action (approach -> table A -> table B), so the workspace is lit like the office_walkthrough."""
    from pxr import UsdLux

    midy = 0.5 * (ta_xy[1] + tb_xy[1])
    pts = [(ta_xy[0], ta_xy[1] + 1.0), (ta_xy[0], midy), (tb_xy[0], tb_xy[1] - 0.2)]
    for i, (x, y) in enumerate(pts):
        lt = UsdLux.RectLight.Define(stage, f"/World/OfficeLights/L{i}")
        lt.GetIntensityAttr().Set(11000.0)
        lt.GetWidthAttr().Set(3.5)
        lt.GetHeightAttr().Set(3.5)
        lt.CreateColorAttr((1.0, 0.97, 0.92))
        fpv_scene.set_world_pose(
            lt.GetPrim(), (float(x), float(y), 2.6)
        )  # identity -> emits down (-Z)


def _add_office_desk(
    stage: "Usd.Stage", name: str, xy: tuple[float, float], floor_z: float
) -> float:
    """Reference SM_TableB.usd (meter-scale, base-pivot, top 0.75) at xy on the floor; return top-z."""
    from pxr import UsdGeom

    prim = fpv_scene.add_reference(stage, f"/World/{name}", OFFICE_DESK_USD)
    fpv_scene.set_world_pose(prim, (float(xy[0]), float(xy[1]), floor_z))
    min_z = fpv_scene.local_bbox_min_z(prim)
    fpv_scene.set_world_pose(prim, (float(xy[0]), float(xy[1]), floor_z - min_z))
    rng = _world_bbox_safe(stage, prim, UsdGeom)
    return float(rng.GetMax()[2])


def _world_bbox_safe(stage: "Usd.Stage", prim: "Usd.Prim", UsdGeom) -> object:
    """World-aligned bbox of a prim (purposed for default+render). Thin wrapper over fpv_scene."""
    return fpv_scene.compute_world_bbox(prim)
