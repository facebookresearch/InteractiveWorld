"""General, reusable PhysX plumbing for a SMPL+H human in Isaac Sim (Milestone 0).

This module is the REUSABLE physics/collision/grasp layer that later milestones (mocap
retarget) bolt onto the existing kinematic SMPL+H pipeline (simulator.physical_world/smplh_anim_core.py +
hand/smplh_fpv.py). It owns NOTHING per-object or per-scene: every size derives from the
body skeleton / the object's own geometry. It does not replace the visible skinned mesh --
PhysX collides an INVISIBLE kinematic capsule proxy rig that follows the joints the pipeline
already computes (the per-joint world positions ``J`` returned by ``smplh_anim_core.pose_body``).

What it provides (all pip / pxr / omni.physx -- NO Isaac Sim source build, NO data download):

  * ``build_physics_scene(stage)``         -- UsdPhysics.Scene + gravity (-Z, 9.81).
  * ``make_static_collider(prim)``         -- table: CollisionAPI (+ convexDecomposition on child meshes).
  * ``make_dynamic_rigid(prim, density)``  -- cup: RigidBodyAPI + CollisionAPI + MassAPI (convexHull).
  * ``build_proxy_rig(stage, J, ...)``     -- one invisible kinematic capsule per major bone (sphere=head),
                                              sized per body region, with FilteredPairsAPI between adjacent
                                              bones. The HAND bone is a real rigid body (FixedJoint anchor).
  * ``update_proxy_rig(proxy, J)``         -- per frame: write each capsule's world xform from the current J.
  * ``human_scene_collision_pairs(stage)`` -- the penetration detector: get_initial_collider_pairs(stage)
                                              filtered to human-vs-scene (exactly one path under the human root).
  * ``grasp_weld`` / ``release_weld``      -- create/delete a UsdPhysics.FixedJoint hand-proxy <-> object.
  * ``settle(...)``                        -- step physics (render=False) until the object's |v| < thresh.

Design notes honored throughout:
  * ``from __future__ import annotations`` + LAZY pxr/omni imports inside each function, so importing
    this module before ``launch_app()`` never pulls in Kit (the proven SimulationApp-first ordering).
  * The bone set + radii come from the SMPL+H kinematic tree (parent->child joint segments), not constants
    tied to any one behavior: a "bone" is the segment ``J[a] -> J[b]`` for a parent/child joint pair.
  * A capsule's default local axis in USD is +Z (UsdGeom.Capsule), so we author the per-frame transform
    that maps local +Z onto the world bone direction and the origin onto the bone midpoint.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd  # noqa: F401

LOG = logging.getLogger("physics_human")

# Small manipulated props are released only a few centimetres above their support.  PhysX's default
# rigid-body settings are tuned for unrestricted simulation: a tiny initial overlap can therefore eject
# a prop quickly, and a round prop can keep rolling for a long time because Coulomb friction does not
# dissipate pure rolling.  These bounded values model the missing rolling/contact losses without freezing
# bodies or depending on an asset category.  They apply to every dynamic prop, including BEHAVIOR assets.
_DYNAMIC_MAX_DEPEN_VELOCITY = 0.35
_DYNAMIC_ANGULAR_DAMPING = 5.0


# ----------------------------------------------------------------------------------------------
# Bone table: a bone is the segment J[a] -> J[b] for a (parent_joint, child_joint) pair of the
# SMPL+H kinematic tree (verified parents[0:22] = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,
# 18,19]). We pick ~14 major bones (spine, head, 2x upper-arm, 2x fore-arm, 2x hand, 2x thigh,
# 2x shin, pelvis); fingers are lumped into the hand capsule for M0. Radii are sized per body
# region (general, not per-behavior). The HAND bones are flagged so the grasp can weld to one.
# Joint indices (VERIFIED, SMPL+H body block 0..21):
#   0 pelvis  1 L_hip  2 R_hip  3 spine1  4 L_knee  5 R_knee  6 spine2  9 spine3  12 neck
#   15 head  16 L_shoulder 17 R_shoulder 18 L_elbow 19 R_elbow 20 L_wrist 21 R_wrist
#   7 L_ankle 8 R_ankle  (knee->ankle = shin)
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _BoneSpec:
    name: str
    a: int  # proximal joint index (bone start)
    b: int  # distal joint index (bone end)
    radius: float  # capsule radius (m), sized for the body region
    is_hand: bool = False  # the FixedJoint grasp anchor welds to a hand bone


# Radii sized to the SMPL+H cross-section so contact-based penetration validation is TRUSTWORTHY (the
# old thin radii under-matched the visible mesh, so a contact check would miss real clipping). Torso is
# the worst under-match -> widened most. Biased outward (conservative): the validator should fire a hair
# early rather than miss a real overlap.
_BONES: tuple[_BoneSpec, ...] = (
    _BoneSpec("pelvis", 0, 3, 0.140),  # pelvis -> spine1 (lower torso/hips)
    _BoneSpec("spine", 3, 12, 0.130),  # spine1 -> neck (upper torso column)
    # neck bridges the spine top (neck joint 12) and the head sphere (joint 15). Without it a thin
    # horizontal slab in the ~9cm neck/upper-chest gap penetrates the true mesh but touches no capsule
    # (open_risks: the proxy's un-modeled vertical blind band). Radius sized between spine and head so
    # its caps overlap both -> continuous pelvis->head coverage with no hole.
    _BoneSpec("neck", 12, 15, 0.075),
    # Collar capsules span the spine top (via the L/R collar joints 13/14) OUT to each shoulder joint
    # (16/17) -> they cover the true ~0.20 m shoulder half-width the narrow r=0.13 spine capsule misses.
    # They END at the shoulder joints (16/17), so they are NOT the reaching arm (the reaching arm is the
    # r_upper_arm bone (17,19) + r_forearm (19,21) + r_hand). Radius sized to the shoulder cross-section
    # so a thin vertical obstacle grazing the shoulder outside the spine centerline is caught.
    _BoneSpec("l_collar", 13, 16, 0.070),
    _BoneSpec("r_collar", 14, 17, 0.070),
    _BoneSpec("l_upper_arm", 16, 18, 0.058),
    _BoneSpec("r_upper_arm", 17, 19, 0.058),
    _BoneSpec("l_forearm", 18, 20, 0.052),
    _BoneSpec("r_forearm", 19, 21, 0.052),
    _BoneSpec(
        "l_hand", 20, 22, 0.055, is_hand=True
    ),  # b=22 is a synthetic hand tip (see _bone_endpoints)
    _BoneSpec("r_hand", 21, 23, 0.055, is_hand=True),
    _BoneSpec("l_thigh", 1, 4, 0.105),
    _BoneSpec("r_thigh", 2, 5, 0.105),
    _BoneSpec("l_shin", 4, 7, 0.075),
    _BoneSpec("r_shin", 5, 8, 0.075),
    # Foot capsules span each ankle (7/8) -> foot/toe joint (10/11). Without them the shins stop at the
    # ankles and a thin LOW obstacle at foot height (a raised threshold, a low bar, the near lip of an
    # open drawer, a foot-stool edge) reads CLEAN even with the body's foot inside it. Foot-sized radius.
    _BoneSpec("l_foot", 7, 10, 0.055),
    _BoneSpec("r_foot", 8, 11, 0.055),
)
# Head is a sphere centered at the head joint (no clean child to span), sized larger.
_HEAD_JOINT = 15
_HEAD_RADIUS = 0.100
# The hand has no body joint past the wrist in the 0..21 block (fingers are 37..51); span the
# wrist toward the middle-finger MCP so the hand capsule covers the palm. MANO MCP middle = 40
# (left wrist 20 -> ... but MANO finger joints are the RIGHT hand only in this rig). For a general
# hand capsule we extend the wrist along the wrist->mid-finger direction by a fixed palm length.
_PALM_LEN = (
    0.09  # m, length of the hand capsule past the wrist (covers palm + knuckles)
)
_RHAND_MID_MCP = 40  # right-hand middle MCP (for the right hand direction); left falls back to a default


# Adjacent-bone pairs that SHARE a joint -> FilteredPairsAPI so a continuous limb does not self-report.
# (Derived from the bone table: two bones are adjacent iff they share an endpoint joint.)
def _adjacent_bone_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    n = len(_BONES)
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = _BONES[i], _BONES[j]
            if {bi.a, bi.b} & {bj.a, bj.b}:
                pairs.append((bi.name, bj.name))
    # The pelvis/spine sit next to both thighs and the spine sits next to both upper-arms via the
    # shoulders' parent chain; the shared-joint test above already covers pelvis<->thighs (joint 0/1/2
    # are not shared, so add the torso<->thigh and torso<->upper-arm neighbors explicitly: they abut
    # at the body even though the capsule endpoints differ).
    extra = [
        ("pelvis", "l_thigh"),
        ("pelvis", "r_thigh"),
        ("pelvis", "spine"),
        ("spine", "l_upper_arm"),
        ("spine", "r_upper_arm"),
        ("spine", "neck"),
        ("neck", "head"),
        ("neck", "l_upper_arm"),
        ("neck", "r_upper_arm"),
        ("spine", "head"),
        # collars abut the spine top / neck and their own shoulder (endpoints differ from spine's).
        ("spine", "l_collar"),
        ("spine", "r_collar"),
        ("neck", "l_collar"),
        ("neck", "r_collar"),
        ("l_collar", "l_upper_arm"),
        ("r_collar", "r_upper_arm"),
        # feet abut their shin (share the ankle joint, so the shared-joint test already covers them, but
        # keep them explicit for clarity/robustness if the shin radius shrinks).
        ("l_shin", "l_foot"),
        ("r_shin", "r_foot"),
    ]
    for p in extra:
        if p not in pairs and (p[1], p[0]) not in pairs:
            pairs.append(p)
    return pairs


# ----------------------------------------------------------------------------------------------
# Physics scene + colliders.
# ----------------------------------------------------------------------------------------------
def build_physics_scene(
    stage: "Usd.Stage", scene_path: str = "/World/PhysicsScene", gravity: float = 9.81
) -> object:
    """Define a UsdPhysics.Scene with gravity pointing down (-Z) at ``gravity`` m/s^2.

    Returns the UsdPhysics.Scene wrapper. Idempotent: re-defining the same path returns the
    existing scene (UsdPhysics.Scene.Define is define-or-get).
    """
    from pxr import Gf, UsdPhysics

    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(float(gravity))
    LOG.info("physics scene %s: gravity %.2f m/s^2 along -Z", scene_path, gravity)
    return scene


def _child_meshes(prim: "Usd.Prim") -> list["Usd.Prim"]:
    """All Mesh prims at or under ``prim`` (a collider may live on child meshes, e.g. a referenced USD)."""
    from pxr import Usd

    out = []
    for p in Usd.PrimRange(prim):
        if p.GetTypeName() == "Mesh":
            out.append(p)
    return out


def _has_collisions_scope(prim: "Usd.Prim") -> bool:
    """True iff ``prim`` ships OmniGibson's authored per-link collision meshes under a ``/collisions/``
    scope (each carries CollisionAPI + MeshCollisionAPI approximation=convexHull, purpose=guide). Traverses
    INTO instance proxies read-only (BEHAVIOR references are instanceable) so it never mutates/de-instances
    -- this is the content gate that routes ONLY BEHAVIOR objects to ``behavior_collider``; Isaac-native
    prims (bare gprims / a single referenced mesh with no collisions scope) return False and stay
    byte-identical."""
    from pxr import Usd

    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() == "Mesh" and "/collisions/" in str(p.GetPath()):
            return True
    return False


def _collision_meshes(prim: "Usd.Prim") -> list["Usd.Prim"]:
    """OG's authored collision meshes under ``prim``: the ``/collisions/*`` Mesh prims (already carrying
    CollisionAPI + MeshCollisionAPI convexHull), EXCLUDING the high-poly ``/visuals/*`` render mesh. Call
    after ``_deinstance_subtree`` so the real (non-proxy) meshes are editable."""
    from pxr import Usd

    out = []
    for p in Usd.PrimRange(prim):
        if p.GetTypeName() == "Mesh" and "/collisions/" in str(p.GetPath()):
            out.append(p)
    return out


def _deinstance_subtree(prim: "Usd.Prim") -> None:
    """De-instance any instanceable prims at/under ``prim`` so the real meshes become editable (a
    BEHAVIOR reference is instanceable, which hides its ``/collisions/*`` meshes from Usd.PrimRange).
    Bounded 4-pass loop (nested instances resolve one level per pass); idempotent."""
    from pxr import Usd

    for _ in range(4):
        inst = [p for p in Usd.PrimRange(prim) if p.IsInstance()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)


def behavior_collider(
    prim: "Usd.Prim", dynamic: bool = False, density: float = 400.0
) -> int:
    """Collider for a BEHAVIOR-1K object built from OmniGibson's OWN per-link collision meshes.

    OG authors, offline (CoACD), a proper convex decomposition per link under ``/<link>/collisions/*``
    (CollisionAPI + MeshCollisionAPI approximation=convexHull, purpose=guide) and ships it inside the USD;
    our bake preserves those low-poly meshes. This path de-instances the reference and keeps ONLY those
    collision meshes as colliders (never the ``/visuals/*`` render mesh). Static furniture uses the exact
    triangles: convex-hulling an L-shaped desk, open cabinet, or cubicle panel fills its empty workspace
    with phantom solid geometry. Dynamic graspables must remain convex and therefore retain OG's authored
    convexHull approximation. Routed to ONLY when ``_has_collisions_scope(prim)`` is True, so Isaac-native
    prims are byte-identical. Returns the number of collision meshes tagged."""
    from pxr import UsdGeom, UsdPhysics

    _deinstance_subtree(prim)
    meshes = _collision_meshes(prim)
    if not meshes:
        # The gate proved a collisions scope, so this is defensive only -- never leave an object with no
        # collider (fall back to the old all-mesh behavior rather than pass through a phantom).
        LOG.warning(
            "behavior_collider: no /collisions/ meshes under %s; falling back to child meshes",
            prim.GetPath(),
        )
        meshes = _child_meshes(prim)
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr().Set(float(density))
    for m in meshes:
        # RTX can draw a de-instanced guide mesh after PhysX takes ownership of a
        # moving body.  Hide that simulation-only geometry explicitly; static
        # furniture keeps its authored visibility because labels may be mounted in
        # the same asset hierarchy.  Visibility does not affect CollisionAPI.
        if dynamic:
            UsdGeom.Imageable(m).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(m)
        if not dynamic:
            UsdPhysics.MeshCollisionAPI.Apply(m).CreateApproximationAttr().Set("none")
    # OG ships collision GEOMETRY but no physics MATERIAL, and a material-less shape has zero friction
    # (see bind_physics_material). This is the only place BEHAVIOR colliders are made, static or dynamic.
    bind_physics_material(
        prim.GetStage(), meshes, "convexHull" if dynamic else "static"
    )
    LOG.info(
        "behavior_collider: %s -> %d OG convexHull colliders (%s)",
        prim.GetPath(),
        len(meshes),
        "dynamic" if dynamic else "static",
    )
    return len(meshes)


def _configure_dynamic_body(
    prim: "Usd.Prim", max_depen_vel: float | None = None
) -> None:
    """Apply shared release-safe rigid-body settings to every dynamic prop."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI(prim)
    rb.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateMaxDepenetrationVelocityAttr().Set(
        float(_DYNAMIC_MAX_DEPEN_VELOCITY if max_depen_vel is None else max_depen_vel)
    )
    physx_rb.CreateAngularDampingAttr().Set(float(_DYNAMIC_ANGULAR_DAMPING))


def make_static_collider(
    prim: "Usd.Prim", approximation: str = "convexDecomposition"
) -> int:
    """Make ``prim`` (or its child meshes) a STATIC collider: CollisionAPI (+ MeshCollisionAPI approx).

    A static collider has NO RigidBodyAPI -> it never moves and never falls; it is the immovable
    table the human/cup collide against. If the prim itself is a Mesh we tag it; otherwise we tag
    every descendant Mesh (a referenced desk USD nests its geometry under child meshes). Returns
    the number of meshes tagged.
    """
    from pxr import UsdGeom, UsdPhysics

    if _has_collisions_scope(prim):
        # BEHAVIOR-1K object: use OG's own low-poly per-link collision meshes, never a coarse hull over
        # the visual mesh. Static meshes are exact so open furniture stays open.
        return behavior_collider(prim, dynamic=False)
    meshes = _child_meshes(prim)
    if not meshes and prim.GetTypeName() in (
        "Cube",
        "Sphere",
        "Capsule",
        "Cylinder",
        "Cone",
    ):
        # A bare Gprim (no mesh): CollisionAPI directly, no MeshCollisionAPI (approx is mesh-only).
        UsdPhysics.CollisionAPI.Apply(prim)
        bind_physics_material(prim.GetStage(), [prim], "static")
        LOG.info("static collider (gprim): %s", prim.GetPath())
        return 1
    n = 0
    for m in meshes:
        UsdPhysics.CollisionAPI.Apply(m)
        mca = UsdPhysics.MeshCollisionAPI.Apply(m)
        mca.CreateApproximationAttr().Set(approximation)
        n += 1
    # The surface half of every contact. Without a material here the pair's friction is PhysX's default
    # -- measured at zero, so anything set down on this surface slides until a wall stops it.
    bind_physics_material(prim.GetStage(), meshes, "static")
    LOG.info(
        "static collider: %s -> %d mesh colliders (%s)",
        prim.GetPath(),
        n,
        approximation,
    )
    if n == 0:
        # Fallback: tag the prim itself so it is at least a collider (e.g. a Gprim type we missed).
        UsdPhysics.CollisionAPI.Apply(prim)
        bind_physics_material(prim.GetStage(), [prim], "static")
        LOG.warning(
            "static collider: no child meshes under %s; applied CollisionAPI to the prim",
            prim.GetPath(),
        )
        n = 1
    _ = UsdGeom  # keep the import explicit for symmetry with the dynamic path
    return n


def set_rigid_kinematic(prim: "Usd.Prim", kinematic: bool = True) -> None:
    """Toggle a rigid body between DYNAMIC (gravity/contacts move it) and KINEMATIC (driven by its
    xform, immune to gravity). The cup is made dynamic for the one-time pre-settle, then kinematic so
    the per-frame pose writes carry it during the (kinematic) render loop without physics pulling it off
    the desk or out of the hand. RigidBodyAPI.Apply is get-or-create (idempotent)."""
    from pxr import UsdPhysics

    rb = (
        UsdPhysics.RigidBodyAPI(prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        else UsdPhysics.RigidBodyAPI.Apply(prim)
    )
    attr = rb.GetKinematicEnabledAttr()
    if not attr or not attr.IsValid():
        attr = rb.CreateKinematicEnabledAttr()
    attr.Set(bool(kinematic))


# Synced orchestrator steps that make a kinematic->dynamic flip LIVE in PhysX (see release_to_dynamic).
_RELEASE_RESYNC_STEPS: int = 4


def release_to_dynamic(prims, step_fn=None) -> None:
    """Flip each prim kinematic->DYNAMIC AND drive the resync that makes the flip take effect in PhysX.

    ``set_rigid_kinematic(p, False)`` is a pure USD attribute write; PhysX only re-parses an EXISTING
    actor's kinematic->dynamic flip when the render orchestrator resyncs USD->fabric (``step_fn`` =
    ``fpv_render.advance_once`` with the timeline PLAYING) -- NOT on a bare ``SimulationContext.step``.
    This is the ONE chokepoint that couples the flip with its resync, so no caller can unfreeze a body
    without it (the bug that free-fell every object after the first, previously papered over by a
    per-object warmup inside ``object_physics.settle_bodies``). Pass ALL prims to unfreeze TOGETHER so a
    SINGLE resync makes the whole batch live BEFORE any settle -- the multi-object case correct by
    construction.

    ``step_fn=None`` -> attribute write only (no resync). Two callers use that legitimately and stay
    byte-identical to a bare ``set_rigid_kinematic(p, False)``: an init-time settle creates FRESH dynamic
    bodies a bare step already handles, and the render-loop release relies on the loop's own per-frame
    ``advance_once`` for the resync (extra steps there would capture stray frames).

    The resync is a bounded warmup of ``_RELEASE_RESYNC_STEPS`` rather than a verified transition: a body
    that lands ON a support does not move under gravity, so "responds to gravity" is not observable, and
    USD exposes no live PhysX kinematic-state query. One synced step suffices for a single flip; fabric
    double-buffering + PhysX's deferred actor re-creation can lag, so a few steps is the safe upper bound
    (extra steps on an already-live body are harmless no-ops). 4 matches the verified warmup it replaces.
    """
    for p in prims:
        set_rigid_kinematic(p, False)
    if step_fn is None:
        return
    for _ in range(_RELEASE_RESYNC_STEPS):
        step_fn()


# Per-tier physics material defaults (staticFriction, dynamicFriction, restitution), keyed off the
# object_physics.classify_collider tier a caller passes: a single convex mesh ('convexHull' -- a can /
# box / cup) does not bounce and grips a slight tilt; a concave decomposition ('convexDecomposition' --
# a bowl / mug / banana) is a touch bouncier. A caller can override the restitution per-call. Only bound
# when a tier is passed, so every existing make_dynamic_rigid call (tier=None) is byte-identical.
_MATERIAL_TIERS: dict[str, tuple[float, float, float]] = {
    "convexHull": (0.6, 0.5, 0.05),
    "convexDecomposition": (0.7, 0.6, 0.02),
    # Immovable scene geometry (floor, benchtop, shelf board). Same grip, but a table does not bounce.
    "static": (0.6, 0.5, 0.0),
}


def _get_or_create_physics_material(
    stage: "Usd.Stage", tier: str, restitution: float | None
) -> "Usd.Prim":
    """Get-or-create a UsdPhysics.MaterialAPI prim at /World/PhysicsMaterials/<tier> with the tier's
    static/dynamic friction + restitution (restitution overridable per-call). Returns the material prim
    (the bind target). Reused across every dynamic body of the same tier (one material prim per tier)."""
    from pxr import UsdPhysics

    sf, df, rest = _MATERIAL_TIERS.get(tier, _MATERIAL_TIERS["convexHull"])
    if restitution is not None:
        rest = float(restitution)
    mat_path = f"/World/PhysicsMaterials/{tier}"
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not (mat_prim and mat_prim.IsValid()):
        stage.DefinePrim("/World/PhysicsMaterials", "Scope")
        mat_prim = stage.DefinePrim(mat_path, "Material")
        mapi = UsdPhysics.MaterialAPI.Apply(mat_prim)
        mapi.CreateStaticFrictionAttr().Set(float(sf))
        mapi.CreateDynamicFrictionAttr().Set(float(df))
        mapi.CreateRestitutionAttr().Set(float(rest))
        LOG.info(
            "physics material %s: staticFriction=%.2f dynamicFriction=%.2f restitution=%.2f",
            mat_path,
            sf,
            df,
            rest,
        )
    return mat_prim


def bind_physics_material(stage, meshes, tier: str, restitution=None) -> None:
    """Bind the tier's physics material to every collider in ``meshes``. NOTHING ELSE SETS FRICTION.

    A shape with no bound material falls back to PhysX's own default, whose friction here measures ZERO.
    Traced in bt_chemlab: a bottle seated correctly on a work surface leaves at 0.12 m/s (the tiny nudge
    the seat's de-penetration produces) and then slides at a DEAD CONSTANT speed -- exactly 10.0 mm per
    1/12 s step for 15 consecutive steps, no decay -- until it ran off the open deck and fell 0.95 m to
    the floor. With this material's 0.5 dynamic friction the same 0.12 m/s stops the object in 1.6 mm.

    Closed containers hid it for months: their walls catch the slide, so the object still scores INSIDE.
    Measured slides in the SAME run, all inside cabinets and all previously read as "fine": 594, 203, 187,
    122, 89 mm. Only the one open work surface in the set had no wall to catch them.

    ``materialPurpose='physics'`` is load-bearing -- PhysX ignores default/render-purpose binds."""
    from pxr import UsdShade

    mat = UsdShade.Material(_get_or_create_physics_material(stage, tier, restitution))
    for m in meshes:
        UsdShade.MaterialBindingAPI(m).Bind(mat, materialPurpose="physics")


def make_dynamic_rigid(
    prim: "Usd.Prim",
    density: float = 400.0,
    approximation: str = "convexHull",
    tier: str | None = None,
    restitution: float | None = None,
    max_depen_vel: float | None = None,
) -> int:
    """Make ``prim`` a DYNAMIC rigid body: RigidBodyAPI on the prim + CollisionAPI/MassAPI on its meshes.

    The carried object (cup) becomes a dynamic body so physics can rest it on a surface and resolve
    contact. RigidBodyAPI goes on the top prim (the body); CollisionAPI + the convexHull approx go on
    the child meshes; MassAPI(density) sets its mass from its own geometry (general -- no per-object
    magic). Returns the number of meshes given colliders.

    ``tier`` (P0-remaining, PhysicsMaterialAPI): when set (e.g. ``classify_collider(prim)``), bind a
    per-tier physics material (friction/restitution) to each collider MESH via
    ``UsdShade.MaterialBindingAPI(mesh).Bind(mat, materialPurpose='physics')`` -- the 'physics' purpose
    is load-bearing (PhysX ignores default/render-purpose binds). ``restitution`` overrides the tier's
    default bounce. ``max_depen_vel`` overrides the shared conservative
    PhysxRigidBodyAPI.maxDepenetrationVelocity so a body resolving an initial overlap slides apart at a
    bounded speed rather than launching. Every dynamic prop also receives the same angular damping so
    round objects have ordinary rolling resistance; neither control depends on an asset or task category.
    """
    from pxr import UsdPhysics, UsdShade

    if _has_collisions_scope(prim):
        # BEHAVIOR-1K graspable: RigidBodyAPI on top + OG's own convexHull collision meshes (skip the
        # visual mesh, preserve the authored approximation). Isaac-native prims fall through unchanged.
        n = behavior_collider(prim, dynamic=True, density=density)
        _configure_dynamic_body(prim, max_depen_vel)
        return n
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateDensityAttr().Set(float(density))

    # A material is bound ALWAYS, not only when a caller asks: a shape without one has zero friction and
    # slides forever on the smallest nudge (see bind_physics_material). ``tier`` now only OVERRIDES which
    # one; unset falls back to the collider's own approximation.
    _tier = tier if tier is not None else approximation
    mat_prim = _get_or_create_physics_material(
        prim.GetStage(),
        _tier if _tier in _MATERIAL_TIERS else "convexHull",
        restitution,
    )

    meshes = _child_meshes(prim)
    n = 0
    for m in meshes:
        UsdPhysics.CollisionAPI.Apply(m)
        mca = UsdPhysics.MeshCollisionAPI.Apply(m)
        mca.CreateApproximationAttr().Set(approximation)
        UsdShade.MaterialBindingAPI(m).Bind(
            UsdShade.Material(mat_prim), materialPurpose="physics"
        )
        n += 1
    if n == 0 and prim.GetTypeName() in (
        "Cube",
        "Sphere",
        "Capsule",
        "Cylinder",
        "Cone",
    ):
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(mat_prim), materialPurpose="physics"
        )
        n = 1
    _configure_dynamic_body(prim, max_depen_vel)
    LOG.info(
        "dynamic rigid: %s density=%.0f -> %d mesh colliders (%s)%s%s",
        prim.GetPath(),
        density,
        n,
        approximation,
        f" tier={tier}" if tier is not None else "",
        f" max_depen_vel={max_depen_vel}" if max_depen_vel is not None else "",
    )
    return n


# ----------------------------------------------------------------------------------------------
# Proxy rig (invisible kinematic capsules driven by J each frame).
# ----------------------------------------------------------------------------------------------
@dataclass
class _ProxyBone:
    name: str
    path: str
    a: int
    b: int
    radius: float
    is_hand: bool
    xform_op: object = (
        None  # the MakeMatrixXform op (write the world matrix each frame)
    )
    is_sphere: bool = False  # head is a sphere (centered at the head joint)


@dataclass
class ProxyRig:
    """Handle for the invisible kinematic proxy rig: the bones + the human root path."""

    root: str
    bones: list[_ProxyBone] = field(default_factory=list)
    hand_paths: dict[str, str] = field(
        default_factory=dict
    )  # 'l'/'r' -> hand-proxy prim path
    stage: object | None = (
        None  # set at build; used to re-acquire fresh xform ops each frame
    )
    refetch_ops: bool = (
        True  # PhysX/detector stale cached ops -> re-fetch per frame (always on)
    )

    def bone(self, name: str) -> _ProxyBone | None:
        for b in self.bones:
            if b.name == name:
                return b
        return None


def _bone_endpoints(
    J: np.ndarray, spec_a: int, spec_b: int, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """World endpoints (pa, pb) of a bone for the current joints J.

    For real joint pairs both indices index into J directly. The HAND bones use a synthetic distal
    index (>= len(J)); we extend the wrist along the wrist->mid-finger direction (right hand uses the
    MANO middle MCP joint 40; a left hand -- absent finger joints in this single-hand rig -- falls
    back to the bone's parent direction) by a fixed palm length so the hand capsule covers the palm.
    """
    nj = J.shape[0]
    pa = np.asarray(J[spec_a], dtype=np.float64)
    if spec_b < nj:
        pb = np.asarray(J[spec_b], dtype=np.float64)
        return pa, pb
    # Synthetic hand tip: extend past the wrist.
    if name.startswith("r_") and _RHAND_MID_MCP < nj:
        d = np.asarray(J[_RHAND_MID_MCP], dtype=np.float64) - pa
    else:
        # No finger joints for this hand: extend along the forearm direction (elbow->wrist).
        elbow = (
            spec_a - 1
        )  # wrist's parent in the body chain (20<-18 left, 21<-19 right) is spec_a-? ; use chain
        elbow = 18 if name.startswith("l_") else 19
        d = pa - np.asarray(J[elbow], dtype=np.float64)
    nrm = np.linalg.norm(d)
    if nrm < 1e-6:
        d = np.array([0.0, 0.0, -1.0])
        nrm = 1.0
    pb = pa + (d / nrm) * _PALM_LEN
    return pa, pb


def _bone_matrix(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """4x4 (column-vector) world transform mapping a +Z-axis unit capsule (origin-centered) onto the
    bone pa->pb: origin -> midpoint, local +Z -> the bone direction. Returns a numpy 4x4.

    UsdGeom.Capsule's axis defaults to +Z and it is centered at its local origin, so a capsule of the
    right ``height`` placed by this transform spans pa..pb."""
    mid = 0.5 * (pa + pb)
    z = pb - pa
    ln = np.linalg.norm(z)
    if ln < 1e-9:
        z = np.array([0.0, 0.0, 1.0])
        ln = 1.0
    z = z / ln
    # Build an orthonormal basis with the 3rd column == z.
    up = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0] = x
    M[:3, 1] = y
    M[:3, 2] = z
    M[:3, 3] = mid
    return M


def _set_world_matrix(xform_op: object, M: np.ndarray) -> None:
    """Write a column-vector 4x4 onto a MakeMatrixXform op (USD is row-vector -> transpose)."""
    from pxr import Gf

    xform_op.Set(Gf.Matrix4d(*[float(v) for v in M.T.flatten()]))


def build_proxy_rig(
    stage: "Usd.Stage", rig: object, J: np.ndarray, root: str = "/World/HumanProxy"
) -> ProxyRig:
    """Author ONE invisible kinematic capsule per major bone (sphere for the head) from the skeleton J.

    Each proxy: a UsdGeom.Capsule (or Sphere for the head) with RigidBodyAPI + kinematicEnabled=True +
    CollisionAPI, made invisible (visibility='invisible'). The capsule axis is +Z (USD default); the
    length is the bone length and the radius is the per-region size from the bone table. Adjacent bones
    that share a joint get a FilteredPairsAPI so a continuous limb never self-reports a collision.

    The HAND bones are full rigid bodies (same as the rest, but flagged) -- the grasp FixedJoint welds
    its body0 to a hand proxy, so the hand proxy must be a proper PhysX rigid body.

    ``rig`` is accepted for API generality (later milestones may size from the body mesh) but M0 sizes
    purely from the bone table + J. Returns a ProxyRig handle.
    """
    from pxr import Sdf, UsdGeom, UsdPhysics

    stage.DefinePrim(root, "Xform")
    proxy = ProxyRig(root=root, stage=stage)

    # ---- Capsule bones. ------------------------------------------------------------------------
    for spec in _BONES:
        path = f"{root}/{spec.name}"
        pa, pb = _bone_endpoints(J, spec.a, spec.b, spec.name)
        ln = float(np.linalg.norm(pb - pa))
        cap = UsdGeom.Capsule.Define(stage, path)
        cap.CreateAxisAttr(UsdGeom.Tokens.z)
        cap.CreateRadiusAttr(float(spec.radius))
        # Capsule height is the CYLINDER part length (caps add radius on each end); subtract the two
        # hemispherical caps so the total span ~= the bone length, clamped non-negative.
        cap.CreateHeightAttr(max(0.0, ln - 2.0 * spec.radius))
        op = UsdGeom.Xformable(cap).MakeMatrixXform()
        _set_world_matrix(op, _bone_matrix(pa, pb))
        _make_invisible(cap.GetPrim())
        _apply_kinematic_collider(cap.GetPrim(), UsdPhysics)
        b = _ProxyBone(
            spec.name,
            path,
            spec.a,
            spec.b,
            spec.radius,
            spec.is_hand,
            op,
            is_sphere=False,
        )
        proxy.bones.append(b)
        if spec.is_hand:
            proxy.hand_paths["l" if spec.name.startswith("l_") else "r"] = path

    # ---- Head sphere. --------------------------------------------------------------------------
    head_path = f"{root}/head"
    sp = UsdGeom.Sphere.Define(stage, head_path)
    sp.CreateRadiusAttr(_HEAD_RADIUS)
    head_op = UsdGeom.Xformable(sp).MakeMatrixXform()
    Mh = np.eye(4)
    Mh[:3, 3] = np.asarray(J[_HEAD_JOINT], dtype=np.float64)
    _set_world_matrix(head_op, Mh)
    _make_invisible(sp.GetPrim())
    _apply_kinematic_collider(sp.GetPrim(), UsdPhysics)
    proxy.bones.append(
        _ProxyBone(
            "head",
            head_path,
            _HEAD_JOINT,
            _HEAD_JOINT,
            _HEAD_RADIUS,
            False,
            head_op,
            is_sphere=True,
        )
    )

    # ---- Filter adjacent bones (shared joint) so a continuous limb does not self-report. --------
    name_to_path = {b.name: b.path for b in proxy.bones}
    n_filt = 0
    for n0, n1 in _adjacent_bone_pairs():
        p0, p1 = name_to_path.get(n0), name_to_path.get(n1)
        if not p0 or not p1:
            continue
        fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(p0))
        fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(p1))
        n_filt += 1

    LOG.info(
        "proxy rig %s: %d capsules + head sphere, %d adjacent-pair filters, hands=%s",
        root,
        len(_BONES),
        n_filt,
        proxy.hand_paths,
    )
    return proxy


def _make_invisible(prim: "Usd.Prim") -> None:
    """Hide a proxy prim from rendering (it is collision-only)."""
    from pxr import UsdGeom

    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)


def _apply_kinematic_collider(prim: "Usd.Prim", UsdPhysics) -> None:
    """RigidBodyAPI + kinematicEnabled=True + CollisionAPI: a kinematic collider moved by its transform."""
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(prim)


def update_proxy_rig(proxy: ProxyRig, J: np.ndarray) -> None:
    """Per frame: rewrite each proxy capsule/sphere world transform from the current joints J.

    Kinematic bodies are moved by writing their world xform (PhysX picks the new pose up on the next
    query/step). The capsule heights were fixed at build time from the rest bone lengths; small
    per-frame length changes are absorbed by the capsule's rounded caps (M0 does not re-author heights).

    The xform op is re-acquired fresh each call (``proxy.refetch_ops``): PhysX simulation + the
    collision detector's stage attach/detach STALE cached op handles ('Accessed schema on invalid
    prim'); a freshly-fetched op always works and the cost is negligible.
    """
    from pxr import UsdGeom

    for b in proxy.bones:
        op = b.xform_op
        if proxy.refetch_ops and proxy.stage is not None:
            op = UsdGeom.Xformable(proxy.stage.GetPrimAtPath(b.path)).MakeMatrixXform()
            b.xform_op = op
        if b.is_sphere:
            M = np.eye(4)
            M[:3, 3] = np.asarray(J[b.a], dtype=np.float64)
            _set_world_matrix(op, M)
        else:
            pa, pb = _bone_endpoints(J, b.a, b.b, b.name)
            _set_world_matrix(op, _bone_matrix(pa, pb))


# ----------------------------------------------------------------------------------------------
# Penetration detector (static -- runs an internal single PhysX step, no timeline play needed).
# ----------------------------------------------------------------------------------------------
def human_scene_collision_pairs(
    stage: "Usd.Stage", human_root: str = "/World/HumanProxy"
) -> list[tuple[str, str]]:
    """Return the colliding pairs where EXACTLY ONE path is under ``human_root`` (human-vs-scene).

    Uses ``omni.physx.scripts.physicsUtils.get_initial_collider_pairs(stage)``, which authors a temp
    contact-report layer, runs ONE internal PhysX step, reads the contact report, and restores state
    (it needs the stage registered in the USD StageCache + a UsdPhysics.Scene -- both true once the
    demo has built the scene; it does NOT need ``timeline.play()``). We then filter OUT human-vs-human
    pairs (both under the root) and scene-vs-scene pairs (neither under the root), leaving only the
    human-vs-scene penetrations -- the M0 detector. Pairs are returned sorted for stable logging.
    """
    from omni.physx.scripts.physicsUtils import get_initial_collider_pairs

    pairs = get_initial_collider_pairs(stage)
    root = human_root.rstrip("/")

    def _under(path: str) -> bool:
        return path == root or path.startswith(root + "/")

    out: list[tuple[str, str]] = []
    for a, b in pairs:
        if _under(a) != _under(b):  # XOR: exactly one side is the human
            out.append(tuple(sorted((a, b))))
    out.sort()
    return out


def all_collision_pairs(stage: "Usd.Stage") -> list[tuple[str, str]]:
    """All colliding pairs on the stage (debug helper; unfiltered get_initial_collider_pairs)."""
    from omni.physx.scripts.physicsUtils import get_initial_collider_pairs

    return sorted(tuple(sorted(p)) for p in get_initial_collider_pairs(stage))


def _velocity_hold_step(
    prim: "Usd.Prim", step_fn: Callable[[], None] | None
) -> tuple[Callable[[], None], str]:
    if step_fn is not None:
        return step_fn, "orchestrator-hold"

    import object_physics as ophys

    sim = ophys.get_sim()
    stage = prim.GetStage()

    def _bare_step() -> None:
        with pinned_scene_timestep(stage, quiet=True):
            sim.step(render=False)

    return _bare_step, "bare-hold"


def zero_body_velocity(prim: "Usd.Prim", step_fn=None) -> str:
    """Stop the carried body's derived motion while it remains kinematic.

    This must run after the caller seats the body at its release pose. PhysX derives velocity from a
    kinematic pose jump, so the same pose is held for a bounded resync window before the dynamic flip.
    USD velocity attributes are also reset for actor recreation, but the stationary pose window is what
    clears the live actor on both headless checks and rendered runs.
    """
    from pxr import Gf, UsdPhysics

    set_rigid_kinematic(prim, True)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    velocity = rb.CreateVelocityAttr()
    angular_velocity = rb.CreateAngularVelocityAttr()
    zero = Gf.Vec3f(0.0, 0.0, 0.0)
    velocity.Set(zero)
    angular_velocity.Set(zero)

    hold_step, mechanism = _velocity_hold_step(prim, step_fn)
    for _ in range(_RELEASE_RESYNC_STEPS):
        hold_step()

    velocity.Set(zero)
    angular_velocity.Set(zero)
    return mechanism


def collider_margin(
    prim_path: str,
    center: tuple[float, float, float],
    half_xy: tuple[float, float],
    obj_h: float,
    step: float = 0.008,
    max_steps: int = 14,
) -> dict:
    """How far ``prim_path``'s OWN physics collider sticks out past the box the renderer draws.

    Every measurement in the placement path -- the drop inset, the overlap test, the support probe, the
    does-it-fit gate -- is taken from ``compute_world_bbox``, i.e. the VISUAL geometry. PhysX moves a
    COLLISION shape instead, which for a graspable is a generated approximation (a convex hull over a
    spray bottle swallows the gap between its trigger and its body, so the hull is fatter than the thing
    you see). When they disagree the whole chain is measuring the wrong object: every probe reports
    clear, and the release then interpenetrates and gets thrown out.

    Walks outward from each visual face until the object's own collider stops answering. Returns
    ``{"x_mm", "y_mm"}`` -- millimetres of collider BEYOND the visual box (0 = they agree), and
    ``"n"`` = how many of the 4 directions answered at all. Diagnosis only; never gates anything."""
    out = {"x_mm": None, "y_mm": None, "n": 0}
    try:
        import carb
        from omni.physx import get_physx_scene_query_interface

        sq = get_physx_scene_query_interface()
    except Exception:  # noqa: BLE001
        return out
    root = str(prim_path).rstrip("/")
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    hx, hy = float(half_xy[0]), float(half_xy[1])
    r = 0.4 * step

    def _self_reaches(dx, dy, half):
        """Largest outward offset past ``half`` at which this prim's own collider still answers."""
        best = None
        for k in range(max_steps + 1):
            d = k * step
            px, py = cx + dx * (half + d), cy + dy * (half + d)
            found = False
            for frac in (0.25, 0.55, 0.85):
                hits: list[str] = []

                def _rep(h, _h=hits):
                    _h.append(str(h.rigid_body) or str(h.collision))
                    return True

                sq.overlap_sphere(
                    r,
                    carb.Float3(px, py, cz + (frac - 0.5) * float(obj_h)),
                    _rep,
                    False,
                )
                if any(h == root or h.startswith(root + "/") for h in hits):
                    found = True
                    break
            if found:
                best = d
            elif best is not None:
                break  # left the body for good
        return best

    try:
        xs = [
            v
            for v in (_self_reaches(1, 0, hx), _self_reaches(-1, 0, hx))
            if v is not None
        ]
        ys = [
            v
            for v in (_self_reaches(0, 1, hy), _self_reaches(0, -1, hy))
            if v is not None
        ]
        out["n"] = len(xs) + len(ys)
        if xs:
            out["x_mm"] = max(xs) * 1000.0
        if ys:
            out["y_mm"] = max(ys) * 1000.0
    except Exception:  # noqa: BLE001
        pass
    return out


_SUPPORT_TOL_EPS_M = 1e-3
_COPLANAR_SUPPORT_CONTACT_TOL_M = 0.006
_EMBEDDED_PROBE_RADIUS_M = 0.008


def _embedded_probe_center_z(
    rest_z: float, hit_z: float | None, radius: float = _EMBEDDED_PROBE_RADIUS_M
) -> float:
    """Probe wholly above the discovered support plane, not across its contact boundary."""
    plane_z = max(float(rest_z), float(hit_z) if hit_z is not None else float(rest_z))
    return plane_z + float(radius) + _SUPPORT_TOL_EPS_M


def _within_support_tolerance(hit_z: float | None, rest_z: float, z_tol: float) -> bool:
    return hit_z is not None and abs(float(hit_z) - float(rest_z)) <= (
        float(z_tol) + _SUPPORT_TOL_EPS_M
    )


def _same_physics_owner(first: str, second: str) -> bool:
    first = str(first).rstrip("/")
    second = str(second).rstrip("/")
    if not first or not second:
        return False
    if (
        first == second
        or first.startswith(second + "/")
        or second.startswith(first + "/")
    ):
        return True

    def collision_parent(path: str) -> str:
        return path.split("/collisions/", 1)[0]

    return collision_parent(first) == collision_parent(second)


def _is_coplanar_support_contact(
    blocker_path: str,
    support_path: str | None,
    *,
    hit_z: float | None,
    seat_z: float,
) -> bool:
    return (
        support_path is not None
        and hit_z is not None
        and _same_physics_owner(blocker_path, support_path)
        and abs(float(hit_z) - float(seat_z)) <= _COPLANAR_SUPPORT_CONTACT_TOL_M
    )


def probe_drop_site(
    xy: tuple[float, float],
    rest_z: float,
    obj_half_xy: tuple[float, float],
    obj_h: float,
    exclude_prefixes: tuple[str, ...] = (),
    z_tol: float = 0.03,
) -> dict:
    """Ask PhysX -- not a bounding box -- whether an object can actually REST at ``(xy, rest_z)`` NOW.

    Two independent questions, because "the object ended up on the floor" has two different causes that
    a bbox cannot tell apart and that need different answers:

      SUPPORT   a ray straight down from just above ``rest_z``. Nothing within ``z_tol`` of ``rest_z``
                means the drop point is over a HOLE -- an authored placement space that no real surface
                backs (a deck slot spread across a unit's outer bbox, a shelf board that stops short of
                where the space claims to extend). The object falls.
      CLEARANCE spheres strictly INSIDE the volume the object will occupy. A hit there means the release
                pose INTERSECTS existing geometry, so PhysX resolves the penetration by ejecting the
                object -- measured in bt_chemlab as a bottle launched 1.1 m and another 2.8 m sideways,
                which a "did it fall" test alone reads as the same symptom as a hole.

    Callable only while the simulation is live (colliders are applied AFTER the scene builder returns, so
    this cannot run at build time -- which is exactly why placement spaces are authored blind). Returns
    ``{"ok", "support", "clear", "hit_z", "support_path", "blockers", "reason"}``; ``ok`` is False only on
    a definite negative, and any query failure returns ok=True so a probe that cannot run never blocks a
    placement that works today."""
    # UNKNOWN (None) is not NO. Every failure path below leaves the corresponding verdict None and ``ok``
    # True, so a probe that cannot run -- missing binding, changed return shape, query error -- can only
    # ever leave the caller's existing choice alone. A checker that silently starts rejecting everything is
    # far more damaging than one that silently checks nothing.
    out = {
        "ok": True,
        "support": None,
        "clear": None,
        "hit_z": None,
        "support_path": None,
        "support_frac": None,
        "blockers": [],
        "self_blockers": [],
        "seat_z": None,
        "embedded": None,
        "reason": "not-probed",
    }
    try:
        import carb
        from omni.physx import get_physx_scene_query_interface

        sq = get_physx_scene_query_interface()
    except Exception:  # noqa: BLE001
        return out
    x, y, z = float(xy[0]), float(xy[1]), float(rest_z)
    excl = tuple(p.rstrip("/") for p in exclude_prefixes if p)

    def _excluded(path: str) -> bool:
        return any(path == e or path.startswith(e + "/") for e in excl)

    # ---- SUPPORT: straight down from just above the intended rest plane. Hit shape follows the working
    # probe in probe_rearr.py: a dict with "hit" / "position" / "collision".
    #
    # Sampled over the object's own FOOTPRINT, not just its axis. One centre ray only says "something is
    # under this point"; an object standing on a narrow ledge, over the lip of a hole, or on a grating gets
    # the same answer and then tips off. ``support_frac`` is the share of the footprint that lands on the
    # same plane, which is the difference between "there is a surface" and "the object can stand on it" --
    # and it is the only signal that separates a genuinely broken asset from a merely marginal placement.
    # ``support`` (the pass/fail the selector reads) stays keyed on the CENTRE alone, so this is diagnosis
    # only until we decide what threshold, if any, should gate a choice.
    lift = max(0.02, z_tol)

    def _ray_z(px, py, plane_z=z):
        try:
            _h = sq.raycast_closest(
                carb.Float3(float(px), float(py), float(plane_z) + lift),
                carb.Float3(0.0, 0.0, -1.0),
                lift + 0.30,
            )
            if not _h["hit"]:
                return None, None
            return float(_h["position"][2]), str(_h["collision"])
        except Exception:  # noqa: BLE001
            return None, "ray-failed"

    # Every mesh a downward ray finds AT the rest plane SUPPORTS this site: the object's base comes to rest
    # in contact with it. Collected here so the clearance block below cannot read that contact as a
    # penetration -- you cannot be blocked by what holds you up.
    support_paths: set[str] = set()
    try:
        _hz, _hp = _ray_z(x, y)
        if _hp == "ray-failed":
            out["reason"] = "ray-failed"
        else:
            out["hit_z"], out["support_path"] = _hz, _hp
            out["support"] = _within_support_tolerance(_hz, z, z_tol)
            if out["support"] and _hp:
                support_paths.add(_hp)
            _hxo, _hyo = float(obj_half_xy[0]), float(obj_half_xy[1])
            _on = 0
            for _fx, _fy in ((-0.7, 0.0), (0.7, 0.0), (0.0, -0.7), (0.0, 0.7)):
                _pz, _pp = _ray_z(x + _fx * _hxo, y + _fy * _hyo)
                _hit = _within_support_tolerance(_pz, z, z_tol)
                _on += int(_hit)
                if _hit and _pp and _pp != "ray-failed":
                    support_paths.add(_pp)
            out["support_frac"] = (_on + int(bool(out["support"]))) / 5.0
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"ray-failed:{type(exc).__name__}"

    # ---- CLEARANCE: spheres wholly inside the object's own volume, so any hit IS interpenetration.
    # Sampled over the CROSS-SECTION, not just the central axis: what an object actually collides with on
    # release is usually beside it (a swung-open door leaf, a neighbouring bottle, a sash), and a single
    # axial column walks straight past all of them. Offsets and radius are fractions of the object's own
    # half-extent chosen so every sphere stays strictly inside its silhouette -- 0.6*half + 0.35*half < half
    # -- so a surface merely TOUCHING the object's side is never mistaken for a penetration.
    try:
        hxo, hyo = float(obj_half_xy[0]), float(obj_half_xy[1])
        r = max(0.012, 0.35 * min(hxo, hyo))
        # SEAT ON THE PLANE THE OBJECT WILL ACTUALLY USE. ``rest_z`` is the space's DECLARED board top;
        # the placement seats the base on the surface the ray FOUND (``_measured_seat_z``), and the two
        # differ by a few mm. Probing the declared plane when the real collider is above it evaluates a
        # pose no placement ever adopts -- one whose bottom is inside the board -- so the site is condemned
        # for a penetration that only the probe's own choice of plane created.
        base = float(out["hit_z"]) if out["support"] and out["hit_z"] is not None else z
        out["seat_z"] = base
        # VERTICAL containment, the twin of the lateral guarantee above. Ring heights are fractions of the
        # object's HEIGHT while the radius is a fraction of its WIDTH, so on anything wider than it is tall
        # (r > 0.15*obj_h: a book, a tray, a punnet, a plate) the bottom ring hangs BELOW the base plane and
        # lands inside the very board the object rests on. Clamp the centres into [base+r, base+obj_h-r] --
        # shrinking r when the object is thinner than one sphere -- so every sphere stays strictly inside
        # the object's own silhouette on ALL THREE axes, which is the invariant this block always claimed.
        r_raw, z_raw = r, z  # the pre-fix sphere, kept only for the AUDIT below
        r = min(r, max(0.004, 0.49 * float(obj_h)))
        lo_z, hi_z = base + r, base + float(obj_h) - r
        blockers: set[str] = set()
        self_blockers: set[str] = set()

        def _supports(p: str) -> bool:
            return any(_same_physics_owner(p, s) for s in support_paths)

        def _overlap(cx_, cy_, cz_, rr) -> list[str]:
            hits: list[str] = []

            def _rep(h, _h=hits):
                _h.append(str(h.rigid_body) or str(h.collision))
                return True

            sq.overlap_sphere(float(rr), carb.Float3(cx_, cy_, float(cz_)), _rep, False)
            return [h for h in hits if h and not _excluded(h)]

        local_support_cache: dict[
            tuple[float, float], tuple[float | None, str | None]
        ] = {}

        def _local_support(px: float, py: float) -> tuple[float | None, str | None]:
            key = (float(px), float(py))
            if key not in local_support_cache:
                local_support_cache[key] = _ray_z(px, py, base)
            return local_support_cache[key]

        # ⚠️ REVERTED 2026-08-06 -- do NOT re-densify these rings before reading this. Covering the
        # object's full height (11 overlapping rings, top crown on the object's top) is obviously right on
        # paper: the fume hood's FIXED glass sash starts 209 mm above its deck and the top 25 mm of a
        # 297 mm bottle was never sampled. Measured, it did NOT work. The site still came back
        # ok/support_frac=1.0 -- and the sash is not excluded either (the caller passes only the object,
        # the human and /World/Obj), so ``overlap_sphere`` is blind to that collider for a reason not yet
        # found. Worse, the extra rings shifted the CHOSEN drop point 47 mm deeper into the sash, turning
        # ammonia from "slid 106 mm, still counted placed" into a 1.09 m fall: chemlab 0.7333 -> 0.6667.
        # Find out why the sash reads clear FIRST; the sampling was never the binding constraint.
        for frac in (0.15, 0.5, 0.85):
            cz = min(max(base + frac * float(obj_h), lo_z), hi_z)
            for fx, fy in (
                (0.0, 0.0),
                (-0.6, 0.0),
                (0.6, 0.0),
                (0.0, -0.6),
                (0.0, 0.6),
            ):
                probe_x = x + fx * hxo
                probe_y = y + fy * hyo
                hits = _overlap(probe_x, probe_y, cz, r)
                local_hit_z, local_support = (
                    _local_support(probe_x, probe_y) if hits else (None, None)
                )
                for h in hits:
                    supported = _supports(h) or _is_coplanar_support_contact(
                        h,
                        local_support,
                        hit_z=local_hit_z,
                        seat_z=base,
                    )
                    (self_blockers if supported else blockers).add(h)
        out["blockers"] = sorted(blockers)[:4]
        out["self_blockers"] = sorted(self_blockers)[:4]
        out["clear"] = not blockers
        if os.environ.get("DROP_PROBE_AUDIT"):
            # Re-ask the PRE-FIX question at this same site: the bottom ring at its unclamped centre,
            # around the DECLARED plane. Off by default because it costs five extra overlap queries per
            # site; on, it is the only way to count how many placements the fix actually changed without
            # running a second build against the old code.
            _old = set()
            for fx, fy in (
                (0.0, 0.0),
                (-0.6, 0.0),
                (0.6, 0.0),
                (0.0, -0.6),
                (0.0, 0.6),
            ):
                _old.update(
                    _overlap(
                        x + fx * hxo, y + fy * hyo, z_raw + 0.15 * float(obj_h), r_raw
                    )
                )
            if _old and not blockers:
                print(
                    f"[drop-probe-audit] ({x:.3f},{y:.3f}) declared_z={z_raw:.4f} seat_z={base:.4f} "
                    f"r {r_raw * 1000:.0f}->{r * 1000:.0f}mm dip={(r_raw - 0.15 * float(obj_h)) * 1000:+.0f}mm: "
                    f"PRE-FIX blocked by {[b.rsplit('/', 1)[-1] for b in sorted(_old)[:3]]}, "
                    f"POST-FIX clear"
                )
        if self_blockers and not blockers:
            # The bug this block used to have, now visible whenever it would have fired: the ONLY thing in
            # the object's volume is the surface holding it up, and the site used to be discarded for it.
            print(
                f"[drop-probe] self-support ignored at ({x:.3f},{y:.3f},{base:.4f}): "
                f"{[b.rsplit('/', 1)[-1] for b in sorted(self_blockers)[:3]]} supports this site "
                f"-> site is CLEAR (was: blocked)"
            )
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"overlap-failed:{type(exc).__name__}"

    # ---- EMBEDDED: is the drop point itself INSIDE solid geometry?
    #
    # The support ray cannot answer this, and worse, it answers WRONG. ``lift == z_tol`` by construction
    # above, and PhysX reports a ray that STARTS inside a solid as a hit at distance ~0 -- so an embedded
    # origin returns ``hit_z == z + lift == z + z_tol``, and ``abs(hit_z - z) <= z_tol`` passes by exactly
    # the tolerance. An embedded drop point and a perfectly seated one are today indistinguishable, and
    # the embedded one is where the body is released inside the furniture and PhysX throws it out (a
    # measured 70.7 m/s on house_single_floor).
    #
    # Measured directly by the placement diagnostic rather than assumed: a small overlap sphere DOES
    # see static colliders -- 10 mm inside the washer's shell it names the shape, and in the genuinely
    # hollow drum 50 mm further in it reports nothing, with both negative controls clean. So containment
    # is a reliable test; the standing "overlap_sphere is blind" note in this file is wrong.
    #
    # The containment sphere must sit WHOLLY ABOVE the surface found by the ray. Centering it on the ray
    # origin straddles a legitimate surface whenever the live collision plane is at the top edge of the
    # tolerance band, making simple contact indistinguishable from containment. A vertical wall or closed
    # volume still intersects the raised sphere; a horizontal supporting slab does not.
    try:
        _oz = z + lift
        _telltale = out["hit_z"] is not None and float(out["hit_z"]) >= _oz - 1e-4
        _probe_z = _embedded_probe_center_z(z, out["hit_z"])
        # Require BOTH the ray telltale and positive containment: either alone is too easy to trip.
        out["embedded"] = bool(
            _telltale and _overlap(x, y, _probe_z, _EMBEDDED_PROBE_RADIUS_M)
        )
    except Exception:  # noqa: BLE001
        out["embedded"] = (
            None  # unknown is not "no" -- the caller must treat None as today's behaviour
        )

    if out["support"] is False:
        out["ok"], out["reason"] = False, "no-support"
    elif out["clear"] is False:
        out["ok"], out["reason"] = False, "blocked"
    elif out["support"] and out["clear"]:
        out["reason"] = "ok"
    return out


def _support_reach(sq, carb, org, u, reach, step, z_tol, excluded) -> float:
    """March a DOWNWARD ray outward from ``org`` along unit ``u`` while the surface stays within ``z_tol``
    of ``org``'s z. Returns the distance at which it stops -- i.e. where the supporting surface ENDS."""
    x, y, z = org
    lift = max(0.02, float(z_tol))
    d = 0.0
    while d + step <= reach:
        px, py = x + u[0] * (d + step), y + u[1] * (d + step)
        try:
            h = sq.raycast_closest(
                carb.Float3(px, py, z + lift), carb.Float3(0.0, 0.0, -1.0), lift + 0.30
            )
        except Exception:  # noqa: BLE001
            break
        if not h["hit"] or excluded(str(h["collision"])):
            break
        if abs(float(h["position"][2]) - z) > z_tol:
            break
        d += step
    return d


def _obstacle_reach(sq, carb, org, obj_h, u, reach, excluded) -> float:
    """Nearest thing STANDING on the surface along unit ``u``, sampled at three heights inside the object's
    own band. Capped at (and defaulting to) ``reach`` -- nothing found means nothing in the way."""
    x, y, z = org
    best = float(reach)
    for frac in (0.15, 0.5, 0.85):
        try:
            h = sq.raycast_closest(
                carb.Float3(x, y, z + frac * float(obj_h)),
                carb.Float3(float(u[0]), float(u[1]), 0.0),
                float(reach),
            )
        except Exception:  # noqa: BLE001
            continue
        if not h["hit"] or excluded(str(h["collision"])):
            continue
        p = h["position"]
        best = min(best, ((float(p[0]) - x) ** 2 + (float(p[1]) - y) ** 2) ** 0.5)
    return best


def free_rest_window(
    xy: tuple[float, float],
    rest_z: float,
    obj_h: float,
    exclude_prefixes: tuple[str, ...] = (),
    reach: float = 0.25,
    step: float = 0.02,
    z_tol: float = 0.03,
) -> dict | None:
    """How much usable room surrounds ``(xy, rest_z)`` along each axis, measured on the LIVE colliders.

    Two independent limits per direction, because a drop point can be unusable in two ways that need
    different instruments and that a single query confuses:

      SUPPORT      a downward ray marched OUTWARD in ``step`` increments; the surface must stay within
                   ``z_tol`` of ``rest_z``. This is the only way to see an EDGE -- a horizontal ray sails
                   straight over a bench lip and reports wide-open space right where the object would
                   topple off.
      OBSTRUCTION  a horizontal ray at three heights inside the object's own band: the first solid thing
                   STANDING on the surface (a sash, a splashback, a rail, a neighbouring bottle). A
                   downward ray never sees these -- it passes beside them onto good surface.

    Returns ``{"x": (neg, pos), "y": (neg, pos)}``: usable metres from the axis in each of the four
    directions, each capped at ``reach``. Returns None when the query cannot run, which callers must read
    as "unknown", never as "no room" -- same contract as ``probe_drop_site``."""
    try:
        import carb
        from omni.physx import get_physx_scene_query_interface

        sq = get_physx_scene_query_interface()
    except Exception:  # noqa: BLE001
        return None
    org = (float(xy[0]), float(xy[1]), float(rest_z))
    excl = tuple(p.rstrip("/") for p in exclude_prefixes if p)

    def _excluded(path: str) -> bool:
        return any(path == e or path.startswith(e + "/") for e in excl)

    out: dict = {}
    for axis, (ux, uy) in (("x", (1.0, 0.0)), ("y", (0.0, 1.0))):
        side = []
        for sgn in (-1.0, 1.0):
            u = (ux * sgn, uy * sgn)
            side.append(
                min(
                    _support_reach(sq, carb, org, u, reach, step, z_tol, _excluded),
                    _obstacle_reach(sq, carb, org, obj_h, u, reach, _excluded),
                )
            )
        out[axis] = (float(side[0]), float(side[1]))
    return out


def human_scene_overlap(
    stage: "Usd.Stage",
    J: np.ndarray,
    scene_paths: list[str],
    joint_subset: tuple[int, ...] | None = None,
    human_root: str = "/World/HumanProxy",
    exclude_sphere: tuple[np.ndarray, float] | None = None,
) -> list[tuple[str, str]]:
    """LAG-FREE per-pose human-vs-scene penetration gate via PhysX SCENE-QUERY overlaps (no sim step, no
    USD write, no session-layer mutation). For the body posed at world joints ``J``, sample
    ``overlap_sphere`` along every proxy capsule (``_bone_overlap_samples``) and report which
    ``scene_paths`` prims any sample overlaps. Returns ``[(capsule_name, scene_owner), ...]`` (deduped,
    sorted) for penetrating pairs -- empty => the body is clear of every listed scene prim.

    This is the INNER gate the RL path uses for the REPEATED per-candidate queries (free-cell start +
    stand ladder): the depth-bearing CONTACT gate (``human_scene_separation``) is correct for a SINGLE
    pose but its SimulationContext step LAGS the contact report by several frames when a KINEMATIC proxy
    is re-posed between queries (measured: a pose reposed 6 m away still read as penetrating the old
    obstacle) -- non-deterministic. The overlap query reads the CURRENT collider set directly and is
    exact + deterministic. Boolean (no depth); the free-cell prior + the ladder need only accept/reject.
    ``joint_subset`` (SMPL+H joint indices) restricts sampling to those bones' capsules (e.g. body-CORE
    only, excluding the arm/hand which legitimately reach furniture). ``scene_paths`` should EXCLUDE the
    floor (a body stands ON it) -- the caller passes the non-floor collidable owners.

    ``exclude_sphere`` = (center xyz, radius): any capsule SAMPLE whose center falls inside this world
    sphere is skipped. Used to gate the reaching ARM against furniture MINUS the intended CONTACT region
    (a small sphere around the handle/grasp point) -- so an arm reaching THROUGH a cabinet body is caught
    while the arm legitimately AT the handle is not. Only samples near the contact are dropped; the rest
    of the arm still reports, so a deep-through-furniture overlap outside the contact sphere surfaces."""
    import carb
    from omni.physx import get_physx_scene_query_interface

    sq = get_physx_scene_query_interface()
    root = human_root.rstrip("/")
    scene_roots = [p.rstrip("/") for p in scene_paths]

    def _owner(path: str) -> str | None:
        for r in scene_roots:
            if path == r or path.startswith(r + "/"):
                return r
        return None

    excl_c, excl_r = (None, 0.0)
    if exclude_sphere is not None:
        excl_c = np.asarray(exclude_sphere[0], dtype=np.float64)
        excl_r = float(exclude_sphere[1])

    samples = _bone_overlap_samples_subset(J, joint_subset)
    found: set[tuple[str, str]] = set()
    for center, radius, cap_name in samples:
        if excl_c is not None and float(np.linalg.norm(center - excl_c)) <= excl_r:
            continue  # this sample is inside the intended contact region -> not a violation
        hits: list[str] = []

        def _rep(hit, _h=hits):
            _h.append(str(hit.rigid_body) or str(hit.collision))
            return True

        sq.overlap_sphere(
            float(radius),
            carb.Float3(float(center[0]), float(center[1]), float(center[2])),
            _rep,
            False,
        )
        for h in hits:
            if h == root or h.startswith(root + "/"):
                continue
            owner = _owner(h)
            if owner is not None:
                found.add((cap_name, owner))
    return sorted(found)


def scene_sphere_overlap(
    stage: "Usd.Stage",
    center: np.ndarray,
    radius: float,
    scene_paths: list[str],
    human_root: str = "/World/HumanProxy",
) -> str | None:
    """Return the first scene owner intersecting one world-space sphere, else None."""
    import carb
    from omni.physx import get_physx_scene_query_interface

    sq = get_physx_scene_query_interface()
    root = human_root.rstrip("/")
    scene_roots = [path.rstrip("/") for path in scene_paths]
    owner = None

    def _rep(hit):
        nonlocal owner
        path = str(hit.rigid_body) or str(hit.collision)
        if path == root or path.startswith(root + "/"):
            return True
        owner = next(
            (
                scene_root
                for scene_root in scene_roots
                if path == scene_root or path.startswith(scene_root + "/")
            ),
            None,
        )
        return owner is None

    point = np.asarray(center, dtype=np.float64)
    sq.overlap_sphere(
        max(0.0, float(radius)),
        carb.Float3(float(point[0]), float(point[1]), float(point[2])),
        _rep,
        False,
    )
    return owner


def _bone_overlap_samples_subset(
    J: np.ndarray, joint_subset: tuple[int, ...] | None
) -> list[tuple[np.ndarray, float, str]]:
    """Like ``_bone_overlap_samples`` but tagged with the capsule name + restricted to bones whose BOTH
    endpoints are in ``joint_subset`` (None -> all). Lets a caller gate BODY-CORE only (arm/hand out)."""
    out: list[tuple[np.ndarray, float, str]] = []
    subset = set(joint_subset) if joint_subset is not None else None
    for spec in _BONES:
        if subset is not None and not ({spec.a, spec.b} <= subset):
            continue
        pa, pb = _bone_endpoints(J, spec.a, spec.b, spec.name)
        ln = float(np.linalg.norm(pb - pa))
        n = max(2, int(ln / max(spec.radius, 0.03)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            out.append((pa + (pb - pa) * t, float(spec.radius), spec.name))
    if subset is None or _HEAD_JOINT in subset:
        out.append(
            (np.asarray(J[_HEAD_JOINT], dtype=np.float64), float(_HEAD_RADIUS), "head")
        )
    return out


def human_scene_exceeds_depth(
    stage: "Usd.Stage",
    J: np.ndarray,
    scene_paths: list[str],
    allowed_m: float,
    joint_subset: tuple[int, ...] | None = None,
    human_root: str = "/World/HumanProxy",
) -> bool:
    """Whether any capsule sample penetrates a scene owner by more than ``allowed_m``.

    For a capsule of radius ``r0``, penetration exceeds ``d`` exactly when a sphere of radius
    ``r0 - d`` at the same center overlaps. This turns the hot planner gate into one scene query per
    sample with an early exit, while the full bisection remains reserved for final diagnostics.
    """
    import carb
    from omni.physx import get_physx_scene_query_interface

    sq = get_physx_scene_query_interface()
    root = human_root.rstrip("/")
    scene_roots = [path.rstrip("/") for path in scene_paths]
    allowed = max(0.0, float(allowed_m))

    for center, radius, _cap_name in _bone_overlap_samples_subset(J, joint_subset):
        query_radius = float(radius) - allowed
        if query_radius <= 0.0:
            continue
        exceeded = False

        def _rep(hit):
            nonlocal exceeded
            path = str(hit.rigid_body) or str(hit.collision)
            if path == root or path.startswith(root + "/"):
                return True
            exceeded = any(
                path == owner or path.startswith(owner + "/") for owner in scene_roots
            )
            return not exceeded

        sq.overlap_sphere(
            query_radius,
            carb.Float3(float(center[0]), float(center[1]), float(center[2])),
            _rep,
            False,
        )
        if exceeded:
            return True
    return False


def human_scene_depth(
    stage: "Usd.Stage",
    J: np.ndarray,
    scene_paths: list[str],
    joint_subset: tuple[int, ...] | None = None,
    human_root: str = "/World/HumanProxy",
    iters: int = 12,
) -> tuple[float, str, bool]:
    """Worst capsule penetration DEPTH in metres for the pose ``J``, with a description + a SATURATION flag.

    ``overlap_sphere(c, r)`` is true iff ``dist(c, hull) < r``, so bisecting ``r`` on a sample that
    already overlaps at its own capsule radius recovers ``dist(c, hull)``; the depth is
    ``capsule_radius - dist``. This reuses the SAME lag-free scene query as ``human_scene_overlap`` --
    no sim step, no contact report, no kinematic-repose lag -- so unlike ``human_scene_separation`` it
    is deterministic and safe to call repeatedly mid-episode.

    SATURATION IS LOAD-BEARING. ``dist >= 0``, so the returned depth can never exceed the capsule
    radius (140 mm pelvis, 105 mm thigh): a sample whose CENTRE is inside the hull bisects all the way
    down and reports exactly the radius, when the true penetration is unbounded above. The third
    element is True whenever any measured sample bottomed out, i.e. the depth is a LOWER BOUND, and a
    caller must never present it as "at most".

    Returns ``(0.0, "", False)`` when nothing overlaps.
    """
    import carb
    from omni.physx import get_physx_scene_query_interface

    sq = get_physx_scene_query_interface()
    root = human_root.rstrip("/")
    scene_roots = [p.rstrip("/") for p in scene_paths]

    def _hit_owner(center: np.ndarray, radius: float) -> str | None:
        hits: list[str] = []

        def _rep(hit, _h=hits):
            _h.append(str(hit.rigid_body) or str(hit.collision))
            return True

        sq.overlap_sphere(
            float(radius),
            carb.Float3(float(center[0]), float(center[1]), float(center[2])),
            _rep,
            False,
        )
        for h in hits:
            if h == root or h.startswith(root + "/"):
                continue
            for r in scene_roots:
                if h == r or h.startswith(r + "/"):
                    return r
        return None

    worst, desc, saturated = 0.0, "", False
    floor = 0.5 ** int(iters)  # the bisection's own resolution, as a fraction of r0
    for center, r0, cap_name in _bone_overlap_samples_subset(J, joint_subset):
        owner = _hit_owner(center, r0)
        if owner is None:
            continue
        lo, hi = 0.0, float(r0)
        for _ in range(int(iters)):
            mid = 0.5 * (lo + hi)
            if _hit_owner(center, mid) is not None:
                hi = mid
            else:
                lo = mid
        clearance = 0.5 * (lo + hi)
        if clearance <= float(r0) * floor:
            saturated = True
        pen = float(r0) - clearance
        if pen > worst:
            worst = pen
            desc = (
                f"{cap_name} vs {owner} r={r0 * 1000:.0f}mm "
                f"clearance={clearance * 1000:.1f}mm"
            )
    return worst, desc, saturated


def arm_kinematic_static_gate(
    sim,
    stage: "Usd.Stage",
    collider_paths: list[str],
    scene_path: str = "/World/PhysicsScene",
) -> None:
    """One-time setup so KINEMATIC-vs-STATIC contacts actually REPORT (the load-bearing ordering fix).

    ``reportKinematicStaticPairs`` is a PhysX SCENE flag read when the PhysX scene is CREATED. If the
    SimulationContext already initialized physics (e.g. an object settle ran first) the flag was OFF at
    creation, so setting it afterwards + stepping yields ZERO kinematic-static contacts (measured: the
    same false-negative as the bare detector). This authors the contact-report APIs + the scene flag,
    then RESETS the SimulationContext so PhysX re-creates its scene WITH the flag live. Idempotent per
    stage via a custom bool attr on the scene prim -- run once, cheap thereafter (later gate calls just
    step + read). ``dynamic-static`` pairs (the object gate) report by default and don't need this."""
    scene_prim = stage.GetPrimAtPath(scene_path)
    done_attr = scene_prim.GetAttribute("initval:ksReportReady")
    already = bool(done_attr.Get()) if done_attr and done_attr.IsValid() else False
    enable_contact_reports(stage, list(collider_paths), scene_path=scene_path)
    if already:
        return
    # Re-create the PhysX scene with reportKinematicStaticPairs live. The step(s) right after a reset do
    # NOT settle contact reporting (the same warm-up stall physcheck documents after timeline.play), so
    # take several uncounted warm-up ticks -> the FIRST real query returns a steady, correct contact set.
    # Without a long-enough warm-up the FIRST stand-pose gate after the reset FALSE-POSITIVES on a clean
    # pose (measured: seed 0's first query diverged from the warm seeds until the warm-up matched the
    # per-query flush+double-step). Each warm-up tick mirrors a real query (app tick + step).
    import omni.kit.app

    sim.reset()
    for _ in range(8):
        omni.kit.app.get_app().update()
        sim.step(render=False)
    from pxr import Sdf

    scene_prim.CreateAttribute("initval:ksReportReady", Sdf.ValueTypeNames.Bool).Set(
        True
    )
    LOG.info(
        "arm_kinematic_static_gate: PhysX scene reset + warm-up with reportKinematicStaticPairs live"
    )


def human_scene_separation(
    sim,
    stage: "Usd.Stage",
    scene_paths: list[str],
    human_root: str = "/World/HumanProxy",
    sep_tol: float = 0.002,
) -> list[tuple[str, str, float]]:
    """AUTHORITATIVE human-vs-scene penetration gate: depth-bearing, KINEMATIC-STATIC reporting ON.

    THE init gate for the KINEMATIC human proxy (do NOT use ``human_scene_collision_pairs`` for this:
    it routes to ``get_initial_collider_pairs`` which NEVER sets ``reportKinematicStaticPairs``, so a
    torso embedded in a STATIC table reads CLEAN -- a silent false-negative). This function is the
    ``object_physics.all_pairs_separation`` pattern specialized to the human: it ensures kinematic-static
    reporting is live (``arm_kinematic_static_gate`` -- one-time flag + PhysX reset), takes ONE
    ``sim.step(render=False)``, reads ``read_contacts()`` signed ``ContactData.separation`` (the
    penetration DEPTH), then filters to pairs where EXACTLY ONE side is under ``human_root``
    (human-vs-scene). Returns ``[(a, b, min_sep_m), ...]`` for penetrating pairs only (min_sep <
    -sep_tol), deepest first. Empty => the human is clear.

    ``sim`` is the shared render=False SimulationContext (object_physics.get_sim()); reporting is
    enabled on the proxy capsule paths + ``scene_paths`` (the furniture/floor)."""
    from pxr import PhysicsSchemaTools

    proxy_paths = [b.path for b in _proxy_bone_paths(stage, human_root)]
    arm_kinematic_static_gate(sim, stage, proxy_paths + list(scene_paths))
    # FLUSH + DOUBLE-STEP so the contact report reflects the CURRENT proxy pose, not a stale one. A
    # KINEMATIC body's freshly-written USD xform is not picked up by the SimulationContext side-channel
    # in a single step: measured, the report LAGS one pose (a proxy reposed 6 m away still read as
    # penetrating the old obstacle) -> a non-deterministic false-positive. One Kit app tick propagates
    # the USD change; the FIRST step then moves the kinematic target into place (its contacts are still
    # for the previous pose) and the SECOND step generates the steady contact report for THIS pose. The
    # proxy is kinematic (does not drift), so two identical-pose steps are safe + deterministic.
    import omni.kit.app

    omni.kit.app.get_app().update()
    sim.step(render=False)
    sim.step(render=False)
    headers, data = read_contacts()

    root = human_root.rstrip("/")

    def _under(path: str) -> bool:
        return path == root or path.startswith(root + "/")

    scene_roots = [p.rstrip("/") for p in scene_paths]

    def _scene_owner(path: str) -> str | None:
        for r in scene_roots:
            if path == r or path.startswith(r + "/"):
                return r
        return None

    worst: dict = {}
    for h in headers:
        a = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
        b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
        a_h, b_h = _under(a), _under(b)
        if a_h == b_h:  # both human, or neither human (scene-vs-scene) -> not our pair
            continue
        scene_side = _scene_owner(b if a_h else a)
        if scene_side is None:  # human vs an unlisted prim -> ignore
            continue
        off, num = int(h.contact_data_offset), int(h.num_contact_data)
        seps = [float(data[k].separation) for k in range(off, off + num)]
        if not seps:
            continue
        key = ("HUMAN", scene_side)
        worst[key] = min(worst.get(key, 0.0), min(seps))

    out = [(a, b, sep) for (a, b), sep in worst.items() if sep < -float(sep_tol)]
    out.sort(key=lambda t: t[2])
    return out


def _proxy_bone_paths(stage: "Usd.Stage", human_root: str) -> list["_ProxyBone"]:
    """Every collider prim under ``human_root`` (the proxy capsules + head sphere), as lightweight
    handles carrying only a ``.path``. Used by ``human_scene_separation`` so the caller need not thread
    the ProxyRig object through -- the gate reads the live stage."""
    from pxr import Usd, UsdPhysics

    root_prim = stage.GetPrimAtPath(human_root)
    out: list[_ProxyBone] = []
    if not (root_prim and root_prim.IsValid()):
        return out
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            out.append(
                _ProxyBone(prim.GetName(), str(prim.GetPath()), 0, 0, 0.0, False)
            )
    return out


# ----------------------------------------------------------------------------------------------
# Grasp: a PhysX FixedJoint between the hand proxy (body0) and the object (body1).
# Minimal port of OmniGibson usd_utils.create_joint (UsdPhysics.FixedJoint.Define + body0/body1 rels);
# we do NOT import OmniGibson (heavy) -- just the load-bearing two lines.
# ----------------------------------------------------------------------------------------------
def grasp_weld(
    stage: "Usd.Stage",
    hand_proxy_path: str,
    obj_path: str,
    joint_path: str = "/World/GraspJoint",
) -> object:
    """Weld the object rigidly to the hand proxy via a UsdPhysics.FixedJoint (body0=hand, body1=object).

    The hand proxy is a kinematic rigid body and the object is dynamic; a FixedJoint from a kinematic
    body to a dynamic one is supported -- the kinematic body drives the dynamic one (so the cup rides
    the hand). Returns the created joint prim. ``exclude_from_articulation`` is set True (standalone
    joint, not part of any articulation) per the OmniGibson pattern.
    """
    from pxr import PhysxSchema, Sdf, UsdPhysics

    assert stage.GetPrimAtPath(hand_proxy_path).IsValid(), (
        f"invalid hand proxy path: {hand_proxy_path}"
    )
    assert stage.GetPrimAtPath(obj_path).IsValid(), f"invalid object path: {obj_path}"

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.GetBody0Rel().SetTargets([Sdf.Path(hand_proxy_path)])
    joint.GetBody1Rel().SetTargets([Sdf.Path(obj_path)])
    jp = stage.GetPrimAtPath(joint_path)
    PhysxSchema.PhysxJointAPI.Apply(jp)
    jp.GetAttribute("physics:jointEnabled").Set(True)
    excl = jp.GetAttribute("physics:excludeFromArticulation")
    if excl:
        excl.Set(True)
    LOG.info(
        "grasp weld: FixedJoint %s  hand=%s <-> obj=%s",
        joint_path,
        hand_proxy_path,
        obj_path,
    )
    return jp


def release_weld(stage: "Usd.Stage", joint_path: str = "/World/GraspJoint") -> bool:
    """Delete the grasp FixedJoint so the object is free again (then physics can settle it). Returns
    True if a joint was removed."""
    prim = stage.GetPrimAtPath(joint_path)
    if not prim or not prim.IsValid():
        LOG.warning("release_weld: no joint at %s", joint_path)
        return False
    stage.RemovePrim(joint_path)
    LOG.info("grasp release: removed FixedJoint %s", joint_path)
    return True


# ----------------------------------------------------------------------------------------------
# Physics settle without rendering (advance PhysX only until the object comes to rest).
# ----------------------------------------------------------------------------------------------
def settle(
    sim: object,
    get_obj_lin_vel_fn,
    max_steps: int = 120,
    vel_thresh: float = 1e-3,
    step_fn=None,
) -> int:
    """Step physics (render=False) until the object's linear speed < ``vel_thresh`` or ``max_steps``.

    ``sim`` is a SimulationContext (isaacsim.core.api) -- we call ``sim.step(render=False)`` which
    advances PhysX ONLY (no RTX render), the clean "let physics rest the object" primitive.
    ``get_obj_lin_vel_fn()`` returns the object's current linear velocity (a 3-vector or its norm).
    Returns the number of physics steps taken.

    ``step_fn`` (optional): a zero-arg callable that advances the sim ONE step INSTEAD of the bare
    ``sim.step(render=False)``. Pass the render orchestrator's per-step advance (``advance_once`` with the
    timeline PLAYING) when a body was just FLIPPED kinematic->dynamic: a bare ``SimulationContext.step`` does
    NOT re-parse that flip, so the just-unfrozen body never collides/settles. None -> the bare step
    (byte-identical default for every existing caller).
    """
    steps = 0
    for _ in range(int(max_steps)):
        if step_fn is not None:
            step_fn()
        else:
            sim.step(render=False)
        steps += 1
        v = get_obj_lin_vel_fn()
        speed = (
            float(np.linalg.norm(np.asarray(v, dtype=np.float64)))
            if np.ndim(v)
            else float(abs(v))
        )
        if speed < float(vel_thresh):
            break
    LOG.info(
        "settle: %d physics steps (render=False), final |v|=%.5f m/s", steps, speed
    )
    return steps


# ----------------------------------------------------------------------------------------------
# Physics-STEPPED loop helpers: run PhysX every captured frame so the object is truly simulated
# (rests / is carried / settles) and validation is REAL PhysX contact reports, not geometry.
# ----------------------------------------------------------------------------------------------
def set_scene_timestep(
    stage: "Usd.Stage",
    fps: float,
    scene_path: str = "/World/PhysicsScene",
    substeps: int = 1,
) -> None:
    """Set the PhysX scene TimeStepsPerSecond so ONE rep.orchestrator.step(delta_time=1/fps) integrates
    ``substeps`` physics substeps at the video rate (TimeStepsPerSecond = fps*substeps). If physx_dt >
    delta_time the cup would fall in SLOW MOTION relative to the video (the orchestrator warns); a rate
    of fps*substeps with substeps>=1 keeps physx_dt <= delta_time so no slow-motion.

    ``substeps`` > 1 (the P3 free-dynamic path: N substeps per captured frame so a released object
    de-penetrates + settles smoothly ON CAMERA) additionally sets the DETERMINISM flags that make each
    fixed-dt orchestrator.step deliver its FULL substep budget instead of a wall-clock-limited partial
    advance (= slow motion): useFixedTimeStepping + set_play_every_frame, plus minFrameRate=fps as the
    max-substeps-per-render clamp. substeps=1 (the default) is byte-identical to the old one-substep
    behaviour and touches NO carb/timeline settings, so the kinematic scenes (openclose/highreach/sit/
    walkby + the frozen clutter replay) are unchanged."""
    from pxr import PhysxSchema

    sapi = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
    rate = float(fps) * int(substeps)
    sapi.CreateTimeStepsPerSecondAttr().Set(rate)
    if int(substeps) > 1:
        import carb

        settings = carb.settings.get_settings()
        settings.set("/persistent/simulation/minFrameRate", int(fps))
        settings.set("/app/player/useFixedTimeStepping", True)
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().set_play_every_frame(True)
        except Exception as exc:  # noqa: BLE001 - determinism nicety; not fatal if the method moved
            LOG.warning(
                "set_scene_timestep: set_play_every_frame unavailable (%s); relying on fixed timestep",
                exc,
            )
    LOG.info(
        "physx scene %s: TimeStepsPerSecond=%.1f (%d substep(s) per captured frame at %.1f fps)",
        scene_path,
        rate,
        int(substeps),
        float(fps),
    )


SETTLE_HZ: float = 60.0


@contextmanager
def pinned_scene_timestep(
    stage: "Usd.Stage",
    rate: float = SETTLE_HZ,
    scene_path: str = "/World/PhysicsScene",
    quiet: bool = False,
):
    """Force the PhysX scene rate for the duration of a block, then restore the previous value verbatim.

    A SAFETY NET, not a fix for anything currently broken -- measured: it never fires on any of the five
    regression scenes, because ``_ensure_settle_clock`` already raises the rate to FPS*5 (=60 Hz) before
    the first settle and every settle observed ran at 60. It exists because nothing else GUARANTEES that:
    ``timeStepsPerSecond`` lives on the SCENE PRIM, so it is global and last-writer-wins, and it beats the
    ``physics_dt`` a SimulationContext was constructed with -- ``object_physics.get_sim()`` asks for 1/60
    only on the call that CREATES the singleton, then reuses the cached instance forever without
    re-asserting it. The scene genuinely does sit at 12 Hz between settles (``set_scene_timestep(stage,
    FPS)``), so any future settle path that forgets to start the clock would silently integrate 5x coarser
    with nothing to catch it.

    ⚠️ Do NOT read a per-settle-step displacement as a per-INTEGRATION-step displacement. One
    ``step_fn()`` under ``substeps=5`` advances 1/12 s of simulated time as 5 PhysX substeps of 1/60. A
    trace showing 10 mm per observed step at |v|=0.125 m/s is 1/12 s of TIME, not a 1/12 s timestep -- the
    integration underneath is fine. Misreading exactly that cost this investigation a round.

    Restores rather than sets-and-forgets: the render loop owns the rate outside a settle (its substep
    budget is tied to the carb determinism flags ``set_scene_timestep`` toggles), so leaving 60 Hz behind
    would silently re-time every kinematic scene."""
    from pxr import PhysxSchema

    prim = stage.GetPrimAtPath(scene_path)
    if not prim or not prim.IsValid():
        yield  # no physics scene (pure-kinematic stage) -- nothing to pin, never fail a settle over it
        return
    attr = PhysxSchema.PhysxSceneAPI.Apply(prim).CreateTimeStepsPerSecondAttr()
    prev = attr.Get()
    if prev is not None and abs(float(prev) - float(rate)) < 1e-9:
        yield
        return
    attr.Set(float(rate))
    if (
        not quiet
    ):  # quiet=True for a per-STEP pin, which would otherwise log once per step
        LOG.info(
            "settle timestep: pinned physx scene %.1f -> %.1f Hz (restored afterwards)",
            -1.0 if prev is None else float(prev),
            float(rate),
        )
    try:
        yield
    finally:
        if prev is not None:
            attr.Set(float(prev))


def enable_contact_reports(
    stage: "Usd.Stage",
    collider_paths: list[str],
    scene_path: str = "/World/PhysicsScene",
    report_kinematic_static: bool = True,
) -> int:
    """Turn on REAL per-step contact reports (replaces the one-shot get_initial_collider_pairs).

    (1) On the scene: report KINEMATIC-vs-STATIC pairs -- OFF by default, so without this the KINEMATIC
    human proxy vs the STATIC desk yields ZERO contacts (a false "no penetration"). (2) On each collider:
    PhysxContactReportAPI with Threshold 0.0 (report every touch, not only hard impacts). The cup is
    dynamic, so cup-vs-desk (dynamic-static) and cup-vs-hand (dynamic-kinematic) already report.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    sapi = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
    sapi.CreateReportKinematicStaticPairsAttr().Set(bool(report_kinematic_static))
    n = 0
    for p in collider_paths:
        root_prim = stage.GetPrimAtPath(p)
        if not root_prim or not root_prim.IsValid():
            continue
        # Apply to the prim itself (rigid body / capsule collider) AND any descendant Mesh colliders
        # (the desk + cup have CollisionAPI on child meshes, not the parent Xform).
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim == root_prim:
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(
                    0.0
                )
                n += 1
    LOG.info(
        "contact reports: enabled on %d colliders (reportKinematicStaticPairs=%s)",
        n,
        report_kinematic_static,
    )
    return n


def read_contacts():
    """Return (contact_headers, contact_data) for the PhysX step JUST taken. The buffer is valid ONLY for
    that step, so call this IMMEDIATELY after the single render/physics step (advance_once)."""
    from omni.physx import get_physx_simulation_interface

    return get_physx_simulation_interface().get_contact_report()


def classify_contacts(
    headers,
    data,
    human_root: str = "/World/HumanProxy",
    obj_path: str = "/World/Obj",
    desk_path: str | None = None,
    sep_tol: float = 0.002,
) -> dict:
    """Bucket the raw contact report into human-vs-desk / cup-vs-desk / cup-vs-hand and the worst
    human-vs-desk penetration depth. Each collider int is decoded to its SdfPath (intToSdfPath); the
    desk/cup colliders live on CHILD meshes so we prefix-match. Returns {n_human_desk, min_sep_human_desk
    (most negative = deepest penetration, m), cup_desk, cup_hand, pen} (pen=True iff a human-vs-desk
    contact is deeper than sep_tol)."""
    from pxr import PhysicsSchemaTools

    root = human_root.rstrip("/")

    def _is(p: str, base: str) -> bool:
        return p == base or p.startswith(base.rstrip("/") + "/")

    res = {
        "n_human_desk": 0,
        "min_sep_human_desk": 0.0,
        "cup_desk": False,
        "cup_hand": False,
        "pen": False,
    }
    for h in headers:
        a = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
        b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
        off, num = int(h.contact_data_offset), int(h.num_contact_data)
        seps = [float(data[k].separation) for k in range(off, off + num)]
        min_sep = min(seps) if seps else 0.0
        a_h, b_h = _is(a, root), _is(b, root)
        a_o, b_o = _is(a, obj_path), _is(b, obj_path)
        a_d = desk_path is not None and _is(a, desk_path)
        b_d = desk_path is not None and _is(b, desk_path)
        if (a_h and b_d) or (b_h and a_d):
            res["n_human_desk"] += 1
            res["min_sep_human_desk"] = min(res["min_sep_human_desk"], min_sep)
            if min_sep < -float(sep_tol):
                res["pen"] = True
        elif (a_o and b_d) or (b_o and a_d):
            res["cup_desk"] = True
        elif (a_o and b_h) or (b_o and a_h):
            res["cup_hand"] = True
    return res


def apply_hand_cup_filter(stage: "Usd.Stage", hand_path: str, obj_path: str) -> None:
    """Disable hand<->cup collision (FilteredPairsAPI on the cup) so the approaching KINEMATIC hand never
    KNOCKS the resting DYNAMIC cup before the grasp weld. Cleared/kept per the carry strategy."""
    from pxr import Sdf, UsdPhysics

    fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(obj_path))
    fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(hand_path))
    LOG.info(
        "hand-cup collision FILTERED (%s <-> %s) -- no pre-grasp knock",
        hand_path,
        obj_path,
    )


def grasp_weld_framed(
    stage: "Usd.Stage",
    hand_proxy_path: str,
    obj_path: str,
    joint_path: str = "/World/GraspJoint",
) -> object:
    """FixedJoint that welds the cup to the hand at the EXACT current relative pose (no origin snap).

    A bare FixedJoint welds the two bodies' ORIGINS together -> the cup jumps to the hand-bone midpoint.
    Instead we author localPos0/localRot0 (the joint frame in the HAND's local space) = the cup's current
    pose expressed in the hand frame, and localPos1/localRot1 = identity (joint frame == cup origin). So
    PhysX freezes the in-hand offset the cup already has, and body0=hand (kinematic) rigidly drives
    body1=cup (dynamic)."""
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    th = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(hand_proxy_path))
    tc = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(obj_path))
    rel = (
        tc * th.GetInverse()
    )  # cup-local -> hand-local (USD row-vector: p_hand = p_cup * rel)
    t0 = rel.ExtractTranslation()
    q0 = rel.ExtractRotation().GetQuat()

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.GetBody0Rel().SetTargets([Sdf.Path(hand_proxy_path)])
    joint.GetBody1Rel().SetTargets([Sdf.Path(obj_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(t0[0], t0[1], t0[2]))
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(float(q0.GetReal()), Gf.Vec3f(q0.GetImaginary()))
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    jp = stage.GetPrimAtPath(joint_path)
    PhysxSchema.PhysxJointAPI.Apply(jp)
    jp.GetAttribute("physics:jointEnabled").Set(True)
    excl = jp.GetAttribute("physics:excludeFromArticulation")
    if excl:
        excl.Set(True)
    LOG.info(
        "grasp weld (framed): FixedJoint %s hand=%s <-> obj=%s, in-hand offset t=%s",
        joint_path,
        hand_proxy_path,
        obj_path,
        [round(float(x), 4) for x in t0],
    )
    return jp
