"""Reusable Isaac rendering and signed-distance helpers."""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
import sys
import time
import zipfile

# Resolve imports regardless of cwd: this file's dir (for fpv_* + the new modules) and the hand dir
# (for the grasp math via smplh_anim_core). Done BEFORE any project import.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRY_DIR = os.path.dirname(_THIS_DIR)
_HUMAN_CODE_DIR = os.path.join(_TRY_DIR, "human_model")
_HAND_DIR = os.environ.get("INTERACTIVEWORLD_HUMAN_ASSET_ROOT", "")
for _p in (_THIS_DIR, _HUMAN_CODE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG = logging.getLogger("render_runtime")

DEFAULT_OUT = os.path.join(_TRY_DIR, "fpv_outputs")

# SMPL+H right-arm joints used by the M0 positive-control (force the hand+forearm into the desk).
# 19 = right elbow, 21 = right wrist, 40 = right middle-finger MCP (the hand-capsule direction joint).
H_ELBOW_J = 19
H_WRIST_J = 21
H_RMID_MCP = 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dynamic FPV pick-walk-place (SMPL+H, locked side grip)."
    )
    p.add_argument("--obj", default="cup")
    p.add_argument("--grip", default="side")
    p.add_argument(
        "--stage", default="reach", choices=["reach", "walk", "place", "full"]
    )
    p.add_argument("--scene", default="simple-room", choices=["office", "simple-room"])
    p.add_argument(
        "--stills-only", dest="stills_only", action="store_true", default=False
    )
    p.add_argument(
        "--physics",
        dest="physics",
        action="store_true",
        default=False,
        help="Milestone 0: build the PhysX scene + invisible kinematic proxy rig, run the reach with "
        "the body root height HELD CONSTANT (no dip), log human-vs-scene collision pairs, FixedJoint-"
        "grasp the cup, and physics-settle it on release. Default OFF (the existing path is untouched).",
    )
    p.add_argument(
        "--print-geom", dest="print_geom", action="store_true", default=False
    )
    p.add_argument(
        "--motion",
        default="none",
        choices=["none", "grab"],
        help="Milestone 2: drive the SMPL+H body with REAL retargeted GRAB whole-body mocap "
        "(natural reach->grasp->lift, no splayed arms, no vertical sink). Default 'none' (the "
        "existing scripted side-grip path is untouched). '--motion grab' takes its own setup path.",
    )
    p.add_argument(
        "--grab-seq",
        default=os.path.join(
            _TRY_DIR, "migration_eval", "GRAB", "grab", "s1", "mug_lift.npz"
        ),
        help="GRAB sequence npz for --motion grab (any GRAB s*/<seq>.npz; default s1/mug_lift.npz).",
    )
    p.add_argument(
        "--grab-fps",
        type=int,
        default=24,
        help="Output fps for the retargeted GRAB clip (subsampled from 120 fps source).",
    )
    p.add_argument(
        "--hand-mode",
        dest="hand_mode",
        default="contact",
        choices=["contact", "grab"],
        help="For --motion grab only. 'contact' (default): graft OUR contact-based side-grip fingers "
        "(rig.rhp from grasp_<obj>_side.npz) onto the GRAB reach + ride the cup wrist-relative. 'grab': "
        "a COMPARISON baseline that keeps GRAB's OWN captured hand pose AND its recorded object "
        "trajectory unchanged (no contact solver) -- shows what GRAB mocap alone gives for the grasp.",
    )
    p.add_argument("--res", default="960x760", help="WIDTHxHEIGHT")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--max-frames", type=int, default=320)
    p.add_argument("--capture-fps", type=int, default=12)
    p.add_argument(
        "--obj-offset",
        dest="obj_offset",
        type=float,
        nargs=2,
        default=[0.10, 0.13],
        metavar=("DX", "DY"),
        help="cup-first: the object's rest XY as an OFFSET (m) from the hand's natural grasp landing, on "
        "the desk plane. The GRAB arm is WARPED so the hand reaches it (object-driven). Stay within the "
        "arm reach envelope (the warp reach-guard warns past it). Render several to show the hand FOLLOWS "
        "the object.",
    )
    p.add_argument(
        "--place-tag",
        dest="place_tag",
        default="",
        help="Optional output-filename suffix so multiple --obj-offset placements do not overwrite each "
        "other (e.g. --place-tag right).",
    )
    p.add_argument(
        "--family",
        default="",
        help="Grasp FAMILY template id supplying the seed fingers + seat template (default: side_wrap). "
        "grasp_select picks WRAP/PINCH from geometry at grasp time; this flag only chooses the template assets.",
    )
    p.add_argument(
        "--obj-mesh",
        dest="obj_mesh",
        default="",
        help="P1 any-object: path to an ARBITRARY object mesh (.obj/.ply/.stl/.usd) to grasp. No per-"
        "object grasp file needed -- the seat is geometry-derived. Omit to grasp a known object def.",
    )
    p.add_argument(
        "--seed-grasp",
        dest="seed_grasp",
        action="store_true",
        help="Seed the finger wrap from the retargeted GRAB grasp shape (from --grab-seq) instead of "
        "opening flat -- the family's chosen natural grasp drives the render, contact only tightens the "
        "loose fingers. For cup/box/plate this also defaults --grab-seq to that family's chosen clip.",
    )
    p.add_argument(
        "--seed-wrist",
        dest="seed_wrist",
        action="store_true",
        help="Also align the INITIAL wrist orientation to the family's GRAB grasp (undo the arm-warp "
        "tilt), so the hand's approach matches the chosen exemplar. Compare with/without this flag.",
    )
    p.add_argument(
        "--seed-seat",
        dest="seed_seat",
        action="store_true",
        help="Set the object's seat (obj_T_in_wrist) from the GRAB grasp frame's REAL cup-in-wrist "
        "transform instead of the stored side-wrap seat, so the object sits in the hand exactly as in the "
        "GRAB exemplar. Combine with --seed-grasp/--seed-wrist (and a small --obj-offset) to reproduce it.",
    )
    p.add_argument(
        "--seed-seat-full",
        dest="seed_seat_full",
        action="store_true",
        help="Like --seed-seat but use the FULL GRAB cup-in-wrist (orientation AND the ~24cm reach that "
        "holds the object out in the fingertips). Only sensible when the proxy is the SAME size as the "
        "GRAB object (e.g. --obj-mesh the GRAB mesh); on a smaller proxy the fingers won't reach.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    width, height = (int(x) for x in args.res.lower().split("x"))
    # Milestone 2: a fully separate setup + timeline path driven by retargeted GRAB mocap. Gated so
    # the existing scripted side-grip (default) and the M0 --physics paths below are never disturbed.
    if args.motion == "grab":
        return main_grab(args, width, height)
    if args.obj != "cup" or args.grip != "side":
        LOG.warning(
            "Only --obj cup --grip side is wired; proceeding with the cup side grasp."
        )
    grasp_npz = os.path.join(_HAND_DIR, f"grasp_{args.obj}_side.npz")

    # ---- 1. Import the hand-math bridge BEFORE launch (it imports torch; torch-first is the proven
    # order). smplh_anim_core has no pxr at module scope, so this is launch-safe.
    import smplh_anim_core as core

    rig = core.load_rig(grasp_npz)

    # ---- 2. Build configs + launch Isaac FIRST, then import the pxr/omni-dependent modules. -------
    from fpv_core import AppCfg, RenderCfg

    app_cfg = AppCfg(
        headless=True, width=width, height=height, renderer="RayTracedLighting"
    )
    render_cfg = RenderCfg(
        capture_fps=args.capture_fps,
        playback_fps=args.capture_fps,
        max_frames=args.max_frames,
    )

    import fpv_core

    app = fpv_core.launch_app(app_cfg)
    fpv_core.post_launch_setup(app_cfg, _loco_cfg())

    import fpv_locomotion  # noqa: F401  (plan_path)
    import fpv_render
    import office_scene
    import smplh_camera
    from fpv_segments import run_segment, SegmentCtx

    # Milestone 0 physics plumbing (general, reusable). Imported only here so the module stays
    # import-safe pre-launch; the --physics path uses it, the default path never touches it.
    physics_human = __import__("physics_human") if args.physics else None

    # ---- 3. Stage + geometry + scene + body + cup + camera + lights. ----------------------------
    import numpy as np

    stage = fpv_core.get_stage()

    # Body geometry at the origin (torch FK, no Isaac needed): feet min-z for the stand lift, and the
    # canonical grip-seat offset (the cup's forward-left XY + its body-frame seat height at the chest).
    V0, J0 = core.pose_body(rig, rig.arm_grasp, (0.0, 0.0), 0.0)
    body_min_z = float(V0[:, 2].min())
    canon_seat = core.canonical_seat_local(rig)  # world [dx, dy, seat_z@origin]
    off_xy = (float(canon_seat[0]), float(canon_seat[1]))

    # The body stands at start_xy and walks to end_xy along -Y (native forward). Tables sit under the
    # cup reach points (body XY + the canonical forward-left offset) so the cup rests where it is grasped.
    start_xy = (0.0, 0.0)
    end_xy = (0.0, -2.0)
    far_xy = (
        0.0,
        1.6,
    )  # approach-walk start: ~1.6 m behind, so the person walks up to table A first
    ta_xy = (start_xy[0] + off_xy[0], start_xy[1] + off_xy[1])
    tb_xy = (end_xy[0] + off_xy[0], end_xy[1] + off_xy[1])
    layout = _build_scene(args.scene, office_scene, stage, ta_xy, tb_xy)

    core.author_body_once(stage, rig, V0)
    core.build_object(stage, rig)

    # Cup base offset (its local bbox bottom vs its origin), measured at identity, so we seat the cup's
    # BASE on the desk top (IsaacLab's object-on-surface convention is bottom == table_top, no float).
    import fpv_scene

    mug_min_z = fpv_scene.local_bbox_min_z(stage.GetPrimAtPath("/World/Obj"))

    # One-time penetration backoff (needs the live USD object + trimesh).
    pen_fn = _make_place_and_measure(stage, rig, app)
    core.bake_side_backoff(rig, pen_fn)
    core.derive_rest_place_arms(rig)

    # Stand lift (feet on the floor); the body stays UPRIGHT and the ARM reaches down to the desk.
    stand_z = layout.floor_z - body_min_z

    # Solve the reach-DOWN grasp arm: the arm articulates to the cup on the desk, but the wrist is
    # rotated so the grip + upright cup match the LOCKED side-grip at pickup. cup_local z = table_top
    # - stand_z so that, at the standing body, the grasp seat sits exactly on the desk top.
    cup_local = np.array([off_xy[0], off_xy[1], layout.table_a_top_z - stand_z])
    arm_grasp_table, ik_err, orient_err, wrist_bend = core.solve_grasp_arm(
        rig, cup_local
    )

    # The near-canonical reach lands a little short of the low desk; close the residual with a SMALL
    # body dip (a natural lean) so the cup seats ON the desk without a teleport pop at grasp. dip_z is
    # the body root height at grasp; reach lerps stand_z -> dip_z and the lift lerps back to stand_z.
    _, Jg0 = core.pose_body(rig, arm_grasp_table)  # grasp arm at the origin
    cup_seat_local_z = float(core.seat_object(rig, Jg0, baked=True)[2, 3])
    cup_rest_origin_z = (
        layout.table_a_top_z - mug_min_z
    )  # cup origin so its base contacts the desk top
    dip_z = cup_rest_origin_z - cup_seat_local_z
    LOG.info(
        "grasp-at-table solve: ik_err=%.1fmm orient_err=%.4f wrist_bend=%.1fdeg dip=%.3fm",
        ik_err * 1000.0,
        orient_err,
        wrist_bend,
        stand_z - dip_z,
    )

    # Resting / placed cup poses = the EXACT reach-down grasp seat at the DIPPED body (cup ON the desk,
    # no pop at grasp/release).
    _, Jga = core.pose_body(rig, arm_grasp_table, start_xy, 0.0, root_z=dip_z)
    rest_cup_pos = core.seat_object(rig, Jga, baked=True)[:3, 3].copy()
    _, Jgb = core.pose_body(rig, arm_grasp_table, end_xy, 0.0, root_z=dip_z)
    place_cup_pos = core.seat_object(rig, Jgb, baked=True)[:3, 3].copy()
    LOG.info(
        "geom: body_min_z=%.3f stand_z=%.3f dip_z=%.3f tableA_top=%.3f restA=%s placeB=%s",
        body_min_z,
        stand_z,
        dip_z,
        layout.table_a_top_z,
        rest_cup_pos.round(3).tolist(),
        place_cup_pos.round(3).tolist(),
    )
    # Monotonic straight-line reach path (rest -> ... -> grasp), so the hand moves directly to the cup's
    # side instead of swinging up and over it (a single big axis-angle lerp curved the wrist path).
    reach_path = core.solve_reach_path(rig, rig.arm_rest, arm_grasp_table)
    geom = {
        "start_xy": start_xy,
        "end_xy": end_xy,
        "far_xy": far_xy,
        "stand_z": stand_z,
        "dip_z": dip_z,
        "arm_grasp_table": arm_grasp_table,
        "reach_path": reach_path,
        "rest_cup_pos": rest_cup_pos,
        "place_cup_pos": place_cup_pos,
        "desk_top_z": layout.table_a_top_z,  # M0 settle: compare the rested cup base against this
    }

    cop = smplh_camera.make_fpv_camera(stage, "/World/FpvCam", width, height)
    # External 3/4 verification camera framing table A (to check cup-on-desk, open-hand reach, grip,
    # lift unambiguously -- the FPV can't show the whole body). Reuses the otherwise-wasted 2nd product.
    tpv_cop = smplh_camera.make_tpv_camera(stage, "/World/TpvCam", width, height)
    smplh_camera.drive_tpv(
        tpv_cop, (ta_xy[0] + 2.4, ta_xy[1] + 0.6, 1.55), (ta_xy[0], ta_xy[1], 0.95)
    )
    fpv_core.add_lighting(stage, app_cfg)
    _boost_lights(stage)  # static pipeline's proven dome 700 + distant 1800

    # ---- 3b. Milestone-0 physics plumbing (only when --physics). Build the PhysX scene + colliders +
    # the invisible kinematic proxy rig from the SMPL+H skeleton. Everything here is gated; the default
    # path never reaches it. The proxy is posed from the CONSTANT-height rest J (no dip in the physics
    # path). desk_a_path is the static collider; /World/Obj (the cup) is the dynamic rigid + weld target.
    proxy = None
    if args.physics:
        physics_human.build_physics_scene(stage)
        desk_a_path = "/World/TableA" if args.scene == "simple-room" else "/World/DeskA"
        # Static collider uses the EXACT triangle mesh ("none"), valid for a non-moving body: the desk
        # asset is a single high-poly container whose convexDecomposition leaves gaps (objects tunnel
        # through), so the exact mesh is the reliable, general choice for a static surface.
        physics_human.make_static_collider(
            stage.GetPrimAtPath(desk_a_path), approximation="none"
        )
        physics_human.make_dynamic_rigid(
            stage.GetPrimAtPath("/World/Obj"), density=400.0
        )
        _, J_rest0 = core.pose_body(rig, rig.arm_rest, start_xy, 0.0, root_z=stand_z)
        proxy = physics_human.build_proxy_rig(stage, rig, J_rest0)
        LOG.info(
            "physics M0: scene+colliders+proxy built (desk=%s, cup=/World/Obj, root held at stand_z=%.3f)",
            desk_a_path,
            stand_z,
        )

    # ---- 4. --print-geom: report the standing reach-down grasp geometry + frustum, then exit. ----
    if args.print_geom:
        _print_geom(
            core,
            smplh_camera,
            rig,
            core.seat_object(rig, Jga, baked=True),
            Jga,
            width,
            height,
        )
        app.close()
        return 0

    # ---- 5. Render products + timeline + settle. ------------------------------------------------
    # Pose the standing idle (arm relaxed, cup resting on table A) + aim the camera at the cup before
    # products so the first validated frame is framed (not black / sunk-body / degenerate).
    Vst, Jst = core.pose_body(rig, rig.arm_rest, start_xy, 0.0, root_z=stand_z)
    core.update_body(rig, Vst)
    core.place_obj_resting(rig, rest_cup_pos)
    smplh_camera.drive_fpv(cop, Jst, "idle", rest_cup_pos, 0, 0.0)

    # Single FPV product. setup_products takes two camera paths (FPV + a verification TPV); this demo
    # has only the head camera, so we point BOTH at /World/FpvCam (the TPV product is a harmless
    # duplicate routed to a throwaway dir so the main frame sequence stays clean).
    rp_fpv, rp_tpv, a_fpv, a_tpv = fpv_render.setup_products(
        "/World/FpvCam", "/World/TpvCam", render_cfg, width, height
    )
    timeline = fpv_core.start_timeline(render_cfg.playback_fps)
    fpv_render.settle_and_validate(app, timeline, a_fpv, app_cfg, annot_tpv=None)

    fpv_dir = os.path.join(args.out, f"cup_{args.scene}_{args.stage}_frames")
    tpv_dir = os.path.join(args.out, f"cup_{args.scene}_{args.stage}_tpv_frames")
    ctx = SegmentCtx(
        a_fpv=a_fpv,
        a_tpv=a_tpv,
        fpv_dir=fpv_dir,
        tpv_dir=tpv_dir,
        rcfg=render_cfg,
        dt=1.0 / render_cfg.capture_fps,
        width=width,
        height=height,
    )

    # ---- 6. Run the timeline (the segment sequence for the requested stage). ---------------------
    if args.physics:
        gidx = _run_physics_m0(
            args,
            core,
            physics_human,
            smplh_camera,
            run_segment,
            ctx,
            rig,
            geom,
            cop,
            proxy,
            app,
        )
    else:
        gidx = _run_timeline(
            args, core, smplh_camera, fpv_locomotion, run_segment, ctx, rig, geom, cop
        )

    # ---- 7. Encode (unless stills-only). --------------------------------------------------------
    fpv_render.teardown([rp_fpv, rp_tpv], [a_fpv, a_tpv])
    if not args.stills_only:
        out_mp4 = os.path.join(args.out, f"cup_{args.scene}_{args.stage}_fpv.mp4")
        fpv_render.encode_mp4(fpv_dir, out_mp4, render_cfg)
        tpv_mp4 = os.path.join(args.out, f"cup_{args.scene}_{args.stage}_tpv.mp4")
        fpv_render.encode_mp4(tpv_dir, tpv_mp4, render_cfg)
        LOG.info("DONE. %d frames -> %s (+ TPV %s)", gidx, out_mp4, tpv_mp4)
    else:
        LOG.info(
            "DONE (stills-only). %d frames + boundary stills in %s", gidx, args.out
        )

    app.close()
    return 0


# --------------------------------------------------------------------------------------
# Milestone 2: REAL retargeted GRAB whole-body motion (natural reach->grasp->lift).
# --------------------------------------------------------------------------------------
def main_grab(args, width: int, height: int) -> int:
    """Drive our SMPL+H human with retargeted GRAB mocap on the M0 physics infra. Returns exit code.

    Fully self-contained setup (does NOT touch the scripted side-grip / --physics paths): load + trim
    a GRAB clip, place our office desk under the GRAB mug resting XY at the GRAB resting z, build the
    M0 proxy rig + colliders + dynamic cup, then for each retargeted+subsampled frame build (V,J) from
    the GRAB pose, author/update the body mesh, update the proxy from J, place/weld the cup, and drive
    FPV + TPV. Logs human-vs-desk collision pairs at sampled frames (target ~0). Encodes FPV + TPV mp4.
    """
    # ---- 1. Import the hand-math bridge BEFORE launch (it imports torch; torch-first is the proven
    # order). grab_retarget + smplh_anim_core have no pxr at module scope, so this is launch-safe.
    import grab_retarget as gr
    import numpy as np
    import smplh_anim_core as core

    # FINAL grasp for the 3 tuned families (cup/box/plate): swap in the GRAB-derived grasp (seed the
    # fingers from that family's chosen GRAB clip) AND align the initial wrist to that grasp's orientation,
    # on the UNCHANGED object-follow setup -- the object still rests at its --obj-offset spot and the arm
    # WARPS to reach it. NO seat / object-position change (that would break reaching objects at arbitrary
    # positions, which is a core capability). Other objects keep the plain live-wrap unless flags are set.
    if args.obj in ("cup", "box", "plate"):
        args.seed_grasp = True
        args.seed_wrist = True
    # Proxy swaps: cup -> the TALL GRAB cup (extends above the hand -> a real side-wrap, unlike the short
    # mug); plate -> an UPRIGHT bowl (the YCB bowl USD is authored Y-up so place_obj_resting laid it on its
    # side + sank it; a flat plate disk is ungraspable lying on a desk -> fingers can't reach). The GRAB
    # bowl mesh, oriented Z-up base-pivot + opening-up, sits correctly AND is graspable (the plate family's
    # grasp is the bowl_lift rim grip). Both keep the object-follow setup (rest at --obj-offset, arm warps
    # to it). An explicit --obj-mesh still overrides.
    if not args.obj_mesh:
        _atlas = os.path.join(_TRY_DIR, "human_model", "atlas", "objects")
        if args.obj == "cup":
            args.obj_mesh = os.path.join(_atlas, "grabcup_m.obj")
        elif args.obj == "plate":
            args.obj_mesh = os.path.join(_atlas, "bowl_upright.obj")

    # P1: the grasp is now a SHARED family template (grip: seed fingers + grip_type), the OBJECT supplies
    # only geometry + seat. This drops the per-object grasp file: any object of a given family reuses one
    # template, and an arbitrary mesh needs no grasp file at all (geometry-derived seat). The GRAB clip
    # still supplies the natural body REACH; the live contact-wrap curls the fingers to the real mesh.
    family_id = args.family or "side_wrap"
    rig = core.load_family(family_id)
    if args.obj_mesh:
        # any-object: an arbitrary mesh with NO per-object def -> geometry-derived seat.
        core.set_object(rig, {"shape": "mesh", "mesh": args.obj_mesh})
        LOG.info(
            "[GRAB] P1 any-object: family=%s mesh=%s (geometry-derived seat)",
            family_id,
            args.obj_mesh,
        )
    else:
        # known object: its GEOMETRY + solved seat come from the object def; the grip stays the family's.
        build, otw = core.load_object_def(args.obj)
        core.set_object(rig, build, otw)
        LOG.info(
            "[GRAB] family=%s + object def '%s' (grip from family; per-object grasp file dropped)",
            family_id,
            args.obj,
        )

    # BODY MOTION: always the default one-handed reach clip (s1/mug_lift.npz -- left hand never touches the
    # object). This is the SAME body for every object, so no per-family "drink"/two-handed motion sneaks in.
    # The FAMILY only supplies the RIGHT-HAND grasp (finger pose + wrist orientation RELATIVE TO the object),
    # extracted below from grasp_seq and grafted onto this body -- so only the right hand changes per family.
    grab_seq = args.grab_seq
    grasp_seq = None
    if getattr(args, "seed_grasp", False):
        _default_seq = os.path.join(
            _TRY_DIR, "migration_eval", "GRAB", "grab", "s1", "mug_lift.npz"
        )
        _family_seq = {
            "cup": os.path.join(
                _TRY_DIR, "migration_eval", "GRAB", "grab", "s1", "cup_drink_2.npz"
            ),
            "box": os.path.join(
                _TRY_DIR, "migration_eval", "GRAB", "grab", "s3", "cubelarge_lift.npz"
            ),
            "plate": os.path.join(
                _TRY_DIR, "migration_eval", "GRAB", "grab", "s1", "bowl_lift.npz"
            ),
        }
        if args.grab_seq == _default_seq and args.obj in _family_seq:
            # WORKING STATE: the family clip drives BOTH body + grasp (good grasps). Decoupling the body to a
            # uniform mug_lift and transplanting the family's wrist (route A) OR whole right arm (route B) both
            # CONTORT: the grasp's arm-approach is coupled to the reach, and warping it to the object-follow
            # position distorts it -> ~45-49deg wrist bend, near=1/5. A clean uniform body needs a grasp-aware
            # whole-arm IK (the general-human-motion project). grasp_seq stays None (no decouple).
            grab_seq = _family_seq[args.obj]
            LOG.info(
                "[GRAB] seed-grasp: family '%s' -> clip %s (body+grasp)",
                args.obj,
                grab_seq,
            )
    clip = gr.load_grab(grab_seq)
    trim = gr.trim_and_subsample(clip, fps_out=float(args.grab_fps))
    # Place the human at the world origin (identity yaw); the GRAB scene is already Z-up + grounded.
    place_xy = (0.0, 0.0)
    face_yaw_deg = 0.0
    frames = gr.retarget(clip, place_xy, face_yaw_deg, frames=trim.frames)
    # Map the detected grasp SOURCE frame to the index within our subsampled frame list.
    grasp_pos = _nearest_frame_pos(frames, trim.grasp_src_idx)
    # OPT-IN (--seed-seat): replace the stored object seat with the GRAB grasp frame's REAL cup-in-wrist
    # transform, so the object sits in the hand exactly as in the GRAB exemplar (fixes the ~88deg/184mm
    # seat mismatch that made the render's hand-vs-object geometry unlike the PICK image). Everything
    # downstream (cup_rest_xy, bake_side_backoff, warp_arm_to_seat) then uses the GRAB-consistent seat.
    if getattr(args, "seed_seat", False) or getattr(args, "seed_seat_full", False):
        _gf = frames[grasp_pos]
        _otw_grab = np.linalg.inv(core.wrist_world_T(_gf.J)) @ np.asarray(
            _gf.obj_T_world, dtype=np.float64
        )
        if getattr(args, "seed_seat_full", False):
            # FULL GRAB seat: orientation AND the ~24cm reach (object held out in the fingertips, so a
            # tall object extends above the hand like the exemplar). Needs a same-size proxy to contact.
            _otw = _otw_grab
            _msg = "FULL GRAB cup-in-wrist (orientation + ~24cm fingertip reach)"
        else:
            # Take the GRAB grasp's ORIENTATION (fixes the ~88deg cup-in-hand mismatch -- the real "feel"
            # bug) but KEEP the stored seat's TRANSLATION distance: the GRAB cup center sits ~24cm from the
            # wrist, which lands our smaller proxy too far for the fingers to reach. The stored translation
            # is proxy-sized (contact-good), so this gives the GRAB orientation AND contact on our proxy.
            _otw = _otw_grab.copy()
            _stored = getattr(rig, "obj_T_in_wrist", None)
            if _stored is not None:
                _otw[:3, 3] = np.asarray(_stored, dtype=np.float64)[:3, 3]
            _msg = "GRAB orientation + stored (proxy-sized) translation"
        rig.obj_T_in_wrist = _otw
        rig.obj_T_in_wrist_baked = _otw.copy()  # re-baked below by bake_side_backoff
        LOG.info("[GRAB] seed-seat: obj_T_in_wrist = %s", _msg)
    mug_rest_xy = gr.object_rest_xy(clip)
    mug_rest_z = gr.object_rest_z(clip)
    # The cup's ACTUAL rest XY = the contact side-grip seat at the grasp frame (where OUR retargeted hand
    # ends up holding it). It differs from GRAB's recorded mug XY by the body-shape retarget + the grip
    # offset, so we must seat the desk under THIS point -- otherwise the cup rests just off the desk's
    # front edge (over the carpet) and reads as floating in the FPV. Pure FK (torch), computable pre-launch.
    g_pose = np.asarray(frames[grasp_pos].pose, dtype=np.float32).copy()
    g_pose[core.RHAND_LO : core.RHAND_LO + 45] = (
        rig.rhp
    )  # contact grip at the grasp frame
    _, g_J = core.H.fwd(g_pose)
    g_J = (frames[grasp_pos].rz @ g_J.T).T + frames[grasp_pos].world_off
    cup_rest_xy = tuple(
        float(v) for v in core.seat_object(rig, g_J, baked=True)[:3, 3][:2]
    )
    # cup-first v1: the object rests at an INDEPENDENT desk spot (a fixed visible offset from the hand's
    # natural landing, to PROVE the hand follows the object). The GRAB arm is WARPED so the hand reaches
    # obj_place -- object-driven, not hand-driven. v1 offset is within the arm envelope (the warp reach-
    # guard warns past it). Later: make obj_place a benchmark input (any reachable desk position).
    OBJ_PLACE_OFFSET = tuple(float(v) for v in args.obj_offset)
    obj_place_xy = (
        cup_rest_xy[0] + OBJ_PLACE_OFFSET[0],
        cup_rest_xy[1] + OBJ_PLACE_OFFSET[1],
    )
    LOG.info(
        "[GRAB] clip=%s frames=%d grasp@list[%d](src=%d) mug_rest_xy=%s cup_rest_xy=%s mug_rest_z=%.3f clip=%.2fs",
        os.path.basename(args.grab_seq),
        len(frames),
        grasp_pos,
        trim.grasp_src_idx,
        np.round(mug_rest_xy, 3).tolist(),
        np.round(cup_rest_xy, 3).tolist(),
        mug_rest_z,
        trim.seconds,
    )

    # ---- 2. Build configs + launch Isaac FIRST, then import the pxr/omni-dependent modules. -------
    from fpv_core import AppCfg, RenderCfg

    app_cfg = AppCfg(
        headless=True, width=width, height=height, renderer="RayTracedLighting"
    )
    render_cfg = RenderCfg(
        capture_fps=args.grab_fps,
        playback_fps=args.grab_fps,
        max_frames=max(args.max_frames, len(frames) + 8),
    )

    import fpv_core

    app = fpv_core.launch_app(app_cfg)
    fpv_core.post_launch_setup(app_cfg, _loco_cfg())

    import fpv_render
    import fpv_scene
    import office_scene
    import physics_human as ph
    import smplh_camera
    from fpv_segments import run_segment, SegmentCtx

    stage = fpv_core.get_stage()

    # ---- 3. Scene: place ONE desk so the cup rests squarely ON it and the body stands clear IN FRONT.
    # The GRAB table heading gives the desk yaw; _place_desk_at_grab_table seats the top at mug_rest_z
    # (the GRAB surface) and shifts the desk so its NEAR (+Y, body-side) edge sits a small margin BEHIND
    # the cup -- i.e. the cup is ~0.3 m ONTO the top, not off its front edge -- while the body (~0.5 m
    # further +Y) never reaches the desk solid.
    table_world_xy, table_yaw_deg = gr.table_world_pose(clip, place_xy, face_yaw_deg)
    # ONE desk only (table_b_xy=None): the pick clip needs a single table. Passing the GRAB-table XY for
    # BOTH A and B spawned a second, never-repositioned 2.8 m SM_TableB slab centered on the body's XY --
    # that un-collided slab is what cut through the legs/hips in every frame.
    _build_scene(args.scene, office_scene, stage, table_world_xy, None)
    desk_a_path = "/World/TableA" if args.scene == "simple-room" else "/World/DeskA"
    # Desk stays at the body-CLEAR placement (near edge behind cup_rest_xy, body in front) -- it is long
    # enough to also cover obj_place. (v1: obj_place is a small offset ON this same desk; moving the desk
    # itself toward the body clipped the legs -> keep it put. Later: re-seat the desk per obj_place.)
    _place_desk_at_grab_table(
        stage, fpv_scene, desk_a_path, table_world_xy, cup_rest_xy, mug_rest_z
    )
    # The office desk ships a GLASS top (invisible from the top-down FPV -> the mug looks like it floats
    # over the carpet). Override it opaque so the FPV clearly shows the mug resting on a real surface.
    _make_prim_opaque(stage, desk_a_path)

    # ---- 4. Body + cup geometry (authored from the FIRST retargeted frame). ----------------------
    f0 = frames[0]
    core.author_body_once(stage, rig, f0.V)
    core.build_object(stage, rig)
    core.set_object_world(rig, f0.obj_T_world)

    # ---- 4b. Bake the ONE-TIME side penetration backoff into obj_T_in_wrist_baked (so the cup rides the
    # wrist with the contact-based seat). The bake needs the live USD cup + trimesh, so it runs after
    # build_object. r_canon (canonical side-grip wrist frame) is passed to the per-frame graft ramp.
    pen_fn = _make_place_and_measure(stage, rig, app)
    core.bake_side_backoff(rig, pen_fn)
    r_canon = core.canonical_wrist_orient(rig)

    # ---- 5. Physics plumbing: PhysX scene + static desk collider + dynamic cup. -------------------
    ph.build_physics_scene(stage)
    ph.make_static_collider(stage.GetPrimAtPath(desk_a_path), approximation="none")
    ph.make_dynamic_rigid(stage.GetPrimAtPath("/World/Obj"), density=400.0)
    # PhysX sim + the detector's stage attach/detach stale cached USD handles -> per-frame refetch.
    core.enable_physics_refetch(rig, stage, "/World/Obj", "/World/Body")

    # ---- 5b. Rest the cup ON the desk with a ONE-TIME render=False PhysX settle: drop it just above the
    # realized desk top and let gravity rest it. This is the physically-true resting height -- robust to
    # the mug-pivot / GRAB-origin / desk-bbox mismatch that floated the kinematic cup (the old path just
    # wrote the GRAB mug-CENTER z as the origin). Done BEFORE build_proxy_rig so the cup falls onto the
    # DESK, not a hand capsule. The rested origin z becomes the pre-grasp rest height the graft uses.
    desk_top = float(
        fpv_scene.compute_world_bbox(stage.GetPrimAtPath(desk_a_path)).GetMax()[2]
    )
    rested_z, settle_ok = _presettle_cup_on_desk(
        ph, stage, rig, core, desk_top, obj_place_xy
    )

    # cup-first + LIVE CONTACT-WRAP. (1) WARP the arm so the WRIST reaches the object preserving GRAB's
    # natural hand-object approach; (2) close the SMPL+H fingers to the REAL object surface (no fixed
    # grip, no obj_T_in_wrist, no upright snap) -- the object is grasped at its actual upright resting
    # pose, so the grip fits and the fingers do not penetrate; (3) capture the LIVE cup-in-wrist so the
    # cup rides the wrist exactly during the carry (GRAB-native orientation, no snap).
    obj_place_world = np.array(
        [obj_place_xy[0], obj_place_xy[1], float(rested_z)], dtype=np.float64
    )
    gfr = frames[grasp_pos]
    # place the cup upright at obj_place + build a signed-distance query of its real world mesh.
    T_cup = core.place_obj_resting(rig, obj_place_world)
    # ROUTE B: build the grasp base pose = the fixed mug_lift body with the FAMILY's whole RIGHT-ARM grasp
    # posture (shoulder+elbow+wrist joints + the 45-D fingers) spliced in. Splicing the ARM (not just the
    # wrist) means the forearm APPROACHES the object the family's way, so the wrist sits naturally (no 44deg
    # contortion); warp_arm_to_seat then adjusts shoulder+elbow to actually reach obj_place. The collar and
    # everything else stay mug_lift (and are what the graft ramps from), so torso/left-hand/lift are uniform.
    grasp_base_pose = gfr.pose
    if grasp_seq is not None:
        _gclip = gr.load_grab(grasp_seq)
        _gc_gf = gr.grasp_window(_gclip)
        _gcf = gr.retarget(_gclip, frames=[_gc_gf])[0]
        grasp_base_pose = np.asarray(gfr.pose, dtype=np.float32).copy()
        for _j in (core.R_SHOULDER, core.R_ELBOW, core.WRIST_J):
            grasp_base_pose[_j * 3 : _j * 3 + 3] = _gcf.pose[_j * 3 : _j * 3 + 3]
        grasp_base_pose[core.RHAND_LO : core.RHAND_LO + 45] = _gcf.pose[
            core.RHAND_LO : core.RHAND_LO + 45
        ]
        LOG.info(
            "[GRAB] route-B: spliced family right ARM (shoulder+elbow+wrist+fingers) from %s @%d onto mug_lift body",
            os.path.basename(grasp_seq),
            _gc_gf,
        )
    for _ in range(2):
        app.update()
    sd_world = _make_sd_world(stage, "/World/Obj", cache_key=args.obj)
    if os.environ.get("WRAP_BENCH") == "1":
        import time as _time

        _t = _time.perf_counter()
        _make_sd_world(
            stage, "/World/Obj", cache_key=args.obj
        )  # 2nd call: demonstrates the mesh-hash cache hit
        LOG.info(
            "[wrap] SDF 2nd build for the SAME object took %.3fs (cache hit; 1st paid the voxelize)",
            _time.perf_counter() - _t,
        )
    wrap_masks = core.right_hand_masks()
    # UNIFIED geometric grasp (the single path): pick ONE grip family + a wrist approach frame from the
    # object's own geometry + the body->object approach direction, score a small candidate fan by the
    # REAL seat lever (predicted wrist == where warp_arm_to_seat lands), and realize the best few
    # (warp+orient the arm, live-wrap the fingers to the real mesh). Object-general, no per-object wrist
    # tuning, no VLM. See grasp_select.select_and_grasp.
    import grasp_select

    sel = grasp_select.select_and_grasp(
        core,
        rig,
        grasp_base_pose,  # route B: mug_lift body + family right-ARM grasp posture
        gfr.rz,
        gfr.world_off,
        obj_place_world,
        sd_world,
        wrap_masks,
        seed_grasp=getattr(args, "seed_grasp", False),
        seed_wrist=getattr(args, "seed_wrist", False),
    )
    warped_grasp_pose = sel["pose"]
    warp_err = float(sel["seat_err"])
    wrist_bend = float(sel["wrist_bend"])
    LOG.info(
        "[GRAB] grasp_select: family=%s wrist_bend=%.1fdeg seat_err=%.1fmm n_tried=%d",
        sel["family"],
        sel["wrist_bend"],
        sel["seat_err"] * 1000.0,
        sel.get("n_tried", -1),
    )
    # The selector already contact-wrapped the fingers (with the family's own thumb opposition); reuse
    # its result. open_grasp_pose (arm posed, hand open) is kept only for the graft delta below.
    open_grasp_pose = warped_grasp_pose.copy()
    open_grasp_pose[core.RHAND_LO : core.RHAND_LO + 45] = 0.0
    wrapped_rhp = sel["rhp"]
    wrap_stats = sel["wrap_stats"]

    def _jdelta(j):
        return (
            open_grasp_pose[j * 3 : j * 3 + 3] - gfr.pose[j * 3 : j * 3 + 3]
        ).astype(np.float64)

    d_sh, d_el = _jdelta(core.R_SHOULDER), _jdelta(core.R_ELBOW)
    d_wrist = _jdelta(
        core.WRIST_J
    )  # P3: wrist-orientation delta (GRAB -> family approach), ramped in the graft
    # (3) capture the LIVE cup-in-wrist (cup at obj_place, wrist at the warped+wrapped grasp) -> the cup
    # rides the wrist via seat_object_nosnap during the carry (GRAB-native, no upright forcing).
    wrapped_grasp_pose = open_grasp_pose.copy()
    wrapped_grasp_pose[core.RHAND_LO : core.RHAND_LO + 45] = wrapped_rhp
    Jg = (gfr.rz @ core.H.fwd(wrapped_grasp_pose)[1].T).T + gfr.world_off
    otw_live = np.linalg.inv(core.wrist_world_T(Jg)) @ T_cup
    warp = {
        "obj_place": obj_place_world,
        "d_sh": d_sh,
        "d_el": d_el,
        "d_wrist": d_wrist,
        "rhp": wrapped_rhp,
        "otw": otw_live,
    }
    if warp_err > 0.02:
        LOG.warning(
            "[GRAB] WARP reach-guard: cup seat lands %.1fmm from obj_place (object may be beyond the "
            "arm envelope; v1 = arbitrary-but-REACHABLE).",
            warp_err * 1000.0,
        )
    LOG.info(
        "[GRAB] cup-first + live-wrap: obj_place=%s warp_err=%.1fmm  per-finger (deg/pad_sd_mm/pen_mm)=%s",
        np.round(obj_place_world, 3).tolist(),
        warp_err * 1000.0,
        {k: (v["deg"], v["pad_sd_mm"], v["pen_mm"]) for k, v in wrap_stats.items()},
    )

    proxy = ph.build_proxy_rig(stage, rig, f0.J)

    # ---- 5c. TRUE physics every captured frame. The cup stays DYNAMIC (NOT made kinematic) so PhysX
    # simulates it throughout: it rests on the desk under gravity pre-grasp, is carried by a FixedJoint to
    # the hand at grasp, and its pose is READ from the sim each frame (never written by coordinates). We
    # (1) match the physx step rate to the capture rate, (2) enable REAL contact reports on the desk + cup
    # + every human proxy bone (so human-vs-desk / cup-vs-desk / cup-vs-hand are validated by PhysX, not
    # by AABB math), (3) filter hand<->cup collision so the approaching kinematic hand can't knock the
    # resting cup before the weld, and (4) force deterministic fixed-timestep playback.
    ph.set_scene_timestep(
        stage, 120.0
    )  # 120 Hz physx vs 24 fps capture -> ~5 stable substeps/frame
    contact_paths = [desk_a_path, "/World/Obj"] + [b.path for b in proxy.bones]
    ph.enable_contact_reports(stage, contact_paths)
    hand_r = proxy.hand_paths.get("r")
    if hand_r:
        ph.apply_hand_cup_filter(stage, hand_r, "/World/Obj")
    import carb
    import omni.timeline

    carb.settings.get_settings().set("/app/player/useFixedTimeStepping", True)
    try:
        omni.timeline.get_timeline_interface().set_play_every_frame(True)
    except Exception as exc:  # noqa: BLE001 - determinism nicety; not fatal if the method moved
        LOG.warning(
            "[GRAB] set_play_every_frame unavailable (%s); relying on fixed timestep",
            exc,
        )
    LOG.info(
        "[GRAB] PHYSICS-STEPPED: cup DYNAMIC, physx=120Hz, contact reports on (desk+cup+%d proxy bones), "
        "hand-cup filtered; desk=%s top=%.3f cup rested z=%.3f",
        len(proxy.bones),
        desk_a_path,
        desk_top,
        rested_z,
    )

    # ---- 6. Cameras + lights. -------------------------------------------------------------------
    cop = smplh_camera.make_fpv_camera(stage, "/World/FpvCam", width, height)
    tpv_cop = smplh_camera.make_tpv_camera(stage, "/World/TpvCam", width, height)
    # 3/4 TPV from the body's front-right, aimed at the grip height, close enough to read the side-grip +
    # confirm the body stands clear IN FRONT of the desk every frame. The body faces -Y with its head at
    # ~+0.6 y; we sit the camera in front-right of the chest and look back at the cup-hold height (~0.95).
    smplh_camera.drive_tpv(
        tpv_cop,
        (cup_rest_xy[0] + 1.7, cup_rest_xy[1] + 0.2, 1.25),
        (cup_rest_xy[0] - 0.1, cup_rest_xy[1] - 0.1, 0.95),
    )
    fpv_core.add_lighting(stage, app_cfg)
    _boost_lights(stage)
    # Re-position the office interior RectLights OVER the actual GRAB body+desk world XY so the desk + cup
    # + hand are lit (the default office lights cluster on the table XY only and leave the action a black
    # void). Span from the body head (+Y) across the mug/desk (-Y) at the action XY. (No-op for simple-room.)
    if args.scene == "office":
        _relight_grab(stage, obj_place_xy, float(f0.head[1]))

    # ---- 7. Render products + timeline + validate (frame the first pose first). ------------------
    core.update_body(rig, f0.V)
    ph.update_proxy_rig(proxy, f0.J)
    # cup-first: rest the cup UPRIGHT at the INDEPENDENT obj_place (decoupled from the hand). The GRAB
    # arm is warped so the hand comes HERE; the object stays put until grasped.
    core.place_obj_resting(rig, obj_place_world)
    if not settle_ok:
        # the dynamic drop was unreliable for this object (a sphere rolls/tunnels, a box falls through the
        # triangle desk collider) -> hold it KINEMATIC at the analytic desk seat so it rests visibly on the
        # table through the whole free phase, instead of sinking to the floor and teleporting into the hand.
        ph.set_rigid_kinematic(stage.GetPrimAtPath("/World/Obj"), True)
        LOG.info(
            "[GRAB] pre-settle unreliable -> object held KINEMATIC on the desk (free phase)"
        )
    smplh_camera.drive_fpv_gaze(cop, f0.head, f0.gaze_fwd, obj_place_world)

    rp_fpv, rp_tpv, a_fpv, a_tpv = fpv_render.setup_products(
        "/World/FpvCam", "/World/TpvCam", render_cfg, width, height
    )
    timeline = fpv_core.start_timeline(render_cfg.playback_fps)
    fpv_render.settle_and_validate(app, timeline, a_fpv, app_cfg, annot_tpv=a_tpv)

    # Tag by the GRASP object (args.obj), not the GRAB clip object: cup and can both ride the mug clip,
    # so a clip-name tag made them write the SAME files (the can clobbered the cup). args.obj is unique
    # per render and object-general (no per-object branch). -> cup_grab_office_*, can_grab_office_*.
    tag = f"{args.obj}_grab_{args.scene}"
    if args.place_tag:
        tag += f"_{args.place_tag}"  # distinguish multiple obj-offset placements (no overwrite)
    if args.hand_mode != "contact":
        tag += f"_{args.hand_mode}hand"  # keep the default contact filenames; suffix only the baseline
    fpv_dir = os.path.join(args.out, f"{tag}_frames")
    tpv_dir = os.path.join(args.out, f"{tag}_tpv_frames")
    ctx = SegmentCtx(
        a_fpv=a_fpv,
        a_tpv=a_tpv,
        fpv_dir=fpv_dir,
        tpv_dir=tpv_dir,
        rcfg=render_cfg,
        dt=1.0 / render_cfg.capture_fps,
        width=width,
        height=height,
    )

    # ---- 8. Drive the retargeted frames (REACH->GRASP->LIFT). ------------------------------------
    # Clear any stale PNGs from a previous (e.g. longer/broken) run so the encoded mp4 contains ONLY
    # this run's frames (ffmpeg globs frame_%05d.png sequentially and would otherwise append leftovers).
    _clear_frame_pngs(fpv_dir, tpv_dir)
    gidx = _run_grab_segments(
        args,
        core,
        ph,
        smplh_camera,
        run_segment,
        ctx,
        rig,
        frames,
        grasp_pos,
        cop,
        proxy,
        stage,
        r_canon,
        rested_z,
        warp,
    )

    # ---- 9. Encode. -----------------------------------------------------------------------------
    fpv_render.teardown([rp_fpv, rp_tpv], [a_fpv, a_tpv])
    out_mp4 = os.path.join(args.out, f"{tag}_fpv.mp4")
    tpv_mp4 = os.path.join(args.out, f"{tag}_tpv.mp4")
    if not args.stills_only:
        fpv_render.encode_mp4(fpv_dir, out_mp4, render_cfg)
        fpv_render.encode_mp4(tpv_dir, tpv_mp4, render_cfg)
        LOG.info("[GRAB] DONE. %d frames -> %s (+ TPV %s)", gidx, out_mp4, tpv_mp4)
    else:
        LOG.info("[GRAB] DONE (stills-only). %d frames in %s", gidx, args.out)

    app.close()
    return 0


def _graft_sidegrip_frames(
    core,
    rig,
    frames,
    grasp_pos,
    r_canon,
    rest_z: float,
    warp: dict | None = None,
    ramp_len: int = 6,
    hand_mode: str = "contact",
):
    """Pre-compute, per GRAB frame, the side-grip-GRAFTED (V, J, cup_T, c).

    GENERAL contact-based graft: the GRAB clip gives the natural body REACH; at contact we replace ONLY
    the wrist orientation + the fingers with THIS object's contact-based side-grip (rig.rhp + R_canon
    from grasp_<obj>_side.npz), then ride the cup on the wrist via seat_object (obj_T_in_wrist_baked).

    Per frame i, a 0->1 ramp ``c`` rises over the last ``ramp_len`` frames ending at the grasp frame and
    is held at 1 after. We post-process fr.pose:
      * wrist joint = geodesic interpolation GRAB-wrist (c=0) -> solve_wrist_orient(...,R_canon) (c=1),
      * fingers     = linear interpolation GRAB right-hand-aa (c=0) -> rig.rhp (c=1),
    re-FK the grafted pose and re-apply the SAME fr.rz / fr.world_off to get the placed (V, J).

    CUP placement (always UPRIGHT, no teleport pop):
      * c == 0 (pre-grasp): the cup rests on the DESK at the grasp-seat XY, top at ``rest_z`` -- the GRAB
        surface where it actually sits before pickup,
      * 0 < c < 1 (closing): the cup position lerps from that desk rest -> the wrist-relative seat, so it
        lifts smoothly out of the desk into the hand as the fingers close,
      * c == 1 (held): the cup rides the wrist (tight in hand).
    Returns (grafted list of (V,J,cup_T,c), wrist_bend_deg_at_grasp). Object-general: everything derives
    from rig + R_canon (no cup constants).
    """
    import numpy as np

    H = core.H
    n = len(frames)
    g0 = max(0, grasp_pos - (ramp_len - 1))  # first frame of the ramp

    def _c(i: int) -> float:
        if i <= g0:
            return 0.0
        if i >= grasp_pos:
            return 1.0
        return float(i - g0) / float(max(1, grasp_pos - g0))

    def _graft_pose(fr, c: float) -> np.ndarray:
        pose = np.asarray(fr.pose, dtype=np.float32).copy()
        if c <= 0.0 or hand_mode == "grab":
            # 'grab' BASELINE: keep GRAB's OWN captured hand pose entirely (no contact-based rig.rhp).
            return pose
        # cup-first WARP: ramp the one-time grasp-frame joint delta (shoulder+elbow, from
        # warp_arm_to_seat) in with a smoothstep so the GRAB arm travels naturally and ARRIVES at the
        # resting cup (obj_place) by the grasp frame -- object-driven, not hand-driven.
        if warp is not None:
            sm = c * c * (3.0 - 2.0 * c)  # smoothstep ramp
            sh0, el0 = core.R_SHOULDER * 3, core.R_ELBOW * 3
            pose[sh0 : sh0 + 3] = pose[sh0 : sh0 + 3] + (warp["d_sh"] * sm).astype(
                np.float32
            )
            pose[el0 : el0 + 3] = pose[el0 : el0 + 3] + (warp["d_el"] * sm).astype(
                np.float32
            )
            if (
                "d_wrist" in warp
            ):  # P3: ramp the wrist GRAB -> family approach orientation
                w0 = core.WRIST_J * 3
                pose[w0 : w0 + 3] = pose[w0 : w0 + 3] + (warp["d_wrist"] * sm).astype(
                    np.float32
                )
        # Close the fingers GRAB-open -> the contact-based grip (rig.rhp), at GRAB's own (now warped)
        # wrist orientation; the fixed side-grip wrist frame is NOT forced here.
        grab_rh = pose[core.RHAND_LO : core.RHAND_LO + 45].astype(np.float64)
        rhp_t = np.asarray(
            warp["rhp"] if (warp is not None and "rhp" in warp) else rig.rhp,
            dtype=np.float64,
        )  # the LIVE-wrapped fingers (object-general); falls back to the offline grip
        pose[core.RHAND_LO : core.RHAND_LO + 45] = grab_rh + (rhp_t - grab_rh) * c
        return pose

    def _place(fr, pose):
        V, J = H.fwd(pose)  # body-local (pelvis at origin), our Z-up world
        return (fr.rz @ V.T).T + fr.world_off, (fr.rz @ J.T).T + fr.world_off

    # wrist_bend is informational only (how far the GRAB wrist is from the canonical side-grip frame);
    # the live-wrap path does NOT force it -- the cup rides the wrist as-grasped (seat_object_nosnap).
    _, _, wrist_bend = core.solve_wrist_orient(rig, frames[grasp_pos].pose, r_canon)

    out = []
    for i in range(n):
        c = _c(i)
        Vw, Jw = _place(frames[i], _graft_pose(frames[i], c))
        # cup_T=None on purpose: on_frame seats the cup LIVE (dynamic on the desk pre-grasp; the live
        # cup-in-wrist via seat_object_nosnap once held). The old precomputed upright-snap cup_T is unused.
        out.append((Vw, Jw, None, c))
    return out, None, float(wrist_bend)


def _run_grab_segments(
    args,
    core,
    ph,
    cam_mod,
    run_segment,
    ctx,
    rig,
    frames,
    grasp_pos,
    cop,
    proxy,
    stage,
    r_canon,
    rest_z,
    warp=None,
) -> int:
    """Play the GRAB reach with a PHYSICS-DRIVEN cup. Returns gidx.

    The body (visible mesh + kinematic proxy colliders) is written from mocap each frame BEFORE the single
    render/physics step; the CUP is a DYNAMIC rigid body whose pose is READ from PhysX AFTER the step (never
    written by coordinates). It rests on the desk under gravity pre-grasp; at the contact frame a framed
    FixedJoint welds it to the (kinematic) hand so the hand carries it. Validation is REAL PhysX contact
    reports read each step (human-vs-desk penetration, cup-vs-desk rest, cup carried), not geometry.
    """
    import numpy as np

    grafted, _rest_pos, wrist_bend = _graft_sidegrip_frames(
        core,
        rig,
        frames,
        grasp_pos,
        r_canon,
        rest_z,
        warp=warp,
        hand_mode=args.hand_mode,
    )
    LOG.info(
        "[GRAB] PHYSICS side-grip: fingers ramp GRAB-open -> rig.rhp at grasp (wrist re-orient bend "
        "%.1f deg); the cup is DYNAMIC (rests + is carried by physics, pose read from the sim each frame).",
        wrist_bend,
    )
    desk_a_path = "/World/TableA" if args.scene == "simple-room" else "/World/DeskA"
    saver = _StillSaver(
        ctx,
        args.out,
        f"{args.obj}_grab" + (f"_{args.place_tag}" if args.place_tag else ""),
    )  # per-object/placement kf prefix (no clobber across placements)
    sample_at = _sample_indices(len(frames))
    beats = {
        "idle": 0,
        "reach": max(0, grasp_pos - 4),
        "grasp": grasp_pos,
        "lift": len(frames) - 1,
    }
    beat_gidx: dict[str, int] = {}
    state = {
        "last_cup_ctr": np.asarray(_read_cup_pos(stage, core, rig), dtype=np.float64),
        "held": False,
        "grasp_gidx": None,
        "kinematic": False,  # True once /World/Obj has been switched kinematic (at the grasp frame)
    }
    clog: list[tuple] = []  # per frame: (gidx, c, cup_z, classify_contacts dict)
    # Per-frame object world-pose trace (numerical continuity check). Written from the REALIZED origin.
    # Per-object path so concurrent cup/can renders never clobber each other.
    csv_path = "/tmp/posetrace_%s.csv" % args.obj
    csv_f = open(csv_path, "w")
    csv_f.write("frame,c,x,y,z\n")

    def on_frame(t: float, gidx: int) -> None:
        # Runs BEFORE the single render/physics step. STATE MACHINE (the HAND goes to the cup, not the
        # cup to the hand):
        #  * reach + finger-close (c<1): the cup is DYNAMIC and stays PUT on the desk under gravity. The
        #    GRAB arm reaches to it and the contact fingers close around it (hand<->cup collision filtered,
        #    so the closing hand never knocks the resting cup). The cup does NOT move to meet the hand.
        #  * grasp + lift (c>=1): only NOW the cup attaches -- switched KINEMATIC and driven to cup_T =
        #    the LOCKED holding state (seat_object: GRAB wrist + contact fingers + upright), so the hand
        #    carries it up. It stays a collider, so it still participates in PhysX detection.
        i = min(int(round(t * (len(frames) - 1))), len(frames) - 1)
        fr = frames[i]
        Vw, Jw, cup_T, c = grafted[i]
        core.update_body(rig, Vw)
        ph.update_proxy_rig(proxy, Jw)
        # cup-first attach: the cup STAYS DYNAMIC at obj_place through the whole reach (c<1) -- the WARPED
        # hand goes to it, the cup never moves to meet the hand. Only at the grasp frame (c>=1) does it
        # attach: flip kinematic + drive to seat_object(Jw). Because the warp lands seat_object(grasp Jw)
        # AT obj_place (where the cup already rests), the attach is seamless (no snap, no bridge wiggle);
        # the cup then rides the warped wrist up for the lift.
        if c >= 1.0:
            if not state["kinematic"]:
                ph.set_rigid_kinematic(stage.GetPrimAtPath("/World/Obj"), True)
                state["kinematic"] = True
                state["held"], state["grasp_gidx"] = True, gidx
            otw = warp.get("otw") if warp else None
            obj_mat = np.asarray(
                core.seat_object_nosnap(rig, Jw, otw)
                if otw is not None
                else core.seat_object(rig, Jw, baked=True),
                dtype=np.float64,
            )  # cup rides the wrist at the LIVE grasp-relative pose (no upright snap)
            core.set_object_world(rig, obj_mat)
            state["last_cup_ctr"] = np.asarray(obj_mat[:3, 3], dtype=np.float64)
        cam_mod.drive_fpv_gaze(
            cop,
            fr.head,
            fr.gaze_fwd,
            state["last_cup_ctr"],
            aim_blend=0.9 if c >= 1.0 else 0.6,
        )
        for name, pos in beats.items():
            if i == pos and name not in beat_gidx:
                beat_gidx[name] = gidx

    def on_post_step(t: float, gidx: int) -> None:
        # Runs IMMEDIATELY after the single render/physics step: read the PhysX contact buffer (valid only
        # for the step just taken). While resting (c<=0) the cup is DYNAMIC, so read its SIMULATED pose
        # from physics (proves it truly sits on the desk, not a coordinate write); while held it is the
        # kinematic seat pose already recorded in on_frame.
        i = min(int(round(t * (len(frames) - 1))), len(frames) - 1)
        c = float(grafted[i][3])
        if not state[
            "kinematic"
        ]:  # cup is DYNAMIC only while c<=0 -> read its rested (sim) pose
            state["last_cup_ctr"] = np.asarray(
                _read_cup_pos(stage, core, rig), dtype=np.float64
            )
        # Per-frame REALIZED object world-pose trace (continuity check): dynamic read pre-bridge,
        # bridged/seat pose during+after. Written from state["last_cup_ctr"] (the realized origin).
        csv_f.write(
            "%d,%.4f,%.5f,%.5f,%.5f\n"
            % (
                gidx,
                c,
                float(state["last_cup_ctr"][0]),
                float(state["last_cup_ctr"][1]),
                float(state["last_cup_ctr"][2]),
            )
        )
        headers, data = ph.read_contacts()
        cc = ph.classify_contacts(
            headers,
            data,
            human_root="/World/HumanProxy",
            obj_path="/World/Obj",
            desk_path=desk_a_path,
        )
        clog.append((gidx, c, float(state["last_cup_ctr"][2]), cc))
        if i in sample_at:
            LOG.info(
                "[PHYS frame %d c=%.2f] cup_z=%.3f  human-desk pairs=%d min_sep=%.1fmm  cup-desk=%s cup-hand=%s",
                gidx,
                c,
                state["last_cup_ctr"][2],
                cc["n_human_desk"],
                cc["min_sep_human_desk"] * 1000.0,
                cc["cup_desk"],
                cc["cup_hand"],
            )

    gidx = run_segment(
        ctx,
        "grab_reach_grasp_lift",
        len(frames),
        on_frame,
        0,
        ctx.rcfg.max_frames,
        on_post_step=on_post_step,
    )
    for name, g in beat_gidx.items():
        saver.maybe_kf(name, g + 1)
        saver.maybe_kf_tpv(name, g + 1)
    # ---- REAL PhysX contact-based validation (read every step from the simulation, not from geometry).
    # This is the "每个物体用物理引擎校验" the user asked for: PhysX itself reports human-vs-desk contact,
    # the cup resting on the desk, and the cup being carried. ----
    worst_sep = min((cc["min_sep_human_desk"] for (_, _, _, cc) in clog), default=0.0)
    pen = [g for (g, _, _, cc) in clog if cc["pen"]]
    pre_z = [z for (_, c, z, cc) in clog if c <= 0.0]
    cup_desk_pre = sum(1 for (_, c, _, cc) in clog if c <= 0.0 and cc["cup_desk"])
    held_z = [z for (_, c, z, cc) in clog if c >= 1.0]
    if pen:
        LOG.error(
            "[PHYS CHECK] PROBLEM 1: human-vs-desk PENETRATION on %d frame(s) (worst min_sep=%.1fmm) "
            "frames=%s -- fix the placement/standoff!",
            len(pen),
            worst_sep * 1000.0,
            pen[:6],
        )
    else:
        LOG.info(
            "[PHYS CHECK] problem 1 OK (PhysX contact-based): human-vs-desk penetration-free across %d "
            "frames (worst min_sep=%.1fmm; >= 0 means a clear gap, the engine confirms no overlap)",
            len(clog),
            worst_sep * 1000.0,
        )
    if pre_z:
        LOG.info(
            "[PHYS CHECK] problem 2 (cup RESTS by physics): pre-grasp cup z (READ from the DYNAMIC sim) in "
            "[%.3f,%.3f] -- a free rigid body that stays put == it really sits on the desk (a fallen cup "
            "would drop to ~0); cup-vs-desk contact events on %d/%d frames",
            min(pre_z),
            max(pre_z),
            cup_desk_pre,
            len(pre_z),
        )
    if len(held_z) >= 2:
        LOG.info(
            "[PHYS CHECK] problem 3 (cup carried, held state UNCHANGED): held frames=%d cup z %.3f->%.3f "
            "(rose %.3fm) via the locked seat_object grip (kinematic collider -- still PhysX-detected).",
            len(held_z),
            held_z[0],
            held_z[-1],
            held_z[-1] - held_z[0],
        )
    csv_f.flush()
    csv_f.close()
    LOG.info("[GRAB] wrote per-frame object world-pose trace -> %s", csv_path)
    return gidx


def _nearest_frame_pos(frames, src_idx: int) -> int:
    """Index into ``frames`` whose src_idx is closest to ``src_idx`` (maps a source grasp frame)."""
    best, bd = 0, 1 << 30
    for k, fr in enumerate(frames):
        d = abs(int(fr.src_idx) - int(src_idx))
        if d < bd:
            best, bd = k, d
    return best


def _sample_indices(n: int) -> set[int]:
    """A handful of frame indices to probe collisions at (start, ~quarters, end)."""
    if n <= 1:
        return {0}
    return {0, n // 4, n // 2, (3 * n) // 4, n - 1}


def _place_desk_at_grab_table(
    stage,
    fpv_scene,
    desk_path: str,
    table_xy,
    cup_xy,
    top_z: float,
    front_margin: float = 0.32,
) -> None:
    """Seat the desk so its TOP is at ``top_z`` and the CUP sits ``front_margin`` ONTO the top (not off
    its front edge), with the body standing clear IN FRONT (+Y) of it.

    The body faces -Y (pelvis at +Y, cup/desk at -Y). SM_TableB's long axis is along Y (2.8 m deep);
    centered on the table XY it overlaps the body and fills the FPV with a deep slab. We therefore:
    (1) yaw it 90deg so the long axis is along X (shallow front-to-back, wide left-right), (2) raise it
    so its world-bbox top == top_z (the GRAB surface), (3) shift it in Y so its NEAR (max-Y, body-side)
    edge sits ``front_margin`` BEHIND the cup -- i.e. the cup is ``front_margin`` m onto the top. The cup
    XY is the realized side-grip seat (where OUR hand holds it), so the cup rests squarely on the desk;
    the body (a further ~0.5 m +Y) never reaches the desk solid. No per-asset constant.
    """
    prim = stage.GetPrimAtPath(desk_path)
    yaw_wxyz = _yaw_wxyz(90.0)  # rotate the long axis from Y to X
    # Raise to top_z at the table XY first (with the rotation applied so the bbox is the rotated one).
    fpv_scene.set_world_pose(
        prim, (float(table_xy[0]), float(table_xy[1]), 0.0), yaw_wxyz
    )
    rng = fpv_scene.compute_world_bbox(prim)
    cur_top = float(rng.GetMax()[2])
    cur_min = float(rng.GetMin()[2])
    base_z = float(top_z) - cur_top + cur_min if cur_top > cur_min else float(top_z)
    fpv_scene.set_world_pose(
        prim, (float(table_xy[0]), float(table_xy[1]), base_z), yaw_wxyz
    )
    # Measure the seated (rotated) desk bbox; shift in Y so its NEAR (+Y) edge sits front_margin BEHIND
    # the cup, AND center it in X under the cup, so the cup rests squarely ON the top.
    rng2 = fpv_scene.compute_world_bbox(prim)
    near_edge_y = float(rng2.GetMax()[1])
    target_near_edge_y = float(cup_xy[1]) + float(front_margin)
    new_y = float(table_xy[1]) + (target_near_edge_y - near_edge_y)
    new_x = float(cup_xy[0])
    fpv_scene.set_world_pose(prim, (new_x, new_y, base_z), yaw_wxyz)
    rng3 = fpv_scene.compute_world_bbox(prim)
    LOG.info(
        "[GRAB] desk %s seated xy=(%.3f,%.3f) yaw90: top->%.3f (target %.3f) near(+Y)edge->%.3f "
        "(cup y=%.3f, %.0fmm onto top) bbox x[%.2f,%.2f] y[%.2f,%.2f]",
        desk_path,
        new_x,
        new_y,
        float(rng3.GetMax()[2]),
        float(top_z),
        float(rng3.GetMax()[1]),
        float(cup_xy[1]),
        front_margin * 1000.0,
        float(rng3.GetMin()[0]),
        float(rng3.GetMax()[0]),
        float(rng3.GetMin()[1]),
        float(rng3.GetMax()[1]),
    )


def _yaw_wxyz(deg: float) -> tuple[float, float, float, float]:
    """(w,x,y,z) quaternion for a yaw of ``deg`` about +Z (for fpv_scene.set_world_pose)."""
    import math

    a = math.radians(float(deg)) * 0.5
    return (math.cos(a), 0.0, 0.0, math.sin(a))


def _make_prim_opaque(
    stage, root_path: str, color: tuple[float, float, float] = (0.38, 0.32, 0.27)
) -> int:
    """Bind an OPAQUE UsdPreviewSurface to every Mesh under ``root_path``, overriding the asset's own
    material. The office desk (SM_TableB) ships a GLASS/transparent top -- from the top-down FPV that
    top is invisible, so a mug resting on it LOOKS like it floats over the carpet. A local opaque
    binding (authored in the root layer -> stronger than the referenced binding) makes the surface read
    solid so the FPV clearly shows the mug on a real tabletop. Returns the number of meshes rebound."""
    from pxr import Sdf, Usd, UsdShade

    mtl = UsdShade.Material.Define(stage, root_path + "/OpaqueMat")
    sh = UsdShade.Shader.Define(stage, root_path + "/OpaqueMat/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if p.GetTypeName() == "Mesh":
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                mtl, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
            n += 1
    LOG.info("[GRAB] desk %s -> opaque material bound to %d meshes", root_path, n)
    return n


def _clear_frame_pngs(*dirs: str) -> None:
    """Delete frame_*.png in each dir so a shorter re-render never leaves STALE frames behind. ffmpeg
    globs frame_%05d.png sequentially, so leftover higher-numbered frames from a previous (longer, e.g.
    broken) run would be appended to the new mp4 -- exactly what made the fixed clip look broken at the
    tail. Clearing the per-frame PNGs (not the dir) before each render makes the output deterministic."""
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("frame_") and f.endswith(".png"):
                os.remove(os.path.join(d, f))


# --------------------------------------------------------------------------------------
# Timeline.
# --------------------------------------------------------------------------------------
def _run_timeline(args, core, cam_mod, loco, run_segment, ctx, rig, geom, cop) -> int:
    """Run the approach / pick / walk / place segments for --stage. Returns the final frame index.

    The body stands UPRIGHT on the floor and the hand is EXTENDED during the approach -- the locked
    side-grip (curled fingers + the wrist rotation) engages only at CONTACT (the grasp segment). Beats:
      approach: walk up to table A, arm hanging (open hand, mostly out of the FPV)
      idle:     stand at the table, cup resting on it
      reach:    OPEN hand (straight wrist) moves to the cup + a small lean
      grasp:    fingers CLOSE + wrist rotates into the side-grip on the cup (cup still on the desk)
      lift:     the side-grip carry rises to the chest (cup now rides the wrist)
      walk/place/release/retract: carry A->B, lower, open the hand, return to rest
    `hand_close` (0=open, 1=grip) animates in grasp/release; the arm LERPs (incl. the wrist joint).
    """
    from fpv_core import smoothstep

    sz, dz = geom["stand_z"], geom["dip_z"]
    start_xy, end_xy, far_xy = geom["start_xy"], geom["end_xy"], geom["far_xy"]
    agt, rp = geom["arm_grasp_table"], geom["reach_path"]
    open_grasp = rp[
        -1
    ]  # grasp shoulder/elbow with a STRAIGHT wrist (= the open-hand reach endpoint)
    rest_cup_pos, place_cup_pos = geom["rest_cup_pos"], geom["place_cup_pos"]
    speed, fps = _loco_cfg().speed_mps, ctx.rcfg.capture_fps

    def _dip(t):
        return sz + (dz - sz) * smoothstep(
            t
        )  # body root: stand -> dip (a small lean) over the reach

    def _rise(t):
        return dz + (sz - dz) * smoothstep(t)  # body root: dip -> stand over the lift

    def _path_pose(s):
        u = max(0.0, min(1.0, s)) * (len(rp) - 1)
        k = min(int(u), len(rp) - 2)
        return core.lerp_arm(rp[k], rp[k + 1], u - k)

    def frame_pose(arm, root_xy, mode, frame, rz, hand_close, attach, cup_pos):
        """Pose body (root rz, finger curl hand_close); attach the cup to the wrist OR rest it; drive FPV."""
        V, J = core.pose_body(rig, arm, root_xy, 0.0, root_z=rz, hand_close=hand_close)
        core.update_body(rig, V)
        if attach:
            aim = core.attach_obj(rig, J)[:3, 3]
        else:
            core.place_obj_resting(rig, cup_pos)
            aim = cup_pos
        cam_mod.drive_fpv(cop, J, mode, aim, frame, 0.0)
        return aim

    saver = _StillSaver(
        ctx, args.out, "pick" if args.stage == "reach" else f"cup_{args.stage}"
    )
    gidx = 0
    maxf = ctx.rcfg.max_frames
    want = _stages_for(args.stage)

    if "approach" in want:
        wps = loco.plan_path(far_xy, start_xy, speed, fps)
        n_ap = len(wps)

        def f_approach(t, frame):
            wi = min(int(round(t * (n_ap - 1))), n_ap - 1)
            frame_pose(
                rig.arm_rest, wps[wi], "walk", frame, sz, 0.0, False, rest_cup_pos
            )

        gidx = run_segment(ctx, "approach", n_ap, f_approach, gidx, maxf)
        saver.maybe_kf("approach", gidx)

    if "idle" in want:

        def f_idle(t, frame):
            mode = "idle" if t < 0.5 else "action"
            frame_pose(
                rig.arm_rest, start_xy, mode, frame, sz, 0.0, False, rest_cup_pos
            )

        gidx = run_segment(ctx, "idle", 8, f_idle, gidx, maxf)
        saver.maybe_kf("idle", gidx)

    if "reach" in want:

        def f_reach(t, frame):
            arm = _path_pose(
                smoothstep(t)
            )  # OPEN hand, straight wrist, monotonic side approach
            frame_pose(
                arm, start_xy, "action", frame, _dip(t), 0.0, False, rest_cup_pos
            )

        gidx = run_segment(ctx, "reach", 16, f_reach, gidx, maxf)
        saver.maybe_kf("reach", gidx)

    if "grasp" in want:

        def f_grasp(t, frame):
            s = smoothstep(t)
            arm = core.lerp_arm(
                open_grasp, agt, s
            )  # wrist rotates straight -> grip; cup still on desk
            frame_pose(
                arm, start_xy, "action", frame, dz, s, False, rest_cup_pos
            )  # fingers close 0->1

        gidx = run_segment(ctx, "grasp", 8, f_grasp, gidx, maxf)
        saver.maybe_kf("grasp", gidx)

    if "lift" in want:

        def f_lift(t, frame):
            arm = core.lerp_arm(
                agt, rig.arm_grasp, smoothstep(t)
            )  # side-grip carry rises to the chest
            mode = "walk" if t > 0.4 else "action"
            frame_pose(arm, start_xy, mode, frame, _rise(t), 1.0, True, None)

        gidx = run_segment(ctx, "lift", 14, f_lift, gidx, maxf)
        saver.maybe_kf("lift", gidx)

    if "walk" in want:
        wps = loco.plan_path(start_xy, end_xy, speed, fps)
        n_w = len(wps)

        def f_walk(t, frame):
            wi = min(int(round(t * (n_w - 1))), n_w - 1)
            frame_pose(
                rig.arm_grasp, wps[wi], "walk", frame, sz, 1.0, True, None
            )  # chest carry

        gidx = run_segment(ctx, "walk", n_w, f_walk, gidx, maxf)
        saver.maybe_kf("walk", gidx)

    if "place" in want:

        def f_place(t, frame):
            arm = core.lerp_arm(
                rig.arm_grasp, agt, smoothstep(t)
            )  # lower the side-grip onto table B
            frame_pose(arm, end_xy, "action", frame, _dip(t), 1.0, True, None)

        gidx = run_segment(ctx, "place", 18, f_place, gidx, maxf)
        saver.maybe_kf("place", gidx)

    if "release" in want:

        def f_release(t, frame):
            s = smoothstep(t)
            arm = core.lerp_arm(
                agt, open_grasp, s
            )  # wrist relaxes grip -> straight as the hand opens
            frame_pose(
                arm, end_xy, "action", frame, dz, 1.0 - s, False, place_cup_pos
            )  # cup rests on B

        gidx = run_segment(ctx, "release", 10, f_release, gidx, maxf)
        saver.maybe_kf("release", gidx)

    if "retract" in want:

        def f_retract(t, frame):
            arm = _path_pose(
                1.0 - smoothstep(t)
            )  # reverse the open-hand reach back to the rest pose
            mode = "idle" if t > 0.6 else "action"
            frame_pose(arm, end_xy, mode, frame, _rise(t), 0.0, False, place_cup_pos)

        gidx = run_segment(ctx, "retract", 12, f_retract, gidx, maxf)
        saver.maybe_kf("retract", gidx)

    return gidx


# --------------------------------------------------------------------------------------
# Milestone-0 PHYSICS sequence: prove the physics/collision/grasp plumbing. The body root
# height is HELD CONSTANT (no dip) the whole time -- this proves the REACH mechanism without
# the vertical sink. We (1) log the human-vs-desk collision pairs at rest (expect ~0), (2) as a
# positive control force the OLD sinking/penetrating pose once and log that pairs ARE detected,
# (3) FixedJoint-grasp the cup to the hand proxy, (4) on release delete the joint and let physics
# settle the cup onto the desk, logging the settle frames + final cup-base height vs the desk top.
# --------------------------------------------------------------------------------------
def _run_physics_m0(
    args, core, ph, cam_mod, run_segment, ctx, rig, geom, cop, proxy, app
) -> int:
    """Run the M0 physics demonstration. Returns the final frame index.

    Reuses the EXISTING arm IK / pose path (rig.arm_rest, the reach_path, the canonical grasp arm)
    but pins root_z = stand_z for every frame (no dip). The proxy rig is updated from J each frame
    so the penetration detector + the FixedJoint anchor track the real body.
    """
    import numpy as np
    from fpv_core import smoothstep

    stage = core_stage(
        core, rig
    )  # live USD stage (hoisted: used by the detector + weld + settle)
    # PhysX simulation (RTX warmup + per-frame steps) and the collision detector's stage attach/detach
    # + session-layer edits STALE cached USD schema handles -> turn ON per-frame handle re-acquisition
    # for the cup writes, the body mesh, and the proxy ops, so every .Set uses a live handle ('Accessed
    # schema on invalid prim' otherwise). The non-physics path never flips this (its fast path stands).
    core.enable_physics_refetch(rig, stage, "/World/Obj", "/World/Body")
    sz, dz = geom["stand_z"], geom["dip_z"]
    start_xy = geom["start_xy"]
    agt, rp = geom["arm_grasp_table"], geom["reach_path"]
    open_grasp = rp[-1]
    rest_cup_pos = geom["rest_cup_pos"]
    saver = _StillSaver(ctx, args.out, "phys")

    def _pose_update(arm, frame, mode, hand_close, attach, cup_pos):
        """Pose body at CONSTANT stand_z, update the proxy from J, attach/rest the cup, drive FPV."""
        V, J = core.pose_body(rig, arm, start_xy, 0.0, root_z=sz, hand_close=hand_close)
        core.update_body(rig, V)
        ph.update_proxy_rig(proxy, J)
        if attach:
            aim = core.attach_obj(rig, J)[:3, 3]
        else:
            core.place_obj_resting(rig, cup_pos)
            aim = cup_pos
        cam_mod.drive_fpv(cop, J, mode, aim, frame, 0.0)
        return J, aim

    gidx = 0
    maxf = ctx.rcfg.max_frames

    # ---- (1) REST: pose idle at constant height, update proxy, log human-vs-desk pairs (~0). ------
    Vr, Jr = core.pose_body(rig, rig.arm_rest, start_xy, 0.0, root_z=sz)
    core.update_body(rig, Vr)
    ph.update_proxy_rig(proxy, Jr)
    core.place_obj_resting(rig, rest_cup_pos)
    rest_pairs = ph.human_scene_collision_pairs(stage=stage)
    LOG.info(
        "[M0 DETECTOR] REST pose (root held at stand_z=%.3f): human-vs-scene pairs = %d  %s",
        sz,
        len(rest_pairs),
        rest_pairs,
    )

    # ---- (2) POSITIVE CONTROL: force the OLD sinking/penetrating configuration and confirm the
    # detector NOW reports pairs (>0). The old pipeline's bug was the body/hand SINKING into the
    # surface; we reproduce that penetration directly by driving the right-hand + forearm proxies to a
    # point INSIDE the desk solid (just under its top). To isolate human-vs-DESK (not human-vs-cup) we
    # park the cup far away during this probe, then restore it. This proves the detector fires on a real
    # human-vs-desk overlap, independent of any particular arm pose or desk height.
    desk_prim = stage.GetPrimAtPath(
        "/World/TableA" if args.scene == "simple-room" else "/World/DeskA"
    )
    desk_inside_pt, desk_z_min, desk_z_max = _desk_bbox(desk_prim)
    core.place_obj_resting(
        rig, np.array([10.0, 10.0, desk_z_max])
    )  # park the cup far away for the probe
    Jpen = Jr.copy()
    Jpen[H_WRIST_J] = (
        desk_inside_pt  # right wrist driven into the desk slab (the "sunk hand" penetration)
    )
    Jpen[H_ELBOW_J] = desk_inside_pt + np.array(
        [0.12, 0.0, 0.0]
    )  # forearm also into the slab
    Jpen[H_RMID_MCP] = desk_inside_pt + np.array(
        [0.03, 0.0, 0.0]
    )  # keep the hand capsule direction sane
    ph.update_proxy_rig(proxy, Jpen)
    pen_pairs = ph.human_scene_collision_pairs(stage=stage)
    LOG.info(
        "[M0 DETECTOR] FORCED penetrating pose (right hand+forearm driven into the desk solid at %s; "
        "cup parked away): human-vs-scene pairs = %d  %s",
        desk_inside_pt.round(3).tolist(),
        len(pen_pairs),
        pen_pairs,
    )
    LOG.info(
        "[M0 DETECTOR] SUMMARY: rest=%d pair(s) (target ~0), forced-penetration=%d pair(s) "
        "(target >0) -> detector works %s",
        len(rest_pairs),
        len(pen_pairs),
        "BOTH WAYS"
        if len(pen_pairs) > len(rest_pairs)
        else "(check: penetration not larger!)",
    )
    # restore rest
    _pose_update(rig.arm_rest, 0, "idle", 0.0, False, rest_cup_pos)

    # ---- (3) Render the constant-height pick: idle -> reach -> grasp(weld) -> lift. ---------------
    def _path_pose(s):
        u = max(0.0, min(1.0, s)) * (len(rp) - 1)
        k = min(int(u), len(rp) - 2)
        return core.lerp_arm(rp[k], rp[k + 1], u - k)

    def f_idle(t, frame):
        _pose_update(
            rig.arm_rest,
            frame,
            "idle" if t < 0.5 else "action",
            0.0,
            False,
            rest_cup_pos,
        )

    gidx = run_segment(ctx, "idle", 6, f_idle, gidx, maxf)
    saver.maybe_kf("idle", gidx)
    saver.maybe_kf_tpv("idle", gidx)

    def f_reach(t, frame):
        arm = _path_pose(
            smoothstep(t)
        )  # OPEN hand, straight wrist, monotonic side approach
        _pose_update(
            arm, frame, "action", 0.0, False, rest_cup_pos
        )  # CONSTANT height (no dip)

    gidx = run_segment(ctx, "reach", 14, f_reach, gidx, maxf)
    saver.maybe_kf("reach", gidx)
    saver.maybe_kf_tpv("reach", gidx)

    # Log collision pairs at the END of the reach (hand near the cup, root still at stand_z).
    reach_pairs = ph.human_scene_collision_pairs(stage=stage)
    LOG.info(
        "[M0 DETECTOR] end-of-reach (constant height): human-vs-scene pairs = %d  %s",
        len(reach_pairs),
        reach_pairs,
    )

    def f_grasp(t, frame):
        s = smoothstep(t)
        arm = core.lerp_arm(
            open_grasp, agt, s
        )  # wrist rotates straight -> grip; fingers close 0->1
        _pose_update(arm, frame, "action", s, False, rest_cup_pos)

    gidx = run_segment(ctx, "grasp", 6, f_grasp, gidx, maxf)
    saver.maybe_kf("grasp", gidx)
    saver.maybe_kf_tpv("grasp", gidx)

    # WELD the cup to the right-hand proxy at the moment of grasp.
    hand_path = proxy.hand_paths.get("r")
    ph.grasp_weld(stage=stage, hand_proxy_path=hand_path, obj_path="/World/Obj")

    # ---- (4) LIFT: the welded cup rides the wrist (constant height). The cup pose still comes from the
    # kinematic attach (the FixedJoint is the PHYSICS proof that hand->cup holds; the render uses the
    # same kinematic seat so the visible cup matches). Confirm the joint exists + drives the cup.
    cup_lift_positions = []

    def f_lift(t, frame):
        arm = core.lerp_arm(
            agt, rig.arm_grasp, smoothstep(t)
        )  # side-grip carry rises to the chest
        _, aim = _pose_update(
            arm, frame, "action", 1.0, True, None
        )  # attach=True -> cup rides wrist
        cup_lift_positions.append(np.asarray(aim, dtype=np.float64).copy())

    gidx = run_segment(ctx, "lift", 12, f_lift, gidx, maxf)
    saver.maybe_kf("lift", gidx)
    saver.maybe_kf_tpv("lift", gidx)
    if len(cup_lift_positions) >= 2:
        rose = float(cup_lift_positions[-1][2] - cup_lift_positions[0][2])
        LOG.info(
            "[M0 GRASP] FixedJoint held: cup followed the hand through the lift (cup rose %.3f m)",
            rose,
        )

    # ---- (5) RELEASE + SETTLE: delete the joint, let physics rest the cup on the desk. ------------
    settle_steps, final_cup_z = _physics_release_and_settle(ph, stage, rig, core, app)
    cup_base_z = final_cup_z + core.local_obj_min_z(rig)
    # The packing-table asset is a CONTAINER (a bin): a dropped object rests on its interior floor
    # (~z_min of the desk bbox), not on the rim (~z_max). So judge "rested ON the structure" by the
    # cup base landing WITHIN the desk's vertical extent [z_min, z_max] rather than against the rim.
    on_struct = desk_z_min - 0.02 <= cup_base_z <= desk_z_max + 0.02
    LOG.info(
        "[M0 SETTLE] %d physics steps -> cup origin z=%.4f, cup BASE z=%.4f; desk z-extent=[%.4f,%.4f] "
        "-> cup rests %s the desk structure (not floating above z_max, not sunk to the floor z~0)",
        settle_steps,
        final_cup_z,
        cup_base_z,
        desk_z_min,
        desk_z_max,
        "ON/IN" if on_struct else "OUTSIDE (CHECK!)",
    )

    # Render a final still of the rested cup (camera peering at the cup on the desk).
    Vf, Jf = core.pose_body(rig, rig.arm_rest, start_xy, 0.0, root_z=sz)
    core.update_body(rig, Vf)
    ph.update_proxy_rig(proxy, Jf)
    cup_now = _read_cup_pos(stage, core, rig)

    def f_settled(t, frame):
        cam_mod.drive_fpv(cop, Jf, "idle", cup_now, frame, 0.0)

    gidx = run_segment(ctx, "settled", 4, f_settled, gidx, maxf)
    saver.maybe_kf("settled", gidx)
    saver.maybe_kf_tpv("settled", gidx)

    return gidx


def core_stage(core, rig) -> object:
    """Return the live USD stage (via the cached body mesh's stage). Small helper so the physics M0
    code can fetch the stage without threading it through every call."""
    return rig._body_mesh.GetPrim().GetStage()


def _desk_bbox(desk_prim) -> tuple["object", float, float]:
    """Return (a point inside the desk solid, slab z_min, slab z_max) from the desk's actual MESH verts.

    We compute the world-space AABB from the desk's child-mesh vertices (NOT compute_world_bbox, whose
    extent hints can differ from the real geometry -- here the asset's reported bbox is offset from the
    actual container mesh). The returned point sits at the mesh XY center just under its top surface, so
    it lies inside the desk solid (the M0 positive-control penetration target). z_min/z_max are the
    slab's true vertical extent, used to judge whether the settled cup rests ON the structure.
    """
    import numpy as np
    from pxr import Usd, UsdGeom

    xc = UsdGeom.XformCache(Usd.TimeCode(0))
    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo
    for p in Usd.PrimRange(desk_prim):
        if p.GetTypeName() != "Mesh":
            continue
        pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
        if pts.size == 0:
            continue
        xf = np.array(xc.GetLocalToWorldTransform(p)).reshape(4, 4)
        w = (np.c_[pts, np.ones(len(pts))] @ xf)[:, :3]
        lo = np.minimum(lo, w.min(0))
        hi = np.maximum(hi, w.max(0))
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    inside_pt = np.array(
        [cx, cy, hi[2] - 0.03], dtype=np.float64
    )  # just under the top -> inside the slab
    return inside_pt, float(lo[2]), float(hi[2])


def _read_cup_pos(stage, core, rig) -> "object":
    """World-space translation of the cup origin (/World/Obj) from the current USD xform."""
    import numpy as np
    from pxr import Usd, UsdGeom

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    m = np.array(
        xc.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Obj"))
    ).reshape(4, 4)
    return m[3, :3]  # USD row-vector layout -> translation is the last ROW


def _settle_physics(ph, stage, max_steps: int = 180) -> tuple[int, object]:
    """Create/reuse a SimulationContext and step PhysX (render=False) until the cup comes to rest.

    Returns (steps, cup_origin_xyz). The cup velocity is finite-differenced from its USD world pose
    between physics steps (robust without a physics view): step(render=False) advances PhysX and writes
    the new pose back so the XformCache sees it move. Shared by the GRAB cup pre-settle and the M0
    release-settle so there is ONE settle code path.
    """
    import numpy as np
    from isaacsim.core.api.simulation_context import SimulationContext
    from pxr import Usd, UsdGeom

    sim = SimulationContext.instance()
    if sim is None:
        sim = SimulationContext(
            physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, backend="numpy"
        )
    sim.initialize_physics()

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    def _cup_pos():
        xc.Clear()
        m = np.array(
            xc.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Obj"))
        ).reshape(4, 4)
        return m[3, :3]

    prev = {"p": _cup_pos()}

    def _vel():
        p = _cup_pos()
        v = (p - prev["p"]) * 60.0  # finite-difference over the 1/60 s physics dt
        prev["p"] = p
        return v

    steps = ph.settle(sim, _vel, max_steps=max_steps, vel_thresh=1e-3)
    return steps, _cup_pos()


def _presettle_cup_on_desk(
    ph, stage, rig, core, desk_top_z: float, seat_xy, z_offset: float = 0.03
) -> "tuple[float, bool]":
    """Drop the dynamic cup just above the desk top and let PhysX rest it -> the true resting ORIGIN z.

    Robust to the mug-pivot / GRAB-origin / desk-bbox mismatch that floated the kinematic cup: instead
    of trusting arithmetic, we place the cup base ``z_offset`` above the realized desk top at the grasp
    XY and run the render=False settle, then read back where it ACTUALLY rested. If the settle lands
    off-target (didn't converge, sank, or rolled off), fall back to the analytic desk-top seat.

    Returns (rested_origin_z, settled_ok). settled_ok is False when the drop was unreliable (sphere
    rolled/tunneled, box fell through the desk) -> the caller should hold the object KINEMATIC on the
    desk for the free phase instead of trusting the dynamic sim (which would drop it to the floor).
    """
    import numpy as np

    drop_origin_z = float(desk_top_z) - core.local_obj_min_z(rig) + float(z_offset)
    core.place_obj_resting(
        rig, np.array([float(seat_xy[0]), float(seat_xy[1]), drop_origin_z])
    )
    steps, pos = _settle_physics(ph, stage)
    base_z = float(pos[2]) + core.local_obj_min_z(rig)
    xy_roll = float(np.hypot(pos[0] - seat_xy[0], pos[1] - seat_xy[1]))
    LOG.info(
        "[GRAB] cup pre-settle: %d steps -> origin z=%.4f (base z=%.4f vs desk top %.4f, dz=%.1fmm) "
        "xy-roll=%.1fmm",
        steps,
        float(pos[2]),
        base_z,
        float(desk_top_z),
        (base_z - float(desk_top_z)) * 1000.0,
        xy_roll * 1000.0,
    )
    if abs(base_z - float(desk_top_z)) > 0.01 or xy_roll > 0.03:
        LOG.warning(
            "[GRAB] cup pre-settle landed off-target -> analytic desk-top seat (base on top)"
        )
        return float(desk_top_z) - core.local_obj_min_z(rig), False
    return float(pos[2]), True


def _physics_release_and_settle(ph, stage, rig, core, app) -> tuple[int, float]:
    """M0 place/release: delete the grasp joint and let PhysX settle the released cup (render=False).

    Returns (settle_steps, final_cup_origin_z).
    """
    ph.release_weld(stage=stage, joint_path="/World/GraspJoint")
    steps, pos = _settle_physics(ph, stage)
    return steps, float(pos[2])


def _stages_for(stage: str) -> set[str]:
    """Which segment names to run for a --stage. reach=pick sub-clip (approach->idle->reach->grasp->lift)."""
    pick = {"approach", "idle", "reach", "grasp", "lift"}
    if stage == "reach":
        return pick
    if stage == "walk":
        return pick | {"walk"}
    if stage == "place":
        return pick | {"walk", "place", "release", "retract"}
    return pick | {"walk", "place", "release", "retract"}  # full


# --------------------------------------------------------------------------------------
# Setup helpers.
# --------------------------------------------------------------------------------------
def _loco_cfg():
    from fpv_core import LocoCfg

    return LocoCfg(drive_skel_anim=False)


def _build_scene(scene: str, office_scene, stage, ta_xy, tb_xy):
    """Build the requested scene with tables A/B at the given world XY (under the cup reach points)."""
    if scene == "office":
        return office_scene.build_office(stage, ta_xy, tb_xy)
    return office_scene.build_simple_room(stage, ta_xy, tb_xy)


def _boost_lights(stage) -> None:
    """Raise the dome+distant intensities to the static pipeline's proven values (dome 700, distant 1800)."""
    from pxr import UsdLux

    dome = UsdLux.DomeLight.Get(stage, "/World/Lights/Dome")
    if dome:
        dome.GetIntensityAttr().Set(150.0)
    dist = UsdLux.DistantLight.Get(stage, "/World/Lights/Distant")
    if dist:
        dist.GetIntensityAttr().Set(400.0)


def _relight_grab(stage, mug_xy, body_y: float) -> None:
    """Re-pose the office interior RectLights to span the GRAB action (body head Y -> mug/desk Y).

    The office is an enclosed interior whose dome is occluded by the ceiling, so the desk + cup + hand
    fall into a black void unless lit from above the action. ``office_scene`` already authored RectLights
    at ``/World/OfficeLights/L{i}`` over the (table, table) XY, which leaves the body and the desk near
    edge dark. We re-pose them (and brighten) over three points spanning the body (+Y) to the mug/desk
    (-Y) at the mug X, just under the office ceiling, emitting down (-Z). Derived from the GRAB body +
    mug XY, no per-asset constant.
    """
    from fpv_scene import set_world_pose
    from pxr import UsdLux

    mx, my = float(mug_xy[0]), float(mug_xy[1])
    midy = 0.5 * (float(body_y) + my)
    # Lights seated LOW (z=1.7, well BELOW the office ceiling so they are not occluded) emitting down
    # (-Z), spanning the body (+Y) -> mug/desk (-Y), bright so the desk top + cup + hand are clearly lit
    # (the enclosed office ceiling blocks the dome, so the action needs its own near-overhead lights).
    pts = [(mx, float(body_y) + 0.1, 1.9), (mx, midy, 1.7), (mx, my - 0.3, 1.7)]
    for i, (x, y, z) in enumerate(pts):
        path = f"/World/OfficeLights/L{i}"
        lt = UsdLux.RectLight.Get(stage, path) or UsdLux.RectLight.Define(stage, path)
        lt.GetIntensityAttr().Set(3500.0)
        lt.CreateWidthAttr(2.6)
        lt.CreateHeightAttr(2.6)
        lt.CreateColorAttr((1.0, 0.97, 0.92))
        set_world_pose(lt.GetPrim(), (x, y, z))
    # A dedicated bright light right ABOVE the desk near-edge + cup so the grasp/lift surface and the
    # held cup read clearly (not a dark slab), and a front fill from the camera side at chest height.
    spot = UsdLux.RectLight.Get(
        stage, "/World/OfficeLights/Cup"
    ) or UsdLux.RectLight.Define(stage, "/World/OfficeLights/Cup")
    spot.GetIntensityAttr().Set(2500.0)
    spot.CreateWidthAttr(1.4)
    spot.CreateHeightAttr(1.4)
    spot.CreateColorAttr((1.0, 0.98, 0.95))
    set_world_pose(spot.GetPrim(), (mx, my - 0.1, 1.5))
    fill = UsdLux.RectLight.Get(
        stage, "/World/OfficeLights/Fill"
    ) or UsdLux.RectLight.Define(stage, "/World/OfficeLights/Fill")
    fill.GetIntensityAttr().Set(1500.0)
    fill.CreateWidthAttr(2.0)
    fill.CreateHeightAttr(2.0)
    fill.CreateColorAttr((1.0, 0.98, 0.95))
    # Aim the fill at the cup-hold height from the camera side (front-right): a RectLight emits along its
    # local -Z, so SetLookAt(eye,target).GetInverse() points the emission at the target.
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(fill.GetPrim())
    xf.ClearXformOpOrder()
    m = Gf.Matrix4d(1.0)
    m.SetLookAt(
        Gf.Vec3d(mx + 1.0, my + 0.3, 1.6),
        Gf.Vec3d(mx - 0.1, my - 0.1, 0.95),
        Gf.Vec3d(0, 0, 1),
    )
    xf.MakeMatrixXform().Set(m.GetInverse())
    LOG.info(
        "[GRAB] re-lit office: 3 down RectLights (z~1.7, int 45000) over y=[%.2f..%.2f] at x=%.2f "
        "+ a cup spot + a front fill",
        my - 0.3,
        float(body_y) + 0.1,
        mx,
    )


def _make_place_and_measure(stage, rig, app):
    """Return a place_and_measure(T_world_colvec, Vq) closure for the one-time side backoff bake.

    Sets the cup's USD matrix, steps the app a couple frames so the transform propagates, rebuilds
    the cup's world trimesh, and returns (mesh, max penetration mm of Vq into it). Mirrors
    smplh_fpv.place_and_measure but uses the dynamic stage + object.
    """
    import numpy as np
    import trimesh
    from pxr import Gf, Usd, UsdGeom

    def _world_trimesh(parent: str):
        parts = []
        xc = UsdGeom.XformCache(Usd.TimeCode(0))
        for p in Usd.PrimRange(stage.GetPrimAtPath(parent)):
            tp = p.GetTypeName()
            xf = np.array(xc.GetLocalToWorldTransform(p)).reshape(4, 4)
            if tp == "Mesh":
                mm = UsdGeom.Mesh(p)
                pts = np.array(mm.GetPointsAttr().Get())
                pts = (np.c_[pts, np.ones(len(pts))] @ xf)[:, :3]
                fc = np.array(mm.GetFaceVertexCountsAttr().Get())
                fi = np.array(mm.GetFaceVertexIndicesAttr().Get())
                tris, o = [], 0
                for c in fc:
                    for k in range(1, c - 1):
                        tris.append([fi[o], fi[o + k], fi[o + k + 1]])
                    o += c
                parts.append(
                    trimesh.Trimesh(vertices=pts, faces=np.array(tris), process=True)
                )
            elif tp == "Sphere":
                r = float(UsdGeom.Sphere(p).GetRadiusAttr().Get())
                s = trimesh.creation.icosphere(4, r)
                s.apply_transform(xf.T)
                parts.append(s)
        return (
            trimesh.util.concatenate(parts)
            if len(parts) > 1
            else (parts[0] if parts else None)
        )

    def place_and_measure(T_world, Vq):
        rig._obj_xf.Set(Gf.Matrix4d(*[float(x) for x in T_world.T.flatten()]))
        for _ in range(2):
            app.update()
        m = _world_trimesh("/World/Obj")
        if m is None or not m.is_watertight:
            return m, float("nan")
        sd = trimesh.proximity.ProximityQuery(m).signed_distance(Vq)
        return m, float(sd.max()) * 1000.0

    return place_and_measure


# Object SDF cache: (local-mesh content hash, pitch) -> (signed_grid_local, local->grid inverse transform,
# pitch). The voxel SDF is built in the object's LOCAL frame and is therefore pose-INDEPENDENT, so ONE build
# serves the object at any world pose. The in-memory layer serves repeated grasps in one warm process; the
# file-backed layer serves later processes and concurrent workers on the same node. Both are content keyed,
# so an instance name can never alias different geometry.
_SDF_CACHE: dict = {}
_SDF_CACHE_SCHEMA = 1


def _sdf_cache_path(mesh_hash: str, pitch: float) -> str | None:
    enabled = (os.getenv("ISAAC_SDF_DISK_CACHE") or "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    root = os.getenv("ISAAC_SDF_CACHE_DIR") or "/tmp/isaacsim_try_sdf_v1"
    pitch_nm = int(round(float(pitch) * 1e9))
    return os.path.join(root, f"v{_SDF_CACHE_SCHEMA}_{mesh_hash}_{pitch_nm}.npz")


def _load_sdf_cache_file(path, mesh_hash, pitch, np):
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema"].item()) != _SDF_CACHE_SCHEMA:
                return None
            if str(data["mesh_hash"].item()) != mesh_hash:
                return None
            if float(data["pitch"].item()) != float(pitch):
                return None
            signed = np.asarray(data["signed"], dtype=np.float64)
            t_inv = np.asarray(data["t_inv"], dtype=np.float64)
        if signed.ndim != 3 or t_inv.shape != (4, 4):
            return None
        if not np.isfinite(signed).all() or not np.isfinite(t_inv).all():
            return None
        return signed, t_inv, float(pitch)
    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None


def _store_sdf_cache_file(path, mesh_hash, pitch, signed, t_inv, np) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "wb") as handle:
            np.savez(
                handle,
                schema=np.asarray(_SDF_CACHE_SCHEMA, dtype=np.int64),
                mesh_hash=np.asarray(mesh_hash),
                pitch=np.asarray(float(pitch), dtype=np.float64),
                signed=signed,
                t_inv=t_inv,
            )
        os.replace(tmp, path)
    except OSError as exc:
        LOG.warning("[wrap] voxel SDF disk-cache write failed: %s", exc)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _build_voxel_sdf(vertices, faces, pitch, np, ndi, trimesh):
    vg = trimesh.Trimesh(vertices, faces, process=False).voxelized(pitch=pitch).fill()
    occ = np.asarray(vg.matrix, dtype=bool)
    if not occ.any() or occ.all():
        raise ValueError("degenerate voxelization")
    # EDT needs an explicit empty border; otherwise occupied edge cells have no outside neighbour.
    pad = max(4, int(np.ceil(0.02 / pitch)))
    occ = np.pad(occ, pad, mode="constant", constant_values=False)
    transform = np.asarray(vg.transform, dtype=np.float64).copy()
    transform[:3, 3] -= pad * pitch
    t_inv = np.linalg.inv(transform)
    signed = (
        ndi.distance_transform_edt(occ) - ndi.distance_transform_edt(~occ)
    ).astype(np.float64) * float(pitch)
    return signed, t_inv


def _cached_voxel_sdf(key, vertices, faces, pitch, label, np, ndi, trimesh):
    entry = _SDF_CACHE.get(key)
    if entry is not None:
        LOG.info(
            "[wrap] voxel SDF CACHE HIT %s key=%s (build skipped)",
            label,
            key[0][:12],
        )
        signed, t_inv, _ = entry
        return signed, t_inv, True

    cache_path = _sdf_cache_path(key[0], pitch)
    lock_handle = None
    if cache_path is not None:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            lock_handle = open(f"{cache_path}.lock", "a+b")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            LOG.warning("[wrap] voxel SDF disk-cache lock failed: %s", exc)
            if lock_handle is not None:
                lock_handle.close()
            lock_handle = None
            cache_path = None
    try:
        disk_entry = (
            _load_sdf_cache_file(cache_path, key[0], pitch, np)
            if cache_path is not None
            else None
        )
        if disk_entry is not None:
            signed, t_inv, _ = disk_entry
            LOG.info(
                "[wrap] voxel SDF DISK CACHE HIT %s key=%s (build skipped)",
                label,
                key[0][:12],
            )
            cached = True
        else:
            started = time.perf_counter()
            signed, t_inv = _build_voxel_sdf(vertices, faces, pitch, np, ndi, trimesh)
            if cache_path is not None:
                _store_sdf_cache_file(cache_path, key[0], pitch, signed, t_inv, np)
            LOG.info(
                "[wrap] voxel SDF built in %.2fs: grid=%s pitch=%.1fmm %s key=%s "
                "(local frame, positive=inside)",
                time.perf_counter() - started,
                tuple(int(size) for size in signed.shape),
                pitch * 1000.0,
                label,
                key[0][:12],
            )
            cached = False
        _SDF_CACHE[key] = (signed, t_inv, float(pitch))
        return signed, t_inv, cached
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def _make_sd_world(
    stage, obj_path: str = "/World/Obj", pitch: float = 0.002, cache_key=None
):
    """Return sd_world(Vw) -> signed distances (POSITIVE = inside) for the live object's world mesh.

    P0: the contact-wrap gate is a PRECOMPUTED voxel SDF (scipy EDT), not a per-call
    ``trimesh.signed_distance`` (~94ms/call without embree here). P0.5: the field is built ONCE in the
    object's LOCAL frame and CACHED under a hash of the build's only inputs -- the quantised local vertices,
    the faces and ``pitch``. Being pose-independent, the cached grid serves the object at any world pose, so
    repeated grasps of one object -- and every OTHER instance or process using the same model -- skip the
    build. A per-key file lock lets concurrent workers share the first build without racing.
    ``cache_key`` is a human-readable LOG LABEL only; it is deliberately NOT the cache key, because an
    instance name is neither unique across the scenes a warm RL process rotates through nor pitch-aware.
    The returned callable maps world -> object-local -> grid (trilinear, POSITIVE inside) and
    carries ``.exact`` (trimesh query, for the final penetration re-check / bench), ``.mesh``, ``.kind``,
    ``.cached``. Falls back to the exact query if voxelization is unavailable/degenerate. (subdivide is the
    only correct trimesh voxelizer here -- ray returns a hollow shell, decimation needs fast_simplification;
    a fast C++ voxelizer such as open3d would additionally cut the first build.)
    """
    import hashlib

    import numpy as np
    import scipy.ndimage as ndi
    import trimesh
    from pxr import Usd, UsdGeom

    xc = UsdGeom.XformCache(Usd.TimeCode(0))
    xf_obj = np.array(
        xc.GetLocalToWorldTransform(stage.GetPrimAtPath(obj_path))
    ).reshape(4, 4)  # row convention: world_row = local_row @ xf_obj
    parts = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(obj_path)):
        tp = p.GetTypeName()
        xf = np.array(xc.GetLocalToWorldTransform(p)).reshape(4, 4)
        if tp == "Mesh":
            mm = UsdGeom.Mesh(p)
            pts = np.array(mm.GetPointsAttr().Get())
            pts = (np.c_[pts, np.ones(len(pts))] @ xf)[:, :3]
            fc = np.array(mm.GetFaceVertexCountsAttr().Get())
            fi = np.array(mm.GetFaceVertexIndicesAttr().Get())
            tris, o = [], 0
            for c in fc:
                for k in range(1, c - 1):
                    tris.append([fi[o], fi[o + k], fi[o + k + 1]])
                o += c
            parts.append(
                trimesh.Trimesh(vertices=pts, faces=np.array(tris), process=True)
            )
        elif tp == "Sphere":
            r = float(UsdGeom.Sphere(p).GetRadiusAttr().Get())
            s = trimesh.creation.icosphere(4, r)
            s.apply_transform(xf.T)
            parts.append(s)
    m = (
        trimesh.util.concatenate(parts)
        if len(parts) > 1
        else (parts[0] if parts else None)
    )
    if m is None:
        raise RuntimeError("_make_sd_world: no object mesh under " + obj_path)

    pq = trimesh.proximity.ProximityQuery(m)

    def sd_exact(Vw):
        return pq.signed_distance(np.asarray(Vw, dtype=np.float64))

    # object-LOCAL mesh vertices (pose-independent): world_row @ inv(xf_obj)
    xf_inv = np.linalg.inv(xf_obj)
    lv = (np.c_[m.vertices, np.ones(len(m.vertices))] @ xf_inv)[:, :3]
    # Quantising to 10 um absorbs the float jitter of the world->local round trip, so two instances of one
    # model standing at different world poses hash the same, while staying far finer than any real geometric
    # difference. Measured at ~0.6 ms on a 3.2k-vertex mesh, against the 2-215 s voxelize+EDT it guards.
    qi = np.rint(lv / 1e-5).astype(
        np.int64
    )  # integer quantize -> robust to -0.0 / float jitter
    key = (
        hashlib.blake2b(
            qi.tobytes() + np.asarray(m.faces, dtype=np.int64).tobytes(),
            digest_size=16,
        ).hexdigest(),
        round(float(pitch), 9),
    )
    label = str(cache_key) if cache_key is not None else key[0][:12]

    try:
        signed, t_inv, cached = _cached_voxel_sdf(
            key, lv, m.faces, pitch, label, np, ndi, trimesh
        )

        def sd_fast(Vw):
            Vw = np.asarray(Vw, dtype=np.float64)
            loc = (np.c_[Vw, np.ones(len(Vw))] @ xf_inv)[:, :3]  # world -> object-local
            ijk = (np.c_[loc, np.ones(len(loc))] @ t_inv.T)[
                :, :3
            ]  # local -> grid index
            return ndi.map_coordinates(signed, ijk.T, order=1, mode="nearest")

        sd_fast.exact = sd_exact
        sd_fast.mesh = m
        sd_fast.kind = "voxel"
        sd_fast.cached = cached
        return sd_fast
    except Exception as exc:  # noqa: BLE001 - any voxel failure -> safe exact fallback
        LOG.warning(
            "[wrap] voxel SDF unavailable (%s); using exact trimesh signed_distance",
            exc,
        )
        sd_exact.exact = sd_exact
        sd_exact.mesh = m
        sd_exact.kind = "trimesh"
        sd_exact.cached = False
        return sd_exact


def _print_geom(core, cam_mod, rig, seat0, J0, width, height) -> None:
    """Print the frozen-arm geometry at root (0,0,0): wrist, head, obj_ctr, and FPV frustum membership."""
    import numpy as np

    wrist = J0[core.WRIST_J]
    head = J0[core.HEAD_J]
    obj_ctr = seat0[:3, 3]
    eye, tgt = cam_mod.fpv_eye_and_aim(J0, "idle", obj_ctr, 0, 0.0)
    inside, ndc_x, ndc_y = cam_mod.point_in_fpv_frustum(
        eye, tgt, obj_ctr, width, height
    )
    print(
        "================ GEOM (frozen canonical arm @ root (0,0,0)) ================"
    )
    print(f"J[WRIST_J={core.WRIST_J}] = {np.asarray(wrist).round(4).tolist()}")
    print(f"J[HEAD_J={core.HEAD_J}]  = {np.asarray(head).round(4).tolist()}")
    print(
        f"seated obj_ctr        = {np.asarray(obj_ctr).round(4).tolist()}  (world-Z={float(obj_ctr[2]):.4f})"
    )
    print(f"FPV eye               = {np.asarray(eye).round(4).tolist()}")
    print(f"FPV target            = {np.asarray(tgt).round(4).tolist()}")
    print(
        f"obj_ctr in FPV frustum: {inside}  (ndc_x={ndc_x:.3f} ndc_y={ndc_y:.3f}; inside iff |both|<=1)"
    )
    print(
        f"obj_T_in_wrist_baked_t = {rig.obj_T_in_wrist_baked[:3, 3].round(4).tolist()}"
    )
    print(
        "============================================================================"
    )


class _StillSaver:
    """Copy the latest captured frame to a named keyframe PNG at each segment boundary (stills)."""

    def __init__(self, ctx, out_dir: str, prefix: str) -> None:
        self.ctx = ctx
        self.out_dir = out_dir
        self.prefix = prefix

    def maybe_kf(self, name: str, gidx: int) -> None:
        """Copy frame_<gidx-1>.png (the segment's last captured FPV frame) to <prefix>_kf_<name>.png."""
        self._copy(self.ctx.fpv_dir, f"{self.prefix}_kf_{name}.png", gidx)

    def maybe_kf_tpv(self, name: str, gidx: int) -> None:
        """Copy the segment's last captured TPV frame to <prefix>_tpv_kf_<name>.png (verification view)."""
        self._copy(self.ctx.tpv_dir, f"{self.prefix}_tpv_kf_{name}.png", gidx)

    def _copy(self, src_dir: str, dst_name: str, gidx: int) -> None:
        import shutil

        if gidx <= 0:
            return
        src = os.path.join(src_dir, f"frame_{gidx - 1:05d}.png")
        dst = os.path.join(self.out_dir, dst_name)
        if os.path.isfile(src):
            os.makedirs(self.out_dir, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"KF {dst}")
