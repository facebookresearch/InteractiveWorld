"""Core launch/config/timeline/lighting/math helpers shared by the FPV demos.

This module is the single place that owns:
  * the Isaac Sim 5.1 ``SimulationApp`` headless RTX bootstrap (``launch_app`` MUST
    be called before ANY ``pxr``/``omni`` import),
  * the post-launch carb settings (renderer normalized literal, ACES tonemap, and
    the opt-in animation extensions),
  * stage acquisition (Z-up, meters),
  * the mandatory timeline setup: a LARGE end time (so the clock never clamps mid-render)
    + ``timeline.play()`` (without play() UsdSkel/physics never advance and replicator
    annotators stay empty),
  * headless lighting (headless has NO default light => black frames otherwise),
  * asset-existence assertions (fail loud at startup), and
  * the small math helpers (smoothstep / lerp / slerp) used for every keyframed
    object-motion segment.

All public symbols match the frozen ``fpv_core`` spec exactly. Asset paths are ALWAYS
local; ``get_assets_root_path`` / ``omniverse://`` / nucleus are never used (offline).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These names are only available once Isaac Sim is installed AND launch_app has
    # run. They are imported lazily inside the functions; the TYPE_CHECKING block
    # keeps type hints honest without importing pxr/omni at module import time.
    from isaacsim import SimulationApp  # noqa: F401
    from pxr import Gf, Usd  # noqa: F401


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][fpv][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("fpv_core")


# --------------------------------------------------------------------------------------
# Module constants — NEVER call get_assets_root_path; these are the only roots we use.
# --------------------------------------------------------------------------------------
ASSET_ROOT: str = os.environ.get(
    "INTERACTIVEWORLD_ISAAC_ASSET_ROOT", "/opt/isaac-sim/Assets/Isaac/5.1/Isaac"
)
OUT_ROOT: str = os.environ.get(
    "INTERACTIVEWORLD_OUTPUT_DIR", "/tmp/interactiveworld-output"
)


# --------------------------------------------------------------------------------------
# Config dataclasses (frozen API surface).
# --------------------------------------------------------------------------------------
@dataclass
class AppCfg:
    """Launch + lighting config for the SimulationApp + RTX pipeline."""

    headless: bool = True
    width: int = 1280
    height: int = 720
    renderer: str = "RayTracedLighting"
    # RayTracedLighting (RTX real-time) needs a long pre-roll for the lighting/denoiser to
    # converge -- the in-repo rendering skill notes ~200 settle frames before rgb.mean()
    # lands in a healthy 30-200; 120 risked a dim-but-nonzero frame that passes the max>0
    # guard yet looks underexposed. Bumped to 200.
    warmup_frames: int = 200
    physics_dt: float = 1.0 / 60.0
    dome_intensity: float = 400.0
    dome_exposure: float = 0.0
    distant_intensity: float = 1500.0
    # GPU pinning. SimulationApp defaults to multi_gpu=True; on a 2x A100 box multi-GPU
    # RENDER can cause device-mismatch on annotator readback in some 5.x headless builds.
    # We pin one A100 by default (multi_gpu=False, active_gpu=0); set multi_gpu=True to opt
    # back into multi-GPU. active_gpu=None lets the launcher pick the default device.
    active_gpu: int | None = 0
    physics_gpu: int = 0
    multi_gpu: bool = False
    # Symbolic simulator gates never consume the default viewport. Leaving it live makes each Kit update
    # synchronize the full RTX/Hydra scene even though no pixels are read. SimulationApp exposes this as
    # a first-class headless optimization; render paths retain the default so their products are unchanged.
    disable_viewport_updates: bool = False
    sync_loads: bool = True


@dataclass
class CameraCfg:
    """Ego-cam intrinsics + gaze. aperture+focal EXPLICIT => ~90deg HFOV (glasses-like)."""

    horizontal_aperture: float = 36.0
    focal_length_mm: float = 18.0
    near_clip: float = 0.012
    far_clip: float = 100000.0
    eye_fwd_offset: float = 0.10
    eye_up_offset: float = 0.05
    rest_pitch_deg: float = 10.0
    walk_pitch_deg: float = 14.0
    pitch_clamp_max_deg: float = 34.0
    pitch_clamp_min_deg: float = -14.0
    turn_lo_dps: float = 22.0
    turn_hi_dps: float = 60.0
    turn_gain: float = 3.5
    head_lambda: float = 3.0
    bob_vert_amp: float = 0.030
    bob_vert_hz: float = 2.8
    bob_lat_amp: float = 0.020
    bob_lat_hz: float = 1.4
    bob_pitch_amp_deg: float = 1.5


@dataclass
class GraspCfg:
    """Carry geometry + mode. Pure-USD camera-anchor carry is the default (no physics)."""

    use_physics_weld: bool = False
    carry_dist: float = 0.40
    carry_drop: float = 0.13
    carry_side: float = 0.05
    two_handed: bool = False
    carry_mode: str = "grip"  # one of {grip, cradle, two_handed}


@dataclass
class LocoCfg:
    """Locomotion. ``drive_skel_anim`` opts into the UsdSkel-eval path; default analytic."""

    walk_clip_rel: str = "People/Animations/stand_walk_loop_in_place.skelanim.usd"
    idle_clip_rel: str = "People/Animations/stand_idle_loop.skelanim.usd"
    speed_mps: float = 1.0
    drive_skel_anim: bool = False


@dataclass
class RenderCfg:
    """Capture/encode cadence and the SINGLE-CLOCK capture contract.

    CONTRACT (owned here; enforced by fpv_render + fpv_segments):
      * ONE advance of simulated time per CAPTURED FRAME ==
        ``rep.orchestrator.step(delta_time=1/capture_fps, pause_timeline=False)``.
        (Verified 5.1 signature: custom_fps_writer_annotator.py L116, sdg_deformables.py
        L194, simulation_get_data.py L105.)
      * ``capture_fps == playback_fps`` so wall-clock pacing of the encoded MP4 matches the
        simulated cadence (no slow-mo / speed-up).
      * The capture loop MUST NOT interleave ``app.update()`` between captures -- a stray
        ``app.update()`` advances the timeline OUTSIDE this clock and causes drift (the
        spec's FLAW 5). ``app.update()`` is permitted ONLY during the pre-roll warmup
        (``AppCfg.warmup_frames``) BEFORE frame 0, never inside ``run_segment``.

    TWO-PRODUCT SINGLE STEP (resolved): each captured frame renders TWO products (FPV +
    TPV). ``fpv_render`` exposes ``advance_once`` (the lone ``orchestrator.step`` per frame)
    + ``read_and_save`` (read an already-advanced annotator, no step). ``fpv_segments
    .run_segment`` calls ``advance_once`` ONCE then ``read_and_save`` for BOTH products, so
    the clock advances exactly once per frame and the FPV/TPV PNGs share a simulation time.
    """

    capture_fps: int = 12
    playback_fps: int = 12
    max_frames: int = 320
    rt_subframes: int = 16
    crf: int = 18
    hide_body_in_fpv: bool = False
    draw_proxy_forearm: bool = True


# --------------------------------------------------------------------------------------
# Launch + post-launch setup.
# --------------------------------------------------------------------------------------
def launch_app(cfg: AppCfg) -> "SimulationApp":
    """Build the headless RTX SimulationApp. MUST be the first call before pxr/omni.

    Sets ``DISPLAY=:0`` if unset (mitigates the 5.1.0 headless black-frame issue), accepts the
    EULA non-interactively, and disables the kit registry (stay offline), THEN constructs
    ``SimulationApp({headless,width,height,renderer})`` and returns it live.

    CONTRACT: this module (and fpv_scene/fpv_grasp/fpv_locomotion, which import from it at
    top level) must have NO pxr/omni import at MODULE import time -- every pxr/omni import
    is function-local so importing fpv_core before launch_app() never pulls in Kit. Keep it
    that way or the SimulationApp-first ordering breaks.

    NOTE on LD_LIBRARY_PATH: on this CentOS Stream 9 / glibc-2.34 box the loader needs the
    conda env's libstdc++ (GLIBCXX_3.4.34) ahead of the system /lib64 one, and that can only
    be set BEFORE the process execs (the dynamic loader reads LD_LIBRARY_PATH at startup, so
    setting it here is too late). Launch via simulator.physical_world/run_isaac.sh (or export it yourself)
    -- see the design record. The two env vars below CAN be set in-process because Kit reads them
    when SimulationApp() builds, which is after this point.
    """
    # DISPLAY must exist before SimulationApp builds the RTX context, even headless.
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        LOG.info("DISPLAY was unset; forced DISPLAY=:0 for headless RTX")
    # Accept the EULA non-interactively (else first run blocks on a prompt) and disable the
    # kit registry so Kit never reaches a nucleus/omniverse server (offline box). Verified
    # against the working probe runs (run_isaac.sh sets both).
    if not os.environ.get("OMNI_KIT_ACCEPT_EULA"):
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        LOG.info(
            "OMNI_KIT_ACCEPT_EULA was unset; forced YES for non-interactive first run"
        )
    if not os.environ.get("OMNI_KIT_REGISTRY_DISABLED"):
        os.environ["OMNI_KIT_REGISTRY_DISABLED"] = "1"
        LOG.info("OMNI_KIT_REGISTRY_DISABLED was unset; forced 1 to stay offline")

    # MDL search path for the OmniGibson VRay materials (BEHAVIOR-1K assets). MUST be set BEFORE the RTX
    # MDL translator initializes (during the SimulationApp build below) -- setting the carb MDL path AFTER
    # launch is too late, so every BEHAVIOR surface falls back to flat RED ("could not find module
    # omnigibson_vray_mtl"). The MDL neuraylib reads MDL_USER_PATH / MDL_SYSTEM_PATH at init. Global fix
    # (all demos launch through here); additive + harmless for non-BEHAVIOR scenes (an unused search dir).
    _mdl_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "migration_eval",
        "OmniGibson",
        "OmniGibson",
        "omnigibson",
        "materials",
    )
    if os.path.isdir(_mdl_dir):
        for _mv in ("MDL_USER_PATH", "MDL_SYSTEM_PATH"):
            _cur = os.environ.get(_mv, "")
            if _mdl_dir not in _cur.split(os.pathsep):
                os.environ[_mv] = _mdl_dir + os.pathsep + _cur if _cur else _mdl_dir
        LOG.info(f"MDL search path set (MDL_USER_PATH/MDL_SYSTEM_PATH): {_mdl_dir}")

    # Import is deliberately local: importing isaacsim is only valid at call time, and
    # SimulationApp() must run before any other pxr/omni import in the whole process.
    from isaacsim import SimulationApp

    launch_config = {
        "headless": bool(cfg.headless),
        "width": int(cfg.width),
        "height": int(cfg.height),
        "renderer": str(cfg.renderer),
        # GPU pinning (keys verified against simulation_app.py default config). active_gpu
        # may be None == launcher default; multi_gpu False pins a single device for stable
        # headless annotator readback on the 2x A100 box.
        "multi_gpu": bool(cfg.multi_gpu),
        "physics_gpu": int(cfg.physics_gpu),
        "active_gpu": (None if cfg.active_gpu is None else int(cfg.active_gpu)),
        "disable_viewport_updates": bool(cfg.disable_viewport_updates),
        "sync_loads": bool(cfg.sync_loads),
    }
    LOG.info("Launching SimulationApp: %s", launch_config)
    app = SimulationApp(launch_config)
    LOG.info(
        "SimulationApp launched (headless=%s, %dx%d, renderer=%s)",
        cfg.headless,
        cfg.width,
        cfg.height,
        cfg.renderer,
    )
    return app


def post_launch_setup(cfg: AppCfg, loco: LocoCfg) -> None:
    """Carb settings + opt-in anim extensions. Call after launch_app, before Biped use.

    * If ``loco.drive_skel_anim``: enable ``omni.anim.skelJoint`` + ``omni.anim.graph.core``
      (the UsdSkel-eval runtime; absent from the default standalone experience).
    * Re-assert ``/rtx/rendermode`` to the 5.1 canonical spelling ``RaytracedLighting``
      (renamed from ``RayTracedLighting`` in simulation_app 2.0.2) and set ACES tonemap op 4.
      This re-set is REDUNDANT, not a different-spelling requirement: SimulationApp matches
      the launch ``renderer`` kwarg case-insensitively (simulation_app.py lowercases it) and
      already sets ``/rtx/rendermode`` at launch. We re-assert defensively + set the tonemap.
    """
    import carb.settings

    if loco.drive_skel_anim:
        from isaacsim.core.utils.extensions import enable_extension

        for ext in ("omni.anim.skelJoint", "omni.anim.graph.core"):
            try:
                ok = enable_extension(ext)
                LOG.info("enable_extension(%s) -> %s", ext, ok)
            except Exception as exc:  # noqa: BLE001 - report but keep going; assert_skel_animates is the real gate
                LOG.warning(
                    "enable_extension(%s) raised %s (will rely on assert_skel_animates)",
                    ext,
                    exc,
                )

    settings = carb.settings.get_settings()
    # Redundant re-assert of the renderer already set at launch; use the 5.1 canonical
    # spelling RaytracedLighting (renamed from RayTracedLighting in simulation_app 2.0.2).
    settings.set("/rtx/rendermode", "RaytracedLighting")
    settings.set("/rtx/post/tonemap/op", 4)  # ACES
    # Disable RTX auto-exposure (histogram): headless it over-brightens the bright desk (blowing it to
    # white) AND flickers frame-to-frame. With it OFF the tonemap uses a fixed exposure, so the light
    # intensities directly + deterministically control the exposure (we tune those instead).
    for _k in (
        "/rtx/post/histogram/enabled",
        "/rtx/post/tonemap/autoExposure",
        "/rtx/post/tonemapping/autoExposure/enabled",
    ):
        try:
            settings.set(_k, False)
        except Exception:  # noqa: BLE001 - key may not exist on this Kit build; the others cover it
            pass
    LOG.info(
        "Re-asserted /rtx/rendermode=RaytracedLighting, ACES tonemap (op=4), auto-exposure OFF"
    )


def get_stage() -> "Usd.Stage":
    """Return the current USD stage, forced to Z-up + metersPerUnit=1.0."""
    import omni.usd
    from pxr import UsdGeom

    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        # Standalone apps usually have a stage already; create one defensively.
        ctx.new_stage()
        stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Could not acquire a USD stage from omni.usd context")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    LOG.info("Stage ready: up-axis=Z, metersPerUnit=1.0")
    return stage


def start_timeline(fps: float | None = None) -> "object":
    """MANDATORY: configure the timeline range, disable looping, play, commit.

    Without ``play()`` the UsdSkel evaluator and physics never advance and replicator
    annotators stay empty (verified across the replicator examples).

    CRITICAL (was a freeze bug): the stage's default end time is tiny (a fraction of a
    second). With looping disabled, the timeline reaches that end and STOPS after the first
    few capture steps -- which freezes ``drive_skel_anim`` mid-clip and stalls the
    ``use_physics_weld`` path (the analytic+keyframe default mostly survives because object
    poses are authored at the Default TimeCode, but the warmup ``app.update()`` loop can
    still exhaust a tiny range before frame 0). Every robust standalone example that manually
    steps the orchestrator sets a LARGE end time before play() -- e.g.
    ``object_based_sdg.py`` L513-516 and ``infinigen_sdg_utils.py`` L631-633 both do
    ``set_start_time(0); set_end_time(1000000); set_looping(False)``. We mirror that so the
    timeline keeps advancing across all ``RenderCfg.max_frames`` capture steps for EVERY mode.

    ``fps`` (optional) sets ``timeCodesPerSecond`` so authored time samples and the capture
    cadence agree; pass ``RenderCfg.playback_fps``. If None, the stage default is left alone.
    """
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    # Order matters: set the range + rate BEFORE play() (mirrors object_based_sdg.py).
    timeline.set_start_time(0.0)
    timeline.set_end_time(1.0e6)
    timeline.set_looping(False)
    if fps is not None:
        timeline.set_time_codes_per_second(float(fps))
    timeline.play()
    timeline.commit()
    LOG.info(
        "Timeline playing (start=0.0, end=1.0e6, looping=False, tcps=%s, committed)",
        f"{float(fps):.1f}" if fps is not None else "stage-default",
    )
    return timeline


def add_lighting(stage: "Usd.Stage", cfg: AppCfg) -> None:
    """Add a DomeLight + DistantLight under /World/Lights.

    Headless renders have NO default light, so without this every frame is black.
    """
    from pxr import Gf, UsdGeom, UsdLux

    scope_path = "/World/Lights"
    UsdGeom.Scope.Define(stage, scope_path)

    dome = UsdLux.DomeLight.Define(stage, scope_path + "/Dome")
    dome.CreateIntensityAttr(float(cfg.dome_intensity))
    # Exposure is a UsdLux.LightAPI attr (stops, additive in log2); lift it if the RTX frame
    # comes out underexposed (rgb.mean below ~30) after warmup. Default 0.0 == no change.
    if abs(float(cfg.dome_exposure)) > 1e-9:
        dome.CreateExposureAttr(float(cfg.dome_exposure))

    distant = UsdLux.DistantLight.Define(stage, scope_path + "/Distant")
    distant.CreateIntensityAttr(float(cfg.distant_intensity))
    distant.CreateAngleAttr(1.0)
    # Tilt the distant light down so the table tops + objects get a clear key light.
    _set_xform_rotate_xyz(distant.GetPrim(), Gf.Vec3f(-45.0, 0.0, 25.0))

    LOG.info(
        "Lighting added: DomeLight(intensity=%.0f) + DistantLight(intensity=%.0f)",
        cfg.dome_intensity,
        cfg.distant_intensity,
    )


def _set_xform_rotate_xyz(prim: "Usd.Prim", rot_deg: "Gf.Vec3f") -> None:
    """Set a single xformOp:rotateXYZ on a prim (used for the distant light)."""
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    op = xform.AddRotateXYZOp()
    op.Set(rot_deg)


def assert_assets_exist(rel_paths: list[str]) -> None:
    """Raise FileNotFoundError listing EVERY missing asset (fail loud at startup)."""
    missing: list[str] = []
    for rel in rel_paths:
        abs_path = os.path.join(ASSET_ROOT, rel)
        if not os.path.isfile(abs_path):
            missing.append(abs_path)
    if missing:
        raise FileNotFoundError(
            "Required Isaac assets not found (offline; local-only paths). Missing:\n  "
            + "\n  ".join(missing)
            + f"\nASSET_ROOT={ASSET_ROOT}"
        )
    LOG.info("All %d required assets present under %s", len(rel_paths), ASSET_ROOT)


# --------------------------------------------------------------------------------------
# Math helpers (used by every keyframed object-motion segment).
# --------------------------------------------------------------------------------------
def smoothstep(t: float) -> float:
    """Clamp ``t`` to [0,1] then ease-in-out: t*t*(3-2t)."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _clamp01(t: float) -> float:
    """Clamp ``t`` to [0,1]. Gf.Slerp is undefined outside this range."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t


def lerp_vec(a: "Gf.Vec3d", b: "Gf.Vec3d", t: float) -> "Gf.Vec3d":
    """Component-wise linear interpolation; ``t`` is clamped to [0,1].

    NOTE: this applies NO easing -- callers that want ease-in-out must pass ``smoothstep(t)``
    (every current caller does). The clamp only guards against out-of-range t; it does not
    add easing, so a raw segment t yields LINEAR motion (the snap the spec avoids).
    """
    from pxr import Gf

    t = _clamp01(float(t))
    return Gf.Vec3d(
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def slerp_quat(a: "Gf.Quatd", b: "Gf.Quatd", t: float) -> "Gf.Quatd":
    """Spherical interpolation for OBJECT orientation only (never the camera basis).

    ``t`` is clamped to [0,1] because ``Gf.Slerp`` is undefined outside that range. As with
    ``lerp_vec`` the easing convention is the caller's: pass ``smoothstep(t)`` for ease.
    """
    from pxr import Gf

    return Gf.Slerp(_clamp01(float(t)), a, b)
