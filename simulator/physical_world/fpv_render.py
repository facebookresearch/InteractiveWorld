"""Render pipeline: replicator render-products + rgb annotators, capture, validate, encode.

SINGLE CLOCK rule: ``rep.orchestrator.step(...)`` is the ONLY time advance during capture
— there is NO extra ``app.update()`` between captures. capture_fps == playback_fps (1:1) for
natural pacing.

The annotator path mirrors the verified multi_camera.py example: ``rep.annotators.get('rgb')``
+ ``.attach(rp)`` before the first step, then ``annot.get_data()`` after each step. The
orchestrator.step kwargs (delta_time / pause_timeline / rt_subframes) are the 5.1 form per
the spec; the live orchestrator adjusts if a kwarg is absent (we degrade gracefully).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fpv_core import AppCfg, RenderCfg
    from isaacsim import SimulationApp  # noqa: F401

LOG = logging.getLogger("fpv_render")

DEFAULT_READBACK_RETRIES = 1


class BlackOrEmptyFrameError(RuntimeError):
    """A render product stayed empty or all-black after a bounded warmup."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


class FrameQuality(str, Enum):
    VALID = "valid"
    EMPTY = "empty"


@dataclass(frozen=True)
class FrameDiagnostics:
    quality: FrameQuality
    mean: float | None
    max: float | None
    retries: int
    reason: str


@dataclass(frozen=True)
class FrameReadback:
    rgb: np.ndarray
    diagnostics: FrameDiagnostics


@dataclass(frozen=True)
class CameraRenderPose:
    eye: tuple[float, float, float]
    look: tuple[float, float, float]
    focal_mm: float
    # Roll reference. ``None`` keeps the historical world-+Z levelling; a rigid head mount
    # supplies the head link's own up so the horizon tilts with the wearer.
    up: tuple[float, float, float] | None = None


def setup_products(
    fpv_cam_path: str,
    tpv_cam_path: str,
    cfg: "RenderCfg",
    width: int,
    height: int,
) -> tuple[object, object, object, object]:
    """Create two render products (FPV primary + TPV verification) + rgb annotators.

    Disables capture-on-play, creates both render products, gets two ``rgb`` annotators and
    attaches them BEFORE the first step. Returns (rp_fpv, rp_tpv, a_fpv, a_tpv). No device
    kwarg on the annotator path (not present in this tree).
    """
    import omni.replicator.core as rep

    rep.orchestrator.set_capture_on_play(False)

    rp_fpv = rep.create.render_product(fpv_cam_path, (int(width), int(height)))
    rp_tpv = rep.create.render_product(tpv_cam_path, (int(width), int(height)))

    # Annotator getter + attach form. The PROVEN form on this box (verified against the
    # working render_test2/probe_b/probe_c runs) is rep.AnnotatorRegistry.get_annotator('rgb')
    # + .attach([rp]) (LIST). The rep.annotators.get('rgb') + .attach(rp) (scalar) form exists
    # in some Isaac builds but was NOT exercised here, so prefer the proven path and fall back.
    a_fpv = _get_rgb_annotator(rep)
    a_tpv = _get_rgb_annotator(rep)
    _attach_annotator(a_fpv, rp_fpv)
    _attach_annotator(a_tpv, rp_tpv)

    LOG.info(
        "Render products created (FPV+TPV) at %dx%d; rgb annotators attached",
        width,
        height,
    )
    return rp_fpv, rp_tpv, a_fpv, a_tpv


def _get_rgb_annotator(rep) -> object:
    """Return an 'rgb' annotator, preferring the proven AnnotatorRegistry form."""
    get_annotator = getattr(
        getattr(rep, "AnnotatorRegistry", None), "get_annotator", None
    )
    if get_annotator is not None:
        return get_annotator("rgb")
    # Fallback for builds that only expose rep.annotators.get.
    return rep.annotators.get("rgb")


def _attach_annotator(annot, rp) -> None:
    """Attach an annotator to a render product, trying the proven list form first.

    The verified runs call ``annot.attach([rp])``; some builds also accept a bare ``rp``.
    Try the list form, then degrade to the scalar form on a TypeError so a signature
    difference never aborts setup.
    """
    try:
        annot.attach([rp])
    except TypeError:
        annot.attach(rp)


def settle_and_validate(
    app: "SimulationApp",
    timeline,
    annot,
    cfg: "AppCfg",
    annot_tpv=None,
    *,
    warmup_frames: int | None = None,
    warm_budget_s: float | None = None,
    context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Warm up the RTX pipeline, then one step, then assert the frame is not black/empty.

    Runs ``app.update()`` warmup_frames times, takes ONE zero-dt orchestrator step (single
    clock -- no per-product extra step), and reads the FPV annotator. Aborts with an
    actionable message if the FPV frame is black/empty (usual causes: timeline not playing,
    no lights, wrong rendermode literal). If ``annot_tpv`` is given, the TPV product is read
    after the SAME step and WARNED (not aborted) on black/empty -- cheap insurance that the
    verification product is wired, without failing the run on a degenerate first TPV look-at.
    """
    import omni.replicator.core as rep

    context = dict(context or {})
    initial_updates = (
        int(cfg.warmup_frames) if warmup_frames is None else int(warmup_frames)
    )
    if initial_updates < 0:
        raise ValueError("warmup_frames must be non-negative")
    for _ in range(initial_updates):
        app.update()

    # Prime the readback: on some headless GPU nodes (e.g. MAST) the FIRST orchestrator-step readback
    # returns EMPTY (size=0) -- a one-time async-readback latency the devvm hides (its first read is
    # already ready). Step until the annotator populates; this also WARMS the pipeline so the subsequent
    # capture loop is never empty. A NO-OP where the first read is already non-empty (devvm) -> exactly
    # one step, so that render output is byte-identical to before.
    # KEEP WARMING while the frame is empty OR BLACK. The old loop broke on `size > 0` alone while the
    # acceptance test below also demands `_rgb_max > 0`, so a non-empty-but-black read exited on the first
    # iteration and went straight to the raise -- the 8 retries were unreachable for the very failure they
    # existed to absorb. Seen for real on MAST: one transient RTX warmup black frame aborted an 8-node job
    # before its first rollout. Byte-identical on a healthy node (first read good -> exactly one step);
    # only the non-empty-but-black path behaves differently, and only by trying longer before failing.
    rgb = None
    _warm_t0 = time.perf_counter()
    _warm_budget = (
        float(os.getenv("FPV_WARMUP_BUDGET_S", "120"))
        if warm_budget_s is None
        else float(warm_budget_s)
    )
    if _warm_budget <= 0.0:
        raise ValueError("warm_budget_s must be positive")
    _warm_n = 0
    while True:
        _safe_step(rep, delta_time=0.0, rt_subframes=1)
        rgb = annot.get_data()
        _warm_n += 1
        if rgb is not None and getattr(rgb, "size", 0) > 0 and _rgb_max(rgb) > 0:
            break
        if _warm_n >= 8 and time.perf_counter() - _warm_t0 >= _warm_budget:
            LOG.error(
                "settle_and_validate: still BLACK/EMPTY after %d steps / %.0fs -- giving up",
                _warm_n,
                time.perf_counter() - _warm_t0,
            )
            break
    if _warm_n > 1:
        LOG.info(
            "settle_and_validate: warmup needed %d steps (%.1fs) before a non-black frame",
            _warm_n,
            time.perf_counter() - _warm_t0,
        )
    elapsed_s = time.perf_counter() - _warm_t0
    array = None if rgb is None else np.asarray(rgb)
    empty = array is None or array.size == 0
    maximum = None if empty else _rgb_max(array)
    mean = None if empty else float(array[..., :3].mean())
    diagnostics = {
        **context,
        "readback_steps": _warm_n,
        "elapsed_s": round(float(elapsed_s), 3),
        "shape": None if empty else tuple(int(value) for value in array.shape),
        "mean": mean,
        "max": maximum,
        "frame_state": "empty" if empty else ("black" if maximum == 0 else "valid"),
    }
    LOG.info(
        "settle_and_validate probe candidate=%s attempt=%s eye=%s look=%s "
        "state=%s shape=%s mean=%s max=%s readback_steps=%d elapsed=%.1fs",
        diagnostics.get("candidate", "default"),
        diagnostics.get("attempt", 1),
        diagnostics.get("eye"),
        diagnostics.get("look"),
        diagnostics["frame_state"],
        diagnostics["shape"],
        diagnostics["mean"],
        diagnostics["max"],
        _warm_n,
        elapsed_s,
    )
    if empty or maximum == 0:
        raise BlackOrEmptyFrameError(
            "BLACK/EMPTY FRAME after warmup. Check: (1) timeline.play() was called, "
            "(2) lights were added (DomeLight+DistantLight), (3) /rtx/rendermode='RaytracedLighting' "
            "(normalized literal), (4) DISPLAY is set for headless. See the 5.1.0 headless "
            "black-frame note in the repo rendering skill. "
            f"probe={diagnostics}",
            diagnostics=diagnostics,
        )
    LOG.info(
        "settle_and_validate OK (FPV): frame mean=%.1f max=%d",
        mean,
        maximum,
    )

    if annot_tpv is not None:
        rgb_tpv = annot_tpv.get_data()
        if (
            rgb_tpv is None
            or getattr(rgb_tpv, "size", 0) == 0
            or _rgb_max(rgb_tpv) == 0
        ):
            LOG.warning(
                "settle_and_validate: TPV product read BLACK/EMPTY after warmup (FPV is fine). "
                "The TPV is the verification artifact; check its camera xform / look-at. "
                "Continuing (warn-only)."
            )
        else:
            LOG.info(
                "settle_and_validate OK (TPV): frame mean=%.1f max=%d",
                float(np.asarray(rgb_tpv)[..., :3].mean()),
                _rgb_max(rgb_tpv),
            )
    return diagnostics


def advance_once(rcfg: "RenderCfg") -> None:
    """Advance the SINGLE render clock by exactly one capture period.

    THIS is the one and only time advance per captured frame. Call it ONCE, then
    ``read_and_save`` BOTH render products' annotators so the FPV and TPV PNGs for a given
    frame index correspond to the SAME simulation time. (Calling capture-with-its-own-step
    per product would step the clock twice per frame and desync the two videos -- the spec's
    single-clock violation.)
    """
    import omni.replicator.core as rep

    _safe_step(
        rep,
        delta_time=1.0 / rcfg.capture_fps,
        rt_subframes=rcfg.rt_subframes,
        pause_timeline=False,
    )


def refresh_once(rcfg: "RenderCfg") -> None:
    """Refresh RTX at the current simulation time without advancing physics."""
    import omni.replicator.core as rep

    _safe_step(
        rep,
        delta_time=0.0,
        rt_subframes=rcfg.rt_subframes,
        pause_timeline=False,
    )


def converge_at_current_time(rcfg: "RenderCfg", refresh_frames: int = 16) -> None:
    """Rebuild temporal render history at one authored pose without moving time."""
    if refresh_frames < 0:
        raise ValueError("refresh_frames must be non-negative")
    for _ in range(refresh_frames):
        refresh_once(rcfg)


def read_rgb(
    annot,
    width: int,
    height: int,
    *,
    max_retries: int = DEFAULT_READBACK_RETRIES,
) -> FrameReadback:
    """Read without advancing time; retry only an empty annotator result."""
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    data = None
    retries = 0
    for attempt in range(max_retries + 1):
        data = annot.get_data()
        if data is not None and getattr(data, "size", 0) > 0:
            retries = attempt
            break
    if data is None or getattr(data, "size", 0) == 0:
        rgb = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        quality = FrameQuality.EMPTY
        mean = maximum = None
        reason = "empty annotator readback"
    else:
        rgb = np.asarray(data)[..., :3].astype(np.uint8)
        quality = FrameQuality.VALID
        mean = float(rgb.mean())
        maximum = float(rgb.max())
        reason = "ok"
    diagnostics = FrameDiagnostics(
        quality=quality,
        mean=mean,
        max=maximum,
        retries=retries,
        reason=reason,
    )
    return FrameReadback(rgb=rgb, diagnostics=diagnostics)


def read_and_save(
    annot,
    frame_dir: str,
    frame_idx: int,
    width: int,
    height: int,
    rcfg: "RenderCfg",
    *,
    max_retries: int = DEFAULT_READBACK_RETRIES,
    on_readback: Callable[[FrameReadback], None] | None = None,
    save: bool = True,
) -> "np.ndarray":
    """Read an already-advanced annotator, save frame_<idx>.png, return the rgb array.

    Does NOT advance the clock (``advance_once`` already did). Retries the READ once without
    a second step; if still empty, black-fills at the ACTUAL (height, width) so the PNG
    sequence keeps constant dimensions for libx264 (mixing sizes corrupts the MP4).
    """
    del rcfg
    readback = read_rgb(
        annot,
        width,
        height,
        max_retries=max_retries,
    )
    if on_readback is not None:
        on_readback(readback)
    rgb = readback.rgb
    if readback.diagnostics.quality is FrameQuality.EMPTY:
        LOG.warning(
            "read_and_save frame %d: annotator returned empty twice; black-filling %dx%d",
            frame_idx,
            int(width),
            int(height),
        )
    if save:
        os.makedirs(frame_dir, exist_ok=True)
        out_png = os.path.join(frame_dir, f"frame_{frame_idx:05d}.png")
        _save_png(out_png, rgb)
    return rgb


def validate_frame(rgb: "np.ndarray", frame_idx: int, logger) -> None:
    """Raise on a fully black frame; otherwise log the mean (good range ~40-180)."""
    arr = np.asarray(rgb)
    if arr[..., :3].max() == 0:
        raise RuntimeError(f"black frame {frame_idx}")
    logger.info("frame %d mean=%.1f", frame_idx, float(arr[..., :3].mean()))


def _ffmpeg_exe() -> str:
    """The ffmpeg binary to shell out to.

    A system ffmpeg wins, so a box that has one encodes exactly as it always did. MAST nodes have
    none and ``hand/run_isaac.sh`` never puts ``$ISAAC_ENV/bin`` on PATH, so fall back to the static
    build the conda env already ships inside ``imageio_ffmpeg``; without it ``encode_mp4`` raises and
    a finished render loses its products (the cv2 fallback has no usable codec there either).
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        LOG.warning("no system ffmpeg and imageio_ffmpeg unusable (%s)", exc)
        return "ffmpeg"


def encode_mp4(frame_dir: str, out_path: str, rcfg: "RenderCfg") -> None:
    """Encode the PNG sequence to MP4 via ffmpeg; fall back to cv2.VideoWriter.

    1280x720 is even, so no pad filter is needed. Raises an actionable message if both
    ffmpeg and cv2 fail (frames are still on disk).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        _ffmpeg_exe(),
        "-y",
        "-framerate",
        str(rcfg.playback_fps),
        "-i",
        os.path.join(frame_dir, "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(rcfg.crf),
        "-movflags",
        "+faststart",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.isfile(out_path):
            LOG.info("encode_mp4 (ffmpeg) -> %s", out_path)
            return
        LOG.warning(
            "ffmpeg failed (rc=%s): %s", result.returncode, result.stderr[-2000:]
        )
    except FileNotFoundError:
        LOG.warning("ffmpeg not on PATH; trying cv2.VideoWriter fallback")

    if _encode_with_cv2(frame_dir, out_path, rcfg):
        LOG.info("encode_mp4 (cv2 fallback) -> %s", out_path)
        return

    raise RuntimeError(
        f"Failed to encode MP4 to {out_path} via both ffmpeg and cv2. PNG frames remain in "
        f"{frame_dir}. Install ffmpeg (or opencv-python) in the isaac311 env, or run "
        "imageio-ffmpeg over the frame sequence manually."
    )


def teardown(rps: list, annots: list) -> None:
    """Wait for in-flight writes, detach annotators, destroy render products."""
    import omni.replicator.core as rep

    try:
        rep.orchestrator.wait_until_complete()
    except Exception as exc:  # noqa: BLE001 - cleanup best-effort
        LOG.warning("wait_until_complete raised %s", exc)
    for a in annots:
        try:
            a.detach()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("annotator detach raised %s", exc)
    for rp in rps:
        try:
            rp.destroy()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("render product destroy raised %s", exc)
    LOG.info("teardown complete (%d products, %d annotators)", len(rps), len(annots))


# --------------------------------------------------------------------------------------
# Internal helpers.
# --------------------------------------------------------------------------------------
def _safe_step(
    rep, delta_time: float, rt_subframes: int, pause_timeline: bool | None = None
) -> None:
    """Call orchestrator.step with the 5.1 kwargs, degrading if a kwarg is unsupported.

    The verified 6.0-era source exposes ``rt_subframes``; ``delta_time`` / ``pause_timeline``
    are the spec's 5.1 form. We try the full signature, then progressively drop kwargs so a
    signature mismatch never aborts the render.
    """
    kwargs_full = {"delta_time": delta_time, "rt_subframes": rt_subframes}
    if pause_timeline is not None:
        kwargs_full["pause_timeline"] = pause_timeline
    try:
        rep.orchestrator.step(**kwargs_full)
        return
    except TypeError:
        pass
    # Drop pause_timeline.
    try:
        rep.orchestrator.step(delta_time=delta_time, rt_subframes=rt_subframes)
        return
    except TypeError:
        pass
    # Drop delta_time too (oldest signature only has rt_subframes).
    rep.orchestrator.step(rt_subframes=rt_subframes)


def _rgb_max(rgb) -> int:
    """Max value over the RGB channels (handles RGBA arrays)."""
    arr = np.asarray(rgb)
    if arr.size == 0:
        return 0
    return int(arr[..., :3].max())


def _save_png(path: str, rgb: "np.ndarray") -> None:
    """Save an HxWx3 uint8 array as PNG via PIL, falling back to imageio."""
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        return
    except Exception as exc:  # noqa: BLE001 - try imageio next
        LOG.debug("PIL save failed (%s); trying imageio", exc)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, rgb)
        return
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not save PNG {path} via PIL or imageio: {exc}"
        ) from exc


def _encode_with_cv2(frame_dir: str, out_path: str, rcfg: "RenderCfg") -> bool:
    """Encode PNG frames to MP4 with cv2.VideoWriter (mp4v), RGB->BGR. Returns success."""
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        LOG.warning("cv2 unavailable for fallback encode: %s", exc)
        return False

    frames = sorted(
        f
        for f in os.listdir(frame_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )
    if not frames:
        LOG.warning("cv2 fallback: no frames in %s", frame_dir)
        return False

    first = cv2.imread(os.path.join(frame_dir, frames[0]))
    if first is None:
        return False
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(rcfg.playback_fps), (w, h))
    try:
        for fname in frames:
            img = cv2.imread(os.path.join(frame_dir, fname))  # cv2 reads BGR already
            if img is not None:
                writer.write(img)
    finally:
        writer.release()
    return os.path.isfile(out_path)
