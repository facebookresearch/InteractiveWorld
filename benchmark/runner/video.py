from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report import load_transcript, ReportError


VIDEO_MODES = ("off", "failures", "all")
CAPTION_MODES = ("off", "speaker", "full")
_SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REQUIRED_VALIDITY_CHECKS = (
    "scene_setup_ok",
    "runtime_complete",
    "trace_complete",
    "frames_usable",
    "assistant_transport_ok",
    "human_executor_ok",
    "simulator_ok",
    "no_material_human_leakage",
)


class VideoError(ValueError):
    """The requested observation video cannot be created safely."""


@dataclass(frozen=True)
class VideoConfig:
    mode: str = "off"
    fps: float = 4.0
    width: int = 1280
    height: int = 720
    captions: str = "speaker"
    ffmpeg: str = "ffmpeg"

    def __post_init__(self) -> None:
        if self.mode not in VIDEO_MODES:
            raise VideoError(f"video mode must be one of {VIDEO_MODES}")
        if self.captions not in CAPTION_MODES:
            raise VideoError(f"caption mode must be one of {CAPTION_MODES}")
        if self.fps <= 0:
            raise VideoError("video fps must be positive")
        if self.fps > 120:
            raise VideoError("video fps must not exceed 120")
        if self.width < 2 or self.height < 2:
            raise VideoError("video resolution must be at least 2x2")
        if self.width % 2 or self.height % 2:
            raise VideoError("video width and height must be even for yuv420p")
        if (
            self.width > 8192
            or self.height > 8192
            or self.width * self.height > 33_177_600
        ):
            raise VideoError("video resolution must not exceed 8K pixel area")
        if not math.isfinite(self.fps):
            raise VideoError("video fps must be finite")
        _validate_ffmpeg_setting(self.ffmpeg)


def add_video_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--video",
        choices=VIDEO_MODES,
        default="off",
        help="create observation videos for no tasks, failed tasks, or all tasks",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=4.0,
        help="playback rate for saved observation frames",
    )
    parser.add_argument(
        "--video-resolution",
        default="1280x720",
        metavar="WIDTHxHEIGHT",
        help="output dimensions; both values must be positive even integers",
    )
    parser.add_argument(
        "--video-captions",
        choices=CAPTION_MODES,
        default="speaker",
        help="off, clipped speaker-prefixed utterances, or full spoken utterances",
    )


def video_config_from_namespace(args: argparse.Namespace) -> VideoConfig:
    resolution = getattr(args, "video_resolution", None)
    if resolution is not None:
        width, height = parse_resolution(str(resolution))
    else:
        width = int(getattr(args, "video_width", 1280))
        height = int(getattr(args, "video_height", 720))
    return VideoConfig(
        mode=str(getattr(args, "video", "off")),
        fps=float(getattr(args, "video_fps", 4.0)),
        width=width,
        height=height,
        captions=str(getattr(args, "video_captions", "speaker")),
        ffmpeg="ffmpeg",
    )


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise VideoError("video resolution must use WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise VideoError("video resolution must use integer WIDTHxHEIGHT") from exc
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise VideoError("video resolution must contain positive even dimensions")
    return width, height


def build_videos(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    config: VideoConfig,
) -> list[dict[str, Any]]:
    """Create one result row per runner record; one failure never hides the rest."""
    return [build_video(record, output_dir, config) for record in records]


def validate_video_environment(config: VideoConfig) -> None:
    """Fail before simulation when requested video encoding is unavailable."""
    if config.mode != "off":
        _resolve_executable(config.ffmpeg)


def build_video(
    record: Mapping[str, Any], output_dir: str | Path, config: VideoConfig
) -> dict[str, Any]:
    """Post-process real runner artifacts into an MP4 observation video."""
    task_id = str(record.get("task_id") or "")
    tag = str(record.get("tag") or task_id)
    base = {"task_id": task_id, "tag": tag, "mode": config.mode}
    if not task_id:
        return {
            **base,
            "status": "UNAVAILABLE",
            "reason": "runner record has no task_id",
            "path": None,
        }
    if not _SAFE_TAG.fullmatch(tag) or tag in {".", ".."}:
        return {
            **base,
            "status": "UNAVAILABLE",
            "reason": f"unsafe artifact tag: {tag!r}",
            "path": None,
        }
    if not _selected(record, config.mode):
        return {
            **base,
            "status": "SKIPPED",
            "reason": _skip_reason(record, config.mode),
            "path": None,
        }
    artifact_dir = _artifact_dir(record)
    if artifact_dir is None:
        return {
            **base,
            "status": "UNAVAILABLE",
            "reason": "runner record has no artifact_dir",
            "path": None,
        }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{tag}_observation.mp4"
    try:
        return _encode_video(record, artifact_dir, destination, config, base)
    except (
        OSError,
        ReportError,
        VideoError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as exc:
        return {**base, "status": "UNAVAILABLE", "reason": str(exc), "path": None}


def _encode_video(
    record: Mapping[str, Any],
    artifact_dir: Path,
    destination: Path,
    config: VideoConfig,
    base: Mapping[str, Any],
) -> dict[str, Any]:
    _reject_symlink(destination)
    source = _discover_source(record, artifact_dir)
    if (
        source["kind"] == "existing_mp4"
        and config.captions == "off"
        and source["path"].resolve() == destination
    ):
        return {
            **base,
            "status": "EXISTING",
            "source": "existing_fpv_mp4",
            "path": str(destination),
        }
    command, sidecars, source_kind, frame_count = _prepare_command(
        record, source, destination, config
    )
    completed = _run_ffmpeg(command, config.ffmpeg)
    if completed.returncode != 0:
        reason = completed.stderr.strip()[-1000:] or (
            f"ffmpeg exited {completed.returncode}"
        )
        return _video_error(base, reason, source_kind)
    if not destination.is_file() or destination.stat().st_size == 0:
        return _video_error(
            base, "ffmpeg reported success but produced no video", source_kind
        )
    return {
        **base,
        "status": "CREATED",
        "path": str(destination),
        "source": source_kind,
        "frame_count": frame_count,
        "fps": config.fps,
        "resolution": [config.width, config.height],
        "captions": config.captions,
        "caption_track": (
            "mov_text" if any(path.suffix == ".srt" for path in sidecars) else None
        ),
        "support_files": [str(path) for path in sidecars],
    }


def _run_ffmpeg(
    command: list[str], executable_setting: str
) -> subprocess.CompletedProcess[str]:
    command[0] = _resolve_executable(executable_setting)
    command[1:1] = ["-nostdin", "-loglevel", "error"]
    # This release cannot depend on Meta-only TrustedSubprocess. Every dynamic
    # value is validated and remains a distinct argv element; no shell is used.
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        timeout=600,
    )


def _video_error(base: Mapping[str, Any], reason: str, source: str) -> dict[str, Any]:
    return {
        **base,
        "status": "ERROR",
        "reason": reason,
        "path": None,
        "source": source,
    }


def _selected(record: Mapping[str, Any], mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "all":
        return True
    return _failed(record)


def _failed(record: Mapping[str, Any]) -> bool:
    if record.get("error"):
        return True
    validity = record.get("validity_evidence")
    if not isinstance(validity, Mapping):
        return True
    if set(validity) != set(_REQUIRED_VALIDITY_CHECKS):
        return True
    if any(validity.get(name) is not True for name in _REQUIRED_VALIDITY_CHECKS):
        return True
    native = record.get("native_taskpack_outcome")
    if _native_failed(record, native):
        return True
    release = record.get("release_rubric_outcome")
    adjudicated_status = record.get("release_score_status")
    if adjudicated_status in {"VOID", "UNSCORABLE"}:
        return True
    return bool(
        isinstance(release, Mapping)
        and release.get("status") == "SCORED"
        and release.get("strict_pass") is False
    )


def _native_failed(record: Mapping[str, Any], native: Any) -> bool:
    if not isinstance(native, Mapping):
        return True
    names = (
        "contract_world_success",
        "process_requirements_met",
        "verification_points_met",
        "assistant_declared_done",
        "requires_assistant_done",
    )
    if any(type(native.get(name)) is not bool for name in names):
        return True
    strict = bool(
        native["contract_world_success"]
        and native["process_requirements_met"]
        and native["verification_points_met"]
        and (native["assistant_declared_done"] or not native["requires_assistant_done"])
    )
    if type(native.get("strict_success")) is not bool:
        return True
    if native["strict_success"] != strict or not strict:
        return True
    return not _has_valid_completion(record, native)


def _has_valid_completion(record: Mapping[str, Any], native: Mapping[str, Any]) -> bool:
    task_goal = record.get("task_goal")
    candidates = (
        record.get("progress"),
        task_goal.get("progress") if isinstance(task_goal, Mapping) else None,
        native.get("completion_score"),
    )
    supplied = [value for value in candidates if value is not None]
    if not supplied:
        return False
    if any(
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
        for value in supplied
    ):
        return False
    numbers = [float(value) for value in supplied]
    return all(value == numbers[0] for value in numbers)


def _skip_reason(record: Mapping[str, Any], mode: str) -> str:
    if mode == "off":
        return "video disabled"
    return "task did not have an explicit failed outcome"


def _artifact_dir(record: Mapping[str, Any]) -> Path | None:
    value = record.get("artifact_dir")
    if isinstance(value, str) and value:
        return Path(value).expanduser().resolve()
    manifest = record.get("manifest")
    tag = record.get("tag")
    if isinstance(manifest, str) and manifest and isinstance(tag, str) and tag:
        path = Path(manifest).expanduser()
        if path.is_absolute() and path.name == f"{tag}_manifest.json":
            return path.resolve().parent
    return None


def _discover_source(record: Mapping[str, Any], artifact_dir: Path) -> dict[str, Any]:
    tag = str(record.get("tag") or "")
    if tag:
        existing = (artifact_dir / f"{tag}_fpv.mp4").resolve()
        _require_within(existing, artifact_dir)
        if existing.is_file() and existing.stat().st_size:
            return {"kind": "existing_mp4", "path": existing}
    io_dir = (artifact_dir / f"{tag}_io").resolve()
    _require_within(io_dir, artifact_dir)
    index_path = (io_dir / "index.json").resolve()
    _require_within(index_path, io_dir)
    if not index_path.is_file():
        raise VideoError(
            "no existing FPV MP4 or saved model-input frames; enable IO capture for observation video"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, Mapping):
        raise VideoError(f"IO capture index must be a JSON object: {index_path}")
    frames = _collect_frames(io_dir, index)
    if not frames:
        raise VideoError(f"IO capture contains no readable PNG frames: {index_path}")
    return {"kind": "observation_frames", "path": index_path, "frames": frames}


def _collect_frames(io_dir: Path, index: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    turns = index.get("turns")
    if not isinstance(turns, list):
        raise VideoError("IO capture index has no turns array")
    for turn_row in turns:
        if not isinstance(turn_row, Mapping):
            continue
        turn = turn_row.get("turn")
        requests = turn_row.get("requests")
        request = requests[-1] if isinstance(requests, list) and requests else None
        if isinstance(request, Mapping):
            candidates.extend(_request_frames(io_dir, request, turn))
    if not candidates:
        candidates.extend(_human_frames(io_dir, turns))
    return _deduplicate_frames(candidates)


def _request_frames(
    io_dir: Path, request: Mapping[str, Any], turn: Any
) -> list[dict[str, Any]]:
    request_file = request.get("file")
    records: Any = None
    if isinstance(request_file, str) and request_file:
        path = (io_dir / request_file).resolve()
        _require_within(path, io_dir)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("images") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        records = [{"file": name} for name in request.get("images") or ()]
    return _frame_records(io_dir, records, turn)


def _human_frames(io_dir: Path, turns: Sequence[Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for turn_row in turns:
        if not isinstance(turn_row, Mapping):
            continue
        for human_input in turn_row.get("human_inputs") or ():
            if not isinstance(human_input, Mapping):
                continue
            records = human_input.get("frame_records")
            if not isinstance(records, list):
                records = [{"file": name} for name in human_input.get("images") or ()]
            candidates.extend(_frame_records(io_dir, records, turn_row.get("turn")))
    return candidates


def _frame_records(
    io_dir: Path, records: Sequence[Any], turn: Any
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for order, frame in enumerate(records):
        if not isinstance(frame, Mapping) or not isinstance(frame.get("file"), str):
            continue
        path = (io_dir / str(frame["file"])).resolve()
        _require_within(path, io_dir)
        if not path.is_file():
            continue
        frames.append(
            {
                "path": path,
                "turn": turn,
                "global_index": frame.get("global_index"),
                "digest": frame.get("sha256") or frame.get("same_as") or str(path),
                "order": order,
            }
        )
    return frames


def _deduplicate_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, Any], dict[str, Any]] = {}
    sequence = 0
    for frame in frames:
        global_index = frame.get("global_index")
        key = (
            ("global", global_index)
            if type(global_index) is int
            else ("digest", frame.get("digest"))
        )
        if key not in unique:
            unique[key] = {**frame, "sequence": sequence}
            sequence += 1
    return sorted(
        unique.values(),
        key=lambda frame: (
            0 if type(frame.get("global_index")) is int else 1,
            (
                frame.get("global_index")
                if type(frame.get("global_index")) is int
                else frame["sequence"]
            ),
        ),
    )


def _prepare_command(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    destination: Path,
    config: VideoConfig,
) -> tuple[list[str], list[Path], str, int | None]:
    sidecars: list[Path] = []
    source_kind = str(source["kind"])
    if source_kind == "existing_mp4":
        command = [config.ffmpeg, "-y", "-i", str(source["path"])]
        frame_count: int | None = None
        duration = None
    else:
        frames = list(source["frames"])
        concat_path = destination.with_suffix(".ffconcat")
        _write_concat(concat_path, frames, config.fps)
        sidecars.append(concat_path)
        command = [
            config.ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
        ]
        frame_count = len(frames)
        duration = frame_count / config.fps
    subtitle = _write_subtitles(record, destination, config, source, duration)
    if subtitle is not None:
        sidecars.append(subtitle)
        command.extend(["-i", str(subtitle), "-map", "0:v:0", "-map", "1:0"])
    scale = (
        f"scale={config.width}:{config.height}:force_original_aspect_ratio=decrease,"
        f"pad={config.width}:{config.height}:(ow-iw)/2:(oh-ih)/2"
    )
    command.extend(["-vf", scale, "-c:v", "libx264", "-pix_fmt", "yuv420p"])
    if subtitle is not None:
        command.extend(["-c:s", "mov_text", "-metadata:s:s:0", "language=eng"])
    command.extend(["-movflags", "+faststart", str(destination)])
    return command, sidecars, source_kind, frame_count


def _write_concat(path: Path, frames: Sequence[Mapping[str, Any]], fps: float) -> None:
    _reject_symlink(path)
    duration = 1.0 / fps
    lines = ["ffconcat version 1.0"]
    for frame in frames:
        lines.append(f"file '{_ffconcat_escape(Path(frame['path']))}'")
        lines.append(f"duration {duration:.9f}")
    lines.append(f"file '{_ffconcat_escape(Path(frames[-1]['path']))}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_subtitles(
    record: Mapping[str, Any],
    destination: Path,
    config: VideoConfig,
    source: Mapping[str, Any],
    duration: float | None,
) -> Path | None:
    if config.captions == "off":
        return None
    transcript, _source = load_transcript(record)
    if not transcript:
        return None
    cues = _caption_cues(
        transcript,
        source,
        config,
        duration,
        event_fps=_record_fps(record),
    )
    if not cues:
        return None
    path = destination.with_suffix(".srt")
    _reject_symlink(path)
    blocks = []
    for index, (start, end, text) in enumerate(cues, 1):
        blocks.append(f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def _caption_cues(
    transcript: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    config: VideoConfig,
    duration: float | None,
    *,
    event_fps: float | None,
) -> list[tuple[float, float, str]]:
    messages = [
        message for message in transcript if _caption_text(message, config.captions)
    ]
    if not messages:
        return []
    if source["kind"] == "observation_frames":
        frames = list(source["frames"])
        by_turn: dict[Any, list[str]] = {}
        for message in messages:
            by_turn.setdefault(message.get("turn"), []).append(
                _caption_text(message, config.captions)
            )
        cues = []
        for index, frame in enumerate(frames):
            text = "\n".join(by_turn.get(frame.get("turn"), ()))
            if text:
                cues.append((index / config.fps, (index + 1) / config.fps, text))
        if cues:
            return cues
    timed = _timed_caption_cues(messages, config.captions, event_fps)
    if timed:
        return timed
    total = duration or max(2.0, len(messages) * 2.0)
    slot = total / len(messages)
    return [
        (index * slot, (index + 1) * slot, _caption_text(message, config.captions))
        for index, message in enumerate(messages)
    ]


def _timed_caption_cues(
    messages: Sequence[Mapping[str, Any]], mode: str, event_fps: float | None
) -> list[tuple[float, float, str]]:
    timed = [
        (timestamp, index, message)
        for index, message in enumerate(messages)
        if (timestamp := _message_time(message, event_fps)) is not None
    ]
    if not timed:
        return []
    timed.sort(key=lambda row: (row[0], row[1]))
    cues: list[tuple[float, float, str]] = []
    for index, (start, _order, message) in enumerate(timed):
        next_start = timed[index + 1][0] if index + 1 < len(timed) else start + 2.0
        end = max(start + 0.1, next_start)
        cues.append((start, end, _caption_text(message, mode)))
    return cues


def _message_time(message: Mapping[str, Any], event_fps: float | None) -> float | None:
    value = message.get("time_s")
    if type(value) in (int, float) and math.isfinite(float(value)) and value >= 0:
        return float(value)
    frame = message.get("frame")
    if type(frame) is int and frame >= 0 and event_fps is not None:
        return frame / event_fps
    return None


def _record_fps(record: Mapping[str, Any]) -> float | None:
    value = record.get("fps")
    if type(value) in (int, float) and math.isfinite(float(value)) and value > 0:
        return float(value)
    artifact_dir = _artifact_dir(record)
    if artifact_dir is None:
        return None
    path = _owned_manifest_path(record, artifact_dir)
    if path is None:
        return None
    _require_within(path, artifact_dir)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("fps") if isinstance(payload, Mapping) else None
    if type(value) in (int, float) and math.isfinite(float(value)) and value > 0:
        return float(value)
    return None


def _owned_manifest_path(record: Mapping[str, Any], artifact_dir: Path) -> Path | None:
    tag = record.get("tag")
    if not isinstance(tag, str) or not tag:
        return None
    expected = f"{tag}_manifest.json"
    value = record.get("manifest")
    if isinstance(value, str) and value:
        if "\x00" in value:
            raise VideoError("invalid manifest artifact reference")
        supplied = Path(value).expanduser()
        if supplied.name != expected or (
            not supplied.is_absolute() and supplied != Path(expected)
        ):
            raise VideoError("manifest does not identify the task-owned artifact")
    return (artifact_dir / expected).resolve()


def _caption_text(message: Mapping[str, Any], mode: str) -> str:
    speaker = str(message.get("speaker") or "").capitalize()
    text = str(message.get("text") or "").strip()
    if not text:
        return ""
    value = _sanitize_subtitle(f"{speaker}: {text}" if speaker else text)
    return value if mode == "full" else value[:160]


def _sanitize_subtitle(value: str) -> str:
    safe = "".join(
        character
        for character in value
        if character in {"\n", "\t"} or character.isprintable()
    )
    lines = [line.strip() for line in safe.replace("-->", "→").splitlines()]
    return "\n".join(line for line in lines if line)


def _resolve_executable(value: str) -> str:
    _validate_ffmpeg_setting(value)
    path = shutil.which(value)
    if path is None:
        raise VideoError(f"ffmpeg executable not found: {value}")
    return path


def _ffconcat_escape(path: Path) -> str:
    if "\n" in str(path) or "\r" in str(path):
        raise VideoError("frame path contains a newline")
    return str(path).replace("'", "'\\''")


def _validate_ffmpeg_setting(value: str) -> None:
    if value == "ffmpeg":
        return
    path = Path(value).expanduser()
    if (
        not path.is_absolute()
        or path.name != "ffmpeg"
        or not path.is_file()
        or path.is_symlink()
        or not os.access(path, os.X_OK)
    ):
        raise VideoError(
            "ffmpeg must be the fixed PATH command or an absolute executable named ffmpeg"
        )


def _require_within(path: Path, parent: Path) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise VideoError(f"artifact path escapes its run directory: {path}") from exc


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise VideoError(f"refusing to overwrite symlink: {path}")


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
