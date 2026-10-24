"""Validated, secret-safe configuration for the public benchmark runner."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import tempfile
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = "interactiveworld-runner-config/1.0"
PROVIDERS = ("http-json", "openai-compatible", "openai", "anthropic", "gemini")
VIDEO_MODES = ("off", "failures", "all")
VIDEO_CAPTIONS = ("off", "speaker", "full")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RUN_PLAN_SCHEMA_VERSION = "interactiveworld-run-plan/1.0"


class ConfigError(ValueError):
    """The public runner configuration is invalid or unsafe."""


@dataclass(frozen=True)
class EndpointConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    timeout_s: float
    frame_cap: int
    tokens: int = 0

    def public_dict(self) -> dict[str, object]:
        """Return serializable configuration without resolving secret values."""
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkConfig:
    task_ids: tuple[str, ...]
    categories: tuple[str, ...]
    assistant_control: str
    assistant: EndpointConfig
    human: EndpointConfig
    gpus: tuple[str, ...]
    output_dir: Path
    run_id: str
    seed: int
    max_ticks: int
    frame_budget: int
    frames_per_tick: int
    max_frames: int
    frame_max_side: int
    human_invalid_attempt_limit: int
    human_http_attempt_limit: int
    human_http_retry_backoff_s: float
    io_capture: str
    graceful_exit: bool
    resume: bool
    fail_fast: bool
    dry_run: bool
    adjudication: Path | None
    video_mode: str
    video_fps: float
    video_resolution: str
    video_captions: str

    def public_dict(self) -> dict[str, object]:
        return {
            "tasks": {
                "ids": list(self.task_ids),
                "categories": list(self.categories),
            },
            "assistant_control": self.assistant_control,
            "assistant": self.assistant.public_dict(),
            "human": self.human.public_dict(),
            "gpus": list(self.gpus),
            "seed": self.seed,
            "max_ticks": self.max_ticks,
            "frame_budget": self.frame_budget,
            "frames_per_tick": self.frames_per_tick,
            "max_frames": self.max_frames,
            "frame_max_side": self.frame_max_side,
            "human_invalid_attempt_limit": self.human_invalid_attempt_limit,
            "human_http_attempt_limit": self.human_http_attempt_limit,
            "human_http_retry_backoff_s": self.human_http_retry_backoff_s,
            "io_capture": self.io_capture,
            "graceful_exit": self.graceful_exit,
            "resume": self.resume,
            "fail_fast": self.fail_fast,
            "dry_run": self.dry_run,
            "adjudication": str(self.adjudication) if self.adjudication else None,
            "video": {
                "mode": self.video_mode,
                "fps": self.video_fps,
                "resolution": self.video_resolution,
                "captions": self.video_captions,
            },
        }


_TOP_LEVEL_KEYS = {
    "schema_version",
    "tasks",
    "assistant_control",
    "assistant",
    "human",
    "runtime",
    "output",
    "evaluation",
    "video",
}
_TASK_KEYS = {"ids", "categories"}
_ENDPOINT_KEYS = {
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "timeout_s",
    "frame_cap",
    "tokens",
}
_RUNTIME_KEYS = {
    "gpus",
    "num_gpus",
    "seed",
    "max_ticks",
    "frame_budget",
    "frames_per_tick",
    "max_frames",
    "frame_max_side",
    "human_invalid_attempt_limit",
    "human_http_attempt_limit",
    "human_http_retry_backoff_s",
    "io_capture",
    "graceful_exit",
    "fail_fast",
}
_OUTPUT_KEYS = {"directory", "run_id", "resume"}
_EVALUATION_KEYS = {"adjudication"}
_VIDEO_KEYS = {"mode", "fps", "resolution", "captions"}


def _object(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a JSON object")
    return value


def _reject_unknown(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown {label} field(s): {', '.join(unknown)}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ConfigError(f"duplicate JSON field: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ConfigError(f"non-finite JSON number is not allowed: {value}")


def load_config_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ConfigError) as exc:
        raise ConfigError(f"cannot read config {path!r}: {exc}") from exc
    root = _object(payload, "configuration")
    _reject_unknown(root, _TOP_LEVEL_KEYS, "top-level")
    schema = root.get("schema_version", CONFIG_SCHEMA_VERSION)
    if schema != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    sections = (
        ("tasks", _TASK_KEYS),
        ("assistant", _ENDPOINT_KEYS),
        ("human", _ENDPOINT_KEYS),
        ("runtime", _RUNTIME_KEYS),
        ("output", _OUTPUT_KEYS),
        ("evaluation", _EVALUATION_KEYS),
        ("video", _VIDEO_KEYS),
    )
    for name, keys in sections:
        section = _object(root.get(name), name)
        _reject_unknown(section, keys, name)
    return root


def _value(namespace: argparse.Namespace, name: str, fallback: Any) -> Any:
    value = getattr(namespace, name, None)
    return fallback if value is None else value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is bool:
        raise ConfigError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    raise ConfigError(f"{label} must be an integer")


def _finite_float(value: Any, label: str) -> float:
    if type(value) is bool:
        raise ConfigError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{label} must be a finite number")
    return result


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{label} must be a boolean")
    return value


def _list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{label} must be a JSON array of strings")
    return value


def _endpoint(
    role: str,
    section: dict[str, Any],
    namespace: argparse.Namespace,
) -> EndpointConfig:
    prefix = f"{role}_"
    provider = _text(
        _value(namespace, prefix + "provider", section.get("provider", "http-json")),
        f"{role}.provider",
    )
    if provider not in PROVIDERS:
        raise ConfigError(f"{role}.provider must be one of {', '.join(PROVIDERS)}")
    model = _text(
        _value(namespace, prefix + "model", section.get("model", "")),
        f"{role}.model",
    )
    base_url = _text(
        _value(namespace, prefix + "base_url", section.get("base_url", "")),
        f"{role}.base_url",
    )
    base_url = _endpoint_base_url(role, provider, base_url)
    api_key_env = _endpoint_key_environment(role, section, namespace)
    timeout_default = 180.0 if role == "assistant" else 360.0
    timeout_s = _finite_float(
        _value(
            namespace, prefix + "timeout_s", section.get("timeout_s", timeout_default)
        ),
        f"{role}.timeout_s",
    )
    frame_cap = _integer(
        _value(namespace, prefix + "frame_cap", section.get("frame_cap", 32)),
        f"{role}.frame_cap",
    )
    tokens = _integer(
        _value(namespace, prefix + "tokens", section.get("tokens", 0)),
        f"{role}.tokens",
    )
    endpoint = EndpointConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_s=timeout_s,
        frame_cap=frame_cap,
        tokens=tokens,
    )
    _validate_endpoint(role, endpoint)
    return endpoint


def _endpoint_base_url(role: str, provider: str, value: str) -> str:
    if provider != "http-json" or value:
        return value
    default_port = "8802" if role == "assistant" else "8801"
    return f"http://127.0.0.1:{default_port}/generate"


def _endpoint_key_environment(
    role: str,
    section: dict[str, Any],
    namespace: argparse.Namespace,
) -> str:
    value = _text(
        _value(
            namespace,
            f"{role}_api_key_env",
            section.get("api_key_env", ""),
        ),
        f"{role}.api_key_env",
    )
    if value and (len(value) > 128 or not _ENV_NAME.fullmatch(value)):
        raise ConfigError(f"{role}.api_key_env is not a valid environment name")
    return value


def _validate_endpoint(role: str, endpoint: EndpointConfig) -> None:
    _validate_endpoint_types(role, endpoint)
    if endpoint.frame_cap < 0 or endpoint.tokens < 0:
        raise ConfigError(f"{role} frame_cap/tokens must be non-negative")
    if endpoint.provider != "http-json" and not endpoint.model:
        raise ConfigError(f"{role}.model is required for {endpoint.provider}")
    if endpoint.provider == "openai-compatible" and not endpoint.base_url:
        raise ConfigError(
            f"{role}.base_url is required for openai-compatible; use provider "
            "'openai' for the hosted OpenAI endpoint"
        )
    if len(endpoint.model) > 512 or any(
        ord(character) < 32 or ord(character) == 127 for character in endpoint.model
    ):
        raise ConfigError(f"{role}.model contains unsupported characters")
    if endpoint.base_url:
        _validate_base_url(role, endpoint.base_url)


def _validate_endpoint_types(role: str, endpoint: EndpointConfig) -> None:
    if not isinstance(endpoint, EndpointConfig):
        raise ConfigError(f"{role} must be an EndpointConfig")
    if endpoint.provider not in PROVIDERS:
        raise ConfigError(f"{role}.provider must be one of {', '.join(PROVIDERS)}")
    for name, value in (
        ("model", endpoint.model),
        ("base_url", endpoint.base_url),
        ("api_key_env", endpoint.api_key_env),
    ):
        if not isinstance(value, str):
            raise ConfigError(f"{role}.{name} must be a string")
    if endpoint.api_key_env and (
        len(endpoint.api_key_env) > 128 or not _ENV_NAME.fullmatch(endpoint.api_key_env)
    ):
        raise ConfigError(f"{role}.api_key_env is not a valid environment name")
    if (
        type(endpoint.timeout_s) not in {int, float}
        or not math.isfinite(endpoint.timeout_s)
        or endpoint.timeout_s <= 0
    ):
        raise ConfigError(f"{role}.timeout_s must be a positive finite number")
    if type(endpoint.frame_cap) is not int or type(endpoint.tokens) is not int:
        raise ConfigError(f"{role} frame_cap/tokens must be integers")


def _validate_base_url(role: str, value: str) -> None:
    if len(value) > 4096 or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise ConfigError(f"{role}.base_url contains unsupported characters")
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError as exc:
        raise ConfigError(f"{role}.base_url is not a valid URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"{role}.base_url must be an http(s) URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigError(f"{role}.base_url has an invalid port") from exc
    if port == 0:
        raise ConfigError(f"{role}.base_url has an invalid port")
    if "@" in parsed.netloc or "#" in value or parsed.query:
        raise ConfigError(
            f"{role}.base_url must not contain userinfo, a query, or a fragment"
        )
    if parsed.scheme == "https" or _is_loopback_host(parsed.hostname):
        return
    raise ConfigError(f"{role}.base_url must use HTTPS unless it is loopback")


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _gpu_items(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = value.split(",")
        if not raw_items or any(not item.strip() for item in raw_items):
            raise ConfigError(f"{label} must not contain empty GPU identifiers")
        items = [item.strip() for item in raw_items]
    elif isinstance(value, list):
        if not value or not all(isinstance(item, (str, int)) for item in value):
            raise ConfigError(f"{label} must contain GPU identifiers")
        items = [str(item).strip() for item in value]
    else:
        raise ConfigError(f"{label} must be a string or JSON array")
    if len(items) != len(set(items)):
        raise ConfigError("GPU identifiers must be unique")
    gpu_index = re.compile(r"^(?:0|[1-9][0-9]*)$")
    if any(len(item) > 20 or not gpu_index.fullmatch(item) for item in items):
        raise ConfigError("GPUs must be decimal physical indices")
    return tuple(items)


def _parse_gpus(value: Any, num_gpus: Any) -> tuple[str, ...]:
    if value is not None and num_gpus is not None:
        raise ConfigError("gpus and num_gpus are mutually exclusive")
    if value is not None:
        return _gpu_items(value, "runtime.gpus")
    count = 1 if num_gpus is None else _integer(num_gpus, "num_gpus")
    if not 1 <= count <= 64:
        raise ConfigError("num_gpus must be between 1 and 64")
    return tuple(str(index) for index in range(count))


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> int:
    result = _integer(value, label)
    invalid = result < 0 if allow_zero else result <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{label} must be {qualifier}")
    return result


def _run_id(
    namespace: argparse.Namespace, output: dict[str, Any], generated: str
) -> str:
    value = _text(
        _value(namespace, "run_id", output.get("run_id", generated)),
        "run_id",
    )
    if len(value) > 64 or not _RUN_ID.fullmatch(value):
        raise ConfigError("run_id may contain only letters, digits, '.', '_' and '-'")
    return value


def _output_dir(
    namespace: argparse.Namespace,
    output: dict[str, Any],
    run_id: str,
) -> Path:
    explicit = _value(namespace, "output_dir", output.get("directory"))
    if explicit is not None:
        value = _text(explicit, "output.directory")
        return _safe_path(value, "output.directory", reject_root=True)
    configured_root = os.environ.get("INTERACTIVEWORLD_RUN_ROOT")
    root = (
        _safe_path(configured_root, "INTERACTIVEWORLD_RUN_ROOT", reject_root=True)
        if configured_root is not None
        else Path(tempfile.gettempdir()).resolve() / "interactiveworld-runs"
    )
    return (root / run_id).resolve()


def _optional_path(value: Any, label: str) -> Path | None:
    if value is None:
        return None
    return _safe_path(_text(value, label), label)


def _safe_path(value: str, label: str, *, reject_root: bool = False) -> Path:
    if not value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ConfigError(f"{label} contains unsupported characters")
    path = Path(value).expanduser().resolve()
    if reject_root and path == Path(path.anchor):
        raise ConfigError(f"{label} must not be a filesystem root")
    return path


def _task_selection(
    namespace: argparse.Namespace, tasks: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cli_task_ids = getattr(namespace, "task_ids", None)
    cli_categories = getattr(namespace, "categories", None)
    if getattr(namespace, "all_tasks", None):
        if cli_task_ids or cli_categories:
            raise ConfigError("--all cannot be combined with --task-id or --category")
        return (), ()
    if cli_task_ids is not None or cli_categories is not None:
        return tuple(cli_task_ids or ()), tuple(cli_categories or ())
    return (
        tuple(_list(tasks.get("ids"), "tasks.ids")),
        tuple(_list(tasks.get("categories"), "tasks.categories")),
    )


def _gpu_selection(
    namespace: argparse.Namespace, runtime: dict[str, Any]
) -> tuple[str, ...]:
    cli_gpus = getattr(namespace, "gpus", None)
    cli_num_gpus = getattr(namespace, "num_gpus", None)
    if cli_gpus is not None:
        return _parse_gpus(cli_gpus, cli_num_gpus)
    if cli_num_gpus is not None:
        return _parse_gpus(None, cli_num_gpus)
    return _parse_gpus(runtime.get("gpus"), runtime.get("num_gpus"))


def _runtime_fields(
    namespace: argparse.Namespace, runtime: dict[str, Any]
) -> dict[str, Any]:
    def value(name: str, default: Any) -> Any:
        return _value(namespace, name, runtime.get(name, default))

    return {
        "seed": _integer(value("seed", 0), "seed"),
        "max_ticks": _positive(value("max_ticks", 0), "max_ticks", allow_zero=True),
        "frame_budget": _positive(value("frame_budget", 24), "frame_budget"),
        "frames_per_tick": _positive(value("frames_per_tick", 4), "frames_per_tick"),
        "max_frames": _positive(value("max_frames", 256), "max_frames"),
        "frame_max_side": _positive(value("frame_max_side", 512), "frame_max_side"),
        "human_invalid_attempt_limit": _positive(
            value("human_invalid_attempt_limit", 8), "human_invalid_attempt_limit"
        ),
        "human_http_attempt_limit": _positive(
            value("human_http_attempt_limit", 3), "human_http_attempt_limit"
        ),
        "human_http_retry_backoff_s": _finite_float(
            value("human_http_retry_backoff_s", 0.25),
            "human_http_retry_backoff_s",
        ),
        "io_capture": _text(value("io_capture", "auto"), "io_capture"),
        "graceful_exit": _boolean(value("graceful_exit", False), "graceful_exit"),
        "fail_fast": _boolean(value("fail_fast", False), "fail_fast"),
    }


def _output_fields(
    namespace: argparse.Namespace,
    output: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "resume": _boolean(
            _value(namespace, "resume", output.get("resume", False)), "resume"
        ),
        "dry_run": _boolean(_value(namespace, "dry_run", False), "dry_run"),
        "adjudication": _optional_path(
            _value(namespace, "adjudication", evaluation.get("adjudication")),
            "evaluation.adjudication",
        ),
    }


def _video_fields(
    namespace: argparse.Namespace, video: dict[str, Any]
) -> dict[str, Any]:
    return {
        "video_mode": _text(
            _value(namespace, "video", video.get("mode", "off")), "video.mode"
        ),
        "video_fps": _finite_float(
            _value(namespace, "video_fps", video.get("fps", 4.0)), "video.fps"
        ),
        "video_resolution": _text(
            _value(namespace, "video_resolution", video.get("resolution", "1280x720")),
            "video.resolution",
        ),
        "video_captions": _text(
            _value(namespace, "video_captions", video.get("captions", "speaker")),
            "video.captions",
        ),
    }


def _with_existing_run_defaults(
    namespace: argparse.Namespace, file_config: dict[str, Any]
) -> dict[str, Any]:
    output = _object(file_config.get("output"), "output")
    resume = _boolean(
        _value(namespace, "resume", output.get("resume", False)), "resume"
    )
    configured = _value(namespace, "output_dir", output.get("directory"))
    if configured is None:
        if resume:
            raise ConfigError("--resume requires an existing explicit output directory")
        return file_config
    directory = _safe_path(
        _text(configured, "output.directory"),
        "output.directory",
        reject_root=True,
    )
    manifest_path = directory / "run_manifest.json"
    if manifest_path.is_symlink():
        raise ConfigError("existing run manifest must not be a symbolic link")
    if not manifest_path.exists():
        if resume:
            raise ConfigError("--resume requires output-dir/run_manifest.json")
        return file_config
    manifest = _load_run_manifest(manifest_path)
    if not resume and manifest.get("status") != "DRY_RUN":
        return file_config
    defaults = _manifest_configuration_defaults(manifest)
    merged = _merge_configuration(defaults, file_config)
    merged_output = dict(_object(merged.get("output"), "output"))
    if getattr(namespace, "run_id", None) is None and "run_id" not in output:
        merged_output["run_id"] = _manifest_run_id(manifest)
    merged["output"] = merged_output
    return merged


def _load_run_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ConfigError) as exc:
        raise ConfigError(f"cannot read existing run manifest: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != _RUN_PLAN_SCHEMA_VERSION
    ):
        raise ConfigError("existing run manifest has an unsupported schema version")
    return value


def _manifest_run_id(manifest: dict[str, Any]) -> str:
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or len(run_id) > 64 or not _RUN_ID.fullmatch(run_id):
        raise ConfigError("existing run manifest has an invalid run_id")
    return run_id


def _manifest_configuration_defaults(manifest: dict[str, Any]) -> dict[str, Any]:
    config = _object(manifest.get("configuration"), "existing configuration")
    runtime_names = _RUNTIME_KEYS - {"num_gpus"}
    runtime = {name: config[name] for name in runtime_names if name in config}
    defaults: dict[str, Any] = {"runtime": runtime}
    for name in ("tasks", "assistant", "human"):
        if name in config:
            defaults[name] = config[name]
    if "assistant_control" in config:
        defaults["assistant_control"] = config["assistant_control"]
    return defaults


def _merge_configuration(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(defaults)
    for name, value in overrides.items():
        prior = merged.get(name)
        if isinstance(prior, dict) and isinstance(value, dict):
            combined = {**prior, **value}
            _drop_overridden_alternative(name, value, combined)
            merged[name] = combined
        else:
            merged[name] = value
    return merged


def _drop_overridden_alternative(
    section: str, supplied: dict[str, Any], combined: dict[str, Any]
) -> None:
    alternatives = {
        "tasks": ("ids", "categories"),
        "runtime": ("gpus", "num_gpus"),
    }
    pair = alternatives.get(section)
    if pair is None or all(name in supplied for name in pair):
        return
    first, second = pair
    if first in supplied:
        combined.pop(second, None)
    if second in supplied:
        combined.pop(first, None)


def resolve_config(
    namespace: argparse.Namespace,
    file_config: dict[str, Any],
    generated_run_id: str,
) -> BenchmarkConfig:
    if getattr(namespace, "runner_args", None) is not None:
        raise ConfigError(
            "raw evaluator arguments are unsupported; use the validated runner options"
        )
    file_config = _with_existing_run_defaults(namespace, file_config)
    tasks = _object(file_config.get("tasks"), "tasks")
    runtime = _object(file_config.get("runtime"), "runtime")
    output = _object(file_config.get("output"), "output")
    evaluation = _object(file_config.get("evaluation"), "evaluation")
    video = _object(file_config.get("video"), "video")
    task_ids, categories = _task_selection(namespace, tasks)
    run_id = _run_id(namespace, output, generated_run_id)
    output_fields = _output_fields(namespace, output, evaluation)
    config = BenchmarkConfig(
        task_ids=task_ids,
        categories=categories,
        assistant_control=_text(
            _value(
                namespace,
                "assistant_control",
                file_config.get("assistant_control", "model"),
            ),
            "assistant_control",
        ),
        assistant=_endpoint(
            "assistant", _object(file_config.get("assistant"), "assistant"), namespace
        ),
        human=_endpoint("human", _object(file_config.get("human"), "human"), namespace),
        gpus=_gpu_selection(namespace, runtime),
        output_dir=_output_dir(namespace, output, run_id),
        run_id=run_id,
        **_runtime_fields(namespace, runtime),
        **output_fields,
        **_video_fields(namespace, video),
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    if not isinstance(config, BenchmarkConfig):
        raise ConfigError("configuration must be a BenchmarkConfig")
    _validate_endpoint("assistant", config.assistant)
    _validate_endpoint("human", config.human)
    _validate_control_fields(config)
    _validate_runtime_types(config)
    _validate_output_config(config)
    _validate_gpu_config(config)
    _validate_video(config)
    _validate_frame_window(config)
    _validate_selection_types(config)


def _validate_control_fields(config: BenchmarkConfig) -> None:
    if not isinstance(
        config.assistant_control, str
    ) or config.assistant_control not in {
        "model",
        "oracle-verbal",
        "silent",
    }:
        raise ConfigError("assistant_control must be model, oracle-verbal, or silent")
    if not isinstance(config.io_capture, str) or config.io_capture not in {
        "auto",
        "off",
    }:
        raise ConfigError("io_capture must be auto or off")


def _validate_output_config(config: BenchmarkConfig) -> None:
    if not isinstance(config.output_dir, Path) or not config.output_dir.is_absolute():
        raise ConfigError("output.directory must resolve to an absolute path")
    if any(
        ord(character) < 32 or ord(character) == 127
        for character in str(config.output_dir)
    ):
        raise ConfigError("output.directory contains unsupported characters")
    if config.output_dir == Path(config.output_dir.anchor):
        raise ConfigError("output.directory must not be a filesystem root")
    if (
        not isinstance(config.run_id, str)
        or len(config.run_id) > 64
        or not _RUN_ID.fullmatch(config.run_id)
    ):
        raise ConfigError("run_id may contain only letters, digits, '.', '_' and '-'")
    if config.adjudication is not None and (
        not isinstance(config.adjudication, Path)
        or not config.adjudication.is_absolute()
    ):
        raise ConfigError("evaluation.adjudication must resolve to an absolute path")
    if config.adjudication is not None and any(
        ord(character) < 32 or ord(character) == 127
        for character in str(config.adjudication)
    ):
        raise ConfigError("evaluation.adjudication contains unsupported characters")


def _validate_gpu_config(config: BenchmarkConfig) -> None:
    if not isinstance(config.gpus, tuple):
        raise ConfigError("runtime.gpus must be a tuple of GPU identifiers")
    if _gpu_items(list(config.gpus), "runtime.gpus") != config.gpus:
        raise ConfigError("runtime.gpus must contain canonical GPU identifiers")


def _validate_runtime_types(config: BenchmarkConfig) -> None:
    integer_fields = {
        "seed": (config.seed, 0),
        "max_ticks": (config.max_ticks, 0),
        "frame_budget": (config.frame_budget, 1),
        "frames_per_tick": (config.frames_per_tick, 1),
        "max_frames": (config.max_frames, 1),
        "frame_max_side": (config.frame_max_side, 1),
        "human_invalid_attempt_limit": (config.human_invalid_attempt_limit, 1),
        "human_http_attempt_limit": (config.human_http_attempt_limit, 1),
    }
    for name, (value, minimum) in integer_fields.items():
        if type(value) is not int or value < minimum:
            raise ConfigError(f"{name} must be an integer of at least {minimum}")
    backoff = config.human_http_retry_backoff_s
    if type(backoff) not in {int, float} or not math.isfinite(backoff) or backoff < 0:
        raise ConfigError("human_http_retry_backoff_s must be non-negative and finite")
    for name in ("graceful_exit", "resume", "fail_fast", "dry_run"):
        if type(getattr(config, name)) is not bool:
            raise ConfigError(f"{name} must be a boolean")


def _validate_selection_types(config: BenchmarkConfig) -> None:
    for name in ("task_ids", "categories"):
        values = getattr(config, name)
        if not isinstance(values, tuple) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ConfigError(f"{name} must be a tuple of non-empty strings")
        if len(values) != len(set(values)):
            raise ConfigError(f"{name} must not contain duplicates")
    folded_categories = [category.casefold() for category in config.categories]
    if len(folded_categories) != len(set(folded_categories)):
        raise ConfigError("task categories must not be repeated")


def _validate_video(config: BenchmarkConfig) -> None:
    if not all(
        isinstance(value, str)
        for value in (
            config.video_mode,
            config.video_resolution,
            config.video_captions,
        )
    ):
        raise ConfigError("video mode, resolution, and captions must be strings")
    if type(config.video_fps) not in {int, float} or not math.isfinite(
        config.video_fps
    ):
        raise ConfigError("video.fps must be a finite number")
    if config.video_mode not in VIDEO_MODES:
        raise ConfigError(f"video.mode must be one of {', '.join(VIDEO_MODES)}")
    if config.video_captions not in VIDEO_CAPTIONS:
        raise ConfigError(f"video.captions must be one of {', '.join(VIDEO_CAPTIONS)}")
    if config.video_fps <= 0:
        raise ConfigError("video.fps must be positive")
    if config.video_fps > 120:
        raise ConfigError("video.fps must not exceed 120")
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", config.video_resolution)
    if match is None:
        raise ConfigError("video.resolution must have WIDTHxHEIGHT form")
    width, height = (int(value) for value in match.groups())
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ConfigError("video.resolution dimensions must be positive even integers")
    if width > 8192 or height > 8192 or width * height > 33_177_600:
        raise ConfigError("video.resolution must not exceed 8K pixel area")
    if config.video_mode != "off" and config.io_capture == "off":
        raise ConfigError("video rendering requires io_capture=auto")
    if config.video_mode != "off" and config.assistant_control == "oracle-verbal":
        raise ConfigError(
            "video rendering is unavailable with oracle-verbal because that "
            "baseline does not produce task-tagged first-person capture"
        )


def _validate_frame_window(config: BenchmarkConfig) -> None:
    if config.max_frames < config.frames_per_tick:
        raise ConfigError("max_frames must be at least frames_per_tick")
    for role, endpoint in (("assistant", config.assistant), ("human", config.human)):
        if endpoint.frame_cap < config.frames_per_tick:
            raise ConfigError(f"{role}.frame_cap must be at least frames_per_tick")


def validate_secret_environment(config: BenchmarkConfig) -> None:
    active: list[tuple[str, EndpointConfig]] = []
    if config.assistant_control == "model":
        active.append(("assistant", config.assistant))
    if config.assistant_control != "oracle-verbal":
        active.append(("human", config.human))
    for role, endpoint in active:
        if endpoint.api_key_env:
            secret = os.environ.get(endpoint.api_key_env, "")
            if not secret:
                raise ConfigError(
                    f"environment variable {endpoint.api_key_env!r} configured "
                    f"for {role} is empty"
                )
            _validate_secret_value(role, endpoint.api_key_env, secret)
            continue
        hosted_candidates = _hosted_key_candidates(endpoint)
        candidates = (f"INTERACTIVEWORLD_{role.upper()}_API_KEY",) + hosted_candidates
        selected = next((name for name in candidates if os.environ.get(name)), None)
        if hosted_candidates and selected is None:
            joined = " or ".join(candidates)
            raise ConfigError(f"{role} hosted endpoint requires {joined}")
        if selected is not None:
            _validate_secret_value(role, selected, os.environ[selected])


def _validate_secret_value(role: str, name: str, value: str) -> None:
    try:
        oversized = len(value.encode("utf-8")) > 8 * 1024
    except UnicodeEncodeError:
        oversized = True
    if oversized or any(character in value for character in "\r\n\x00"):
        raise ConfigError(
            f"{role} API-key environment variable {name!r} contains an unsafe value"
        )


def _hosted_key_candidates(endpoint: EndpointConfig) -> tuple[str, ...]:
    defaults = {
        "openai": ("api.openai.com", ("OPENAI_API_KEY",)),
        "openai-compatible": ("api.openai.com", ("OPENAI_API_KEY",)),
        "anthropic": ("api.anthropic.com", ("ANTHROPIC_API_KEY",)),
        "gemini": (
            "generativelanguage.googleapis.com",
            ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        ),
    }
    hosted = defaults.get(endpoint.provider)
    if hosted is None:
        return ()
    hostname = urllib.parse.urlsplit(endpoint.base_url).hostname
    if not endpoint.base_url or hostname == hosted[0]:
        return hosted[1]
    return ()
