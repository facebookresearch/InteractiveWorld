"""Validated configuration for launching or attaching to a simulator process."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .errors import SimulatorConfigError


_OBSERVATION_PROFILES = frozenset({"agent", "debug"})


def _optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def _require_boolean(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        if type(getattr(instance, name)) is not bool:
            raise SimulatorConfigError(f"{name} must be a boolean")


def _require_positive_number(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(instance, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SimulatorConfigError(f"{name} must be a number")
        if float(value) <= 0:
            raise SimulatorConfigError(f"{name} must be positive")


def _normalize_absolute_paths(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = _optional_path(getattr(instance, name))
        if value is not None and not value.is_absolute():
            raise SimulatorConfigError(f"{name} must be an absolute path")
        object.__setattr__(instance, name, value)


@dataclass(frozen=True)
class AssetConfig:
    """Paths to separately distributed simulator assets.

    Restricted assets stay outside the release. Environment variables remain a
    compatibility fallback, while this object gives applications an explicit and
    inspectable configuration.
    """

    behavior_cache: Path | str | None = None
    behavior_data: Path | str | None = None
    mano_python: Path | str | None = None
    smplh_model: Path | str | None = None
    walk_clip: Path | str | None = None
    human_asset_root: Path | str | None = None
    label_asset_root: Path | str | None = None

    def __post_init__(self) -> None:
        for name in (
            "behavior_cache",
            "behavior_data",
            "mano_python",
            "smplh_model",
            "walk_clip",
            "human_asset_root",
            "label_asset_root",
        ):
            path = _optional_path(getattr(self, name))
            if path is not None and not path.is_absolute():
                raise SimulatorConfigError(f"assets.{name} must be an absolute path")
            object.__setattr__(self, name, path)

    @classmethod
    def from_environment(cls) -> AssetConfig:
        """Resolve the public asset environment variables without changing them."""
        return cls(
            behavior_cache=_optional_path(
                os.environ.get("INTERACTIVEWORLD_BEHAVIOR_CACHE")
            ),
            behavior_data=_optional_path(
                os.environ.get("INTERACTIVEWORLD_BEHAVIOR_DATA")
            ),
            mano_python=_optional_path(os.environ.get("INTERACTIVEWORLD_MANO_PYTHON")),
            smplh_model=_optional_path(os.environ.get("INTERACTIVEWORLD_SMPLH_MODEL")),
            walk_clip=_optional_path(os.environ.get("INTERACTIVEWORLD_WALK_CLIP")),
            human_asset_root=_optional_path(
                os.environ.get("INTERACTIVEWORLD_HUMAN_ASSET_ROOT")
            ),
            label_asset_root=_optional_path(
                os.environ.get("INTERACTIVEWORLD_LABEL_ASSET_ROOT")
            ),
        )

    def environment(self) -> dict[str, str]:
        """Return only configured path overrides for a simulator subprocess."""
        values = {
            "INTERACTIVEWORLD_BEHAVIOR_CACHE": self.behavior_cache,
            "INTERACTIVEWORLD_BEHAVIOR_DATA": self.behavior_data,
            "INTERACTIVEWORLD_MANO_PYTHON": self.mano_python,
            "INTERACTIVEWORLD_SMPLH_MODEL": self.smplh_model,
            "INTERACTIVEWORLD_WALK_CLIP": self.walk_clip,
            "INTERACTIVEWORLD_HUMAN_ASSET_ROOT": self.human_asset_root,
            "INTERACTIVEWORLD_LABEL_ASSET_ROOT": self.label_asset_root,
        }
        return {name: str(path) for name, path in values.items() if path is not None}


@dataclass(frozen=True)
class SimulatorConfig:
    """Stable user-facing controls for one persistent simulator process."""

    gpu: int = 0
    seed: int = 0
    render_observations: bool = True
    observation_profile: str = "agent"
    frames_per_action: int = 4
    render_selected_only: bool = True
    visibility_gating: bool = False
    visibility_checks: bool = True
    strict_acceptance: bool = True
    output_dir: Path | str | None = None
    isaac_python: Path | str | None = None
    isaac_env_dir: Path | str | None = None
    open_timeout_s: float = 2400.0
    step_timeout_s: float = 600.0
    connect_timeout_s: float = 120.0
    assets: AssetConfig | None = None

    def __post_init__(self) -> None:
        if type(self.gpu) is not int or self.gpu < 0:
            raise SimulatorConfigError("gpu must be a non-negative physical index")
        if type(self.seed) is not int:
            raise SimulatorConfigError("seed must be an integer")
        _require_boolean(
            self,
            (
                "render_observations",
                "render_selected_only",
                "visibility_gating",
                "visibility_checks",
                "strict_acceptance",
            ),
        )
        if self.observation_profile not in _OBSERVATION_PROFILES:
            raise SimulatorConfigError("observation_profile must be 'agent' or 'debug'")
        if self.assets is not None and not isinstance(self.assets, AssetConfig):
            raise SimulatorConfigError("assets must be an AssetConfig")
        if type(self.frames_per_action) is not int or self.frames_per_action < 1:
            raise SimulatorConfigError("frames_per_action must be positive")
        _require_positive_number(
            self,
            ("open_timeout_s", "step_timeout_s", "connect_timeout_s"),
        )
        _normalize_absolute_paths(
            self,
            ("output_dir", "isaac_python", "isaac_env_dir"),
        )

    @property
    def resolved_output_dir(self) -> Path:
        """Return the external runtime directory; never point inside the release."""
        if self.output_dir is not None:
            return Path(self.output_dir).expanduser().resolve()
        return Path(tempfile.gettempdir()) / "interactiveworld-simulator"

    @property
    def resolved_assets(self) -> AssetConfig:
        return self.assets or AssetConfig.from_environment()

    def child_environment(self) -> dict[str, str]:
        """Environment overrides required by the isolated Isaac subprocess."""
        values = self.resolved_assets.environment()
        values["INTERACTIVEWORLD_OUTPUT_DIR"] = str(self.resolved_output_dir)
        values["ISAAC_FRAMES_PER_SEG"] = str(self.frames_per_action)
        values["ISAAC_RENDER_SELECTED_ONLY"] = "1" if self.render_selected_only else "0"
        values["ISAAC_VIS_GATE"] = "1" if self.visibility_gating else "0"
        values["ISAAC_VISCHECK"] = "1" if self.visibility_checks else "0"
        if self.isaac_python is not None:
            values["ISAAC_PYTHON"] = str(self.isaac_python)
        if self.isaac_env_dir is not None:
            values["ISAAC_ENV_DIR"] = str(self.isaac_env_dir)
        return values


__all__ = ["AssetConfig", "SimulatorConfig"]
