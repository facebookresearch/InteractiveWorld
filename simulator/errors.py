"""Stable exception hierarchy for the public InteractiveWorld simulator API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SimulatorError(RuntimeError):
    """Base class for public simulator failures."""


class SimulatorConfigError(SimulatorError, ValueError):
    """The requested simulator configuration is invalid or unsafe."""


class TaskCatalogError(SimulatorError):
    """The TaskPack catalog could not be read or verified."""


class TaskNotFoundError(TaskCatalogError, KeyError):
    """The requested task identifier is not present in the selected catalog."""


class TaskValidationError(TaskCatalogError, ValueError):
    """A TaskPack or its declared digest does not satisfy the public contract."""


class SimulatorProtocolError(SimulatorError):
    """The simulator request or response violates the wire contract."""


class SimulatorConnectionError(SimulatorError, ConnectionError):
    """A simulator process or endpoint could not be reached."""


class SimulatorTimeoutError(SimulatorError, TimeoutError):
    """A simulator operation exceeded its configured timeout."""


class SimulatorRemoteError(SimulatorError):
    """A valid request was rejected by the simulator process."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"{code}: {message}")


__all__ = [
    "SimulatorConfigError",
    "SimulatorConnectionError",
    "SimulatorError",
    "SimulatorProtocolError",
    "SimulatorRemoteError",
    "SimulatorTimeoutError",
    "TaskCatalogError",
    "TaskNotFoundError",
    "TaskValidationError",
]
