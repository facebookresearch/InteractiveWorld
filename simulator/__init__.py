"""InteractiveWorld simulator release package.

Importing this package is intentionally lightweight: Isaac Sim, PyTorch, and the
SMPL+H model are loaded only when an environment is opened.
"""

from ._bootstrap import activate_runtime_paths

activate_runtime_paths()

from .actions import Action, action_capabilities, ActionSpec
from .api import (
    available_worlds,
    create_environment,
    install_taskpack,
    Simulator,
    TaskInput,
    TaskPreparation,
)
from .catalog import load_taskpack, TaskCatalog, TaskInfo, TaskSpec
from .client import SimulatorClient
from .config import AssetConfig, SimulatorConfig
from .errors import (
    SimulatorConfigError,
    SimulatorConnectionError,
    SimulatorError,
    SimulatorProtocolError,
    SimulatorRemoteError,
    SimulatorTimeoutError,
    TaskCatalogError,
    TaskNotFoundError,
    TaskValidationError,
)
from .types import (
    ActionOutcome,
    Observation,
    Relation,
    Score,
    SimulatorInfo,
    StepResult,
)


__all__ = [
    "Action",
    "ActionOutcome",
    "ActionSpec",
    "AssetConfig",
    "Observation",
    "Relation",
    "Score",
    "Simulator",
    "SimulatorClient",
    "SimulatorConfig",
    "SimulatorConfigError",
    "SimulatorConnectionError",
    "SimulatorError",
    "SimulatorInfo",
    "SimulatorProtocolError",
    "SimulatorRemoteError",
    "SimulatorTimeoutError",
    "StepResult",
    "TaskCatalog",
    "TaskCatalogError",
    "TaskInfo",
    "TaskInput",
    "TaskNotFoundError",
    "TaskPreparation",
    "TaskSpec",
    "TaskValidationError",
    "action_capabilities",
    "activate_runtime_paths",
    "available_worlds",
    "create_environment",
    "install_taskpack",
    "load_taskpack",
]
