from __future__ import annotations

import importlib.util
import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


_CLI_PATH = Path(__file__).resolve().parents[2] / "launch" / "simulator.py"


def _load_cli() -> object:
    spec = importlib.util.spec_from_file_location(
        "interactiveworld_test_cli", _CLI_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load simulator CLI from {_CLI_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Task:
    task_id: str
    taskpack_sha256: str
    scene: str
    source_path: Path
    taskpack: dict[str, object]


@dataclass(frozen=True)
class _Result:
    terminated: bool
    truncated: bool
    frame_png: bytes


@dataclass(frozen=True)
class _Preparation:
    steps: tuple[_Result, ...]
    observation: dict[str, object]
    score_baseline: dict[str, object]


class _Config:
    __slots__ = ("values",)

    def __init__(self, **values: object) -> None:
        self.values = values


class SimulatorCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cli = _load_cli()

    def test_import_does_not_load_heavy_runtime(self) -> None:
        self.assertNotIn("isaacsim", sys.modules)
        self.assertNotIn("omni", sys.modules)

    def test_tasks_list_filters_and_emits_summaries(self) -> None:
        task = _Task(
            task_id="WATCH-1",
            taskpack_sha256="abc",
            scene="kitchen",
            source_path=Path("taskpacks/prevention/WATCH-1.taskpack.json"),
            taskpack={
                "executable_spec": {"meta": {"family": "WATCH", "difficulty": "easy"}}
            },
        )
        catalog = MagicMock()
        catalog.list.return_value = (task,)
        public = SimpleNamespace(
            TaskCatalog=SimpleNamespace(bundled=MagicMock(return_value=catalog))
        )
        output = io.StringIO()

        with patch.object(self.cli, "_load_public_api", return_value=public):
            status = self.cli.main(
                [
                    "tasks",
                    "list",
                    "--category",
                    "prevention",
                    "--source-family",
                    "WATCH",
                ],
                stdout=output,
            )

        self.assertEqual(0, status)
        catalog.list.assert_called_once_with(
            category="prevention", source_family="WATCH"
        )
        payload = json.loads(output.getvalue())
        self.assertEqual(1, payload["count"])
        self.assertEqual("WATCH-1", payload["tasks"][0]["task_id"])
        self.assertEqual("prevention", payload["tasks"][0]["category"])

    def test_tasks_show_emits_complete_task(self) -> None:
        task = _Task(
            task_id="WATCH-1",
            taskpack_sha256="abc",
            scene="kitchen",
            source_path=Path("WATCH-1.taskpack.json"),
            taskpack={"task_id": "WATCH-1", "goal": {"predicate": "on"}},
        )
        catalog = MagicMock()
        catalog.get.return_value = task
        public = SimpleNamespace(
            TaskCatalog=SimpleNamespace(bundled=MagicMock(return_value=catalog))
        )
        output = io.StringIO()

        with patch.object(self.cli, "_load_public_api", return_value=public):
            status = self.cli.main(["tasks", "show", "WATCH-1"], stdout=output)

        self.assertEqual(0, status)
        catalog.get.assert_called_once_with("WATCH-1")
        self.assertEqual("WATCH-1", json.loads(output.getvalue())["task_id"])

    def test_capabilities_does_not_launch_simulator(self) -> None:
        capability = {"primitive": "walk", "status": "implemented"}
        action_capabilities = MagicMock(return_value=(capability,))
        launch = MagicMock()
        public = SimpleNamespace(
            action_capabilities=action_capabilities,
            Simulator=SimpleNamespace(launch=launch),
        )
        output = io.StringIO()

        with patch.object(self.cli, "_load_public_api", return_value=public):
            status = self.cli.main(["capabilities", "--all"], stdout=output)

        self.assertEqual(0, status)
        action_capabilities.assert_called_once_with(include_unavailable=True)
        launch.assert_not_called()
        self.assertEqual(
            "walk", json.loads(output.getvalue())["actions"][0]["primitive"]
        )

    def test_worlds_lists_scenes_without_launching_simulator(self) -> None:
        available_worlds = MagicMock(return_value=("Rs_int", "office_large"))
        launch = MagicMock()
        public = SimpleNamespace(
            available_worlds=available_worlds,
            Simulator=SimpleNamespace(launch=launch),
        )
        output = io.StringIO()

        with patch.object(self.cli, "_load_public_api", return_value=public):
            status = self.cli.main(["worlds"], stdout=output)

        self.assertEqual(0, status)
        available_worlds.assert_called_once_with()
        launch.assert_not_called()
        self.assertEqual(
            {"count": 2, "worlds": ["Rs_int", "office_large"]},
            json.loads(output.getvalue()),
        )

    def test_doctor_returns_nonzero_for_an_incomplete_install(self) -> None:
        output = io.StringIO()
        result = {"ok": False, "checks": [{"name": "asset", "ok": False}]}

        with patch("launch.check_install.check_install", return_value=result):
            status = self.cli.main(["doctor"], stdout=output)

        self.assertEqual(1, status)
        self.assertEqual(result, json.loads(output.getvalue()))

    def test_run_task_is_unprepared_by_default_and_passes_controls(self) -> None:
        simulator = MagicMock()
        simulator.__enter__.return_value = simulator
        simulator.current_scene = "kitchen"
        simulator.observe.return_value = {"task_id": "WATCH-1", "state": "raw"}
        simulator.step.return_value = _Result(
            terminated=True,
            truncated=False,
            frame_png=b"png",
        )
        simulator.score.return_value = {"success": True, "progress": 1.0}
        launch = MagicMock(return_value=simulator)

        public = SimpleNamespace(
            Simulator=SimpleNamespace(launch=launch), SimulatorConfig=_Config
        )
        output = io.StringIO()

        with patch.object(self.cli, "_load_public_api", return_value=public):
            status = self.cli.main(
                [
                    "run",
                    "--task",
                    "WATCH-1",
                    "--action",
                    "[Walk] <item_1> (1)",
                    "--action",
                    "[Grab] <item_1> (1)",
                    "--gpu",
                    "2",
                    "--no-render",
                    "--observation-profile",
                    "debug",
                    "--seed",
                    "7",
                    "--frames-per-action",
                    "2",
                    "--visibility-gating",
                    "--no-visibility-checks",
                    "--no-render-selected-only",
                    "--no-strict-acceptance",
                ],
                stdout=output,
            )

        self.assertEqual(0, status)
        config = launch.call_args.kwargs["config"]
        self.assertEqual(
            {
                "gpu": 2,
                "seed": 7,
                "render_observations": False,
                "observation_profile": "debug",
                "frames_per_action": 2,
                "render_selected_only": False,
                "visibility_gating": True,
                "visibility_checks": False,
                "strict_acceptance": False,
                "output_dir": None,
                "isaac_python": None,
                "isaac_env_dir": None,
            },
            config.values,
        )
        launch.assert_called_once_with(task_id="WATCH-1", config=config)
        simulator.prepare_task.assert_not_called()
        simulator.observe.assert_called_once_with()
        simulator.step.assert_called_once_with("[Walk] <item_1> (1)")
        payload = json.loads(output.getvalue())
        self.assertEqual(1, len(payload["steps"]))
        self.assertEqual(3, payload["steps"][0]["result"]["frame_png"]["size_bytes"])
        self.assertFalse(payload["prepared"])
        self.assertEqual("raw", payload["initial_observation"]["state"])
        self.assertTrue(payload["score"]["success"])

    def test_prepare_task_is_explicit_and_uses_handoff_observation(self) -> None:
        simulator = MagicMock()
        simulator.__enter__.return_value = simulator
        simulator.current_scene = "kitchen"
        simulator.prepare_task.return_value = _Preparation(
            steps=(
                _Result(terminated=False, truncated=False, frame_png=b"first"),
                _Result(terminated=False, truncated=False, frame_png=b"second"),
            ),
            observation={"task_id": "NARRATE-1", "state": "handoff"},
            score_baseline={"success": False, "progress": 0.0},
        )
        simulator.score.return_value = {"success": False, "progress": 0.0}
        launch = MagicMock(return_value=simulator)
        public = SimpleNamespace(
            Simulator=SimpleNamespace(launch=launch), SimulatorConfig=_Config
        )
        output = io.StringIO()

        with patch.object(self.cli, "_load_public_api", return_value=public):
            status = self.cli.main(
                ["run", "--task", "NARRATE-1", "--prepare-task"],
                stdout=output,
            )

        self.assertEqual(0, status)
        simulator.prepare_task.assert_called_once_with()
        simulator.observe.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["prepared"])
        self.assertEqual(2, payload["preparation"]["action_count"])
        self.assertEqual("handoff", payload["initial_observation"]["state"])

    def test_run_scene_opens_raw_world_without_task_preparation(self) -> None:
        simulator = MagicMock()
        simulator.__enter__.return_value = simulator
        simulator.current_scene = "Rs_int"
        simulator.observe.return_value = {"scene": "Rs_int", "objects": ["desk"]}
        simulator.step.return_value = _Result(
            terminated=False,
            truncated=False,
            frame_png=b"scene",
        )
        launch = MagicMock(return_value=simulator)
        public = SimpleNamespace(
            Simulator=SimpleNamespace(launch=launch), SimulatorConfig=_Config
        )
        output = io.StringIO()

        with patch.object(self.cli, "_load_public_api", return_value=public):
            status = self.cli.main(
                [
                    "run",
                    "--scene",
                    "Rs_int",
                    "--action",
                    "[LookAround]",
                    "--no-render",
                ],
                stdout=output,
            )

        self.assertEqual(0, status)
        config = launch.call_args.kwargs["config"]
        launch.assert_called_once_with(scene="Rs_int", config=config)
        simulator.prepare_task.assert_not_called()
        simulator.observe.assert_called_once_with()
        simulator.step.assert_called_once_with("[LookAround]")
        simulator.score.assert_not_called()
        payload = json.loads(output.getvalue())
        self.assertIsNone(payload["task_id"])
        self.assertEqual("Rs_int", payload["scene"])
        self.assertFalse(payload["prepared"])
        self.assertIsNone(payload["score"])

    def test_custom_taskpack_requires_an_explicit_digest(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.cli.main(["run", "--taskpack", "/tmp/custom.taskpack.json"])

    def test_prepare_task_is_rejected_for_a_raw_scene(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.cli.main(["run", "--scene", "Rs_int", "--prepare-task"])
