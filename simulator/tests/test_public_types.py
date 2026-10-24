from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from typing import Any

from simulator.actions import Action, action_capabilities
from simulator.config import AssetConfig, SimulatorConfig
from simulator.errors import SimulatorConfigError
from simulator.types import Observation, StepResult


class PublicActionTest(unittest.TestCase):
    def test_action_round_trips_command_syntax(self) -> None:
        action = Action.parse("[PutInside] <pie> (2) <oven> (1)")

        self.assertEqual("PutInside", action.verb)
        self.assertEqual(("pie", "oven"), action.arguments)
        self.assertEqual((2, 1), action.instances)
        self.assertEqual("[PutInside] <pie> (2) <oven> (1)", action.to_command())

    def test_convenience_actions_use_existing_grammar(self) -> None:
        self.assertEqual("[Walk] <oven> (1)", Action.walk("oven").to_command())
        self.assertEqual("[WalkForward]", Action.walk_forward().to_command())
        self.assertEqual("[TurnLeft]", Action.turn_left().to_command())
        self.assertEqual("[TurnRight]", Action.turn_right().to_command())
        self.assertEqual("[LookAround]", Action.look_around().to_command())
        self.assertEqual("[LookAt] <oven> (1)", Action.look_at("oven").to_command())
        self.assertEqual(
            "[PutOn] <mitt> (1) <table> (1)",
            Action.put_on("mitt", "table").to_command(),
        )
        primitives = {capability.primitive for capability in action_capabilities()}
        self.assertIn("navigate_to", primitives)
        self.assertIn("pick", primitives)
        self.assertIn("place_inside", primitives)

    def test_action_rejects_ambiguous_or_malformed_references(self) -> None:
        with self.assertRaises(SimulatorConfigError):
            Action("Grab", ("cup",), (1, 2))
        with self.assertRaises(SimulatorConfigError):
            Action.parse("[Grab] cup")
        with self.assertRaises(SimulatorConfigError):
            Action("Teleport", ("cup",))
        with self.assertRaises(SimulatorConfigError):
            Action("Grab")


class PublicObservationTest(unittest.TestCase):
    def setUp(self) -> None:
        png = b"\x89PNG\r\n\x1a\nexample"
        encoded = base64.b64encode(png).decode("ascii")
        self.payload = {
            "task_id": "WATCH-28008",
            "scene": "Wainscott_0_int",
            "taskpack_sha256": "sha256:" + "a" * 64,
            "taskpack_digest": "sha256:" + "b" * 64,
            "task": "spec:WATCH-28008",
            "task_completed": True,
            "held": ["mitt"],
            "at": ["oven"],
            "objects": ["oven", "pie"],
            "objects_are_visible_only": True,
            "put_verb": "PutInside",
            "frame": encoded,
            "frames": [encoded],
            "frame_segs": ["hold"],
            "relations": [["INSIDE", "pie", "oven"]],
            "score": {
                "success": True,
                "scoreable": True,
                "n_done": 1,
                "n_total": 1,
                "progress": 1.0,
                "prohibitions_ok": True,
            },
            "outcomes": [
                {
                    "primitive": "place_inside",
                    "ok": True,
                    "reason": "OK",
                    "detail": "",
                }
            ],
            "_visible_object_keys_full": ["private-answer-bearing-object"],
        }

    def test_agent_observation_excludes_privileged_state(self) -> None:
        observation = Observation.from_payload(self.payload, profile="agent")

        self.assertEqual(("mitt",), observation.held)
        self.assertEqual("WATCH-28008", observation.task_id)
        self.assertEqual("Wainscott_0_int", observation.scene)
        self.assertEqual("sha256:" + "b" * 64, observation.taskpack_digest)
        self.assertTrue(observation.task_completed)
        self.assertEqual(b"\x89PNG\r\n\x1a\nexample", observation.frame_png)
        self.assertEqual((), observation.relations)
        self.assertIsNone(observation.score)
        self.assertNotIn("private-answer-bearing-object", observation.objects)

    def test_debug_observation_exposes_typed_state(self) -> None:
        observation = Observation.from_payload(self.payload, profile="debug")

        self.assertEqual("INSIDE", observation.relations[0].predicate)
        self.assertEqual("oven", observation.relations[0].target)
        self.assertIsNotNone(observation.score)
        self.assertTrue(observation.score.success if observation.score else False)

    def test_step_result_reports_action_and_task_success_separately(self) -> None:
        result = StepResult.from_payload(
            self.payload,
            profile="agent",
            meta={"phases": {"step.total": {"total_s": 1.5}}},
        )

        self.assertTrue(result.success)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual("place_inside", result.outcomes[0].primitive)
        self.assertIn("step.total", result.timings)


class PublicConfigTest(unittest.TestCase):
    def test_config_normalizes_paths_and_builds_child_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = SimulatorConfig(
                output_dir=str(root / "run"),
                isaac_python=str(root / "isaac-python"),
                assets=AssetConfig(behavior_cache=str(root / "cache")),
            )

            environment = config.child_environment()

        self.assertEqual(str(root / "run"), environment["INTERACTIVEWORLD_OUTPUT_DIR"])
        self.assertEqual(
            str(root / "cache"), environment["INTERACTIVEWORLD_BEHAVIOR_CACHE"]
        )
        self.assertEqual("4", environment["ISAAC_FRAMES_PER_SEG"])
        self.assertEqual("1", environment["ISAAC_RENDER_SELECTED_ONLY"])
        self.assertEqual("0", environment["ISAAC_VIS_GATE"])
        self.assertEqual("1", environment["ISAAC_VISCHECK"])

    def test_config_rejects_invalid_public_values(self) -> None:
        invalid_seed: Any = "7"
        invalid_boolean: Any = "yes"
        with self.assertRaises(SimulatorConfigError):
            SimulatorConfig(gpu=-1)
        with self.assertRaises(SimulatorConfigError):
            SimulatorConfig(observation_profile="hidden-answer")
        with self.assertRaises(SimulatorConfigError):
            SimulatorConfig(output_dir="relative/path")
        with self.assertRaises(SimulatorConfigError):
            SimulatorConfig(seed=invalid_seed)
        with self.assertRaises(SimulatorConfigError):
            SimulatorConfig(render_observations=invalid_boolean)
