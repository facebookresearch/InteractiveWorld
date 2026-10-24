# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from __future__ import annotations

import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


_RELEASE_ROOT = Path(__file__).resolve().parents[2]
_TASKPACK_PATH = (
    _RELEASE_ROOT
    / "benchmark"
    / "tasks"
    / "taskpacks"
    / "prevention"
    / "WATCH-28008.taskpack.json"
)


def _runtime_import_stubs() -> dict[str, types.ModuleType]:
    """Stub render-only imports so the real dispatch module can load offline."""
    stubs = {
        name: types.ModuleType(name)
        for name in (
            "clutter_render_replay",
            "egocentric_camera",
            "gait_player",
            "humanoid_motion",
            "numpy",
            "placement_slots",
            "semantic_executor",
            "smplh_anim_core",
        )
    }
    scene_state = types.ModuleType("scene_state")
    scene_state.Role = types.SimpleNamespace(GRASPABLE="graspable")
    scene_state.SceneState = object
    stubs["scene_state"] = scene_state
    return stubs


def _isaac_env_type():
    """Load the environment contract without importing numpy or Isaac modules."""
    reset = types.ModuleType("simulator.environment.reset")
    placement = types.ModuleType("simulator.environment.interaction_placement")
    placement.settle_or_lock_taskpack_placement = lambda *_args, **_kwargs: None
    with mock.patch.dict(
        sys.modules,
        {
            "numpy": types.ModuleType("numpy"),
            "simulator.environment.reset": reset,
            "simulator.environment.interaction_placement": placement,
        },
    ):
        sys.modules.pop("simulator.environment.isaac_env", None)
        module = importlib.import_module("simulator.environment.isaac_env")
        return module.IsaacEnv


class RuntimeBuilderRegistryTest(unittest.TestCase):
    def test_every_admitted_world_has_an_exact_seeded_generic_builder(self) -> None:
        from simulator.world_catalog.scene_manifest import admitted_scenes

        generic = mock.Mock(return_value=("scene", {}, "", (), ()))
        behavior_import = types.ModuleType("behavior_import")
        behavior_import.build_behavior_generic = generic
        with mock.patch.dict(
            sys.modules,
            {**_runtime_import_stubs(), "behavior_import": behavior_import},
        ):
            sys.modules.pop("simulator.physical_world.episode_runtime", None)
            episode_runtime = importlib.import_module(
                "simulator.physical_world.episode_runtime"
            )
            builders = episode_runtime._scene_builders(seed=37)
            worlds = tuple(admitted_scenes())

            self.assertTrue(worlds)
            self.assertEqual(
                set(worlds),
                {name for name in builders if not name.startswith("spec:")},
            )
            self.assertTrue(all(callable(builders[name]) for name in worlds))
            self.assertTrue(any(name.startswith("spec:") for name in builders))
            builders[worlds[0]]("stage", "app")

        generic.assert_called_once_with(
            "stage",
            "app",
            worlds[0],
            seed=37,
        )

    def test_dynamic_spec_is_visible_to_episode_runtime_dispatch(self) -> None:
        from simulator.environment.runtime_builder_spec import (
            register_runtime_builder_spec,
        )
        from simulator.interaction.task_runtime import builder_spec_from_taskpack
        from simulator.physical_world import task_world_builder

        taskpack = json.loads(_TASKPACK_PATH.read_text(encoding="utf-8"))
        builder_spec, digest = builder_spec_from_taskpack(taskpack)
        scene = f"spec:{taskpack['task_id']}"
        self.addCleanup(task_world_builder.clear_runtime_spec)
        register_runtime_builder_spec(scene, builder_spec, digest)

        with mock.patch.dict(sys.modules, _runtime_import_stubs()):
            sys.modules.pop("task_world_builder", None)
            sys.modules.pop("simulator.physical_world.episode_runtime", None)
            episode_runtime = importlib.import_module(
                "simulator.physical_world.episode_runtime"
            )
            builders = episode_runtime._scene_builders()

        self.assertIn(scene, builders)
        self.assertTrue(callable(builders[scene]))


class GoalOverrideContractTest(unittest.TestCase):
    @staticmethod
    def _environment():
        role = lambda name: types.SimpleNamespace(name=name)
        scene = types.SimpleNamespace(
            entities={
                "cup": types.SimpleNamespace(role=role("GRASPABLE")),
                "spare": types.SimpleNamespace(role=role("GRASPABLE")),
                "bin": types.SimpleNamespace(role=role("CONTAINER")),
                "other_bin": types.SimpleNamespace(role=role("CONTAINER")),
                "table": types.SimpleNamespace(role=role("SURFACE")),
            },
            behavior_task={
                "grasp": ["cup"],
                "container": "bin",
                "goals": [("cup", "bin"), ("cup", "other_bin")],
                "existential_witness": {"cup": ["bin"]},
            },
        )
        environment_type = _isaac_env_type()
        environment = environment_type.__new__(environment_type)
        environment.scene_obj = scene
        return environment, scene

    def test_goal_override_accepts_only_validated_roles_and_reachability(self) -> None:
        environment, scene = self._environment()

        environment._apply_goal_override([("cup", "bin")])

        self.assertEqual([("cup", "bin")], scene.behavior_task["goals"])
        self.assertEqual(["cup"], scene.behavior_task["grasp"])
        self.assertNotIn("container", scene.behavior_task)

    def test_goal_override_rejects_unknown_or_wrong_role_names(self) -> None:
        cases = (
            (("missing", "bin"), "registered graspables"),
            (("cup", "table"), "registered containers"),
            (("spare", "bin"), "validated vocabulary"),
            (("cup", "other_bin"), "reachability witness"),
        )
        for goal, message in cases:
            with self.subTest(goal=goal):
                environment, scene = self._environment()
                original = dict(scene.behavior_task)

                with self.assertRaisesRegex(ValueError, message):
                    environment._apply_goal_override([goal])

                self.assertEqual(original, scene.behavior_task)
