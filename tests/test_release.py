# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validate_release import validate


def _run_benchmark(
    *arguments: str, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    child_environment = dict(os.environ if environment is None else environment)
    child_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(ROOT / "launch" / "interactiveworld.py"), *arguments],
        cwd=ROOT,
        env=child_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


class CompleteReleaseTest(unittest.TestCase):
    def test_release_contract(self) -> None:
        result = validate()
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["tasks"], 120)
        self.assertEqual(set(result["assigned_categories"].values()), {20})

    def test_taskpack_launcher_help_does_not_import_isaac(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, str(ROOT / "launch" / "run_taskpack.py"), "--help"],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("--task-id", completed.stdout)
        self.assertIn("--assistant", completed.stdout)

    def test_benchmark_launcher_help_is_available_without_isaac(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["ISAAC_PYTHON"] = "/definitely/not/an/isaac/python"
        completed = subprocess.run(
            [str(ROOT / "launch" / "interactiveworld.sh"), "--help"],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        for option in ("--task-id", "--num-gpus", "--video", "--adjudication"):
            self.assertIn(option, completed.stdout)

    def test_benchmark_list_tasks_reports_frozen_120(self) -> None:
        completed = _run_benchmark("--list-tasks")

        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(120, payload["task_count"])
        self.assertEqual(120, len(payload["tasks"]))
        categories: dict[str, int] = {}
        for task in payload["tasks"]:
            category = task["category"]
            categories[category] = categories.get(category, 0) + 1
        self.assertEqual(6, len(categories))
        self.assertEqual({20}, set(categories.values()))

    def test_benchmark_120_task_four_gpu_dry_run_does_not_start_isaac(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            environment["ISAAC_PYTHON"] = "/definitely/not/an/isaac/python"
            completed = _run_benchmark(
                "--all",
                "--assistant-control",
                "silent",
                "--num-gpus",
                "4",
                "--output-dir",
                temporary,
                "--run-id",
                "static-four-gpu-test",
                "--dry-run",
                environment=environment,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual("DRY_RUN", payload["status"])
            self.assertEqual(120, payload["task_count"])
            self.assertEqual(["0", "1", "2", "3"], payload["configuration"]["gpus"])
            self.assertEqual(4, len(payload["workers"]))
            self.assertEqual(
                [30, 30, 30, 30], [len(row["task_ids"]) for row in payload["workers"]]
            )
            self.assertFalse(any(Path(temporary).rglob("results.jsonl")))
            self.assertFalse(any(Path(temporary).rglob("runner.log")))

    def test_example_config_drives_a_single_task_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = "7"
            environment["ISAAC_PYTHON"] = "/definitely/not/an/isaac/python"
            completed = _run_benchmark(
                "--config",
                str(ROOT / "benchmark" / "runner" / "config.example.json"),
                "--task-id",
                "ASK-13000",
                "--output-dir",
                temporary,
                "--run-id",
                "example-config-test",
                "--dry-run",
                environment=environment,
            )

            self.assertEqual(0, completed.returncode, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(1, payload["task_count"])
            self.assertEqual(["0"], payload["configuration"]["gpus"])
            self.assertEqual(
                "openai", payload["configuration"]["assistant"]["provider"]
            )
            self.assertEqual(
                "openai-compatible",
                payload["configuration"]["human"]["provider"],
            )
