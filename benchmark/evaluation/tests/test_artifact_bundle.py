# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import tempfile
import tracemalloc
import unittest
from pathlib import Path

from benchmark.evaluation.artifact_bundle import (
    ArtifactBundleError,
    compute_run_evidence_sha256,
    reset_task_artifacts,
    verify_run_evidence_sha256,
)


class ArtifactBundleTest(unittest.TestCase):
    def _write_bundle(self, root: Path, tag: str) -> Path:
        for suffix in ("dialogue", "eval", "manifest", "taskpack_eval"):
            (root / f"{tag}_{suffix}.json").write_text(
                json.dumps({"kind": suffix, "task_id": "ASK-12802"}),
                encoding="utf-8",
            )
        capture = root / f"{tag}_io"
        capture.mkdir()
        (capture / "index.json").write_text(
            json.dumps({"turns": [{"request": "turn00_request.json"}]}),
            encoding="utf-8",
        )
        (capture / "turn00_request.json").write_text(
            json.dumps({"images": [{"file": "frame.png"}]}),
            encoding="utf-8",
        )
        frame = capture / "frame.png"
        frame.write_bytes(b"png evidence")
        return frame

    def test_binary_evidence_is_hashed_without_loading_the_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tag = "TP_ASK-12802_model_s0"
            self._write_bundle(root, tag)
            video = root / f"{tag}_fpv.mp4"
            with video.open("wb") as handle:
                handle.truncate(8 * 1024 * 1024)

            # Independent oracle for the existing versioned wire framing.
            expected = hashlib.sha256(b"interactiveworld-run-evidence/1.0\x00")
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                content = path.read_bytes()
                if path.suffix == ".json":
                    content = json.dumps(
                        json.loads(content),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                expected.update(path.relative_to(root).as_posix().encode() + b"\x00")
                expected.update(str(len(content)).encode() + b"\x00")
                expected.update(content)
            del content

            tracemalloc.start()
            try:
                actual = compute_run_evidence_sha256(root, tag)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertEqual(actual, "sha256:" + expected.hexdigest())
            self.assertLess(peak, 4 * 1024 * 1024)
            with video.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"x")
            self.assertNotEqual(actual, compute_run_evidence_sha256(root, tag))

    def test_digest_binds_all_files_and_is_stable_after_embedding_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tag = "TP_ASK-12802_model_s0"
            frame = self._write_bundle(root, tag)

            original = compute_run_evidence_sha256(root, tag)
            for suffix in ("eval", "taskpack_eval"):
                path = root / f"{tag}_{suffix}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["run_evidence_sha256"] = original
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            self.assertEqual(original, compute_run_evidence_sha256(root, tag))
            verify_run_evidence_sha256(root, tag, original)
            path = root / f"{tag}_eval.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["run_evidence_sha256"] = "sha256:" + "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactBundleError, "embedded"):
                verify_run_evidence_sha256(root, tag, original)
            payload["run_evidence_sha256"] = original
            path.write_text(json.dumps(payload), encoding="utf-8")
            frame.write_bytes(b"changed png evidence")
            self.assertNotEqual(original, compute_run_evidence_sha256(root, tag))

    def test_digest_is_independent_of_json_formatting_and_directory_location(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory() as second,
        ):
            tag = "TP_ASK-12802_model_s0"
            first_root = Path(first)
            second_root = Path(second)
            self._write_bundle(first_root, tag)
            self._write_bundle(second_root, tag)
            path = second_root / f"{tag}_manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps(payload, indent=4), encoding="utf-8")

            self.assertEqual(
                compute_run_evidence_sha256(first_root, tag),
                compute_run_evidence_sha256(second_root, tag),
            )

    def test_missing_core_artifact_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tag = "TP_ASK-12802_model_s0"
            frame = self._write_bundle(root, tag)
            (root / f"{tag}_dialogue.json").unlink()
            with self.assertRaisesRegex(ArtifactBundleError, "missing required"):
                compute_run_evidence_sha256(root, tag)

            (root / f"{tag}_dialogue.json").write_text("{}", encoding="utf-8")
            frame.unlink()
            frame.symlink_to(root / f"{tag}_manifest.json")
            with self.assertRaisesRegex(ArtifactBundleError, "symlink"):
                compute_run_evidence_sha256(root, tag)

    def test_unsafe_tag_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ArtifactBundleError, "unsafe"):
                compute_run_evidence_sha256(temporary, "../escape")

    def test_retry_reset_removes_only_exact_owned_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tag = "TP_ASK-12802_model_s0"
            self._write_bundle(root, tag)
            optional = root / f"{tag}_replay.json.gz"
            optional.write_bytes(b"old replay")
            unrelated = root / f"{tag}_notes.txt"
            unrelated.write_text("user file", encoding="utf-8")

            reset_task_artifacts(root, tag)

            self.assertTrue(unrelated.is_file())
            self.assertFalse(optional.exists())
            self.assertFalse((root / f"{tag}_io").exists())
            self.assertFalse((root / f"{tag}_eval.json").exists())
