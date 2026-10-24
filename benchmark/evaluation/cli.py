# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .prepare import PreparationError, prepare_evidence
from .rubric import load_rubric, RubricError, validate_task_manifest
from .score import aggregate_results, EvidenceError, score_evidence


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_or_jsonl(path: str) -> Any:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as document_error:
        rows = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as line_error:
                raise PreparationError(
                    "runner results are neither JSON nor JSONL; "
                    f"invalid line {line_number}: "
                    f"{line_error}"
                ) from document_error
        if not rows:
            raise PreparationError("runner results file is empty") from document_error
        return rows


def _prepare_payload(payload: Any, rubric: dict[str, Any]) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        payload = payload["results"]
    if isinstance(payload, list):
        if not all(isinstance(row, dict) for row in payload):
            raise PreparationError("every runner results row must be a JSON object")
        return [prepare_evidence(row, rubric) for row in payload]
    if not isinstance(payload, dict):
        raise PreparationError(
            "runner input must be one result object, a results array/manifest, or JSONL"
        )
    return prepare_evidence(payload, rubric)


def _write_json(value: Any, path: str | None) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path:
        Path(path).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interactiveworld-eval")
    parser.add_argument("--rubric", help="alternate rubric.json path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the frozen 120-task rubric")
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="create a criterion-keyed evidence template from a runner artifact",
    )
    prepare_parser.add_argument(
        "artifact",
        help="runner artifact, JSON results array/manifest, or JSONL results file",
    )
    prepare_parser.add_argument("--output")
    score_parser = subparsers.add_parser("score", help="score one evidence JSON object")
    score_parser.add_argument("evidence")
    score_parser.add_argument("--output")
    batch_parser = subparsers.add_parser(
        "score-batch", help="score a JSON array and aggregate it"
    )
    batch_parser.add_argument("evidence")
    batch_parser.add_argument("--allow-partial", action="store_true")
    batch_parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rubric = load_rubric(args.rubric)
        if args.command == "validate":
            validate_task_manifest(rubric)
            _write_json(
                {
                    "status": "OK",
                    "tasks": len(rubric["tasks"]),
                    "protocols": len(rubric["protocols"]),
                },
                None,
            )
            return 0
        if args.command == "prepare":
            payload = _read_json_or_jsonl(args.artifact)
            _write_json(_prepare_payload(payload, rubric), args.output)
            return 0
        payload = _read_json(args.evidence)
        if args.command == "score":
            if not isinstance(payload, dict):
                raise EvidenceError("single evidence input must be a JSON object")
            result = score_evidence(payload, rubric)
            _write_json(result, args.output)
            return 0 if result["status"] == "SCORED" else 1
        if not isinstance(payload, list):
            raise EvidenceError("batch evidence input must be a JSON array")
        results = [score_evidence(item, rubric) for item in payload]
        aggregate = aggregate_results(
            results,
            rubric,
            require_complete=not args.allow_partial,
        )
        _write_json({"results": results, "aggregate": aggregate}, args.output)
        return 0 if aggregate["status"] in {"SCORED", "PARTIAL"} else 1
    except (
        OSError,
        json.JSONDecodeError,
        RubricError,
        EvidenceError,
        PreparationError,
    ) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
