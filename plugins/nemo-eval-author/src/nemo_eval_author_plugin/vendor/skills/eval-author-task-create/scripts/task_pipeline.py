#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select actionable audit gaps, scaffold Gym or Harbor drafts, and verify closure."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

_ACTIONABLE_REASON = "not_covered_by_any_input_report"
_DRAFT_PARTS = (".eval-author", "task-drafts")
_PROPOSAL_PARTS = (".eval-author", "proposals")
_SLUG_PREFIX = "cover-"
_MIN_VERIFY_REPORTS = 2


class PipelineError(ValueError):
    """A user-correctable pipeline input error."""


def _read_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from ``path`` or raise ``PipelineError``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"expected a JSON object: {path}")
    return payload


def task_slug_for_tool(tool_name: str) -> str:
    """Return the base Harbor artifact slug for one uncovered audit tool name."""
    normalized = tool_name.strip()
    if not normalized:
        raise PipelineError("tool name is empty")
    slug_body = re.sub(r"[^A-Za-z0-9]+", "-", normalized.replace(".", "-").replace(":", "-"))
    slug_body = slug_body.strip("-").lower()
    if not slug_body or not slug_body[0].isalpha():
        slug_body = f"tool-{slug_body}" if slug_body else "tool"
    return f"{_SLUG_PREFIX}{slug_body}"


def _assign_unique_task_slugs(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return gaps with unique ``task_slug`` values, suffixing ``-2``, ``-3``, ... on collisions."""
    assigned: set[str] = set()
    enriched: list[dict[str, Any]] = []
    for gap in sorted(gaps, key=lambda item: str(item["name"])):
        base_slug = task_slug_for_tool(str(gap["name"]))
        task_slug = base_slug
        suffix = 2
        while task_slug in assigned:
            task_slug = f"{base_slug}-{suffix}"
            suffix += 1
        assigned.add(task_slug)
        enriched.append({**gap, "task_slug": task_slug, "paths": _artifact_paths(task_slug)})
    return enriched


def _task_slug_for_target(report: dict[str, Any], target: str) -> str:
    """Return the assigned task slug for one actionable uncovered tool in ``report``."""
    for gap in _actionable_tools(report):
        if gap["name"] == target:
            return gap["task_slug"]
    raise PipelineError(f"{target!r} is not an actionable uncovered tool")


def _artifact_paths(task_slug: str) -> dict[str, str]:
    """Return the canonical ``.eval-author/`` paths for one task slug."""
    return {
        "proposal": f".eval-author/proposals/{task_slug}-instruction.md",
        "draft": f".eval-author/task-drafts/{task_slug}",
        "measurements": f".eval-author/task-measurements/{task_slug}",
        "task_id": task_slug,
    }


def _gap_from_uncovered_item(item: dict[str, Any]) -> dict[str, Any]:
    """Build one actionable tool gap record from an aggregate ``uncovered_items`` entry."""
    generation = item.get("generation") or {}
    return {
        "name": item.get("name"),
        "kind": "tool",
        "reason": item.get("reason"),
        "description": item.get("description"),
        "focus": generation.get("focus"),
        "needed_tools": generation.get("needed_tools") or [],
        "evidence_required": generation.get("evidence_required") or [],
    }


def _actionable_tools(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return uncovered tool items that are eligible for task generation."""
    gaps: list[dict[str, Any]] = []
    for item in report.get("uncovered_items") or []:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != "tool" or item.get("reason") != _ACTIONABLE_REASON:
            continue
        gaps.append(_gap_from_uncovered_item(item))
    return _assign_unique_task_slugs(gaps)


def _select(report_path: Path, target: str | None) -> dict[str, Any]:
    """List actionable uncovered tools from one aggregate coverage report."""
    gaps = _actionable_tools(_read_json(report_path))
    if target is not None:
        gaps = [gap for gap in gaps if gap["name"] == target]
        if not gaps:
            raise PipelineError(f"{target!r} is not an actionable uncovered tool in {report_path}")
    return {
        "schema": "nemo.eval_author.task_gap_selection.v1",
        "report": str(report_path),
        "actionable_count": len(gaps),
        "actionable_tools": gaps,
        "valid": True,
    }


def _require_draft_destination(path: Path) -> None:
    """Require ``path`` to live under ``.eval-author/task-drafts/``."""
    parts = path.resolve().parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == _DRAFT_PARTS:
            return
    raise PipelineError("draft output must be under .eval-author/task-drafts/")


def _require_proposal_destination(path: Path) -> None:
    """Require ``path`` to live under ``.eval-author/proposals/``."""
    parts = path.resolve().parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == _PROPOSAL_PARTS:
            return
    raise PipelineError("instruction file must be under .eval-author/proposals/")


def _require_task_slug_paths(*, task_slug: str, output: Path, task_name: str, instruction_file: Path) -> None:
    """Require draft, Harbor task name, and proposal filenames to match ``task_slug``."""
    if output.name != task_slug:
        raise PipelineError(f"draft directory must be named {task_slug!r} for the selected tool, got {output.name!r}")
    if Path(task_name).name != task_slug:
        raise PipelineError("draft directory name must match the final component of --task-name")
    expected_instruction = f"{task_slug}-instruction.md"
    if instruction_file.name != expected_instruction:
        raise PipelineError(f"instruction file must be named {expected_instruction!r}, got {instruction_file.name!r}")
    _require_proposal_destination(instruction_file)


def _scaffold(
    *,
    report_path: Path,
    target: str,
    output: Path,
    task_name: str,
    description: str,
    author: str,
    instruction_file: Path,
    provider: str = "harbor",
    gym_python: str | None = None,
) -> dict[str, Any]:
    """Initialize a provider-native draft and install the supplied instruction."""
    report = _read_json(report_path)
    task_slug = _task_slug_for_target(report, target)
    _require_draft_destination(output)
    _require_task_slug_paths(
        task_slug=task_slug,
        output=output,
        task_name=task_name,
        instruction_file=instruction_file,
    )
    if output.exists():
        raise PipelineError(f"draft output already exists: {output}")
    try:
        instruction = instruction_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineError(f"cannot read instruction {instruction_file}: {exc}") from exc
    if not instruction.strip():
        raise PipelineError("instruction file is empty")
    if provider == "gym":
        if not gym_python:
            raise PipelineError("--gym-python must select an existing Gym runtime")
        command = [
            gym_python,
            str(Path(__file__).with_name("gym_scaffold.py")),
            "--out",
            str(output.resolve()),
            "--name",
            task_slug.replace("-", "_"),
            "--instruction-file",
            str(instruction_file.resolve()),
            "--description",
            description,
            "--author",
            author,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
            payload = json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise PipelineError("Gym scaffolder unavailable or returned no report") from exc
        if result.returncode or not isinstance(payload, dict) or not payload.get("written"):
            raise PipelineError("Gym scaffolding failed; inspect the selected runtime and destination")
        return {**payload, "target_tool": target, "task_slug": task_slug, "paths": _artifact_paths(task_slug)}
    harbor = shutil.which("harbor")
    if harbor is None:
        raise PipelineError("harbor executable is not available")

    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            harbor,
            "task",
            "init",
            task_name,
            "--tasks-dir",
            str(output.parent),
            "--description",
            description,
            "--author",
            author,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PipelineError(f"harbor task init failed: {result.stderr or result.stdout}")
    if not output.is_dir():
        raise PipelineError(f"harbor did not create the expected draft: {output}")
    (output / "instruction.md").write_text(instruction, encoding="utf-8")
    return {
        "schema": "nemo.eval_author.harbor_task_draft.v1",
        "target_tool": target,
        "task_slug": task_slug,
        "paths": _artifact_paths(task_slug),
        "draft": str(output),
        "task_name": task_name,
        "scaffolder": "harbor task init",
        "valid": True,
        "written": True,
    }


def _run_ids_from_report(report: dict[str, Any], *, report_path: Path, task_id: str) -> list[str]:
    """Return ATIF ``subject.run_id`` values recorded in one aggregate coverage report."""
    input_reports = report.get("input_reports")
    if not isinstance(input_reports, list) or not input_reports:
        raise PipelineError(f"after report must include input_reports with subject.run_id: {report_path}")
    run_ids: list[str] = []
    for index, entry in enumerate(input_reports):
        if not isinstance(entry, dict):
            raise PipelineError(f"after report input_reports[{index}] must be an object: {report_path}")
        subject = entry.get("subject")
        if not isinstance(subject, dict):
            raise PipelineError(f"after report input_reports[{index}] must include subject: {report_path}")
        if subject.get("task_id") != task_id:
            raise PipelineError(f"after report subject.task_id must be {task_id!r}: {report_path}")
        run_id = subject.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise PipelineError(
                f"after report input_reports[{index}].subject.run_id must be a non-empty string: {report_path}"
            )
        run_ids.append(run_id)
    return run_ids


def _verify(
    before_path: Path, after_paths: list[Path], target: str, evidence_paths: list[Path] | None = None
) -> tuple[int, dict[str, Any]]:
    """Accept only when ``target`` was an actionable gap before and covered in every repeat report."""
    before = _read_json(before_path)
    task_id = _task_slug_for_target(before, target)
    if len(after_paths) < _MIN_VERIFY_REPORTS:
        raise PipelineError(f"at least {_MIN_VERIFY_REPORTS} --after reports are required")
    resolved_paths = [path.resolve() for path in after_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise PipelineError("--after reports must be distinct")

    runs = []
    all_covered = True
    run_ids: list[str] = []
    reports = [_read_json(path) for path in after_paths]
    for path, report in zip(after_paths, reports):
        report_run_ids = _run_ids_from_report(report, report_path=path, task_id=task_id)
        if len(set(report_run_ids)) != 1:
            raise PipelineError("each after report must describe exactly one agent run")
        # A report may contain several measurement methods for the same run.
        report_run_ids = sorted(set(report_run_ids))
        run_ids.extend(report_run_ids)
        covered = target in set(report.get("covered") or [])
        still_uncovered = target in set(report.get("uncovered") or [])
        passed = covered and not still_uncovered
        all_covered = all_covered and passed
        runs.append(
            {
                "report": str(path),
                "run_ids": report_run_ids,
                "covered": covered,
                "passed": passed,
            }
        )

    if len(set(run_ids)) != len(run_ids):
        raise PipelineError("--after reports must come from distinct ATIF subject.run_id values")

    from task_evidence import EvidenceError, verify_receipts

    try:
        agents = verify_receipts(evidence_paths or [], task_id)
        if set(agents) != set(run_ids):
            raise EvidenceError("coverage report run IDs must match all supplied agent receipts")
        for report in reports:
            for entry in report["input_reports"]:
                subject = entry["subject"]
                trace = subject.get("trace")
                if not isinstance(trace, str) or Path(trace).resolve() != Path(agents[subject["run_id"]]["trace"]):
                    raise EvidenceError(
                        "coverage subject.trace must identify the recorded agent trace (use absolute paths)"
                    )
            if target in report.get("covered", []) and not any(
                entry.get("item_kind") == "tool" and target in entry.get("covered", [])
                for entry in report["input_reports"]
            ):
                raise EvidenceError("aggregate tool coverage lacks supporting input evidence")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise PipelineError(f"task execution evidence: {exc}") from exc

    payload = {
        "schema": "nemo.eval_author.task_gap_verification.v2",
        "task_id": task_id,
        "task_validation": "passed",
        "agent_execution": "passed",
        "coverage_closed": all_covered,
        "evidence": [str(path) for path in evidence_paths or []],
        "target_tool": target,
        "before": str(before_path),
        "repeat_count": len(runs),
        "runs": runs,
        "accepted": all_covered,
        "valid": True,
    }
    return (0 if all_covered else 1), payload


def _parser() -> argparse.ArgumentParser:
    """Build the ``select``, ``scaffold``, and ``verify`` subcommand parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="list actionable uncovered tool gaps")
    select.add_argument("--report", type=Path, required=True)
    select.add_argument("--target")

    scaffold = subparsers.add_parser("scaffold", help="initialize a Gym or Harbor native task draft")
    scaffold.add_argument("--provider", choices=("gym", "harbor"), default="harbor")
    scaffold.add_argument("--gym-python", help="Existing Gym Python interpreter (required for Gym)")
    scaffold.add_argument("--report", type=Path, required=True)
    scaffold.add_argument("--target", required=True)
    scaffold.add_argument("--out", type=Path, required=True)
    scaffold.add_argument("--task-name", required=True)
    scaffold.add_argument("--description", required=True)
    scaffold.add_argument("--author", required=True)
    scaffold.add_argument("--instruction-file", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="require two distinct repeats to close one gap")
    verify.add_argument("--before", type=Path, required=True)
    verify.add_argument("--after", type=Path, action="append", required=True)
    verify.add_argument("--target", required=True)
    verify.add_argument(
        "--evidence", type=Path, action="append", help="execution receipt; repeat for all controls and agents"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one pipeline subcommand and print a JSON verdict on stdout."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "select":
            payload = _select(args.report, args.target)
            exit_code = 0
        elif args.command == "scaffold":
            payload = _scaffold(
                report_path=args.report,
                target=args.target,
                output=args.out,
                task_name=args.task_name,
                description=args.description,
                author=args.author,
                instruction_file=args.instruction_file,
                provider=args.provider,
                gym_python=args.gym_python,
            )
            exit_code = 0
        else:
            exit_code, payload = _verify(args.before, args.after, args.target, args.evidence)
    except PipelineError as exc:
        payload = {"valid": False, "error": str(exc)}
        exit_code = 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
