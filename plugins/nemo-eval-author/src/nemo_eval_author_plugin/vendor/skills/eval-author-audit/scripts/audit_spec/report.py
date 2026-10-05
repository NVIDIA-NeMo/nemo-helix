#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-trace audit coverage files into one coverage report."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _markdown import AuditMarkdownError  # noqa: E402
from _schema import ITEM_KINDS as AUDIT_ITEM_KINDS  # noqa: E402
from _schema import AuditEnvironmentError, AuditSpecError, item_counts, load_audit_spec  # noqa: E402
from _values import dedupe_names  # noqa: E402

JsonObject: TypeAlias = dict[str, Any]

REPORT_SCHEMA = "nemo.eval_author.audit_coverage_report.v1"
ITEM_KINDS = tuple(sorted(AUDIT_ITEM_KINDS))
SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
COVERAGE_SCHEMA_PATH = SCHEMAS_DIR / "audit_coverage.schema.json"
REPORT_SCHEMA_PATH = SCHEMAS_DIR / "audit_coverage_report.schema.json"


class AuditCoverageInputError(ValueError):
    """Raised when an input coverage file cannot be aggregated."""


class AuditCoverageReportError(ValueError):
    """Raised when the generated aggregate report is invalid."""


@dataclass(frozen=True)
class CoverageInput:
    """One validated input coverage file and its filesystem path."""

    path: Path
    payload: JsonObject


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True, help="audit.md file used as the denominator")
    parser.add_argument("--coverage", type=Path, action="append", help="coverage.json file; repeatable")
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        action="append",
        help="directory to scan recursively for coverage.json files; repeatable",
    )
    parser.add_argument("--out", type=Path, required=True, help="coverage report JSON file to write")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)

    try:
        audit = load_audit_spec(args.audit)
        coverage_paths = _coverage_paths(files=args.coverage, dirs=args.coverage_dir)
        coverage_inputs = [_load_coverage(path) for path in coverage_paths]
        report = _aggregate(audit=audit, audit_path=args.audit, coverage_inputs=coverage_inputs)
        _validate_report(report)
        _write_report(report, out=args.out, compact=args.compact)
    except AuditEnvironmentError as exc:
        _print({"valid": None, "written": False, "error_type": "environment", "error": str(exc)}, compact=args.compact)
        return 2
    except (AuditMarkdownError, AuditSpecError) as exc:
        _print({"valid": False, "written": False, "error_type": "audit_spec", "error": str(exc)}, compact=args.compact)
        return 1
    except AuditCoverageInputError as exc:
        _print(
            {"valid": True, "written": False, "error_type": "coverage_input", "error": str(exc)}, compact=args.compact
        )
        return 1
    except AuditCoverageReportError as exc:
        _print(
            {"valid": True, "written": False, "error_type": "coverage_report", "error": str(exc)}, compact=args.compact
        )
        return 1

    summary: JsonObject = {
        "valid": True,
        "written": True,
        "audit": str(args.audit),
        "coverage_report": str(args.out),
        "coverage_input_count": len(coverage_inputs),
        "warning_count": len(report["warnings"]),
        "warnings": report["warnings"],
        "measured_kinds": report["measured_kinds"],
        "covered_count": report["coverage"]["overall"]["covered_count"],
        "uncovered_count": report["coverage"]["overall"]["uncovered_count"],
        "uncovered": report["uncovered"],
    }
    _print(summary, compact=args.compact)
    return 0


def _coverage_paths(*, files: list[Path] | None, dirs: list[Path] | None) -> list[Path]:
    """Resolve explicit coverage files and recursively discovered coverage files."""
    paths: list[Path] = []
    for path in files or []:
        if not path.is_file():
            raise AuditCoverageInputError(f"coverage file does not exist: {path}")
        paths.append(path)
    for directory in dirs or []:
        if not directory.is_dir():
            raise AuditCoverageInputError(f"coverage directory does not exist: {directory}")
        paths.extend(sorted(directory.rglob("coverage.json")))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)

    if not deduped:
        raise AuditCoverageInputError("no coverage.json files found; pass --coverage or --coverage-dir")
    return deduped


def _load_coverage(path: Path) -> CoverageInput:
    """Load and validate one per-trace coverage.json file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditCoverageInputError(f"could not read coverage file at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditCoverageInputError(f"coverage file at {path} must be a JSON object")

    _validate_json(
        payload,
        schema_path=COVERAGE_SCHEMA_PATH,
        label=f"coverage input {path}",
        error_type=AuditCoverageInputError,
    )
    return CoverageInput(path=path, payload=payload)


def _aggregate(*, audit: JsonObject, audit_path: Path, coverage_inputs: list[CoverageInput]) -> JsonObject:
    """Union covered audit item names and format uncovered items for task generation."""
    audit_items = audit["items"]
    audit_items_by_name = {item["name"]: item for item in audit_items}
    audit_item_kinds = {item["name"]: item["kind"] for item in audit_items}
    audit_counts = item_counts(audit)
    covered_names: set[str] = set()
    input_reports: list[JsonObject] = []
    measured_kinds: set[str] = set()
    warnings: list[JsonObject] = []

    for coverage_input in coverage_inputs:
        payload = coverage_input.payload
        warnings.extend(
            _validate_coverage_matches_audit(
                coverage=payload,
                coverage_path=coverage_input.path,
                audit=audit,
                audit_counts=audit_counts,
                audit_item_kinds=audit_item_kinds,
            )
        )
        input_report = _input_report(coverage_input)
        input_reports.append(input_report)
        measured_kinds.add(payload["item_kind"])
        covered_names.update(payload["covered"])

    covered = [item["name"] for item in audit_items if item["name"] in covered_names]
    uncovered = [item["name"] for item in audit_items if item["name"] not in covered_names]
    measured_kinds_list = [kind for kind in ITEM_KINDS if kind in measured_kinds]
    return {
        "schema": REPORT_SCHEMA,
        "audit": _audit_info(audit, audit_path, audit_counts),
        "measured_kinds": measured_kinds_list,
        "warnings": warnings,
        "input_reports": input_reports,
        "coverage": _coverage_summary(audit_items, covered),
        "covered": covered,
        "uncovered": uncovered,
        "uncovered_items": [
            _uncovered_item(item, audit_items_by_name, measured_kinds=measured_kinds)
            for item in audit_items
            if item["name"] not in covered_names
        ],
    }


def _validate_coverage_matches_audit(
    *,
    coverage: JsonObject,
    coverage_path: Path,
    audit: JsonObject,
    audit_counts: dict[str, int],
    audit_item_kinds: dict[str, str],
) -> list[JsonObject]:
    """Reject stale or incompatible coverage files before aggregating them."""
    warnings: list[JsonObject] = []
    coverage_audit = coverage["audit"]
    for field in ("schema", "agent", "item_count"):
        expected = len(audit["items"]) if field == "item_count" else audit[field]
        if coverage_audit[field] != expected:
            raise AuditCoverageInputError(
                f"{coverage_path}: audit.{field} {coverage_audit[field]!r} does not match current audit {expected!r}"
            )

    if coverage_audit["status"] != audit["status"]:
        warnings.append(
            {
                "kind": "audit_status_mismatch",
                "coverage": str(coverage_path),
                "input_status": coverage_audit["status"],
                "current_status": audit["status"],
            }
        )

    item_kind = coverage["item_kind"]
    if coverage["item_kind_count"] != audit_counts[item_kind]:
        raise AuditCoverageInputError(
            f"{coverage_path}: item_kind_count {coverage['item_kind_count']!r} does not match current "
            f"{item_kind} count {audit_counts[item_kind]!r}"
        )

    for name in coverage["covered"]:
        if name not in audit_item_kinds:
            raise AuditCoverageInputError(f"{coverage_path}: covered item {name!r} is not in the current audit")
        if audit_item_kinds[name] != item_kind:
            raise AuditCoverageInputError(
                f"{coverage_path}: covered item {name!r} has kind {audit_item_kinds[name]!r}, not {item_kind!r}"
            )
    return warnings


def _input_report(coverage_input: CoverageInput) -> JsonObject:
    """Keep one compact traceability record per aggregated coverage file."""
    payload = coverage_input.payload
    return {
        "path": str(coverage_input.path),
        "method": payload["method"]["name"],
        "item_kind": payload["item_kind"],
        "item_kind_count": payload["item_kind_count"],
        "subject": payload["subject"],
        "covered": payload["covered"],
        "covered_count": len(payload["covered"]),
    }


def _coverage_summary(audit_items: list[JsonObject], covered: list[str]) -> JsonObject:
    """Build overall and per-kind count summaries from the audit denominator."""
    covered_set = set(covered)
    by_kind: JsonObject = {}
    for kind in ITEM_KINDS:
        kind_names = [item["name"] for item in audit_items if item["kind"] == kind]
        covered_names = [name for name in kind_names if name in covered_set]
        by_kind[kind] = _count_summary(item_count=len(kind_names), covered_count=len(covered_names))
    return {
        "overall": _count_summary(item_count=len(audit_items), covered_count=len(covered)),
        "by_kind": by_kind,
    }


def _count_summary(*, item_count: int, covered_count: int) -> JsonObject:
    return {
        "item_count": item_count,
        "covered_count": covered_count,
        "uncovered_count": item_count - covered_count,
    }


def _uncovered_item(
    item: JsonObject, audit_items_by_name: dict[str, JsonObject], *, measured_kinds: set[str]
) -> JsonObject:
    """Format one uncovered audit item as direct input to a later generation step."""
    return {
        "name": item["name"],
        "kind": item["kind"],
        "reason": _uncovered_reason(item, measured_kinds),
        "description": item["description"],
        "source_refs": item.get("source_refs", []),
        "generation": {
            "focus": _generation_focus(item),
            "needed_tools": _needed_tools(item, audit_items_by_name),
            "evidence_required": item["evidence_required"],
        },
        "audit_item": item,
    }


def _uncovered_reason(item: JsonObject, measured_kinds: set[str]) -> str:
    """Distinguish true coverage gaps from item kinds no selected method measured."""
    if item["kind"] not in measured_kinds:
        return "not_measured_by_any_method"
    return "not_covered_by_any_input_report"


def _generation_focus(item: JsonObject) -> str:
    """Describe the missing behavior in task-generation terms without inventing a task."""
    if item["kind"] == "tool":
        return f"Exercise a scenario where the agent should call {item['name']}: {item['expected_use']}"
    if item["kind"] == "capability":
        return f"Exercise capability {item['name']}: {item['expected_behavior']}"
    trigger = item["trigger"].rstrip(". ")
    return f"Exercise failure case {item['name']} by triggering {trigger}: {item['expected_behavior']}"


def _needed_tools(item: JsonObject, audit_items_by_name: dict[str, JsonObject]) -> list[str]:
    """Return the declared tool names a generator should consider when closing this gap."""
    if item["kind"] == "tool":
        return dedupe_names([item["name"]])
    if item["kind"] == "capability":
        return dedupe_names(item["required_tools"])

    if "expected_tools" in item:
        tools = item["expected_tools"]
    else:
        tools = _tools_from_capabilities(item, audit_items_by_name)
    prohibited_tools = set(item.get("prohibited_tools", []))
    return dedupe_names(tool for tool in tools if tool not in prohibited_tools)


def _tools_from_capabilities(item: JsonObject, audit_items_by_name: dict[str, JsonObject]) -> list[str]:
    """Collect required tools from capabilities linked by a failure-case item."""
    tools: list[str] = []
    for capability_name in item["applies_to"]:
        capability = audit_items_by_name.get(capability_name)
        if capability is not None and capability["kind"] == "capability":
            tools.extend(capability["required_tools"])
    return tools


def _audit_info(audit: JsonObject, audit_path: Path, counts: dict[str, int]) -> JsonObject:
    """Record denominator identity and counts for aggregate report consumers."""
    return {
        "path": str(audit_path),
        "schema": audit["schema"],
        "agent": audit["agent"],
        "status": audit["status"],
        "item_count": len(audit["items"]),
        "item_counts": counts,
    }


def _validate_report(report: JsonObject) -> None:
    """Validate the aggregate report before writing it."""
    _validate_json(
        report,
        schema_path=REPORT_SCHEMA_PATH,
        label="coverage report",
        error_type=AuditCoverageReportError,
    )


def _validate_json(
    payload: JsonObject,
    *,
    schema_path: Path,
    label: str,
    error_type: type[AuditCoverageInputError] | type[AuditCoverageReportError],
) -> None:
    """Validate JSON with a bundled schema, preserving environment failures."""
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError
    except ImportError as exc:
        raise AuditEnvironmentError(f"jsonschema is required to validate {label}") from exc

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditEnvironmentError(f"could not load {label} JSON Schema from {schema_path}: {exc}") from exc

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise AuditEnvironmentError(f"bundled {label} JSON Schema is invalid: {exc.message}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: (list(error.absolute_path), error.message),
    )
    if errors:
        raise error_type(
            f"{label} failed its JSON Schema: "
            + "\n".join(f"{list(error.absolute_path)}: {error.message}" for error in errors)
        )


def _write_report(report: JsonObject, *, out: Path, compact: bool) -> None:
    """Write the aggregate report after every validation step has passed."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json(report, compact=compact) + "\n", encoding="utf-8")


def _json(payload: JsonObject, *, compact: bool) -> str:
    if compact:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return json.dumps(payload, sort_keys=True, indent=2)


def _print(payload: JsonObject, *, compact: bool) -> None:
    print(_json(payload, compact=compact))


if __name__ == "__main__":
    raise SystemExit(main())
