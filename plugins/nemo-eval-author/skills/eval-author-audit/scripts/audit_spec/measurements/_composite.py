#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared measurement engine for composite audit items."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

from measurements.trace_tools import collect_tool_calls, tool_call_counts

try:
    from harbor.models.trajectories import Trajectory  # ty: ignore[unresolved-import]
except ImportError:
    Trajectory = Any  # type: ignore[assignment,misc]

JsonObject: TypeAlias = dict[str, Any]
EvidenceTarget: TypeAlias = tuple[str, int]

DETERMINISTIC_EVIDENCE_KINDS = frozenset({"tool_call"})
JUDGEABLE_EVIDENCE_KINDS = frozenset(
    {
        "environment_state",
        "outcome",
        "output",
        "policy_boundary",
        "state_change",
        "trace_span",
        "user_intent",
        "verifier",
    }
)


@dataclass(frozen=True)
class ToolGateSpec:
    """Describe a tool list that must be present or absent for coverage."""

    item_field: str
    result_field: str
    should_be_observed: bool
    failure_reason: str


@dataclass(frozen=True)
class CompositeSpec:
    """Describe one audit item method handled by the composite engine."""

    method_name: str
    item_kind: str
    details_schema: str
    judgments_schema: str
    tool_gate: ToolGateSpec


def measure_composite(
    audit: JsonObject,
    trajectory: Trajectory,
    *,
    judgments: JsonObject | None,
    spec: CompositeSpec,
) -> JsonObject:
    """Measure one composite item kind against a trace and optional judgments."""
    audit_items = [item for item in audit["items"] if item["kind"] == spec.item_kind]
    observed_tool_calls = collect_tool_calls(trajectory)
    calls_by_tool = _calls_by_tool(observed_tool_calls)
    judgments_by_target = _judgments_by_target(audit_items, judgments, spec=spec)
    item_results = {
        item["name"]: _item_result(
            item,
            calls_by_tool=calls_by_tool,
            judgments_by_target=judgments_by_target,
            spec=spec,
        )
        for item in audit_items
    }
    covered = [name for name, result in item_results.items() if result["covered"]]
    missing = [name for name in item_results if name not in covered]
    return {
        "item_kind": spec.item_kind,
        "covered": covered,
        "details": {
            "schema": spec.details_schema,
            "covered": covered,
            "missing": missing,
            "judgment_input": _judgment_input_summary(judgments, spec=spec),
            "observed_tool_calls": observed_tool_calls,
            "tool_call_counts": tool_call_counts(observed_tool_calls),
            f"{spec.item_kind}_results": item_results,
        },
    }


def _item_result(
    item: JsonObject,
    *,
    calls_by_tool: dict[str, list[JsonObject]],
    judgments_by_target: dict[EvidenceTarget, JsonObject],
    spec: CompositeSpec,
) -> JsonObject:
    """Combine the configured tool gate with every declared evidence predicate."""
    tool_results = _tool_results(item, calls_by_tool=calls_by_tool, spec=spec.tool_gate)
    evidence_results = [
        _evidence_result(
            item["name"],
            evidence_index,
            evidence,
            calls_by_tool=calls_by_tool,
            judgments_by_target=judgments_by_target,
        )
        for evidence_index, evidence in enumerate(item["evidence_required"])
    ]
    missing_reasons = _missing_reasons(tool_results=tool_results, evidence_results=evidence_results, spec=spec)
    return {
        "covered": not missing_reasons,
        spec.tool_gate.result_field: tool_results,
        "evidence_results": evidence_results,
        "missing_reasons": missing_reasons,
    }


def _tool_results(
    item: JsonObject,
    *,
    calls_by_tool: dict[str, list[JsonObject]],
    spec: ToolGateSpec,
) -> list[JsonObject]:
    """Evaluate each configured tool name using presence or absence semantics."""
    results: list[JsonObject] = []
    failure_status = "missing" if spec.should_be_observed else "violated"
    for tool in _dedupe_names(item.get(spec.item_field, [])):
        matches = calls_by_tool.get(tool, [])
        satisfied = bool(matches) is spec.should_be_observed
        results.append(
            {
                "tool": tool,
                "status": "satisfied" if satisfied else failure_status,
                "matches": matches,
            }
        )
    return results


def _evidence_result(
    item_name: str,
    evidence_index: int,
    evidence: JsonObject,
    *,
    calls_by_tool: dict[str, list[JsonObject]],
    judgments_by_target: dict[EvidenceTarget, JsonObject],
) -> JsonObject:
    """Return deterministic or judged status for one evidence requirement."""
    kind = evidence["kind"]
    result: JsonObject = {
        "kind": kind,
        "evidence_index": evidence_index,
        "description": evidence["description"],
    }
    if kind in DETERMINISTIC_EVIDENCE_KINDS:
        tool = evidence["tool"]
        matches = calls_by_tool.get(tool, [])
        result.update(
            {
                "measurement": "deterministic",
                "tool": tool,
                "status": "satisfied" if matches else "missing",
                "matches": matches,
            }
        )
        return result

    if kind in JUDGEABLE_EVIDENCE_KINDS:
        judgment = judgments_by_target.get((item_name, evidence_index))
        if judgment is None:
            result.update(
                {
                    "measurement": "judgment_required",
                    "status": "unjudged",
                }
            )
            return result
        result.update(
            {
                "measurement": "judged",
                "status": judgment["status"],
                "confidence": judgment["confidence"],
                "rationale": judgment["rationale"],
            }
        )
        if "supporting_trace_refs" in judgment:
            result["supporting_trace_refs"] = judgment["supporting_trace_refs"]
        return result

    result["measurement"] = "unsupported"
    result["status"] = "unsupported"
    return result


def _missing_reasons(
    *,
    tool_results: list[JsonObject],
    evidence_results: list[JsonObject],
    spec: CompositeSpec,
) -> list[str]:
    """Return stable reason codes for why one composite item was not covered."""
    reasons: list[str] = []
    if any(result["status"] != "satisfied" for result in tool_results):
        reasons.append(spec.tool_gate.failure_reason)
    if any(result["measurement"] == "deterministic" and result["status"] == "missing" for result in evidence_results):
        reasons.append("missing_tool_call_evidence")
    if any(result["status"] == "unjudged" for result in evidence_results):
        reasons.append("unjudged_evidence")
    if any(result["measurement"] == "judged" and result["status"] != "satisfied" for result in evidence_results):
        reasons.append("judged_evidence_not_satisfied")
    if any(result["status"] == "unsupported" for result in evidence_results):
        reasons.append("unsupported_evidence_kind")
    return reasons


def _judgments_by_target(
    audit_items: list[JsonObject],
    judgments: JsonObject | None,
    *,
    spec: CompositeSpec,
) -> dict[EvidenceTarget, JsonObject]:
    """Index judgments and reject stale, duplicate, or deterministic targets."""
    if judgments is None:
        return {}

    items_by_name = {item["name"]: item for item in audit_items}
    indexed: dict[EvidenceTarget, JsonObject] = {}
    errors: list[str] = []
    item_label = spec.item_kind.replace("_", " ")
    for index, judgment in enumerate(judgments["judgments"]):
        item_name = judgment[spec.item_kind]
        evidence_index = judgment["evidence_index"]
        target = (item_name, evidence_index)
        item = items_by_name.get(item_name)
        if item is None:
            errors.append(f"judgments[{index}] references unknown {item_label} {item_name!r}")
            continue
        evidence_items = item["evidence_required"]
        if evidence_index < 0 or evidence_index >= len(evidence_items):
            errors.append(f"judgments[{index}] references missing evidence index {evidence_index}")
            continue

        target_errors: list[str] = []
        evidence = evidence_items[evidence_index]
        if evidence["kind"] in DETERMINISTIC_EVIDENCE_KINDS:
            target_errors.append(f"judgments[{index}] targets deterministic evidence kind {evidence['kind']!r}")
        if evidence["kind"] != judgment["kind"]:
            target_errors.append(
                f"judgments[{index}] kind {judgment['kind']!r} does not match audit evidence kind {evidence['kind']!r}"
            )
        if evidence["description"] != judgment["description"]:
            target_errors.append(f"judgments[{index}] description does not match audit evidence description")
        if target in indexed:
            target_errors.append(
                f"judgments[{index}] duplicates {item_label} {item_name!r} evidence index {evidence_index}"
            )
        if target_errors:
            errors.extend(target_errors)
            continue
        indexed[target] = judgment

    if errors:
        judgment_label = spec.item_kind.replace("_", "-")
        raise ValueError(f"invalid {judgment_label} judgments:\n" + "\n".join(errors))
    return indexed


def _judgment_input_summary(judgments: JsonObject | None, *, spec: CompositeSpec) -> JsonObject:
    """Return reproducibility metadata about the optional judgment input."""
    if judgments is None:
        return {"provided": False, "judgment_count": 0}
    summary: JsonObject = {
        "provided": True,
        "schema": spec.judgments_schema,
        "trace_sha256": judgments["trace_sha256"],
        "judgment_count": len(judgments["judgments"]),
    }
    judged_by = judgments.get("judged_by")
    if isinstance(judged_by, str) and judged_by.strip():
        summary["judged_by"] = judged_by.strip()
    return summary


def _dedupe_names(names: Iterable[str]) -> list[str]:
    """Dedupe declared names while preserving audit order."""
    return list(dict.fromkeys(names))


def _calls_by_tool(tool_calls: list[JsonObject]) -> dict[str, list[JsonObject]]:
    """Group observed ATIF tool calls by function name."""
    grouped: dict[str, list[JsonObject]] = {}
    for call in tool_calls:
        grouped.setdefault(call["tool"], []).append(call)
    return grouped
