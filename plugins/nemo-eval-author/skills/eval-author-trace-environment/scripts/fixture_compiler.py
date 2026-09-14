# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure derivation helpers for trace-backed tool-call fixtures."""

from __future__ import annotations

import hashlib
import json
from typing import Any

INVENTORY_SCHEMA = "nemo.eval_author.trace_environment_tool_calls.v1"
PLAN_SCHEMA = "nemo.eval_author.trace_environment_tool_call_plan.v1"
DECISIONS_SCHEMA = "nemo.eval_author.trace_environment_tool_access_decisions.v1"
ACCESS_SCHEMA = "nemo.eval_author.trace_environment_tool_access.v1"
CALL_FIXTURES_SCHEMA = "nemo.eval_author.trace_environment_call_fixtures.v1"
MCP_SCENARIO_SCHEMA = "nemo.eval_author.trace_environment_mcp_scenario.v1"
_UNUSABLE_MARKERS = ("<redacted:", "<omitted:image")
_ACCESS_STATES = frozenset({"real", "mock", "none"})
_MOCK_ADAPTERS = frozenset({"mcp"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def json_sha256(value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _contains_unusable_marker(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in _UNUSABLE_MARKERS)
    if isinstance(value, list):
        return any(_contains_unusable_marker(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_unusable_marker(key) or _contains_unusable_marker(item) for key, item in value.items())
    return False


def _definition_name(definition: dict[str, Any]) -> str | None:
    name = definition.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function = definition.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _definition_schema(definition: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("inputSchema", "input_schema", "parameters"):
        value = definition.get(key)
        if isinstance(value, dict):
            return value
    function = definition.get("function")
    if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
        return function["parameters"]
    return None


def _definition_description(definition: dict[str, Any]) -> str | None:
    description = definition.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    function = definition.get("function")
    if isinstance(function, dict):
        description = function.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
    return None


def _definition_annotations(definition: dict[str, Any]) -> dict[str, Any]:
    annotations = definition.get("annotations")
    if isinstance(annotations, dict):
        return annotations
    function = definition.get("function")
    if isinstance(function, dict) and isinstance(function.get("annotations"), dict):
        return function["annotations"]
    return {}


def _normalized_definition(definition: dict[str, Any]) -> dict[str, Any] | None:
    name = _definition_name(definition)
    input_schema = _definition_schema(definition)
    if name is None or input_schema is None:
        return None
    normalized: dict[str, Any] = {"name": name, "inputSchema": input_schema}
    if description := _definition_description(definition):
        normalized["description"] = description
    annotations = _definition_annotations(definition)
    if annotations:
        normalized["annotations"] = annotations
    return normalized


def _definitions_equivalent(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Compare complete definitions by meaning and incomplete definitions by raw shape."""
    first_normalized = _normalized_definition(first)
    second_normalized = _normalized_definition(second)
    if first_normalized is not None and second_normalized is not None:
        return canonical_json(first_normalized) == canonical_json(second_normalized)
    return canonical_json(first) == canonical_json(second)


def _trajectories(trajectory: dict[str, Any], *, trajectory_path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    trajectories = [(trajectory_path, trajectory)]
    for index, child in enumerate(trajectory.get("subagent_trajectories") or []):
        if isinstance(child, dict):
            trajectories.extend(
                _trajectories(child, trajectory_path=f"{trajectory_path}.subagent_trajectories[{index}]")
            )
    return trajectories


def derive_tool_call_inventory(trajectory: dict[str, Any], *, safe_atif_sha256: str) -> dict[str, Any]:
    definitions_by_name: dict[str, list[dict[str, Any]]] = {}
    trajectories = _trajectories(trajectory)
    for _, current in trajectories:
        agent = current.get("agent")
        raw_definitions = agent.get("tool_definitions") if isinstance(agent, dict) else None
        for definition in raw_definitions or []:
            if not isinstance(definition, dict) or not (name := _definition_name(definition)):
                continue
            known = definitions_by_name.setdefault(name, [])
            if all(not _definitions_equivalent(existing, definition) for existing in known):
                known.append(definition)

    tools: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    unresolved_calls: list[dict[str, Any]] = []
    for trajectory_path, current in trajectories:
        for step in current.get("steps") or []:
            if not isinstance(step, dict):
                continue
            step_id = step.get("step_id")
            observation = step.get("observation")
            raw_results = observation.get("results") if isinstance(observation, dict) else None
            results = [item for item in raw_results or [] if isinstance(item, dict)]
            for call in step.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name = call.get("function_name")
                if not isinstance(name, str) or not name.strip():
                    unresolved_calls.append(
                        {
                            "trajectory_path": trajectory_path,
                            "step_id": step_id,
                            "tool_call_id": call.get("tool_call_id"),
                            "reason": "function_name_missing",
                        }
                    )
                    continue
                name = name.strip()
                if name not in tools:
                    order.append(name)
                    matches = definitions_by_name.get(name, [])
                    normalized = [_normalized_definition(item) for item in matches]
                    complete = [item for item in normalized if item is not None]
                    if not matches:
                        definition_status = "missing"
                        selected_definition = None
                        normalized_definition = None
                    elif len(matches) == 1 and len(complete) == 1:
                        definition_status = "complete"
                        selected_definition = matches[0]
                        normalized_definition = complete[0]
                    elif len(matches) == 1:
                        definition_status = "incomplete"
                        selected_definition = matches[0]
                        normalized_definition = None
                    else:
                        definition_status = "ambiguous"
                        selected_definition = None
                        normalized_definition = None
                    annotations = normalized_definition.get("annotations", {}) if normalized_definition else {}
                    read_only = annotations.get("readOnlyHint")
                    tools[name] = {
                        "name": name,
                        "definition_status": definition_status,
                        "definition": selected_definition,
                        "normalized_definition": normalized_definition,
                        "read_only": read_only if type(read_only) is bool else None,
                        "read_only_evidence": "mcp_annotation" if type(read_only) is bool else None,
                        "calls": [],
                        "uncertainties": [],
                    }
                call_id = call.get("tool_call_id")
                arguments = call.get("arguments")
                tools[name]["calls"].append(
                    {
                        "trajectory_path": trajectory_path,
                        "step_id": step_id,
                        "tool_call_id": call_id,
                        "arguments": arguments,
                        "arguments_status": "object" if isinstance(arguments, dict) else "missing_or_non_object",
                        "matching_observations": [item for item in results if item.get("source_call_id") == call_id],
                    }
                )

    inventory_tools: list[dict[str, Any]] = []
    for name in order:
        tool = tools[name]
        uncertainties = tool["uncertainties"]
        if tool["definition_status"] != "complete":
            uncertainties.append(f"tool_definition_{tool['definition_status']}")
        if any(call["arguments_status"] != "object" for call in tool["calls"]):
            uncertainties.append("arguments_missing_or_non_object")
        if any(len(call["matching_observations"]) != 1 for call in tool["calls"]):
            uncertainties.append("observation_pairing_incomplete")
        inventory_tools.append(tool)

    return {
        "schema": INVENTORY_SCHEMA,
        "safe_atif_sha256": safe_atif_sha256,
        "tool_count": len(inventory_tools),
        "call_count": sum(len(tool["calls"]) for tool in inventory_tools) + len(unresolved_calls),
        "unresolved_call_count": len(unresolved_calls),
        "unresolved_calls": unresolved_calls,
        "tools": inventory_tools,
    }


def _plan_tool(tool: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    warnings: list[str] = []
    calls = tool["calls"]
    if tool["definition_status"] != "complete":
        reason_codes.append(f"tool_definition_{tool['definition_status']}")
    if any(call["arguments_status"] != "object" for call in calls):
        reason_codes.append("arguments_missing_or_non_object")
    if any(len(call["matching_observations"]) != 1 for call in calls):
        reason_codes.append("observation_pairing_incomplete")
    if any(
        len(call["matching_observations"]) == 1 and not isinstance(call["matching_observations"][0].get("content"), str)
        for call in calls
    ):
        reason_codes.append("non_text_result_unsupported")
    if any(
        _contains_unusable_marker(call["arguments"])
        or any(_contains_unusable_marker(result.get("content")) for result in call["matching_observations"])
        for call in calls
    ) or _contains_unusable_marker(tool["normalized_definition"]):
        reason_codes.append("redacted_value_required")

    responses_by_arguments: dict[str, set[str]] = {}
    for call in calls:
        if call["arguments_status"] != "object" or len(call["matching_observations"]) != 1:
            continue
        content = call["matching_observations"][0].get("content")
        if not isinstance(content, str):
            continue
        key = canonical_json(call["arguments"])
        responses_by_arguments.setdefault(key, set()).add(content)
    if any(len(responses) > 1 for responses in responses_by_arguments.values()):
        reason_codes.append("conflicting_results_for_identical_arguments")

    if tool["read_only"] is False:
        warnings.append("tool_declared_mutating")
    elif tool["read_only"] is None:
        warnings.append("side_effects_unproven")

    return {
        "name": tool["name"],
        "mock_support": "exact_replay" if not reason_codes else "unsupported",
        "reason_codes": reason_codes,
        "warnings": warnings,
        "evidence_steps": sorted({call["step_id"] for call in calls}),
        "call_count": len(calls),
        "case_count": len(responses_by_arguments),
    }


def derive_tool_call_plan(inventory: dict[str, Any]) -> dict[str, Any]:
    tools = [_plan_tool(tool) for tool in inventory["tools"]]
    counts: dict[str, int] = {}
    for tool in tools:
        support = tool["mock_support"]
        counts[support] = counts.get(support, 0) + 1
    return {
        "schema": PLAN_SCHEMA,
        "inventory_sha256": json_sha256(inventory),
        "tools": tools,
        "mock_support_counts": dict(sorted(counts.items())),
    }


def resolve_tool_access(
    inventory: dict[str, Any],
    plan: dict[str, Any],
    requested: dict[str, Any],
    *,
    reviewer_kind: str,
) -> dict[str, Any]:
    if set(requested) != {"schema", "decisions"} or requested.get("schema") != DECISIONS_SCHEMA:
        raise ValueError("tool access decisions do not match the versioned contract")
    decisions = requested.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("tool access decisions must contain a decisions list")
    if reviewer_kind not in {"agent", "human"}:
        raise ValueError("tool access reviewer_kind must be agent or human")

    inventory_order = [tool["name"] for tool in inventory["tools"]]
    plan_by_name = {tool["name"]: tool for tool in plan["tools"]}
    requested_by_name: dict[str, dict[str, Any]] = {}
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict) or set(decision) != {"name", "access", "adapter", "note"}:
            raise ValueError(f"tool access decision {index} fields do not match the versioned contract")
        name = decision.get("name")
        access = decision.get("access")
        adapter = decision.get("adapter")
        note = decision.get("note")
        if not isinstance(name, str) or name not in plan_by_name or name in requested_by_name:
            raise ValueError(f"tool access decision {index} has an unknown or duplicate name")
        if access not in _ACCESS_STATES:
            raise ValueError(f"tool access decision for {name} must select real, mock, or none")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"tool access decision for {name} requires a non-empty note")
        if access == "mock":
            if adapter not in _MOCK_ADAPTERS:
                raise ValueError(f"mock access for {name} requires a supported adapter")
            if plan_by_name[name]["mock_support"] != "exact_replay":
                reasons = ", ".join(plan_by_name[name]["reason_codes"])
                raise ValueError(f"mock access for {name} is unsupported: {reasons}")
        elif adapter is not None:
            raise ValueError(f"{access} access for {name} must not select a mock adapter")
        requested_by_name[name] = {
            "name": name,
            "access": access,
            "adapter": adapter,
            "note": note.strip(),
            "warnings": plan_by_name[name]["warnings"],
            "evidence_steps": plan_by_name[name]["evidence_steps"],
        }
    missing = [name for name in inventory_order if name not in requested_by_name]
    if missing:
        raise ValueError(f"tool access decisions are missing: {', '.join(missing)}")

    counts: dict[str, int] = {}
    resolved = [requested_by_name[name] for name in inventory_order]
    for decision in resolved:
        access = decision["access"]
        counts[access] = counts.get(access, 0) + 1
    return {
        "schema": ACCESS_SCHEMA,
        "inventory_sha256": json_sha256(inventory),
        "plan_sha256": json_sha256(plan),
        "reviewer_kind": reviewer_kind,
        "decisions": resolved,
        "access_counts": dict(sorted(counts.items())),
    }


def build_call_fixtures(inventory: dict[str, Any], access: dict[str, Any]) -> dict[str, Any]:
    access_by_name = {decision["name"]: decision for decision in access["decisions"]}
    fixture_tools: list[dict[str, Any]] = []
    for tool in inventory["tools"]:
        decision = access_by_name[tool["name"]]
        if decision["access"] != "mock":
            continue
        cases_by_input: dict[str, dict[str, Any]] = {}
        for call in tool["calls"]:
            key = canonical_json(call["arguments"])
            result = call["matching_observations"][0]
            case = cases_by_input.setdefault(
                key,
                {"input": call["arguments"], "output": result["content"], "evidence_steps": []},
            )
            case["evidence_steps"].append(call["step_id"])
        fixture_tools.append(
            {
                "name": tool["name"],
                "definition": tool["normalized_definition"],
                "adapter": decision["adapter"],
                "cases": list(cases_by_input.values()),
            }
        )
    return {
        "schema": CALL_FIXTURES_SCHEMA,
        "inventory_sha256": json_sha256(inventory),
        "access_sha256": json_sha256(access),
        "matching": "tool_name_and_canonical_json_input",
        "unknown_call_policy": "error",
        "tools": fixture_tools,
    }


def build_mcp_scenario(call_fixtures: dict[str, Any]) -> dict[str, Any]:
    tools = []
    for fixture in call_fixtures["tools"]:
        if fixture["adapter"] != "mcp":
            continue
        tools.append(
            {
                "definition": fixture["definition"],
                "cases": [
                    {
                        "arguments": case["input"],
                        "result": {"content": [{"type": "text", "text": case["output"]}], "isError": False},
                        "evidence_steps": case["evidence_steps"],
                    }
                    for case in fixture["cases"]
                ],
            }
        )
    return {
        "schema": MCP_SCENARIO_SCHEMA,
        "call_fixtures_sha256": json_sha256(call_fixtures),
        "unknown_call_policy": call_fixtures["unknown_call_policy"],
        "tools": tools,
    }
