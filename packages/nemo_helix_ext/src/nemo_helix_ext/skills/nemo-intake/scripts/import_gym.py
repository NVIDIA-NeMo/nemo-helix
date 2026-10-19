# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import Gym rollout JSONL containing ng_trajectory 1.0 into NeMo Intake."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from _import_common import (
    ImportBundle,
    add_common_arguments,
    normalize_datetime,
    normalize_status,
    parse_bound,
    project_signal,
    run_import,
    validate_common_arguments,
)
from pydantic import JsonValue

JsonObject = dict[str, JsonValue]


def _object(value: JsonValue, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _records(value: JsonValue, label: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return [_object(item, label) for item in value]


def _identity(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _id(*parts: str) -> str:
    # Length-delimited JSON avoids collisions from delimiters inside native IDs.
    return "gym-" + hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def _time(value: JsonValue) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError("Gym timestamps must be epoch seconds or timezone-aware ISO-8601 strings")
    if isinstance(value, str):
        parse_bound(value)
    return normalize_datetime(value)


def _span(
    *,
    span_id: str,
    trace_id: str,
    parent_id: str | None,
    name: str,
    kind: str,
    record: JsonObject,
    anchor: str,
    attributes: JsonObject,
) -> JsonObject:
    started = record.get("started_at")
    ended = record.get("completed_at")
    attrs = dict(attributes)
    attrs["gym.raw"] = record
    attrs["gym.timing"] = "observed" if started is not None else "anchor_only"
    if record.get("duration_ms") is not None:
        attrs["gym.observed_duration_ms"] = record["duration_ms"]
    start_time = _time(started) if started is not None else anchor
    # Missing start time must not turn a known completion into a fabricated duration.
    end_time = _time(ended) if started is not None and ended is not None else None
    if end_time is not None and datetime.fromisoformat(end_time) < datetime.fromisoformat(start_time):
        raise ValueError(f"completed_at precedes started_at for {name}")
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "session_id": trace_id,
        "parent_span_id": parent_id,
        "name": name,
        "kind": kind,
        "status": normalize_status(record.get("status"), error=record.get("error_type")),
        "started_at": start_time,
        "ended_at": end_time,
        "attributes": attrs,
    }


def map_gym_rollouts(
    rows: list[JsonObject],
    *,
    run_id: str,
    agent_name: str,
    started_at: datetime | None = None,
    include_feedback: bool = True,
) -> ImportBundle:
    """Map versioned Gym evidence without inferring ownership or execution order.

    ``started_at`` is an explicit fallback anchor for trajectories with no absolute
    timestamps. Untimed spans retain null end times and carry ``gym.timing=anchor_only``.
    """
    _identity(run_id, "run_id")
    _identity(agent_name, "agent_name")
    if started_at is not None and started_at.utcoffset() is None:
        raise ValueError("started_at must include a UTC offset")
    bundle = ImportBundle(source="gym")
    for row in rows:
        trajectory = _object(row.get("ng_trajectory"), "ng_trajectory")
        if trajectory.get("schema_version") != "1.0":
            raise ValueError("Only ng_trajectory schema_version 1.0 is supported")
        task_id = _identity(trajectory.get("task_id"), "task_id")
        rollout_id = _identity(trajectory.get("rollout_id"), "rollout_id")
        trace_id = _id(run_id, task_id, rollout_id)
        root_id = _id(trace_id, "rollout")
        invocations = _records(trajectory.get("invocations", []), "invocations")
        calls = _records(trajectory.get("model_calls", []), "model_calls")
        tools = _records(trajectory.get("tool_calls", []), "tool_calls")
        turns = _records(trajectory.get("turns", []), "turns")
        observed = [
            _time(item["started_at"])
            for item in [row, *invocations, *calls, *tools]
            if item.get("started_at") is not None
        ]
        observed.extend(_time(item["timestamp"]) for item in turns if item.get("timestamp") is not None)
        if observed:
            anchor = min(observed, key=datetime.fromisoformat)
        elif started_at is not None:
            anchor = started_at.isoformat()
        else:
            raise ValueError(
                f"Rollout {rollout_id} has no absolute timestamps; supply --started-at as an explicit anchor"
            )
        attributes: JsonObject = {
            "gen_ai.agent.name": agent_name,
            "gym.run_id": run_id,
            "gym.task_id": task_id,
            "gym.rollout_id": rollout_id,
            "gym.schema_version": "1.0",
        }
        root = _span(
            span_id=root_id,
            trace_id=trace_id,
            parent_id=None,
            name="gym.rollout",
            kind="CHAIN",
            record=row,
            anchor=anchor,
            attributes=attributes,
        )
        rollout_start = len(bundle.spans)
        # Preserve the complete evidence envelope, including gaps, turns and supplemental observations.
        bundle.spans.append(root)
        invocation_map = {_identity(item.get("invocation_id"), "invocation_id"): item for item in invocations}
        if len(invocation_map) != len(invocations):
            raise ValueError("Duplicate invocation_id")
        for invocation_id, invocation in invocation_map.items():
            visited = {invocation_id}
            parent = invocation.get("parent_invocation_id")
            while isinstance(parent, str) and parent in invocation_map:
                if parent in visited:
                    raise ValueError("Invocation parents form a cycle")
                visited.add(parent)
                parent = invocation_map[parent].get("parent_invocation_id")
            parent = invocation.get("parent_invocation_id")
            parent_id = (
                _id(trace_id, "invocation", parent) if isinstance(parent, str) and parent in invocation_map else root_id
            )
            span = _span(
                span_id=_id(trace_id, "invocation", invocation_id),
                trace_id=trace_id,
                parent_id=parent_id,
                name=invocation_id,
                kind="AGENT",
                record=invocation,
                anchor=anchor,
                attributes=attributes,
            )
            span["output"] = invocation.get("conversation")
            bundle.spans.append(span)
        owners_by_call: dict[int, set[str]] = {}
        for invocation_id, invocation in invocation_map.items():
            for ref in _records(invocation.get("model_calls", []), "model_calls references"):
                matches = [
                    index
                    for index, candidate in enumerate(calls)
                    if _matches(ref, candidate, _object(candidate.get("response_metadata", {}), "response_metadata"))
                ]
                if len(matches) == 1:
                    owners_by_call.setdefault(matches[0], set()).add(invocation_id)
        for index, call in enumerate(calls):
            metadata = _object(call.get("response_metadata", {}), "response_metadata")
            owners = owners_by_call.get(index, set())
            owner = next(iter(owners)) if len(owners) == 1 else None
            call_attributes = dict(attributes)
            call_attributes["gym.ownership"] = "explicit" if owner is not None else "unavailable_or_ambiguous"
            if metadata.get("model") is not None:
                call_attributes["gen_ai.request.model"] = metadata["model"]
            for native, semantic in (
                ("prompt_tokens", "input_tokens"),
                ("completion_tokens", "output_tokens"),
                ("total_tokens", "total_tokens"),
            ):
                value = _object(call.get("token_stats", {}), "token_stats").get(native)
                if value is not None:
                    call_attributes[f"gen_ai.usage.{semantic}"] = value
            call_id = call.get("model_call_id")
            # Anonymous calls have no stable source identity beyond position in this artifact.
            identity_parts = (
                ("id", _identity(call_id, "model_call_id")) if call_id is not None else ("index", str(index))
            )
            span = _span(
                span_id=_id(trace_id, "model", *identity_parts),
                trace_id=trace_id,
                parent_id=_id(trace_id, "invocation", owner) if owner is not None else root_id,
                name="gym.model_call",
                kind="LLM",
                record=call,
                anchor=anchor,
                attributes=call_attributes,
            )
            span["status"] = normalize_status(metadata.get("response_status"), error=metadata.get("error_category"))
            status_code = metadata.get("status_code")
            if isinstance(status_code, int) and status_code >= 400:
                span["status"] = "error"
            span["input"] = call.get("request")
            span["output"] = call.get("response")
            bundle.spans.append(span)
        for tool in tools:
            invocation_id = _identity(tool.get("invocation_id"), "tool invocation_id")
            tool_id = _identity(tool.get("tool_call_id"), "tool_call_id")
            tool_attributes = dict(attributes)
            if tool.get("tool_name") is not None:
                tool_attributes["tool.name"] = tool["tool_name"]
            span = _span(
                span_id=_id(trace_id, "tool", invocation_id, tool_id),
                trace_id=trace_id,
                parent_id=_id(trace_id, "invocation", invocation_id) if invocation_id in invocation_map else root_id,
                name=str(tool.get("tool_name") or "gym.tool_call"),
                kind="TOOL",
                record=tool,
                anchor=anchor,
                attributes=tool_attributes,
            )
            if tool.get("status") == "timeout":
                span["status"] = "error"
            span["output"] = tool.get("output")
            invocation = invocation_map.get(invocation_id)
            if invocation is not None:
                conversation = _records(invocation.get("conversation", []), "conversation")
                arguments = [
                    item.get("arguments")
                    for item in conversation
                    if item.get("type") == "function_call" and item.get("call_id") == tool_id
                ]
                if len(arguments) == 1:
                    span["input"] = arguments[0]
            bundle.spans.append(span)
        _roll_up_rollout(root, row, invocations, bundle.spans[rollout_start + 1 :])
        if include_feedback and row.get("reward") is not None:
            results, annotations = project_signal(
                provider="gym",
                span_id=root_id,
                session_id=trace_id,
                name="reward",
                value=row["reward"],
                comment=None,
                automated=True,
            )
            bundle.evaluator_results.extend(results)
            bundle.annotations.extend(annotations)
    bundle.validate()
    return bundle


def _roll_up_rollout(root: JsonObject, row: JsonObject, invocations: list[JsonObject], spans: list[JsonObject]) -> None:
    """Summarize execution outcomes and observed bounds without using reward as status."""
    attributes = _object(root["attributes"], "attributes")
    if row.get("status") is not None or row.get("error_type"):
        attributes["gym.status_source"] = "rollout"
    else:
        top_level = [item for item in invocations if item.get("parent_invocation_id") is None]
        statuses = [normalize_status(item.get("status"), error=item.get("error_type")) for item in top_level]
        identities = {item["invocation_id"] for item in invocations}
        missing_parent = any(
            item.get("parent_invocation_id") is not None and item["parent_invocation_id"] not in identities
            for item in invocations
        )
        if "error" in statuses:
            root["status"] = "error"
        elif "cancelled" in statuses:
            root["status"] = "cancelled"
        elif statuses and all(status == "success" for status in statuses) and not missing_parent:
            root["status"] = "success"
        attributes["gym.status_source"] = "top_level_invocations"

    perf = _object(row.get("ng_perf") or {}, "ng_perf")
    duration = perf.get("total_latency_ms")
    if duration is not None:
        if (
            isinstance(duration, bool)
            or not isinstance(duration, int | float)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("ng_perf.total_latency_ms must be a finite nonnegative number")
        attributes["gym.observed_duration_ms"] = duration
        attributes["gym.duration_source"] = "ng_perf.total_latency_ms"

    # A recorded elapsed duration does not establish an absolute start or end.
    # Only project a child window when the rollout has neither boundary itself.
    if row.get("started_at") is None and row.get("completed_at") is None:
        timed = [span for span in spans if _object(span["attributes"], "attributes")["gym.timing"] == "observed"]
        if timed and all(span.get("ended_at") is not None for span in timed):
            # Retain the source anchor, which also includes observed turn timestamps.
            # start_time is part of Intake's storage key, so replay must not shift it.
            root["ended_at"] = max(
                (span["ended_at"] for span in timed), key=lambda value: datetime.fromisoformat(str(value))
            )
            attributes["gym.timing"] = "observed_child_window"


def _matches(ref: JsonObject, call: JsonObject, metadata: JsonObject) -> bool:
    if ref.get("model_call_id"):
        return ref["model_call_id"] == call.get("model_call_id") and all(
            ref.get(key) is None or ref[key] == metadata.get(key) for key in ("model_ref", "response_id")
        )
    return bool(ref.get("model_ref") and ref.get("response_id")) and all(
        ref[key] == metadata.get(key) for key in ("model_ref", "response_id")
    )


def load_rollouts(path: Path) -> list[JsonObject]:
    """Read rollout JSONL, identifying malformed records before any writes."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(_object(json.loads(line), "rollout"))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--run-id", required=True, help="Stable, globally unique evaluation run identifier.")
    parser.add_argument("--agent-name", required=True, help="Stable agent name for Intake filters.")
    parser.add_argument(
        "--started-at", type=parse_bound, help="Explicit timestamp anchor when evidence has no absolute time."
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.input is None:
        parser.error("--input is required (Gym rollout JSONL)")
    validate_common_arguments(parser, args)
    bundle = map_gym_rollouts(
        load_rollouts(args.input),
        run_id=args.run_id,
        agent_name=args.agent_name,
        started_at=args.started_at,
        include_feedback=args.include_feedback,
    )
    return run_import(bundle, args)


if __name__ == "__main__":
    raise SystemExit(main())
