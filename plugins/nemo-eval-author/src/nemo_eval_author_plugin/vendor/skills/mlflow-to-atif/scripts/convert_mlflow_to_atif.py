#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert MLflow traces into canonical ATIF trajectory files."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from enum import Enum
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

SCHEMA_VERSION = "ATIF-v1.7"
CANONICALIZER = "mlflow-to-atif@2"
EVENT_TIME_MICROSECONDS_NORMALIZED = "event_time_microseconds_normalized_to_nanoseconds"
LLM_TYPES = frozenset({"CHAT_MODEL", "LLM", "MODEL", "PROMPT"})
TOOL_TYPES = frozenset({"EMBEDDING", "FUNCTION", "GUARDRAIL", "RERANKER", "RETRIEVER", "SEARCH", "TOOL"})
ORCHESTRATION_TYPES = frozenset({"AGENT", "CHAIN", "TASK", "WORKFLOW"})


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    if callable(model_dump := getattr(value, "model_dump", None)):
        return _jsonable(model_dump(mode="json"))
    if callable(to_dict := getattr(value, "to_dict", None)):
        return _jsonable(to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def _object(value: Any, label: str) -> dict[str, Any]:
    converted = _jsonable(value)
    if not isinstance(converted, dict):
        raise ValueError(f"{label} must be an object")
    return converted


def _parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _mlflow_id(value: Any, *, byte_length: int) -> str:
    text = str(value or "")
    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return text
    return decoded.hex() if len(decoded) == byte_length else text


def _span_id(value: Any) -> str:
    return _mlflow_id(value, byte_length=8)


def _trace_id(value: Any) -> str:
    return _mlflow_id(value, byte_length=16)


def _trace_identity(value: Any) -> str:
    """Canonicalize IDs for comparison while preserving MLflow's external ``tr-`` ID for output."""
    trace_id = _trace_id(value)
    if match := re.fullmatch(r"tr-([0-9a-fA-F]{32})", trace_id):
        return match.group(1).lower()
    return trace_id.lower() if re.fullmatch(r"[0-9a-fA-F]{32}", trace_id) else trace_id


def _iso_from_nanos(value: Any) -> str:
    nanoseconds = int(value)
    seconds, remainder = divmod(nanoseconds, 1_000_000_000)
    timestamp = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=remainder // 1000)
    return timestamp.isoformat()


def _normalize_event_times(span: dict[str, Any]) -> int:
    """Correct microsecond event times when nanosecond parent bounds prove the unit mismatch."""
    events = span.get("events")
    start = span.get("start_time_unix_nano")
    end = span.get("end_time_unix_nano")
    if (
        not isinstance(events, list)
        or not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        return 0
    if start > end:
        return 0

    normalized = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        value = event.get("time_unix_nano")
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        event_time_nanos = value * 1_000
        if not start <= value <= end and start <= event_time_nanos <= end:
            event["time_unix_nano"] = event_time_nanos
            normalized += 1
    return normalized


def _parse_bound(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO-8601 timestamp: {value}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("time bounds must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _attributes(span: dict[str, Any]) -> dict[str, Any]:
    serialized = _object(span.get("attributes") or {}, "MLflow span attributes")
    return {key: _parse_json(value) for key, value in serialized.items()}


def _span_type(attributes: dict[str, Any]) -> str:
    value = attributes.get("mlflow.spanType") or attributes.get("openinference.span.kind") or "UNKNOWN"
    return str(value).upper().removeprefix("SPAN_KIND_")


def _input(attributes: dict[str, Any]) -> Any:
    for key in ("mlflow.spanInputs", "input.value", "gen_ai.input.messages"):
        if key in attributes:
            return _parse_json(attributes[key])
    return None


def _output(attributes: dict[str, Any]) -> Any:
    _, value = _output_entry(attributes)
    return value


def _output_entry(attributes: dict[str, Any]) -> tuple[str | None, Any]:
    for key in ("mlflow.spanOutputs", "output.value", "gen_ai.output.messages"):
        if key in attributes:
            return key, _parse_json(attributes[key])
    return None, None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "message", "text", "response", "answer", "output"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(choice.get("text"), str):
                    return choice["text"]
    return _text(value)


def _message_records(value: Any) -> list[dict[str, Any]] | None:
    """Return a non-empty message list, leaving shape validation to the caller."""
    candidate = value.get("messages") if isinstance(value, dict) else value
    if not isinstance(candidate, list) or not candidate:
        return None
    if not all(isinstance(item, dict) and ("role" in item or "source" in item) for item in candidate):
        return None
    return candidate


def _is_chat_shaped(value: Any) -> bool:
    """Recognize explicit or partially recognizable chat input that must not degrade to opaque JSON."""
    if isinstance(value, dict):
        return "messages" in value
    return isinstance(value, list) and any(
        isinstance(item, dict) and ("role" in item or "source" in item) for item in value
    )


def _input_steps(value: Any, *, timestamp: str) -> list[dict[str, Any]]:
    """Project valid chat history or one genuinely non-chat input into ATIF steps."""
    messages = _message_records(value)
    if messages is None:
        if _is_chat_shaped(value):
            raise ValueError("MLflow trace input contains an empty or malformed message list")
        return [
            {
                "step_id": 0,
                "timestamp": timestamp,
                "source": "user",
                "message": _text(value),
                "extra": {"mlflow_to_atif": {"canonical_role": "human_instruction"}},
            }
        ]

    last_user = max(
        (index for index, item in enumerate(messages) if str(item.get("role") or item.get("source")).lower() == "user"),
        default=-1,
    )
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(messages):
        role = str(item.get("role") or item.get("source") or "").lower()
        source = {"assistant": "agent", "developer": "system", "human": "user"}.get(role, role)
        if source not in {"agent", "system", "user"}:
            source = "system"
        content = item.get("content", item.get("message", item.get("text", "")))
        step: dict[str, Any] = {
            "step_id": 0,
            "timestamp": timestamp,
            "source": source,
            "message": _message_text(content),
            "is_copied_context": index != last_user,
        }
        if source == "user":
            canonical_role = "human_instruction" if index == last_user else "conversation_context"
            step["extra"] = {"mlflow_to_atif": {"canonical_role": canonical_role}}
        steps.append(step)
    if last_user < 0:
        raise ValueError("MLflow trace input messages do not contain a user message")
    return steps


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metrics(attributes: dict[str, Any]) -> dict[str, Any] | None:
    usage: dict[str, Any] = {}
    for key in ("mlflow.chat.tokenUsage", "mlflow.chat.token_usage", "mlflow.llm.tokenUsage"):
        candidate = _parse_json(attributes.get(key))
        if isinstance(candidate, dict):
            usage.update(candidate)
    usage.update({key: value for key, value in attributes.items() if key.startswith("gen_ai.usage.")})

    def first(*keys: str) -> Any:
        return next((usage[key] for key in keys if key in usage), None)

    mapped: dict[str, Any] = {}
    values = {
        "prompt_tokens": _positive_int(first("input_tokens", "prompt_tokens", "gen_ai.usage.input_tokens")),
        "completion_tokens": _positive_int(first("output_tokens", "completion_tokens", "gen_ai.usage.output_tokens")),
        "cached_tokens": _positive_int(first("cached_tokens", "cache_read_input_tokens", "gen_ai.usage.cached_tokens")),
        "cost_usd": _number(
            attributes.get("mlflow.cost.total")
            or attributes.get("llm.cost.total")
            or attributes.get("gen_ai.usage.cost")
        ),
    }
    mapped.update({key: value for key, value in values.items() if value is not None})
    return mapped or None


def _model_name(attributes: dict[str, Any]) -> str | None:
    for key in ("mlflow.llm.model", "gen_ai.request.model", "llm.model_name"):
        value = attributes.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _span_extra(span: dict[str, Any], attributes: dict[str, Any], span_type: str) -> dict[str, Any]:
    return {
        "mlflow": {
            "span_id": _span_id(span.get("span_id")),
            "trace_id": _trace_id(span.get("trace_id")),
            "parent_span_id": _span_id(span.get("parent_span_id")) or None,
            "name": str(span.get("name") or ""),
            "span_type": span_type,
            "status": _jsonable(span.get("status")),
            "attributes": attributes,
            "events": _jsonable(span.get("events") or []),
            "links": _jsonable(span.get("links") or []),
        }
    }


def _tool_step(span: dict[str, Any], attributes: dict[str, Any], span_type: str) -> tuple[dict[str, Any], str]:
    span_id = _span_id(span.get("span_id"))
    raw_input = _input(attributes)
    arguments = (
        raw_input if isinstance(raw_input, dict) else ({"value": _jsonable(raw_input)} if raw_input is not None else {})
    )
    function_name = attributes.get("mlflow.spanFunctionName") or span.get("name") or span_type.lower()
    call = {
        "tool_call_id": span_id,
        "function_name": str(function_name),
        "arguments": arguments,
        "extra": {"mlflow": {"span_type": span_type}},
    }
    output_attribute, raw_output = _output_entry(attributes)
    output_state = "missing" if output_attribute is None else ("null" if raw_output is None else "value")
    result = {
        "source_call_id": span_id,
        "content": _text(raw_output),
        "extra": {
            "mlflow": {
                "status": _jsonable(span.get("status")),
                "output_state": output_state,
                "output_attribute": output_attribute,
            }
        },
    }
    return (
        {
            "step_id": 0,
            "timestamp": _iso_from_nanos(span["start_time_unix_nano"]),
            "source": "agent",
            "message": "",
            "llm_call_count": 0,
            "tool_calls": [call],
            "observation": {"results": [result]},
            "extra": _span_extra(span, attributes, span_type),
        },
        output_state,
    )


def _agent_step(span: dict[str, Any], attributes: dict[str, Any], span_type: str) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": 0,
        "timestamp": _iso_from_nanos(span["start_time_unix_nano"]),
        "source": "agent",
        "message": _message_text(_output(attributes)),
        "extra": _span_extra(span, attributes, span_type),
    }
    if span_type in LLM_TYPES:
        step["llm_call_count"] = 1
        if model_name := _model_name(attributes):
            step["model_name"] = model_name
        if metrics := _metrics(attributes):
            step["metrics"] = metrics
    return step


def _trace_input(info: dict[str, Any], root_attributes: dict[str, Any]) -> Any:
    for key in ("mlflow.spanInputs", "input.value", "gen_ai.input.messages"):
        if key in root_attributes:
            value = _parse_json(root_attributes[key])
            if _has_trace_input(value):
                return value
    metadata = _object(info.get("trace_metadata") or {}, "MLflow trace metadata")
    for key in ("mlflow.traceInputs", "inputs"):
        if key in metadata:
            value = _parse_json(metadata[key])
            if _has_trace_input(value):
                return value
    return _parse_json(info.get("request_preview"))


def _has_trace_input(value: Any) -> bool:
    """Whether an input carries an instruction rather than an empty export placeholder."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict | list | tuple):
        return bool(value)
    return True


def _trace_output(info: dict[str, Any], root_attributes: dict[str, Any]) -> Any:
    value = _output(root_attributes)
    if value is not None:
        return value
    metadata = _object(info.get("trace_metadata") or {}, "MLflow trace metadata")
    for key in ("mlflow.traceOutputs", "outputs"):
        if key in metadata:
            return _parse_json(metadata[key])
    return _parse_json(info.get("response_preview"))


def _require_span_fields(span: dict[str, Any], *, trace_index: int, span_index: int) -> None:
    label = f"trace[{trace_index}].span[{span_index}]"
    if not _span_id(span.get("span_id")):
        raise ValueError(f"{label} requires span_id")
    if span.get("start_time_unix_nano") is None:
        raise ValueError(f"{label} requires start_time_unix_nano")


def _validate_span_graph(spans: list[dict[str, Any]], *, trace_index: int) -> tuple[list[dict[str, Any]], set[str]]:
    """Validate parent references and cycles, returning roots and spans that have children."""
    span_ids = {_span_id(span.get("span_id")) for span in spans}
    parent_by_id = {_span_id(span.get("span_id")): _span_id(span.get("parent_span_id")) for span in spans}
    if any(parent_id and parent_id not in span_ids for parent_id in parent_by_id.values()):
        raise ValueError(f"trace[{trace_index}] contains a span with an unresolved parent span ID")

    complete: set[str] = set()
    for start_id in span_ids:
        path: set[str] = set()
        current_id = start_id
        while current_id and current_id not in complete:
            if current_id in path:
                raise ValueError(f"trace[{trace_index}] span parent graph contains a cycle")
            path.add(current_id)
            current_id = parent_by_id[current_id]
        complete.update(path)

    roots = [span for span in spans if not _span_id(span.get("parent_span_id"))]
    if not roots:
        raise ValueError(f"trace[{trace_index}] span parent graph does not contain a root span")
    child_ids = {parent_id for parent_id in parent_by_id.values() if parent_id}
    return roots, child_ids


def _final_metrics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_cached_tokens": 0,
        "total_cost_usd": 0.0,
    }
    present: set[str] = set()
    mapping = {
        "prompt_tokens": "total_prompt_tokens",
        "completion_tokens": "total_completion_tokens",
        "cached_tokens": "total_cached_tokens",
        "cost_usd": "total_cost_usd",
    }
    for step in steps:
        for source, target in mapping.items():
            value = (step.get("metrics") or {}).get(source)
            if value is not None:
                totals[target] += value
                present.add(target)
    return {"total_steps": len(steps), **{key: totals[key] for key in totals if key in present}}


def convert_trace(
    trace_value: Any,
    *,
    trace_index: int,
    agent_name: str,
    agent_version: str,
) -> dict[str, Any]:
    """Convert one internally consistent MLflow trace into a canonical ATIF trajectory."""
    trace = _object(trace_value, f"trace[{trace_index}]")
    info = _object(trace.get("info") or {}, f"trace[{trace_index}].info")
    data = _object(trace.get("data") or {}, f"trace[{trace_index}].data")
    span_values = data.get("spans") or []
    if not isinstance(span_values, list):
        raise ValueError(f"trace[{trace_index}].data.spans must be an array")
    spans = [_object(value, f"trace[{trace_index}].span") for value in span_values]
    if not spans:
        raise ValueError(f"trace[{trace_index}] does not contain spans")
    for span_index, span in enumerate(spans):
        _require_span_fields(span, trace_index=trace_index, span_index=span_index)
    normalized_event_times = sum(_normalize_event_times(span) for span in spans)
    canonical_span_ids = [_span_id(span.get("span_id")) for span in spans]
    if len(canonical_span_ids) != len(set(canonical_span_ids)):
        raise ValueError(f"trace[{trace_index}] contains duplicate span IDs")
    native_trace_ids = {
        _trace_identity(span.get("trace_id")) for span in spans if _trace_identity(span.get("trace_id"))
    }
    if len(native_trace_ids) > 1:
        raise ValueError(f"trace[{trace_index}] contains spans from different trace IDs")
    info_trace_id = _trace_id(info.get("trace_id"))
    info_trace_identity = _trace_identity(info_trace_id)
    if info_trace_identity and native_trace_ids and native_trace_ids != {info_trace_identity}:
        raise ValueError(f"trace[{trace_index}] info trace ID does not match its spans")
    spans.sort(key=lambda item: (int(item["start_time_unix_nano"]), _span_id(item.get("span_id"))))

    roots, child_ids = _validate_span_graph(spans, trace_index=trace_index)
    root = roots[0]
    root_attributes = _attributes(root)
    trace_input = _trace_input(info, root_attributes)
    if not _has_trace_input(trace_input):
        raise ValueError(f"trace[{trace_index}] has no recoverable root input for an ATIF human instruction")

    external_trace_id = info_trace_id or _trace_id(root.get("trace_id"))
    if not external_trace_id:
        raise ValueError(f"trace[{trace_index}] requires trace_id")
    metadata = _object(info.get("trace_metadata") or {}, "MLflow trace metadata")
    session_id = str(
        metadata.get("mlflow.trace.session")
        or metadata.get("mlflow.trace.session_id")
        or metadata.get("session_id")
        or external_trace_id
    )
    root_started_at = _iso_from_nanos(root["start_time_unix_nano"])
    steps = _input_steps(trace_input, timestamp=root_started_at)
    loss_codes: set[str] = set()
    if len(roots) > 1:
        loss_codes.add("multiple_root_spans_linearized")
    if child_ids:
        loss_codes.add("mlflow_span_tree_linearized")

    for span in spans:
        attributes = _attributes(span)
        span_type = _span_type(attributes)
        is_orchestration_parent = _span_id(span.get("span_id")) in child_ids and span_type in ORCHESTRATION_TYPES
        if is_orchestration_parent:
            loss_codes.add("orchestration_parent_not_emitted_as_step")
            continue
        if span_type in TOOL_TYPES:
            tool_step, output_state = _tool_step(span, attributes, span_type)
            steps.append(tool_step)
            if output_state == "missing":
                loss_codes.add("missing_tool_output_rendered_as_empty_string")
            elif output_state == "null":
                loss_codes.add("null_tool_output_rendered_as_empty_string")
        elif span_type in LLM_TYPES or span_type in ORCHESTRATION_TYPES:
            steps.append(_agent_step(span, attributes, span_type))
        else:
            loss_codes.add("unknown_span_type_not_emitted_as_step")

    final_output = _trace_output(info, root_attributes)
    final_text = _message_text(final_output)
    last_agent_text = next((str(step["message"]) for step in reversed(steps) if step["source"] == "agent"), None)
    if final_output is not None and final_text != last_agent_text:
        ended_at = root.get("end_time_unix_nano") or root["start_time_unix_nano"]
        steps.append(
            {
                "step_id": 0,
                "timestamp": _iso_from_nanos(ended_at),
                "source": "agent",
                "message": final_text,
                "extra": {"mlflow": {"role": "trace_output", "root_span_id": _span_id(root.get("span_id"))}},
            }
        )
    for step_id, step in enumerate(steps, start=1):
        step["step_id"] = step_id

    models = {str(step["model_name"]) for step in steps if step.get("model_name")}
    agent: dict[str, Any] = {"name": agent_name, "version": agent_version}
    if len(models) == 1:
        agent["model_name"] = models.pop()
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "trajectory_id": external_trace_id,
        "agent": agent,
        "steps": steps,
        "notes": "Converted from MLflow spans; nested span structure is retained in namespaced extra metadata.",
        "final_metrics": _final_metrics(steps),
        "extra": {
            "mlflow": {
                "trace_id": external_trace_id,
                "info": info,
                "spans": [_jsonable(span) for span in spans],
            },
            "mlflow_to_atif": {
                "canonicalizer": CANONICALIZER,
                "loss_codes": sorted(loss_codes),
                "normalization_codes": ([EVENT_TIME_MICROSECONDS_NORMALIZED] if normalized_event_times else []),
            },
        },
    }


def convert_export(
    payload: Any,
    *,
    agent_name: str,
    agent_version: str,
) -> list[dict[str, Any]]:
    """Convert one trace or a complete trace collection while rejecting ambiguous exports."""
    if isinstance(payload, dict) and "traces" in payload:
        if payload.get("next_page_token") not in (None, ""):
            raise ValueError("MLflow export is incomplete: fetch all pages before conversion")
        traces = payload["traces"]
    elif isinstance(payload, dict) and "info" in payload and "data" in payload:
        traces = [payload]
    else:
        traces = payload
    if not isinstance(traces, list):
        raise ValueError("MLflow export must be one trace, a trace array, or an object with `traces`")
    if not traces:
        raise ValueError("MLflow export does not contain traces")
    trajectories = [
        convert_trace(
            value,
            trace_index=index,
            agent_name=agent_name,
            agent_version=agent_version,
        )
        for index, value in enumerate(traces)
    ]
    trajectory_ids = [str(item["trajectory_id"]) for item in trajectories]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("MLflow export contains duplicate trace IDs")
    return trajectories


def _load_input(path: Path) -> Any:
    if str(path) == "-":
        return json.load(sys.stdin)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _validate_mlflow_transport(tracking_uri: str | None) -> None:
    """Reject remote cleartext or TLS-bypass configuration before MLflow can send credentials."""
    if os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").strip().lower() == "true":
        raise ValueError("MLFLOW_TRACKING_INSECURE_TLS=true is not allowed for live conversion")
    if not tracking_uri:
        return
    try:
        parsed = urlparse(tracking_uri)
        hostname = parsed.hostname
    except ValueError as error:
        raise ValueError("MLflow tracking URI is invalid") from error
    if parsed.scheme.lower() != "http":
        if parsed.scheme.lower() == "https" and not hostname:
            raise ValueError("HTTPS MLflow tracking URI requires a hostname")
        return
    if not hostname:
        raise ValueError("HTTP MLflow tracking URI requires a loopback hostname")
    try:
        is_loopback = ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = hostname.lower() == "localhost"
    if not is_loopback:
        raise ValueError("remote MLflow tracking URI must use HTTPS")


def _fetch_mlflow(args: argparse.Namespace) -> dict[str, Any]:
    """Fetch every page in the bounded trace query without allowing credential-bearing redirects."""
    _validate_mlflow_transport(args.tracking_uri)
    previous_redirect_setting = os.environ.get("MLFLOW_ALLOW_HTTP_REDIRECTS")
    os.environ["MLFLOW_ALLOW_HTTP_REDIRECTS"] = "false"
    try:
        if importlib.util.find_spec("mlflow") is None:
            raise RuntimeError("live conversion requires MLflow in the selected Python environment")
        mlflow = importlib.import_module("mlflow")
        lower_ms = int(args.since.timestamp() * 1000)
        upper_ms = int(args.until.timestamp() * 1000)
        client = mlflow.MlflowClient(tracking_uri=args.tracking_uri)
        traces: list[Any] = []
        page_token: str | None = None
        while True:
            page = client.search_traces(
                experiment_ids=[args.experiment_id],
                filter_string=f"timestamp_ms >= {lower_ms} AND timestamp_ms < {upper_ms}",
                page_token=page_token,
            )
            traces.extend(page)
            page_token = getattr(page, "token", None)
            if not page_token:
                break
    finally:
        if previous_redirect_setting is None:
            os.environ.pop("MLFLOW_ALLOW_HTTP_REDIRECTS", None)
        else:
            os.environ["MLFLOW_ALLOW_HTTP_REDIRECTS"] = previous_redirect_setting
    return {"traces": [_jsonable(trace) for trace in traces]}


def _safe_filename(trajectory_id: str) -> str:
    """Create a stable restricted filename without trusting an external trace ID as a path."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", trajectory_id).strip("._-")[:80] or "trace"
    digest = hashlib.sha256(trajectory_id.encode()).hexdigest()[:12]
    return f"{slug}-{digest}.atif.json"


def _validate_structure(trajectory: dict[str, Any]) -> None:
    if trajectory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("converter emitted an unsupported ATIF version")
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("converter emitted an ATIF trajectory without steps")
    if [step.get("step_id") for step in steps] != list(range(1, len(steps) + 1)):
        raise ValueError("converter emitted non-sequential ATIF step IDs")
    for step in steps:
        call_ids = {call["tool_call_id"] for call in step.get("tool_calls") or []}
        for result in (step.get("observation") or {}).get("results") or []:
            source_call_id = result.get("source_call_id")
            if source_call_id is not None and source_call_id not in call_ids:
                raise ValueError(f"observation references unknown tool call {source_call_id!r}")


def _validate_with_harbor(trajectories: list[dict[str, Any]]) -> None:
    if importlib.util.find_spec("harbor") is None:
        raise RuntimeError("--validate-with-harbor requires Harbor in the selected Python environment")
    module = importlib.import_module("harbor.models.trajectories")
    trajectory_model = module.Trajectory
    for trajectory in trajectories:
        trajectory_model.model_validate(trajectory)


def _write_trajectories(trajectories: list[dict[str, Any]], *, output_dir: Path, overwrite: bool) -> list[Path]:
    """Stage private files and publish them atomically, with no-clobber semantics by default."""
    output_existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output_existed:
        if not output_dir.is_dir():
            raise NotADirectoryError(f"ATIF output is not a directory: {output_dir}")
        if os.name == "posix" and output_dir.stat().st_mode & 0o077:
            raise PermissionError(f"ATIF output directory must not be accessible by group or other users: {output_dir}")
    else:
        output_dir.chmod(0o700)
    targets = [output_dir / _safe_filename(str(item["trajectory_id"])) for item in trajectories]
    if len(targets) != len(set(targets)):
        raise ValueError("MLflow trace IDs resolve to duplicate ATIF output paths")
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing ATIF output: {existing[0]}")
    staged: list[tuple[Path, Path]] = []
    try:
        for trajectory, target in zip(trajectories, targets, strict=True):
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=output_dir)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(trajectory, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                temporary.chmod(0o600)
                staged.append((temporary, target))
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        for temporary, target in staged:
            if overwrite:
                temporary.replace(target)
            else:
                os.link(temporary, target)
                temporary.unlink()
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="MLflow Trace.to_dict() JSON, or '-' for standard input")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment-id", help="MLflow experiment ID for live conversion")
    parser.add_argument("--since", type=_parse_bound, help="Inclusive live-query lower bound")
    parser.add_argument("--until", type=_parse_bound, help="Exclusive live-query upper bound")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agent-name", required=True, help="Stable agent name recorded in ATIF")
    parser.add_argument("--agent-version", default="unknown")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-with-harbor", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.input is None:
        if not args.experiment_id or args.since is None or args.until is None:
            parser.error("live conversion requires --experiment-id, --since, and --until")
        if args.since >= args.until:
            parser.error("--since must be earlier than --until")
    elif any(value is not None for value in (args.experiment_id, args.since, args.until)):
        parser.error("--input cannot be combined with --experiment-id, --since, or --until")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    try:
        payload = _load_input(args.input) if args.input is not None else _fetch_mlflow(args)
        trajectories = convert_export(
            payload,
            agent_name=args.agent_name,
            agent_version=args.agent_version,
        )
        for trajectory in trajectories:
            _validate_structure(trajectory)
        if args.validate_with_harbor:
            _validate_with_harbor(trajectories)
        paths = _write_trajectories(trajectories, output_dir=args.output_dir, overwrite=args.overwrite)
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(
        json.dumps(
            {
                "converted": len(paths),
                "files": [str(path) for path in paths],
                "schema_version": SCHEMA_VERSION,
                "validated_with_harbor": bool(args.validate_with_harbor),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
