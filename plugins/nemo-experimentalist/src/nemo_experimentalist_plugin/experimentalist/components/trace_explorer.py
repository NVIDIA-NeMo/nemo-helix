# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from nemo_experimentalist_plugin.entities import ResourceRef
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.intake.client import AsyncIntakeClient
from nemo_platform_plugin.intake.types import ListEvaluatorResultsQueryParams, ListSpansQueryParams
from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """A tool/function call made by the LLM."""

    function_name: str
    arguments: str = "{}"
    tool_call_id: str = ""


class LLMMessage(BaseModel):
    """A single message in an LLM conversation."""

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str = ""

    def format_preview(self, max_length: int = 100) -> str:
        """Format a short preview of the message content."""
        if len(self.content) <= max_length:
            return self.content.replace("\n", " ")
        return self.content[:max_length].replace("\n", " ") + "..."


class ToolDefinition(BaseModel):
    """A tool definition available to the LLM."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class SpanStatus(BaseModel):
    """OpenTelemetry span status."""

    status_code: str = "OK"
    description: str | None = None


class Span(BaseModel):
    """Base OpenTelemetry/OpenInference span.

    Spans are raw trace units. Sessions and turns are derived from spans.
    """

    model_config = ConfigDict(extra="allow")

    span_id: str
    kind: str
    trace_id: str = ""
    parent_span_id: str | None = None
    name: str = ""
    start_time: int = 0
    end_time: int = 0
    duration_ns: int = 0
    status: SpanStatus = Field(default_factory=SpanStatus)
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_ns / 1_000_000


class LLMSpan(Span):
    """Represents an OpenInference LLM span.

    https://arize-ai.github.io/openinference/spec/llm_spans.html
    """

    kind: Literal["LLM"] = "LLM"
    system: str | None = None
    provider: str | None = None
    model_name: str | None = None
    invocation_parameters: dict[str, Any] = Field(default_factory=dict)
    input_value: str | None = None
    input_mime_type: str | None = None
    output_value: str | None = None
    output_mime_type: str | None = None
    input_messages: list[LLMMessage] = Field(default_factory=list)
    output_messages: list[LLMMessage] = Field(default_factory=list)
    tools: list[ToolDefinition] = Field(default_factory=list)
    token_counts: dict[str, int] = Field(default_factory=dict)


class ToolSpan(Span):
    """Represents an OpenInference TOOL span."""

    kind: Literal["TOOL"] = "TOOL"
    tool_name: str | None = None
    tool_call_id: str = ""
    input_value: str | None = None
    input_mime_type: str | None = None
    output_value: str | None = None
    output_mime_type: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class AgentSpan(Span):
    """Represents an OpenInference AGENT span.

    TraceExplorer derives AgentSession objects from these spans.
    """

    kind: Literal["AGENT"] = "AGENT"
    agent_name: str | None = None
    method_name: str | None = None
    input_value: str | None = None
    output_value: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class ChainSpan(Span):
    """Represents an OpenInference CHAIN span."""

    kind: Literal["CHAIN"] = "CHAIN"
    input_value: str | None = None
    output_value: str | None = None


class EvaluatorSpan(Span):
    """Represents an evaluation span."""

    kind: Literal["EVALUATOR"] = "EVALUATOR"
    evaluator_name: str | None = None
    input_value: str | None = None
    input_mime_type: str | None = None
    output_value: str | None = None
    output_mime_type: str | None = None
    score: Any = None


TraceSpan = LLMSpan | ToolSpan | AgentSpan | ChainSpan | EvaluatorSpan | Span


class LLMTurn(BaseModel):
    """Derived TraceExplorer view of one LLM generation turn."""

    session_id: str
    messages: list[LLMMessage]
    response: str
    model: str
    token_counts: dict[str, int] | None = None
    duration_ms: float | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_content: str = ""
    span_id: str = ""
    provider: str = ""
    invocation_parameters: dict[str, Any] = Field(default_factory=dict)
    tools: list[ToolDefinition] = Field(default_factory=list)
    start_time: int = 0
    end_time: int = 0


class ToolTurn(BaseModel):
    """Derived TraceExplorer view of one tool call result."""

    tool_name: str = ""
    input: str
    stdout: str = ""
    error: str | None = None
    output: Any = None
    status: str = "OK"
    duration_ms: float | None = None
    execution_id: str = ""
    generation_id: str = ""
    error_type: str | None = None
    span_id: str = ""
    tool_call_id: str = ""
    start_time: int = 0
    end_time: int = 0


class AgentSession(BaseModel):
    """A single agent method/session invocation."""

    session_id: str
    agent_name: str
    method_name: str
    parent_session_id: str | None
    depth: int = 0
    turns: list[LLMTurn | ToolTurn] = Field(default_factory=list)
    children: list[AgentSession] = Field(default_factory=list)
    start_time: int = 0
    end_time: int = 0
    result: Any = None
    status: str = "OK"
    span_id: str = ""
    method_signature: str = ""
    docstring: str = ""
    file_path: str = ""
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    strategy: str | None = None
    call_id: str = ""

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) / 1_000_000

    @property
    def full_name(self) -> str:
        """Full method name like 'RouterTestWrapper.process'."""
        return f"{self.agent_name}.{self.method_name}"

    def get_error_turns(self) -> list[ToolTurn]:
        """Get all turns that have errors."""
        return [t for t in self.turns if isinstance(t, ToolTurn) and t.error]

    def get_llm_turns(self) -> list[LLMTurn]:
        """Get all LLM turns."""
        return [t for t in self.turns if isinstance(t, LLMTurn)]

    def get_tool_turns(self) -> list[ToolTurn]:
        """Get all tool turns."""
        return [t for t in self.turns if isinstance(t, ToolTurn)]

    def get_execution_turns(self) -> list[ToolTurn]:
        """Compatibility alias for code that still says execution turns."""
        return self.get_tool_turns()


class SessionSummary(BaseModel):
    """Summary of one agent session."""

    session_id: str
    agent_name: str
    method_name: str
    status: str
    turn_count: int
    llm_turns: int
    tool_turns: int
    execution_turns: int = 0
    duration_ms: float
    parent_session_id: str | None = None
    has_children: bool = False
    result_preview: str | None = None


class TurnInfo(BaseModel):
    """Structured turn detail for programmatic use."""

    session_id: str
    turn_index: int
    turn_type: Literal["llm", "tool"]
    messages: list[LLMMessage] | None = None
    response: str | None = None
    model: str | None = None
    token_counts: dict[str, int] | None = None
    tool_calls: list[ToolCall] | None = None
    provider: str | None = None
    invocation_parameters: dict[str, Any] | None = None
    tools: list[ToolDefinition] | None = None
    code: str | None = None
    tool_name: str | None = None
    stdout: str | None = None
    error: str | None = None
    error_type: str | None = None
    output: Any = None
    status: str | None = None
    duration_ms: float | None = None
    span_id: str = ""
    tool_call_id: str = ""
    start_time: int = 0
    end_time: int = 0


class SessionData(BaseModel):
    """Structured session data."""

    session: SessionSummary
    turns: list[TurnInfo]


class SearchResult(BaseModel):
    """Result of searching trace content."""

    session_id: str
    turn_index: int
    turn_type: str
    location: str
    match_text: str
    line_number: int | None = None


class SearchMatches(BaseModel):
    """Structured search results."""

    pattern: str
    match_count: int
    matches: list[SearchResult]
    by_location: dict[str, int]

    def __bool__(self) -> bool:
        """Return true when at least one match exists."""
        return self.match_count > 0


class TimelineEvent(BaseModel):
    """One chronological event in a trace."""

    time_ns: int
    span_id: str
    event_type: str
    summary: str


class TimelineData(BaseModel):
    """Structured timeline output."""

    total_events: int
    max_events: int
    events: list[TimelineEvent]


class OverviewStats(BaseModel):
    """Trace overview counters."""

    duration_ms: float
    session_count: int
    turn_count: int
    runtime_errors: int
    eval_passed: bool | None = None


class RootSessionInfo(BaseModel):
    """Root/main session metadata."""

    agent_name: str
    method_name: str
    session_id: str


class EvalContextData(BaseModel):
    """Evaluation metadata attached to a trace, if available.

    This model also represents Intake evaluator result rows. When a trace has
    multiple evaluator rows, the top-level context is an aggregate and `results`
    contains one EvalContextData per row.
    """

    model_config = ConfigDict(extra="allow")

    test_id: str | None = None
    passed: bool | None = None
    input: Any = None
    expected: Any = None
    output: Any = None
    error: str | None = None
    evaluator_result_id: str = ""
    span_id: str = ""
    session_id: str = ""
    workspace: str = ""
    name: str = ""
    value: float | None = None
    string_value: str | None = None
    data_type: str = ""
    comment: str | None = None
    created_by: str | None = None
    created_at: datetime | None = None
    ingested_at: datetime | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)


class OverviewData(BaseModel):
    """Structured trace overview."""

    trace_id: str
    root: RootSessionInfo
    stats: OverviewStats
    sessions: list[SessionSummary]
    call_graph: list[dict[str, Any]]
    eval_result: EvalContextData | None = None
    benchmark_context: str | None = None


def _short_id(value: str | None) -> str:
    return (value or "")[:6]


def _kind(value: Any) -> str:
    text = str(value or "").split(".")[-1].upper()
    return text if text in {"AGENT", "CHAIN", "EVALUATOR", "LLM", "TOOL"} else "UNKNOWN"


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _preview(value: Any, limit: int = 160) -> str:
    text = _json_text(value).replace("\n", "\\n")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _attr(attrs: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = attrs.get(key)
        if value not in (None, ""):
            return value
    return None


def _time_ns(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value)
    if text.isdigit():
        return int(text)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _otel_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "doubleValue", "boolValue", "bytesValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value:
        return [_otel_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {item.get("key", ""): _otel_value(item.get("value")) for item in value["kvlistValue"].get("values", [])}
    return value


def _otel_attrs_to_dict(attributes: Any) -> dict[str, Any]:
    if isinstance(attributes, dict):
        return dict(attributes)
    result: dict[str, Any] = {}
    for item in attributes or []:
        if isinstance(item, dict) and "key" in item:
            result[item["key"]] = _otel_value(item.get("value"))
    return result


def _status_from_raw(raw: Any, attrs: dict[str, Any]) -> SpanStatus:
    error_message = _attr(attrs, "error.message", "exception.message")
    if isinstance(raw, dict):
        code = raw.get("status_code", raw.get("code"))
        description = raw.get("description", raw.get("message", error_message))
        if code in (2, "2", "ERROR", "error"):
            return SpanStatus(status_code="ERROR", description=description)
        return SpanStatus(status_code="OK", description=description)
    if str(raw or "").lower() == "error" or error_message:
        return SpanStatus(status_code="ERROR", description=error_message)
    return SpanStatus(status_code="OK", description=None)


def _tool_call_from_raw(raw: Any) -> ToolCall | None:
    if not isinstance(raw, dict):
        return None
    function = raw.get("function")
    if not isinstance(function, dict):
        function = {}
    name = function.get("name") or raw.get("name") or raw.get("tool_call.function.name")
    if not name:
        return None
    arguments = function.get("arguments") or raw.get("arguments") or raw.get("tool_call.function.arguments") or "{}"
    tool_call_id = raw.get("id") or raw.get("tool_call.id") or raw.get("tool_call_id") or ""
    return ToolCall(
        function_name=str(name),
        arguments=_json_text(arguments),
        tool_call_id=str(tool_call_id),
    )


def _messages_from_payload(value: Any) -> list[LLMMessage]:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, dict):
        messages = parsed.get("messages")
    elif isinstance(parsed, list):
        messages = parsed
    else:
        messages = None
    if not isinstance(messages, list):
        return []

    result: list[LLMMessage] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        nested = item.get("message")
        source = {**item, **nested} if isinstance(nested, dict) else item
        raw_tool_calls = source.get("tool_calls") or []
        tool_calls = [call for raw_call in raw_tool_calls if (call := _tool_call_from_raw(raw_call))]
        result.append(
            LLMMessage(
                role=str(source.get("role") or "unknown"),
                content=_json_text(source.get("content")),
                tool_calls=tool_calls,
                tool_call_id=str(source.get("tool_call_id") or source.get("tool_call.id") or ""),
            )
        )
    return result


def _messages_from_attrs(attrs: dict[str, Any], prefix: str) -> list[LLMMessage]:
    message_pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.message\.(.+)$")
    tool_pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\.(.+)$")
    messages: dict[int, dict[str, Any]] = {}
    tool_calls: dict[tuple[int, int], dict[str, Any]] = {}

    for key, value in attrs.items():
        if tool_match := tool_pattern.match(key):
            msg_idx = int(tool_match.group(1))
            call_idx = int(tool_match.group(2))
            tool_calls.setdefault((msg_idx, call_idx), {})[tool_match.group(3)] = value
            continue
        if msg_match := message_pattern.match(key):
            msg_idx = int(msg_match.group(1))
            messages.setdefault(msg_idx, {})[msg_match.group(2)] = value

    result: list[LLMMessage] = []
    for msg_idx in sorted(messages):
        data = messages[msg_idx]
        calls: list[ToolCall] = []
        for (owner_idx, _), call_data in sorted(tool_calls.items()):
            if owner_idx != msg_idx:
                continue
            name = _attr(call_data, "function.name", "name")
            if not name:
                continue
            calls.append(
                ToolCall(
                    function_name=str(name),
                    arguments=_json_text(_attr(call_data, "function.arguments", "arguments") or "{}"),
                    tool_call_id=str(_attr(call_data, "id", "tool_call_id") or ""),
                )
            )
        content = _attr(data, "content", "message_content.text")
        result.append(
            LLMMessage(
                role=str(data.get("role") or "unknown"),
                content=_json_text(content),
                tool_calls=calls,
                tool_call_id=str(_attr(data, "tool_call_id", "tool_call.id") or ""),
            )
        )
    return result


def _tools_from_attrs(attrs: dict[str, Any]) -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []
    pattern = re.compile(r"^llm\.tools\.(\d+)\.tool\.json_schema$")
    for key, value in sorted(attrs.items()):
        if not pattern.match(key):
            continue
        schema = _parse_jsonish(value)
        if not isinstance(schema, dict):
            continue
        raw_func = schema.get("function")
        function: dict[str, Any] = raw_func if isinstance(raw_func, dict) else schema
        name = function.get("name")
        if not name:
            continue
        raw_params = function.get("parameters")
        tools.append(
            ToolDefinition(
                name=str(name),
                description=str(function.get("description") or ""),
                parameters=raw_params if isinstance(raw_params, dict) else {},
            )
        )
    return tools


def _token_counts(attrs: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    prefix = "llm.token_count."
    for key, value in attrs.items():
        if key.startswith(prefix) and isinstance(value, (int, float, str)):
            try:
                counts[key.removeprefix(prefix)] = int(value)
            except ValueError:
                pass
    return counts


def _raw_attrs(raw: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if isinstance(raw.get("attributes"), list):
        attrs.update(_otel_attrs_to_dict(raw["attributes"]))
    elif isinstance(raw.get("attributes"), dict):
        attrs.update(raw["attributes"])
    parsed_raw = _parse_jsonish(raw.get("raw_attributes"))
    if isinstance(parsed_raw, dict):
        attrs.update(parsed_raw)
    return attrs


def _span_from_otlp_record(raw: dict[str, Any], resource_attrs: dict[str, Any] | None = None) -> TraceSpan:
    attrs = _raw_attrs(raw)
    if resource_attrs:
        for key, value in resource_attrs.items():
            attrs.setdefault(key, value)

    span_id = str(raw.get("span_id") or raw.get("spanId") or raw.get("external_span_id") or "")
    trace_id = str(raw.get("trace_id") or raw.get("traceId") or "")
    parent_span_id = raw.get("parent_span_id") or raw.get("parentSpanId") or raw.get("external_parent_span_id")
    parent_span_id = str(parent_span_id) if parent_span_id else None
    start_time = _time_ns(raw.get("start_time") or raw.get("started_at") or raw.get("startTimeUnixNano"))
    end_time = _time_ns(raw.get("end_time") or raw.get("ended_at") or raw.get("endTimeUnixNano"))
    status = _status_from_raw(raw.get("status"), attrs)
    kind = _kind(_attr(attrs, "openinference.span.kind") or raw.get("kind"))
    if kind == "UNKNOWN":
        kind = _kind(raw.get("openinference.span.kind"))

    input_value = raw.get("input")
    if input_value is None:
        input_value = _attr(attrs, "input.value")
    output_value = raw.get("output")
    if output_value is None:
        output_value = _attr(attrs, "output.value")

    raw_events = raw.get("events")
    base: dict[str, Any] = dict(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        name=str(raw.get("name") or ""),
        start_time=start_time,
        end_time=end_time,
        duration_ns=end_time - start_time if start_time and end_time else int(raw.get("duration_ns") or 0),
        status=status,
        attributes=attrs,
        events=raw_events if isinstance(raw_events, list) else [],
    )

    if kind == "LLM":
        input_messages = _messages_from_payload(input_value) or _messages_from_attrs(attrs, "llm.input_messages")
        output_messages = _messages_from_payload(output_value) or _messages_from_attrs(attrs, "llm.output_messages")
        return LLMSpan(
            **base,
            provider=str(raw.get("provider") or _attr(attrs, "llm.provider") or ""),
            model_name=str(raw.get("model") or _attr(attrs, "gen_ai.request.model", "llm.model_name") or ""),
            invocation_parameters=_parse_jsonish(_attr(attrs, "llm.invocation_parameters")) or {},
            input_value=_json_text(input_value) if input_value is not None else None,
            input_mime_type=_attr(attrs, "input.mime_type"),
            output_value=_json_text(output_value) if output_value is not None else None,
            output_mime_type=_attr(attrs, "output.mime_type"),
            input_messages=input_messages,
            output_messages=output_messages,
            tools=_tools_from_attrs(attrs),
            token_counts=_token_counts(attrs),
        )

    if kind == "TOOL":
        return ToolSpan(
            **base,
            tool_name=str(
                raw.get("tool_name") or _attr(attrs, "tool.name", "tool_call.function.name") or raw.get("name") or ""
            ),
            tool_call_id=str(_attr(attrs, "tool_call.id", "tool.id", "tool_call_id") or ""),
            input_value=_json_text(input_value) if input_value is not None else None,
            input_mime_type=_attr(attrs, "input.mime_type"),
            output_value=_json_text(output_value) if output_value is not None else None,
            output_mime_type=_attr(attrs, "output.mime_type"),
            error_type=_attr(attrs, "error.type", "exception.type"),
            error_message=_attr(attrs, "error.message", "exception.message") or status.description,
        )

    if kind == "AGENT":
        return AgentSpan(
            **base,
            agent_name=str(
                _attr(attrs, "gen_ai.agent.name", "agent.name", "nemo.agent.name") or raw.get("agent_name") or "Agent"
            ),
            method_name=str(_attr(attrs, "agent.method", "method.name") or raw.get("name") or "run"),
            input_value=_json_text(input_value) if input_value is not None else None,
            output_value=_json_text(output_value) if output_value is not None else None,
            error_type=_attr(attrs, "error.type", "exception.type"),
            error_message=_attr(attrs, "error.message", "exception.message") or status.description,
        )

    if kind == "CHAIN":
        return ChainSpan(
            **base,
            input_value=_json_text(input_value) if input_value is not None else None,
            output_value=_json_text(output_value) if output_value is not None else None,
        )

    if kind == "EVALUATOR":
        return EvaluatorSpan(
            **base,
            evaluator_name=str(raw.get("name") or _attr(attrs, "name", "evaluator.name") or ""),
            input_value=_json_text(input_value) if input_value is not None else None,
            input_mime_type=_attr(attrs, "input.mime_type"),
            output_value=_json_text(output_value) if output_value is not None else None,
            output_mime_type=_attr(attrs, "output.mime_type"),
            score=_attr(attrs, "score", "evaluator.score") or raw.get("score"),
        )

    return Span(**base, kind=kind)


# Intake column -> the token_counts key trace_explorer exposes.
_TOKEN_COLUMN_BY_KEY: tuple[tuple[str, str], ...] = (
    ("prompt", "input_tokens"),
    ("completion", "output_tokens"),
    ("cached", "cached_tokens"),
    ("total", "total_tokens"),
)

# ATIF step source -> chat message role.
_ROLE_BY_ATIF_SOURCE: dict[str, str] = {"user": "user", "system": "system", "agent": "assistant"}
_ATIF_INPUT_ROLES = frozenset({"user", "system"})


def _column_int(value: Any) -> int | None:
    """Coerce an Intake numeric column, tolerating strings and rejecting bools."""
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _intake_token_counts(row: dict[str, Any]) -> dict[str, int]:
    """Read token counts from the row's typed columns.

    Intake strips the ``llm.token_count.*`` keys from raw_attributes because it
    exposes them as columns, so the columns are the only source here.
    """
    counts: dict[str, int] = {}
    for key, column in _TOKEN_COLUMN_BY_KEY:
        parsed = _column_int(row.get(column))
        if parsed is not None:
            counts[key] = parsed
    return counts


def _intake_error_fields(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read error type and message from the row's typed columns."""

    def pick(value: Any) -> str | None:
        return str(value) if value not in (None, "") else None

    return pick(row.get("error_type")), pick(row.get("error_message"))


def _atif_messages(attrs: dict[str, Any]) -> tuple[list[LLMMessage], list[LLMMessage]]:
    """Synthesize messages from an ATIF step's source and message.

    ATIF records a step's text as a bare string plus a ``source`` discriminator,
    so there is no message list to recover — one is built from those two fields.
    """
    message = attrs.get("message")
    if not isinstance(message, str) or not message:
        return [], []
    role = _ROLE_BY_ATIF_SOURCE.get(str(attrs.get("source")), "assistant")
    entry = LLMMessage(role=role, content=message)
    if role in _ATIF_INPUT_ROLES:
        return [entry], []
    return [], [entry]


def _atif_tool_definitions(attrs: dict[str, Any]) -> list[ToolDefinition]:
    """Return the tool definitions ATIF preserved on the trajectory root span."""
    agent = attrs.get("agent")
    if not isinstance(agent, dict):
        return []
    definitions = agent.get("tool_definitions")
    if not isinstance(definitions, list):
        return []
    return [
        ToolDefinition(
            name=str(definition.get("name") or ""),
            description=str(definition.get("description") or ""),
            parameters=definition.get("parameters") if isinstance(definition.get("parameters"), dict) else {},
        )
        for definition in definitions
        if isinstance(definition, dict) and definition.get("name")
    ]


def _atif_tool_call_id(attrs: dict[str, Any]) -> str:
    """Return the tool call id ATIF preserved on a TOOL span."""
    tool_call = attrs.get("tool_call")
    if isinstance(tool_call, dict):
        call_id = tool_call.get("tool_call_id")
        if isinstance(call_id, str):
            return call_id
    return ""


def _looks_like_intake_row(record: dict[str, Any]) -> bool:
    """Intake rows always carry the source column; OTLP file records never do."""
    return "source" in record and ("span_id" in record or "external_span_id" in record)


def _span_from_intake_row(row: dict[str, Any]) -> TraceSpan:
    """Build a span from an Intake row, reading promoted values from columns.

    Intake lifts known attributes into typed columns and strips those keys from
    ``raw_attributes``, so the columns are authoritative here. Only values Intake
    preserved rather than promoted — ATIF's raw step payload, mime types,
    invocation parameters — are read from the attribute bag.
    """
    attrs = _raw_attrs(row)
    start_time = _time_ns(row.get("started_at"))
    end_time = _time_ns(row.get("ended_at"))
    status = _status_from_raw(row.get("status"), attrs)
    kind = _kind(row.get("kind"))
    error_type, error_message = _intake_error_fields(row)
    parent_span_id = row.get("parent_span_id") or row.get("external_parent_span_id")
    input_value = row.get("input")
    output_value = row.get("output")
    is_atif = row.get("source") == "atif"

    base: dict[str, Any] = dict(
        span_id=str(row.get("span_id") or row.get("external_span_id") or ""),
        trace_id=str(row.get("trace_id") or ""),
        parent_span_id=str(parent_span_id) if parent_span_id else None,
        name=str(row.get("name") or ""),
        start_time=start_time,
        end_time=end_time,
        duration_ns=end_time - start_time if start_time and end_time else 0,
        status=status,
        attributes=attrs,
        events=[],
    )

    if kind == "LLM":
        atif_input, atif_output = _atif_messages(attrs) if is_atif else ([], [])
        return LLMSpan(
            **base,
            provider=str(row.get("provider") or ""),
            model_name=str(row.get("model") or ""),
            invocation_parameters=_parse_jsonish(_attr(attrs, "llm.invocation_parameters")) or {},
            input_value=_json_text(input_value) if input_value is not None else None,
            input_mime_type=_attr(attrs, "input.mime_type"),
            output_value=_json_text(output_value) if output_value is not None else None,
            output_mime_type=_attr(attrs, "output.mime_type"),
            input_messages=_messages_from_payload(input_value) or atif_input,
            output_messages=_messages_from_payload(output_value) or atif_output,
            tools=_tools_from_attrs(attrs) or (_atif_tool_definitions(attrs) if is_atif else []),
            token_counts=_intake_token_counts(row),
        )

    if kind == "TOOL":
        return ToolSpan(
            **base,
            tool_name=str(row.get("tool_name") or row.get("name") or ""),
            # Intake has no tool_call_id column, so these keys survive in raw_attributes
            # for OTLP rows. ATIF preserves the id inside its tool_call payload instead.
            tool_call_id=str(
                _attr(attrs, "tool_call.id", "tool.id", "tool_call_id")
                or (_atif_tool_call_id(attrs) if is_atif else "")
            ),
            input_value=_json_text(input_value) if input_value is not None else None,
            input_mime_type=_attr(attrs, "input.mime_type"),
            output_value=_json_text(output_value) if output_value is not None else None,
            output_mime_type=_attr(attrs, "output.mime_type"),
            error_type=error_type,
            error_message=error_message or status.description,
        )

    if kind == "AGENT":
        return AgentSpan(
            **base,
            agent_name=str(row.get("agent_name") or "Agent"),
            method_name=str(row.get("name") or "run"),
            input_value=_json_text(input_value) if input_value is not None else None,
            output_value=_json_text(output_value) if output_value is not None else None,
            error_type=error_type,
            error_message=error_message or status.description,
        )

    if kind == "CHAIN":
        return ChainSpan(
            **base,
            input_value=_json_text(input_value) if input_value is not None else None,
            output_value=_json_text(output_value) if output_value is not None else None,
        )

    if kind == "EVALUATOR":
        return EvaluatorSpan(
            **base,
            evaluator_name=str(row.get("name") or ""),
            input_value=_json_text(input_value) if input_value is not None else None,
            input_mime_type=_attr(attrs, "input.mime_type"),
            output_value=_json_text(output_value) if output_value is not None else None,
            output_mime_type=_attr(attrs, "output.mime_type"),
            score=_attr(attrs, "score", "evaluator.score"),
        )

    return Span(**base, kind=kind)


def _looks_like_span_record(record: dict[str, Any]) -> bool:
    if "evaluator_result_id" in record:
        return False
    return any(
        key in record
        for key in (
            "span_id",
            "spanId",
            "external_span_id",
            "trace_id",
            "traceId",
            "attributes",
            "raw_attributes",
        )
    )


def _iter_span_records(
    document: Any,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    if isinstance(document, list):
        records: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        for item in document:
            records.extend(_iter_span_records(item))
        return records

    if not isinstance(document, dict):
        return []

    if "resourceSpans" in document:
        records = []
        for resource_spans in document.get("resourceSpans", []):
            resource_attrs = _otel_attrs_to_dict(resource_spans.get("resource", {}).get("attributes", []))
            for scope_spans in resource_spans.get("scopeSpans", []):
                for span in scope_spans.get("spans", []):
                    records.append((span, resource_attrs))
        return records

    if isinstance(document.get("data"), list):
        return [(span, None) for span in document["data"] if isinstance(span, dict) and _looks_like_span_record(span)]

    if "spans" in document and isinstance(document["spans"], list):
        return [(span, None) for span in document["spans"] if isinstance(span, dict) and _looks_like_span_record(span)]

    if _looks_like_span_record(document):
        return [(document, None)]

    return []


def _eval_context_from_record(raw: dict[str, Any]) -> EvalContextData:
    context = EvalContextData.model_validate(raw)
    passed = _eval_context_passed(context)
    if passed is not None and context.passed is None:
        context = context.model_copy(update={"passed": passed})
    if not context.test_id and context.session_id:
        context = context.model_copy(update={"test_id": context.session_id})
    return context


def _iter_evaluator_result_records(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        records: list[dict[str, Any]] = []
        for item in document:
            records.extend(_iter_evaluator_result_records(item))
        return records

    if not isinstance(document, dict):
        return []

    rows: list[dict[str, Any]] = []
    for key in ("evaluator_results", "evaluatorResults"):
        value = document.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))

    data = document.get("data")
    if isinstance(data, list):
        rows.extend(item for item in data if isinstance(item, dict) and "evaluator_result_id" in item)

    if "evaluator_result_id" in document:
        rows.append(document)

    return rows


def _load_trace_models(
    path: str | Path,
) -> tuple[list[TraceSpan], list[EvalContextData]]:
    text = Path(path).read_text()
    documents: list[Any] = []
    try:
        documents.append(json.loads(text))
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    spans: list[TraceSpan] = []
    eval_contexts: list[EvalContextData] = []
    for document in documents:
        for record, resource_attrs in _iter_span_records(document):
            # This path deliberately accepts either shape — _looks_like_span_record
            # matches Intake-shaped records, so a dumped Intake payload must keep
            # working here. Only from_intake, which knows its source, skips the check.
            span = (
                _span_from_intake_row(record)
                if _looks_like_intake_row(record)
                else _span_from_otlp_record(record, resource_attrs)
            )
            if span.span_id:
                spans.append(span)
        eval_contexts.extend(_eval_context_from_record(record) for record in _iter_evaluator_result_records(document))
    return spans, eval_contexts


def _load_span_models(path: str | Path) -> list[TraceSpan]:
    spans, _ = _load_trace_models(path)
    return spans


async def _fetch_intake_eval_contexts(
    *,
    client: AsyncNemoClient,
    workspace: str,
    session_ids: set[str],
    page_size: int,
) -> list[EvalContextData]:
    results: list[EvalContextData] = []
    seen: set[str] = set()

    intake = AsyncIntakeClient.from_client(client)
    for session_id in sorted(session_ids):
        query_params: ListEvaluatorResultsQueryParams = {
            "filter": {"session_id": session_id},
            "page_size": max(1, page_size),
            "sort": "created_at",
        }
        paginator = await intake.list_evaluator_results(workspace=workspace, query_params=query_params)
        async for item in paginator.items():
            row = item.model_dump(mode="json", exclude_none=True)
            result = _eval_context_from_record(row)
            dedupe_key = result.evaluator_result_id or json.dumps(row, sort_keys=True, default=str)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            results.append(result)

    return results


def _eval_context_value(context: EvalContextData) -> Any:
    if context.value is not None:
        return context.value
    if context.string_value is not None:
        return context.string_value
    return context.output


def _eval_context_passed(context: EvalContextData) -> bool | None:
    if context.passed is not None:
        return context.passed
    data_type = context.data_type.upper()
    if data_type == "BOOLEAN" and context.value is not None:
        return context.value == 1.0
    if context.string_value is None:
        return None
    text = context.string_value.strip().lower()
    if text in {"true", "pass", "passed", "success", "ok"}:
        return True
    if text in {"false", "fail", "failed", "failure", "error"}:
        return False
    return None


def _eval_context_result_dict(context: EvalContextData) -> dict[str, Any]:
    return {
        "test_id": context.test_id,
        "passed": context.passed,
        "input": context.input,
        "expected": context.expected,
        "output": context.output,
        "error": context.error,
        "evaluator_result_id": context.evaluator_result_id,
        "span_id": context.span_id,
        "session_id": context.session_id,
        "workspace": context.workspace,
        "name": context.name,
        "value": context.value,
        "string_value": context.string_value,
        "data_type": context.data_type,
        "comment": context.comment,
        "created_by": context.created_by,
        "created_at": context.created_at,
        "ingested_at": context.ingested_at,
    }


def _merge_eval_contexts(
    eval_contexts: list[EvalContextData],
) -> EvalContextData | None:
    if not eval_contexts:
        return None

    normalized = []
    for context in eval_contexts:
        passed = _eval_context_passed(context)
        if passed is not None and context.passed is None:
            context = context.model_copy(update={"passed": passed})
        if not context.test_id and context.session_id:
            context = context.model_copy(update={"test_id": context.session_id})
        normalized.append(context)

    if len(normalized) == 1:
        return normalized[0]

    pass_values = [context.passed for context in normalized if context.passed is not None]
    session_ids = sorted({context.session_id for context in normalized if context.session_id})
    output: dict[str, Any] = {}
    for index, context in enumerate(normalized):
        key = context.name or context.evaluator_result_id or f"result_{index}"
        value = _eval_context_value(context)
        if key not in output:
            output[key] = value
            continue
        existing = output[key]
        if not isinstance(existing, list):
            output[key] = [existing]
        output[key].append(value)

    return EvalContextData(
        test_id=session_ids[0] if len(session_ids) == 1 else None,
        passed=all(pass_values) if pass_values else None,
        output=output or None,
        results=[_eval_context_result_dict(context) for context in normalized],
    )


def _flatten_sessions(sessions: list[AgentSession]) -> list[AgentSession]:
    result: list[AgentSession] = []
    for session in sessions:
        result.append(session)
        result.extend(_flatten_sessions(session.children))
    return result


def _turn_from_llm_span(span: LLMSpan, session_id: str) -> LLMTurn:
    tool_calls = [tc for message in span.output_messages for tc in message.tool_calls]
    response_parts = [message.content for message in span.output_messages if message.content]
    if not response_parts and span.output_value:
        parsed = _parse_jsonish(span.output_value)
        if not (isinstance(parsed, dict) and isinstance(parsed.get("messages"), list)):
            response_parts = [span.output_value]
    return LLMTurn(
        session_id=_short_id(_attr(span.attributes, "generation.id") or session_id),
        messages=span.input_messages,
        response="\n".join(response_parts),
        model=span.model_name or "",
        token_counts=span.token_counts or None,
        duration_ms=span.duration_ms,
        tool_calls=tool_calls,
        reasoning_content=str(_attr(span.attributes, "llm.reasoning_content") or ""),
        span_id=span.span_id,
        provider=span.provider or "",
        invocation_parameters=span.invocation_parameters,
        tools=span.tools,
        start_time=span.start_time,
        end_time=span.end_time,
    )


def _turn_from_tool_span(span: ToolSpan) -> ToolTurn:
    parsed_input = _parse_jsonish(span.input_value)
    parsed_output = _parse_jsonish(span.output_value)
    stdout = ""
    output: Any = parsed_output
    error = span.error_message

    if isinstance(parsed_output, dict):
        stdout = str(parsed_output.get("stdout") or "")
        error = error or parsed_output.get("error")
        output = parsed_output.get("returned_value", parsed_output.get("result", parsed_output))
    input_text = _json_text(parsed_input)
    status = "ERROR" if error or span.status.status_code == "ERROR" else "OK"

    return ToolTurn(
        tool_name=span.tool_name or span.name,
        input=input_text,
        stdout=stdout,
        error=error,
        output=output,
        status=status,
        duration_ms=span.duration_ms,
        execution_id=_short_id(_attr(span.attributes, "execution.id") or span.span_id),
        generation_id=_short_id(_attr(span.attributes, "generation.id")),
        error_type=span.error_type,
        span_id=span.span_id,
        tool_call_id=span.tool_call_id,
        start_time=span.start_time,
        end_time=span.end_time,
    )


def _session_from_agent_span(span: AgentSpan, parent_session_id: str | None) -> AgentSession:
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    parsed_input = _parse_jsonish(span.input_value)
    if isinstance(parsed_input, dict):
        parsed_args = _parse_jsonish(parsed_input.get("args"))
        parsed_kwargs = _parse_jsonish(parsed_input.get("kwargs"))
        if isinstance(parsed_args, list):
            args = parsed_args
        elif parsed_args not in (None, ""):
            args = [parsed_args]
        if isinstance(parsed_kwargs, dict):
            kwargs = parsed_kwargs

    return AgentSession(
        session_id=_short_id(span.span_id),
        agent_name=span.agent_name or "Agent",
        method_name=span.method_name or span.name or "run",
        parent_session_id=parent_session_id,
        start_time=span.start_time,
        end_time=span.end_time,
        result=_parse_jsonish(_attr(span.attributes, "agent.result") or span.output_value),
        status="ERROR" if span.status.status_code == "ERROR" or span.error_message else "OK",
        span_id=span.span_id,
        method_signature=str(_attr(span.attributes, "agent.method_signature") or ""),
        docstring=str(_attr(span.attributes, "agent.docstring") or ""),
        file_path=str(_attr(span.attributes, "agent.file_path") or ""),
        args=args,
        kwargs=kwargs,
        error_message=span.error_message,
        strategy=_attr(span.attributes, "agent.strategy.name", "agent.strategy"),
        call_id=str(_attr(span.attributes, "agent.call_id") or ""),
    )


def _session_from_chain_span(span: ChainSpan) -> AgentSession:
    return AgentSession(
        session_id=_short_id(span.span_id),
        agent_name="Chain",
        method_name=span.name or "root",
        parent_session_id=None,
        start_time=span.start_time,
        end_time=span.end_time,
        result=_parse_jsonish(span.output_value),
        status="ERROR" if span.status.status_code == "ERROR" else "OK",
        span_id=span.span_id,
    )


def _build_sessions(spans: list[TraceSpan]) -> list[AgentSession]:
    span_by_id = {span.span_id: span for span in spans if span.span_id}
    agent_spans = sorted(
        [span for span in spans if isinstance(span, AgentSpan)],
        key=lambda span: (span.start_time, span.span_id),
    )

    def nearest_agent_id(span: TraceSpan) -> str | None:
        parent_id = span.parent_span_id
        seen: set[str] = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = span_by_id.get(parent_id)
            if isinstance(parent, AgentSpan):
                return parent.span_id
            parent_id = parent.parent_span_id if parent else None
        return None

    if not agent_spans:
        root_chain_spans = sorted(
            [
                span
                for span in spans
                if isinstance(span, ChainSpan) and (not span.parent_span_id or span.parent_span_id not in span_by_id)
            ],
            key=lambda span: (span.start_time, span.span_id),
        )
        if not root_chain_spans:
            trace_id = next((span.trace_id for span in spans if span.trace_id), "trace")
            session = AgentSession(
                session_id=_short_id(trace_id) or "trace",
                agent_name="Trace",
                method_name="root",
                parent_session_id=None,
                start_time=min((span.start_time for span in spans if span.start_time), default=0),
                end_time=max((span.end_time for span in spans if span.end_time), default=0),
            )
            session.turns = _turns_for_spans(spans, session.session_id)
            return [session]

        session_by_span_id = {span.span_id: _session_from_chain_span(span) for span in root_chain_spans}

        def root_chain_id(span: TraceSpan) -> str | None:
            parent_id = span.parent_span_id
            root_id: str | None = None
            seen: set[str] = set()
            while parent_id and parent_id not in seen:
                seen.add(parent_id)
                parent = span_by_id.get(parent_id)
                if isinstance(parent, ChainSpan):
                    root_id = parent.span_id
                parent_id = parent.parent_span_id if parent else None
            if root_id in session_by_span_id:
                return root_id
            if len(session_by_span_id) == 1:
                return next(iter(session_by_span_id))
            return None

        turn_buckets: dict[str, list[tuple[int, LLMTurn | ToolTurn]]] = {span_id: [] for span_id in session_by_span_id}
        for span in spans:
            if isinstance(span, (LLMSpan, ToolSpan)):
                owner_id = root_chain_id(span)
                if not owner_id:
                    continue
                turn = (
                    _turn_from_llm_span(span, _short_id(owner_id))
                    if isinstance(span, LLMSpan)
                    else _turn_from_tool_span(span)
                )
                turn_buckets[owner_id].append((span.start_time, turn))

        for span_id, bucket in turn_buckets.items():
            session_by_span_id[span_id].turns = [turn for _, turn in sorted(bucket, key=lambda item: item[0])]

        return [session_by_span_id[span.span_id] for span in root_chain_spans]

    session_by_span_id: dict[str, AgentSession] = {}
    for span in agent_spans:
        parent_agent_id = nearest_agent_id(span)
        parent_session_id = _short_id(parent_agent_id) if parent_agent_id else None
        session_by_span_id[span.span_id] = _session_from_agent_span(span, parent_session_id)

    roots: list[AgentSession] = []
    for span in agent_spans:
        session = session_by_span_id[span.span_id]
        parent_agent_id = nearest_agent_id(span)
        parent = session_by_span_id.get(parent_agent_id or "")
        if parent is None:
            roots.append(session)
        else:
            parent.children.append(session)

    def set_depth(session: AgentSession, depth: int) -> None:
        session.depth = depth
        session.children.sort(key=lambda child: (child.start_time, child.session_id))
        for child in session.children:
            set_depth(child, depth + 1)

    for root in sorted(roots, key=lambda item: (item.start_time, item.session_id)):
        set_depth(root, 0)

    turn_buckets: dict[str, list[tuple[int, LLMTurn | ToolTurn]]] = {span_id: [] for span_id in session_by_span_id}
    for span in spans:
        if isinstance(span, (LLMSpan, ToolSpan)):
            owner_id = nearest_agent_id(span)
            if not owner_id or owner_id not in turn_buckets:
                continue
            turn = (
                _turn_from_llm_span(span, _short_id(owner_id))
                if isinstance(span, LLMSpan)
                else _turn_from_tool_span(span)
            )
            turn_buckets[owner_id].append((span.start_time, turn))

    for span_id, bucket in turn_buckets.items():
        session_by_span_id[span_id].turns = [turn for _, turn in sorted(bucket, key=lambda item: item[0])]

    return sorted(roots, key=lambda item: (item.start_time, item.session_id))


def _turns_for_spans(spans: list[TraceSpan], session_id: str) -> list[LLMTurn | ToolTurn]:
    turns: list[tuple[int, LLMTurn | ToolTurn]] = []
    for span in spans:
        if isinstance(span, LLMSpan):
            turns.append((span.start_time, _turn_from_llm_span(span, session_id)))
        elif isinstance(span, ToolSpan):
            turns.append((span.start_time, _turn_from_tool_span(span)))
    return [turn for _, turn in sorted(turns, key=lambda item: item[0])]


class TraceExplorer:
    """Programmatic interface for exploring traces.

    Load traces with `from_file(path)` for local JSON/JSONL, `from_intake(...)`
    for Intake API data, or `from_spans(...)` when another adapter already
    produced semantic spans.

    Public query methods return either readable text or structured `*_data`
    models. They operate on normalized sessions and turns, not on source-
    specific raw span envelopes.

    TraceExplorer accepts semantic OpenInference spans from any source. Source
    adapters (`from_file`, `from_intake`, `from_spans`) produce `TraceSpan`
    models; query methods operate only on sessions and turns derived from those
    models.
    """

    def __init__(
        self,
        trace_id: str,
        sessions: list[AgentSession],
        eval_context: EvalContextData | None = None,
        raw_spans: list[TraceSpan] | None = None,
        eval_contexts: list[EvalContextData] | None = None,
        benchmark_context: str | None = None,
    ):
        self.sessions = sessions
        self.raw_spans = raw_spans or []
        self.eval_context = eval_context or _merge_eval_contexts(eval_contexts or [])
        self.trace_id = trace_id
        self.benchmark_context = benchmark_context
        self._all_sessions = _flatten_sessions(sessions)
        self._session_by_id = {session.session_id: session for session in self._all_sessions}
        self._span_by_id = {span.span_id: span for span in self.raw_spans if span.span_id}

    @property
    def eval_result(self) -> EvalContextData | None:
        """Evaluation context, if provided."""
        return self.eval_context

    @property
    def task_name(self) -> str | None:
        """Task identifier recorded by trace or evaluator metadata."""
        attribute_names = (
            "nemo.test_case.name",
            "nemo.test_case.id",
            "test_case.id",
            "task.name",
            "task_name",
            "task.id",
            "task_id",
        )
        for span in self.raw_spans:
            for attribute_name in attribute_names:
                value = span.attributes.get(attribute_name)
                if value is not None and str(value).strip():
                    return str(value)

        if self.eval_context is not None and self.eval_context.test_id:
            return self.eval_context.test_id
        return None

    @property
    def agent_count(self) -> int:
        """Total number of agent sessions."""
        return len(self._all_sessions)

    @property
    def max_agent_depth(self) -> int:
        """Maximum session depth."""
        return max((session.depth for session in self._all_sessions), default=0)

    @classmethod
    def from_spans(
        cls,
        spans: list[TraceSpan],
        *,
        eval_result: EvalContextData | None = None,
        benchmark_context: str | None = None,
        eval_contexts: list[EvalContextData] | None = None,
        trace_id: str = "",
    ) -> TraceExplorer:
        """Build a TraceExplorer from already-normalized semantic spans."""
        return cls(
            sessions=_build_sessions(spans),
            eval_context=eval_result,
            raw_spans=spans,
            eval_contexts=eval_contexts,
            trace_id=trace_id,
            benchmark_context=benchmark_context,
        )

    @classmethod
    async def from_file(
        cls,
        trace_path: str | Path,
        eval_result: EvalContextData | None = None,
        benchmark_context: str | None = None,
    ) -> TraceExplorer:
        """Load a trace from JSON/JSONL without assuming one raw source envelope."""
        path = Path(trace_path)
        spans, eval_contexts = await asyncio.to_thread(_load_trace_models, path)
        if not spans:
            raise ValueError(f"No spans found in trace file: {path}")
        return cls.from_spans(
            spans,
            eval_result=eval_result,
            benchmark_context=benchmark_context,
            eval_contexts=eval_contexts,
            trace_id=str(path),
        )

    @classmethod
    async def from_ref(
        cls, ref: ResourceRef, client: AsyncNemoClient | None = None, workspace: str | None = None
    ) -> TraceExplorer:
        """Load a trace from a resource reference."""
        if ref.uri.startswith("file://"):
            if ref.uri.endswith(".atif.json"):
                raise ValueError(
                    f"Cannot read ATIF trajectory from disk: {ref.uri}. ATIF becomes spans only "
                    "through Intake ingest, so this trace was never uploaded — check for an earlier "
                    "persistence failure, or supply a client and workspace to load it from Intake."
                )
            return await cls.from_file(ref.uri[len("file://") :])
        elif ref.uri.startswith("intake://") and client is not None and workspace is not None:
            # Trial/eval traces are stored as ``intake://traces/<id>`` (see the
            # backend's persist path), while Eval Author traces attached to an Insight
            # use the bare ``intake://<id>`` form. Strip both prefixes so either resolves to the
            # raw trace id that Intake filters on.
            trace_id = ref.uri.removeprefix("intake://").removeprefix("traces/")
            return await cls.from_intake(client, trace_id, workspace=workspace)
        else:
            raise ValueError(f"Unsupported resource type: {ref.uri} or missing client/workspace")

    @classmethod
    async def from_intake(
        cls,
        client: AsyncNemoClient,
        trace_id: str,
        *,
        workspace: str,
        page_size: int = 1000,
        eval_result: EvalContextData | None = None,
        benchmark_context: str | None = None,
    ) -> TraceExplorer:
        """Load a trace from Intake by adapting returned rows into semantic spans."""
        spans: list[TraceSpan] = []
        session_ids: set[str] = set()

        query_params: ListSpansQueryParams = {
            "filter": {"trace_id": trace_id},
            "mode": "detailed",
            "page_size": max(1, page_size),
            "sort": "started_at",
        }
        paginator = await AsyncIntakeClient.from_client(client).list_spans(
            workspace=workspace, query_params=query_params
        )
        async for item in paginator.items():
            row = item.model_dump(mode="json", exclude_none=True)
            if row.get("session_id"):
                session_ids.add(str(row["session_id"]))
            spans.append(_span_from_intake_row(row))

        eval_contexts = await _fetch_intake_eval_contexts(
            client=client,
            workspace=workspace,
            session_ids=session_ids,
            page_size=page_size,
        )

        if not spans:
            raise ValueError(f"No spans found in Intake for trace: {trace_id}")
        return cls.from_spans(
            spans,
            eval_result=eval_result,
            benchmark_context=benchmark_context,
            eval_contexts=eval_contexts,
            trace_id=f"intake://{workspace}/{trace_id}",
        )

    async def help(self) -> str:
        """Return API documentation."""
        return inspect.cleandoc(self.__doc__ or "")

    def _find_session(self, session_id: str) -> AgentSession | None:
        if session_id in self._session_by_id:
            return self._session_by_id[session_id]
        for sid, session in self._session_by_id.items():
            if sid.startswith(session_id) or session_id.startswith(sid):
                return session
        return None

    def _summary(self, session: AgentSession) -> SessionSummary:
        return SessionSummary(
            session_id=session.session_id,
            agent_name=session.agent_name,
            method_name=session.method_name,
            status=session.status,
            turn_count=len(session.turns),
            llm_turns=len(session.get_llm_turns()),
            tool_turns=len(session.get_tool_turns()),
            execution_turns=len(session.get_tool_turns()),
            duration_ms=session.duration_ms,
            parent_session_id=session.parent_session_id,
            has_children=bool(session.children),
            result_preview=_preview(session.result) if session.result is not None else None,
        )

    def _turn_info(self, session_id: str, turn_index: int, turn: LLMTurn | ToolTurn) -> TurnInfo:
        if isinstance(turn, LLMTurn):
            return TurnInfo(
                session_id=session_id,
                turn_index=turn_index,
                turn_type="llm",
                messages=turn.messages,
                response=turn.response,
                model=turn.model,
                token_counts=turn.token_counts,
                tool_calls=turn.tool_calls,
                provider=turn.provider,
                invocation_parameters=turn.invocation_parameters,
                tools=turn.tools,
                duration_ms=turn.duration_ms,
                span_id=turn.span_id,
                start_time=turn.start_time,
                end_time=turn.end_time,
            )
        return TurnInfo(
            session_id=session_id,
            turn_index=turn_index,
            turn_type="tool",
            code=turn.input,
            tool_name=turn.tool_name,
            stdout=turn.stdout,
            error=turn.error,
            error_type=turn.error_type,
            output=turn.output,
            status=turn.status,
            duration_ms=turn.duration_ms,
            span_id=turn.span_id,
            tool_call_id=turn.tool_call_id,
            start_time=turn.start_time,
            end_time=turn.end_time,
        )

    async def get_session_list(self) -> list[SessionSummary]:
        """Return sessions root-first, including children."""
        return [self._summary(session) for session in self._all_sessions]

    async def get_span_id(self, session_id: str, turn_index: int) -> str | None:
        """Return the span ID for a turn."""
        session = self._find_session(session_id)
        if not session or turn_index < 0 or turn_index >= len(session.turns):
            return None
        return session.turns[turn_index].span_id

    async def get_overview_data(self) -> OverviewData | None:
        """Return structured trace overview."""
        if not self.sessions:
            return None
        total_turns = sum(len(session.turns) for session in self._all_sessions)
        runtime_errors = sum(
            1
            for session in self._all_sessions
            if session.status != "OK" or any(turn.error for turn in session.get_tool_turns())
        )
        root = self.sessions[0]
        return OverviewData(
            trace_id=self.trace_id,
            root=RootSessionInfo(
                agent_name=root.agent_name,
                method_name=root.method_name,
                session_id=root.session_id,
            ),
            stats=OverviewStats(
                duration_ms=sum(session.duration_ms for session in self._all_sessions),
                session_count=len(self._all_sessions),
                turn_count=total_turns,
                runtime_errors=runtime_errors,
                eval_passed=self.eval_context.passed if self.eval_context else None,
            ),
            sessions=[self._summary(session) for session in self._all_sessions],
            call_graph=[
                {
                    "session_id": session.session_id,
                    "full_name": session.full_name,
                    "depth": session.depth,
                    "status": session.status,
                    "turn_count": len(session.turns),
                    "duration_ms": session.duration_ms,
                    "parent_session_id": session.parent_session_id,
                }
                for session in self._all_sessions
            ],
            eval_result=self.eval_context,
            benchmark_context=self.benchmark_context,
        )

    async def get_overview(self, *, concise: bool = True) -> str:
        """Return readable high-level trace overview."""
        del concise
        data = await self.get_overview_data()
        if data is None:
            return "No sessions found in trace."
        lines = [
            f"# {data.root.agent_name}.{data.root.method_name}()",
            "",
            (
                f"Duration: {data.stats.duration_ms:.1f}ms | Sessions: {data.stats.session_count} | "
                f"Turns: {data.stats.turn_count} | Runtime Errors: {data.stats.runtime_errors}"
            ),
            "",
            "## Call Graph",
        ]
        for session in self._all_sessions:
            indent = "  " * session.depth
            lines.append(
                f"{indent}- [{session.session_id}] {session.full_name} "
                f"{len(session.turns)}t {session.duration_ms:.1f}ms [{session.status}]"
            )
        return "\n".join(lines)

    async def get_session_data(self, session_id: str) -> SessionData:
        """Return structured session detail."""
        session = self._find_session(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found.")
        return SessionData(
            session=self._summary(session),
            turns=[self._turn_info(session.session_id, idx, turn) for idx, turn in enumerate(session.turns)],
        )

    async def get_session(self, session_id: str, *, concise: bool = False) -> str:
        """Return readable session detail."""
        session = self._find_session(session_id)
        if not session:
            return f"Session '{session_id}' not found. Use get_overview() to see available sessions."
        lines = [
            f"# {session.full_name} [{session.session_id}]",
            f"Status: {session.status} | Turns: {len(session.turns)} | Duration: {session.duration_ms:.1f}ms",
        ]
        if session.result is not None:
            lines.append(f"Result: {_preview(session.result, 500 if not concise else 160)}")
        if session.error_message:
            lines.append(f"Error: {session.error_message}")
        lines.append("")
        lines.append("## Turns")
        for idx, turn in enumerate(session.turns):
            if isinstance(turn, LLMTurn):
                tool_names = ", ".join(call.function_name for call in turn.tool_calls)
                suffix = f" -> {tool_names}" if tool_names else ""
                lines.append(f"- {idx}: LLM {len(turn.messages)} messages{suffix}")
            else:
                lines.append(f"- {idx}: TOOL {turn.tool_name} [{turn.status}]")
        return "\n".join(lines)

    async def get_turn_data(self, session_id: str, turn_index: int) -> TurnInfo | None:
        """Return structured turn data."""
        session = self._find_session(session_id)
        if not session or turn_index < 0 or turn_index >= len(session.turns):
            return None
        return self._turn_info(session.session_id, turn_index, session.turns[turn_index])

    async def get_turn(self, session_id: str, turn_index: int) -> str:
        """Return readable turn detail."""
        info = await self.get_turn_data(session_id, turn_index)
        if info is None:
            session = self._find_session(session_id)
            size = len(session.turns) if session else 0
            return f"Turn index {turn_index} out of range (session has {size} turns)"
        if info.turn_type == "llm":
            lines = [f"# Turn {turn_index}: LLM", ""]
            for idx, message in enumerate(info.messages or []):
                lines.append(f"## Message {idx}: {message.role}")
                lines.append(message.content)
                for call in message.tool_calls:
                    lines.append(f"TOOL CALL {call.function_name}: {call.arguments}")
            if info.response:
                lines.extend(["", "## Response", info.response])
            if info.tool_calls:
                lines.append("")
                lines.append("## Tool Calls")
                for call in info.tool_calls:
                    lines.append(f"- {call.function_name}: {call.arguments}")
            return "\n".join(lines)
        lines = [f"# Turn {turn_index}: Tool", f"Tool input: {info.code or ''}"]
        if info.stdout:
            lines.extend(["", "## Stdout", info.stdout])
        if info.output is not None:
            lines.extend(["", "## Output", _json_text(info.output)])
        if info.error:
            lines.extend(["", "## Error", info.error])
        return "\n".join(lines)

    async def get_errors_data(self) -> dict[str, Any]:
        """Return structured errors."""
        errors: list[dict[str, Any]] = []
        for session in self._all_sessions:
            if session.status != "OK" or session.error_message:
                errors.append(
                    {
                        "session_id": session.session_id,
                        "turn_index": None,
                        "error_message": session.error_message or session.status,
                        "context": session.full_name,
                    }
                )
            for idx, turn in enumerate(session.turns):
                if isinstance(turn, ToolTurn) and turn.error:
                    errors.append(
                        {
                            "session_id": session.session_id,
                            "turn_index": idx,
                            "error_type": turn.error_type,
                            "error_message": turn.error,
                            "context": turn.tool_name,
                        }
                    )
        return {"count": len(errors), "errors": errors}

    async def get_errors(self) -> str:
        """Return readable errors."""
        data = await self.get_errors_data()
        if data["count"] == 0:
            return "No errors found in trace."
        lines = [f"Found {data['count']} error(s):", ""]
        for error in data["errors"]:
            turn = "" if error["turn_index"] is None else f" turn {error['turn_index']}"
            lines.append(f"- [{error['session_id']}]{turn}: {error['error_message']}")
        return "\n".join(lines)

    async def get_eval_context(self, concise: bool = True) -> str:
        """Return eval context if available."""
        del concise
        if not self.eval_context:
            return "No evaluation result provided."
        return json.dumps(self.eval_context.model_dump(), indent=2, default=str)

    async def get_eval_context_data(self) -> dict[str, Any]:
        """Return structured eval context."""
        if not self.eval_context:
            return {"error": "No evaluation result provided."}
        return {
            "eval_result": self.eval_context,
            "benchmark_context": self.benchmark_context,
        }

    def _search_turn(self, pattern: str) -> list[SearchResult]:
        if not pattern or len(pattern) > 256:
            return []
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        results: list[SearchResult] = []
        for session in self._all_sessions:
            for idx, turn in enumerate(session.turns):
                fields: list[tuple[str, str]] = []
                if isinstance(turn, LLMTurn):
                    fields.extend((f"message[{msg.role}]", msg.content) for msg in turn.messages)
                    fields.append(("response", turn.response))
                    fields.extend((f"tool_call[{call.function_name}]", call.arguments) for call in turn.tool_calls)
                    turn_type = "llm"
                else:
                    fields.extend(
                        [
                            ("tool_input", turn.input),
                            ("stdout", turn.stdout),
                            ("output", _json_text(turn.output)),
                            ("error", turn.error or ""),
                        ]
                    )
                    turn_type = "tool"
                for location, text in fields:
                    for match in regex.finditer(text or ""):
                        start = max(0, match.start() - 40)
                        end = min(len(text), match.end() + 40)
                        results.append(
                            SearchResult(
                                session_id=session.session_id,
                                turn_index=idx,
                                turn_type=turn_type,
                                location=location,
                                match_text=text[start:end],
                            )
                        )
        return results

    async def search_data(self, pattern: str) -> SearchMatches:
        """Search trace content and return structured matches."""
        matches = self._search_turn(pattern)
        counts = Counter(match.location for match in matches)
        return SearchMatches(
            pattern=pattern,
            match_count=len(matches),
            matches=matches,
            by_location=dict(counts),
        )

    async def search(self, pattern: str, *, concise: bool = True) -> str:
        """Search trace content."""
        del concise
        data = await self.search_data(pattern)
        if not data.matches:
            return f"No matches found for pattern: {pattern}"
        lines = [f"Found {data.match_count} match(es) for '{pattern}':", ""]
        for match in data.matches[:20]:
            lines.append(
                f"- [{match.session_id} t{match.turn_index}] {match.location}: {_preview(match.match_text, 120)}"
            )
        if data.match_count > 20:
            lines.append(f"... and {data.match_count - 20} more matches")
        return "\n".join(lines)

    async def get_turn_context(
        self,
        session_id: str,
        turn_index: int,
        max_length: int | None = None,
        include_system: bool = False,
    ) -> str:
        """Return message context for an LLM turn."""
        session = self._find_session(session_id)
        if not session or turn_index < 0 or turn_index >= len(session.turns):
            return ""
        turn = session.turns[turn_index]
        if not isinstance(turn, LLMTurn):
            return ""
        parts = [message.content for message in turn.messages if include_system or message.role.lower() != "system"]
        text = "\n".join(parts)
        return f"{text[:max_length]}... [truncated]" if max_length and len(text) > max_length else text

    async def search_in_turn_context(
        self,
        session_id: str,
        turn_index: int,
        pattern: str,
        max_matches: int = 10,
        include_system: bool = False,
    ) -> list[SearchResult]:
        """Search inside one turn context."""
        context = await self.get_turn_context(session_id, turn_index, include_system=include_system)
        if not context or not pattern or len(pattern) > 256:
            return []
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
        matches: list[SearchResult] = []
        for match in regex.finditer(context):
            if len(matches) >= max_matches:
                break
            start = max(0, match.start() - 40)
            end = min(len(context), match.end() + 40)
            matches.append(
                SearchResult(
                    session_id=session_id,
                    turn_index=turn_index,
                    turn_type="llm",
                    location="context",
                    match_text=context[start:end],
                )
            )
        return matches

    async def get_method_counts(self) -> dict[str, int]:
        """Return invocation count per method."""
        return dict(Counter(session.full_name for session in self._all_sessions))

    async def get_recursion_pattern(self) -> str:
        """Return high-level method involvement description."""
        counts = await self.get_method_counts()
        if not counts:
            return "no agent calls"
        repeated = [name for name, count in counts.items() if count > 1]
        if repeated:
            return f"repeated method calls ({', '.join(repeated)})"
        if len(counts) == 1:
            return f"single method ({next(iter(counts))})"
        return f"{len(counts)} methods involved"

    async def get_timeline_data(self, max_events: int = 50) -> TimelineData:
        """Return chronological span/session event timeline."""
        events: list[TimelineEvent] = []
        for session in self._all_sessions:
            events.append(
                TimelineEvent(
                    time_ns=session.start_time,
                    span_id=session.span_id,
                    event_type="AGENT_START",
                    summary=session.full_name,
                )
            )
            for idx, turn in enumerate(session.turns):
                if isinstance(turn, LLMTurn):
                    events.append(
                        TimelineEvent(
                            time_ns=turn.start_time or session.start_time + idx,
                            span_id=turn.span_id,
                            event_type="LLM",
                            summary=f"{len(turn.messages)} messages",
                        )
                    )
                else:
                    events.append(
                        TimelineEvent(
                            time_ns=turn.start_time or session.start_time + idx,
                            span_id=turn.span_id,
                            event_type="TOOL",
                            summary=f"{turn.tool_name} [{turn.status}]",
                        )
                    )
            events.append(
                TimelineEvent(
                    time_ns=session.end_time,
                    span_id=session.span_id,
                    event_type="AGENT_END",
                    summary=session.status,
                )
            )
        events.sort(key=lambda event: (event.time_ns, event.span_id))
        return TimelineData(total_events=len(events), max_events=max_events, events=events[:max_events])

    async def get_timeline(self, max_events: int = 50) -> str:
        """Return readable chronological timeline."""
        data = await self.get_timeline_data(max_events=max_events)
        lines = ["# Timeline", ""]
        for event in data.events:
            lines.append(f"[{_short_id(event.span_id)}] {event.event_type}: {event.summary}")
        if data.total_events > len(data.events):
            lines.append(f"... ({data.total_events - len(data.events)} more events)")
        return "\n".join(lines)

    async def find_first_error_data(self) -> dict[str, Any]:
        """Return first error by timeline order."""
        errors = (await self.get_errors_data())["errors"]
        if not errors:
            return {"error": "No errors found in trace."}
        return errors[0]

    async def find_first_error(self) -> str:
        """Return first error."""
        error = await self.find_first_error_data()
        if "error" in error and len(error) == 1:
            return error["error"]
        return json.dumps(error, indent=2, default=str)

    def _session_span_ids(self, session: AgentSession) -> set[str]:
        """Return the session span and all descendants by parent relationship."""
        span_ids = {session.span_id} if session.span_id else set()
        span_ids.update(turn.span_id for turn in session.turns if turn.span_id)
        changed = True
        while changed:
            changed = False
            for span in self.raw_spans:
                if span.parent_span_id in span_ids and span.span_id not in span_ids:
                    span_ids.add(span.span_id)
                    changed = True
        return span_ids

    async def get_harness_telemetry_data(self, session_id: str | None = None) -> dict[str, Any]:
        """Return any harness telemetry attributes exposed by spans.

        This method intentionally treats harness telemetry as optional
        key-value data. It does not require a specific span name or runtime
        package.
        """
        title = "Harness Telemetry (all sessions)"
        spans = self.raw_spans
        if session_id:
            session = self._find_session(session_id)
            if not session:
                return {"error": f"Session not found: {session_id}"}
            title = f"Harness Telemetry for session {session.session_id}"
            span_ids = self._session_span_ids(session)
            spans = [span for span in self.raw_spans if span.span_id in span_ids]

        merged: dict[str, Any] = {}
        prefill_type = ""
        for span in spans:
            for key, value in span.attributes.items():
                if not key.startswith("harness."):
                    continue
                if key == "harness.prefill_type":
                    prefill_type = str(value)
                    continue
                if key not in merged:
                    merged[key] = value
                    continue
                existing = merged[key]
                if isinstance(existing, int) and isinstance(value, int):
                    merged[key] = existing + value
                elif isinstance(existing, list) and isinstance(value, list):
                    merged[key] = existing + value
                elif existing != value:
                    values = existing if isinstance(existing, list) else [existing]
                    values.append(value)
                    merged[key] = values

        return {"title": title, "metrics": merged, "prefill_type": prefill_type}

    async def get_harness_telemetry(self, session_id: str | None = None) -> str:
        """Return readable harness telemetry, if the trace includes it."""
        data = await self.get_harness_telemetry_data(session_id)
        if "error" in data:
            return str(data["error"])
        title = str(data.get("title") or "Harness Telemetry")
        metrics = data.get("metrics", {})
        lines = [title, "-" * len(title), ""]
        if not metrics:
            lines.append("(no harness telemetry found)")
            return "\n".join(lines)
        for key, value in sorted(metrics.items()):
            lines.append(f"{key}: {_preview(value, 240)}")
        if data.get("prefill_type"):
            lines.append(f"harness.prefill_type: {data['prefill_type']}")
        return "\n".join(lines)

    async def find_span(self, span_id: str, *, json_output: bool = False) -> str:
        """Find a span by ID or prefix and show where it appears."""
        for session in self._all_sessions:
            for idx, turn in enumerate(session.turns):
                if turn.span_id and (turn.span_id == span_id or turn.span_id.startswith(span_id)):
                    turn_data = await self.get_turn_data(session.session_id, idx)
                    if json_output:
                        payload = {
                            "span_id": turn.span_id,
                            "session_id": session.session_id,
                            "agent": session.full_name,
                            "turn_index": idx,
                            "turn_count": len(session.turns),
                            "turn": turn_data.model_dump(mode="json") if turn_data else None,
                        }
                        return json.dumps(payload, indent=2, default=str)
                    header = (
                        f"# Span {_short_id(turn.span_id)} -> session {session.session_id} "
                        f"({session.full_name}), turn {idx} of {len(session.turns)}"
                    )
                    return f"{header}\n\n{await self.get_turn(session.session_id, idx)}"

        raw = await self.get_raw_span_data(span_id)
        if "error" in raw:
            return f"Span {span_id} not found in any session or raw spans."
        if json_output:
            return json.dumps(raw, indent=2, default=str)
        return f"# Span {_short_id(str(raw.get('span_id')))} (raw)\n\n{json.dumps(raw, indent=2, default=str)}"

    async def compare(self, other: TraceExplorer) -> str:
        """Compare this trace with another trace."""
        return await TraceExplorer.diff(self, other)

    async def compare_data(self, other: TraceExplorer) -> dict[str, Any]:
        """Return structured comparison data for this trace and another trace."""
        return await TraceExplorer.diff_data(self, other)

    @staticmethod
    def _turn_type(turn: LLMTurn | ToolTurn) -> str:
        return "llm" if isinstance(turn, LLMTurn) else "tool"

    @staticmethod
    def _turn_brief(turn: LLMTurn | ToolTurn) -> dict[str, Any]:
        if isinstance(turn, LLMTurn):
            return {
                "type": "llm",
                "model": turn.model,
                "message_count": len(turn.messages),
                "tool_calls": [call.function_name for call in turn.tool_calls],
                "span_id": turn.span_id,
            }
        return {
            "type": "tool",
            "tool_name": turn.tool_name,
            "status": turn.status,
            "has_error": bool(turn.error),
            "span_id": turn.span_id,
        }

    @classmethod
    def _find_prompt_diffs(cls, turn1: LLMTurn, turn2: LLMTurn) -> list[str]:
        """Find prompt-level differences between two LLM turns."""

        def extract_expr_paths(messages: list[LLMMessage]) -> dict[str, set[str]]:
            paths: dict[str, set[str]] = {}
            for message in messages:
                for match in re.finditer(r"<(\w+)[^>]*\bexpr=\"([^\"]+)\"", message.content):
                    paths.setdefault(match.group(1), set()).add(match.group(2))
            return paths

        diffs: list[str] = []
        paths1 = extract_expr_paths(turn1.messages)
        paths2 = extract_expr_paths(turn2.messages)
        for tag in sorted(set(paths1) | set(paths2)):
            left = paths1.get(tag, set())
            right = paths2.get(tag, set())
            if left and right:
                if left != right:
                    diffs.append(f"`<{tag}>` expr differs: {sorted(left)} vs {sorted(right)}")
            elif left:
                diffs.append(f"`<{tag}>` present in trace 1 only")
            elif right:
                diffs.append(f"`<{tag}>` present in trace 2 only")
        if len(turn1.messages) != len(turn2.messages):
            diffs.append(f"message count differs: {len(turn1.messages)} vs {len(turn2.messages)}")
        return diffs

    @classmethod
    def _turn_differences(cls, turn1: LLMTurn | ToolTurn, turn2: LLMTurn | ToolTurn) -> list[str]:
        diffs: list[str] = []
        if cls._turn_type(turn1) != cls._turn_type(turn2):
            return [f"turn type differs: {cls._turn_type(turn1)} vs {cls._turn_type(turn2)}"]
        if isinstance(turn1, LLMTurn) and isinstance(turn2, LLMTurn):
            if turn1.model != turn2.model:
                diffs.append(f"model differs: {turn1.model!r} vs {turn2.model!r}")
            calls1 = [call.function_name for call in turn1.tool_calls]
            calls2 = [call.function_name for call in turn2.tool_calls]
            if calls1 != calls2:
                diffs.append(f"tool calls differ: {calls1} vs {calls2}")
            diffs.extend(cls._find_prompt_diffs(turn1, turn2))
            return diffs
        if isinstance(turn1, ToolTurn) and isinstance(turn2, ToolTurn):
            if turn1.tool_name != turn2.tool_name:
                diffs.append(f"tool differs: {turn1.tool_name!r} vs {turn2.tool_name!r}")
            if turn1.status != turn2.status:
                diffs.append(f"status differs: {turn1.status} vs {turn2.status}")
            if bool(turn1.error) != bool(turn2.error):
                diffs.append("error presence differs")
        return diffs

    @classmethod
    def _matched_sessions(
        cls, trace1: TraceExplorer, trace2: TraceExplorer
    ) -> list[tuple[AgentSession | None, AgentSession | None]]:
        by_name1: dict[str, list[AgentSession]] = {}
        by_name2: dict[str, list[AgentSession]] = {}
        for session in trace1._all_sessions:
            by_name1.setdefault(session.full_name, []).append(session)
        for session in trace2._all_sessions:
            by_name2.setdefault(session.full_name, []).append(session)

        pairs: list[tuple[AgentSession | None, AgentSession | None]] = []
        for name in sorted(set(by_name1) | set(by_name2)):
            left = by_name1.get(name, [])
            right = by_name2.get(name, [])
            for index in range(max(len(left), len(right))):
                pairs.append(
                    (
                        left[index] if index < len(left) else None,
                        right[index] if index < len(right) else None,
                    )
                )
        return pairs

    @classmethod
    def _collect_prompt_expr_diffs(
        cls,
        matched_pairs: list[tuple[AgentSession | None, AgentSession | None]],
    ) -> list[dict[str, Any]]:
        diffs: list[dict[str, Any]] = []
        for session1, session2 in matched_pairs:
            if not session1 or not session2:
                continue
            for idx, (turn1, turn2) in enumerate(zip(session1.turns, session2.turns, strict=False)):
                if isinstance(turn1, LLMTurn) and isinstance(turn2, LLMTurn):
                    for diff in cls._find_prompt_diffs(turn1, turn2):
                        diffs.append(
                            {
                                "session": session1.full_name,
                                "session_id_1": session1.session_id,
                                "session_id_2": session2.session_id,
                                "turn_index": idx,
                                "diff": diff,
                            }
                        )
        return diffs

    @classmethod
    async def diff_data(cls, trace1: TraceExplorer, trace2: TraceExplorer) -> dict[str, Any]:
        """Return structured comparison data for two traces."""
        pairs = cls._matched_sessions(trace1, trace2)
        turn_differences: list[dict[str, Any]] = []
        missing_sessions: list[dict[str, Any]] = []

        for session1, session2 in pairs:
            if session1 is None and session2 is not None:
                missing_sessions.append({"trace": 1, "session": session2.full_name})
                continue
            if session2 is None and session1 is not None:
                missing_sessions.append({"trace": 2, "session": session1.full_name})
                continue
            if session1 is None or session2 is None:
                continue
            if len(session1.turns) != len(session2.turns):
                turn_differences.append(
                    {
                        "session": session1.full_name,
                        "session_id_1": session1.session_id,
                        "session_id_2": session2.session_id,
                        "kind": "turn_count",
                        "trace_1": len(session1.turns),
                        "trace_2": len(session2.turns),
                    }
                )
            for idx, (turn1, turn2) in enumerate(zip(session1.turns, session2.turns, strict=False)):
                diffs = cls._turn_differences(turn1, turn2)
                if diffs:
                    turn_differences.append(
                        {
                            "session": session1.full_name,
                            "session_id_1": session1.session_id,
                            "session_id_2": session2.session_id,
                            "turn_index": idx,
                            "differences": diffs,
                            "trace_1": cls._turn_brief(turn1),
                            "trace_2": cls._turn_brief(turn2),
                        }
                    )

        prompt_differences = cls._collect_prompt_expr_diffs(pairs)
        summary = {
            "file_1": Path(trace1.trace_id).name or trace1.trace_id or "(memory)",
            "file_2": Path(trace2.trace_id).name or trace2.trace_id or "(memory)",
            "sessions_1": len(trace1._all_sessions),
            "sessions_2": len(trace2._all_sessions),
            "total_turns_1": sum(len(session.turns) for session in trace1._all_sessions),
            "total_turns_2": sum(len(session.turns) for session in trace2._all_sessions),
            "status_1": "ERROR" if (await trace1.get_errors_data())["count"] else "OK",
            "status_2": "ERROR" if (await trace2.get_errors_data())["count"] else "OK",
            "eval_1": trace1.eval_context.passed if trace1.eval_context else None,
            "eval_2": trace2.eval_context.passed if trace2.eval_context else None,
        }
        return {
            "summary": summary,
            "equivalent": not missing_sessions and not turn_differences,
            "missing_sessions": missing_sessions,
            "turn_differences": turn_differences,
            "prompt_expression_differences": prompt_differences,
            "call_graphs": {
                "trace_1": [
                    {
                        **trace1._summary(session).model_dump(mode="json"),
                        "full_name": session.full_name,
                        "depth": session.depth,
                    }
                    for session in trace1._all_sessions
                ],
                "trace_2": [
                    {
                        **trace2._summary(session).model_dump(mode="json"),
                        "full_name": session.full_name,
                        "depth": session.depth,
                    }
                    for session in trace2._all_sessions
                ],
            },
            "first_difference": (missing_sessions or turn_differences or [None])[0],
        }

    @classmethod
    async def diff(cls, trace1: TraceExplorer, trace2: TraceExplorer) -> str:
        """Return a readable diff report for two traces."""
        data = await cls.diff_data(trace1, trace2)
        summary = data["summary"]
        lines = [
            "# Trace Comparison",
            "",
            "## Summary",
            "",
            "| Metric | Trace 1 | Trace 2 |",
            "|--------|---------|---------|",
            f"| File | {summary['file_1']} | {summary['file_2']} |",
            f"| Sessions | {summary['sessions_1']} | {summary['sessions_2']} |",
            f"| Total Turns | {summary['total_turns_1']} | {summary['total_turns_2']} |",
            f"| Status | {summary['status_1']} | {summary['status_2']} |",
            f"| Eval | {summary['eval_1']} | {summary['eval_2']} |",
            "",
            "## Call Graphs",
            "",
            "### Trace 1",
        ]
        for item in data["call_graphs"]["trace_1"]:
            lines.append(f"{'  ' * item['depth']}- {item['full_name']} ({item['turn_count']} turns, {item['status']})")
        lines.extend(["", "### Trace 2"])
        for item in data["call_graphs"]["trace_2"]:
            lines.append(f"{'  ' * item['depth']}- {item['full_name']} ({item['turn_count']} turns, {item['status']})")

        lines.extend(["", "## Differences"])
        if data["equivalent"]:
            lines.append("No structural differences found.")
        for missing in data["missing_sessions"]:
            lines.append(f"- Missing in trace {missing['trace']}: {missing['session']}")
        for diff in data["turn_differences"][:20]:
            if diff.get("kind") == "turn_count":
                lines.append(f"- {diff['session']}: turn count {diff['trace_1']} vs {diff['trace_2']}")
            else:
                lines.append(f"- {diff['session']} turn {diff['turn_index']}: {'; '.join(diff['differences'])}")
        if len(data["turn_differences"]) > 20:
            lines.append(f"... and {len(data['turn_differences']) - 20} more turn differences")
        return "\n".join(lines)

    async def get_raw_span_data(self, span_id: str) -> dict[str, Any]:
        """Return raw semantic span data."""
        for sid, span in self._span_by_id.items():
            if sid == span_id or sid.startswith(span_id):
                return span.model_dump(mode="json")
        return {"error": f"No span found matching '{span_id}'"}

    async def get_raw_span(self, span_id: str) -> str:
        """Return readable raw semantic span JSON."""
        data = await self.get_raw_span_data(span_id)
        if "error" in data:
            return data["error"]
        return json.dumps(data, indent=2, default=str)

    async def get_raw_spans(self, session_id: str) -> str:
        """Return raw spans associated with a session subtree."""
        session = self._find_session(session_id)
        if not session:
            return f"Session not found: {session_id}"
        span_ids = {session.span_id}
        changed = True
        while changed:
            changed = False
            for span in self.raw_spans:
                if span.parent_span_id in span_ids and span.span_id not in span_ids:
                    span_ids.add(span.span_id)
                    changed = True
        spans = [span.model_dump(mode="json") for span in self.raw_spans if span.span_id in span_ids]
        return json.dumps(spans, indent=2, default=str)
