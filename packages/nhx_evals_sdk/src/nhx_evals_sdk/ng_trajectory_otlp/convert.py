# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project one NeMo-Gym rollout record onto OTLP spans.

Gym records a rollout as an OpenAI Responses API request/response pair, not as a trace: there is no
span tree and no timing anywhere in the record. What can be said honestly is that the agent ran
(one AGENT span), that a model was called (one LLM span carrying the usage the response reports),
and that any tool the model invoked was executed (one TOOL span per call). That is what this
builds; it does not synthesise structure the record does not evidence.

The result is the generated protobuf message rather than its JSON rendering, so a mistyped field or
a wrong-width id fails here rather than somewhere downstream, and ids stay bytes instead of picking
up a hex-or-base64 ambiguity on the way. It is a ``ResourceSpans`` and not an
``ExportTraceServiceRequest``: the latter is the envelope of OTLP's Export RPC, and whether these
spans are ever sent over a wire is the caller's business, not this projection's.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from hashlib import sha256
from math import isfinite
from typing import Any

from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

SPAN_KIND_ATTRIBUTE = "openinference.span.kind"

_SCOPE_NAME = "ng_trajectory_otlp"
_TRACE_ID_BYTES = 16
_SPAN_ID_BYTES = 8

# Both vocabularies are in play: Gym's model servers speak the Responses API's
# ``input_tokens``/``output_tokens`` or Chat Completions' ``prompt_tokens``/``completion_tokens``
# depending on the adapter, and reading only one set reports a real call as zero.
_PROMPT_TOKEN_KEYS = ("input_tokens", "prompt_tokens")
_COMPLETION_TOKEN_KEYS = ("output_tokens", "completion_tokens")


def rollout_to_resource_spans(
    record: Mapping[str, Any],
    *,
    rollout_id: str,
    task_id: str | None = None,
    model_calls: Sequence[Mapping[str, Any]] = (),
) -> list[ResourceSpans]:
    """Return the OTLP spans for one rollout, under a single resource.

    Ids are derived from ``rollout_id`` rather than generated, so converting the same record twice
    yields the same ids and a re-publish replaces its spans instead of duplicating them. That makes
    ``rollout_id`` load-bearing: two rollouts sharing one collide, and a consumer keyed on span
    identity would store only the last. A task id is not enough — one task can be attempted
    repeatedly, and Gym does not always record which attempt a record is.

    Args:
        record: One Gym rollout record.
        rollout_id: Identifies this attempt uniquely among all rollouts in the run.
        task_id: The task this rollout answers, used to name the root span.
        model_calls: This rollout's captured model exchanges, when Gym recorded them. Each becomes
            its own timed LLM span; without them the rollout record supports only one untimed span
            carrying the turn's summed usage.

    Returns:
        The rollout's resource spans, ready to hand to ``ExportTraceServiceRequest`` or to store:
        one AGENT root span with its LLM and TOOL children.
    """
    trace_id = _id_bytes(rollout_id, "trace", width=_TRACE_ID_BYTES)
    root_id = _id_bytes(rollout_id, "root", width=_SPAN_ID_BYTES)
    raw_response = record.get("response")
    # Gym records `response` as the Responses API payload, but some adapters write the answer as a
    # bare string. Both are answers; only the mapping carries usage and tool calls.
    answer = raw_response if isinstance(raw_response, str) and raw_response else None
    response = raw_response if isinstance(raw_response, Mapping) else {}

    llm_spans = [
        _captured_llm_span(call, trace_id=trace_id, parent_id=root_id, seed=rollout_id, ordinal=ordinal)
        for ordinal, call in enumerate(model_calls)
    ] or [_llm_span(record, response, trace_id=trace_id, parent_id=root_id, seed=rollout_id, answer=answer)]
    spans = [
        _root_span(record, response, trace_id=trace_id, span_id=root_id, task_id=task_id, answer=answer),
        *llm_spans,
        *_tool_spans(response, trace_id=trace_id, parent_id=root_id, seed=rollout_id),
    ]
    return [
        ResourceSpans(
            resource=Resource(attributes=_attributes({"service.name": _SCOPE_NAME})),
            scope_spans=[ScopeSpans(spans=spans)],
        )
    ]


def _root_span(
    record: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    trace_id: bytes,
    span_id: bytes,
    task_id: str | None,
    answer: str | None,
) -> Span:
    """The span standing for the whole rollout, carrying the answer a content metric scores."""
    attributes: dict[str, Any] = {SPAN_KIND_ATTRIBUTE: "AGENT"}
    if (prompt := _request_text(record)) is not None:
        attributes["input.value"] = prompt
    if (text := answer or response_output_text(response)) is not None:
        attributes["output.value"] = text
    reward = record.get("reward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        attributes["nemo.gym.reward"] = reward
    return Span(trace_id=trace_id, span_id=span_id, name=task_id or "rollout", attributes=_attributes(attributes))


def _llm_span(
    record: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    trace_id: bytes,
    parent_id: bytes,
    seed: str,
    answer: str | None,
) -> Span:
    """The model call the response reports, with whichever usage vocabulary it used."""
    attributes: dict[str, Any] = {SPAN_KIND_ATTRIBUTE: "LLM"}
    params = record.get("responses_create_params")
    model = response.get("model") or (params.get("model") if isinstance(params, Mapping) else None)
    if isinstance(model, str):
        attributes["gen_ai.request.model"] = model
    usage = response.get("usage")
    if isinstance(usage, Mapping):
        if (prompt := _token_count(usage, _PROMPT_TOKEN_KEYS)) is not None:
            attributes["gen_ai.usage.input_tokens"] = prompt
        if (completion := _token_count(usage, _COMPLETION_TOKEN_KEYS)) is not None:
            attributes["gen_ai.usage.output_tokens"] = completion
    if (text := answer or response_output_text(response)) is not None:
        attributes["output.value"] = text
    return Span(
        trace_id=trace_id,
        span_id=_id_bytes(seed, "llm", width=_SPAN_ID_BYTES),
        parent_span_id=parent_id,
        name="model call",
        attributes=_attributes(attributes),
    )


def _captured_llm_span(call: Mapping[str, Any], *, trace_id: bytes, parent_id: bytes, seed: str, ordinal: int) -> Span:
    """One captured model exchange, with the timing only the capture records.

    The rollout record reports a single usage block summed over the turn, so per-call tokens and
    latency exist here or nowhere.
    """
    attributes: dict[str, Any] = {SPAN_KIND_ATTRIBUTE: "LLM"}
    model_ref = call.get("model_ref")
    model = model_ref.get("name") if isinstance(model_ref, Mapping) else None
    if isinstance(model, str):
        attributes["gen_ai.request.model"] = model
    response = call.get("response")
    usage = response.get("usage") if isinstance(response, Mapping) else None
    if isinstance(usage, Mapping):
        if (prompt := _token_count(usage, _PROMPT_TOKEN_KEYS)) is not None:
            attributes["gen_ai.usage.input_tokens"] = prompt
        if (completion := _token_count(usage, _COMPLETION_TOKEN_KEYS)) is not None:
            attributes["gen_ai.usage.output_tokens"] = completion
        details = usage.get("input_tokens_details")
        if isinstance(details, Mapping) and (cached := _token_count(details, ("cached_tokens",))) is not None:
            attributes["gen_ai.usage.cache_read.input_tokens"] = cached
    if isinstance(ttft := call.get("latency_ttft_ms"), (int, float)) and not isinstance(ttft, bool):
        # No OTLP semantic convention names time-to-first-token on a span, so this stays namespaced
        # rather than borrowing a metric's name and implying a meaning it does not have.
        attributes["nemo.gym.latency_ttft_ms"] = float(ttft)
    span = Span(
        trace_id=trace_id,
        span_id=_id_bytes(seed, f"call:{ordinal}", width=_SPAN_ID_BYTES),
        parent_span_id=parent_id,
        name="model call",
        attributes=_attributes(attributes),
    )
    if (start := _nanos(call.get("started_at"))) is not None:
        span.start_time_unix_nano = start
    if (end := _nanos(call.get("completed_at"))) is not None:
        span.end_time_unix_nano = end
    status = call.get("status_code")
    if isinstance(status, int) and not 200 <= status < 300:
        span.status.code = Status.STATUS_CODE_ERROR
        if isinstance(category := call.get("error_category"), str):
            span.status.message = category
    return span


def _nanos(value: Any) -> int | None:
    """Convert a capture's epoch-seconds timestamp to the unix nanos OTLP spans carry.

    Scales the whole and fractional parts separately. Multiplying an epoch float by a billion in
    one step spends the mantissa on the seconds and rounds the nanoseconds: 1788534870.75 comes
    back as ...750000128.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value) or value <= 0:
        # ``json.loads`` accepts ``Infinity`` and ``NaN``, and ``int(inf)`` raises OverflowError --
        # which is neither a TypeError nor a ValueError, so it would escape the caller's guard and
        # take the whole rollout with it rather than costing this one timestamp.
        return None
    seconds = int(value)
    return seconds * 1_000_000_000 + round((value - seconds) * 1_000_000_000)


def _tool_spans(response: Mapping[str, Any], *, trace_id: bytes, parent_id: bytes, seed: str) -> Iterator[Span]:
    """One TOOL span per function call the model made, paired with its result where recorded."""
    items = response.get("output")
    if not isinstance(items, list):
        return
    # Keyed on the call id, so an output without a usable one is dropped rather than filed under
    # ``None``: every such output would share that key, and a call whose own id is missing would
    # then read the last one back as its result and attribute another tool's output to itself.
    results = {
        item.get("call_id"): item.get("output")
        for item in items
        if isinstance(item, Mapping)
        and item.get("type") == "function_call_output"
        and isinstance(item.get("call_id"), str)
    }
    ordinal = 0
    for item in items:
        if not isinstance(item, Mapping) or item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            # A call with no usable name cannot be attributed to a tool, and inventing one would
            # make a metric asking "was this tool used" answer about a tool that does not exist.
            continue
        attributes: dict[str, Any] = {SPAN_KIND_ATTRIBUTE: "TOOL", "tool.name": name}
        if isinstance(call_id := item.get("call_id"), str):
            attributes["tool_call.id"] = call_id
        if isinstance(arguments := item.get("arguments"), str):
            attributes["input.value"] = arguments
        if isinstance(result := results.get(item.get("call_id")), str):
            attributes["output.value"] = result
        yield Span(
            trace_id=trace_id,
            span_id=_id_bytes(seed, f"tool:{ordinal}", width=_SPAN_ID_BYTES),
            parent_span_id=parent_id,
            name=name,
            attributes=_attributes(attributes),
        )
        ordinal += 1


def response_output_text(response: Mapping[str, Any]) -> str | None:
    """Return the assistant's text from a Responses API payload, or ``None`` when it has none.

    Prefers the ``output_text`` convenience field, then the text parts of the last message item.
    A tool call is not an answer, so an output list carrying only calls yields nothing.
    """
    if isinstance(text := response.get("output_text"), str) and text:
        return text
    items = response.get("output")
    if not isinstance(items, list):
        return None
    for item in reversed(items):
        if not isinstance(item, Mapping) or item.get("type") not in (None, "message"):
            continue
        content = item.get("content")
        if isinstance(content, str) and content:
            return content
        if not isinstance(content, list):
            continue
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(text := part.get("text"), str) and text:
                parts.append(text)
        if parts:
            return "".join(parts)
    return None


def _request_text(record: Mapping[str, Any]) -> str | None:
    """Return the prompt the rollout was given, as text."""
    params = record.get("responses_create_params")
    if not isinstance(params, Mapping):
        return None
    value = params.get("input")
    if isinstance(value, str):
        return value or None
    if value in (None, [], {}):
        instructions = params.get("instructions")
        return instructions if isinstance(instructions, str) and instructions else None
    return json.dumps(value, default=str)


def _token_count(usage: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    """Return the first reported count among ``keys``, preserving an explicit zero."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _attributes(values: Mapping[str, Any]) -> list[KeyValue]:
    """Render a mapping as OTLP attributes, typed by value."""
    return [KeyValue(key=key, value=_any_value(value)) for key, value in values.items()]


def _any_value(value: Any) -> AnyValue:
    # bool before int: bool is an int subclass, so the reverse order writes True as an int_value.
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    if isinstance(value, float):
        return AnyValue(double_value=value)
    return AnyValue(string_value=value if isinstance(value, str) else json.dumps(value, default=str))


def _id_bytes(seed: str, role: str, *, width: int) -> bytes:
    """A stable id of exactly ``width`` bytes for one part of a rollout's trace."""
    return sha256(f"{seed}/{role}".encode()).digest()[:width]
