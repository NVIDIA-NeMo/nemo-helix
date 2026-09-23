# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads over OTLP traces.

Two levels, deliberately: the protobuf message Intake ingests, for anything that needs the
schema honoured, and the decoded JSON, for text searches that must survive a trace one
malformed field away from failing to parse.
"""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Any, TypeAlias, cast

from google.protobuf import json_format
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span, Status

_MESSAGE_ATTRIBUTE_KEY = "gen_ai.output.messages"
_SPAN_KIND_ATTRIBUTE = "openinference.span.kind"

# Span attributes that can carry an agent's final answer, in precedence order.
#
# Deliberately narrower than Intake's ``OTLP_OUTPUT_PAYLOAD_ATTRIBUTE_KEYS``
# (``services/intake/src/nhx/intake/spans/ingest/otlp.py``), which answers "what payload did
# this span emit" for storage and so also admits ``gen_ai.tool.call.result`` and
# ``tool_response``. A tool result is not the agent's answer, and this value is compared
# against a reference by the content metrics, so admitting one would score a tool's output
# as the model's. The list is duplicated rather than imported because the SDK ships
# independently of the services and must not depend on one.
FINAL_OUTPUT_ATTRIBUTE_KEYS = (
    "output.value",
    _MESSAGE_ATTRIBUTE_KEY,
    "final_result",
)

# Span kinds whose output can be the agent's answer. An include list, so a kind this does not
# know about is ineligible: a tool result and a judge's score are both recorded in the same
# `output.value` attribute an answer uses, and scoring one as the model's answer is silent.
ANSWER_SPAN_KINDS = frozenset({"AGENT", "CHAIN", "LLM"})

#: Attribute value types OTLP's ``AnyValue`` carries that a caller here needs to write.
AttributeValue: TypeAlias = str | int | float | bool


def validate_resource_spans(value: Any) -> list[dict[str, Any]]:
    """Check that an OTLP ``resourceSpans`` value is a list of objects.

    Args:
        value: Candidate resource-span list.

    Returns:
        Resource-span objects in their original order.

    Raises:
        ValueError: The value is not a list of objects.
    """
    if not isinstance(value, list):
        raise ValueError("OTLP resourceSpans must be a list")
    resource_spans: list[dict[str, Any]] = []
    for index, resource_span in enumerate(value):
        if not isinstance(resource_span, dict):
            raise ValueError(f"OTLP resourceSpans[{index}] must be an object")
        resource_spans.append(cast(dict[str, Any], resource_span))
    return resource_spans


def resource_spans_from_request(request: Any) -> list[dict[str, Any]]:
    """Extract the validated ``resourceSpans`` of one OTLP/JSON request object.

    Args:
        request: Decoded OTLP/JSON request value.

    Returns:
        The request's resource-span objects.

    Raises:
        ValueError: The request is not an object or declares no ``resourceSpans``.
    """
    if not isinstance(request, dict):
        raise ValueError("OTLP trace request must be an object")
    if "resourceSpans" not in request:
        raise ValueError("OTLP trace request requires a resourceSpans list")
    return validate_resource_spans(request["resourceSpans"])


def resource_spans_from_text(raw: str) -> list[dict[str, Any]]:
    """Decode OTLP/JSON text into resource spans, in request and line order.

    Accepts either one request object or JSONL, which is how exporters that flush
    repeatedly write a trace to a single file.

    Args:
        raw: File or inline OTLP/JSON text.

    Returns:
        Concatenated resource-span objects; empty when the text is blank.

    Raises:
        ValueError: A line does not parse, or a request has the wrong shape.
    """
    if not raw.strip():
        return []
    try:
        requests = [_loads(raw)]
    except json.JSONDecodeError:
        requests = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                requests.append(_loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid OTLP JSON on line {line_number}") from exc

    resource_spans: list[dict[str, Any]] = []
    for request in requests:
        resource_spans.extend(resource_spans_from_request(request))
    return resource_spans


def parse_resource_spans(resource_spans: list[dict[str, Any]]) -> list[ResourceSpans]:
    """Parse OTLP/JSON resource spans into their protobuf messages.

    Returns the spans themselves rather than an ``ExportTraceServiceRequest``: that is the envelope
    of OTLP's Export RPC, and only a caller actually sending them over a wire needs one.

    Args:
        resource_spans: Decoded OTLP/JSON ``resourceSpans`` objects.

    Returns:
        The parsed resource spans, in the order given.

    Raises:
        ValueError: The payload does not match the OTLP schema.
    """
    try:
        request = json_format.ParseDict(
            {"resourceSpans": [_base64_ids(resource_span) for resource_span in resource_spans]},
            ExportTraceServiceRequest(),
            # A producer on a newer OTLP schema stays readable rather than failing wholesale.
            ignore_unknown_fields=True,
        )
        return list(request.resource_spans)
    except json_format.ParseError as error:
        raise ValueError(f"invalid OTLP payload: {error}") from error
    except RecursionError as error:
        raise ValueError("OTLP payload is nested too deeply to convert") from error


def final_output_text(resource_spans: list[ResourceSpans]) -> str | None:
    """Return the agent's final answer from an OTLP trace.

    A root span represents the run as a whole, so its output is the answer when it has one.
    Otherwise the latest-ending span carrying an output attribute is the closest equivalent
    to ATIF's "last agent step wins". Only :data:`ANSWER_SPAN_KINDS` are considered, so a
    trace of nothing but tool calls has no answer rather than a misattributed one.

    Args:
        resource_spans: Parsed OTLP resource spans.

    Returns:
        The answer text, or ``None`` when no span carries one.
    """
    spans = [span for rs in resource_spans for ss in rs.scope_spans for span in ss.spans if _is_answer_span(span)]
    by_recency = sorted(spans, key=lambda span: span.end_time_unix_nano, reverse=True)
    roots = [span for span in by_recency if not span.parent_span_id]
    for span in roots + by_recency:
        text = _span_output_text(span)
        if text:
            return text
    return None


def _is_answer_span(span: Span) -> bool:
    """Whether this span's output could be the agent's answer, by its declared kind."""
    for attribute in span.attributes:
        if attribute.key == _SPAN_KIND_ATTRIBUTE:
            return attribute.value.string_value.upper() in ANSWER_SPAN_KINDS
    return False


def _span_output_text(span: Span) -> str | None:
    """Return the answer text a single span carries, honouring key precedence."""
    attributes = {attribute.key: attribute.value for attribute in span.attributes}
    for key in FINAL_OUTPUT_ATTRIBUTE_KEYS:
        value = attributes.get(key)
        if value is None:
            continue
        raw = value.string_value
        if not raw:
            continue
        text = _assistant_text(raw) if key == _MESSAGE_ATTRIBUTE_KEY else raw
        if text:
            return text
    return None


def _assistant_text(raw: str) -> str | None:
    """Extract the assistant's text from a ``gen_ai.output.messages`` payload.

    The payload is a JSON array of ``{"role", "parts": [{"type", "content"}]}`` messages.
    Structured data carrying no assistant text yields ``None``, so a caller can fall back to
    another key or another trace format rather than scoring serialized telemetry. Text that
    is not JSON at all is returned unchanged: a producer writing the answer as plain text
    here is the one case where the raw value is the answer.
    """
    try:
        messages = _loads(raw)
    except ValueError:
        return raw
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        contents: list[str] = []
        for part in parts:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            content = part.get("content")
            if isinstance(content, str):
                contents.append(content)
        if contents:
            return "".join(contents)
    return None


def span_text_strings(resource_spans: list[ResourceSpans]) -> Iterator[str]:
    """Yield every string carried by the spans' and span events' attributes.

    Args:
        resource_spans: Parsed OTLP resource spans.

    Yields:
        String leaves, including those nested in array and key-value attribute values.
    """
    for resource_span in resource_spans:
        for scope_span in resource_span.scope_spans:
            for span in scope_span.spans:
                for attribute in span.attributes:
                    yield from _any_value_strings(attribute.value)
                for event in span.events:
                    for attribute in event.attributes:
                        yield from _any_value_strings(attribute.value)


def _objects(value: Any) -> list[dict[str, Any]]:
    """Return only the object elements of a decoded OTLP repeated field."""
    if not isinstance(value, list):
        return []
    return [cast(dict[str, Any], item) for item in value if isinstance(item, dict)]


def _any_value_strings(value: AnyValue) -> Iterator[str]:
    """Yield the string leaves of one ``AnyValue``, in no particular order.

    Walks an explicit stack rather than recursing, so nesting depth is bounded by what the
    protobuf parser already accepted rather than by this process's recursion limit.
    """
    pending = [value]
    while pending:
        current = pending.pop()
        field = current.WhichOneof("value")
        if field == "string_value":
            yield current.string_value
        elif field == "array_value":
            pending.extend(current.array_value.values)
        elif field == "kvlist_value":
            pending.extend(item.value for item in current.kvlist_value.values)


def _loads(text: str) -> Any:
    """Decode JSON, normalizing ``RecursionError`` into this module's ``ValueError`` contract.

    ``RecursionError`` is not a ``ValueError``, so a deeply nested payload would otherwise
    escape every caller's guard.
    """
    try:
        return json.loads(text)
    except RecursionError as error:
        raise ValueError("OTLP JSON is nested too deeply to parse") from error


_ID_BYTE_LENGTHS = {"traceId": 16, "spanId": 8, "parentSpanId": 8}


def _base64_ids(resource_span: dict[str, Any]) -> dict[str, Any]:
    """Return a resource span with its trace and span ids re-encoded as base64.

    OTLP/JSON writes ids as hex, which is a deliberate departure from the protobuf JSON
    mapping that ``ParseDict`` implements. Left alone, ``ParseDict`` reads a 16-character
    hex span id as base64 and yields twelve unrelated bytes, so ids silently stop matching
    the ones the producer recorded.
    """
    converted = deepcopy(resource_span)
    for scope_span in _objects(converted.get("scopeSpans")):
        for span in _objects(scope_span.get("spans")):
            _base64_span_ids(span)
            for link in _objects(span.get("links")):
                _base64_span_ids(link)
    return converted


def _base64_span_ids(span: dict[str, Any]) -> None:
    """Rewrite one span's hex id fields to base64 in place, leaving other encodings alone."""
    for field, byte_length in _ID_BYTE_LENGTHS.items():
        value = span.get(field)
        if not isinstance(value, str) or len(value) != byte_length * 2:
            continue
        try:
            raw = bytes.fromhex(value)
        except ValueError:
            continue
        span[field] = b64encode(raw).decode("ascii")


def set_span_attributes(resource_spans: list[ResourceSpans], attributes: Mapping[str, AttributeValue]) -> None:
    """Set attributes on every span of a trace, replacing any the producer set.

    Span attributes are written rather than resource attributes because Intake merges the two
    layers as ``{**resource, **span}``. A producer that publishes one of these keys itself
    would win from the resource layer and silently redirect whatever the value identifies.
    """
    for span in _spans(resource_spans):
        _set_attributes(span, attributes)


def set_root_span_attributes(resource_spans: list[ResourceSpans], attributes: Mapping[str, AttributeValue]) -> None:
    """Set attributes on the trace's root span alone, if it has exactly one.

    Totals for a whole run belong on the one span that represents it. Writing them to every
    span would have any rollup that sums across a trace count them once per span.
    """
    root = _root_span(resource_spans)
    if root is not None:
        _set_attributes(root, attributes)


def set_root_span_error(resource_spans: list[ResourceSpans], *, message: str | None = None) -> None:
    """Mark the trace's root span as failed, if it has exactly one.

    Intake reads span status, not attributes, to decide a span failed.
    """
    root = _root_span(resource_spans)
    if root is None:
        return
    root.status.code = Status.STATUS_CODE_ERROR
    if message is not None:
        root.status.message = message


def fill_missing_start_times(resource_spans: list[ResourceSpans], *, start_time_unix_nano: int) -> None:
    """Give spans without a start time one, so re-publishing the same trace replaces it.

    Intake stores a span with no start time against its own ingest clock, and start time is
    part of the key its spans table replaces on — so an absent one makes every publish a new
    row rather than a replacement of the last.
    """
    for span in _spans(resource_spans):
        if not span.start_time_unix_nano:
            span.start_time_unix_nano = start_time_unix_nano


def _root_span(resource_spans: list[ResourceSpans]) -> Span | None:
    """The single parentless span of a trace, or ``None`` when there is not exactly one."""
    roots = [span for span in _spans(resource_spans) if not span.parent_span_id]
    return roots[0] if len(roots) == 1 else None


def _spans(resource_spans: list[ResourceSpans]) -> Iterator[Span]:
    """Yield every span, in resource and scope order."""
    for resource_span in resource_spans:
        for scope_span in resource_span.scope_spans:
            yield from scope_span.spans


def _set_attributes(span: Span, attributes: Mapping[str, AttributeValue]) -> None:
    """Write attributes onto one span, replacing keys it already carries."""
    existing = {attribute.key: attribute for attribute in span.attributes}
    for key, value in attributes.items():
        attribute = existing.get(key)
        if attribute is None:
            attribute = span.attributes.add()
            attribute.key = key
        if isinstance(value, bool):
            attribute.value.bool_value = value
        elif isinstance(value, int):
            attribute.value.int_value = value
        elif isinstance(value, float):
            attribute.value.double_value = value
        else:
            attribute.value.string_value = value


def root_span_id(resource_spans: list[ResourceSpans]) -> str | None:
    """Return the hex id of the trace's root span, or ``None`` when it has no single root.

    A trace with several roots has no one span that stands for the whole run, so there is
    nothing to attach a trial-level score to.
    """
    roots = [span for span in _spans(resource_spans) if not span.parent_span_id]
    if len(roots) != 1:
        return None
    span_id = bytes(roots[0].span_id)
    # Intake requires a present, non-zero id and drops the span otherwise, which would leave
    # anything scored against this id pointing at a span that was never stored.
    if not any(span_id):
        return None
    return span_id.hex()


# Byte lengths of the OTLP id fields, which decide whether a string is hex for that field.
