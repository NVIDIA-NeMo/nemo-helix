# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP/HTTP trace ingest for ClickHouse-backed Intake spans."""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from nmp.intake.config import IntakeConfig
from nmp.intake.spans.api.dependencies import DenormalizerDep, SpansServiceDep, require_workspace_access
from nmp.intake.spans.domain import (
    IntakeSpan,
    SpanStatus,
    TraceBatch,
)
from nmp.intake.spans.span_attribute_catalog import SpanAttributeField, spec_for_field
from nmp.intake.spans.span_semantic_attributes import SpanSemanticAttributes
from nmp.intake.spans.storage import json_dumps_preserve, normalize_span_kind, stable_id, utc_now
from pydantic import BaseModel, Field

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Ingest"

# Bag key the evaluation id lands under on a built span (from the attribute catalog). The ingest loop
# reads it back off each span to learn which evaluations a batch touched, for the denormalizer.
_EVALUATION_NAME_BAG_KEY = spec_for_field(SpanAttributeField.EVALUATION_NAME).bag_key

# Ordered by precedence. Keep direct input.value/output.value first so existing
# OpenInference/LangChain payloads continue to win over framework-specific fallbacks.
OTLP_INPUT_PAYLOAD_ATTRIBUTE_KEYS = (
    "input.value",
    # Pydantic AI v2+ follows the newer OTel GenAI semantic conventions and
    # emits model request content as JSON strings.
    "gen_ai.input.messages",
    # Tool spans use GenAI tool-call fields instead of input.value.
    "gen_ai.tool.call.arguments",
    "tool_arguments",
)

OTLP_OUTPUT_PAYLOAD_ATTRIBUTE_KEYS = (
    "output.value",
    "gen_ai.output.messages",
    # Pydantic AI agent run spans store the typed/validated result here.
    "final_result",
    # Current Pydantic AI tool spans may use either the OTel semantic convention
    # name or the older Pydantic-specific attribute.
    "gen_ai.tool.call.result",
    "tool_response",
)


class IngestResponse(BaseModel):
    errors: list[str] = Field(default_factory=list)


@router.post(
    "/v2/workspaces/{workspace}/ingest/otlp/v1/traces",
    response_model=IngestResponse,
    tags=[API_TAG],
    # The body is streamed off the raw Request, so FastAPI infers no schema for it and
    # generated clients get no body argument unless the spec declares one here.
    openapi_extra={
        "requestBody": {
            "content": {
                "application/x-protobuf": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                        "description": "Serialized OTLP ExportTraceServiceRequest protobuf",
                    }
                }
            },
            "required": True,
        }
    },
)
async def ingest_otlp_traces(
    workspace: str,
    request: Request,
    service: SpansServiceDep,
    denormalizer: DenormalizerDep,
    content_length: int | None = Header(default=None),
) -> IngestResponse:
    content_type = request.headers.get("content-type", "")
    if "application/x-protobuf" not in content_type.lower():
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="OTLP trace ingest only accepts application/x-protobuf",
        )

    max_bytes = _otlp_max_body_bytes(request)
    if content_length is not None and content_length > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"OTLP body of {content_length} bytes exceeds limit of {max_bytes} bytes",
        )

    body = await _read_limited_body(request, max_bytes=max_bytes)
    export_request = _parse_export_request(body)
    ingested_at = utc_now()
    spans: list[IntakeSpan] = []
    evaluation_names: set[str] = set()
    errors: list[str] = []

    for resource_spans in export_request.resource_spans:
        resource_attributes = _attributes_to_dict(resource_spans.resource.attributes)
        for scope_spans in resource_spans.scope_spans:
            scope = getattr(scope_spans, "scope", None)
            scope_data = {
                "name": getattr(scope, "name", None),
                "version": getattr(scope, "version", None),
            }
            for span in scope_spans.spans:
                try:
                    span_domain = _span_to_domain(
                        workspace=workspace,
                        span=span,
                        resource_attributes=resource_attributes,
                        scope_data=scope_data,
                        ingested_at=ingested_at,
                    )
                except (OverflowError, OSError, TypeError, ValueError) as exc:
                    span_hex = bytes(getattr(span, "span_id", b"")).hex() or "<missing>"
                    errors.append(f"span {span_hex}: {exc}")
                    continue
                spans.append(span_domain)
                evaluation_name = span_domain.attributes_string.get(_EVALUATION_NAME_BAG_KEY)
                if evaluation_name:
                    evaluation_names.add(evaluation_name)

    await service.ingest_batch(TraceBatch(spans=spans))
    # A single OTLP batch can carry spans for several evaluations (associated via the
    # nemo.evaluation.name attribute), so refresh the denormalized name facets for every one it touched.
    if denormalizer is not None:
        for evaluation_name in evaluation_names:
            denormalizer.mark_dirty(workspace=workspace, evaluation_name=evaluation_name)
    return IngestResponse(errors=errors)


def _parse_export_request(body: bytes) -> Any:
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

    request = trace_service_pb2.ExportTraceServiceRequest()
    try:
        request.ParseFromString(body)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid OTLP protobuf: {exc}") from exc
    return request


async def _read_limited_body(request: Request, *, max_bytes: int) -> bytes:
    size = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"OTLP body exceeds limit of {max_bytes} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _otlp_max_body_bytes(request: Request) -> int:
    """Read the configured OTLP body-size cap from the attached Intake service."""

    service = getattr(request.app.state, "intake_service", None) or getattr(request.app.state, "service", None)
    cfg: IntakeConfig | None = getattr(service, "service_config", None) if service is not None else None
    if cfg is None:
        cfg = IntakeConfig()
    return cfg.otlp_max_body_bytes


def _span_to_domain(
    *,
    workspace: str,
    span: Any,
    resource_attributes: dict[str, Any],
    scope_data: dict[str, Any],
    ingested_at: datetime,
) -> IntakeSpan:
    trace_id = _required_otlp_id(span.trace_id, field_name="trace_id")
    trace_id_hex = trace_id.hex()
    source_span_id = _required_otlp_id(span.span_id, field_name="span_id")
    source_span_id_hex = source_span_id.hex()

    attributes = _attributes_to_dict(span.attributes)
    events = [_event_to_dict(event) for event in span.events]
    normalized_attributes = SpanSemanticAttributes.from_source_attribute_layers(
        resource_attributes=resource_attributes,
        span_attributes=attributes,
    )
    semantic_attributes = normalized_attributes.semantic
    source_attributes = normalized_attributes.source_attributes

    attribute_bags = semantic_attributes.to_bags()
    attribute_bags.put_unhandled_source_attributes(source_attributes, consumed_keys=normalized_attributes.consumed_keys)

    session_id = _first_str(
        source_attributes.get("gen_ai.conversation.id"),
        source_attributes.get("session.id"),
    ) or stable_id(workspace, trace_id_hex, prefix="session")
    parent_span_id = bytes(span.parent_span_id)
    if parent_span_id and not any(parent_span_id):
        raise ValueError("parent_span_id must not be all zero")
    parent_source_id = parent_span_id.hex()
    original_kind = _as_str(source_attributes.get("openinference.span.kind"))
    kind = normalize_span_kind(original_kind)
    status_message = _as_str(getattr(span.status, "message", None))
    if status_message and "exception.message" not in attribute_bags.string:
        attribute_bags.put_field(SpanAttributeField.ERROR_MESSAGE, status_message)
    if events:
        attribute_bags.put_json("otel.events", events)
    if any(value is not None for value in scope_data.values()):
        attribute_bags.put_json("otel.scope", scope_data)

    # TODO: After the POC lands, switch non-openinference/non-gen_ai spans from UNKNOWN rows to hard drops.
    span_domain = IntakeSpan(
        session_id=session_id,
        workspace=workspace,
        trace_id=trace_id_hex,
        source_format="otel",
        external_span_id=source_span_id_hex,
        external_parent_span_id=parent_source_id,
        kind=kind,
        name=span.name or "",
        status=_span_status(span, attributes),
        start_time=_nanos_to_datetime(span.start_time_unix_nano) or ingested_at,
        end_time=_nanos_to_datetime(span.end_time_unix_nano),
        attributes_string=attribute_bags.string,
        attributes_number=attribute_bags.number,
        attributes_bool=attribute_bags.boolean,
        input=_input_payload(attributes, events, kind=kind.value) or "",
        output=_output_payload(attributes, events, kind=kind.value) or "",
        event_ts=ingested_at,
    )
    return span_domain


def _required_otlp_id(value: Any, *, field_name: str) -> bytes:
    value_bytes = bytes(value)
    if not value_bytes or not any(value_bytes):
        raise ValueError(f"{field_name} is required")
    return value_bytes


def _input_payload(attributes: dict[str, Any], events: list[dict[str, Any]], *, kind: str) -> str | None:
    if kind == "LLM":
        messages_payload = _message_payload(attributes, prefix="llm.input_messages")
        if messages_payload is not None:
            return messages_payload
    return _first_payload_value(
        attributes,
        keys=OTLP_INPUT_PAYLOAD_ATTRIBUTE_KEYS,
    )


def _output_payload(attributes: dict[str, Any], events: list[dict[str, Any]], *, kind: str) -> str | None:
    if kind == "LLM":
        messages_payload = _message_payload(attributes, prefix="llm.output_messages")
        if messages_payload is not None:
            return messages_payload
    return _first_payload_value(
        attributes,
        keys=OTLP_OUTPUT_PAYLOAD_ATTRIBUTE_KEYS,
    )


def _attributes_to_dict(attributes: Any) -> dict[str, Any]:
    return {item.key: _any_value_to_python(item.value) for item in attributes}


def _event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "name": event.name,
        "time_unix_nano": event.time_unix_nano,
        "attributes": _attributes_to_dict(event.attributes),
    }


def _any_value_to_python(value: Any) -> Any:
    field = value.WhichOneof("value")
    if field is None:
        return None
    if field == "string_value":
        return value.string_value
    if field == "bool_value":
        return value.bool_value
    if field == "int_value":
        return value.int_value
    if field == "double_value":
        return value.double_value
    if field == "bytes_value":
        return bytes(value.bytes_value).hex()
    if field == "array_value":
        return [_any_value_to_python(item) for item in value.array_value.values]
    if field == "kvlist_value":
        return _attributes_to_dict(value.kvlist_value.values)
    raise ValueError(f"Unsupported OTLP AnyValue field: {field}")


def _span_status(span: Any, attributes: dict[str, Any]) -> SpanStatus:
    status_value = str(attributes.get("status") or attributes.get("otel.status_code") or "").lower()
    if status_value in {"error", "status_code_error"}:
        return SpanStatus.ERROR
    if status_value in {"cancelled", "canceled"}:
        return SpanStatus.CANCELLED
    code = getattr(span.status, "code", 0)
    if code == 2:
        return SpanStatus.ERROR
    return SpanStatus.SUCCESS


def _payload_value(
    attributes: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    direct_key: str,
    event_markers: tuple[str, ...],
) -> str | None:
    direct = attributes.get(direct_key)
    if direct is not None:
        if isinstance(direct, str):
            return direct
        return json_dumps_preserve(direct)

    matched_events = [event for event in events if any(marker in event["name"].lower() for marker in event_markers)]
    if matched_events:
        return json_dumps_preserve(matched_events)
    return None


def _first_payload_value(
    attributes: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        payload = _payload_value(attributes, [], direct_key=key, event_markers=())
        if payload is not None:
            return payload
    return None


def _message_payload(attributes: dict[str, Any], *, prefix: str) -> str | None:
    messages_by_index: dict[int, dict[str, Any]] = {}
    tool_calls_by_message: dict[int, dict[int, dict[str, Any]]] = {}
    message_prefix = f"{prefix}."

    for key, value in attributes.items():
        if not key.startswith(message_prefix):
            continue
        index_text, separator, field = key.removeprefix(message_prefix).partition(".message.")
        if not separator or not index_text.isdigit():
            continue
        message_index = int(index_text)
        message = messages_by_index.setdefault(message_index, {})
        _set_message_field(
            message=message,
            tool_calls_by_index=tool_calls_by_message.setdefault(message_index, {}),
            field=field,
            value=value,
        )

    if not messages_by_index:
        return None

    messages: list[dict[str, Any]] = []
    for message_index in sorted(messages_by_index):
        message = messages_by_index[message_index]
        tool_calls = tool_calls_by_message.get(message_index, {})
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        messages.append(message)
    return json_dumps_preserve({"messages": messages})


def _set_message_field(
    *,
    message: dict[str, Any],
    tool_calls_by_index: dict[int, dict[str, Any]],
    field: str,
    value: Any,
) -> None:
    if field in {"role", "content", "name", "tool_call_id", "status"}:
        message[field] = value
        return

    tool_call_prefix = "tool_calls."
    if not field.startswith(tool_call_prefix):
        message[field] = value
        return

    index_text, separator, tool_call_field = field.removeprefix(tool_call_prefix).partition(".tool_call.")
    if not separator or not index_text.isdigit():
        message[field] = value
        return

    tool_call = tool_calls_by_index.setdefault(int(index_text), {})
    if tool_call_field == "id":
        tool_call["id"] = value
    elif tool_call_field == "type":
        tool_call["type"] = value
    elif tool_call_field == "function.name":
        tool_call.setdefault("function", {})["name"] = value
        tool_call.setdefault("type", "function")
    elif tool_call_field == "function.arguments":
        tool_call.setdefault("function", {})["arguments"] = value
        tool_call.setdefault("type", "function")
    else:
        tool_call[tool_call_field] = value


def _first_str(*values: Any) -> str | None:
    for value in values:
        str_value = _as_str(value)
        if str_value is not None:
            return str_value
    return None


def _nanos_to_datetime(value: int) -> datetime | None:
    if not value:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc)


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    string_value = str(value)
    if string_value == "":
        return None
    return string_value
