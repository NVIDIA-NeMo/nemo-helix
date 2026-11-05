# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP/HTTP trace ingest for ClickHouse-backed Intake spans."""

import json
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.intake.dependencies import IntakeConfigDep
from nhx.intake.spans.api.dependencies import DenormalizerDep, SpansServiceDep, require_workspace_access
from nhx.intake.spans.domain import (
    IntakeSpan,
    SpanKind,
    SpanStatus,
    TraceBatch,
)
from nhx.intake.spans.ingest.evaluation_context import EvaluationContext
from nhx.intake.spans.ingest.evaluation_context_validation import validate_evaluation_context
from nhx.intake.spans.span_attribute_catalog import SpanAttributeField, spec_for_field
from nhx.intake.spans.span_semantic_attributes import SpanSemanticAttributes
from nhx.intake.spans.storage import json_dumps_preserve, normalize_span_kind, stable_id, utc_now
from pydantic import BaseModel, Field

router = APIRouter(dependencies=[Depends(require_workspace_access)])
EntityClientDep = Annotated[EntityClient, Depends(get_entity_client)]
API_TAG = "Ingest"

# Bag key the evaluation id lands under on a built span (from the attribute catalog). The ingest loop
# reads it back off each span to learn which evaluations a batch touched, for the denormalizer.
_EVALUATION_NAME_BAG_KEY = spec_for_field(SpanAttributeField.EVALUATION_NAME).bag_key

# OTel GenAI operation names, used only when the span has no OpenInference kind. Names that are not
# listed stay UNKNOWN, so a non-root execute_tool still counts as a model call.
_SPAN_KIND_BY_GEN_AI_OPERATION = {
    "chat": SpanKind.LLM,
    "generate_content": SpanKind.LLM,
    "text_completion": SpanKind.LLM,
    "embeddings": SpanKind.EMBEDDING,
    "invoke_agent": SpanKind.AGENT,
}

# Relay 0.9.3 full exporter: crates/core/src/observability/otel.rs::scope_type_name.
_SPAN_KIND_BY_RELAY_SCOPE = {
    "agent": SpanKind.AGENT,
    "function": SpanKind.CHAIN,
    "tool": SpanKind.TOOL,
    "llm": SpanKind.LLM,
    "retriever": SpanKind.RETRIEVER,
    "embedder": SpanKind.EMBEDDING,
    "reranker": SpanKind.RERANKER,
    "guardrail": SpanKind.GUARDRAIL,
    "evaluator": SpanKind.EVALUATOR,
}
_RELAY_RESPONSE_KEY = "__nemo_relay_integrations_langchain_model_response"

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
    config: IntakeConfigDep,
    service: SpansServiceDep,
    entity_client: EntityClientDep,
    denormalizer: DenormalizerDep,
    content_length: int | None = Header(default=None),
) -> IngestResponse:
    content_type = request.headers.get("content-type", "")
    if "application/x-protobuf" not in content_type.lower():
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="OTLP trace ingest only accepts application/x-protobuf",
        )

    max_bytes = config.otlp_max_body_bytes
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

    rejected_names: set[str] = set()
    for evaluation_name in sorted(evaluation_names):
        try:
            await validate_evaluation_context(
                workspace=workspace,
                context=EvaluationContext(evaluation_name=evaluation_name),
                entity_client=entity_client,
            )
        except HTTPException as exc:
            rejected_names.add(evaluation_name)
            errors.append(f"evaluation {evaluation_name!r}: {exc.detail}")
    if rejected_names:
        # One entry per evaluation, not per span: thousands of spans can share one bad name.
        evaluation_names -= rejected_names
        spans = [span for span in spans if span.attributes_string.get(_EVALUATION_NAME_BAG_KEY) not in rejected_names]

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
    # Native captures remain usable for replay/debugging even when a catalog alias consumed a key.
    for key, value in source_attributes.items():
        if key.startswith("nemo_relay."):
            attribute_bags.put_unhandled_source_attribute(key, value)

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
    if kind is SpanKind.UNKNOWN:
        operation = _as_str(source_attributes.get("gen_ai.operation.name"))
        if operation is not None:
            kind = _SPAN_KIND_BY_GEN_AI_OPERATION.get(operation, SpanKind.UNKNOWN)
    if original_kind is None and "gen_ai.operation.name" not in source_attributes:
        kind = _SPAN_KIND_BY_RELAY_SCOPE.get(source_attributes.get("nemo_relay.scope_type"), kind)
    for field, value in _relay_semantic_fields(attributes, kind=kind, name=span.name).items():
        if attribute_bags.get_field(field) is None:
            attribute_bags.put_field(field, value)
    status_message = _as_str(getattr(span.status, "message", None))
    if status_message and not _as_str(source_attributes.get("exception.message")):
        attribute_bags.put_field(SpanAttributeField.ERROR_MESSAGE, status_message)
    if events:
        attribute_bags.put_json("otel.events", events)
    if any(value is not None for value in scope_data.values()):
        attribute_bags.put_json("otel.scope", scope_data)

    # Resource-level payloads describe the task boundary, not every child call.
    # At that boundary use the same span-over-resource layering as semantic fields;
    # a child's missing payload must not become a copy of the root request/answer.
    payload_attributes = source_attributes if not parent_span_id else attributes

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
        input=_input_payload(payload_attributes, events, kind=kind.value) or "",
        output=_output_payload(payload_attributes, events, kind=kind.value) or "",
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
    payload = _first_payload_value(attributes, keys=OTLP_INPUT_PAYLOAD_ATTRIBUTE_KEYS)
    return payload if payload is not None else _relay_payload(attributes, direction="input", kind=kind)


def _output_payload(attributes: dict[str, Any], events: list[dict[str, Any]], *, kind: str) -> str | None:
    if kind == "LLM":
        messages_payload = _message_payload(attributes, prefix="llm.output_messages")
        if messages_payload is not None:
            return messages_payload
    payload = _first_payload_value(attributes, keys=OTLP_OUTPUT_PAYLOAD_ATTRIBUTE_KEYS)
    return payload if payload is not None else _relay_payload(attributes, direction="output", kind=kind)


def _relay_value(attributes: dict[str, Any], prefix: str) -> Any:
    """Read Relay 0.9.3 full export's *top-level-only* JSON projection.

    Nested objects/arrays are JSON strings, not recursively dotted paths. Never
    interpret dots in user keys as ancestry or mutate the recorded attributes.
    """
    if prefix in attributes:
        return _relay_json(attributes[prefix])
    fields = {
        key.removeprefix(prefix + "."): _relay_json(value)
        for key, value in attributes.items()
        if key.startswith(prefix + ".")
    }
    return fields or None


def _relay_json(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def _relay_messages(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return None
    messages = []
    for item in value:
        if isinstance(item.get("data"), dict) and "type" in item:
            # LangChain messages_to_dict: preserve recorded content blocks,
            # tool calls, IDs, metadata and any application-specific fields.
            message = dict(item["data"])
            message_type = item["type"]
            if isinstance(message_type, str):
                role = {"human": "user", "ai": "assistant"}.get(message_type, message_type)
                message.setdefault("role", role)
            messages.append(message)
        else:
            messages.append(dict(item))
    return messages


def _relay_payload(attributes: dict[str, Any], *, direction: str, kind: str) -> str | None:
    if "nemo_relay.scope_type" not in attributes:
        return None
    prefix = "nemo_relay.start.input" if direction == "input" else "nemo_relay.end.output"
    value = _relay_value(attributes, prefix)
    if value is None:
        return None
    if kind == "LLM" and isinstance(value, dict):
        content = value.get("content", value) if direction == "input" else value.get(_RELAY_RESPONSE_KEY, value)
        if isinstance(content, dict):
            messages = _relay_messages(content.get("messages"))
            if messages is not None:
                return json_dumps_preserve({**content, "messages": messages})
    return value if isinstance(value, str) else json_dumps_preserve(value)


def _relay_semantic_fields(attributes: dict[str, Any], *, kind: SpanKind, name: str) -> dict[SpanAttributeField, Any]:
    """Only recorded native values; no provider, token-total or reasoning inference."""
    fields: dict[SpanAttributeField, Any] = {}
    scope = attributes.get("nemo_relay.scope_type")
    if scope == "agent" and kind is SpanKind.AGENT:
        fields[SpanAttributeField.AGENT_NAME] = name
    if scope == "tool" and kind is SpanKind.TOOL:
        fields[SpanAttributeField.TOOL_NAME] = name
    if scope != "llm" or kind is not SpanKind.LLM:
        return fields
    output = _relay_value(attributes, "nemo_relay.end.output")
    if not isinstance(output, dict):
        return fields
    response = output.get(_RELAY_RESPONSE_KEY)
    if not isinstance(response, dict):
        return fields
    messages = _relay_messages(response.get("messages"))
    if not messages:
        return fields
    # Matches Relay 0.9.3 LangChain _model_response_to_annotated selection.
    message = next((item for item in reversed(messages) if item.get("role") == "assistant"), messages[-1])
    metadata = message.get("response_metadata")
    if isinstance(metadata, dict):
        fields[SpanAttributeField.MODEL] = next(
            (
                metadata[key]
                for key in ("model_name", "model", "model_id")
                if isinstance(metadata.get(key), str) and metadata[key]
            ),
            None,
        )
        provider = metadata.get("model_provider")
        if isinstance(provider, str) and provider:
            fields[SpanAttributeField.PROVIDER] = provider
    usage = message.get("usage_metadata")
    if not isinstance(usage, dict):
        return fields
    for key, field in (
        ("input_tokens", SpanAttributeField.INPUT_TOKENS),
        ("output_tokens", SpanAttributeField.OUTPUT_TOKENS),
        ("total_tokens", SpanAttributeField.TOTAL_TOKENS),
    ):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            fields[field] = value
    for key, mappings in (
        (
            "input_token_details",
            (
                ("cache_read", SpanAttributeField.CACHED_TOKENS),
                ("cache_creation", SpanAttributeField.PROMPT_CACHE_WRITE_TOKENS),
                ("audio", SpanAttributeField.PROMPT_AUDIO_TOKENS),
            ),
        ),
        (
            "output_token_details",
            (
                ("reasoning", SpanAttributeField.COMPLETION_REASONING_TOKENS),
                ("audio", SpanAttributeField.COMPLETION_AUDIO_TOKENS),
            ),
        ),
    ):
        details = usage.get(key)
        if isinstance(details, dict):
            for detail, field in mappings:
                value = details.get(detail)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    fields[field] = value
    return fields


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
    if code == 0 and not status_value:
        relay_status = attributes.get("nemo_relay.end.metadata.otel.status_code")
        if relay_status == "ERROR":
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
