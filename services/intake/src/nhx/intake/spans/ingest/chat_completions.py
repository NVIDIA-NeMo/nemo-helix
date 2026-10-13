# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible chat-completions ingest.

Producers POST a captured chat-completion request + response. Intake stores
one IntakeSpan per response (one model invocation), preserving the raw
request and response payloads verbatim for downstream telemetry consumers.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Annotated, Any, Self

from fastapi import APIRouter, Depends, status
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.intake.spans.api.dependencies import DenormalizerDep, SpansServiceDep, require_workspace_access
from nhx.intake.spans.domain import IntakeSpan, SpanKind, SpanStatus, TraceBatch
from nhx.intake.spans.ingest.evaluation_context import EvaluationContext
from nhx.intake.spans.ingest.evaluation_context_validation import validate_evaluation_context
from nhx.intake.spans.span_attribute_bags import SpanAttributeBags
from nhx.intake.spans.span_semantic_attributes import SpanSemanticAttributes
from nhx.intake.spans.storage import json_dumps_preserve, stable_id, utc_now
from pydantic import BaseModel, ConfigDict, Field, model_validator

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Ingest"
EntityClientDep = Annotated[EntityClient, Depends(get_entity_client)]

SOURCE_FORMAT = "chat_completions"
NonNegativeFloat = Annotated[float, Field(ge=0)]


class ChatMessageRole(str, Enum):
    """Valid role values for captured chat-completions messages."""

    user = "user"
    system = "system"
    assistant = "assistant"
    developer = "developer"
    tool = "tool"
    function = "function"


class CapturedChatMessage(BaseModel):
    """A flexible message model that requires a valid role field but allows provider-specific fields."""

    model_config = ConfigDict(extra="allow")

    role: ChatMessageRole = Field(
        description="The role of the message sender.",
    )


class CapturedChatCompletionsRequest(BaseModel):
    """Flexible captured chat-completions request."""

    model_config = ConfigDict(extra="allow")

    messages: list[CapturedChatMessage] = Field(
        description="Messages comprising the conversation.",
    )
    model: str = Field(
        description="The model identifier used for this request.",
    )


class CapturedChatCompletionsResponse(BaseModel):
    """Flexible captured chat-completions response."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "oneOf": [
                {"required": ["choices"]},
                {"required": ["error"]},
            ],
        },
    )

    choices: list[dict[str, Any]] | None = Field(default=None)
    error: dict[str, Any] | None = Field(default=None)

    @model_validator(mode="after")
    def _require_choices_or_error(self) -> Self:
        if (self.choices is None) == (self.error is None):
            if self.choices is None:
                raise ValueError("response must include either `choices` or `error`")
            raise ValueError("response cannot include both `choices` and `error`")
        return self


class ChatCompletionsIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_context: EvaluationContext | None = None
    request: CapturedChatCompletionsRequest
    response: CapturedChatCompletionsResponse

    session_id: str | None = Field(
        default=None,
        description="Groups related chat-completions calls without forcing them into the same trace.",
    )
    trace_id: str | None = Field(
        default=None,
        description=(
            "Opt into joining an existing trace built via OTel or ATIF. This is not a grouping mechanism for "
            "chat-completions calls; use session_id to group related calls."
        ),
    )
    provider: str | None = None
    cost_usd: NonNegativeFloat | None = Field(
        default=None,
        description=(
            "Total estimated cost of this model call in USD. This matches ATIF step "
            "metrics; Intake stores it as semantic cost_total_usd on spans."
        ),
    )
    cost_input_usd: NonNegativeFloat | None = Field(
        default=None,
        description="Estimated input-token cost of this model call in USD.",
    )
    cost_output_usd: NonNegativeFloat | None = Field(
        default=None,
        description="Estimated output-token cost of this model call in USD.",
    )
    cost_details: dict[str, NonNegativeFloat] = Field(
        default_factory=dict,
        description="Additional estimated cost breakdown fields in USD.",
    )


class ChatCompletionsIngestResponse(BaseModel):
    session_id: str
    span_id: str


@router.post(
    "/v2/workspaces/{workspace}/ingest/chat-completions",
    response_model=ChatCompletionsIngestResponse,
    tags=[API_TAG],
    status_code=status.HTTP_201_CREATED,
)
async def ingest_chat_completion(
    workspace: str,
    body: ChatCompletionsIngestRequest,
    service: SpansServiceDep,
    entity_client: EntityClientDep,
    denormalizer: DenormalizerDep,
) -> ChatCompletionsIngestResponse:
    await validate_evaluation_context(
        workspace=workspace,
        context=body.evaluation_context,
        entity_client=entity_client,
    )
    ingested_at = utc_now()
    span = _chat_completion_to_span(workspace=workspace, body=body, ingested_at=ingested_at)
    await service.ingest_batch(TraceBatch(spans=[span]))
    context = body.evaluation_context
    if denormalizer is not None and context is not None and context.evaluation_name:
        denormalizer.mark_dirty(workspace=workspace, evaluation_name=context.evaluation_name)
    return ChatCompletionsIngestResponse(
        session_id=span.session_id,
        span_id=span.external_span_id,
    )


def _chat_completion_to_span(
    *, workspace: str, body: ChatCompletionsIngestRequest, ingested_at: datetime
) -> IntakeSpan:
    request = body.request.model_dump(mode="json")
    response = body.response.model_dump(mode="json")
    usage = _dict_or_empty(response.get("usage"))

    external_span_id = _external_span_id(response, request)
    trace_id = body.trace_id or external_span_id
    session_id = body.session_id or trace_id

    error = response.get("error") if isinstance(response.get("error"), dict) else None
    span_status = SpanStatus.ERROR if error is not None else SpanStatus.SUCCESS

    parsed_start = _datetime_from_unix_seconds(response.get("created"))
    start_time = parsed_start if parsed_start is not None and parsed_start <= ingested_at else ingested_at

    model = _as_str(response.get("model")) or _as_str(request.get("model")) or ""
    attribute_bags = _build_attribute_bags(body=body, request=request, response=response, usage=usage, error=error)

    return IntakeSpan(
        workspace=workspace,
        session_id=session_id,
        trace_id=trace_id,
        source_format=SOURCE_FORMAT,
        external_span_id=external_span_id,
        external_parent_span_id="",
        kind=SpanKind.LLM,
        name=model,
        status=span_status,
        start_time=start_time,
        end_time=ingested_at,
        attributes_string=attribute_bags.string,
        attributes_number=attribute_bags.number,
        attributes_bool=attribute_bags.boolean,
        input=json_dumps_preserve(request),
        output=json_dumps_preserve(response),
        event_ts=ingested_at,
    )


def _build_attribute_bags(
    *,
    body: ChatCompletionsIngestRequest,
    request: dict[str, Any],
    response: dict[str, Any],
    usage: dict[str, Any],
    error: dict[str, Any] | None,
) -> SpanAttributeBags:
    prompt_details = _dict_or_empty(usage.get("prompt_tokens_details"))
    completion_details = _dict_or_empty(usage.get("completion_tokens_details"))
    input_tokens = _clean_int(usage.get("prompt_tokens"))
    output_tokens = _clean_int(usage.get("completion_tokens"))
    total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None

    error_type: str | None = None
    error_message: str | None = None
    if error is not None:
        error_type = _as_str(error.get("type")) or _as_str(error.get("code"))
        error_message = _as_str(error.get("message"))

    evaluation_context = body.evaluation_context
    semantic = SpanSemanticAttributes(
        model=_as_str(response.get("model")) or _as_str(request.get("model")),
        provider=body.provider or _infer_provider(response),
        evaluation_name=evaluation_context.evaluation_name if evaluation_context is not None else None,
        test_case_name=evaluation_context.test_case_name if evaluation_context is not None else None,
        error_type=error_type,
        error_message=error_message,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=_clean_int(prompt_details.get("cached_tokens")),
        prompt_audio_tokens=_clean_int(prompt_details.get("audio_tokens")),
        completion_reasoning_tokens=_clean_int(completion_details.get("reasoning_tokens")),
        completion_audio_tokens=_clean_int(completion_details.get("audio_tokens")),
        cost_total_usd=_decimal_or_none(body.cost_usd),
        cost_input_usd=_decimal_or_none(body.cost_input_usd),
        cost_output_usd=_decimal_or_none(body.cost_output_usd),
    )
    attribute_bags = semantic.to_bags()
    for key, value in body.cost_details.items():
        bag_key = f"cost.{key}"
        if bag_key in attribute_bags.number:
            continue
        attribute_bags.put_unhandled_source_attribute(f"llm.cost.{key}", value)
    return attribute_bags


def _external_span_id(response: dict[str, Any], request: dict[str, Any]) -> str:
    """OpenAI chat-completion responses carry a stable `id` like 'chatcmpl-...'.

    When that id is missing (e.g. self-rolled providers, partial captures),
    fall back to a deterministic hash of the request+response so retries of
    the same payload still dedupe.
    """

    response_id = _as_str(response.get("id"))
    if response_id:
        return response_id
    return stable_id(
        json_dumps_preserve(request),
        json_dumps_preserve(response),
        prefix="chatcmpl-hash",
    )


def _infer_provider(response: dict[str, Any]) -> str | None:
    response_id = _as_str(response.get("id")) or ""
    if response_id.startswith("chatcmpl-"):
        return "openai"
    return None


def _datetime_from_unix_seconds(value: Any) -> datetime | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_int(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value if value >= 0 else None


def _decimal_or_none(value: float | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
