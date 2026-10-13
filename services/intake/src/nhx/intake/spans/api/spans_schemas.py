# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for ClickHouse-backed Intake spans."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Self

from nhx.common.api.common import Page
from nhx.common.entities.values import DatetimeFilter
from nhx.intake.spans.domain import (
    INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT,
    IntakeResponseMode,
    IntakeSpan,
    SpanKind,
    SpanStatus,
)
from nhx.intake.spans.domain import SpanGroup as IntakeSpanGroup
from nhx.intake.spans.ingest.evaluation_context import EvaluationContext
from nhx.intake.spans.span_attribute_bags import SpanAttributeBags
from nhx.intake.spans.span_semantic_attributes import SpanSemanticAttributes
from nhx.intake.spans.storage import text_for_mode
from pydantic import BaseModel, ConfigDict, Field


class SpanSortField(StrEnum):
    STARTED_AT_ASC = "started_at"
    STARTED_AT_DESC = "-started_at"


class SpanGroupSortField(StrEnum):
    SPAN_COUNT_ASC = "span_count"
    SPAN_COUNT_DESC = "-span_count"
    STARTED_AT_ASC = "started_at"
    STARTED_AT_DESC = "-started_at"


class SpanGroupBy(StrEnum):
    TRACE_ID = "trace_id"
    SESSION_ID = "session_id"


SpanMode = IntakeResponseMode
SPAN_SUMMARY_ERROR_MESSAGE_CHAR_LIMIT = 1000


class SpanFilter(BaseModel):
    session_id: str | None = Field(default=None, description="Filter by span session id.")
    trace_id: str | None = Field(default=None, description="Filter by canonical trace id.")
    project: str | None = Field(default=None, description="Filter by project name.")
    evaluation_name: str | None = Field(default=None, description="Filter by Evaluation name.")
    test_case_name: str | None = Field(default=None, description="Filter by test case name.")
    evaluation_id: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated alias for evaluation_name. Use evaluation_name instead.",
    )
    test_case_id: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated alias for test_case_name. Use test_case_name instead.",
    )
    source: str | None = Field(
        default=None, description="Filter by ingest source (e.g. 'otel', 'atif', 'chat_completions')."
    )
    kind: SpanKind | None = Field(default=None, description="Filter by normalized span kind.")
    status: SpanStatus | None = Field(default=None, description="Filter by normalized span status.")
    model: str | None = Field(default=None, description="Filter by model name.")
    tool_name: str | None = Field(default=None, description="Filter by tool name.")
    provider: str | None = Field(default=None, description="Filter by provider (e.g. 'openai', 'nim', 'anthropic').")
    agent_id: str | None = Field(default=None, description="Filter by agent identifier.")
    agent_name: str | None = Field(
        default=None, description="Filter by agent application name (e.g. 'claude-code', 'codex')."
    )
    parent_span_id: str | None = Field(
        default=None, description="Filter by parent span id. Use to fetch direct children of a span."
    )
    started_at: DatetimeFilter | None = Field(default=None, description="Filter by span start timestamp.")


class SpanEvaluationContext(EvaluationContext):
    # Read model for span evaluation context, aligned with the ingest EvaluationContext.
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_semantic_attributes(cls, attributes: SpanSemanticAttributes) -> Self | None:
        context = cls(
            evaluation_name=attributes.evaluation_name,
            test_case_name=attributes.test_case_name,
        )
        if not context.has_values():
            return None
        return context


class Span(BaseModel):
    span_id: str
    session_id: str
    workspace: str
    project: str | None = None
    evaluation_context: SpanEvaluationContext | None = None
    parent_span_id: str | None = None
    kind: SpanKind
    name: str | None = None
    source: str
    trace_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    status: SpanStatus
    error_type: str | None = None
    error_message: str | None = Field(
        default=None,
        description=(
            "Normalized error message. In summary and preview modes this is truncated to "
            f"{SPAN_SUMMARY_ERROR_MESSAGE_CHAR_LIMIT} characters."
        ),
    )
    provider: str | None = None
    model: str | None = None
    prompt_id: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_details: dict[str, int] = Field(default_factory=dict)
    cost_total_usd: float | None = None
    cost_input_usd: float | None = None
    cost_output_usd: float | None = None
    cost_details: dict[str, float] = Field(default_factory=dict)
    input: str | None = Field(
        default=None,
        description=(
            f"Span input text. Omitted in summary mode and truncated to {INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT} "
            "characters in preview mode."
        ),
    )
    output: str | None = Field(
        default=None,
        description=(
            f"Span output text. Omitted in summary mode and truncated to {INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT} "
            "characters in preview mode."
        ),
    )
    raw_attributes: str | None = None
    ingested_at: datetime

    @classmethod
    def from_domain(cls, span: IntakeSpan, *, mode: SpanMode = "detailed") -> Self:
        compact = mode != "detailed"
        attribute_bags = SpanAttributeBags.from_domain_maps(
            attributes_string=span.attributes_string,
            attributes_number=span.attributes_number,
            attributes_bool=span.attributes_bool,
        )
        semantic_attributes = SpanSemanticAttributes.from_bags(attribute_bags)
        cost_total = _float_or_none(semantic_attributes.cost_total_usd)
        return cls(
            span_id=span.external_span_id,
            session_id=span.session_id,
            workspace=span.workspace,
            project=semantic_attributes.project,
            evaluation_context=_evaluation_context(semantic_attributes),
            parent_span_id=span.external_parent_span_id or None,
            kind=span.kind,
            name=span.name or None,
            source=span.source_format,
            trace_id=span.trace_id,
            started_at=span.start_time,
            ended_at=span.end_time,
            status=span.status,
            error_type=semantic_attributes.error_type,
            error_message=_error_message_for_mode(semantic_attributes.error_message, compact=compact),
            provider=semantic_attributes.provider,
            model=semantic_attributes.model,
            prompt_id=semantic_attributes.prompt_id,
            agent_id=semantic_attributes.agent_id,
            agent_name=semantic_attributes.agent_name,
            tool_name=semantic_attributes.tool_name,
            input_tokens=semantic_attributes.input_tokens,
            output_tokens=semantic_attributes.output_tokens,
            cached_tokens=semantic_attributes.cached_tokens,
            total_tokens=semantic_attributes.total_tokens,
            usage_details=_usage_details(semantic_attributes),
            cost_total_usd=cost_total,
            cost_input_usd=_float_or_none(semantic_attributes.cost_input_usd),
            cost_output_usd=_float_or_none(semantic_attributes.cost_output_usd),
            cost_details=attribute_bags.cost_details(),
            input=text_for_mode(span.input or None, mode=mode),
            output=text_for_mode(span.output or None, mode=mode),
            raw_attributes=None if compact else attribute_bags.raw_attributes_json(),
            ingested_at=span.event_ts,
        )


class SpanGroup(BaseModel):
    group: dict[str, str] = Field(description="Group key values, keyed by the requested group-by fields.")
    span_count: int = Field(ge=0, description="Number of matching spans in this group.")
    started_at: datetime = Field(
        description=(
            "Start time of the earliest matching span in this group. Only matching spans count, so a filter "
            "narrows this to when the filtered work began rather than when the whole trace or session began."
        )
    )

    @classmethod
    def from_domain(cls, group: IntakeSpanGroup) -> Self:
        return cls(group=group.group, span_count=group.span_count, started_at=group.started_at)


class SpanGroupsPage(Page[SpanGroup]):
    grouped_by: list[SpanGroupBy] = Field(description="Span fields used to group the matching spans.")


def _evaluation_context(
    attributes: SpanSemanticAttributes,
) -> SpanEvaluationContext | None:
    return SpanEvaluationContext.from_semantic_attributes(attributes)


def _usage_details(attributes: SpanSemanticAttributes) -> dict[str, int]:
    details: dict[str, int] = {}
    if attributes.prompt_cache_write_tokens is not None:
        details["prompt_details.cache_write"] = attributes.prompt_cache_write_tokens
    if attributes.prompt_audio_tokens is not None:
        details["prompt_details.audio"] = attributes.prompt_audio_tokens
    if attributes.completion_reasoning_tokens is not None:
        details["completion_details.reasoning"] = attributes.completion_reasoning_tokens
    if attributes.completion_audio_tokens is not None:
        details["completion_details.audio"] = attributes.completion_audio_tokens
    return details


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str | int | float | Decimal):
        return float(value)
    raise TypeError(f"Expected float-compatible span value, got {type(value).__name__}")


def _error_message_for_mode(value: str | None, *, compact: bool) -> str | None:
    if value is None or not compact:
        return value
    return value[:SPAN_SUMMARY_ERROR_MESSAGE_CHAR_LIMIT]
