# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for ClickHouse-backed Intake trace summaries."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from nhx.common.entities.values import DatetimeFilter
from nhx.intake.spans.domain import (
    INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT,
    IntakeResponseMode,
    IntakeTrace,
    SpanStatus,
)
from nhx.intake.spans.ingest.evaluation_context import EvaluationContext
from nhx.intake.spans.storage import text_for_mode
from pydantic import BaseModel, Field


class TraceSortField(StrEnum):
    STARTED_AT_ASC = "started_at"
    STARTED_AT_DESC = "-started_at"


TraceMode = IntakeResponseMode


class TraceFilter(BaseModel):
    id: str | None = Field(default=None, description="Filter by canonical Intake trace id.")
    session_id: str | None = Field(default=None, description="Filter by session id.")
    status: SpanStatus | None = Field(default=None, description="Filter by root span status.")
    started_at: DatetimeFilter | None = Field(default=None, description="Filter by root span start timestamp.")
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
    agent_name: str | None = Field(default=None, description="Filter by root-span agent name.")


class Trace(BaseModel):
    id: str
    root_span_id: str | None = None
    session_id: str
    workspace: str
    name: str | None = None
    input: str | None = Field(
        default=None,
        description=(
            f"Root-span input text. Omitted in summary mode and truncated to {INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT} "
            "characters in preview mode."
        ),
    )
    output: str | None = Field(
        default=None,
        description=(
            f"Root-span output text. Omitted in summary mode and truncated to {INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT} "
            "characters in preview mode."
        ),
    )
    evaluation_context: EvaluationContext | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: SpanStatus
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = None
    cost_input_usd: float | None = None
    cost_output_usd: float | None = None
    models: list[str] | None = Field(
        default=None,
        description="Distinct models used across the trace's spans. Omitted in summary mode.",
    )
    providers: list[str] | None = Field(
        default=None,
        description="Distinct inference providers used across the trace's spans. Omitted in summary mode.",
    )
    span_count: int | None = Field(default=None, ge=0)
    error_count: int | None = Field(default=None, ge=0)

    @classmethod
    def from_domain(cls, trace: IntakeTrace, *, mode: TraceMode = "detailed") -> Self:
        return cls(
            id=trace.id,
            root_span_id=trace.root_span_id,
            session_id=trace.session_id,
            workspace=trace.workspace,
            name=trace.name,
            input=text_for_mode(trace.input, mode=mode),
            output=text_for_mode(trace.output, mode=mode),
            evaluation_context=_evaluation_context(trace),
            agent_name=trace.agent_name,
            agent_version=trace.agent_version,
            started_at=trace.started_at,
            ended_at=trace.ended_at,
            duration_ms=trace.duration_ms,
            status=trace.status,
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            cached_tokens=trace.cached_tokens,
            total_tokens=trace.total_tokens,
            cost_usd=trace.cost_usd,
            cost_input_usd=trace.cost_input_usd,
            cost_output_usd=trace.cost_output_usd,
            models=trace.models,
            providers=trace.providers,
            span_count=trace.span_count,
            error_count=trace.error_count,
        )


def _evaluation_context(trace: IntakeTrace) -> EvaluationContext | None:
    if trace.evaluation_name is None:
        return None
    return EvaluationContext(
        evaluation_name=trace.evaluation_name,
        test_case_name=trace.test_case_name,
    )
