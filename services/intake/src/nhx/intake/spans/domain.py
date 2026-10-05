# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain model for Intake trace spans."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

NEMO_STEP_ID_ATTRIBUTE = "nemo.step_id"


class SpanKind(StrEnum):
    LLM = "LLM"
    CHAIN = "CHAIN"
    TOOL = "TOOL"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    AGENT = "AGENT"
    RERANKER = "RERANKER"
    EVALUATOR = "EVALUATOR"
    GUARDRAIL = "GUARDRAIL"
    UNKNOWN = "UNKNOWN"


class SpanStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class SpanAttributeFilter(BaseModel):
    field: str
    operator: str
    value: Any


class IntakeSpan(BaseModel):
    workspace: str
    session_id: str
    trace_id: str
    source_format: str
    external_span_id: str
    start_time: datetime
    event_ts: datetime
    id: int | None = Field(default=None, ge=0, le=(1 << 64) - 1)
    external_parent_span_id: str = ""
    parent_id: int | None = None
    kind: SpanKind = SpanKind.UNKNOWN
    name: str = ""
    status: SpanStatus = SpanStatus.UNKNOWN
    end_time: datetime | None = None
    attributes_string: dict[str, str] = Field(default_factory=dict)
    attributes_number: dict[str, float] = Field(default_factory=dict)
    attributes_bool: dict[str, bool] = Field(default_factory=dict)
    input: str = ""
    output: str = ""
    is_deleted: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_parent(self) -> Self:
        if not self.external_span_id:
            raise ValueError("external_span_id must not be empty")
        if self.external_parent_span_id == self.external_span_id:
            raise ValueError("external_parent_span_id must differ from external_span_id")
        return self


class SpanListFilter(BaseModel):
    workspace: str
    session_id: str | None = None
    trace_id: str | None = None
    external_parent_span_id: str | None = None
    source_format: str | None = None
    kind: SpanKind | None = None
    status: SpanStatus | None = None
    started_at_gte: datetime | None = None
    started_at_lte: datetime | None = None
    attribute_filters: list[SpanAttributeFilter] = Field(default_factory=list)


class SpanGroup(BaseModel):
    group: dict[str, str]
    span_count: int = Field(ge=0)
    started_at: datetime


class TraceListFilter(BaseModel):
    workspace: str
    trace_ids: list[str] | None = None
    session_id: str | None = None
    source_format: str | None = None
    status: SpanStatus | None = None
    started_at_gte: datetime | None = None
    started_at_lte: datetime | None = None
    evaluation_name: str | None = None
    test_case_name: str | None = None
    agent_name: str | None = None


TraceMetricBucket = Literal["total", "hour", "day", "week", "month"]


class TokenRollup(BaseModel):
    """Distribution of a per-run token count across the runs in one bucket."""

    sum: int | None = Field(default=None, ge=0)
    mean: float | None = None
    p90: float | None = None
    p99: float | None = None


class CostRollup(BaseModel):
    """Distribution of per-run cost across the runs in one bucket."""

    sum: float | None = None
    mean: float | None = None
    p90: float | None = None
    p99: float | None = None


class LatencyRollup(BaseModel):
    """Distribution of end-to-end run duration across the runs in one bucket."""

    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None


class TraceMetricPoint(BaseModel):
    """One time bucket of agent-scoped trace metrics.

    ``bucket_start`` is None only for the ``total`` bucket, which collapses the
    whole filtered range into a single row.
    """

    bucket_start: datetime | None = None
    run_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    input_tokens: TokenRollup
    output_tokens: TokenRollup
    cached_tokens: TokenRollup
    total_tokens: TokenRollup
    cost_usd: CostRollup
    latency_ms: LatencyRollup


IntakeResponseMode = Literal["summary", "preview", "detailed"]
TraceMode = IntakeResponseMode
INTAKE_PREVIEW_PAYLOAD_CHAR_LIMIT = 300


class IntakeTrace(BaseModel):
    id: str
    root_span_id: str | None = None
    workspace: str
    session_id: str
    source_format: str
    name: str | None = None
    input: str | None = None
    output: str | None = None
    project: str | None = None
    evaluation_name: str | None = None
    test_case_name: str | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    ingested_at: datetime
    status: SpanStatus
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = None
    cost_input_usd: float | None = None
    cost_output_usd: float | None = None
    models: list[str] | None = None
    providers: list[str] | None = None
    span_count: int | None = Field(default=None, ge=0)
    error_count: int | None = Field(default=None, ge=0)


class IntakeSession(BaseModel):
    id: str
    workspace: str
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
    trace_count: int = Field(ge=0)
    span_count: int = Field(ge=0)


class EvaluatorResultDataType(StrEnum):
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    BOOLEAN = "BOOLEAN"
    TEXT = "TEXT"


class EvaluatorResult(BaseModel):
    evaluator_result_id: str
    span_id: str
    session_id: str
    workspace: str
    name: str
    value: float | None = None
    string_value: str | None = None
    data_type: EvaluatorResultDataType
    comment: str | None = None
    created_by: str | None = None
    created_at: datetime
    ingested_at: datetime


class EvaluatorResultListFilter(BaseModel):
    workspace: str
    span_id: str | None = None
    session_id: str | None = None
    name: str | None = None
    data_type: EvaluatorResultDataType | None = None
    created_by: str | None = None
    value_gte: float | None = None
    value_lte: float | None = None
    created_at_gte: datetime | None = None
    created_at_lte: datetime | None = None


class AnnotationKind(StrEnum):
    FEEDBACK = "feedback"
    LABEL = "label"
    NOTE = "note"
    METADATA = "metadata"


class Annotation(BaseModel):
    """Post-hoc human-supplied signal on a span or session.

    Distinct from `EvaluatorResult` (automated eval pipeline output). Annotations
    are added after ingestion by humans or downstream processes: feedback (positive/
    negative), labels (categorical or numeric), free-text notes, or structured
    metadata.
    """

    annotation_id: str
    workspace: str
    span_id: str | None = None
    session_id: str

    kind: AnnotationKind
    name: str | None = None
    value_text: str | None = None
    value_numeric: float | None = None
    text: str | None = None
    metadata: dict[str, Any] | None = None

    created_by: str | None = None
    created_at: datetime
    ingested_at: datetime


class AnnotationListFilter(BaseModel):
    workspace: str
    span_id: str | None = None
    session_id: str | None = None
    kind: AnnotationKind | None = None
    name: str | None = None
    value_text: str | None = None
    value_numeric_gte: float | None = None
    value_numeric_lte: float | None = None
    created_by: str | None = None
    created_at_gte: datetime | None = None
    created_at_lte: datetime | None = None


class TraceBatch(BaseModel):
    spans: list[IntakeSpan] = Field(default_factory=list)
    evaluator_results: list[EvaluatorResult] = Field(default_factory=list)
