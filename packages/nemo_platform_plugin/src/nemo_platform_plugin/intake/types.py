# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed wire shapes for the Intake APIs used by evaluator and Experimentalist."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, NotRequired, Required, TypedDict

from nemo_platform_plugin.schema import Page
from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, TypeAdapter, model_validator, with_config

EvaluatorResultDataType = Literal["NUMERIC", "BOOLEAN", "CATEGORICAL", "TEXT"]
TraceMode = Literal["summary", "preview", "detailed"]
TraceStatus = Literal["OK", "ERROR", "UNSET"] | str
SpanMode = Literal["summary", "preview", "detailed"]
# The server's SpanKind / SpanStatus enums; direct span ingest is validated against them.
SpanKind = (
    Literal["LLM", "CHAIN", "TOOL", "RETRIEVER", "EMBEDDING", "AGENT", "RERANKER", "EVALUATOR", "GUARDRAIL", "UNKNOWN"]
    | str
)

SpanStatus = Literal["success", "error", "cancelled", "unknown"] | str


@with_config(ConfigDict(extra="allow"))
class EvaluationContextParam(TypedDict, total=False):
    evaluation_name: str
    test_case_name: str
    evaluation_id: str
    test_case_id: str


@with_config(ConfigDict(extra="allow"))
class AtifAgentParam(TypedDict, total=False):
    name: Required[str]
    version: Required[str]
    model_name: NotRequired[str]
    tool_definitions: NotRequired[list[dict[str, Any]]]
    extra: NotRequired[dict[str, Any]]


class AtifFinalMetricsParam(TypedDict, total=False):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_cost_usd: float
    total_steps: int
    extra: dict[str, Any]


class AtifStepAgentParam(TypedDict, total=False):
    source: Required[Literal["agent"]]
    step_id: Required[int]
    timestamp: str | datetime
    message: str | list[dict[str, Any]]
    model_name: str
    reasoning_effort: str | float
    reasoning_content: str
    tool_calls: list[dict[str, Any]]
    metrics: dict[str, Any]
    observation: dict[str, Any]
    llm_call_count: int
    is_copied_context: bool
    extra: dict[str, Any]


AtifStepParam = dict[str, Any]


@with_config(ConfigDict(extra="allow"))
class AtifCreateParams(TypedDict, total=False):
    workspace: str
    evaluation_context: EvaluationContextParam
    schema_version: Required[str]
    session_id: str
    trajectory_id: str
    agent: Required[AtifAgentParam]
    final_metrics: AtifFinalMetricsParam
    continued_trajectory_ref: str
    notes: str
    extra: dict[str, Any]
    steps: list[AtifStepParam]
    subagent_trajectories: list[dict[str, Any]]


class AtifCreateRequest(RootModel[AtifCreateParams]):
    """Request body for POST /ingest/atif."""


class IngestResponse(BaseModel):
    """Response body for OTLP trace ingest."""

    errors: list[str] = Field(default_factory=list)


class EvaluatorResultCreateParams(TypedDict, total=False):
    workspace: str
    span_id: Required[str]
    session_id: Required[str]
    name: Required[str]
    value: float
    string_value: str
    data_type: Required[EvaluatorResultDataType]
    comment: str


class EvaluatorResultCreateRequest(BaseModel):
    """Request body for POST /evaluator-results."""

    model_config = ConfigDict(extra="forbid")

    span_id: str = Field(description="Target span id.")
    session_id: str = Field(description="Session id the target span belongs to.")
    name: str = Field(description="Evaluator / metric identity.")
    value: float | None = None
    string_value: str | None = None
    data_type: EvaluatorResultDataType
    comment: str | None = None

    @model_validator(mode="after")
    def _enforce_value_coherence(self) -> EvaluatorResultCreateRequest:
        if self.data_type in ("NUMERIC", "BOOLEAN") and self.value is None:
            raise ValueError(f"`value` is required when data_type is {self.data_type}.")
        if self.data_type in ("CATEGORICAL", "TEXT") and self.string_value is None:
            raise ValueError(f"`string_value` is required when data_type is {self.data_type}.")
        if self.data_type == "BOOLEAN" and self.value not in (0, 1, 0.0, 1.0):
            raise ValueError("`value` must be 0 or 1 when data_type is BOOLEAN.")
        return self


class EvaluatorResult(BaseModel):
    """Response model for evaluator-result reads."""

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


class EvaluatorAggregate(BaseModel):
    """Aggregate stats hydrated onto evaluation responses."""

    sum: float | None = None
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    count: int = 0


class EvaluationCreateRequest(BaseModel):
    """Create/full-update body for an Evaluation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    experiment_ids: list[str] = Field(default_factory=list)
    experiment_group_id: str | None = None
    dataset_name: str
    dataset_version: str | None = None
    source_link: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    description: str | None = None
    parent_evaluation_id: str | None = None
    status: str | None = None
    root_cause: str | None = None

    @model_validator(mode="after")
    def _resolve_group_membership(self) -> EvaluationCreateRequest:
        if not self.experiment_ids and self.experiment_group_id:
            self.experiment_ids = [self.experiment_group_id]
        if not self.experiment_ids:
            raise ValueError("Evaluation requires at least one experiment id.")
        self.experiment_ids = list(dict.fromkeys(self.experiment_ids))
        return self


class EvaluationPatchRequest(BaseModel):
    """Partial-update body for an Evaluation."""

    model_config = ConfigDict(extra="forbid")

    experiment_ids: list[str] | None = None
    source_link: str | None = None
    metadata: dict[str, str] | None = None
    description: str | None = None
    parent_evaluation_id: str | None = None
    status: str | None = None
    root_cause: str | None = None


class EvaluationResponse(BaseModel):
    """Evaluation as served by the Intake API."""

    id: str
    name: str
    workspace: str
    experiment_ids: list[str]
    dataset_name: str
    dataset_version: str | None = None
    source_link: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    description: str | None = None
    parent_evaluation_id: str | None = None
    status: str | None = None
    root_cause: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pinned_at: datetime | None = None
    evaluator_names: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(default_factory=list)
    agent_names: list[str] = Field(default_factory=list)
    agent_versions: list[str] = Field(default_factory=list)
    aggregate_scores: dict[str, EvaluatorAggregate] | None = None
    run_count: int = 0
    test_case_count: int = 0
    cost_usd: EvaluatorAggregate | None = None
    latency_ms: EvaluatorAggregate | None = None
    tokens: EvaluatorAggregate | None = None


class ExperimentCreateRequest(BaseModel):
    """Create body for an Experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    insight_id: str | None = None
    summary: str | None = None
    metadata: dict[str, str] | None = None
    default_sort: str = "-created_at"
    pareto: dict[str, Any] | None = None
    column_layout: dict[str, Any] | None = None
    is_favorite: bool = False
    show_evaluations_over_time: bool = False


class ExperimentUpdateRequest(ExperimentCreateRequest):
    """Full-update body for an Experiment."""

    baseline_evaluation_name: str | None = None


class ExperimentResponse(BaseModel):
    """Experiment as served by the Intake API."""

    id: str
    name: str
    workspace: str
    description: str | None = None
    insight_id: str | None = None
    summary: str | None = None
    metadata: dict[str, str] | None = None
    default_sort: str
    pareto: dict[str, Any] | None = None
    column_layout: dict[str, Any] | None = None
    is_favorite: bool = False
    show_evaluations_over_time: bool = False
    baseline_evaluation_name: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    evaluation_count: int = 0


class Trace(BaseModel):
    """Trace summary returned by Intake trace listing."""

    id: str
    root_span_id: str | None = None
    session_id: str
    workspace: str
    name: str | None = None
    input: str | None = None
    output: str | None = None
    evaluation_context: EvaluationContextParam | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: TraceStatus
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


class SpanEvaluationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_name: str | None = None
    test_case_name: str | None = None
    evaluation_id: str | None = None
    test_case_id: str | None = None


class Span(BaseModel):
    """Span row returned by the Intake spans API."""

    model_config = ConfigDict(extra="allow")

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
    error_message: str | None = None
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
    input: str | None = None
    output: str | None = None
    raw_attributes: str | None = None
    ingested_at: datetime


class SpanGroup(BaseModel):
    """Grouped span row returned by Intake."""

    group: dict[str, str]
    span_count: int = Field(ge=0)
    started_at: datetime


class SpanGroupsPage(Page[SpanGroup]):
    grouped_by: list[str] = Field(default_factory=list)


class Annotation(BaseModel):
    """Flattened read shape for Intake post-hoc annotations."""

    model_config = ConfigDict(extra="allow")

    annotation_id: str
    workspace: str
    span_id: str | None = None
    session_id: str
    kind: str
    name: str | None = None
    value: str | float | None = None
    value_type: str | None = None
    text: str | None = None
    metadata: dict[str, JsonValue] | None = None
    created_by: str | None = None
    created_at: datetime
    ingested_at: datetime


class TraceFilterParam(TypedDict, total=False):
    id: str | dict[str, list[str]]
    session_id: str
    status: str
    started_at: dict[str, str]
    evaluation_name: str
    test_case_name: str
    evaluation_id: str
    test_case_id: str
    agent_name: str


class ListTracesQueryParams(TypedDict, total=False):
    page: int
    page_size: int
    sort: str
    mode: TraceMode
    filter: TraceFilterParam | str


class RetrieveTraceQueryParams(TypedDict, total=False):
    mode: TraceMode


class SpanFilterParam(TypedDict, total=False):
    session_id: str
    trace_id: str
    parent_span_id: str
    project: str
    evaluation_name: str
    test_case_name: str
    evaluation_id: str
    test_case_id: str
    source: str
    kind: str
    status: str
    model: str
    tool_name: str
    provider: str
    agent_id: str
    agent_name: str
    started_at: dict[str, str]


class ListSpansQueryParams(TypedDict, total=False):
    page: int
    page_size: int
    sort: str
    mode: SpanMode
    filter: SpanFilterParam | dict[str, JsonValue] | str


class ListSpanGroupsQueryParams(TypedDict, total=False):
    by: Required[str]
    page: int
    page_size: int
    sort: str
    filter: SpanFilterParam | dict[str, JsonValue] | str


class AnnotationFilterParam(TypedDict, total=False):
    span_id: str
    session_id: str
    kind: str
    name: str
    value_text: str
    value_numeric: dict[str, float]
    created_by: str
    created_at: dict[str, str]


class ListAnnotationsQueryParams(TypedDict, total=False):
    page: int
    page_size: int
    sort: str
    filter: AnnotationFilterParam | dict[str, JsonValue] | str


class ListEvaluatorResultsQueryParams(TypedDict, total=False):
    page: int
    page_size: int
    sort: str
    filter: dict[str, Any] | str


TraceMetricBucket = Literal["total", "hour", "day", "week", "month"]

AnnotationKind = Literal["feedback", "note", "metadata", "label"]


class TraceMetricsQueryParams(TypedDict, total=False):
    bucket: TraceMetricBucket
    timezone: str
    filter: TraceFilterParam | str


class TokenRollup(BaseModel):
    sum: int | None = Field(default=None, ge=0)
    mean: float | None = None
    p90: float | None = None
    p99: float | None = None


class CostRollup(BaseModel):
    sum: float | None = None
    mean: float | None = None
    p90: float | None = None
    p99: float | None = None


class LatencyRollup(BaseModel):
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None


class TraceMetricPoint(BaseModel):
    """One time bucket of trace metric rollups."""

    bucket_start: datetime | None = Field(
        default=None, description="Start of the bucket in the requested timezone. Omitted when bucket=total."
    )
    run_count: int = Field(ge=0, description="Agent runs started in this bucket.")
    failed_run_count: int = Field(ge=0, description="Runs whose root span ended in error.")
    input_tokens: TokenRollup
    output_tokens: TokenRollup
    cached_tokens: TokenRollup
    total_tokens: TokenRollup
    cost_usd: CostRollup
    latency_ms: LatencyRollup


class TraceMetrics(BaseModel):
    """Response body for GET /traces/metrics."""

    bucket: TraceMetricBucket
    timezone: str
    data: list[TraceMetricPoint]


class Session(BaseModel):
    """Aggregate telemetry for one Intake session."""

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


class _AnnotationInputBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span_id: str | None = Field(
        default=None,
        description="Id of the span this annotation applies to. Omit to annotate the whole session.",
    )
    session_id: str = Field(description="Id of the session this annotation belongs to. Always required.")


class FeedbackAnnotationInput(_AnnotationInputBase):
    kind: Literal["feedback"]
    value: Literal["positive", "negative"]


class NoteAnnotationInput(_AnnotationInputBase):
    kind: Literal["note"]
    text: str = Field(min_length=1, max_length=10_000)


class MetadataAnnotationInput(_AnnotationInputBase):
    kind: Literal["metadata"]
    metadata: dict[str, Any] = Field(min_length=1)


class LabelAnnotationInput(_AnnotationInputBase):
    kind: Literal["label"]
    value_type: Literal["text", "numeric"]
    value: str | float
    name: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _validate(self) -> LabelAnnotationInput:
        if self.value_type == "numeric":
            if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
                raise ValueError("value_type=numeric requires a numeric `value`")
            if self.name is None:
                raise ValueError("value_type=numeric requires `name`")
        elif not isinstance(self.value, str):
            raise ValueError("value_type=text requires a string `value`")
        return self


AnnotationInput = Annotated[
    FeedbackAnnotationInput | NoteAnnotationInput | MetadataAnnotationInput | LabelAnnotationInput,
    Field(discriminator="kind"),
]

ANNOTATION_INPUT_ADAPTER: TypeAdapter[
    FeedbackAnnotationInput | NoteAnnotationInput | MetadataAnnotationInput | LabelAnnotationInput
] = TypeAdapter(AnnotationInput)


class ChatCompletionsIngestRequest(BaseModel):
    """Request body for POST /ingest/chat-completions."""

    model_config = ConfigDict(extra="forbid")

    evaluation_context: EvaluationContextParam | None = None
    request: dict[str, Any] = Field(description="Flexible captured chat-completions request.")
    response: dict[str, Any] = Field(description="Flexible captured chat-completions response.")
    session_id: str | None = None
    trace_id: str | None = None
    provider: str | None = None
    cost_usd: float | None = Field(default=None, ge=0)
    cost_input_usd: float | None = Field(default=None, ge=0)
    cost_output_usd: float | None = Field(default=None, ge=0)
    cost_details: dict[str, float] = Field(default_factory=dict)


class ChatCompletionsIngestResponse(BaseModel):
    session_id: str
    span_id: str


class DirectSpanInput(BaseModel):
    """One provider-neutral span supplied by a historical trace importer."""

    model_config = ConfigDict(extra="forbid")

    span_id: str
    trace_id: str
    session_id: str | None = None
    parent_span_id: str | None = None
    name: str = ""
    kind: SpanKind = "UNKNOWN"
    status: SpanStatus = "unknown"
    started_at: datetime
    ended_at: datetime | None = None
    input: JsonValue | None = None
    output: JsonValue | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)


class DirectSpansIngestRequest(BaseModel):
    """Request body for POST /ingest/spans."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Stable name for the source trace store, such as `langsmith` or `mlflow`.")
    spans: list[DirectSpanInput] = Field(min_length=1, max_length=1000)


class ExperimentFilterParam(TypedDict, total=False):
    name: str
    insight_id: str
    is_favorite: bool
    show_evaluations_over_time: bool
    baseline_evaluation_name: str
    is_deleted: bool
    metadata: dict[str, str]


class ListExperimentsQueryParams(TypedDict, total=False):
    page: int
    page_size: int
    sort: str
    filter: ExperimentFilterParam | str


ExperimentPage = Page[ExperimentResponse]

TracePage = Page[Trace]
SpanPage = Page[Span]
SpanGroupPage = Page[SpanGroup]
EvaluatorResultPage = Page[EvaluatorResult]
AnnotationPage = Page[Annotation]
