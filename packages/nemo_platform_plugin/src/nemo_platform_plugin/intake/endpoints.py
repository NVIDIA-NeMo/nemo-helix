# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Intake APIs used by evaluator and Experimentalist."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterable, Iterable

from nemo_platform_plugin.client.endpoint import delete, get, patch, post, put
from nemo_platform_plugin.client.types import Paginated, PreparedRequest
from nemo_platform_plugin.intake.types import (
    Annotation,
    AnnotationInput,
    AtifCreateRequest,
    ChatCompletionsIngestRequest,
    ChatCompletionsIngestResponse,
    DirectSpansIngestRequest,
    EvaluationCreateRequest,
    EvaluationPatchRequest,
    EvaluationResponse,
    EvaluatorResult,
    EvaluatorResultCreateRequest,
    ExperimentCreateRequest,
    ExperimentResponse,
    ExperimentUpdateRequest,
    IngestResponse,
    ListAnnotationsQueryParams,
    ListEvaluatorResultsQueryParams,
    ListExperimentsQueryParams,
    ListSpanGroupsQueryParams,
    ListSpansQueryParams,
    ListTracesQueryParams,
    RetrieveTraceQueryParams,
    Session,
    Span,
    SpanGroup,
    Trace,
    TraceMetrics,
    TraceMetricsQueryParams,
)

_INTAKE_BASE = "/apis/intake/v2/workspaces/{workspace}"


@post(f"{_INTAKE_BASE}/ingest/atif")
@abstractmethod
def create_atif(*, workspace: str | None = None, body: AtifCreateRequest) -> None: ...


def create_otlp_traces(
    *,
    workspace: str | None = None,
    content: bytes | Iterable[bytes] | AsyncIterable[bytes],
) -> PreparedRequest[IngestResponse]:
    return PreparedRequest(
        path_template=f"{_INTAKE_BASE}/ingest/otlp/v1/traces",
        path_params={} if workspace is None else {"workspace": workspace},
        method="POST",
        content=content,
        content_type="application/x-protobuf",
        response_type=IngestResponse,
    )


@get(f"{_INTAKE_BASE}/traces")
@abstractmethod
def list_traces(
    *,
    workspace: str | None = None,
    query_params: ListTracesQueryParams | None = None,
) -> Paginated[Trace]: ...


@get(f"{_INTAKE_BASE}/traces/{{id}}")
@abstractmethod
def get_trace(
    *,
    workspace: str | None = None,
    id: str,
    query_params: RetrieveTraceQueryParams | None = None,
) -> Trace: ...


@get(f"{_INTAKE_BASE}/spans")
@abstractmethod
def list_spans(
    *,
    workspace: str | None = None,
    query_params: ListSpansQueryParams | None = None,
) -> Paginated[Span]: ...


@get(f"{_INTAKE_BASE}/spans/groups")
@abstractmethod
def list_span_groups(
    *,
    workspace: str | None = None,
    query_params: ListSpanGroupsQueryParams | None = None,
) -> Paginated[SpanGroup]: ...


@post(f"{_INTAKE_BASE}/evaluations")
@abstractmethod
def create_evaluation(
    *,
    workspace: str | None = None,
    body: EvaluationCreateRequest,
) -> EvaluationResponse: ...


@post(f"{_INTAKE_BASE}/evaluator-results")
@abstractmethod
def create_evaluator_result(
    *,
    workspace: str | None = None,
    body: EvaluatorResultCreateRequest,
) -> EvaluatorResult: ...


@get(f"{_INTAKE_BASE}/evaluations/{{name}}")
@abstractmethod
def get_evaluation(*, workspace: str | None = None, name: str) -> EvaluationResponse: ...


@put(f"{_INTAKE_BASE}/evaluations/{{name}}")
@abstractmethod
def update_evaluation(
    *,
    workspace: str | None = None,
    name: str,
    body: EvaluationCreateRequest,
) -> EvaluationResponse: ...


@patch(f"{_INTAKE_BASE}/evaluations/{{name}}")
@abstractmethod
def patch_evaluation(
    *,
    workspace: str | None = None,
    name: str,
    body: EvaluationPatchRequest,
) -> EvaluationResponse: ...


@get(f"{_INTAKE_BASE}/evaluator-results")
@abstractmethod
def list_evaluator_results(
    *,
    workspace: str | None = None,
    query_params: ListEvaluatorResultsQueryParams | None = None,
) -> Paginated[EvaluatorResult]: ...


@get(f"{_INTAKE_BASE}/spans/{{span_id}}/evaluator-results")
@abstractmethod
def list_evaluator_results_for_span(*, workspace: str | None = None, span_id: str) -> list[EvaluatorResult]: ...


@get(f"{_INTAKE_BASE}/spans/{{span_id}}")
@abstractmethod
def get_span(*, workspace: str | None = None, span_id: str) -> Span: ...


@get(f"{_INTAKE_BASE}/annotations")
@abstractmethod
def list_annotations(
    *,
    workspace: str | None = None,
    query_params: ListAnnotationsQueryParams | None = None,
) -> Paginated[Annotation]: ...


@get(f"{_INTAKE_BASE}/annotations/{{annotation_id}}")
@abstractmethod
def get_annotation(*, workspace: str | None = None, annotation_id: str) -> Annotation: ...


@post(f"{_INTAKE_BASE}/ingest/chat-completions")
@abstractmethod
def create_chat_completion(
    *, workspace: str | None = None, body: ChatCompletionsIngestRequest
) -> ChatCompletionsIngestResponse: ...


@post(f"{_INTAKE_BASE}/ingest/spans")
@abstractmethod
def create_spans(*, workspace: str | None = None, body: DirectSpansIngestRequest) -> None: ...


@get(f"{_INTAKE_BASE}/traces/metrics")
@abstractmethod
def get_trace_metrics(
    *, workspace: str | None = None, query_params: TraceMetricsQueryParams | None = None
) -> TraceMetrics: ...


@get(f"{_INTAKE_BASE}/sessions/{{id}}")
@abstractmethod
def get_session(*, workspace: str | None = None, id: str) -> Session: ...


@post(f"{_INTAKE_BASE}/annotations")
@abstractmethod
def create_annotation(*, workspace: str | None = None, body: AnnotationInput) -> Annotation: ...


@delete(f"{_INTAKE_BASE}/annotations/{{annotation_id}}")
@abstractmethod
def delete_annotation(*, workspace: str | None = None, annotation_id: str) -> None: ...


@get(f"{_INTAKE_BASE}/evaluator-results/{{evaluator_result_id}}")
@abstractmethod
def get_evaluator_result(*, workspace: str | None = None, evaluator_result_id: str) -> EvaluatorResult: ...


@get(f"{_INTAKE_BASE}/experiments/{{name}}")
@abstractmethod
def get_experiment(*, workspace: str | None = None, name: str) -> ExperimentResponse: ...


def _get_experiment_on_conflict(
    body: ExperimentCreateRequest, workspace: str | None
) -> PreparedRequest[ExperimentResponse]:
    """Build the retrieve request replayed when ``create_experiment(exist_ok=True)`` 409s."""
    return get_experiment(name=body.name, workspace=workspace)


@post(f"{_INTAKE_BASE}/experiments", get_on_conflict=_get_experiment_on_conflict)
@abstractmethod
def create_experiment(
    *, workspace: str | None = None, body: ExperimentCreateRequest, exist_ok: bool = False
) -> ExperimentResponse: ...


@get(f"{_INTAKE_BASE}/experiments")
@abstractmethod
def list_experiments(
    *,
    workspace: str | None = None,
    query_params: ListExperimentsQueryParams | None = None,
) -> Paginated[ExperimentResponse]: ...


@put(f"{_INTAKE_BASE}/experiments/{{name}}")
@abstractmethod
def update_experiment(
    *, workspace: str | None = None, name: str, body: ExperimentUpdateRequest
) -> ExperimentResponse: ...


@delete(f"{_INTAKE_BASE}/experiments/{{name}}")
@abstractmethod
def delete_experiment(*, workspace: str | None = None, name: str) -> None: ...
