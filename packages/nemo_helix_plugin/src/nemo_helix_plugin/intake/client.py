# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Intake APIs used by evaluator and Insights."""

from __future__ import annotations

from collections.abc import (
    AsyncIterable,
    Iterable,
    Mapping,
    Sequence,
)
from functools import cached_property
from typing import Any

from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.method import method
from nemo_helix_plugin.client.response import AsyncNemoPaginatedCall, NemoPaginatedResponse
from nemo_helix_plugin.client.types import OffsetPagination
from nemo_helix_plugin.intake import endpoints
from nemo_helix_plugin.intake.types import (
    ANNOTATION_INPUT_ADAPTER,
    Annotation,
    AnnotationInput,
    AtifCreateRequest,
    ChatCompletionsIngestRequest,
    ChatCompletionsIngestResponse,
    DirectSpanInput,
    DirectSpansIngestRequest,
    EvaluationContextParam,
    EvaluatorResult,
    EvaluatorResultCreateRequest,
    IngestResponse,
    ListAnnotationsQueryParams,
    ListEvaluatorResultsQueryParams,
    ListSpanGroupsQueryParams,
    ListSpansQueryParams,
    ListTracesQueryParams,
    RetrieveTraceQueryParams,
    Session,
    Span,
    SpanGroup,
    SpanGroupsPage,
    SpanMode,
    Trace,
    TraceFilterParam,
    TraceMetricBucket,
    TraceMetrics,
    TraceMetricsQueryParams,
    TraceMode,
)
from nemo_helix_plugin.schema import PaginationData
from pydantic import JsonValue

FilterQueryParam = dict[str, JsonValue]
EvaluatorResults = list[EvaluatorResult]
DirectSpanInputLike = DirectSpanInput | Mapping[str, object]


class _IntakeMethods:
    create_atif = method(endpoints.create_atif)
    create_otlp_traces = method(endpoints.create_otlp_traces)
    list_traces = method(endpoints.list_traces)
    get_trace = method(endpoints.get_trace)
    list_spans = method(endpoints.list_spans)
    list_span_groups = method(endpoints.list_span_groups)
    create_experiment = method(endpoints.create_experiment)
    get_experiment = method(endpoints.get_experiment)
    update_experiment = method(endpoints.update_experiment)
    create_evaluation = method(endpoints.create_evaluation)
    create_evaluator_result = method(endpoints.create_evaluator_result)
    get_evaluation = method(endpoints.get_evaluation)
    update_evaluation = method(endpoints.update_evaluation)
    patch_evaluation = method(endpoints.patch_evaluation)
    list_evaluator_results = method(endpoints.list_evaluator_results)
    list_evaluator_results_for_span = method(endpoints.list_evaluator_results_for_span)
    get_span = method(endpoints.get_span)
    list_annotations = method(endpoints.list_annotations)
    get_annotation = method(endpoints.get_annotation)
    create_chat_completion = method(endpoints.create_chat_completion)
    create_spans = method(endpoints.create_spans)
    get_trace_metrics = method(endpoints.get_trace_metrics)
    get_session = method(endpoints.get_session)
    create_annotation = method(endpoints.create_annotation)
    delete_annotation = method(endpoints.delete_annotation)
    get_evaluator_result = method(endpoints.get_evaluator_result)
    list_experiments = method(endpoints.list_experiments)
    delete_experiment = method(endpoints.delete_experiment)


def _list_traces_params(
    *,
    page: int | None = None,
    page_size: int | None = None,
    sort: str | None = None,
    filter: TraceFilterParam | str | None = None,
    mode: TraceMode | None = None,
) -> ListTracesQueryParams | None:
    params: ListTracesQueryParams = {}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if sort is not None:
        params["sort"] = sort
    if filter is not None:
        params["filter"] = filter
    if mode is not None:
        params["mode"] = mode
    return params or None


def _trace_metrics_params(
    *,
    bucket: TraceMetricBucket | None = None,
    timezone: str | None = None,
    filter: TraceFilterParam | str | None = None,
) -> TraceMetricsQueryParams | None:
    params: TraceMetricsQueryParams = {}
    if bucket is not None:
        params["bucket"] = bucket
    if timezone is not None:
        params["timezone"] = timezone
    if filter is not None:
        params["filter"] = filter
    return params or None


def _list_evaluator_results_params(
    *,
    page: int | None = None,
    page_size: int | None = None,
    sort: str | None = None,
    filter: dict[str, Any] | str | None = None,
) -> ListEvaluatorResultsQueryParams | None:
    params: ListEvaluatorResultsQueryParams = {}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if sort is not None:
        params["sort"] = sort
    if filter is not None:
        params["filter"] = filter
    return params or None


def _list_spans_params(
    *,
    page: int | None = None,
    page_size: int | None = None,
    sort: str | None = None,
    filter: FilterQueryParam | None = None,
    mode: SpanMode | None = None,
) -> ListSpansQueryParams | None:
    params: ListSpansQueryParams = {}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if sort is not None:
        params["sort"] = sort
    if filter is not None:
        params["filter"] = filter
    if mode is not None:
        params["mode"] = mode
    return params or None


def _list_span_groups_params(
    *,
    by: str,
    page: int | None = None,
    page_size: int | None = None,
    sort: str | None = None,
    filter: FilterQueryParam | None = None,
) -> ListSpanGroupsQueryParams:
    params: ListSpanGroupsQueryParams = {"by": by}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if sort is not None:
        params["sort"] = sort
    if filter is not None:
        params["filter"] = filter
    return params


def _list_annotations_params(
    *,
    page: int | None = None,
    page_size: int | None = None,
    sort: str | None = None,
    filter: FilterQueryParam | None = None,
) -> ListAnnotationsQueryParams | None:
    params: ListAnnotationsQueryParams = {}
    if page is not None:
        params["page"] = page
    if page_size is not None:
        params["page_size"] = page_size
    if sort is not None:
        params["sort"] = sort
    if filter is not None:
        params["filter"] = filter
    return params or None


def _grouped_by_fields(by: str) -> list[str]:
    return [field.strip() for field in by.split(",") if field.strip()]


def _span_groups_page(
    *,
    items: list[SpanGroup],
    metadata: object,
    by: str,
    sort: str | None,
    filter: FilterQueryParam | None,
) -> SpanGroupsPage:
    return SpanGroupsPage(
        data=items,
        grouped_by=_grouped_by_fields(by),
        pagination=PaginationData.model_validate(metadata),
        sort=sort,
        filter=filter,
    )


def _annotation_body(
    *,
    session_id: str,
    kind: str,
    span_id: str | None,
    name: str | None,
    value: str | float | None,
    value_type: str | None,
    text: str | None,
    metadata: dict[str, Any] | None,
) -> AnnotationInput:
    payload: dict[str, Any] = {"session_id": session_id, "kind": kind}
    if span_id is not None:
        payload["span_id"] = span_id
    if name is not None:
        payload["name"] = name
    if value is not None:
        payload["value"] = value
    if value_type is not None:
        payload["value_type"] = value_type
    if text is not None:
        payload["text"] = text
    if metadata is not None:
        payload["metadata"] = metadata
    return ANNOTATION_INPUT_ADAPTER.validate_python(payload)


def _atif_request(
    *,
    schema_version: str,
    agent: Mapping[str, object],
    evaluation_context: EvaluationContextParam | None,
    session_id: str | None,
    trajectory_id: str | None,
    final_metrics: Mapping[str, object] | None,
    continued_trajectory_ref: str | None,
    notes: str | None,
    extra: Mapping[str, object] | None,
    steps: Sequence[Mapping[str, object]] | None,
    subagent_trajectories: Sequence[Mapping[str, object]] | None,
) -> AtifCreateRequest:
    payload: dict[str, object] = {"schema_version": schema_version, "agent": dict(agent)}
    if evaluation_context is not None:
        payload["evaluation_context"] = evaluation_context
    if session_id is not None:
        payload["session_id"] = session_id
    if trajectory_id is not None:
        payload["trajectory_id"] = trajectory_id
    if final_metrics is not None:
        payload["final_metrics"] = dict(final_metrics)
    if continued_trajectory_ref is not None:
        payload["continued_trajectory_ref"] = continued_trajectory_ref
    if notes is not None:
        payload["notes"] = notes
    if extra is not None:
        payload["extra"] = dict(extra)
    if steps is not None:
        payload["steps"] = [dict(step) for step in steps]
    if subagent_trajectories is not None:
        payload["subagent_trajectories"] = [dict(trajectory) for trajectory in subagent_trajectories]
    return AtifCreateRequest.model_validate(payload)


def _evaluator_result_request(
    *,
    span_id: str,
    session_id: str,
    name: str,
    data_type: str,
    value: float | None,
    string_value: str | None,
    comment: str | None,
) -> EvaluatorResultCreateRequest:
    payload: dict[str, object] = {
        "span_id": span_id,
        "session_id": session_id,
        "name": name,
        "data_type": data_type,
    }
    if value is not None:
        payload["value"] = value
    if string_value is not None:
        payload["string_value"] = string_value
    if comment is not None:
        payload["comment"] = comment
    return EvaluatorResultCreateRequest.model_validate(payload)


def _chat_completions_request(
    *,
    request: Mapping[str, object],
    response: Mapping[str, object],
    evaluation_context: EvaluationContextParam | None,
    session_id: str | None,
    trace_id: str | None,
    provider: str | None,
    cost_usd: float | None,
    cost_input_usd: float | None,
    cost_output_usd: float | None,
    cost_details: dict[str, float] | None,
) -> ChatCompletionsIngestRequest:
    payload: dict[str, object] = {"request": dict(request), "response": dict(response)}
    if evaluation_context is not None:
        payload["evaluation_context"] = evaluation_context
    if session_id is not None:
        payload["session_id"] = session_id
    if trace_id is not None:
        payload["trace_id"] = trace_id
    if provider is not None:
        payload["provider"] = provider
    if cost_usd is not None:
        payload["cost_usd"] = cost_usd
    if cost_input_usd is not None:
        payload["cost_input_usd"] = cost_input_usd
    if cost_output_usd is not None:
        payload["cost_output_usd"] = cost_output_usd
    if cost_details is not None:
        payload["cost_details"] = dict(cost_details)
    return ChatCompletionsIngestRequest.model_validate(payload)


async def _async_bytes_content(body: AsyncIterable[object]) -> AsyncIterable[bytes]:
    async for chunk in body:
        if not isinstance(chunk, bytes):
            raise TypeError("OTLP trace chunks must be bytes.")
        yield chunk


def _async_otlp_content(body: bytes | Iterable[bytes] | AsyncIterable[bytes]) -> bytes | AsyncIterable[bytes]:
    if isinstance(body, bytes):
        return body
    if isinstance(body, Iterable) and not isinstance(body, AsyncIterable):
        return b"".join(body)
    return _async_bytes_content(body)


class _TracesCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: TraceFilterParam | str | None = None,
        mode: TraceMode | None = None,
    ) -> NemoPaginatedResponse[Trace, OffsetPagination]:
        return self._client.list_traces(
            workspace=workspace,
            query_params=_list_traces_params(page=page, page_size=page_size, sort=sort, filter=filter, mode=mode),
        )

    def retrieve(self, id: str, *, workspace: str | None = None, mode: TraceMode | None = None) -> Trace:
        query_params: RetrieveTraceQueryParams | None = {"mode": mode} if mode is not None else None
        return self._client.get_trace(id=id, workspace=workspace, query_params=query_params).data()

    def get_metrics(
        self,
        *,
        workspace: str | None = None,
        bucket: TraceMetricBucket | None = None,
        timezone: str | None = None,
        filter: TraceFilterParam | str | None = None,
    ) -> TraceMetrics:
        return self._client.get_trace_metrics(
            workspace=workspace,
            query_params=_trace_metrics_params(bucket=bucket, timezone=timezone, filter=filter),
        ).data()


class _AsyncTracesCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: TraceFilterParam | str | None = None,
        mode: TraceMode | None = None,
    ) -> AsyncNemoPaginatedCall[Trace, OffsetPagination]:
        return AsyncNemoPaginatedCall(
            lambda: self._client.list_traces(
                workspace=workspace,
                query_params=_list_traces_params(page=page, page_size=page_size, sort=sort, filter=filter, mode=mode),
            )
        )

    async def retrieve(self, id: str, *, workspace: str | None = None, mode: TraceMode | None = None) -> Trace:
        query_params: RetrieveTraceQueryParams | None = {"mode": mode} if mode is not None else None
        return (await self._client.get_trace(id=id, workspace=workspace, query_params=query_params)).data()

    async def get_metrics(
        self,
        *,
        workspace: str | None = None,
        bucket: TraceMetricBucket | None = None,
        timezone: str | None = None,
        filter: TraceFilterParam | str | None = None,
    ) -> TraceMetrics:
        return (
            await self._client.get_trace_metrics(
                workspace=workspace,
                query_params=_trace_metrics_params(bucket=bucket, timezone=timezone, filter=filter),
            )
        ).data()


class _SessionsCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def retrieve(self, id: str, *, workspace: str | None = None) -> Session:
        return self._client.get_session(id=id, workspace=workspace).data()


class _AsyncSessionsCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    async def retrieve(self, id: str, *, workspace: str | None = None) -> Session:
        return (await self._client.get_session(id=id, workspace=workspace)).data()


class _EvaluatorResultsCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def create(
        self,
        *,
        span_id: str,
        session_id: str,
        name: str,
        data_type: str,
        workspace: str | None = None,
        value: float | None = None,
        string_value: str | None = None,
        comment: str | None = None,
    ) -> EvaluatorResult:
        body = _evaluator_result_request(
            span_id=span_id,
            session_id=session_id,
            name=name,
            data_type=data_type,
            value=value,
            string_value=string_value,
            comment=comment,
        )
        return self._client.create_evaluator_result(workspace=workspace, body=body).data()

    def retrieve(self, evaluator_result_id: str, *, workspace: str | None = None) -> EvaluatorResult:
        return self._client.get_evaluator_result(evaluator_result_id=evaluator_result_id, workspace=workspace).data()

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: dict[str, Any] | str | None = None,
    ) -> NemoPaginatedResponse[EvaluatorResult, OffsetPagination]:
        return self._client.list_evaluator_results(
            workspace=workspace,
            query_params=_list_evaluator_results_params(
                page=page,
                page_size=page_size,
                sort=sort,
                filter=filter,
            ),
        )


class _AsyncEvaluatorResultsCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    async def create(
        self,
        *,
        span_id: str,
        session_id: str,
        name: str,
        data_type: str,
        workspace: str | None = None,
        value: float | None = None,
        string_value: str | None = None,
        comment: str | None = None,
    ) -> EvaluatorResult:
        body = _evaluator_result_request(
            span_id=span_id,
            session_id=session_id,
            name=name,
            data_type=data_type,
            value=value,
            string_value=string_value,
            comment=comment,
        )
        return (await self._client.create_evaluator_result(workspace=workspace, body=body)).data()

    async def retrieve(self, evaluator_result_id: str, *, workspace: str | None = None) -> EvaluatorResult:
        return (
            await self._client.get_evaluator_result(evaluator_result_id=evaluator_result_id, workspace=workspace)
        ).data()

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: dict[str, Any] | str | None = None,
    ) -> AsyncNemoPaginatedCall[EvaluatorResult, OffsetPagination]:
        return AsyncNemoPaginatedCall(
            lambda: self._client.list_evaluator_results(
                workspace=workspace,
                query_params=_list_evaluator_results_params(
                    page=page,
                    page_size=page_size,
                    sort=sort,
                    filter=filter,
                ),
            )
        )


class _SpansCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    @cached_property
    def groups(self) -> "_SpanGroupsCompat":
        return _SpanGroupsCompat(self._client)

    @cached_property
    def evaluator_results(self) -> "_SpanEvaluatorResultsCompat":
        return _SpanEvaluatorResultsCompat(self._client)

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: FilterQueryParam | None = None,
        mode: SpanMode | None = None,
    ) -> NemoPaginatedResponse[Span, OffsetPagination]:
        return self._client.list_spans(
            workspace=workspace,
            query_params=_list_spans_params(page=page, page_size=page_size, sort=sort, filter=filter, mode=mode),
        )

    def retrieve(self, span_id: str, *, workspace: str | None = None) -> Span:
        return self._client.get_span(span_id=span_id, workspace=workspace).data()


class _AsyncSpansCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    @cached_property
    def groups(self) -> "_AsyncSpanGroupsCompat":
        return _AsyncSpanGroupsCompat(self._client)

    @cached_property
    def evaluator_results(self) -> "_AsyncSpanEvaluatorResultsCompat":
        return _AsyncSpanEvaluatorResultsCompat(self._client)

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: FilterQueryParam | None = None,
        mode: SpanMode | None = None,
    ) -> AsyncNemoPaginatedCall[Span, OffsetPagination]:
        return AsyncNemoPaginatedCall(
            lambda: self._client.list_spans(
                workspace=workspace,
                query_params=_list_spans_params(page=page, page_size=page_size, sort=sort, filter=filter, mode=mode),
            )
        )

    async def retrieve(self, span_id: str, *, workspace: str | None = None) -> Span:
        return (await self._client.get_span(span_id=span_id, workspace=workspace)).data()


class _SpanGroupsCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def list(
        self,
        *,
        workspace: str | None = None,
        by: str,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: FilterQueryParam | None = None,
    ) -> SpanGroupsPage:
        response = self._client.list_span_groups(
            workspace=workspace,
            query_params=_list_span_groups_params(by=by, page=page, page_size=page_size, sort=sort, filter=filter),
        )
        page_result = response.page()
        return _span_groups_page(
            items=page_result.items, metadata=page_result.metadata, by=by, sort=sort, filter=filter
        )


class _AsyncSpanGroupsCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    async def list(
        self,
        *,
        workspace: str | None = None,
        by: str,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: FilterQueryParam | None = None,
    ) -> SpanGroupsPage:
        response = await self._client.list_span_groups(
            workspace=workspace,
            query_params=_list_span_groups_params(by=by, page=page, page_size=page_size, sort=sort, filter=filter),
        )
        page_result = response.page()
        return _span_groups_page(
            items=page_result.items, metadata=page_result.metadata, by=by, sort=sort, filter=filter
        )


class _SpanEvaluatorResultsCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def list(self, span_id: str, *, workspace: str | None = None) -> EvaluatorResults:
        return self._client.list_evaluator_results_for_span(span_id=span_id, workspace=workspace).data()


class _AsyncSpanEvaluatorResultsCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    async def list(self, span_id: str, *, workspace: str | None = None) -> EvaluatorResults:
        return (await self._client.list_evaluator_results_for_span(span_id=span_id, workspace=workspace)).data()


class _AnnotationsCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: FilterQueryParam | None = None,
    ) -> NemoPaginatedResponse[Annotation, OffsetPagination]:
        return self._client.list_annotations(
            workspace=workspace,
            query_params=_list_annotations_params(page=page, page_size=page_size, sort=sort, filter=filter),
        )

    def create(
        self,
        *,
        session_id: str,
        kind: str,
        workspace: str | None = None,
        span_id: str | None = None,
        name: str | None = None,
        value: str | float | None = None,
        value_type: str | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Annotation:
        return self._client.create_annotation(
            workspace=workspace,
            body=_annotation_body(
                session_id=session_id,
                kind=kind,
                span_id=span_id,
                name=name,
                value=value,
                value_type=value_type,
                text=text,
                metadata=metadata,
            ),
        ).data()

    def retrieve(self, annotation_id: str, *, workspace: str | None = None) -> Annotation:
        return self._client.get_annotation(annotation_id=annotation_id, workspace=workspace).data()

    def delete(self, annotation_id: str, *, workspace: str | None = None) -> None:
        return self._client.delete_annotation(annotation_id=annotation_id, workspace=workspace).data()


class _AsyncAnnotationsCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: FilterQueryParam | None = None,
    ) -> AsyncNemoPaginatedCall[Annotation, OffsetPagination]:
        return AsyncNemoPaginatedCall(
            lambda: self._client.list_annotations(
                workspace=workspace,
                query_params=_list_annotations_params(page=page, page_size=page_size, sort=sort, filter=filter),
            )
        )

    async def create(
        self,
        *,
        session_id: str,
        kind: str,
        workspace: str | None = None,
        span_id: str | None = None,
        name: str | None = None,
        value: str | float | None = None,
        value_type: str | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Annotation:
        return (
            await self._client.create_annotation(
                workspace=workspace,
                body=_annotation_body(
                    session_id=session_id,
                    kind=kind,
                    span_id=span_id,
                    name=name,
                    value=value,
                    value_type=value_type,
                    text=text,
                    metadata=metadata,
                ),
            )
        ).data()

    async def retrieve(self, annotation_id: str, *, workspace: str | None = None) -> Annotation:
        return (await self._client.get_annotation(annotation_id=annotation_id, workspace=workspace)).data()

    async def delete(self, annotation_id: str, *, workspace: str | None = None) -> None:
        return (await self._client.delete_annotation(annotation_id=annotation_id, workspace=workspace)).data()


class _AtifCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def create(
        self,
        *,
        schema_version: str,
        agent: Mapping[str, object],
        workspace: str | None = None,
        evaluation_context: EvaluationContextParam | None = None,
        session_id: str | None = None,
        trajectory_id: str | None = None,
        final_metrics: Mapping[str, object] | None = None,
        continued_trajectory_ref: str | None = None,
        notes: str | None = None,
        extra: Mapping[str, object] | None = None,
        steps: Sequence[Mapping[str, object]] | None = None,
        subagent_trajectories: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        return self._client.create_atif(
            workspace=workspace,
            body=_atif_request(
                schema_version=schema_version,
                agent=agent,
                evaluation_context=evaluation_context,
                session_id=session_id,
                trajectory_id=trajectory_id,
                final_metrics=final_metrics,
                continued_trajectory_ref=continued_trajectory_ref,
                notes=notes,
                extra=extra,
                steps=steps,
                subagent_trajectories=subagent_trajectories,
            ),
        ).data()


class _AsyncAtifCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    async def create(
        self,
        *,
        schema_version: str,
        agent: Mapping[str, object],
        workspace: str | None = None,
        evaluation_context: EvaluationContextParam | None = None,
        session_id: str | None = None,
        trajectory_id: str | None = None,
        final_metrics: Mapping[str, object] | None = None,
        continued_trajectory_ref: str | None = None,
        notes: str | None = None,
        extra: Mapping[str, object] | None = None,
        steps: Sequence[Mapping[str, object]] | None = None,
        subagent_trajectories: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        return (
            await self._client.create_atif(
                workspace=workspace,
                body=_atif_request(
                    schema_version=schema_version,
                    agent=agent,
                    evaluation_context=evaluation_context,
                    session_id=session_id,
                    trajectory_id=trajectory_id,
                    final_metrics=final_metrics,
                    continued_trajectory_ref=continued_trajectory_ref,
                    notes=notes,
                    extra=extra,
                    steps=steps,
                    subagent_trajectories=subagent_trajectories,
                ),
            )
        ).data()


class _ChatCompletionsCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def create(
        self,
        *,
        request: Mapping[str, object],
        response: Mapping[str, object],
        workspace: str | None = None,
        evaluation_context: EvaluationContextParam | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        provider: str | None = None,
        cost_usd: float | None = None,
        cost_input_usd: float | None = None,
        cost_output_usd: float | None = None,
        cost_details: dict[str, float] | None = None,
    ) -> ChatCompletionsIngestResponse:
        body = _chat_completions_request(
            request=request,
            response=response,
            evaluation_context=evaluation_context,
            session_id=session_id,
            trace_id=trace_id,
            provider=provider,
            cost_usd=cost_usd,
            cost_input_usd=cost_input_usd,
            cost_output_usd=cost_output_usd,
            cost_details=cost_details,
        )
        return self._client.create_chat_completion(workspace=workspace, body=body).data()


class _AsyncChatCompletionsCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    async def create(
        self,
        *,
        request: Mapping[str, object],
        response: Mapping[str, object],
        workspace: str | None = None,
        evaluation_context: EvaluationContextParam | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        provider: str | None = None,
        cost_usd: float | None = None,
        cost_input_usd: float | None = None,
        cost_output_usd: float | None = None,
        cost_details: dict[str, float] | None = None,
    ) -> ChatCompletionsIngestResponse:
        body = _chat_completions_request(
            request=request,
            response=response,
            evaluation_context=evaluation_context,
            session_id=session_id,
            trace_id=trace_id,
            provider=provider,
            cost_usd=cost_usd,
            cost_input_usd=cost_input_usd,
            cost_output_usd=cost_output_usd,
            cost_details=cost_details,
        )
        return (await self._client.create_chat_completion(workspace=workspace, body=body)).data()


class _IngestSpansCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def create(self, *, source: str, spans: Sequence[DirectSpanInputLike], workspace: str | None = None) -> None:
        body = DirectSpansIngestRequest.model_validate({"source": source, "spans": list(spans)})
        return self._client.create_spans(workspace=workspace, body=body).data()


class _AsyncIngestSpansCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    async def create(self, *, source: str, spans: Sequence[DirectSpanInputLike], workspace: str | None = None) -> None:
        body = DirectSpansIngestRequest.model_validate({"source": source, "spans": list(spans)})
        return (await self._client.create_spans(workspace=workspace, body=body)).data()


class _OtlpTracesCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    def create(self, *, body: bytes | Iterable[bytes], workspace: str | None = None) -> IngestResponse:
        return self._client.create_otlp_traces(workspace=workspace, content=body).data()


class _AsyncOtlpTracesCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    async def create(
        self,
        *,
        body: bytes | Iterable[bytes] | AsyncIterable[bytes],
        workspace: str | None = None,
    ) -> IngestResponse:
        return (await self._client.create_otlp_traces(workspace=workspace, content=_async_otlp_content(body))).data()


class _OtlpV1Compat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    @cached_property
    def traces(self) -> _OtlpTracesCompat:
        return _OtlpTracesCompat(self._client)


class _AsyncOtlpV1Compat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    @cached_property
    def traces(self) -> _AsyncOtlpTracesCompat:
        return _AsyncOtlpTracesCompat(self._client)


class _OtlpCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    @cached_property
    def v1(self) -> _OtlpV1Compat:
        return _OtlpV1Compat(self._client)


class _AsyncOtlpCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    @cached_property
    def v1(self) -> _AsyncOtlpV1Compat:
        return _AsyncOtlpV1Compat(self._client)


class _IngestCompat:
    def __init__(self, client: "IntakeClient") -> None:
        self._client = client

    @cached_property
    def atif(self) -> _AtifCompat:
        return _AtifCompat(self._client)

    @cached_property
    def chat_completions(self) -> _ChatCompletionsCompat:
        return _ChatCompletionsCompat(self._client)

    @cached_property
    def spans(self) -> _IngestSpansCompat:
        return _IngestSpansCompat(self._client)

    @cached_property
    def otlp(self) -> _OtlpCompat:
        return _OtlpCompat(self._client)


class _AsyncIngestCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    @cached_property
    def atif(self) -> _AsyncAtifCompat:
        return _AsyncAtifCompat(self._client)

    @cached_property
    def chat_completions(self) -> _AsyncChatCompletionsCompat:
        return _AsyncChatCompletionsCompat(self._client)

    @cached_property
    def spans(self) -> _AsyncIngestSpansCompat:
        return _AsyncIngestSpansCompat(self._client)

    @cached_property
    def otlp(self) -> _AsyncOtlpCompat:
        return _AsyncOtlpCompat(self._client)


class IntakeClient(_IntakeMethods, NemoClient):
    """Sync client for the Intake API subset evaluator and Insights use."""

    @cached_property
    def evaluator_results(self) -> _EvaluatorResultsCompat:
        return _EvaluatorResultsCompat(self)

    @cached_property
    def ingest(self) -> _IngestCompat:
        return _IngestCompat(self)

    @cached_property
    def spans(self) -> _SpansCompat:
        return _SpansCompat(self)

    @cached_property
    def annotations(self) -> _AnnotationsCompat:
        return _AnnotationsCompat(self)

    @cached_property
    def sessions(self) -> _SessionsCompat:
        return _SessionsCompat(self)

    @cached_property
    def traces(self) -> _TracesCompat:
        return _TracesCompat(self)


class AsyncIntakeClient(_IntakeMethods, AsyncNemoClient):
    """Async client for the Intake API subset evaluator and Insights use."""

    @cached_property
    def evaluator_results(self) -> _AsyncEvaluatorResultsCompat:
        return _AsyncEvaluatorResultsCompat(self)

    @cached_property
    def ingest(self) -> _AsyncIngestCompat:
        return _AsyncIngestCompat(self)

    @cached_property
    def spans(self) -> _AsyncSpansCompat:
        return _AsyncSpansCompat(self)

    @cached_property
    def annotations(self) -> _AsyncAnnotationsCompat:
        return _AsyncAnnotationsCompat(self)

    @cached_property
    def sessions(self) -> _AsyncSessionsCompat:
        return _AsyncSessionsCompat(self)

    @cached_property
    def traces(self) -> _AsyncTracesCompat:
        return _AsyncTracesCompat(self)
