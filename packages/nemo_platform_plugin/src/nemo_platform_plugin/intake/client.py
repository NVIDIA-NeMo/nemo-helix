# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed HTTP clients for the Intake APIs used by evaluator, Experimentalist, and Insights."""

from __future__ import annotations

from functools import cached_property

from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.client.method import method
from nemo_platform_plugin.client.response import AsyncNemoPaginatedResponse, NemoPaginatedResponse
from nemo_platform_plugin.client.types import OffsetPagination
from nemo_platform_plugin.intake import endpoints
from nemo_platform_plugin.intake.types import (
    Annotation,
    EvaluatorResult,
    ListAnnotationsQueryParams,
    ListSpanGroupsQueryParams,
    ListSpansQueryParams,
    Span,
    SpanGroup,
    SpanGroupsPage,
    SpanMode,
)
from nemo_platform_plugin.schema import PaginationData
from pydantic import JsonValue

FilterQueryParam = dict[str, JsonValue]
EvaluatorResults = list[EvaluatorResult]


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

    async def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: FilterQueryParam | None = None,
        mode: SpanMode | None = None,
    ) -> AsyncNemoPaginatedResponse[Span, OffsetPagination]:
        return await self._client.list_spans(
            workspace=workspace,
            query_params=_list_spans_params(page=page, page_size=page_size, sort=sort, filter=filter, mode=mode),
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

    def retrieve(self, annotation_id: str, *, workspace: str | None = None) -> Annotation:
        return self._client.get_annotation(annotation_id=annotation_id, workspace=workspace).data()


class _AsyncAnnotationsCompat:
    def __init__(self, client: "AsyncIntakeClient") -> None:
        self._client = client

    async def list(
        self,
        *,
        workspace: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        sort: str | None = None,
        filter: FilterQueryParam | None = None,
    ) -> AsyncNemoPaginatedResponse[Annotation, OffsetPagination]:
        return await self._client.list_annotations(
            workspace=workspace,
            query_params=_list_annotations_params(page=page, page_size=page_size, sort=sort, filter=filter),
        )

    async def retrieve(self, annotation_id: str, *, workspace: str | None = None) -> Annotation:
        return (await self._client.get_annotation(annotation_id=annotation_id, workspace=workspace)).data()


class IntakeClient(_IntakeMethods, NemoClient):
    """Sync client for the Intake API subset evaluator, Experimentalist, and Insights use."""

    @cached_property
    def spans(self) -> _SpansCompat:
        return _SpansCompat(self)

    @cached_property
    def annotations(self) -> _AnnotationsCompat:
        return _AnnotationsCompat(self)


class AsyncIntakeClient(_IntakeMethods, AsyncNemoClient):
    """Async client for the Intake API subset evaluator, Experimentalist, and Insights use."""

    @cached_property
    def spans(self) -> _AsyncSpansCompat:
        return _AsyncSpansCompat(self)

    @cached_property
    def annotations(self) -> _AsyncAnnotationsCompat:
        return _AsyncAnnotationsCompat(self)
