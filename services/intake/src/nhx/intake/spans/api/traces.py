# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read API for ClickHouse-backed Intake trace summaries."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from nhx.common.api.common import Page
from nhx.common.api.filter import FilterOperator
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.intake.spans.api.dependencies import SpansServiceDep, require_workspace_access, validate_list_query_params
from nhx.intake.spans.api.query_filters import (
    filter_comparisons,
    require_datetime_value,
    require_enum_value,
    require_string_or_list_value,
    require_string_value,
)
from nhx.intake.spans.api.traces_schemas import Trace, TraceFilter, TraceMode, TraceSortField
from nhx.intake.spans.domain import SpanStatus, TraceListFilter
from nhx.intake.spans.service import TraceNotFoundError

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Traces"
TRACE_INDEX_FILTER_FIELDS = frozenset(
    {
        "evaluation_name",
        "test_case_name",
        "evaluation_id",
        "test_case_id",
        "agent_name",
    }
)
TRACE_INDEX_FILTER_ALIASES = {
    "evaluation_name": "evaluation_name",
    "evaluation_id": "evaluation_name",
    "test_case_name": "test_case_name",
    "test_case_id": "test_case_name",
    "agent_name": "agent_name",
}


@router.get(
    "/v2/workspaces/{workspace}/traces",
    response_model=Page[Trace],
    response_model_exclude_none=True,
    tags=[API_TAG],
    openapi_extra=generate_openapi_extra_params(
        filter_schema=TraceFilter,
        filter_description=(
            "Filter root-span-backed traces by id, session_id, root status, root span started_at, "
            "evaluation_name, test_case_name, and agent_name."
        ),
    ),
)
async def list_traces(
    workspace: str,
    request: Request,
    service: SpansServiceDep,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=10, ge=1, le=1000, description="Page size."),
    sort: TraceSortField = Query(default=TraceSortField.STARTED_AT_DESC),
    mode: TraceMode = Query(
        default="preview",
        description=(
            "Response mode. summary returns root-span fields without payloads or rollups; preview adds token, cost, "
            "and span-count rollups plus 300-character input/output previews; detailed returns rollups and full payloads."
        ),
    ),
    parsed: ParsedFilter = Depends(make_filter_dep(TraceFilter)),
) -> Page[Trace]:
    validate_list_query_params(request, additional_params={"mode"})
    filters = _trace_filter(workspace, parsed)
    result = await service.list_traces(
        filters=filters,
        page=page,
        page_size=page_size,
        sort=sort.value,
        mode=mode,
    )
    traces = [Trace.from_domain(trace, mode=mode) for trace in result.data]
    return Page[Trace](
        data=traces,
        pagination=result.pagination,
        sort=sort,
        filter=parsed.to_response(),
    )


@router.get(
    "/v2/workspaces/{workspace}/traces/{id}",
    response_model=Trace,
    response_model_exclude_none=True,
    tags=[API_TAG],
)
async def get_trace(
    workspace: str,
    id: str,
    service: SpansServiceDep,
    mode: TraceMode = Query(
        default="detailed",
        description=(
            "Response mode. summary returns root-span fields without payloads or rollups; preview adds token, cost, "
            "and span-count rollups plus 300-character input/output previews; detailed returns rollups and full payloads."
        ),
    ),
) -> Trace:
    try:
        trace = await service.get_trace(workspace=workspace, trace_id=id, mode=mode)
    except TraceNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Trace {workspace}/{id} not found")
    return Trace.from_domain(trace, mode=mode)


def _trace_filter(workspace: str, parsed: ParsedFilter) -> TraceListFilter:
    filters = TraceListFilter(workspace=workspace)
    for comparison in filter_comparisons(parsed):
        if comparison.field == "id":
            filters.trace_ids = require_string_or_list_value(comparison)
        elif comparison.field == "session_id":
            filters.session_id = require_string_value(comparison)
        elif comparison.field == "status":
            filters.status = require_enum_value(comparison, SpanStatus)
        elif comparison.field == "started_at" and comparison.operator == FilterOperator.GTE:
            filters.started_at_gte = require_datetime_value(comparison)
        elif comparison.field == "started_at" and comparison.operator == FilterOperator.LTE:
            filters.started_at_lte = require_datetime_value(comparison)
        elif comparison.field in TRACE_INDEX_FILTER_FIELDS:
            if comparison.operator != FilterOperator.EQ:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported trace filter: {comparison.field} {comparison.operator.value}",
                )
            _set_trace_index_filter(filters, comparison.field, require_string_value(comparison))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported trace filter: {comparison.field} {comparison.operator.value}",
            )
    return filters


def _set_trace_index_filter(filters: TraceListFilter, public_field: str, value: str) -> None:
    field = TRACE_INDEX_FILTER_ALIASES[public_field]
    current_value = getattr(filters, field)
    if current_value is not None and current_value != value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Conflicting trace filters for {field}",
        )
    setattr(filters, field, value)
