# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read API for time-bucketed Intake trace metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.intake.spans.api.dependencies import SpansServiceDep, require_workspace_access
from nhx.intake.spans.api.trace_metrics_schemas import (
    TraceMetricBucketParam,
    TraceMetricPointResponse,
    TraceMetrics,
)
from nhx.intake.spans.api.traces import _trace_filter
from nhx.intake.spans.api.traces_schemas import TraceFilter
from nhx.intake.spans.domain import TraceListFilter

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Traces"
_ALLOWED_QUERY_PARAMS = frozenset({"bucket", "timezone", "filter"})
_DEFAULT_WINDOW = timedelta(days=7)
_DETAIL_BODY = {
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {"detail": {"type": "string"}},
                "required": ["detail"],
            }
        }
    },
}


@router.get(
    "/v2/workspaces/{workspace}/traces/metrics",
    response_model=TraceMetrics,
    response_model_exclude_none=True,
    tags=[API_TAG],
    openapi_extra=generate_openapi_extra_params(
        filter_schema=TraceFilter,
        filter_description=(
            "Filter the traces the metrics are computed over. Accepts the same fields as the traces "
            "list, so agent_name scopes the rollup to one agent. Without a started_at "
            "lower bound the rollup covers the last 7 days."
        ),
    ),
    responses={
        400: {"description": "Unsupported query parameter or unknown timezone", **_DETAIL_BODY},
        503: {"description": "ClickHouse spans storage unavailable", **_DETAIL_BODY},
    },
)
async def get_trace_metrics(
    workspace: str,
    request: Request,
    service: SpansServiceDep,
    bucket: TraceMetricBucketParam = Query(
        default=TraceMetricBucketParam.DAY,
        description="Time bucket granularity. total collapses the filtered range into a single row.",
    ),
    timezone: str = Query(
        default="UTC",
        description="IANA timezone the buckets are aligned to, e.g. America/Los_Angeles.",
    ),
    parsed: ParsedFilter = Depends(make_filter_dep(TraceFilter)),
) -> TraceMetrics:
    _validate_query_params(request)
    _validate_timezone(timezone)
    filters = _default_started_at_window(_trace_filter(workspace, parsed))
    points = await service.trace_metrics(filters=filters, bucket=bucket.value, timezone_name=timezone)
    return TraceMetrics(
        bucket=bucket,
        timezone=timezone,
        data=[TraceMetricPointResponse.from_domain(point) for point in points],
    )


def _default_started_at_window(filters: TraceListFilter) -> TraceListFilter:
    """Bound an open-ended rollup so a missing filter cannot scan the whole workspace."""

    if filters.started_at_gte is None:
        window_end = filters.started_at_lte or datetime.now(UTC)
        filters.started_at_gte = window_end - _DEFAULT_WINDOW
    return filters


def _validate_query_params(request: Request) -> None:
    """Reject anything this endpoint does not implement.

    The shared list validator always permits page/page_size/sort, which this
    endpoint has none of; accepting them silently would imply the response is
    paginated when it returns every bucket in the filtered range.
    """
    unsupported = sorted(
        {key for key in request.query_params if key not in _ALLOWED_QUERY_PARAMS and not key.startswith("filter[")}
    )
    if unsupported:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported query parameter(s): {', '.join(unsupported)}",
        )


def _validate_timezone(timezone: str) -> None:
    # ClickHouse would reject an unknown zone with an opaque server error; fail here instead.
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown timezone: {timezone}",
        ) from None
