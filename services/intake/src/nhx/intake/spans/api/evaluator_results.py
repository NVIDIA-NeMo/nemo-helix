# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API for ClickHouse-backed evaluator_results."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from nhx.common.api.common import Page
from nhx.common.api.filter import FilterOperator
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.auth import AuthClient, get_auth_client
from nhx.intake.spans.api.dependencies import SpansServiceDep, require_workspace_access, validate_list_query_params
from nhx.intake.spans.api.evaluator_results_schemas import (
    EvaluatorResult,
    EvaluatorResultDataType,
    EvaluatorResultFilter,
    EvaluatorResultInput,
    EvaluatorResultSortField,
)
from nhx.intake.spans.api.query_filters import (
    filter_comparisons,
    require_datetime_value,
    require_enum_value,
    require_float_value,
    require_string_value,
)
from nhx.intake.spans.domain import EvaluatorResult as DomainEvaluatorResult
from nhx.intake.spans.domain import EvaluatorResultListFilter
from nhx.intake.spans.service import EvaluatorResultNotFoundError
from nhx.intake.spans.storage import stable_id

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Evaluator Results"


@router.post(
    "/v2/workspaces/{workspace}/evaluator-results",
    response_model=EvaluatorResult,
    response_model_exclude_none=True,
    tags=[API_TAG],
    status_code=status.HTTP_201_CREATED,
)
async def create_evaluator_result(
    workspace: str,
    body: EvaluatorResultInput,
    service: SpansServiceDep,
    auth_client: AuthClient = Depends(get_auth_client),
) -> EvaluatorResult:
    now = datetime.now(timezone.utc)
    # Identity-derived id: one result per (workspace, session, span, evaluator name). Re-POSTing
    # the same target hashes to the same id, so ReplacingMergeTree upserts (latest write wins)
    # rather than accumulating duplicate or stale rows.
    evaluator_result_id = stable_id(
        workspace,
        body.session_id,
        body.span_id,
        body.name,
        prefix="eval",
    )
    domain_result = DomainEvaluatorResult(
        evaluator_result_id=evaluator_result_id,
        span_id=body.span_id,
        session_id=body.session_id,
        workspace=workspace,
        name=body.name,
        value=body.value,
        string_value=body.string_value,
        data_type=body.data_type,
        comment=body.comment,
        created_by=_resolve_created_by(auth_client),
        created_at=now,
        ingested_at=now,
    )
    saved = await service.create_evaluator_result(domain_result)
    return EvaluatorResult.from_domain(saved)


@router.get(
    "/v2/workspaces/{workspace}/evaluator-results",
    response_model=Page[EvaluatorResult],
    response_model_exclude_none=True,
    tags=[API_TAG],
    openapi_extra=generate_openapi_extra_params(
        filter_schema=EvaluatorResultFilter,
        filter_description=(
            "Filter evaluator results by span_id, session_id, name, data_type, created_by, value range, "
            "and created_at range."
        ),
    ),
)
async def list_evaluator_results(
    workspace: str,
    request: Request,
    service: SpansServiceDep,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=10, ge=1, le=1000, description="Page size."),
    sort: EvaluatorResultSortField = Query(default=EvaluatorResultSortField.CREATED_AT_DESC),
    parsed: ParsedFilter = Depends(make_filter_dep(EvaluatorResultFilter)),
) -> Page[EvaluatorResult]:
    validate_list_query_params(request)
    result = await service.list_evaluator_results(
        filters=_evaluator_result_filter(workspace, parsed),
        page=page,
        page_size=page_size,
        sort=sort.value,
    )
    rows = [EvaluatorResult.from_domain(item) for item in result.data]
    return Page[EvaluatorResult](
        data=rows,
        pagination=result.pagination,
        sort=sort,
        filter=parsed.to_response(),
    )


@router.get(
    "/v2/workspaces/{workspace}/evaluator-results/{evaluator_result_id}",
    response_model=EvaluatorResult,
    response_model_exclude_none=True,
    tags=[API_TAG],
)
async def get_evaluator_result(
    workspace: str,
    evaluator_result_id: str,
    service: SpansServiceDep,
) -> EvaluatorResult:
    try:
        result = await service.get_evaluator_result(workspace=workspace, evaluator_result_id=evaluator_result_id)
    except EvaluatorResultNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluator result {workspace}/{evaluator_result_id} not found",
        ) from None
    return EvaluatorResult.from_domain(result)


@router.get(
    "/v2/workspaces/{workspace}/spans/{span_id}/evaluator-results",
    response_model=list[EvaluatorResult],
    response_model_exclude_none=True,
    tags=[API_TAG],
)
async def list_evaluator_results_for_span(
    workspace: str,
    span_id: str,
    service: SpansServiceDep,
) -> list[EvaluatorResult]:
    results = await service.list_evaluator_results_for_span(workspace=workspace, span_id=span_id)
    return [EvaluatorResult.from_domain(item) for item in results]


def _resolve_created_by(auth_client: AuthClient) -> str | None:
    if not getattr(auth_client, "auth_enabled", False):
        return None
    principal_id = getattr(getattr(auth_client, "principal", None), "id", None)
    return principal_id or None


def _evaluator_result_filter(workspace: str, parsed: ParsedFilter) -> EvaluatorResultListFilter:
    filters = EvaluatorResultListFilter(workspace=workspace)
    for comparison in filter_comparisons(parsed):
        if comparison.field == "span_id":
            filters.span_id = require_string_value(comparison)
        elif comparison.field == "session_id":
            filters.session_id = require_string_value(comparison)
        elif comparison.field == "name":
            filters.name = require_string_value(comparison)
        elif comparison.field == "data_type":
            filters.data_type = require_enum_value(comparison, EvaluatorResultDataType)
        elif comparison.field == "created_by":
            filters.created_by = require_string_value(comparison)
        elif comparison.field == "value" and comparison.operator == FilterOperator.GTE:
            filters.value_gte = require_float_value(comparison)
        elif comparison.field == "value" and comparison.operator == FilterOperator.LTE:
            filters.value_lte = require_float_value(comparison)
        elif comparison.field == "created_at" and comparison.operator == FilterOperator.GTE:
            filters.created_at_gte = require_datetime_value(comparison)
        elif comparison.field == "created_at" and comparison.operator == FilterOperator.LTE:
            filters.created_at_lte = require_datetime_value(comparison)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported evaluator_result filter: {comparison.field} {comparison.operator.value}",
            )
    return filters
