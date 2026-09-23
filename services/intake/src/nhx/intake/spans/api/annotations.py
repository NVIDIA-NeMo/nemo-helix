# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API for post-hoc annotations on spans and sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from nhx.common.api.common import Page
from nhx.common.api.filter import FilterOperator
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.auth import AuthClient, get_auth_client
from nhx.intake.spans.api.annotations_schemas import (
    Annotation,
    AnnotationFilter,
    AnnotationInput,
    AnnotationSortField,
    annotation_from_domain,
    annotation_input_to_domain_fields,
)
from nhx.intake.spans.api.dependencies import SpansServiceDep, require_workspace_access, validate_list_query_params
from nhx.intake.spans.api.query_filters import (
    filter_comparisons,
    require_datetime_value,
    require_enum_value,
    require_float_value,
    require_string_value,
)
from nhx.intake.spans.domain import Annotation as DomainAnnotation
from nhx.intake.spans.domain import AnnotationKind, AnnotationListFilter
from nhx.intake.spans.service import AnnotationNotFoundError

router = APIRouter(dependencies=[Depends(require_workspace_access)])
API_TAG = "Annotations"


@router.post(
    "/v2/workspaces/{workspace}/annotations",
    response_model=Annotation,
    response_model_exclude_none=True,
    tags=[API_TAG],
    status_code=status.HTTP_201_CREATED,
)
async def create_annotation(
    workspace: str,
    body: AnnotationInput,
    service: SpansServiceDep,
    auth_client: AuthClient = Depends(get_auth_client),
) -> Annotation:
    now = datetime.now(timezone.utc)
    fields = annotation_input_to_domain_fields(body.root)
    domain_annotation = DomainAnnotation(
        annotation_id=f"ann-{uuid4().hex}",
        workspace=workspace,
        created_by=_resolve_created_by(auth_client),
        created_at=now,
        ingested_at=now,
        **fields,
    )
    saved = await service.create_annotation(domain_annotation)
    return annotation_from_domain(saved)


@router.get(
    "/v2/workspaces/{workspace}/annotations",
    response_model=Page[Annotation],
    response_model_exclude_none=True,
    tags=[API_TAG],
    openapi_extra=generate_openapi_extra_params(
        filter_schema=AnnotationFilter,
        filter_description=("Filter annotations by span_id, session_id, kind, name, created_by, and created_at range."),
    ),
)
async def list_annotations(
    workspace: str,
    request: Request,
    service: SpansServiceDep,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=10, ge=1, le=1000, description="Page size."),
    sort: AnnotationSortField = Query(default=AnnotationSortField.CREATED_AT_DESC),
    parsed: ParsedFilter = Depends(make_filter_dep(AnnotationFilter)),
) -> Page[Annotation]:
    validate_list_query_params(request)
    result = await service.list_annotations(
        filters=_annotation_filter(workspace, parsed),
        page=page,
        page_size=page_size,
        sort=sort.value,
    )
    rows = [annotation_from_domain(item) for item in result.data]
    return Page[Annotation](
        data=rows,
        pagination=result.pagination,
        sort=sort,
        filter=parsed.to_response(),
    )


@router.get(
    "/v2/workspaces/{workspace}/annotations/{annotation_id}",
    response_model=Annotation,
    response_model_exclude_none=True,
    tags=[API_TAG],
    responses={404: {"description": "Annotation not found"}},
)
async def get_annotation(
    workspace: str,
    annotation_id: str,
    service: SpansServiceDep,
) -> Annotation:
    try:
        annotation = await service.get_annotation(workspace=workspace, annotation_id=annotation_id)
    except AnnotationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation {workspace}/{annotation_id} not found",
        ) from None
    return annotation_from_domain(annotation)


@router.delete(
    "/v2/workspaces/{workspace}/annotations/{annotation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=[API_TAG],
    responses={404: {"description": "Annotation not found"}},
)
async def delete_annotation(
    workspace: str,
    annotation_id: str,
    service: SpansServiceDep,
) -> None:
    try:
        await service.delete_annotation(workspace=workspace, annotation_id=annotation_id)
    except AnnotationNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation {workspace}/{annotation_id} not found",
        ) from None


def _resolve_created_by(auth_client: AuthClient) -> str | None:
    if not getattr(auth_client, "auth_enabled", False):
        return None
    principal_id = getattr(getattr(auth_client, "principal", None), "id", None)
    return principal_id or None


def _annotation_filter(workspace: str, parsed: ParsedFilter) -> AnnotationListFilter:
    filters = AnnotationListFilter(workspace=workspace)
    for comparison in filter_comparisons(parsed):
        if comparison.field == "span_id":
            filters.span_id = require_string_value(comparison)
        elif comparison.field == "session_id":
            filters.session_id = require_string_value(comparison)
        elif comparison.field == "kind":
            filters.kind = require_enum_value(comparison, AnnotationKind)
        elif comparison.field == "name":
            filters.name = require_string_value(comparison)
        elif comparison.field == "value_text":
            filters.value_text = require_string_value(comparison)
        elif comparison.field == "value_numeric" and comparison.operator == FilterOperator.GTE:
            filters.value_numeric_gte = require_float_value(comparison)
        elif comparison.field == "value_numeric" and comparison.operator == FilterOperator.LTE:
            filters.value_numeric_lte = require_float_value(comparison)
        elif comparison.field == "created_by":
            filters.created_by = require_string_value(comparison)
        elif comparison.field == "created_at" and comparison.operator == FilterOperator.GTE:
            filters.created_at_gte = require_datetime_value(comparison)
        elif comparison.field == "created_at" and comparison.operator == FilterOperator.LTE:
            filters.created_at_lte = require_datetime_value(comparison)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported annotation filter: {comparison.field} {comparison.operator.value}",
            )
    return filters
