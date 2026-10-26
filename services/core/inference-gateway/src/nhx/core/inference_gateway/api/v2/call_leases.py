# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call lease endpoints: the platform service that creates a sandbox leases its address to the sandbox's trial.

POST  /v2/workspaces/{workspace}/call-leases
GET   /v2/workspaces/{workspace}/call-leases
GET   /v2/workspaces/{workspace}/call-leases/{name}
PATCH /v2/workspaces/{workspace}/call-leases/{name}
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from nemo_helix_plugin.call_leases.types import CallLease, CreateCallLeaseRequest, EndCallLeaseRequest
from nhx.common.api.common import Page, PaginationData
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.entities import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.common.entities.values import DatetimeFilter, Filter, StringFilter
from nhx.common.service.dependencies import get_entity_client
from pydantic import Field

logger = logging.getLogger(__name__)

router = APIRouter()

EntityClientDep = Annotated[EntityClient, Depends(get_entity_client)]

_LEASE_FIELDS = set(CreateCallLeaseRequest.model_fields) - {"name"}


class CallLeaseFilter(Filter):
    """Filter for call lease list queries."""

    client_address: StringFilter | str | None = Field(None, description="Filter by leased address.")
    session_id: StringFilter | str | None = Field(None, description="Filter by trial session.")
    role: str | None = Field(None, description="Filter by sandbox role: agent or verifier.")
    created_at: DatetimeFilter | None = Field(None, description="Filter by creation date.")


def _not_found(workspace: str, name: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"Call lease '{name}' not found in workspace '{workspace}'."
    )


@router.post(
    "/v2/workspaces/{workspace}/call-leases",
    summary="Create Call Lease",
    response_description="Created call lease",
    operation_id="create_call_lease",
    status_code=status.HTTP_201_CREATED,
    response_model=CallLease,
    responses={
        200: {"description": "The same lease already exists; creating it again is a no-op."},
        409: {"description": "A different lease with that name already exists."},
    },
)
async def create_call_lease(
    workspace: str, body: CreateCallLeaseRequest, entity_client: EntityClientDep, response: Response
) -> CallLease:
    """Lease a sandbox's address to its trial. Repeating an identical create returns the existing lease."""
    lease = CallLease(workspace=workspace, **body.model_dump())
    try:
        return await entity_client.create(lease)
    except EntityConflictError:
        existing = await entity_client.get(CallLease, name=body.name, workspace=workspace)
        if existing.model_dump(include=_LEASE_FIELDS) != lease.model_dump(include=_LEASE_FIELDS):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A different call lease '{body.name}' already exists in workspace '{workspace}'.",
            ) from None
        response.status_code = status.HTTP_200_OK
        return existing


@router.patch(
    "/v2/workspaces/{workspace}/call-leases/{name}",
    summary="End Call Lease",
    response_description="Ended call lease",
    operation_id="end_call_lease",
    response_model=CallLease,
    responses={
        404: {"description": "Call lease not found."},
        409: {"description": "The lease already ended at a different time."},
        422: {"description": "ended_at is before the lease started."},
    },
)
async def end_call_lease(
    workspace: str, name: str, body: EndCallLeaseRequest, entity_client: EntityClientDep
) -> CallLease:
    """End a lease. Ending it again at the same time is a no-op."""
    try:
        lease = await entity_client.get(CallLease, name=name, workspace=workspace)
    except EntityNotFoundError as exc:
        raise _not_found(workspace, name) from exc
    if lease.ended_at is not None:
        if lease.ended_at != body.ended_at:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Call lease '{name}' already ended at {lease.ended_at.isoformat()}.",
            )
        return lease
    if body.ended_at < lease.started_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"ended_at {body.ended_at.isoformat()} is before the lease started at {lease.started_at.isoformat()}.",
        )
    try:
        return await entity_client.update(lease.model_copy(update={"ended_at": body.ended_at}))
    except EntityConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Concurrent modification - please retry."
        ) from exc


@router.get(
    "/v2/workspaces/{workspace}/call-leases/{name}",
    summary="Get Call Lease",
    response_description="Call lease",
    operation_id="get_call_lease",
    response_model=CallLease,
    responses={404: {"description": "Call lease not found."}},
)
async def get_call_lease(workspace: str, name: str, entity_client: EntityClientDep) -> CallLease:
    """Get a call lease by workspace and name."""
    try:
        return await entity_client.get(CallLease, name=name, workspace=workspace)
    except EntityNotFoundError as exc:
        raise _not_found(workspace, name) from exc


@router.get(
    "/v2/workspaces/{workspace}/call-leases",
    summary="List Call Leases",
    response_description="Paginated list of call leases",
    operation_id="list_call_leases",
    response_model=Page[CallLease],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=CallLeaseFilter,
        filter_description="Filter call leases by client_address, session_id, role, and created_at.",
    ),
)
async def list_call_leases(
    workspace: str,
    entity_client: EntityClientDep,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=200, description="Number of results per page."),
    sort: str = Query(default="-created_at", description="Sort field. Prefix with ``-`` for descending order."),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(CallLeaseFilter)),
) -> Page[CallLease]:
    """List call leases in a workspace; ``workspace=-`` lists across every workspace the caller can read."""
    result = await entity_client.list(
        CallLease,
        workspace=workspace,
        page=page,
        page_size=page_size,
        sort=sort,
        filter_operation=parsed_filter.operation,
    )
    pagination = PaginationData(
        page=result.pagination.page,
        page_size=result.pagination.page_size,
        current_page_size=len(result.data),
        total_pages=result.pagination.total_pages,
        total_results=result.pagination.total_results,
    )
    return Page(data=result.data, pagination=pagination, sort=sort, filter=parsed_filter.to_response())
