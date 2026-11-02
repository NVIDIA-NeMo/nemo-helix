# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job group API endpoints.

A job group is a first-class entity representing a set of related jobs
submitted and managed as one unit. Membership is the Entity Store ``parent``
relationship: each member job is parented to the group. The group's ``status``
is persisted on the group entity (set at create time and kept current by the
controller's group roll-up), so reads return the stored value without fanning
out over every member.
"""

import logging
import math

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from nhx.common.api.common import Page, PaginationData
from nhx.common.auth import AuthClient, AuthContext, get_auth_client
from nhx.common.entities.client import EntityConflictError, EntityValidationError
from nhx.common.jobs.exceptions import HelixJobCompilationError
from nhx.common.observability import scoped_app_ctx
from nhx.core.jobs.api.dependencies import dep_dispatcher
from nhx.core.jobs.api.v2.jobs.endpoints import validate_job_spec
from nhx.core.jobs.api.v2.jobs.schemas import (
    CreateJobGroupRequest,
    HelixJobListSortField,
    JobGroupResponse,
    UpdateJobGroupRequest,
)
from nhx.core.jobs.app.ctx import JobContext
from nhx.core.jobs.app.dispatcher import (
    JobAlreadyExistsError,
    JobDeletionConflictError,
    JobDispatcher,
    JobOperationConflictError,
)
from nhx.core.jobs.config import profiles

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/v2/workspaces/{workspace}/job-groups",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Successful Response"},
        status.HTTP_409_CONFLICT: {"description": "A job group with this name already exists"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid job group request"},
    },
)
async def create_job_group(
    workspace: str,
    request: CreateJobGroupRequest,
    auth_client: AuthClient = Depends(get_auth_client),
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> JobGroupResponse:
    """Create a job group and expand its members into parented jobs."""
    # Validate the shared template against the configured execution profiles once;
    # every member job is built from it, so an invalid template fails the whole request up front.
    validate_job_spec(request.template, profiles)
    try:
        return await dispatcher.create_job_group(
            request,
            workspace,
            auth_context=AuthContext.from_principal(auth_client.principal),
        )
    except JobAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except EntityConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (EntityValidationError, HelixJobCompilationError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get(
    "/v2/workspaces/{workspace}/job-groups",
    response_model=Page[JobGroupResponse],
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
)
async def list_job_groups(
    workspace: str,
    page: int = Query(default=1, description="Page number.", gt=0),
    page_size: int = Query(default=10, description="Page size.", gt=0),
    sort: HelixJobListSortField = Query(
        default=HelixJobListSortField.CREATED_AT_DESC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> Page[JobGroupResponse]:
    """List job groups with their aggregate derived status."""
    offset = (page - 1) * page_size
    groups, total_count = await dispatcher.list_job_groups(workspace, limit=page_size, offset=offset, sort=sort)
    return Page(
        data=groups,
        pagination=PaginationData(
            page=page,
            page_size=page_size,
            current_page_size=len(groups),
            total_pages=int(math.ceil(total_count / page_size)) if total_count > 0 else 1,
            total_results=total_count,
        ),
        sort=sort,
    )


@router.get(
    "/v2/workspaces/{workspace}/job-groups/{name}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job group not found"},
    },
)
async def get_job_group(
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> JobGroupResponse:
    """Get a job group by name or ID, including its aggregate status."""
    with scoped_app_ctx(JobContext(id=name)):
        group = await dispatcher.get_job_group(name, workspace)
        if not group:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job group '{name}' not found in workspace '{workspace}'.",
            )
        return group


@router.patch(
    "/v2/workspaces/{workspace}/job-groups/{name}",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job group not found"},
        status.HTTP_409_CONFLICT: {"description": "Version conflict"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid job group update"},
    },
)
async def update_job_group(
    name: str,
    request: UpdateJobGroupRequest,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> JobGroupResponse:
    """Update a job group's mutable metadata."""
    with scoped_app_ctx(JobContext(id=name)):
        try:
            group = await dispatcher.update_job_group(name, workspace, request)
        except EntityConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except EntityValidationError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        if group is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job group '{name}' not found in workspace '{workspace}'.",
            )
        return group


@router.delete(
    "/v2/workspaces/{workspace}/job-groups/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {"description": "Successful Response"},
        status.HTTP_404_NOT_FOUND: {"description": "Job group not found"},
        status.HTTP_409_CONFLICT: {
            "description": "A member job is not terminal; members must reach a terminal state before deletion."
        },
    },
)
async def delete_job_group(
    name: str,
    workspace: str,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> None:
    """Delete a job group and all of its member jobs."""
    with scoped_app_ctx(JobContext(id=name)):
        try:
            deleted = await dispatcher.delete_job_group(name, workspace)
        except JobDeletionConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job group '{name}' not found in workspace '{workspace}'.",
            )


@router.post(
    "/v2/workspaces/{workspace}/job-groups/{name}/cancel",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Cancellation requested; the group is now cancelling"},
        status.HTTP_404_NOT_FOUND: {"description": "Job group not found"},
        status.HTTP_409_CONFLICT: {"description": "The job group has already reached a terminal state"},
    },
)
async def cancel_job_group(
    name: str,
    workspace: str,
    background_tasks: BackgroundTasks,
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> JobGroupResponse:
    """Request cancellation of a job group.

    Cancelling a group of thousands of jobs is one operation for the caller, not a cancel
    per member: this records the group's ``cancelling`` intent in a single write and returns
    immediately. The per-member teardown -- cancelling every non-terminal member through the
    normal per-job cancel path -- then runs as a background task, off the caller's request
    path. Once every member reaches a terminal state the group settles to ``cancelled`` via
    the member-status roll-up. The background teardown is best-effort and not durable: if it
    is lost before it runs, members finish naturally and the group settles to its real
    terminal status (not ``cancelled``), so re-requesting cancellation is how a caller drives
    it again -- there is no server-side reaper yet.
    """
    with scoped_app_ctx(JobContext(id=name)):
        try:
            group = await dispatcher.cancel_job_group(name, workspace)
        except JobOperationConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        if group is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job group '{name}' not found in workspace '{workspace}'.",
            )
        # Fan the per-member cancel out off the caller's request path: the intent write has
        # already returned the group, and cancel_group_members walks the members itself.
        background_tasks.add_task(dispatcher.cancel_group_members, group.id, workspace)
        return group


@router.post(
    "/v2/workspaces/{workspace}/job-groups/{name}/retry",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Retry requested; failed members are being rerun"},
        status.HTTP_404_NOT_FOUND: {"description": "Job group not found"},
        status.HTTP_409_CONFLICT: {"description": "The job group is cancelling and cannot be retried"},
    },
)
async def retry_job_group(
    name: str,
    workspace: str,
    background_tasks: BackgroundTasks,
    auth_client: AuthClient = Depends(get_auth_client),
    dispatcher: JobDispatcher = Depends(dep_dispatcher),
) -> JobGroupResponse:
    """Retry the non-succeeded members of a job group.

    Recovering a large run where some members did not succeed is one operation for the
    caller, not a rerun per member: this resolves the group and returns it immediately,
    then fans the per-member retry out as a background task off the caller's request path.
    Each ``error`` or ``cancelled`` member is rerun through the normal per-job rerun path
    (a fresh attempt, not a delete-and-recreate); only ``completed`` members are left
    untouched. Retry is a group-level operation, so a fail-fast group's collateral-
    cancelled members restart alongside the errored one. As retried members re-enter
    ``created`` the group re-opens to a non-terminal status.

    A group that is cancelling is refused (409): retrying members while a cancel sweep drives
    them to ``cancelled`` would fight that sweep. A group with no non-succeeded members is a
    no-op that returns the group unchanged, so a repeat or racing retry is harmless. The
    background retry is best-effort and not durable: if it is lost before it runs, re-
    requesting retry is how a caller drives it again.
    """
    with scoped_app_ctx(JobContext(id=name)):
        try:
            group = await dispatcher.retry_job_group(name, workspace)
        except JobOperationConflictError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        if group is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job group '{name}' not found in workspace '{workspace}'.",
            )
        # Fan the per-member rerun out off the caller's request path, carrying the caller's
        # auth so each rerun runs as the requester (job-level rerun honors auth_context).
        background_tasks.add_task(
            dispatcher.retry_group_members,
            group.id,
            workspace,
            AuthContext.from_principal(auth_client.principal),
        )
        return group
