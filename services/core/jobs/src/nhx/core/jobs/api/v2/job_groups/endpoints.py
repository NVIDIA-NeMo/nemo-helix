# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accept JobGroup submissions and record controller actions."""

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.entities.base import ID_PATTERN
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.jobs.types import HelixJobListSortField
from nhx.common.api.common import Page, PaginationData
from nhx.common.auth import AuthClient, AuthContext, get_auth_client
from nhx.common.client_factory import get_async_nemo_client
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError, EntityValidationError
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.common.service.dependencies import get_nemo_client
from nhx.core.jobs.api.v2.jobs.endpoints import (
    configured_subprocess_translation_profiles,
    translate_cpu_container_steps_to_subprocess,
    validate_job_spec,
)
from nhx.core.jobs.api.v2.jobs.schemas import get_model_id
from nhx.core.jobs.app.dispatcher import JobEnvironmentValidationError, generate_job_name, validate_job_environment
from nhx.core.jobs.config import profiles
from nhx.core.jobs.job_groups import (
    CreateJobGroupRequest,
    HelixJobGroup,
    JobGroupControllerState,
    JobGroupMemberResponse,
    JobGroupMemberState,
    JobGroupResponse,
    JobGroupStatus,
    JobGroupSubmission,
    JobGroupTransitionError,
    UpdateJobGroupRequest,
    log_unreadable_job_group,
)

router = APIRouter(prefix="/v2/workspaces/{workspace}/job-groups")
PageNumber = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]


def get_job_group_store(
    client: Annotated[AsyncNemoClient, Depends(get_nemo_client)],
    auth: Annotated[AuthClient, Depends(get_auth_client)],
) -> EntityClient:
    principal = auth.principal.effective_principal
    caller = principal if principal.id and principal.caller_kind != "service_principal" else None
    jobs_client = get_async_nemo_client(
        as_service="jobs",
        internal=True,
        on_behalf_of=caller,
        http_client=client.http_client,
    )
    return EntityClient(AsyncEntitiesClient.from_client(jobs_client))


Store = Annotated[EntityClient, Depends(get_job_group_store)]


def require_permission(permission: str) -> Callable[[str, AuthClient], Awaitable[None]]:
    async def check(workspace: str, auth: Annotated[AuthClient, Depends(get_auth_client)]) -> None:
        if not auth.auth_enabled:
            return
        if auth.principal.on_behalf_of:
            allowed = await auth.check_permissions(workspace, subject=[permission])
        else:
            allowed = await auth.check_permissions(workspace, actor=[permission])
        if not allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail=f"Permission {permission} required")

    return check


def conflict(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_409_CONFLICT, detail=detail)


def require_open(group: HelixJobGroup) -> None:
    if group.controller_state.status == JobGroupStatus.DELETION_PENDING:
        raise conflict("JobGroup deletion is pending")


async def find_group(store: EntityClient, workspace: str, identifier: str) -> HelixJobGroup:
    try:
        group = await store.get_by_id(HelixJobGroup, identifier)
        if group.workspace == workspace:
            return group
    except EntityNotFoundError:
        pass
    try:
        group = await store.get(HelixJobGroup, name=identifier, workspace=workspace)
        if group.workspace != workspace:
            raise EntityNotFoundError(identifier)
        return group
    except EntityNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="JobGroup not found") from exc


async def mutate_group(
    store: EntityClient,
    workspace: str,
    identifier: str,
    change: Callable[[HelixJobGroup], bool],
) -> JobGroupResponse:
    group = await find_group(store, workspace, identifier)
    for _ in range(3):
        if not change(group):
            return JobGroupResponse.from_entity(group)
        try:
            updated = await store.update(group)
            return JobGroupResponse.from_entity(updated)
        except EntityConflictError:
            try:
                group = await store.get_by_id(HelixJobGroup, group.id)
            except EntityNotFoundError as exc:
                raise HTTPException(status.HTTP_404_NOT_FOUND, detail="JobGroup not found") from exc
        except EntityNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="JobGroup not found") from exc
    raise conflict("JobGroup changed concurrently")


@router.post("", status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_permission("jobs.create"))])
async def create_job_group(
    workspace: Annotated[str, Path(pattern=ID_PATTERN)],
    request: CreateJobGroupRequest,
    store: Store,
    auth: Annotated[AuthClient, Depends(get_auth_client)],
) -> JobGroupResponse:
    """Save a JobGroup submission for the controller."""
    template = translate_cpu_container_steps_to_subprocess(
        request.template, configured_subprocess_translation_profiles()
    )
    validate_job_spec(template, profiles)
    try:
        validate_job_environment(template, workspace)
    except JobEnvironmentValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    group = HelixJobGroup(
        workspace=workspace,
        name=request.name or generate_job_name(request.source, get_model_id("jobgroup")[-8:]),
        project=request.project,
        description=request.description,
        custom_fields=request.custom_fields,
        ownership=request.ownership,
        submission=JobGroupSubmission(
            source=request.source,
            template=template,
            spec=request.spec,
            members=request.members,
            max_concurrent_jobs=request.max_concurrent_jobs,
            fail_fast=request.fail_fast,
            default_timeout_seconds=request.default_timeout_seconds,
        ),
        controller_state=JobGroupControllerState(),
    ).with_auth_context(AuthContext.from_principal(auth.principal))
    try:
        return JobGroupResponse.from_entity(await store.create(group))
    except EntityConflictError as exc:
        raise conflict("A JobGroup with this name already exists") from exc
    except EntityValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("", dependencies=[Depends(require_permission("jobs.list"))])
async def list_job_groups(
    workspace: str,
    store: Store,
    page: PageNumber = 1,
    page_size: PageSize = 10,
    sort: HelixJobListSortField = HelixJobListSortField.CREATED_AT_DESC,
) -> Page[JobGroupResponse]:
    """List JobGroups and their current progress."""
    match sort:
        case HelixJobListSortField.SOURCE_ASC:
            sort_field = "submission.source"
        case HelixJobListSortField.SOURCE_DESC:
            sort_field = "-submission.source"
        case _:
            sort_field = sort.value
    groups = await store.list(
        HelixJobGroup,
        workspace=workspace,
        page=page,
        page_size=page_size,
        sort=sort_field,
        on_decode_error=log_unreadable_job_group,
    )
    return Page(
        data=[JobGroupResponse.from_entity(group) for group in groups.data],
        pagination=PaginationData(**groups.pagination.model_dump()),
        sort=sort,
    )


@router.get("/{identifier}", dependencies=[Depends(require_permission("jobs.read"))])
async def get_job_group(workspace: str, identifier: str, store: Store) -> JobGroupResponse:
    """Get a JobGroup by name or ID."""
    group = await find_group(store, workspace, identifier)
    require_open(group)
    return JobGroupResponse.from_entity(group)


def member_response(index: int, member: JobGroupMemberState) -> JobGroupMemberResponse:
    return JobGroupMemberResponse(
        index=index,
        job_id=member.job_id,
        attempt_id=member.attempt_id,
        status=HelixJobStatus(member.outcome.value) if member.outcome is not None else HelixJobStatus.PENDING,
        reason=member.reason,
    )


@router.get("/{identifier}/members", dependencies=[Depends(require_permission("jobs.read"))])
async def list_job_group_members(
    workspace: str,
    identifier: str,
    store: Store,
    page: PageNumber = 1,
    page_size: PageSize = 10,
) -> Page[JobGroupMemberResponse]:
    """List submitted members and their recorded group outcomes.

    Status stays pending until the group records an outcome. Fetch the member's
    Attempt for live execution status when an Attempt ID is available.
    """
    group = await find_group(store, workspace, identifier)
    require_open(group)
    offset = (page - 1) * page_size
    total = len(group.submission.members)
    data = [
        member_response(index, group.member_state(index)) for index in range(offset, min(total, offset + page_size))
    ]
    return Page(
        data=data,
        pagination=PaginationData(
            page=page,
            page_size=page_size,
            current_page_size=len(data),
            total_pages=(total + page_size - 1) // page_size,
            total_results=total,
        ),
    )


@router.patch("/{identifier}", dependencies=[Depends(require_permission("jobs.update"))])
async def update_job_group(
    workspace: str, identifier: str, request: UpdateJobGroupRequest, store: Store
) -> JobGroupResponse:
    """Update JobGroup metadata."""

    def update(group: HelixJobGroup) -> bool:
        require_open(group)
        if "description" in request.model_fields_set:
            group.description = request.description
        if "custom_fields" in request.model_fields_set:
            group.custom_fields = request.custom_fields
        return True

    return await mutate_group(store, workspace, identifier, update)


@router.post(
    "/{identifier}/cancel",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_permission("jobs.cancel"))],
)
async def cancel_job_group(workspace: str, identifier: str, store: Store) -> JobGroupResponse:
    """Request cancellation of outstanding JobGroup work."""

    def cancel(group: HelixJobGroup) -> bool:
        require_open(group)
        try:
            return group.controller_state.cancel()
        except JobGroupTransitionError as exc:
            raise conflict(str(exc)) from exc

    return await mutate_group(store, workspace, identifier, cancel)


@router.post(
    "/{identifier}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_permission("jobs.update"))],
)
async def retry_job_group(workspace: str, identifier: str, store: Store) -> JobGroupResponse:
    """Request new work for unsuccessful members of a settled JobGroup."""

    def retry(group: HelixJobGroup) -> bool:
        try:
            return group.controller_state.retry_unsuccessful_members()
        except JobGroupTransitionError as exc:
            raise conflict(str(exc)) from exc

    return await mutate_group(store, workspace, identifier, retry)


@router.delete(
    "/{identifier}",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_permission("jobs.delete"))],
)
async def delete_job_group(workspace: str, identifier: str, store: Store) -> JobGroupResponse:
    """Request deletion of a settled JobGroup and its owned Jobs."""

    def delete(group: HelixJobGroup) -> bool:
        try:
            return group.controller_state.request_deletion()
        except JobGroupTransitionError as exc:
            raise conflict(str(exc)) from exc

    return await mutate_group(store, workspace, identifier, delete)
