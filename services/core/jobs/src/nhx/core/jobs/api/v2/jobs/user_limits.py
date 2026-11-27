# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admin API for per-user concurrency limit overrides.

An override replaces the ``max_active_jobs_per_user`` config default for one user. The jobs
controller reads all overrides in bulk once per scheduling pass, so a change takes effect within
one pass.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nemo_helix_plugin.jobs.types import (
    CreateUserConcurrencyLimitRequest,
    UpdateUserConcurrencyLimitRequest,
    UserConcurrencyLimitResponse,
)
from nhx.common.api.common import Page, PaginationData
from nhx.common.entities import SYSTEM_WORKSPACE
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError
from nhx.core.jobs.api.dependencies import dep_system_entity_client
from nhx.core.jobs.entities import JobsUserConcurrencyLimit

router = APIRouter()


def _to_response(entity: JobsUserConcurrencyLimit) -> UserConcurrencyLimitResponse:
    return UserConcurrencyLimitResponse(
        user_id=entity.user_id,
        max_concurrent_jobs=entity.max_concurrent_jobs,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


async def _get_or_404(store: EntityClient, user_id: str) -> JobsUserConcurrencyLimit:
    try:
        return await store.get(
            JobsUserConcurrencyLimit,
            JobsUserConcurrencyLimit.name_for_user(user_id),
            workspace=SYSTEM_WORKSPACE,
        )
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No concurrency limit override for user '{user_id}'") from exc


@router.post(
    "/v2/user-concurrency-limits",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_409_CONFLICT: {"description": "An override already exists for this user"}},
)
async def create_user_concurrency_limit(
    request: CreateUserConcurrencyLimitRequest,
    store: EntityClient = Depends(dep_system_entity_client),
) -> UserConcurrencyLimitResponse:
    """Grant a user a non-default concurrency limit."""
    entity = JobsUserConcurrencyLimit(
        name=JobsUserConcurrencyLimit.name_for_user(request.user_id),
        workspace=SYSTEM_WORKSPACE,
        user_id=request.user_id,
        max_concurrent_jobs=request.max_concurrent_jobs,
    )
    try:
        created = await store.create(entity)
    except EntityConflictError as exc:
        raise HTTPException(
            status_code=409, detail=f"A concurrency limit override already exists for user '{request.user_id}'"
        ) from exc
    return _to_response(created)


@router.get("/v2/user-concurrency-limits", response_model=Page[UserConcurrencyLimitResponse])
async def list_user_concurrency_limits(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=1000)] = 100,
    store: EntityClient = Depends(dep_system_entity_client),
) -> Page[UserConcurrencyLimitResponse]:
    """List all per-user concurrency limit overrides."""
    result = await store.list(
        JobsUserConcurrencyLimit,
        workspace=SYSTEM_WORKSPACE,
        sort="created_at",
        page=page,
        page_size=page_size,
    )
    return Page(
        data=[_to_response(e) for e in result.data],
        pagination=PaginationData(**result.pagination.model_dump()),
    )


@router.get(
    "/v2/user-concurrency-limits/{user_id}",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No override for this user"}},
)
async def get_user_concurrency_limit(
    user_id: str,
    store: EntityClient = Depends(dep_system_entity_client),
) -> UserConcurrencyLimitResponse:
    """Get a user's concurrency limit override."""
    return _to_response(await _get_or_404(store, user_id))


@router.put(
    "/v2/user-concurrency-limits/{user_id}",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No override for this user"}},
)
async def update_user_concurrency_limit(
    user_id: str,
    request: UpdateUserConcurrencyLimitRequest,
    store: EntityClient = Depends(dep_system_entity_client),
) -> UserConcurrencyLimitResponse:
    """Change a user's concurrency limit override."""
    entity = await _get_or_404(store, user_id)
    entity.max_concurrent_jobs = request.max_concurrent_jobs
    try:
        updated = await store.update(entity)
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No concurrency limit override for user '{user_id}'") from exc
    except EntityConflictError as exc:
        raise HTTPException(status_code=409, detail="The override changed during the request; retry") from exc
    return _to_response(updated)


@router.delete(
    "/v2/user-concurrency-limits/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "No override for this user"}},
)
async def delete_user_concurrency_limit(
    user_id: str,
    store: EntityClient = Depends(dep_system_entity_client),
) -> None:
    """Remove a user's override so they fall back to the default limit."""
    entity = await _get_or_404(store, user_id)
    try:
        await store.delete_by_id(JobsUserConcurrencyLimit, entity.id)
    except EntityNotFoundError:
        # Deleted concurrently; the override is gone either way.
        return
