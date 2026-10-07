# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CRUD routes for stored tasksets under /apis/evaluator/v2/workspaces/{workspace}/tasksets."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status
from nemo_evaluator.api.dependencies import get_taskset_service
from nemo_evaluator.api.schemas import Revision, Taskset, TasksetFilter, TasksetInput, TasksetSort
from nemo_evaluator.api.service.taskset_service import (
    DuplicateTaskRefError,
    TaskRefNotFoundError,
    TasksetExistsError,
    TasksetService,
)
from nemo_evaluator.authz import scope
from nemo_evaluator.entities import MAX_NAME_LENGTH, NAME_PATTERN
from nemo_evaluator.revisions import RevisionConflictError, RevisionNotFoundError
from nemo_helix_plugin.api.parsed_filter import ParsedFilter, make_filter_dep
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.entities import EntityValidationError
from nemo_helix_plugin.entity_client import NemoEntityConflictError
from nemo_helix_plugin.jobs.openapi_utils import generate_openapi_extra_params
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.schema import Page

logger = logging.getLogger(__name__)


class TasksetPerms(PermissionSet, namespace="evaluator.tasksets"):
    """Permissions for the stored-taskset CRUD collection."""

    CREATE = perm("Create a stored taskset")
    LIST = perm("List stored tasksets")
    READ = perm("Read a stored taskset")
    DELETE = perm("Delete a stored taskset")


router = APIRouter()


@router.get(
    "/tasksets",
    summary="List Tasksets By Workspace",
    response_description="Return stored tasksets for a workspace",
    status_code=status.HTTP_200_OK,
    response_model=Page[Taskset],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=TasksetFilter,
        filter_description="Filter tasksets by workspace, name, created_at, and updated_at.",
    ),
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetPerms.LIST])
async def list_tasksets(
    workspace: str,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Page size."),
    sort: TasksetSort = Query(
        default=TasksetSort.CREATED_AT_ASC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(TasksetFilter)),
    service: TasksetService = Depends(get_taskset_service),
) -> Page[Taskset]:
    """List stored tasksets for a specific workspace."""
    # Discard any workspace override in the filter — always scope to the path workspace.
    parsed_filter.remove("workspace")
    try:
        return await service.list_tasksets(
            workspace=workspace,
            page=page,
            page_size=page_size,
            sort=sort,
            filter_operation=parsed_filter.operation,
        )
    except Exception:
        logger.exception(f"Failed to list tasksets for workspace {sanitize_for_log(workspace)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post(
    "/tasksets/{name}",
    summary="Create Taskset",
    response_description="Store a new taskset",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_409_CONFLICT: {"description": "Taskset already exists"}},
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetPerms.CREATE])
async def create_taskset(
    workspace: str,
    name: Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)],
    taskset: TasksetInput,
    project: str | None = Query(default=None, description="Optional project to associate with the taskset."),
    service: TasksetService = Depends(get_taskset_service),
) -> Taskset:
    """Store a new taskset, addressed by workspace/name, and publish it as revision 1.

    Strict create — 409 if the name is taken. Use ``PUT`` to publish a further revision.
    """
    safe_workspace = sanitize_for_log(workspace)
    safe_name = sanitize_for_log(name)
    logger.info(f"Creating taskset: {safe_workspace}/{safe_name}")
    try:
        created, _ = await service.create_taskset(name, taskset, workspace=workspace, project=project)
        return created
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during taskset creation: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except (TaskRefNotFoundError, DuplicateTaskRefError) as e:
        # A bad member reference (missing task, or two refs to the same task) — a client error.
        logger.warning(f"Taskset has an invalid task reference: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except TasksetExistsError:
        logger.warning(f"Taskset already exists: {safe_workspace}/{safe_name}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Taskset with workspace '{workspace}' and name '{name}' already exists",
        )
    except ValueError as e:
        logger.warning(f"Taskset creation validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid taskset data")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to create taskset")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.put(
    "/tasksets/{name}",
    summary="Replace Taskset",
    response_description="Publish a revision of the taskset",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Content already published; no new revision was cut"},
        status.HTTP_201_CREATED: {"model": Taskset, "description": "A new revision was published"},
        status.HTTP_409_CONFLICT: {"description": "Concurrent write; refresh and retry"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetPerms.CREATE])
async def replace_taskset(
    workspace: str,
    name: Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)],
    taskset: TasksetInput,
    response: Response,
    project: str | None = Query(default=None, description="Optional project to associate with the taskset."),
    service: TasksetService = Depends(get_taskset_service),
) -> Taskset:
    """Replace a taskset's membership and publish the result, creating it if it does not exist.

    Members are re-resolved to exact revision digests on every write, so re-submitting the *same*
    member names can still publish a new revision — if a member task published in the meantime,
    this grouping now names different content and genuinely differs. A taskset's identity is the
    exact revisions it names, not the names alone.
    """
    safe_workspace = sanitize_for_log(workspace)
    safe_name = sanitize_for_log(name)
    logger.info(f"Replacing taskset: {safe_workspace}/{safe_name}")
    try:
        replaced, published = await service.replace_taskset(name, taskset, workspace=workspace, project=project)
    except EntityValidationError as e:
        logger.warning(f"Entity store validation error during taskset replace: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except (TaskRefNotFoundError, DuplicateTaskRefError) as e:
        logger.warning(f"Taskset has an invalid task reference: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except RevisionConflictError as e:
        logger.warning(f"Taskset revision allocation contended: {e}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except NemoEntityConflictError as e:
        # Another request updated this record between our read and our write. A retry against the
        # current state is the caller's move; a 500 would wrongly suggest a server fault.
        logger.warning(f"Taskset modified concurrently during replace: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Taskset was modified by another request: {workspace}/{name}. Refresh and try again.",
        )
    except ValueError as e:
        logger.warning(f"Taskset replace validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid taskset data")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to replace taskset")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    response.status_code = status.HTTP_201_CREATED if published else status.HTTP_200_OK
    return replaced


@router.get(
    "/tasksets/{name}/revisions",
    summary="List Taskset Revisions",
    response_description="Return the taskset's published revisions, newest first",
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Taskset not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetPerms.READ])
async def list_taskset_revisions(
    workspace: str,
    name: Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)],
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Page size."),
    service: TasksetService = Depends(get_taskset_service),
) -> Page[Revision]:
    """List a taskset's published revisions — each entry's ``content_hash`` is what a pinned
    reference carries."""
    try:
        revisions = await service.list_revisions(workspace, name, page=page, page_size=page_size)
    except Exception:
        logger.exception("Failed to list taskset revisions")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    if revisions is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Taskset '{name}' not found")
    return revisions


@router.get(
    "/tasksets/{name}/revisions/{revision}",
    summary="Get Taskset Revision",
    response_description="Return the taskset's membership as of a published revision",
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Taskset or revision not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetPerms.READ])
async def get_taskset_revision(
    workspace: str,
    name: Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)],
    revision: Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)],
    service: TasksetService = Depends(get_taskset_service),
) -> Taskset:
    """Get a taskset's membership as of a published revision.

    ``revision`` is a content digest or a tag — what a reference's ``#fragment`` carries — so a
    pinned consumer reads the exact grouping that was published.
    """
    try:
        taskset = await service.get_taskset(workspace, name, revision)
    except RevisionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    if taskset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Taskset not found: {workspace}/{name}")
    return taskset


@router.put(
    "/tasksets/{name}/tags/{tag}",
    summary="Tag Taskset Revision",
    response_description="Point a tag at an existing revision",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Taskset or revision not found"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Tag is reserved"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetPerms.CREATE])
async def tag_taskset_revision(
    workspace: str,
    name: Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)],
    tag: Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)],
    revision: str = Query(description="Revision to tag — a content digest, or another tag."),
    service: TasksetService = Depends(get_taskset_service),
) -> Taskset:
    """Point a tag at an already-published revision. ``latest`` is reserved and machine-managed."""
    try:
        tagged = await service.tag_revision(workspace, name, tag, revision)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e))
    except RevisionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    if tagged is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Taskset '{name}' not found")
    return tagged


@router.get(
    "/tasksets/{name}",
    summary="Get Taskset",
    response_description="Return stored taskset details",
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Taskset not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetPerms.READ])
async def get_taskset(
    workspace: str,
    name: str,
    service: TasksetService = Depends(get_taskset_service),
) -> Taskset:
    """Get a stored taskset's current membership.

    Use ``GET /tasksets/{name}/revisions/{revision}`` to read it as of a published revision.
    """
    logger.debug(f"Getting taskset: {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
    # Only the service call can fail unexpectedly; wrap just that so the 404 below is raised outside
    # the try (no catching HTTPException only to re-raise it).
    try:
        taskset = await service.get_taskset(workspace, name)
    except Exception:
        logger.exception(f"Failed to get taskset {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    if not taskset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Taskset not found: {workspace}/{name}")
    return taskset


@router.delete(
    "/tasksets/{name}",
    summary="Delete Taskset",
    response_description="Delete a stored taskset",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Taskset not found"}},
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TasksetPerms.DELETE])
async def delete_taskset(
    workspace: str,
    name: str,
    service: TasksetService = Depends(get_taskset_service),
):
    """Delete a stored taskset by workspace and name."""
    logger.info(f"Deleting taskset: {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
    # Wrap only the service call so the 404 below is raised outside the try (no catch-and-re-raise).
    try:
        deleted = await service.delete_taskset(workspace, name)
    except NemoEntityConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Taskset was modified by another request: {workspace}/{name}. Refresh and try again.",
        ) from exc
    except Exception:
        logger.exception(f"Failed to delete taskset {sanitize_for_log(workspace)}/{sanitize_for_log(name)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Taskset not found: {workspace}/{name}")
    return None
