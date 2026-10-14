# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container image builds of a task revision, under
/apis/evals/v2/workspaces/{workspace}/tasks/{name}/revisions/{revision}/image-builds."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from nemo_evals.api.dependencies import get_image_build_service
from nemo_evals.api.v2.tasks import TaskPerms
from nemo_evals.authz import scope
from nemo_evals.entities import MAX_NAME_LENGTH, NAME_PATTERN
from nemo_evals.image_builds.models import ImageBuild, ImageBuildRequest
from nemo_evals.image_builds.planning import ImagePlanError
from nemo_evals.image_builds.service import ImageBuildForbiddenError, ImageBuildJobConflictError, ImageBuildService
from nemo_evals.revisions import RevisionNotFoundError
from nemo_evals.task_identity import UnsupportedTaskKindError
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.client.errors import NemoHTTPError, PermissionDeniedError
from nemo_helix_plugin.schema import Page

logger = logging.getLogger(__name__)

_BUILDS = "/tasks/{name}/revisions/{revision}/image-builds"

TaskName = Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)]
RevisionSelector = Annotated[
    str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN, description="Revision digest or tag.")
]


class TaskImageBuildPerms(PermissionSet, namespace="evaluator.tasks.image-builds"):
    """Permissions for container image builds of stored tasks."""

    CREATE = perm("Request a container image build of a task revision")
    LIST = perm("List the container image builds of a task revision")
    READ = perm("Read a container image build of a task revision")


router = APIRouter()


@router.post(
    _BUILDS,
    summary="Request Task Image Build",
    response_description="The build started, or the one already running or done",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_403_FORBIDDEN: {"description": "The caller may not submit container image builds"},
        status.HTTP_404_NOT_FOUND: {"description": "Task or revision not found"},
        status.HTTP_409_CONFLICT: {"description": "The build's Job name is held by another Job"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "The task's images can't be built"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TaskPerms.READ, TaskImageBuildPerms.CREATE])
async def request_task_image_build(
    workspace: str,
    name: TaskName,
    revision: RevisionSelector,
    request: Annotated[ImageBuildRequest | None, Body()] = None,
    service: ImageBuildService = Depends(get_image_build_service),
) -> ImageBuild:
    """Build every container image a Harbor task revision needs.

    Repeating the request while a build is running, or after it succeeded, returns that build.
    After a failed build, it starts the next attempt.
    """
    try:
        build = await service.request_build(workspace, name, revision, request or ImageBuildRequest())
    except RevisionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except (UnsupportedTaskKindError, ImagePlanError) as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e
    except ImageBuildForbiddenError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e)) from e
    except ImageBuildJobConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e
    except PermissionDeniedError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Could not start the build: {e}") from e
    except NemoHTTPError as e:
        logger.warning("Jobs refused the image build job: %s", e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Could not start the build: {e}") from e
    if build is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task not found: {workspace}/{name}")
    return build


@router.get(
    _BUILDS,
    summary="List Task Image Builds",
    response_description="The revision's builds, newest first",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Task or revision not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TaskPerms.READ, TaskImageBuildPerms.LIST])
async def list_task_image_builds(
    workspace: str,
    name: TaskName,
    revision: RevisionSelector,
    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=100, ge=1, le=1000, description="Page size."),
    service: ImageBuildService = Depends(get_image_build_service),
) -> Page[ImageBuild]:
    """List the container image builds of a task revision."""
    try:
        builds = await service.list_builds(workspace, name, revision, page=page, page_size=page_size)
    except RevisionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    if builds is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task not found: {workspace}/{name}")
    return builds


@router.get(
    _BUILDS + "/{build_id}",
    summary="Get Task Image Build",
    response_description="One build and the status of each image",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Task, revision or build not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[TaskPerms.READ, TaskImageBuildPerms.READ])
async def get_task_image_build(
    workspace: str,
    name: TaskName,
    revision: RevisionSelector,
    build_id: str,
    service: ImageBuildService = Depends(get_image_build_service),
) -> ImageBuild:
    """Get one container image build of a task revision."""
    try:
        build = await service.get_build(workspace, name, revision, build_id)
    except RevisionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    if build is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Image build {build_id} of {workspace}/{name} not found"
        )
    return build
