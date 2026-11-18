# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored-skill routes under ``/apis/agents/v2/workspaces/{workspace}/skills``.

These mirror the evals taskset routes: ``POST`` creates a skill and publishes revision 1, ``PUT``
publishes the next revision (creating the skill if absent), and a revision is addressed by a tag
or its content digest. Archives are never uploaded or downloaded through these routes; a publish
names a source directory in a fileset the caller can read.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from typing import Annotated, NoReturn, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status
from nemo_agents_plugin.api.v2._perms import SkillPerms
from nemo_agents_plugin.api.v2.dependencies import get_skill_service
from nemo_agents_plugin.authz import scope
from nemo_agents_plugin.schema import SkillFilter, SkillSort
from nemo_agents_plugin.skill_service import (
    LinkedTasksetNotFoundError,
    SkillExistsError,
    SkillRevisionConflictError,
    SkillRevisionContentMismatchError,
    SkillRevisionNotFoundError,
    SkillService,
    SkillValidationError,
)
from nemo_agents_plugin.skill_storage import SkillSourceError
from nemo_helix_plugin.agents.skills import SKILL_NAME_MAX_LENGTH, SKILL_NAME_PATTERN, Skill, SkillInput
from nemo_helix_plugin.api.pagination import PageParams, make_page_dep
from nemo_helix_plugin.api.parsed_filter import ParsedFilter, make_filter_dep
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entities import CLIENT_ERRORS
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityValidationError
from nemo_helix_plugin.jobs.openapi_utils import generate_openapi_extra_params
from nemo_helix_plugin.log_utils import sanitize_for_log
from nemo_helix_plugin.schema import Page

logger = logging.getLogger(__name__)

router = APIRouter()

_T = TypeVar("_T")

#: A revision or tag path segment: what a reference's ``#fragment`` carries (a tag or a 64-hex digest).
_FRAGMENT_PATTERN = r"^[\w\-.]+$"
_FRAGMENT_MAX_LENGTH = 255

SkillName = Annotated[str, Path(max_length=SKILL_NAME_MAX_LENGTH, pattern=SKILL_NAME_PATTERN)]
Fragment = Annotated[str, Path(max_length=_FRAGMENT_MAX_LENGTH, pattern=_FRAGMENT_PATTERN)]

_NOT_FOUND: dict[int | str, dict[str, object]] = {
    status.HTTP_404_NOT_FOUND: {"description": "Skill or revision not found"}
}


def _not_found(workspace: str, name: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Skill not found: {workspace}/{name}")


async def _call(action: str, call: Awaitable[_T]) -> _T:
    """Run a service call and map its typed errors to HTTP statuses.

    Typed-client and entity-client 4xx errors (``CLIENT_ERRORS``) and ``HTTPException`` (for example
    a caller authorization denial) pass through unchanged.
    """
    try:
        return await call
    except HTTPException:
        raise
    except Exception as exc:
        _raise_mapped(action, exc)


def _raise_mapped(action: str, exc: Exception) -> NoReturn:
    if isinstance(exc, SkillExistsError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if isinstance(exc, (SkillRevisionConflictError, NemoEntityConflictError)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The skill was modified by another request ({exc}). Refresh and try again.",
        ) from exc
    if isinstance(exc, SkillRevisionNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, (SkillValidationError, SkillSourceError, LinkedTasksetNotFoundError, NemoEntityValidationError)):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    if isinstance(exc, SkillRevisionContentMismatchError):
        logger.error("Stored skill content failed verification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The stored skill revision failed content verification",
        ) from exc
    if isinstance(exc, CLIENT_ERRORS):
        raise exc
    logger.exception("Failed to %s", action)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error") from exc


@router.get(
    "/skills",
    summary="List Skills",
    response_description="Return stored skills at their latest revision",
    status_code=status.HTTP_200_OK,
    response_model=Page[Skill],
    response_model_exclude_none=True,
    openapi_extra=generate_openapi_extra_params(
        filter_schema=SkillFilter,
        filter_description="Filter skills by name and by the latest revision's frontmatter description and tags.",
    ),
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SkillPerms.LIST])
async def list_skills(
    workspace: str,
    page_params: PageParams = Depends(make_page_dep()),
    sort: SkillSort = Query(
        default=SkillSort.CREATED_AT_ASC,
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed_filter: ParsedFilter = Depends(make_filter_dep(SkillFilter)),
    service: SkillService = Depends(get_skill_service),
) -> Page[Skill]:
    """List stored skills in a workspace, each at its latest revision."""
    # Always scope to the path workspace.
    parsed_filter.remove("workspace")
    page = await _call(
        "list skills",
        service.list_skills(
            workspace,
            page=page_params.page,
            page_size=page_params.page_size,
            sort=sort.value,
            filter_operation=parsed_filter.operation,
        ),
    )
    page.filter = parsed_filter.to_response()
    return page


@router.post(
    "/skills/{name}",
    summary="Create Skill",
    response_description="Store a new skill and publish revision 1",
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_409_CONFLICT: {"description": "Skill already exists"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid source, tags or linked taskset"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SkillPerms.CREATE])
async def create_skill(
    workspace: str,
    name: SkillName,
    body: SkillInput,
    project: str | None = Query(default=None, description="Optional project to associate with the skill."),
    service: SkillService = Depends(get_skill_service),
) -> Skill:
    """Store a new skill from a source directory and publish it as revision 1.

    Strict create: 409 if the name is taken. Use ``PUT`` to publish a further revision. The
    ``SKILL.md`` frontmatter ``name`` must equal the skill name.
    """
    logger.info("Creating skill %s/%s", sanitize_for_log(workspace), sanitize_for_log(name))
    return await _call("create skill", service.create_skill(workspace, name, body, project=project))


@router.put(
    "/skills/{name}",
    summary="Publish Skill",
    response_description="Publish a revision of the skill",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {"description": "Content already published; no new revision was cut"},
        status.HTTP_201_CREATED: {"model": Skill, "description": "A new revision was published"},
        status.HTTP_409_CONFLICT: {"description": "Concurrent write; refresh and retry"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Invalid source, tags or linked taskset"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SkillPerms.CREATE])
async def publish_skill(
    workspace: str,
    name: SkillName,
    body: SkillInput,
    response: Response,
    project: str | None = Query(default=None, description="Optional project to associate with a new skill."),
    service: SkillService = Depends(get_skill_service),
) -> Skill:
    """Publish the source as the skill's next revision, creating the skill if it does not exist.

    Publishing content identical to the latest revision cuts no new revision (200); the requested
    tags are still applied.
    """
    logger.info("Publishing skill %s/%s", sanitize_for_log(workspace), sanitize_for_log(name))
    skill, published = await _call("publish skill", service.publish_skill(workspace, name, body, project=project))
    response.status_code = status.HTTP_201_CREATED if published else status.HTTP_200_OK
    return skill


@router.get(
    "/skills/{name}/revisions",
    summary="List Skill Revisions",
    response_description="Return the skill's published revisions, newest first",
    status_code=status.HTTP_200_OK,
    response_model=Page[Skill],
    response_model_exclude_none=True,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Skill not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SkillPerms.READ])
async def list_skill_revisions(
    workspace: str,
    name: SkillName,
    page_params: PageParams = Depends(make_page_dep()),
    service: SkillService = Depends(get_skill_service),
) -> Page[Skill]:
    """List a skill's published revisions; each entry's ``content_hash`` is what a pinned reference carries."""
    revisions = await _call(
        "list skill revisions",
        service.list_revisions(workspace, name, page=page_params.page, page_size=page_params.page_size),
    )
    if revisions is None:
        raise _not_found(workspace, name)
    return revisions


@router.get(
    "/skills/{name}/revisions/{revision}",
    summary="Get Skill Revision",
    response_description="Return the skill as of a published revision",
    status_code=status.HTTP_200_OK,
    responses=_NOT_FOUND,
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SkillPerms.READ])
async def get_skill_revision(
    workspace: str,
    name: SkillName,
    revision: Fragment,
    service: SkillService = Depends(get_skill_service),
) -> Skill:
    """Get a skill as of a published revision, named by a tag or its content digest."""
    skill = await _call("get skill revision", service.get_skill(workspace, name, revision))
    if skill is None:
        raise _not_found(workspace, name)
    return skill


@router.put(
    "/skills/{name}/tags/{tag}",
    summary="Tag Skill Revision",
    response_description="Point a tag at an existing revision",
    status_code=status.HTTP_200_OK,
    responses={
        **_NOT_FOUND,
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Tag is reserved or invalid"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SkillPerms.CREATE])
async def tag_skill_revision(
    workspace: str,
    name: SkillName,
    tag: Fragment,
    revision: str = Query(description="Revision to tag: a content digest, or another tag."),
    service: SkillService = Depends(get_skill_service),
) -> Skill:
    """Point a tag at an already-published revision. ``latest`` is reserved and machine-managed."""
    tagged = await _call("tag skill revision", service.tag_revision(workspace, name, tag, revision))
    if tagged is None:
        raise _not_found(workspace, name)
    return tagged


@router.get(
    "/skills/{name}",
    summary="Get Skill",
    response_description="Return the skill at its latest revision",
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Skill not found"}},
)
@scope.read
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SkillPerms.READ])
async def get_skill(
    workspace: str,
    name: SkillName,
    service: SkillService = Depends(get_skill_service),
) -> Skill:
    """Get a skill at its ``latest`` revision, with every tag pointer.

    Use ``GET /skills/{name}/revisions/{revision}`` to read it as of another revision.
    """
    skill = await _call("get skill", service.get_skill(workspace, name))
    if skill is None:
        raise _not_found(workspace, name)
    return skill


@router.delete(
    "/skills/{name}",
    summary="Delete Skill",
    response_description="Delete a stored skill and its revisions",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Skill not found"},
        status.HTTP_409_CONFLICT: {"description": "Concurrent write; refresh and retry"},
    },
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[SkillPerms.DELETE])
async def delete_skill(
    workspace: str,
    name: SkillName,
    service: SkillService = Depends(get_skill_service),
) -> None:
    """Delete a stored skill and its revisions.

    Archives stay in the service-owned skills fileset, since a recorded run may pin one.
    """
    logger.info("Deleting skill %s/%s", sanitize_for_log(workspace), sanitize_for_log(name))
    if not await _call("delete skill", service.delete_skill(workspace, name)):
        raise _not_found(workspace, name)
