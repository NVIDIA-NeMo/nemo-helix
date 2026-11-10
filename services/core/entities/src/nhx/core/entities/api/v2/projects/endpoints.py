# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Project API Endpoints v2.

Core Operations:
- POST /apis/entities/v2/workspaces/{workspace}/projects              - Create project
- GET /apis/entities/v2/workspaces/{workspace}/projects               - List all projects
- GET /apis/entities/v2/workspaces/{workspace}/projects/{name}        - Get project by name
- PUT /apis/entities/v2/workspaces/{workspace}/projects/{name}        - Update project
- DELETE /apis/entities/v2/workspaces/{workspace}/projects/{name}     - Delete project
"""

import logging
import textwrap

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nemo_helix_plugin.api.pagination import PageParams, make_page_dep
from nhx.common.api.common import DeleteResponse, ErrorResponse, GenericSortField, Page, PaginationData
from nhx.core.entities.api.dependencies import EntityRepository
from nhx.core.entities.api.v2.projects.schemas import (
    Project,
    ProjectInput,
    ProjectUpdate,
)
from nhx.core.entities.app.repository.exceptions import (
    EntityNotFoundError,
    EntityVersionConflictError,
    InvalidFilterError,
)
from nhx.core.entities.utils.filter import FilterDep
from sqlalchemy.exc import IntegrityError

router = APIRouter()
API_TAG = "Entity Store"
logger = logging.getLogger(__name__)

# Entity type for projects stored in the entities table
PROJECT_ENTITY_TYPE = "project"


def _entity_to_project(entity) -> Project:
    """Convert an Entity to a Project response."""
    return Project(
        id=entity.id,
        name=entity.name,
        workspace=entity.workspace,
        description=entity.data.get("description"),
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


@router.post(
    "/v2/workspaces/{workspace}/projects",
    response_model=Project,
    tags=[API_TAG],
    status_code=201,
    summary="Create a new project",
    description=textwrap.dedent("""
        Create a new project in the given workspace.

        Example:
        ```
        POST /apis/entities/v2/workspaces/default/projects
        {
            "name": "ml-project",
            "description": "Machine Learning project"
        }
        ```
    """),
)
async def create_project(
    workspace: str,
    project: ProjectInput,
    repository: EntityRepository,
) -> Project:
    """Create a new project."""
    try:
        new_entity = await repository.create_entity(
            workspace=workspace,
            entity_type=PROJECT_ENTITY_TYPE,
            name=project.name,
            data={"description": project.description},
        )
        return _entity_to_project(new_entity)
    except IntegrityError as e:
        error_msg = str(e.orig) if hasattr(e, "orig") else str(e)
        logger.warning(f"Integrity error creating project: {error_msg}")

        if "foreign key" in error_msg.lower():
            if "fk_entities_workspace" in error_msg or "workspace" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Workspace '{workspace}' does not exist. Please create the workspace first.",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Referenced resource does not exist.",
                )
        elif (
            "duplicate key" in error_msg.lower()
            or "unique constraint" in error_msg.lower()
            or "unique constraint failed" in error_msg.lower()
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Project '{project.name}' already exists in workspace '{workspace}'.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data provided. Please check your input.",
            )


@router.get(
    "/v2/workspaces/{workspace}/projects",
    response_model=Page[Project],
    tags=[API_TAG],
    summary="List all projects",
    description=textwrap.dedent("""
        List all projects in a workspace with pagination.

        Query Parameters:
        - page, page_size: Pagination
        - sort: Sort field
        - filter: Advanced filters

        Example:
        ```
        GET /apis/entities/v2/workspaces/default/projects?sort=-created_at&page=1&page_size=10
        ```
    """),
)
async def list_projects(
    workspace: str,
    repository: EntityRepository,
    filter: FilterDep,
    page_params: PageParams = Depends(make_page_dep()),
    sort: GenericSortField = Query(GenericSortField.CREATED_AT_DESC, description="Sort field"),
) -> Page[Project]:
    """List projects in the workspace."""
    try:
        entities, total = await repository.list_entities(
            workspace=workspace,
            entity_type=PROJECT_ENTITY_TYPE,
            page=page_params.page,
            page_size=page_params.page_size,
            sort=sort,
            filter_op=filter,
        )
    except InvalidFilterError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    projects = [_entity_to_project(e) for e in entities]

    return Page(
        data=projects,
        pagination=PaginationData(
            page=page_params.page,
            page_size=page_params.page_size,
            total_results=total,
            total_pages=(total + page_params.page_size - 1) // page_params.page_size,
            current_page_size=len(projects),
        ),
        sort=sort,
        filter=filter.to_dict() if filter else None,
    )


@router.get(
    "/v2/workspaces/{workspace}/projects/{name}",
    response_model=Project,
    tags=[API_TAG],
    summary="Get project by name",
    responses={404: {"model": ErrorResponse, "description": "Project not found"}},
    description=textwrap.dedent("""
        Get a specific project by its workspace and name.

        Example:
        ```
        GET /apis/entities/v2/workspaces/default/projects/ml-project
        ```
    """),
)
async def get_project(
    workspace: str,
    name: str,
    repository: EntityRepository,
) -> Project:
    """Get project by name."""
    entity = await repository.get_entity_by_name(
        workspace=workspace,
        entity_type=PROJECT_ENTITY_TYPE,
        name=name,
    )
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{name}' not found in workspace '{workspace}'",
        )
    return _entity_to_project(entity)


@router.put(
    "/v2/workspaces/{workspace}/projects/{name}",
    response_model=Project,
    tags=[API_TAG],
    summary="Update project",
    description=textwrap.dedent("""
        Update a project's description.

        Example:
        ```
        PUT /apis/entities/v2/workspaces/default/projects/ml-project
        {
            "description": "Updated description for ML project"
        }
        ```
    """),
)
async def update_project(
    workspace: str,
    name: str,
    project_data: ProjectUpdate,
    repository: EntityRepository,
) -> Project:
    """Update project."""
    try:
        entity = await repository.update_entity_by_name(
            workspace=workspace,
            entity_type=PROJECT_ENTITY_TYPE,
            name=name,
            data={"description": project_data.description},
        )
        return _entity_to_project(entity)
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except EntityVersionConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except IntegrityError as e:
        error_msg = str(e.orig) if hasattr(e, "orig") else str(e)
        logger.warning(f"Integrity error updating project: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid data provided.",
        )


@router.delete(
    "/v2/workspaces/{workspace}/projects/{name}",
    response_model=DeleteResponse,
    tags=[API_TAG],
    summary="Delete project",
    description=textwrap.dedent("""
        Delete a project.

        Example:
        ```
        DELETE /apis/entities/v2/workspaces/default/projects/ml-project
        ```
    """),
)
async def delete_project(
    workspace: str,
    name: str,
    repository: EntityRepository,
) -> DeleteResponse:
    """Delete project."""
    try:
        selected = await repository.get_entity_by_name(workspace=workspace, entity_type=PROJECT_ENTITY_TYPE, name=name)
        deleted_count = (
            await repository.delete_entity(entity_id=selected.id, expected_db_version=selected.db_version)
            if selected
            else 0
        )
        if deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project '{name}' not found in workspace '{workspace}'",
            )
        return DeleteResponse(
            id=f"{workspace}/{name}",
            message="Project deleted successfully",
        )
    except EntityVersionConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except IntegrityError as e:
        error_msg = str(e.orig) if hasattr(e, "orig") else str(e)
        logger.warning(f"Integrity error deleting project: {error_msg}")

        if "foreign key" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot delete project '{name}'. There are entities that reference this project. Please delete all entities in this project first.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete project due to data constraints.",
            )
