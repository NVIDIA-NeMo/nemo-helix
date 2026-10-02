# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generic Entity API Endpoints v2.

Name-Based Operations (Primary):
- POST   /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}               - Create entity
- GET    /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}               - List entities
- GET    /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}        - Get entity by name
- PUT    /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}        - Update entity by name
- DELETE /apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}        - Delete entity by name

ID-Based Operations (Debug/Internal):
- GET    /apis/entities/v2/entities/{entity_id}                                        - Get entity by ID

Cross-Workspace Queries:
- GET    /apis/entities/v2/workspaces/-/entities/{entity_type}                         - List entities across all accessible workspaces
"""

import logging
import textwrap

from fastapi import APIRouter, HTTPException, Query, status
from nhx.common.api.common import DeleteResponse, Page, PaginationData
from nhx.common.auth import ALL_WORKSPACES
from nhx.common.auth.client import AuthClient
from nhx.core.entities.api.dependencies import (
    AuthClientDep,
    EntityRepository,
    WorkspaceRepository,
)
from nhx.core.entities.api.v2.entities.schemas import EntityCreateInput, EntityUpdate
from nhx.core.entities.api.v2.utils import (
    ROLE_BINDING_ENTITY_TYPE,
    add_workspace_filtering,
    bindings_cache_delete,
    get_accessible_workspaces,
    raise_if_workspace_inaccessible,
    require_workspace_access,
)
from nhx.core.entities.app.repository import WorkspaceRepositoryInterface
from nhx.core.entities.app.repository.exceptions import EntityNotFoundError, EntityVersionConflictError
from nhx.core.entities.entities import Entity
from nhx.core.entities.utils.filter import FilterDep
from nhx.core.entities.utils.identifiers import generate_entity_name
from sqlalchemy.exc import IntegrityError


class EntitiesPage(Page[Entity]):
    group_counts: dict[str, int] | None = None


router = APIRouter()
API_TAG = "Entity Store"
logger = logging.getLogger(__name__)

PROJECT_ENTITY_TYPE = "project"


async def validate_project_exists(repository: EntityRepository, workspace: str, project: str | None) -> None:
    """Validate that the specified project exists in the workspace.

    Args:
        repository: Entity repository for database access
        workspace: Workspace identifier
        project: Project name to validate (if None, no validation is performed)

    Raises:
        HTTPException: 422 if project does not exist in workspace
    """
    if project is None:
        return

    project_entity = await repository.get_entity_by_name(
        workspace=workspace,
        entity_type=PROJECT_ENTITY_TYPE,
        name=project,
    )
    if project_entity is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Project '{project}' does not exist in workspace '{workspace}'. Please create the project first.",
        )


async def _validate_parent_access(
    accessible: set[str] | None,
    repository: EntityRepository,
    parent_id: str,
) -> None:
    """Ensure the parent exists and the caller may reference it (parent may live in another workspace).

    The new row is always created in the path ``workspace``; that is already authorized for this
    request. Here we only ensure the **parent row's** workspace is in
    ``get_accessible_workspaces`` (same role-binding / OBO logic as list filters), so a child in W1
    cannot point at a parent in W2 unless the effective user has access to W2.
    """

    parent = await repository.get_entity_by_id(entity_id=parent_id)
    if not parent or (accessible is not None and parent.workspace not in accessible):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Parent entity '{parent_id}' not found or not in accessible workspaces",
        )


async def validate_workspace_not_deleting(
    workspace_repository: WorkspaceRepositoryInterface,
    auth_client: AuthClient,
    workspace: str,
    *,
    missing_workspace_status: int = status.HTTP_404_NOT_FOUND,
) -> None:
    """Validate that a workspace exists and is not being deleted.

    Always checks workspace existence. For the deletion-stage check, service
    principals are allowed through (for cleanup), but all other requests
    receive 404 as if the workspace doesn't exist.

    Args:
        missing_workspace_status: HTTP status code to return when the workspace
            does not exist. Use 422 for write operations (POST/PUT) where the
            workspace is input being validated, and 404 for read/delete
            operations (GET/DELETE) where a missing workspace means "not found".
    """
    ws = await workspace_repository.get_workspace_by_name(name=workspace)
    if ws is None:
        if missing_workspace_status == status.HTTP_422_UNPROCESSABLE_CONTENT:
            detail = f"Workspace '{workspace}' does not exist. Please create the workspace first."
        else:
            detail = f"Workspace '{workspace}' not found"
        raise HTTPException(status_code=missing_workspace_status, detail=detail)

    # Service principals can access workspaces being deleted (for cleanup)
    if ws._deletion_stage is not None and auth_client.principal.caller_kind != "service_principal":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace}' not found",
        )


async def _invalidate_role_binding_cache_if_present(
    repository: EntityRepository,
    workspace: str,
    entity_type: str,
    name: str,
    parent: str | None = None,
) -> None:
    """For ``role_binding`` rows, evict the principal's cached bindings before an update or delete.

    No-op for other entity types. If the type is ``role_binding`` and no row
    exists at the given name, raises 404.
    """
    if entity_type != ROLE_BINDING_ENTITY_TYPE:
        return

    entity = await repository.get_entity_by_name(
        workspace=workspace,
        entity_type=entity_type,
        name=name,
        parent=parent,
    )

    if entity is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Entity not found")

    principal = entity.data.get("principal")
    if principal is not None:
        await bindings_cache_delete(principal)


@router.post(
    "/v2/workspaces/{workspace}/entities/{entity_type}",
    response_model=Entity,
    tags=[API_TAG],
    status_code=201,
    summary="Create a new entity",
    description=textwrap.dedent("""
        Create a new entity of the specified type in the given workspace.

        If name is not provided, it will be auto-generated based on the entity type.

        Example:
        ```
        POST /apis/entities/v2/workspaces/default/entities/customization_config
        {
            "name": "my-config",
            "data": {
                "target_id": "llama-2-7b",
                "training_options": {"learning_rate": 0.01}
            }
        }
        ```
    """),
)
async def create_entity(
    workspace: str,
    entity_type: str,
    entity: EntityCreateInput,
    repository: EntityRepository,
    workspace_repository: WorkspaceRepository,
    auth_client: AuthClientDep,
) -> Entity:
    """Create a new entity."""
    # Check if workspace is being deleted (404 for user requests)
    await validate_workspace_not_deleting(
        workspace_repository,
        auth_client,
        workspace,
        missing_workspace_status=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )

    # Auto-generate name if not provided (treat empty string as not provided)
    name = entity.name
    if not name:
        name = generate_entity_name(entity_type)

    # Validate project exists if specified
    await validate_project_exists(repository, workspace, entity.project)

    accessible = await require_workspace_access(
        repository,
        workspace,
    )

    if entity.parent is not None:
        await _validate_parent_access(accessible, repository, entity.parent)
    try:
        new_entity = await repository.create_entity(
            workspace=workspace,
            entity_type=entity_type,
            name=name,
            data=entity.data,
            parent=entity.parent,
            project=entity.project,
            created_by=auth_client.principal.effective_id,
        )
        if entity_type == ROLE_BINDING_ENTITY_TYPE:
            principal = entity.data.get("principal")
            if principal is not None:
                await bindings_cache_delete(str(principal))
        return new_entity
    except IntegrityError as e:
        error_msg = str(e.orig) if hasattr(e, "orig") else str(e)
        error_msg_lower = error_msg.lower()

        is_unique_violation = (
            "duplicate key" in error_msg_lower
            or "unique constraint" in error_msg_lower
            or "unique constraint failed" in error_msg_lower  # SQLite format
        )

        # Idempotent reconcilers (e.g. ModelProviderReconciler.ensure_passthrough_virtual_model)
        # rely on the 409 response and intentionally swallow ConflictError.  Logging those
        # at WARNING drowns the platform log in repetitive noise every reconcile cycle, so
        # demote the expected case to DEBUG.  Genuine integrity errors stay at WARNING.
        if is_unique_violation:
            logger.debug(
                "Entity %s/%s of type %s already exists (idempotent create): %s",
                workspace,
                name,
                entity_type,
                error_msg,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Entity '{name}' of type '{entity_type}' already exists in workspace '{workspace}'.",
            ) from e

        logger.warning(f"Integrity error creating entity: {error_msg}")

        if "foreign key" in error_msg_lower:
            if "fk_entities_workspace" in error_msg or "workspace" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Workspace '{workspace}' does not exist. Please create the workspace first.",
                ) from e
            # SQLite FK errors don't include constraint names — check workspace explicitly
            ws = await workspace_repository.get_workspace_by_name(name=workspace)
            if ws is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Workspace '{workspace}' does not exist. Please create the workspace first.",
                ) from e
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Referenced resource does not exist.",
            ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data provided. Please check your input.",
            ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get(
    "/v2/workspaces/{workspace}/entities/{entity_type}",
    response_model=EntitiesPage,
    tags=[API_TAG],
    summary="List entities",
    description=textwrap.dedent("""
        List all entities of a specific type in the given workspace.

        Use workspace="-" to list entities across all workspaces the principal has
        access to.

        Query Parameters:
        - sort: Sort field
        - page, page_size: Pagination
        - filter: Advanced filters (JSON, text, or bracket notation)

        Examples:
        ```
        GET /apis/entities/v2/workspaces/default/entities/customization_config?sort=-created_at
        GET /apis/entities/v2/workspaces/-/entities/customization_config  # Cross-workspace query
        ```
    """),
)
async def list_entities(
    workspace: str,
    entity_type: str,
    repository: EntityRepository,
    workspace_repository: WorkspaceRepository,
    filter: FilterDep,
    auth_client: AuthClientDep,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Items per page"),
    # TODO: eventually validate this, for now trust people to submit valid fields. -md
    sort: str = Query(
        "-created_at",
        description="Sort field",
        examples=["-created_at", "created_at", "-updated_at", "updated_at", "-name", "name"],
    ),
    count_by: str | None = Query(
        default=None,
        description="Optional direct string data field whose matching values should be counted.",
    ),
) -> EntitiesPage:
    """List entities with filtering, supporting cross-workspace queries."""
    accessible_workspaces = await get_accessible_workspaces(repository)
    # Handle cross-workspace query (workspace = "*")
    if workspace == ALL_WORKSPACES:
        # Build combined filter for workspace access and user's filter
        query_workspace = ALL_WORKSPACES
        effective_filter = add_workspace_filtering(accessible_workspaces, filter, field="workspace")
    else:
        raise_if_workspace_inaccessible(
            accessible_workspaces,
            workspace,
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
        # Check if workspace is being deleted (404 for user requests)
        await validate_workspace_not_deleting(workspace_repository, auth_client, workspace)

        query_workspace = workspace
        effective_filter = filter

    entities, total = await repository.list_entities(
        workspace=query_workspace,
        entity_type=entity_type,
        page=page,
        page_size=page_size,
        sort=sort,
        filter_op=effective_filter,
        relationship_child_workspaces=accessible_workspaces,
    )
    group_counts = None
    if count_by is not None:
        try:
            group_counts = await repository.count_entities_by(
                workspace=query_workspace,
                entity_type=entity_type,
                group_by=count_by,
                filter_op=effective_filter,
                relationship_child_workspaces=accessible_workspaces,
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    return EntitiesPage(
        data=entities,
        pagination=PaginationData(
            page=page,
            page_size=page_size,
            current_page_size=len(entities),
            total_results=total,
            total_pages=(total + page_size - 1) // page_size,
        ),
        sort=sort,
        filter=filter.to_dict() if filter else None,
        group_counts=group_counts,
    )


@router.get(
    "/v2/workspaces/{workspace}/entities/{entity_type}/{name}",
    response_model=Entity,
    tags=[API_TAG],
    summary="Get entity by name",
    description=textwrap.dedent("""
        Get a specific entity by its workspace, type, and name.

        Example:
        ```
        GET /apis/entities/v2/workspaces/default/entities/customization_config/my-config
        ```
    """),
)
async def get_entity_by_name(
    workspace: str,
    entity_type: str,
    name: str,
    repository: EntityRepository,
    workspace_repository: WorkspaceRepository,
    auth_client: AuthClientDep,
    parent: str | None = Query(default=None, description="Parent entity ID for nested entities"),
) -> Entity:
    """Get entity by name."""
    # Check if workspace is being deleted (404 for user requests)
    await validate_workspace_not_deleting(workspace_repository, auth_client, workspace)

    await require_workspace_access(
        repository,
        workspace,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )

    entity = await repository.get_entity_by_name(
        workspace=workspace,
        entity_type=entity_type,
        name=name,
        parent=parent,
    )
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity


@router.put(
    "/v2/workspaces/{workspace}/entities/{entity_type}/{name}",
    response_model=Entity,
    tags=[API_TAG],
    summary="Update entity by name",
    description=textwrap.dedent("""
        Update an entity by its name. Optionally change the entity's name.

        Example:
        ```
        PUT /apis/entities/v2/workspaces/default/entities/customization_config/my-config
        {
            "data": {
                "target_id": "llama-2-7b",
                "training_options": {"learning_rate": 0.02}
            }
        }
        ```
    """),
)
async def update_entity_by_name(
    workspace: str,
    entity_type: str,
    name: str,
    entity_data: EntityUpdate,
    repository: EntityRepository,
    workspace_repository: WorkspaceRepository,
    auth_client: AuthClientDep,
    parent: str | None = Query(default=None, description="Parent entity ID for nested entities"),
) -> Entity:
    """Update entity by name."""
    # Check if workspace is being deleted (404 for user requests)
    await validate_workspace_not_deleting(
        workspace_repository,
        auth_client,
        workspace,
        missing_workspace_status=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )

    await require_workspace_access(
        repository,
        workspace,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )

    await _invalidate_role_binding_cache_if_present(repository, workspace, entity_type, name, parent)

    # Validate project exists if specified
    await validate_project_exists(repository, workspace, entity_data.project)

    try:
        entity = await repository.update_entity_by_name(
            workspace=workspace,
            entity_type=entity_type,
            name=name,
            data=entity_data.data,
            new_name=entity_data.new_name,
            parent=parent,
            project=entity_data.project,
            updated_by=auth_client.principal.effective_id,
            expected_db_version=entity_data.expected_db_version,
        )
        return entity
    except EntityVersionConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except IntegrityError as e:
        error_msg = str(e.orig) if hasattr(e, "orig") else str(e)
        error_msg_lower = error_msg.lower()

        is_unique_violation = (
            "duplicate key" in error_msg_lower
            or "unique constraint" in error_msg_lower
            or "unique constraint failed" in error_msg_lower  # SQLite format
        )

        if is_unique_violation:
            logger.debug(
                "Conflict updating entity %s/%s of type %s: %s",
                workspace,
                name,
                entity_type,
                error_msg,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A conflicting entity already exists.",
            ) from e

        logger.warning(f"Integrity error updating entity: {error_msg}")

        if "foreign key" in error_msg_lower:
            if "fk_entities_workspace" in error_msg or "workspace" in error_msg:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Workspace '{workspace}' does not exist. Please create the workspace first.",
                ) from e
            # SQLite FK errors don't include constraint names — check workspace explicitly
            ws = await workspace_repository.get_workspace_by_name(name=workspace)
            if ws is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=f"Workspace '{workspace}' does not exist. Please create the workspace first.",
                ) from e
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Referenced resource does not exist.",
            ) from e
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data provided.",
            ) from e


@router.delete(
    "/v2/workspaces/{workspace}/entities/{entity_type}/{name}",
    response_model=DeleteResponse,
    tags=[API_TAG],
    summary="Delete entity by name",
    description=textwrap.dedent("""
        Delete an entity by its name.

        Example:
        ```
        DELETE /apis/entities/v2/workspaces/default/entities/customization_config/my-config
        ```
    """),
)
async def delete_entity_by_name(
    workspace: str,
    entity_type: str,
    name: str,
    repository: EntityRepository,
    workspace_repository: WorkspaceRepository,
    auth_client: AuthClientDep,
    parent: str | None = Query(default=None, description="Parent entity ID for nested entities"),
    expected_db_version: int | None = Query(
        default=None,
        description="Optional database version for optimistic locking. Delete only succeeds if the entity still has this version.",
    ),
) -> DeleteResponse:
    """Delete entity by name."""
    # Check if workspace is being deleted (404 for user requests)
    await validate_workspace_not_deleting(workspace_repository, auth_client, workspace)
    await require_workspace_access(
        repository,
        workspace,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )

    await _invalidate_role_binding_cache_if_present(repository, workspace, entity_type, name, parent)

    try:
        deleted_count = await repository.delete_entity_by_name(
            workspace=workspace,
            entity_type=entity_type,
            name=name,
            parent=parent,
            expected_db_version=expected_db_version,
        )
    except EntityVersionConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Entity not found")
    return DeleteResponse(id=f"{workspace}/{entity_type}/{name}")


@router.get(
    "/v2/entities/{id}",
    response_model=Entity,
    tags=[API_TAG],
    summary="Get entity by ID (debug/internal)",
    description=textwrap.dedent("""
        Get a specific entity by its unique identifier.
        This endpoint is primarily for debugging and internal use.

        Example:
        ```
        GET /apis/entities/v2/entities/customization-config-5Q2LoF8z8M9JZxZsHwJKNn
        ```
    """),
)
async def get_entity_by_id(
    id: str,
    repository: EntityRepository,
    workspace_repository: WorkspaceRepository,
    auth_client: AuthClientDep,
) -> Entity:
    """Get entity by id (debug/internal use)."""

    entity = await repository.get_entity_by_id(entity_id=id)

    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    await validate_workspace_not_deleting(workspace_repository, auth_client, entity.workspace)

    return entity
