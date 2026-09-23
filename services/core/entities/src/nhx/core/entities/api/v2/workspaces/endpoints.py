# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Workspace API Endpoints v2.

Core Operations:
- POST /apis/entities/v2/workspaces              - Create workspace
- GET /apis/entities/v2/workspaces               - List all workspaces
- GET /apis/entities/v2/workspaces/{id}          - Get workspace by ID
- PUT /apis/entities/v2/workspaces/{id}          - Update workspace
- DELETE /apis/entities/v2/workspaces/{id}       - Delete workspace

Member Management:
- GET /apis/entities/v2/workspaces/{workspace}/members                  - List workspace members
- POST /apis/entities/v2/workspaces/{workspace}/members                 - Add workspace member
- PUT /apis/entities/v2/workspaces/{workspace}/members/{principal_id}   - Update member roles
- DELETE /apis/entities/v2/workspaces/{workspace}/members/{principal_id} - Remove workspace member
"""

import hashlib
import logging
import textwrap
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status
from nhx.common.api.common import DeleteResponse, GenericSortField, Page, PaginationData
from nhx.common.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nhx.common.auth.models import Principal
from nhx.core.entities.api.dependencies import AuthClientDep, EntityRepository, WorkspaceRepository
from nhx.core.entities.api.v2.utils import (
    ROLE_BINDING_ENTITY_TYPE,
    add_workspace_filtering,
    bindings_cache_delete,
    get_accessible_workspaces,
)
from nhx.core.entities.api.v2.workspaces.schemas import (
    WorkspaceInput,
    WorkspaceMember,
    WorkspaceMemberInput,
    WorkspaceMemberListResponse,
    WorkspaceMemberUpdate,
    WorkspaceUpdate,
)
from nhx.core.entities.entities import Entity, Workspace, WorkspaceDeletionStage
from nhx.core.entities.utils.filter import FilterDep
from sqlalchemy.exc import IntegrityError

router = APIRouter()
API_TAG = "Entity Store"
logger = logging.getLogger(__name__)

_ACTIVE_WORKSPACE_FILTER = ComparisonOperation(
    operator=FilterOperator.EQ,
    field="deletion_stage",
    value=None,
)


def _exclude_deleting_workspaces(filter_op: FilterOperation | None) -> FilterOperation:
    """AND the caller filter with deletion_stage IS NULL so list matches GET/DELETE."""
    if filter_op is None:
        return _ACTIVE_WORKSPACE_FILTER
    return LogicalOperation(
        operator=FilterOperator.AND,
        operations=[filter_op, _ACTIVE_WORKSPACE_FILTER],
    )


def _principal_for_role_binding(principal: Principal) -> str | None:
    """Return the identifier to store on role bindings (email preferred for human-readable membership).

    Falls back to principal ID when email is absent (e.g. some service accounts).
    """
    if principal.email:
        email = principal.email.strip()
        if email:
            return email
    if principal.id:
        return principal.id.strip() or None
    return None


def _generate_binding_name(principal: str, workspace: str, role: str) -> str:
    """Generate a deterministic short name for a role binding.

    Uses a hash of the composite key to ensure uniqueness while staying
    within the 32-character limit for entity names.
    """
    composite_key = f"{principal}:{workspace}:{role}"
    hash_digest = hashlib.sha256(composite_key.encode()).hexdigest()[:24]
    return f"rb-{hash_digest}"  # rb- prefix + 24 hex chars = 27 chars


async def _count_active_admins(
    entity_repository: EntityRepository, workspace: str, exclude_principal: str | None = None
) -> int:
    """Count the number of active Admin role bindings for a workspace.

    Args:
        entity_repository: The entity repository to query
        workspace: The workspace to check
        exclude_principal: Optional principal to exclude from the count (for checking
                          if removing this principal would leave no admins)

    Returns:
        Number of active Admin role bindings
    """
    # Role bindings are scoped by the entity workspace. Older seeded bindings do
    # not carry a duplicate data.workspace field, so do not filter on it.
    admin_filter = ComparisonOperation(
        operator=FilterOperator.EQ,
        field="data.role",
        value="Admin",
    )

    bindings, _ = await entity_repository.list_entities(
        workspace=workspace,
        entity_type=ROLE_BINDING_ENTITY_TYPE,
        filter_op=admin_filter,
        page_size=1000,
    )

    # Count active bindings (not revoked), excluding the specified principal if any
    count = 0
    for binding in bindings:
        if binding.data.get("revoked_at") is not None:
            continue
        if exclude_principal and binding.data.get("principal") == exclude_principal:
            continue
        count += 1

    return count


async def _delete_all_role_bindings(entity_repository: EntityRepository, workspace: str) -> list[Entity]:
    """Delete all role bindings in a workspace.

    Args:
        entity_repository: The entity repository
        workspace: The workspace whose role bindings should be deleted

    Returns:
        List of deleted role binding entities
    """
    # Get all role bindings for this workspace
    all_bindings: list[Entity] = []
    page = 1
    page_size = 1000
    while True:
        bindings, total = await entity_repository.list_entities(
            workspace=workspace,
            entity_type=ROLE_BINDING_ENTITY_TYPE,
            page=page,
            page_size=page_size,
        )
        all_bindings.extend(bindings)
        if len(all_bindings) >= total or len(bindings) < page_size:
            break
        page += 1

    # Delete all role bindings
    for binding in all_bindings:
        await entity_repository.delete_entity(entity_id=binding.id)

    return all_bindings


@router.post(
    "/v2/workspaces",
    response_model=Workspace,
    tags=[API_TAG],
    status_code=201,
    summary="Create a new workspace",
    description=textwrap.dedent("""
        Create a new workspace.

        The creator is automatically granted Admin role on the workspace.
        By default, this endpoint waits for the Admin role to propagate before returning.
        Use `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:
        ```
        POST /apis/entities/v2/workspaces
        {
            "name": "ml-team",
            "description": "Machine Learning Team workspace"
        }
        ```
    """),
)
async def create_workspace(
    workspace: WorkspaceInput,
    workspace_repository: WorkspaceRepository,
    entity_repository: EntityRepository,
    auth_client: AuthClientDep,
    wait_role_propagation: bool = Query(
        default=True,
        description="If true, wait for Admin role to propagate before returning (default: true). Set to false for bulk operations.",
    ),
) -> Workspace:
    """Create a new workspace.

    The creator is automatically granted Admin role on the workspace, keyed by email when
    available (otherwise principal ID).
    """
    existing = await workspace_repository.get_workspace_by_name(name=workspace.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Workspace '{workspace.name}' already exists",
        )

    try:
        new_workspace = await workspace_repository.create_workspace(
            name=workspace.name,
            description=workspace.description,
            created_by=auth_client.principal.effective_id,
        )
    except IntegrityError as e:
        error_msg = str(e.orig) if hasattr(e, "orig") else str(e)
        logger.warning(
            "Integrity error creating workspace",
            extra={"workspace": workspace.name, "error": error_msg},
        )

        if "duplicate key" in error_msg.lower() or "unique constraint" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workspace '{workspace.name}' already exists.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data provided. Please check your input.",
            )

    # Grant Admin role to the creator (prefer email for stored principal so listings match IdP email)
    binding_principal = _principal_for_role_binding(auth_client.principal)
    if auth_client.auth_enabled and binding_principal:
        granted_by = auth_client.principal.id if auth_client.principal.id else binding_principal
        granted_at = datetime.now(timezone.utc)
        binding_name = _generate_binding_name(binding_principal, workspace.name, "Admin")

        await entity_repository.create_entity(
            workspace=workspace.name,
            entity_type=ROLE_BINDING_ENTITY_TYPE,
            name=binding_name,
            data={
                "principal": binding_principal,
                "workspace": workspace.name,
                "role": "Admin",
                "granted_by": granted_by,  # Self-granted (actor id when available)
                "granted_at": granted_at.isoformat(),
                "revoked_at": None,
            },
        )
        await bindings_cache_delete(binding_principal)

        # Wait for Admin role to propagate if requested
        if wait_role_propagation:
            if await auth_client.wait_role(binding_principal, workspace.name, "Admin"):
                logger.info(
                    "Admin role granted for workspace creator",
                    extra={"workspace": workspace.name, "principal": binding_principal},
                )
            else:
                logger.warning(
                    "Timeout waiting for Admin role propagation",
                    extra={"workspace": workspace.name, "principal": binding_principal},
                )

    return new_workspace


@router.get(
    "/v2/workspaces",
    response_model=Page[Workspace],
    tags=[API_TAG],
    summary="List all workspaces",
    description=textwrap.dedent("""
        List all workspaces with pagination.

        Workspaces marked for deletion (non-null deletion_stage) are omitted so the
        list matches GET/DELETE, which treat those workspaces as not found.

        When authentication is enabled, only workspaces the principal has access to
        are returned. Service principals and platform admins have access to all workspaces.

        Query Parameters:
        - page, page_size: Pagination
        - sort: Sort field
        - filter: Advanced filters (JSON, text, or bracket notation)

        Example:
        ```
        GET /apis/entities/v2/workspaces?sort=-created_at&page=1&page_size=10
        ```
    """),
)
async def list_workspaces(
    repository: WorkspaceRepository,
    entity_repository: EntityRepository,
    filter: FilterDep,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Items per page"),
    sort: GenericSortField = Query(GenericSortField.CREATED_AT_DESC, description="Sort field"),
) -> Page[Workspace]:
    """List workspaces accessible to the current principal."""
    # Get accessible workspaces for access control
    accessible_workspaces = await get_accessible_workspaces(entity_repository)

    combined_filter = _exclude_deleting_workspaces(add_workspace_filtering(accessible_workspaces, filter, field="name"))

    workspaces, total = await repository.list_workspaces(
        page=page,
        page_size=page_size,
        sort=sort,
        filter_op=combined_filter,
    )

    return Page(
        data=workspaces,
        pagination=PaginationData(
            page=page,
            page_size=page_size,
            total_results=total,
            total_pages=(total + page_size - 1) // page_size,
            current_page_size=len(workspaces),
        ),
        sort=sort,
        filter=filter.to_dict() if filter else None,
    )


@router.get(
    "/v2/workspaces/{name}",
    response_model=Workspace,
    tags=[API_TAG],
    summary="Get workspace by ID",
    description=textwrap.dedent("""
        Get a specific workspace by ID.

        Example:
        ```
        GET /apis/entities/v2/workspaces/ml-team
        ```
    """),
)
async def get_workspace(
    name: str,
    repository: WorkspaceRepository,
) -> Workspace:
    """Get workspace by name."""
    workspace = await repository.get_workspace_by_name(name=name)
    if workspace is None or workspace._deletion_stage is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{name}' not found",
        )

    return workspace


@router.put(
    "/v2/workspaces/{name}",
    response_model=Workspace,
    tags=[API_TAG],
    summary="Update workspace",
    description=textwrap.dedent("""
        Update a workspace's description.

        Example:
        ```
        PUT /apis/entities/v2/workspaces/ml-team
        {
            "description": "Updated description for ML Team"
        }
        ```
    """),
)
async def update_workspace(
    name: str,
    workspace_data: WorkspaceUpdate,
    repository: WorkspaceRepository,
    auth_client: AuthClientDep,
) -> Workspace:
    """Update workspace."""
    try:
        workspace = await repository.update_workspace(
            name=name,
            description=workspace_data.description,
            updated_by=auth_client.principal.effective_id,
        )
        if workspace is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{name}' not found",
            )
        return workspace
    except IntegrityError as e:
        error_msg = str(e.orig) if hasattr(e, "orig") else str(e)
        logger.warning(
            "Integrity error updating workspace",
            extra={"workspace": name, "error": error_msg},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid data provided.",
        )


@router.delete(
    "/v2/workspaces/{name}",
    response_model=DeleteResponse,
    tags=[API_TAG],
    summary="Delete workspace",
    description=textwrap.dedent("""
        Delete a workspace.

        This marks the workspace for deletion and returns immediately. The workspace
        will no longer be accessible via the API. An asynchronous cleanup controller
        will handle deletion of all entities and external resources.

        Role bindings are immediately deleted to revoke access.

        Example:
        ```
        DELETE /apis/entities/v2/workspaces/ml-team
        ```
    """),
)
async def delete_workspace(
    name: str,
    repository: WorkspaceRepository,
    entity_repository: EntityRepository,
) -> DeleteResponse:
    """Mark workspace for deletion."""
    # Check if workspace exists first
    workspace = await repository.get_workspace_by_name(name=name)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{name}' not found",
        )

    # Mark workspace for deletion (sets deletion_stage = 'pending')
    marked = await repository.mark_workspace_for_deletion(
        name=name,
        deletion_stage=WorkspaceDeletionStage.PENDING,
    )

    if not marked:
        # Workspace was deleted by someone else between our check and mark
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{name}' not found",
        )

    logger.info(
        "Marked workspace for deletion",
        extra={"workspace": name, "deletion_stage": WorkspaceDeletionStage.PENDING},
    )

    # Delete all role bindings immediately (revoke access)
    deleted_bindings = await _delete_all_role_bindings(entity_repository, name)
    for binding in deleted_bindings:
        principal = binding.data.get("principal")
        if principal is not None:
            await bindings_cache_delete(str(principal))
    logger.info(
        "Deleted role bindings for workspace deletion",
        extra={"workspace": name, "deleted_count": len(deleted_bindings)},
    )

    return DeleteResponse(
        id=name,
        message="Workspace marked for deletion",
    )


# =================== Workspace Member Management ===================


@router.get(
    "/v2/workspaces/{workspace}/members",
    response_model=WorkspaceMemberListResponse,
    tags=[API_TAG],
    summary="List workspace members",
    description=textwrap.dedent("""
        List all members of a workspace with their roles.

        Returns a list of all principals with active role bindings in the workspace.

        Example:
        ```
        GET /apis/entities/v2/workspaces/ml-team/members
        ```
    """),
)
async def list_workspace_members(
    workspace: str,
    workspace_repository: WorkspaceRepository,
    entity_repository: EntityRepository,
) -> WorkspaceMemberListResponse:
    """List all members of a workspace."""
    # Check if workspace exists
    ws = await workspace_repository.get_workspace_by_name(name=workspace)
    if ws is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace}' not found",
        )

    # Get all role bindings for this workspace
    all_bindings = []
    page = 1
    page_size = 1000
    while True:
        bindings, total = await entity_repository.list_entities(
            workspace=workspace,
            entity_type=ROLE_BINDING_ENTITY_TYPE,
            page=page,
            page_size=page_size,
        )
        all_bindings.extend(bindings)
        if len(all_bindings) >= total or len(bindings) < page_size:
            break
        page += 1

    # Filter to only active bindings (revoked_at is None)
    active_bindings = [b for b in all_bindings if b.data.get("revoked_at") is None]

    # Group by principal and collect roles
    members_dict: dict[str, dict] = {}
    for binding in active_bindings:
        principal = binding.data.get("principal")
        if principal not in members_dict:
            members_dict[principal] = {
                "principal": principal,
                "roles": [],
                "granted_at": binding.data.get("granted_at"),
                "granted_by": binding.data.get("granted_by"),
            }
        role = binding.data.get("role")
        if role not in members_dict[principal]["roles"]:
            members_dict[principal]["roles"].append(role)
        # Keep the earliest granted_at date
        binding_granted_at = binding.data.get("granted_at")
        if binding_granted_at and (
            members_dict[principal]["granted_at"] is None or binding_granted_at < members_dict[principal]["granted_at"]
        ):
            members_dict[principal]["granted_at"] = binding_granted_at

    return WorkspaceMemberListResponse(data=list(members_dict.values()))


@router.post(
    "/v2/workspaces/{workspace}/members",
    response_model=WorkspaceMember,
    tags=[API_TAG],
    status_code=201,
    summary="Add workspace member",
    description=textwrap.dedent("""
        Add a new member to the workspace with specified roles.

        This creates role bindings for the specified principal with the given roles.
        By default, this endpoint waits for the roles to propagate before returning.
        Use `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:
        ```
        POST /apis/entities/v2/workspaces/ml-team/members
        {
            "principal": "user@example.com",
            "roles": ["Editor"]
        }
        ```
    """),
)
async def add_workspace_member(
    workspace: str,
    member: WorkspaceMemberInput,
    workspace_repository: WorkspaceRepository,
    entity_repository: EntityRepository,
    auth_client: AuthClientDep,
    wait_role_propagation: bool = Query(
        default=True,
        description="If true, wait for roles to propagate before returning (default: true). Set to false for bulk operations.",
    ),
) -> WorkspaceMember:
    """Add a new member to the workspace."""
    # Check if workspace exists
    ws = await workspace_repository.get_workspace_by_name(name=workspace)
    if ws is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace}' not found",
        )

    granted_by = auth_client.principal.id if auth_client.principal.id else "system"
    granted_at = datetime.now(timezone.utc)

    # Create role bindings for each role
    for role in member.roles:
        # Check if binding already exists
        binding_name = _generate_binding_name(member.principal, workspace, role)
        existing = await entity_repository.get_entity_by_name(
            workspace=workspace,
            entity_type=ROLE_BINDING_ENTITY_TYPE,
            name=binding_name,
        )

        if existing is None:
            # Create new binding
            await entity_repository.create_entity(
                workspace=workspace,
                entity_type=ROLE_BINDING_ENTITY_TYPE,
                name=binding_name,
                data={
                    "principal": member.principal,
                    "workspace": workspace,
                    "role": role,
                    "granted_by": granted_by,
                    "granted_at": granted_at.isoformat(),
                    "revoked_at": None,
                },
            )
        elif existing.data.get("revoked_at") is not None:
            # Reactivate a previously revoked binding
            await entity_repository.update_entity(
                entity_id=existing.id,
                data={
                    **existing.data,
                    "granted_by": granted_by,
                    "granted_at": granted_at.isoformat(),
                    "revoked_at": None,
                },
            )

    await bindings_cache_delete(member.principal)

    # Wait for roles to propagate if requested
    if wait_role_propagation and member.roles:
        for role in member.roles:
            if await auth_client.wait_role(member.principal, workspace, role):
                logger.info(
                    "Role granted for workspace member",
                    extra={"workspace": workspace, "principal": member.principal, "role": role},
                )
            else:
                logger.warning(
                    "Timeout waiting for role propagation",
                    extra={"workspace": workspace, "principal": member.principal, "role": role},
                )

    return WorkspaceMember(
        principal=member.principal,
        roles=member.roles,
        granted_at=granted_at,
        granted_by=granted_by,
    )


@router.put(
    "/v2/workspaces/{workspace}/members/{principal_id}",
    response_model=WorkspaceMember,
    tags=[API_TAG],
    summary="Update workspace member roles",
    description=textwrap.dedent("""
        Update the roles for a workspace member.

        This will revoke existing roles not in the new list and add new roles.
        By default, this endpoint waits for the roles to propagate before returning.
        Use `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:
        ```
        PUT /apis/entities/v2/workspaces/ml-team/members/user@example.com
        {
            "roles": ["Viewer", "Editor"]
        }
        ```
    """),
)
async def update_workspace_member(
    workspace: str,
    principal_id: str,
    member_update: WorkspaceMemberUpdate,
    workspace_repository: WorkspaceRepository,
    entity_repository: EntityRepository,
    auth_client: AuthClientDep,
    wait_role_propagation: bool = Query(
        default=True,
        description="If true, wait for roles to propagate before returning (default: true). Set to false for bulk operations.",
    ),
) -> WorkspaceMember:
    """Update the roles for a workspace member."""
    # Check if workspace exists
    ws = await workspace_repository.get_workspace_by_name(name=workspace)
    if ws is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace}' not found",
        )

    granted_by = auth_client.principal.id if auth_client.principal.id else "system"
    now = datetime.now(timezone.utc)

    # Role bindings are already scoped to the workspace by the entity workspace.
    # Seeded role bindings predate data.workspace, so filtering on data.workspace
    # would miss active seeded bindings and leave PDP authorization stale.
    principal_filter = ComparisonOperation(
        operator=FilterOperator.EQ,
        field="data.principal",
        value=principal_id,
    )

    # Get current role bindings for this principal
    current_bindings, _ = await entity_repository.list_entities(
        workspace=workspace,
        entity_type=ROLE_BINDING_ENTITY_TYPE,
        filter_op=principal_filter,
        page_size=1000,
    )

    # Filter to only active bindings
    active_bindings = [b for b in current_bindings if b.data.get("revoked_at") is None]

    current_roles = {str(b.data.get("role")) for b in active_bindings}
    new_roles = set(member_update.roles)

    # Track which roles are being added and removed
    roles_to_add = new_roles - current_roles
    roles_to_remove = current_roles - new_roles

    # Check if removing Admin role would leave workspace without any admins
    if "Admin" in roles_to_remove:
        other_admins = await _count_active_admins(entity_repository, workspace, exclude_principal=principal_id)
        if other_admins == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot remove Admin role from '{principal_id}'. This is the last Admin for workspace '{workspace}'. "
                "Add another Admin before removing this one.",
            )

    # Revoke roles that are no longer needed
    for binding in active_bindings:
        if binding.data.get("role") not in new_roles:
            await entity_repository.update_entity(
                entity_id=binding.id,
                data={
                    **binding.data,
                    "revoked_at": now.isoformat(),
                },
            )

    # Add new roles
    for role in roles_to_add:
        binding_name = _generate_binding_name(principal_id, workspace, role)

        # Check if there's a revoked binding we can reactivate
        existing = await entity_repository.get_entity_by_name(
            workspace=workspace,
            entity_type=ROLE_BINDING_ENTITY_TYPE,
            name=binding_name,
        )

        if existing is None:
            # Create new binding
            await entity_repository.create_entity(
                workspace=workspace,
                entity_type=ROLE_BINDING_ENTITY_TYPE,
                name=binding_name,
                data={
                    "principal": principal_id,
                    "workspace": workspace,
                    "role": role,
                    "granted_by": granted_by,
                    "granted_at": now.isoformat(),
                    "revoked_at": None,
                },
            )
        elif existing.data.get("revoked_at") is not None:
            # Reactivate a previously revoked binding
            await entity_repository.update_entity(
                entity_id=existing.id,
                data={
                    **existing.data,
                    "workspace": workspace,  # Ensure workspace field is present
                    "granted_by": granted_by,
                    "granted_at": now.isoformat(),
                    "revoked_at": None,
                },
            )

    await bindings_cache_delete(principal_id)

    # Wait for roles to propagate if requested
    if wait_role_propagation:
        # Wait for all added roles to be granted
        for role in roles_to_add:
            if await auth_client.wait_role(principal_id, workspace, role, is_present=True):
                logger.info(
                    "Role granted for workspace member",
                    extra={"workspace": workspace, "principal": principal_id, "role": role},
                )
            else:
                logger.warning(
                    "Timeout waiting for role to be granted",
                    extra={"workspace": workspace, "principal": principal_id, "role": role},
                )

        # Wait for all removed roles to be revoked
        for role in roles_to_remove:
            if await auth_client.wait_role(principal_id, workspace, role, is_present=False):
                logger.info(
                    "Role revoked for workspace member",
                    extra={"workspace": workspace, "principal": principal_id, "role": role},
                )
            else:
                logger.warning(
                    "Timeout waiting for role to be revoked",
                    extra={"workspace": workspace, "principal": principal_id, "role": role},
                )

    return WorkspaceMember(
        principal=principal_id,
        roles=list(new_roles),
        granted_by=granted_by,
        granted_at=now,
    )


@router.delete(
    "/v2/workspaces/{workspace}/members/{principal_id}",
    response_model=DeleteResponse,
    tags=[API_TAG],
    summary="Remove workspace member",
    description=textwrap.dedent("""
        Remove a member from the workspace by revoking all their roles.

        This revokes all active role bindings for the principal in the workspace.
        By default, this endpoint waits for all roles to be revoked before returning.
        Use `wait_role_propagation=false` to skip waiting (useful for bulk operations).

        Example:
        ```
        DELETE /apis/entities/v2/workspaces/ml-team/members/user@example.com
        ```
    """),
)
async def remove_workspace_member(
    workspace: str,
    principal_id: str,
    workspace_repository: WorkspaceRepository,
    entity_repository: EntityRepository,
    auth_client: AuthClientDep,
    wait_role_propagation: bool = Query(
        default=True,
        description="If true, wait for roles to propagate before returning (default: true). Set to false for bulk operations.",
    ),
) -> DeleteResponse:
    """Remove a member from the workspace."""
    # Check if workspace exists
    ws = await workspace_repository.get_workspace_by_name(name=workspace)
    if ws is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace}' not found",
        )

    # Role bindings are already scoped to the workspace by the entity workspace.
    # Seeded role bindings predate data.workspace, so filtering on data.workspace
    # would miss active seeded bindings.
    principal_filter = ComparisonOperation(
        operator=FilterOperator.EQ,
        field="data.principal",
        value=principal_id,
    )

    # Get all role bindings for this principal in this workspace
    bindings, _ = await entity_repository.list_entities(
        workspace=workspace,
        entity_type=ROLE_BINDING_ENTITY_TYPE,
        filter_op=principal_filter,
        page_size=1000,
    )

    # Filter to only active bindings
    active_bindings = [b for b in bindings if b.data.get("revoked_at") is None]

    if not active_bindings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Member '{principal_id}' not found in workspace '{workspace}'",
        )

    # Collect the roles being revoked for waiting
    revoked_roles = [str(binding.data.get("role")) for binding in active_bindings]

    # Check if removing this member would leave workspace without any admins
    if "Admin" in revoked_roles:
        other_admins = await _count_active_admins(entity_repository, workspace, exclude_principal=principal_id)
        if other_admins == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot remove member '{principal_id}'. This is the last Admin for workspace '{workspace}'. "
                "Add another Admin before removing this member.",
            )

    # Revoke all bindings
    now = datetime.now(timezone.utc)
    for binding in active_bindings:
        await entity_repository.update_entity(
            entity_id=binding.id,
            data={
                **binding.data,
                "revoked_at": now.isoformat(),
            },
        )

    await bindings_cache_delete(principal_id)

    # Wait for roles to be revoked if requested
    if wait_role_propagation and revoked_roles:
        for role in revoked_roles:
            if await auth_client.wait_role(principal_id, workspace, role, is_present=False):
                logger.info(
                    "Role revoked for workspace member",
                    extra={"workspace": workspace, "principal": principal_id, "role": role},
                )
            else:
                logger.warning(
                    "Timeout waiting for role to be revoked",
                    extra={"workspace": workspace, "principal": principal_id, "role": role},
                )

    return DeleteResponse(
        id=principal_id,
        message=f"Member removed from workspace '{workspace}'",
    )
