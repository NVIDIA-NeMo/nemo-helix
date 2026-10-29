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
- GET /apis/entities/v2/workspace-deletions/{name} - Read in-progress deletion status

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
from nemo_helix_plugin.workspaces.constants import SAMPLE_WORKSPACE_PREFIX
from nemo_helix_plugin.workspaces.types import WorkspaceDeletionStatus
from nhx.common.api.common import DeleteResponse, GenericSortField, Page, PaginationData
from nhx.common.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.auth.principal_identifier import SERVICE_ACCOUNT_PRINCIPAL_PREFIX
from nhx.core.entities.api.dependencies import (
    AccountCredentialStoreDep,
    AccountIdentityStoreDep,
    AuthClientDep,
    EntityRepository,
    WorkspaceRepository,
)
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
from nhx.core.entities.app.repository import (
    SERVICE_ACCOUNT_IDENTITY_ISSUER,
    AccountCredentialStore,
    AccountCredentialType,
    AccountIdentityStore,
    WorkspaceRepositoryInterface,
)
from nhx.core.entities.entities import Entity, Workspace, WorkspaceDeletionStage
from nhx.core.entities.utils.filter import FilterDep
from nhx.core.entities.utils.identifiers import entity_id_prefix
from sqlalchemy.exc import IntegrityError

router = APIRouter()
API_TAG = "Entity Store"
logger = logging.getLogger(__name__)

# Scoped Access Keys, the only credential type bound to a workspace today.
_ACCESS_KEY_CREDENTIAL_TYPE: AccountCredentialType = "access_key"
# Prefix of a stable account ID, which a Scoped Access Key also authenticates as.
_ACCOUNT_ID_PREFIX = entity_id_prefix("account")

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


def _workspace_name_conflict_detail(existing: Workspace) -> str:
    """Distinguish a live workspace from a name reserved by cleanup."""
    if existing._deletion_stage is None:
        return f"Workspace '{existing.name}' already exists"
    detail = f"Workspace '{existing.name}' is still being deleted"
    if existing._deletion_error:
        return f"{detail} ({existing._deletion_error})"
    return detail


def _principal_for_role_binding(principal: Principal) -> str | None:
    """Use the acting principal for role bindings, including on-behalf-of requests.

    Effective identity matches the principal for direct requests. Prefer email for
    human-readable membership, falling back to ID when email is absent.
    """
    if principal.effective_email:
        email = principal.effective_email.strip()
        if email:
            return email
    if principal.effective_id:
        return principal.effective_id.strip() or None
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

        Names beginning with `sample-` are reserved for the Agents sample flow.

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
    if workspace.name.startswith(SAMPLE_WORKSPACE_PREFIX) and auth_client.principal.id != "service:agents":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Workspace names beginning with '{SAMPLE_WORKSPACE_PREFIX}' are reserved for the sample flow",
        )

    existing = await workspace_repository.get_workspace_by_name(name=workspace.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_workspace_name_conflict_detail(existing),
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
            raced = await workspace_repository.get_workspace_by_name(name=workspace.name)
            if raced is not None and raced._deletion_stage is not None:
                detail = _workspace_name_conflict_detail(raced)
            else:
                detail = f"Workspace '{workspace.name}' already exists."
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=detail,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data provided. Please check your input.",
            )

    # Grant Admin role to the creator (prefer email for stored principal so listings match IdP email)
    #
    # Deliberately not subject to _require_bound_workspace_admin: the creator acts for itself here,
    # and there is no one else to ask. A service account whose Scoped Access Key is bound to
    # another workspace therefore becomes Admin of a workspace outside its binding. That is access
    # outside the binding, so the bound workspace's Admins can no longer rotate or unsuspend its
    # keys until a HelixAdmin removes the role.
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


def _workspace_not_found(name: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Workspace '{name}' not found",
    )


async def _may_read_workspace_deletion(auth_client: AuthClient, workspace: Workspace) -> bool:
    """Allow the creator, a platform admin, or a service principal.

    Auth off allows the read. Anyone else gets the same not-found response as a
    missing workspace, so the route does not confirm that the name exists.
    """
    if not auth_client.auth_enabled or auth_client.principal.caller_kind == "service_principal":
        return True
    created_by = (workspace.created_by or "").strip()
    principal = auth_client.principal
    identities = {principal.effective_id.strip()}
    email = (principal.effective_email or "").strip()
    if email:
        identities.add(email)
    identities.discard("")
    if created_by and created_by in identities:
        return True
    return await auth_client.has_role("system", "HelixAdmin")


@router.get(
    "/v2/workspace-deletions/{name}",
    response_model=WorkspaceDeletionStatus,
    tags=[API_TAG],
    summary="Get workspace deletion status",
    description=textwrap.dedent("""
        Read the cleanup stage for a workspace whose name is still reserved.

        The workspace itself is hidden from get and list while deletion is in
        progress. This route is system-scoped because delete already removed the
        workspace role bindings. With authentication enabled, the original
        creator, a platform admin, and service principals can read it. A missing
        workspace, a live workspace, and a deletion the caller may not see all
        return not found.

        Example:
        ```
        GET /apis/entities/v2/workspace-deletions/ml-team
        ```
    """),
)
async def get_workspace_deletion(
    name: str,
    repository: WorkspaceRepository,
    auth_client: AuthClientDep,
) -> WorkspaceDeletionStatus:
    """Return the stored cleanup stage without changing the deletion lease."""
    workspace = await repository.get_workspace_by_name(name=name)
    if workspace is None or workspace._deletion_stage is None:
        raise _workspace_not_found(name)
    if not await _may_read_workspace_deletion(auth_client, workspace):
        raise _workspace_not_found(name)
    return WorkspaceDeletionStatus(
        name=workspace.name,
        deletion_stage=workspace._deletion_stage.value,
        deletion_error=workspace._deletion_error,
        updated_at=workspace.updated_at,
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


async def _require_bound_workspace_admin(
    principal: str,
    workspace: Workspace,
    *,
    auth_client: AuthClient,
    workspace_repository: WorkspaceRepositoryInterface,
    identity_store: AccountIdentityStore,
    credential_store: AccountCredentialStore,
) -> None:
    """Refuse to grant a service account roles outside the workspace its Scoped Access Keys are bound to.

    Roles belong to the service account, not to a key, so a role granted here reaches every live
    key of that account — including one bound to another workspace and managed by that workspace's
    Admins. Granting it is therefore that workspace's decision: only a HelixAdmin, or an Admin of
    every workspace the account's live keys are bound to, may widen the account. Without this, any
    Admin (and anyone can create a workspace and be its Admin) could hand their own access to
    someone else's service credential.

    Granting roles in the workspace a key is already bound to is unrestricted: the endpoint's own
    Admin check already covers it. So is an account with no live bound keys.
    """
    if not auth_client.auth_enabled:
        return
    if principal.startswith(SERVICE_ACCOUNT_PRINCIPAL_PREFIX):
        account_id = await identity_store.get_account_id(
            issuer=SERVICE_ACCOUNT_IDENTITY_ISSUER,
            subject=principal.removeprefix(SERVICE_ACCOUNT_PRINCIPAL_PREFIX),
        )
    elif principal.startswith(_ACCOUNT_ID_PREFIX):
        # A Scoped Access Key also authenticates as its account ID, so a binding can be keyed by it.
        account_id = principal
    else:
        return
    if account_id is None:
        # Never issued a key, so it has none to widen.
        return

    bound_ids = await credential_store.bound_workspace_ids_for_subject(_ACCESS_KEY_CREDENTIAL_TYPE, account_id)
    elsewhere = sorted(bound_ids - {workspace.id})
    if not elsewhere:
        return
    # A service principal acts for the platform, not for a person, and already passed the route's
    # own authorization.
    if auth_client.principal.effective_principal.is_privileged():
        return
    if await auth_client.has_role("system", "HelixAdmin"):
        return

    names: list[str] = []
    for workspace_id in elsewhere:
        bound = await workspace_repository.get_workspace_by_id(workspace_id=workspace_id)
        # A workspace that is gone can't be administered, so its keys stay HelixAdmin-only. Keep
        # the ID in the message rather than dropping it, so the caller can name it to an admin.
        if bound is None:
            names.append(workspace_id)
            continue
        # Confirm the name still maps back to the ID: a workspace deleted and recreated under the
        # same name is a different workspace, and being Admin of the new one grants nothing here.
        if not await auth_client.has_role(bound.name, "Admin"):
            names.append(bound.name)
            continue
        current = await workspace_repository.get_workspace_by_name(name=bound.name)
        if current is None or current.id != workspace_id:
            names.append(bound.name)

    if names:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"'{principal}' has live Scoped Access Keys bound to workspace(s) "
                f"{', '.join(repr(name) for name in names)}. Roles belong to the service account, so those "
                f"keys would also get the roles granted here. Only a HelixAdmin, or an Admin of every "
                f"workspace they're bound to, can grant it roles in '{workspace.name}'."
            ),
        )


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

        A service account with live Scoped Access Keys bound to another workspace can only be
        granted roles here by a HelixAdmin or an Admin of every workspace those keys are bound
        to; other callers get 403.

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
    identity_store: AccountIdentityStoreDep,
    credential_store: AccountCredentialStoreDep,
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

    if member.roles:
        await _require_bound_workspace_admin(
            member.principal,
            ws,
            auth_client=auth_client,
            workspace_repository=workspace_repository,
            identity_store=identity_store,
            credential_store=credential_store,
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

        Adding roles for a service account with live Scoped Access Keys bound to another
        workspace requires HelixAdmin or Admin of every workspace those keys are bound to, as
        when adding a member; other callers get 403. Removing roles is never restricted.

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
    identity_store: AccountIdentityStoreDep,
    credential_store: AccountCredentialStoreDep,
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

    # Only widening the account's access is restricted; taking roles away never is.
    if roles_to_add:
        await _require_bound_workspace_admin(
            principal_id,
            ws,
            auth_client=auth_client,
            workspace_repository=workspace_repository,
            identity_store=identity_store,
            credential_store=credential_store,
        )

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
