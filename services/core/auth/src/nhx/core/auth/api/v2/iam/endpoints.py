# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IAM endpoints for Auth Service."""

import hashlib
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from nhx.common.api.common import DeleteResponse, Page
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nhx.common.api.parsed_filter import ParsedFilter, make_filter_dep
from nhx.common.api.utils import generate_openapi_extra_params
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.entities import ALL_WORKSPACES, EntityClient, EntityConflictError, EntityNotFoundError
from nhx.common.service.dependencies import get_entity_client
from nhx.core.auth.entities import RoleBindingEntity

from .schemas import (
    RoleBinding,
    RoleBindingFilter,
    RoleBindingInput,
)

router = APIRouter(tags=["IAM"])
logger = logging.getLogger(__name__)


def require_service_principal_for_iam_role_bindings(auth_client: AuthClient) -> None:
    """Enforce access rules for IAM role-binding routes."""
    if not auth_client.auth_enabled:
        return
    principal_id = auth_client.principal.id or ""
    if principal_id.startswith("service:"):
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


def _generate_binding_name(principal: str, workspace: str, role: str) -> str:
    """Generate a deterministic short name for a role binding.

    Uses a hash of the composite key to ensure uniqueness while staying
    within the 32-character limit for entity names.
    """
    composite_key = f"{principal}:{workspace}:{role}"
    hash_digest = hashlib.sha256(composite_key.encode()).hexdigest()[:24]
    return f"rb-{hash_digest}"  # rb- prefix + 24 hex chars = 27 chars


async def _get_role_binding_by_name(
    entities_client: EntityClient,
    name: str,
) -> RoleBindingEntity:
    """Resolve a role binding by its deterministic name across workspaces."""
    try:
        return await entities_client.find_one(
            RoleBindingEntity,
            workspace=ALL_WORKSPACES,
            filter_obj={"name": name},
        )
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail="Role binding not found") from e
    except EntityConflictError as e:
        raise HTTPException(status_code=409, detail=f"Multiple role bindings found for name '{name}'") from e


@router.get(
    "/v2/iam/role-bindings",
    response_model=Page[RoleBinding],
    summary="List role bindings",
    description="List all role bindings (Platform Admin only)",
    openapi_extra=generate_openapi_extra_params(
        filter_schema=RoleBindingFilter,
        filter_description="Filter role bindings by principal, workspace, role, granted_by, is_active, granted_at, and revoked_at.",
    ),
)
async def list_role_bindings(
    page: int = Query(default=1, description="Page number."),
    page_size: int = Query(default=10, description="Page size."),
    # TODO(v2): Support sorting by granted_at once entities API supports custom sort fields
    sort: str = Query(
        default="created_at",
        description="The field to sort by. To sort in decreasing order, use `-` in front of the field name.",
    ),
    parsed: ParsedFilter = Depends(make_filter_dep(RoleBindingFilter)),
    auth_client: AuthClient = Depends(get_auth_client),
    entities_client: EntityClient = Depends(get_entity_client),
) -> Page[RoleBinding]:
    """List all role bindings.

    This endpoint requires Platform Admin permissions.
    """
    require_service_principal_for_iam_role_bindings(auth_client)

    # Translate is_active filter to revoked_at EXISTS filter for database query
    is_active_value = parsed.remove("is_active")
    if is_active_value is not None:
        # is_active=True means revoked_at IS NULL (EXISTS=false)
        # is_active=False means revoked_at IS NOT NULL (EXISTS=true)
        revoked_condition = ComparisonOperation(
            field="revoked_at",
            operator=FilterOperator.EXISTS,
            value=not is_active_value,
        )
        if parsed.operation is not None:
            parsed.operation = LogicalOperation(
                operator=FilterOperator.AND,
                operations=[revoked_condition, parsed.operation],
            )
        else:
            parsed.operation = revoked_condition

    res = await entities_client.list(
        RoleBindingEntity,
        workspace=ALL_WORKSPACES,
        page=page,
        page_size=page_size,
        sort=sort,
        filter_operation=parsed.operation,
    )

    return Page(  # type: ignore[return-value]
        data=[RoleBinding.model_validate(r.model_dump()) for r in res.data],
        pagination=res.pagination.model_dump(),
        sort=sort,
        filter=parsed.to_response(),
    )


@router.post(
    "/v2/iam/role-bindings",
    response_model=RoleBinding,
    summary="Create role binding",
    description="Create a new role binding (Platform Admin only)",
)
async def create_role_binding(
    role_binding: RoleBindingInput,
    auth_client: AuthClient = Depends(get_auth_client),
    entities_client: EntityClient = Depends(get_entity_client),
    wait_role_propagation: bool = Query(
        default=True,
        description="If true, wait for role to propagate before returning (default: true). Set to false for bulk operations.",
    ),
) -> RoleBinding:
    """Create a new role binding.

    This endpoint requires Platform Admin permissions.
    By default, this endpoint waits for the role to propagate before returning.
    Use `wait_role_propagation=false` to skip waiting (useful for bulk operations).
    """
    require_service_principal_for_iam_role_bindings(auth_client)
    # Get the principal ID of the user making the request
    granted_by = auth_client.principal.id if auth_client.principal.id else "system"

    # Check if there's already an active binding for this principal/workspace/role
    target_workspace = role_binding.workspace or "system"
    existing_all = await entities_client.list(
        RoleBindingEntity,
        workspace=ALL_WORKSPACES,
        filter_obj={
            "principal": role_binding.principal,
            "workspace": target_workspace,
            "role": role_binding.role,
        },
    )
    # Filter to only active bindings
    existing = [b for b in existing_all.data if b.revoked_at is None]

    if existing:
        raise HTTPException(
            status_code=409,
            detail="Active role binding already exists for this principal/workspace/role combination",
        )

    # Create the role binding
    # Name is a required field in EntityBase, so we generate a unique name from the binding fields
    # workspace is both the storage scope and the workspace this binding grants access to
    binding_name = _generate_binding_name(role_binding.principal, target_workspace, role_binding.role)
    obj = RoleBindingEntity(
        name=binding_name,
        workspace=target_workspace,
        principal=role_binding.principal,
        role=role_binding.role,
        granted_by=granted_by,
        granted_at=datetime.now(timezone.utc),
    )

    saved = await entities_client.add(obj)

    # Wait for role to propagate if requested
    if wait_role_propagation and saved.workspace:
        if await auth_client.wait_role(saved.principal, saved.workspace, saved.role):
            logger.info(f"Role '{saved.role}' granted for {saved.principal} in workspace {saved.workspace}")
        else:
            logger.warning(f"Timeout waiting for role '{saved.role}' for {saved.principal}")

    return RoleBinding.model_validate(saved.model_dump())


@router.get(
    "/v2/iam/role-bindings/{name}",
    response_model=RoleBinding,
    summary="Get role binding",
    description="Get a specific role binding (Platform Admin only)",
)
async def get_role_binding(
    name: str,
    auth_client: AuthClient = Depends(get_auth_client),
    entities_client: EntityClient = Depends(get_entity_client),
) -> RoleBinding:
    """Get a specific role binding.

    This endpoint requires Platform Admin permissions.
    """
    require_service_principal_for_iam_role_bindings(auth_client)
    obj = await _get_role_binding_by_name(entities_client, name)

    return RoleBinding.model_validate(obj.model_dump())


@router.delete(
    "/v2/iam/role-bindings/{name}",
    response_model=DeleteResponse,
    summary="Revoke role binding",
    description="Revoke a role binding (Platform Admin only)",
)
async def revoke_role_binding(
    name: str,
    auth_client: AuthClient = Depends(get_auth_client),
    entities_client: EntityClient = Depends(get_entity_client),
    wait_role_propagation: bool = Query(
        default=True,
        description="If true, wait for role to propagate before returning (default: true). Set to false for bulk operations.",
    ),
) -> DeleteResponse:
    """Revoke a role binding.

    This endpoint requires Platform Admin permissions.
    Note: This performs a soft delete by setting revoked_at timestamp.
    By default, this endpoint waits for the role to be revoked before returning.
    Use `wait_role_propagation=false` to skip waiting (useful for bulk operations).
    """
    require_service_principal_for_iam_role_bindings(auth_client)
    obj = await _get_role_binding_by_name(entities_client, name)

    if obj.revoked_at is not None:
        raise HTTPException(status_code=409, detail="Role binding is already revoked")

    # Soft delete by setting revoked_at
    obj.revoked_at = datetime.now(timezone.utc)
    await entities_client.update(obj)

    # Wait for role to be revoked if requested
    if wait_role_propagation and obj.workspace:
        if await auth_client.wait_role(obj.principal, obj.workspace, obj.role, is_present=False):
            logger.info(f"Role '{obj.role}' revoked for {obj.principal} in workspace {obj.workspace}")
        else:
            logger.warning(f"Timeout waiting for role '{obj.role}' to be revoked for {obj.principal}")

    return DeleteResponse(id=obj.id)
