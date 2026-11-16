# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Additional authorization for core control entities and resolved references."""

from typing import Literal

from fastapi import HTTPException, status
from nhx.common.api.filter import FilterOperation, LogicalOperation
from nhx.common.auth import ALL_WORKSPACES
from nhx.common.auth.client import AuthClient
from nhx.core.entities.app.repository import WorkspaceRepositoryInterface
from nhx.core.entities.utils.filter import RelationshipFilterOperation

CONTROL_ENTITY_TYPES = frozenset({"role_binding", "access_key"})
EntityOperation = Literal["create", "read", "list", "update", "delete"]


async def require_control_entity_access(
    auth_client: AuthClient,
    workspace: str,
    entity_type: str,
    operation: EntityOperation,
) -> None:
    if entity_type not in CONTROL_ENTITY_TYPES or not auth_client.auth_enabled:
        return
    permissions = {
        "create": ["iam.create"],
        "read": ["iam.read"],
        "list": ["iam.list"],
        "update": ["iam.create", "iam.delete"],
        "delete": ["iam.delete"],
    }[operation]
    allowed = (
        await auth_client.check_permissions(workspace, subject=permissions)
        if auth_client.principal.on_behalf_of
        else await auth_client.check_permissions(workspace, actor=permissions)
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not allowed to {operation} control entities in workspace '{workspace}'",
        )


async def get_control_entity_list_workspaces(
    auth_client: AuthClient,
    workspace_repository: WorkspaceRepositoryInterface,
    entity_type: str,
    accessible_workspaces: set[str] | None,
) -> set[str] | None:
    """Restrict control collections to real workspaces with the caller's IAM grant."""
    if entity_type not in CONTROL_ENTITY_TYPES or not auth_client.auth_enabled:
        return accessible_workspaces

    if accessible_workspaces is None:
        candidates: set[str] = set()
        page = 1
        page_size = 1000
        while True:
            workspaces, total = await workspace_repository.list_workspaces(page=page, page_size=page_size, sort="name")
            candidates.update(workspace.name for workspace in workspaces)
            if not workspaces or page * page_size >= total:
                break
            page += 1
    else:
        candidates = accessible_workspaces

    candidates = candidates - {ALL_WORKSPACES}
    if auth_client.principal.on_behalf_of:
        return await auth_client.filter_workspaces(candidates, subject=["iam.list"])
    return await auth_client.filter_workspaces(candidates, actor=["iam.list"])


async def require_entity_read_access(auth_client: AuthClient, workspace: str, entity_type: str) -> None:
    if auth_client.auth_enabled and not await auth_client.check_permissions(workspace, actor=["entities.read"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not allowed to read entities in workspace '{workspace}'",
        )
    await require_control_entity_access(auth_client, workspace, entity_type, "read")


async def require_relationship_access(auth_client: AuthClient, operation: FilterOperation | None) -> None:
    if isinstance(operation, RelationshipFilterOperation):
        await require_control_entity_access(
            auth_client, ALL_WORKSPACES, operation.relationship.target_entity_type, "list"
        )
        await require_relationship_access(auth_client, operation.condition)
    elif isinstance(operation, LogicalOperation):
        for child in operation.operations:
            await require_relationship_access(auth_client, child)
